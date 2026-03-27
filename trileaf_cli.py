#!/usr/bin/env python3
"""
trileaf — command-line interface for the Trileaf writing optimiser.

Usage
-----
  trileaf run              Start the dashboard server
  trileaf stop             Stop the server and release GPU memory
  trileaf setup            First-time setup: download models + configure provider
  trileaf config           Add or edit rewrite provider profiles
  trileaf doctor           Check that all models and configuration are in place
  trileaf remove           Remove Trileaf and its generated files

Run  trileaf <command> --help  for per-command options.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Ensure the project root is on sys.path so `run` and `scripts.*` are importable
# regardless of how the console-scripts entry point was invoked.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_RC_MARKER = "# Added by Trileaf installer"
_RC_FILES = (".zprofile", ".zshrc", ".bashrc", ".bash_profile", ".profile")
_INSTALL_META_NAME = "install.json"
_MANAGED_MODEL_DIR_NAMES = {
    "desklib-ai-text-detector-v1.01",
    "sentence-transformers-paraphrase-mpnet-base-v2",
}


# ── command handlers ──────────────────────────────────────────────────────────

def _cmd_run(args: argparse.Namespace) -> None:
    """Start the dashboard server.

    If the environment check fails on the first attempt, runs setup once as
    an automatic fallback, then retries.  The retry is not repeated — if it
    still fails, the error propagates normally.
    """
    import run as _run

    argv: list[str] = []
    if getattr(args, "leafhub_alias", None):
        argv += ["--leafhub-alias", args.leafhub_alias]
    if args.reload:
        argv.append("--reload")

    try:
        _run.main(argv)
    except SystemExit as exc:
        if exc.code == 0:
            raise  # clean exit — propagate as-is
        # Environment check returned a non-zero exit — run setup once as a
        # fallback, then retry.  _cmd_onboard always raises SystemExit(0) on
        # completion; catch it so execution can continue to the retry.
        print(
            "\n[run] Environment check failed — running setup as fallback (once only) ...",
            file=sys.stderr,
        )
        try:
            _cmd_onboard(argparse.Namespace())
        except SystemExit:
            pass
        print("[run] Retrying startup after setup ...\n", file=sys.stderr)
        _run.main(argv)  # if this still fails, the error propagates


def _get_default_alias() -> str:
    """Read the primary alias from leafhub.toml, fall back to 'rewrite'."""
    try:
        from leafhub_sdk.manifest import get_default_alias
        return get_default_alias(project_dir=_ROOT, fallback="rewrite")
    except ImportError:
        pass
    return "rewrite"

_LEAFHUB_ALIAS = _get_default_alias()


def _ensure_leafhub_pip() -> None:
    """Install the leafhub pip package into the running venv if it is absent.

    Uses sys.executable so the install always targets the same venv that is
    running trileaf — no risk of installing into the wrong interpreter.
    """
    try:
        import leafhub  # noqa: F401
        return  # already present
    except ImportError:
        pass

    print("[setup] leafhub pip package not found — installing into current venv ...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", f"{_ROOT}[leafhub]"],
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        print(
            f"[!] leafhub install failed:\n{result.stderr.strip()}",
            file=sys.stderr,
        )
        print(
            "    Manual fix:  pip install -e '.[leafhub]'  (from Trileaf directory)",
            file=sys.stderr,
        )
    else:
        print("[setup] leafhub installed.")


def _read_dotfile() -> dict:
    """Read .leafhub and return its parsed content, or {} on any error."""
    import json as _json
    try:
        return _json.loads((_ROOT / ".leafhub").read_text(encoding="utf-8"))
    except Exception:
        return {}


def _ensure_leafhub_binding() -> bool:
    """Verify the 'rewrite' alias is bound in LeafHub; auto-repair if missing.

    Token-first: the project name is read from .leafhub rather than hardcoded,
    so renaming the project in LeafHub does not break this function.

    Returns True when credentials are fully resolved, False otherwise.

    States handled:
      .leafhub absent              → not registered; return False.
      token valid, alias bound     → return True immediately.
      token valid, alias missing   → auto-bind to first available provider.
      token invalid / project gone → project was deleted; print re-register guide.
      no providers in vault        → print guidance; return False.
      leafhub binary absent        → print guidance; return False.
    """
    import shutil

    _leafhub_token = _ROOT / ".leafhub"
    if not _leafhub_token.exists():
        return False

    # ── Read project name from dotfile (token-first; never hardcoded) ─────────
    dotfile = _read_dotfile()
    project_name = dotfile.get("project") or "trileaf"

    leafhub_bin = shutil.which("leafhub")
    if not leafhub_bin:
        print(
            "[setup] LeafHub binary not found — cannot verify binding.\n"
            "    Install LeafHub first, then re-run trileaf setup.",
            file=sys.stderr,
        )
        return False

    # ── Fast path: try full credential resolution ─────────────────────────────
    try:
        from scripts.rewrite_config import _try_leafhub  # type: ignore[attr-defined]
        if _try_leafhub():
            return True
    except Exception:
        pass

    # ── Check project health via `leafhub project show <name>` ────────────────
    show_result = subprocess.run(
        [leafhub_bin, "project", "show", project_name],
        capture_output=True, text=True,
    )

    if show_result.returncode != 0 or "not found" in show_result.stdout.lower():
        # Project was deleted from LeafHub vault but .leafhub file still exists.
        print(
            f"[!] LeafHub: project '{project_name}' not found in vault.\n"
            f"    The .leafhub token is stale — re-register to create a fresh link:\n"
            f"      leafhub register {project_name} --path {_ROOT} --alias {_LEAFHUB_ALIAS}\n"
            f"    Or run the full setup:  ./setup.sh",
            file=sys.stderr,
        )
        return False

    if _LEAFHUB_ALIAS in show_result.stdout:
        # Binding exists — credentials failed for a different reason (token mismatch, etc.)
        return False

    # ── Alias missing — attempt auto-bind ─────────────────────────────────────
    print(f"[setup] alias '{_LEAFHUB_ALIAS}' is not bound — attempting auto-bind ...")

    prov_result = subprocess.run(
        [leafhub_bin, "provider", "list", "--json"],
        capture_output=True, text=True,
    )
    provider_name: str | None = None
    try:
        import json as _json
        providers_data = _json.loads(prov_result.stdout)
        if providers_data:
            provider_name = providers_data[0].get("label")
    except Exception:
        pass

    if not provider_name:
        print(
            f"[!] No providers in LeafHub vault — cannot auto-bind.\n"
            f"    Add one:  leafhub manage  (Web UI at http://localhost:8765)\n"
            f"    Then:     leafhub project bind {project_name}"
            f" --alias {_LEAFHUB_ALIAS} --provider <name>",
            file=sys.stderr,
        )
        return False

    bind_result = subprocess.run(
        [leafhub_bin, "project", "bind", project_name,
         "--alias", _LEAFHUB_ALIAS, "--provider", provider_name],
        capture_output=True, text=True,
    )
    if bind_result.returncode == 0:
        print(f"[setup] {bind_result.stdout.strip()}")
        return True

    print(
        f"[!] Auto-bind failed: {bind_result.stderr.strip()}\n"
        f"    Manual fix: leafhub project bind {project_name}"
        f" --alias {_LEAFHUB_ALIAS} --provider {provider_name}",
        file=sys.stderr,
    )
    return False


def _cmd_onboard(args: argparse.Namespace) -> None:
    """Ensure pip dependencies, detection models, and LeafHub binding are all in place.

    Steps:
      1. Install leafhub pip package if missing (uses current venv's Python).
      2. Run environment check (informational only — always proceed regardless of result).
      3. Download desklib and mpnet models to <project_root>/models/.
      4. Verify 'rewrite' alias is bound in LeafHub; auto-bind to first available
         provider if missing.
    """
    # ── 1. pip dependencies ───────────────────────────────────────────────────
    _ensure_leafhub_pip()

    # ── 2. environment check (informational) ──────────────────────────────────
    from scripts import check_env

    print("\nChecking environment ...")
    try:
        check_env.main()
    except SystemExit:
        pass  # env check during setup is informational only — always proceed to downloads

    # ── 3. model downloads ────────────────────────────────────────────────────
    # The download scripts use argparse on sys.argv.  Clear sys.argv to just
    # the script name so the trileaf subcommand ("setup", "--yes", etc.) is not
    # misinterpreted by the download argparse.
    from scripts.download_scripts import (
        desklib_detector_download,
        mpnet_download,
    )
    _saved_argv = sys.argv[:]
    sys.argv = sys.argv[:1]
    try:
        desklib_detector_download.main()
        mpnet_download.main()
    finally:
        sys.argv = _saved_argv

    # ── 4. LeafHub binding verification + auto-repair ─────────────────────────
    _leafhub_token = _ROOT / ".leafhub"
    if not _leafhub_token.exists():
        print(
            "\n[setup] Models downloaded.  LeafHub is not linked yet.\n"
            "    Run the full setup:  ./setup.sh\n"
            "    Or link manually:   trileaf config\n"
        )
    elif _ensure_leafhub_binding():
        print("\n[setup] Done — models downloaded, LeafHub linked and provider bound.")
    else:
        print(
            "\n[setup] Models downloaded.  Credentials still need attention.\n"
            "    Check:   leafhub project show trileaf\n"
            "    Verify:  trileaf doctor\n"
        )

    raise SystemExit(0)


def _cmd_config(_args: argparse.Namespace) -> None:
    """Show LeafHub registration status for this project."""
    import subprocess, shutil

    leafhub = shutil.which("leafhub")
    if not leafhub:
        print("LeafHub is not installed.")
        print("  Install: https://github.com/Rebas9512/Leafhub")
        raise SystemExit(1)

    print("\nLeafHub status:")
    subprocess.run([leafhub, "status"], check=False)
    print("\nProject binding:")
    subprocess.run([leafhub, "project", "show", "trileaf"], check=False)


def _cmd_doctor(_args: argparse.Namespace) -> None:
    """Run an environment and model health check."""
    from scripts import check_env

    try:
        check_env.main()
    except SystemExit:
        raise
    raise SystemExit(0)


def _cmd_stop(_args: argparse.Namespace) -> None:
    """Stop a running Trileaf server and release GPU memory."""
    import socket

    pid_file = Path.home() / ".trileaf" / "run.pid"

    if not pid_file.exists():
        print("No running Trileaf instance found (no PID file at ~/.trileaf/run.pid).")
        print("If the server is still running, stop it with:  kill <PID>")
        raise SystemExit(0)

    try:
        pid = int(pid_file.read_text().strip())
    except (ValueError, OSError):
        print("Corrupt PID file — removing it.")
        pid_file.unlink(missing_ok=True)
        raise SystemExit(1)

    # Check if the process is still alive before sending a signal.
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        print(f"Process {pid} is no longer running. Removing stale PID file.")
        pid_file.unlink(missing_ok=True)
        raise SystemExit(0)

    print(f"Sending SIGTERM to Trileaf (PID {pid})...")
    try:
        os.kill(pid, signal.SIGTERM)
    except PermissionError:
        print(f"Permission denied — cannot signal PID {pid}.")
        raise SystemExit(1)

    # Determine the dashboard port so we can wait for it to be released.
    port = 8001
    try:
        from scripts import app_config
        dash = app_config.get_dashboard_config()
        port = int(os.getenv("DASHBOARD_PORT") or dash.get("port", 8001))
    except Exception:
        pass

    # Poll until the port is free (server fully stopped) or timeout.
    print(f"Waiting for port {port} to be released...", end="", flush=True)
    deadline = time.time() + 12.0
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                break
        time.sleep(0.3)
        print(".", end="", flush=True)
    else:
        print(f"\nWarning: port {port} still in use after 12 s — server may not have stopped cleanly.")
        raise SystemExit(1)

    print("\nTrileaf stopped. GPU memory has been released.")
    pid_file.unlink(missing_ok=True)


def _project_root() -> Path:
    return _ROOT


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _read_pid_file(pid_file: Path) -> int | None:
    if not pid_file.exists():
        return None
    try:
        return int(pid_file.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _load_install_metadata(config_dir: Path) -> dict[str, object] | None:
    meta_path = config_dir / _INSTALL_META_NAME
    if not meta_path.exists():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _resolve_install_dir(config_dir: Path) -> Path | None:
    payload = _load_install_metadata(config_dir)
    if not payload:
        return None
    raw = payload.get("install_dir")
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return Path(raw).expanduser().resolve()
    except OSError:
        return Path(raw).expanduser()


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_pid(pid: int, timeout_s: float = 12.0) -> None:
    os.kill(pid, signal.SIGTERM)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not _pid_exists(pid):
            return
        time.sleep(0.25)
    raise TimeoutError(f"process {pid} did not exit within {timeout_s:.0f}s")


def _find_installer_rcs(home: Path) -> list[Path]:
    dirty: list[Path] = []
    for name in _RC_FILES:
        rc = home / name
        if not rc.exists():
            continue
        try:
            src = rc.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if _RC_MARKER in src:
            dirty.append(rc)
    return dirty


def _load_windows_user_path() -> str:
    if os.name != "nt":
        return ""
    try:
        import winreg  # type: ignore[import]

        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Environment",
            0,
            winreg.KEY_READ,
        )
        try:
            value, _ = winreg.QueryValueEx(key, "Path")
        finally:
            winreg.CloseKey(key)
        return str(value or "")
    except OSError:
        return ""


def _resolve_from_project(project_root: Path, raw_path: object) -> Path:
    path = Path(str(raw_path))
    return path if path.is_absolute() else (project_root / path).resolve()


def _is_safe_managed_model_dir(path: Path, project_root: Path) -> bool:
    resolved = path.resolve()
    if resolved in {project_root.resolve(), Path.home(), Path("/")}:
        return False
    if _path_is_within(resolved, project_root):
        return True
    return resolved.name in _MANAGED_MODEL_DIR_NAMES


def _collect_managed_model_dirs(project_root: Path) -> list[Path]:
    paths: list[Path] = []
    try:
        from scripts import app_config

        paths.append(app_config.resolve_model_path("desklib"))
        paths.append(app_config.resolve_model_path("mpnet"))
    except Exception:
        paths.extend(
            [
                project_root / "models" / "desklib-ai-text-detector-v1.01",
                project_root / "models" / "sentence-transformers-paraphrase-mpnet-base-v2",
            ]
        )

    out: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        key = str(resolved)
        if key in seen or not resolved.exists():
            continue
        if not _is_safe_managed_model_dir(resolved, project_root):
            continue
        seen.add(key)
        out.append(resolved)
    return out


def _collect_generated_project_paths(project_root: Path) -> list[Path]:
    candidates = [
        project_root / ".venv",
        project_root / "build",
        project_root / ".pytest_cache",
        project_root / "trileaf.egg-info",
        project_root / ".env",
        project_root / "__pycache__",
        project_root / "api" / "__pycache__",
        project_root / "scripts" / "__pycache__",
        project_root / "scripts" / "download_scripts" / "__pycache__",
        project_root / "tests" / "__pycache__",
    ]
    out: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen or not path.exists():
            continue
        seen.add(key)
        out.append(path)
    return out


def _schedule_delete_after_exit(target: Path) -> None:
    if os.name == "nt":
        script = Path(tempfile.gettempdir()) / f"trileaf-remove-{os.getpid()}.cmd"
        script.write_text(
            "@echo off\r\n"
            f'set "TARGET={target}"\r\n'
            ":retry\r\n"
            "timeout /t 2 /nobreak >nul\r\n"
            'rmdir /s /q "%TARGET%" >nul 2>&1\r\n'
            'if exist "%TARGET%" goto retry\r\n'
            'del "%~f0" >nul 2>&1\r\n',
            encoding="utf-8",
        )
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.Popen(
            ["cmd", "/c", str(script)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        return

    subprocess.Popen(
        ["/bin/sh", "-c", f"sleep 1; rm -rf -- {shlex.quote(str(target))}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _should_remove_unix_link(link: Path, config_dir: Path, project_root: Path) -> bool:
    if not (link.exists() or link.is_symlink()):
        return False
    if not link.is_symlink():
        return False
    try:
        target = link.resolve(strict=False)
    except OSError:
        return True
    return _path_is_within(target, config_dir) or _path_is_within(target, project_root)


def _cmd_weight(args: argparse.Namespace) -> None:
    """Show or update Pareto utility weights."""
    from scripts import app_config

    config = app_config.load_config()
    pipeline = config.setdefault("pipeline", {})

    # Fill defaults for any missing weight keys
    defaults = app_config._DEFAULTS["pipeline"]
    for key in ("w_ai", "w_sem", "w_risk"):
        pipeline.setdefault(key, defaults[key])

    # No args → print current weights
    if args.ai is None and args.sem is None and args.risk is None:
        print(f"Current Pareto weights:")
        print(f"  --ai   (AI reduction)  {pipeline['w_ai']:.2f}")
        print(f"  --sem  (Semantic)      {pipeline['w_sem']:.2f}")
        print(f"  --risk (Risk penalty)  {pipeline['w_risk']:.2f}")
        print(f"  Sum: {pipeline['w_ai'] + pipeline['w_sem'] + pipeline['w_risk']:.2f}")
        print(f"\nTo update: trileaf weight --ai 0.60 --sem 0.35 --risk 0.05")
        raise SystemExit(0)

    # Apply any provided values, keep existing for omitted ones
    new_ai   = args.ai   if args.ai   is not None else pipeline["w_ai"]
    new_sem  = args.sem  if args.sem  is not None else pipeline["w_sem"]
    new_risk = args.risk if args.risk is not None else pipeline["w_risk"]

    total = round(new_ai + new_sem + new_risk, 6)
    if abs(total - 1.0) > 0.001:
        print(f"Error: weights must sum to 1.0 (got {total:.4f}).")
        print(f"  --ai {new_ai:.2f}  --sem {new_sem:.2f}  --risk {new_risk:.2f}")
        raise SystemExit(1)

    pipeline["w_ai"]   = new_ai
    pipeline["w_sem"]  = new_sem
    pipeline["w_risk"] = new_risk
    config["pipeline"] = pipeline
    app_config.save_config(config)

    print(f"Weights updated:")
    print(f"  AI reduction : {new_ai:.2f}")
    print(f"  Semantic     : {new_sem:.2f}")
    print(f"  Risk penalty : {new_risk:.2f}")


def _find_venv_python(project_root: Path) -> str:
    """Return the Python executable inside the project venv, falling back to sys.executable."""
    for candidate in (
        project_root / ".venv" / "bin" / "python",
        project_root / ".venv" / "Scripts" / "python.exe",
    ):
        if candidate.exists():
            return str(candidate)
    return sys.executable


def _cmd_update(args: argparse.Namespace) -> None:
    """Pull the latest changes from the git remote and refresh Python packages."""
    import shutil

    project_root = _project_root()

    if not (project_root / ".git").exists():
        print(f"Error: project directory is not a git repository.")
        print(f"  {project_root}")
        raise SystemExit(1)

    git = shutil.which("git")
    if not git:
        print("Error: git is not available on PATH.")
        raise SystemExit(1)

    # Warn if server is still running
    pid_file = Path.home() / ".trileaf" / "run.pid"
    pid = _read_pid_file(pid_file)
    if pid and _pid_exists(pid):
        print(f"Warning: Trileaf server is running (PID {pid}).")
        print("  Stop it before updating to avoid conflicts:")
        print("    trileaf stop")
        if not args.yes:
            try:
                answer = input("Continue anyway? [y/N] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nAborted.")
                raise SystemExit(1)
            if answer not in ("y", "yes"):
                print("Aborted.")
                raise SystemExit(0)

    # Snapshot current commit
    res = subprocess.run(
        [git, "rev-parse", "--short", "HEAD"],
        cwd=project_root, capture_output=True, text=True,
    )
    old_sha = res.stdout.strip() if res.returncode == 0 else "unknown"

    print("Pulling latest changes…")
    ret = subprocess.run([git, "pull"], cwd=project_root)
    if ret.returncode != 0:
        print("Error: git pull failed.")
        raise SystemExit(ret.returncode)

    res = subprocess.run(
        [git, "rev-parse", "--short", "HEAD"],
        cwd=project_root, capture_output=True, text=True,
    )
    new_sha = res.stdout.strip() if res.returncode == 0 else "unknown"

    if old_sha == new_sha:
        print("Already up to date.")
        raise SystemExit(0)

    # Refresh Python packages
    python = _find_venv_python(project_root)
    req = project_root / "requirements.txt"
    print("Updating Python packages…")
    if req.exists():
        ret = subprocess.run(
            [python, "-m", "pip", "install", "-r", str(req), "-q"],
            cwd=project_root,
        )
    else:
        ret = subprocess.run(
            [python, "-m", "pip", "install", "-e", ".", "-q"],
            cwd=project_root,
        )
    if ret.returncode != 0:
        print("Error: package update failed — code was updated but dependencies may be incomplete.")
        print("  Run manually:  pip install -r requirements.txt")
        print(f"\nTrileaf code updated:  {old_sha} → {new_sha}  (packages incomplete)")
        raise SystemExit(ret.returncode)

    print(f"\nTrileaf updated:  {old_sha} → {new_sha}")
    print("Restart the server to apply changes:  trileaf run")


def _cmd_remove(args: argparse.Namespace) -> None:
    """Remove Trileaf and its generated files."""
    import shutil

    project_root = _project_root()
    home = Path.home()
    config_dir = home / ".trileaf"
    legacy_config_dir = home / ".llm-writing-optimizer"
    pid_file = config_dir / "run.pid"
    recorded_install_dir = _resolve_install_dir(config_dir)

    this_file = Path(__file__).resolve()
    legacy_one_liner = config_dir in this_file.parents
    managed_one_liner = (
        recorded_install_dir is not None
        and recorded_install_dir == project_root.resolve()
    )
    is_one_liner = legacy_one_liner or managed_one_liner

    plan: list[tuple[str, object, str]] = []
    deleted_roots: list[Path] = []

    pid = _read_pid_file(pid_file)
    if pid is None:
        if pid_file.exists():
            plan.append(("stale PID file", pid_file, "unlink"))
    elif _pid_exists(pid):
        plan.append(("running Trileaf process", pid, "stop_process"))
    else:
        plan.append(("stale PID file", pid_file, "unlink"))

    if os.name != "nt":
        link = home / ".local" / "bin" / "trileaf"
        if _should_remove_unix_link(link, config_dir, project_root):
            plan.append(("symlink", link, "unlink"))
        dirty_rcs = _find_installer_rcs(home)
        if dirty_rcs:
            plan.append(("PATH entries in rc files", dirty_rcs, "clean_rc"))
    else:
        user_path = _load_windows_user_path()
        for scripts_dir in (
            config_dir / ".venv" / "Scripts",
            project_root / ".venv" / "Scripts",
        ):
            if not scripts_dir.exists():
                continue
            if str(scripts_dir).lower() in user_path.lower():
                plan.append(("PATH registry entry", str(scripts_dir), "win_path"))

    if is_one_liner:
        if managed_one_liner and recorded_install_dir and recorded_install_dir.exists():
            plan.append(("install directory", recorded_install_dir, "self_delete"))
            deleted_roots.append(recorded_install_dir)
            if config_dir.exists():
                plan.append(("config directory (~/.trileaf/)", config_dir, "rmtree"))
        elif config_dir.exists():
            plan.append(("install + config directory", config_dir, "self_delete"))
            deleted_roots.append(config_dir)
    else:
        if config_dir.exists():
            plan.append(("config directory (~/.trileaf/)", config_dir, "rmtree"))

        if args.purge_source and project_root.exists():
            plan.append(("source checkout", project_root, "self_delete"))
            deleted_roots.append(project_root)
        else:
            for path in _collect_generated_project_paths(project_root):
                plan.append(("generated project file", path, "remove_path"))

    if legacy_config_dir.exists():
        plan.append(("legacy config directory", legacy_config_dir, "rmtree"))

    for model_dir in _collect_managed_model_dirs(project_root):
        if any(_path_is_within(model_dir, root) for root in deleted_roots):
            continue
        plan.append(("managed model directory", model_dir, "rmtree"))

    seen_steps: set[tuple[str, str]] = set()
    deduped_plan: list[tuple[str, object, str]] = []
    for label, target, action in plan:
        if isinstance(target, list):
            key = (label, "|".join(str(t) for t in target))
        else:
            key = (label, str(target))
        if key in seen_steps:
            continue
        seen_steps.add(key)
        deduped_plan.append((label, target, action))
    plan = deduped_plan

    if not plan:
        print("Nothing to remove.")
        raise SystemExit(0)

    # ── Confirm ───────────────────────────────────────────────────────────────
    print("\nThe following will be permanently removed:")
    for label, target, _ in plan:
        if isinstance(target, list):
            for t in target:
                print(f"  {label}: {t}")
        else:
            print(f"  {label}: {target}")

    if not args.yes:
        try:
            answer = input("\nProceed? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            raise SystemExit(1)
        if answer not in ("y", "yes"):
            print("Aborted.")
            raise SystemExit(0)

    print()

    # ── Execute ───────────────────────────────────────────────────────────────
    for label, target, action in plan:
        try:
            if action == "stop_process":
                _terminate_pid(int(target))
                pid_file.unlink(missing_ok=True)
                print(f"  stopped  PID {target}")

            elif action == "unlink":
                target.unlink(missing_ok=True)
                print(f"  removed  {target}")

            elif action == "clean_rc":
                for rc in target:
                    _clean_installer_block(rc)
                    print(f"  cleaned  {rc}")

            elif action == "win_path":
                import winreg  # type: ignore[import]
                key = winreg.OpenKey(
                    winreg.HKEY_CURRENT_USER,
                    r"Environment",
                    0, winreg.KEY_READ | winreg.KEY_WRITE,
                )
                cur, _ = winreg.QueryValueEx(key, "Path")
                new_val = ";".join(
                    p for p in cur.split(";")
                    if p.strip().lower() != target.lower()
                )
                winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, new_val)
                winreg.CloseKey(key)
                print(f"  removed  {target} from user PATH")

            elif action == "rmtree":
                shutil.rmtree(target)
                print(f"  removed  {target}")

            elif action == "remove_path":
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink(missing_ok=True)
                print(f"  removed  {target}")

            elif action == "self_delete":
                _schedule_delete_after_exit(target)
                print(f"  scheduled removal  {target}")

        except Exception as exc:
            print(f"  warning: could not remove {target}: {exc}")

    print()
    if is_one_liner:
        print("Trileaf removed.")
    elif args.purge_source:
        print("Trileaf source checkout scheduled for removal.")
    else:
        print("Trileaf generated files removed.")
        print(f"Source checkout kept:  {project_root}")
        print("Re-run with --purge-source to delete the checkout as well.")


def _clean_installer_block(rc: Path) -> None:
    """Strip the PATH block that install.sh wrote into a shell rc file."""
    lines = rc.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
    out: list[str] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].rstrip("\n\r")
        if stripped == _RC_MARKER:
            # Remove the marker line and the export PATH line that follows it.
            i += 1  # skip marker
            if i < len(lines) and lines[i].startswith("export PATH="):
                i += 1  # skip PATH line
            # Also drop the preceding blank line if we added it.
            if out and out[-1].strip() == "":
                out.pop()
        else:
            out.append(lines[i])
            i += 1
    rc.write_text("".join(out), encoding="utf-8")


# ── CLI definition ────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="trileaf",
        description="Trileaf — AI writing humaniser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  trileaf setup              # first-time setup (models + provider)\n"
            "  trileaf run                # start the dashboard\n"
            "  trileaf stop               # stop the server and release GPU memory\n"
            "  trileaf config             # configure rewrite provider (.env)\n"
            "  trileaf config show        # show current .env config\n"
            "  trileaf weight             # show current Pareto weights\n"
            "  trileaf weight --ai 0.5 --sem 0.45 --risk 0.05  # update weights\n"
            "  trileaf update             # pull latest version and refresh packages\n"
            "  trileaf doctor             # environment health check\n"
            "  trileaf remove             # uninstall Trileaf and all derived files\n"
        ),
    )

    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # trileaf run
    p_run = sub.add_parser("run", help="Start the Trileaf dashboard server")
    p_run.add_argument(
        "--leafhub-alias", default="", metavar="ALIAS",
        help="LeafHub alias to use for the rewrite API key (overrides LEAFHUB_ALIAS in .env)",
    )
    p_run.add_argument(
        "--reload", action="store_true",
        help="Enable uvicorn auto-reload (development mode)",
    )

    # trileaf setup  — download detection models
    p_setup = sub.add_parser(
        "setup", help="Download detection models (desklib + mpnet, ~0.9 GB)"
    )
    p_setup.add_argument(
        "-y", "--yes", action="store_true",
        help="Non-interactive: skip confirmation prompts",
    )
    p_setup.add_argument(
        "--models-only", action="store_true", dest="models_only",
        help="Internal flag used by install scripts",
    )

    # trileaf stop
    sub.add_parser("stop", help="Stop a running Trileaf server and release GPU memory")

    # trileaf config — show LeafHub registration status
    p_cfg = sub.add_parser("config", help="Show LeafHub registration status for this project")
    p_cfg.add_argument(
        "config_sub", nargs="?", default="show",
        choices=["show"],
        help="wizard (default) | show current config | clear rewrite keys",
    )

    # trileaf doctor
    sub.add_parser("doctor", help="Check that all models and config are in place")

    # trileaf weight
    p_weight = sub.add_parser(
        "weight", help="Show or update Pareto utility weights"
    )
    p_weight.add_argument(
        "--ai", type=float, default=None, metavar="W",
        help="Weight for AI-score reduction (e.g. 0.60)",
    )
    p_weight.add_argument(
        "--sem", type=float, default=None, metavar="W",
        help="Weight for semantic similarity (e.g. 0.35)",
    )
    p_weight.add_argument(
        "--risk", type=float, default=None, metavar="W",
        help="Weight for risk penalty (e.g. 0.05)",
    )

    # trileaf update
    p_update = sub.add_parser(
        "update",
        help="Pull the latest version from git and refresh Python packages",
    )
    p_update.add_argument(
        "-y", "--yes", action="store_true",
        help="Skip the running-server warning prompt",
    )

    # trileaf remove
    p_remove = sub.add_parser(
        "remove",
        help="Remove Trileaf, its generated files, and installer side effects",
    )
    p_remove.add_argument(
        "-y", "--yes", action="store_true",
        help="Skip the confirmation prompt",
    )
    p_remove.add_argument(
        "--purge-source", action="store_true",
        help="Also delete the current source checkout after cleanup",
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        raise SystemExit(0)

    dispatch = {
        "run":     _cmd_run,
        "stop":    _cmd_stop,
        "setup":   _cmd_onboard,
        "config":  _cmd_config,
        "weight":  _cmd_weight,
        "update":  _cmd_update,
        "doctor":  _cmd_doctor,
        "remove":  _cmd_remove,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
