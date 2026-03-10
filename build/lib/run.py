#!/usr/bin/env python3
import argparse
import gc
import os
import signal
import sys
import time
import webbrowser
import subprocess
from contextlib import suppress
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

# ── PID file (used by `trileaf stop`) ────────────────────────────────────────
_PID_FILE = Path.home() / ".trileaf" / "run.pid"


def _write_pid_file() -> None:
    with suppress(OSError):
        _PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        _PID_FILE.write_text(str(os.getpid()))


def _remove_pid_file() -> None:
    with suppress(OSError):
        _PID_FILE.unlink(missing_ok=True)


def _graceful_shutdown(signum, frame) -> None:
    """Signal handler: convert SIGHUP / SIGTERM into KeyboardInterrupt so the
    existing try/finally cleanup path in main() runs."""
    raise KeyboardInterrupt

def run_env_check():
    try:
        from scripts import check_env
        check_env.main()
    except SystemExit:
        raise
    except Exception:
        print("[run] Env check skipped (error).")


def set_device_hint() -> None:
    """Expose a lightweight device hint for /api/health without forcing backend imports."""
    try:
        import torch

        if torch.cuda.is_available():
            device = f"cuda ({torch.cuda.get_device_name(0)})"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps (Apple Silicon)"
        else:
            device = "cpu"
        os.environ["OPTIMIZER_DEVICE_HINT"] = device
    except Exception:
        os.environ.setdefault("OPTIMIZER_DEVICE_HINT", "unknown")


def clear_model_memory() -> None:
    with suppress(Exception):
        mr = sys.modules.get("scripts.models_runtime")
        if mr is not None:
            mr._DESKLIB_CACHE.clear()
            mr._MPNET_CACHE.clear()
            mr._REWRITE_CACHE.clear()

    gc.collect()
    with suppress(Exception):
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            with suppress(Exception):
                torch.cuda.ipc_collect()


def is_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        "out of memory" in message
        or "cuda oom" in message
        or "cuda out of memory" in message
    )


def preflight_models() -> tuple[list[dict], str | None]:
    print("[run] Clearing model memory before preflight...")
    clear_model_memory()

    try:
        import scripts.models_runtime as mr

        print("[run] Preloading models for status check...")
        results = mr.preload_models()
        return results, None
    except Exception as exc:
        if is_oom_error(exc):
            return [], f"OOM during model preload: {exc}"
        return [], f"Model preload failed: {exc}"
    finally:
        print("[run] Clearing model memory after preflight...")
        clear_model_memory()


def wait_for_backend(
    ready_url: str,
    proc: subprocess.Popen[bytes] | None = None,
    timeout_s: float = 600.0,
    poll_s: float = 1.0,
) -> None:
    deadline = time.time() + timeout_s
    last_err = None

    while time.time() < deadline:
        if proc is not None:
            code = proc.poll()
            if code is not None:
                raise RuntimeError(
                    f"Backend process exited before readiness check passed (exit code {code})."
                )
        try:
            with urlopen(ready_url, timeout=5) as resp:
                if 200 <= getattr(resp, "status", 200) < 300:
                    print("[run] Backend startup complete.")
                    return
        except (URLError, OSError) as exc:
            last_err = exc
        time.sleep(poll_s)

    raise TimeoutError(f"Backend did not become ready in {timeout_s:.0f}s: {last_err}")


def terminate_process_tree(proc: subprocess.Popen[bytes], grace_s: float = 5.0) -> None:
    if proc.poll() is not None:
        return

    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
        with suppress(Exception):
            proc.wait(timeout=2)
    except ProcessLookupError:
        pass


def _free_port(host: str, port: int) -> None:
    """If host:port is already bound, kill the holding process and wait for release."""
    import socket

    check_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if sock.connect_ex((check_host, int(port))) != 0:
            return  # Port is free

    print(f"[run] Port {port} is already in use — stopping existing process...")
    killed = False

    # Method 1: fuser (Linux/WSL) — fast, direct
    with suppress(Exception):
        r = subprocess.run(
            ["fuser", f"{port}/tcp"],
            capture_output=True, text=True, timeout=5,
        )
        for tok in r.stdout.split():
            if tok.strip().lstrip("-").isdigit():
                pid = int(tok.strip())
                with suppress(ProcessLookupError, PermissionError):
                    os.kill(pid, signal.SIGTERM)
                    killed = True
                    print(f"[run]   SIGTERM → PID {pid}")

    # Method 2: lsof (macOS / Linux fallback)
    if not killed:
        with suppress(Exception):
            r = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True, text=True, timeout=5,
            )
            for line in r.stdout.splitlines():
                if line.strip().isdigit():
                    pid = int(line.strip())
                    with suppress(ProcessLookupError, PermissionError):
                        os.kill(pid, signal.SIGTERM)
                        killed = True
                        print(f"[run]   SIGTERM → PID {pid}")

    if killed:
        time.sleep(1.5)  # Give the process time to release the port
        print(f"[run] Port {port} cleared.")
    else:
        print(f"[run] Warning: could not identify process on port {port}; startup may fail.")


def _bust_static_cache(root: Path) -> None:
    """Touch static assets so browsers revalidate instead of serving stale cache.

    FastAPI's StaticFiles derives ETags from file mtime.  Updating mtime on
    every server start ensures the browser's conditional GET gets a fresh ETag,
    causing a 200 (new content) rather than a stale 304.
    """
    static_dir = root / "api" / "static"
    if not static_dir.is_dir():
        return
    for path in static_dir.rglob("*"):
        if path.is_file() and path.suffix in {".html", ".js", ".css"}:
            with suppress(OSError):
                path.touch()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the Trileaf dashboard.")
    parser.add_argument(
        "--configure-rewrite",
        action="store_true",
        help="Open the guided rewrite-provider setup before starting the backend.",
    )
    parser.add_argument(
        "--configure-only",
        action="store_true",
        help="Open the guided rewrite-provider setup, save config, and exit.",
    )
    parser.add_argument(
        "--rewrite-profile",
        default="",
        help="Temporarily use a specific local rewrite profile for this run.",
    )
    parser.add_argument(
        "--onboard",
        action="store_true",
        help="Run the first-time setup wizard (model download + provider config), then exit.",
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Run an environment/model health check and exit (non-zero on failure).",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn auto-reload. Slower and less robust than the default single-process mode.",
    )
    return parser.parse_args(argv)


def _apply_rewrite_profile(profile_name: str = "") -> None:
    from scripts import rewrite_config

    selected = rewrite_config.load_selected_profile(profile_name or None)
    if selected is None:
        if profile_name:
            raise SystemExit(f"[run] Unknown rewrite profile: {profile_name}")
        return

    name = str(selected["name"])
    profile = selected["profile"]
    os.environ["REWRITE_PROFILE"] = name
    backend = str(profile.get("backend") or "local")
    print(f"[run] Rewrite profile: {name} ({backend})")


def _rewrite_backend_is_external() -> bool:
    """Return True if the active rewrite profile (or env) uses an external API."""
    from scripts import rewrite_config

    selected = rewrite_config.load_selected_profile()
    if isinstance(selected, dict):
        profile = selected.get("profile") or {}
        backend = str(profile.get("backend") or "").strip().lower()
        if backend in {"external", "openai_api"}:
            return True
    raw = rewrite_config.first_defined(
        os.getenv("REWRITE_BACKEND"),
        rewrite_config.legacy_env_first("backend"),
        "local",
    ) or "local"
    raw = raw.strip().lower()
    return raw in {"external", "openai_api"}


def main(argv: list[str] | None = None):
    from scripts import app_config as _app_config

    args = parse_args(argv)

    root = Path(__file__).resolve().parent
    _dash = _app_config.get_dashboard_config()
    host = os.getenv("DASHBOARD_HOST") or str(_dash.get("host", "127.0.0.1"))
    port = os.getenv("DASHBOARD_PORT") or str(_dash.get("port", 8001))
    app_target = os.getenv("DASHBOARD_APP", "api.optimizer_api:app")
    url = os.getenv("DASHBOARD_URL", f"http://{host}:{port}/static/index.html")
    ready_url = os.getenv("DASHBOARD_READY_URL", f"http://{host}:{port}/api/ready")
    reload_enabled = args.reload or os.getenv("DASHBOARD_RELOAD", "").lower() in {"1", "true", "yes", "on"}

    if args.onboard:
        from scripts import onboarding

        raise SystemExit(onboarding.main())

    if args.doctor:
        from scripts import check_env

        try:
            check_env.main()
        except SystemExit:
            raise
        raise SystemExit(0)

    if args.configure_rewrite or args.configure_only:
        from scripts import rewrite_provider_cli

        exit_code = rewrite_provider_cli.main(["wizard"])
        if exit_code:
            raise SystemExit(exit_code)

    _apply_rewrite_profile(args.rewrite_profile)

    if args.configure_only:
        return

    run_env_check()
    set_device_hint()

    if _rewrite_backend_is_external():
        # External API backend: no local rewrite model to check for OOM, and
        # Desklib / MPNet are small enough to load lazily on the first request.
        # Skipping preflight eliminates the transformers → sklearn → scipy cold
        # import (which can stall startup by 10-60 s on first run).
        print("[run] External rewrite backend — skipping model preflight (models load on first request).")
        os.environ["OPTIMIZER_SKIP_STARTUP_PRELOAD"] = "1"
        os.environ["OPTIMIZER_PREFLIGHT_ERROR"] = ""
    else:
        preflight_results, preflight_error = preflight_models()
        if preflight_results:
            for item in preflight_results:
                detail = f" ({item['detail']})" if item.get("detail") else ""
                print(
                    f"[run] Preflight {item['name']}: "
                    f"{item['status']} in {item['seconds']:.3f}s{detail}"
                )

        if preflight_error:
            print(f"[run] {preflight_error}")
            print("[run] Falling back to backend startup without startup preload.")
            os.environ["OPTIMIZER_SKIP_STARTUP_PRELOAD"] = "1"
            os.environ["OPTIMIZER_PREFLIGHT_ERROR"] = preflight_error
            clear_model_memory()
        else:
            print("[run] Preflight succeeded; backend startup preload will be skipped to avoid double-loading models.")
            os.environ["OPTIMIZER_SKIP_STARTUP_PRELOAD"] = "1"
            os.environ["OPTIMIZER_PREFLIGHT_ERROR"] = ""

        clear_model_memory()

    # Install signal handlers so terminal-close (SIGHUP) and explicit SIGTERM
    # both run the same cleanup path as Ctrl+C (KeyboardInterrupt → finally block).
    if os.name == "posix":
        signal.signal(signal.SIGHUP, _graceful_shutdown)
        signal.signal(signal.SIGTERM, _graceful_shutdown)

    _free_port(host, int(port))
    _bust_static_cache(root)

    cmd = [
        sys.executable, "-m", "uvicorn",
        app_target,
        "--host", host,
        "--port", str(port),
    ]
    if reload_enabled:
        cmd.append("--reload")

    print("[run] Starting backend:", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=root,
        env=os.environ.copy(),
        start_new_session=(os.name == "posix"),
    )
    _write_pid_file()

    try:
        print("[run] Waiting for application startup to complete...")
        wait_for_backend(ready_url, proc=proc)
        print("[run] Opening:", url)
        webbrowser.open(url)
        proc.wait()
    except KeyboardInterrupt:
        print("\n[run] Stopping...")
    except Exception as exc:
        print(f"[run] Startup failed: {exc}")
    finally:
        terminate_process_tree(proc)
        _remove_pid_file()


if __name__ == "__main__":
    main()
