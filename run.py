#!/usr/bin/env python3
import argparse
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
        "--leafhub-alias",
        default="",
        help="LeafHub alias to use for the rewrite API key (overrides LEAFHUB_ALIAS env var).",
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


def _load_rewrite_credentials() -> None:
    """
    Resolve rewrite provider credentials and inject into os.environ.

    Resolution order:
      1. LeafHub probe (.leafhub dotfile) — API key, optionally base_url/model
      2. Existing os.environ — REWRITE_API_KEY / provider-specific key fallbacks

    Must be called before models_runtime is imported.
    """
    from scripts import rewrite_config

    result = rewrite_config.resolve_credentials()
    cred = result.get("credential_source", "none")
    print(f"[run] Rewrite backend: external  credentials: {cred}")


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

    if args.doctor:
        from scripts import check_env

        try:
            check_env.main()
        except SystemExit:
            raise
        raise SystemExit(0)

    if args.leafhub_alias:
        os.environ["LEAFHUB_ALIAS"] = args.leafhub_alias
    _load_rewrite_credentials()

    run_env_check()
    set_device_hint()

    # External API backend — Desklib/MPNet load lazily on first request.
    # Skipping preflight avoids the transformers cold-import stall (10-60 s).
    print("[run] External rewrite backend — models load on first request.")
    os.environ["OPTIMIZER_SKIP_STARTUP_PRELOAD"] = "1"
    os.environ["OPTIMIZER_PREFLIGHT_ERROR"] = ""

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
