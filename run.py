#!/usr/bin/env python3
import gc
import os
import sys
import time
import webbrowser
import subprocess
from contextlib import suppress
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from dotenv import load_dotenv


def run_env_check():
    try:
        from scripts import check_env
        check_env.main()
    except SystemExit:
        raise
    except Exception:
        print("[run] Env check skipped (error).")


def clear_model_memory() -> None:
    with suppress(Exception):
        import scripts.models_runtime as mr

        mr._DESKLIB_CACHE.clear()
        mr._MPNET_CACHE.clear()
        mr._QWEN_CACHE.clear()

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


def wait_for_backend(health_url: str, timeout_s: float = 600.0, poll_s: float = 1.0) -> None:
    deadline = time.time() + timeout_s
    last_err = None

    while time.time() < deadline:
        try:
            with urlopen(health_url, timeout=5) as resp:
                if 200 <= getattr(resp, "status", 200) < 300:
                    print("[run] Backend startup complete.")
                    return
        except (URLError, OSError) as exc:
            last_err = exc
        time.sleep(poll_s)

    raise TimeoutError(f"Backend did not become ready in {timeout_s:.0f}s: {last_err}")


def main():
    load_dotenv()

    root = Path(__file__).resolve().parent
    host = os.getenv("DASHBOARD_HOST", "127.0.0.1")
    port = os.getenv("DASHBOARD_PORT", "8001")
    app_target = os.getenv("DASHBOARD_APP", "api.optimizer_api:app")
    url = os.getenv("DASHBOARD_URL", f"http://{host}:{port}/static/index.html")
    health_url = os.getenv("DASHBOARD_HEALTH_URL", f"http://{host}:{port}/api/health")

    run_env_check()

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

    cmd = [
        sys.executable, "-m", "uvicorn",
        app_target,
        "--host", host,
        "--port", str(port),
        "--reload",
    ]

    print("[run] Starting backend:", " ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=root, env=os.environ.copy())

    try:
        print("[run] Waiting for application startup to complete...")
        wait_for_backend(health_url)
        print("[run] Opening:", url)
        webbrowser.open(url)
        proc.wait()
    except KeyboardInterrupt:
        print("\n[run] Stopping...")
    except Exception as exc:
        print(f"[run] Startup failed: {exc}")
    finally:
        if proc.poll() is None:
            proc.terminate()


if __name__ == "__main__":
    main()
