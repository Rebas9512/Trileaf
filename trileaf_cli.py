#!/usr/bin/env python3
"""
trileaf — command-line interface for the Trileaf writing optimiser.

Usage
-----
  trileaf run              Start the dashboard server
  trileaf stop             Stop the server and release GPU memory
  trileaf onboard          First-time setup: download models + configure provider
  trileaf config           Add or edit rewrite provider profiles
  trileaf doctor           Check that all models and configuration are in place

Run  trileaf <command> --help  for per-command options.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `run` and `scripts.*` are importable
# regardless of how the console-scripts entry point was invoked.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ── command handlers ──────────────────────────────────────────────────────────

def _cmd_run(args: argparse.Namespace) -> None:
    """Start the dashboard server."""
    import run as _run

    argv: list[str] = []
    if args.profile:
        argv += ["--rewrite-profile", args.profile]
    if args.reload:
        argv.append("--reload")
    _run.main(argv)


def _cmd_onboard(args: argparse.Namespace) -> None:
    """Run the first-time onboarding wizard."""
    from scripts import onboarding

    argv: list[str] = []
    if args.yes:
        argv.append("--yes")
    raise SystemExit(onboarding.main(argv))


def _cmd_config(_args: argparse.Namespace) -> None:
    """Open the interactive provider configuration wizard."""
    from scripts import rewrite_provider_cli

    raise SystemExit(rewrite_provider_cli.main(["wizard"]))


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
    import signal
    import socket
    import time

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


# ── CLI definition ────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="trileaf",
        description="Trileaf — AI writing humaniser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  trileaf onboard            # first-time setup\n"
            "  trileaf run                # start the dashboard\n"
            "  trileaf run --profile gpt  # use a specific provider profile\n"
            "  trileaf stop               # stop the server and release GPU memory\n"
            "  trileaf config             # manage provider profiles\n"
            "  trileaf doctor             # environment health check\n"
        ),
    )

    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # trileaf run
    p_run = sub.add_parser("run", help="Start the Trileaf dashboard server")
    p_run.add_argument(
        "--profile", default="", metavar="NAME",
        help="Use a specific rewrite provider profile for this session",
    )
    p_run.add_argument(
        "--reload", action="store_true",
        help="Enable uvicorn auto-reload (development mode)",
    )

    # trileaf onboard
    p_onboard = sub.add_parser(
        "onboard", help="First-time setup: download models and configure a provider"
    )
    p_onboard.add_argument(
        "-y", "--yes", action="store_true",
        help="Non-interactive mode: accept all defaults",
    )

    # trileaf stop
    sub.add_parser("stop", help="Stop a running Trileaf server and release GPU memory")

    # trileaf config
    sub.add_parser("config", help="Add or edit rewrite provider profiles")

    # trileaf doctor
    sub.add_parser("doctor", help="Check that all models and config are in place")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        raise SystemExit(0)

    dispatch = {
        "run":     _cmd_run,
        "stop":    _cmd_stop,
        "onboard": _cmd_onboard,
        "config":  _cmd_config,
        "doctor":  _cmd_doctor,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
