#!/usr/bin/env python3
"""
Guided first-time setup for Trileaf.

Steps
-----
1. Python environment  — torch, sentence-transformers, huggingface_hub
2. Detection models    — Desklib + MPNet (required; auto-download offered)
3. Rewrite provider    — External API wizard  OR  local Qwen3-VL-8B download
4. Final validation    — re-runs check_env to confirm everything is ready

Run:
    python scripts/onboarding.py
or:
    trileaf onboard

Non-interactive / CI:
    python scripts/onboarding.py --yes   # accept all download prompts automatically
"""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

# ── project root on sys.path ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import app_config as _app_config

# ── constants ────────────────────────────────────────────────────────────────
_DIVIDER = "─" * 62

# Module-level flag set by parse_args(); checked by _prompt_yn.
_YES_ALL: bool = False


# ── tiny UI helpers ───────────────────────────────────────────────────────────

def _header(title: str) -> None:
    print()
    print(_DIVIDER)
    print(f"  {title}")
    print(_DIVIDER)


def _prompt_yn(question: str, default: bool = True) -> bool:
    """Interactive yes/no prompt.  Returns *default* immediately in --yes mode."""
    if _YES_ALL:
        hint = "Y (auto)" if default else "N (auto)"
        print(f"{question} [{hint}]")
        return default
    hint = "[Y/n]" if default else "[y/N]"
    while True:
        try:
            raw = input(f"{question} {hint}: ").strip().lower()
        except EOFError:
            return default
        if raw == "":
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  Please enter y or n.")


def _prompt_choice(prompt: str, options: list[tuple[str, str]]) -> str:
    """Display a numbered menu and return the chosen key.  In --yes mode picks first option."""
    for i, (_key, label) in enumerate(options, 1):
        print(f"    {i}. {label}")
    if _YES_ALL:
        print(f"  {prompt}: 1 (auto)")
        return options[0][0]
    while True:
        try:
            raw = input(f"  {prompt}: ").strip()
        except EOFError:
            return options[0][0]
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx][0]
        except ValueError:
            pass
        print(f"  Please enter a number between 1 and {len(options)}.")


def _resolve_path(config_key: str) -> Path:
    return _app_config.resolve_model_path(config_key)


def _model_dir_ok(path: Path) -> bool:
    """True when the directory exists and contains at least one file."""
    return path.exists() and any(path.iterdir())


def _load_download_module(script_name: str):
    """Import a packaged download helper from scripts.download_scripts."""
    return importlib.import_module(f"scripts.download_scripts.{script_name}")


# ── Step 1: environment ───────────────────────────────────────────────────────

def step_env() -> bool:
    _header("Step 1 / 4  —  Python environment")

    all_ok = True
    ver = sys.version_info

    if ver >= (3, 10):
        print(f"  [OK]      Python {ver.major}.{ver.minor}.{ver.micro}")
    else:
        print(
            f"  [WARN]    Python {ver.major}.{ver.minor}.{ver.micro}"
            " — Python 3.10+ is recommended"
        )

    try:
        import torch

        if torch.cuda.is_available():
            device = f"cuda  ({torch.cuda.get_device_name(0)})"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps   (Apple Silicon)"
        else:
            device = "cpu"
        print(f"  [OK]      PyTorch {torch.__version__}  —  device: {device}")
    except ImportError:
        print("  [MISSING] torch is not installed")
        all_ok = False

    if importlib.util.find_spec("sentence_transformers") is not None:
        print("  [OK]      sentence-transformers installed")
    else:
        print("  [MISSING] sentence-transformers is not installed")
        all_ok = False

    if importlib.util.find_spec("huggingface_hub") is not None:
        print("  [OK]      huggingface_hub installed")
    else:
        print("  [MISSING] huggingface_hub is not installed")
        all_ok = False

    if not all_ok:
        print()
        print("  Install missing packages first, then re-run this wizard:")
        print()
        print("    pip install -r requirements.txt")
        print("    trileaf onboard")

    return all_ok


# ── Step 2: detection models ──────────────────────────────────────────────────

def _download_model(
    label: str,
    script_name: str,
    output_dir: Path,
) -> bool:
    """Download a single model via its download_scripts/* module."""
    print()
    print(f"  Downloading {label}")
    print(f"  → {output_dir}")
    try:
        mod = _load_download_module(script_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        for filename in mod.REQUIRED_FILES:
            mod.download_if_needed(
                model_id=mod.DEFAULT_MODEL_ID,
                output_dir=output_dir,
                filename=filename,
                token=None,
                resume_download=True,
                force_download=False,
            )
        missing = mod.validate_download(output_dir)
        if missing:
            print(f"  [ERROR] Incomplete — missing files: {', '.join(missing)}")
            return False
        print(f"  [OK] {label} ready.")
        return True
    except Exception as exc:
        print(f"  [ERROR] Download failed: {exc}")
        return False


def step_detection_models() -> bool:
    _header("Step 2 / 4  —  Detection models  (required)")

    desklib_path = _resolve_path("desklib")
    mpnet_path   = _resolve_path("mpnet")

    desklib_ok = _model_dir_ok(desklib_path)
    mpnet_ok = _model_dir_ok(mpnet_path)

    print(
        f"  {'[OK]     ' if desklib_ok else '[MISSING]'}"
        f"  Desklib AI detector      → {desklib_path}"
    )
    print(
        f"  {'[OK]     ' if mpnet_ok else '[MISSING]'}"
        f"  MPNet sentence encoder   → {mpnet_path}"
    )

    if desklib_ok and mpnet_ok:
        print()
        print("  Both detection models are present.")
        print("  This is the only required local model set when you use an external rewrite API.")
        return True

    missing_names: list[str] = []
    if not desklib_ok:
        missing_names.append("Desklib  (~0.5 GB)")
    if not mpnet_ok:
        missing_names.append("MPNet  (~0.4 GB)")

    print()
    print(f"  Missing: {',  '.join(missing_names)}")
    print("  These models are required for AI-score and semantic-similarity scoring.")
    print("  They are the only mandatory local models when rewrites are handled by an API.")
    print("  Both are publicly available — no HuggingFace account needed.")
    print()

    if not _prompt_yn("  Download missing detection models now?", default=True):
        print()
        print("  Cannot continue without detection models.")
        print("  To download them manually:")
        if not desklib_ok:
            print("    python -m scripts.download_scripts.desklib_detector_download")
        if not mpnet_ok:
            print("    python -m scripts.download_scripts.mpnet_download")
        return False

    if not desklib_ok and not _download_model(
        "desklib/ai-text-detector-v1.01", "desklib_detector_download", desklib_path
    ):
        return False
    if not mpnet_ok and not _download_model(
        "sentence-transformers/paraphrase-mpnet-base-v2", "mpnet_download", mpnet_path
    ):
        return False

    return True


# ── Step 3: rewrite provider ──────────────────────────────────────────────────

def _dot_env_summary() -> Optional[str]:
    """Return a one-line summary of the current .env config, or None."""
    try:
        from scripts import rewrite_config as rc

        backend = rc._read_dot_env_key(rc.ENV_FILE, "REWRITE_BACKEND") or ""
        if not backend:
            return None
        if backend == "local":
            model_path = rc._read_dot_env_key(rc.ENV_FILE, "REWRITE_MODEL_PATH") or "./models/Qwen3-VL-8B-Instruct"
            return f"local → {model_path}"
        model = rc._read_dot_env_key(rc.ENV_FILE, "REWRITE_MODEL") or "(unset)"
        base_url = rc._read_dot_env_key(rc.ENV_FILE, "REWRITE_BASE_URL") or "(unset)"
        alias = rc._read_dot_env_key(rc.ENV_FILE, "LEAFHUB_ALIAS")
        if alias:
            return f"external via LeafHub alias='{alias}'  model={model}  url={base_url}"
        has_key = bool(rc._read_dot_env_key(rc.ENV_FILE, "REWRITE_API_KEY"))
        return f"external  model={model}  url={base_url}  key={'set' if has_key else '(missing)'}"
    except Exception:
        return None


def _dot_env_is_complete() -> bool:
    """True when .env has a complete external or local config."""
    try:
        from scripts import rewrite_config as rc

        backend = rc._read_dot_env_key(rc.ENV_FILE, "REWRITE_BACKEND") or ""
        if backend == "local":
            # Verify the recorded model path actually exists on disk.
            raw_path = rc._read_dot_env_key(rc.ENV_FILE, "REWRITE_MODEL_PATH") or ""
            if not raw_path:
                return True  # path not stored yet; let step_rewrite_provider re-check
            p = Path(raw_path)
            if not p.is_absolute():
                p = (PROJECT_ROOT / p).resolve()
            return _model_dir_ok(p)
        if backend not in ("external", "openai_api"):
            return False
        has_url = bool(rc._read_dot_env_key(rc.ENV_FILE, "REWRITE_BASE_URL"))
        has_model = bool(rc._read_dot_env_key(rc.ENV_FILE, "REWRITE_MODEL"))
        # Key comes from LeafHub or directly from .env
        has_key = bool(
            rc._read_dot_env_key(rc.ENV_FILE, "REWRITE_API_KEY")
            or rc._read_dot_env_key(rc.ENV_FILE, "LEAFHUB_ALIAS")
        )
        return has_url and has_model and has_key
    except Exception:
        return False


def _download_local_rewrite_model(output_dir: Path) -> bool:
    print()
    print("  Downloading Qwen/Qwen3-VL-8B-Instruct")
    print(f"  → {output_dir}")
    print("  (Large model — approximately 16 GB.  This may take a while.)")
    try:
        mod = _load_download_module("qwen3_vl_download")
        output_dir.mkdir(parents=True, exist_ok=True)
        mod.ensure_required_files(
            model_id=mod.DEFAULT_MODEL_ID,
            output_dir=output_dir,
            token=None,
            resume_download=True,
            force_download=False,
        )
        missing = mod.validate_download(output_dir)
        if missing:
            shown = ", ".join(missing[:3])
            extra = f" … (+{len(missing) - 3} more)" if len(missing) > 3 else ""
            print(f"  [ERROR] Incomplete — missing files: {shown}{extra}")
            return False
        print("  [OK] Qwen3-VL-8B-Instruct ready.")
        return True
    except Exception as exc:
        print(f"  [ERROR] Download failed: {exc}")
        return False


_LH_INSTALL_URL = "https://raw.githubusercontent.com/Rebas9512/Leafhub/main/install.sh"


def _leafhub_fallback(reason: str) -> bool:
    """
    Called when LeafHub setup fails mid-wizard.

    Prints the failure reason, then falls back to the interactive .env external
    wizard (or skips it in --yes mode). Returns True if fallback succeeds.
    """
    print()
    print("  [!] LeafHub setup could not complete:")
    for line in reason.splitlines():
        print(f"      {line}")
    _lh_hint = (
        "If you just installed LeafHub, open a new terminal first so the CLI\n"
        "  is on your PATH, then re-run:  trileaf onboard"
    )
    print()
    print(f"  Hint: {_lh_hint}")
    print()
    print("  Falling back to local .env credential storage.")
    print("  [!] The API key will be stored in plain text (chmod 600, git-ignored).")
    print()

    if _YES_ALL:
        print("  [--yes mode] Skipping interactive provider wizard.")
        print("  Configure the rewrite provider later with:  trileaf config")
        return True

    print("  Launching the .env provider wizard…")
    print()
    try:
        from scripts import rewrite_provider_cli
        exit_code = rewrite_provider_cli.main(["wizard"])
        return exit_code in (None, 0)
    except SystemExit as exc:
        return exc.code in (None, 0)
    except Exception as exc:
        print(f"  [ERROR] .env wizard also failed: {exc}")
        print("  Configure the rewrite provider later with:  trileaf config")
        return False


def step_rewrite_provider() -> bool:
    _header("Step 3 / 4  —  Rewrite provider")

    from scripts import rewrite_config as rc

    # ── Show current .env config if it exists ────────────────────────────
    summary = _dot_env_summary()
    if summary:
        print(f"  Current config (.env): {summary}")
        print()

    if _dot_env_is_complete():
        if _prompt_yn(
            "  Provider already configured. Keep this configuration?",
            default=True,
        ):
            print("  Keeping existing configuration.")
            return True

    # ── Check for LeafHub ────────────────────────────────────────────────
    try:
        from leafhub.probe import detect as _lh_detect
        _lh_found = _lh_detect(project_dir=PROJECT_ROOT)
    except ImportError:
        _lh_found = None
    except Exception as _det_exc:
        # detect() should never raise, but guard against unexpected errors
        # (corrupted dotfile, permission denied, etc.) and treat as absent.
        _lh_found = None
        import logging as _logging
        _logging.getLogger(__name__).debug("LeafHub detect() error: %s", _det_exc)

    lh_available = _lh_found is not None and (_lh_found.ready or _lh_found.can_link)
    lh_linked    = _lh_found is not None and _lh_found.ready

    # ── Upfront credential-manager question (only when LeafHub not present) ──
    # When LeafHub is already installed or linked, the existing menu below
    # already surfaces it as an option — no extra prompt needed.
    if not lh_available and not _YES_ALL:
        print()
        print("  How would you like to manage your API credentials?")
        print()
        print("    1. LeafHub  (recommended)")
        print("       Keys stored in an AES-256-GCM encrypted local vault.")
        print("       Never written to .env or shell history.")
        print(f"       https://github.com/Rebas9512/Leafhub")
        print()
        print("    2. Local .env file")
        print("       API key stored in plain text (chmod 600, git-ignored).")
        print("       Simpler setup — suitable for personal machines.")
        print()
        mgr = _prompt_choice(
            "Credential storage",
            [("leafhub", "LeafHub — encrypted vault"), ("dotenv", "Local .env file")],
        )
        if mgr == "leafhub":
            print()
            print("  Install LeafHub, then re-run  trileaf onboard.")
            print()
            print("  macOS / Linux / WSL:")
            print(f"    curl -fsSL {_LH_INSTALL_URL} | bash")
            print()
            print("  Windows (PowerShell):")
            print("    irm https://raw.githubusercontent.com/Rebas9512/Leafhub/main/install.ps1 | iex")
            print()
            print("  After the installer completes, open a new terminal and run:")
            print("    trileaf onboard")
            print()
            return False
        else:
            # User chose local .env — show security note and continue.
            print()
            print("  [!] Security note: the API key will be stored in plain text in")
            print("      PROJECT_ROOT/.env (chmod 600).  Anyone with read access to")
            print("      this directory can read it.  Use LeafHub on shared machines.")
            print()

    print()
    print("  Choose how the rewrite pipeline should obtain credentials:")
    print()
    options: list[tuple[str, str]] = []
    if lh_linked:
        options.append(("leafhub", f"LeafHub  (already linked as '{_lh_found.project_name}')"))
    elif lh_available:
        options.append(("leafhub", f"LeafHub  (server at {_lh_found.manage_url} — link this directory first)"))
    options += [
        ("external", "External API  (OpenAI-compatible — store key in .env)"),
        ("local",    "Local Qwen3-VL-8B  (offline — requires ≥16 GB VRAM)"),
    ]
    choice = _prompt_choice("Select option", options)

    # ── LeafHub path ──────────────────────────────────────────────────────
    if choice == "leafhub":
        if not lh_linked:
            print()
            print("  This directory is not yet linked to a LeafHub project.")
            if _YES_ALL:
                proj_name = "trileaf"
                print(f"  Project name: {proj_name} (auto)")
            else:
                proj_name = input("  Project name to create [trileaf]: ").strip() or "trileaf"
            print()
            try:
                from leafhub.probe import register as _lh_register
                _lh_found = _lh_register(proj_name, project_dir=PROJECT_ROOT, probe=_lh_found)
                print(f"  [OK] Project '{proj_name}' created and linked.")
            except Exception as _exc:
                return _leafhub_fallback(str(_exc))

        print()
        if _YES_ALL:
            alias    = "rewrite"
            base_url = ""
            model    = ""
            print(f"  LeafHub alias: {alias} (auto)")
            print()
        else:
            try:
                alias    = input("  LeafHub key alias (name of the key stored in LeafHub) [rewrite]: ").strip() or "rewrite"
                print()
                # base_url and model are optional — LeafHub provides them at runtime via get_config()
                base_url = input("  Provider base URL (e.g. https://api.openai.com/v1): ").strip()
                model    = input("  Model name (e.g. gpt-4o): ").strip()
            except EOFError:
                alias, base_url, model = "rewrite", "", ""

        try:
            env_values: dict[str, str] = {
                "LEAFHUB_ALIAS":    alias,
                "REWRITE_BACKEND":  "external",
            }
            if base_url:
                env_values["REWRITE_BASE_URL"] = base_url
            if model:
                env_values["REWRITE_MODEL"] = model

            rc.write_dot_env(
                env_values,
                header="Trileaf rewrite provider — credentials managed by LeafHub",
            )
        except Exception as _write_exc:
            return _leafhub_fallback(f"Failed to write .env: {_write_exc}")

        print(f"  [OK] Wrote .env  (alias={alias})")
        print(f"  API key will be fetched from LeafHub at runtime.")
        return True

    # ── External API path (key stored in .env) ────────────────────────────
    if choice == "external":
        if _YES_ALL:
            print()
            print("  [--yes mode] Skipping interactive provider wizard.")
            print("  Configure the rewrite provider later with:  trileaf config")
            return True

        print()
        print("  Launching the provider setup wizard…")
        print()
        try:
            from scripts import rewrite_provider_cli

            exit_code = rewrite_provider_cli.main(["wizard"])
            if exit_code not in (None, 0):
                print(f"  [WARN] Wizard exited with code {exit_code}.")
                return False
        except SystemExit as exc:
            if exc.code not in (None, 0):
                print(f"  [WARN] Wizard exited with code {exc.code}.")
                return False
        except Exception as exc:
            print(f"  [ERROR] Wizard failed: {exc}")
            return False

        if not _dot_env_is_complete():
            print()
            print("  [WARN] Provider does not appear to be fully configured.")
            print("  Re-run at any time with:  trileaf config")
        return True

    # ── Local rewrite-model path ──────────────────────────────────────────
    _raw_rewrite_path = (
        rc._read_dot_env_key(rc.ENV_FILE, "REWRITE_MODEL_PATH")
        or rc.legacy_env_first("model_path")
        or os.getenv("REWRITE_MODEL_PATH", "")
        or "./models/Qwen3-VL-8B-Instruct"
    )
    local_model_path = Path(_raw_rewrite_path)
    if not local_model_path.is_absolute():
        local_model_path = (PROJECT_ROOT / local_model_path).resolve()

    if _model_dir_ok(local_model_path):
        print()
        print(f"  [OK] Local rewrite model found at: {local_model_path}")
        rc.write_dot_env({"REWRITE_BACKEND": "local", "REWRITE_MODEL_PATH": str(local_model_path)})
        return True

    print()
    print(f"  [MISSING] Local rewrite model not found at: {local_model_path}")
    print()
    print("  Downloading requires approximately 16 GB of disk space.")
    print("  The model is publicly available — no HuggingFace account needed.")
    print("  Make sure you have a stable internet connection before proceeding.")
    print()

    if not _prompt_yn("  Download Qwen3-VL-8B-Instruct now?", default=True):
        print()
        print("  You can download it later with:")
        print("    python -m scripts.download_scripts.qwen3_vl_download")
        print()
        print("  Or switch to an external API instead:")
        print("    trileaf config")
        return False

    if not _download_local_rewrite_model(local_model_path):
        return False

    rc.write_dot_env({"REWRITE_BACKEND": "local", "REWRITE_MODEL_PATH": str(local_model_path)})
    return True


# ── Step 4: final validation ──────────────────────────────────────────────────

def step_final_validation() -> bool:
    _header("Step 4 / 4  —  Final validation")
    print()
    try:
        from scripts import check_env

        check_env.main()
        return True
    except SystemExit as exc:
        if exc.code not in (None, 0):
            print()
            print("  [FAIL] Environment check did not pass.")
            print("  Review the errors above, then re-run:")
            print("    trileaf onboard")
            return False
        return True
    except Exception as exc:
        print(f"  [ERROR] Validation error: {exc}")
        return False


# ── argument parsing ──────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trileaf — first-time setup wizard.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  trileaf onboard                        # interactive\n"
            "  python scripts/onboarding.py --yes     # headless / CI\n"
            "  trileaf config                         # configure an external rewrite provider\n"
        ),
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help=(
            "Non-interactive mode: automatically accept all download prompts "
            "and use defaults. Exits non-zero if any required step fails. "
            "The interactive rewrite-provider wizard is skipped; configure it "
            "later with: trileaf config"
        ),
    )
    return parser.parse_args(argv)


# ── main ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    global _YES_ALL
    args = _parse_args(argv)
    _YES_ALL = args.yes

    # Ensure ~/.trileaf/config.json exists with defaults before any step reads it.
    _app_config.ensure_config_exists()

    print()
    print("=" * 62)
    print("  Trileaf  —  First-time Setup")
    print("=" * 62)
    print()
    print("  This wizard will guide you through four steps:")
    print("    1. Verify your Python environment")
    print("    2. Download required detection models (if needed)")
    print("    3. Configure the rewrite provider (external API or optional local model)")
    print("    4. Run a final environment check")
    print()
    print("  Minimum local requirement: the two detection models.")
    print("  Rewrites can come from an external API, so a local rewrite model is optional.")
    print()
    if _YES_ALL:
        print("  Running in non-interactive mode (--yes).")
        print()
    else:
        print("  Press Ctrl+C at any time to cancel.")

    try:
        if not step_env():
            return 1

        if not step_detection_models():
            return 1

        if not step_rewrite_provider():
            return 1

        if not step_final_validation():
            return 1

    except KeyboardInterrupt:
        print()
        print()
        print("  Setup cancelled.")
        return 1

    print()
    print("=" * 62)
    print("  Setup complete!  Start Trileaf with:")
    print()
    if shutil.which("trileaf"):
        print("    trileaf run")
    else:
        print("    source .venv/bin/activate   # activate the venv first")
        print("    trileaf run")
        print()
        print("  Or directly without activating:")
        print("    .venv/bin/trileaf run")
    print()
    print("  To reconfigure the rewrite provider at any time:")
    print("    trileaf config")
    print("=" * 62)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
