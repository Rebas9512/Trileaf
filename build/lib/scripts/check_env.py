"""
Environment validator for Trileaf.

Checks
------
- torch device info
- sentence-transformers availability
- Each required model directory exists (DESKLIB, MPNET, local rewrite model if used)
- Model shard file completeness
- External rewrite API config if backend = external

Run standalone:  trileaf doctor
Called by run.py on startup.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Optional

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import app_config
from scripts import rewrite_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _check_shard_completeness(model_dir: Path) -> Optional[str]:
    """Return an error string if model shard files are incomplete, else None."""
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        # Single-file model — nothing to cross-check
        return None
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return f"Cannot read index file: {exc}"

    weight_map = data.get("weight_map", {})
    shard_names = sorted(set(weight_map.values()))
    missing = [n for n in shard_names if not (model_dir / n).exists()]
    if missing:
        return "Missing shards: " + ", ".join(missing[:3]) + (
            f" … (+{len(missing)-3} more)" if len(missing) > 3 else ""
        )
    return None



def main() -> None:
    try:
        from scripts._version import __version__
    except Exception:
        __version__ = "unknown"
    print(f"=== Trileaf v{__version__} — Environment Check ===")
    selected_profile = rewrite_config.load_selected_profile()
    active_profile_name = (
        str(selected_profile["name"])
        if isinstance(selected_profile, dict) and selected_profile.get("name")
        else os.getenv("REWRITE_PROFILE", "")
    )
    active_profile = (
        selected_profile.get("profile")
        if isinstance(selected_profile, dict)
        else None
    )

    # ── Device ────────────────────────────────────────────────────────────────
    try:
        import torch

        if torch.cuda.is_available():
            device = f"cuda ({torch.cuda.get_device_name(0)})"
        elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            device = "mps (Apple Silicon)"
        else:
            device = "cpu"
        print(f"Device:               {device}")
        print(f"PyTorch version:      {torch.__version__}")
    except ImportError:
        print("WARNING: torch not installed")

    # ── sentence-transformers ─────────────────────────────────────────────────
    # Use find_spec instead of a full import to avoid triggering the slow
    # sentence_transformers → sklearn → pandas cold-start import chain, which
    # can take 10-30 s on first run and make startup appear to hang.
    if importlib.util.find_spec("sentence_transformers") is not None:
        print("sentence-transformers: installed")
    else:
        print("WARNING: sentence-transformers not installed  (pip install sentence-transformers)")

    print()

    # ── Model paths ───────────────────────────────────────────────────────────
    rewrite_backend = rewrite_config.first_defined(
        rewrite_config.resolve_profile_value(active_profile, "backend"),
        os.getenv("REWRITE_BACKEND"),
        rewrite_config.legacy_env_first("backend"),
        "local",
    ).lower()
    if rewrite_backend == "openai_api":
        rewrite_backend = "external"

    if active_profile_name:
        print(f"Rewrite profile:      {active_profile_name}")

    local_models = {
        "desklib": str(app_config.resolve_model_path("desklib")),
        "mpnet":   str(app_config.resolve_model_path("mpnet")),
    }
    if rewrite_backend == "local":
        local_models["rewrite"] = str(_resolve(
            rewrite_config.first_defined(
                rewrite_config.resolve_profile_value(active_profile, "model_path"),
                os.getenv("REWRITE_MODEL_PATH"),
                rewrite_config.legacy_env_first("model_path"),
                "./models/Qwen3-VL-8B-Instruct",
            )
        ))

    all_ok = True
    print("Local model directories:")
    for env_key, path_str in local_models.items():
        path = _resolve(path_str)
        exists = path.exists()
        tag = "OK     " if exists else "MISSING"
        if not exists:
            all_ok = False
        print(f"  [{tag}] {env_key}")
        print(f"           → {path}")

        if exists:
            shard_err = _check_shard_completeness(path)
            if shard_err:
                print(f"           ⚠ {shard_err}")
                all_ok = False

    # ── External rewrite API config ──────────────────────────────────────────
    if rewrite_backend == "external":
        provider_id = rewrite_config.resolve_provider_id(active_profile)
        api_kind = rewrite_config.first_defined(
            rewrite_config.resolve_profile_value(active_profile, "api_kind"),
            os.getenv("REWRITE_API_KIND"),
            "openai-chat-completions",
        )
        api_url = rewrite_config.first_defined(
            rewrite_config.resolve_profile_value(active_profile, "base_url"),
            os.getenv("REWRITE_BASE_URL"),
            rewrite_config.legacy_env_first("base_url"),
            "",
        )
        api_model = rewrite_config.first_defined(
            rewrite_config.resolve_profile_value(active_profile, "model"),
            os.getenv("REWRITE_MODEL"),
            rewrite_config.legacy_env_first("model"),
            "",
        )
        api_key_info = rewrite_config.resolve_api_key(
            active_profile,
            provider_id=provider_id,
        )
        api_key = str(api_key_info.get("value") or "")
        api_key_source = str(api_key_info.get("source") or "")
        api_key_candidates = rewrite_config.format_env_var_list(
            api_key_info.get("candidates") or ["REWRITE_API_KEY"]
        )
        print()
        print("Rewrite backend: external")
        if provider_id:
            print(f"  provider: {provider_id}")
        print(f"  api_kind: {api_kind}")
        print(
            "  base_url: "
            f"{api_url or '(not set — configure in rewrite profile or set an env override)'}"
        )
        print(f"  model:    {api_model or '(not set)'}")
        if api_key:
            source_label = api_key_source or "profile"
            print(f"  api_key:  set ({source_label})")
        else:
            print(f"  api_key:  (not set; accepted env vars: {api_key_candidates})")
        if not api_url:
            print("  ERROR: REWRITE_BASE_URL is required for external backend")
            all_ok = False
        if not api_model:
            print("  ERROR: REWRITE_MODEL is required for external backend")
            all_ok = False
        if not api_key:
            print(
                "  ERROR: rewrite API key is required for external backend "
                f"(profile api_key or env: {api_key_candidates})"
            )
            all_ok = False

    print()
    if all_ok:
        if rewrite_backend == "external":
            print("All checks passed. Detection models are ready; rewrites will use the configured external API.")
        else:
            print("All checks passed. Detection models and the local rewrite model are ready.")
    else:
        print("Some checks failed. Download or configure missing models before running.")
        print("  HuggingFace repos:")
        print("    desklib/ai-text-detector-v1.01")
        print("    sentence-transformers/paraphrase-mpnet-base-v2  (downloads to sentence-transformers-paraphrase-mpnet-base-v2/)")
        print("    Qwen/Qwen3-VL-8B-Instruct  (if using local rewrite backend)")
        print("  Or re-run:  trileaf onboard")
    print("=================================================")

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
