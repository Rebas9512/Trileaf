"""
Environment validator for Trileaf.

Checks
------
- torch device info
- sentence-transformers availability
- Each required model directory exists (DESKLIB, MPNET)
- Model shard file completeness
- External rewrite API config (base_url, model, api_key)

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

    credential_source = os.getenv("REWRITE_CREDENTIAL_SOURCE", "")

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

    if credential_source:
        print(f"Credential source:    {credential_source}")

    # ── Detection model paths ─────────────────────────────────────────────────
    local_models = {
        "desklib": str(app_config.resolve_model_path("desklib")),
        "mpnet":   str(app_config.resolve_model_path("mpnet")),
    }

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
    provider_id = rewrite_config.normalize_provider_id(os.getenv("REWRITE_PROVIDER_ID", ""))
    api_kind  = os.getenv("REWRITE_API_KIND") or "openai-chat-completions"
    api_url   = os.getenv("REWRITE_BASE_URL") or ""
    api_model = os.getenv("REWRITE_MODEL") or ""
    api_key   = os.getenv("REWRITE_API_KEY", "")
    leafhub_alias = os.getenv("LEAFHUB_ALIAS") or ""
    api_key_candidates = rewrite_config.format_env_var_list(
        rewrite_config.get_provider_env_api_key_candidates(provider_id)
    )
    print()
    cred_label = "via LeafHub" if leafhub_alias else (credential_source or "env var")
    print(f"Rewrite backend: external  (credential: {cred_label})")
    if provider_id:
        print(f"  provider:  {provider_id}")
    print(f"  api_kind:  {api_kind}")
    print(f"  base_url:  {api_url or '(not set — run: trileaf config)'}")
    print(f"  model:     {api_model or '(not set)'}")
    if leafhub_alias:
        print(f"  api_key:   via LeafHub alias='{leafhub_alias}' ({credential_source or 'pending'})")
    elif api_key:
        print(f"  api_key:   set ({credential_source or 'env'})")
    else:
        print(f"  api_key:   (not set; run 'trileaf config' or set {api_key_candidates})")
    if not api_url:
        print("  ERROR: REWRITE_BASE_URL is required")
        all_ok = False
    if not api_model:
        print("  ERROR: REWRITE_MODEL is required")
        all_ok = False
    if not api_key and not leafhub_alias:
        print(
            "  ERROR: rewrite API key is required — "
            f"link LeafHub ('trileaf config') or set {api_key_candidates}"
        )
        all_ok = False

    print()
    if all_ok:
        print("All checks passed. Detection models are ready; rewrites will use the configured external API.")
    else:
        print("Some checks failed. Download or configure missing models before running.")
        print("  HuggingFace repos:")
        print("    desklib/ai-text-detector-v1.01")
        print("    sentence-transformers/paraphrase-mpnet-base-v2  (downloads to sentence-transformers-paraphrase-mpnet-base-v2/)")
        print("  Or re-run:  trileaf setup")
    print("=================================================")

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
