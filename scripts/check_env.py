"""
Environment validator for the writing optimizer.

Checks
------
- torch device info
- sentence-transformers availability
- Each required model directory exists (DESKLIB, MPNET, QWEN if local)
- Model shard file completeness
- Qwen API config if backend = openai_api

Run standalone:  python scripts/check_env.py
Called by run.py on startup.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

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
    print("=== LLM Writing Optimizer — Environment Check ===")

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
    try:
        import sentence_transformers as _st
        print(f"sentence-transformers: {_st.__version__}")
    except ImportError:
        print("WARNING: sentence-transformers not installed  (pip install sentence-transformers)")

    print()

    # ── Model paths ───────────────────────────────────────────────────────────
    qwen_backend = os.getenv("QWEN_BACKEND", "local").lower()

    local_models = {
        "DESKLIB_MODEL_PATH": os.getenv("DESKLIB_MODEL_PATH", "./models/desklib-ai-text-detector-v1.01"),
        "MPNET_MODEL_PATH":   os.getenv("MPNET_MODEL_PATH",   "./models/paraphrase-mpnet-base-v2"),
    }
    if qwen_backend == "local":
        local_models["QWEN_MODEL_PATH"] = os.getenv("QWEN_MODEL_PATH", "./models/Qwen3-VL-8B-Instruct")

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

    # ── Qwen API config ───────────────────────────────────────────────────────
    if qwen_backend == "openai_api":
        api_url   = os.getenv("QWEN_API_BASE_URL", "")
        api_model = os.getenv("QWEN_API_MODEL", "")
        print()
        print("Qwen backend: openai_api")
        print(f"  base_url: {api_url or '(not set — set QWEN_API_BASE_URL in .env)'}")
        print(f"  model:    {api_model or '(not set)'}")
        if not api_url:
            print("  ERROR: QWEN_API_BASE_URL is required for openai_api backend")
            all_ok = False

    print()
    if all_ok:
        print("All checks passed. Ready to run.")
    else:
        print("Some checks failed. Download or configure missing models before running.")
        print("  HuggingFace repos:")
        print("    desklib/ai-text-detector-v1.01")
        print("    sentence-transformers/paraphrase-mpnet-base-v2")
        print("    Qwen/Qwen3-VL-8B-Instruct  (if using local backend)")
    print("=================================================")

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
