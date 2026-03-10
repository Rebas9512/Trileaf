#!/usr/bin/env python3
"""
Local Qwen rewrite pipeline test.

Tests run_rewrite_candidate (all 3 styles) + Desklib + MPNet scoring
against the local Qwen3-VL-8B-Instruct model, using 3 sample chunks.

Run:
    python scripts/test_local_qwen.py
    python scripts/test_local_qwen.py --debug    # show raw model output
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Force local backend BEFORE importing models_runtime ───────────────────────
# models_runtime reads all config at module import time, so env vars must be
# set before the first import.  The active profile (e.g. minimax) takes
# precedence over REWRITE_BACKEND, so we clear REWRITE_PROFILE first so that
# the module falls through to plain env-var resolution.
# Select the built-in "local-rewrite" profile so the saved active profile
# (e.g. minimax) is ignored.  The profile store's "local-rewrite" entry has
# backend=local and model_path=./models/Qwen3-VL-8B-Instruct by default.
os.environ["REWRITE_PROFILE"]    = "local-rewrite"
os.environ["REWRITE_MODEL_PATH"] = str(PROJECT_ROOT / "models" / "Qwen3-VL-8B-Instruct")
os.environ["REWRITE_DISABLE_THINKING"] = "true"   # skip chain-of-thought for speed

# ── Sample chunks ─────────────────────────────────────────────────────────────
CHUNKS = [
    # chunk 0 — expository AI prose (typical pipeline input)
    (
        "Artificial intelligence has fundamentally transformed the way modern "
        "businesses operate across every sector. Companies are increasingly adopting "
        "machine learning solutions to automate repetitive tasks and improve "
        "decision-making processes across all departments."
    ),
    # chunk 1 — dialogue-heavy fiction (stress-tests quote preservation)
    (
        '"Sir," the boy said nervously, "is there another train tonight?" '
        'Mr. Harris shook his head gently. "Not until morning."'
    ),
    # chunk 2 — data/numbers (checks factual preservation)
    (
        "In 2023, global cloud computing expenditure reached $591 billion, "
        "representing a 21 % year-over-year increase. Analysts expect that figure "
        "to exceed $1 trillion by 2027, driven largely by AI workload demand."
    ),
]

_DIV  = "─" * 70
_DIV2 = "═" * 70


def _pct(score: float) -> str:
    return f"{score * 100:.1f}%"


def _diff(orig: float, new: float) -> str:
    delta = new - orig
    sign  = "+" if delta >= 0 else ""
    return f"{sign}{delta * 100:.1f}pp"


def run_test(debug: bool) -> None:
    if debug:
        os.environ["REWRITE_DEBUG"] = "1"

    import logging
    if debug:
        logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)
    else:
        logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

    print()
    print(_DIV2)
    print("  Trileaf — Local Qwen Pipeline Test")
    print(_DIV2)

    # ── Import runtime (happens after env vars are set) ───────────────────────
    print()
    print("Loading runtime module…")
    t0 = time.time()
    import scripts.models_runtime as mr
    print(f"  REWRITE_BACKEND:  {mr.REWRITE_BACKEND}")
    print(f"  REWRITE_MODEL:    {mr.REWRITE_MODEL}")
    print(f"  REWRITE_MODEL_PATH: {mr.REWRITE_MODEL_PATH}")
    print(f"  DEVICE:           {mr.DEVICE}")
    print(f"  DISABLE_THINKING: {mr.REWRITE_DISABLE_THINKING}")
    print(f"  (import took {time.time()-t0:.2f}s)")

    if mr.REWRITE_BACKEND != "local":
        print(f"\n[ABORT] Backend resolved to '{mr.REWRITE_BACKEND}', expected 'local'.")
        print("  Check that REWRITE_BACKEND env var is set before models_runtime is imported.")
        sys.exit(1)

    # ── Pre-load local model once (warm-up) ───────────────────────────────────
    print()
    print("Pre-loading local Qwen model…")
    t0 = time.time()
    try:
        mr._load_local_rewrite_model()
        print(f"  Model loaded in {time.time()-t0:.1f}s")
    except Exception as exc:
        print(f"  [FAIL] Model load: {type(exc).__name__}: {exc}")
        sys.exit(1)

    # ── Per-chunk tests ───────────────────────────────────────────────────────
    passed = 0
    failed = 0

    for chunk_idx, chunk_text in enumerate(CHUNKS):
        print()
        print(_DIV)
        print(f"  Chunk {chunk_idx + 1} / {len(CHUNKS)}")
        print(_DIV)
        print(f"  Original ({len(chunk_text)} chars):")
        print(f"    {chunk_text[:200]}")
        print()

        # Baseline AI score
        t0 = time.time()
        orig_ai = mr.run_desklib(chunk_text)
        print(f"  Baseline AI score: {_pct(orig_ai)}  ({time.time()-t0:.2f}s)")
        print()

        for style in mr.REWRITE_STYLES:
            t0 = time.time()
            try:
                result = mr.run_rewrite_candidate(chunk_text, style=style)
                elapsed = time.time() - t0

                ai_score  = mr.run_desklib(result)
                sem_score = mr.run_mpnet_similarity(chunk_text, result)
                identical = result.strip() == chunk_text.strip()
                gate_pass = (ai_score < orig_ai) and (sem_score >= 0.65)

                status = "PASS" if gate_pass else "fail"
                if identical:
                    status = "IDENTICAL — silent fallback!"

                print(f"  [{style:>12}]  {elapsed:.1f}s  "
                      f"AI {_pct(orig_ai)} → {_pct(ai_score)} ({_diff(orig_ai, ai_score)})  "
                      f"sem {_pct(sem_score)}  gate={status}")
                print(f"                → {result[:160]!r}")

                if identical:
                    failed += 1
                else:
                    passed += 1

            except Exception as exc:
                elapsed = time.time() - t0
                print(f"  [{style:>12}]  {elapsed:.1f}s  [EXCEPTION] {type(exc).__name__}: {exc}")
                failed += 1
            print()

    # ── Summary ───────────────────────────────────────────────────────────────
    total = passed + failed
    print(_DIV2)
    print(f"  Results: {passed}/{total} rewrites differ from original")
    if failed:
        print(f"  {failed} case(s) returned identical text or raised exceptions")
    print(_DIV2)
    print()

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local Qwen pipeline test")
    parser.add_argument("--debug", action="store_true", help="Print raw model output to stderr")
    args = parser.parse_args()
    run_test(debug=args.debug)
