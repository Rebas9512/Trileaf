#!/usr/bin/env python3
"""
Pipeline timing diagnostic — measures where the gap is between
POST /api/optimize and the first chunk_start WS event.

Run (from project root, with venv active):
    python scripts/test_pipeline_timing.py
    python scripts/test_pipeline_timing.py --text "Your text here."
    python scripts/test_pipeline_timing.py --model-load  # also time first model load
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SAMPLE_TEXT = (
    "Artificial intelligence has transformed the way modern businesses operate. "
    "Companies are increasingly adopting machine learning solutions to automate "
    "repetitive tasks and improve decision-making processes across all departments. "
    "The integration of AI tools into daily workflows has significantly changed "
    "how employees interact with data and make decisions in real time."
)

_T0 = None  # global start time


def _elapsed(since: float | None = None) -> str:
    ref = since if since is not None else _T0
    return f"{(time.perf_counter() - ref) * 1000:.1f}ms"


def _mark(label: str, since: float | None = None) -> float:
    t = time.perf_counter()
    print(f"  [{_elapsed(since)}]  {label}")
    return t


async def run_timing_test(text: str, also_time_model_load: bool = False) -> None:
    global _T0

    print()
    print("=" * 70)
    print("  Pipeline Timing Diagnostic")
    print("=" * 70)
    print(f"  Text: {text[:80]}{'…' if len(text) > 80 else ''}")
    print(f"  Text length: {len(text)} chars")
    print()

    # ── 1. Import cost ────────────────────────────────────────────────────────
    print("── Phase 1: imports ──────────────────────────────────────────────────")
    t_imp = time.perf_counter()
    from scripts import chunker
    _mark("chunker imported", t_imp)

    t_imp2 = time.perf_counter()
    import scripts.models_runtime as mr
    _mark("models_runtime imported", t_imp2)

    t_imp3 = time.perf_counter()
    from scripts.orchestrator import run_pipeline
    _mark("orchestrator imported", t_imp3)
    print()

    # ── 2. Model load (optional cold-start measure) ───────────────────────────
    if also_time_model_load:
        print("── Phase 2: model pre-load ───────────────────────────────────────────")
        t_ml = time.perf_counter()
        results = await asyncio.to_thread(mr.preload_models)
        for r in results:
            print(f"  [{(time.perf_counter() - t_ml)*1000:.1f}ms]  "
                  f"{r['name']}: {r['status']} in {r['seconds']:.3f}s")
        print()

    # ── 3. Pipeline entry-point timing ───────────────────────────────────────
    print("── Phase 3: pipeline event timing ───────────────────────────────────")

    events: list[dict] = []

    async def mock_broadcast(payload: dict) -> None:
        t = time.perf_counter()
        evt_type = payload.get("type", "?")
        data = payload.get("data", {})
        chunk_idx = data.get("chunk_idx", "")
        tag = f"{evt_type}" + (f"[{chunk_idx}]" if chunk_idx != "" else "")
        elapsed_ms = (t - _T0) * 1000
        events.append({"type": evt_type, "chunk_idx": chunk_idx, "t_ms": elapsed_ms})
        print(f"  [{elapsed_ms:7.1f}ms]  WS event: {tag}")

    _T0 = time.perf_counter()
    print(f"  [      0ms]  run_pipeline() called")

    try:
        await run_pipeline(
            text=text,
            broadcast=mock_broadcast,
            run_id="timing-test",
        )
    except Exception as exc:
        print(f"\n  ERROR during pipeline: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        return

    total_ms = (time.perf_counter() - _T0) * 1000
    print(f"\n  [  total]  {total_ms:.0f}ms\n")

    # ── 4. Gap analysis ───────────────────────────────────────────────────────
    print("── Phase 4: gap analysis ─────────────────────────────────────────────")

    run_start_evt  = next((e for e in events if e["type"] == "run_start"), None)
    chunk0_start   = next((e for e in events if e["type"] == "chunk_start" and e["chunk_idx"] == 0), None)
    first_rw_cand  = next((e for e in events if e["type"] == "rewrite_candidate"), None)

    if run_start_evt:
        print(f"  run_start arrived at      {run_start_evt['t_ms']:.1f}ms   ← gap from POST submit")
    if chunk0_start:
        print(f"  chunk_start[0] arrived at {chunk0_start['t_ms']:.1f}ms")
        if run_start_evt:
            gap = chunk0_start["t_ms"] - run_start_evt["t_ms"]
            print(f"  run_start → chunk_start   {gap:.1f}ms gap")
    if first_rw_cand and chunk0_start:
        gap2 = first_rw_cand["t_ms"] - chunk0_start["t_ms"]
        print(f"  chunk_start → 1st rewrite {gap2:.1f}ms gap  (baseline Desklib on chunk)")

    print()
    print("  Key question: is run_start fast (<200ms)?")
    if run_start_evt:
        ms = run_start_evt["t_ms"]
        if ms < 200:
            print(f"  → YES ({ms:.0f}ms). Backend gap is fine.")
            print("  → If UI still feels slow, the delay is WebSocket round-trip")
            print("    or JS rendering. Pre-render 1 placeholder card on run_start.")
        else:
            print(f"  → NO ({ms:.0f}ms). There is a backend bottleneck before run_start.")
            print("  → Check what runs before the run_start broadcast in orchestrator.")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline timing diagnostic")
    parser.add_argument("--text", default=SAMPLE_TEXT, help="Input text to test")
    parser.add_argument("--model-load", action="store_true",
                        help="Also measure cold model load time")
    args = parser.parse_args()

    asyncio.run(run_timing_test(args.text, also_time_model_load=args.model_load))


if __name__ == "__main__":
    main()
