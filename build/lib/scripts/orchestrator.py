"""
Pipeline orchestrator for the writing optimizer.

Full flow per run
-----------------
1.  Chunk input text (chunker.split_text).
2.  Per chunk — multi-criteria Pareto selection:
    Step 0  Baseline scoring: AI score, length.
    Step 1  Ensemble rewrite: generate 3 candidates (conservative / balanced / aggressive).
    Step 2  Batch scoring: for every candidate compute AI score, chunk-level semantic
            similarity, length ratio, and worst sentence-level similarity.
    Step 3  Hard gate: drop candidates that violate any quality floor.
    Step 4  Pareto front (non-dominated sorting on ai_score / sem_score);
            among rank-0 candidates compute a weighted utility score and select the best.
            If no candidate passes the gate → fallback to original chunk.
3.  Reassemble chunks into final text.
4.  Desklib + MPNet full-text final scores.
5.  Broadcast run_done.

All blocking inference calls run in asyncio.to_thread() so WebSocket
broadcasts are not blocked between model calls.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from scripts import app_config as _app_config

_pipeline = _app_config.get_pipeline_config()

MAX_CHUNK_CHARS   = int(_pipeline["max_chunk_chars"])

# Hard gate thresholds
SEM_GATE          = float(_pipeline["sem_gate"])
MIN_SENT_SIM_GATE = float(_pipeline["min_sent_sim_gate"])
LEN_RATIO_MIN     = float(_pipeline["len_ratio_min"])
LEN_RATIO_MAX     = float(_pipeline["len_ratio_max"])

# Utility weights
W_AI      = float(_pipeline["w_ai"])
W_SEM     = float(_pipeline["w_sem"])
W_RISK    = float(_pipeline["w_risk"])


# ─── Data structures ───────────────────────────────────────────────────────────


@dataclass
class CandidateScore:
    """All metrics for one rewrite candidate."""
    style:          str
    text:           str
    ai_score:       float   # Desklib AI probability [0,1]; lower = more human
    sem_score:      float   # chunk-level cosine similarity to original [0,1]
    length_ratio:   float   # len(text) / len(original); 1.0 = same length
    min_sent_sim:   float   # worst sentence-level similarity in MPNet alignment
    gate_pass:      bool  = False
    gate_fail_reason: str = ""
    pareto_rank:    int   = -1   # 0 = Pareto-optimal (non-dominated)
    utility:        float = 0.0  # final weighted utility among gate-passing candidates


@dataclass
class ChunkResult:
    chunk_idx:          int
    original:           str
    orig_ai_score:      float
    candidates:         List[CandidateScore]
    selected:           Optional[CandidateScore]  # None if fallback to original
    final_text:         str
    final_ai_score:     float
    final_sem_score:    float
    reverted_to_original: bool
    status_label:       str = "Edited"


@dataclass
class PipelineResult:
    run_id:            str
    input_text:        str
    chunks:            List[ChunkResult]
    output_text:       str
    original_ai_score: float
    final_ai_score:    float
    final_sem_score:   float


# ─── Pareto / utility helpers ──────────────────────────────────────────────────


def _z_normalize(values: List[float]) -> List[float]:
    if len(values) <= 1:
        return [0.0] * len(values)
    mean = sum(values) / len(values)
    var  = sum((v - mean) ** 2 for v in values) / len(values)
    std  = var ** 0.5
    if std < 1e-9:
        return [0.0] * len(values)
    return [(v - mean) / std for v in values]


def _compute_pareto_ranks(candidates: List[CandidateScore]) -> None:
    """
    Assign pareto_rank in-place.  rank=0 means non-dominated (Pareto-optimal).

    Objectives (all "maximise"):
      • −ai_score   (lower AI prob = better)
      • sem_score   (higher semantic similarity = better)
    """
    for i, b in enumerate(candidates):
        dominated = False
        for j, a in enumerate(candidates):
            if i == j:
                continue
            a_obj = (-a.ai_score, a.sem_score)
            b_obj = (-b.ai_score, b.sem_score)
            if all(ao >= bo for ao, bo in zip(a_obj, b_obj)) and any(
                ao > bo for ao, bo in zip(a_obj, b_obj)
            ):
                dominated = True
                break
        b.pareto_rank = 0 if not dominated else 1


def _compute_utility(
    all_pass: List[CandidateScore],
    orig_ai:  float,
    w_ai:     float = W_AI,
    w_sem:    float = W_SEM,
    w_risk:   float = W_RISK,
) -> None:
    """
    Compute and assign utility scores in-place for every candidate in *all_pass*.

    Z-normalisation is performed across the entire gate-passing set so that
    scores are calibrated even when only one Pareto-optimal candidate exists.

    U = w_ai * ai_gain_z + w_sem * sem_z − w_risk * risk_penalty

    risk_penalty = length deviation >10% + worst-sentence penalty below 0.5
    """
    ai_gains = [orig_ai - c.ai_score for c in all_pass]
    sem_vals = [c.sem_score           for c in all_pass]

    ai_z  = _z_normalize(ai_gains)
    sem_z = _z_normalize(sem_vals)

    for i, c in enumerate(all_pass):
        len_dev  = max(0.0, abs(c.length_ratio - 1.0) - 0.1)
        sent_pen = max(0.0, 0.5 - c.min_sent_sim) if c.min_sent_sim < 0.5 else 0.0
        risk     = len_dev + sent_pen
        c.utility = (
            w_ai  * ai_z[i]
            + w_sem * sem_z[i]
            - w_risk * risk
        )


def _apply_hard_gate(
    candidates: List[CandidateScore],
    orig_ai:    float,
) -> None:
    """Set gate_pass / gate_fail_reason in-place for each candidate."""
    for c in candidates:
        reasons: List[str] = []

        if c.ai_score >= orig_ai:
            reasons.append(f"ai_score {c.ai_score:.4f} ≥ orig {orig_ai:.4f}")
        if c.sem_score < SEM_GATE:
            reasons.append(f"sem {c.sem_score:.4f} < gate {SEM_GATE}")
        if c.min_sent_sim < MIN_SENT_SIM_GATE:
            reasons.append(f"min_sent_sim {c.min_sent_sim:.4f} < gate {MIN_SENT_SIM_GATE}")
        if not (LEN_RATIO_MIN <= c.length_ratio <= LEN_RATIO_MAX):
            reasons.append(
                f"len_ratio {c.length_ratio:.2f} outside [{LEN_RATIO_MIN}, {LEN_RATIO_MAX}]"
            )

        if reasons:
            c.gate_pass        = False
            c.gate_fail_reason = "; ".join(reasons)
        else:
            c.gate_pass = True


# ─── Public entry point ────────────────────────────────────────────────────────


async def run_pipeline(
    text:      str,
    broadcast: Callable[[Dict[str, Any]], Any],
    run_id:    str,
    w_ai:      float = W_AI,
    w_sem:     float = W_SEM,
    w_risk:    float = W_RISK,
) -> PipelineResult:
    """Run the full writing-optimizer pipeline, streaming events via *broadcast*."""
    from scripts import chunker
    from scripts import models_runtime as mr

    text = chunker.clean_text(text)

    chunks: List[str] = await asyncio.to_thread(chunker.split_text, text, MAX_CHUNK_CHARS)
    if not chunks:
        chunks = [text]

    await broadcast(
        {
            "type": "run_start",
            "data": {
                "run_id":       run_id,
                "total_chunks": len(chunks),
                "weights":      {"w_ai": w_ai, "w_sem": w_sem, "w_risk": w_risk},
            },
        }
    )

    chunk_results: List[ChunkResult] = []
    for i, chunk in enumerate(chunks):
        result = await _process_chunk(
            chunk=chunk,
            chunk_idx=i,
            total_chunks=len(chunks),
            broadcast=broadcast,
            mr=mr,
            chunker=chunker,
            w_ai=w_ai,
            w_sem=w_sem,
            w_risk=w_risk,
        )
        chunk_results.append(result)

    output_text = "\n\n".join(r.final_text for r in chunk_results)

    await broadcast({"type": "final_scoring", "data": {"message": "Computing final scores…"}})

    original_ai, final_ai, final_sem = await asyncio.gather(
        asyncio.to_thread(mr.run_desklib,          text),
        asyncio.to_thread(mr.run_desklib,          output_text),
        asyncio.to_thread(mr.run_mpnet_similarity, text, output_text),
    )

    # Build per-chunk payloads; attach best_candidate for reverted chunks
    chunk_payloads = []
    for r in chunk_results:
        best_cand_payload = None
        if r.reverted_to_original and r.candidates:
            best = min(r.candidates, key=lambda c: c.ai_score)
            best_cand_payload = {
                "text":      best.text,
                "style":     best.style,
                "ai_score":  round(best.ai_score, 4),
                "sem_score": round(best.sem_score, 4),
                "gate_fail_reason": best.gate_fail_reason,
            }
        chunk_payloads.append({
            "chunk_idx":            r.chunk_idx,
            "original_text":        r.original,
            "final_text":           r.final_text,
            "reverted_to_original": r.reverted_to_original,
            "status_label":         r.status_label,
            "original_ai_score":    round(r.orig_ai_score, 4),
            "final_ai_score":       round(r.final_ai_score, 4),
            "final_sem_score":      round(r.final_sem_score, 4),
            "selected_style":       r.selected.style if r.selected else "original",
            "best_candidate":       best_cand_payload,
            "candidates": [
                {
                    "style":            c.style,
                    "ai_score":         round(c.ai_score, 4),
                    "sem_score":        round(c.sem_score, 4),
                    "length_ratio":     round(c.length_ratio, 3),
                    "min_sent_sim":     round(c.min_sent_sim, 4),
                    "gate_pass":        c.gate_pass,
                    "gate_fail_reason": c.gate_fail_reason,
                    "pareto_rank":      c.pareto_rank,
                    "utility":          round(c.utility, 4),
                }
                for c in r.candidates
            ],
        })

    await broadcast(
        {
            "type": "run_done",
            "data": {
                "run_id":            run_id,
                "output":            output_text,
                "original_ai_score": original_ai,
                "final_ai_score":    final_ai,
                "final_sem_score":   final_sem,
                "chunks":            chunk_payloads,
            },
        }
    )

    return PipelineResult(
        run_id=run_id,
        input_text=text,
        chunks=chunk_results,
        output_text=output_text,
        original_ai_score=original_ai,
        final_ai_score=final_ai,
        final_sem_score=final_sem,
    )


# ─── Per-chunk processing ──────────────────────────────────────────────────────


async def _process_chunk(
    chunk:        str,
    chunk_idx:    int,
    total_chunks: int,
    broadcast:    Callable,
    mr:           Any,
    chunker:      Any,
    w_ai:         float = W_AI,
    w_sem:        float = W_SEM,
    w_risk:       float = W_RISK,
) -> ChunkResult:

    # ── helpers ──────────────────────────────────────────────────────────────

    async def emit_stage(stage: str, state: str, message: str, **extra: Any) -> None:
        payload: Dict[str, Any] = {
            "type": "chunk_stage",
            "data": {
                "chunk_idx": chunk_idx,
                "stage":     stage,
                "state":     state,
                "message":   message,
            },
        }
        if extra:
            payload["data"].update(extra)
        await broadcast(payload)

    async def emit_log(message: str, level: str = "info", **extra: Any) -> None:
        payload: Dict[str, Any] = {
            "type": "chunk_log",
            "data": {
                "chunk_idx": chunk_idx,
                "level":     level,
                "message":   message,
            },
        }
        if extra:
            payload["data"].update(extra)
        await broadcast(payload)

    # ── Step 0: baseline scoring ──────────────────────────────────────────────

    await broadcast(
        {
            "type": "chunk_start",
            "data": {"chunk_idx": chunk_idx, "total": total_chunks, "text": chunk},
        }
    )

    orig_ai  = await asyncio.to_thread(mr.run_desklib, chunk)
    orig_len = max(1, len(chunk))

    orig_sents: List[str] = await asyncio.to_thread(chunker.split_sentences, chunk)
    if not orig_sents:
        orig_sents = [chunk]

    await emit_log(
        f"Baseline — AI: {orig_ai:.4f}  len: {orig_len}",
        level="info",
    )
    await broadcast(
        {
            "type": "chunk_baseline",
            "data": {
                "chunk_idx": chunk_idx,
                "ai_score":  round(orig_ai, 4),
                "length":    orig_len,
            },
        }
    )

    # ── Step 1: ensemble rewrite ──────────────────────────────────────────────
    # Generate each candidate individually so the UI can track per-draft progress.

    _styles = list(mr.REWRITE_STYLES)
    await emit_stage("rewrite", "active", f"Generating {len(_styles)} drafts")
    candidates_raw: List[Dict[str, Any]] = []
    rewrite_error_count = 0

    for _s_idx, _style in enumerate(_styles):
        await broadcast({
            "type": "rewrite_candidate",
            "data": {
                "chunk_idx":    chunk_idx,
                "style":        _style,
                "style_idx":    _s_idx,
                "total_styles": len(_styles),
                "status":       "generating",
            },
        })
        try:
            _cand_text = await asyncio.to_thread(mr.run_rewrite_candidate, chunk, _style)
            candidates_raw.append({"style": _style, "text": _cand_text})
            await broadcast({
                "type": "rewrite_candidate",
                "data": {
                    "chunk_idx":    chunk_idx,
                    "style":        _style,
                    "style_idx":    _s_idx,
                    "total_styles": len(_styles),
                    "status":       "done",
                },
            })
        except Exception as _exc:
            rewrite_error_count += 1
            candidates_raw.append({
                "style":               _style,
                "text":                chunk,
                "error":               f"{type(_exc).__name__}: {_exc}",
                "reverted_to_original": True,
            })
            await broadcast({
                "type": "rewrite_candidate",
                "data": {
                    "chunk_idx":    chunk_idx,
                    "style":        _style,
                    "style_idx":    _s_idx,
                    "total_styles": len(_styles),
                    "status":       "error",
                },
            })
            await emit_log(
                f"  [{_style:>12}] {type(_exc).__name__}: {_exc} — using original text for this candidate.",
                level="warn",
            )

    if rewrite_error_count:
        await emit_log(
            f"Rewrite backend returned errors for {rewrite_error_count}/{len(_styles)} drafts; "
            "failed drafts were replaced with the original chunk.",
            level="warn",
        )

    await emit_log("Candidates generated — running batch scoring…")

    # ── Step 2: batch scoring ─────────────────────────────────────────────────

    await emit_stage("batch_score", "active", "Scoring all candidates (AI + Sem)")

    scored: List[CandidateScore] = []
    for c in candidates_raw:
        text = c["text"]

        ai_score  = await asyncio.to_thread(mr.run_desklib,         text)
        sem_score = await asyncio.to_thread(mr.run_mpnet_similarity, chunk, text)

        length_ratio = len(text) / orig_len

        cand_sents: List[str] = await asyncio.to_thread(chunker.split_sentences, text)
        if not cand_sents:
            cand_sents = [text]

        sent_pairs = await asyncio.to_thread(
            mr.run_mpnet_sentence_align, orig_sents, cand_sents
        )
        min_sent_sim = (
            min(p["similarity"] for p in sent_pairs) if sent_pairs else 1.0
        )

        cand = CandidateScore(
            style=c["style"],
            text=text,
            ai_score=ai_score,
            sem_score=sem_score,
            length_ratio=length_ratio,
            min_sent_sim=min_sent_sim,
        )
        scored.append(cand)
        await emit_log(
            f"  [{c['style']:>12}]  AI {ai_score:.4f}  "
            f"sem {sem_score:.4f}  "
            f"len_r {length_ratio:.2f}  min_sim {min_sent_sim:.4f}"
        )

    await broadcast(
        {
            "type": "ensemble_candidates",
            "data": {
                "chunk_idx":  chunk_idx,
                "orig_ai":    round(orig_ai, 4),
                "candidates": [
                    {
                        "style":        c.style,
                        "ai_score":     round(c.ai_score, 4),
                        "sem_score":    round(c.sem_score, 4),
                        "length_ratio": round(c.length_ratio, 3),
                        "min_sent_sim": round(c.min_sent_sim, 4),
                    }
                    for c in scored
                ],
            },
        }
    )

    # ── Step 3: hard gate ─────────────────────────────────────────────────────

    _apply_hard_gate(scored, orig_ai)
    gate_pass = [c for c in scored if c.gate_pass]
    gate_fail = [c for c in scored if not c.gate_pass]

    for c in gate_fail:
        await emit_log(
            f"  [{c.style:>12}] GATE FAIL: {c.gate_fail_reason}",
            level="warn",
        )

    await emit_stage(
        "batch_score",
        "done" if gate_pass else "warn",
        f"{len(gate_pass)}/{len(scored)} candidates passed gate",
    )

    # ── Step 4: Pareto front + utility selection ──────────────────────────────

    if not gate_pass:
        # Fallback: no candidate improved on original within quality constraints
        await emit_log(
            "No candidate passed hard gate — reverting to original.", level="warn"
        )
        await emit_stage("pareto", "warn", "Fallback to original (no gate-pass candidates)")

        final_text        = chunk
        final_ai_score    = orig_ai
        final_sem_score   = 1.0
        reverted          = True
        selected_cand     = None
        status_label      = "Reverted to original"

    else:
        # Pareto ranks across all gate-passing candidates
        _compute_pareto_ranks(gate_pass)
        pareto_front = [c for c in gate_pass if c.pareto_rank == 0]

        # Utility: z-normalise across ALL gate-passing candidates for calibration
        _compute_utility(gate_pass, orig_ai, w_ai, w_sem, w_risk)

        # Select the Pareto-optimal candidate with highest utility
        selected_cand = max(pareto_front, key=lambda c: c.utility)

        await broadcast(
            {
                "type": "pareto_selection",
                "data": {
                    "chunk_idx":     chunk_idx,
                    "pareto_front":  [
                        {
                            "style":   c.style,
                            "utility": round(c.utility, 4),
                            "rank":    c.pareto_rank,
                        }
                        for c in gate_pass
                    ],
                    "selected_style":   selected_cand.style,
                    "selected_utility": round(selected_cand.utility, 4),
                },
            }
        )

        final_text      = selected_cand.text
        final_ai_score  = selected_cand.ai_score
        final_sem_score = selected_cand.sem_score
        reverted        = False
        status_label    = "Edited"

        await emit_stage(
            "pareto",
            "done",
            f"Selected '{selected_cand.style}'  "
            f"AI {final_ai_score:.4f}  sem {final_sem_score:.4f}",
        )
        await emit_log(
            f"Selected '{selected_cand.style}' (utility {selected_cand.utility:.4f}).  "
            f"AI {final_ai_score:.4f}  sem {final_sem_score:.4f}",
            level="success",
        )

    # Best candidate for display: the selected one, or (on fallback) lowest-AI scored candidate
    if selected_cand:
        _best_cand_payload = {
            "text":             selected_cand.text,
            "style":            selected_cand.style,
            "ai_score":         round(selected_cand.ai_score, 4),
            "sem_score":        round(selected_cand.sem_score, 4),
            "gate_fail_reason": "",
        }
    elif scored:
        _best = min(scored, key=lambda c: c.ai_score)
        _best_cand_payload = {
            "text":             _best.text,
            "style":            _best.style,
            "ai_score":         round(_best.ai_score, 4),
            "sem_score":        round(_best.sem_score, 4),
            "gate_fail_reason": _best.gate_fail_reason,
        }
    else:
        _best_cand_payload = None

    await broadcast(
        {
            "type": "chunk_done",
            "data": {
                "chunk_idx":            chunk_idx,
                "final_text":           final_text,
                "original_ai_score":    round(orig_ai, 4),
                "final_ai_score":       round(final_ai_score, 4),
                "final_sem_score":      round(final_sem_score, 4),
                "reverted_to_original": reverted,
                "status_label":         status_label,
                "selected_style":       selected_cand.style if selected_cand else "original",
                "best_candidate":       _best_cand_payload,
                "candidates": [
                    {
                        "style":            c.style,
                        "ai_score":         round(c.ai_score, 4),
                        "sem_score":        round(c.sem_score, 4),
                        "length_ratio":     round(c.length_ratio, 3),
                        "min_sent_sim":     round(c.min_sent_sim, 4),
                        "gate_pass":        c.gate_pass,
                        "gate_fail_reason": c.gate_fail_reason,
                        "pareto_rank":      c.pareto_rank,
                        "utility":          round(c.utility, 4),
                    }
                    for c in scored
                ],
            },
        }
    )

    return ChunkResult(
        chunk_idx=chunk_idx,
        original=chunk,
        orig_ai_score=orig_ai,
        candidates=scored,
        selected=selected_cand,
        final_text=final_text,
        final_ai_score=final_ai_score,
        final_sem_score=final_sem_score,
        reverted_to_original=reverted,
        status_label=status_label,
    )
