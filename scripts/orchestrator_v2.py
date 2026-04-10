"""
V2 Pipeline orchestrator — five-stage progressive architecture.

Stage 1: Global scoring (detector + rule analysis)
Stage 2: Per-sentence tagging (dual-signal fusion)
Stage 3: Standardized rewrite (guided by violation annotations)
Stage 4: Stubborn sentence attack (rewrite / delete / unfixed_risky)
Stage 5: Human touch polish (technique budget + deficit-aware)
"""

from __future__ import annotations

import asyncio
import logging
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from scripts import rule_detector as rd
from scripts import prompt_builder as pb
from scripts import post_processor as pp
from scripts.detector_interface import BaseDetector, DesklibDetector
from scripts.rule_detector import SentenceAnalysis, DocumentAnalysis

_log = logging.getLogger(__name__)

# Thresholds (can be overridden via config later)
_MODEL_USEFUL_MIN_STD = 0.03
_STAGE4_REWRITE_SEM_GATE = 0.65
_STAGE4_DELETE_PARA_SEM_GATE = 0.85
_STAGE4_RELATIVE_Z = 0.5
_STAGE3_SHORT_MAX_WORDS = 1000


# ═══════════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SentenceResult:
    idx: int
    original: str
    rule_severity: str
    ai_score: float
    ai_z_score: float
    severity: str
    flags: List[str]
    after_stage3: str
    after_stage4: Optional[str] = None
    after_stage4_action: str = "kept"
    final_text: str = ""
    stage4_sem_score: Optional[float] = None


@dataclass
class StageMetrics:
    ai_score: float
    violation_count: int
    sem_score: float


@dataclass
class PipelineV2Result:
    run_id: str
    input_text: str
    output_text: str
    sentences: List[SentenceResult]
    stage_metrics: Dict[str, StageMetrics]
    original_ai_score: float
    final_ai_score: float
    final_sem_score: float
    unfixed_sentences: List[int]


# ═══════════════════════════════════════════════════════════════════════════════
# Dependency accessors (patchable in tests)
# ═══════════════════════════════════════════════════════════════════════════════

_detector_instance: Optional[BaseDetector] = None
_rewriter_instance: Optional[Any] = None
_similarity_instance: Optional[Any] = None


def _get_detector():
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = DesklibDetector()
    return _detector_instance


def _get_rewriter():
    """Returns an object with .run_rewrite_with_prompt(text, prompt, temperature)."""
    global _rewriter_instance
    if _rewriter_instance is None:
        from scripts import models_runtime as mr

        class _DefaultRewriter:
            def __init__(self):
                self.calls = []

            def run_rewrite_with_prompt(self, text, prompt, temperature=0.7):
                self.calls.append((text, prompt, temperature))
                return mr.run_rewrite_with_prompt(text, prompt, temperature)

        _rewriter_instance = _DefaultRewriter()
    return _rewriter_instance


def _get_similarity():
    """Returns an object with .run_mpnet_similarity(text1, text2)."""
    global _similarity_instance
    if _similarity_instance is None:
        from scripts import models_runtime as mr

        class _DefaultSimilarity:
            def run_mpnet_similarity(self, text1, text2):
                return mr.run_mpnet_similarity(text1, text2)

        _similarity_instance = _DefaultSimilarity()
    return _similarity_instance


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 4 decision function
# ═══════════════════════════════════════════════════════════════════════════════

def _decide_stage4_action(
    candidate_text: str,
    candidate_sem: float,
    candidate_rules_pass: bool,
    para_sem_without: Optional[float],
) -> str:
    """
    Decide action for a stubborn sentence in Stage 4.

    Returns: "rewritten" | "deleted" | "unfixed_risky"
    """
    if candidate_sem >= _STAGE4_REWRITE_SEM_GATE and candidate_rules_pass:
        return "rewritten"

    if para_sem_without is not None and para_sem_without >= _STAGE4_DELETE_PARA_SEM_GATE:
        return "deleted"

    return "unfixed_risky"


# ═══════════════════════════════════════════════════════════════════════════════
# Sentence splitting helper
# ═══════════════════════════════════════════════════════════════════════════════

def _split_and_align(original_sentences: List[SentenceAnalysis], new_text: str) -> List[str]:
    """
    Split rewritten text back into sentences, aligned 1:1 with original.
    If counts don't match, fall back to simple splitting.
    """
    new_sents = rd._split_sentences(new_text)
    orig_count = len(original_sentences)

    if len(new_sents) == orig_count:
        return new_sents

    # Fallback: if more new sentences than original, merge extras into last
    if len(new_sents) > orig_count:
        merged = new_sents[:orig_count - 1]
        merged.append(" ".join(new_sents[orig_count - 1:]))
        return merged

    # If fewer new sentences, pad with empty strings
    return new_sents + [""] * (orig_count - len(new_sents))


# ═══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════════

async def run_pipeline_v2(
    text: str,
    broadcast: Callable,
    run_id: str,
    chunk_mode: str = "short",
) -> PipelineV2Result:
    """Execute the five-stage V2 pipeline."""

    detector = _get_detector()
    rewriter = _get_rewriter()
    similarity = _get_similarity()

    stage_metrics: Dict[str, StageMetrics] = {}

    # Helper: run blocking function off the event loop
    _t = asyncio.to_thread

    # ── Stage 1: Global scoring ─────────────────────────────────────────────
    await broadcast("stage_start", {"stage": 1, "name": "Global scoring"})

    doc_analysis = await _t(rd.analyze_document, text)
    overall_result = await _t(detector.score_text, text)
    overall_score = overall_result.score

    # Per-sentence AI scores
    sentence_scores = await _t(detector.score_batch, [s.text for s in doc_analysis.sentences])
    scores = [r.score for r in sentence_scores]
    doc_mean = statistics.mean(scores) if scores else 0.0
    doc_std = statistics.stdev(scores) if len(scores) > 1 else 0.0
    model_useful = doc_std >= _MODEL_USEFUL_MIN_STD

    doc_analysis.ai_score = overall_score
    doc_analysis.doc_ai_mean = doc_mean
    doc_analysis.doc_ai_std = doc_std
    doc_analysis.model_useful = model_useful

    s1_violations = sum(len(s.violations) for s in doc_analysis.sentences)

    # Genre detection via LLM
    genre_prompt = pb.build_genre_detect_prompt(text)
    try:
        genre_raw = await _t(rewriter.run_rewrite_with_prompt, "", genre_prompt, 0.1)
        genre = pb.parse_genre_response(genre_raw)
    except Exception:
        genre = "general"
    _log.info("Detected genre: %s", genre)

    stage_metrics["stage1"] = StageMetrics(
        ai_score=overall_score,
        violation_count=s1_violations,
        sem_score=1.0,
    )

    await broadcast("stage1_done", {
        "ai_score": overall_score,
        "violation_count": s1_violations,
        "summary": doc_analysis.summary,
        "top_issues": doc_analysis.top_issues,
        "model_useful": model_useful,
        "doc_mean": doc_mean,
        "doc_std": doc_std,
        "genre": genre,
    })

    # ── Stage 2: Per-sentence tagging ───────────────────────────────────────
    await broadcast("stage_start", {"stage": 2, "name": "Per-sentence tagging"})

    for sa, result in zip(doc_analysis.sentences, sentence_scores):
        sa.ai_score = result.score
        sa.ai_z_score = ((result.score - doc_mean) / doc_std) if doc_std >= _MODEL_USEFUL_MIN_STD else 0.0
        sa.severity = rd.fuse_severity(sa.rule_severity, sa.ai_z_score, model_useful)

    for sa in doc_analysis.sentences:
        await broadcast("sentence_tagged", {
            "idx": sa.idx,
            "text": sa.text,
            "severity": sa.severity,
            "rule_severity": sa.rule_severity,
            "ai_score": round(sa.ai_score, 4),
            "ai_z_score": round(sa.ai_z_score, 2),
            "flags": [v.rule_id for v in sa.violations],
        })

    # Recount summary after fusion
    severity_dist = {}
    for sa in doc_analysis.sentences:
        severity_dist[sa.severity] = severity_dist.get(sa.severity, 0) + 1

    await broadcast("stage2_done", {
        "total_sentences": len(doc_analysis.sentences),
        "severity_distribution": severity_dist,
        "model_useful": model_useful,
    })

    # ── Determine rewrite intensity based on how "AI" the text already is ──
    clean_ratio = sum(1 for sa in doc_analysis.sentences if sa.rule_severity == "clean") / max(len(doc_analysis.sentences), 1)
    already_human = overall_score < 0.35 or (overall_score < 0.45 and clean_ratio >= 0.70)

    # ── Stage 3: Standardized rewrite (with retry loop) ──────────────────
    await broadcast("stage_start", {"stage": 3, "name": "Standardized rewrite"})

    async def _do_stage3_single(source_text, sents, temp, is_retry=False):
        """Run one Stage 3 rewrite at a given temperature."""
        prompt = pb.build_stage3_prompt(
            sents,
            overall_ai_score=overall_score,
            is_retry=is_retry,
            genre=genre,
        )
        wc = len(source_text.split())
        if wc <= _STAGE3_SHORT_MAX_WORDS:
            return await _t(rewriter.run_rewrite_with_prompt, source_text, prompt, temp)
        else:
            segments = pp.split_into_segments(source_text, sents)
            parts = []
            for si, seg in enumerate(segments):
                seg_sents = [s for s in sents if s.idx in seg.sentence_indices]
                seg_prompt = pb.build_stage3_prompt(
                    seg_sents,
                    overall_ai_score=overall_score,
                    is_retry=is_retry,
                    genre=genre,
                )
                part = await _t(rewriter.run_rewrite_with_prompt, seg.text, seg_prompt, temp)
                parts.append(part)
                await broadcast("stage3_progress", {
                    "segment_idx": si,
                    "total_segments": len(segments),
                    "status": "done",
                })
            return " ".join(parts)

    async def _do_stage3_multi_candidate(source_text, sents, is_retry=False):
        """Generate 2 candidates at different temperatures, pick the one with lower AI score."""
        temps = [0.85, 0.65] if is_retry else [0.7, 0.9]
        source_words = len(source_text.split())
        candidates = []
        for ti, temp in enumerate(temps):
            try:
                raw = await _do_stage3_single(source_text, sents, temp, is_retry)
                cleaned, _ = pp.run_post_process(raw)
                # Length guard: reject if output shrunk/grew too much (>40% deviation)
                out_words = len(cleaned.split())
                if out_words < 5 or (source_words > 20 and abs(out_words - source_words) / source_words > 0.40):
                    _log.warning("Stage 3 candidate temp=%.2f rejected: length %d→%d (%.0f%% deviation)",
                                 temp, source_words, out_words, abs(out_words - source_words) / source_words * 100)
                    continue
                score = (await _t(detector.score_text, cleaned)).score
                sem = await _t(similarity.run_mpnet_similarity, text, cleaned)
                candidates.append({"text": cleaned, "ai_score": score, "sem": sem, "temp": temp})
            except Exception as exc:
                _log.warning("Stage 3 candidate at temp=%.2f failed: %s", temp, exc)
        if not candidates:
            return source_text, overall_score  # fallback to original
        # Filter: semantic similarity must be reasonable (>0.5)
        viable = [c for c in candidates if c["sem"] > 0.5] or candidates
        best = min(viable, key=lambda c: c["ai_score"])
        _log.info(
            "Stage 3 candidates: %s → selected temp=%.2f (AI=%.3f, sem=%.2f)",
            [(c["temp"], round(c["ai_score"], 3)) for c in candidates],
            best["temp"], best["ai_score"], best["sem"],
        )
        return best["text"], best["ai_score"]

    # Light-touch mode for already-human text
    if already_human:
        _log.info("Text appears already human (AI=%.2f, clean_ratio=%.0f%%), using light-touch mode", overall_score, clean_ratio * 100)
        # Only do a single pass at low temperature, no retry
        try:
            rewritten = await _do_stage3_single(text, doc_analysis.sentences, 0.5, is_retry=False)
            rewritten, _ = pp.run_post_process(rewritten)
            out_words = len(rewritten.split())
            in_words = len(text.split())
            if out_words < 5 or (in_words > 20 and abs(out_words - in_words) / in_words > 0.40):
                _log.warning("Light-touch rewrite rejected (length %d→%d), keeping original", in_words, out_words)
                rewritten = text
        except Exception:
            rewritten = text
        s3_result = await _t(detector.score_text, rewritten)
        s3_score = s3_result.score
    else:
        # Normal mode: multi-candidate
        rewritten, s3_score = await _do_stage3_multi_candidate(
            text, doc_analysis.sentences, is_retry=False
        )

    improvement_pct = (overall_score - s3_score) / max(overall_score, 0.01)

    # Retry if improvement < 15% and original score was meaningful (skip retry for already-human)
    if not already_human and improvement_pct < 0.15 and overall_score > 0.25:
        await broadcast("stage3_progress", {
            "segment_idx": 0,
            "total_segments": 1,
            "status": "retry",
        })
        retry_analysis = await _t(rd.analyze_document, rewritten)
        # Feed AI scores from first pass into retry sentences
        retry_scores = await _t(detector.score_batch, [s.text for s in retry_analysis.sentences])
        for sa, rs in zip(retry_analysis.sentences, retry_scores):
            sa.ai_score = rs.score

        rewritten, s3_score = await _do_stage3_multi_candidate(
            rewritten, retry_analysis.sentences, is_retry=True
        )

    # Post-rewrite analysis
    post_analysis = await _t(rd.analyze_document, rewritten)
    s3_sem = await _t(similarity.run_mpnet_similarity, text, rewritten)
    s3_violations = sum(len(s.violations) for s in post_analysis.sentences)

    stage_metrics["stage3"] = StageMetrics(
        ai_score=s3_score,
        violation_count=s3_violations,
        sem_score=s3_sem,
    )

    # Align rewritten sentences with original
    rewritten_sents = _split_and_align(doc_analysis.sentences, rewritten)

    await broadcast("stage3_done", {
        "ai_score": s3_score,
        "violation_count": s3_violations,
        "violation_delta": s3_violations - s1_violations,
        "sem_score": s3_sem,
    })

    # ── Stage 4: Stubborn sentence attack ───────────────────────────────────
    await broadcast("stage_start", {"stage": 4, "name": "Stubborn sentence attack"})

    # Build sentence results
    sentence_results: List[SentenceResult] = []
    current_text_parts: List[str] = []  # builds the running text
    unfixed_indices: List[int] = []
    s4_rewritten = 0
    s4_deleted = 0
    s4_unfixed = 0

    for sa in doc_analysis.sentences:
        rw_text = rewritten_sents[sa.idx] if sa.idx < len(rewritten_sents) else sa.text

        sr = SentenceResult(
            idx=sa.idx,
            original=sa.text,
            rule_severity=sa.rule_severity,
            ai_score=sa.ai_score,
            ai_z_score=sa.ai_z_score,
            severity=sa.severity,
            flags=[v.rule_id for v in sa.violations],
            after_stage3=rw_text,
        )

        # Check if this sentence still has issues after Stage 3
        if sa.idx < len(post_analysis.sentences):
            post_sa = post_analysis.sentences[sa.idx]
            still_flagged = post_sa.rule_severity in ("critical", "high")
        else:
            still_flagged = sa.severity in ("critical", "high")

        if still_flagged:
            # Try single-sentence rewrite
            ctx_before = rewritten_sents[sa.idx - 1] if sa.idx > 0 and sa.idx - 1 < len(rewritten_sents) else ""
            ctx_after = rewritten_sents[sa.idx + 1] if sa.idx + 1 < len(rewritten_sents) else ""
            s4_prompt = pb.build_stage4_prompt(
                sentence=rw_text,
                violations=sa.violations,
                context_before=ctx_before,
                context_after=ctx_after,
                ai_score=sa.ai_score,
            )
            candidate = await _t(rewriter.run_rewrite_with_prompt, rw_text, s4_prompt, 0.85)
            cand_sem = await _t(similarity.run_mpnet_similarity, sa.text, candidate)
            cand_analysis = rd.analyze_sentence(candidate)
            cand_rules_pass = cand_analysis.rule_severity in ("clean", "low")

            # Try deletion: compute paragraph similarity without this sentence
            para_without = " ".join(
                rewritten_sents[j]
                for j in range(len(rewritten_sents))
                if j != sa.idx and j < len(rewritten_sents) and rewritten_sents[j]
            )
            para_sem = (await _t(similarity.run_mpnet_similarity, text, para_without)) if para_without else 0.0

            action = _decide_stage4_action(candidate, cand_sem, cand_rules_pass, para_sem)

            sr.stage4_sem_score = cand_sem

            if action == "rewritten":
                sr.after_stage4 = candidate
                sr.after_stage4_action = "rewritten"
                sr.final_text = candidate
                s4_rewritten += 1
            elif action == "deleted":
                sr.after_stage4 = ""
                sr.after_stage4_action = "deleted"
                sr.final_text = ""
                s4_deleted += 1
            else:
                sr.after_stage4_action = "unfixed_risky"
                sr.final_text = rw_text
                unfixed_indices.append(sa.idx)
                s4_unfixed += 1

            await broadcast("stage4_sentence", {
                "idx": sa.idx,
                "action": action,
                "sem_score": cand_sem,
                "detail": f"rule_severity={cand_analysis.rule_severity}",
            })
        else:
            sr.after_stage4_action = "kept"
            sr.final_text = rw_text

        sentence_results.append(sr)
        if sr.final_text:
            current_text_parts.append(sr.final_text)

    s4_text = " ".join(current_text_parts)
    s4_score = (await _t(detector.score_text, s4_text)).score if s4_text else s3_score
    s4_sem = (await _t(similarity.run_mpnet_similarity, text, s4_text)) if s4_text else s3_sem

    stage_metrics["stage4"] = StageMetrics(
        ai_score=s4_score,
        violation_count=s3_violations,  # approximate
        sem_score=s4_sem,
    )

    await broadcast("stage4_done", {
        "rewritten": s4_rewritten,
        "deleted": s4_deleted,
        "unfixed": s4_unfixed,
        "ai_score": s4_score,
    })

    # ── Stage 5: Human touch polish ─────────────────────────────────────────
    await broadcast("stage_start", {"stage": 5, "name": "Human touch polish"})

    # For already-human text, use lower AI score for budget calculation to avoid over-processing
    s5_ai_for_budget = s4_score if not already_human else max(0.0, s4_score - 0.15)
    budget = pb.compute_technique_budget(len(s4_text.split()), current_ai_score=s5_ai_for_budget)

    # Check human deficit
    human_analysis = await _t(rd.analyze_document, s4_text)
    human_deficit = []
    for sa in human_analysis.sentences:
        for v in sa.violations:
            if v.rule_id.startswith("human.") and v.rule_id not in human_deficit:
                human_deficit.append(v.rule_id)

    s5_prompt = pb.build_stage5_prompt(s4_text, budget, human_deficit, current_ai_score=s4_score, genre=genre)
    s5_temp = 0.8 if s4_score > 0.35 else 0.7  # higher temp when score is still high
    polished = await _t(rewriter.run_rewrite_with_prompt, s4_text, s5_prompt, s5_temp)
    polished, _ = pp.run_post_process(polished)

    final_ai = (await _t(detector.score_text, polished)).score
    final_sem = await _t(similarity.run_mpnet_similarity, text, polished)

    final_doc = await _t(rd.analyze_document, polished)
    stage_metrics["stage5"] = StageMetrics(
        ai_score=final_ai,
        violation_count=sum(len(s.violations) for s in final_doc.sentences),
        sem_score=final_sem,
    )

    await broadcast("stage5_done", {
        "ai_score": final_ai,
        "sem_score": final_sem,
        "techniques_used": budget,
    })

    # ── Stage 6: Formality calibration (conditional) ───────────────────────
    output_text = polished
    if not pb.needs_formality_pass(genre):
        await broadcast("stage6_skipped", {
            "genre": genre,
            "reason": f"Not required for {genre} genre",
        })
    else:
        await broadcast("stage_start", {"stage": 6, "name": "Formality calibration"})

        s6_prompt = pb.build_stage6_prompt(polished, genre)
        calibrated = await _t(rewriter.run_rewrite_with_prompt, polished, s6_prompt, 0.5)
        calibrated, _ = pp.run_post_process(calibrated)

        s6_ai = (await _t(detector.score_text, calibrated)).score
        s6_sem = await _t(similarity.run_mpnet_similarity, text, calibrated)

        # Only accept if formality pass didn't make AI score significantly worse
        if s6_ai <= final_ai + 0.05:
            output_text = calibrated
            final_ai = s6_ai
            final_sem = s6_sem
            _log.info("Stage 6 accepted: AI %.3f → %.3f, sem %.2f", final_ai, s6_ai, s6_sem)
        else:
            _log.info("Stage 6 rejected: AI would increase %.3f → %.3f, keeping Stage 5 output", final_ai, s6_ai)

        stage_metrics["stage6"] = StageMetrics(
            ai_score=final_ai,
            violation_count=sum(len(s.violations) for s in (await _t(rd.analyze_document, output_text)).sentences),
            sem_score=final_sem,
        )

        await broadcast("stage6_done", {
            "ai_score": final_ai,
            "sem_score": final_sem,
            "accepted": output_text == calibrated,
            "genre": genre,
        })

    # Update stage5 metrics to reflect final state
    stage_metrics["stage5"] = StageMetrics(
        ai_score=final_ai,
        violation_count=sum(len(s.violations) for s in final_doc.sentences),
        sem_score=final_sem,
    )

    # Update final_text on sentence results
    final_sents = rd._split_sentences(output_text)
    for i, sr in enumerate(sentence_results):
        if i < len(final_sents):
            sr.final_text = final_sents[i]

    # ── Final result ────────────────────────────────────────────────────────
    result = PipelineV2Result(
        run_id=run_id,
        input_text=text,
        output_text=output_text,
        sentences=sentence_results,
        stage_metrics=stage_metrics,
        original_ai_score=overall_score,
        final_ai_score=final_ai,
        final_sem_score=final_sem,
        unfixed_sentences=unfixed_indices,
    )

    # Collect unfixed sentence texts for frontend highlighting
    unfixed_texts = []
    for sr in sentence_results:
        if sr.after_stage4_action == "unfixed_risky" and sr.final_text:
            unfixed_texts.append(sr.final_text)

    await broadcast("run_done_v2", {
        "run_id": run_id,
        "output": output_text,
        "original_ai_score": overall_score,
        "final_ai_score": final_ai,
        "final_sem_score": final_sem,
        "unfixed_sentences": unfixed_indices,
        "unfixed_texts": unfixed_texts,
    })

    return result
