"""
Deterministic post-processing for V2 pipeline.

Handles:
- Punctuation fixes (em dash, semicolon) that LLM rewrites may miss
- Whitespace normalization
- Long-text segmentation for Stage 3
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.rule_detector import SentenceAnalysis


# ═══════════════════════════════════════════════════════════════════════════════
# Segment dataclass
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Segment:
    text: str                       # text to rewrite
    sentence_indices: List[int]     # corresponding SentenceAnalysis.idx values
    context_before: str             # read-only leading context
    context_after: str              # read-only trailing context
    word_count: int


# ═══════════════════════════════════════════════════════════════════════════════
# Punctuation fixes
# ═══════════════════════════════════════════════════════════════════════════════

# Em dash / en dash surrounded by optional spaces
_RE_DASH = re.compile(r"\s*[—–]\s*")
# Double-hyphen with spaces (not inside a word)
_RE_DOUBLE_HYPHEN = re.compile(r"\s+--\s+")


def fix_punctuation(text: str) -> str:
    """
    Replace all em dashes, en dashes, and double-hyphens with commas.
    Replace all semicolons with periods (capitalizing the next word).
    Preserves normal hyphens in compound words.
    """
    # 1. Em dash / en dash → comma
    text = _RE_DASH.sub(", ", text)

    # 2. Double-hyphen (space-surrounded) → comma
    text = _RE_DOUBLE_HYPHEN.sub(", ", text)

    # 3. Semicolons → period + capitalize
    def _semi_to_period(m: re.Match) -> str:
        after = m.group(1)
        return ". " + after.upper()

    text = re.sub(r";\s*([a-zA-Z])", _semi_to_period, text)
    # Catch trailing semicolons
    text = text.replace(";", ".")

    # Clean up artifacts: ", ," or double commas
    text = re.sub(r",\s*,", ",", text)
    # Double periods
    text = re.sub(r"\.(\s*\.)+", ".", text)

    return text


# ═══════════════════════════════════════════════════════════════════════════════
# Whitespace fixes
# ═══════════════════════════════════════════════════════════════════════════════

def fix_whitespace(text: str) -> str:
    """Normalize whitespace: collapse runs, strip trailing, limit blank lines."""
    # Collapse multiple spaces to one
    text = re.sub(r"[^\S\n]{2,}", " ", text)
    # Strip trailing whitespace per line
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    # Collapse 3+ consecutive newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


# ═══════════════════════════════════════════════════════════════════════════════
# Combined post-process
# ═══════════════════════════════════════════════════════════════════════════════

def run_post_process(text: str) -> Tuple[str, List[str]]:
    """
    Apply all deterministic fixes.

    Returns (fixed_text, log) where log lists what was changed.
    """
    log: List[str] = []
    original = text

    text = fix_punctuation(text)
    if text != original:
        log.append("Fixed punctuation (dashes/semicolons)")

    prev = text
    text = fix_whitespace(text)
    if text != prev:
        log.append("Normalized whitespace")

    return text, log


# ═══════════════════════════════════════════════════════════════════════════════
# Segmentation
# ═══════════════════════════════════════════════════════════════════════════════

def split_into_segments(
    text: str,
    sentences: "List[SentenceAnalysis]",
    target_words: int = 800,
    overlap_sentences: int = 2,
) -> List[Segment]:
    """
    Split text into segments of ~target_words, cutting at sentence boundaries.
    Each segment (except first/last) carries overlap context from neighbors.
    """
    if not sentences:
        return [Segment(
            text=text,
            sentence_indices=[],
            context_before="",
            context_after="",
            word_count=len(text.split()),
        )]

    total_words = sum(len(s.text.split()) for s in sentences)
    if total_words <= target_words:
        return [Segment(
            text=text,
            sentence_indices=[s.idx for s in sentences],
            context_before="",
            context_after="",
            word_count=total_words,
        )]

    # Greedy partition into segments
    partitions: List[List[int]] = []  # each is a list of sentence indices
    current: List[int] = []
    current_words = 0

    for sa in sentences:
        wc = len(sa.text.split())
        if current_words + wc > target_words and current:
            partitions.append(current)
            current = []
            current_words = 0
        current.append(sa.idx)
        current_words += wc

    if current:
        # Merge small tail into previous segment to avoid undersized last segment
        if len(partitions) > 0 and current_words < target_words * 0.4:
            partitions[-1].extend(current)
        else:
            partitions.append(current)

    # Build Segments with overlap context
    sent_by_idx = {s.idx: s for s in sentences}
    segments: List[Segment] = []

    for pi, part_indices in enumerate(partitions):
        seg_text = " ".join(sent_by_idx[i].text for i in part_indices)
        seg_words = len(seg_text.split())

        # Context before: last `overlap_sentences` of previous partition
        ctx_before = ""
        if pi > 0:
            prev = partitions[pi - 1]
            overlap_idx = prev[-overlap_sentences:]
            ctx_before = " ".join(sent_by_idx[i].text for i in overlap_idx)

        # Context after: first `overlap_sentences` of next partition
        ctx_after = ""
        if pi < len(partitions) - 1:
            nxt = partitions[pi + 1]
            overlap_idx = nxt[:overlap_sentences]
            ctx_after = " ".join(sent_by_idx[i].text for i in overlap_idx)

        segments.append(Segment(
            text=seg_text,
            sentence_indices=part_indices,
            context_before=ctx_before,
            context_after=ctx_after,
            word_count=seg_words,
        ))

    return segments
