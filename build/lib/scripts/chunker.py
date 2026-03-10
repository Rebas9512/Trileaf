"""
Text chunking utilities for the writing optimizer pipeline.

Strategy:
  1. Split by natural paragraphs (double newline).
  2. If no paragraphs exist, split by sentences grouping up to MAX_CHUNK_CHARS.
  3. Oversized paragraphs are sub-split at sentence boundaries.
  4. split_finer() provides progressively smaller chunks for AI-detection retries.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List

def clean_text(text: str) -> str:
    """
    Lightweight sanitization applied before chunking.

    Removes encoding artifacts and pathological whitespace that would confuse
    downstream models.  Normal prose — including smart quotes, em dashes,
    emojis, and non-ASCII scripts — is left completely unchanged.

    Steps (in order):
    1. Unicode NFC normalization (fix combining-character sequences).
    2. Strip zero-width / invisible format characters (BOM, ZWSP, ZWJ, …).
    3. Remove Unicode replacement character U+FFFD (garbled-byte marker).
    4. Remove ASCII control characters (keep tab, newline, carriage-return).
    5. Normalize line endings to \\n.
    6. Strip trailing whitespace from every line.
    7. Collapse 3+ consecutive blank lines to 2 (one paragraph break).
    8. Collapse runs of 2+ horizontal whitespace chars to a single space.
    9. Cap 5+ consecutive identical punctuation chars to 3
       (e.g. "!!!!!" → "!!!", "....." → "...").
    """
    # 1. NFC normalization
    text = unicodedata.normalize("NFC", text)

    # 2. Zero-width / invisible Unicode format chars
    #    U+FEFF BOM · U+200B ZWSP · U+200C ZWNJ · U+200D ZWJ
    #    U+00AD soft-hyphen · U+2028 line-sep · U+2029 para-sep
    text = re.sub(r"[\ufeff\u200b\u200c\u200d\u00ad\u2028\u2029]", "", text)

    # 3. Encoding-error marker
    text = text.replace("\ufffd", "")

    # 4. ASCII control chars (keep \x09=tab, \x0a=LF, \x0d=CR)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # 5. Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 6. Trailing whitespace per line
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    # 7. Collapse 3+ blank lines → 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 8. Collapse multiple horizontal whitespace to single space
    text = re.sub(r"[ \t]{2,}", " ", text)

    # 9. Cap egregiously repeated identical punctuation (5+ → 3)
    text = re.sub(r"([!?.,;:\-~])\1{4,}", lambda m: m.group(1) * 3, text)

    return text.strip()


# Sentence-ending boundaries (English + Chinese punctuation)
_SENT_BOUNDARY = re.compile(
    r"(?<=[.!?。！？])\s+"
    r"|(?<=[.!?。！？])(?=[A-Z\"\u4e00-\u9fff])"
)


def split_text(text: str, max_chunk_chars: int = 200) -> List[str]:
    """
    Main entry point: split *text* into chunks suitable for the pipeline.

    Rules
    -----
    - If text has natural paragraphs (double newline), use them as chunk
      boundaries.  Oversized paragraphs are further split by sentences.
    - If text has no clear paragraph structure, split by sentences grouping
      up to *max_chunk_chars* characters; prefer cuts at punctuation marks.
    - Text shorter than *max_chunk_chars* is returned as a single chunk.
    """
    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    if len(paragraphs) <= 1:
        return _sentence_group_split(text, max_chunk_chars)

    result: List[str] = []
    for para in paragraphs:
        if len(para) > max_chunk_chars:
            result.extend(_sentence_group_split(para, max_chunk_chars))
        else:
            result.append(para)
    return result


def split_sentences(text: str) -> List[str]:
    """Split text into individual sentences (English + Chinese aware)."""
    parts = _SENT_BOUNDARY.split(text.strip())
    out = [p.strip() for p in parts if p.strip()]
    return out if out else [text.strip()]


def split_finer(text: str, level: int) -> List[str]:
    """
    Return progressively finer sub-chunks for the AI-detection retry loop.

    level 1 → 2-sentence groups
    level 2 → individual sentences
    level 3 → individual sentences (same granularity; orchestrator uses a
               more aggressive Qwen prompt at this level)
    """
    sents = split_sentences(text)
    if len(sents) <= 1 or level <= 0:
        return [text]

    if level == 1:
        groups: List[str] = []
        for i in range(0, len(sents), 2):
            groups.append(" ".join(sents[i : i + 2]))
        return groups

    # level >= 2 → sentence-level
    return sents


# ─── Internal helpers ──────────────────────────────────────────────────────────


def _sentence_group_split(text: str, max_chars: int) -> List[str]:
    """Group sentences into chunks not exceeding *max_chars*."""
    if len(text) <= max_chars:
        return [text]

    sents = split_sentences(text)
    if len(sents) <= 1:
        return _hard_split(text, max_chars)

    chunks: List[str] = []
    current = ""
    for sent in sents:
        candidate = (current + " " + sent).strip() if current else sent
        if len(candidate) > max_chars and current:
            chunks.append(current.strip())
            current = sent
        else:
            current = candidate
    if current:
        chunks.append(current.strip())
    return chunks or [text]


def _hard_split(text: str, max_chars: int) -> List[str]:
    """Last-resort split: cut at the last space before *max_chars*."""
    chunks: List[str] = []
    while len(text) > max_chars:
        cut = text.rfind(" ", 0, max_chars)
        if cut <= 0:
            cut = max_chars
        chunks.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        chunks.append(text)
    return chunks
