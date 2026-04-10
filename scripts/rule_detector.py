"""
Rule-based AI writing trace detector.

Pure deterministic detection (regex + word lists, no ML models, no spaCy).
This module is the "eyes" of the V2 pipeline — Stages 1, 2, 3-verify,
and 4 all depend on it.

Rule categories
───────────────
A. Punctuation     — em dash, semicolon, colon overuse
B. Structure       — not-X-but-Y, triple parallel, abstract noun stack
C. Vocabulary      — high-risk transitions, cliche phrases, AI filler words
D. Syntax          — uniform sentence rhythm, passive voice stacking
E. Human deficit   — missing colloquial markers, short sentences, questions
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Violation:
    rule_id: str                    # e.g. "punct.em_dash"
    severity: str                   # "critical" | "high" | "medium" | "low"
    span: Tuple[int, int]           # char offset within the sentence
    context: str                    # snippet around the violation
    suggestion: str                 # brief fix hint for prompt injection


@dataclass
class SentenceAnalysis:
    idx: int
    text: str
    violations: List[Violation]
    rule_severity: str              # severity from rules alone
    ai_score: float = 0.0          # filled by orchestrator
    ai_z_score: float = 0.0        # filled by orchestrator
    severity: str = ""             # fused severity; defaults to rule_severity

    def __post_init__(self):
        if not self.severity:
            self.severity = self.rule_severity


@dataclass
class DocumentAnalysis:
    sentences: List[SentenceAnalysis]
    summary: Dict[str, int]         # severity → count
    top_issues: List[str]           # most frequent rule_ids
    ai_score: float = 0.0          # filled by caller
    doc_ai_mean: float = 0.0
    doc_ai_std: float = 0.0
    model_useful: bool = False


# ═══════════════════════════════════════════════════════════════════════════════
# Word lists & patterns
# ═══════════════════════════════════════════════════════════════════════════════

# ── A-class: Punctuation ────────────────────────────────────────────────────

# Match em dash, en dash, or double-hyphen that is NOT inside a compound word.
# A compound-word hyphen has \w on both sides with no spaces.
_RE_EM_DASH = re.compile(r"—|–|(?<!\w)--(?!\w)|(?<=\s)--(?=\s)")
_RE_SEMICOLON = re.compile(r";")
# Colon not at end-of-sentence (list intro usually ends the sentence / line)
_RE_COLON_OVERUSE = re.compile(r":(?=\s+[a-z])")

# ── B-class: Structure ──────────────────────────────────────────────────────

_RE_NOT_X_BUT_Y = re.compile(
    r"\bnot\s+(?:to\s+)?[\w\s,]+,\s*but\s+(?:to\s+)?[\w]"
    r"|\brather\s+than\s+[\w]"
    r"|\bnot\s+because\b",
    re.IGNORECASE,
)

_RE_TRIPLE_PARALLEL = re.compile(
    r"(?:no|the|a|an|its|their|our|my|his|her)?\s*[\w\s-]{3,40},"
    r"\s*(?:no|the|a|an|its|their|our|my|his|her)?\s*[\w\s-]{3,40},"
    r"\s*and\s+(?:no|the|a|an|its|their|our|my|his|her)?\s*[\w\s-]{3,40}",
    re.IGNORECASE,
)

_ABSTRACT_NOUNS = {
    "legitimacy", "alignment", "framework", "mechanism", "optimization",
    "outcomes", "methodology", "implementation", "infrastructure",
    "paradigm", "dimension", "trajectory", "intersection", "synergy",
    "resilience", "governance", "sustainability", "accountability",
    "transparency", "stakeholder", "facilitation", "utilization",
    "prioritization", "transformation", "integration", "complexity",
    "foundation", "effectiveness", "efficiency", "productivity",
    "capability", "functionality", "architecture", "ecosystem",
    "landscape", "narrative", "discourse", "implication",
    "suboptimal", "inherently",
}

# ── C-class: Vocabulary ─────────────────────────────────────────────────────

_HIGH_RISK_TRANSITIONS = [
    "however", "furthermore", "moreover", "that said", "in essence",
    "fundamentally", "ultimately", "indeed", "notably",
]

_CLICHE_PHRASES = [
    "fresh in everyone's memory", "fresh in memory",
    "by default", "there wasn't much to go on",
    "walked away feeling", "in practice",
    "constructed metric", "empirical foundation",
    "a nuanced approach", "at its core",
    "navigate the complexities of", "navigate the complexities",
]

_AI_FILLER_WORDS = [
    "delve into", "delve", "nuanced", "multifaceted", "pivotal",
    "comprehensive", "notably", "it is worth noting",
    "it is important to", "in today's world", "in conclusion",
    "it's crucial",
]

# ── E-class: Human deficit ──────────────────────────────────────────────────

_COLLOQUIAL_MARKERS = [
    "actually", "just", "pretty much", "kind of", "sort of",
    "really", "a whole lot", "at least", "honestly", "to be fair",
    "sure", "though", "well",
]

# ═══════════════════════════════════════════════════════════════════════════════
# Sentence splitting
# ═══════════════════════════════════════════════════════════════════════════════

_RE_SENT_SPLIT = re.compile(
    r'(?<=[.!?])\s+'           # standard sentence boundary
    r'|(?<=[.!?])\n+'          # newline after terminal punct
)


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences. Simple regex-based, no ML."""
    raw = _RE_SENT_SPLIT.split(text.strip())
    return [s.strip() for s in raw if s.strip()]


# ═══════════════════════════════════════════════════════════════════════════════
# Individual rule checkers
# ═══════════════════════════════════════════════════════════════════════════════

def _check_punct_em_dash(text: str) -> List[Violation]:
    violations = []
    for m in _RE_EM_DASH.finditer(text):
        violations.append(Violation(
            rule_id="punct.em_dash",
            severity="critical",
            span=(m.start(), m.end()),
            context=text[max(0, m.start() - 15):m.end() + 15],
            suggestion="Replace dash with comma, period, or parentheses",
        ))
    return violations


def _check_punct_semicolon(text: str) -> List[Violation]:
    violations = []
    for m in _RE_SEMICOLON.finditer(text):
        violations.append(Violation(
            rule_id="punct.semicolon",
            severity="medium",
            span=(m.start(), m.end()),
            context=text[max(0, m.start() - 15):m.end() + 15],
            suggestion="Replace semicolon with period",
        ))
    return violations


def _check_punct_colon_overuse(text: str) -> List[Violation]:
    violations = []
    for m in _RE_COLON_OVERUSE.finditer(text):
        violations.append(Violation(
            rule_id="punct.colon_overuse",
            severity="medium",
            span=(m.start(), m.end()),
            context=text[max(0, m.start() - 15):m.end() + 15],
            suggestion="Remove colon or rephrase; reserve colons for list introductions",
        ))
    return violations


def _check_struct_not_x_but_y(text: str) -> List[Violation]:
    violations = []
    for m in _RE_NOT_X_BUT_Y.finditer(text):
        violations.append(Violation(
            rule_id="struct.not_x_but_y",
            severity="high",
            span=(m.start(), m.end()),
            context=text[max(0, m.start() - 10):m.end() + 10],
            suggestion="Break the symmetric opposition into two separate statements",
        ))
    return violations


def _check_struct_triple_parallel(text: str) -> List[Violation]:
    violations = []
    for m in _RE_TRIPLE_PARALLEL.finditer(text):
        violations.append(Violation(
            rule_id="struct.triple_parallel",
            severity="high",
            span=(m.start(), m.end()),
            context=text[max(0, m.start() - 5):m.end() + 5],
            suggestion="Break the three-item list into separate sentences",
        ))
    return violations


def _check_struct_abstract_noun_stack(text: str) -> List[Violation]:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    found = [w for w in words if w in _ABSTRACT_NOUNS]
    if len(found) >= 3:
        return [Violation(
            rule_id="struct.abstract_noun_stack",
            severity="high",
            span=(0, len(text)),
            context=f"Abstract nouns: {', '.join(found[:5])}",
            suggestion="Use clauses and pronouns to break up abstract noun density",
        )]
    return []


def _check_vocab_high_risk_transition(text: str) -> List[Violation]:
    violations = []
    text_lower = text.lower()
    for phrase in _HIGH_RISK_TRANSITIONS:
        pattern = re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE)
        for m in pattern.finditer(text):
            violations.append(Violation(
                rule_id="vocab.high_risk_transition",
                severity="high",
                span=(m.start(), m.end()),
                context=text[max(0, m.start() - 10):m.end() + 10],
                suggestion=f"Remove or replace '{phrase}'",
            ))
    return violations


def _check_vocab_cliche_phrase(text: str) -> List[Violation]:
    violations = []
    for phrase in _CLICHE_PHRASES:
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        for m in pattern.finditer(text):
            violations.append(Violation(
                rule_id="vocab.cliche_phrase",
                severity="medium",
                span=(m.start(), m.end()),
                context=text[max(0, m.start() - 10):m.end() + 10],
                suggestion=f"Replace cliche '{phrase}' with a more specific expression",
            ))
    return violations


def _check_vocab_ai_filler(text: str) -> List[Violation]:
    violations = []
    for phrase in _AI_FILLER_WORDS:
        pattern = re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE)
        for m in pattern.finditer(text):
            # Don't double-count "notably" (already in high_risk_transitions)
            if phrase == "notably":
                continue
            violations.append(Violation(
                rule_id="vocab.ai_filler",
                severity="low",
                span=(m.start(), m.end()),
                context=text[max(0, m.start() - 10):m.end() + 10],
                suggestion=f"Remove or replace AI filler '{phrase}'",
            ))
    return violations


# ── D-class: Syntax (document-level) ────────────────────────────────────────

_RE_PASSIVE = re.compile(
    r"\b(?:was|were|been|being|is|are)\s+\w+(?:ed|en)\b",
    re.IGNORECASE,
)


def _check_syntax_uniform_rhythm(sentences: List[str]) -> List[Violation]:
    """Flag if ≥3 consecutive sentences have similar word counts (std < 3)."""
    if len(sentences) < 3:
        return []

    violations = []
    word_counts = [len(s.split()) for s in sentences]
    window = 3

    for i in range(len(word_counts) - window + 1):
        chunk = word_counts[i:i + window]
        mean_wc = sum(chunk) / len(chunk)
        spread = max(chunk) - min(chunk)
        # Uniform = all within a narrow band AND in the AI-typical mid-range
        # Short sentences (mean < 14) are natural human variation, not a signal
        if spread <= 4 and mean_wc >= 14:
            std = statistics.stdev(chunk) if len(chunk) > 1 else 0.0
            if std < 3:
                    violations.append(Violation(
                        rule_id="syntax.uniform_rhythm",
                        severity="medium",
                        span=(0, 0),
                        context=f"Sentences {i}-{i + window - 1}: word counts {chunk}, std={std:.1f}",
                        suggestion="Vary sentence length: insert a short sentence (≤8 words) to break the rhythm",
                    ))
                    break  # one flag per document is enough
    return violations


def _check_syntax_passive_stack(sentences: List[str]) -> List[Violation]:
    """Flag if passive voice density is too high across consecutive sentences."""
    if len(sentences) < 2:
        return []

    violations = []
    window = 3
    for i in range(len(sentences) - min(window, len(sentences)) + 1):
        chunk = sentences[i:i + window]
        text_chunk = " ".join(chunk)
        passive_count = len(_RE_PASSIVE.findall(text_chunk))
        word_count = len(text_chunk.split())
        if word_count > 0 and passive_count >= 3 and (passive_count / word_count) > 0.04:
            violations.append(Violation(
                rule_id="syntax.passive_stack",
                severity="medium",
                span=(0, 0),
                context=f"Sentences {i}-{i + len(chunk) - 1}: {passive_count} passive constructions",
                suggestion="Convert passive voice to active voice; use simple past tense",
            ))
            break
    return violations


# ── E-class: Human deficit (document-level) ─────────────────────────────────

def _check_human_no_colloquial(text: str) -> List[Violation]:
    text_lower = text.lower()
    for marker in _COLLOQUIAL_MARKERS:
        if re.search(r"\b" + re.escape(marker) + r"\b", text_lower):
            return []  # found at least one → no violation
    return [Violation(
        rule_id="human.no_colloquial",
        severity="low",
        span=(0, len(text)),
        context="No colloquial markers found in text",
        suggestion="Add 1-2 colloquial markers (actually, just, pretty much, kind of)",
    )]


def _check_human_no_short_sentence(sentences: List[str]) -> List[Violation]:
    for s in sentences:
        if len(s.split()) <= 8:
            return []  # has at least one short sentence
    return [Violation(
        rule_id="human.no_short_sentence",
        severity="low",
        span=(0, 0),
        context="All sentences are >8 words",
        suggestion="Insert at least one short sentence (≤8 words) for rhythm variation",
    )]


def _check_human_no_question(text: str) -> List[Violation]:
    if "?" in text:
        return []
    return [Violation(
        rule_id="human.no_question",
        severity="low",
        span=(0, len(text)),
        context="No question marks found in text",
        suggestion="Consider adding 1-2 direct questions to the reader",
    )]


# ═══════════════════════════════════════════════════════════════════════════════
# Severity grading
# ═══════════════════════════════════════════════════════════════════════════════

_SEVERITY_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1, "clean": 0}


def compute_rule_severity(violations: List[Violation]) -> str:
    """
    Compute severity from a list of violations.

    Logic:
      - Any punct.em_dash → critical
      - ≥2 high-severity violations → critical
      - 1 high-severity violation → high
      - Any medium-severity violation → medium
      - Only low-severity violations → low
      - No violations → clean
    """
    if not violations:
        return "clean"

    has_em_dash = any(v.rule_id == "punct.em_dash" for v in violations)
    if has_em_dash:
        return "critical"

    high_count = sum(1 for v in violations if v.severity == "high")
    if high_count >= 2:
        return "critical"
    if high_count == 1:
        return "high"

    if any(v.severity == "medium" for v in violations):
        return "medium"

    if any(v.severity == "low" for v in violations):
        return "low"

    return "clean"


def fuse_severity(
    rule_severity: str,
    ai_z_score: float,
    model_useful: bool,
) -> str:
    """
    Dual-signal fusion: rule severity + AI probability z-score → final severity.

    Fusion rules:
      - rule = critical → critical (never downgrade)
      - rule = clean AND z > 1.5 AND model_useful → medium
      - rule = medium AND z > 1.0 AND model_useful → high
      - rule = high AND z < -1.0 AND model_useful → medium
      - otherwise → keep rule_severity
    """
    if rule_severity == "critical":
        return "critical"

    if not model_useful:
        return rule_severity

    if rule_severity == "clean" and ai_z_score > 1.5:
        return "medium"

    if rule_severity == "medium" and ai_z_score > 1.0:
        return "high"

    if rule_severity == "high" and ai_z_score < -1.0:
        return "medium"

    return rule_severity


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_sentence(sentence: str) -> SentenceAnalysis:
    """
    Run all sentence-level rule checks on a single sentence.

    Returns SentenceAnalysis with ai_score/ai_z_score/severity defaulted
    (to be filled by orchestrator after AI model scoring).
    """
    violations: List[Violation] = []

    # A-class
    violations.extend(_check_punct_em_dash(sentence))
    violations.extend(_check_punct_semicolon(sentence))
    violations.extend(_check_punct_colon_overuse(sentence))

    # B-class
    violations.extend(_check_struct_not_x_but_y(sentence))
    violations.extend(_check_struct_triple_parallel(sentence))
    violations.extend(_check_struct_abstract_noun_stack(sentence))

    # C-class
    violations.extend(_check_vocab_high_risk_transition(sentence))
    violations.extend(_check_vocab_cliche_phrase(sentence))
    violations.extend(_check_vocab_ai_filler(sentence))

    rule_sev = compute_rule_severity(violations)

    return SentenceAnalysis(
        idx=0,  # caller should set the real index
        text=sentence,
        violations=violations,
        rule_severity=rule_sev,
    )


def analyze_document(text: str) -> DocumentAnalysis:
    """
    Full document analysis: split into sentences, run per-sentence rules,
    then run document-level rules (D-class syntax, E-class human deficit).

    ai_score / ai_z_score / fused severity are left at defaults —
    the orchestrator fills them after running the detector model.
    """
    sentences = _split_sentences(text)

    # Per-sentence analysis
    analyses: List[SentenceAnalysis] = []
    for i, sent in enumerate(sentences):
        sa = analyze_sentence(sent)
        sa.idx = i
        analyses.append(sa)

    # D-class: document-level syntax checks
    sent_texts = [sa.text for sa in analyses]
    rhythm_violations = _check_syntax_uniform_rhythm(sent_texts)
    passive_violations = _check_syntax_passive_stack(sent_texts)

    # Attach D-class violations to the first sentence in the flagged window
    for v in rhythm_violations:
        if analyses:
            analyses[0].violations.append(v)
            analyses[0].rule_severity = compute_rule_severity(analyses[0].violations)
            analyses[0].severity = analyses[0].rule_severity
    for v in passive_violations:
        if analyses:
            # Attach to the sentence where passive stacking starts
            analyses[0].violations.append(v)
            analyses[0].rule_severity = compute_rule_severity(analyses[0].violations)
            analyses[0].severity = analyses[0].rule_severity

    # E-class: document-level human deficit checks
    e_violations: List[Violation] = []
    e_violations.extend(_check_human_no_colloquial(text))
    e_violations.extend(_check_human_no_short_sentence(sent_texts))
    e_violations.extend(_check_human_no_question(text))

    # Attach E-class to the first sentence (they're document-level signals)
    for v in e_violations:
        if analyses:
            analyses[0].violations.append(v)
            analyses[0].rule_severity = compute_rule_severity(analyses[0].violations)
            analyses[0].severity = analyses[0].rule_severity

    # Summary
    severity_counts: Dict[str, int] = {
        "critical": 0, "high": 0, "medium": 0, "low": 0, "clean": 0,
    }
    for sa in analyses:
        severity_counts[sa.rule_severity] = severity_counts.get(sa.rule_severity, 0) + 1

    # Top issues by frequency
    rule_freq: Dict[str, int] = {}
    for sa in analyses:
        for v in sa.violations:
            rule_freq[v.rule_id] = rule_freq.get(v.rule_id, 0) + 1
    top_issues = sorted(rule_freq, key=rule_freq.get, reverse=True)

    return DocumentAnalysis(
        sentences=analyses,
        summary=severity_counts,
        top_issues=top_issues,
    )
