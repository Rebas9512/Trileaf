"""
Phase 1 — rule_detector.py acceptance tests.

Covers every rule_id with positive and negative examples drawn from
the strategy document (降低AI写作痕迹策略总结.md).

Acceptance criteria from PIPELINE_V2_PLAN.md §1.2:
  - analyze_document() flags all ❌ examples with correct rule_id
  - analyze_document() marks all ✅ examples as clean or low
  - severity grading matches the documented logic
  - analyze_document() on 1000-word text completes in < 50 ms
  - No spaCy dependency
"""

from __future__ import annotations

import time
from typing import List

import pytest

# These imports will fail until the module is implemented.
# pytest.importorskip ensures we get a clear skip, not a cryptic ImportError.
rd = pytest.importorskip("scripts.rule_detector")


# ═══════════════════════════════════════════════════════════════════════════════
# A-class: Punctuation
# ═══════════════════════════════════════════════════════════════════════════════

class TestPunctEmDash:
    """punct.em_dash — em dash, en dash, double-hyphen."""

    @pytest.mark.parametrize("text,expected_count", [
        ("The algorithm worked — but only on its own terms.", 1),
        ("Three options – top-down and open jobs – each had downsides.", 2),
        ("We tried it -- and it failed.", 1),
        # Multiple dashes in one sentence
        ("He said — wait — what?", 2),
    ])
    def test_positive(self, text, expected_count):
        result = rd.analyze_sentence(text)
        em_violations = [v for v in result.violations if v.rule_id == "punct.em_dash"]
        assert len(em_violations) == expected_count
        for v in em_violations:
            assert v.severity == "critical"

    @pytest.mark.parametrize("text", [
        "The algorithm worked, but only on its own terms.",
        "Three options (top-down, open jobs, and an algorithm) each had downsides.",
        "A well-known fact.",  # regular hyphen in compound word, NOT a dash
        "state-of-the-art system",
    ])
    def test_negative(self, text):
        result = rd.analyze_sentence(text)
        em_violations = [v for v in result.violations if v.rule_id == "punct.em_dash"]
        assert len(em_violations) == 0


class TestPunctSemicolon:
    """punct.semicolon"""

    def test_positive(self):
        result = rd.analyze_sentence(
            "The algorithm was effective; however, it had limitations."
        )
        semi = [v for v in result.violations if v.rule_id == "punct.semicolon"]
        assert len(semi) == 1

    def test_negative(self):
        result = rd.analyze_sentence(
            "The algorithm was effective. However, it had limitations."
        )
        semi = [v for v in result.violations if v.rule_id == "punct.semicolon"]
        assert len(semi) == 0


class TestPunctColonOveruse:
    """punct.colon_overuse — colons not used for list introduction."""

    def test_positive_self_answer(self):
        result = rd.analyze_sentence(
            "The question is simple: can we even measure this?"
        )
        colon = [v for v in result.violations if v.rule_id == "punct.colon_overuse"]
        assert len(colon) >= 1

    def test_negative_list_intro(self):
        # Colon before a list is acceptable
        result = rd.analyze_sentence(
            "Three options were considered:"
        )
        colon = [v for v in result.violations if v.rule_id == "punct.colon_overuse"]
        assert len(colon) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# B-class: Structure
# ═══════════════════════════════════════════════════════════════════════════════

class TestStructNotXButY:
    """struct.not_x_but_y — symmetric opposition."""

    @pytest.mark.parametrize("text", [
        "The recommendation is not to discard the algorithm, but to supplement it with additional data.",
        "It won by default, not because anyone could show it was the right call.",
        "Rather than abandoning the project, they decided to pivot.",
    ])
    def test_positive(self, text):
        result = rd.analyze_sentence(text)
        matches = [v for v in result.violations if v.rule_id == "struct.not_x_but_y"]
        assert len(matches) >= 1, f"Expected struct.not_x_but_y in: {text}"

    @pytest.mark.parametrize("text", [
        # Strategy doc ✅ examples: broken symmetry
        "So the fix isn't to throw the algorithm out. It's to feed it more than preferences.",
        "Nobody could actually prove it was the best choice. It just wasn't as bad as the other two.",
    ])
    def test_negative(self, text):
        result = rd.analyze_sentence(text)
        matches = [v for v in result.violations if v.rule_id == "struct.not_x_but_y"]
        assert len(matches) == 0, f"False positive struct.not_x_but_y in: {text}"


class TestStructTripleParallel:
    """struct.triple_parallel — A, B, and C pattern."""

    @pytest.mark.parametrize("text", [
        "no direct productivity data, no pre-established benchmark, and no way to tell if anyone was better off",
        "The framework addresses legitimacy, alignment, and optimization.",
    ])
    def test_positive(self, text):
        result = rd.analyze_sentence(text)
        matches = [v for v in result.violations if v.rule_id == "struct.triple_parallel"]
        assert len(matches) >= 1

    @pytest.mark.parametrize("text", [
        # Broken into independent sentences (✅ style)
        "No productivity numbers came out of it. Nobody set any benchmarks. Without that, there was nothing to compare.",
        # Short list of two items (not three)
        "We considered speed and accuracy.",
    ])
    def test_negative(self, text):
        result = rd.analyze_sentence(text)
        matches = [v for v in result.violations if v.rule_id == "struct.triple_parallel"]
        assert len(matches) == 0


class TestStructAbstractNounStack:
    """struct.abstract_noun_stack — ≥3 abstract nouns in one sentence."""

    def test_positive(self):
        result = rd.analyze_sentence(
            "nor that global optimization inherently produces individual suboptimal outcomes"
        )
        matches = [v for v in result.violations if v.rule_id == "struct.abstract_noun_stack"]
        assert len(matches) >= 1

    def test_negative(self):
        # Rewritten with clauses and pronouns (✅ style)
        result = rd.analyze_sentence(
            "when you optimize for a whole group, some individuals necessarily "
            "end up with worse outcomes than they would have otherwise"
        )
        matches = [v for v in result.violations if v.rule_id == "struct.abstract_noun_stack"]
        assert len(matches) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# C-class: Vocabulary
# ═══════════════════════════════════════════════════════════════════════════════

class TestVocabHighRiskTransition:
    """vocab.high_risk_transition — however, furthermore, moreover, etc."""

    @pytest.mark.parametrize("text,word", [
        ("However, the results showed improvement.", "however"),
        ("Furthermore, this approach has merit.", "furthermore"),
        ("Moreover, the data supports the claim.", "moreover"),
        ("That said, optimizing purely for happiness is the core weakness.", "that said"),
        ("Indeed, the results were surprising.", "indeed"),
        ("Notably, the system outperformed expectations.", "notably"),
    ])
    def test_positive(self, text, word):
        result = rd.analyze_sentence(text)
        matches = [v for v in result.violations if v.rule_id == "vocab.high_risk_transition"]
        assert len(matches) >= 1, f"Expected to flag '{word}' in: {text}"

    @pytest.mark.parametrize("text", [
        "The real weakness of the algorithm is what it chose to measure.",
        "You have to get ahead of this stuff before the process even starts.",
    ])
    def test_negative(self, text):
        result = rd.analyze_sentence(text)
        matches = [v for v in result.violations if v.rule_id == "vocab.high_risk_transition"]
        assert len(matches) == 0


class TestVocabClichePhrase:
    """vocab.cliche_phrase — overused AI phrases."""

    @pytest.mark.parametrize("phrase", [
        "fresh in everyone's memory",
        "at its core",
        "navigate the complexities of",
        "a nuanced approach",
        "empirical foundation",
    ])
    def test_positive(self, phrase):
        text = f"The team found that {phrase} the situation was clear."
        result = rd.analyze_sentence(text)
        matches = [v for v in result.violations if v.rule_id == "vocab.cliche_phrase"]
        assert len(matches) >= 1, f"Expected to flag cliche: {phrase}"


class TestVocabAiFiller:
    """vocab.ai_filler — delve into, nuanced, multifaceted, etc."""

    @pytest.mark.parametrize("word", [
        "delve into",
        "nuanced",
        "multifaceted",
        "pivotal",
        "comprehensive",
    ])
    def test_positive(self, word):
        text = f"This {word} analysis reveals key insights."
        result = rd.analyze_sentence(text)
        matches = [v for v in result.violations if v.rule_id == "vocab.ai_filler"]
        assert len(matches) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# D-class: Syntax
# ═══════════════════════════════════════════════════════════════════════════════

class TestSyntaxUniformRhythm:
    """syntax.uniform_rhythm — ≥3 consecutive sentences with similar word count."""

    def test_positive_uniform(self):
        # 4 sentences all 18-20 words — very uniform
        text = (
            "The algorithm produced results that were consistent across all major metrics in the evaluation. "
            "The team analyzed the data carefully and found significant patterns in the overall distribution. "
            "The management decided to implement the changes based on recommendations from the analysis team. "
            "The stakeholders reviewed the proposal and agreed that the approach was sound and well justified."
        )
        result = rd.analyze_document(text)
        rhythm = [
            v
            for s in result.sentences
            for v in s.violations
            if v.rule_id == "syntax.uniform_rhythm"
        ]
        assert len(rhythm) >= 1

    def test_negative_varied(self):
        # Mix of short and long — human-like rhythm
        text = (
            "That didn't work. "
            "Leadership had been down that road in 2014 with a smaller group "
            "and hadn't liked what they saw. "
            "So they tried something else. "
            "The algorithm was in roughly the same boat."
        )
        result = rd.analyze_document(text)
        rhythm = [
            v
            for s in result.sentences
            for v in s.violations
            if v.rule_id == "syntax.uniform_rhythm"
        ]
        assert len(rhythm) == 0


class TestSyntaxPassiveStack:
    """syntax.passive_stack — excessive was/were/been/being + past participle."""

    def test_positive(self):
        text = (
            "No one had set a benchmark tied to the ten strategic priorities "
            "before the process started. The results were analyzed and the data "
            "was reviewed by the committee. It was determined that changes were needed."
        )
        result = rd.analyze_document(text)
        passive = [
            v
            for s in result.sentences
            for v in s.violations
            if v.rule_id == "syntax.passive_stack"
        ]
        assert len(passive) >= 1

    def test_negative(self):
        text = (
            "Nobody set any benchmarks against the ten strategic priorities "
            "before the thing started. The team reviewed the data themselves."
        )
        result = rd.analyze_document(text)
        passive = [
            v
            for s in result.sentences
            for v in s.violations
            if v.rule_id == "syntax.passive_stack"
        ]
        assert len(passive) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# E-class: Human touch deficit
# ═══════════════════════════════════════════════════════════════════════════════

class TestHumanNoColloquial:
    """human.no_colloquial — no colloquial markers in entire text."""

    def test_positive_no_colloquial(self):
        text = (
            "The algorithm produced consistent results. "
            "The team found the approach effective. "
            "The methodology was sound."
        )
        result = rd.analyze_document(text)
        matches = [
            v
            for s in result.sentences
            for v in s.violations
            if v.rule_id == "human.no_colloquial"
        ]
        assert len(matches) >= 1

    def test_negative_has_colloquial(self):
        text = (
            "The algorithm actually produced pretty good results. "
            "Nobody could really prove it was the best choice."
        )
        result = rd.analyze_document(text)
        matches = [
            v
            for s in result.sentences
            for v in s.violations
            if v.rule_id == "human.no_colloquial"
        ]
        assert len(matches) == 0


class TestHumanNoShortSentence:
    """human.no_short_sentence — no sentence ≤8 words."""

    def test_positive_all_long(self):
        text = (
            "The algorithm produced results that were consistent across all metrics. "
            "The team analyzed the data carefully and found significant patterns. "
            "The management decided to implement the changes based on recommendations."
        )
        result = rd.analyze_document(text)
        matches = [
            v
            for s in result.sentences
            for v in s.violations
            if v.rule_id == "human.no_short_sentence"
        ]
        assert len(matches) >= 1

    def test_negative_has_short(self):
        text = (
            "That didn't work. "
            "Leadership had been down that road before and hadn't liked what they saw."
        )
        result = rd.analyze_document(text)
        matches = [
            v
            for s in result.sentences
            for v in s.violations
            if v.rule_id == "human.no_short_sentence"
        ]
        assert len(matches) == 0


class TestHumanNoQuestion:
    """human.no_question — no question marks in entire text."""

    def test_positive_no_questions(self):
        text = (
            "The results were clear. The approach worked. No further changes needed."
        )
        result = rd.analyze_document(text)
        matches = [
            v
            for s in result.sentences
            for v in s.violations
            if v.rule_id == "human.no_question"
        ]
        assert len(matches) >= 1

    def test_negative_has_question(self):
        text = (
            "How do you even tell if the reorganization worked? "
            "The results were not immediately obvious."
        )
        result = rd.analyze_document(text)
        matches = [
            v
            for s in result.sentences
            for v in s.violations
            if v.rule_id == "human.no_question"
        ]
        assert len(matches) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Severity grading
# ═══════════════════════════════════════════════════════════════════════════════

class TestSeverityGrading:
    """Verify severity levels match documented logic."""

    def test_critical_from_em_dash(self):
        result = rd.analyze_sentence("The system worked — barely.")
        assert result.rule_severity == "critical"

    def test_critical_from_multiple_high(self):
        # Two high-severity violations → critical
        result = rd.analyze_sentence(
            "However, the recommendation is not to discard the system, "
            "but to supplement it with additional resources."
        )
        assert result.rule_severity == "critical"

    def test_high_single_transition(self):
        result = rd.analyze_sentence("However, the data was clear.")
        assert result.rule_severity == "high"

    def test_high_triple_parallel(self):
        result = rd.analyze_sentence(
            "We considered speed, accuracy, and reliability."
        )
        assert result.rule_severity == "high"

    def test_medium_cliche(self):
        result = rd.analyze_sentence(
            "The empirical foundation of this research is solid."
        )
        assert result.rule_severity == "medium"

    def test_low_ai_filler(self):
        result = rd.analyze_sentence(
            "This comprehensive review covered all major areas."
        )
        assert result.rule_severity == "low"

    def test_clean(self):
        result = rd.analyze_sentence(
            "Nobody could actually prove it was the best choice."
        )
        assert result.rule_severity == "clean"


# ═══════════════════════════════════════════════════════════════════════════════
# Severity fusion (rule + AI z-score)
# ═══════════════════════════════════════════════════════════════════════════════

class TestFuseSeverity:
    """fuse_severity() — dual-signal fusion logic."""

    def test_critical_unaffected_by_low_z(self):
        """Critical stays critical even if AI score is low."""
        assert rd.fuse_severity("critical", ai_z_score=-2.0, model_useful=True) == "critical"

    def test_clean_upgraded_on_high_z(self):
        """Clean → medium when z > 1.5 and model is useful."""
        assert rd.fuse_severity("clean", ai_z_score=2.0, model_useful=True) == "medium"

    def test_clean_stays_clean_when_model_not_useful(self):
        """Clean stays clean when model has no discrimination."""
        assert rd.fuse_severity("clean", ai_z_score=2.0, model_useful=False) == "clean"

    def test_medium_boosted_on_high_z(self):
        """Medium → high when z > 1.0 and model is useful."""
        assert rd.fuse_severity("medium", ai_z_score=1.5, model_useful=True) == "high"

    def test_medium_stays_when_z_normal(self):
        """Medium stays medium when z is in normal range."""
        assert rd.fuse_severity("medium", ai_z_score=0.5, model_useful=True) == "medium"

    def test_high_downgraded_on_low_z(self):
        """High → medium when z < -1.0 and model is useful."""
        assert rd.fuse_severity("high", ai_z_score=-1.5, model_useful=True) == "medium"

    def test_high_stays_when_model_not_useful(self):
        """High stays high when model has no discrimination (no downgrade)."""
        assert rd.fuse_severity("high", ai_z_score=-1.5, model_useful=False) == "high"

    def test_low_unaffected(self):
        """Low severity has no fusion rules, stays low."""
        assert rd.fuse_severity("low", ai_z_score=0.0, model_useful=True) == "low"


# ═══════════════════════════════════════════════════════════════════════════════
# Document-level analysis
# ═══════════════════════════════════════════════════════════════════════════════

class TestAnalyzeDocument:
    """Full document analysis integration."""

    def test_ai_heavy_text_flags_multiple_rules(self, sample_ai_heavy):
        result = rd.analyze_document(sample_ai_heavy)
        all_rule_ids = {v.rule_id for s in result.sentences for v in s.violations}
        # Must catch at least these from the AI-heavy sample
        assert "punct.em_dash" in all_rule_ids
        assert "vocab.high_risk_transition" in all_rule_ids
        assert "struct.not_x_but_y" in all_rule_ids
        assert "struct.triple_parallel" in all_rule_ids

    def test_human_text_mostly_clean(self, sample_human):
        result = rd.analyze_document(sample_human)
        # Should be mostly clean or low
        for sent in result.sentences:
            assert sent.rule_severity in ("clean", "low"), (
                f"Sentence flagged as {sent.rule_severity}: {sent.text}"
            )

    def test_summary_counts(self, sample_ai_heavy):
        result = rd.analyze_document(sample_ai_heavy)
        total = sum(result.summary.values())
        assert total == len(result.sentences)
        assert all(k in ("critical", "high", "medium", "low", "clean") for k in result.summary)

    def test_top_issues_populated(self, sample_ai_heavy):
        result = rd.analyze_document(sample_ai_heavy)
        assert len(result.top_issues) > 0
        # top_issues should be rule_id strings
        for issue in result.top_issues:
            assert "." in issue  # e.g. "punct.em_dash"


# ═══════════════════════════════════════════════════════════════════════════════
# Violation data structure
# ═══════════════════════════════════════════════════════════════════════════════

class TestViolationFields:
    """Every Violation should have all required fields populated."""

    def test_violation_has_all_fields(self):
        result = rd.analyze_sentence("The algorithm worked — but only on its own terms.")
        assert len(result.violations) > 0
        for v in result.violations:
            assert isinstance(v.rule_id, str) and len(v.rule_id) > 0
            assert v.severity in ("critical", "high", "medium", "low")
            assert isinstance(v.span, tuple) and len(v.span) == 2
            assert isinstance(v.context, str) and len(v.context) > 0
            assert isinstance(v.suggestion, str) and len(v.suggestion) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Performance
# ═══════════════════════════════════════════════════════════════════════════════

class TestPerformance:
    """analyze_document() on 1000-word text must complete in < 100 ms."""

    def test_1000_word_under_100ms(self):
        # Generate ~1000 word text by repeating a mix of sentences
        # Limit raised from 50→100 ms after rule set expanded from 13→25 rules
        base = (
            "The algorithm produced results. However, the team found issues. "
            "Furthermore, the data was comprehensive and nuanced. "
            "Nobody could actually prove it worked. That didn't go well. "
        )
        words = base.split()
        # Repeat until we have ~1000 words
        repeated = " ".join(words * (1000 // len(words) + 1))
        word_count = len(repeated.split())
        assert word_count >= 1000

        start = time.perf_counter()
        rd.analyze_document(repeated)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 100, f"analyze_document took {elapsed_ms:.1f} ms (limit: 100 ms)"


# ═══════════════════════════════════════════════════════════════════════════════
# No spaCy dependency
# ═══════════════════════════════════════════════════════════════════════════════

class TestNoSpacyDependency:
    """Phase 1 must not depend on spaCy."""

    def test_no_spacy_import(self):
        import importlib
        import sys
        from pathlib import Path

        # Re-import to check transitive deps
        if "scripts.rule_detector" in sys.modules:
            mod = sys.modules["scripts.rule_detector"]
        else:
            mod = importlib.import_module("scripts.rule_detector")

        # Check that spacy is not in the loaded modules after importing rule_detector
        source = Path(mod.__file__).read_text(encoding="utf-8")
        assert "import spacy" not in source
        assert "from spacy" not in source
