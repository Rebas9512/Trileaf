"""
Phase 2 — prompt_builder.py acceptance tests.

Acceptance criteria from PIPELINE_V2_PLAN.md §2.2:
  - Stage 3 prompt includes all CRITICAL/HIGH sentence annotations
  - Stage 3 prompt token count ≤ 4000 for ≤1000-word input
  - Stage 5 technique_budget scales correctly with text length
  - All prompts require JSON output format
  - build functions produce well-structured output
"""

from __future__ import annotations

import re
from typing import Dict, List

import pytest

pb = pytest.importorskip("scripts.prompt_builder")
rd = pytest.importorskip("scripts.rule_detector")


# ── Helpers ─────────────────────────────────────────────────────────────────

def _make_sentence_analysis(idx, text, rule_severity, flags=None):
    """Create a SentenceAnalysis for testing prompt builders."""
    violations = []
    if flags:
        for f in flags:
            violations.append(rd.Violation(
                rule_id=f,
                severity="high",
                span=(0, len(text)),
                context=text[:50],
                suggestion=f"Fix {f}",
            ))
    return rd.SentenceAnalysis(
        idx=idx,
        text=text,
        violations=violations,
        rule_severity=rule_severity,
        ai_score=0.0,
        ai_z_score=0.0,
        severity=rule_severity,
    )


def _approx_token_count(text: str) -> int:
    """Rough token count: ~4 chars per token for English."""
    return len(text) // 4


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 3 Prompt
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildStage3Prompt:
    """build_stage3_prompt() — standardized rewrite prompt."""

    def _sample_sentences(self) -> List:
        return [
            _make_sentence_analysis(
                0,
                "The algorithm worked — but only on its own terms.",
                "critical",
                ["punct.em_dash", "struct.not_x_but_y"],
            ),
            _make_sentence_analysis(
                1,
                "However, the results showed improvement.",
                "high",
                ["vocab.high_risk_transition"],
            ),
            _make_sentence_analysis(
                2,
                "Employees felt uncertain about the changes.",
                "clean",
            ),
            _make_sentence_analysis(
                3,
                "The empirical foundation was solid.",
                "medium",
                ["vocab.cliche_phrase"],
            ),
        ]

    def test_includes_critical_annotations(self):
        sentences = self._sample_sentences()
        prompt = pb.build_stage3_prompt(sentences, "rules summary text")
        # Must contain [CRITICAL] tag for sentence 0
        assert "[CRITICAL" in prompt
        assert "punct.em_dash" in prompt
        assert "struct.not_x_but_y" in prompt

    def test_includes_high_annotations(self):
        sentences = self._sample_sentences()
        prompt = pb.build_stage3_prompt(sentences, "rules summary text")
        assert "[HIGH" in prompt
        assert "vocab.high_risk_transition" in prompt

    def test_includes_clean_sentences(self):
        """Clean sentences should appear (for context) but with CLEAN tag."""
        sentences = self._sample_sentences()
        prompt = pb.build_stage3_prompt(sentences, "rules summary text")
        assert "[CLEAN]" in prompt or "CLEAN" in prompt
        assert "Employees felt uncertain" in prompt

    def test_includes_rules_summary(self):
        sentences = self._sample_sentences()
        custom_rules = "Do not use em dashes. Avoid however."
        prompt = pb.build_stage3_prompt(sentences, custom_rules)
        assert "Do not use em dashes" in prompt

    def test_requires_json_output(self):
        sentences = self._sample_sentences()
        prompt = pb.build_stage3_prompt(sentences, "rules")
        assert '{"rewrite"' in prompt or '"rewrite"' in prompt

    def test_includes_hard_constraints(self):
        sentences = self._sample_sentences()
        prompt = pb.build_stage3_prompt(sentences, "rules")
        prompt_lower = prompt.lower()
        # Must mention preserving facts
        assert "fact" in prompt_lower or "entity" in prompt_lower or "data" in prompt_lower
        # Must mention no em dashes
        assert "dash" in prompt_lower or "—" in prompt_lower

    def test_token_count_within_budget(self):
        """For ≤1000-word input, prompt token count should be ≤ 4000."""
        # Simulate ~40 sentences of ~25 words each = ~1000 words
        sentences = []
        for i in range(40):
            text = f"This is sentence number {i} with approximately twenty five words in it to fill the space needed."
            severity = "clean" if i % 5 != 0 else "high"
            flags = ["vocab.high_risk_transition"] if severity == "high" else None
            sentences.append(_make_sentence_analysis(i, text, severity, flags))

        prompt = pb.build_stage3_prompt(sentences, "Standard rules summary.")
        token_est = _approx_token_count(prompt)
        assert token_est <= 4000, f"Estimated {token_est} tokens exceeds 4000 limit"

    def test_sentence_order_preserved(self):
        """Sentences should appear in original order in the prompt."""
        sentences = self._sample_sentences()
        prompt = pb.build_stage3_prompt(sentences, "rules")
        idx_0 = prompt.find("algorithm worked")
        idx_1 = prompt.find("results showed")
        idx_2 = prompt.find("Employees felt")
        assert idx_0 < idx_1 < idx_2


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 4 Prompt
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildStage4Prompt:
    """build_stage4_prompt() — single sentence attack prompt."""

    def test_includes_sentence_text(self):
        violations = [
            rd.Violation("punct.em_dash", "critical", (20, 21), "—", "Replace with comma")
        ]
        prompt = pb.build_stage4_prompt(
            sentence="The algorithm worked — but only.",
            violations=violations,
            context_before="Some context before.",
            context_after="Some context after.",
        )
        assert "algorithm worked" in prompt

    def test_includes_violations(self):
        violations = [
            rd.Violation("punct.em_dash", "critical", (20, 21), "—", "Replace with comma"),
            rd.Violation("struct.not_x_but_y", "high", (0, 30), "not X but Y", "Break symmetry"),
        ]
        prompt = pb.build_stage4_prompt(
            sentence="Not to discard it — but to supplement.",
            violations=violations,
            context_before="Previous sentence.",
            context_after="Next sentence.",
        )
        assert "punct.em_dash" in prompt
        assert "struct.not_x_but_y" in prompt

    def test_includes_context(self):
        violations = [rd.Violation("vocab.high_risk_transition", "high", (0, 7), "However", "Remove")]
        prompt = pb.build_stage4_prompt(
            sentence="However, the data was clear.",
            violations=violations,
            context_before="The experiment ran for three months.",
            context_after="Management decided to act.",
        )
        assert "experiment ran" in prompt
        assert "Management decided" in prompt

    def test_requires_json_output(self):
        violations = [rd.Violation("vocab.ai_filler", "low", (0, 5), "nuanced", "Remove")]
        prompt = pb.build_stage4_prompt(
            sentence="A nuanced approach.",
            violations=violations,
            context_before="",
            context_after="",
        )
        assert '{"rewrite"' in prompt or '"rewrite"' in prompt

    def test_includes_suggestions(self):
        """Violation suggestions should appear in the prompt."""
        violations = [
            rd.Violation("punct.em_dash", "critical", (5, 6), "—", "Replace dash with comma or period"),
        ]
        prompt = pb.build_stage4_prompt(
            sentence="Test — sentence.",
            violations=violations,
            context_before="",
            context_after="",
        )
        assert "comma" in prompt.lower() or "period" in prompt.lower() or "Replace" in prompt


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 5 Prompt
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildStage5Prompt:
    """build_stage5_prompt() — human touch polish prompt."""

    def test_includes_text(self):
        budget = {"understatement": 2, "lazy_pronoun": 1}
        prompt = pb.build_stage5_prompt(
            text="The results were clear and convincing.",
            technique_budget=budget,
            human_deficit=["human.no_colloquial"],
        )
        assert "results were clear" in prompt

    def test_includes_budget(self):
        budget = {
            "understatement": 2,
            "lazy_pronoun": 1,
            "half_comparison": 1,
            "sentence_tail": 2,
            "redundant_modifier": 2,
            "short_sentence_break": 1,
        }
        prompt = pb.build_stage5_prompt(
            text="Some text.",
            technique_budget=budget,
            human_deficit=[],
        )
        # Budget numbers should appear
        for technique, count in budget.items():
            assert str(count) in prompt or technique in prompt.lower().replace("_", " ")

    def test_includes_deficit_hints(self):
        prompt = pb.build_stage5_prompt(
            text="Some text.",
            technique_budget={"understatement": 1},
            human_deficit=["human.no_colloquial", "human.no_question"],
        )
        prompt_lower = prompt.lower()
        assert "colloquial" in prompt_lower or "oral" in prompt_lower or "casual" in prompt_lower

    def test_requires_json_output(self):
        prompt = pb.build_stage5_prompt(
            text="Some text.",
            technique_budget={"understatement": 1},
            human_deficit=[],
        )
        assert '{"rewrite"' in prompt or '"rewrite"' in prompt

    def test_no_dash_constraint(self):
        """Prompt should explicitly forbid dashes."""
        prompt = pb.build_stage5_prompt(
            text="Some text.",
            technique_budget={"understatement": 1},
            human_deficit=[],
        )
        prompt_lower = prompt.lower()
        assert "dash" in prompt_lower or "—" in prompt_lower

    def test_distribution_constraint(self):
        """Prompt should mention distributing techniques evenly."""
        prompt = pb.build_stage5_prompt(
            text="Some text.",
            technique_budget={"understatement": 2, "lazy_pronoun": 2},
            human_deficit=[],
        )
        prompt_lower = prompt.lower()
        assert any(
            kw in prompt_lower
            for kw in ["distribute", "spread", "evenly", "scatter", "disperse",
                        "均匀", "分散", "分布"]
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Dynamic budget calculation
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeTechniqueBudget:
    """compute_technique_budget() should scale with word count."""

    def test_short_text_minimum_budget(self):
        budget = pb.compute_technique_budget(200)
        # All values should be at least 1
        for k, v in budget.items():
            assert v >= 1, f"{k} budget is {v}, expected >= 1"

    def test_medium_text_scaling(self):
        budget_500 = pb.compute_technique_budget(500)
        budget_1000 = pb.compute_technique_budget(1000)
        # 1000-word budget should generally be >= 500-word budget
        for k in budget_500:
            assert budget_1000[k] >= budget_500[k], (
                f"{k}: 1000w={budget_1000[k]} < 500w={budget_500[k]}"
            )

    def test_max_cap_at_3(self):
        """No technique should exceed 3 per article."""
        budget = pb.compute_technique_budget(5000)
        for k, v in budget.items():
            assert v <= 3, f"{k} budget is {v}, exceeds max of 3"

    def test_all_techniques_present(self):
        """Budget should include all 6 technique types."""
        budget = pb.compute_technique_budget(1000)
        expected_keys = {
            "understatement",
            "lazy_pronoun",
            "half_comparison",
            "sentence_tail",
            "redundant_modifier",
            "short_sentence_break",
        }
        assert set(budget.keys()) == expected_keys

    @pytest.mark.parametrize("word_count", [100, 499, 500, 501, 999, 1000, 1500, 2000, 3000])
    def test_budget_values_are_positive_integers(self, word_count):
        budget = pb.compute_technique_budget(word_count)
        for k, v in budget.items():
            assert isinstance(v, int) and v >= 1, f"{k}={v} at {word_count} words"

    def test_half_comparison_conservative(self):
        """half_comparison should always be ≤ 2 (it's a strong technique)."""
        for wc in [200, 500, 1000, 2000, 5000]:
            budget = pb.compute_technique_budget(wc)
            assert budget["half_comparison"] <= 2
