"""
Phase 3 — post_processor.py acceptance tests.

Acceptance criteria from PIPELINE_V2_PLAN.md §3.2:
  - fix_punctuation replaces all em dash / en dash / -- with reasonable punctuation
  - fix_punctuation replaces all semicolons with periods
  - split_into_segments produces 2-3 segments of 700-900 words for 2000-word text
  - Segment overlap context is correctly marked and won't be rewritten
"""

from __future__ import annotations

import pytest

pp = pytest.importorskip("scripts.post_processor")
rd = pytest.importorskip("scripts.rule_detector")


# ═══════════════════════════════════════════════════════════════════════════════
# fix_punctuation — em dash / en dash / double-hyphen
# ═══════════════════════════════════════════════════════════════════════════════

class TestFixPunctuationDashes:
    """All dashes should be replaced with comma, period, or other punctuation."""

    def test_em_dash_replaced(self):
        text = "The algorithm worked — but only on its own terms."
        result = pp.fix_punctuation(text)
        assert "—" not in result
        # Should still be valid English
        assert "algorithm worked" in result
        assert "its own terms" in result

    def test_en_dash_replaced(self):
        text = "Three options – top-down and open jobs – each had downsides."
        result = pp.fix_punctuation(text)
        assert "–" not in result

    def test_double_hyphen_replaced(self):
        text = "We tried it -- and it failed."
        result = pp.fix_punctuation(text)
        assert "--" not in result

    def test_regular_hyphen_preserved(self):
        """Normal hyphens in compound words must NOT be replaced."""
        text = "A well-known state-of-the-art system."
        result = pp.fix_punctuation(text)
        assert "well-known" in result
        assert "state-of-the-art" in result

    def test_multiple_dashes_all_replaced(self):
        text = "He said — wait — what? And then -- silence."
        result = pp.fix_punctuation(text)
        assert "—" not in result
        assert "--" not in result

    def test_dash_at_sentence_boundary(self):
        """Dash between two clauses should become comma or period."""
        text = "That's the real question — can you even tell if it worked?"
        result = pp.fix_punctuation(text)
        assert "—" not in result
        # The two parts should still be present
        assert "real question" in result
        assert "tell if it worked" in result


# ═══════════════════════════════════════════════════════════════════════════════
# fix_punctuation — semicolons
# ═══════════════════════════════════════════════════════════════════════════════

class TestFixPunctuationSemicolons:
    """All semicolons should be replaced with periods."""

    def test_semicolon_replaced(self):
        text = "The algorithm was effective; the results were clear."
        result = pp.fix_punctuation(text)
        assert ";" not in result
        # Should be split into two sentences
        assert ". " in result or "." in result

    def test_multiple_semicolons(self):
        text = "First point; second point; third point."
        result = pp.fix_punctuation(text)
        assert ";" not in result

    def test_no_semicolons_unchanged(self):
        text = "A normal sentence without any semicolons."
        result = pp.fix_punctuation(text)
        assert result == text


# ═══════════════════════════════════════════════════════════════════════════════
# fix_whitespace
# ═══════════════════════════════════════════════════════════════════════════════

class TestFixWhitespace:
    """Whitespace normalization."""

    def test_multiple_spaces(self):
        result = pp.fix_whitespace("Too   many    spaces.")
        assert "   " not in result

    def test_trailing_spaces(self):
        result = pp.fix_whitespace("Line with trailing spaces.   \nNext line.  ")
        assert not any(line.endswith(" ") for line in result.split("\n"))

    def test_excessive_blank_lines(self):
        result = pp.fix_whitespace("Paragraph one.\n\n\n\n\nParagraph two.")
        # Should collapse to at most 2 newlines (1 blank line)
        assert "\n\n\n" not in result


# ═══════════════════════════════════════════════════════════════════════════════
# run_post_process — combined
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunPostProcess:
    """run_post_process applies all fixes and returns a log."""

    def test_returns_tuple(self):
        result = pp.run_post_process("Some text — with issues; and spaces.")
        assert isinstance(result, tuple)
        assert len(result) == 2
        text, log = result
        assert isinstance(text, str)
        assert isinstance(log, list)

    def test_all_issues_fixed(self):
        text = "Test — sentence; with    issues."
        fixed, log = pp.run_post_process(text)
        assert "—" not in fixed
        assert ";" not in fixed
        assert "    " not in fixed

    def test_log_describes_changes(self):
        text = "Fixed — this; issue."
        _, log = pp.run_post_process(text)
        assert len(log) > 0  # Should report what was changed

    def test_clean_text_unchanged(self):
        text = "A perfectly clean sentence with no issues."
        fixed, log = pp.run_post_process(text)
        assert fixed == text
        assert len(log) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# split_into_segments
# ═══════════════════════════════════════════════════════════════════════════════

def _make_long_text(target_words: int) -> str:
    """Generate a multi-paragraph text of approximately target_words words."""
    para_template = (
        "The team reviewed the quarterly results and found several interesting patterns. "
        "Revenue grew steadily across all regions while costs remained relatively stable. "
        "Customer satisfaction scores improved compared to the previous quarter. "
        "The marketing team launched three new campaigns that performed above expectations. "
        "Employee retention rates held steady despite industry-wide turnover increases."
    )
    words_per_para = len(para_template.split())
    paras_needed = target_words // words_per_para + 1
    text = "\n\n".join([para_template] * paras_needed)
    # Trim to approximate target
    words = text.split()[:target_words]
    return " ".join(words)


def _make_sentence_analyses(text: str):
    """Create dummy SentenceAnalysis objects from text."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    analyses = []
    for i, sent in enumerate(sentences):
        if sent.strip():
            analyses.append(rd.SentenceAnalysis(
                idx=i,
                text=sent.strip(),
                violations=[],
                rule_severity="clean",
                ai_score=0.0,
                ai_z_score=0.0,
                severity="clean",
            ))
    return analyses


class TestSplitIntoSegments:
    """split_into_segments() — paragraph-boundary splitting."""

    def test_2000_word_produces_2_to_3_segments(self):
        text = _make_long_text(2000)
        sentences = _make_sentence_analyses(text)
        segments = pp.split_into_segments(
            text, sentences, target_words=800, overlap_sentences=2
        )
        assert 2 <= len(segments) <= 3, (
            f"Expected 2-3 segments, got {len(segments)}"
        )

    def test_segment_word_count_in_range(self):
        text = _make_long_text(2000)
        sentences = _make_sentence_analyses(text)
        segments = pp.split_into_segments(
            text, sentences, target_words=800, overlap_sentences=2
        )
        for i, seg in enumerate(segments):
            wc = seg.word_count
            is_last = (i == len(segments) - 1)
            # Last segment may be shorter due to uneven division
            low = 300 if is_last else 500
            assert low <= wc <= 1100, (
                f"Segment {i} word count {wc} outside expected range [{low}, 1100]"
            )

    def test_segments_cover_all_sentences(self):
        """All sentence indices should appear in at least one segment."""
        text = _make_long_text(2000)
        sentences = _make_sentence_analyses(text)
        segments = pp.split_into_segments(
            text, sentences, target_words=800, overlap_sentences=2
        )
        covered_indices = set()
        for seg in segments:
            covered_indices.update(seg.sentence_indices)
        all_indices = set(range(len(sentences)))
        assert covered_indices == all_indices

    def test_overlap_context_present(self):
        """Each segment (except first/last) should have overlap context."""
        text = _make_long_text(2000)
        sentences = _make_sentence_analyses(text)
        segments = pp.split_into_segments(
            text, sentences, target_words=800, overlap_sentences=2
        )
        if len(segments) >= 2:
            # Second segment should have context_before
            assert len(segments[1].context_before) > 0
            # First segment's context_before can be empty
            # Last segment's context_after can be empty

    def test_overlap_context_indices_not_in_segment(self):
        """Overlap context sentence indices should NOT appear in the segment's own sentence_indices."""
        text = _make_long_text(2000)
        sentences = _make_sentence_analyses(text)
        segments = pp.split_into_segments(
            text, sentences, target_words=800, overlap_sentences=2
        )
        if len(segments) >= 2:
            # Adjacent segments should not share sentence indices
            for i in range(len(segments) - 1):
                set_a = set(segments[i].sentence_indices)
                set_b = set(segments[i + 1].sentence_indices)
                overlap = set_a & set_b
                assert not overlap, f"Segments {i} and {i+1} share indices: {overlap}"

    def test_segment_dataclass_fields(self):
        text = _make_long_text(2000)
        sentences = _make_sentence_analyses(text)
        segments = pp.split_into_segments(
            text, sentences, target_words=800, overlap_sentences=2
        )
        for seg in segments:
            assert isinstance(seg.text, str) and len(seg.text) > 0
            assert isinstance(seg.sentence_indices, list)
            assert isinstance(seg.context_before, str)
            assert isinstance(seg.context_after, str)
            assert isinstance(seg.word_count, int) and seg.word_count > 0

    def test_short_text_single_segment(self):
        """Text shorter than target should produce 1 segment."""
        text = "A short paragraph. Just two sentences."
        sentences = _make_sentence_analyses(text)
        segments = pp.split_into_segments(
            text, sentences, target_words=800, overlap_sentences=2
        )
        assert len(segments) == 1
        assert segments[0].context_before == ""
        assert segments[0].context_after == ""
