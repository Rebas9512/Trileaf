"""
Phase 1.5 — detector_interface.py acceptance tests.

Covers:
  - BaseDetector ABC contract enforcement
  - DesklibDetector wraps run_desklib correctly
  - DetectorResult dataclass fields
  - score_batch default implementation
  - Model swappability (MockDetector passes same interface)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
from unittest.mock import patch, MagicMock

import pytest

di = pytest.importorskip("scripts.detector_interface")


# ═══════════════════════════════════════════════════════════════════════════════
# DetectorResult
# ═══════════════════════════════════════════════════════════════════════════════

class TestDetectorResult:
    """DetectorResult dataclass should have score, label, confidence."""

    def test_fields(self):
        r = di.DetectorResult(score=0.75, label="ai", confidence=0.75)
        assert r.score == 0.75
        assert r.label == "ai"
        assert r.confidence == 0.75

    def test_score_range(self):
        """Score should be in [0, 1]."""
        r = di.DetectorResult(score=0.0, label="human", confidence=0.0)
        assert 0.0 <= r.score <= 1.0
        r = di.DetectorResult(score=1.0, label="ai", confidence=1.0)
        assert 0.0 <= r.score <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# BaseDetector ABC
# ═══════════════════════════════════════════════════════════════════════════════

class TestBaseDetectorABC:
    """BaseDetector cannot be instantiated directly."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            di.BaseDetector()

    def test_subclass_must_implement_score_text(self):
        """A subclass that doesn't implement score_text should raise TypeError."""

        class Incomplete(di.BaseDetector):
            @property
            def name(self) -> str:
                return "incomplete"

        with pytest.raises(TypeError):
            Incomplete()

    def test_subclass_must_implement_name(self):
        """A subclass that doesn't implement name should raise TypeError."""

        class NoName(di.BaseDetector):
            def score_text(self, text: str) -> di.DetectorResult:
                return di.DetectorResult(score=0.5, label="ai", confidence=0.5)

        with pytest.raises(TypeError):
            NoName()

    def test_valid_subclass(self):
        """A complete subclass should instantiate fine."""

        class Valid(di.BaseDetector):
            @property
            def name(self) -> str:
                return "valid"

            def score_text(self, text: str) -> di.DetectorResult:
                return di.DetectorResult(score=0.5, label="ai", confidence=0.5)

        v = Valid()
        assert v.name == "valid"
        result = v.score_text("hello")
        assert isinstance(result, di.DetectorResult)


# ═══════════════════════════════════════════════════════════════════════════════
# score_batch default implementation
# ═══════════════════════════════════════════════════════════════════════════════

class TestScoreBatch:
    """Default score_batch calls score_text for each input."""

    def test_batch_returns_list(self):
        class SimpleDetector(di.BaseDetector):
            @property
            def name(self) -> str:
                return "simple"

            def score_text(self, text: str) -> di.DetectorResult:
                return di.DetectorResult(
                    score=len(text) / 100.0,
                    label="ai",
                    confidence=len(text) / 100.0,
                )

        det = SimpleDetector()
        texts = ["short", "a medium length sentence", "x" * 100]
        results = det.score_batch(texts)

        assert len(results) == 3
        assert all(isinstance(r, di.DetectorResult) for r in results)

    def test_batch_order_preserved(self):
        """Results should correspond 1:1 with inputs."""
        scores = [0.1, 0.5, 0.9]

        class OrderedDetector(di.BaseDetector):
            def __init__(self):
                self._idx = 0

            @property
            def name(self) -> str:
                return "ordered"

            def score_text(self, text: str) -> di.DetectorResult:
                s = scores[self._idx]
                self._idx += 1
                return di.DetectorResult(score=s, label="ai", confidence=s)

        det = OrderedDetector()
        results = det.score_batch(["a", "b", "c"])
        assert [r.score for r in results] == scores


# ═══════════════════════════════════════════════════════════════════════════════
# DesklibDetector
# ═══════════════════════════════════════════════════════════════════════════════

class TestDesklibDetector:
    """DesklibDetector should wrap models_runtime.run_desklib."""

    def test_is_base_detector_subclass(self):
        assert issubclass(di.DesklibDetector, di.BaseDetector)

    def test_name(self):
        # Don't actually load the model — just check the name property
        det = di.DesklibDetector.__new__(di.DesklibDetector)
        assert isinstance(det.name, str)
        assert len(det.name) > 0

    @patch("scripts.detector_interface.mr.run_desklib", return_value=0.72)
    def test_score_text_delegates_to_run_desklib(self, mock_desklib):
        det = di.DesklibDetector.__new__(di.DesklibDetector)
        result = det.score_text("Some AI-generated text.")
        mock_desklib.assert_called_once_with("Some AI-generated text.")
        assert result.score == 0.72
        assert result.label == "ai"

    @patch("scripts.detector_interface.mr.run_desklib", return_value=0.30)
    def test_score_text_human_label(self, mock_desklib):
        det = di.DesklibDetector.__new__(di.DesklibDetector)
        result = det.score_text("Human-written text.")
        assert result.label == "human"
        assert result.score == 0.30


# ═══════════════════════════════════════════════════════════════════════════════
# Swappability
# ═══════════════════════════════════════════════════════════════════════════════

class TestSwappability:
    """Any BaseDetector subclass should work interchangeably in the pipeline."""

    def _make_detector(self, fixed_score: float):
        class FixedDetector(di.BaseDetector):
            @property
            def name(self) -> str:
                return f"fixed-{fixed_score}"

            def score_text(self, text: str) -> di.DetectorResult:
                return di.DetectorResult(
                    score=fixed_score,
                    label="ai" if fixed_score > 0.5 else "human",
                    confidence=fixed_score,
                )

        return FixedDetector()

    def test_pipeline_code_works_with_any_detector(self):
        """Simulate how orchestrator_v2 would use a detector."""
        for score in [0.2, 0.5, 0.8]:
            det = self._make_detector(score)
            texts = ["sentence one.", "sentence two.", "sentence three."]
            results = det.score_batch(texts)

            scores = [r.score for r in results]
            mean = sum(scores) / len(scores)
            std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5

            # All same score → std = 0 → model_useful = False
            assert std < 0.03
            assert all(r.score == score for r in results)

    def test_discriminating_detector(self):
        """A detector with variance should set model_useful = True."""

        class VaryingDetector(di.BaseDetector):
            @property
            def name(self) -> str:
                return "varying"

            def score_text(self, text: str) -> di.DetectorResult:
                s = len(text) % 10 / 10.0  # deterministic but varying
                return di.DetectorResult(score=s, label="ai" if s > 0.5 else "human", confidence=s)

        det = VaryingDetector()
        texts = ["a", "ab", "abc", "abcd", "abcde", "abcdef", "abcdefg"]
        results = det.score_batch(texts)
        scores = [r.score for r in results]
        mean = sum(scores) / len(scores)
        std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
        assert std > 0.03  # model_useful = True
