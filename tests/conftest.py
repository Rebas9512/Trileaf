"""
Shared pytest fixtures for the Trileaf V2 test suite.

Design constraints
──────────────────
- No real model downloads — fake model dirs contain a single sentinel file.
- Rewrite-related env vars are cleared around tests via clean_env.
- All LLM / detector calls are mocked unless explicitly testing integration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ── Fake model directory ────────────────────────────────────────────────────

@pytest.fixture
def fake_model_dir(tmp_path: Path) -> Path:
    """A directory that passes _model_dir_ok() — non-empty, exists."""
    d = tmp_path / "fake_model"
    d.mkdir()
    (d / "config.json").write_text('{"model_type": "test"}', encoding="utf-8")
    return d


# ── Env isolation ───────────────────────────────────────────────────────────

@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch):
    """Remove model-path and rewrite-backend env vars so tests start from defaults."""
    for var in (
        "DESKLIB_MODEL_PATH", "MPNET_MODEL_PATH",
        "REWRITE_BASE_URL", "REWRITE_MODEL",
        "REWRITE_API_KEY", "REWRITE_PROVIDER_ID",
        "REWRITE_CREDENTIAL_SOURCE",
        "LEAFHUB_ALIAS",
    ):
        monkeypatch.delenv(var, raising=False)


# ── Sample texts ────────────────────────────────────────────────────────────

# A paragraph full of AI markers (from the strategy doc ❌ examples)
SAMPLE_AI_HEAVY = (
    "The algorithm worked — but only on its own terms. "
    "However, the results showed no direct productivity data, "
    "no pre-established benchmark, and no way to tell if anyone was better off. "
    "Furthermore, the recommendation is not to discard the algorithm, "
    "but to supplement it with a more nuanced approach. "
    "It is worth noting that this comprehensive framework navigates "
    "the complexities of organizational alignment; indeed, the empirical "
    "foundation was fresh in everyone's memory. "
    "That said, optimizing purely for 'happiness' is the core weakness."
)

# A paragraph that reads human (from the strategy doc ✅ examples)
SAMPLE_HUMAN = (
    "Top-down assignment was the fast option. "
    "Leadership had been down that road in 2014 with a smaller group "
    "and hadn't liked what they saw. "
    "Nobody could actually prove it was the best choice. "
    "It just wasn't as bad as the other two. "
    "You have to get ahead of this stuff before the thing even starts."
)

# A longer text (~300 words) for pipeline-level testing
SAMPLE_LONG = (
    "The algorithm worked — but only on its own terms. However, the results "
    "showed no direct productivity data, no pre-established benchmark, and "
    "no way to tell if anyone was better off. Furthermore, the recommendation "
    "is not to discard the algorithm, but to supplement it with a more nuanced "
    "approach. It is worth noting that this comprehensive framework navigates "
    "the complexities of organizational alignment.\n\n"
    "That said, optimizing purely for 'happiness' is the core weakness. "
    "No one had set a benchmark tied to the ten strategic priorities before "
    "the process started. The 2014 attempt was a failure and left employees "
    "resentful. This risk cannot be managed without explicitly communicating "
    "process boundaries upfront. Moreover, the constructed metric lacked an "
    "empirical foundation.\n\n"
    "The real question is whether the organization can sustain this model. "
    "Employees weren't told the full picture — manual adjustments, global "
    "tradeoffs, everything. Three options — top-down, open jobs, and an "
    "algorithm — each had downsides. Indeed, navigating the complexities "
    "of modern workforce planning requires a multifaceted, pivotal strategy "
    "that delves into the fundamental mechanisms of talent optimization."
)


@pytest.fixture
def sample_ai_heavy() -> str:
    return SAMPLE_AI_HEAVY


@pytest.fixture
def sample_human() -> str:
    return SAMPLE_HUMAN


@pytest.fixture
def sample_long() -> str:
    return SAMPLE_LONG


# ── Mock detector ───────────────────────────────────────────────────────────

@dataclass
class MockDetectorResult:
    score: float
    label: str = ""
    confidence: float = 0.0

    def __post_init__(self):
        if not self.label:
            self.label = "ai" if self.score > 0.5 else "human"
        if self.confidence == 0.0:
            self.confidence = self.score


class MockDetector:
    """
    A fake detector for testing.

    By default returns scores drawn from a pre-set list (cycling).
    Can also be configured with a dict mapping text → score.
    """

    def __init__(
        self,
        scores: Optional[List[float]] = None,
        text_scores: Optional[Dict[str, float]] = None,
        name: str = "mock-detector",
    ):
        self._name = name
        self._scores = scores or [0.45, 0.48, 0.42, 0.50, 0.47]
        self._text_scores = text_scores or {}
        self._call_idx = 0

    @property
    def name(self) -> str:
        return self._name

    def score_text(self, text: str) -> MockDetectorResult:
        if text in self._text_scores:
            s = self._text_scores[text]
        else:
            s = self._scores[self._call_idx % len(self._scores)]
            self._call_idx += 1
        return MockDetectorResult(score=s)

    def score_batch(self, texts: List[str]) -> List[MockDetectorResult]:
        return [self.score_text(t) for t in texts]


@pytest.fixture
def mock_detector() -> MockDetector:
    """A detector that returns low-variance scores (simulating poor discrimination)."""
    return MockDetector(scores=[0.45, 0.48, 0.42, 0.50, 0.47])


@pytest.fixture
def mock_detector_discriminating() -> MockDetector:
    """A detector that returns high-variance scores (good discrimination)."""
    return MockDetector(scores=[0.30, 0.85, 0.25, 0.90, 0.35, 0.80, 0.20, 0.75])


# ── Mock LLM rewrite ───────────────────────────────────────────────────────

class MockRewriter:
    """Simulates LLM rewrite calls, returning deterministic results."""

    def __init__(self, transform=None):
        """
        transform: Optional callable (text, prompt) -> str.
        If None, returns text with minor modification (lowercased first word).
        """
        self._transform = transform
        self.calls: List[Tuple[str, str, float]] = []  # (text, prompt, temperature)

    def run_rewrite_with_prompt(
        self, text: str, prompt: str, temperature: float = 0.7
    ) -> str:
        self.calls.append((text, prompt, temperature))
        if self._transform:
            return self._transform(text, prompt)
        # Default: simple deterministic "rewrite"
        words = text.split()
        if words:
            words[0] = words[0].lower()
        return " ".join(words)


@pytest.fixture
def mock_rewriter() -> MockRewriter:
    return MockRewriter()


# ── Mock MPNet similarity ───────────────────────────────────────────────────

class MockSimilarity:
    """Returns a configurable similarity score."""

    def __init__(self, default_score: float = 0.85):
        self.default_score = default_score
        self.calls: List[Tuple[str, str]] = []

    def run_mpnet_similarity(self, text1: str, text2: str) -> float:
        self.calls.append((text1, text2))
        if text1 == text2:
            return 1.0
        return self.default_score


@pytest.fixture
def mock_similarity() -> MockSimilarity:
    return MockSimilarity(default_score=0.85)
