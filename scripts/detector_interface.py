"""
AI detection model abstraction layer.

Provides a pluggable interface so the pipeline can swap detection models
without changing orchestration logic. All detectors implement BaseDetector
and return DetectorResult.

Current implementations
───────────────────────
- DesklibDetector  — wraps models_runtime.run_desklib (DebertaV2 classifier)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import scripts.models_runtime as mr


# ═══════════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DetectorResult:
    """Result from a single detection call."""
    score: float            # model raw output [0, 1]
    label: str              # "ai" | "human"
    confidence: float       # model confidence (some models lack this; default = score)


# ═══════════════════════════════════════════════════════════════════════════════
# Abstract base
# ═══════════════════════════════════════════════════════════════════════════════

class BaseDetector(ABC):
    """All AI detection models must subclass this."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name for logs and UI."""

    @abstractmethod
    def score_text(self, text: str) -> DetectorResult:
        """Score a single text. Returns DetectorResult."""

    def score_batch(self, texts: List[str]) -> List[DetectorResult]:
        """Score multiple texts. Default: sequential calls to score_text."""
        return [self.score_text(t) for t in texts]


# ═══════════════════════════════════════════════════════════════════════════════
# Desklib implementation
# ═══════════════════════════════════════════════════════════════════════════════

class DesklibDetector(BaseDetector):
    """Wraps the existing Desklib DebertaV2 AI-text detector."""

    @property
    def name(self) -> str:
        return "desklib-v1.01"

    def score_text(self, text: str) -> DetectorResult:
        score = mr.run_desklib(text)
        return DetectorResult(
            score=score,
            label="ai" if score > 0.5 else "human",
            confidence=score,
        )
