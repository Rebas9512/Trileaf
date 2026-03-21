"""
Shared pytest fixtures.

Design constraints
──────────────────
• No real model downloads — fake model dirs contain a single sentinel file.
• Rewrite-related env vars are cleared around tests via clean_env.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ── Fake model directory ──────────────────────────────────────────────────────

@pytest.fixture
def fake_model_dir(tmp_path: Path) -> Path:
    """A directory that passes _model_dir_ok() — non-empty, exists."""
    d = tmp_path / "fake_model"
    d.mkdir()
    (d / "config.json").write_text('{"model_type": "test"}', encoding="utf-8")
    return d


# ── Env isolation ─────────────────────────────────────────────────────────────

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
