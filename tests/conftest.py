"""
Shared pytest fixtures.

Design constraints
──────────────────
• No real model downloads — fake model dirs contain a single sentinel file.
• Profile store is redirected to a tmp_path so tests never touch ~/.trileaf.
• _YES_ALL in onboarding is reset between tests via the yes_all fixture.
• Legacy env aliases are cleared around config tests to avoid cross-contamination.
"""

from __future__ import annotations

import importlib
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


# ── Isolated profile store ────────────────────────────────────────────────────

@pytest.fixture
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Redirect rewrite_config's CONFIG_PATH and USER_CONFIG_DIR to a temp location.
    Also neutralise the legacy path so migration tests are explicit.
    """
    import scripts.rewrite_config as rc

    config_dir = tmp_path / "user-config"
    config_dir.mkdir()
    config_file = config_dir / "rewrite_profiles.json"
    legacy_file = tmp_path / "legacy_profiles.json"  # does NOT exist by default

    monkeypatch.setattr(rc, "USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr(rc, "CONFIG_PATH", config_file)
    monkeypatch.setattr(rc, "_LEGACY_CONFIG_PATH", legacy_file)
    monkeypatch.setattr(rc, "_LEGACY_USER_CONFIG_PATH", tmp_path / "legacy_user_profiles.json")

    return {"dir": config_dir, "config": config_file, "legacy": legacy_file}


# ── Headless onboarding ───────────────────────────────────────────────────────

@pytest.fixture
def yes_all(monkeypatch: pytest.MonkeyPatch):
    """Force onboarding._YES_ALL = True and reset it after the test."""
    import scripts.onboarding as ob
    monkeypatch.setattr(ob, "_YES_ALL", True)
    yield
    monkeypatch.setattr(ob, "_YES_ALL", False)


# ── Env isolation ─────────────────────────────────────────────────────────────

@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch):
    """Remove model-path and rewrite-backend env vars so tests start from defaults."""
    for var in (
        "DESKLIB_MODEL_PATH", "MPNET_MODEL_PATH",
        "REWRITE_BACKEND", "QWEN_BACKEND",
        "REWRITE_MODEL_PATH", "QWEN_MODEL_PATH",
        "REWRITE_PROFILE", "REWRITE_BASE_URL", "REWRITE_MODEL", "REWRITE_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
