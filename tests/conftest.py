"""
Shared pytest fixtures.

Design constraints
──────────────────
• No real model downloads — fake model dirs contain a single sentinel file.
• ENV_FILE is redirected to a tmp_path so tests never touch PROJECT_ROOT/.env.
• _YES_ALL in onboarding is reset between tests via the yes_all fixture.
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


# ── Isolated .env file ────────────────────────────────────────────────────────

@pytest.fixture
def isolated_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """
    Redirect rewrite_config.ENV_FILE to a temp location.
    Tests that call rewrite_config functions (load_dot_env, write_dot_env,
    resolve_credentials, _read_dot_env_key) will use this temp file instead
    of the real PROJECT_ROOT/.env.
    """
    import scripts.rewrite_config as rc

    env_file = tmp_path / ".env"
    monkeypatch.setattr(rc, "ENV_FILE", env_file)
    return env_file


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
        "REWRITE_BACKEND",
        "REWRITE_MODEL_PATH", "QWEN_MODEL_PATH",
        "REWRITE_BASE_URL", "REWRITE_MODEL",
        "REWRITE_API_KEY", "REWRITE_PROVIDER_ID",
        "REWRITE_CREDENTIAL_SOURCE",
        "QWEN_BACKEND", "QWEN_API_BASE_URL", "QWEN_API_MODEL",
        "LEAFHUB_ALIAS",
    ):
        monkeypatch.delenv(var, raising=False)
