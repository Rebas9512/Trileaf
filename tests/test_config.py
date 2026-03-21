"""
Credential resolution tests (scripts/rewrite_config.py).

.env file functions have been removed — credentials now come exclusively
from LeafHub or existing env vars.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import scripts.rewrite_config as rc


# ── resolve_credentials ───────────────────────────────────────────────────────

def test_resolve_credentials_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REWRITE_API_KEY already in process env → picked up directly."""
    monkeypatch.setenv("REWRITE_API_KEY", "sk-from-env")
    result = rc.resolve_credentials()
    assert result["credential_source"] == "env"
    assert os.environ["REWRITE_API_KEY"] == "sk-from-env"


def test_resolve_credentials_provider_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OPENAI_API_KEY used as fallback when REWRITE_API_KEY absent."""
    monkeypatch.delenv("REWRITE_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-fallback")
    monkeypatch.setenv("REWRITE_PROVIDER_ID", "openai")
    result = rc.resolve_credentials()
    assert os.environ.get("REWRITE_API_KEY") == "sk-openai-fallback"
    assert "env:OPENAI_API_KEY" in result["credential_source"]


def test_resolve_credentials_no_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing set → credential_source = 'none'."""
    monkeypatch.delenv("REWRITE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("REWRITE_PROVIDER_ID", raising=False)
    result = rc.resolve_credentials()
    assert result["credential_source"] == "none"


def test_resolve_credentials_leafhub_takes_priority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When LeafHub returns a key it must override the env key."""
    monkeypatch.delenv("REWRITE_API_KEY", raising=False)

    monkeypatch.setattr(
        rc, "_try_leafhub",
        lambda *a, **k: {"REWRITE_API_KEY": "sk-from-leafhub"},
    )
    result = rc.resolve_credentials()
    assert os.environ.get("REWRITE_API_KEY") == "sk-from-leafhub"
    assert result["leafhub_active"] is True


# ── Placeholder rejection ─────────────────────────────────────────────────────

def test_unresolved_placeholder_rejected() -> None:
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = rc._trim_credential("${MY_API_KEY}")
    assert result is None
    assert len(w) == 1


def test_dollar_only_placeholder_rejected() -> None:
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = rc._trim_credential("$MY_API_KEY")
    assert result is None
    assert len(w) == 1


def test_real_key_not_rejected() -> None:
    assert rc._trim_credential("sk-real-key-abc123") == "sk-real-key-abc123"


# ── Provider key candidates ───────────────────────────────────────────────────

def test_provider_candidates_openai() -> None:
    cands = rc.get_provider_env_api_key_candidates("openai")
    assert "REWRITE_API_KEY" in cands
    assert "OPENAI_API_KEY" in cands
    assert cands[0] == "REWRITE_API_KEY"


def test_provider_candidates_unknown() -> None:
    cands = rc.get_provider_env_api_key_candidates("unknown-provider")
    assert cands == ["REWRITE_API_KEY"]


def test_provider_candidates_alias() -> None:
    """custom-openai should resolve to openai's candidates."""
    cands = rc.get_provider_env_api_key_candidates("custom-openai")
    assert "OPENAI_API_KEY" in cands


# ── first_defined ─────────────────────────────────────────────────────────────

def test_first_defined_picks_first_non_empty() -> None:
    assert rc.first_defined(None, "", "found", "second") == "found"


def test_first_defined_all_empty_returns_none() -> None:
    assert rc.first_defined(None, "", None) is None


def test_first_defined_single_value() -> None:
    assert rc.first_defined("only") == "only"


# ── normalize_provider_id ─────────────────────────────────────────────────────

def test_normalize_provider_id_basic() -> None:
    assert rc.normalize_provider_id("OpenAI") == "openai"


def test_normalize_provider_id_spaces() -> None:
    assert rc.normalize_provider_id("My Provider") == "my-provider"


def test_normalize_provider_id_empty() -> None:
    assert rc.normalize_provider_id("") == ""


# ── mask_secret ───────────────────────────────────────────────────────────────

def test_mask_secret_long() -> None:
    masked = rc.mask_secret("sk-abcdefghijklmn")
    assert "sk-a" in masked
    assert "lmn" in masked
    assert "..." in masked


def test_mask_secret_short() -> None:
    assert rc.mask_secret("abc") == "***"


def test_mask_secret_empty() -> None:
    assert rc.mask_secret("") == "(empty)"


# ── LeafHub optional-field propagation ────────────────────────────────────────

def test_resolve_credentials_leafhub_propagates_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REWRITE_BASE_URL from LeafHub is injected into os.environ."""
    monkeypatch.delenv("REWRITE_API_KEY", raising=False)
    monkeypatch.delenv("REWRITE_BASE_URL", raising=False)

    monkeypatch.setattr(
        rc, "_try_leafhub",
        lambda *a, **k: {
            "REWRITE_API_KEY": "sk-lh",
            "REWRITE_BASE_URL": "https://api.minimax.io/anthropic",
        },
    )
    rc.resolve_credentials()
    assert os.environ.get("REWRITE_BASE_URL") == "https://api.minimax.io/anthropic"


def test_resolve_credentials_leafhub_propagates_auth_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_LEAFHUB_AUTH_MODE is applied to REWRITE_AUTH_MODE when not already set."""
    monkeypatch.delenv("REWRITE_API_KEY", raising=False)
    monkeypatch.delenv("REWRITE_AUTH_MODE", raising=False)

    monkeypatch.setattr(
        rc, "_try_leafhub",
        lambda *a, **k: {
            "REWRITE_API_KEY": "sk-lh",
            "_LEAFHUB_AUTH_MODE": "x-api-key",
        },
    )
    rc.resolve_credentials()
    assert os.environ.get("REWRITE_AUTH_MODE") == "x-api-key"
