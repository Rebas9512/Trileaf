"""
.env credential store tests (scripts/rewrite_config.py).

All tests use the isolated_env_file fixture so they never touch
PROJECT_ROOT/.env on the real filesystem.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

import scripts.rewrite_config as rc


# ── _parse_env_line ───────────────────────────────────────────────────────────

def test_parse_simple_kv() -> None:
    assert rc._parse_env_line("KEY=value") == ("KEY", "value")


def test_parse_quoted_double() -> None:
    assert rc._parse_env_line('KEY="hello world"') == ("KEY", "hello world")


def test_parse_quoted_single() -> None:
    assert rc._parse_env_line("KEY='hello world'") == ("KEY", "hello world")


def test_parse_comment_line() -> None:
    assert rc._parse_env_line("# comment") is None


def test_parse_empty_line() -> None:
    assert rc._parse_env_line("  ") is None


def test_parse_no_equals() -> None:
    assert rc._parse_env_line("KEYONLY") is None


def test_parse_empty_value() -> None:
    assert rc._parse_env_line("KEY=") == ("KEY", "")


def test_parse_strips_whitespace_key() -> None:
    assert rc._parse_env_line("  KEY  =value") == ("KEY", "value")


# ── load_dot_env ──────────────────────────────────────────────────────────────

def test_load_dot_env_basic(
    isolated_env_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    isolated_env_file.write_text("FOO=bar\nBAZ=qux\n", encoding="utf-8")
    monkeypatch.delenv("FOO", raising=False)
    monkeypatch.delenv("BAZ", raising=False)
    loaded = rc.load_dot_env(isolated_env_file)
    assert loaded == {"FOO": "bar", "BAZ": "qux"}
    assert os.environ["FOO"] == "bar"
    assert os.environ["BAZ"] == "qux"


def test_load_dot_env_skips_comments(
    isolated_env_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    isolated_env_file.write_text("# comment\nKEY=val\n", encoding="utf-8")
    monkeypatch.delenv("KEY", raising=False)
    loaded = rc.load_dot_env(isolated_env_file)
    assert loaded == {"KEY": "val"}


def test_load_dot_env_no_override(
    isolated_env_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    isolated_env_file.write_text("FOO=from_file\n", encoding="utf-8")
    monkeypatch.setenv("FOO", "from_env")
    rc.load_dot_env(isolated_env_file, override=False)
    assert os.environ["FOO"] == "from_env"


def test_load_dot_env_override(
    isolated_env_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    isolated_env_file.write_text("FOO=from_file\n", encoding="utf-8")
    monkeypatch.setenv("FOO", "from_env")
    rc.load_dot_env(isolated_env_file, override=True)
    assert os.environ["FOO"] == "from_file"


def test_load_dot_env_missing_file(tmp_path: Path) -> None:
    result = rc.load_dot_env(tmp_path / "nonexistent.env")
    assert result == {}


# ── write_dot_env ─────────────────────────────────────────────────────────────

def test_write_dot_env_creates_file(isolated_env_file: Path) -> None:
    rc.write_dot_env({"A": "1", "B": "2"}, path=isolated_env_file, merge=False)
    assert isolated_env_file.exists()
    content = isolated_env_file.read_text(encoding="utf-8")
    assert "A=1" in content
    assert "B=2" in content


def test_write_dot_env_chmod_600(isolated_env_file: Path) -> None:
    if os.name == "nt":
        pytest.skip("chmod 600 not applicable on Windows")
    rc.write_dot_env({"K": "v"}, path=isolated_env_file)
    mode = stat.S_IMODE(isolated_env_file.stat().st_mode)
    assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"


def test_write_dot_env_with_header(isolated_env_file: Path) -> None:
    rc.write_dot_env(
        {"KEY": "val"},
        path=isolated_env_file,
        header="Test header",
        merge=False,
    )
    content = isolated_env_file.read_text(encoding="utf-8")
    assert "# Test header" in content
    assert "KEY=val" in content


def test_write_dot_env_merge_updates_key(isolated_env_file: Path) -> None:
    isolated_env_file.write_text("KEY=old\nOTHER=keep\n", encoding="utf-8")
    rc.write_dot_env({"KEY": "new"}, path=isolated_env_file, merge=True)
    content = isolated_env_file.read_text(encoding="utf-8")
    assert "KEY=new" in content
    assert "OTHER=keep" in content
    assert "KEY=old" not in content


def test_write_dot_env_merge_appends_new_key(isolated_env_file: Path) -> None:
    isolated_env_file.write_text("EXISTING=yes\n", encoding="utf-8")
    rc.write_dot_env({"NEWKEY": "newval"}, path=isolated_env_file, merge=True)
    content = isolated_env_file.read_text(encoding="utf-8")
    assert "EXISTING=yes" in content
    assert "NEWKEY=newval" in content


def test_write_dot_env_fresh_on_empty_merge(isolated_env_file: Path) -> None:
    """merge=True on a nonexistent file should create fresh."""
    assert not isolated_env_file.exists()
    rc.write_dot_env({"X": "1"}, path=isolated_env_file, merge=True)
    assert "X=1" in isolated_env_file.read_text(encoding="utf-8")


# ── _read_dot_env_key ─────────────────────────────────────────────────────────

def test_read_dot_env_key_found(isolated_env_file: Path) -> None:
    isolated_env_file.write_text("A=1\nTARGET=found\nB=2\n", encoding="utf-8")
    assert rc._read_dot_env_key(isolated_env_file, "TARGET") == "found"


def test_read_dot_env_key_missing(isolated_env_file: Path) -> None:
    isolated_env_file.write_text("A=1\n", encoding="utf-8")
    assert rc._read_dot_env_key(isolated_env_file, "MISSING") is None


def test_read_dot_env_key_no_file(tmp_path: Path) -> None:
    assert rc._read_dot_env_key(tmp_path / "no.env", "KEY") is None


# ── resolve_credentials ───────────────────────────────────────────────────────

def test_resolve_credentials_from_env(
    isolated_env_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REWRITE_API_KEY already in process env → picked up directly."""
    monkeypatch.setenv("REWRITE_API_KEY", "sk-from-env")
    monkeypatch.setenv("REWRITE_BACKEND", "external")
    result = rc.resolve_credentials()
    assert result["credential_source"] in ("env", "dotenv")
    assert os.environ["REWRITE_API_KEY"] == "sk-from-env"


def test_resolve_credentials_from_dot_env(
    isolated_env_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Key in .env → loaded into os.environ."""
    isolated_env_file.write_text(
        "REWRITE_BACKEND=external\nREWRITE_API_KEY=sk-dotenv\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("REWRITE_API_KEY", raising=False)
    monkeypatch.delenv("REWRITE_BACKEND", raising=False)
    result = rc.resolve_credentials()
    assert os.environ.get("REWRITE_API_KEY") == "sk-dotenv"
    assert result["dot_env_loaded"] is True


def test_resolve_credentials_provider_fallback(
    isolated_env_file: Path,
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
    isolated_env_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing set → credential_source = 'none'."""
    monkeypatch.delenv("REWRITE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("REWRITE_PROVIDER_ID", raising=False)
    result = rc.resolve_credentials()
    assert result["credential_source"] == "none"


def test_resolve_credentials_leafhub_takes_priority(
    isolated_env_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When LeafHub returns a key it must override the .env key."""
    isolated_env_file.write_text(
        "REWRITE_BACKEND=external\nREWRITE_API_KEY=sk-from-dotenv\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("REWRITE_API_KEY", raising=False)

    # Patch _try_leafhub to simulate a linked project
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
    isolated_env_file: Path,
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
    isolated_env_file: Path,
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


def test_resolve_credentials_dotenv_auth_mode_not_overridden(
    isolated_env_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """.env REWRITE_AUTH_MODE takes precedence over LeafHub's _LEAFHUB_AUTH_MODE."""
    isolated_env_file.write_text(
        "REWRITE_BACKEND=external\nREWRITE_AUTH_MODE=bearer\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("REWRITE_API_KEY", raising=False)
    monkeypatch.delenv("REWRITE_AUTH_MODE", raising=False)

    monkeypatch.setattr(
        rc, "_try_leafhub",
        lambda *a, **k: {
            "REWRITE_API_KEY": "sk-lh",
            "_LEAFHUB_AUTH_MODE": "x-api-key",  # should lose to .env
        },
    )
    rc.resolve_credentials()
    # .env loaded first (override=False) sets REWRITE_AUTH_MODE=bearer;
    # _LEAFHUB_AUTH_MODE should not overwrite it
    assert os.environ.get("REWRITE_AUTH_MODE") == "bearer"
