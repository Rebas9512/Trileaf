"""
Profile store tests (scripts/rewrite_config.py).

All tests use the isolated_config fixture so they never touch
~/.trileaf on the real system.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

import scripts.rewrite_config as rc


# ── Default store ─────────────────────────────────────────────────────────────

def test_default_store_structure(isolated_config) -> None:
    store = rc.load_store()
    assert store["version"] == rc.STORE_VERSION
    assert store["active_profile"] == rc.DEFAULT_LOCAL_PROFILE
    assert rc.DEFAULT_LOCAL_PROFILE in store["profiles"]


def test_default_local_profile(isolated_config) -> None:
    store = rc.load_store()
    profile = store["profiles"][rc.DEFAULT_LOCAL_PROFILE]
    assert profile["backend"] == "local"
    assert "model_path" in profile


# ── Save / load round-trip ────────────────────────────────────────────────────

def test_save_and_reload(isolated_config) -> None:
    store = rc.load_store()
    rc.upsert_profile(store, "test-profile", {
        "backend": "external",
        "base_url": "https://api.example.com/v1",
        "model": "test-model",
        "api_key": "sk-test",
    })
    rc.save_store(store)

    reloaded = rc.load_store()
    assert "test-profile" in reloaded["profiles"]
    assert reloaded["profiles"]["test-profile"]["model"] == "test-model"


def test_saved_file_permissions(isolated_config) -> None:
    """Profile file must be chmod 600 on POSIX so secrets aren't world-readable."""
    if os.name == "nt":
        pytest.skip("chmod 600 semantics are Windows-specific — skipping on non-POSIX")
    store = rc.load_store()
    rc.save_store(store)
    mode = stat.S_IMODE(rc.CONFIG_PATH.stat().st_mode)
    assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"


# ── Profile CRUD ──────────────────────────────────────────────────────────────

def test_upsert_and_get_profile(isolated_config) -> None:
    store = rc.load_store()
    rc.upsert_profile(store, "my-profile", {"backend": "external", "model": "gpt-4o"})
    profile = rc.get_profile(store, "my-profile")
    assert profile is not None
    assert profile["model"] == "gpt-4o"


def test_set_active_profile(isolated_config) -> None:
    store = rc.load_store()
    rc.upsert_profile(store, "alt", {"backend": "external", "model": "m"})
    rc.set_active_profile(store, "alt")
    assert store["active_profile"] == "alt"


def test_set_unknown_profile_raises(isolated_config) -> None:
    store = rc.load_store()
    with pytest.raises(KeyError):
        rc.set_active_profile(store, "does-not-exist")


def test_delete_profile(isolated_config) -> None:
    store = rc.load_store()
    rc.upsert_profile(store, "disposable", {"backend": "external"})
    rc.delete_profile(store, "disposable")
    assert rc.get_profile(store, "disposable") is None
    # Active profile falls back to default after deletion
    assert store["active_profile"] == rc.DEFAULT_LOCAL_PROFILE


def test_delete_default_profile_raises(isolated_config) -> None:
    store = rc.load_store()
    with pytest.raises(ValueError):
        rc.delete_profile(store, rc.DEFAULT_LOCAL_PROFILE)


# ── Legacy migration ──────────────────────────────────────────────────────────

def test_migrate_legacy_store(isolated_config) -> None:
    """If old project-root store exists and new one doesn't, it should be migrated."""
    legacy_path: Path = isolated_config["legacy"]
    new_path: Path = isolated_config["config"]

    # Write a fake legacy store
    legacy_data = {
        "version": 1,
        "active_profile": rc.LEGACY_LOCAL_PROFILE,
        "profiles": {
            rc.LEGACY_LOCAL_PROFILE: rc._default_local_profile(),
            "my-old-api": {"backend": "external", "model": "claude-3"},
        },
    }
    legacy_path.write_text(json.dumps(legacy_data), encoding="utf-8")

    assert not new_path.exists()
    rc._migrate_legacy_store()

    assert new_path.exists(), "New config should have been created by migration"
    assert not legacy_path.exists(), "Legacy file should have been removed after migration"

    migrated = json.loads(new_path.read_text(encoding="utf-8"))
    assert "my-old-api" in migrated["profiles"]
    assert rc.DEFAULT_LOCAL_PROFILE in migrated["profiles"]
    assert rc.LEGACY_LOCAL_PROFILE not in migrated["profiles"]
    assert migrated["active_profile"] == rc.DEFAULT_LOCAL_PROFILE


def test_migration_skipped_when_new_exists(isolated_config) -> None:
    """Migration must not overwrite an existing new-format store."""
    legacy_path: Path = isolated_config["legacy"]
    new_path: Path = isolated_config["config"]

    legacy_path.write_text(
        json.dumps({"version": 1, "active_profile": rc.LEGACY_LOCAL_PROFILE, "profiles": {}}),
        encoding="utf-8",
    )
    new_path.write_text(
        json.dumps({"version": 1, "active_profile": rc.DEFAULT_LOCAL_PROFILE, "profiles": {"sentinel": {}}}),
        encoding="utf-8",
    )

    rc._migrate_legacy_store()

    # New file should be untouched
    data = json.loads(new_path.read_text(encoding="utf-8"))
    assert "sentinel" in data["profiles"]


# ── API key resolution ────────────────────────────────────────────────────────

def test_api_key_from_profile(isolated_config) -> None:
    profile = {"backend": "external", "api_key": "sk-from-profile"}
    result = rc.resolve_api_key(profile, provider_id="openai")
    assert result["value"] == "sk-from-profile"
    assert result["source"] == "profile"


def test_api_key_from_env_fallback(isolated_config, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REWRITE_API_KEY", "sk-from-env")
    profile = {"backend": "external"}  # no api_key in profile
    result = rc.resolve_api_key(profile, provider_id="openai")
    assert result["value"] == "sk-from-env"
    assert "env:" in result["source"]


def test_api_key_provider_specific_fallback(isolated_config, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REWRITE_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-specific")
    profile = {"backend": "external"}
    result = rc.resolve_api_key(profile, provider_id="openai")
    assert result["value"] == "sk-openai-specific"


def test_unresolved_placeholder_rejected(isolated_config) -> None:
    """A profile api_key that looks like ${MY_VAR} must not be treated as a real key."""
    profile = {"backend": "external", "api_key": "${MY_API_KEY}"}
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = rc.resolve_api_key(profile)
    assert result["value"] == "" or result["value"] is None or not result["value"]


# ── Env injection (_profile_to_env) ──────────────────────────────────────────

def test_local_profile_env_injection(isolated_config) -> None:
    profile = {"backend": "local", "model_path": "./models/Qwen3-VL-8B-Instruct"}
    env = rc._profile_to_env(profile)
    assert env["REWRITE_BACKEND"] == "local"
    assert "Qwen3" in env["REWRITE_MODEL_PATH"]


def test_external_profile_env_injection(isolated_config) -> None:
    profile = {
        "backend": "external",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
        "api_key": "sk-xxx",
    }
    env = rc._profile_to_env(profile)
    assert env["REWRITE_BACKEND"] == "external"
    assert env["REWRITE_BASE_URL"] == "https://api.openai.com/v1"
    assert env["REWRITE_MODEL"] == "gpt-4o"
