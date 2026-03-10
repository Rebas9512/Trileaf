"""
Environment check tests (scripts/check_env.py).

Redirects the profile store and model paths via fixtures so no real models
or real ~/.trileaf state is required.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import scripts.check_env as ce
import scripts.rewrite_config as rc


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ── Version header ────────────────────────────────────────────────────────────

def test_check_env_prints_version(isolated_config, capsys) -> None:
    """check_env output must include a version string."""
    from scripts._version import __version__

    # Run with fake model dirs so it doesn't exit(1)
    # We patch the local_models dict via monkeypatching os.getenv
    # Actually just test that it starts with the version header
    # by intercepting the first print — simplest is to let it run to completion.
    # For this test we only care about the header line.
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            ce.main()
    except SystemExit:
        pass  # check_env may exit(1) if models are missing — that's fine here

    out = buf.getvalue()
    assert f"v{__version__}" in out


# ── All models present → passes ───────────────────────────────────────────────

def test_check_env_all_ok(
    fake_model_dir: Path,
    isolated_config,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    """check_env must succeed (no SystemExit) when all required model dirs exist."""
    monkeypatch.setenv("DESKLIB_MODEL_PATH", str(fake_model_dir))
    monkeypatch.setenv("MPNET_MODEL_PATH", str(fake_model_dir))
    # External backend → no local rewrite model needed
    monkeypatch.setenv("REWRITE_BACKEND", "external")

    # Provide a complete external profile so check_env doesn't report missing API config
    store = rc.load_store()
    rc.upsert_profile(store, "ci-ext", {
        "backend": "external",
        "base_url": "https://api.example.com/v1",
        "model": "test-model",
        "api_key": "sk-ci-test",
    })
    rc.set_active_profile(store, "ci-ext")
    rc.save_store(store)

    # Should NOT raise SystemExit
    ce.main()

    out = capsys.readouterr().out
    assert "All checks passed" in out
    assert "Detection models are ready" in out


# ── Missing model → fails with exit code 1 ───────────────────────────────────

def test_check_env_missing_desklib_exits_1(
    fake_model_dir: Path,
    isolated_config,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DESKLIB_MODEL_PATH", str(tmp_path / "missing_desklib"))
    monkeypatch.setenv("MPNET_MODEL_PATH", str(fake_model_dir))
    monkeypatch.setenv("REWRITE_BACKEND", "external")

    store = rc.load_store()
    rc.upsert_profile(store, "ci-ext2", {
        "backend": "external",
        "base_url": "https://api.example.com/v1",
        "model": "test-model",
        "api_key": "sk-ci-test",
    })
    rc.set_active_profile(store, "ci-ext2")
    rc.save_store(store)

    with pytest.raises(SystemExit) as exc_info:
        ce.main()
    assert exc_info.value.code == 1


def test_check_env_missing_mpnet_exits_1(
    fake_model_dir: Path,
    isolated_config,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DESKLIB_MODEL_PATH", str(fake_model_dir))
    monkeypatch.setenv("MPNET_MODEL_PATH", str(tmp_path / "missing_mpnet"))
    monkeypatch.setenv("REWRITE_BACKEND", "external")

    store = rc.load_store()
    rc.upsert_profile(store, "ci-ext3", {
        "backend": "external",
        "base_url": "https://api.example.com/v1",
        "model": "test-model",
        "api_key": "sk-ci-test",
    })
    rc.set_active_profile(store, "ci-ext3")
    rc.save_store(store)

    with pytest.raises(SystemExit) as exc_info:
        ce.main()
    assert exc_info.value.code == 1


# ── Local backend: also checks rewrite model ──────────────────────────────────

def test_check_env_local_backend_missing_rewrite_model_exits_1(
    fake_model_dir: Path,
    isolated_config,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DESKLIB_MODEL_PATH", str(fake_model_dir))
    monkeypatch.setenv("MPNET_MODEL_PATH", str(fake_model_dir))

    store = rc.load_store()
    # local rewrite profile with a path that doesn't exist
    rc.upsert_profile(store, rc.DEFAULT_LOCAL_PROFILE, {
        "backend": "local",
        "model_path": str(tmp_path / "missing_qwen"),
    })
    rc.set_active_profile(store, rc.DEFAULT_LOCAL_PROFILE)
    rc.save_store(store)

    with pytest.raises(SystemExit) as exc_info:
        ce.main()
    assert exc_info.value.code == 1


def test_check_env_local_backend_all_present(
    fake_model_dir: Path,
    isolated_config,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    monkeypatch.setenv("DESKLIB_MODEL_PATH", str(fake_model_dir))
    monkeypatch.setenv("MPNET_MODEL_PATH", str(fake_model_dir))

    store = rc.load_store()
    rc.upsert_profile(store, rc.DEFAULT_LOCAL_PROFILE, {
        "backend": "local",
        "model_path": str(fake_model_dir),
    })
    rc.set_active_profile(store, rc.DEFAULT_LOCAL_PROFILE)
    rc.save_store(store)

    ce.main()  # must not raise
    out = capsys.readouterr().out
    assert "All checks passed" in out


# ── External backend: checks API config ──────────────────────────────────────

def test_check_env_external_missing_api_key_exits_1(
    fake_model_dir: Path,
    isolated_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DESKLIB_MODEL_PATH", str(fake_model_dir))
    monkeypatch.setenv("MPNET_MODEL_PATH", str(fake_model_dir))
    monkeypatch.delenv("REWRITE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    store = rc.load_store()
    rc.upsert_profile(store, "ext-no-key", {
        "backend": "external",
        "base_url": "https://api.example.com/v1",
        "model": "test-model",
        # deliberately no api_key
    })
    rc.set_active_profile(store, "ext-no-key")
    rc.save_store(store)

    with pytest.raises(SystemExit) as exc_info:
        ce.main()
    assert exc_info.value.code == 1
