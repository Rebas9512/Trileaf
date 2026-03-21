"""
Environment check tests (scripts/check_env.py).

Uses monkeypatch.setenv so no real models or .env file are required.
The rewrite backend is always 'external' (local model support removed).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import scripts.check_env as ce


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ── Version header ────────────────────────────────────────────────────────────

def test_check_env_prints_version(capsys) -> None:
    """check_env output must include a version string."""
    from scripts._version import __version__

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


# ── All models present + external backend → passes ───────────────────────────

def test_check_env_all_ok(
    fake_model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    """check_env must succeed (no SystemExit) when detection models exist and API config is set."""
    monkeypatch.setenv("DESKLIB_MODEL_PATH", str(fake_model_dir))
    monkeypatch.setenv("MPNET_MODEL_PATH", str(fake_model_dir))

    monkeypatch.setenv("REWRITE_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("REWRITE_MODEL", "test-model")
    monkeypatch.setenv("REWRITE_API_KEY", "sk-ci-test")

    ce.main()  # must not raise

    out = capsys.readouterr().out
    assert "All checks passed" in out
    assert "Detection models are ready" in out


# ── Missing model → fails with exit code 1 ───────────────────────────────────

def test_check_env_missing_desklib_exits_1(
    fake_model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DESKLIB_MODEL_PATH", str(tmp_path / "missing_desklib"))
    monkeypatch.setenv("MPNET_MODEL_PATH", str(fake_model_dir))

    monkeypatch.setenv("REWRITE_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("REWRITE_MODEL", "test-model")
    monkeypatch.setenv("REWRITE_API_KEY", "sk-ci-test")

    with pytest.raises(SystemExit) as exc_info:
        ce.main()
    assert exc_info.value.code == 1


def test_check_env_missing_mpnet_exits_1(
    fake_model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DESKLIB_MODEL_PATH", str(fake_model_dir))
    monkeypatch.setenv("MPNET_MODEL_PATH", str(tmp_path / "missing_mpnet"))

    monkeypatch.setenv("REWRITE_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("REWRITE_MODEL", "test-model")
    monkeypatch.setenv("REWRITE_API_KEY", "sk-ci-test")

    with pytest.raises(SystemExit) as exc_info:
        ce.main()
    assert exc_info.value.code == 1


# ── External backend: checks API config ──────────────────────────────────────

def test_check_env_external_missing_api_key_exits_1(
    fake_model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DESKLIB_MODEL_PATH", str(fake_model_dir))
    monkeypatch.setenv("MPNET_MODEL_PATH", str(fake_model_dir))

    monkeypatch.setenv("REWRITE_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("REWRITE_MODEL", "test-model")
    monkeypatch.delenv("REWRITE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LEAFHUB_ALIAS", raising=False)

    with pytest.raises(SystemExit) as exc_info:
        ce.main()
    assert exc_info.value.code == 1


def test_check_env_external_missing_base_url_exits_1(
    fake_model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DESKLIB_MODEL_PATH", str(fake_model_dir))
    monkeypatch.setenv("MPNET_MODEL_PATH", str(fake_model_dir))

    monkeypatch.delenv("REWRITE_BASE_URL", raising=False)
    monkeypatch.setenv("REWRITE_MODEL", "test-model")
    monkeypatch.setenv("REWRITE_API_KEY", "sk-ci-test")

    with pytest.raises(SystemExit) as exc_info:
        ce.main()
    assert exc_info.value.code == 1
