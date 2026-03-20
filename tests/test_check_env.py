"""
Environment check tests (scripts/check_env.py).

Uses isolated_env_file + monkeypatch.setenv so no real models or
real PROJECT_ROOT/.env are required.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import scripts.check_env as ce


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ── Version header ────────────────────────────────────────────────────────────

def test_check_env_prints_version(isolated_env_file: Path, capsys) -> None:
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


# ── All models present → passes ───────────────────────────────────────────────

def test_check_env_all_ok(
    fake_model_dir: Path,
    isolated_env_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    """check_env must succeed (no SystemExit) when all required model dirs exist."""
    monkeypatch.setenv("DESKLIB_MODEL_PATH", str(fake_model_dir))
    monkeypatch.setenv("MPNET_MODEL_PATH", str(fake_model_dir))
    monkeypatch.setenv("REWRITE_BACKEND", "external")

    # Write a complete external config to the isolated .env
    import scripts.rewrite_config as rc
    rc.write_dot_env(
        {
            "REWRITE_BACKEND": "external",
            "REWRITE_BASE_URL": "https://api.example.com/v1",
            "REWRITE_MODEL": "test-model",
            "REWRITE_API_KEY": "sk-ci-test",
        },
        path=isolated_env_file,
        merge=False,
    )

    # Should NOT raise SystemExit
    ce.main()

    out = capsys.readouterr().out
    assert "All checks passed" in out
    assert "Detection models are ready" in out


# ── Missing model → fails with exit code 1 ───────────────────────────────────

def test_check_env_missing_desklib_exits_1(
    fake_model_dir: Path,
    isolated_env_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DESKLIB_MODEL_PATH", str(tmp_path / "missing_desklib"))
    monkeypatch.setenv("MPNET_MODEL_PATH", str(fake_model_dir))
    monkeypatch.setenv("REWRITE_BACKEND", "external")

    import scripts.rewrite_config as rc
    rc.write_dot_env(
        {
            "REWRITE_BACKEND": "external",
            "REWRITE_BASE_URL": "https://api.example.com/v1",
            "REWRITE_MODEL": "test-model",
            "REWRITE_API_KEY": "sk-ci-test",
        },
        path=isolated_env_file,
        merge=False,
    )

    with pytest.raises(SystemExit) as exc_info:
        ce.main()
    assert exc_info.value.code == 1


def test_check_env_missing_mpnet_exits_1(
    fake_model_dir: Path,
    isolated_env_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DESKLIB_MODEL_PATH", str(fake_model_dir))
    monkeypatch.setenv("MPNET_MODEL_PATH", str(tmp_path / "missing_mpnet"))
    monkeypatch.setenv("REWRITE_BACKEND", "external")

    import scripts.rewrite_config as rc
    rc.write_dot_env(
        {
            "REWRITE_BACKEND": "external",
            "REWRITE_BASE_URL": "https://api.example.com/v1",
            "REWRITE_MODEL": "test-model",
            "REWRITE_API_KEY": "sk-ci-test",
        },
        path=isolated_env_file,
        merge=False,
    )

    with pytest.raises(SystemExit) as exc_info:
        ce.main()
    assert exc_info.value.code == 1


# ── Local backend: also checks rewrite model ──────────────────────────────────

def test_check_env_local_backend_missing_rewrite_model_exits_1(
    fake_model_dir: Path,
    isolated_env_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DESKLIB_MODEL_PATH", str(fake_model_dir))
    monkeypatch.setenv("MPNET_MODEL_PATH", str(fake_model_dir))
    monkeypatch.setenv("REWRITE_BACKEND", "local")
    monkeypatch.setenv("REWRITE_MODEL_PATH", str(tmp_path / "missing_qwen"))

    with pytest.raises(SystemExit) as exc_info:
        ce.main()
    assert exc_info.value.code == 1


def test_check_env_local_backend_all_present(
    fake_model_dir: Path,
    isolated_env_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    monkeypatch.setenv("DESKLIB_MODEL_PATH", str(fake_model_dir))
    monkeypatch.setenv("MPNET_MODEL_PATH", str(fake_model_dir))
    monkeypatch.setenv("REWRITE_BACKEND", "local")
    monkeypatch.setenv("REWRITE_MODEL_PATH", str(fake_model_dir))

    ce.main()  # must not raise
    out = capsys.readouterr().out
    assert "All checks passed" in out


# ── External backend: checks API config ──────────────────────────────────────

def test_check_env_external_missing_api_key_exits_1(
    fake_model_dir: Path,
    isolated_env_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DESKLIB_MODEL_PATH", str(fake_model_dir))
    monkeypatch.setenv("MPNET_MODEL_PATH", str(fake_model_dir))
    monkeypatch.setenv("REWRITE_BACKEND", "external")
    monkeypatch.delenv("REWRITE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # .env has URL + model but no API key
    import scripts.rewrite_config as rc
    rc.write_dot_env(
        {
            "REWRITE_BACKEND": "external",
            "REWRITE_BASE_URL": "https://api.example.com/v1",
            "REWRITE_MODEL": "test-model",
        },
        path=isolated_env_file,
        merge=False,
    )

    with pytest.raises(SystemExit) as exc_info:
        ce.main()
    assert exc_info.value.code == 1
