"""
Onboarding wizard tests (scripts/onboarding.py).

No model downloads — fake_model_dir fixtures substitute for real models.
The --yes flag (via yes_all fixture) keeps the wizard fully non-interactive.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import scripts.onboarding as ob


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ── Argument parsing ──────────────────────────────────────────────────────────

def test_parse_args_defaults() -> None:
    args = ob._parse_args([])
    assert args.yes is False


def test_parse_args_yes_flag() -> None:
    for flag in ("--yes", "-y"):
        args = ob._parse_args([flag])
        assert args.yes is True


# ── _prompt_yn in headless mode ───────────────────────────────────────────────

def test_prompt_yn_yes_all_returns_default_true(yes_all) -> None:
    assert ob._prompt_yn("Question?", default=True) is True


def test_prompt_yn_yes_all_returns_default_false(yes_all) -> None:
    assert ob._prompt_yn("Question?", default=False) is False


def test_prompt_yn_interactive_yes(monkeypatch: pytest.MonkeyPatch) -> None:
    ob._YES_ALL = False
    monkeypatch.setattr("builtins.input", lambda _: "y")
    assert ob._prompt_yn("Question?") is True
    ob._YES_ALL = False


def test_prompt_yn_interactive_no(monkeypatch: pytest.MonkeyPatch) -> None:
    ob._YES_ALL = False
    monkeypatch.setattr("builtins.input", lambda _: "n")
    assert ob._prompt_yn("Question?") is False
    ob._YES_ALL = False


def test_prompt_yn_eof_returns_default(monkeypatch: pytest.MonkeyPatch) -> None:
    ob._YES_ALL = False

    def raise_eof(_):
        raise EOFError

    monkeypatch.setattr("builtins.input", raise_eof)
    assert ob._prompt_yn("Question?", default=True) is True
    assert ob._prompt_yn("Question?", default=False) is False
    ob._YES_ALL = False


# ── _prompt_choice in headless mode ──────────────────────────────────────────

def test_prompt_choice_yes_all_picks_first(yes_all) -> None:
    options = [("a", "Option A"), ("b", "Option B")]
    assert ob._prompt_choice("Pick", options) == "a"


# ── _model_dir_ok ─────────────────────────────────────────────────────────────

def test_model_dir_ok_nonempty(fake_model_dir: Path) -> None:
    assert ob._model_dir_ok(fake_model_dir) is True


def test_model_dir_ok_missing(tmp_path: Path) -> None:
    assert ob._model_dir_ok(tmp_path / "nonexistent") is False


def test_model_dir_ok_empty(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert ob._model_dir_ok(empty) is False


# ── step_env ──────────────────────────────────────────────────────────────────

def test_step_env_passes(capsys) -> None:
    """step_env must return True in a properly installed environment."""
    result = ob.step_env()
    assert result is True
    out = capsys.readouterr().out
    assert "[OK]" in out


# ── step_detection_models ─────────────────────────────────────────────────────

def test_step_detection_models_all_present(
    fake_model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    yes_all,
    capsys,
) -> None:
    """When both models are already present, step returns True without downloading."""
    monkeypatch.setenv("DESKLIB_MODEL_PATH", str(fake_model_dir))
    monkeypatch.setenv("MPNET_MODEL_PATH", str(fake_model_dir))
    result = ob.step_detection_models()
    assert result is True
    out = capsys.readouterr().out
    assert "Both detection models are present" in out
    assert "only required local model set" in out


def test_step_detection_models_missing_declines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    """When models are missing and user declines download, step returns False."""
    ob._YES_ALL = False
    monkeypatch.setenv("DESKLIB_MODEL_PATH", str(tmp_path / "desklib"))
    monkeypatch.setenv("MPNET_MODEL_PATH", str(tmp_path / "mpnet"))

    # Simulate user typing "n"
    monkeypatch.setattr("builtins.input", lambda _: "n")
    result = ob.step_detection_models()
    assert result is False
    ob._YES_ALL = False


# ── step_rewrite_provider ─────────────────────────────────────────────────────

def test_step_rewrite_provider_external_configured(
    isolated_config,
    monkeypatch: pytest.MonkeyPatch,
    yes_all,
    capsys,
) -> None:
    """If an external profile is already complete, --yes keeps it."""
    import scripts.rewrite_config as rc

    store = rc.load_store()
    rc.upsert_profile(store, "test-ext", {
        "backend": "external",
        "base_url": "https://api.example.com/v1",
        "model": "test-model",
        "api_key": "sk-test",
    })
    rc.set_active_profile(store, "test-ext")
    rc.save_store(store)

    result = ob.step_rewrite_provider()
    assert result is True
    out = capsys.readouterr().out
    assert "Keeping existing configuration" in out


def test_step_rewrite_provider_yes_external_skips_wizard(
    isolated_config,
    yes_all,
    capsys,
) -> None:
    """In --yes mode, selecting 'external' skips the interactive wizard."""
    # No external profile configured → wizard would be needed, but --yes skips it
    result = ob.step_rewrite_provider()
    assert result is True  # succeeds by skipping wizard
    out = capsys.readouterr().out
    assert "trileaf config" in out


# ── main() end-to-end (all models present, --yes) ────────────────────────────

def test_main_yes_all_models_present(
    fake_model_dir: Path,
    isolated_config,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    """Full wizard in --yes mode with all detection models present should return 0."""
    monkeypatch.setenv("DESKLIB_MODEL_PATH", str(fake_model_dir))
    monkeypatch.setenv("MPNET_MODEL_PATH", str(fake_model_dir))

    # Patch step_final_validation to avoid needing real models in check_env
    with patch.object(ob, "step_final_validation", return_value=True):
        result = ob.main(["--yes"])

    assert result == 0
    out = capsys.readouterr().out
    assert "Setup complete" in out
    assert "trileaf run" in out
