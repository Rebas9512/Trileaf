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
    isolated_env_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    yes_all,
    capsys,
) -> None:
    """If an external .env config is already complete, --yes keeps it."""
    import scripts.rewrite_config as rc

    rc.write_dot_env(
        {
            "REWRITE_BACKEND": "external",
            "REWRITE_BASE_URL": "https://api.example.com/v1",
            "REWRITE_MODEL": "test-model",
            "REWRITE_API_KEY": "sk-test",
        },
        path=isolated_env_file,
        merge=False,
    )

    result = ob.step_rewrite_provider()
    assert result is True
    out = capsys.readouterr().out
    assert "Keeping existing configuration" in out


def test_step_rewrite_provider_yes_external_skips_wizard(
    isolated_env_file: Path,
    yes_all,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    """In --yes mode with no .env config and no LeafHub, wizard is skipped."""
    # Suppress LeafHub detection so "external" is the first option
    monkeypatch.setattr("leafhub.probe.detect", lambda **kw: None, raising=False)
    result = ob.step_rewrite_provider()
    assert result is True  # succeeds by skipping wizard
    out = capsys.readouterr().out
    assert "trileaf config" in out


# ── main() end-to-end (all models present, --yes) ────────────────────────────

def test_main_yes_all_models_present(
    fake_model_dir: Path,
    isolated_env_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    """Full wizard in --yes mode with all detection models present should return 0."""
    monkeypatch.setenv("DESKLIB_MODEL_PATH", str(fake_model_dir))
    monkeypatch.setenv("MPNET_MODEL_PATH", str(fake_model_dir))
    # Suppress LeafHub detection so --yes picks "external" (skip wizard path)
    monkeypatch.setattr("leafhub.probe.detect", lambda **kw: None, raising=False)

    # Patch step_final_validation to avoid needing real models in check_env
    with patch.object(ob, "step_final_validation", return_value=True):
        result = ob.main(["--yes"])

    assert result == 0
    out = capsys.readouterr().out
    assert "Setup complete" in out
    assert "trileaf run" in out


# ── _dot_env_is_complete: local backend path validation ──────────────────────

def test_dot_env_is_complete_local_path_exists(
    isolated_env_file: Path,
    fake_model_dir: Path,
) -> None:
    """local backend + recorded path exists → complete."""
    import scripts.rewrite_config as rc

    rc.write_dot_env(
        {"REWRITE_BACKEND": "local", "REWRITE_MODEL_PATH": str(fake_model_dir)},
        path=isolated_env_file,
        merge=False,
    )
    assert ob._dot_env_is_complete() is True


def test_dot_env_is_complete_local_path_missing(
    isolated_env_file: Path,
    tmp_path: Path,
) -> None:
    """local backend + recorded path does not exist → incomplete."""
    import scripts.rewrite_config as rc

    rc.write_dot_env(
        {"REWRITE_BACKEND": "local", "REWRITE_MODEL_PATH": str(tmp_path / "nonexistent")},
        path=isolated_env_file,
        merge=False,
    )
    assert ob._dot_env_is_complete() is False


def test_dot_env_is_complete_local_no_path_recorded(
    isolated_env_file: Path,
) -> None:
    """local backend + no REWRITE_MODEL_PATH in .env → conservative True (re-check deferred)."""
    import scripts.rewrite_config as rc

    rc.write_dot_env(
        {"REWRITE_BACKEND": "local"},
        path=isolated_env_file,
        merge=False,
    )
    assert ob._dot_env_is_complete() is True


def test_dot_env_is_complete_leafhub_alias_without_url(
    isolated_env_file: Path,
) -> None:
    """external backend with LeafHub alias but missing base_url → incomplete."""
    import scripts.rewrite_config as rc

    rc.write_dot_env(
        {
            "REWRITE_BACKEND": "external",
            "LEAFHUB_ALIAS": "minimax",
            # REWRITE_BASE_URL intentionally absent
            "REWRITE_MODEL": "MiniMax-M2.5",
        },
        path=isolated_env_file,
        merge=False,
    )
    assert ob._dot_env_is_complete() is False


def test_dot_env_is_complete_leafhub_alias_full(
    isolated_env_file: Path,
) -> None:
    """external backend with LeafHub alias, base_url, and model → complete."""
    import scripts.rewrite_config as rc

    rc.write_dot_env(
        {
            "REWRITE_BACKEND": "external",
            "LEAFHUB_ALIAS": "minimax",
            "REWRITE_BASE_URL": "https://api.minimax.io/anthropic",
            "REWRITE_MODEL": "MiniMax-M2.5",
        },
        path=isolated_env_file,
        merge=False,
    )
    assert ob._dot_env_is_complete() is True


# ── step_rewrite_provider: upfront credential-manager preamble ────────────────

def test_step_rewrite_provider_leafhub_not_installed_user_wants_leafhub(
    isolated_env_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    """When LeafHub is absent and user picks LeafHub, show install instructions and return False."""
    ob._YES_ALL = False

    # LeafHub not importable
    monkeypatch.setitem(__import__("sys").modules, "leafhub", None)
    monkeypatch.setitem(__import__("sys").modules, "leafhub.probe", None)

    # Simulate: user picks "1" (LeafHub) at the credential-manager menu
    inputs = iter(["1"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))

    result = ob.step_rewrite_provider()

    assert result is False
    out = capsys.readouterr().out
    assert "install.sh" in out or "Leafhub" in out
    assert "trileaf onboard" in out


def test_step_rewrite_provider_leafhub_not_installed_user_picks_dotenv(
    isolated_env_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    """When LeafHub is absent and user picks .env, show security note and continue."""
    ob._YES_ALL = False

    monkeypatch.setitem(__import__("sys").modules, "leafhub", None)
    monkeypatch.setitem(__import__("sys").modules, "leafhub.probe", None)

    # user picks "2" (.env), then "1" (external API in the next menu)
    inputs = iter(["2", "1"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))

    # Patch the external wizard so we don't need a full config
    with patch.object(ob, "_configure_external_provider", return_value=True, create=True):
        # step will reach the existing options menu and pick "external"
        # We just verify the security note appeared and step didn't early-exit
        try:
            ob.step_rewrite_provider()
        except (StopIteration, SystemExit):
            pass  # exhausted inputs — that's fine, we only care about the output

    out = capsys.readouterr().out
    assert "Security note" in out or "plain text" in out


# ── _leafhub_fallback edge cases ──────────────────────────────────────────────

class TestLeafhubFallback:
    """_leafhub_fallback() should always land cleanly in .env path."""

    def test_fallback_yes_mode_returns_true(self, yes_all, capsys) -> None:
        """In --yes mode, fallback skips the wizard and returns True."""
        result = ob._leafhub_fallback("server not reachable")
        assert result is True
        out = capsys.readouterr().out
        assert "Falling back" in out
        assert "trileaf config" in out

    def test_fallback_prints_reason(self, yes_all, capsys) -> None:
        """Fallback output includes the failure reason."""
        ob._leafhub_fallback("connection refused on port 8765")
        out = capsys.readouterr().out
        assert "connection refused on port 8765" in out

    def test_fallback_wizard_success(
        self, isolated_env_file: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        """Fallback calls the external wizard; returns True when wizard exits 0."""
        ob._YES_ALL = False
        monkeypatch.setattr(
            "scripts.rewrite_provider_cli.main", lambda *a, **kw: 0
        )
        result = ob._leafhub_fallback("test error")
        assert result is True

    def test_fallback_wizard_nonzero_exit(
        self, isolated_env_file: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        """Fallback returns False when the wizard exits non-zero."""
        ob._YES_ALL = False
        monkeypatch.setattr(
            "scripts.rewrite_provider_cli.main", lambda *a, **kw: 1
        )
        result = ob._leafhub_fallback("test error")
        assert result is False

    def test_fallback_wizard_raises_system_exit_0(
        self, isolated_env_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fallback treats SystemExit(0) from wizard as success."""
        ob._YES_ALL = False
        monkeypatch.setattr(
            "scripts.rewrite_provider_cli.main", lambda *a, **kw: (_ for _ in ()).throw(SystemExit(0))
        )
        result = ob._leafhub_fallback("test error")
        assert result is True

    def test_fallback_wizard_exception(
        self, isolated_env_file: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        """When the fallback wizard itself raises, returns False gracefully."""
        ob._YES_ALL = False
        monkeypatch.setattr(
            "scripts.rewrite_provider_cli.main",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("disk full")),
        )
        result = ob._leafhub_fallback("test error")
        assert result is False
        out = capsys.readouterr().out
        assert "disk full" in out or "trileaf config" in out


# ── step_rewrite_provider: LeafHub register() failure → fallback ──────────────

class TestLeafhubRegisterFallback:
    """register() failures should fall through to _leafhub_fallback(), not return False."""

    def _make_linked_probe(self):
        """Return a fake ProbeResult with ready=False, can_link=True."""
        from types import SimpleNamespace
        return SimpleNamespace(ready=False, can_link=True, manage_url="http://127.0.0.1:8765")

    def test_register_runtime_error_falls_back(
        self, isolated_env_file: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        """RuntimeError from register() triggers fallback, not a hard failure."""
        ob._YES_ALL = True

        fake_probe = self._make_linked_probe()
        monkeypatch.setattr(
            "leafhub.probe.detect", lambda **kw: fake_probe, raising=False
        )
        monkeypatch.setattr(
            "leafhub.probe.register",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("LeafHub server not running")),
            raising=False,
        )

        result = ob.step_rewrite_provider()

        assert result is True  # fallback succeeded (--yes mode)
        out = capsys.readouterr().out
        assert "Falling back" in out
        assert "LeafHub server not running" in out

    def test_register_generic_exception_falls_back(
        self, isolated_env_file: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        """Any exception from register() is caught and triggers fallback."""
        ob._YES_ALL = True

        fake_probe = self._make_linked_probe()
        monkeypatch.setattr(
            "leafhub.probe.detect", lambda **kw: fake_probe, raising=False
        )
        monkeypatch.setattr(
            "leafhub.probe.register",
            lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("timeout")),
            raising=False,
        )

        result = ob.step_rewrite_provider()

        assert result is True
        out = capsys.readouterr().out
        assert "Falling back" in out

    def test_detect_unexpected_exception_treated_as_absent(
        self, isolated_env_file: Path, monkeypatch: pytest.MonkeyPatch, capsys, yes_all
    ) -> None:
        """If detect() raises an unexpected error, LeafHub is treated as not available."""
        monkeypatch.setattr(
            "leafhub.probe.detect",
            lambda **kw: (_ for _ in ()).throw(PermissionError("no access to ~/.leafhub")),
            raising=False,
        )

        # --yes mode: should still succeed (picks external/skip)
        result = ob.step_rewrite_provider()
        assert result is True
        out = capsys.readouterr().out
        # No "Falling back" message — LeafHub is simply absent; normal flow runs
        assert "trileaf config" in out

    def test_write_env_failure_falls_back(
        self, isolated_env_file: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        """If write_dot_env raises after LeafHub link, fallback is triggered."""
        ob._YES_ALL = True

        from types import SimpleNamespace
        fake_probe = SimpleNamespace(
            ready=True, can_link=True, project_name="trileaf",
        )
        monkeypatch.setattr(
            "leafhub.probe.detect", lambda **kw: fake_probe, raising=False
        )
        import scripts.rewrite_config as rc
        monkeypatch.setattr(
            rc, "write_dot_env",
            lambda *a, **kw: (_ for _ in ()).throw(OSError("permission denied")),
        )

        result = ob.step_rewrite_provider()

        assert result is True  # fallback in --yes mode
        out = capsys.readouterr().out
        assert "Falling back" in out
        assert "permission denied" in out

    def test_eof_on_alias_input_uses_default(
        self, isolated_env_file: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        """EOFError during alias/url input falls back to defaults silently."""
        ob._YES_ALL = False

        from types import SimpleNamespace
        fake_probe = SimpleNamespace(
            ready=True, can_link=True, project_name="trileaf",
        )
        monkeypatch.setattr(
            "leafhub.probe.detect", lambda **kw: fake_probe, raising=False
        )

        # Simulate EOF on the first input() call
        monkeypatch.setattr("builtins.input", lambda _p="": (_ for _ in ()).throw(EOFError()))

        result = ob.step_rewrite_provider()

        assert result is True
        out = capsys.readouterr().out
        assert "alias=rewrite" in out
