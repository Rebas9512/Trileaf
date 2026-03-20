"""
Tests for scripts/rewrite_provider_cli.py — LeafHub wizard path.

All tests mock the LeafHub SDK so they run without a real LeafHub install
or a running LeafHub server.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import scripts.rewrite_provider_cli as cli


# ── _lh_prefill ───────────────────────────────────────────────────────────────

class TestLhPrefill:
    """_lh_prefill(hub, alias) pulls base_url/model/api_kind/auth_mode from LeafHub."""

    def _make_hub(self, **cfg_fields):
        cfg = SimpleNamespace(**cfg_fields)
        hub = MagicMock()
        hub.get_config.return_value = cfg
        return hub

    def test_returns_all_fields(self):
        hub = self._make_hub(
            base_url="https://api.minimax.io/anthropic",
            model="MiniMax-M2.5",
            api_format="anthropic-messages",
            auth_mode="bearer",
        )
        result = cli._lh_prefill(hub, "minimax")
        assert result["base_url"] == "https://api.minimax.io/anthropic"
        assert result["model"] == "MiniMax-M2.5"
        assert result["api_kind"] == "anthropic-messages"
        assert result["auth_mode"] == "bearer"

    def test_openai_completions_normalised(self):
        """LeafHub api_format 'openai-completions' → Trileaf 'openai-chat-completions'."""
        hub = self._make_hub(
            base_url="https://api.openai.com/v1",
            model="gpt-4o",
            api_format="openai-completions",
            auth_mode="bearer",
        )
        result = cli._lh_prefill(hub, "openai")
        assert result["api_kind"] == "openai-chat-completions"

    def test_openai_completions_underscore_variant_normalised(self):
        hub = self._make_hub(
            base_url="https://api.openai.com/v1",
            model="gpt-4o",
            api_format="openai_completions",
            auth_mode="bearer",
        )
        result = cli._lh_prefill(hub, "openai")
        assert result["api_kind"] == "openai-chat-completions"

    def test_unknown_format_passed_through(self):
        """Formats not in the mapping are kept as-is."""
        hub = self._make_hub(
            base_url="http://localhost:8080",
            model="llama3",
            api_format="ollama",
            auth_mode="none",
        )
        result = cli._lh_prefill(hub, "local")
        assert result["api_kind"] == "ollama"

    def test_exception_returns_empty_dict(self):
        """If get_config raises, _lh_prefill must not propagate the exception."""
        hub = MagicMock()
        hub.get_config.side_effect = RuntimeError("alias not bound")
        result = cli._lh_prefill(hub, "missing")
        assert result == {}

    def test_missing_attribute_handled_gracefully(self):
        """ProviderConfig with missing fields falls back to empty strings."""
        hub = self._make_hub(base_url="https://example.com")  # no model / api_format
        result = cli._lh_prefill(hub, "alias")
        assert result["base_url"] == "https://example.com"
        assert result["model"] == ""


# ── _wizard_leafhub: full profile → no follow-up questions ────────────────────

class TestWizardLeafhubFullProfile:
    """When LeafHub provides a complete profile, the wizard writes .env without prompting."""

    def _make_lh_found(self, hub: MagicMock) -> MagicMock:
        lh_found = MagicMock()
        lh_found.ready = True
        lh_found.open_sdk.return_value = hub
        return lh_found

    def _make_hub(self, aliases, cfg) -> MagicMock:
        hub = MagicMock()
        hub.list_aliases.return_value = aliases
        hub.get_config.return_value = SimpleNamespace(**cfg)
        return hub

    def test_full_profile_writes_env_without_prompts(
        self, isolated_env_file: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Full base_url + model from LeafHub → no _ask() calls after alias selection."""
        hub = self._make_hub(
            aliases=["minimax"],
            cfg={
                "base_url": "https://api.minimax.io/anthropic",
                "model": "MiniMax-M2.5",
                "api_format": "anthropic-messages",
                "auth_mode": "bearer",
            },
        )
        lh_found = self._make_lh_found(hub)

        # _choose selects index 0 → "minimax"
        monkeypatch.setattr(cli, "_choose", lambda prompt, options: options[0][0])
        # _ask should NOT be called for base_url/model/api_kind/auth_mode
        ask_calls: list[str] = []
        monkeypatch.setattr(cli, "_ask", lambda prompt, **_kw: ask_calls.append(prompt) or "")

        result = cli._wizard_leafhub(lh_found)

        assert result == 0
        # Only alias selection calls _ask if alias list is shown via _choose
        # base_url / model prompts must NOT appear
        for call in ask_calls:
            assert "base URL" not in call
            assert "Model" not in call

    def test_full_profile_values_written_to_env(
        self, isolated_env_file: Path, monkeypatch: pytest.MonkeyPatch
    ):
        import scripts.rewrite_config as rc

        hub = self._make_hub(
            aliases=["minimax"],
            cfg={
                "base_url": "https://api.minimax.io/anthropic",
                "model": "MiniMax-M2.5",
                "api_format": "anthropic-messages",
                "auth_mode": "bearer",
            },
        )
        lh_found = self._make_lh_found(hub)

        monkeypatch.setattr(cli, "_choose", lambda prompt, options: options[0][0])
        monkeypatch.setattr(cli, "_ask", lambda prompt, **kw: kw.get("default", ""))

        cli._wizard_leafhub(lh_found)

        assert rc._read_dot_env_key(isolated_env_file, "LEAFHUB_ALIAS") == "minimax"
        assert rc._read_dot_env_key(isolated_env_file, "REWRITE_BASE_URL") == "https://api.minimax.io/anthropic"
        assert rc._read_dot_env_key(isolated_env_file, "REWRITE_MODEL") == "MiniMax-M2.5"
        assert rc._read_dot_env_key(isolated_env_file, "REWRITE_AUTH_MODE") == "bearer"
        # anthropic-messages is not the default api_kind so it gets written
        assert rc._read_dot_env_key(isolated_env_file, "REWRITE_API_KIND") == "anthropic-messages"

    def test_partial_profile_falls_back_to_prompts(
        self, isolated_env_file: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Only base_url available (model empty) → _ask() called for missing fields."""
        hub = self._make_hub(
            aliases=["partial"],
            cfg={"base_url": "https://api.example.com", "model": "", "api_format": "", "auth_mode": "bearer"},
        )
        lh_found = self._make_lh_found(hub)

        monkeypatch.setattr(cli, "_choose", lambda prompt, options: options[0][0])
        ask_calls: list[str] = []

        def fake_ask(prompt, **kw):
            ask_calls.append(prompt)
            return kw.get("default", "")

        monkeypatch.setattr(cli, "_ask", fake_ask)

        cli._wizard_leafhub(lh_found)

        # Model name prompt must have been asked
        assert any("Model" in c for c in ask_calls)

    def test_no_bound_aliases_falls_back_to_manual_entry(
        self, isolated_env_file: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """list_aliases() returns [] → wizard falls back to manual alias entry."""
        hub = self._make_hub(
            aliases=[],
            cfg={"base_url": "", "model": "", "api_format": "", "auth_mode": "bearer"},
        )
        lh_found = self._make_lh_found(hub)

        ask_calls: list[str] = []

        def fake_ask(prompt, **kw):
            ask_calls.append(prompt)
            return kw.get("default", "rewrite")

        monkeypatch.setattr(cli, "_ask", fake_ask)

        cli._wizard_leafhub(lh_found)

        # Manual alias prompt must appear
        assert any("alias" in c.lower() for c in ask_calls)

    def test_list_aliases_exception_falls_back_to_manual_entry(
        self, isolated_env_file: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """If list_aliases() raises, wizard gracefully falls back to manual alias entry."""
        hub = MagicMock()
        hub.list_aliases.side_effect = RuntimeError("SDK error")
        cfg = SimpleNamespace(base_url="", model="", api_format="", auth_mode="bearer")
        hub.get_config.return_value = cfg
        lh_found = self._make_lh_found(hub)

        ask_calls: list[str] = []

        def fake_ask(prompt, **kw):
            ask_calls.append(prompt)
            return kw.get("default", "rewrite")

        monkeypatch.setattr(cli, "_ask", fake_ask)

        result = cli._wizard_leafhub(lh_found)

        assert result == 0
        assert any("alias" in c.lower() for c in ask_calls)
