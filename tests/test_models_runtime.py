"""
Runtime device-selection and rewrite extraction tests.

Device tests focus on the two required local scoring models so distribution
stays portable on CPU-only and Apple Silicon machines.

Rewrite extraction tests cover the four-layer JSON parsing chain and the
identity-check that catches silent model failures (the "always returns
original text" symptom).
"""

from __future__ import annotations

import importlib
import sys

import pytest


def _reload_models_runtime():
    sys.modules.pop("scripts.models_runtime", None)
    return importlib.import_module("scripts.models_runtime")


def test_select_device_falls_back_to_cpu_without_accelerators(monkeypatch: pytest.MonkeyPatch) -> None:
    mr = _reload_models_runtime()

    monkeypatch.setattr(mr.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(mr.platform, "system", lambda: "Linux")

    class _FakeMps:
        @staticmethod
        def is_available() -> bool:
            return False

    monkeypatch.setattr(mr.torch.backends, "mps", _FakeMps, raising=False)

    assert mr.select_device() == "cpu"


# ── Rewrite extraction tests ──────────────────────────────────────────────────
#
# These tests do NOT load any real model.  They patch _rewrite_generate so
# we can drive every code path in the extraction + identity-check pipeline.


class TestExtractRewriteOutput:
    """Unit tests for _extract_rewrite_output's four-layer parsing chain."""

    def setup_method(self):
        self.mr = _reload_models_runtime()

    def test_layer1_valid_json(self):
        """Layer-1: well-formed JSON with 'rewrite' key is extracted correctly."""
        raw = '{"rewrite": "A completely different sentence."}'
        result = self.mr._extract_rewrite_output(raw, "Original sentence.")
        assert result == "A completely different sentence."

    def test_layer1_json_with_escaped_quotes(self):
        """Layer-1: JSON where rewrite value contains escaped double-quotes."""
        raw = r'{"rewrite": "He said \"hello\" to her."}'
        result = self.mr._extract_rewrite_output(raw, "He said hello.")
        assert 'hello' in result

    def test_layer2_regex_fallback_broken_outer_json(self):
        """Layer-2: broken outer JSON but rewrite key is findable by regex."""
        raw = '{"rewrite": "Rewritten text here." trailing garbage!!'
        result = self.mr._extract_rewrite_output(raw, "Original.")
        assert "Rewritten text here" in result

    def test_layer3_truncated_json(self):
        """Layer-3: JSON truncated before the closing quote/brace."""
        raw = '{"rewrite": "Truncated rewrite without closing'
        result = self.mr._extract_rewrite_output(raw, "Original sentence.")
        assert result  # should recover something, not crash

    def test_layer4_plain_text_strips_outer_brace(self):
        """
        Layer-4 fallback: model outputs plain prose accidentally wrapped in braces
        (e.g. '{Some rewritten prose}'). Layer-4 strips the outer braces and returns
        the inner text.  If that content equals the source, the *caller*
        (run_rewrite_candidate) detects the identity and raises RewriteResponseError.
        """
        source = "The quick brown fox jumps over the lazy dog."
        raw = "{" + source + "}"
        result = self.mr._extract_rewrite_output(raw, source)
        # Extraction strips braces; content equals source → caller rejects it
        assert result.strip() == source.strip()

    def test_layer1_ignores_preamble_before_json(self):
        """
        Layer-1 scans for the first valid JSON object, skipping any preamble.
        This handles thinking-mode output where <think>...</think> precedes the JSON.
        """
        raw = "<think>Let me rewrite this carefully.</think>\n{\"rewrite\": \"Better phrasing here.\"}"
        result = self.mr._extract_rewrite_output(raw, "Original.")
        assert "Better phrasing here" in result


class TestRunRewriteCandidate:
    """
    Tests for run_rewrite_candidate's identity check.

    The 'always returns original text' symptom occurs when the model:
      (a) echoes the source text verbatim inside valid JSON, or
      (b) outputs non-JSON that extraction reduces to the original text.

    In both cases run_rewrite_candidate should raise RewriteResponseError
    rather than silently returning unchanged text.
    """

    def setup_method(self):
        self.mr = _reload_models_runtime()

    def test_raises_when_model_echoes_source_in_json(self, monkeypatch: pytest.MonkeyPatch):
        """Model returns source text unchanged inside valid JSON → RewriteResponseError."""
        source = "The cat sat on the mat."
        monkeypatch.setattr(
            self.mr,
            "_rewrite_generate",
            lambda prompt, **kw: '{"rewrite": "The cat sat on the mat."}',
        )
        with pytest.raises(self.mr.RewriteResponseError, match="unchanged"):
            self.mr.run_rewrite_candidate(source, style="balanced")

    def test_raises_when_brace_wrapped_output_matches_original(self, monkeypatch: pytest.MonkeyPatch):
        """
        Model outputs '{<source text>}' — prose wrapped in braces.
        Layer-4 strips braces → extraction equals source → RewriteResponseError.
        """
        source = "Artificial intelligence is transforming many industries."
        monkeypatch.setattr(
            self.mr,
            "_rewrite_generate",
            lambda prompt, **kw: "{" + source + "}",
        )
        with pytest.raises(self.mr.RewriteResponseError):
            self.mr.run_rewrite_candidate(source, style="conservative")

    def test_raises_on_empty_output(self, monkeypatch: pytest.MonkeyPatch):
        """Empty model output → RewriteResponseError (empty check, not identity check)."""
        monkeypatch.setattr(
            self.mr, "_rewrite_generate", lambda prompt, **kw: ""
        )
        with pytest.raises(self.mr.RewriteResponseError):
            self.mr.run_rewrite_candidate("Some source text.", style="aggressive")

    def test_succeeds_with_valid_distinct_rewrite(self, monkeypatch: pytest.MonkeyPatch):
        """When model returns a genuine rewrite, run_rewrite_candidate returns it."""
        source = "The cat sat on the mat."
        rewrite = "A feline rested upon the rug."
        monkeypatch.setattr(
            self.mr,
            "_rewrite_generate",
            lambda prompt, **kw: f'{{"rewrite": "{rewrite}"}}',
        )
        result = self.mr.run_rewrite_candidate(source, style="balanced")
        assert result == rewrite

    def test_whitespace_normalised_identity_check(self, monkeypatch: pytest.MonkeyPatch):
        """
        Identity check normalises whitespace before comparing.
        A rewrite that differs only in internal whitespace is still considered identical
        and should raise RewriteResponseError.
        """
        source = "Hello   world."
        # Model returns same words with normalised spacing
        monkeypatch.setattr(
            self.mr,
            "_rewrite_generate",
            lambda prompt, **kw: '{"rewrite": "Hello world."}',
        )
        with pytest.raises(self.mr.RewriteResponseError):
            self.mr.run_rewrite_candidate(source, style="balanced")


class TestRunRewriteEnsemble:
    """Tests for the ensemble-level fallback behaviour."""

    def setup_method(self):
        self.mr = _reload_models_runtime()

    def test_all_candidates_marked_reverted_when_model_echoes_source(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """
        When every candidate fails (model always echoes source), ensemble marks
        all three candidates as reverted_to_original and preserves the source text.
        This is the end-to-end symptom of the 'always returns original text' bug.
        """
        source = "Every candidate fails."
        monkeypatch.setattr(
            self.mr,
            "_rewrite_generate",
            lambda prompt, **kw: f'{{"rewrite": "{source}"}}',
        )
        results = self.mr.run_rewrite_ensemble(source)

        assert len(results) == 3
        styles = {r["style"] for r in results}
        assert styles == {"conservative", "balanced", "aggressive"}
        for r in results:
            assert r["text"] == source
            assert r.get("reverted_to_original") is True
            assert "error" in r

    def test_partial_failure_preserves_successful_candidates(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """
        When only one style fails, the other two return real rewrites.
        Verifies that ensemble doesn't short-circuit on partial failure.
        """
        source = "Some input text."
        call_count = {"n": 0}

        def _fake_generate(prompt, **kw):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # First call (conservative): echo source → will be rejected
                return f'{{"rewrite": "{source}"}}'
            n = call_count["n"]
            return f'{{"rewrite": "Rewritten variant {n}"}}'

        monkeypatch.setattr(self.mr, "_rewrite_generate", _fake_generate)
        results = self.mr.run_rewrite_ensemble(source)

        assert len(results) == 3
        failed = [r for r in results if r.get("reverted_to_original")]
        succeeded = [r for r in results if not r.get("reverted_to_original")]
        assert len(failed) == 1
        assert len(succeeded) == 2


# ── Device-selection tests ─────────────────────────────────────────────────────


def test_load_mpnet_uses_selected_device(monkeypatch: pytest.MonkeyPatch) -> None:
    mr = _reload_models_runtime()

    calls: dict[str, str] = {}

    class _FakeSentenceTransformer:
        def __init__(self, model_path: str, device: str | None = None) -> None:
            calls["model_path"] = model_path
            calls["device"] = device or ""

    mr._MPNET_CACHE.clear()
    monkeypatch.setattr(mr, "_HAS_ST", True)
    monkeypatch.setattr(mr, "SentenceTransformer", _FakeSentenceTransformer)
    monkeypatch.setattr(mr, "MPNET_MODEL_PATH", "/tmp/fake-mpnet")
    monkeypatch.setattr(mr, "DEVICE", "cpu")

    model = mr._load_mpnet()

    assert isinstance(model, _FakeSentenceTransformer)
    assert calls == {"model_path": "/tmp/fake-mpnet", "device": "cpu"}
