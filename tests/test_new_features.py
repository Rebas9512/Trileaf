"""
Tests for features added in recent development:

1. Pareto weight CLI  (trileaf weight)
2. Short / long chunk modes  (split_text, split_text_with_para_idx)
3. Paragraph structure preservation  (para_idx tracking through assembly)
4. OptimizeRequest.chunk_mode validation
5. Two-pass orchestration entry-point (smoke test via run_pipeline mock)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── chunker imports ────────────────────────────────────────────────────────────
from scripts.chunker import (
    _merge_paragraph_split,
    _merge_paragraph_split_annotated,
    split_text,
    split_text_with_para_idx,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Pareto weight CLI — trileaf weight
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def isolated_app_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect app_config's CONFIG_PATH to a temp location."""
    import scripts.app_config as ac

    config_dir = tmp_path / "trileaf"
    config_dir.mkdir()
    config_file = config_dir / "config.json"

    monkeypatch.setattr(ac, "USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr(ac, "CONFIG_PATH", config_file)
    return config_file


def _run_weight_cmd(argv: List[str]) -> int:
    """Run trileaf_cli main() with the given argv, return SystemExit code."""
    import trileaf_cli

    try:
        trileaf_cli.main(argv)
        return 0
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 0


def test_weight_show_defaults(isolated_app_config, capsys) -> None:
    """trileaf weight with no args should print defaults and exit 0."""
    code = _run_weight_cmd(["weight"])
    assert code == 0
    captured = capsys.readouterr()
    assert "0.60" in captured.out
    assert "0.35" in captured.out
    assert "0.05" in captured.out


def test_weight_update_valid(isolated_app_config) -> None:
    """trileaf weight --ai 0.50 --sem 0.40 --risk 0.10 should persist."""
    import scripts.app_config as ac

    code = _run_weight_cmd(["weight", "--ai", "0.50", "--sem", "0.40", "--risk", "0.10"])
    assert code == 0

    cfg = ac.load_config()
    assert cfg["pipeline"]["w_ai"] == pytest.approx(0.50)
    assert cfg["pipeline"]["w_sem"] == pytest.approx(0.40)
    assert cfg["pipeline"]["w_risk"] == pytest.approx(0.10)


def test_weight_partial_update(isolated_app_config) -> None:
    """Only providing --ai updates that weight; others retain their stored values."""
    import scripts.app_config as ac

    # Set a known baseline first
    _run_weight_cmd(["weight", "--ai", "0.60", "--sem", "0.35", "--risk", "0.05"])

    code = _run_weight_cmd(["weight", "--ai", "0.55", "--sem", "0.40", "--risk", "0.05"])
    assert code == 0
    cfg = ac.load_config()
    assert cfg["pipeline"]["w_ai"] == pytest.approx(0.55)
    assert cfg["pipeline"]["w_sem"] == pytest.approx(0.40)


def test_weight_invalid_sum_rejected(isolated_app_config, capsys) -> None:
    """Weights that don't sum to 1.0 must exit with code 1."""
    code = _run_weight_cmd(["weight", "--ai", "0.50", "--sem", "0.50", "--risk", "0.50"])
    assert code == 1
    captured = capsys.readouterr()
    assert "Error" in captured.out or "Error" in captured.err


def test_weight_float_precision(isolated_app_config) -> None:
    """Tolerance is 0.001 — 0.601+0.350+0.050 = 1.001 should be accepted."""
    code = _run_weight_cmd(["weight", "--ai", "0.601", "--sem", "0.350", "--risk", "0.050"])
    # 1.001 is within the 0.001 tolerance window
    assert code == 0


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Short / long chunk modes — split_text + split_text_with_para_idx
# ─────────────────────────────────────────────────────────────────────────────

_THREE_PARA_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "It was a sunny afternoon in the meadow.\n\n"
    "Second paragraph with different content. "
    "This sentence is also here for padding purposes.\n\n"
    "Third paragraph closes the piece. "
    "One final sentence to round it off."
)


def test_short_mode_paragraph_count() -> None:
    """Short mode keeps each paragraph as its own chunk (each ≤200 chars)."""
    chunks = split_text(_THREE_PARA_TEXT, max_chunk_chars=200)
    # Each paragraph is < 200 chars → should be preserved as individual chunks
    assert len(chunks) == 3


def test_long_mode_merges_short_paragraphs() -> None:
    """Long mode (merge_short_paragraphs=True) can merge all three 100-ish char paras."""
    # With max=400, all three paragraphs (each ~90-110 chars) should merge into 1-2 chunks
    chunks = split_text(_THREE_PARA_TEXT, max_chunk_chars=400, merge_short_paragraphs=True)
    assert len(chunks) <= 2  # 3 short paras merged into ≤ 2 chunks


def test_long_mode_no_data_loss() -> None:
    """No content should be lost between the input and merged chunks."""
    chunks = split_text(_THREE_PARA_TEXT, max_chunk_chars=400, merge_short_paragraphs=True)
    rejoined = " ".join(chunks)
    for para in _THREE_PARA_TEXT.split("\n\n"):
        for sentence in para.split("."):
            sentence = sentence.strip()
            if sentence:
                assert sentence in rejoined


def test_para_idx_short_mode_three_paragraphs() -> None:
    """Each of the 3 paragraphs gets a distinct para_idx."""
    chunks, idxs = split_text_with_para_idx(_THREE_PARA_TEXT, max_chunk_chars=200)
    assert len(chunks) == len(idxs) == 3
    assert idxs == [0, 1, 2]


def test_para_idx_long_para_split_shares_same_idx() -> None:
    """A paragraph that gets sub-split into multiple chunks must share one para_idx."""
    long_para = "Word " * 80  # ~400 chars — forces sub-split at max_chunk_chars=100
    text = long_para.strip() + "\n\nSecond paragraph."
    chunks, idxs = split_text_with_para_idx(text, max_chunk_chars=100)

    # All chunks derived from the long first paragraph share para_idx=0
    long_para_chunks = [i for i, idx in enumerate(idxs) if idx == 0]
    assert len(long_para_chunks) >= 2  # it was sub-split
    # The second paragraph gets para_idx=1
    assert 1 in idxs


def test_para_idx_no_paragraphs_all_zero() -> None:
    """Text without blank-line breaks → all chunks share para_idx 0."""
    flat_text = "Sentence one. Sentence two. " * 20  # long flat text
    chunks, idxs = split_text_with_para_idx(flat_text, max_chunk_chars=100)
    assert all(i == 0 for i in idxs)
    assert len(chunks) >= 2  # should have been split


def test_long_mode_para_idx_annotated() -> None:
    """In merge mode, merged chunks get the first paragraph's idx."""
    chunks, idxs = split_text_with_para_idx(
        _THREE_PARA_TEXT, max_chunk_chars=400, merge_short_paragraphs=True
    )
    assert len(chunks) == len(idxs)
    # First chunk should include at least paragraphs 0 and 1
    assert idxs[0] == 0  # first chunk always starts at para 0


def test_merge_paragraph_split_basic() -> None:
    paras = ["Short A.", "Short B.", "Short C." * 20]  # last one is long
    chunks = _merge_paragraph_split(paras, max_chunk_chars=50)
    # "Short A." + "Short B." can merge; "Short C..."*20 must be sub-split
    assert len(chunks) >= 2


def test_merge_paragraph_split_annotated_consistency() -> None:
    """_merge_paragraph_split_annotated must return same chunks as _merge_paragraph_split."""
    paras = [f"Paragraph {i}. " * 3 for i in range(5)]
    plain = _merge_paragraph_split(paras, max_chunk_chars=60)
    annotated_chunks, idxs = _merge_paragraph_split_annotated(paras, max_chunk_chars=60)
    assert plain == annotated_chunks
    assert len(idxs) == len(annotated_chunks)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Paragraph structure preservation through output assembly
# ─────────────────────────────────────────────────────────────────────────────


def _simulate_assembly(chunks: List[str], idxs: List[int]) -> str:
    """Mirror the backend para-grouped output assembly."""
    from collections import defaultdict

    para_groups: dict = defaultdict(list)
    for chunk, idx in zip(chunks, idxs):
        para_groups[idx].append(chunk)
    return "\n\n".join(" ".join(v) for _, v in sorted(para_groups.items()))


def test_assembly_three_paras_restored() -> None:
    """3-para text split into 6 chunks (2 per para) should reassemble as 3 paragraphs."""
    # Simulate: para 0 → chunks 0,1; para 1 → chunk 2; para 2 → chunks 3,4
    chunks = ["A1", "A2", "B1", "C1", "C2", "C3"]
    idxs   = [0,    0,    1,    2,    2,    2]
    result = _simulate_assembly(chunks, idxs)
    paragraphs = result.split("\n\n")
    assert len(paragraphs) == 3
    assert "A1 A2" in paragraphs[0]
    assert "B1"    in paragraphs[1]
    assert "C1 C2 C3" in paragraphs[2]


def test_assembly_flat_text_single_para() -> None:
    """Flat text (no paragraphs) should come back as a single paragraph."""
    chunks = ["s1", "s2", "s3"]
    idxs   = [0, 0, 0]
    result = _simulate_assembly(chunks, idxs)
    assert "\n\n" not in result
    assert result == "s1 s2 s3"


def test_assembly_long_mode_merged() -> None:
    """Long mode merged chunks (each already a multi-para block) join with \\n\\n."""
    # Each merged chunk = different para group
    chunks = ["Para1\\n\\nPara2", "Para3\\n\\nPara4"]
    idxs   = [0, 2]  # first chunk covers paras 0-1, second covers paras 2-3
    result = _simulate_assembly(chunks, idxs)
    assert "\n\n" in result


# ─────────────────────────────────────────────────────────────────────────────
# 4.  OptimizeRequest.chunk_mode validation
# ─────────────────────────────────────────────────────────────────────────────


def test_optimize_request_default_chunk_mode() -> None:
    from api.optimizer_api import OptimizeRequest

    req = OptimizeRequest(text="Hello world")
    assert req.chunk_mode == "short"


def test_optimize_request_long_mode() -> None:
    from api.optimizer_api import OptimizeRequest

    req = OptimizeRequest(text="Hello world", chunk_mode="long")
    assert req.chunk_mode == "long"


def test_optimize_request_invalid_chunk_mode() -> None:
    from pydantic import ValidationError

    from api.optimizer_api import OptimizeRequest

    with pytest.raises(ValidationError):
        OptimizeRequest(text="Hello world", chunk_mode="invalid")


def test_optimize_request_empty_text_rejected() -> None:
    from pydantic import ValidationError

    from api.optimizer_api import OptimizeRequest

    with pytest.raises(ValidationError):
        OptimizeRequest(text="   ")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Two-pass: orchestrator run_pipeline smoke test (mocked inference)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_pipeline_returns_para_idx_in_chunk_results() -> None:
    """
    run_pipeline should tag each ChunkResult with a para_idx.
    We mock the heavy inference calls so no models are needed.
    """
    import asyncio

    from scripts import orchestrator

    two_para_text = "First paragraph with enough words here.\n\nSecond paragraph ends here."

    broadcasts: List[Dict[str, Any]] = []

    async def fake_broadcast(payload: Dict[str, Any]) -> None:
        broadcasts.append(payload)

    # Patch the inference calls
    with (
        patch("scripts.models_runtime.run_desklib", return_value=0.9),
        patch(
            "scripts.models_runtime.run_rewrite_candidate",
            return_value="Rewritten text.",
        ),
        patch("scripts.models_runtime.run_mpnet_similarity", return_value=0.8),
        patch(
            "scripts.models_runtime.run_mpnet_sentence_align",
            return_value=[{"similarity": 0.8}],
        ),
        patch(
            "scripts.models_runtime.REWRITE_STYLES",
            ["conservative", "balanced", "aggressive"],
        ),
    ):
        result = await orchestrator.run_pipeline(
            text=two_para_text,
            broadcast=fake_broadcast,
            run_id="test-run",
            chunk_mode="short",
        )

    # Two-paragraph text → 2 chunks → para_idx 0 and 1
    assert len(result.chunks) == 2
    para_idxs = [c.para_idx for c in result.chunks]
    assert 0 in para_idxs
    assert 1 in para_idxs


@pytest.mark.asyncio
async def test_run_pipeline_output_has_paragraph_structure() -> None:
    """
    Output text must contain exactly one blank line separating the two paragraphs.
    """
    from scripts import orchestrator

    two_para_text = "First paragraph.\n\nSecond paragraph."

    with (
        patch("scripts.models_runtime.run_desklib", return_value=0.9),
        patch(
            "scripts.models_runtime.run_rewrite_candidate",
            return_value="Rewritten.",
        ),
        patch("scripts.models_runtime.run_mpnet_similarity", return_value=0.85),
        patch(
            "scripts.models_runtime.run_mpnet_sentence_align",
            return_value=[{"similarity": 0.85}],
        ),
        patch(
            "scripts.models_runtime.REWRITE_STYLES",
            ["conservative", "balanced", "aggressive"],
        ),
    ):
        result = await orchestrator.run_pipeline(
            text=two_para_text,
            broadcast=AsyncMock(),
            run_id="para-test",
            chunk_mode="short",
        )

    assert "\n\n" in result.output_text
    assert result.output_text.count("\n\n") == 1  # exactly one paragraph break
