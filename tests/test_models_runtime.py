"""
Runtime device-selection tests.

These focus on the two required local scoring models so distribution stays
portable on CPU-only and Apple Silicon machines.
"""

from __future__ import annotations

import importlib
import sys

import pytest


def _reload_models_runtime():
    sys.modules.pop("scripts.models_runtime", None)
    return importlib.import_module("scripts.models_runtime")


def test_select_device_falls_back_to_cpu_without_accelerators(isolated_config, monkeypatch: pytest.MonkeyPatch) -> None:
    mr = _reload_models_runtime()

    monkeypatch.setattr(mr.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(mr.platform, "system", lambda: "Linux")

    class _FakeMps:
        @staticmethod
        def is_available() -> bool:
            return False

    monkeypatch.setattr(mr.torch.backends, "mps", _FakeMps, raising=False)

    assert mr.select_device() == "cpu"


def test_load_mpnet_uses_selected_device(isolated_config, monkeypatch: pytest.MonkeyPatch) -> None:
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
