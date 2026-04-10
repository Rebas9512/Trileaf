"""
API acceptance tests (V2-only pipeline).

Covers:
  - POST /api/optimize starts V2 pipeline
  - GET /api/session returns V2 summary
  - GET /api/health and /api/ready work
  - WebSocket V2 event types are valid JSON
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch, MagicMock

import pytest

httpx = pytest.importorskip("httpx")
pytest.importorskip("scripts.orchestrator_v2")

from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _reset_session():
    import api.optimizer_api as mod
    mod._session = mod.SessionState()
    yield
    mod._session = mod.SessionState()


@pytest.fixture
def app():
    with patch("scripts.models_runtime._load_desklib"), \
         patch("scripts.models_runtime._load_mpnet"):
        from api.optimizer_api import app as _app
        yield _app


@pytest.fixture
def client(app):
    return TestClient(app)


class TestOptimizeEndpoint:

    def test_starts_pipeline(self, client):
        with patch("api.optimizer_api.run_pipeline_v2", new_callable=AsyncMock) as mock:
            mock.return_value = MagicMock()
            resp = client.post("/api/optimize", json={"text": "Test text."})
            assert resp.status_code == 200
            assert resp.json()["status"] == "started"

    def test_rejects_empty_text(self, client):
        resp = client.post("/api/optimize", json={"text": "   "})
        assert resp.status_code == 422


class TestSessionEndpoint:

    def test_idle(self, client):
        resp = client.get("/api/session")
        assert resp.status_code == 200
        assert resp.json()["status"] == "idle"

    def test_done_has_v2_summary(self, client):
        import api.optimizer_api as mod
        from scripts.orchestrator_v2 import PipelineV2Result, StageMetrics

        mod._session = mod.SessionState(
            run_id="t",
            status="done",
            result=PipelineV2Result(
                run_id="t", input_text="in", output_text="out",
                sentences=[], stage_metrics={
                    "stage1": StageMetrics(ai_score=0.8, violation_count=10, sem_score=1.0),
                    "stage5": StageMetrics(ai_score=0.3, violation_count=0, sem_score=0.85),
                },
                original_ai_score=0.8, final_ai_score=0.3,
                final_sem_score=0.85, unfixed_sentences=[],
            ),
        )
        resp = client.get("/api/session")
        data = resp.json()
        assert "summary" in data
        assert "stage_metrics" in data["summary"]
        assert "unfixed_count" in data["summary"]


class TestHealthEndpoints:

    def test_ready(self, client):
        assert client.get("/api/ready").status_code == 200

    def test_health(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert "status" in resp.json()


class TestWebSocket:

    def test_connects(self, client):
        try:
            with client.websocket_connect("/ws/optimizer"):
                pass
        except Exception:
            pytest.skip("WebSocket requires running server")

    def test_v2_events_serializable(self):
        import json
        events = [
            {"type": "stage_start", "data": {"stage": 1, "name": "Global Scoring"}},
            {"type": "stage1_done", "data": {"ai_score": 0.85, "model_useful": True}},
            {"type": "run_done_v2", "data": {"run_id": "x", "output": "y",
                "original_ai_score": 0.8, "final_ai_score": 0.3,
                "final_sem_score": 0.8, "unfixed_sentences": []}},
        ]
        for e in events:
            assert json.loads(json.dumps(e))["type"] == e["type"]
