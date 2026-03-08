"""
FastAPI application for the LLM Writing Optimizer.

Endpoints
---------
POST /api/optimize          Start a new pipeline run (async, results streamed via WS)
GET  /api/session           Current run status and summary
GET  /api/health            Device + uptime info
WS   /ws/optimizer          Real-time event stream for the dashboard

Session state is in-memory: each new /api/optimize call overwrites the previous
session.  No database persistence is required.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from scripts.orchestrator import PipelineResult, run_pipeline
import scripts.models_runtime as mr


# ─── Request / response models ─────────────────────────────────────────────────


class OptimizeRequest(BaseModel):
    text:   str
    w_ai:   float = 0.60
    w_sem:  float = 0.35
    w_risk: float = 0.05


# ─── In-memory session state ───────────────────────────────────────────────────


@dataclass
class SessionState:
    run_id:  str                      = ""
    status:  str                      = "idle"   # idle | running | done | error
    result:  Optional[PipelineResult] = None
    error:   Optional[str]            = None


_session = SessionState()


# ─── WebSocket connection manager ──────────────────────────────────────────────


class _ConnectionManager:
    def __init__(self) -> None:
        self._connections: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.add(ws)

    async def disconnect(self, ws: WebSocket) -> None:
        self._connections.discard(ws)

    async def broadcast(self, payload: Dict[str, Any]) -> None:
        dead = []
        for conn in list(self._connections):
            try:
                await conn.send_json(payload)
            except Exception:
                dead.append(conn)
        for d in dead:
            self._connections.discard(d)


_manager = _ConnectionManager()


# ─── FastAPI app ───────────────────────────────────────────────────────────────


app = FastAPI(title="LLM Writing Optimizer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
async def on_startup() -> None:
    if os.getenv("OPTIMIZER_SKIP_STARTUP_PRELOAD") == "1":
        reason = os.getenv("OPTIMIZER_PREFLIGHT_ERROR", "preflight failure")
        if reason:
            print(f"[startup] Skipping preload because run.py requested fallback: {reason}")
        else:
            print("[startup] Skipping preload because run.py already completed model preflight.")
        return

    print("[startup] Preloading local model weights...")
    try:
        results = await asyncio.to_thread(mr.preload_models)
        for item in results:
            detail = f" ({item['detail']})" if item.get("detail") else ""
            print(f"[startup] {item['name']}: {item['status']} in {item['seconds']:.3f}s{detail}")
    except RuntimeError as exc:
        print(f"[startup] WARNING: preload failed — {exc}")
        print("[startup] Server is still running; models will load lazily on first request.")


# ─── WebSocket ────────────────────────────────────────────────────────────────


@app.websocket("/ws/optimizer")
async def ws_optimizer(ws: WebSocket) -> None:
    await _manager.connect(ws)
    try:
        while True:
            await ws.receive_text()   # keep-alive; client sends nothing meaningful
    except WebSocketDisconnect:
        await _manager.disconnect(ws)


# ─── REST endpoints ───────────────────────────────────────────────────────────


@app.post("/api/optimize")
async def api_optimize(req: OptimizeRequest):
    global _session

    if _session.status == "running":
        return {"error": "A run is already in progress. Wait for it to finish."}

    run_id  = str(uuid.uuid4())
    _session = SessionState(run_id=run_id, status="running")

    async def _broadcast(payload: Dict[str, Any]) -> None:
        await _manager.broadcast(payload)

    async def _run() -> None:
        global _session
        try:
            result = await run_pipeline(
                text=req.text,
                broadcast=_broadcast,
                run_id=run_id,
                w_ai=req.w_ai,
                w_sem=req.w_sem,
                w_risk=req.w_risk,
            )
            _session.status = "done"
            _session.result = result
        except Exception as exc:
            _session.status = "error"
            _session.error  = str(exc)
            await _manager.broadcast(
                {"type": "error", "data": {"message": str(exc)}}
            )

    asyncio.create_task(_run())
    return {"run_id": run_id, "status": "started"}


@app.get("/api/session")
async def api_session():
    s = _session
    out: Dict[str, Any] = {
        "run_id": s.run_id,
        "status": s.status,
        "error":  s.error,
    }
    if s.result is not None:
        r = s.result
        out["summary"] = {
            "total_chunks":      len(r.chunks),
            "original_ai_score": round(r.original_ai_score, 4),
            "final_ai_score":    round(r.final_ai_score,    4),
            "final_sem_score":   round(r.final_sem_score,   4),
            "output_length":     len(r.output_text),
            "reverted_chunks":   sum(1 for c in r.chunks if c.reverted_to_original),
        }
    return out


@app.get("/api/health")
async def api_health():
    return {
        "status": "ok",
        "device": mr.DEVICE,
        "qwen_backend": mr.QWEN_BACKEND,
    }
