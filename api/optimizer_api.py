"""
FastAPI application for Trileaf.

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
import signal
import sys
import uuid
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set

import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

_log = logging.getLogger(__name__)

from scripts.orchestrator import PipelineResult, run_pipeline


# Set to True only after models_runtime has been fully imported (all module-level
# constants assigned).  Prevents a race where _get_runtime_health_snapshot reads
# a partially-initialised module that Python placed in sys.modules mid-import.
_RUNTIME_READY: bool = False
_STATIC_DIR = Path(__file__).resolve().parent / "static"


def _get_loaded_models_runtime():
    return sys.modules.get("scripts.models_runtime")


def _get_runtime_health_snapshot() -> Dict[str, str]:
    if _RUNTIME_READY:
        runtime = _get_loaded_models_runtime()
        if runtime is not None:
            # OPTIMIZER_DEVICE_HINT (set by run.py before launching uvicorn) carries
            # the full "cuda (NVIDIA GeForce RTX 5090)" string; mr.DEVICE is just "cuda".
            device = (
                os.getenv("OPTIMIZER_DEVICE_HINT")
                or getattr(runtime, "DEVICE", "unknown")
            )
            return {
                "device": device,
                "rewrite_backend": getattr(runtime, "REWRITE_BACKEND", "unknown"),
                "rewrite_api_kind": getattr(runtime, "REWRITE_API_KIND", ""),
                "rewrite_profile": getattr(runtime, "ACTIVE_REWRITE_PROFILE_NAME", os.getenv("REWRITE_PROFILE", "")),
                "rewrite_model": getattr(runtime, "REWRITE_MODEL", ""),
            }

    from scripts import rewrite_config

    selected = rewrite_config.load_selected_profile()
    profile = selected.get("profile") if isinstance(selected, dict) else None
    backend = rewrite_config.first_defined(
        rewrite_config.resolve_profile_value(profile, "backend"),
        os.getenv("REWRITE_BACKEND"),
        rewrite_config.legacy_env_first("backend"),
        "local",
    ) or "local"
    if backend == "openai_api":
        backend = "external"

    # For local backend the model identity comes from the path basename.
    if backend == "local":
        model_path = rewrite_config.first_defined(
            rewrite_config.resolve_profile_value(profile, "model_path"),
            os.getenv("REWRITE_MODEL_PATH"),
            rewrite_config.legacy_env_first("model_path"),
            "./models/Qwen3-VL-8B-Instruct",
        ) or "./models/Qwen3-VL-8B-Instruct"
        rewrite_model = Path(model_path).name
    else:
        rewrite_model = rewrite_config.first_defined(
            rewrite_config.resolve_profile_value(profile, "model"),
            os.getenv("REWRITE_MODEL"),
            rewrite_config.legacy_env_first("model"),
            "",
        ) or ""

    return {
        "device": os.getenv("OPTIMIZER_DEVICE_HINT", "unknown"),
        "rewrite_backend": backend,
        "rewrite_api_kind": rewrite_config.first_defined(
            rewrite_config.resolve_profile_value(profile, "api_kind"),
            os.getenv("REWRITE_API_KIND"),
            "",
        ) or "",
        "rewrite_profile": (
            str(selected["name"])
            if isinstance(selected, dict) and selected.get("name")
            else os.getenv("REWRITE_PROFILE", "")
        ),
        "rewrite_model": rewrite_model,
    }


# ─── Request / response models ─────────────────────────────────────────────────


class OptimizeRequest(BaseModel):
    text:       str
    chunk_mode: str = "short"   # "short" | "long"

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty")
        if len(v) > 50_000:
            raise ValueError("text is too long (max 50 000 characters)")
        return v

    @field_validator("chunk_mode")
    @classmethod
    def validate_chunk_mode(cls, v: str) -> str:
        if v not in ("short", "long"):
            raise ValueError("chunk_mode must be 'short' or 'long'")
        return v


# ─── In-memory session state ───────────────────────────────────────────────────


@dataclass
class SessionState:
    run_id:  str                      = ""
    status:  str                      = "idle"   # idle | running | done | error
    result:  Optional[PipelineResult] = None
    error:   Optional[str]            = None


_session = SessionState()


# ─── Idle-shutdown (browser-close) ────────────────────────────────────────────

# Seconds to wait after the last client disconnects before shutting down.
# A page refresh reconnects within ~500 ms, so 15 s is safe while still
# releasing VRAM promptly after the user genuinely closes the browser.
_IDLE_SHUTDOWN_GRACE_S: float = 15.0


async def _idle_shutdown_task(grace_s: float) -> None:
    """Wait grace_s seconds then send SIGTERM to self → triggers on_shutdown."""
    await asyncio.sleep(grace_s)
    print(f"[server] No active clients for {grace_s:.0f}s — initiating idle shutdown.")
    os.kill(os.getpid(), signal.SIGTERM)


# ─── WebSocket connection manager ──────────────────────────────────────────────


class _ConnectionManager:
    def __init__(self) -> None:
        self._connections: Set[WebSocket] = set()
        self._shutdown_handle: Optional[asyncio.Task] = None

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.add(ws)
        # Cancel any pending idle shutdown (e.g. browser refresh)
        if self._shutdown_handle and not self._shutdown_handle.done():
            self._shutdown_handle.cancel()
            self._shutdown_handle = None
            print("[server] Client reconnected — idle shutdown cancelled.")

    async def disconnect(self, ws: WebSocket) -> None:
        self._connections.discard(ws)
        if not self._connections:
            print(
                f"[server] All clients disconnected — "
                f"shutting down in {_IDLE_SHUTDOWN_GRACE_S:.0f}s "
                f"(reconnect to cancel)."
            )
            self._shutdown_handle = asyncio.create_task(
                _idle_shutdown_task(_IDLE_SHUTDOWN_GRACE_S)
            )

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


app = FastAPI(title="Trileaf")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # cannot combine wildcard origin with credentials=True
    allow_methods=["*"],
    allow_headers=["*"],
)


class _NoCacheStaticFiles(StaticFiles):
    """StaticFiles that forces browsers to revalidate on every request.

    Without Cache-Control headers, browsers apply heuristic caching and may
    serve a stale app.js even after the file has been updated on disk.
    ``no-cache`` tells the browser to always send a conditional GET; the server
    still responds 304 if the file is unchanged (no extra bandwidth), but will
    correctly return 200 when the file has changed.
    """

    async def get_response(self, path: str, scope: dict) -> Any:
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/static", _NoCacheStaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.on_event("shutdown")
async def on_shutdown() -> None:
    """Release model memory (VRAM + CPU) when the server shuts down.

    Fires when uvicorn receives SIGTERM — which happens both on Ctrl+C (via
    run.py's terminate_process_tree) and on `trileaf stop`.
    """
    from contextlib import suppress
    import gc

    print("[shutdown] Releasing model memory...")
    with suppress(Exception):
        mr = sys.modules.get("scripts.models_runtime")
        if mr is not None:
            # Explicitly del the model objects before clearing the cache dicts so
            # CPython's reference counting drops them immediately (rather than
            # waiting for the next GC cycle).  This matters for --reload restarts
            # where the process doesn't exit but the app is reloaded.
            for _cache in (mr._DESKLIB_CACHE, mr._MPNET_CACHE, mr._REWRITE_CACHE):
                for _key in list(_cache.keys()):
                    _obj = _cache.pop(_key)
                    with suppress(Exception):
                        # Model objects are tuples (tokenizer, model) or a plain model
                        _parts = _obj if isinstance(_obj, tuple) else (_obj,)
                        for _part in _parts:
                            if hasattr(_part, "cpu"):
                                _part.cpu()  # move off GPU before deleting
                    del _obj

    gc.collect()

    with suppress(Exception):
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()   # ensure all CUDA ops finished before freeing
            torch.cuda.empty_cache()
            with suppress(Exception):
                torch.cuda.ipc_collect()

    print("[shutdown] Model memory released.")


async def _warm_runtime_import() -> None:
    """Import models_runtime in the background so the first request is fast.

    For external-backend runs we skip model preloading to keep startup snappy,
    but the cold import of torch + sentence-transformers takes ~5 s on first
    use.  Running the import in a background thread during the few seconds
    between server start and the user's first click eliminates that gap.
    """
    global _RUNTIME_READY
    try:
        await asyncio.to_thread(__import__, "scripts.models_runtime")
        _RUNTIME_READY = True
        print("[startup] models_runtime import warm-up complete.")
    except Exception as exc:
        print(f"[startup] WARNING: background import warm-up failed — {exc}")


@app.on_event("startup")
async def on_startup() -> None:
    if os.getenv("OPTIMIZER_SKIP_STARTUP_PRELOAD") == "1":
        reason = os.getenv("OPTIMIZER_PREFLIGHT_ERROR", "preflight failure")
        if reason:
            print(f"[startup] Skipping preload because run.py requested fallback: {reason}")
        else:
            print("[startup] Skipping preload because run.py already completed model preflight.")
        # Warm up models_runtime import in background so the first /api/optimize
        # request doesn't hit a 5-second cold-import stall.
        asyncio.create_task(_warm_runtime_import())
        return

    global _RUNTIME_READY
    print("[startup] Preloading local model weights...")
    try:
        import scripts.models_runtime as mr

        results = await asyncio.to_thread(mr.preload_models)
        _RUNTIME_READY = True
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


@app.get("/api/ready")
async def api_ready():
    return {"status": "ok"}


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
            from scripts.app_config import get_pipeline_config
            pcfg = get_pipeline_config()
            result = await run_pipeline(
                text=req.text,
                broadcast=_broadcast,
                run_id=run_id,
                w_ai=pcfg["w_ai"],
                w_sem=pcfg["w_sem"],
                w_risk=pcfg["w_risk"],
                chunk_mode=req.chunk_mode,
            )
            _session.status = "done"
            _session.result = result
        except Exception as exc:
            _log.exception("Pipeline error in run %s", run_id)
            _session.status = "error"
            _session.error  = "An internal error occurred."
            await _manager.broadcast(
                {"type": "error", "data": {"message": "An internal error occurred."}}
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
    return {"status": "ok", **_get_runtime_health_snapshot()}
