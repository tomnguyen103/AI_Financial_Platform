"""FastAPI gateway (arch §2.7, §3).

Wires the six modules behind one ASGI app with cross-cutting middleware:
  * CORS (origins from CORS_ALLOW_ORIGINS env; "*" only as an explicit dev opt-in)
  * structured request logging with a per-request id and latency
  * a simple in-memory token-bucket rate limiter (stands in for API Management)
  * uniform JSON error envelope

Auth + RBAC live in the route dependencies (app.security.auth.require).
"""
from __future__ import annotations

import os
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.db import init_db
from app.logging_config import configure_logging, get_logger, request_id_var
from app.routers import admin, alerts, auth, chatbot, forecasts, nl2sql

STATIC_DIR = Path(__file__).resolve().parent / "static"

configure_logging()
log = get_logger("app.gateway")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # Render's filesystem is ephemeral, so the SQLite DB is empty after every
    # cold start/redeploy. The data is deterministic synthetic (SYNTH_SEED), so
    # we just regenerate it on boot when the tables are empty — no persistent
    # disk required. Set SEED_ON_STARTUP=0 to disable (e.g. local dev).
    if os.getenv("SEED_ON_STARTUP", "1") != "0":
        from app.db import tx
        with tx() as conn:
            has_data = conn.execute("SELECT COUNT(*) FROM collections").fetchone()[0]
        if not has_data:
            log.info("empty database — seeding synthetic data")
            from scripts.seed_data import main as seed
            seed()
    log.info("startup complete")
    yield
    log.info("shutdown")


app = FastAPI(
    title="AI Financial Reporting & Revenue Intelligence Platform",
    version="0.1.0",
    description="Local runnable MVP — 6 modules behind one gateway.",
    lifespan=lifespan,
)

# CORS: default to localhost dev origins. Set CORS_ALLOW_ORIGINS to a
# comma-separated list in prod; "*" requires an explicit opt-in and is
# incompatible with credentials per the CORS spec, so we drop credentials then.
_origins_env = os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000")
_allow_origins = [o.strip() for o in _origins_env.split(",") if o.strip()]
_allow_all = _allow_origins == ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_credentials=not _allow_all,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- naive in-memory rate limiter (per client IP) --------------------------
# NOTE: per-process state. Behind multiple workers/replicas the effective limit
# is N x _RATE_MAX; swap for a shared store (Redis) when scaling horizontally.
_RATE_MAX = 120          # requests
_RATE_WINDOW = 60.0      # seconds
_hits: dict[str, deque] = defaultdict(deque)


@app.middleware("http")
async def rate_limit_and_log(request: Request, call_next):
    client = request.client.host if request.client else "unknown"
    now = time.time()
    window = _hits[client]
    while window and now - window[0] > _RATE_WINDOW:
        window.popleft()
    if len(window) >= _RATE_MAX:
        log.warning("rate limit exceeded", extra={"client": client, "path": request.url.path})
        return JSONResponse(status_code=429, content={"error": "rate limit exceeded"})
    window.append(now)

    req_id = str(uuid.uuid4())[:8]
    token = request_id_var.set(req_id)
    start = time.time()
    try:
        try:
            response = await call_next(request)
        except Exception:  # noqa: BLE001 - uniform error envelope
            latency = int((time.time() - start) * 1000)
            log.exception(
                "unhandled error",
                extra={"method": request.method, "path": request.url.path,
                       "latency_ms": latency},
            )
            return JSONResponse(
                status_code=500,
                content={"error": "internal error", "request_id": req_id},
            )
        latency = int((time.time() - start) * 1000)
        response.headers["X-Request-ID"] = req_id
        log.info(
            "request",
            extra={"method": request.method, "path": request.url.path,
                   "status": response.status_code, "latency_ms": latency,
                   "client": client},
        )
        return response
    finally:
        request_id_var.reset(token)


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", tags=["health"])
def health() -> dict:
    return {"status": "ok"}


for r in (auth.router, forecasts.router, alerts.router, chatbot.router,
          nl2sql.router, admin.router):
    app.include_router(r)


if __name__ == "__main__":
    import uvicorn
    # Bind 0.0.0.0 and honor $PORT so the container is reachable on Render
    # (Render injects PORT, default 10000). Locally defaults to 8000.
    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
