"""AuTrad AI API.  uvicorn api.main:app --workers 4"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, Response

from agents.override import OverrideBook
from audit.log import AuditLog
from risk.gate import KillSwitch

from .db import InMemoryRepo
from .routes import router
from .security import bucket, client_key


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.kill = KillSwitch()
    app.state.overrides = OverrideBook()
    app.state.audit = AuditLog(os.getenv("AUDIT_PATH"))
    pool = None
    if os.getenv("DATABASE_URL"):  # pragma: no cover
        from .db import PgRepo, create_pool

        pool = await create_pool()
        app.state.repo = PgRepo(pool)
    else:
        app.state.repo = InMemoryRepo()
    yield
    if pool is not None:  # pragma: no cover
        await pool.close()


app = FastAPI(title="AuTrad AI", version="0.1.0", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=6)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if o],
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.middleware("http")
async def edge(request: Request, call_next):
    if request.url.path.startswith("/api/") and not bucket.allow(client_key(request)):
        return JSONResponse({"detail": "rate limit exceeded"}, status_code=429, headers={"Retry-After": "1"})
    t0 = time.perf_counter()
    resp: Response = await call_next(request)
    resp.headers["Server-Timing"] = f"app;dur={(time.perf_counter() - t0) * 1000:.1f}"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    if request.method == "GET" and request.url.path.startswith("/api/"):
        resp.headers.setdefault("Cache-Control", "private, no-store")
    return resp


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/ready")
async def ready(request: Request) -> dict:
    return {"status": "ready", "kill_switch": request.app.state.kill.active}


app.include_router(router)
