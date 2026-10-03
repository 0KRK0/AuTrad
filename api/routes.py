from __future__ import annotations

import time

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from agents.override import ACTIONS, Override
from agents.taxonomy import POLICIES, SPECULATIVE, SPECULATIVE_SLEEVE, Category
from backtest.engine import BacktestConfig, run_backtest
from marketdata.loaders import synthetic
from risk.gate import RiskLimits

from .cache import cached, stats
from .security import require

router = APIRouter(prefix="/api/v1")


class BacktestRequest(BaseModel):
    symbols: list[str] = Field(default_factory=lambda: ["SYN0", "SYN1", "SYN2", "SYN3"], max_length=50)
    n_bars: int = Field(1500, ge=200, le=5000)
    seed: int = 0
    folds: int = Field(4, ge=1, le=10)


def _downsample(eq, max_points: int = 400) -> list[list]:
    idx = np.unique(np.linspace(0, len(eq) - 1, min(max_points, len(eq))).astype(int))
    return [[eq.index[i].isoformat(), round(float(eq.iloc[i]), 2)] for i in idx]


@cached("backtest", ttl=3600)
async def _cached_backtest(symbols: tuple[str, ...], n_bars: int, seed: int, folds: int) -> dict:
    def work() -> dict:
        data = {s: synthetic(s, n=n_bars, seed=seed + i) for i, s in enumerate(symbols)}
        res = run_backtest(data, BacktestConfig(folds=folds))
        res["equity"] = _downsample(res["equity"])
        res["data_source"] = "synthetic (mechanics test only)"
        return res

    return await run_in_threadpool(work)  # CPU-bound: keep the event loop free


@router.post("/backtests")
async def create_backtest(req: BacktestRequest, _: str = Depends(require("researcher"))) -> dict:
    t0 = time.perf_counter()
    res = await _cached_backtest(tuple(req.symbols), req.n_bars, req.seed, req.folds)
    return {**res, "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)}


@router.get("/orders")
async def list_orders(request: Request, cursor: int | None = None, limit: int = Query(50, ge=1, le=500),
                      _: str = Depends(require("viewer"))) -> dict:
    page = await request.app.state.repo.list_orders(cursor, limit)
    return {"items": page.items, "next_cursor": page.next_cursor}


@router.get("/evidence")
async def list_evidence(request: Request, symbol: str | None = None, cursor: int | None = None,
                        limit: int = Query(50, ge=1, le=200), _: str = Depends(require("viewer"))) -> dict:
    page = await request.app.state.repo.list_evidence(symbol, cursor, limit)
    return {"items": page.items, "next_cursor": page.next_cursor}


@router.get("/risk/limits")
async def risk_limits(_: str = Depends(require("viewer"))) -> dict:
    return RiskLimits().__dict__


@router.get("/killswitch")
async def killswitch_status(request: Request, _: str = Depends(require("viewer"))) -> dict:
    k = request.app.state.kill
    return {"active": k.active, "reason": k.reason, "tripped_at": k.tripped_at}


class TripRequest(BaseModel):
    reason: str = Field("manual", max_length=200)


@router.post("/killswitch/trip")
async def killswitch_trip(body: TripRequest, request: Request, role: str = Depends(require("trader"))) -> dict:
    request.app.state.kill.trip(f"{body.reason} (by {role})")
    request.app.state.audit.write("kill_switch", {"action": "trip", "reason": body.reason, "role": role})
    return {"active": True}


@router.post("/killswitch/reset")
async def killswitch_reset(request: Request, role: str = Depends(require("risk_officer"))) -> dict:
    request.app.state.kill.reset()
    request.app.state.audit.write("kill_switch", {"action": "reset", "role": role})
    return {"active": False}


@router.get("/system/cache")
async def cache_stats(_: str = Depends(require("admin"))) -> dict:
    return stats


# ---------- Category agents + manual override ----------

@router.get("/agents")
async def list_agents(request: Request, _: str = Depends(require("viewer"))) -> dict:
    paused = request.app.state.overrides.paused
    return {
        "speculative_sleeve": SPECULATIVE_SLEEVE,
        "agents": [{"category": c.value, "speculative": c in SPECULATIVE, "paused": c in paused, **p.__dict__}
                   for c, p in POLICIES.items()],
    }


class OverrideRequest(BaseModel):
    action: str = Field(pattern="|".join(sorted(ACTIONS)))
    symbol: str | None = Field(None, max_length=40)
    category: Category | None = None
    weight: float = Field(0.0, ge=-1.0, le=1.0)
    pin_hours: float = Field(24.0, ge=0, le=24 * 30)
    reason: str = Field(..., min_length=3, max_length=200)


@router.post("/overrides")
async def create_override(body: OverrideRequest, request: Request, role: str = Depends(require("trader"))) -> dict:
    try:
        o = Override(body.action, role, body.reason, body.symbol, body.category, body.weight,
                     expires_at=time.time() + body.pin_hours * 3600)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    request.app.state.overrides.add(o)
    request.app.state.audit.write("override", {**o.__dict__, "category": o.category.value if o.category else None})
    return {"id": o.id, "status": "queued", "note": "executes next cycle through the Risk Gate (kill switch still applies)"}


@router.get("/overrides")
async def list_overrides(request: Request, _: str = Depends(require("viewer"))) -> dict:
    book = request.app.state.overrides
    return {"paused": sorted(c.value for c in book.paused),
            "items": [{**o.__dict__, "category": o.category.value if o.category else None} for o in book.items[-200:]]}
