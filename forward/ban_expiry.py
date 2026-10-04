"""v1.3 Part D: forward-only tracking of the v1.2 in-sample observation "stocks in an F&O ban during the last 3
sessions before expiry rose +0.9 % (market-adjusted) from T-4 to T". Not a strategy; no backfill ever: only expiries on
or after FORWARD_START are recorded. Hash-chained ledger: data/forward/ban_expiry/ledger.jsonl.

    python -m forward.ban_expiry update        # after new F&O/ban/price data is in the store
    python -m forward.ban_expiry status        # evaluation only when >= 24 months AND >= 300 events
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from forward.ledger import Ledger

FORWARD_START = pd.Timestamp("2026-10-05")
LEDGER = Path("data/forward/ban_expiry/ledger.jsonl")
MIN_MONTHS, MIN_EVENTS = 24, 300


def candidate_events(asof: pd.Timestamp) -> pd.DataFrame:
    from marketdata.nse_fo import load_fo
    ban = pd.read_csv("data/events/fo_ban.csv", parse_dates=["date"])
    f = load_fo(2026, 2100)                                   # no research END cut-off: forward data must be visible
    f["date"], f["expiry"] = pd.to_datetime(f["date"]), pd.to_datetime(f["expiry"])
    s = pd.DatetimeIndex(np.sort(f["date"].unique()))
    pos = {d: i for i, d in enumerate(s)}
    exps = sorted({pd.Timestamp(e) for e in f.loc[f["kind"] == "STK", "expiry"].unique()})
    banned = set(zip(ban["symbol"], ban["date"]))
    return events_from(s, pos, exps, banned, asof, car_fn)


def car_fn(sym, a, b):
    from scripts import v12_events as E
    return E.car(sym, a, b)


def events_from(s, pos, exps, banned, asof, car) -> pd.DataFrame:
    rows = []
    for T in exps:
        if T < FORWARD_START or T > asof or T not in pos or pos[T] < 4:
            continue
        k = pos[T]
        w3 = [s[k - 3], s[k - 2], s[k - 1]]
        syms = {sy for sy, d in banned if d in w3}
        for sy in sorted(syms):
            c = car(sy, s[k - 4], T)
            rows.append({"expiry": T.date().isoformat(), "symbol": sy, "t_minus_4": s[k - 4].date().isoformat(), "car": c})
    return pd.DataFrame(rows, columns=["expiry", "symbol", "t_minus_4", "car"])


def update(asof: pd.Timestamp | None = None) -> dict:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    led = Ledger(LEDGER)
    have = {(r["body"]["expiry"], r["body"]["symbol"]) for r in led.of_kind("event")}
    asof = asof or pd.Timestamp.today().normalize()
    ev = candidate_events(asof)
    n = 0
    for r in ev.itertuples(index=False):
        if (r.expiry, r.symbol) in have or r.car is None or (isinstance(r.car, float) and np.isnan(r.car)):
            continue
        led.append("event", {"expiry": r.expiry, "symbol": r.symbol, "t_minus_4": r.t_minus_4, "car": float(r.car)})
        n += 1
    return {"appended": n, "total": len(led.of_kind("event")), "verify_errors": led.verify()}


def status() -> dict:
    led = Ledger(LEDGER) if LEDGER.exists() else None
    ev = pd.DataFrame([r["body"] for r in led.of_kind("event")]) if led else pd.DataFrame()
    if ev.empty:
        return {"events": 0, "evaluable": False}
    months = (pd.Timestamp(ev["expiry"].max()) - FORWARD_START).days / 30.44
    out = {"events": int(len(ev)), "months": round(months, 1), "evaluable": bool(months >= MIN_MONTHS and len(ev) >= MIN_EVENTS)}
    if out["evaluable"]:
        from scripts.v11_basis import nwt
        g = ev.groupby("expiry")["car"].mean()
        out["mean"], out["t"], out["p"], out["expiries"] = nwt(g.to_numpy())
        out["verdict"] = "SUPPORTED forward" if out["t"] > 2 and out["mean"] > 0 else "NOT SUPPORTED forward"
    return out


if __name__ == "__main__":  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["update", "status"])
    a = ap.parse_args()
    print(json.dumps(update() if a.cmd == "update" else status(), indent=1, default=str))
