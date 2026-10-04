"""v1.3 Part C: NSE depth execution study (sealed). Refuses to analyse until >= 20 sealed recorded days exist for the
instrument; then splits days chronologically 60 % train / 40 % test and runs the v1.0 M1 design (P0 cross now, P1 post
then cross at 60 s, P2 P1 + depth-imbalance trigger with theta = measured half-spread + 0.2 bp) on the 1 s NSE grid.

    python -m scripts.v13_depth status
    python -m scripts.v13_depth run --symbol NIFTY-FUT --out reports/v1.3_depth_<ts>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from marketdata.nse_depth_recorder import ROOT, nse_index_futures_round_trip_bp, to_grid

MIN_DAYS = 20


def sealed_days(symbol: str, root: Path = ROOT) -> list[str]:
    return sorted(p.parent.name for p in (Path(root) / symbol).glob("*/manifest.json")
                  if json.loads(p.read_text()).get("qc", {}).get("ok"))


def status(root: Path = ROOT) -> dict:
    syms = sorted(p.name for p in Path(root).glob("*") if p.is_dir()) if Path(root).exists() else []
    return {s: {"sealed_ok_days": len(sealed_days(s, root)), "required": MIN_DAYS, "unsealed": True} for s in syms} or \
        {"recorded": 0, "note": "no NSE depth recorded yet: needs the owner's Kite Connect key and `python -m scripts.record_nse_depth`"}


def split(days: list[str]) -> tuple[list[str], list[str]]:
    k = int(round(len(days) * 0.6))
    return days[:k], days[k:]


def run(symbol: str, out: Path, root: Path = ROOT) -> dict:  # pragma: no cover - needs recorded data
    days = sealed_days(symbol, root)
    if len(days) < MIN_DAYS:
        raise SystemExit(f"sealed: only {len(days)} of {MIN_DAYS} required days for {symbol}; no analysis is run")
    from scripts import v10_micro as V
    train, test = split(days)
    grids = {d: to_grid(pd.read_parquet(Path(root) / symbol / d / "day.parquet"), d) for d in days}
    hs = float(np.nanmedian(np.concatenate([((g["ask"] - g["bid"]) / ((g["ask"] + g["bid"]) / 2) * 1e4 / 2).to_numpy() for d, g in grids.items() if d in train])))
    theta = hs + 0.2
    rt = nse_index_futures_round_trip_bp()
    V.EVERY, V.DEADLINE = 300, 60                           # 5 minutes and 60 s on the 1 s grid
    res = {"train_days": train, "test_days": test, "half_spread_bp": hs, "theta_bp": theta, "fee_round_trip_bp": rt}
    # the M0-style predictor on NSE: order-book imbalance only (fitted on train days), pred in bp of the next 5 s
    X, y = [], []
    for d in train:
        g = grids[d]
        obi = ((g["bidq"] - g["askq"]) / (g["bidq"] + g["askq"])).to_numpy()
        mid = ((g["bid"] + g["ask"]) / 2).to_numpy()
        r5 = (np.r_[np.log(mid[5:]), np.full(5, np.nan)] - np.log(mid)) * 1e4
        m = ~(np.isnan(obi) | np.isnan(r5))
        X.append(obi[m]); y.append(r5[m])
    beta = float(np.polyfit(np.concatenate(X), np.concatenate(y), 1)[0])
    rows = []
    for d in test:
        g = grids[d]
        pred = (beta * ((g["bidq"] - g["askq"]) / (g["bidq"] + g["askq"]))).to_numpy(np.float32)
        o = V.m1_orders(g, pred, theta, lat=1)
        x = V.is_bp(o, (rt / 2, rt / 2))
        x["day"] = d
        rows.append(x)
    x = pd.concat(rows)
    res.update({"orders": int(len(x)), "mean_IS_bp": {k: float(x[k].mean()) for k in ("IS0", "IS1", "IS2")}})
    out.mkdir(parents=True, exist_ok=True)
    (out / "depth_study.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["status", "run"])
    ap.add_argument("--symbol", default="NIFTY-FUT")
    ap.add_argument("--out", default="reports/v1.3_depth")
    a = ap.parse_args()
    print(json.dumps(status() if a.cmd == "status" else run(a.symbol, Path(a.out)), indent=1))
