"""Walk-forward validation.

Every ARGO candidate is an online learner: its decision at t uses only data <= t. So one full
run per candidate yields honest out-of-sample returns for every period. What walk-forward adds
is honest *selection*: for each test year Y, the candidate is chosen using only its performance
before Y (expanding window), then judged on Y. The stitched test years are AuTrad's OOS record.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from multiprocessing import get_context

import numpy as np
import pandas as pd

from agents.taxonomy import Instrument
from argo import ArgoConfig
from backtest.sim import Market, SimConfig, SimResult, simulate
from backtest.strategies import ArgoAgents, BuyHold, MACross, Momentum, RandomTrader
from execution.costs import PROFILES
from marketdata.store import DataStore
from marketdata.universe import UNIVERSE, Asset

from .metrics import perf


@dataclass(frozen=True)
class Candidate:
    name: str
    horizon: int
    decide_every: int
    band: float
    edge_multiple: float = 1.5


GRID: list[Candidate] = [
    Candidate("daily", 1, 1, 0.0),
    Candidate("h5_weekly", 5, 5, 0.3),
    Candidate("h5_weekly_strict", 5, 5, 0.3, 3.0),
    Candidate("h20_weekly", 20, 5, 0.5),
    Candidate("h20_monthly", 20, 20, 0.5),
    Candidate("h20_monthly_strict", 20, 20, 0.5, 3.0),
]
# Configurations evaluated on the full sample during v0.2 research before this grid was frozen.
# They count as trials for the deflated Sharpe ratio.
EXTRA_TRIALS = 2

GROUPS: dict[str, set[str]] = {
    "all": {"equity", "etf", "crypto", "fx"},
    "nse_equity": {"equity", "etf_index"},
    "crypto": {"crypto"},
    "fx": {"fx"},
    "commodity": {"etf_commodity"},
}


def group_assets(group: str) -> list[Asset]:
    want = GROUPS[group]
    out = []
    for a in UNIVERSE:
        tags = {a.asset_class}
        if a.asset_class == "etf":
            tags.add("etf_commodity" if a.category == "commodity" else "etf_index")
        if tags & want:
            out.append(a)
    return out


def build_market(group: str, data_dir: str) -> tuple[Market, dict[str, Instrument], dict[str, float]]:
    assets = group_assets(group)
    data = DataStore(data_dir).load([a.symbol for a in assets])
    assets = [a for a in assets if a.symbol in data]
    mk = Market(data, {a.symbol: a.sector for a in assets}, {a.symbol: a.category for a in assets},
                {a.symbol: PROFILES[a.cost_profile] for a in assets})
    inst = {a.symbol: Instrument(a.symbol, a.asset_class, category=a.category, sector=a.sector) for a in assets}
    cost = {a.symbol: PROFILES[a.cost_profile].round_trip_bps() / 2 for a in assets}
    return mk, inst, cost


# ---------- parallel execution (portable: works with fork on Linux and spawn on Windows) ----------
_W: dict = {}


def _init(group: str, data_dir: str, sim_cfg: SimConfig) -> None:
    _W["mk"], _W["inst"], _W["cost"] = build_market(group, data_dir)
    _W["cfg"] = sim_cfg


def _run(job: tuple) -> SimResult:
    kind, arg = job
    mk, inst, cost, cfg = _W["mk"], _W["inst"], _W["cost"], _W["cfg"]
    if kind == "argo":
        c: Candidate = arg
        strat = ArgoAgents(inst, ArgoConfig(horizon=c.horizon, edge_multiple=c.edge_multiple), cost,
                           decide_every=c.decide_every, band=c.band, name=c.name)
    elif kind == "baseline":
        strat = {"buy_hold": BuyHold, "momentum_6m": Momentum, "ma_50_200": MACross}[arg]()
    else:
        seed, size, expo, hold = arg
        strat = RandomTrader(seed, size, expo, hold, name=f"random_{seed}")
    res = simulate(mk, strat, cfg)
    res.log = []  # decision logs stay in the worker; keep results light
    if kind == "random":  # only the equity curve is needed for the null distribution
        res.weights, res.ledger = pd.DataFrame(), pd.DataFrame()
        res.costs = res.equity.iloc[:0]
    return res


def run_jobs(group: str, data_dir: str, sim_cfg: SimConfig, jobs: list[tuple], workers: int | None = None) -> list[SimResult]:
    workers = workers or max(1, (os.cpu_count() or 2))
    if workers == 1:
        _init(group, data_dir, sim_cfg)
        return [_run(j) for j in jobs]
    ctx = get_context("fork") if hasattr(os, "fork") else get_context("spawn")
    with ctx.Pool(min(workers, len(jobs)), initializer=_init, initargs=(group, data_dir, sim_cfg)) as pool:
        return pool.map(_run, jobs)


# ---------- walk-forward selection and stitching ----------
@dataclass
class WalkForward:
    folds: pd.DataFrame           # per test year: chosen candidate, train sharpe, test return/sharpe
    equity: pd.Series             # stitched OOS equity
    costs: pd.Series              # stitched cumulative costs
    exposure: pd.Series
    ledger: pd.DataFrame          # trades of the chosen candidate that closed inside its test year
    turnover: float


def walk_forward(results: dict[str, SimResult], first_test_year: int, cash_rate: float, start_value: float) -> WalkForward:
    any_res = next(iter(results.values()))
    idx = any_res.equity.index
    years = sorted({y for y in idx.year if y >= first_test_year})
    rows, rets, cost_inc, expo, trades = [], [], [], [], []
    turnover = 0.0
    for y in years:
        t0, t1 = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y + 1}-01-01")
        scores = {}
        for name, r in results.items():
            m = perf(r.equity[r.equity.index < t0], cash_rate)
            scores[name] = m.get("sharpe", -np.inf) if m else -np.inf
        best = max(scores, key=scores.get)
        r = results[best]
        mask = (r.equity.index >= t0) & (r.equity.index < t1)
        prev = r.equity[r.equity.index < t0]
        seg = pd.concat([prev.iloc[-1:], r.equity[mask]])
        rets.append(seg.pct_change().dropna())
        cseg = pd.concat([r.costs[r.costs.index < t0].iloc[-1:], r.costs[mask]])
        cost_inc.append(cseg.diff().dropna() / seg.iloc[:-1].to_numpy())   # costs as fraction of equity
        expo.append(r.exposure[mask])
        led = r.ledger
        if not led.empty:
            trades.append(led[(led["exit"] >= t0) & (led["exit"] < t1)].assign(candidate=best))
        test = perf(seg, cash_rate)
        span_years = (r.equity.index[-1] - r.equity.index[0]).days / 365.25
        turnover += r.turnover / max(span_years, 1e-9) * test.get("years", 0.0)
        rows.append({"year": y, "chosen": best, "train_sharpe": scores[best],
                     "test_return": test.get("total_return", np.nan), "test_sharpe": test.get("sharpe", np.nan),
                     "test_max_dd": test.get("max_drawdown", np.nan)})
    r_all = pd.concat(rets)
    eq = start_value * (1 + r_all).cumprod()
    first = pd.Series([start_value], index=[r_all.index[0] - pd.Timedelta(days=1)])
    eq = pd.concat([first, eq])
    c_frac = pd.concat(cost_inc)
    costs = (c_frac * eq.shift(1).reindex(c_frac.index)).cumsum()
    costs = pd.concat([pd.Series([0.0], index=first.index), costs])
    return WalkForward(pd.DataFrame(rows).set_index("year"), eq, costs, pd.concat(expo),
                       pd.concat(trades) if trades else pd.DataFrame(), turnover)


def calibrate_random(results: dict[str, SimResult], wf: WalkForward) -> tuple[float, float, float]:
    """Position size, exposure and holding period of the candidate chosen most often."""
    modal = wf.folds["chosen"].mode().iloc[0]
    r = results[modal]
    w = r.weights.abs()
    size = float(w[w > 1e-6].stack().median()) if (w > 1e-6).any().any() else 0.02
    expo = float(r.exposure.mean())
    led = r.ledger
    days = float(led["days"].mean()) if not led.empty else 20.0
    idx = r.equity.index
    bars_per_day = (len(idx) - 1) / max((idx[-1] - idx[0]).days, 1)
    return size, expo, max(days * bars_per_day, 1.0)


def candidate_dicts() -> list[dict]:
    return [asdict(c) for c in GRID]
