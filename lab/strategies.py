"""AuTrad strategy library. Every strategy: fn(panel, **params) -> target weights at decision dates.

Families and horizons (the judge treats them identically):
  CORE        months      xs_momentum, low_vol, index_trend
  SWING       days-weeks  breakout, rsi2_pullback, st_reversal, rel_strength
  CROSS_ASSET weeks-months tsmom_multi
  REL_VALUE   days-weeks  pairs_sector (stock futures, long/short)
  INTRADAY    hours       see lab/intraday.py
  BLOCKED     value, quality, event_driven (data not available point-in-time; nothing is faked)
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .panel import Panel
from .portfolio import cap_weights, every, features, month_starts, rank_select, targets_frame, tradable


@dataclass
class StrategySpec:
    name: str
    family: str
    horizon: str
    description: str
    fn: Callable[..., pd.DataFrame] | None
    grid: dict[str, list] = field(default_factory=dict)
    universe: str = "india"            # india | cross_asset | pairs | intraday
    status: str = "ACTIVE"             # ACTIVE | BLOCKED
    blocked_reason: str = ""
    post_test_peek: bool = False       # added AFTER the sealed test was looked at: its TEST numbers are not clean

    def configs(self) -> list[dict]:
        keys = list(self.grid)
        out = [{}]
        for k in keys:
            out = [{**c, k: v} for c in out for v in self.grid[k]]
        return out


def _market_on(P: Panel, t: int) -> bool:
    ma = features(P).bench_ma200.iloc[t]
    return bool(np.isfinite(ma) and P.bench.iloc[t] > ma)


# ------------------------------------------------------------------ CORE
def xs_momentum(P: Panel, top_n: int = 10, signal: str = "12-1", market_filter: bool = False) -> pd.DataFrame:
    """Cross-sectional momentum with absolute-momentum filter; monthly; inverse-vol weights."""
    F = features(P)
    r12 = F.ret_over(252, 21)
    if signal == "ensemble":
        parts = [F.ret_over(L, 21) for L in (63, 126, 189, 252)]
        z = [(p.sub(p.mean(axis=1), axis=0)).div(p.std(axis=1), axis=0) for p in parts]
        score = sum(z) / len(z)
    else:
        score = r12
    rows = {}
    for t in month_starts(P):
        if t < 252:
            continue
        if market_filter and not _market_on(P, t):
            rows[t] = pd.Series(dtype=float)
            continue
        ok = tradable(P, t, {"largecap"})
        s = score.iloc[t].where(ok & (r12.iloc[t] > 0).to_numpy())        # absolute momentum: own trend must be up
        rows[t] = rank_select(P, t, s, top_n)
    return targets_frame(rows, P)


def low_vol(P: Panel, top_n: int = 12, market_filter: bool = False) -> pd.DataFrame:
    F = features(P)
    v = F.vol(252)
    rows = {}
    for t in month_starts(P):
        if t < 252:
            continue
        if market_filter and not _market_on(P, t):
            rows[t] = pd.Series(dtype=float)
            continue
        ok = tradable(P, t, {"largecap"})
        rows[t] = rank_select(P, t, (-v.iloc[t]).where(ok), top_n, weighting="equal")
    return targets_frame(rows, P)


def index_trend(P: Panel, ma: int = 200) -> pd.DataFrame:
    """Hold the NIFTY ETF while it is above its moving average, otherwise cash. Trades only on signal changes."""
    F = features(P)
    i = P.idx["NIFTYBEES"]
    c, m = F.Cdf.iloc[:, i], F.ma(ma).iloc[:, i]
    on = (c > m).to_numpy()
    rows, prev = {}, None
    for t in range(ma, len(P.dates)):
        if on[t] != prev:
            rows[t] = pd.Series({"NIFTYBEES": 1.0}) if on[t] else pd.Series(dtype=float)
            prev = on[t]
    return targets_frame(rows, P)


def strategic_mix(P: Panel, gold: float = 0.3) -> pd.DataFrame:
    """Fixed NIFTY ETF / gold ETF mix, rebalanced monthly. No forecasting at all: the edge, if any, is
    diversification (Indian equities and INR gold have had low correlation) plus the rebalancing premium."""
    rows = {}
    for t in month_starts(P):
        ok = tradable(P, t)
        w = pd.Series(0.0, index=P.symbols)
        for sym, wt in (("NIFTYBEES", 1 - gold), ("GOLDBEES", gold)):
            if sym in P.idx and ok[P.idx[sym]]:
                w[sym] = wt
        rows[t] = w
    return targets_frame(rows, P)


# ------------------------------------------------------------------ SWING
def _event_book(P: Panel, entry: pd.DataFrame, exit_: pd.DataFrame, max_pos: int, max_hold: int | None,
                klass: set[str], rank: pd.DataFrame | None = None) -> pd.DataFrame:
    """Daily position book: enter on entry signals (best-ranked first), exit on exit signals or time."""
    held: dict[str, int] = {}
    rows = {}
    E, X = entry.to_numpy(), exit_.to_numpy()
    R = rank.to_numpy() if rank is not None else None
    for t in range(len(P.dates)):
        changed = False
        for s in list(held):
            i = P.idx[s]
            if X[t, i] or (max_hold and t - held[s] >= max_hold) or np.isnan(P.C[t, i]):
                del held[s]
                changed = True
        ok = tradable(P, t, klass)
        cand = [j for j in np.nonzero(E[t] & ok)[0] if P.symbols[j] not in held]
        if R is not None:
            cand.sort(key=lambda j: -np.nan_to_num(R[t, j], nan=-1e9))
        for j in cand[: max(0, max_pos - len(held))]:
            held[P.symbols[j]] = t
            changed = True
        if changed:
            w = pd.Series(1.0 / max_pos, index=list(held)) if held else pd.Series(dtype=float)
            rows[t] = cap_weights(P, w)
    return targets_frame(rows, P)


def breakout(P: Panel, entry_n: int = 55, exit_n: int = 20, max_pos: int = 10) -> pd.DataFrame:
    """Donchian breakout: close above the prior N-day high; exit below the prior M-day low."""
    F = features(P)
    entry = F.Cdf > F.high_n(entry_n).shift(1)
    exit_ = F.Cdf < F.low_n(exit_n).shift(1)
    return _event_book(P, entry.fillna(False), exit_.fillna(False), max_pos, None, {"largecap"}, F.ret_over(126))


def rsi2_pullback(P: Panel, threshold: int = 10, max_hold: int = 10, max_pos: int = 10) -> pd.DataFrame:
    """Buy short pullbacks inside long uptrends (close > 200d avg, RSI(2) below threshold); exit on close > 5d avg."""
    F = features(P)
    entry = (F.Cdf > F.ma200) & (F.rsi(2) < threshold)
    exit_ = F.Cdf > F.ma(5)
    return _event_book(P, entry.fillna(False), exit_.fillna(False), max_pos, max_hold, {"largecap"}, -F.rsi(2))


def st_reversal(P: Panel, k: int = 8, hold: int = 5) -> pd.DataFrame:
    """Weekly: buy the k biggest 5-day losers among index members, hold one week."""
    F = features(P)
    r5 = F.ret_over(5)
    rows = {}
    for t in every(P, hold, 260):
        ok = tradable(P, t, {"largecap"})
        rows[t] = rank_select(P, t, (-r5.iloc[t]).where(ok), k, weighting="equal")
    return targets_frame(rows, P)


def rel_strength(P: Panel, k: int = 8, lookback: int = 20) -> pd.DataFrame:
    """Weekly: strongest stocks vs the NIFTY over `lookback` days, only if above their 50-day average."""
    F = features(P)
    r = F.ret_over(lookback)
    b = P.bench / P.bench.shift(lookback) - 1
    ex = r.sub(b.to_numpy(), axis=0)
    up = F.Cdf > F.ma(50)
    rows = {}
    for t in every(P, 5, 260):
        ok = tradable(P, t, {"largecap"}) & up.iloc[t].to_numpy()
        rows[t] = rank_select(P, t, ex.iloc[t].where(ok), k)
    return targets_frame(rows, P)


# ------------------------------------------------------------------ CROSS-ASSET
def tsmom_multi(P: Panel, lookback: int = 252, vol_target: float = 0.10) -> pd.DataFrame:
    """Time-series momentum across NIFTY, bank index, gold, silver, USDINR (long/short) and BTC (long only),
    each position sized to an equal share of a portfolio volatility target; monthly."""
    F = features(P)
    r = F.ret_over(lookback, 21)
    v = F.vol(63) * np.sqrt(252)
    rows = {}
    n = len(P.symbols)
    for t in month_starts(P):
        if t < lookback:
            continue
        w = {}
        for j, s in enumerate(P.symbols):
            if np.isnan(P.C[t, j]) or not np.isfinite(r.iat[t, j]) or not np.isfinite(v.iat[t, j]) or v.iat[t, j] <= 0:
                continue
            sign = np.sign(r.iat[t, j])
            if sign < 0 and not P.profiles[j].shortable:
                continue
            w[s] = sign * vol_target / np.sqrt(n) / v.iat[t, j]
        rows[t] = cap_weights(P, pd.Series(w, dtype=float))
    return targets_frame(rows, P)


# ------------------------------------------------------------------ RELATIVE VALUE
def pairs_sector(P: Panel, entry_z: float = 2.0, lookback: int = 60, leg: float = 0.05, max_pairs: int = 6,
                 max_hold: int = 20) -> pd.DataFrame:
    """Within-sector pairs on stock futures: each month pick the most correlated pair per sector;
    trade the z-score of the log-price spread (hedge ratio by rolling OLS); exit at |z| < 0.5 or on time."""
    F = features(P)
    lp = np.log(F.Cdf)
    rets = F.Cdf.pct_change(fill_method=None)
    pairs: list[tuple[str, str]] = []
    open_: dict[tuple[str, str], tuple[int, float, int]] = {}   # pair -> (side, beta, t0)
    rows = {}
    months = set(month_starts(P))
    for t in range(260, len(P.dates)):
        changed = False
        if t in months:
            pairs = []
            ok = tradable(P, t, {"futures"})
            by_sec: dict[str, list[str]] = {}
            for j in np.nonzero(ok)[0]:
                by_sec.setdefault(P.sector[j], []).append(P.symbols[j])
            window = rets.iloc[t - 250:t + 1]
            for _sec, names in by_sec.items():
                if len(names) < 2:
                    continue
                cm = window[names].corr().to_numpy(copy=True)
                np.fill_diagonal(cm, np.nan)
                if np.all(np.isnan(cm)):
                    continue
                i, j = np.unravel_index(np.nanargmax(cm), cm.shape)
                best = (names[i], names[j])
                if best:
                    pairs.append(tuple(sorted(best)))
            pairs = pairs[:max_pairs]
            for p in list(open_):
                if p not in pairs:
                    del open_[p]
                    changed = True
        for a, b in pairs:
            ya, yb = lp[a].iloc[t - lookback:t + 1], lp[b].iloc[t - lookback:t + 1]
            if ya.isna().any() or yb.isna().any():
                continue
            beta = float(np.polyfit(yb, ya, 1)[0])
            spread = ya - beta * yb
            z = (spread.iloc[-1] - spread.mean()) / (spread.std() + 1e-12)
            key = (a, b)
            if key in open_:
                side, _, t0 = open_[key]
                if abs(z) < 0.5 or t - t0 >= max_hold:
                    del open_[key]
                    changed = True
            elif abs(z) > entry_z and np.isfinite(beta) and 0.2 < beta < 5:
                open_[key] = (-int(np.sign(z)), beta, t)          # spread high -> short A, long B
                changed = True
        if changed:
            w: dict[str, float] = {}
            for (a, b), (side, beta, _) in open_.items():
                w[a] = w.get(a, 0.0) + side * leg
                w[b] = w.get(b, 0.0) - side * leg * min(beta, 2.0)
            rows[t] = cap_weights(P, pd.Series(w, dtype=float))
    return targets_frame(rows, P)


LIBRARY: list[StrategySpec] = [
    StrategySpec("xs_momentum", "CORE", "1-12 months", "Top-N NIFTY 50 stocks by momentum (12-1 or 3/6/9/12-month ensemble), "
                 "own trend must be positive, optional market filter; monthly, inverse-vol weights", xs_momentum,
                 {"top_n": [8, 12], "signal": ["12-1", "ensemble"], "market_filter": [False, True]}),
    StrategySpec("low_vol", "CORE", "1-12 months", "Lowest 1-year volatility NIFTY 50 stocks, equal weight, monthly", low_vol,
                 {"top_n": [10, 15], "market_filter": [False, True]}),
    StrategySpec("index_trend", "CORE", "weeks-months", "NIFTY ETF when above its moving average, else cash", index_trend,
                 {"ma": [100, 150, 200]}),
    StrategySpec("strategic_mix", "CORE", "years", "NIFTY ETF + gold ETF at fixed weights, monthly rebalance (no signal)",
                 strategic_mix, {"gold": [0.25, 0.4, 0.5]}, post_test_peek=True),
    StrategySpec("breakout", "SWING", "days-weeks", "Donchian breakout on NIFTY 50 stocks", breakout,
                 {"entry_n": [20, 55], "exit_n": [10, 20]}),
    StrategySpec("rsi2_pullback", "SWING", "days", "Buy RSI(2) pullbacks in long-term uptrends", rsi2_pullback,
                 {"threshold": [5, 10], "max_hold": [5, 10]}),
    StrategySpec("st_reversal", "SWING", "1 week", "Buy last week's biggest losers, hold a week", st_reversal,
                 {"k": [5, 10]}),
    StrategySpec("rel_strength", "SWING", "1-4 weeks", "Strongest stocks vs NIFTY, weekly", rel_strength,
                 {"k": [5, 10], "lookback": [20, 60]}),
    StrategySpec("tsmom_multi", "CROSS_ASSET", "1-12 months", "Trend following across NIFTY, banks, gold, silver, USDINR, BTC",
                 tsmom_multi, {"lookback": [126, 252], "vol_target": [0.10, 0.15]}, universe="cross_asset"),
    StrategySpec("pairs_sector", "REL_VALUE", "days-weeks", "Within-sector pairs on stock futures (long/short)", pairs_sector,
                 {"entry_z": [2.0, 2.5], "lookback": [60, 120]}, universe="pairs"),
    StrategySpec("value", "CORE", "months", "P/E, P/B, EV/EBITDA, FCF yield", None, status="BLOCKED",
                 blocked_reason="needs point-in-time fundamentals (as reported on the filing date); none available yet"),
    StrategySpec("quality", "CORE", "months", "ROE, ROIC, leverage, earnings stability", None, status="BLOCKED",
                 blocked_reason="needs point-in-time fundamentals; none available yet"),
    StrategySpec("event_driven", "EVENT", "minutes-days", "Earnings, RBI/Fed, filings, regulatory events", None, status="BLOCKED",
                 blocked_reason="needs timestamped historical news/filings; the evidence collector must run forward to build history"),
]
