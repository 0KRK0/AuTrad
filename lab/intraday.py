"""Intraday lab: same-day (MIS) strategies on hourly bars, judged like everything else.

Data: Yahoo 1-hour bars (~2 years, 2023-10 onward) for 22 liquid NSE names + NIFTY; 5-minute bars (~60 days)
used only for a resolution check. No bid/ask or order book, so nothing here can model queue position or
order flow; strategies are limited to bar-level rules (opening range, intraday momentum, VWAP trend).

Execution model (conservative):
  * signals use completed bars only; entry at the NEXT bar's open; stops fill at min(open, stop) for longs
    (max for shorts) when the bar's range touches the stop; everything is flat by the last bar's close.
  * costs per trade = Indian intraday statutory charges + brokerage min(0.03%, Rs20) + GST, a half spread on
    each side, and sqrt market impact against the day's median bar volume ("nse_intraday" profile).
  * sizing: each trade gets 1/max_pos of start-of-day capital (1x, no MIS leverage). Idle capital is assumed
    to sit in a pledged liquid fund earning the cash rate, so Sharpe is measured on returns above cash.
  * universe = today's liquid large caps (survivorship bias over 3 years is small but not zero; flagged).

Sealed split: selection on 2023-10 .. 2025-03, TEST 2025-04 onward (seen once).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from execution.costs import COST_MODEL_VERSION, PROFILES, CostProfile
from marketdata.intraday import load
from validation import stats
from validation.metrics import excess_returns, perf, periods_per_year

from .pbo import pbo

VAL_END = "2025-04-01"          # selection data ends here; test begins
TRADE_COLS = ["day", "symbol", "side", "entry_ts", "entry_px", "exit_ts", "exit_px", "i0", "i1"]


# ----------------------------------------------------------------- data
@dataclass
class DayBars:
    minutes: np.ndarray      # minutes since 09:15 at bar START
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray  # noqa: E741  (OHLC convention)
    c: np.ndarray
    v: np.ndarray
    ts: pd.DatetimeIndex


@dataclass
class IntradayData:
    interval: str
    bar_minutes: int
    symbols: list[str]
    days: list[pd.Timestamp]
    bars: dict[str, dict[pd.Timestamp, DayBars]]
    prev_close: dict[str, dict[pd.Timestamp, float]]
    index: str = "NIFTY50"
    notes: list[str] = field(default_factory=list)

    def subset_days(self, start: str | None = None, end: str | None = None) -> IntradayData:
        keep = [d for d in self.days if (start is None or d >= pd.Timestamp(start)) and (end is None or d < pd.Timestamp(end))]
        ks = set(keep)
        return IntradayData(self.interval, self.bar_minutes, self.symbols, keep,
                            {s: {d: b for d, b in m.items() if d in ks} for s, m in self.bars.items()},
                            self.prev_close, self.index, self.notes)


def _split_days(df: pd.DataFrame) -> tuple[dict, dict]:
    out, prev, last = {}, {}, np.nan
    day = df.index.normalize().tz_localize(None)
    for d, g in df.groupby(day):
        g = g[(g.index.hour * 60 + g.index.minute) < 15 * 60 + 30]
        if len(g) < 2:
            continue
        m = (g.index.hour * 60 + g.index.minute - (9 * 60 + 15)).to_numpy()
        out[d] = DayBars(m, *(g[k].to_numpy(float) for k in ["open", "high", "low", "close", "volume"]), g.index)
        prev[d] = last
        last = float(g["close"].iloc[-1])
    return out, prev


def load_intraday(data_dir: str, interval: str = "1h") -> IntradayData:
    raw = load(data_dir, interval)
    if not raw:
        raise FileNotFoundError(f"no intraday {interval} bars under {data_dir}/intraday — run scripts.fetch_intraday")
    bars, prev = {}, {}
    for s, df in raw.items():
        df = df.dropna(subset=["close"])
        df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
        bars[s], prev[s] = _split_days(df)
    days = sorted(set().union(*[set(b) for b in bars.values()]))
    syms = [s for s in raw if s != "NIFTY50"]
    bm = {"1h": 60, "5m": 5, "15m": 15}[interval]
    return IntradayData(interval, bm, syms, days, bars, prev,
                        notes=[f"{len(syms)} symbols, {len(days)} sessions, {days[0].date()} .. {days[-1].date()}"])


# ----------------------------------------------------------------- strategies -> trade lists
def _index_first(D: IntradayData, d: pd.Timestamp, or_min: int) -> float:
    b = D.bars.get(D.index, {}).get(d)
    pc = D.prev_close.get(D.index, {}).get(d, np.nan)
    if b is None:
        return np.nan
    k = np.nonzero(b.minutes < or_min)[0]
    if not len(k):
        return np.nan
    ref = pc if np.isfinite(pc) else b.o[0]
    return b.c[k[-1]] / ref - 1


def _exit_scan(b: DayBars, i0: int, side: int, stop: float | None, exit_rule: Callable[[int], bool] | None = None) -> tuple[int, float]:
    """Walk bars from the entry bar; return (exit bar index, exit price)."""
    n = len(b.c)
    for j in range(i0, n):
        if stop is not None:
            if side > 0 and b.l[j] <= stop:
                return j, min(b.o[j], stop)
            if side < 0 and b.h[j] >= stop:
                return j, max(b.o[j], stop)
        if exit_rule is not None and j < n - 1 and exit_rule(j):
            return j + 1, b.o[j + 1]           # decided at bar close, filled next open
    return n - 1, b.c[n - 1]


def orb(D: IntradayData, or_min: int = 60, confirm: bool = False, direction: str = "both", use_stop: bool = True) -> pd.DataFrame:
    """Opening-range breakout: range = first `or_min` minutes; a bar CLOSING beyond it triggers entry at the
    next open; stop at the other side of the range; flat at the close. Optional: trade only in the direction
    of NIFTY's opening-range move."""
    rows = []
    for d in D.days:
        idx_move = _index_first(D, d, or_min) if confirm else 0.0
        for s in D.symbols:
            b = D.bars[s].get(d)
            if b is None:
                continue
            rng = b.minutes < or_min
            if rng.sum() == 0 or rng.all():
                continue
            H, L = b.h[rng].max(), b.l[rng].min()
            first = int(np.nonzero(~rng)[0][0])
            for i in range(first, len(b.c) - 1):
                side = 1 if b.c[i] > H else (-1 if b.c[i] < L else 0)
                if side == 0 or (direction == "long" and side < 0):
                    continue
                if confirm and not (np.isfinite(idx_move) and np.sign(idx_move) == side):
                    break
                stop = (L if side > 0 else H) if use_stop else None
                j, px = _exit_scan(b, i + 1, side, stop)
                rows.append((d, s, side, b.ts[i + 1], b.o[i + 1], b.ts[j], px, i + 1, j))
                break
    return pd.DataFrame(rows, columns=TRADE_COLS)


def intraday_momentum(D: IntradayData, or_min: int = 60, mode: str = "late", threshold: float = 0.0) -> pd.DataFrame:
    """Gao-Han-Li-Zhou (2018): the first part of the day (from yesterday's close) predicts the last part.
    mode 'late': trade the final bar only;  mode 'day': hold from the end of the opening window to the close."""
    rows = []
    for d in D.days:
        for s in D.symbols:
            b = D.bars[s].get(d)
            pc = D.prev_close[s].get(d, np.nan)
            if b is None or not np.isfinite(pc) or len(b.c) < 3:
                continue
            k = np.nonzero(b.minutes < or_min)[0]
            if not len(k) or k[-1] >= len(b.c) - 1:
                continue
            sig = b.c[k[-1]] / pc - 1
            if abs(sig) <= threshold:
                continue
            side = int(np.sign(sig))
            i = len(b.c) - 1 if mode == "late" else k[-1] + 1
            if mode == "late":
                # decision at the close of the second-to-last bar using the day so far
                sig2 = b.c[-2] / pc - 1
                side = int(np.sign(sig2)) if abs(sig2) > threshold else 0
                if side == 0:
                    continue
            rows.append((d, s, side, b.ts[i], b.o[i], b.ts[-1], b.c[-1], i, len(b.c) - 1))
    return pd.DataFrame(rows, columns=TRADE_COLS)


def vwap_trend(D: IntradayData, or_min: int = 60, direction: str = "both", gap: float = 0.0) -> pd.DataFrame:
    """After the opening window: price above session VWAP and above the open -> long (mirror for short);
    exit when a bar closes back through VWAP, else at the close. One trade per name per day."""
    rows = []
    for d in D.days:
        for s in D.symbols:
            b = D.bars[s].get(d)
            if b is None or len(b.c) < 3:
                continue
            tp = (b.h + b.l + b.c) / 3
            vol = np.where(b.v > 0, b.v, np.nan)
            if np.isnan(vol).all():
                vw = np.cumsum(tp) / np.arange(1, len(tp) + 1)
            else:
                vol = np.nan_to_num(vol, nan=np.nanmedian(vol))
                vw = np.cumsum(tp * vol) / np.cumsum(vol)
            start = np.nonzero(b.minutes >= or_min)[0]
            if not len(start):
                continue
            for i in range(max(int(start[0]) - 1, 0), len(b.c) - 1):
                if b.minutes[i] + D.bar_minutes < or_min:
                    continue
                up = b.c[i] > vw[i] * (1 + gap) and b.c[i] > b.o[0]
                dn = b.c[i] < vw[i] * (1 - gap) and b.c[i] < b.o[0]
                side = 1 if up else (-1 if dn and direction == "both" else 0)
                if side == 0:
                    continue
                def rule(j: int, sd: int = side, c: np.ndarray = b.c, v: np.ndarray = vw) -> bool:
                    return bool(c[j] < v[j]) if sd > 0 else bool(c[j] > v[j])

                j, px = _exit_scan(b, i + 1, side, None, rule)
                rows.append((d, s, side, b.ts[i + 1], b.o[i + 1], b.ts[j], px, i + 1, j))
                break
    return pd.DataFrame(rows, columns=TRADE_COLS)


@dataclass
class IntradaySpec:
    name: str
    description: str
    fn: Callable[..., pd.DataFrame]
    grid: dict[str, list]
    family: str = "INTRADAY"
    horizon: str = "minutes-hours"

    def configs(self) -> list[dict]:
        out = [{}]
        for k, vs in self.grid.items():
            out = [{**c, k: v} for c in out for v in vs]
        return out


INTRADAY_LIBRARY = [
    IntradaySpec("orb", "Opening-range breakout (first hour), stop at the other side, flat by close", orb,
                 {"confirm": [False, True], "direction": ["both", "long"], "use_stop": [True, False]}),
    IntradaySpec("intraday_momentum", "Day-so-far return predicts the last hour / rest of day (Gao et al. 2018)",
                 intraday_momentum, {"mode": ["late", "day"], "threshold": [0.0, 0.005]}),
    IntradaySpec("vwap_trend", "Above VWAP and the open -> long until it closes back under VWAP", vwap_trend,
                 {"direction": ["both", "long"], "gap": [0.0, 0.002]}),
]


# ----------------------------------------------------------------- portfolio accounting
@dataclass
class IntradayResult:
    nav: pd.Series
    trades: pd.DataFrame
    costs: pd.Series
    exposure: pd.Series
    params: dict
    cost_model: str = COST_MODEL_VERSION

    @property
    def returns(self) -> pd.Series:
        return self.nav.pct_change().dropna()


def simulate(D: IntradayData, trades: pd.DataFrame, capital: float = 10_00_000, max_pos: int = 5,
             cost_mult: float = 1.0, cash_rate: float = 0.06, profile: CostProfile | None = None,
             params: dict | None = None) -> IntradayResult:
    """Daily accounting of a trade list. At most `max_pos` trades per day (earliest entries first)."""
    prof = profile or PROFILES["nse_intraday"]
    days = pd.DatetimeIndex(D.days)
    tr = trades.sort_values(["day", "entry_ts", "symbol"]).groupby("day", sort=False).head(max_pos).copy() if len(trades) else trades.copy()
    nav = np.empty(len(days))
    costs = np.zeros(len(days))
    expo = np.zeros(len(days))
    by_day = {d: g for d, g in tr.groupby("day")} if len(tr) else {}
    gross_col, cost_col = np.zeros(len(tr)), np.zeros(len(tr))
    pos_in_tr = {ix: k for k, ix in enumerate(tr.index)}
    v = capital
    prev_d = None
    for t, d in enumerate(days):
        gap = (d - prev_d).days if prev_d is not None else 0
        v *= (1 + cash_rate) ** (gap / 365.0)
        g = by_day.get(d)
        if g is not None:
            slot = v / max_pos
            pnl = 0.0
            for ix, r in g.iterrows():
                qty = slot / r.entry_px
                gross = r.side * qty * (r.exit_px - r.entry_px)
                b = D.bars[r.symbol][d]
                vols = b.v[b.v > 0]
                med = float(np.median(vols)) if len(vols) else 0.0
                imp = prof.impact_coef * 0.015 * np.sqrt(qty / (med * len(b.v))) if med > 0 else 0.0
                exit_notional = qty * r.exit_px
                c = (prof.fees(r.side, slot) + prof.fees(-r.side, exit_notional)
                     + prof.half_spread_bps / 1e4 * (slot + exit_notional) + imp * (slot + exit_notional)) * cost_mult
                pnl += gross - c
                costs[t] += c
                k = pos_in_tr[ix]
                gross_col[k], cost_col[k] = gross, c
            expo[t] = len(g) / max_pos
            v += pnl
        nav[t] = v
        prev_d = d
    if len(tr):
        tr["gross"], tr["cost"] = gross_col, cost_col
        tr["pnl"] = tr["gross"] - tr["cost"]
        tr["notional"] = [v0 for v0 in (nav[days.get_loc(d) - 1] if days.get_loc(d) > 0 else capital for d in tr["day"])]
        tr["notional"] = tr["notional"] / max_pos
        tr["gross_bps"] = tr["gross"] / tr["notional"] * 1e4
        tr["cost_bps"] = tr["cost"] / tr["notional"] * 1e4
    return IntradayResult(pd.Series(nav, days), tr, pd.Series(np.cumsum(costs), days), pd.Series(expo, days), params or {})


# ----------------------------------------------------------------- nulls
def random_direction(trades: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = trades.copy()
    out["side"] = rng.choice([-1, 1], size=len(out))
    return out


def random_entry(D: IntradayData, trades: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Same names, days and sides; entry bar drawn at random, held to the close."""
    rows = []
    for r in trades.itertuples(index=False):
        b = D.bars[r.symbol][r.day]
        i = int(rng.integers(1, len(b.c)))
        rows.append((r.day, r.symbol, r.side, b.ts[i], b.o[i], b.ts[-1], b.c[-1], i, len(b.c) - 1))
    return pd.DataFrame(rows, columns=TRADE_COLS)


# ----------------------------------------------------------------- judge
def _sh(nav: pd.Series, rf: float) -> float:
    ex = excess_returns(nav, rf)
    return float(ex.mean() / ex.std() * np.sqrt(periods_per_year(nav.index))) if len(ex) > 5 and ex.std() > 0 else 0.0


def _win(nav: pd.Series, start: str | None = None, end: str | None = None) -> pd.Series:
    if start:
        prev = nav[nav.index < pd.Timestamp(start)]
        seg = nav[nav.index >= pd.Timestamp(start)]
        return pd.concat([prev.iloc[-1:], seg]) if len(prev) else seg
    return nav[nav.index < pd.Timestamp(end)] if end else nav


def judge_intraday(spec: IntradaySpec, D: IntradayData, n_trials_before: int, capital: float = 10_00_000,
                   cash_rate: float = 0.06, n_null: int = 100, max_pos: int = 5, D5: IntradayData | None = None,
                   log=print) -> dict:
    from lab.judge import GATES
    rf = cash_rate
    cfgs = spec.configs()
    trade_lists = {repr(c): spec.fn(D, **c) for c in cfgs}
    runs = {k: simulate(D, t, capital, max_pos, cash_rate=rf, params=c) for (k, t), c in zip(trade_lists.items(), cfgs, strict=True)}
    val = {k: _sh(_win(r.nav, end=VAL_END), rf) for k, r in runs.items()}
    sel = max(val, key=val.get)
    R = runs[sel]
    tw, vw = _win(R.nav, VAL_END), _win(R.nav, end=VAL_END)
    m_test, m_val = perf(tw, rf, R.exposure, R.costs), perf(vw, rf, R.exposure, R.costs)
    s_test = _sh(tw, rf)
    ttest = R.trades[R.trades["day"] >= pd.Timestamp(VAL_END)] if len(R.trades) else R.trades

    shock = {}
    for mult in (0.0, 1.0, 1.5, 2.0, 3.0):
        r = R if mult == 1.0 else simulate(D, trade_lists[sel], capital, max_pos, cost_mult=mult, cash_rate=rf)
        shock[str(mult)] = {"test_sharpe": _sh(_win(r.nav, VAL_END), rf), "val_sharpe": _sh(_win(r.nav, end=VAL_END), rf)}
    capacity = {}
    for cap in (1e5, 5e5, 1e6, 5e6, 1e7, 5e7):
        r = R if cap == capital else simulate(D, trade_lists[sel], cap, max_pos, cash_rate=rf)
        capacity[f"{cap:,.0f}"] = {"test_sharpe": _sh(_win(r.nav, VAL_END), rf), "test_cagr": perf(_win(r.nav, VAL_END), rf).get("cagr")}

    log(f"  [{spec.name}] nulls x{2 * n_null}")
    rng = np.random.default_rng(7)
    test_tr = trade_lists[sel][trade_lists[sel]["day"] >= pd.Timestamp(VAL_END)]
    Dt = D.subset_days(start=VAL_END)
    base = _sh(simulate(Dt, test_tr, capital, max_pos, cash_rate=rf).nav, rf)
    null_dir = [_sh(simulate(Dt, random_direction(test_tr, rng), capital, max_pos, cash_rate=rf).nav, rf) for _ in range(n_null)]
    null_ent = [_sh(simulate(Dt, random_entry(Dt, test_tr, rng), capital, max_pos, cash_rate=rf).nav, rf) for _ in range(n_null)]
    p_dir, p_ent = stats.random_pvalue(base, null_dir), stats.random_pvalue(base, null_ent)

    ex = excess_returns(tw, rf)
    trial_sr = [float(e.mean() / e.std()) if e.std() > 0 else 0.0 for e in (excess_returns(_win(r.nav, VAL_END), rf) for r in runs.values())]
    n_trials = n_trials_before + len(cfgs)
    dsr = stats.deflated_sharpe(ex, trial_sr, n_trials)
    M = np.column_stack([r.nav.pct_change().dropna().to_numpy() for r in runs.values()])
    p_bo = pbo(M, 8)
    q = tw.resample("QE").last()
    q = pd.concat([tw.iloc[:1], q]).pct_change().dropna()
    pos_q = float((q > (1 + rf) ** 0.25 - 1).mean()) if len(q) else 0.0
    neigh = [k for k, r in runs.items() if k != sel and sum(r.params.get(p) != R.params.get(p) for p in R.params) == 1]
    stability = float(np.mean([val[k] > 0 for k in neigh])) if neigh else float("nan")

    res5 = None
    if D5 is not None and len(D5.days) > 20:
        d0 = D5.days[0]
        t5 = simulate(D5, spec.fn(D5, **R.params), capital, max_pos, cash_rate=rf)
        D1s = D.subset_days(start=str(d0.date()))
        t1 = simulate(D1s, spec.fn(D1s, **R.params), capital, max_pos, cash_rate=rf)
        res5 = {"days": len(D5.days), "sharpe_5m": _sh(t5.nav, rf), "sharpe_1h_same_days": _sh(t1.nav, rf),
                "avg_net_bps_5m": float(t5.trades["pnl"].sum() / t5.trades["notional"].sum() * 1e4) if len(t5.trades) else 0.0,
                "avg_net_bps_1h": float(t1.trades["pnl"].sum() / t1.trades["notional"].sum() * 1e4) if len(t1.trades) else 0.0,
                "note": "~60 sessions: a consistency check, not evidence"}

    tstat = {}
    if len(ttest):
        tstat = {"trades": int(len(ttest)), "win_rate": float((ttest.pnl > 0).mean()),
                 "avg_gross_bps": float(ttest.gross.sum() / ttest.notional.sum() * 1e4),
                 "avg_cost_bps": float(ttest.cost.sum() / ttest.notional.sum() * 1e4),
                 "avg_net_bps": float(ttest.pnl.sum() / ttest.notional.sum() * 1e4),
                 "profit_factor": float(ttest.pnl[ttest.pnl > 0].sum() / -ttest.pnl[ttest.pnl <= 0].sum()) if (ttest.pnl <= 0).any() else float("inf"),
                 "long_share": float((ttest.side > 0).mean())}
    G = GATES
    gates = [
        ("Sealed test Sharpe", s_test >= G["min_test_sharpe"], f"{s_test:.2f}", f">= {G['min_test_sharpe']}"),
        ("Selection-period Sharpe", val[sel] >= G["min_oos_sharpe"], f"{val[sel]:.2f}", f">= {G['min_oos_sharpe']}"),
        ("Beats cash in test", m_test.get("excess_cagr", -1) > 0, f"{m_test.get('excess_cagr', float('nan')):+.1%}", "> 0"),
        ("Positive net edge per trade (test)", tstat.get("avg_net_bps", -1) > 0, f"{tstat.get('avg_net_bps', float('nan')):+.1f} bps", "> 0"),
        ("Random direction null (test)", p_dir < G["max_null_p"], f"p = {p_dir:.3f}", f"< {G['max_null_p']}"),
        ("Random entry-time null (test)", p_ent < G["max_null_p"], f"p = {p_ent:.3f}", f"< {G['max_null_p']}"),
        ("Deflated Sharpe", dsr >= G["min_dsr"], f"{dsr:.2f} ({n_trials} trials)", f">= {G['min_dsr']}"),
        ("Overfitting (PBO)", np.isfinite(p_bo["pbo"]) and p_bo["pbo"] < G["max_pbo"], f"{p_bo['pbo']:.2f}", f"< {G['max_pbo']}"),
        ("Survives 2x costs", shock["2.0"]["test_sharpe"] > 0, f"{shock['2.0']['test_sharpe']:.2f}", "> 0"),
        ("Parameter stability", not np.isfinite(stability) or stability >= G["min_stability"], f"{stability:.0%}" if np.isfinite(stability) else "n/a", f">= {G['min_stability']:.0%}"),
        ("Max drawdown (test)", m_test.get("max_drawdown", 1) <= G["max_dd"], f"{m_test.get('max_drawdown', float('nan')):.1%}", f"<= {G['max_dd']:.0%}"),
        ("Profitable quarters (test)", pos_q >= G["min_pos_years"], f"{pos_q:.0%}", f">= {G['min_pos_years']:.0%}"),
    ]
    passed = all(g[1] for g in gates)
    verdict = "QUALIFIED_FOR_PAPER" if passed else ("REJECT" if s_test <= 0 and val[sel] <= 0 else "RESEARCH")
    return {
        "strategy": spec.name, "family": spec.family, "horizon": spec.horizon, "description": spec.description,
        "selected": R.params, "val_sharpe": val, "verdict": verdict, "gates_passed": sum(g[1] for g in gates), "gates_total": len(gates),
        "gates": [{"name": n, "passed": bool(p), "value": v, "target": t} for n, p, v, t in gates],
        "test": m_test, "oos": m_val, "trades": tstat, "cost_shock": shock, "capacity": capacity,
        "nulls": {"p_direction": p_dir, "p_entry": p_ent, "median_direction": float(np.median(null_dir)),
                  "median_entry": float(np.median(null_ent)), "strategy_test_sharpe": base},
        "stats": {"psr": stats.psr(ex), "dsr": dsr, "n_trials": n_trials, "pbo": p_bo,
                  "sharpe_ci": stats.bootstrap_sharpe_ci(ex, periods_per_year(tw.index))},
        "perturbation": {"stability": stability, "neighbours": {k: val[k] for k in neigh}},
        "resolution_check": res5, "cost_model": R.cost_model, "split": {"selection_end": VAL_END},
        "_nav": R.nav, "_returns": R.returns, "_configs": [r.params for r in runs.values()],
    }
