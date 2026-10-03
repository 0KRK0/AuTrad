"""Fast research backtester (vectorised per day, exact per-symbol costs).

Timeline for decision date t (after the close of t):
  t+1 open : overnight gap applied to held weights, then trade to the target
             (fees by side, brokerage caps, DP charge, half-spread, sqrt impact vs traded value,
              participation cap vs the day's traded value, entries blocked if not eligible at t)
  t+1 close: intraday move on the new weights; cash earns the cash rate; futures pay roll carry.
A target row fully specifies the portfolio (symbols missing from the row go to zero); rows are
only needed on rebalance dates. Symbols with no price at t+1 keep their order pending.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from execution.costs import COST_MODEL_VERSION, GST

from .panel import Panel


@dataclass
class RunResult:
    name: str
    nav: pd.Series
    costs: pd.Series            # cumulative costs, Rs
    exposure: pd.Series         # gross |w|
    net_exposure: pd.Series
    weights: np.ndarray         # T x N end-of-day weights
    trades: pd.DataFrame        # position episodes
    turnover: float             # sum |traded| / average NAV
    capital: float
    cost_mult: float
    cost_model: str = COST_MODEL_VERSION
    params: dict = field(default_factory=dict)

    @property
    def returns(self) -> pd.Series:
        return self.nav.pct_change().fillna(0.0)


def _profile_arrays(panel: Panel):
    P = panel.profiles
    f = lambda attr: np.array([getattr(p, attr) for p in P], dtype=float)  # noqa: E731
    return {k: f(k) for k in ["buy_bps", "sell_bps", "half_spread_bps", "impact_coef", "max_participation",
                              "brokerage_pct", "brokerage_cap", "flat_per_sell", "carry_bps_per_day"]} | {
        "cash_settled": np.array([p.cash_settled for p in P]), "shortable": np.array([p.shortable for p in P])}


def run(panel: Panel, targets: pd.DataFrame, capital: float = 10_00_000.0, cost_mult: float = 1.0,
        cash_rate: float = 0.06, band: float = 0.0, name: str = "", params: dict | None = None,
        fills: list | None = None) -> RunResult:
    """fills: optional list that receives one dict per executed order (date index, symbol, weight change, traded
    value, open price, modelled cost, NAV before the trade). Passing it does not change any result."""
    T, N = panel.C.shape
    pa = _profile_arrays(panel)
    tgt = targets.reindex(columns=panel.symbols).fillna(0.0)
    row_of = {panel.dates.get_loc(d): r for d, r in zip(tgt.index, tgt.to_numpy(), strict=True) if d in panel.dates}
    Cf = panel.ffill_close()
    w = np.zeros(N)
    nav = capital
    pending: np.ndarray | None = None
    navs, cum_cost, exp_g, exp_n = np.empty(T), np.empty(T), np.empty(T), np.empty(T)
    W = np.zeros((T, N), dtype=np.float32)
    total_cost, traded = 0.0, 0.0
    # episode accounting
    ep_pnl, ep_cost, ep_start, ep_notional = np.zeros(N), np.zeros(N), np.full(N, -1), np.zeros(N)
    ep_side = np.zeros(N)
    episodes: list[tuple] = []
    days = np.r_[0, np.diff(panel.dates.values).astype("timedelta64[D]").astype(float)]
    rf = (1 + cash_rate) ** (days / 365.0) - 1

    for t in range(T):
        if t > 0:
            o, c = panel.O[t], panel.C[t]
            has_o = ~np.isnan(o)
            # 1) overnight gap on held weights
            with np.errstate(invalid="ignore", divide="ignore"):
                rg = np.where(has_o & (Cf[t - 1] > 0), o / Cf[t - 1] - 1, 0.0)
            pnl = w * rg * nav
            nav_new = nav + pnl.sum()
            ep_pnl += pnl
            w = w * (1 + rg) * nav / nav_new if nav_new > 0 else w * 0
            nav = nav_new
            # 2) trade at the open
            if pending is not None:
                can = has_o & ~np.isnan(pending)
                desired = np.where(can, pending, w)
                grow = np.abs(desired) > np.abs(w) + 1e-12
                blocked = grow & ~panel.eligible[t - 1]
                desired = np.where(blocked, np.sign(w) * np.minimum(np.abs(desired), np.abs(w)), desired)
                desired = np.where(~pa["shortable"] & (desired < 0), 0.0, desired)
                delta = desired - w
                if band > 0:
                    small = (np.abs(delta) < band) & (desired != 0)
                    delta = np.where(small, 0.0, delta)
                val_cap = pa["max_participation"] * panel.value[t]
                limited = (val_cap > 0) & (np.abs(delta) * nav > val_cap)
                delta = np.where(limited, np.sign(delta) * val_cap / max(nav, 1e-9), delta)
                tv = np.abs(delta) * nav
                buy = delta > 0
                fee = tv * np.where(buy, pa["buy_bps"], pa["sell_bps"]) / 1e4
                brok = tv * pa["brokerage_pct"] / 100
                brok = np.where(pa["brokerage_cap"] > 0, np.minimum(brok, pa["brokerage_cap"]), brok) * (1 + GST)
                flat = np.where(~buy & (tv > 0), pa["flat_per_sell"], 0.0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    part = np.where(panel.value[t] > 0, tv / panel.value[t], 0.0)
                vol = np.nan_to_num(panel.vol20[t - 1], nan=0.02)
                slip = tv * (pa["half_spread_bps"] / 1e4 + pa["impact_coef"] * vol * np.sqrt(part))
                cost_i = np.where(tv > 0, (fee + brok + flat + slip) * cost_mult, 0.0)
                cost = float(cost_i.sum())
                if fills is not None:
                    for i in np.nonzero(tv > 0)[0]:
                        fills.append({"t": t, "symbol": panel.symbols[i], "delta_w": float(delta[i]), "value": float(tv[i]),
                                      "price": float(o[i]), "cost": float(cost_i[i]), "nav_before": float(nav)})
                # episodes: open / close / flip
                new_w = w + delta
                for i in np.nonzero(tv > 0)[0]:
                    if abs(w[i]) < 1e-12 and abs(new_w[i]) > 1e-12:
                        ep_start[i], ep_pnl[i], ep_cost[i], ep_notional[i], ep_side[i] = t, 0.0, 0.0, 0.0, np.sign(new_w[i])
                ep_cost += cost_i
                ep_notional = np.maximum(ep_notional, np.abs(new_w) * nav)
                pos = new_w * nav
                nav -= cost
                total_cost += cost
                traded += tv.sum()
                w = pos / nav if nav > 0 else pos * 0
                for i in np.nonzero((tv > 0) & (np.abs(new_w) < 1e-12) & (ep_start >= 0))[0]:
                    episodes.append((panel.symbols[i], int(ep_start[i]), t, ep_pnl[i] - ep_cost[i], ep_cost[i], ep_notional[i], ep_side[i]))
                    ep_start[i] = -1
                # unfilled remainders (participation cap) keep working on the next bars, like a real order
                pending = np.where(can & ~limited, np.nan, pending)
                if np.all(np.isnan(pending)):
                    pending = None
            # 3) intraday move
            with np.errstate(invalid="ignore", divide="ignore"):
                ri = np.where(has_o & ~np.isnan(c), c / o - 1, np.where(~has_o & ~np.isnan(c) & (Cf[t - 1] > 0), c / Cf[t - 1] - 1, 0.0))
            pnl = w * ri * nav
            nav_new = nav + pnl.sum()
            ep_pnl += pnl
            w = w * (1 + ri) * nav / nav_new if nav_new > 0 else w * 0
            nav = nav_new
            # 4) cash yield and futures carry
            cash_w = 1.0 - np.sum(np.where(pa["cash_settled"], np.maximum(w, 0.0), 0.0))
            carry = np.sum(np.abs(w) * pa["carry_bps_per_day"] / 1e4) * days[t] * nav * cost_mult
            nav = nav * (1 + max(cash_w, 0.0) * rf[t]) - carry
            total_cost += carry
            w = w * (nav_new / nav) if nav > 0 else w * 0
        # decision at the close of t
        if t in row_of:
            pending = row_of[t].astype(float)
        navs[t], cum_cost[t] = nav, total_cost
        exp_g[t], exp_n[t] = np.abs(w).sum(), w.sum()
        W[t] = w
    for i in np.nonzero(ep_start >= 0)[0]:
        episodes.append((panel.symbols[i], int(ep_start[i]), T - 1, ep_pnl[i] - ep_cost[i], ep_cost[i], ep_notional[i], ep_side[i]))
    idx = panel.dates
    tr = pd.DataFrame(episodes, columns=["symbol", "i0", "i1", "pnl", "cost", "notional", "side"])
    if len(tr):
        tr["entry"], tr["exit"] = idx[tr.i0], idx[tr.i1]
        tr["days"] = (tr["exit"] - tr["entry"]).dt.days
        tr["ret"] = tr.pnl / tr.notional.replace(0, np.nan)
    return RunResult(name, pd.Series(navs, idx), pd.Series(cum_cost, idx), pd.Series(exp_g, idx), pd.Series(exp_n, idx),
                     W, tr, traded / max(float(np.mean(navs)), 1e-9), capital, cost_mult, params=params or {})
