"""v0.8.1 crypto studies C1 (BTC/ETH 140-day trend) and C2 (TOP50 21-day momentum, top 5), as pre-registered in
research/preregistration_v0.8.1.md section C.

    python -m scripts.fetch_crypto --budget 160        # data (network)
    python -m scripts.v081_crypto audit                # data audit (no returns)
    python -m scripts.v081_crypto run --out reports/v0.8.1_<ts>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from marketdata.crypto_binance import ROOT, research_symbol

A = Path("research/data_audits")
USDINR = 84.0
CAP_10L, CAP_1CR = 10_00_000 / USDINR, 1_00_00_000 / USDINR
SUBS = [("2018-01", "2019-12"), ("2020-01", "2021-12"), ("2022-01", "2023-12"), ("2024-01", "2026-09")]
FIRST, LAST = pd.Timestamp("2018-01-07"), pd.Timestamp("2026-09-27")


def load_all() -> dict[str, pd.DataFrame]:
    frames = []
    for p in sorted((ROOT / "1d").glob("*.csv.gz")):
        base = research_symbol(p.name.split(".")[0])
        if not base:
            continue
        d = pd.read_csv(p, parse_dates=["date"])
        if d.empty:
            continue
        d["sym"] = base
        frames.append(split_redenominations(d))
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["close"] > 0) & (df["date"] <= LAST + pd.Timedelta(days=8))]
    return {k: df.pivot_table(index="date", columns="sym", values=k, aggfunc="last") for k in ("open", "close", "volume", "quote_volume")}


def split_redenominations(d: pd.DataFrame, gap_days: int = 7, ratio: float = 20.0) -> pd.DataFrame:
    """A coin that stops trading for a week or more and comes back at a price 20x away (a redenomination, a token
    swap, or a different coin re-using the ticker) is split into two series (SYM and SYM#2): no return is ever
    computed across the gap."""
    d = d.sort_values("date").reset_index(drop=True)
    gap = d["date"].diff().dt.days.fillna(0)
    jump = (d["close"].shift(1) / d["open"]).fillna(1.0)
    cut = (gap >= gap_days) & ((jump > ratio) | (jump < 1 / ratio))
    k = cut.cumsum()
    d["sym"] = np.where(k == 0, d["sym"], d["sym"] + "#" + (k + 1).astype(str))
    return d


def audit() -> dict:
    M = load_all()
    c = M["close"]
    life = c.notna()
    first, last = life.idxmax(), life[::-1].idxmax()
    gaps = {s: int(((~life[s]) & (c.index > first[s]) & (c.index < last[s])).sum()) for s in c.columns}
    r = np.log(c).diff()
    ext = int((r.abs() > np.log(3)).sum().sum())
    res = {"coins": int(c.shape[1]), "split_series": [x for x in c.columns if "#" in x], "days": int(c.shape[0]), "first": str(c.index.min().date()), "last": str(c.index.max().date()),
           "delisted_before_end": int((last < c.index.max() - pd.Timedelta(days=7)).sum()),
           "coins_with_internal_gaps": int(sum(v > 0 for v in gaps.values())), "internal_gap_days": int(sum(gaps.values())),
           "daily_moves_beyond_3x": ext,
           "eligible_coins_by_year": {int(y): int((life[(c.index.year == y)].any()).sum()) for y in range(2017, 2027)},
           "made_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    res["accepted"] = bool(res["coins"] > 300 and res["delisted_before_end"] > 0)
    (A / f"crypto_data_{time.strftime('%Y-%m-%d')}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return res


def panel(M):
    from execution.costs import PROFILES
    from lab.panel import Panel

    c = M["close"]
    o = M["open"].reindex_like(c)
    v = M["volume"].reindex_like(c).fillna(0.0)
    syms = list(c.columns)
    prof = [PROFILES["crypto_inr"]] * len(syms)
    return Panel(c.index, syms, o.to_numpy(), o.to_numpy(), o.to_numpy(), c.to_numpy(), v.to_numpy(), c.notna().to_numpy(),
                 ["crypto"] * len(syms), ["crypto"] * len(syms), prof, pd.Series(np.nan, index=c.index), "binance_usdt")


def targets(M) -> dict[str, pd.DataFrame]:
    c, qv = M["close"], M["quote_volume"]
    sundays = [d for d in c.index if d.weekday() == 6 and FIRST <= d <= LAST]
    hist = c.notna().cumsum()
    med30 = qv.where(qv > 0).rolling(30, min_periods=20).median()
    sma = c.rolling(140, min_periods=140).mean()
    mom = c / c.shift(21) - 1
    out = {"C1_BTC": {}, "C1_ETH": {}, "BH_BTC": {}, "BH_ETH": {}, "C2": {}, "C2_BENCH": {}}
    for d in sundays:
        for coin in ("BTC", "ETH"):
            on = bool(c.at[d, coin] > sma.at[d, coin]) if not np.isnan(sma.at[d, coin]) else False
            out[f"C1_{coin}"][d] = pd.Series({coin: 1.0 if on else 0.0})
            out[f"BH_{coin}"][d] = pd.Series({coin: 1.0})
        ok = (hist.loc[d] >= 60) & c.loc[d].notna() & med30.loc[d].notna()
        top = med30.loc[d][ok].sort_values(ascending=False).index[:50]
        m = mom.loc[d, top].dropna().sort_values(ascending=False)
        win = list(m.index[:5])
        out["C2"][d] = pd.Series(1.0 / len(win), index=win) if win else pd.Series(dtype=float)
        out["C2_BENCH"][d] = pd.Series(1.0 / len(top), index=top) if len(top) else pd.Series(dtype=float)
    return {k: pd.DataFrame(v).T.fillna(0.0) for k, v in out.items()}


def run(out: Path) -> dict:
    from scipy import stats as sps

    from accounting.tax_crypto_india import crypto_post_tax
    from lab import engine
    from lab.gates import GateInput, capital_efficiency_gate
    from lab.v05 import nw_ols

    au = json.loads(sorted(A.glob("crypto_data_*.json"))[-1].read_text())
    if not au["accepted"]:
        raise SystemExit("crypto data not accepted")
    out.mkdir(parents=True, exist_ok=True)
    M = load_all()
    P = panel(M)
    T = targets(M)
    Cf = P.ffill_close()
    end_px = {s: float(Cf[-1, P.idx[s]]) for s in P.symbols if not np.isnan(Cf[-1, P.idx[s]])}
    start = T["C2"].index[0]

    def cg(nav):
        n = nav[nav.index >= start]
        return float((n.iloc[-1] / n.iloc[0]) ** (365.25 / (n.index[-1] - n.index[0]).days) - 1)

    def go(tg, **kw):
        fl = []
        r = engine.run(P, tg.reindex(columns=P.symbols).fillna(0.0), capital=kw.pop("capital", CAP_10L), cash_rate=0.0, fills=fl, **kw)
        return r, fl

    def post(r, fl, recoverable=True):
        tx = crypto_post_tax(r.nav, fl, P.dates, end_px, start=start, recoverable=recoverable)
        n = tx["post_tax_nav"]
        return float((tx["liquidation_value"] / n.iloc[0]) ** (365.25 / (n.index[-1] - n.index[0]).days) - 1), tx

    res, weekly = {}, {}
    for name, strat, bench in (("C1_BTC", "C1_BTC", "BH_BTC"), ("C1_ETH", "C1_ETH", "BH_ETH"), ("C2", "C2", "C2_BENCH")):
        rs, fs = go(T[strat])
        rb, fb = go(T[bench])
        rs2, _ = go(T[strat], cost_mult=2.0)
        rb2, _ = go(T[bench], cost_mult=2.0)
        rs0, _ = go(T[strat], cost_mult=0.0)
        rc, _ = go(T[strat], capital=CAP_1CR)
        rc0, _ = go(T[strat], capital=CAP_1CR, cost_mult=0.0)
        rbc, _ = go(T[bench], capital=CAP_1CR)
        idx = T[strat].index
        ws = rs.nav.reindex(idx).pct_change().dropna()
        wb = rb.nav.reindex(idx).pct_change().dropna()
        ex = (ws - wb).dropna()
        weekly[name] = ex
        fit = nw_ols(ex.to_numpy(), np.empty((len(ex), 0)), lags=4)
        t = float(fit["t"][0])
        pt_s, txs = post(rs, fs)
        pt_b, txb = post(rb, fb)
        pt_s_nr, txs_nr = post(rs, fs, recoverable=False)
        mo_s = rs.nav[rs.nav.index >= start].resample("ME").last().pct_change()
        mo_b = rb.nav[rb.nav.index >= start].resample("ME").last().pct_change()
        mo = (mo_s - mo_b).dropna()
        subs = [float(mo[a:b].sum()) for a, b in SUBS if len(mo[a:b])]
        res[name] = {"weeks": int(len(ex)), "mean_weekly_excess": float(ex.mean()), "nw_t": t, "p": float(2 * sps.t.sf(abs(t), len(ex) - 1)),
                     "cagr": cg(rs.nav), "benchmark_cagr": cg(rb.nav),
                     "max_drawdown": float((rs.nav / rs.nav.cummax() - 1).min()), "benchmark_max_drawdown": float((rb.nav / rb.nav.cummax() - 1).min()),
                     "gi": GateInput(statistical_pass=False, net_excess=cg(rs.nav) - cg(rb.nav), post_tax_cagr=pt_s, benchmark_post_tax_cagr=pt_b,
                                     net_excess_2x=cg(rs2.nav) - cg(rb2.nav), cost_drag_10L=cg(rs0.nav) - cg(rs.nav),
                                     cost_drag_1Cr=cg(rc0.nav) - cg(rc.nav), net_excess_1Cr=cg(rc.nav) - cg(rbc.nav), subperiod_excess=subs).__dict__,
                     "post_tax_tds_not_recovered": pt_s_nr, "post_tax_wiped_out_on": txs.get("wiped_out_on"),
                     "tax": {"strategy_transfers": txs["transfers"], "losses_not_set_off_usd": txs["losses_not_set_off"],
                                                                     "benchmark_transfers": txb["transfers"]},
                     "turnover_trades": len(fs)}
        pd.DataFrame({"strategy": rs.nav, "benchmark": rb.nav}).to_csv(out / f"crypto_{name}_navs.csv")
    ps = np.array([res[k]["p"] for k in ("C1_BTC", "C1_ETH", "C2")])
    order = np.argsort(ps)
    holm, run_max = np.empty(3), 0.0
    for j, i in enumerate(order):
        run_max = max(run_max, min(1.0, ps[i] * (3 - j)))
        holm[i] = run_max
    from lab.gates import GateInput as GI
    for k, h in zip(("C1_BTC", "C1_ETH", "C2"), holm):
        res[k]["holm_p"] = float(h)
        gi = GI(**res[k].pop("gi"))
        gi.statistical_pass = bool(h < 0.05 and res[k]["mean_weekly_excess"] > 0)
        res[k]["gate_input"] = gi.__dict__
        res[k]["gate"] = capital_efficiency_gate(gi)
    res["window"] = [str(start.date()), str(LAST.date())]
    (out / "crypto.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk not in ("gate_input",)} if isinstance(v, dict) else v) for k, v in res.items()}, indent=1, default=str))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["audit", "run"])
    ap.add_argument("--out", default=f"reports/v0.8.1_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
    a = ap.parse_args()
    audit() if a.cmd == "audit" else run(Path(a.out))


if __name__ == "__main__":
    main()
