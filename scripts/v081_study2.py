"""v0.8.1 Study 2 — promoter encumbrance (research/preregistration_v0.8.md §2 + Amendments 1, 1a;
research/preregistration_v0.8.1.md §B). Steps:

    python -m scripts.v081_study2 data      # wide shareholding table + coverage audit + power count (NO returns)
    python -m scripts.v081_study2 run --out reports/v0.8.1_<ts>     # only if the audit and the power rule pass
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from marketdata.bse_shp import load as load_shp

BSE = Path("data/pit/bse")
WIDE = BSE / "shareholding_wide.csv"
RET = Path("data/exchange_all/returns.parquet")
UNI = Path("data/exchange_all/universe_monthly.csv")
A = Path("research/data_audits")
START = pd.Timestamp("2016-05-01")
HORIZON = 63
THRESH_EVENT, THRESH_LEVEL = 5.0, 25.0


# ------------------------------------------------------------------------------------------------ data + audit
def wide_codes() -> dict[str, str]:
    w = pd.read_csv("marketdata/reference/bse_codes_wide.csv", dtype=str).dropna(subset=["bse_code"])
    return {r.bse_code: r.sec for r in w.itertuples(index=False) if r.bse_code}


def build_table(budget: float = 140.0) -> pd.DataFrame | None:
    """Parsed per BSE code into data/pit/bse/parsed_wide/<code>.csv (reading ~50k stored files from a mounted folder
    takes several calls); returns the full table once every code is parsed, else None."""
    t0, d = time.time(), BSE / "parsed_wide"
    d.mkdir(exist_ok=True)
    codes = wide_codes()
    import os

    k, n = (int(x) for x in os.environ.get("SHARD", "0/1").split("/"))
    todo = [c for c in sorted(codes)[k::n] if not (d / f"{c}.csv").exists()]
    for c in todo:
        if time.time() - t0 > budget:
            print(f"parsed {len(codes) - len(todo) + todo.index(c)}/{len(codes)} codes; run again")
            return None
        x = load_shp(BSE, {c: codes[c]})
        (x if len(x) else pd.DataFrame(columns=["symbol"])).to_csv(d / f"{c}.csv", index=False)
    if n > 1 or any(not (d / f"{c}.csv").exists() for c in codes):
        print("this shard is parsed; run without SHARD to assemble")
        return None
    parts = [pd.read_csv(d / f"{c}.csv", dtype={"bse_code": str}, low_memory=False) for c in codes]
    df = pd.concat([x for x in parts if len(x)], ignore_index=True).rename(columns={"symbol": "sec"})
    df["sec"] = df["bse_code"].map(codes)                    # security ids from the current mapping (ISIN links can change)
    df.to_csv(WIDE, index=False)
    return df


def read_table() -> pd.DataFrame:
    df = pd.read_csv(WIDE, parse_dates=["period_end"], dtype={"sec": str, "bse_code": str}, low_memory=False)
    df["available_at"] = pd.to_datetime(df["available_at"], utc=True, format="ISO8601").dt.tz_convert("Asia/Kolkata")
    df["E"] = np.where(df["no_promoter"].astype(str) == "True", 0.0, df["encumbered_pct_of_promoter"])
    return df


def quarterly_first(df: pd.DataFrame) -> pd.DataFrame:
    q = df[df["kind"] == "quarterly"].sort_values(["sec", "period_end", "available_at"]).groupby(["sec", "period_end"]).head(1)
    q = q.sort_values(["sec", "period_end"]).copy()
    prev_max = q.groupby("sec")["E"].transform(lambda v: v.shift(1).rolling(4, min_periods=1).max())
    q["dE"] = q["E"] - prev_max
    return q


def membership() -> tuple[np.ndarray, dict]:
    u = pd.read_csv(UNI, parse_dates=["date"], dtype={"sec": str})
    u = u[u["in_top500"]]
    ends = np.sort(u["date"].unique())
    return ends, {pd.Timestamp(d): set(g["sec"]) for d, g in u.groupby("date")}


def member_at(ends, mem, when: pd.Timestamp) -> set:
    i = np.searchsorted(ends, np.datetime64(when.tz_localize(None) if when.tzinfo else when), side="left") - 1
    return mem[pd.Timestamp(ends[i])] if i >= 0 else set()


def data_audit() -> dict | None:
    if build_table() is None:
        return None
    df = read_table()
    q = quarterly_first(df)
    ends, mem = membership()
    # coverage: TOP500 member-quarters with a quarterly filing for that period (mapped companies and all members)
    w = pd.read_csv("marketdata/reference/bse_codes_wide.csv", dtype=str)
    mapped = set(w.dropna(subset=["bse_code"])["sec"])
    have = set(zip(q["sec"], q["period_end"]))
    rows = []
    for qe in pd.date_range("2016-06-30", "2026-06-30", freq="QE"):
        for s in member_at(ends, mem, qe + pd.Timedelta(days=1)):
            rows.append((qe, s, s in mapped, (s, qe) in have))
    c = pd.DataFrame(rows, columns=["q", "sec", "mapped", "has"])
    lag = (q["available_at"].dt.tz_localize(None) - q["period_end"]).dt.days
    # power count (Amendment 1 rule; members at the previous month-end; from START)
    q["member"] = [s in member_at(ends, mem, t) for s, t in zip(q["sec"], q["available_at"])]
    ev = q[q["member"] & (q["available_at"] >= START.tz_localize("Asia/Kolkata")) & (q["dE"] >= THRESH_EVENT)]
    res = {"filings": int(len(df)), "companies": int(df["sec"].nunique()), "quarterly_first_filings": int(len(q)),
           "sources": df["source"].value_counts().to_dict(),
           "member_quarters": int(len(c)), "covered_all_members": round(float(c["has"].mean()), 4),
           "covered_mapped_members": round(float(c[c["mapped"]]["has"].mean()), 4),
           "by_year_covered": {int(k): round(float(v), 4) for k, v in c.groupby(c["q"].dt.year)["has"].mean().items()},
           "filing_lag_days_median": float(lag.median()), "filed_after_45_days_share": round(float((lag > 45).mean()), 4),
           "pct_out_of_range": int(sum(((df[x] < 0) | (df[x] > 100.0001)).sum() for x in ("promoter_pct", "encumbered_pct_of_promoter"))),
           "encumbrance_unknown_with_promoter": int((df["encumbered_pct_of_promoter"].isna() & (df["no_promoter"].astype(str) != "True")).sum()),
           "power": {"rule": "Amendment 1 H2a events among TOP500 members (previous month-end), from 2016-05-01",
                     "events": int(len(ev)), "companies": int(ev["sec"].nunique()), "required": {"events": 100, "companies": 30},
                     "powered": bool(len(ev) >= 100 and ev["sec"].nunique() >= 30)},
           "high_encumbrance_member_quarters": int(((q["E"] >= THRESH_LEVEL) & q["member"]).sum()),
           "made_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    res["accepted"] = bool(res["covered_mapped_members"] >= 0.95 and res["pct_out_of_range"] == 0)
    (A / f"study2_wide_data_{time.strftime('%Y-%m-%d')}.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))
    return res


# ------------------------------------------------------------------------------------------------ returns
def load_returns(secs: set[str]) -> dict[str, pd.DataFrame]:
    r = pd.read_parquet(RET, columns=["date", "sec", "cc", "on", "oc", "value"])
    r = r[r["sec"].isin(secs)]
    return {k: r.pivot(index="date", columns="sec", values=k).astype("float64") for k in ("cc", "on", "oc", "value")}


def make_panel(M: dict[str, pd.DataFrame], ranks: pd.Series, tier_override: str | None = None):
    """lab.panel.Panel with synthetic adjusted price levels (C from cc, O from on) and real traded value."""
    from execution.costs import PROFILES
    from lab.panel import Panel

    cc, on = M["cc"], M["on"]
    dates, secs = cc.index, list(cc.columns)
    C = np.full(cc.shape, np.nan)
    O = np.full(cc.shape, np.nan)
    last = np.full(len(secs), np.nan)
    ccv, onv = cc.to_numpy(), on.to_numpy()
    for t in range(len(dates)):
        base = np.where(np.isnan(last), 100.0, last)
        traded = ~np.isnan(ccv[t])
        C[t] = np.where(traded, base * (1 + ccv[t]), np.nan)
        O[t] = np.where(traded & ~np.isnan(onv[t]), base * (1 + onv[t]), np.where(traded, C[t], np.nan))
        last = np.where(traded, C[t], last)
    val = M["value"].reindex(index=dates, columns=secs).to_numpy()
    V = np.where(C > 0, val / C, 0.0)
    tier = lambda r: "nse_largecap" if r <= 100 else ("nse_midcap" if r <= 250 else "nse_smallcap")  # noqa: E731
    prof = [PROFILES[tier_override or tier(ranks.get(s, 500))] for s in secs]
    bench = pd.Series(np.nan, index=dates)
    return Panel(dates, secs, O, O, O, C, np.nan_to_num(V), ~np.isnan(C), ["eq"] * len(secs), ["default"] * len(secs), prof, bench, "top500_wide")


def run_study(out: Path) -> dict:
    from scipy import stats as sps

    from accounting.tax_india import post_tax_nav
    from lab import engine
    from lab import v05 as V
    from lab.gates import GateInput, capital_efficiency_gate

    audit = json.loads(sorted(A.glob("study2_wide_data_*.json"))[-1].read_text())
    if not (audit["accepted"] and audit["power"]["powered"]):
        raise SystemExit(f"blocked by the pre-registration: accepted={audit['accepted']} powered={audit['power']['powered']}")
    out.mkdir(parents=True, exist_ok=True)
    df = read_table()
    q = quarterly_first(df)
    ends, mem = membership()
    u = pd.read_csv(UNI, parse_dates=["date"], dtype={"sec": str})
    u5 = u[u["in_top500"]]
    secs = set(u5["sec"])
    ranks = u5.groupby("sec")["rank"].median()
    M = load_returns(secs)
    cc, oc = M["cc"], M["oc"]
    dates = cc.index
    last_d = dates[-1]

    # ---- H2a: event CARs (63 sessions from the next open after available_at), vs EW TOP500 buy-and-hold
    q["member"] = [s in member_at(ends, mem, t) for s, t in zip(q["sec"], q["available_at"])]
    ev = q[q["member"] & (q["available_at"] >= START.tz_localize("Asia/Kolkata")) & (q["dE"] >= THRESH_EVENT)].copy()
    rows = []
    for e in ev.itertuples(index=False):
        i0 = dates.searchsorted(pd.Timestamp(e.available_at.date()), side="right")         # first session after the filing day
        if i0 + HORIZON > len(dates) or e.sec not in cc.columns:
            continue
        win = slice(i0, i0 + HORIZON)
        uni = [s for s in member_at(ends, mem, dates[i0]) if s in cc.columns]

        def bh(cols):
            r = cc.iloc[win][cols].copy()
            r.iloc[0] = oc.iloc[i0][cols]                    # entry at the open of the first session
            return (1 + r.fillna(0.0)).prod() - 1
        stock = float(bh([e.sec]).iloc[0])
        bench = float(bh(uni).mean())
        rows.append({"sec": e.sec, "available_at": e.available_at, "entry": dates[i0], "dE": e.dE, "E": e.E, "car": stock - bench})
    C = pd.DataFrame(rows)
    C["month"] = pd.to_datetime(C["entry"]).dt.to_period("M")
    # descriptive robustness (outside the Holm family): events of TOP200 members at the previous month-end
    u2 = u[u["in_top200"]]
    mem2 = {pd.Timestamp(d): set(g["sec"]) for d, g in u2.groupby("date")}
    C["top200"] = [s in member_at(ends, mem2, a) for s, a in zip(C["sec"], C["available_at"])]
    c2 = C[C["top200"]]
    m2 = c2.groupby("month")["car"].mean()
    top200 = {"events": int(len(c2)), "companies": int(c2["sec"].nunique()), "mean_car": float(c2["car"].mean()),
              "t_by_month": float(m2.mean() / (m2.std(ddof=1) / np.sqrt(len(m2)))) if len(m2) > 2 else None}
    m = C.groupby("month")["car"].mean()
    t_a = float(m.mean() / (m.std(ddof=1) / np.sqrt(len(m)))) if len(m) > 2 else float("nan")
    p_a = float(2 * sps.t.sf(abs(t_a), len(m) - 1)) if len(m) > 2 else float("nan")

    # ---- H2b: monthly Fama-MacBeth: next-month return on 1{E>=25} + log median traded value
    from lab.v05 import nw_ols
    dec = [pd.Timestamp(d) for d in ends if pd.Timestamp(d) >= START and pd.Timestamp(d) < last_d]
    q_sorted = df[df["kind"] == "quarterly"].sort_values("available_at")      # every version: the latest public one is used
    slopes, excl_targets, bench_targets = [], {}, {}
    val126 = M["value"].where(M["value"] > 0).rolling(126, min_periods=100).median()
    recent_events = ev[["sec", "available_at"]].copy()
    recent_events["d"] = recent_events["available_at"].dt.tz_localize(None).dt.normalize()
    for k, t in enumerate(dec):
        members = [s for s in mem[t] if s in cc.columns]
        cut = pd.Timestamp(t.date()).tz_localize("Asia/Kolkata") + pd.Timedelta(hours=15, minutes=30)
        Eknown = q_sorted[q_sorted["available_at"] < cut].sort_values(["period_end", "available_at"]).groupby("sec").tail(1).set_index("sec")["E"]
        E = Eknown.reindex(members)
        flag = (E >= THRESH_LEVEL).fillna(False)
        recent = set(recent_events[(recent_events["d"] <= t) & (recent_events["d"] > dates[max(0, dates.get_loc(t) - HORIZON)])]["sec"])
        keep_ = [s for s in members if not flag.get(s, False) and s not in recent]
        excl_targets[t] = pd.Series(1.0 / len(keep_), index=keep_)
        bench_targets[t] = pd.Series(1.0 / len(members), index=members)
        nxt = dec[k + 1] if k + 1 < len(dec) else last_d
        i0, i1 = dates.get_loc(t) + 1, dates.get_loc(nxt)
        if i1 < i0:
            continue
        r = cc.iloc[i0:i1 + 1][members].copy()
        r.iloc[0] = oc.iloc[i0][members]
        y = (1 + r.fillna(0.0)).prod() - 1
        x1 = flag.astype(float).reindex(members)
        x2 = np.log(val126.loc[t].reindex(members))
        d = pd.DataFrame({"y": y, "x1": x1, "x2": x2}).replace([np.inf, -np.inf], np.nan).dropna()
        d = d[E.reindex(d.index).notna()]                      # companies with a known encumbrance only
        if len(d) > 50 and d["x1"].sum() >= 3:
            X = np.column_stack([np.ones(len(d)), d["x1"], d["x2"]])
            b = np.linalg.lstsq(X, d["y"].to_numpy(), rcond=None)[0]
            slopes.append((t, b[1], int(d["x1"].sum()), len(d)))
    S = pd.DataFrame(slopes, columns=["t", "slope", "n_flag", "n"])
    fm = nw_ols(S["slope"].to_numpy(), np.empty((len(S), 0)), lags=6)
    t_b = float(fm["t"][0])
    p_b = float(2 * sps.t.sf(abs(t_b), len(S) - 1))
    ps = np.array([p_a, p_b])
    order = np.argsort(ps)
    holm = np.empty(2)
    for j, i in enumerate(order):
        holm[i] = min(1.0, max(ps[i] * (2 - j), holm[order[j - 1]] if j else 0))
    stat_pass = bool((holm[0] < 0.05 and np.mean(C["car"]) < 0) or (holm[1] < 0.05 and S["slope"].mean() < 0))

    # ---- implementation through the engine (exclusion vs EW TOP500, both paying costs)
    P = make_panel(M, ranks)
    tg = pd.DataFrame(excl_targets).T.reindex(columns=P.symbols).fillna(0.0)
    bm = pd.DataFrame(bench_targets).T.reindex(columns=P.symbols).fillna(0.0)
    start = tg.index[0]
    Cf = P.ffill_close()
    end_px = {s: float(Cf[-1, P.idx[s]]) for s in P.symbols if not np.isnan(Cf[-1, P.idx[s]])}

    def cg(nav):
        n = nav[nav.index >= start]
        return float((n.iloc[-1] / n.iloc[0]) ** (365.25 / (n.index[-1] - n.index[0]).days) - 1)
    fl, fb = [], []
    r1, rb = engine.run(P, tg, fills=fl), engine.run(P, bm, fills=fb)
    r2, r0 = engine.run(P, tg, cost_mult=2.0), engine.run(P, tg, cost_mult=0.0)
    rb2 = engine.run(P, bm, cost_mult=2.0)
    rc, rc0, rbc = engine.run(P, tg, capital=1e7), engine.run(P, tg, capital=1e7, cost_mult=0.0), engine.run(P, bm, capital=1e7)

    def post(nav, fills):
        tx = post_tax_nav(nav, fills, P.dates, end_px, start=start)
        n = tx["post_tax_nav"]
        return float((tx["liquidation_value"] / n.iloc[0]) ** (365.25 / (n.index[-1] - n.index[0]).days) - 1)
    mo = (V.monthly(r1.nav[r1.nav.index >= start]) - V.monthly(rb.nav[rb.nav.index >= start])).dropna()
    subs = [float(mo[a:b].sum()) for a, b in V.SUBPERIODS if len(mo[a:b])]
    gi = GateInput(statistical_pass=stat_pass, net_excess=cg(r1.nav) - cg(rb.nav), post_tax_cagr=post(r1.nav, fl),
                   benchmark_post_tax_cagr=post(rb.nav, fb), net_excess_2x=cg(r2.nav) - cg(rb2.nav),
                   cost_drag_10L=cg(r0.nav) - cg(r1.nav), cost_drag_1Cr=cg(rc0.nav) - cg(rc.nav),
                   net_excess_1Cr=cg(rc.nav) - cg(rbc.nav), subperiod_excess=subs)
    gate = capital_efficiency_gate(gi)
    rb0 = engine.run(P, bm, cost_mult=0.0)
    Ps = make_panel(M, ranks, tier_override="nse_smallcap")       # tiers use a median rank over the whole history: bound it
    sens = {"gross_excess_zero_costs": cg(r0.nav) - cg(rb0.nav),
            "net_excess_all_names_smallcap_costs": cg(engine.run(Ps, tg).nav) - cg(engine.run(Ps, bm).nav),
            "note": "net excess also reflects fewer holdings paying fewer fixed per-scrip charges (DP Rs 15.34 per sale) at Rs 10 lakh"}
    res = {"H2a": {"events": int(len(C)), "companies": int(C["sec"].nunique()), "mean_car": float(C["car"].mean()),
                   "median_car": float(C["car"].median()), "months": int(len(m)), "t": t_a, "p": p_a, "holm_p": float(holm[0])},
           "H2b": {"months": int(len(S)), "mean_slope_monthly": float(S["slope"].mean()), "nw_t": t_b, "p": p_b, "holm_p": float(holm[1]),
                   "avg_flagged": float(S["n_flag"].mean())},
           "H2a_top200_descriptive": top200, "cost_sensitivity": sens,
           "statistical_pass": stat_pass, "gate_input": gi.__dict__, "gate": gate,
           "strategy_cagr": cg(r1.nav), "benchmark_cagr": cg(rb.nav), "window": [str(start.date()), str(last_d.date())],
           "turnover_note": "both books rebalance monthly to equal weight and pay the same costs"}
    C.to_csv(out / "study2_events.csv", index=False)
    S.to_csv(out / "study2_fm_slopes.csv", index=False)
    pd.DataFrame({"strategy": r1.nav, "benchmark": rb.nav}).to_csv(out / "study2_navs.csv")
    (out / "study2.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "gate_input"}, indent=1, default=str))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["data", "run"])
    ap.add_argument("--out", default=f"reports/v0.8.1_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
    a = ap.parse_args()
    data_audit() if a.cmd == "data" else run_study(Path(a.out))


if __name__ == "__main__":
    main()
