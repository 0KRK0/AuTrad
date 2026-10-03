"""AuTrad v0.7 study 1: cash-flow quality (research/preregistration_v0.7.md, section 2 + Amendment 1), the
capital-efficiency gate, and a register of every candidate so far through the same gate.

    python -m scripts.v07 study --out reports/v0.7_<ts>
    python -m scripts.v07 register --out reports/v0.7_<ts>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from accounting.tax_india import LABEL as TAX_LABEL
from accounting.tax_india import post_tax_nav
from lab import engine
from lab import v05 as V
from lab.cfq import cfq_scores
from lab.factors import (FactorContext, decision_dates, forward_returns, fundamental_scores, newey_west_t, price_scores,
                         stock_universe)
from lab.gates import GateInput, capital_efficiency_gate
from lab.panel import build_panel, india_equity_assets
from scripts.v05 import clean

DATA = "data/snap_20261001"
PREREG = Path("research/preregistration_v0.7.md")


def check_prereg() -> str:
    sha = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    logged = [json.loads(x) for x in open("research/trials.jsonl") if '"preregistration"' in x]
    assert any(r.get("sha256") == sha and r.get("version") == "v0.7" for r in logged), "v0.7 pre-registration changed after it was logged"
    return sha


def nw_t(x: np.ndarray, lags: int = 6) -> float:
    return newey_west_t(np.asarray(x, float), lags)


def study(out: Path) -> None:
    from marketdata.exchange import load as load_bse
    from marketdata.exchange import series
    from marketdata.fundamentals import FINANCIAL_SECTORS, PITFundamentals, load_facts, split_events_from_store
    from marketdata.pit import SECTOR
    from marketdata.store import DataStore

    sha = check_prereg()
    P = build_panel(DATA, india_equity_assets(), name="india_pit_xopen", open_source="exchange")
    st = DataStore(DATA)
    st.load(list(P.symbols))
    sev = split_events_from_store(st)
    facts = load_facts("data/pit/nse/facts.csv", split_events=sev)
    fund = PITFundamentals(facts, sev, SECTOR)
    cm = pd.read_csv("marketdata/reference/bse_codes.csv", dtype={"bse_code": str}).set_index("symbol")["bse_code"]
    B = load_bse(Path(DATA).parent / "exchange")
    raw = pd.DataFrame({s: series(B, cm[s])["close"] for s in P.symbols if s in cm.index}).reindex(P.dates).ffill(limit=5)
    ctx = FactorContext(P, raw, fund, SECTOR)
    all_me = decision_dates(P, start="2022-06-01")
    CFQ = cfq_scores(pd.read_csv("data/pit/nse/cashflow.csv"), facts, P, all_me, SECTOR)
    n_scored = CFQ.notna().sum(axis=1)
    dates = [t for t in all_me if n_scored.get(P.dates[t], 0) >= 15]
    first = P.dates[dates[0]]
    fwd = forward_returns(P, dates)
    C = P.ffill_close()
    R = np.vstack([np.full((1, C.shape[1]), np.nan), np.diff(np.log(C), axis=0)])
    m = np.r_[np.nan, np.diff(np.log(P.bench.ffill().to_numpy()))]
    ic, coefs, rows = {}, {}, []
    for t in dates:
        d = P.dates[t]
        if d not in fwd.index:
            continue
        js = [j for j in stock_universe(P, t) if SECTOR.get(P.symbols[j]) not in FINANCIAL_SECTORS]
        ps = price_scores(P, t, js, C, R, m)
        fs = fundamental_scores(ctx, t, js)
        X = pd.DataFrame({"CFQ": CFQ.loc[d], "REV1": ps["REV1"], "MOM12": ps["MOM12"], "EY": fs["EY"], "ROE": fs["ROE"]})
        r = fwd.loc[d]
        k = X["CFQ"].notna() & r.notna()
        if k.sum() >= 15:
            ic[d] = sps.spearmanr(X.loc[k, "CFQ"], r[k]).statistic
        full = X.dropna().index.intersection(r.dropna().index)
        if len(full) >= 15:
            Z = X.loc[full].rank(pct=True)
            A = np.column_stack([np.ones(len(full)), Z.to_numpy()])
            beta, *_ = np.linalg.lstsq(A, r[full].to_numpy(), rcond=None)
            coefs[d] = dict(zip(["const", *X.columns], beta))
        rows.append({"date": d, "n_cfq": int(X["CFQ"].notna().sum()), "n_fm": int(len(full))})
    ic = pd.Series(ic)
    cf = pd.DataFrame(coefs).T
    t1, t2 = nw_t(ic.to_numpy()), nw_t(cf["CFQ"].to_numpy())
    p1, p2 = float(1 - sps.t.cdf(t1, len(ic) - 1)), float(1 - sps.t.cdf(t2, len(cf) - 1))
    hp = V.holm({"S1": p1, "S2": p2})
    res: dict = {"prereg_sha256": sha, "first_scored_month_end": str(first.date()), "months_ic": int(len(ic)), "months_fm": int(len(cf)),
                 "avg_names_scored": float(pd.DataFrame(rows)["n_cfq"].mean()), "avg_names_fm": float(pd.DataFrame(rows)["n_fm"].mean()),
                 "S1": {"mean_ic": float(ic.mean()), "nw_t": t1, "p_one_sided": p1, "holm_p": hp["S1"], "pct_months_positive": float((ic > 0).mean())},
                 "S2": {"mean_coef_bps": float(cf["CFQ"].mean() * 1e4), "nw_t": t2, "p_one_sided": p2, "holm_p": hp["S2"],
                        "other_coefs_bps": {c: float(cf[c].mean() * 1e4) for c in ["REV1", "MOM12", "EY", "ROE"]}}}
    stat_pass = (hp["S1"] < 0.05 and ic.mean() > 0) or (hp["S2"] < 0.05 and cf["CFQ"].mean() > 0)

    # semi-annual portfolio: last session of June and December, from Dec 2022
    sem = [t for t in all_me if P.dates[t].month in (6, 12) and P.dates[t] >= pd.Timestamp("2022-12-01")]
    tg, ub, us_, cov = {}, {}, {}, []
    for t in sem:
        d = P.dates[t]
        s = CFQ.loc[d].dropna()
        if len(s) < 15:
            continue
        n = len(s) // 3
        w = pd.Series(0.0, index=P.symbols)
        w[s.nlargest(n).index] = 1.0 / n
        tg[d] = w
        js = [P.symbols[j] for j in stock_universe(P, t) if SECTOR.get(P.symbols[j]) not in FINANCIAL_SECTORS]
        u = pd.Series(0.0, index=P.symbols)
        u[js] = 1.0 / len(js)
        ub[d] = u
        us = pd.Series(0.0, index=P.symbols)
        us[s.index] = 1.0 / len(s)
        us_[d] = us
        cov.append({"date": str(d.date()), "scored": int(len(s)), "eligible_non_financial": int(len(js))})
    TG, UB, US = pd.DataFrame(tg).T, pd.DataFrame(ub).T, pd.DataFrame(us_).T
    d0 = TG.index[0]

    def run(tg_, **kw):
        fl: list = []
        r = engine.run(P, tg_, fills=fl, **kw)
        return r, fl

    def cagr(nav):
        n = nav[nav.index >= d0]
        return float((n.iloc[-1] / n.iloc[0]) ** (365.25 / (n.index[-1] - n.index[0]).days) - 1)

    r1, f1 = run(TG)
    rb, fb = run(UB)
    r2, _ = run(TG, cost_mult=2.0)
    r0, _ = run(TG, cost_mult=0.0)
    rc, _ = run(TG, capital=1e7)
    rc0, _ = run(TG, capital=1e7, cost_mult=0.0)
    end_px = {s: float(C[-1, P.idx[s]]) for s in P.symbols}
    tx1 = post_tax_nav(r1.nav, f1, P.dates, end_px, start=d0)
    txb = post_tax_nav(rb.nav, fb, P.dates, end_px, start=d0)

    def post(tx):
        n = tx["post_tax_nav"]
        return float((tx["liquidation_value"] / n.iloc[0]) ** (365.25 / (n.index[-1] - n.index[0]).days) - 1)
    mo = (V.monthly(r1.nav[r1.nav.index >= d0]) - V.monthly(rb.nav[rb.nav.index >= d0])).dropna()
    subs = {"2022-12..2024-06": float(mo["2022-12":"2024-06"].sum()), "2024-07..2026-09": float(mo["2024-07":"2026-09"].sum())}
    traded = sum(x["value"] for x in f1 if P.dates[x["t"]] > d0 + pd.Timedelta(days=5))
    live = r1.nav[r1.nav.index >= d0]
    port = {"rebalances": int(len(TG)), "first_rebalance": str(d0.date()), "avg_names": float((TG > 0).sum(axis=1).mean()),
            "cagr": cagr(r1.nav), "benchmark_cagr": cagr(rb.nav), "net_excess": cagr(r1.nav) - cagr(rb.nav),
            "gross_excess": cagr(r0.nav) - cagr(rb.nav), "net_excess_2x": cagr(r2.nav) - cagr(rb.nav),
            "net_excess_1Cr": cagr(rc.nav) - cagr(rb.nav), "cost_drag_10L": cagr(r0.nav) - cagr(r1.nav), "cost_drag_1Cr": cagr(rc0.nav) - cagr(rc.nav),
            "turnover_per_year": float(traded / live.mean() / ((live.index[-1] - live.index[0]).days / 365.25)),
            "post_tax_cagr": post(tx1), "benchmark_post_tax_cagr": post(txb), "tax_label": TAX_LABEL,
            "subperiod_excess": subs, "monthly_excess_t": float(mo.mean() / mo.std() * np.sqrt(len(mo))) if mo.std() > 0 else None}
    rs, _ = run(US)
    ms = (V.monthly(r1.nav[r1.nav.index >= d0]) - V.monthly(rs.nav[rs.nav.index >= d0])).dropna()
    port["coverage_by_rebalance"] = cov
    port["sensitivity_vs_scored_names_only"] = {"note": "descriptive (not a decision input): benchmark = equal weight of the stocks that had a CFQ score",
                                                "net_excess": cagr(r1.nav) - cagr(rs.nav),
                                                "subperiods": {"2022-12..2024-06": float(ms["2022-12":"2024-06"].sum()), "2024-07..2026-09": float(ms["2024-07":"2026-09"].sum())}}
    res["portfolio"] = port
    gate = capital_efficiency_gate(GateInput(stat_pass, port["net_excess"], port["post_tax_cagr"], port["benchmark_post_tax_cagr"],
                                             port["net_excess_2x"], port["cost_drag_10L"], port["cost_drag_1Cr"], port["net_excess_1Cr"],
                                             list(subs.values()), "not started"))
    res["gate"] = gate
    out.mkdir(parents=True, exist_ok=True)
    (out / "cfq.json").write_text(json.dumps(clean(res), indent=1))
    pd.DataFrame({"ic": ic}).join(cf.add_prefix("fm_")).to_csv(out / "cfq_monthly.csv")
    print(json.dumps(clean({k: v for k, v in res.items() if k in ("first_scored_month_end", "months_ic", "S1", "S2")}), indent=1))
    print(json.dumps(clean(port), indent=1)[:1500])
    print(gate["verdict"])


def _metrics(P, tg, bench, start, fills_bench=None):
    """Net excess, 2x costs, Rs 1 Cr, cost drags, after-tax estimate and subperiods of a target book vs a benchmark book."""
    C = P.ffill_close()
    end_px = {s: float(C[-1, P.idx[s]]) for s in P.symbols}

    def cg(nav):
        n = nav[nav.index >= start]
        return float((n.iloc[-1] / n.iloc[0]) ** (365.25 / (n.index[-1] - n.index[0]).days) - 1)
    fl, fb = [], []
    r1 = engine.run(P, tg, fills=fl)
    rb = engine.run(P, bench, fills=fb)
    r2, r0 = engine.run(P, tg, cost_mult=2.0), engine.run(P, tg, cost_mult=0.0)
    rc, rc0 = engine.run(P, tg, capital=1e7), engine.run(P, tg, capital=1e7, cost_mult=0.0)

    def post(nav, fills):
        tx = post_tax_nav(nav, fills, P.dates, end_px, start=start)
        n = tx["post_tax_nav"]
        return float((tx["liquidation_value"] / n.iloc[0]) ** (365.25 / (n.index[-1] - n.index[0]).days) - 1)
    mo = (V.monthly(r1.nav[r1.nav.index >= start]) - V.monthly(rb.nav[rb.nav.index >= start])).dropna()
    subs = [float(mo[a:b].sum()) for a, b in V.SUBPERIODS if len(mo[a:b])]
    return dict(net_excess=cg(r1.nav) - cg(rb.nav), post_tax_cagr=post(r1.nav, fl), benchmark_post_tax_cagr=post(rb.nav, fb),
                net_excess_2x=cg(r2.nav) - cg(rb.nav), cost_drag_10L=cg(r0.nav) - cg(r1.nav), cost_drag_1Cr=cg(rc0.nav) - cg(rc.nav),
                net_excess_1Cr=cg(rc.nav) - cg(rb.nav), subperiod_excess=subs)


def register(out: Path) -> None:
    """Every candidate through the gate, each measured over its own pre-registered window on one consistent source
    (exchange-anchored opens, full costs, the same tax estimate)."""
    from lab.factors import tercile_targets, universe_targets
    from scripts.v06 import bad_events, variants

    P = build_panel(DATA, india_equity_assets(), name="india_pit_xopen", open_source="exchange")
    C = P.ffill_close()
    R = np.vstack([np.full((1, C.shape[1]), np.nan), np.diff(np.log(C), axis=0)])
    m = np.r_[np.nan, np.diff(np.log(P.bench.ffill().to_numpy()))]
    dates = decision_dates(P)
    S = pd.DataFrame({P.dates[t]: price_scores(P, t, stock_universe(P, t), C, R, m)["REV1"] for t in dates}).T.reindex(columns=P.symbols).dropna(how="all")
    uni = universe_targets(S, P)
    rows = []
    rev1 = _metrics(P, tercile_targets(S), uni, S.index[0])
    rows.append(("REV1 one-month reversal (frozen benchmark)", "v0.4 window 2016-05..2026-09, recomputed with exchange opens", True, rev1, "running"))
    ev = bad_events(P)
    rep, _, _, _ = variants(S, P, ev, 20)
    start = S.index[S.index >= P.dates[int(ev.loc[ev["bad"] == 1, "reaction_idx"].min())]][0]
    rows.append(("REV1 + earnings veto (REPLACE)", "v0.6 window 2019-10..2026-09", False, _metrics(P, rep, uni, start), "not started"))
    v5 = sorted(Path("reports").glob("v0.5_*/trackB.json"))[-1]
    b5 = json.loads(v5.read_text())["objects"]["low_vol_sleeve"]
    rows.append(("Low-volatility sleeve", f"v0.5 Track B ({v5.parent.name})", False,
                 dict(net_excess=b5["net_excess_vs_beta_matched"]["ann"], post_tax_cagr=None, benchmark_post_tax_cagr=None,
                      net_excess_2x=b5["cost_shock"]["2.0"]["vs_beta_matched_ann"], cost_drag_10L=b5["costs"]["cost_drag_10L"],
                      cost_drag_1Cr=b5["costs"]["cost_drag_1Cr"], net_excess_1Cr=None, subperiod_excess=[v["vs_beta_matched_ann"] for v in b5["subperiods"].values()]),
                 "not started"))
    c5 = json.loads(sorted(Path("reports").glob("v0.5_*/trackC.json"))[-1].read_text())
    t1 = c5["T1_avoid_overlay"]
    rows.append(("Bad-earnings avoid overlay", "v0.5 Track C (T1)", any((c5["tests"][k]["holm_p"] or 1) < 0.05 for k in ("C1", "C2", "C3")),
                 dict(net_excess=t1["1.0x"]["value_ann"], post_tax_cagr=None, benchmark_post_tax_cagr=None, net_excess_2x=t1["2.0x"]["value_ann"],
                      cost_drag_10L=None, cost_drag_1Cr=None, net_excess_1Cr=None, subperiod_excess=[]), "not started"))
    out_rows = []
    for name, src, stat, mt, fwd in rows:
        g = capital_efficiency_gate(GateInput(stat, mt["net_excess"], mt["post_tax_cagr"], mt["benchmark_post_tax_cagr"], mt["net_excess_2x"],
                                              mt["cost_drag_10L"], mt["cost_drag_1Cr"], mt["net_excess_1Cr"], mt["subperiod_excess"], fwd))
        out_rows.append({"candidate": name, "source": src, "inputs": mt, "verdict": g["verdict"], "steps": g["steps"]})
    p = out / "cfq.json"
    if p.exists():
        c = json.loads(p.read_text())
        out_rows.append({"candidate": "Cash-flow quality, semi-annual (v0.7 study 1)", "source": "v0.7 window 2022-12..2026-09", "verdict": c["gate"]["verdict"],
                         "steps": c["gate"]["steps"]})
    (out / "candidate_register.json").write_text(json.dumps(clean(out_rows), indent=1))
    for r in out_rows:
        print(f"{r['candidate']:52} {r['verdict']}")


def audit(out: Path) -> None:
    """Data audit for study 1 (re-runnable): cash-flow, total-assets and quarterly-net-profit coverage per filing period."""
    from marketdata.fundamentals import FINANCIAL_SECTORS
    from marketdata.pit import SECTOR

    d = pd.read_csv("data/pit/nse/cashflow.csv")
    d["pe"] = pd.to_datetime(d["period_end"])
    f = pd.read_csv("data/pit/nse/facts.csv")
    f["pe"] = pd.to_datetime(f["period_end"])
    nonfin = lambda s: SECTOR.get(s) not in FINANCIAL_SECTORS  # noqa: E731
    d, f = d[d["symbol"].map(nonfin)], f[f["symbol"].map(nonfin)]
    q = f[(f["period_days"].isna() | (f["period_days"] < 120)) & (f["pe"] >= "2021-06-01")]
    syms = sorted(f["symbol"].unique())
    np_cov = {str(pe.date()): int(q.loc[(q["pe"] == pe) & q["net_profit"].notna(), "symbol"].nunique()) for pe in sorted(q["pe"].unique())}
    res = {"generated": "2026-10-03", "non_financial_symbols_with_filings": len(syms),
           "ocf_found_by_period": d[d["status"] == "ok"].groupby(d["pe"].dt.strftime("%Y-%m")).size().to_dict(),
           "total_assets_present_by_period": f[f["pe"].dt.month.isin([3, 9])].groupby(f["pe"].dt.strftime("%Y-%m"))["total_assets"].apply(lambda x: int(x.notna().sum())).to_dict(),
           "quarterly_net_profit_present_by_quarter": np_cov,
           "missing_2022_06_quarter": sorted(set(syms) - set(q.loc[(q["pe"] == "2022-06-30") & q["net_profit"].notna(), "symbol"]))}
    (out / "data_audit_study1.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=0)[:1500])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["audit", "study", "register", "report"])
    ap.add_argument("--out", default=f"reports/v0.7_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    a = ap.parse_args()
    {"study": study, "register": register, "audit": audit}.get(a.step, lambda o: __import__("scripts.v07_report", fromlist=["x"]).write_report(o))(Path(a.out))


if __name__ == "__main__":
    main()
