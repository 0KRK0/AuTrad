"""AuTrad v0.6 (research/preregistration_v0.6.md): exchange-anchored opens, tax, and REV1 x bad-earnings-reaction veto.

    python -m scripts.v06 run --out reports/v0.6_<ts>        # < 3 minutes on a laptop
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from accounting.tax_india import post_tax_nav
from lab import engine
from lab import v05 as V
from lab.factors import FactorContext, decision_dates, earnings_events, price_scores, stock_universe, tercile_targets, universe_targets
from lab.panel import build_panel, india_equity_assets
from scripts.v05 import clean, mean_t

DATA = "data/snap_20261001"
PREREG = Path("research/preregistration_v0.6.md")
RF_M = 1.06 ** (1 / 12) - 1


def check_prereg() -> str:
    sha = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    logged = [json.loads(x) for x in open("research/trials.jsonl") if '"preregistration"' in x]
    assert any(r.get("sha256") == sha and r.get("version") == "v0.6" for r in logged), "v0.6 pre-registration changed after it was logged"
    return sha


def bad_events(P) -> pd.DataFrame:
    from marketdata.fundamentals import PITFundamentals, load_facts, split_events_from_store
    from marketdata.pit import SECTOR
    from marketdata.store import DataStore

    st = DataStore(DATA)
    st.load(list(P.symbols))
    sev = split_events_from_store(st)
    fund = PITFundamentals(load_facts("data/pit/nse/facts.csv", split_events=sev), sev, SECTOR)
    ev = earnings_events(FactorContext(P, pd.DataFrame(index=P.dates), fund, SECTOR)).reset_index(drop=True)
    order = ev["reaction_idx"] + np.arange(len(ev)) * 1e-6
    ev["bad"] = V.past_only_flag(ev["ear"], order, 1 / 3)
    return ev


def variants(S: pd.DataFrame, P, ev: pd.DataFrame, window: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """REPLACE and SKIP targets, the number of REV1 names vetoed per month, and the BAD flag matrix (dates x symbols)."""
    bad = ev[ev["bad"] == 1]
    rep, skp, nveto, flags = {}, {}, {}, {}
    for d, s in S.iterrows():
        t = P.dates.get_loc(d)
        b = set(bad.loc[(bad["reaction_idx"] <= t) & (bad["reaction_idx"] >= t - window + 1), "symbol"])
        s = s.dropna()
        flags[d] = pd.Series({k: float(k in b) for k in s.index})
        if len(s) < 15:
            continue
        n = len(s) // 3
        top = s.nlargest(n).index
        keep = [k for k in top if k not in b]
        nveto[d] = n - len(keep)
        ranked = [k for k in s.sort_values(ascending=False, kind="stable").index if k not in b][:n]
        w = pd.Series(0.0, index=S.columns)
        w[ranked] = 1.0 / n
        rep[d] = w
        w2 = pd.Series(0.0, index=S.columns)
        if keep:
            w2[keep] = 1.0 / len(keep)
        skp[d] = w2
    return pd.DataFrame(rep).T, pd.DataFrame(skp).T, pd.Series(nveto), pd.DataFrame(flags).T.reindex(columns=S.columns)


def fama_macbeth(S: pd.DataFrame, flags: pd.DataFrame, fwd: pd.DataFrame, start: pd.Timestamp) -> dict:
    coefs = {}
    for d in S.index:
        if d < start or d not in fwd.index:
            continue
        s, r, b = S.loc[d], fwd.loc[d], flags.loc[d]
        k = s.notna() & r.notna() & b.notna()
        if k.sum() < 15 or b[k].sum() == 0:
            continue
        rank = s[k].rank(pct=True).to_numpy()
        X = np.column_stack([np.ones(k.sum()), rank, b[k].to_numpy()])
        beta, *_ = np.linalg.lstsq(X, r[k].to_numpy(), rcond=None)
        coefs[d] = beta[2]
    c = pd.Series(coefs)
    nw = V.nw_ols(c.to_numpy(), np.zeros((len(c), 0)))
    t = float(nw["t"][0])
    return {"months": int(len(c)), "mean_coef_bps": float(c.mean() * 1e4), "t": t, "p_one_sided": float(V.sps.t.cdf(t, df=len(c) - 1)),
            "pct_months_negative": float((c < 0).mean())}


def run(out: Path) -> None:
    sha = check_prereg()
    P = build_panel(DATA, india_equity_assets(), name="india_pit_xopen", open_source="exchange")
    Pv = build_panel(DATA, india_equity_assets(), name="india_pit_vendor")
    C = P.ffill_close()
    R = np.vstack([np.full((1, C.shape[1]), np.nan), np.diff(np.log(C), axis=0)])
    m = np.r_[np.nan, np.diff(np.log(P.bench.ffill().to_numpy()))]
    dates = decision_dates(P)
    S = pd.DataFrame({P.dates[t]: price_scores(P, t, stock_universe(P, t), C, R, m)["REV1"] for t in dates}).T
    S = S.reindex(columns=P.symbols).dropna(how="all")
    ev = bad_events(P)
    first_bad = P.dates[int(ev.loc[ev["bad"] == 1, "reaction_idx"].min())]
    start = S.index[S.index >= first_bad][0]
    tg_rev1 = tercile_targets(S)
    tg_rep, tg_skp, nveto, flags = variants(S, P, ev, 20)
    tg_rep60, _, nveto60, _ = variants(S, P, ev, 60)
    uni = universe_targets(S, P)
    res: dict = {"prereg_sha256": sha, "open_stats": P.open_stats, "active_from": str(start.date()),
                 "events": int(len(ev)), "bad_events": int((ev["bad"] == 1).sum())}

    # ---------------- execution-price sensitivity of the frozen REV1 (full sample)
    def excess(Pp, tg, cost=1.0, capital=10_00_000):
        nt = engine.run(Pp, tg, cost_mult=cost, capital=capital).nav
        nb = engine.run(Pp, uni, capital=capital).nav
        st_ = tg.index[0]
        nt, nb = nt[nt.index >= st_], nb[nb.index >= st_]
        yrs = (nt.index[-1] - st_).days / 365.25
        return float((nt.iloc[-1] / nt.iloc[0]) ** (1 / yrs) - (nb.iloc[-1] / nb.iloc[0]) ** (1 / yrs))
    res["execution"] = {"REV1_vendor_open": excess(Pv, tg_rev1), "REV1_exchange_open": excess(P, tg_rev1),
                        "REV1_exchange_open_2x": excess(P, tg_rev1, 2.0)}

    # ---------------- variant runs (exchange opens)
    runs, fills = {}, {}
    for name, tg in (("REV1", tg_rev1), ("REPLACE", tg_rep), ("SKIP", tg_skp), ("REPLACE60", tg_rep60), ("EW", uni)):
        fl: list = []
        runs[name] = {1.0: engine.run(P, tg, fills=fl, name=name)}
        fills[name] = fl
        if name != "EW":
            runs[name][2.0] = engine.run(P, tg, cost_mult=2.0, name=name)
            runs[name]["1Cr"] = engine.run(P, tg, capital=1e7, name=name)
    nb_t = pd.DataFrame([pd.Series({"NIFTYBEES": 1.0})], index=[start]).reindex(columns=P.symbols).fillna(0.0)
    flnb: list = []
    runs["NIFTYBEES"] = {1.0: engine.run(P, nb_t, fills=flnb, name="NIFTYBEES")}
    fills["NIFTYBEES"] = flnb

    def win(nav):
        return nav[nav.index >= start]

    def cagr(nav):
        n = win(nav)
        return float((n.iloc[-1] / n.iloc[0]) ** (365.25 / (n.index[-1] - n.index[0]).days) - 1)

    mon = {k: V.monthly(win(v[1.0].nav)) for k, v in runs.items()}
    stats = {}
    for k in ("REV1", "REPLACE", "SKIP", "REPLACE60", "EW", "NIFTYBEES"):
        n = win(runs[k][1.0].nav)
        r = n.pct_change().dropna()
        dd = float((n / n.cummax() - 1).min())
        traded = sum(f["value"] for f in fills[k] if P.dates[f["t"]] > start + pd.Timedelta(days=5))
        st_ = {"cagr": cagr(runs[k][1.0].nav), "sharpe": float((r.mean() - (1.06 ** (1 / 252) - 1)) / r.std() * np.sqrt(252)),
               "max_dd": dd, "turnover_per_year": float(traded / n.mean() / ((n.index[-1] - n.index[0]).days / 365.25))}
        if k not in ("EW", "NIFTYBEES"):
            st_ |= {"excess_vs_EW": st_["cagr"] - cagr(runs["EW"][1.0].nav),
                    "excess_vs_EW_2x": cagr(runs[k][2.0].nav) - cagr(runs["EW"][1.0].nav),
                    "excess_vs_EW_1Cr": cagr(runs[k]["1Cr"].nav) - cagr(runs["EW"][1.0].nav)}
        # tax (FIFO lots from the fills; open lots liquidated at the end)
        end_px = {s: float(P.ffill_close()[-1, P.idx[s]]) for s in P.symbols}
        tx = post_tax_nav(runs[k][1.0].nav, fills[k], P.dates, end_px, start=start)
        ptn = tx["post_tax_nav"]
        yrs = (ptn.index[-1] - ptn.index[0]).days / 365.25
        st_ |= {"post_tax_cagr_liquidated": float((tx["liquidation_value"] / ptn.iloc[0]) ** (1 / yrs) - 1),
                "tax_paid_along_the_way_pct_of_start": float(tx["tax_paid_along_the_way"] / ptn.iloc[0]),
                "tax_by_fy": {int(fy): float(v) for fy, v in tx["tax_by_fy"]["tax"].items()} if len(tx["tax_by_fy"]) else {}}
        stats[k] = st_
    for k in ("REV1", "REPLACE", "SKIP", "REPLACE60"):
        stats[k]["post_tax_excess_vs_NIFTYBEES"] = stats[k]["post_tax_cagr_liquidated"] - stats["NIFTYBEES"]["post_tax_cagr_liquidated"]
        stats[k]["TAX_NEGATIVE"] = stats[k]["post_tax_excess_vs_NIFTYBEES"] <= 0
    res["stats"] = stats
    res["vetoed_per_month"] = {"BAD20_mean": float(nveto[nveto.index >= start].mean()), "BAD20_months_with_veto": int((nveto[nveto.index >= start] > 0).sum()),
                               "BAD20_total": int(nveto[nveto.index >= start].sum()), "BAD60_mean": float(nveto60[nveto60.index >= start].mean()),
                               "months": int((nveto.index >= start).sum())}

    # ---------------- hypothesis tests
    from lab.factors import forward_returns
    fwd = forward_returns(P, dates)
    tests = {"H1": fama_macbeth(S, flags, fwd, start)}
    for hid, k in (("H2", "REPLACE"), ("H3", "SKIP")):
        d = (mon[k] - mon["REV1"]).dropna()
        mt = mean_t(d)
        tests[hid] = {"variant": k, "months": mt["months"], "mean_diff_ann": mt["ann"], "t": mt["t"],
                      "p_one_sided": float(1 - V.sps.t.cdf(mt["t"], df=mt["months"] - 1)),
                      "subperiods": {f"{a}..{b}": float(d[a:b].sum()) for a, b in V.SUBPERIODS},
                      "pct_months_positive": float((d > 0).mean()), "pct_months_zero": float((d.abs() < 1e-12).mean())}
    d60 = (mon["REPLACE60"] - mon["REV1"]).dropna()
    res["sensitivity_BAD60_replace"] = mean_t(d60)
    hp = V.holm({"H1": tests["H1"]["p_one_sided"], "H2": tests["H2"]["p_one_sided"], "H3": tests["H3"]["p_one_sided"]})
    for k in tests:
        tests[k]["holm_p"] = hp[k]
    res["tests"] = tests
    sp = tests["H2"]["subperiods"]
    crit = {"1 H1 or H2 significant after Holm (right sign)": (hp["H1"] < 0.05 and tests["H1"]["mean_coef_bps"] < 0) or (hp["H2"] < 0.05 and tests["H2"]["mean_diff_ann"] > 0),
            "2 REPLACE beats REV1 vs EW at 1x and is > 0 at 2x costs": stats["REPLACE"]["excess_vs_EW"] > stats["REV1"]["excess_vs_EW"] and stats["REPLACE"]["excess_vs_EW_2x"] > 0,
            "3 REPLACE - REV1 > 0 in >= 3 of 4 subperiods": sum(v > 1e-12 for v in sp.values()) >= 3,
            "4 turnover <= 1.1x REV1": stats["REPLACE"]["turnover_per_year"] <= 1.1 * stats["REV1"]["turnover_per_year"]}
    res["criteria"] = crit
    res["decision"] = "REV2 CANDIDATE (needs its own forward pre-registration)" if all(crit.values()) else "NOT A CANDIDATE"
    res["scorecard"] = {k: V.score_axes(stat_ok=(hp["H2"] < 0.05) if k == "REPLACE" else (hp["H3"] < 0.05) if k == "SKIP" else None,
                                        stat_weak=tests["H2" if k == "REPLACE" else "H3"]["p_one_sided"] < 0.10 if k in ("REPLACE", "SKIP") else None,
                                        econ_value=stats[k]["excess_vs_EW"], econ_kind="portfolio",
                                        tradable={"net_positive": stats[k]["excess_vs_EW"] > 0, "long_only": True, "cost_2x": stats[k]["excess_vs_EW_2x"] > 0,
                                                  "capacity_1Cr": stats[k]["excess_vs_EW_1Cr"] > 0, "after_tax": not stats[k]["TAX_NEGATIVE"]},
                                        forward="REV1 running" if k == "REV1" else "not started") for k in ("REV1", "REPLACE", "SKIP")}
    out.mkdir(parents=True, exist_ok=True)
    (out / "v06.json").write_text(json.dumps(clean(res), indent=1))
    pd.DataFrame({k: v for k, v in mon.items()}).to_csv(out / "v06_monthly.csv")
    print(json.dumps(clean({"execution": res["execution"], "tests": {k: {a: b for a, b in v.items() if a != "subperiods"} for k, v in tests.items()},
                            "criteria": crit, "decision": res["decision"]}), indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run", "report"])
    ap.add_argument("--out", default=f"reports/v0.6_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    a = ap.parse_args()
    if a.step == "run":
        run(Path(a.out))
    else:
        from scripts.v06_report import write_report
        write_report(Path(a.out))


if __name__ == "__main__":
    main()
