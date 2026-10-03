"""AuTrad v0.5 research (research/preregistration_v0.5.md): Track B (low-volatility decomposition) and Track C
(earnings reaction). Track A (the REV1 forward paper test) runs separately: scripts/forward_rev1.py.

    python -m scripts.v05 B --out reports/v0.5_<ts>      # each step stays under ~2 minutes
    python -m scripts.v05 C --out reports/v0.5_<ts>
    python -m scripts.v05 report --out reports/v0.5_<ts>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from lab import engine
from lab import v05 as V
from lab.factors import decision_dates, price_scores, stock_universe, tercile_targets, universe_targets
from lab.panel import build_panel, india_equity_assets
from marketdata.pit import nifty50_assets

DATA = "data/snap_20261001"
PREREG = Path("research/preregistration_v0.5.md")
RF_M = (1.06) ** (1 / 12) - 1


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    return o


def check_prereg() -> str:
    sha = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    logged = [json.loads(x) for x in open("research/trials.jsonl") if '"preregistration"' in x]
    assert any(r.get("sha256") == sha and r.get("version") == "v0.5" for r in logged), "v0.5 pre-registration changed after it was logged"
    return sha


def panels():
    P = build_panel(DATA, india_equity_assets(), name="india_pit")
    stocks = nifty50_assets()
    F = build_panel(DATA, stocks, profile_override={s.symbol: "nse_stock_fut" for s in stocks}, name="futures")
    return P, F


def arrays(P):
    C = P.ffill_close()
    R = np.vstack([np.full((1, C.shape[1]), np.nan), np.diff(np.log(C), axis=0)])
    m = np.r_[np.nan, np.diff(np.log(P.bench.ffill().to_numpy()))]
    return C, R, m


def regress(y: pd.Series, X: pd.DataFrame, lags: int = 3) -> dict:
    d = pd.concat([y.rename("y"), X], axis=1).dropna()
    r = V.nw_ols(d["y"].to_numpy(), d[X.columns].to_numpy(), lags)
    out = {"months": int(r["n"]), "alpha_ann": float(r["beta"][0] * 12), "alpha_t": float(r["t"][0]), "r2": r["r2"],
           "loadings": {c: {"beta": float(b), "t": float(t)} for c, b, t in zip(X.columns, r["beta"][1:], r["t"][1:], strict=True)},
           "waterfall_ann": {"total_over_cash": float(d["y"].mean() * 12), "alpha": float(r["beta"][0] * 12),
                             **{c: float(b * d[c].mean() * 12) for c, b in zip(X.columns, r["beta"][1:], strict=True)}}}
    return out


def mean_t(x: pd.Series) -> dict:
    x = x.dropna()
    if len(x) < 6:
        return {"ann": float(x.mean() * 12) if len(x) else None, "t": None, "months": int(len(x))}
    r = V.nw_ols(x.to_numpy(), np.zeros((len(x), 0)))
    return {"ann": float(x.mean() * 12), "t": float(r["t"][0]), "months": int(len(x))}


# =============================================================================== Track B
def track_b(out: Path) -> None:
    from lab.strategies import low_vol

    sha = check_prereg()
    P, F = panels()
    C, R, m = arrays(P)
    me = decision_dates(P)
    sc = {"LVOL": {}, "MOM12": {}, "SMB": {}}
    for t in me:
        js = stock_universe(P, t)
        ps = price_scores(P, t, js, C, R, m)
        sc["LVOL"][P.dates[t]], sc["MOM12"][P.dates[t]] = ps["LVOL"], ps["MOM12"]
        tv = pd.Series({P.symbols[j]: np.nanmedian(P.value[t - 62:t + 1, j]) for j in js})
        sc["SMB"][P.dates[t]] = -np.log(tv[tv > 0])
    S = {k: pd.DataFrame(v).T.reindex(columns=P.symbols).dropna(how="all") for k, v in sc.items()}

    objects = {"low_vol_sleeve": low_vol(P, top_n=15, market_filter=False), "LVOL_tercile": tercile_targets(S["LVOL"]),
               "sector_neutral_low_vol": V.sector_neutral_low_vol(P)}
    first = min(tg.index[0] for tg in objects.values())
    nb_t = pd.DataFrame([pd.Series({"NIFTYBEES": 1.0})], index=[first]).reindex(columns=P.symbols).fillna(0.0)
    navs = {"NIFTYBEES": engine.run(P, nb_t, name="NIFTYBEES").nav,
            "EW_universe": engine.run(P, universe_targets(S["MOM12"], P), name="EW").nav}
    ls = {k: engine.run(F, tercile_targets(S[k], long_short=True), name=f"{k}_ls").nav for k in ["MOM12", "LVOL", "SMB"]}
    runs = {}
    for k, tg in objects.items():
        runs[k] = {f"cost{c}": engine.run(P, tg, cost_mult=c, name=k) for c in (0.0, 2.0, 3.0)}
        fl: list = []
        runs[k]["cost1.0"] = engine.run(P, tg, cost_mult=1.0, name=k, fills=fl)
        runs[k]["_fills"] = fl
        runs[k] |= {f"cap{int(c)}": engine.run(P, tg, capital=c, name=k) for c in (1e7, 5e7)}
        runs[k] |= {f"cap{int(c)}_gross": engine.run(P, tg, capital=c, cost_mult=0.0, name=k) for c in (1e7, 5e7)}

    win = lambda s: s[(s.index >= "2016-06-30") & (s.index <= "2026-09-30")]  # noqa: E731
    mo = {k: win(V.monthly(v)) for k, v in (navs | {f"LS_{k}": v for k, v in ls.items()}).items()}
    X = pd.DataFrame({"MKT": mo["NIFTYBEES"] - RF_M, "EWCW": mo["EW_universe"] - mo["NIFTYBEES"], "SMB": mo["LS_SMB"] - RF_M,
                      "MOM": mo["LS_MOM12"] - RF_M, "LVOL": mo["LS_LVOL"] - RF_M})
    flags = V.regime_flags(P)
    res: dict = {"prereg_sha256": sha, "window": "2016-06 .. 2026-09", "factor_means_ann": {c: float(X[c].mean() * 12) for c in X},
                 "factor_corr": X.corr().round(2).to_dict(), "objects": {}}

    # size proxy vs market cap (2019+), Spearman across members, averaged over months
    try:
        from marketdata.fundamentals import PITFundamentals, load_facts, split_events_from_store
        from marketdata.pit import SECTOR
        from marketdata.store import DataStore

        st = DataStore(DATA)
        st.load(list(P.symbols))
        sev = split_events_from_store(st)
        fund = PITFundamentals(load_facts("data/pit/nse/facts.csv", split_events=sev), sev, SECTOR)
        from marketdata.exchange import load as load_bse
        from marketdata.exchange import series
        cm = pd.read_csv("marketdata/reference/bse_codes.csv", dtype={"bse_code": str}).set_index("symbol")["bse_code"]
        B = load_bse(Path(DATA).parent / "exchange")
        raw = pd.DataFrame({s: series(B, cm[s])["close"] for s in P.symbols if s in cm.index}).reindex(P.dates).ffill(limit=5)
        cors = []
        for d in S["SMB"].index[S["SMB"].index >= "2019-06-30"][::3]:
            mc = {s: fund.snapshot(s, d, float(raw.at[d, s])).get("mcap", np.nan) for s in S["SMB"].loc[d].dropna().index if s in raw}
            mc = pd.Series(mc).replace([np.inf, -np.inf], np.nan).dropna()
            mc = mc[mc > 0]
            if len(mc) >= 20:
                cors.append(pd.Series(-np.log(mc)).corr(S["SMB"].loc[d, mc.index], method="spearman"))
        res["size_proxy_vs_mcap_spearman"] = {"mean": float(np.mean(cors)), "min": float(np.min(cors)), "dates": len(cors)}
    except Exception as e:  # pragma: no cover - reported, not hidden
        res["size_proxy_vs_mcap_spearman"] = {"error": repr(e)}

    for k, tg in objects.items():
        r1 = runs[k]["cost1.0"]
        y = win(V.monthly(r1.nav)) - RF_M
        o: dict = {"decisions": int(len(tg)), "first_decision": str(tg.index[0].date())}
        o["M0"] = regress(y, X[["MKT", "MOM", "LVOL"]])
        o["M1"] = regress(y, X[["MKT", "EWCW", "SMB", "MOM"]])
        o["M2"] = regress(y, X[["MKT", "EWCW", "SMB", "MOM", "LVOL"]])
        uni = {}
        for d in tg.index:
            t = P.dates.get_loc(d)
            js = np.nonzero(V.tradable(P, t, {"largecap"}))[0] if k != "LVOL_tercile" else stock_universe(P, t)
            w = pd.Series(0.0, index=P.symbols)
            w.iloc[list(js)] = 1.0 / len(js)
            uni[d] = w
        U = pd.DataFrame(uni).T
        br = V.brinson(P, tg.reindex(columns=P.symbols).fillna(0.0), U)
        o["brinson_ann"] = {"allocation": float(br["allocation"].mean() * 12), "selection": float(br["selection"].mean() * 12),
                            "excess_gross": float(br["excess"].mean() * 12), "periods": int(len(br)),
                            "selection_t": mean_t(br["selection"])["t"], "allocation_t": mean_t(br["allocation"])["t"]}
        o["concentration"] = V.concentration(P, tg.reindex(columns=P.symbols).fillna(0.0), U)
        betas = pd.Series({d: V.ex_ante_beta(P, P.dates.get_loc(d), tg.loc[d]) for d in tg.index})
        o["ex_ante_beta_avg"] = float(betas.mean())

        def bm_excess(nav):
            r = win(V.monthly(nav))
            b = pd.Series({M: betas[betas.index <= P.dates[P.dates.to_period("M") == M.to_period("M")][0]].iloc[-1]
                           for M in r.index if (betas.index <= P.dates[P.dates.to_period("M") == M.to_period("M")][0]).any()})
            return (r - (b * mo["NIFTYBEES"].reindex(b.index) + (1 - b) * RF_M)).dropna()

        exb = bm_excess(r1.nav)
        exe = (win(V.monthly(r1.nav)) - mo["EW_universe"]).dropna()
        exn = (win(V.monthly(r1.nav)) - mo["NIFTYBEES"]).dropna()
        o["net_excess_vs_beta_matched"] = mean_t(exb)
        o["net_excess_vs_EW"] = mean_t(exe)
        o["net_excess_vs_NIFTYBEES"] = mean_t(exn)
        o["realized_beta"] = o["M1"]["loadings"]["MKT"]["beta"]
        o["cost_shock"] = {c: {"vs_beta_matched_ann": float(bm_excess(runs[k][f"cost{c}"].nav).mean() * 12),
                               "vs_EW_ann": float((win(V.monthly(runs[k][f"cost{c}"].nav)) - mo["EW_universe"]).mean() * 12)}
                           for c in (0.0, 1.0, 2.0, 3.0)}
        g0, g1 = runs[k]["cost0.0"].nav, r1.nav
        cagr = lambda n: float((n.iloc[-1] / n[n.index >= tg.index[0]].iloc[0]) ** (365.25 / (n.index[-1] - tg.index[0]).days) - 1)  # noqa: E731
        drag10l = cagr(g0) - cagr(g1)
        live = r1.nav[r1.nav.index >= tg.index[0]]
        traded = sum(f["value"] for f in runs[k]["_fills"] if P.dates[f["t"]] > tg.index[0] + pd.Timedelta(days=5))  # excl. first purchase
        o["costs"] = {"turnover_per_year": float(traded / live.mean() / ((live.index[-1] - live.index[0]).days / 365.25)),
                      "cost_drag_10L": drag10l,
                      "cost_drag_1Cr": cagr(runs[k]["cap10000000_gross"].nav) - cagr(runs[k]["cap10000000"].nav),
                      "cost_drag_5Cr": cagr(runs[k]["cap50000000_gross"].nav) - cagr(runs[k]["cap50000000"].nav)}
        # regimes (state at the last session before the month)
        reg = {}
        for name, col in (("bull", "bull"), ("high_vol", "high_vol")):
            st_ = {}
            for M in exb.index:
                prev = flags[flags.index < M.to_period("M").start_time]
                if len(prev) and prev["valid"].iloc[-1]:
                    st_[M] = bool(prev[col].iloc[-1])
            s_ = pd.Series(st_)
            for v_ in (True, False):
                ix = s_[s_ == v_].index
                reg[f"{name}={v_}"] = {"vs_beta_matched": mean_t(exb.reindex(ix)), "vs_EW": mean_t(exe.reindex(ix))}
        o["regimes"] = reg
        mk = mo["NIFTYBEES"]
        ro = win(V.monthly(r1.nav)).reindex(mk.index)
        o["capture"] = {"up": float(ro[mk > 0].mean() / mk[mk > 0].mean()), "down": float(ro[mk < 0].mean() / mk[mk < 0].mean())}
        worst = mk.nsmallest(10).index
        o["worst_10_nifty_months"] = [{"month": str(M.date())[:7], "object": float(ro[M]), "NIFTYBEES": float(mk[M]),
                                        "EW": float(mo["EW_universe"].get(M, np.nan))} for M in sorted(worst)]
        o["episodes"] = {n_: {"object": float((1 + ro[a:b]).prod() - 1), "NIFTYBEES": float((1 + mk[a:b]).prod() - 1),
                              "EW": float((1 + mo["EW_universe"][a:b]).prod() - 1)} for n_, a, b in V.EPISODES}
        o["subperiods"] = {}
        for a, b in V.SUBPERIODS:
            yy, XX = y[a:b], X[["MKT", "EWCW", "SMB", "MOM"]][a:b]
            o["subperiods"][f"{a}..{b}"] = {"M1_alpha_ann": regress(yy, XX)["alpha_ann"], "M1_alpha_t": regress(yy, XX)["alpha_t"],
                                           "vs_beta_matched_ann": float(exb[a:b].mean() * 12), "vs_EW_ann": float(exe[a:b].mean() * 12)}
        res["objects"][k] = o

    p = res["objects"]["low_vol_sleeve"]
    sn = res["objects"]["sector_neutral_low_vol"]
    crit = {"B1 M1 alpha > 0, NW t >= 2": p["M1"]["alpha_ann"] > 0 and p["M1"]["alpha_t"] >= 2.0,
            "B2 Brinson selection > 0 and sector-neutral M1 alpha > 0": p["brinson_ann"]["selection"] > 0 and sn["M1"]["alpha_ann"] > 0,
            "B3 net excess vs beta-matched > 0": p["net_excess_vs_beta_matched"]["ann"] > 0,
            "B4 M1 alpha > 0 in >= 3 of 4 subperiods": sum(v["M1_alpha_ann"] > 0 for v in p["subperiods"].values()) >= 3,
            "B5 vs beta-matched > 0 at 2x costs": p["cost_shock"][2.0]["vs_beta_matched_ann"] > 0}
    res["criteria"] = crit
    res["decision"] = "SERIOUS CANDIDATE (needs its own forward pre-registration)" if all(crit.values()) else "NOT A CANDIDATE"
    for k, o in res["objects"].items():
        a = o["M1"]
        trad = {"net_positive": o["net_excess_vs_beta_matched"]["ann"] > 0, "long_only": True,
                "cost_2x": o["cost_shock"][2.0]["vs_beta_matched_ann"] > 0,
                "subperiods": sum(v["vs_beta_matched_ann"] > 0 for v in o["subperiods"].values()) >= 3,
                "capacity_1Cr": o["costs"]["cost_drag_1Cr"] <= 1.25 * max(o["costs"]["cost_drag_10L"], 1e-9)}
        o["scorecard"] = V.score_axes(stat_ok=a["alpha_t"] >= 2.0 and a["alpha_ann"] > 0, stat_weak=a["alpha_t"] >= 1.645 and a["alpha_ann"] > 0,
                                      econ_value=o["net_excess_vs_beta_matched"]["ann"], econ_kind="portfolio", tradable=trad,
                                      forward="not started")
        o["tradability_checks"] = trad
    (out / "trackB.json").write_text(json.dumps(clean(res), indent=1))
    pd.concat({"y_" + k: win(V.monthly(runs[k]["cost1.0"].nav)) - RF_M for k in objects} | dict(X), axis=1).to_csv(out / "trackB_monthly.csv")
    print("Track B:", res["decision"], {k: bool(v) for k, v in crit.items()})


# =============================================================================== REV1 research-side checks
def rev1_research(out: Path) -> None:
    """Scorecard inputs for the frozen REV1 candidate on the research data (with the fixed quality gate): net excess
    at 1x and 2x costs and by subperiod. Nothing here feeds the forward test."""
    check_prereg()
    P = build_panel(DATA, india_equity_assets(), name="india_pit")
    C, R, m = arrays(P)
    dates = decision_dates(P)
    S = pd.DataFrame({P.dates[t]: price_scores(P, t, stock_universe(P, t), C, R, m)["REV1"] for t in dates}).T
    S = S.reindex(columns=P.symbols).dropna(how="all")
    tg = tercile_targets(S)
    st = tg.index[0]
    nb = engine.run(P, universe_targets(S, P), name="EW").nav
    nb = nb[nb.index >= st]
    res = {}
    for c in (1.0, 2.0):
        nt = engine.run(P, tg, cost_mult=c, name="REV1").nav
        nt = nt[nt.index >= st]
        ex = np.log(nt).diff().dropna() - np.log(nb).diff().reindex(nt.index).dropna()
        yrs = (nt.index[-1] - st).days / 365.25
        res[f"{c}x"] = {"net_excess": float((nt.iloc[-1] / nt.iloc[0]) ** (1 / yrs) - (nb.iloc[-1] / nb.iloc[0]) ** (1 / yrs)),
                        "subperiods": {f"{a}..{b}": float(ex[a:b].sum()) for a, b in V.SUBPERIODS}}
    (out / "rev1_research.json").write_text(json.dumps(clean(res), indent=1))
    print("REV1:", {k: round(v["net_excess"], 4) for k, v in res.items()})


# =============================================================================== Track C
def track_c(out: Path) -> None:
    from lab.factors import FactorContext, earnings_events
    from marketdata.fundamentals import PITFundamentals, load_facts, split_events_from_store
    from marketdata.pit import SECTOR
    from marketdata.store import DataStore
    from execution.costs import PROFILES

    sha = check_prereg()
    P = build_panel(DATA, india_equity_assets(), name="india_pit")
    st = DataStore(DATA)
    st.load(list(P.symbols))
    sev = split_events_from_store(st)
    fund = PITFundamentals(load_facts("data/pit/nse/facts.csv", split_events=sev), sev, SECTOR)
    ev = earnings_events(FactorContext(P, pd.DataFrame(index=P.dates), fund, SECTOR))
    C = P.ffill_close()
    lb = np.log(P.bench.ffill().to_numpy())
    flags = V.regime_flags(P)
    sec = np.array(P.sector)
    rows = []
    for e in ev.itertuples(index=False):
        i, j = e.reaction_idx, P.idx[e.symbol]
        if i + 1 >= len(P.dates) or i < 130 or not np.isfinite(P.O[i + 1, j]) or P.O[i + 1, j] <= 0:
            continue
        absr = float(np.log(C[i, j] / C[i - 1, j]))
        mem = [k for k in stock_universe(P, i)]
        peers = [k for k in mem if k != j and sec[k] == sec[j] and np.isfinite(C[i, k]) and np.isfinite(C[i - 1, k])]
        secr = absr - float(np.mean([np.log(C[i, k] / C[i - 1, k]) for k in peers])) if len(peers) >= 2 else np.nan
        sn = fund.snapshot(e.symbol, P.dates[i], None)
        sue = sn.get("sue", np.nan) if str(pd.Timestamp(sn.get("last_period", "1900-01-01")).date()) == str(pd.Timestamp(e.period_end).date()) else np.nan
        v_prev = P.V[i - 20:i, j]
        vr = float(P.V[i, j] / np.nanmedian(v_prev)) if np.nanmedian(v_prev) > 0 else np.nan
        pm_all = {k: np.log(C[i - 1, k] / C[i - 127, k]) for k in mem if np.isfinite(C[i - 1, k]) and np.isfinite(C[i - 127, k])}
        pm = pm_all.get(j, np.nan) - np.median(list(pm_all.values())) if j in pm_all else np.nan
        js = [k for k in mem if np.isfinite(P.O[i + 1, k]) and P.O[i + 1, k] > 0]
        rec = {"symbol": e.symbol, "reaction_idx": i, "reaction_day": P.dates[i], "ABS": absr, "MKT": float(e.ear), "SEC": secr,
               "SUE": sue, "VR": vr, "PM": pm, "bull": bool(flags["bull"].iloc[i]) if flags["valid"].iloc[i] else None,
               "high_vol": bool(flags["high_vol"].iloc[i]) if flags["valid"].iloc[i] else None}
        for h in (5, 20, 60):
            k = i + h
            if k < len(P.dates) and np.isfinite(C[k, j]):
                rec[f"car{h}"] = float(np.log(C[k, j] / P.O[i + 1, j]) - np.log(np.nanmean(C[k, js] / P.O[i + 1, js])))
        rows.append(rec)
    E = pd.DataFrame(rows)
    order = E["reaction_idx"] + np.arange(len(E)) * 1e-6
    for mcol in ("ABS", "MKT", "SEC"):
        E[f"bad_{mcol}"] = V.past_only_flag(E[mcol], order, 1 / 3)
    E["VR_high"] = V.past_only_flag(E["VR"], order, 0.5, below=False)
    res: dict = {"prereg_sha256": sha, "events": int(len(E)), "symbols": int(E["symbol"].nunique()),
                 "span": f"{E['reaction_day'].min().date()} .. {E['reaction_day'].max().date()}",
                 "sec_defined": int(E["SEC"].notna().sum()), "sue_defined": int(E["SUE"].notna().sum())}
    tests, pv = {}, {}
    for cid, mcol in (("C1", "MKT"), ("C2", "SEC"), ("C3", "ABS")):
        g = E[E[f"bad_{mcol}"] == 1].dropna(subset=["car20"])
        _, t_day, nd = V.day_clustered_mean(g["car20"], g["reaction_day"])
        cr = V.cluster_ols(g["car20"].to_numpy(), np.zeros((len(g), 0)), g["reaction_day"].to_numpy())
        mean, t = float(cr["beta"][0]), float(cr["t"][0])
        p = float(V.sps.t.cdf(t, df=cr["clusters"] - 1))
        tests[cid] = {"measure": mcol, "n": int(len(g)), "days": nd, "mean_bps": mean * 1e4, "t": t, "t_day_average": t_day, "p_one_sided": p,
                      "by_h": {h: dict(zip(("mean_bps", "t", "days"), (lambda r_: (r_[0] * 1e4, r_[1], r_[2]))(
                          V.day_clustered_mean(E[E[f"bad_{mcol}"] == 1][f"car{h}"].dropna(),
                                               E[E[f"bad_{mcol}"] == 1].dropna(subset=[f"car{h}"])["reaction_day"])), strict=True))
                               for h in (5, 20, 60)}}
        pv[cid] = p
    bad = E[E["bad_MKT"] == 1].dropna(subset=["car20"]).copy()
    bad["SUE_neg"] = np.where(bad["SUE"].notna(), (bad["SUE"] < 0).astype(float), np.nan)
    bad["PM_loser"] = np.where(bad["PM"].notna(), (bad["PM"] <= 0).astype(float), np.nan)
    for cid, col, lab1, lab0 in (("C4", "SUE_neg", "SUE<0", "SUE>=0"), ("C5", "VR_high", "high volume", "low volume"),
                                 ("C6", "PM_loser", "prior loser", "prior winner")):
        g = bad.dropna(subset=[col])
        r = V.cluster_ols(g["car20"].to_numpy(), g[col].to_numpy(), g["reaction_day"].to_numpy())
        tests[cid] = {"split": f"{lab1} minus {lab0}", "n": int(len(g)), "n1": int((g[col] == 1).sum()), "n0": int((g[col] == 0).sum()),
                      "mean1_bps": float(g.loc[g[col] == 1, "car20"].mean() * 1e4), "mean0_bps": float(g.loc[g[col] == 0, "car20"].mean() * 1e4),
                      "diff_bps": float(r["beta"][1] * 1e4), "t": float(r["t"][1]), "p_two_sided": float(r["p"][1]), "clusters": r["clusters"]}
        pv[cid] = float(r["p"][1])
    hp = V.holm(pv)
    for k in tests:
        tests[k]["holm_p"] = hp.get(k)
    res["tests"] = tests
    # descriptive cells (not decision tests)
    desc = {}
    med_day = bad["reaction_day"].median()
    for name, mask in (("first half", bad["reaction_day"] <= med_day), ("second half", bad["reaction_day"] > med_day),
                       ("bull", bad["bull"] == True), ("bear", bad["bull"] == False),  # noqa: E712
                       ("high vol", bad["high_vol"] == True), ("low vol", bad["high_vol"] == False)):  # noqa: E712
        g = bad[mask]
        m_, t_, nd = V.day_clustered_mean(g["car20"], g["reaction_day"])
        desc[name] = {"n": int(len(g)), "mean_bps": m_ * 1e4, "t": t_}
    for y, g in bad.groupby(bad["reaction_day"].dt.year):
        m_, t_, nd = V.day_clustered_mean(g["car20"], g["reaction_day"])
        desc[str(y)] = {"n": int(len(g)), "mean_bps": m_ * 1e4, "t": t_}
    res["bad_MKT_cells"] = desc

    # T1 avoid-overlay vs baseline (engine, full costs, 1% no-trade band on both)
    me = decision_dates(P)
    me_dates = [P.dates[t] for t in me]
    uni = {P.dates[t]: [P.symbols[j] for j in stock_universe(P, t)] for t in me}
    ev_bad = E[(E["bad_MKT"] == 1)][["symbol", "reaction_idx"]]
    excl: list[tuple[str, int, int]] = [(r.symbol, r.reaction_idx, r.reaction_idx + 20) for r in ev_bad.itertuples(index=False)]
    change_days = sorted(set(me) | {a for _, a, _ in excl} | {b for _, _, b in excl if b < len(P.dates)})
    start = me[0]
    rows_b, rows_o = {}, {}
    for t in change_days:
        if t < start:
            continue
        d = P.dates[t]
        cur = [x for x in me_dates if x <= d][-1]
        U = uni[cur]
        w = pd.Series(1.0 / len(U), index=U)
        if t in me:
            rows_b[d] = w
        out_ = {s for s, a, b in excl if a <= t < b}
        wo = w.copy()
        gone = [s for s in out_ if s in wo.index]
        wo[gone] = 0.0
        wo["NIFTYBEES"] = len(gone) / len(U)
        rows_o[d] = wo
    TB = pd.DataFrame(rows_b).T.reindex(columns=P.symbols).fillna(0.0)
    TO = pd.DataFrame(rows_o).T.reindex(columns=P.symbols).fillna(0.0)
    t1 = {}
    for c in (1.0, 2.0):
        nb_ = engine.run(P, TB, cost_mult=c, band=0.01, name="baseline").nav
        no_ = engine.run(P, TO, cost_mult=c, band=0.01, name="overlay").nav
        # identical until the first flagged event: measure from the last month-end before it (active period only)
        first_ev = P.dates[min(a_ for _, a_, _ in excl)]
        d0 = TB.index[TB.index <= first_ev][-1]
        nb_, no_ = nb_[nb_.index >= d0], no_[no_.index >= d0]
        ex = (V.monthly(no_) - V.monthly(nb_)).dropna()
        yrs = (nb_.index[-1] - d0).days / 365.25
        sub_pos = sum(float(ex[a_:b_].sum()) > 0 for a_, b_ in V.SUBPERIODS)
        t1[f"{c}x"] = {"subperiods_positive": int(sub_pos), "overlay_cagr": float((no_.iloc[-1] / no_.iloc[0]) ** (1 / yrs) - 1), "baseline_cagr": float((nb_.iloc[-1] / nb_.iloc[0]) ** (1 / yrs) - 1),
                       "value_ann": float((no_.iloc[-1] / no_.iloc[0]) ** (1 / yrs) - (nb_.iloc[-1] / nb_.iloc[0]) ** (1 / yrs)),
                       "value_t": mean_t(ex)["t"], "by_year": {int(y): float(v) for y, v in ex.groupby(ex.index.year).sum().items()}}
    t1["events_used"] = int(len(excl))
    t1["active_from"] = str(d0.date())
    res["T1_avoid_overlay"] = t1
    fut = PROFILES["nse_stock_fut"]
    rt = fut.round_trip_bps(1_00_000) + 20 * fut.carry_bps_per_day * 7 / 5
    x = -bad["car20"] * 1e4 - rt
    m_, t_, _ = V.day_clustered_mean(x / 1e4, bad["reaction_day"])
    res["T2_short_diagnostic"] = {"futures_round_trip_bps": rt, "net_bps_per_event": m_ * 1e4, "t": t_, "n": int(len(bad)),
                                  "note": "diagnostic only; not a proposal"}
    hp_ = lambda k: tests[k]["holm_p"] if tests[k]["holm_p"] is not None else 1.0  # noqa: E731
    crit = {"C1/C2/C3 significant after Holm": any(hp_(k) < 0.05 and tests[k]["mean_bps"] < 0 for k in ("C1", "C2", "C3")),
            "T1 value > 0 at 1x and 2x costs": t1["1.0x"]["value_ann"] > 0 and t1["2.0x"]["value_ann"] > 0,
            "bad-MKT CAR20 < 0 in both halves": desc["first half"]["mean_bps"] < 0 and desc["second half"]["mean_bps"] < 0,
            "bad-MKT CAR20 < 0 in bull and bear": desc["bull"]["mean_bps"] < 0 and desc["bear"]["mean_bps"] < 0}
    res["criteria"] = crit
    res["decision"] = "CANDIDATE avoid-filter (needs its own forward pre-registration)" if all(crit.values()) else "NOT A CANDIDATE"
    best = min(("C1", "C2", "C3"), key=hp_)
    cost2_ok = t1["2.0x"]["value_ann"] > 0
    res["scorecard"] = V.score_axes(stat_ok=hp_(best) < 0.05, stat_weak=tests[best]["p_one_sided"] < 0.10,
                                    econ_value=t1["1.0x"]["value_ann"], econ_kind="portfolio",
                                    tradable={"net_positive": t1["1.0x"]["value_ann"] > 0, "long_only": True, "cost_2x": cost2_ok,
                                              "subperiods": t1["1.0x"]["subperiods_positive"] >= 3}, forward="not started")
    E.to_csv(out / "trackC_events.csv", index=False)
    (out / "trackC.json").write_text(json.dumps(clean(res), indent=1))
    print("Track C:", res["decision"], {k: bool(v) for k, v in crit.items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["A", "B", "C", "report"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out or f"reports/v0.5_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    out.mkdir(parents=True, exist_ok=True)
    if a.step == "A":
        rev1_research(out)
    elif a.step == "B":
        track_b(out)
    elif a.step == "C":
        track_c(out)
    else:
        from scripts.v05_report import write_report
        write_report(out)


if __name__ == "__main__":
    main()
