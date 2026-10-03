"""AuTrad v0.4: run the pre-registered factor and event study (research/preregistration_v0.4.md).

    python -m scripts.factors --data-dir data/snap_20261001 --facts data/pit/nse/facts.csv
Writes reports/v0.4_<ts>/{report.md, results.json, factor_ic.csv, events.csv} and logs every factor as a trial.
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

from execution.costs import COST_MODEL_VERSION
from lab import engine
from lab.factors import (
    ALL_FACTORS,
    FactorContext,
    all_scores,
    decision_dates,
    earnings_events,
    evaluate_factor,
    forward_returns,
    holm,
    passes,
    universe_targets,
)
from lab.panel import build_panel, india_equity_assets
from lab.registry import Registry, git_sha
from marketdata.corporate_actions import TRANSFORMS_VERSION
from marketdata.exchange import load as load_bse
from marketdata.exchange import series
from marketdata.fundamentals import PITFundamentals, load_facts, split_events_from_store
from marketdata.pit import SECTOR, nifty50_assets
from marketdata.store import DataStore

PREREG = Path("research/preregistration_v0.4.md")
ROUND_TRIP_BPS = 29.0                     # NSE large-cap delivery round trip at Rs1L (execution.costs)


def f(x, fmt="{:.3f}"):
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else fmt.format(x)


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items() if not str(k).startswith("_")}
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


def event_study(ev: pd.DataFrame, P, horizons=(5, 20, 60)) -> dict:
    """Drift after the reaction session for point-in-time members, by reaction tercile. Benchmark = equal-weight index
    members over the SAME window (next open to close of day i+h). Terciles use past-only thresholds (expanding
    quantiles of earlier reactions). t-stats are clustered by reaction day."""
    from lab.factors import stock_universe

    C = P.ffill_close()
    rows = []
    ev = ev.sort_values("reaction_idx").reset_index(drop=True)
    past: list[float] = []
    for r in ev.itertuples(index=False):
        j, i = P.idx[r.symbol], r.reaction_idx
        if len(past) >= 200:                                  # past-only tercile thresholds
            lo, hi = np.quantile(past, [1 / 3, 2 / 3])
            terc = "good" if r.ear > hi else ("bad" if r.ear < lo else "neutral")
        else:
            terc = None
        past.append(r.ear)
        if terc is None or i + 1 >= len(P.dates):
            continue
        o = P.O[i + 1, j]
        if not np.isfinite(o) or o <= 0:
            continue
        js = [k for k in stock_universe(P, i) if np.isfinite(P.O[i + 1, k]) and P.O[i + 1, k] > 0]
        rec = {"symbol": r.symbol, "reaction_day": r.reaction_day, "ear": r.ear, "tercile": terc}
        for h in horizons:
            k = i + h
            if k < len(P.dates) and np.isfinite(C[k, j]):
                bench = np.log(np.nanmean(C[k, js] / P.O[i + 1, js]))
                rec[f"car{h}"] = np.log(C[k, j] / o) - bench
        rows.append(rec)
    E = pd.DataFrame(rows)
    if E.empty:
        return {"events": 0}
    out = {"events": int(len(E)), "symbols": int(E["symbol"].nunique()), "start": str(E["reaction_day"].min().date()),
           "end": str(E["reaction_day"].max().date()), "excluded_nonmember": ev.attrs.get("excluded_nonmember"),
           "excluded_stale": ev.attrs.get("excluded_stale"), "by_tercile": {}, "_frame": E}

    def clustered(x: pd.Series, days: pd.Series) -> tuple[float, float]:
        d = x.groupby(days).mean()
        return float(x.mean()), float(d.mean() / d.std() * np.sqrt(len(d))) if len(d) > 2 and d.std() > 0 else float("nan")

    for h in horizons:
        col = f"car{h}"
        for tname, g in E.groupby("tercile"):
            x = g[col].dropna()
            m, t = clustered(x, g.loc[x.index, "reaction_day"])
            out["by_tercile"].setdefault(str(tname), {})[f"{h}d"] = {"n": int(len(x)), "mean_bps": m * 1e4, "t": t,
                                                                    "hit_rate": float((x > 0).mean())}
        sub = E.dropna(subset=[col])
        sign = sub["tercile"].map({"good": 0.5, "bad": -0.5, "neutral": 0.0})
        daily = (sub[col] * sign).groupby(sub["reaction_day"]).sum()
        g_m, b_m = sub.loc[sub.tercile == "good", col].mean(), sub.loc[sub.tercile == "bad", col].mean()
        out.setdefault("spread", {})[f"{h}d"] = {"gross_bps": float((g_m - b_m) * 1e4),
                                                 "t": float(daily.mean() / daily.std() * np.sqrt(len(daily))) if daily.std() > 0 else float("nan"),
                                                 "net_long_good_bps": float(g_m * 1e4 - ROUND_TRIP_BPS),
                                                 "net_long_neutral_bps": float(sub.loc[sub.tercile == "neutral", col].mean() * 1e4 - ROUND_TRIP_BPS)}
    return out


def robustness(P, scores: pd.DataFrame, dates: list[int], name: str) -> list[dict]:
    """Checks of a passing factor (not alternative strategies): later entry, volatile months removed, and for REV1 a
    score that ends one session earlier (cannot be a single bad closing print)."""
    from lab.factors import ic_series, newey_west_t, stock_universe

    C = P.ffill_close()

    def fwd(lag: int) -> pd.DataFrame:
        rows = {}
        for a_, b_ in zip(dates[:-1], dates[1:], strict=True):
            if a_ + 1 + lag < len(P.dates):
                rows[P.dates[a_]] = pd.Series(C[b_] / P.O[a_ + 1 + lag] - 1, index=P.symbols)
        return pd.DataFrame(rows).T

    def row(label, sc, fw):
        ic = ic_series(sc, fw)
        return {"variant": label, "mean_ic": float(ic.mean()), "nw_t": newey_west_t(ic.to_numpy()), "months": int(len(ic)),
                "median_ic": float(ic.median())}

    f0 = fwd(0)
    out = [row("as registered", scores, f0), row("enter one session later", scores, fwd(1)), row("enter five sessions later", scores, fwd(5))]
    mk = pd.Series({P.dates[a_]: abs(np.log(P.bench.iloc[b_] / P.bench.iloc[a_])) for a_, b_ in zip(dates[:-1], dates[1:], strict=True)})
    calm = mk[mk < mk.quantile(0.9)].index
    out.append(row("excluding the 10% most volatile market months", scores.loc[scores.index.intersection(calm)], f0))
    if name == "REV1":
        sk = {}
        for t in dates:
            s_ = {}
            for j in stock_universe(P, t):
                a0, b0 = C[t - 21, j], C[t - 1, j]
                if np.isfinite(a0) and np.isfinite(b0):
                    s_[P.symbols[j]] = -(b0 / a0 - 1)
            sk[P.dates[t]] = pd.Series(s_)
        out.append(row("score ends at the previous close (one session earlier)", pd.DataFrame(sk).T.reindex(columns=P.symbols), f0))
    return out


def attribution(sleeves: pd.DataFrame, factors: dict[str, pd.Series], mkt: pd.Series, cash_rate: float = 0.06) -> dict:
    """Monthly excess returns of each sleeve on market + factor long-short returns (OLS)."""
    me = lambda s: s.resample("ME").last().pct_change().dropna()  # noqa: E731
    rf_m = (1 + cash_rate) ** (1 / 12) - 1
    X = pd.DataFrame({"MKT": me(mkt) - rf_m, **{k: me(v) - rf_m for k, v in factors.items()}}).dropna()
    out = {}
    for name in sleeves.columns:
        y = (me(sleeves[name].dropna()) - rf_m).reindex(X.index).dropna()
        if len(y) < 24:
            continue
        Xm = X.loc[y.index]
        A = np.column_stack([np.ones(len(Xm)), Xm.to_numpy()])
        beta, *_ = np.linalg.lstsq(A, y.to_numpy(), rcond=None)
        resid = y.to_numpy() - A @ beta
        s2 = resid @ resid / (len(y) - A.shape[1])
        se = np.sqrt(np.diag(s2 * np.linalg.inv(A.T @ A)))
        r2 = 1 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))
        out[name] = {"months": int(len(y)), "alpha_ann": float(beta[0] * 12), "alpha_t": float(beta[0] / se[0]),
                     "r2": float(r2), "loadings": {c: {"beta": float(b), "t": float(b / s)} for c, b, s in zip(Xm.columns, beta[1:], se[1:], strict=True)}}
    return {"window": f"{X.index[0].date()} .. {X.index[-1].date()}" if len(X) else "n/a", "sleeves": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/snap_20261001")
    ap.add_argument("--facts", default="data/pit/nse/facts.csv")
    ap.add_argument("--sleeves", default=None, help="navs.csv of a v0.3.x lab run for attribution")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--registry", default="research/trials.jsonl")
    ap.add_argument("--label", default="v0.4")
    ap.add_argument("--compare", default=None, help="results.json of an earlier run to show side by side")
    a = ap.parse_args()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(a.out) / f"{a.label}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    prereg_sha = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    reg = Registry(a.registry)
    logged = [r for r in (json.loads(x) for x in open(a.registry)) if r.get("kind") == "preregistration"]
    assert any(r["sha256"] == prereg_sha for r in logged), "pre-registration changed after it was logged"

    store = DataStore(a.data_dir)
    P = build_panel(a.data_dir, india_equity_assets(), name="india_pit")
    stocks = nifty50_assets()
    F = build_panel(a.data_dir, stocks, profile_override={s.symbol: "nse_stock_fut" for s in stocks}, name="futures")
    store.load([s for s in P.symbols])                               # populates store.events (verified splits/bonuses)
    cm = pd.read_csv("marketdata/reference/bse_codes.csv", dtype={"bse_code": str}).set_index("symbol")["bse_code"]
    B = load_bse(Path(a.data_dir).parent / "exchange")
    raw = pd.DataFrame({s: series(B, cm[s])["close"] for s in P.symbols if s in cm.index}).reindex(P.dates).ffill(limit=5)
    fund = None
    n_issues = 0
    cov_note = "no fundamentals file"
    if Path(a.facts).exists():
        sev = split_events_from_store(store)
        facts = load_facts(a.facts, split_events=sev)
        fund = PITFundamentals(facts, sev, SECTOR)
        n_issues = int(len(facts.attrs.get("issues", [])))
        cov_note = (f"{facts['symbol'].nunique()} companies, {len(facts)} filings, periods {facts['period_end'].min().date()} .. "
                    f"{facts['period_end'].max().date()}, sha256 {hashlib.sha256(Path(a.facts).read_bytes()).hexdigest()[:16]}")
    ctx = FactorContext(P, raw, fund, SECTOR)
    dates = decision_dates(P)
    print(f"{len(dates)} decision dates; fundamentals: {cov_note}", flush=True)
    scores = all_scores(ctx, dates)
    fwd = forward_returns(P, dates)
    any_scores = scores["MOM12"]
    bench = engine.run(P, universe_targets(any_scores.dropna(how="all"), P), name="EW universe").nav
    results, pvals = {}, {}
    for k in ALL_FACTORS:
        sc = scores[k].dropna(how="all")
        cov = float(sc.notna().sum(axis=1).mean()) if len(sc) else 0.0
        r = evaluate_factor(k, sc, fwd, P, F, bench)
        r["avg_names_scored"] = cov
        results[k] = r
        if "p" in r:
            pvals[k] = r["p"]
        print(f"{k}: months {r.get('months')} IC {f(r.get('mean_ic'))} t {f(r.get('nw_t'), '{:.2f}')} net excess {f(r.get('net_excess'), '{:+.2%}')}", flush=True)
    hp = holm(pvals)
    for k, r in results.items():
        r["holm_p"] = hp.get(k, float("nan"))
        r["criteria"] = passes(r, r["holm_p"])
        r["decision"] = "CANDIDATE" if r["criteria"] and all(r["criteria"].values()) else "FAIL"
    passing = [k for k, r in results.items() if r["decision"] == "CANDIDATE"]
    composite = None
    if passing:
        ranks = [scores[k].rank(axis=1, pct=True) for k in passing]
        comp = sum(ranks) / len(ranks)
        composite = evaluate_factor("COMPOSITE", comp.dropna(how="all"), fwd, P, F, bench)
        composite["members"] = passing
    rob = {k: robustness(P, scores[k].dropna(how="all"), dates, k) for k in passing}
    ev = earnings_events(ctx)
    es = event_study(ev, P) if len(ev) else {"events": 0}
    att = {}
    if a.sleeves and Path(a.sleeves).exists():
        nv = pd.read_csv(a.sleeves, index_col=0, parse_dates=True)
        sleeves = nv.drop(columns=[c for c in nv.columns if c == "NIFTYBEES_bh"])
        long_f = {k: results[k]["_nav_ls"] for k in ["MOM12", "LVOL"] if "_nav_ls" in results.get(k, {})}
        short_f = {k: results[k]["_nav_ls"] for k in ["MOM12", "LVOL", "EY", "ROE"] if "_nav_ls" in results.get(k, {})}
        att = {"long": attribution(sleeves, long_f, nv["NIFTYBEES_bh"].dropna()),
               "short": attribution(sleeves, short_f, nv["NIFTYBEES_bh"].dropna())}

    # ledger: every factor is a trial; one experiment record each
    n_prior = reg.count()
    reg.add([{"version": a.label, "strategy": f"factor:{k}", "config": {"definition": "preregistered", "prereg_sha256": prereg_sha[:16]}}
             for k in ALL_FACTORS] + ([{"version": a.label, "strategy": "factor:COMPOSITE", "config": {"members": passing}}] if passing else []))
    exp = []
    for k, r in results.items():
        exp.append(clean({"kind": "experiment", "version": a.label, "experiment_id": hashlib.sha1(f"{ts}{k}".encode()).hexdigest()[:12],
                          "git_sha": git_sha(), "dataset_version": Path(a.data_dir).name, "data_hash": store.manifest_hash(),
                          "exchange_hash": store.exchange_hash(), "fundamentals": cov_note, "transforms_version": TRANSFORMS_VERSION,
                          "universe_definition": "point-in-time NIFTY 50", "point_in_time": True, "preregistration_sha256": prereg_sha,
                          "training_period": "none (no fitting)", "validation_period": "none", "test_period": f"{r.get('start')} .. {r.get('end')} (historical evidence)",
                          "features": k, "model": "rank factor, top tercile", "hyperparameters": "none", "cost_model": COST_MODEL_VERSION,
                          "turnover": r.get("turnover_per_year"), "gross_return": r.get("gross_excess"), "net_return": r.get("net_excess"),
                          "sharpe": r.get("info_ratio"), "p_value": r.get("p"), "holm_p": r.get("holm_p"), "mean_ic": r.get("mean_ic"),
                          "excess_t_monthly": r.get("excess_t_monthly"), "number_of_prior_trials": n_prior,
                          "decision": r["decision"], "criteria": r.get("criteria")}))
    reg.add(exp)

    # outputs
    ic_rows = []
    from lab.factors import ic_series
    for k in ALL_FACTORS:
        ic = ic_series(scores[k].dropna(how="all"), fwd)
        ic_rows += [{"factor": k, "date": str(d.date()), "ic": float(v)} for d, v in ic.items()]
    pd.DataFrame(ic_rows).to_csv(out / "factor_ic.csv", index=False)
    if "_frame" in es:
        es["_frame"].to_csv(out / "events.csv", index=False)
    (out / "results.json").write_text(json.dumps(clean({"generated": ts, "prereg_sha256": prereg_sha, "fundamentals": cov_note,
                                                         "decision_dates": len(dates), "factors": results, "holm": hp, "passing": passing,
                                                         "composite": composite, "event_study": es, "attribution": att,
                                                         "robustness": rob, "label": a.label, "data_issues_blanked": n_issues}), indent=1))
    write_report(out, ts, results, passing, composite, es, att, cov_note, prereg_sha, store, rob, a, n_issues)
    print(f"report written to {out}", flush=True)


def _cell(lo: dict, c: str) -> str:
    return f"{f(lo[c]['beta'], '{:+.2f}')} ({f(lo[c]['t'], '{:.1f}')})" if c in lo else "—"


def write_report(out, ts, results, passing, composite, es, att, cov_note, prereg_sha, store, rob, a, n_issues) -> None:
    prev = json.loads(Path(a.compare).read_text()) if a.compare and Path(a.compare).exists() else None
    L = [f"# AuTrad AI {a.label} — factor and event research\n",
         f"Generated {ts} · pre-registration sha256 `{prereg_sha[:16]}` (logged before any result) · prices: exchange-anchored "
         f"(`{TRANSFORMS_VERSION}`, exchange `{store.exchange_hash()}`) · fundamentals: {cov_note}; {n_issues} parse-error values "
         f"blanked by validation · cost model `{COST_MODEL_VERSION}`\n",
         "**Historical evidence, not a sealed test.** Definitions were fixed in advance (`research/preregistration_v0.4.md`), "
         "but the 2024–26 window had already been seen. No factor is traded on this evidence: a passing factor becomes a "
         "candidate for the forward paper test starting 1 Oct 2026.\n"]
    if prev:
        L.append("**This run applies the fixes from an independent review of the first run** (same definitions; data validation, "
                 "publication-keyed split handling, one statement basis per comparison, members-only events with a same-window "
                 "benchmark, past-only event terciles, day-clustered t-stats, compounded gross returns). The first run's numbers are "
                 "shown beside each factor.\n")
    L += ["## Verdict\n",
          (f"**{len(passing)} of {len(results)} pre-registered factors passed every criterion: {', '.join(passing)}.**" if passing else
           f"**None of the {len(results)} pre-registered factors passed every criterion.**")]
    for k in passing:
        r = results[k]
        L.append(f" {k}'s portfolio excess is {f(r.get('net_excess'), '{:+.2%}')} a year after costs, but its monthly t-stat is "
                 f"{f(r.get('excess_t_monthly'), '{:.2f}')}: the ranking signal is significant, the after-cost portfolio gain is not. "
                 "That makes it a weak candidate.")
    L += ["\n## Factor scorecard\n",
          "IC = Spearman rank correlation of the score with the next month's return (next open to month-end close), point-in-time NIFTY 50 "
          "members only. Excess = equal-weight top tercile minus the equal-weight universe, both through the engine with Indian costs; "
          "gross is the same portfolio run with zero costs (compounded). L/S = top minus bottom tercile on stock futures (diagnostic).\n",
          "| Factor | Months | Mean IC | NW t | Holm p | IC + years | IC halves | Gross excess | Cost drag | Net excess | Excess t | Turnover | L/S CAGR | Decision |"
          + (" First run |" if prev else ""),
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|" + ("---|" if prev else "")]
    for k, r in results.items():
        pv = (prev or {}).get("factors", {}).get(k, {}) if prev else {}
        pcell = (f" IC {f(pv.get('mean_ic'))}, t {f(pv.get('nw_t'), '{:.2f}')}, {pv.get('decision', '—')} |" if prev else "")
        if "mean_ic" not in r:
            L.append(f"| {k} | {r.get('months', 0)} | — | — | — | — | — | — | — | — | — | — | — | no data |" + pcell)
            continue
        L.append(f"| {k} | {r['months']} | {f(r['mean_ic'])} | {f(r['nw_t'], '{:.2f}')} | {f(r['holm_p'], '{:.3f}')} | "
                 f"{f(r['pct_years_pos'], '{:.0%}')} | {f(r['ic_first_half'], '{:+.3f}')} / {f(r['ic_second_half'], '{:+.3f}')} | "
                 f"{f(r.get('gross_excess'), '{:+.2%}')} | {f(r.get('cost_drag'), '{:.2%}')} | {f(r.get('net_excess'), '{:+.2%}')} | "
                 f"{f(r.get('excess_t_monthly'), '{:.2f}')} | {f(r.get('turnover_per_year'), '{:.1f}')}× | {f(r.get('ls_cagr'), '{:+.1%}')} | "
                 f"**{r['decision']}** |" + pcell)
    L.append("")
    for k in passing:
        r = results[k]
        L += [f"### {k}: excess return by year (net of costs, log)\n", " · ".join(f"{y} {v:+.1%}" for y, v in r["excess_by_year"].items()) + "\n",
              f"### {k}: robustness (checks of the candidate, not trials)\n", "| Variant | Mean IC | Median IC | NW t | Months |", "|---|---|---|---|---|"]
        L += [f"| {x['variant']} | {f(x['mean_ic'], '{:+.3f}')} | {f(x['median_ic'], '{:+.3f}')} | {f(x['nw_t'], '{:.2f}')} | {x['months']} |" for x in rob.get(k, [])]
        L.append("")
    if composite:
        L += ["## Composite of passing factors (equal-weight ranks, not fitted)\n",
              f"Members: {', '.join(composite['members'])}. With a single member it is that factor; no further test.\n" if len(composite["members"]) == 1 else
              f"Members: {', '.join(composite['members'])}. Mean IC {f(composite.get('mean_ic'))}, NW t {f(composite.get('nw_t'), '{:.2f}')}, "
              f"net excess {f(composite.get('net_excess'), '{:+.2%}')} a year.\n"]
    L += ["## Earnings event study\n"]
    if es.get("events"):
        L.append(f"{es['events']} results announcements by point-in-time NIFTY 50 members ({es['symbols']} companies, {es['start']} .. {es['end']}). "
                 f"Excluded: {es.get('excluded_nonmember')} events by stocks not in the index at the time, and {es.get('excluded_stale')} with stale "
                 "timestamps (more than 60 days after the quarter; re-filings). The reaction is the market-adjusted return on the first session "
                 "that could trade the filing. Drift runs from the next open, against equal-weight index members over the same window. "
                 "Terciles use only earlier reactions as thresholds, and t-stats are clustered by day.\n")
        L += ["| Reaction tercile | +5 sessions | +20 sessions | +60 sessions |", "|---|---|---|---|"]
        for tname in ["good", "neutral", "bad"]:
            d = es["by_tercile"].get(tname, {})
            L.append(f"| {tname} | " + " | ".join(f"{f(d.get(h, {}).get('mean_bps'), '{:+.0f}')} bps (t {f(d.get(h, {}).get('t'), '{:.1f}')})"
                                                   for h in ["5d", "20d", "60d"]) + " |")
        L.append("| good − bad | " + " | ".join(f"{f(es['spread'][h]['gross_bps'], '{:+.0f}')} bps (t {f(es['spread'][h]['t'], '{:.1f}')})"
                                               for h in ["5d", "20d", "60d"]) + " |")
        L.append("\nLong-only after a ~29 bps round trip: good tercile " + ", ".join(f"{h} {f(es['spread'][h]['net_long_good_bps'], '{:+.0f}')} bps"
                                                                                      for h in ["5d", "20d", "60d"])
                 + "; neutral tercile " + ", ".join(f"{h} {f(es['spread'][h]['net_long_neutral_bps'], '{:+.0f}')} bps" for h in ["5d", "20d", "60d"]) + ".\n")
    for key, title in (("long", "market, MOM12 and LVOL over the full sample"), ("short", "market, MOM12, LVOL, EY and ROE (short window: ROE data)")):
        A_ = att.get(key, {}) if isinstance(att, dict) else {}
        if A_.get("sleeves"):
            cols = ["MKT", "MOM12", "LVOL"] + (["EY", "ROE"] if key == "short" else [])
            L += [f"## Attribution of v0.3 sleeves: {title}\n", f"Window {A_['window']}. Monthly excess returns; OLS t-stats in brackets.\n",
                  "| Sleeve | Months | Alpha / yr | Alpha t | R² | " + " | ".join(cols) + " |", "|---|---|---|---|---|" + "---|" * len(cols)]
            for k, v in A_["sleeves"].items():
                L.append(f"| {k} | {v['months']} | {f(v['alpha_ann'], '{:+.1%}')} | {f(v['alpha_t'], '{:.1f}')} | {f(v['r2'], '{:.2f}')} | "
                         + " | ".join(_cell(v["loadings"], c) for c in cols) + " |")
            L.append("")
    L += ["## Known limitations\n",
          "- Machine-readable results start in mid-2018, so fundamental factors cover about seven years; price factors cover ten.",
          "- About 50 stocks per month: terciles hold about 16 names, so tracking error is high and power is limited.",
          "- No analyst consensus: SUE uses a seasonal random walk (same quarter last year) as the expectation.",
          "- Book equity is reported half-yearly, so BM, ROE and LEV update twice a year and cover fewer months.",
          "- Validation blanks values it cannot trust (it never fills them). Companies whose share count changed through rights "
          "issues or recapitalisation lose some EPS quarters.",
          "- Taxes on gains are not deducted.\n"]
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
