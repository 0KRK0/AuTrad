"""AuTrad v0.3 research lab: judge every strategy at every horizon with one frozen judge, then ask whether
ARGO-Meta adds anything on top. Writes reports/v0.3_<timestamp>/{report.md, results.json, navs.csv}.

    python -m scripts.lab --data-dir data/snap_20261001            # full run (~20-30 min on 2 cores)
    python -m scripts.lab --quick                                    # 20 nulls, for a smoke test
    python -m scripts.lab --only index_trend low_vol --no-intraday --no-meta
"""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from execution.costs import COST_MODEL_VERSION
from lab import engine
from lab.intraday import INTRADAY_LIBRARY, VAL_END, judge_intraday, load_intraday
from lab.judge import GATES, JudgeConfig, judge
from lab.meta import judge_meta
from lab.panel import build_panel, cross_asset_assets, india_equity_assets
from lab.registry import Registry, git_sha
from lab.strategies import LIBRARY
from marketdata.corporate_actions import TRANSFORMS_VERSION
from marketdata.pit import nifty50_assets
from marketdata.store import DataStore


def experiment_record(r: dict, label: str, data_dir: str, store, jc, n_prior: int, universe: str) -> dict:
    """One row of the institutional-style experiment ledger (lab.registry.EXPERIMENT_FIELDS)."""
    import uuid

    o, t = r.get("oos", {}), r.get("test", {})
    nul = r.get("nulls", {})
    ps = [nul.get(k) for k in ("p_timing", "p_selection", "p_direction", "p_entry", "p_allocation_oos")]
    ps = [x for x in ps if x is not None and math.isfinite(x)]
    return clean({
        "kind": "experiment", "version": label, "experiment_id": uuid.uuid4().hex[:12], "git_sha": git_sha(),
        "dataset_version": Path(data_dir).name, "data_hash": store.manifest_hash(), "exchange_hash": store.exchange_hash(),
        "transforms_version": TRANSFORMS_VERSION, "universe_definition": universe, "point_in_time": universe != "cross_asset",
        "training_period": f"warm-up .. {jc.val_start}", "validation_period": f"{jc.val_start} .. {jc.test_start}",
        "test_period": f"{jc.test_start} .. end (sealed)", "features": r.get("description"), "model": r["strategy"],
        "hyperparameters": r.get("selected"), "configs_tried": len(r.get("_configs", [])), "cost_model": COST_MODEL_VERSION,
        "slippage_model": "half-spread by liquidity tier + sqrt impact vs day's traded value, participation cap",
        "turnover": o.get("turnover_per_year"), "capacity": {k: v.get("oos_cagr", v.get("test_cagr")) for k, v in r.get("capacity", {}).items()},
        "gross_return": o.get("gross_cagr"), "net_return": o.get("cagr"), "test_net_return": t.get("cagr"),
        "sharpe": o.get("sharpe"), "test_sharpe": t.get("sharpe"), "sortino": o.get("sortino"), "max_drawdown": o.get("max_drawdown"),
        "calmar": o.get("calmar"), "tail_loss": o.get("cvar_95"), "hit_rate": r.get("trades", {}).get("win_rate"),
        "p_value": max(ps) if ps else None, "deflated_sharpe": r["stats"]["dsr"], "pbo": r["stats"]["pbo"].get("pbo"),
        "number_of_prior_trials": n_prior, "decision": r["verdict"], "gates_passed": f"{r['gates_passed']}/{r['gates_total']}",
        "post_test_peek": bool(r.get("post_test_peek", False)),
    })


def bh_nav(P, symbol: str, capital: float, cash_rate: float) -> pd.Series:
    t = pd.DataFrame({symbol: [1.0]}, index=[P.dates[0]]).reindex(columns=P.symbols, fill_value=0.0)
    return engine.run(P, t, capital=capital, cash_rate=cash_rate).nav


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    return o


def f(x, fmt="{:.2f}"):
    try:
        return "n/a" if x is None or not math.isfinite(float(x)) else fmt.format(x)
    except (TypeError, ValueError):
        return str(x)


def pct(x):
    return f(x, "{:+.1%}")


# ----------------------------------------------------------------- report
def write_report(out: Path, res: list[dict], intra: list[dict], meta: dict | None, bench: dict, ctx: dict) -> None:
    L = []
    L.append(f"# AuTrad AI {ctx.get('label', 'v0.3')} — Multi-Horizon Research Lab report\n")
    L.append(f"Generated {ctx['generated']} · data snapshot `{ctx['data_dir']}` (manifest `{ctx['manifest']}`, exchange check `{ctx.get('exchange')}`) · "
             f"cost model `{COST_MODEL_VERSION}` · corporate actions `{TRANSFORMS_VERSION}` · trials in registry: {ctx['trials']}\n")
    if ctx.get("label", "v0.3") != "v0.3":
        L.append("**Re-run on corrected data.** The sealed test window was first opened by v0.3 on 30 Sep 2026. This run changes "
                 "only the price data (exchange-anchored corporate actions); every strategy, grid and gate is unchanged, so the "
                 "differences measure the data fix, not a new discovery.\n")
    L.append("Split: strategy parameters chosen on **2021–2023 only**; **2024 → today is a sealed test** seen once. "
             "\"OOS\" = 2018 onward (includes the selection window, so it is less clean than TEST). "
             "All figures are net of Indian taxes-on-trade, brokerage, spread and impact; Sharpe is measured above a 6% cash rate.\n")
    all_rows = res + intra + ([meta] if meta else [])
    q = [r for r in all_rows if r["verdict"] == "QUALIFIED_FOR_PAPER"]
    L.append("## Verdict\n")
    if q:
        L.append("Qualified for paper trading: " + ", ".join(f"**{r['strategy']}**" for r in q) + ". Nothing qualifies for live money until paper results match.\n")
    else:
        L.append("**No strategy qualified for paper trading.** Every one failed at least one gate. Details below; the closest candidates are at the top of the leaderboard.\n")
    L.append("## Leaderboard (all horizons, one judge)\n")
    L.append("| Strategy | Family | Horizon | Verdict | Gates | Test Sharpe | OOS Sharpe | OOS CAGR | OOS max DD | Null p | DSR | PBO | Sharpe at 0× cost | Sharpe at 2× cost |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(all_rows, key=lambda r: (-r["gates_passed"], -(r["test"].get("sharpe") or -9))):
        sh = r.get("cost_shock", {})
        nul = r["nulls"]
        ps = [nul.get(k) for k in ("p_timing", "p_selection", "p_direction", "p_entry", "p_allocation_oos")]
        ps = [x for x in ps if x is not None and math.isfinite(x)]
        p = max(ps) if ps else float("nan")
        zero = sh.get("0.0", {}).get("oos_sharpe", sh.get("0.0", {}).get("test_sharpe"))
        two = sh.get("2.0", {}).get("oos_sharpe", sh.get("2.0", {}).get("test_sharpe"))
        name = r["strategy"] + (" †" if r.get("post_test_peek") else "")
        L.append(f"| {name} | {r['family']} | {r['horizon']} | {r['verdict']} | {r['gates_passed']}/{r['gates_total']} | "
                 f"{f(r['test'].get('sharpe'))} | {f(r['oos'].get('sharpe'))} | {pct(r['oos'].get('cagr'))} | {f(r['oos'].get('max_drawdown'), '{:.1%}')} | "
                 f"{f(p, '{:.3f}')} | {f(r['stats']['dsr'])} | {f(r['stats']['pbo'].get('pbo'))} | {f(zero)} | {f(two)} |")
    L.append("\n† added after the sealed test had been seen (gold's 2024-26 run was visible in the benchmarks), so its TEST figures are not "
             "an honest out-of-sample result; treat it as a hypothesis for paper trading, not evidence.\n")
    L.append("Null p = the weakest (largest) of the applicable random-strategy tests.\n")
    L.append("\nIntraday rows: \"OOS\" is the 2023-10 → 2025-03 selection window and null/PBO use the test window (hourly data only goes back ~2 years).\n")
    L.append("## Benchmarks\n")
    L.append("| Benchmark | Test Sharpe | Test CAGR | OOS Sharpe | OOS CAGR |")
    L.append("|---|---|---|---|---|")
    for k, b in bench.items():
        L.append(f"| {k} | {f(b['test_sharpe'])} | {pct(b['test_cagr'])} | {f(b['oos_sharpe'])} | {pct(b['oos_cagr'])} |")
    L.append("")

    if meta:
        L.append("## ARGO-Meta (does an allocator on top help?)\n")
        L.append(f"Eligible sleeves (pre-2024 Sharpe > 0, fixed before the test): {', '.join(meta['eligible_sleeves']) or 'none'}. "
                 f"Chosen settings: {meta['selected']}. Average capital deployed 2018+: {meta['avg_allocation_oos']:.0%} (rest in cash).\n")
        L.append("| Portfolio | Test Sharpe | Test CAGR | OOS Sharpe | OOS CAGR | OOS max DD |")
        L.append("|---|---|---|---|---|---|")
        L.append(f"| **ARGO-Meta** | {f(meta['test'].get('sharpe'))} | {pct(meta['test'].get('cagr'))} | {f(meta['oos'].get('sharpe'))} | {pct(meta['oos'].get('cagr'))} | {f(meta['oos'].get('max_drawdown'), '{:.1%}')} |")
        for k, b in meta["baselines"].items():
            L.append(f"| {k} | {f(b['test_sharpe'])} | {pct(b['test_cagr'])} | {f(b['oos_sharpe'])} | {pct(b['oos_cagr'])} | {f(b['oos_max_dd'], '{:.1%}')} |")
        L.append("\n" + gate_table(meta))
        L.append(f"Random-allocation null: median OOS Sharpe {f(meta['nulls']['median_oos'])}, 95th pct {f(meta['nulls']['p95_oos'])}. "
                 f"Latest allocation: {', '.join(f'{k} {v:.0%}' for k, v in meta['latest_allocation'].items()) or '100% cash'}.\n")
        L.append("Yearly: " + ", ".join(f"{y} {pct(v)}" for y, v in meta["yearly"].items()) + "\n")
        L.append("Regimes (2018+): " + "; ".join(f"{k}: {v['days']}d, excess {pct(v['ann_excess'])}/yr, avg deployed {v['avg_allocation']:.0%}" for k, v in meta["regimes"].items()) + "\n")

    L.append("## Strategy detail\n")
    for r in res:
        L.append(f"### {r['strategy']} — {r['family']}, {r['horizon']} — {r['verdict']} ({r['gates_passed']}/{r['gates_total']})\n")
        L.append(f"{r['description']}. Selected on 2021–23: `{r['selected']}`.\n")
        L.append(gate_table(r))
        t, o = r["test"], r["oos"]
        L.append(f"TEST: CAGR {pct(t.get('cagr'))}, vol {f(t.get('volatility'), '{:.1%}')}, max DD {f(t.get('max_drawdown'), '{:.1%}')}. "
                 f"OOS: CAGR {pct(o.get('cagr'))} (gross {pct(o.get('gross_cagr'))}, cost drag {f(o.get('cost_drag_cagr'), '{:.2%}')}/yr), "
                 f"turnover {f(o.get('turnover_per_year'), '{:.1f}')}×/yr, avg exposure {f(o.get('avg_exposure'), '{:.0%}')}. "
                 f"Sharpe 95% CI {f(r['stats']['sharpe_ci'][0])} … {f(r['stats']['sharpe_ci'][1])}.\n")
        tr = r["trades"]
        L.append(f"Trades 2018+: {tr['trades']}, win rate {tr['win_rate']:.0%}, profit factor {f(tr['profit_factor'])}, "
                 f"avg trade {pct(tr['avg_trade_ret'])}, avg hold {tr['avg_hold_days']:.0f} days.\n")
        L.append("Walk-forward (config picked each year from the past only): " + ", ".join(f"{x['year']} {pct(x['return'])}" for x in r["walk_forward"]["folds"]) + "\n")
        L.append("Cost shock (OOS Sharpe): " + ", ".join(f"{k}× {f(v['oos_sharpe'])}" for k, v in r["cost_shock"].items()) + ". "
                 "Capacity (OOS CAGR): " + ", ".join(f"₹{k} {pct(v['oos_cagr'])}" for k, v in r["capacity"].items()) + "\n")
        L.append("Regimes: " + "; ".join(f"{k} {pct(v['ann_excess'])}/yr (Sharpe {f(v['sharpe'])})" for k, v in r["regimes"].items()) + "\n")
        if r.get("survivorship"):
            s = r["survivorship"]
            L.append(f"Survivorship test: today's-members-only universe gives OOS Sharpe {f(s['survivors_oos_sharpe'])} vs point-in-time {f(s['pit_oos_sharpe'])} "
                     f"(bias {f(s['survivors_oos_sharpe'] - s['pit_oos_sharpe'], '{:+.2f}')}).\n")
    if intra:
        L.append("## Intraday lab (hourly bars, MIS costs)\n")
        L.append(f"Selection on 2023-10 → {VAL_END}, sealed test after. Each trade uses 1/5 of capital, no leverage, flat by the close.\n")
        L.append("| Strategy | Selected | Trades (test) | Gross bps/trade | Cost bps/trade | Net bps/trade | Win rate | Test Sharpe | Sharpe at 0× cost |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for r in intra:
            t = r["trades"]
            L.append(f"| {r['strategy']} | `{r['selected']}` | {t.get('trades', 0)} | {f(t.get('avg_gross_bps'), '{:+.1f}')} | {f(t.get('avg_cost_bps'), '{:.1f}')} | "
                     f"{f(t.get('avg_net_bps'), '{:+.1f}')} | {f(t.get('win_rate'), '{:.0%}')} | {f(r['test'].get('sharpe'))} | {f(r['cost_shock']['0.0']['test_sharpe'])} |")
        L.append("")
        for r in intra:
            L.append(f"### {r['strategy']} — {r['verdict']} ({r['gates_passed']}/{r['gates_total']})\n")
            L.append(gate_table(r))
            if r.get("resolution_check"):
                c = r["resolution_check"]
                L.append(f"5-minute resolution check ({c['days']} sessions): net {f(c['avg_net_bps_5m'], '{:+.1f}')} bps/trade on 5-min bars vs "
                         f"{f(c['avg_net_bps_1h'], '{:+.1f}')} on hourly bars over the same days. {c['note']}.\n")
    L.append("## Blocked (not faked)\n")
    for s in LIBRARY:
        if s.status == "BLOCKED":
            L.append(f"- **{s.name}** ({s.family}): {s.blocked_reason}.")
    L.append("\n## Gates (same for every horizon)\n")
    L.append(", ".join(f"{k} = {v}" for k, v in GATES.items()) + ". A strategy is QUALIFIED_FOR_PAPER only if all gates pass; REJECT if it lost to cash in both TEST and OOS.\n")
    L.append("## Known limitations\n")
    for x in ctx["limitations"]:
        L.append(f"- {x}")
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def gate_table(r: dict) -> str:
    rows = ["| Gate | Result | Value | Target |", "|---|---|---|---|"]
    rows += [f"| {g['name']} | {'PASS' if g['passed'] else 'fail'} | {g['value']} | {g['target']} |" for g in r["gates"]]
    return "\n".join(rows) + "\n"


# ----------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/snap_20261001")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--no-intraday", action="store_true")
    ap.add_argument("--no-meta", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--registry", default="research/trials.jsonl")
    ap.add_argument("--label", default="v0.3.1", help="run label; v0.3.1 = v0.3 judge re-run on exchange-corrected data")
    a = ap.parse_args()
    t0 = time.time()
    jc = JudgeConfig(n_null=20 if a.quick else 200, workers=a.workers)
    reg = Registry(a.registry)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(a.out) / f"{a.label}_{ts}"
    out.mkdir(parents=True, exist_ok=True)

    def log(m: str) -> None:
        print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)

    store = DataStore(a.data_dir)
    log("building panels")
    P = build_panel(a.data_dir, india_equity_assets(), name="india_pit")
    S = build_panel(a.data_dir, india_equity_assets(), survivors_only=True, name="india_survivors")
    X = build_panel(a.data_dir, cross_asset_assets(), calendar="union", name="cross_asset")
    stocks = nifty50_assets()
    F = build_panel(a.data_dir, stocks, profile_override={s.symbol: "nse_stock_fut" for s in stocks}, name="pairs_futures")
    panels = {"india": P, "cross_asset": X, "pairs": F}
    bn = bh_nav(P, "NIFTYBEES", jc.capital, jc.cash_rate)
    gb = bh_nav(P, "GOLDBEES", jc.capital, jc.cash_rate)
    from lab.judge import sharpe, window
    from validation.metrics import perf

    def bstats(n):
        return {"test_sharpe": sharpe(window(n, jc.test_start), jc.cash_rate), "oos_sharpe": sharpe(window(n, jc.oos_start), jc.cash_rate),
                "test_cagr": perf(window(n, jc.test_start), jc.cash_rate)["cagr"], "oos_cagr": perf(window(n, jc.oos_start), jc.cash_rate)["cagr"]}
    bench = {"NIFTY ETF buy & hold": bstats(bn), "Gold ETF buy & hold": bstats(gb),
             "Cash (6%)": {"test_sharpe": 0.0, "oos_sharpe": 0.0, "test_cagr": jc.cash_rate, "oos_cagr": jc.cash_rate}}

    results, navs = [], {}
    for spec in LIBRARY:
        if spec.status != "ACTIVE" or (a.only and spec.name not in a.only):
            continue
        log(f"judging {spec.name} ({len(spec.configs())} configs) on {panels[spec.universe].name}")
        n_before = reg.count()
        r = judge(spec, panels[spec.universe], jc, n_before, benchmark_nav=bn,
                  survivors=S if spec.universe == "india" else None, log=log)
        reg.add([{"version": a.label, "strategy": spec.name, "config": c, "val_sharpe": r["val_sharpe"].get(",".join(f"{k}={v}" for k, v in c.items()) or "default")}
                 for c in r["_configs"]])
        reg.add([experiment_record(r, a.label, a.data_dir, store, jc, n_before, spec.universe)])
        log(f"  -> {r['verdict']} {r['gates_passed']}/{r['gates_total']}  test {r['test'].get('sharpe', float('nan')):.2f}  oos {r['oos'].get('sharpe', float('nan')):.2f}")
        results.append(r)
        if not spec.post_test_peek:          # post-peek sleeves would contaminate the meta test
            navs[spec.name] = r["_nav"]

    intra = []
    if not a.no_intraday:
        try:
            D = load_intraday(a.data_dir, "1h")
            D5 = load_intraday(a.data_dir, "5m")
            for spec in INTRADAY_LIBRARY:
                if a.only and spec.name not in a.only:
                    continue
                log(f"judging intraday {spec.name}")
                r = judge_intraday(spec, D, reg.count(), capital=jc.capital, cash_rate=jc.cash_rate,
                                   n_null=20 if a.quick else 100, D5=D5, log=log)
                n_prior = reg.count()
                reg.add([{"version": a.label, "strategy": spec.name, "config": c} for c in r["_configs"]])
                reg.add([experiment_record(r, a.label, a.data_dir, store, jc, n_prior, "intraday_22_liquid")])
                log(f"  -> {r['verdict']} {r['gates_passed']}/{r['gates_total']}  test {r['test'].get('sharpe', float('nan')):.2f}  "
                    f"net {r['trades'].get('avg_net_bps', float('nan')):+.1f} bps/trade")
                intra.append(r)
        except FileNotFoundError as e:
            log(f"intraday skipped: {e}")

    meta = None
    if not a.no_meta and len(navs) >= 2:
        log("judging ARGO-Meta")
        meta = judge_meta(navs, P.dates, P.bench, jc, reg.count(), benchmark_nav=bn, n_null=50 if a.quick else 200, log=log)
        n_prior = reg.count()
        reg.add([{"version": a.label, "strategy": "argo_meta", "config": c} for c in meta["_configs"]])
        reg.add([experiment_record(meta, a.label, a.data_dir, store, jc, n_prior, "sleeves")])
        log(f"  -> {meta['verdict']} {meta['gates_passed']}/{meta['gates_total']}  test {meta['test'].get('sharpe', float('nan')):.2f}")

    ctx = {"generated": ts, "data_dir": a.data_dir, "manifest": store.manifest_hash(), "exchange": store.exchange_hash(), "label": a.label, "trials": reg.count(),
           "limitations": [
               "Point-in-time universe covers NIFTY 50 only (2016-04 onward; earlier dates use the first snapshot). Midcaps/smallcaps have no PIT list yet, so they are not in v0.3.",
               "Daily bars from Yahoo (NSE) re-anchored on BSE's official unadjusted bhavcopy: every split, bonus, rights issue and demerger is checked against the exchange (see reports/data_integrity_*). FX, crypto and index series have no exchange cross-check.",
               "Futures are modelled as the underlying plus roll carry and futures charges; real futures basis and lot sizes are not modelled.",
               "Intraday uses hourly bars with no bid/ask; half-spread and impact are estimates. First-hour bars from Yahoo report zero volume.",
               "Value, quality and event strategies are blocked until point-in-time fundamentals and timestamped news exist.",
               "Taxes on profits (STCG 20%, LTCG 12.5%, crypto 30%) are NOT deducted in these figures; they reduce every row further.",
               "Sleeve configurations were chosen on 2021-2023, so ARGO-Meta's own validation window is optimistic; only its test window is clean.",
           ]}
    write_report(out, results, intra, meta, bench, ctx)
    nav_df = pd.DataFrame({**navs, "NIFTYBEES_bh": bn, **({"argo_meta": meta["_nav"]} if meta else {})})
    nav_df.to_csv(out / "navs.csv", float_format="%.2f")
    (out / "results.json").write_text(json.dumps(clean({"context": ctx, "benchmarks": bench, "strategies": results,
                                                        "intraday": intra, "meta": meta, "gates": GATES,
                                                        "cost_model": COST_MODEL_VERSION, "transforms": TRANSFORMS_VERSION}), indent=1))
    log(f"report written to {out}")


if __name__ == "__main__":
    main()
