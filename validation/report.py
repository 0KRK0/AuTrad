"""Write a reproducible validation report: results.json, report.md, equity and trade CSVs."""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .scorecard import THRESHOLDS
from .walkforward import GRID


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip() or "uncommitted"
    except Exception:
        return "unknown"


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    return o


def pct(x, d=1, sign=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%"


def write(results: dict[str, dict], meta: dict, out_root: str = "reports") -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(out_root) / f"v0.2_{run_id}"
    out.mkdir(parents=True, exist_ok=True)
    meta = {**meta, "run_id": run_id, "git_sha": _git_sha(), "created_utc": run_id,
            "grid": [asdict(c) for c in GRID], "thresholds": THRESHOLDS}
    (out / "results.json").write_text(json.dumps(_clean({"meta": meta, "groups": results}), indent=2))
    for g, r in results.items():
        pd.DataFrame(r["_series"]).to_csv(out / f"equity_{g}.csv", float_format="%.2f")
        if not r["_ledger"].empty:
            r["_ledger"].to_csv(out / f"trades_{g}.csv", index=False, float_format="%.2f")
    (out / "report.md").write_text(markdown(results, meta))
    return out


def markdown(results: dict[str, dict], meta: dict) -> str:
    L = [f"# AuTrad v0.2 real-market validation — {meta['run_id']}", "",
         f"Data manifest `{meta['data_manifest']}` · code `{meta['git_sha']}` · capital Rs{meta['cash']:,.0f} · "
         f"cash yield {meta['cash_rate']:.1%} · out-of-sample from {meta['first_test_year']}", "",
         "| Group | Verdict | Gates | OOS CAGR | Sharpe | Max DD | Best baseline | Baseline CAGR | Baseline Sharpe |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for g, r in results.items():
        m, b = r["oos"], r["baselines"][r["best_baseline"]]
        sc = r["scorecard"]
        L.append(f"| {g} | {sc['verdict']} | {sc['passed']}/{sc['total']} | {pct(m['cagr'])} | {m['sharpe']:.2f} | "
                 f"{m['max_drawdown']:.1%} | {r['best_baseline']} | {pct(b.get('cagr'))} | {b.get('sharpe', 0):.2f} |")
    for g, r in results.items():
        m, t = r["oos"], r["trades"]
        L += ["", f"## {g}", "", f"Symbols: {', '.join(r['symbols'])}", "",
              "| Metric | AuTrad (walk-forward OOS) |", "| --- | --- |"]
        rows = [("Net CAGR", pct(m["cagr"])), ("Gross CAGR (before costs)", pct(m.get("gross_cagr"))),
                ("Excess over cash", pct(m["excess_cagr"])), ("After-tax CAGR (estimate)", pct(r["after_tax"]["cagr"])),
                ("Sharpe (95% CI)", f"{m['sharpe']:.2f} ({r['stats']['sharpe_ci_95'][0]:.2f} to {r['stats']['sharpe_ci_95'][1]:.2f})"),
                ("Sortino", f"{m['sortino']:.2f}"), ("Max drawdown", f"{m['max_drawdown']:.1%}"), ("Calmar", f"{m['calmar']:.2f}"),
                ("Worst day / month", f"{pct(m['worst_day'])} / {pct(m['worst_month'])}"), ("Cost drag per year", pct(m.get("cost_drag_cagr"), 2, sign=False)), ("CVaR 95% (daily loss)", pct(m["cvar_95"], 2, sign=False)),
                ("Avg exposure", f"{m.get('avg_exposure', 0):.0%}"), ("Turnover per year", f"{m.get('turnover_per_year', 0):.1f}x"),
                ("Trading P&L before / after costs", f"Rs{m.get('trading_pnl_gross', 0):,.0f} / Rs{m.get('trading_pnl_net', 0):,.0f}"),
                ("Costs paid", f"Rs{m.get('costs_paid', 0):,.0f}" + (f" ({m['cost_share_of_gross_profit']:.0%} of gross trading profit)"
                                                                  if np.isfinite(m.get('cost_share_of_gross_profit', np.inf)) else " (no gross trading profit)")),
                ("Trades / win rate / profit factor", f"{t['trades']} / {t['win_rate']:.0%} / {t['profit_factor']:.2f}"),
                ("Expected value per trade", f"Rs{t['avg_trade']:,.0f} ({pct(t['avg_trade_ret'], 2)} of notional)"),
                ("PSR / deflated Sharpe", f"{r['stats']['psr']:.2f} / {r['stats']['deflated_sharpe']:.2f} ({r['stats']['n_trials']} trials)"),
                ("Random traders (p-value)", f"median Sharpe {r['random']['median_sharpe']:.2f}, 95th pct {r['random']['p95_sharpe']:.2f}, p = {r['random']['p_value']:.3f}"),
                ("Random-trader exposure (target / realised)", f"{r['random']['calibration']['target_exposure']:.0%} / {r['random']['calibration']['realised_exposure']:.0%}")]
        L += [f"| {a} | {b} |" for a, b in rows]
        L += ["", "**Baselines (same window, same costs)**", "", "| Strategy | CAGR | Sharpe | Max DD | Cost drag /yr |", "| --- | --- | --- | --- | --- |"]
        for n, b in r["baselines"].items():
            drag = pct(b["cost_drag_cagr"], 2, sign=False) if "cost_drag_cagr" in b else "none (index)"
            L.append(f"| {n} | {pct(b.get('cagr'))} | {b.get('sharpe', 0):.2f} | {b.get('max_drawdown', 0):.1%} | {drag} |")
        L += ["", "**Walk-forward folds**", "", "| Test year | Chosen config | Train Sharpe | Test return | Test Sharpe |", "| --- | --- | --- | --- | --- |"]
        for f in r["folds"]:
            L.append(f"| {f['year']} | {f['chosen']} | {f['train_sharpe']:.2f} | {pct(f['test_return'])} | {f['test_sharpe']:.2f} |")
        L += ["", f"**Scorecard: {r['scorecard']['verdict']}** — {r['scorecard']['action']}", "", "| Gate | Result | Value | Target |", "| --- | --- | --- | --- |"]
        for gate in r["scorecard"]["gates"]:
            L.append(f"| {gate['name']} | {'PASS' if gate['passed'] else 'FAIL'} | {gate['value']} | {gate['target']} |")
        if r["regimes"]:
            L += ["", "**By market regime (NIFTY trend x volatility)**", "", "| Regime | Days | Excess return /yr | Sharpe |", "| --- | --- | --- | --- |"]
            for k, v in sorted(r["regimes"].items()):
                L.append(f"| {k} | {v['days']} | {pct(v['ann_excess_return'])} | {v['sharpe']:.2f} |")
        if r["attribution"]:
            L += ["", "**P&L by category (OOS trades, after fees)**", "", "| Category | Trades | P&L | Fees |", "| --- | --- | --- | --- |"]
            for k, v in r["attribution"].items():
                L.append(f"| {k} | {v['trades']:.0f} | Rs{v['pnl']:,.0f} | Rs{v['fees']:,.0f} |")
    L += ["", "## Assumptions and known biases", "",
          "- Survivorship bias: equities are today's liquid names; delisted/removed stocks are missing. This flatters all long-only strategies.",
          "- Yahoo adjusted prices; crypto INR prices are USD prices converted, not Indian exchange order books.",
          "- Fills at next bar open with estimated spread and square-root impact; spreads must be recalibrated from live paper fills.",
          "- Idle cash earns the cash yield (liquid-fund assumption) for every strategy.",
          "- After a kill-switch trip the backtest stays flat for the cooldown, then resets (models a human review).",
          "- Taxes are an estimate for a resident individual; not tax advice."]
    return "\n".join(L) + "\n"
