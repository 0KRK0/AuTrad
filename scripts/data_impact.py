"""Impact of the data correction on strategy results: v0.3 (vendor data + hand fixes) vs v0.3.1 (same judge,
same strategies and grids, exchange-anchored data). Appends a section to the data-integrity report.

    python -m scripts.data_impact --before reports/v0.3_<ts> --after reports/v0.3.1_<ts> --audit reports/data_integrity_<ts>
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _rows(res: dict) -> dict[str, dict]:
    out = {}
    for r in res.get("strategies", []) + res.get("intraday", []) + ([res["meta"]] if res.get("meta") else []):
        out[r["strategy"]] = r
    return out


def f(x, fmt="{:.2f}"):
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else fmt.format(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--audit", required=True)
    a = ap.parse_args()
    B = json.loads(Path(a.before, "results.json").read_text())
    A = json.loads(Path(a.after, "results.json").read_text())
    rb, ra = _rows(B), _rows(A)
    L = ["## Impact on strategy results (same judge, same strategies, corrected data)\n",
         f"Before: `{Path(a.before).name}` (vendor data with the v0.3 hand fixes). After: `{Path(a.after).name}` "
         "(exchange-anchored data). Strategy code, parameter grids, splits and gates are identical; only the prices changed. "
         "Intraday rows use hourly vendor bars, which BSE does not publish for free, so they are unchanged by this audit.\n",
         "| Strategy | Verdict before → after | Gates | Test Sharpe | OOS Sharpe (2018+) | OOS CAGR | OOS max DD | Selected config changed? |",
         "|---|---|---|---|---|---|---|---|"]
    moved = []
    for k in [*rb.keys(), *[x for x in ra if x not in rb]]:
        b, a_ = rb.get(k, {}), ra.get(k, {})
        if not a_:
            continue
        tb, ta = b.get("test", {}).get("sharpe"), a_.get("test", {}).get("sharpe")
        ob, oa = b.get("oos", {}).get("sharpe"), a_.get("oos", {}).get("sharpe")
        cb, ca = b.get("oos", {}).get("cagr"), a_.get("oos", {}).get("cagr")
        db, da = b.get("oos", {}).get("max_drawdown"), a_.get("oos", {}).get("max_drawdown")
        sel = "yes" if b and b.get("selected") != a_.get("selected") else "no"
        L.append(f"| {k} | {b.get('verdict', '—')} → {a_['verdict']} | {b.get('gates_passed', '—')} → {a_['gates_passed']}/{a_['gates_total']} | "
                 f"{f(tb)} → {f(ta)} | {f(ob)} → {f(oa)} | {f(cb, '{:+.1%}')} → {f(ca, '{:+.1%}')} | "
                 f"{f(db, '{:.1%}')} → {f(da, '{:.1%}')} | {sel} |")
        if tb is not None and ta is not None and (abs(ta - tb) >= 0.05 or (ob is not None and oa is not None and abs(oa - ob) >= 0.05)):
            moved.append(k)
    L.append("")
    bb, ba = B.get("benchmarks", {}), A.get("benchmarks", {})
    L += ["| Benchmark | Test Sharpe | OOS Sharpe | OOS CAGR |", "|---|---|---|---|"]
    for k in ba:
        L.append(f"| {k} | {f(bb.get(k, {}).get('test_sharpe'))} → {f(ba[k].get('test_sharpe'))} | "
                 f"{f(bb.get(k, {}).get('oos_sharpe'))} → {f(ba[k].get('oos_sharpe'))} | "
                 f"{f(bb.get(k, {}).get('oos_cagr'), '{:+.1%}')} → {f(ba[k].get('oos_cagr'), '{:+.1%}')} |")
    q_before = [k for k, r in rb.items() if r.get("verdict") == "QUALIFIED_FOR_PAPER"]
    q_after = [k for k, r in ra.items() if r.get("verdict") == "QUALIFIED_FOR_PAPER"]
    L += ["", f"Strategies whose test or out-of-sample Sharpe moved by 0.05 or more: {', '.join(moved) or 'none'}. "
          f"Qualified for paper trading before: {', '.join(q_before) or 'none'}; after: {', '.join(q_after) or 'none'}.\n"]
    sb = {k: r.get("survivorship") for k, r in rb.items() if r.get("survivorship")}
    sa = {k: r.get("survivorship") for k, r in ra.items() if r.get("survivorship")}
    if sa:
        L += ["Survivorship-bias test (OOS Sharpe, today's members only vs true point-in-time members):\n",
              "| Strategy | Before: survivors / PIT | After: survivors / PIT |", "|---|---|---|"]
        for k in sa:
            x, y = sb.get(k) or {}, sa[k]
            L.append(f"| {k} | {f(x.get('survivors_oos_sharpe'))} / {f(x.get('pit_oos_sharpe'))} | "
                     f"{f(y.get('survivors_oos_sharpe'))} / {f(y.get('pit_oos_sharpe'))} |")
        L.append("")
    text = "\n".join(L) + "\n"
    Path(a.audit, "impact.md").write_text(text, encoding="utf-8")
    rep = Path(a.audit, "report.md")
    body = rep.read_text(encoding="utf-8")
    marker = "## Data-quality score"
    if "## Impact on strategy results" not in body and marker in body:
        body = body.replace(marker, text + "\n" + marker)
        rep.write_text(body, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
