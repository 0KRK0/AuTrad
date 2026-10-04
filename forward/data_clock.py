"""Data clock (v1.4): how far each forward dataset is from the point where its sealed study may be opened.

    python -m forward.data_clock            -> prints the clock, writes research/data_clock.json

Nothing here looks at returns. It only counts what has been collected:

* analyst snapshots   (scripts/fetch_estimates)       sealed until >= 12 months of snapshots AND >= 24 monthly
                                                       cross-sections with >= 100 valid stocks (preregistration_v1.3, Part B)
* NSE depth           (marketdata/nse_depth_recorder) sealed until >= 20 sealed (QC-passed) days for a symbol (Part C)
* promise verdicts    (intel/promise_review)          human verdicts only; no study threshold is fixed yet
* F&O ban at expiry   (forward/ban_expiry)            sealed until >= 24 months and >= 300 events
* candidates          (lab/candidates)                ideas waiting for a pre-registration; never tradeable

A cross-section is counted for a calendar month when its last snapshot day has >= 100 valid stocks (current fiscal
year EPS, >= 3 analysts, no quality flags) and an earlier snapshot exists at least 28 calendar days before it, so a
20-session change can be formed. This is a conservative proxy; the sealed study recomputes the exact count.
"""
from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from datetime import date
from pathlib import Path

EST_ROOT = Path("data/estimates/yahoo")
DEPTH_ROOT = Path("data/micro/nse")
GRAPH = Path("data/intel/graph.sqlite")
OUT = Path("research/data_clock.json")

ANALYST_MIN_MONTHS, ANALYST_MIN_XS, ANALYST_MIN_STOCKS, XS_LOOKBACK_DAYS = 12, 24, 100, 28
DEPTH_MIN_DAYS = 20


def _valid_names(day_dir: Path) -> int:
    from marketdata.estimates import flatten
    n = 0
    for p in day_dir.glob("*.json"):
        try:
            rows = [r for r in flatten(p) if r["horizon"] == "0y"]
        except Exception:
            continue
        if rows and not rows[0]["flags"] and (rows[0]["eps_n"] or 0) >= 3:
            n += 1
    return n


def analyst(root: Path = EST_ROOT) -> dict:
    days = sorted(p.name for p in Path(root).glob("????-??-??") if p.is_dir())
    per_day = {d: {"names": len(list((Path(root) / d).glob("*.json"))), "valid": _valid_names(Path(root) / d)} for d in days}
    months = 0.0
    if days:
        months = round((date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days / 30.44, 1)
    last_in_month: dict[str, str] = {}
    for d in days:
        last_in_month[d[:7]] = d
    xs = []
    for m, d in sorted(last_in_month.items()):
        earlier = [e for e in days if (date.fromisoformat(d) - date.fromisoformat(e)).days >= XS_LOOKBACK_DAYS]
        if earlier and per_day[d]["valid"] >= ANALYST_MIN_STOCKS:
            xs.append(m)
    return {"snapshot_days": len(days), "first": days[0] if days else None, "last": days[-1] if days else None,
            "months": months, "cross_sections": len(xs), "latest_names": per_day[days[-1]]["names"] if days else 0,
            "latest_valid": per_day[days[-1]]["valid"] if days else 0,
            "threshold": f">= {ANALYST_MIN_MONTHS} months and >= {ANALYST_MIN_XS} cross-sections with >= {ANALYST_MIN_STOCKS} valid stocks",
            "unsealed": months >= ANALYST_MIN_MONTHS and len(xs) >= ANALYST_MIN_XS}


def depth(root: Path = DEPTH_ROOT) -> dict:
    sealed: dict[str, int] = defaultdict(int)
    raw: dict[str, int] = defaultdict(int)
    for sym_dir in (p for p in Path(root).glob("*") if p.is_dir()):
        for day in (p for p in sym_dir.glob("*") if p.is_dir()):
            raw[sym_dir.name] += 1
            m = day / "manifest.json"
            if m.exists() and json.loads(m.read_text()).get("qc", {}).get("ok"):
                sealed[sym_dir.name] += 1
    best = max(sealed.values(), default=0)
    return {"symbols": len(raw), "recorded_days_max": max(raw.values(), default=0), "sealed_days_max": best,
            "per_symbol_sealed": dict(sorted(sealed.items())), "threshold": f">= {DEPTH_MIN_DAYS} sealed days per symbol",
            "unsealed": best >= DEPTH_MIN_DAYS,
            "blocker": None if raw else "recorder not running: needs the owner's Kite Connect API key and access token"}


def promises(graph: Path = GRAPH) -> dict:
    out = {"reviewed_claims": 0, "verdicts": {}, "queue": 35, "threshold": "none fixed; a study needs its own pre-registration"}
    if Path(graph).exists():
        con = sqlite3.connect(graph)
        try:
            rows = con.execute("SELECT r.claim_id, r.verdict FROM promise_reviews r JOIN (SELECT claim_id, MAX(reviewed_at) m "
                               "FROM promise_reviews GROUP BY claim_id) l ON l.claim_id=r.claim_id AND l.m=r.reviewed_at").fetchall()
        except sqlite3.OperationalError:
            rows = []
        con.close()
        out["reviewed_claims"] = len({r[0] for r in rows})
        for _, v in rows:
            out["verdicts"][v] = out["verdicts"].get(v, 0) + 1
    return out


def ban() -> dict:
    try:
        from forward.ban_expiry import MIN_EVENTS, MIN_MONTHS, status
        s = status()
        return {**s, "threshold": f">= {MIN_MONTHS} months and >= {MIN_EVENTS} events"}
    except Exception as e:  # pragma: no cover - ledger not initialised
        return {"events": 0, "months": 0.0, "evaluable": False, "error": str(e)}


def candidates() -> dict:
    from lab.candidates import load
    return {k: c["status"] for k, c in load().items()}


def clock() -> dict:
    return {"asof": date.today().isoformat(), "analyst": analyst(), "depth": depth(), "promises": promises(), "ban_expiry": ban(),
            "candidates": candidates(), "proven_alpha": 0, "live_trading": "off"}


def markdown(c: dict) -> str:
    a, d, p, b = c["analyst"], c["depth"], c["promises"], c["ban_expiry"]
    pct = lambda x, n: f"{min(100, round(100 * x / n))} %"
    rows = [
        ("Analyst snapshots", f"{a['snapshot_days']} days ({a['first']} to {a['last']}), {a['latest_names']} names, {a['latest_valid']} valid",
         f"{a['months']} / 12 months; {a['cross_sections']} / 24 cross-sections", pct(min(a["months"] / 12, a["cross_sections"] / 24) * 100, 100),
         "open" if a["unsealed"] else "sealed"),
        ("NSE depth", f"{d['symbols']} symbols, {d['recorded_days_max']} recorded days", f"{d['sealed_days_max']} / 20 sealed days",
         pct(d["sealed_days_max"], 20), d["blocker"] or ("open" if d["unsealed"] else "sealed")),
        ("Promise verdicts", f"{p['reviewed_claims']} of {p['queue']} queued promises reviewed", ", ".join(f"{k} {v}" for k, v in p["verdicts"].items()) or "none",
         pct(p["reviewed_claims"], p["queue"]), "human review"),
        ("Ban at expiry (C-014)", f"{b.get('events', 0)} events", f"{b.get('months', 0)} / 24 months; {b.get('events', 0)} / 300 events",
         pct(min(b.get("months", 0) / 24, b.get("events", 0) / 300) * 100, 100), "open" if b.get("evaluable") else "sealed"),
    ]
    lines = ["## Data clock", "", f"As of {c['asof']}. Counts only; no returns are looked at.", "",
             "| Dataset | Collected | Toward the threshold | Progress | State |", "|---|---|---|---|---|"]
    lines += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |" for r in rows]
    lines += ["", "Candidates (not strategies, never tradeable): " + ", ".join(f"{k} {v}" for k, v in c["candidates"].items()), ""]
    return "\n".join(lines)


if __name__ == "__main__":
    c = clock()
    OUT.write_text(json.dumps(c, indent=2))
    print(markdown(c))
