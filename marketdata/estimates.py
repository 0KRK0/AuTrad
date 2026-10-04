"""Analyst consensus snapshots — collected forward, point-in-time by construction.

No free source offers *historical* point-in-time consensus for Indian stocks (that is LSEG I/B/E/S, FactSet, Bloomberg,
S&P Capital IQ or Trendlyne's paid history). So AuTrad builds its own history from today: once a day it stores what the
public consensus says *now*, stamped with the fetch time. After ~2 years of snapshots, revision and surprise studies
can be run on data that was genuinely known at each date.

Source (tier 6, specialist third party): Yahoo Finance analysis tables via `yfinance` — EPS and revenue consensus for
the current/next quarter and fiscal year, analyst counts, the EPS trend and revision counts, and the last four
quarters of estimate-vs-actual. The vendor's own look-back columns (7/30/60/90 days ago) are stored but labelled
`vendor_lookback`: they are what the vendor says the consensus was, not something AuTrad observed at the time.

Quality flags on every snapshot: too few analysts, EPS trend vs estimate table disagreeing, and non-INR currency.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import pandas as pd

TABLES = ("eps_trend", "earnings_estimate", "revenue_estimate", "eps_revisions", "earnings_history", "analyst_price_targets",
          "recommendations")
MIN_ANALYSTS = 3
TREND_TOLERANCE = 0.05


def yahoo_symbol(sym: str) -> str:
    return f"{sym}.NS"


def _jsonable(v):
    if isinstance(v, pd.DataFrame):
        return json.loads(v.reset_index().to_json(orient="records", date_format="iso"))
    if isinstance(v, dict):
        return {k: (None if isinstance(x, float) and math.isnan(x) else x) for k, x in v.items()}
    return v


def snapshot(sym: str, root: Path, day: str | None = None) -> Path | None:  # pragma: no cover - network
    import yfinance as yf

    day = day or time.strftime("%Y-%m-%d", time.gmtime())
    d = Path(root) / day
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{sym}.json"
    if p.exists():
        return p
    t = yf.Ticker(yahoo_symbol(sym))
    out = {"symbol": sym, "vendor_symbol": yahoo_symbol(sym), "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "source": "Yahoo Finance analysis (yfinance)", "tier": 6, "tables": {}, "errors": {}}
    for name in TABLES:
        try:
            out["tables"][name] = _jsonable(getattr(t, name))
        except Exception as e:  # noqa: BLE001
            out["errors"][name] = str(e)[:200]
    p.write_text(json.dumps(out))
    return p


def _rows(tab) -> list[dict]:
    return tab if isinstance(tab, list) else []


def flatten(path: Path) -> list[dict]:
    """One row per (symbol, horizon) with the consensus, its analyst count and the quality flags."""
    j = json.loads(Path(path).read_text())
    T = j["tables"]
    trend = {r.get("period"): r for r in _rows(T.get("eps_trend"))}
    revs = {r.get("period"): r for r in _rows(T.get("eps_revisions"))}
    rev_est = {r.get("period"): r for r in _rows(T.get("revenue_estimate"))}
    out = []
    for r in _rows(T.get("earnings_estimate")):
        h = r.get("period")
        tr, rv, re_ = trend.get(h, {}), revs.get(h, {}), rev_est.get(h, {})
        avg, cur = r.get("avg"), tr.get("current")
        n = r.get("numberOfAnalysts") or 0
        flags = []
        if n < MIN_ANALYSTS:
            flags.append("few_analysts")
        if avg and cur and abs(cur / avg - 1) > TREND_TOLERANCE:
            flags.append("trend_vs_estimate_mismatch")
        if (r.get("currency") or tr.get("currency") or "INR") != "INR":
            flags.append("non_inr")
        out.append({"symbol": j["symbol"], "fetched_at": j["fetched_at"], "horizon": h, "eps_avg": avg, "eps_low": r.get("low"),
                    "eps_high": r.get("high"), "eps_n": n, "eps_year_ago": r.get("yearAgoEps"),
                    "eps_trend_current": cur, "vendor_lookback_7d": tr.get("7daysAgo"), "vendor_lookback_30d": tr.get("30daysAgo"),
                    "vendor_lookback_90d": tr.get("90daysAgo"), "up_30d": rv.get("upLast30days"), "down_30d": rv.get("downLast30days"),
                    "rev_avg": re_.get("avg"), "rev_n": re_.get("numberOfAnalysts"), "flags": ";".join(flags), "tier": j.get("tier", 6)})
    return out


def surprises(path: Path) -> list[dict]:
    """Vendor-reported estimate vs actual for the last four quarters (cross-checkable against filed EPS)."""
    j = json.loads(Path(path).read_text())
    return [{"symbol": j["symbol"], "fetched_at": j["fetched_at"], "quarter": str(r.get("quarter"))[:10], "eps_actual": r.get("epsActual"),
             "eps_estimate": r.get("epsEstimate"), "surprise_pct": r.get("surprisePercent")} for r in _rows(j["tables"].get("earnings_history"))]


def load(root: Path) -> pd.DataFrame:
    rows = [x for p in sorted(Path(root).glob("*/*.json")) for x in flatten(p)]
    return pd.DataFrame(rows)
