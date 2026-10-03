"""Cash-flow quality (accruals) score, point-in-time (research/preregistration_v0.7.md, section 2).

ACC = (NP_TTM - OCF_TTM) / total assets;  CFQ = -ACC (higher = earnings better backed by operating cash).
OCF_TTM: at a March filing the annual figure; at a September filing H1 + (previous annual - previous H1), all on one
statement basis. NP_TTM: the four quarterly net profits ending at the same period end, same basis. Total assets from
the same filing. Every input must be usable (15:30 IST rule) by the decision date; nothing is filled in.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from marketdata.fundamentals import FINANCIAL_SECTORS, usable_from


def prepare(cf: pd.DataFrame, facts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cf = cf[cf["status"] == "ok"].copy()
    cf["period_end"] = pd.to_datetime(cf["period_end"])
    cf["available_at"] = pd.to_datetime(cf["available_at"], utc=True)
    cf["usable_from"] = usable_from(cf["available_at"])
    f = facts.copy()
    f["period_end"] = pd.to_datetime(f["period_end"])
    if "usable_from" not in f:
        f["available_at"] = pd.to_datetime(f["available_at"], utc=True)
        f["usable_from"] = usable_from(f["available_at"])
    return cf, f


def _latest(g: pd.DataFrame, pe: pd.Timestamp, basis: bool) -> pd.Series | None:
    x = g[(g["period_end"] == pe) & (g["consolidated"].astype(bool) == basis)]
    if x.empty:
        return None
    return x.sort_values(["usable_from", "revision"]).iloc[-1]


def cfq_at(cf: pd.DataFrame, f: pd.DataFrame, symbol: str, t: pd.Timestamp) -> dict:
    c = cf[(cf["symbol"] == symbol) & (cf["usable_from"] <= t)]
    if c.empty:
        return {}
    P = c["period_end"].max()
    row = c[c["period_end"] == P].sort_values(["usable_from", "revision"]).iloc[-1]
    basis = bool(row["consolidated"])
    if P.month == 3:
        ocf = float(row["ocf"])
    else:
        fy = _latest(c, P - pd.DateOffset(months=6) + pd.offsets.MonthEnd(0), basis)
        h1p = _latest(c, P - pd.DateOffset(years=1), basis)
        if fy is None or h1p is None:
            return {"missing": "previous annual or H1 cash flow"}
        ocf = float(row["ocf"]) + float(fy["ocf"]) - float(h1p["ocf"])
    g = f[(f["symbol"] == symbol) & (f["usable_from"] <= t)]
    gq = g[g["period_days"].isna() | (g["period_days"] < 120)] if "period_days" in g else g     # quarterly rows only (not half-year)
    nps = []
    for k in range(4):
        q = _latest(gq, (P - pd.DateOffset(months=3 * k)) + pd.offsets.MonthEnd(0), basis)
        if q is None or pd.isna(q["net_profit"]):
            return {"missing": f"net profit for the quarter ending {(P - pd.DateOffset(months=3 * k)).date()}"}
        nps.append(float(q["net_profit"]))
    b = _latest(g, P, basis)
    if b is None or pd.isna(b.get("total_assets")) or b["total_assets"] <= 0:
        return {"missing": "total assets"}
    acc = (sum(nps) - ocf) / float(b["total_assets"])
    return {"cfq": -acc, "period_end": P, "ocf_ttm": ocf, "np_ttm": sum(nps), "total_assets": float(b["total_assets"]), "basis": basis}


def cfq_scores(cf: pd.DataFrame, facts: pd.DataFrame, P, dates: list[int], sector: dict[str, str]) -> pd.DataFrame:
    from lab.factors import stock_universe

    cf, f = prepare(cf, facts)
    rows = {}
    for t in dates:
        d = P.dates[t]
        s = {}
        for j in stock_universe(P, t):
            sym = P.symbols[j]
            if sector.get(sym) in FINANCIAL_SECTORS:
                continue
            r = cfq_at(cf, f, sym, d)
            if "cfq" in r and np.isfinite(r["cfq"]):
                s[sym] = r["cfq"]
        rows[d] = pd.Series(s, dtype=float)
    return pd.DataFrame(rows).T.reindex(columns=P.symbols)
