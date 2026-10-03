"""Shadow scorecard (research/preregistration_v0.7.md, section 3): does the 10th Man's evidence challenge improve
decisions? For every valid (not superseded) shadow memo whose holding month is complete, record each pick's evidence
status and its realised return minus the equal-weight universe (REV1's recorded universe), from the open after the
decision to the next decision close. Test only after >= 24 completed months and >= 30 MATERIAL picks:
MATERIAL picks underperform NONE picks; one-sided t on month-clustered differences; pass needs p < 0.05 and a mean
difference <= -0.5 % per month. Until then the Head's authority over allocation stays zero."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from forward.ledger import Ledger


def valid_memos(shadow: Ledger) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for r in shadow.records():
        if r["kind"] == "memo":
            latest[r["month"]] = r
        elif r["kind"] == "memo_superseded":
            latest.pop(r["month"], None)
    return latest


def outcomes(shadow: Ledger, rev1: Ledger, P) -> pd.DataFrame:
    decs = {d["month"]: d for d in rev1.of_kind("decision")}
    months = sorted(decs)
    C = P.ffill_close()
    rows = []
    for month, memo in sorted(valid_memos(shadow).items()):
        if month not in decs or months.index(month) + 1 >= len(months):
            continue                                         # holding month not complete yet
        d0, d1 = pd.Timestamp(decs[month]["date"]), pd.Timestamp(decs[months[months.index(month) + 1]]["date"])
        if d0 not in P.dates or d1 not in P.dates:
            continue
        i0, i1 = P.dates.get_loc(d0), P.dates.get_loc(d1)
        ret = {s: C[i1, P.idx[s]] / P.O[i0 + 1, P.idx[s]] - 1 for s in decs[month]["universe"] if s in P.idx}
        uni = float(np.nanmean(list(ret.values())))
        for it in memo["memo"]["items"]:
            c = it["challenge"]
            rows.append({"month": month, "symbol": it["symbol"], "evidence_status": c.get("evidence_status", "MATERIAL" if c.get("status") == "MATERIAL_CHALLENGE" else "NONE"),
                         "historical_p": (c.get("historical") or {}).get("value", c.get("p_fail")), "excess": float(ret.get(it["symbol"], np.nan) - uni)})
    return pd.DataFrame(rows)


def evaluate(df: pd.DataFrame, min_months: int = 24, min_material: int = 30) -> dict:
    if df.empty:
        return {"months": 0, "material_picks": 0, "status": "ACCUMULATING (no completed month yet)"}
    months = df["month"].nunique()
    nm = int((df["evidence_status"] == "MATERIAL").sum())
    out = {"months": int(months), "picks": int(len(df)), "material_picks": nm,
           "mean_excess_material": float(df.loc[df.evidence_status == "MATERIAL", "excess"].mean()) if nm else None,
           "mean_excess_none": float(df.loc[df.evidence_status == "NONE", "excess"].mean())}
    if months < min_months or nm < min_material:
        out["status"] = f"ACCUMULATING (test needs >= {min_months} months and >= {min_material} MATERIAL picks)"
        return out
    g = df.groupby(["month", "evidence_status"])["excess"].mean().unstack()
    d = (g.get("MATERIAL") - g.get("NONE")).dropna()
    t = float(d.mean() / d.std() * np.sqrt(len(d))) if len(d) > 2 and d.std() > 0 else float("nan")
    p = float(sps.t.cdf(t, df=len(d) - 1)) if np.isfinite(t) else float("nan")
    out |= {"months_with_both": int(len(d)), "mean_diff_per_month": float(d.mean()), "t": t, "p_one_sided": p,
            "status": "PASS (challenges add value; authority still needs a separate pre-registration)" if p < 0.05 and d.mean() <= -0.005 else "FAIL"}
    return out
