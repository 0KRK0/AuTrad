"""Corporate-action / data-integrity audit: vendor (Yahoo, NSE, adjusted) vs exchange (BSE, unadjusted).

A correctly adjusted vendor series equals the exchange price times an adjustment factor F_t that is a step
function: flat between events, stepping only on real ex-dates (small steps = dividends, large = splits,
bonuses, demergers). So r_t = log(vendor_close / bse_close) is the vendor's implied factor plus a little
cross-venue noise. The audit fits that step function and classifies every step and every big raw move:

  ADJUSTED_EVENT     vendor step on a day the exchange shows an abnormal raw move  -> vendor adjusted it
  DIVIDEND           small positive step (< 5%)                                      -> vendor's dividend adjustment
  VENDOR_ARTIFACT    large step on a day the exchange price is normal               -> wrong date / bad factor / glitch
  UNADJUSTED_EVENT   both series show the same large abnormal drop, no step          -> possible missed corporate action
  BLIP               ratio departs and returns within 5 sessions                     -> transient vendor print error
Every large event must then match an entry in reference/corporate_actions.csv (verified with a public
source) before the corrected series uses it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

STEP_MIN = 0.004        # 0.4% in log terms: smaller steps are folded into the segment (noise / tiny dividends)
DIV_MAX = 0.05          # dividend-like steps up to 5%
ABNORMAL = 0.12         # |raw exchange move net of market| that counts as an event day


@dataclass
class Step:
    date: pd.Timestamp
    size: float          # log step of vendor/exchange ratio at this date
    bse_ret: float       # raw exchange log return that day (net of market)
    vendor_ret: float    # vendor log return that day (net of market)
    kind: str


def ratio_frame(vendor: pd.DataFrame, bse: pd.DataFrame, bench: pd.Series | None = None) -> pd.DataFrame:
    """Daily frame on dates where both venues traded."""
    idx = vendor.index.intersection(bse.index)
    f = pd.DataFrame({"v": vendor["close"].reindex(idx).astype(float), "b": bse["close"].reindex(idx).astype(float),
                      "b_open": bse["open"].reindex(idx).astype(float)}, index=idx)
    f = f[(f.v > 0) & (f.b > 0)]
    f["lr"] = np.log(f.v / f.b)
    m = np.log(bench.reindex(f.index).ffill()).diff() if bench is not None else 0.0
    f["mkt"] = m if bench is not None else 0.0
    f["bret"] = np.log(f.b).diff() - f["mkt"].fillna(0)
    f["vret"] = np.log(f.v).diff() - f["mkt"].fillna(0)
    return f


def fit_steps(lr: pd.Series, w: int = 5, thr: float = STEP_MIN) -> list[tuple[int, float]]:
    """Change points of a noisy step function: windowed-median difference, local maxima above thr."""
    x = lr.to_numpy()
    n = len(x)
    if n < 2 * w + 1:
        return []
    d = np.zeros(n)
    for t in range(w, n - w + 1):
        d[t] = np.median(x[t:t + w]) - np.median(x[t - w:t])
    out = []
    t = w
    while t < n - w + 1:
        if abs(d[t]) > thr:
            j = t
            while j + 1 < n - w + 1 and abs(d[j + 1]) > thr and np.sign(d[j + 1]) == np.sign(d[t]):
                j += 1
            k = t + int(np.argmax(np.abs(d[t:j + 1])))
            # exact day: the single largest one-day change of lr inside the window around k
            lo, hi = max(k - w + 1, 1), min(k + w, n)
            jumps = np.abs(np.diff(x[lo - 1:hi]))
            day = lo + int(np.argmax(jumps))
            size = float(np.median(x[day:min(day + w, n)]) - np.median(x[max(day - w, 0):day]))
            out.append((day, size))
            t = j + 1
        else:
            t += 1
    return out


def find_blips(lr: pd.Series, max_len: int = 5, thr: float = 0.05) -> list[tuple[pd.Timestamp, pd.Timestamp, float]]:
    x = lr.to_numpy()
    out = []
    t = 1
    while t < len(x):
        base = x[t - 1]
        if abs(x[t] - base) > thr:
            for L in range(1, max_len + 1):
                if t + L < len(x) and abs(x[t + L] - base) < thr / 3:
                    out.append((lr.index[t], lr.index[t + L - 1], float(x[t] - base)))
                    t += L
                    break
        t += 1
    return out


def audit_symbol(symbol: str, vendor: pd.DataFrame, bse: pd.DataFrame, bench: pd.Series | None = None) -> dict:
    f = ratio_frame(vendor, bse, bench)
    blips = find_blips(f["lr"])
    lr = f["lr"].copy()
    for a, b, _ in blips:                       # mask transient prints before fitting steps
        lr[(lr.index >= a) & (lr.index <= b)] = np.nan
    lr = lr.ffill().bfill()
    steps = []
    for i, size in fit_steps(lr):
        d = f.index[i]
        br, vr = float(f["bret"].iloc[i]), float(f["vret"].iloc[i])
        explained = abs(br + size) < max(0.03, 0.25 * abs(size))   # the exchange moved by the step that day
        if explained and abs(vr) < 0.06 and abs(size) >= DIV_MAX:
            kind = "ADJUSTED_EVENT"
        elif explained and abs(vr) < 0.06 and size > 0:
            kind = "DIVIDEND"
        elif 0 < size < DIV_MAX and abs(vr) < ABNORMAL:
            kind = "DIVIDEND"
        elif abs(size) < 0.01 and abs(vr) < ABNORMAL:
            kind = "NOISE"
        else:
            kind = "VENDOR_ARTIFACT"
        steps.append(Step(d, size, br, vr, kind))
    stepped = {s.date for s in steps}
    unadj = [Step(d, 0.0, float(r.bret), float(r.vret), "UNADJUSTED_EVENT") for d, r in f.iterrows()
             if d not in stepped and r.bret < -ABNORMAL and r.vret < -ABNORMAL and abs(r.bret - r.vret) < 0.02]
    resid = (f["vret"] - f["bret"]).abs()
    return {"symbol": symbol, "days": int(len(f)), "first": f.index[0] if len(f) else None, "last": f.index[-1] if len(f) else None,
            "steps": steps, "unadjusted": unadj, "blips": blips,
            "median_abs_venue_gap_bps": float((f["lr"] - f["lr"].rolling(21, center=True, min_periods=5).median()).abs().median() * 1e4),
            "days_return_mismatch_gt2pct": int((resid > 0.02).sum()), "frame": f}
