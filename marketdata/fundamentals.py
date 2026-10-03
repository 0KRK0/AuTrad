"""Point-in-time fundamentals: "what did the market know at 15:30 IST on day t?"

Built from data/pit/nse/facts.csv (marketdata.nse_filings). Each row is one filing for one quarter, with
period_end, published_at, available_at (exchange dissemination), revision and the parsed values.

Rules (see research/preregistration_v0.4.md):
  * a filing is usable for a decision on day t only if available_at < 15:30 IST on t; a filing disseminated
    after the close counts from the next session (usable_from);
  * for each quarter, the latest version usable by t is taken (revisions only replace it from their own date);
  * values are company totals in INR (EPS in INR/share);
  * per-share quantities across a split or bonus are made comparable with the verified corporate-action factors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

CLOSE = pd.Timedelta(hours=15, minutes=30)
FINANCIAL_SECTORS = {"banks", "nbfc", "insurance"}
VALUE_COLS = ["revenue", "total_income", "pbt", "net_profit", "eps_basic", "eps_diluted", "finance_costs", "depreciation",
              "paid_up_capital", "face_value", "equity", "total_assets", "borrowings", "gross_npa_pct", "operating_profit_bank"]


def usable_from(available_at: pd.Series) -> pd.Series:
    """Session date from which a filing may be used at the close (strictly before 15:30 IST that day)."""
    t = pd.to_datetime(available_at, utc=True).dt.tz_convert("Asia/Kolkata")
    day = t.dt.tz_localize(None).dt.normalize()
    after_close = (t.dt.tz_localize(None) - day) >= CLOSE
    return day + pd.to_timedelta(after_close.astype(int), unit="D")


def clean_facts(f: pd.DataFrame, split_events: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Screen OUR parse errors (wrong context, swapped or missing fields). A bad value is set to NaN, never replaced
    with information from other periods. Detection may look at neighbouring quarters (parse errors are our mistakes,
    not market information), but no value is filled in from them. Returns (clean facts, issue log)."""
    f = f.copy()
    issues = []

    def flag(mask, col, why):
        for i in f.index[mask.fillna(False)]:
            issues.append({"symbol": f.at[i, "symbol"], "period_end": f.at[i, "period_end"], "field": col, "value": f.at[i, col], "issue": why})
        f.loc[mask.fillna(False), col] = np.nan

    fv = f["face_value"]
    flag(~fv.isin([0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, 100]) & fv.notna(), "face_value", "implausible face value")
    flag((f["paid_up_capital"] < 1e6) & f["paid_up_capital"].notna(), "paid_up_capital", "paid-up capital < Rs10 lakh (swapped/missing units)")
    flag(f["revenue"] <= 0, "revenue", "zero or negative revenue")
    f = f.sort_values(["symbol", "consolidated", "period_end", "available_at"])
    last = f.groupby(["symbol", "consolidated", "period_end"]).tail(1)
    for col, lim in (("revenue", 8.0), ("eps_basic", None)):
        for _, g in last.groupby(["symbol", "consolidated"]):
            x = g[col]
            med = pd.Series([x.drop(i).loc[g["period_end"].sub(g.at[i, "period_end"]).abs().le(pd.Timedelta(days=400)).drop(i, errors="ignore").index]
                             .median() for i in g.index], index=g.index)
            if col == "revenue":
                bad = (x / med > lim) | (x / med < 1 / lim)
            else:
                # EPS must agree with net profit / shares implied by the neighbouring quarters' own EPS
                ev = (split_events or {}).get(g["symbol"].iloc[0], [])
                # put every quarter's implied share count on today's basis (splits/bonuses after its publication)
                adj = np.array([np.prod([1 / k for d, k in ev if d > u]) for u in g["usable_from"]]) if ev else 1.0
                imp = g["net_profit"] / g["eps_basic"] * adj
                imp_med = pd.Series([imp.drop(i).median() for i in g.index], index=g.index)
                bad = ((imp / imp_med) > 1.8) | ((imp / imp_med) < 0.55)
            idx = bad[bad.fillna(False)].index
            same = f.set_index(["symbol", "consolidated", "period_end"]).index
            keys = set(zip(g.loc[idx, "symbol"], g.loc[idx, "consolidated"], g.loc[idx, "period_end"], strict=True))
            flag(pd.Series([k in keys for k in same], index=f.index), col, f"{col} inconsistent with neighbouring quarters (parse error)")
    return f, pd.DataFrame(issues)


def load_facts(path: str | Path, clean: bool = True, split_events: dict | None = None) -> pd.DataFrame:
    f = pd.read_csv(path)
    f["period_end"] = pd.to_datetime(f["period_end"])
    f["available_at"] = pd.to_datetime(f["available_at"], utc=True)
    f["usable_from"] = usable_from(f["available_at"])
    for c in VALUE_COLS:
        if c not in f:
            f[c] = np.nan
        f[c] = pd.to_numeric(f[c], errors="coerce")
    f = f[f["period_end"].notna() & f["available_at"].notna()]
    f["consolidated"] = f["consolidated"].astype(str).str.lower().isin(["true", "1"])
    f["lag_days"] = (f["available_at"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None) - f["period_end"]).dt.days
    if clean:
        f, issues = clean_facts(f, split_events)
        f.attrs["issues"] = issues
    return f.sort_values(["symbol", "period_end", "available_at"]).reset_index(drop=True)


@dataclass
class PITFundamentals:
    facts: pd.DataFrame
    split_events: dict[str, list[tuple[pd.Timestamp, float]]] = field(default_factory=dict)   # symbol -> [(ex_date, k)]
    sector: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._by = {s: g.reset_index(drop=True) for s, g in self.facts.groupby("symbol")}

    # ----------------------------------------------------------------- core as-of view
    def known(self, symbol: str, t: pd.Timestamp) -> pd.DataFrame:
        """Quarterly history of `symbol` exactly as known at the close of day t (latest usable version per quarter)."""
        g = self._by.get(symbol)
        if g is None:
            return pd.DataFrame(columns=self.facts.columns)
        g = g[g["usable_from"] <= pd.Timestamp(t)]
        if g.empty:
            return g
        return g.sort_values(["period_end", "available_at"]).groupby("period_end", as_index=False).tail(1).sort_values("period_end")

    def split_factor(self, symbol: str, after: pd.Timestamp, upto: pd.Timestamp) -> float:
        """Product of split/bonus factors k with ex-date in (after, upto]: shares multiply by 1/k, per-share values by k."""
        k = 1.0
        for d, f in self.split_events.get(symbol, []):
            if pd.Timestamp(after) < d <= pd.Timestamp(upto):
                k *= f
        return k

    # ----------------------------------------------------------------- derived metrics at t
    def snapshot(self, symbol: str, t: pd.Timestamp, raw_price: float | None) -> dict:
        q = self.known(symbol, t)
        out: dict = {"symbol": symbol, "n_quarters": len(q)}
        if q.empty:
            return out
        q = q.reset_index(drop=True)
        last = q.iloc[-1]
        out["last_period"] = last["period_end"]
        out["last_usable"] = last["usable_from"]
        fin = self.sector.get(symbol) in FINANCIAL_SECTORS
        out["financial"] = fin
        # consecutive last four quarters
        q4 = _last_consecutive(q, 4)
        q8 = _last_consecutive(q, 8)
        if q4 is not None:
            out["ttm_np"] = q4["net_profit"].sum(min_count=4)
            out["ttm_rev"] = q4["revenue"].sum(min_count=4)
            k = [self.split_factor(symbol, u, last["usable_from"]) for u in q4["usable_from"]]
            out["ttm_eps"] = float(np.sum(q4["eps_basic"].to_numpy() * np.array(k))) if q4["eps_basic"].notna().all() else np.nan
        if q8 is not None:
            prev4 = q8.iloc[:4]
            out["ttm_np_prev"] = prev4["net_profit"].sum(min_count=4)
            out["ttm_rev_prev"] = prev4["revenue"].sum(min_count=4)
            k = [self.split_factor(symbol, u, last["usable_from"]) for u in prev4["usable_from"]]
            out["ttm_eps_prev"] = float(np.sum(prev4["eps_basic"].to_numpy() * np.array(k))) if prev4["eps_basic"].notna().all() else np.nan
        eq = q.dropna(subset=["equity"])
        if len(eq):
            out["equity"] = float(eq["equity"].iloc[-1])
        bor = q.dropna(subset=["borrowings"])
        if len(bor):
            out["borrowings"] = float(bor["borrowings"].iloc[-1])
        sh = q.dropna(subset=["paid_up_capital", "face_value"])
        sh = sh[sh["face_value"] > 0]
        if len(sh) and raw_price is not None and np.isfinite(raw_price):
            r = sh.iloc[-1]
            # a filing published after a split already states post-split face value: adjust only for events after publication
            shares = r["paid_up_capital"] / r["face_value"] / self.split_factor(symbol, r["usable_from"], t)
            # cross-check with the share count implied by the same filing's profit / EPS (bonuses restate EPS, not capital)
            if np.isfinite(r["net_profit"]) and np.isfinite(r["eps_basic"]) and r["eps_basic"] > 0 and r["net_profit"] > 0:
                implied = r["net_profit"] / r["eps_basic"] / self.split_factor(symbol, r["usable_from"], t)
                if not 0.6 < shares / implied < 1.6:
                    shares = implied
                    out["shares_from_eps"] = True
            out["shares"] = float(shares)
            out["mcap"] = float(shares * raw_price)
        # seasonal surprise: EPS_q - EPS_{q-4}, scaled by the std of that difference over the previous 8 quarters
        qs = _last_consecutive(q, 13)
        if qs is not None and qs["eps_basic"].notna().all():
            ref = qs["usable_from"].iloc[-1]
            eps = qs["eps_basic"].to_numpy() * np.array([self.split_factor(symbol, u, ref) for u in qs["usable_from"]])
            d = eps[4:] - eps[:-4]                 # 9 seasonal differences, the last is the latest quarter
            sd = np.std(d[:-1], ddof=1)
            out["sue"] = float(d[-1] / sd) if sd > 0 else np.nan
            out["eps_yoy_growth_std"] = float(np.std(_yoy_growth(eps)[-8:], ddof=1)) if len(eps) >= 12 else np.nan
        return out


def _last_consecutive(q: pd.DataFrame, n: int) -> pd.DataFrame | None:
    """The last n quarters if they are consecutive quarter-ends on ONE statement basis (consolidated or standalone)."""
    if len(q) < n:
        return None
    tail = q.iloc[-n:]
    if "consolidated" in tail and tail["consolidated"].nunique() > 1:
        return None
    pe = tail["period_end"].dt.year.to_numpy() * 4 + tail["period_end"].dt.quarter.to_numpy()
    if (np.diff(pe) != 1).any():
        return None
    return tail


def _yoy_growth(eps: np.ndarray) -> np.ndarray:
    prev = eps[:-4]
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where(np.abs(prev) > 1e-9, (eps[4:] - prev) / np.abs(prev), np.nan)
    return g[np.isfinite(g)]


def split_events_from_store(store) -> dict[str, list[tuple[pd.Timestamp, float]]]:
    """Share-count events (splits, bonuses) from the exchange-anchored corporate-action engine of a loaded DataStore."""
    out = {}
    for s, evs in getattr(store, "events", {}).items():
        out[s] = [(e.date, e.factor) for e in evs if e.kind in ("split", "bonus", "ratio_event")]
    return out
