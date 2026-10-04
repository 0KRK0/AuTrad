"""Whole-market NSE equity panel and point-in-time liquidity universes (v0.8.1).

Prices come only from NSE's official bhavcopy (data/exchange_all/nse). NSE's archived prev_close is NOT adjusted on
ex-dates, so every day's return is adjusted with NSE's own corporate-action records (marketdata/nse_corp_actions.py):
splits, bonuses, consolidations, rights (theoretical ex-rights) and cash (dividends, return of capital) -> total
returns. A price factor is skipped when the ex-date open shows that prev_close already carries it.

Security identity: ISIN first. An ISIN that stops trading is linked to a new ISIN that starts (a) on the next session
under the same NSE symbol, or (b) within 60 sessions with the same issuer code (ISIN characters 3-7, e.g. after a
suspension or a re-issued ISIN), when exactly one candidate exists.

Universe (PIT, survivorship-free): at a decision date t, rank companies (ISIN prefix INE, series EQ/BE/BZ) by the
median daily traded value over the previous 126 sessions up to and including t. Eligible only if traded on at least
80% of those sessions and listed for at least 126 sessions. TOP100 / TOP200 / TOP500 are liquidity proxies for the
NIFTY 100 / 200 / 500 (NSE's indices use free-float market cap with liquidity screens; their historical member lists
are not used).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("data/exchange_all")
PANEL = ROOT / "panel.parquet"
LOOKBACK, MIN_TRADED = 126, 0.8


def build_panel(root: Path = ROOT) -> pd.DataFrame:
    frames = []
    for p in sorted((Path(root) / "nse").glob("*.csv.gz")):
        d = pd.read_csv(p, dtype={"isin": str, "symbol": str, "series": str})
        d = d[d["isin"].fillna("").str.startswith("INE")]
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    rank = df["series"].map({"EQ": 0, "BE": 1, "BZ": 2}).fillna(3)
    df = df.assign(_r=rank).sort_values(["date", "isin", "_r"]).drop_duplicates(["date", "isin"]).drop(columns="_r")
    df["sec"] = link_ids(df)
    for c in ("open", "high", "low", "close", "prev_close"):
        df[c] = df[c].astype("float64")
    df["value"] = df["value"].astype("float64")
    df = df.drop_duplicates(["date", "sec"], keep="first")
    df[["date", "sec", "isin", "symbol", "series", "open", "high", "low", "close", "prev_close", "volume", "value"]].to_parquet(Path(root) / "panel.parquet")
    return df


def link_ids(df: pd.DataFrame) -> pd.Series:
    """ISIN -> security id, linking an ISIN that ends to a new ISIN starting next session under the same symbol."""
    first = df.groupby("isin")["date"].min()
    last = df.groupby("isin")["date"].max()
    sym_last = df.sort_values("date").groupby("isin")["symbol"].last()
    sym_first = df.sort_values("date").groupby("isin")["symbol"].first()
    dates = np.sort(df["date"].unique())
    nxt = {pd.Timestamp(a): pd.Timestamp(b) for a, b in zip(dates[:-1], dates[1:])}
    starts = {}
    for isin, d in first.items():
        starts.setdefault((pd.Timestamp(d), sym_first[isin]), []).append(isin)
    parent = {i: i for i in first.index}
    pos = {pd.Timestamp(d): k for k, d in enumerate(dates)}
    issuer_first = {}
    for isin, d in first.items():
        issuer_first.setdefault(isin[2:7], []).append((pos[pd.Timestamp(d)], isin))
    linked_to = set()
    for isin, d in last.items():
        nd = nxt.get(pd.Timestamp(d))
        if nd is None:
            continue
        cand = [c for c in starts.get((nd, sym_last[isin]), []) if c != isin]
        if len(cand) != 1:                                   # same issuer, re-listed within 60 sessions
            k = pos[pd.Timestamp(d)]
            cand = [c for p0, c in issuer_first.get(isin[2:7], []) if c != isin and k < p0 <= k + 60]
        if len(cand) == 1 and cand[0] not in linked_to:
            parent[cand[0]] = isin
            linked_to.add(cand[0])
    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i
    return df["isin"].map({i: root(i) for i in parent})


def load_panel() -> pd.DataFrame:
    return pd.read_parquet(PANEL)


EXTREME_LOG = np.log(1.6)       # |log return| beyond this (+60 % / -37.5 %) with no corporate-action record is checked
STANDARD_RATIOS = (1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0)


def adjust(df: pd.DataFrame, ca: pd.DataFrame | None = None) -> pd.DataFrame:
    """Adds cc / on / oc total returns adjusted with NSE corporate actions (see marketdata.nse_corp_actions)."""
    from marketdata.nse_corp_actions import daily_factors, load, rights_factor

    ca = load() if ca is None else ca
    f = daily_factors(ca)
    df = df[(df["prev_close"] > 0) & (df["close"] > 0)].copy()
    by_isin = f.rename(columns={"ex_date": "date"}).drop(columns="symbol").groupby(["isin", "date"]).first()
    by_sym = f.rename(columns={"ex_date": "date"}).drop(columns="isin").groupby(["symbol", "date"]).first()
    a = df[["isin", "date"]].merge(by_isin, left_on=["isin", "date"], right_index=True, how="left")
    b = df[["symbol", "date"]].merge(by_sym, left_on=["symbol", "date"], right_index=True, how="left")
    for c in ("price_factor", "dividend", "unadjusted", "rights_a", "rights_b", "rights_premium", "face_value"):
        df[c] = a[c].where(a["price_factor"].notna(), b[c]).values
    df["ca"] = df["price_factor"].notna()
    df["price_factor"] = df["price_factor"].fillna(1.0)
    df["dividend"] = df["dividend"].fillna(0.0)
    for c in ("rights_a", "rights_b", "rights_premium"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    # a factor that NSE already applied to prev_close: the ex-date open sits near prev_close instead of near prev_close*f
    pf = df["price_factor"].to_numpy()
    big = (pf < 0.8) | (pf > 1.25)
    with np.errstate(divide="ignore", invalid="ignore"):
        gap = np.abs(np.log(np.where(df["open"] > 0, df["open"], df["close"]) / df["prev_close"]))
    already = big & (gap < np.abs(np.log(np.where(big, pf, 1.0))) / 2)
    df["factor_skipped"] = already
    df.loc[already, "price_factor"] = 1.0
    rf = np.ones(len(df))
    hit = np.flatnonzero((df["rights_a"] > 0).to_numpy())
    for k in hit:
        r = df.iloc[k]
        fv = r["face_value"] if pd.notna(r["face_value"]) else None
        rf[k] = rights_factor(r["prev_close"], int(r["rights_a"]), int(r["rights_b"]), float(r["rights_premium"]), fv)
    df["factor"] = df["price_factor"] * rf
    base = df["prev_close"] * df["factor"]
    cash = df["dividend"] * df["price_factor"]
    df["cc"] = (df["close"] + cash) / base - 1
    df["on"] = np.where(df["open"] > 0, (df["open"] + cash) / base - 1, np.nan)
    df["oc"] = np.where(df["open"] > 0, df["close"] / df["open"] - 1, np.nan)
    df["flag"] = ""
    unpriced = df["unadjusted"].fillna("").str.contains("demerger|arrangement|capital_reduction|rights_unparsed")
    df.loc[unpriced, "flag"] = "unpriced_action"           # value moves to new securities / cash we cannot price:
    df.loc[unpriced, "on"] = 0.0                           # assume the holder's value is conserved overnight,
    df.loc[unpriced, "cc"] = df.loc[unpriced, "oc"].fillna(0.0)   # and keep the day's real move from the open
    # extreme moves on days with no corporate-action record: an unrecorded split/bonus if the price ratio is standard
    ext = ~df["ca"] & (np.log1p(df["cc"]).abs() > EXTREME_LOG)
    if ext.any():
        ratio = (df.loc[ext, "prev_close"] / df.loc[ext, "open"].where(df.loc[ext, "open"] > 0, df.loc[ext, "close"])).to_numpy()
        std = np.array(STANDARD_RATIOS)
        near = np.min(np.abs(np.log(ratio[:, None] / std[None, :])), axis=1) < np.log(1.07)
        idx = df.index[ext]
        df.loc[idx[near], "flag"] = "unrecorded_split"
        df.loc[idx[~near], "flag"] = "unexplained_extreme"
        best = std[np.argmin(np.abs(np.log(ratio[:, None] / std[None, :])), axis=1)]
        k = idx[near]
        base2 = df.loc[k, "prev_close"] / best[near]
        df.loc[k, "cc"] = df.loc[k, "close"] / base2 - 1
        df.loc[k, "on"] = np.where(df.loc[k, "open"] > 0, df.loc[k, "open"] / base2 - 1, np.nan)
        # not a standard ratio: a genuine market move (e.g. YES BANK, 6 Mar 2020) - kept, only flagged
    return df


def matrices(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """date x security matrices of adjusted returns: cc (close to close), on (prev close to open), oc (open to close),
    plus traded value."""
    out = {}
    if "cc" not in df.columns:
        df = adjust(df)
    for k in ("cc", "on", "oc", "value"):
        out[k] = df.pivot(index="date", columns="sec", values=k).astype("float32")
    return out


def liquidity_universe(value: pd.DataFrame, t: pd.Timestamp, n: int) -> list[str]:
    """Top-n securities by median traded value over the LOOKBACK sessions ending at t (inclusive)."""
    i = value.index.get_indexer([t])[0]
    if i < LOOKBACK - 1:
        return []
    w = value.iloc[i - LOOKBACK + 1:i + 1]
    traded = (w > 0).mean()
    listed = value.iloc[:i + 1].notna().sum() >= LOOKBACK
    ok = (traded >= MIN_TRADED) & listed & value.iloc[i].notna()
    med = w.where(w > 0).median()[ok]
    return list(med.sort_values(ascending=False).index[:n])
