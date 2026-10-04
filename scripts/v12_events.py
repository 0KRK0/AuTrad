"""v1.2 Forced Flow & Events (research/preregistration_v1.2.md):
HD  dividend-information incorporation (timestamped NSE announcements; event study + cash-and-carry trade),
HI  NIFTY 50 / NIFTY NEXT 50 inclusion-exclusion forced flow (anticipation drift, reversal, hedged futures trade),
HF  F&O ban measurement (limits to arbitrage; forced unwinding at expiry).

    python -m scripts.v12_events audit|run --out reports/v1.2_<ts>
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from marketdata.nse_events import load_announcements as _load_announcements
from marketdata.nse_fo import load_index
from scripts import v11_basis as B

START, END = B.START, B.END
EVENT_COLS = ["symbol", "ann_ts", "t0", "record_date", "ex_date", "payment_date", "D", "type", "action", "desc", "amount_match"]
FIELDS = {
    "HD": {"economic_mechanism": "futures should re-price by -D when a dividend becomes public; participants incorporate it at different speeds",
           "economic_counterparty": "futures longs and quoting desks using stale dividend assumptions",
           "mechanism_constraint": "information processing of unstructured, mostly after-close filings; cash-and-carry capital; STT on both cash legs",
           "our_advantage": "timestamped parsing of every filing; automated fair value with known dividends; AEM cost-aware entry",
           "failure_condition": "futures fully discount the dividend by the first close after the announcement, or the residual is below costs"},
    "HI": {"economic_mechanism": "passive trackers must buy inclusions and sell exclusions at the rebalance close",
           "economic_counterparty": "index funds and ETFs (and benchmarked funds) paying impact to minimise tracking error",
           "mechanism_constraint": "mandate forces rebalance-close trading; liquidity providers need capital and bear event risk",
           "our_advantage": "systematic parsing of press releases; execution at t0 close; futures for both directions",
           "failure_condition": "the move is complete before t0 close, or front-running is so crowded that the R-close impact vanishes"},
    "HF": {"economic_mechanism": "in an F&O ban no new derivative positions may be opened: arbitrage capital cannot enter and expiring positions cannot be rolled",
           "economic_counterparty": "holders of expiring positions in banned stocks; hedgers",
           "mechanism_constraint": "exchange rule (market-wide position limit)",
           "our_advantage": "none during the ban for a retail trader (measurement only)",
           "failure_condition": "no basis dislocation during bans and no abnormal open-interest unwinding at ban-at-expiry"}}


CACHE = Path("/tmp/autrad_v12")                  # local scratch (the repository mount is slow); rebuilt on demand


@lru_cache(maxsize=1)
def load_announcements() -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    c = CACHE / "announcements.pkl"
    if c.exists():
        return pd.read_pickle(c)
    a = _load_announcements()
    a.to_pickle(c)
    return a


# ------------------------------------------------------------------------------------------------ sessions & prices
@lru_cache(maxsize=1)
def sessions() -> pd.DatetimeIndex:
    """Trading calendar: F&O sessions plus cash sessions (NSE has no F&O file for 2021-03-30; Amendment 1)."""
    cs = B.cash()["date"]
    cs = cs[(cs >= START) & (cs <= END)].unique()
    return pd.DatetimeIndex(np.union1d(np.sort(B.futures()["date"].unique()), np.sort(cs)))


def first_close_after(ts: pd.Timestamp) -> pd.Timestamp | None:
    """t0: the first session whose closing price is formed after the announcement. NSE's closing price is the
    weighted average of 15:00-15:30, so a filing at or after 15:00 belongs to the next session (Amendment 1)."""
    s = sessions()
    d = pd.Timestamp(ts.tz_convert("Asia/Kolkata").date()) if ts.tzinfo else pd.Timestamp(ts.date())
    tm = ts.tz_convert("Asia/Kolkata").time() if ts.tzinfo else ts.time()
    k = s.searchsorted(d)
    if k < len(s) and s[k] == d and (tm.hour, tm.minute) < (15, 0):
        return s[k]
    k = s.searchsorted(d, side="right")
    return s[k] if k < len(s) else None


def next_session(d: pd.Timestamp, n: int = 1) -> pd.Timestamp | None:
    s = sessions()
    k = s.searchsorted(d) + n
    return s[k] if 0 <= k < len(s) else None


@lru_cache(maxsize=1)
def contract_panel() -> pd.DataFrame:
    """Near and next stock futures with the cash close, per (date, symbol, expiry)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    c_ = CACHE / "contract_panel.pkl"
    if c_.exists():
        return pd.read_pickle(c_)
    f = B.futures()
    c = B.contracts(f[f["kind"] == "STK"])
    cs = B.cash()[["date", "symbol", "close"]].rename(columns={"close": "S"})
    p = c.merge(cs, on=["date", "symbol"], how="inner").rename(columns={"close": "F", "value": "fval", "contracts": "fcon"})
    p["tau"] = (p["expiry"] - p["date"]).dt.days
    p = p[["date", "symbol", "expiry", "rank", "F", "S", "fval", "fcon", "tau", "oi"]].reset_index(drop=True)
    p.to_pickle(c_)
    return p


# ------------------------------------------------------------------------------------------------ dividend events
@lru_cache(maxsize=1)
def raw_corporate_dividends() -> pd.DataFrame:
    """Dividend rows of the NSE corporate-action file with ex-date, record date and amount."""
    rows = []
    for p in sorted(Path("data/exchange_all/corp_actions").glob("*.json")):
        for r in json.loads(p.read_text()):
            if "ividend" in (r.get("subject") or "") and r.get("series") in ("EQ", "BE"):
                rows.append({"symbol": r["symbol"], "ex_date": r.get("exDate"), "record_date": r.get("recDate"), "subject": r.get("subject")})
    d = pd.DataFrame(rows)
    d["ex_date"] = pd.to_datetime(d["ex_date"], format="%d-%b-%Y", errors="coerce")
    d["record_date"] = pd.to_datetime(d["record_date"], format="%d-%b-%Y", errors="coerce")
    ca = B.actions()[0].rename(columns={"dividend": "D"})                    # parsed amounts (one row per dividend)
    m = ca.merge(d.drop_duplicates(["symbol", "ex_date"]), on=["symbol", "ex_date"], how="left")
    return m[(m["ex_date"] >= "2015-01-01") & (m["ex_date"] <= "2027-01-01")]


@lru_cache(maxsize=1)
def dividend_events() -> pd.DataFrame:
    """Each corporate-action dividend linked to its earliest timestamped announcement with a matching amount."""
    c_ = CACHE / "dividend_events.pkl"
    if c_.exists():
        return pd.read_pickle(c_)
    out = _dividend_events()
    CACHE.mkdir(parents=True, exist_ok=True)
    out.to_pickle(c_)
    return out


def _dividend_events() -> pd.DataFrame:
    ann = load_announcements()
    ann = ann[~ann["negative"] & ann["ts"].notna()].copy()
    ann["ts"] = pd.to_datetime(ann["ts"], utc=True).dt.tz_convert("Asia/Kolkata")
    by = {s: g.sort_values("ts") for s, g in ann.groupby("symbol")}
    fo_syms = set(B.futures().query("kind == 'STK'")["symbol"].unique())
    ca = raw_corporate_dividends()
    ca = ca[ca["symbol"].isin(fo_syms) & (ca["ex_date"] >= START) & (ca["ex_date"] <= END)].sort_values(["symbol", "ex_date"])
    # the search window starts after the stock's previous ex-date, so a repeated amount cannot link to the last dividend
    prev = ca.drop_duplicates(["symbol", "ex_date"])[["symbol", "ex_date"]].copy()
    prev["prev_ex"] = prev.groupby("symbol")["ex_date"].shift(1)
    ca = ca.merge(prev, on=["symbol", "ex_date"], how="left")
    out = []
    for r in ca.itertuples(index=False):
        g = by.get(r.symbol)
        best, any_ann = None, False
        if g is not None:
            ex = pd.Timestamp(r.ex_date).tz_localize("Asia/Kolkata")
            lo = ex - pd.Timedelta(days=120)
            if pd.notna(r.prev_ex):
                lo = max(lo, pd.Timestamp(r.prev_ex).tz_localize("Asia/Kolkata") + pd.Timedelta(days=1))
            w = g[(g["ts"] >= lo) & (g["ts"] < ex)]
            any_ann = len(w) > 0
            for x in w.itertuples(index=False):
                if any(abs(a - r.D) <= 0.01 * max(r.D, 1e-9) + 1e-9 for a in x.amounts):
                    best = x
                    break
        row = {"symbol": r.symbol, "ex_date": pd.Timestamp(r.ex_date), "record_date": r.record_date, "D": float(r.D),
               "amount_match": best is not None, "any_announcement_in_window": any_ann}
        if best is not None:
            pay = pd.to_datetime(best.payment_text, errors="coerce") if best.payment_text else pd.NaT
            row.update({"ann_ts": best.ts, "t0": first_close_after(best.ts), "type": best.type, "action": best.action, "desc": best.desc,
                        "payment_date": pay})
        out.append(row)
    return pd.DataFrame(out)


def audit_dividends(ev: pd.DataFrame) -> dict:
    m = ev[ev["amount_match"]].copy()
    gap = (m["ex_date"] - m["t0"]).dt.days
    after = m["ann_ts"].map(lambda t: (t.hour, t.minute) >= (15, 30))
    yr = ev.groupby(ev["ex_date"].dt.year)["amount_match"].mean().round(3).to_dict()
    return {"ca_dividends_fo_symbols": int(len(ev)), "linked_with_amount_match": int(ev["amount_match"].sum()),
            "link_rate": float(ev["amount_match"].mean()), "link_rate_by_year": {int(k): v for k, v in yr.items()},
            "no_announcement_in_window": int((~ev["any_announcement_in_window"]).sum()),
            "share_announced_after_close": float(after.mean()),
            "gap_t0_to_ex_days": {q: float(gap.quantile(v)) for q, v in (("p10", .1), ("median", .5), ("p90", .9))},
            "types": m["type"].value_counts(dropna=False).to_dict(), "actions": m["action"].value_counts(dropna=False).to_dict(),
            "payment_date_parsed_share": float(m["payment_date"].notna().mean())}


# ------------------------------------------------------------------------------------------------ HD
def _fair_resid(F, S, r, tau):
    return (F - S * (1 + r * tau / 365)) / S


def hd_events() -> pd.DataFrame:
    """Event rows on t0: the contract covering the ex-date, residual mispricing with the announced dividend(s),
    matched controls on the same contract and day."""
    ev = dividend_events()
    ev = ev[ev["amount_match"] & ev["t0"].notna()].copy()
    p = contract_panel()
    by_day = {d: g for d, g in p.groupby("date")}
    _, adj = B.actions()
    rows = []
    known = ev.sort_values("ann_ts")
    for e in ev.itertuples(index=False):
        t0 = e.t0
        if not (e.ex_date > t0):
            continue
        day = by_day.get(t0)
        if day is None:
            continue
        q = day[(day["symbol"] == e.symbol) & (day["expiry"] >= e.ex_date)].sort_values("expiry")
        if q.empty:
            continue
        c = q.iloc[0]
        if c["tau"] > 65 or c["fval"] < B.MIN_FUT_VALUE:
            continue
        T = c["expiry"]
        # every dividend of this stock announced by t0's close with ex-date in (t0, T]
        kd = known[(known["symbol"] == e.symbol) & (known["t0"] <= t0) & (known["ex_date"] > t0) & (known["ex_date"] <= T)]
        D = float(kd["D"].sum())
        r = float(B.rate_on([t0])[0])
        u = _fair_resid(c["F"], c["S"], r, c["tau"])
        rows.append({"symbol": e.symbol, "t0": t0, "ann_ts": e.ann_ts, "ex_date": e.ex_date, "record_date": e.record_date, "expiry": T,
                     "D": D, "D_frac": D / c["S"], "F": c["F"], "S": c["S"], "tau": c["tau"], "r": r, "u": u, "m": u + D / c["S"],
                     "type": e.type, "action": e.action, "fval": c["fval"],
                     "extraordinary": D / c["S"] >= 0.02})
    x = pd.DataFrame(rows).drop_duplicates(["symbol", "t0", "expiry"]).reset_index(drop=True)
    x["adj_event"] = B._in_window(adj, x, "t0", "expiry", None) > 0
    # controls: same day and contract, liquid, no dividend announcement in the 45 days to t0, no dividend ex-date in (t0, T]
    ca = raw_corporate_dividends()
    ann_all = load_announcements()
    ann_all = ann_all[ann_all["ts"].notna()]
    ann_t = pd.to_datetime(ann_all["ts"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    ann_sym = ann_all["symbol"].to_numpy()
    ctrl_m = []
    for e in x.itertuples(index=False):
        day = by_day.get(e.t0)
        q = day[(day["expiry"] == e.expiry) & (day["fval"] >= B.MIN_FUT_VALUE) & (day["symbol"] != e.symbol)]
        bad = set(ca[(ca["ex_date"] > e.t0) & (ca["ex_date"] <= e.expiry)]["symbol"])
        hi_ = pd.Timestamp(e.t0) + pd.Timedelta(hours=15, minutes=30)
        bad |= set(ann_sym[((ann_t > hi_ - pd.Timedelta(days=45)) & (ann_t <= hi_)).to_numpy()])     # any dividend-related filing
        q = q[~q["symbol"].isin(bad)]
        ctrl_m.append(float(np.median(_fair_resid(q["F"], q["S"], e.r, q["tau"]))) if len(q) >= 5 else np.nan)
    x["control_m"] = ctrl_m
    x["excess_m"] = x["m"] - x["control_m"]
    # incorporation path: residual without the dividend before (t-1) and after (t0, t0+1), relative to controls
    return x


def incorporation_path(x: pd.DataFrame) -> dict:
    p = contract_panel()
    key = p.drop_duplicates(["date", "symbol", "expiry"]).set_index(["date", "symbol", "expiry"])
    out = {}
    for lab, shift in (("t-1", -1), ("t0", 0), ("t0+1", 1), ("t0+5", 5)):
        vals = []
        for e in x.itertuples(index=False):
            d = next_session(e.t0, shift)
            if d is None or (d, e.symbol, e.expiry) not in key.index or (shift > 0 and d >= e.ex_date):
                continue
            c = key.loc[(d, e.symbol, e.expiry)]
            u = _fair_resid(c["F"], c["S"], e.r, (e.expiry - d).days)
            vals.append((u - e.control_m) / e.D_frac * -1 if e.D_frac > 0 else np.nan)      # share of D discounted
        v = np.array([z for z in vals if not np.isnan(z)])
        out[lab] = {"median_share_discounted": float(np.median(v)) if len(v) else None, "n": int(len(v))}
    return out


@lru_cache(maxsize=1)
def ban_days() -> set:
    b = pd.read_csv("data/events/fo_ban.csv", parse_dates=["date"])
    return set(zip(b["symbol"], b["date"]))


def hd_trades(x: pd.DataFrame, mult: float = 1.0, fast: bool = False, include_extra: bool = False, sel_offset: int = 0) -> pd.DataFrame:
    """Cash-and-carry on announced dividends: e_cap > 0 at the t0 close; executed at the t0+1 close.
    fast=True selects AND fills at the same close (t0 + sel_offset); with sel_offset > 0 this is the same-close placebo
    used to separate speed value from the selection-on-noise of filling at the close used to select (Amendment 1).
    No new short future is opened on a day the stock is in an F&O ban."""
    p = contract_panel()
    key = p.drop_duplicates(["date", "symbol", "expiry"]).set_index(["date", "symbol", "expiry"])
    st = B.settle_on_expiry(B.futures(), "STK").drop_duplicates(["symbol", "expiry"]).set_index(["symbol", "expiry"])
    cs = B.cash().drop_duplicates(["date", "symbol"]).set_index(["date", "symbol"])["close"]
    ad = B.adjusted_dividends()
    divs = {sy: (g["ex_date"].to_numpy(), g["dividend"].to_numpy(float)) for sy, g in B.actions()[0].groupby("symbol")}
    adjs = {sy: (g["ex_date"].to_numpy(), g["dividend"].to_numpy(float)) for sy, g in ad[ad["adjusted"]].groupby("symbol")}

    def window_sum(d, sym, a, b):
        if sym not in d:
            return 0.0
        ex, v = d[sym]
        m = (ex > np.datetime64(a)) & (ex <= np.datetime64(b))
        return float(v[m].sum())
    notional = B.BOOK / B.TOPK / (1 + B.CAP_MARGIN)
    rows, open_until = [], {}
    y = x if include_extra else x[~x["extraordinary"]]
    y = y[~y["adj_event"]].sort_values("t0")
    banned = ban_days()
    for e in y.itertuples(index=False):
        ds = e.t0 if sel_offset == 0 else next_session(e.t0, sel_offset)
        if ds is None or ds >= e.ex_date or (ds, e.symbol, e.expiry) not in key.index:
            continue
        cs_ = key.loc[(ds, e.symbol, e.expiry)]
        Fs, Ss, taus = float(cs_["F"]), float(cs_["S"]), (e.expiry - ds).days
        c0 = sum(B.costs_bp(ds, e.expiry, notional, "STK", 1.0, Ss).values()) / 1e4
        e_cap = ((Fs - Ss + e.D) / Ss - c0) / (1 + B.CAP_MARGIN) * 365 / max(taus, 1) - e.r
        if e_cap <= 0:
            continue
        d1 = ds if fast else next_session(ds, 1)
        if d1 is not None and (e.symbol, d1) in banned:
            continue
        if d1 is None or d1 >= e.ex_date or d1 >= e.expiry or (d1, e.symbol, e.expiry) not in key.index:
            continue
        if e.symbol in open_until and d1 <= open_until[e.symbol]:
            continue
        c1 = key.loc[(d1, e.symbol, e.expiry)]
        if c1["fcon"] <= 0 or (e.symbol, e.expiry) not in st.index or (e.expiry, e.symbol) not in cs.index:
            continue
        F1, S1 = float(c1["F"]), float(c1["S"])
        FT, ST = float(st.loc[(e.symbol, e.expiry), "F_T"]), float(cs.loc[(e.expiry, e.symbol)])
        Drec = window_sum(divs, e.symbol, d1, e.expiry)
        Dadj = window_sum(adjs, e.symbol, d1, e.expiry)
        cb = B.costs_bp(d1, e.expiry, notional, "STK", mult, S1)
        days = (e.expiry - d1).days
        r1 = float(B.rate_on([d1])[0])
        cash_leg = (ST - S1 + Drec) / S1
        fut_leg = -(FT + Dadj - F1) / S1
        pnl = cash_leg + fut_leg - (cb["cash"] + cb["fut"]) / 1e4
        rows.append({"symbol": e.symbol, "t0": e.t0, "d1": d1, "expiry": e.expiry, "days": days, "D": e.D, "D_received": Drec,
                     "e_cap_signal": e_cap, "locked_at_entry": (F1 - S1 + e.D) / S1, "cash_leg": cash_leg, "fut_leg": fut_leg,
                     "cost_cash": cb["cash"] / 1e4, "cost_fut": cb["fut"] / 1e4, "pnl": pnl, "r": r1,
                     "ret_cap": pnl / (1 + B.CAP_MARGIN), "excess_cap": pnl / (1 + B.CAP_MARGIN) - r1 * days / 365,
                     "cap_frac": min(1.0, 0.01 * float(c1["fval"]) / notional), "type": e.type})
        open_until[e.symbol] = e.expiry
    return pd.DataFrame(rows)


def trade_summary(t: pd.DataFrame, t2: pd.DataFrame, beta_fn=None) -> dict:
    if t.empty:
        return {"trips": 0}
    t = t.copy()
    t["month"] = t["d1"].dt.to_period("M")
    mm = t.groupby("month")["excess_cap"].mean()
    m, tt, pv, n = B.nwt(mm.to_numpy())
    ann = float(t["excess_cap"].sum() / t["days"].sum() * 365)
    t["w"] = 1.0 * (1 + B.CAP_MARGIN)
    cash_int = float((t["r"] * t["days"] / 365).sum())
    after = {}
    for mode in ("prereg", "carry_forward", "business_cf"):
        tx = B.after_tax_excess(t, None, mode=mode)
        after[mode] = float(((sum(v["pre_tax"] for v in tx.values()) - sum(v["tax"] for v in tx.values())) - cash_int * (1 - B.TAX_CASH)) / t["days"].sum() * 365)
    subs = {}
    for k, (a, b) in B.SUBS.items():
        g = t[(t["d1"] >= a) & (t["d1"] <= b)]
        subs[k] = float(g["excess_cap"].sum() / g["days"].sum() * 365) if len(g) else None
    mret = t.groupby("month")["ret_cap"].mean()
    nm = B.nifty_month_returns(mret.index)
    ok = nm.notna().to_numpy()
    beta = float(np.polyfit(nm.to_numpy()[ok], mret.to_numpy()[ok], 1)[0]) if ok.sum() > 10 else float("nan")
    return {"trips": int(len(t)), "mean_month_excess_per_trip": m, "t": tt, "p": pv, "months": n,
            "mean_net_per_trip_bp": float(t["pnl"].mean() * 1e4), "mean_excess_per_trip_bp": float(t["excess_cap"].mean() * 1e4),
            "ann_excess_on_deployed_capital": ann, "after_tax_ann_excess": after["prereg"],
            "tax_variants_after_tax_ann_excess": {k: v for k, v in after.items() if k != "prereg"},
            "cost2x_ann_excess": float(t2["excess_cap"].sum() / t2["days"].sum() * 365) if len(t2) else None,
            "capacity_ann_excess": float((t["excess_cap"] * t["cap_frac"]).sum() / (t["days"] * t["cap_frac"]).sum() * 365),
            "subperiods_ann_excess": subs, "beta_to_nifty": beta, "mean_days": float(t["days"].mean()),
            "mean_locked_at_entry_bp": float(t["locked_at_entry"].mean() * 1e4), "mean_cost_bp": float((t["cost_cash"] + t["cost_fut"]).mean() * 1e4)}


# ------------------------------------------------------------------------------------------------ HI
@lru_cache(maxsize=1)
def index_events() -> pd.DataFrame:
    d = pd.read_csv("data/events/index_changes.csv", parse_dates=["announce_date", "effective_date"])
    d = d[d["index_key"].isin(["NIFTY50", "NIFTYNEXT50"])].drop_duplicates(["announce_date", "index_key", "action", "symbol"])
    # NEXT 50 events that are the mirror image of a NIFTY 50 change on the same announcement are NIFTY 50 events
    n50 = set(zip(d.loc[d["index_key"] == "NIFTY50", "announce_date"], d.loc[d["index_key"] == "NIFTY50", "symbol"]))
    mirror = np.array([(a, s) in n50 for a, s in zip(d["announce_date"], d["symbol"])], dtype=bool)
    d = d[~((d["index_key"] == "NIFTYNEXT50").to_numpy() & mirror)]
    # a release superseded by a later release for the same stock/index/action before the first effective date is dropped
    d = d.sort_values("announce_date")
    later = d.groupby(["index_key", "action", "symbol"])["announce_date"].shift(-1)
    sup = later.notna() & (later <= d["effective_date"])
    d = d[~sup.reindex(d.index).fillna(False)]
    s = sessions()
    d["t0"] = [s[s.searchsorted(a, side="right")] if s.searchsorted(a, side="right") < len(s) else pd.NaT for a in d["announce_date"]]
    d["R"] = [s[s.searchsorted(e) - 1] if pd.notna(e) and s.searchsorted(e) > 0 else pd.NaT for e in d["effective_date"]]
    d["sign"] = np.where(d["action"] == "include", 1, -1)
    return d[(d["t0"] >= START) & (d["R"] <= END) & (d["R"] > d["t0"])].reset_index(drop=True)


@lru_cache(maxsize=1)
def price_returns() -> tuple[pd.DataFrame, pd.Series]:
    r = B.cash()[["date", "symbol", "cc"]].pivot_table(index="date", columns="symbol", values="cc")
    ix = load_index()
    n = ix[ix["index"] == "NIFTY 50"].copy()
    n["date"] = pd.to_datetime(n["date"])
    m = n.set_index("date")["close"].sort_index().pct_change()
    return r, m


def car(sym: str, a, b) -> float | None:
    r, m = price_returns()
    if sym not in r.columns:
        return None
    rs = r.loc[(r.index > a) & (r.index <= b), sym]
    rm = m[(m.index > a) & (m.index <= b)]
    if rs.isna().any() or len(rs) == 0:
        return None
    return float(np.prod(1 + rs.to_numpy()) - np.prod(1 + rm.reindex(rs.index).fillna(0).to_numpy()))


def hi_tests(ev: pd.DataFrame, key: str) -> dict:
    x = ev[ev["index_key"] == key].copy()
    x["car1"] = [car(s, a, b) for s, a, b in zip(x["symbol"], x["t0"], x["R"])]
    x["R10"] = [next_session(d, 10) for d in x["R"]]
    x["car2"] = [car(s, a, b) if b is not None else None for s, a, b in zip(x["symbol"], x["R"], x["R10"])]
    x["car_ann_to_t0"] = [car(s, next_session(t, -1), t) for s, t in zip(x["symbol"], x["t0"])]      # announcement-day move (descriptive)
    out = {"events": int(len(x)), "inclusions": int((x["sign"] > 0).sum()), "exclusions": int((x["sign"] < 0).sum())}
    for k, col, sign in (("HI1", "car1", 1), ("HI2", "car2", -1)):
        y = x.dropna(subset=[col])
        g = (y[col] * y["sign"]).groupby(y["announce_date"]).mean()
        m, t, p, n = B.nwt((sign * g).to_numpy())
        out[k] = {"mean_signed_car": float(g.mean()), "t_in_hypothesis_direction": t, "p": p, "announcements": n, "events": int(len(y)),
                  "inclusions_mean_car": float(y.loc[y["sign"] > 0, col].mean()), "exclusions_mean_car": float(y.loc[y["sign"] < 0, col].mean())}
    y = x.dropna(subset=["car_ann_to_t0"])
    out["announcement_day_signed_car"] = float((y["car_ann_to_t0"] * y["sign"]).mean())
    out["mean_days_t0_to_R"] = float((x["R"] - x["t0"]).dt.days.mean())
    return out, x


def hi_trades(x: pd.DataFrame, mult: float = 1.0) -> pd.DataFrame:
    """Hedged futures trade over (t0, R]: stock future (contract with expiry >= R) vs NIFTY future (same rule)."""
    f = B.futures()
    k = f.drop_duplicates(["date", "symbol", "expiry"]).set_index(["date", "symbol", "expiry"])
    t0s = set(x["t0"])
    by_day = {d: g for d, g in f[f["date"].isin(t0s)].groupby("date")}
    rows = []
    for e in x.itertuples(index=False):
        def leg(sym, kind):
            day = by_day[e.t0]
            q = day[(day["symbol"] == sym) & (day["kind"] == kind) & (day["expiry"] >= e.R)].sort_values("expiry")
            if q.empty:
                return None
            T = q.iloc[0]["expiry"]
            if (e.R, sym, T) not in k.index:
                return None
            a, b = q.iloc[0], k.loc[(e.R, sym, T)]
            return float(a["close"]), float(b["close"]), float(a["value"]), float(a["contracts"])
        s_, n_ = leg(e.symbol, "STK"), leg("NIFTY", "IDX")
        if s_ is None or n_ is None or s_[3] <= 0:
            continue
        rs, rn = s_[1] / s_[0] - 1, n_[1] / n_[0] - 1
        notional = B.BOOK / max(1, int((x["t0"] == e.t0).sum()))
        brok = 20 * (1 + B.GST) / notional * 1e4
        cost_stk = (B.fut_stt(e.t0) + 0.2 + 2 * (B.TXN_FUT + B.SEBI) * (1 + B.GST) + 2 * brok + 2 * max(3.0, 0.5 * B.TICK / s_[0] * 1e4))
        cost_idx = (B.fut_stt(e.t0) + 0.2 + 2 * (B.TXN_FUT + B.SEBI) * (1 + B.GST) + 2 * brok + 2 * 1.0)
        pnl = e.sign * (rs - rn) - mult * (cost_stk + cost_idx) / 1e4
        days = (e.R - e.t0).days
        rows.append({"symbol": e.symbol, "announce_date": e.announce_date, "t0": e.t0, "d1": e.t0, "R": e.R, "expiry": e.R, "sign": e.sign, "rn": rn,
                     "days": days, "pnl": pnl, "excess_cap": pnl, "ret_cap": pnl, "r": 0.0, "cap_frac": min(1.0, 0.01 * s_[2] / notional),
                     "fut_leg": pnl + mult * (cost_stk + cost_idx) / 1e4, "cost_fut": mult * (cost_stk + cost_idx) / 1e4, "cash_leg": 0.0, "cost_cash": 0.0})
    return pd.DataFrame(rows)


def hi_summary(t: pd.DataFrame, t2: pd.DataFrame) -> dict:
    if t.empty:
        return {"trips": 0}
    ann = float(t["excess_cap"].sum() / t["days"].sum() * 365)
    after = {}
    for mode in ("prereg", "carry_forward", "business_cf"):
        tt = t.assign(w=1.0 * (1 + B.CAP_MARGIN))
        tx = B.after_tax_excess(tt, None, mode=mode)
        after[mode] = float((sum(v["pre_tax"] for v in tx.values()) - sum(v["tax"] for v in tx.values())) / t["days"].sum() * 365)
    subs = {}
    for k, (a, b) in B.SUBS.items():
        g = t[(t["t0"] >= a) & (t["t0"] <= b)]
        subs[k] = float(g["excess_cap"].sum() / g["days"].sum() * 365) if len(g) else None
    return {"trips": int(len(t)), "mean_net_per_trip_bp": float(t["pnl"].mean() * 1e4), "ann_excess_on_deployed_capital": ann,
            "after_tax_ann_excess": after["prereg"], "tax_variants_after_tax_ann_excess": {k: v for k, v in after.items() if k != "prereg"},
            "cost2x_ann_excess": float(t2["excess_cap"].sum() / t2["days"].sum() * 365) if len(t2) else None,
            "capacity_ann_excess": float((t["excess_cap"] * t["cap_frac"]).sum() / (t["days"] * t["cap_frac"]).sum() * 365),
            "subperiods_ann_excess": subs, "beta_to_nifty": float(np.polyfit(t["rn"], t["pnl"], 1)[0]) if len(t) > 10 else float("nan"),
            "beta_note": "slope of the hedged trade's P&L on the NIFTY futures return over the same window",
            "mean_days": float(t["days"].mean())}


# ------------------------------------------------------------------------------------------------ HF
def hf_tests() -> dict:
    ban = pd.read_csv("data/events/fo_ban.csv", parse_dates=["date"])
    oi = pd.read_csv("data/events/mwpl_oi.csv.gz", parse_dates=["date"])
    p = contract_panel()
    near = p[(p["rank"] == 0) & (p["tau"] >= 5)].copy()
    near["us"] = _fair_resid(near["F"], near["S"], B.rate_on(near["date"]), near["tau"]) * 1e4      # signed, bp
    near["u"] = near["us"].abs()
    s = sessions()
    pos = {d: i for i, d in enumerate(s)}
    ban = ban[ban["date"].isin(s)]
    ban["k"] = ban["date"].map(pos)
    ban = ban.sort_values(["symbol", "k"])
    ban["ep"] = (ban.groupby("symbol")["k"].diff() != 1).cumsum()
    eps = ban.groupby(["symbol", "ep"]).agg(start=("date", "min"), end=("date", "max"), n=("date", "size")).reset_index()
    u_by = {sym: g.set_index("date")["u"] for sym, g in near.groupby("symbol")}
    us_by = {sym: g.set_index("date")["us"] for sym, g in near.groupby("symbol")}
    diffs, after1, signed = [], [], []
    banned = set(zip(ban["symbol"], ban["date"]))
    for e in eps.itertuples(index=False):
        u = u_by.get(e.symbol)
        if u is None:
            continue
        a, b = pos[e.start], pos[e.end]
        win = [s[i] for i in range(max(0, a - 60), min(len(s), b + 61)) if (e.symbol, s[i]) not in banned]
        inb = u.reindex([s[i] for i in range(a, b + 1)]).dropna()
        out = u.reindex(win).dropna()
        if len(inb) and len(out) >= 20:
            diffs.append((e.start, e.symbol, float(inb.mean() - out.mean())))
            us = us_by[e.symbol]
            signed.append(float(us.reindex(inb.index).mean() - us.reindex(out.index).mean()))
            nxt = s[b + 1] if b + 1 < len(s) else None
            if nxt is not None and nxt in u.index:
                after1.append((e.start, float(u[nxt] - out.mean())))
    d = pd.DataFrame(diffs, columns=["start", "symbol", "diff"]).sort_values("start")
    m1, t1, p1, n1 = B.nwt(d["diff"].to_numpy())
    # robustness (Amendment 1): clustered by calendar month, median, without the 20 largest episodes, persistence test
    gm = d.groupby(d["start"].dt.to_period("M"))["diff"].mean()
    _, t1m, p1m, n1m = B.nwt(gm.to_numpy())
    trim = d.sort_values("diff").iloc[:-20] if len(d) > 40 else d
    a1 = pd.DataFrame(after1, columns=["start", "v"]).sort_values("start")
    ma, ta, pa, na = B.nwt(a1["v"].to_numpy()) if len(a1) > 10 else (None, None, None, 0)
    top = d.sort_values("diff", ascending=False).head(8)[["symbol", "start", "diff"]].astype(str).values.tolist()
    # HF2: ban during the last 3 sessions before an expiry -> OI change T-4 .. T vs the same stock's other expiries
    f = B.futures()
    exps = np.sort(f[(f["kind"] == "STK")]["expiry"].unique())
    exps = [pd.Timestamp(e) for e in exps if pd.Timestamp(e) in pos and pos[pd.Timestamp(e)] >= 4]
    oi_p = oi.pivot_table(index="date", columns="symbol", values="oi")
    rows = []
    for T in exps:
        k = pos[T]
        w3 = {s[k - 3], s[k - 2], s[k - 1]}
        t4 = s[k - 4]
        if T not in oi_p.index or t4 not in oi_p.index:
            continue
        chg = (oi_p.loc[T] / oi_p.loc[t4] - 1).dropna()
        for sym, v in chg.items():
            rows.append({"T": T, "symbol": sym, "chg": float(v), "ban": any((sym, d) in banned for d in w3)})
    r = pd.DataFrame(rows)
    med = r[~r["ban"]].groupby("symbol")["chg"].median()
    bb = r[r["ban"]].copy()
    bb["rel"] = bb["chg"] - bb["symbol"].map(med)
    bb = bb.dropna(subset=["rel"]).sort_values("T")
    m2, t2, p2, n2 = B.nwt((-bb["rel"]).to_numpy())          # hypothesis: rel < 0 -> test -rel > 0
    # robustness (Amendment 1): cluster by expiry; controls matched on the OI/MWPL level at T-4
    ge = bb.groupby("T")["rel"].mean()
    _, t2e, p2e, n2e = B.nwt((-ge).to_numpy())
    mw = oi.assign(lv=oi["oi"] / oi["mwpl"]).pivot_table(index="date", columns="symbol", values="lv")
    r["lv"] = [float(mw.at[s[pos[T] - 4], sy]) if (s[pos[T] - 4] in mw.index and sy in mw.columns) else np.nan for T, sy in zip(r["T"], r["symbol"])]
    bins = [0.75, 0.85, 0.95, 10]
    r["bin"] = pd.cut(r["lv"], bins, right=False)
    ctrl = r[~r["ban"]].groupby("bin", observed=True)["chg"].mean()
    bm = r[r["ban"] & r["bin"].notna()].copy()
    bm["rel_lv"] = bm["chg"] - bm["bin"].map(ctrl).astype(float)
    gl = bm.groupby("T")["rel_lv"].mean()
    ml, tl, pl, nl = B.nwt((-gl).to_numpy()) if len(gl) > 10 else (None, None, None, 0)
    bb["car"] = [car(sy, s[pos[T] - 4], T) for sy, T in zip(bb["symbol"], bb["T"])]
    cc = bb["car"].dropna()
    mc, tc, pc, nc = B.nwt(cc.to_numpy(), 3, two_sided=True) if len(cc) > 5 else (None, None, None, 0)
    gc = bb.dropna(subset=["car"]).groupby("T")["car"].mean()
    _, tce, pce, nce = B.nwt(gc.to_numpy(), 3, two_sided=True) if len(gc) > 5 else (None, None, None, 0)
    return {"HF1": {"mean_abs_resid_diff_bp": m1, "t": t1, "p": p1, "episodes": n1, "ban_episodes_total": int(len(eps)),
                    "median_episode_sessions": float(eps["n"].median()),
                    "robustness": {"median_diff_bp": float(d["diff"].median()), "t_month_clustered": t1m, "p_month_clustered": p1m, "months": n1m,
                                   "mean_without_top20_bp": float(trim["diff"].mean()), "signed_resid_diff_bp": float(np.mean(signed)),
                                   "persist_next_session": {"mean_bp": ma, "t": ta, "p": pa, "episodes": na},
                                   "largest_episodes": top}},
            "HF2": {"mean_rel_oi_change": float(bb["rel"].mean()), "t_for_negative": t2, "p": p2, "episodes": n2,
                    "median_rel_oi_change": float(bb["rel"].median()),
                    "robustness": {"t_expiry_clustered": t2e, "p_expiry_clustered": p2e, "expiries": n2e,
                                   "level_matched_mean_rel": ml, "level_matched_t_expiry_clustered": tl, "level_matched_p": pl},
                    "car_T-4_to_T": {"mean": mc, "t": tc, "p_two_sided": pc, "n": nc, "t_expiry_clustered": tce, "p_expiry_clustered": pce}}}


# ------------------------------------------------------------------------------------------------ audit / run
def audit(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    ev = dividend_events()
    a = {"dividends": audit_dividends(ev)}
    d = pd.read_csv("data/events/index_changes.csv", parse_dates=["announce_date", "effective_date"])
    s = pd.read_csv("marketdata/reference/nifty50_pit_snapshots.csv")
    dates = sorted(s["effective_date"].unique())
    ch = []
    for x, y in zip(dates[:-1], dates[1:]):
        A_, B_ = set(s[s["effective_date"] == x]["symbol"]), set(s[s["effective_date"] == y]["symbol"])
        ch += [(y, "include", z) for z in B_ - A_] + [(y, "exclude", z) for z in A_ - B_]
    ch = pd.DataFrame(ch, columns=["snap", "action", "symbol"])
    n50 = d[d["index_key"] == "NIFTY50"]
    mm = ch.merge(n50[["symbol", "action"]].drop_duplicates(), on=["symbol", "action"], how="left", indicator=True)
    ie = index_events()
    a["index_changes"] = {"press_releases_parsed": int(pd.read_csv("data/events/index_press_releases.csv").shape[0]),
                          "nifty50_rows": int(len(n50)), "nifty50_announcements": int(n50["announce_date"].nunique()),
                          "snapshot_changes": int(len(ch)), "snapshot_changes_found_in_press_releases": int((mm["_merge"] == "both").sum()),
                          "snapshot_changes_not_found": mm.loc[mm["_merge"] != "both", ["snap", "action", "symbol"]].astype(str).values.tolist(),
                          "events_in_window": ie.groupby("index_key").size().to_dict(),
                          "missing_effective_date": int(d["effective_date"].isna().sum())}
    ban = pd.read_csv("data/events/fo_ban.csv")
    man = json.loads(Path("data/events/ban_manifest.json").read_text())
    a["fo_ban"] = {"ban_rows": int(len(ban)), "symbols": int(ban["symbol"].nunique()), "days_covered": len(man["done"]),
                   "no_ban_file_days": len(man["no_ban_file"]), "no_oi_file_days": len(man["no_oi_file"])}
    (out / "audit.json").write_text(json.dumps(a, indent=1, default=str))
    return a


def run(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    res = {"fields": FIELDS}
    # ---- HD
    x = hd_events()
    xx = x[~x["extraordinary"] & ~x["adj_event"]].dropna(subset=["control_m"]).copy()
    xx["month"] = pd.to_datetime(xx["t0"]).dt.to_period("M")
    g = xx.groupby("month")["excess_m"].mean()
    m, t, p, n = B.nwt(g.to_numpy())
    lo_, hi_ = xx["excess_m"].quantile([0.05, 0.95])
    res["HD1"] = {"events": int(len(xx)), "months": n, "mean_excess_m_bp": float(g.mean() * 1e4), "t": t, "p": p,
                  "median_event_excess_m_bp": float(xx["excess_m"].median() * 1e4), "mean_D_bp": float(xx["D_frac"].mean() * 1e4),
                  "share_positive": float((xx["excess_m"] > 0).mean()),
                  "trimmed_mean_5_95_bp": float(xx.loc[xx["excess_m"].between(lo_, hi_), "excess_m"].mean() * 1e4),
                  "by_type_mean_excess_m_bp": (xx.groupby(xx["type"].fillna("unknown"))["excess_m"].mean() * 1e4).round(2).to_dict(),
                  "incorporation_path": incorporation_path(xx)}
    x.to_csv(out / "hd_events.csv", index=False)
    t1 = hd_trades(x)
    t1x2 = hd_trades(x, mult=2.0)
    res["HD2"] = trade_summary(t1, t1x2)
    tf = hd_trades(x, fast=True)
    fast = {"t0": trade_summary(tf, pd.DataFrame())}
    for k in (1, 3, 5):                                   # same-close placebos: select and fill at the same later close
        tk = hd_trades(x, fast=True, sel_offset=k)
        fast[f"placebo_t0+{k}"] = trade_summary(tk, pd.DataFrame()) if len(tk) else {"trips": 0}
    res["HD2"]["descriptive_same_close_fill"] = {k: {kk: v.get(kk) for kk in ("trips", "ann_excess_on_deployed_capital", "t", "mean_locked_at_entry_bp")}
                                                 for k, v in fast.items()}
    te = hd_trades(x, include_extra=True)
    res["HD2"]["descriptive_including_extraordinary"] = {"trips": int(len(te)), "ann_excess": float(te["excess_cap"].sum() / te["days"].sum() * 365) if len(te) else None}
    t1.to_csv(out / "hd2_trades.csv", index=False)
    # ---- HI
    ie = index_events()
    res["HI"] = {}
    hi_x = {}
    for key in ("NIFTY50", "NIFTYNEXT50"):
        r, xk = hi_tests(ie, key)
        # sensitivity: drop announcements with an exclusion caused by a corporate event (no price 10 sessions after R)
        ce = xk[(xk["sign"] < 0) & xk["car2"].isna()]["announce_date"].unique()
        r2, _ = hi_tests(ie[~((ie["index_key"] == key) & ie["announce_date"].isin(ce))], key)
        r["sensitivity_without_corporate_event_exclusions"] = {"announcements_dropped": [str(a)[:10] for a in ce],
                                                               "HI1": r2["HI1"], "HI2": r2["HI2"]}
        res["HI"][key] = r
        hi_x[key] = xk
        xk.to_csv(out / f"hi_{key}.csv", index=False)
    tr = hi_trades(hi_x["NIFTY50"])
    res["HI_econ"] = hi_summary(tr, hi_trades(hi_x["NIFTY50"], mult=2.0))
    res["HI_econ"]["descriptive_NIFTYNEXT50"] = hi_summary(hi_trades(hi_x["NIFTYNEXT50"]), pd.DataFrame())
    tr.to_csv(out / "hi_trades.csv", index=False)
    # ---- HF
    res["HF"] = hf_tests()
    # ---- Holm across 8
    ps = {"HD1": res["HD1"]["p"], "HD2": res["HD2"].get("p", 1.0), "HI1-N50": res["HI"]["NIFTY50"]["HI1"]["p"], "HI2-N50": res["HI"]["NIFTY50"]["HI2"]["p"],
          "HI1-NN50": res["HI"]["NIFTYNEXT50"]["HI1"]["p"], "HI2-NN50": res["HI"]["NIFTYNEXT50"]["HI2"]["p"], "HF1": res["HF"]["HF1"]["p"], "HF2": res["HF"]["HF2"]["p"]}
    hp = B.holm(ps)
    res["holm"] = hp

    def gate(r, holm_p, positive):
        ae = r.get("ann_excess_on_deployed_capital")
        if ae is None:
            return {"verdict": "no trades"}
        b = r.get("beta_to_nifty")
        steps = [("statistical", holm_p < 0.05 and positive), ("excess_ge_1.5pct_and_mean_net_positive", ae >= 0.015 and r["mean_net_per_trip_bp"] > 0),
                 ("after_tax_excess_ge_1pct", r["after_tax_ann_excess"] >= 0.01), ("cost2x_excess_ge_0.75pct", (r["cost2x_ann_excess"] or -1) >= 0.0075),
                 ("beta_abs_lt_0.10", b is not None and not np.isnan(b) and abs(b) < 0.10), ("capacity", r["capacity_ann_excess"] >= 0.015),
                 ("subperiods_2_of_3", sum(1 for v in r["subperiods_ann_excess"].values() if v is not None and v > 0) >= 2)]
        failed = next((k for k, ok in steps if not ok), None)
        return {"holm_p": holm_p, "steps": {k: bool(v) for k, v in steps}, "first_failed": failed,
                "verdict": "PASS -> forward paper candidate" if failed is None else f"REJECTED at {failed}"}
    res["gate"] = {"HD2": gate(res["HD2"], hp["HD2"], (res["HD2"].get("mean_month_excess_per_trip") or -1) > 0),
                   "HI_econ": gate(res["HI_econ"], hp["HI1-N50"], res["HI"]["NIFTY50"]["HI1"]["mean_signed_car"] > 0)}
    (out / "v12.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["audit", "run"])
    ap.add_argument("--out", default="reports/v1.2_dev")
    a = ap.parse_args()
    r = audit(Path(a.out)) if a.cmd == "audit" else run(Path(a.out))
    print(json.dumps(r, indent=1, default=str)[:5000])


if __name__ == "__main__":
    main()
