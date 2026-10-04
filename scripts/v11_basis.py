"""v1.1 Indian Basis & Relative-Value Lab (research/preregistration_v1.1.md): HB1 monthly stock cash-and-carry book,
HB2 one-day stock basis dislocations, HB3 index carry via ETF (NIFTY/NIFTYBEES, BANKNIFTY/BANKBEES), HB4 roll-window
calendar spread. Signal at t, execution at t+1 closes, exit at expiry (futures at the final settlement price).

    python -m scripts.v11_basis audit|run --out reports/v1.1_<ts>
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from lab.v05 import nw_ols
from marketdata.nse_corp_actions import load as load_ca
from marketdata.nse_fo import load_fo, load_index

START, END = "2015-01-01", "2026-09-30"
CAP_MARGIN = 0.20
BOOK = 1e7                       # ₹1 crore
TOPK = 10
HB2_TH = 0.10
MIN_FUT_VALUE = 1e8              # ₹10 crore on the signal day
TAX_BUS, TAX_CG, TAX_CASH = 0.312, 0.208, 0.312
PHYSICAL_FROM = pd.Timestamp("2019-10-01")       # all stock futures physically settled (phased in from Jul 2018)
CASH_SENSITIVITY = {"on": False}                 # descriptive: liquid-fund rate below repo in May 2020 - Apr 2022


def cg_rate(d) -> float:
    """Short-term capital-gains rate incl. cess: 15 % (15.6 %) until 22 Jul 2024, 20 % (20.8 %) from 23 Jul 2024."""
    return 0.156 if pd.Timestamp(d) < pd.Timestamp("2024-07-23") else 0.208
ETF = {"NIFTY": ("NIFTYBEES", "NIFTY 50"), "BANKNIFTY": ("BANKBEES", "NIFTY BANK")}
SUBS = {"2015-18": ("2015-01-01", "2018-12-31"), "2019-22": ("2019-01-01", "2022-12-31"), "2023-26": ("2023-01-01", "2026-12-31")}
A = Path("research/data_audits")


# ------------------------------------------------------------------------------------------------ inputs
def repo() -> pd.Series:
    r = pd.read_csv("marketdata/reference/rbi_repo.csv", parse_dates=["effective"])
    return r.set_index("effective")["repo_pct"] / 100


def rate_on(dates: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    r = repo()
    dd = pd.to_datetime(dates).values
    idx = np.searchsorted(r.index.values, dd, side="right") - 1
    out = r.values[np.maximum(idx, 0)].astype(float)
    if CASH_SENSITIVITY["on"]:                    # reverse repo 3.35 % (22 May 2020 - 7 Apr 2022) instead of repo 4.0 %
        m = (dd >= np.datetime64("2020-05-22")) & (dd < np.datetime64("2022-04-08"))
        out = np.where(m, out - 0.0065, out)
    return out


def fut_stt(d) -> float:
    d = pd.Timestamp(d)
    return 1.0 if d < pd.Timestamp("2023-04-01") else 1.25 if d < pd.Timestamp("2024-10-01") else 2.0 if d < pd.Timestamp("2026-04-01") else 5.0


GST = 0.18
TXN_EQ, TXN_FUT, SEBI = 0.307, 0.183, 0.01         # bp (SEBI ₹10 per crore = 0.01 bp)


TICK = 0.05                                          # ₹, NSE equity and stock-futures tick (conservative throughout)


def costs_bp(d1, dT, notional: float, kind: str, mult: float = 1.0, price: float | None = None) -> dict:
    """Round-trip cost of a cash-and-carry trip in bp of the cash-leg notional, split by leg (for tax).
    Half-spreads are floored at half a tick (Amendment 1): a ₹5 stock cannot trade inside a 100 bp tick."""
    hs_cash, hs_fut = (3.0, 3.0) if kind == "STK" else (4.0, 1.0)
    if price is not None and kind == "STK":
        floor = 0.5 * TICK / price * 1e4
        hs_cash, hs_fut = max(hs_cash, floor), max(hs_fut, floor)
    brok = 20 * (1 + GST) / notional * 1e4                                   # ₹20 + GST per futures order
    dp = 15.34 / notional * 1e4
    if kind == "STK":
        cash_buy = 10 + 1.5 + (TXN_EQ + SEBI) * (1 + GST) + hs_cash
        # physically settled: the stock is delivered at the final settlement price, no spread (Amendment 1)
        cash_sell = 10 + (TXN_EQ + SEBI) * (1 + GST) + dp + (0.0 if pd.Timestamp(dT) >= PHYSICAL_FROM else hs_cash)
    else:                                                                    # ETF: STT 0.001 % on sells only
        cash_buy = 1.5 + (TXN_EQ + SEBI) * (1 + GST) + hs_cash
        cash_sell = 0.1 + (TXN_EQ + SEBI) * (1 + GST) + dp + hs_cash
    fut_sell = fut_stt(d1) + (TXN_FUT + SEBI) * (1 + GST) + brok + hs_fut
    fut_exit = 0.0                                                           # settles at the final settlement price (Amendment 1)
    return {"cash": mult * (cash_buy + cash_sell), "fut": mult * (fut_sell + fut_exit)}


@lru_cache(maxsize=1)
def futures() -> pd.DataFrame:
    f = load_fo()
    f["date"] = pd.to_datetime(f["date"])
    f["expiry"] = pd.to_datetime(f["expiry"])
    return normalise_expiry(f[(f["date"] >= START) & (f["date"] <= END)].copy())


def normalise_expiry(f: pd.DataFrame) -> pd.DataFrame:
    """One monthly futures contract per (kind, symbol, month). When NSE moved an expiry (holiday shifts such as June
    2023, the 2025 change of expiry weekday) the same contract appears under two expiry labels; every row gets the
    contract's actual last trading day (contracts still alive at the end of the sample keep their listed expiry)."""
    key = [f["kind"], f["symbol"], f["expiry"].dt.to_period("M")]
    last = f.groupby(key)["date"].transform("max")
    listed = f.groupby(key)["expiry"].transform("max")
    f["expiry"] = np.where((listed > pd.Timestamp(END)) | (last >= listed), listed, last)
    f["expiry"] = pd.to_datetime(f["expiry"])
    return f


def contracts(f: pd.DataFrame) -> pd.DataFrame:
    """Near (rank 0) and next (rank 1) contract per (date, symbol), expiry >= date."""
    g = f[f["expiry"] >= f["date"]].sort_values(["date", "symbol", "expiry"])
    g["rank"] = g.groupby(["date", "symbol"]).cumcount()
    return g[g["rank"] <= 1]


@lru_cache(maxsize=1)
def cash() -> pd.DataFrame:
    r = pd.read_parquet("data/exchange_all/returns.parquet", columns=["date", "symbol", "close", "cc", "value"])
    r["date"] = pd.to_datetime(r["date"])
    return r


@lru_cache(maxsize=1)
def actions() -> tuple[pd.DataFrame, pd.DataFrame]:
    c = load_ca()
    div = c[c["dividend"] > 0][["symbol", "ex_date", "dividend"]]
    adj = c[(c["price_factor"] != 1) | (c["unadjusted"].fillna("") != "") | (c["rights_a"] > 0)][["symbol", "ex_date"]]
    return div, adj


def _in_window(events: pd.DataFrame, trips: pd.DataFrame, a: str, b: str, val: str | None) -> np.ndarray:
    """Sum of events[val] (or the count) per trip with event ex_date in (trip[a], trip[b]] (vectorised per symbol)."""
    out = np.zeros(len(trips))
    ev = {s: g.sort_values("ex_date") for s, g in events.groupby("symbol")}
    pos = np.arange(len(trips))
    for s, idx in pd.Series(pos).groupby(trips["symbol"].to_numpy()):
        g = ev.get(s)
        if g is None:
            continue
        ex = g["ex_date"].to_numpy()
        cum = np.r_[0.0, np.cumsum(g[val].to_numpy(float) if val else np.ones(len(g)))]
        ia = np.searchsorted(ex, trips[a].to_numpy()[idx.to_numpy()], side="right")
        ib = np.searchsorted(ex, trips[b].to_numpy()[idx.to_numpy()], side="right")
        out[idx.to_numpy()] = cum[ib] - cum[ia]
    return out


# ------------------------------------------------------------------------------------------------ stock panel
@lru_cache(maxsize=1)
def stock_panel() -> pd.DataFrame:
    f = futures()
    c = contracts(f[f["kind"] == "STK"])
    near = c[c["rank"] == 0].rename(columns={"close": "F", "value": "fval", "contracts": "fcon", "settle": "Fset"})
    cs = cash()
    p = near.merge(cs[["date", "symbol", "close", "value"]].rename(columns={"close": "S", "value": "sval"}), on=["date", "symbol"], how="inner")
    p["tau"] = (p["expiry"] - p["date"]).dt.days
    return p[["date", "symbol", "expiry", "F", "Fset", "fval", "fcon", "S", "sval", "tau"]]


def settle_on_expiry(f: pd.DataFrame, kind: str) -> pd.DataFrame:
    e = f[(f["kind"] == kind) & (f["date"] == f["expiry"])][["symbol", "expiry", "settle", "close"]]
    return e.rename(columns={"settle": "F_T", "close": "F_T_close"})


@lru_cache(maxsize=1)
def adjusted_dividends() -> pd.DataFrame:
    """Dividends whose futures contracts were adjusted by NSE ("extraordinary" dividends). NSE's threshold changed over
    time, so it is detected from the data: for a dividend >= 2 % of the stock price, the contract was adjusted if the
    near-month basis (F - S) did not rise by at least half the dividend on the ex-date (an unadjusted future already
    discounts the dividend, so F - S jumps by about D when the stock goes ex)."""
    p = stock_panel()
    div, _ = actions()
    sess = np.sort(p["date"].unique())
    prev = dict(zip(sess[1:], sess[:-1]))
    q = p.set_index(["date", "symbol"])[["F", "S", "expiry"]]
    rows = []
    for sym, ex, D in div[(div["ex_date"] >= START) & (div["ex_date"] <= END)].itertuples(index=False):
        ex = pd.Timestamp(ex)
        if ex not in prev or (ex, sym) not in q.index or (prev[ex], sym) not in q.index:
            continue
        a, b = q.loc[(prev[ex], sym)], q.loc[(ex, sym)]
        if a["expiry"] != b["expiry"] or D / a["S"] < 0.02:
            continue
        jump = (b["F"] - b["S"]) - (a["F"] - a["S"])
        rows.append({"symbol": sym, "ex_date": ex, "dividend": D, "jump_over_D": jump / D, "adjusted": bool(jump < 0.5 * D)})
    return pd.DataFrame(rows, columns=["symbol", "ex_date", "dividend", "jump_over_D", "adjusted"])


def stock_trips(p: pd.DataFrame, entries: pd.DataFrame, mult: float = 1.0, margin: float = CAP_MARGIN,
                notional: float = BOOK / TOPK / (1 + CAP_MARGIN)) -> pd.DataFrame:
    """entries: rows (signal date t, symbol, expiry) -> executed at the next session t+1, exit at expiry."""
    sess = np.sort(p["date"].unique())
    nxt = dict(zip(sess[:-1], sess[1:]))
    e = entries.copy()
    e["d1"] = e["date"].map(nxt)
    e = e.dropna(subset=["d1"])
    x = e.merge(p[["date", "symbol", "expiry", "F", "S", "fcon", "fval"]].rename(columns={"date": "d1", "F": "F1", "S": "S1", "fcon": "fcon1", "fval": "fval1"}),
                on=["d1", "symbol", "expiry"], how="inner")
    x = x[(x["fcon1"] > 0) & (x["d1"] < x["expiry"])]
    f = futures()
    x = x.merge(settle_on_expiry(f, "STK"), on=["symbol", "expiry"], how="inner")
    cs = cash()
    x = x.merge(cs[["date", "symbol", "close"]].rename(columns={"date": "expiry", "close": "S_T"}), on=["expiry", "symbol"], how="inner")
    div, adj = actions()
    x["D"] = _in_window(div, x, "d1", "expiry", "dividend")
    ad = adjusted_dividends()
    x["D_adj"] = _in_window(ad[ad["adjusted"]], x, "d1", "expiry", "dividend")       # contract re-based by NSE
    x["adj_events"] = _in_window(adj.assign(n=1), x, "date", "expiry", None)
    x = x[x["adj_events"] == 0]
    x["days"] = (x["expiry"] - x["d1"]).dt.days
    x["cash_leg"] = (x["S_T"] - x["S1"] + x["D"]) / x["S1"]
    x["fut_leg"] = -(x["F_T"] + x["D_adj"] - x["F1"]) / x["S1"]
    cb = [costs_bp(a, b, notional, "STK", mult, px) for a, b, px in zip(x["d1"], x["expiry"], x["S1"])]
    x["cost_cash"] = [c["cash"] / 1e4 for c in cb]
    x["cost_fut"] = [c["fut"] / 1e4 for c in cb]
    x["pnl"] = x["cash_leg"] + x["fut_leg"] - x["cost_cash"] - x["cost_fut"]        # per unit notional
    x["r"] = rate_on(x["d1"])
    x["ret_cap"] = x["pnl"] / (1 + margin)
    x["excess_cap"] = x["ret_cap"] - x["r"] * x["days"] / 365
    x["locked_at_entry"] = (x["F1"] - x["S1"]) / x["S1"]
    x["cap_frac"] = np.minimum(1.0, 0.01 * x["fval1"] / notional)               # capacity: <= 1 % of futures value
    return x


def net_excess_signal(p: pd.DataFrame, div: pd.DataFrame, notional: float = BOOK / TOPK / (1 + CAP_MARGIN),
                      include_div: bool = False) -> pd.DataFrame:
    """e = (g - C_rt) * 365/tau - r. Amended (Amendment 1): g = (F - S)/S. include_div=True reproduces the
    pre-registered g = (F - S + D)/S with D = dividends with ex-date in (t, T] (excluding NSE-adjusted ones), which
    uses ex-dates of dividends that may not yet have been announced at t (look-ahead)."""
    p = p.copy()
    if include_div:
        ad = adjusted_dividends()
        d = div.merge(ad[ad["adjusted"]][["symbol", "ex_date"]], on=["symbol", "ex_date"], how="left", indicator=True)
        d = d[d["_merge"] == "left_only"].drop(columns="_merge")
        p["D"] = _in_window(d, p, "date", "expiry", "dividend")
    else:
        p["D"] = 0.0
    p["g"] = (p["F"] - p["S"] + p["D"]) / p["S"]
    c = np.array([sum(costs_bp(a, a, notional, "STK", 1.0, px).values()) for a, px in zip(p["date"], p["S"])]) / 1e4
    r = rate_on(p["date"])
    tau = p["tau"].clip(lower=1)
    p["e"] = (p["g"] - c) * 365 / tau - r
    p["e_cap"] = (p["g"] - c) / (1 + CAP_MARGIN) * 365 / tau - r          # descriptive: excess on capital incl. margin
    p["carry_ann"] = p["g"] * 365 / tau
    return p


# ------------------------------------------------------------------------------------------------ statistics
def nwt(v, lags=3, two_sided=False):
    v = np.asarray([x for x in v if x is not None and not np.isnan(x)], float)
    fit = nw_ols(v, np.empty((len(v), 0)), lags=lags)
    t = float(fit["t"][0])
    p = float(2 * sps.t.sf(abs(t), len(v) - 1)) if two_sided else float(sps.t.sf(t, len(v) - 1))
    return float(v.mean()), t, p, len(v)


def holm(ps: dict) -> dict:
    keys = sorted(ps, key=ps.get)
    out, run = {}, 0.0
    for r, k in enumerate(keys):
        run = max(run, min(1.0, ps[k] * (len(ps) - r)))
        out[k] = run
    return out


def fy(d: pd.Timestamp) -> int:
    return d.year if d.month >= 4 else d.year - 1


def after_tax_excess(trips: pd.DataFrame, cycles: pd.DataFrame | None = None, weight_col: str = "w", mode: str = "prereg") -> dict:
    """Per financial year: B = futures-leg P&L (business income, 31.2 %), C = cash-leg P&L (short-term capital gain,
    15.6 % until 22 Jul 2024, 20.8 % after). Units: fraction of the book's capital.
    mode "prereg": a business loss is set off against capital gains of the same year; a capital loss is not set off
                   against business income; no carry-forward.
    mode "carry_forward": as prereg, plus losses carried forward (business losses against later business income,
                   capital losses against later capital gains; the 8-year limit never binds in this sample).
    mode "business_cf": both legs as business income (how most arbitrageurs file), losses carried forward."""
    t = trips.copy()
    t["FY"] = t["expiry"].map(fy)
    t["C"] = (t["cash_leg"] - t["cost_cash"]) * t[weight_col] / (1 + CAP_MARGIN)
    t["B"] = (t["fut_leg"] - t["cost_fut"]) * t[weight_col] / (1 + CAP_MARGIN)
    t["rate"] = t["expiry"].map(cg_rate)
    out, cfB, cfC = {}, 0.0, 0.0
    for y in sorted(t["FY"].unique()):
        g = t[t["FY"] == y]
        B, C = float(g["B"].sum()), float(g["C"].sum())
        pos = g["C"].clip(lower=0)
        rC = float((g["rate"] * pos).sum() / pos.sum()) if pos.sum() > 0 else float(g["rate"].iloc[-1])
        pre = B + C
        if mode == "business_cf":
            tot = B + C
            if tot > 0:
                use = min(cfB, tot)
                tot, cfB = tot - use, cfB - use
                tax = TAX_BUS * tot
            else:
                cfB += -tot
                tax = 0.0
        else:
            if B < 0 < C:
                off = min(-B, C)
                B, C = B + off, C - off
            if mode == "carry_forward":
                if B > 0:
                    use = min(cfB, B)
                    B, cfB = B - use, cfB - use
                if C > 0:
                    use = min(cfC, C)
                    C, cfC = C - use, cfC - use
                cfB += max(-B, 0.0)
                cfC += max(-C, 0.0)
            tax = TAX_BUS * max(B, 0.0) + rC * max(C, 0.0)
        out[int(y)] = {"pre_tax": pre, "tax": tax}
    return out


# ------------------------------------------------------------------------------------------------ studies
def monthly_expiries(p: pd.DataFrame) -> np.ndarray:
    """The common monthly stock-futures expiry calendar: expiries shared by >= 50 symbols (contracts terminated early,
    e.g. on a merger, carry their own last trading day and are not cycle boundaries)."""
    n = p.groupby("expiry")["symbol"].nunique()
    return np.sort(n[n >= 50].index.values)


def hb1(p: pd.DataFrame, mult: float = 1.0, topk: int = TOPK, margin: float = CAP_MARGIN, include_div: bool = False,
        capital_select: bool = False) -> dict:
    div, _ = actions()
    exp = monthly_expiries(p)
    sess = np.sort(p["date"].unique())
    d0s = np.unique([sess[np.searchsorted(sess, e, side="right")] for e in exp if np.searchsorted(sess, e, side="right") < len(sess)])
    sig = p[p["date"].isin(d0s) & (p["fval"] >= MIN_FUT_VALUE)]
    sig = net_excess_signal(sig, div, BOOK / topk / (1 + margin), include_div)
    key = "e_cap" if capital_select else "e"
    sig = sig[sig[key] > 0].sort_values(["date", key], ascending=[True, False]).groupby("date").head(topk)
    tr = stock_trips(p, sig[["date", "symbol", "expiry", "e", "g"]], mult, margin, BOOK / topk / (1 + margin))
    cyc = []
    for d0 in d0s:
        g = tr[tr["date"] == d0]
        # cycle: from d1 to the near expiry of that cycle
        nxt_exp = exp[np.searchsorted(exp, d0)] if np.searchsorted(exp, d0) < len(exp) else None
        if nxt_exp is None or pd.Timestamp(nxt_exp) > pd.Timestamp(END):
            continue
        d1 = sess[np.searchsorted(sess, d0, side="right")]
        days = (pd.Timestamp(nxt_exp) - pd.Timestamp(d1)).days
        r = float(rate_on([d1])[0])
        w = np.full(len(g), 1 / topk)
        ret = float((w * g["ret_cap"]).sum()) + (1 - w.sum()) * r * days / 365
        cyc.append({"d0": pd.Timestamp(d0), "expiry": pd.Timestamp(nxt_exp), "days": days, "n": len(g), "ret": ret,
                    "cash": r * days / 365, "excess": ret - r * days / 365})
    cyc = pd.DataFrame(cyc)
    tr["w"] = 1 / topk
    return {"trips": tr, "cycles": cyc}


def hb2(p: pd.DataFrame, mult: float = 1.0, th: float = HB2_TH, include_div: bool = False) -> dict:
    div, _ = actions()
    sig = p[(p["tau"] >= 3) & (p["fval"] >= MIN_FUT_VALUE)]
    sig = net_excess_signal(sig, div, include_div=include_div)
    cand = sig[sig["e"] >= th].sort_values("date")
    tr = stock_trips(p, cand[["date", "symbol", "expiry", "e", "g"]], mult)
    tr = tr.sort_values("d1")
    keep, open_until = [], {}
    for i, (s, d1, T) in enumerate(zip(tr["symbol"], tr["d1"], tr["expiry"])):     # one open trip per stock
        if s in open_until and d1 <= open_until[s]:
            continue
        keep.append(i)
        open_until[s] = T
    tr = tr.iloc[keep].copy()
    tr["decay"] = tr["locked_at_entry"] / tr["g"].replace(0, np.nan)            # carry still there at execution / at signal
    return {"trips": tr}


def etf_returns(symbols=("NIFTYBEES", "BANKBEES")) -> pd.DataFrame:
    """ETF closes and close-to-close returns from the raw whole-market bhavcopy (ETFs are not in returns.parquet and
    have no entries in the corporate-action file). NSE's archived prev_close is not split-adjusted, so a day whose
    ratio is within 7 % of a standard split ratio (1/2, 1/5, 1/10) is treated as a split (NIFTYBEES and BANKBEES
    split 1:10 in Dec 2019). Distributions, if any, are missed (conservative)."""
    d = pd.concat([pd.read_csv(f"data/exchange_all/nse/{y}.csv.gz", usecols=["date", "symbol", "series", "close", "prev_close", "value"])
                   for y in range(2015, 2027)], ignore_index=True)
    d = d[d["symbol"].isin(symbols) & (d["series"] == "EQ")].copy()
    d["date"] = pd.to_datetime(d["date"])
    ratio = d["close"] / d["prev_close"]
    split = np.zeros(len(d))
    for k in (2, 5, 10):
        split = np.where(np.abs(ratio * k - 1) < 0.07, k, split)
    d["split"] = split
    d["cc"] = np.where(split > 0, ratio * np.where(split > 0, split, 1) - 1, ratio - 1)
    return d.rename(columns={"close": "E", "cc": "Ecc", "value": "Eval"})[["date", "symbol", "E", "Ecc", "Eval", "split"]]


@lru_cache(maxsize=1)
def index_panel() -> pd.DataFrame:
    f = futures()
    c = contracts(f[f["kind"] == "IDX"])
    ix = load_index()
    ix["date"] = pd.to_datetime(ix["date"])
    cs = etf_returns().rename(columns={"E": "close", "Ecc": "cc", "Eval": "value"})
    rows = []
    for fut, (etf, iname) in ETF.items():
        n = c[(c["symbol"] == fut) & (c["rank"] == 0)][["date", "expiry", "close", "contracts", "value"]].rename(columns={"close": "F", "contracts": "fcon", "value": "fval"})
        i = ix[ix["index"] == iname][["date", "close", "div_yield"]].rename(columns={"close": "I", "div_yield": "dy"})
        e = cs[cs["symbol"] == etf][["date", "close", "cc", "value"]].rename(columns={"close": "E", "cc": "Ecc", "value": "Eval"})
        x = n.merge(i, on="date", how="left").merge(e, on="date", how="left")
        x["fut"] = fut
        rows.append(x)
    x = pd.concat(rows, ignore_index=True)
    x["tau"] = (x["expiry"] - x["date"]).dt.days
    return x


@lru_cache(maxsize=2)
def etf_series(sym: str) -> pd.Series:
    e = etf_returns((sym,))
    return e.set_index("date")["Ecc"].sort_index()


def etf_total_return(sym: str, a, b) -> float:
    """Compounded ETF return over (a, b] on the ETF's own sessions (not the F&O calendar; Amendment 1)."""
    s = etf_series(sym)
    seg = s[(s.index > pd.Timestamp(a)) & (s.index <= pd.Timestamp(b))]
    return float(np.prod(1 + seg.to_numpy()) - 1) if len(seg) and not seg.isna().any() else float("nan")


def hb3(ip: pd.DataFrame, fut: str, mult: float = 1.0, margin: float = CAP_MARGIN) -> pd.DataFrame:
    x = ip[ip["fut"] == fut].sort_values("date").reset_index(drop=True)
    f = futures()
    st = settle_on_expiry(f, "IDX")
    st = st[st["symbol"] == fut].set_index("expiry")["F_T"]
    exp = np.sort(x["expiry"].unique())
    sess = x["date"].values
    notional = BOOK / (1 + margin)
    out = []
    for e_prev, e in zip(exp[:-1], exp[1:]):
        k0 = np.searchsorted(sess, e_prev, side="right")
        if k0 + 1 >= len(sess) or pd.Timestamp(e) > pd.Timestamp(END):
            continue
        d0, d1 = x.loc[k0], x.loc[k0 + 1]
        if d0["expiry"] != e or d1["expiry"] != e:
            continue
        r = float(rate_on([d1["date"]])[0])
        days = (pd.Timestamp(e) - d1["date"]).days
        row = {"d0": d0["date"], "expiry": pd.Timestamp(e), "days": days, "cash": r * days / 365}
        ok = not (np.isnan(d0["I"]) or np.isnan(d0["F"]) or np.isnan(d1["E"]) or e not in st.index)
        if ok:
            g = np.log(d0["F"] / d0["I"]) + (d0["dy"] / 100 if not np.isnan(d0["dy"]) else 0) * d0["tau"] / 365
            c1 = costs_bp(d1["date"], e, notional, "ETF", 1.0)                 # selection always at 1x costs
            c = costs_bp(d1["date"], e, notional, "ETF", mult)
            ann = (g - (c1["cash"] + c1["fut"]) / 1e4) * 365 / max(d0["tau"], 1) - float(rate_on([d0["date"]])[0])
            row.update({"e": ann, "g_ann": g * 365 / max(d0["tau"], 1)})
            if ann > 0:
                R_etf = etf_total_return(ETF[fut][0], d1["date"], e)
                if np.isnan(R_etf):
                    ok = False
                else:
                    I1 = d1["I"] if not np.isnan(d1["I"]) else d1["F"]
                    cash_leg = R_etf
                    fut_leg = -(st[e] - d1["F"]) / I1
                    pnl = cash_leg + fut_leg - (c["cash"] + c["fut"]) / 1e4
                    row.update({"in": True, "cash_leg": cash_leg, "fut_leg": fut_leg, "cost_cash": c["cash"] / 1e4, "cost_fut": c["fut"] / 1e4,
                                "ret": pnl / (1 + margin), "w": 1.0, "fval1": d1["fval"], "eval1": d1["Eval"],
                                "cap_frac": min(1.0, 0.01 * d1["fval"] / notional)})
        if "ret" not in row:
            row.update({"in": False, "ret": row["cash"], "w": 0.0})
        row["excess"] = row["ret"] - row["cash"]
        row["data_ok"] = ok
        out.append(row)
    return pd.DataFrame(out)


def index_replication(ip: pd.DataFrame, fut: str) -> dict:
    """Descriptive: hold the index as near-month futures + cash at the repo rate (rolled at expiry, entered at the next
    session's close) instead of the ETF. Per cycle: futures return on the index level + cash interest - futures
    costs (buy stamp, exchange fees, brokerage, half-spread in and out) vs the ETF's total return."""
    x = ip[ip["fut"] == fut].sort_values("date").reset_index(drop=True)
    st = settle_on_expiry(futures(), "IDX")
    st = st[st["symbol"] == fut].set_index("expiry")["F_T"]
    exp = np.sort(x["expiry"].unique())
    sess = x["date"].values
    rows = []
    for e_prev, e in zip(exp[:-1], exp[1:]):
        k = np.searchsorted(sess, e_prev, side="right")
        if k >= len(sess) or pd.Timestamp(e) > pd.Timestamp(END) or e not in st.index:
            continue
        d1 = x.loc[k]
        if d1["expiry"] != e or np.isnan(d1["I"]):
            continue
        etf = etf_total_return(ETF[fut][0], d1["date"], e)
        if np.isnan(etf):
            continue
        days = (pd.Timestamp(e) - d1["date"]).days
        r = float(rate_on([d1["date"]])[0])
        brok = 20 * (1 + GST) / (BOOK) * 1e4
        cost = (0.2 + (TXN_FUT + SEBI) * (1 + GST) + brok + 1.0 + 1.0) / 1e4          # buy stamp, fees, spread in/out
        synth = (st[e] - d1["F"]) / d1["I"] + r * days / 365 - cost
        rows.append({"d1": d1["date"], "days": days, "synthetic": synth, "etf": etf, "diff": synth - etf})
    d = pd.DataFrame(rows)
    mu, t, pv, n = nwt(d["diff"].to_numpy(), 3, two_sided=True)
    return {"cycles": n, "ann_advantage_futures_plus_cash_over_etf": float(d["diff"].sum() / d["days"].sum() * 365), "t": t, "p_two_sided": pv,
            "note": "pre-tax; futures P&L is business income, ETF gains are capital gains (LTCG after 12 months)"}


def reverse_frequency(p: pd.DataFrame) -> dict:
    """Descriptive: how often a liquid stock future is cheap enough that reverse cash-and-carry (short stock via SLB,
    long future) would clear costs and the cash rate; not retail-feasible (needs stock borrowing)."""
    q = p[(p["tau"] >= 7) & (p["fval"] >= MIN_FUT_VALUE)].copy()
    q["carry_ann"] = (q["F"] / q["S"] - 1) * 365 / q["tau"]
    c = 0.0037 * 365 / q["tau"]                                         # ~37 bp round trip (HB1's measured mean)
    q["rev_e"] = rate_on(q["date"]) - q["carry_ann"] - c
    return {"stock_days": int(len(q)), "share_rev_e_gt_0": float((q["rev_e"] > 0).mean()), "share_rev_e_gt_5pct": float((q["rev_e"] > 0.05).mean()),
            "median_carry_minus_repo": float((q["carry_ann"] - rate_on(q["date"])).median())}


def hb4(ip: pd.DataFrame | None = None) -> dict:
    f = futures()
    res = {}
    for name, sel in (("NIFTY", f["symbol"] == "NIFTY"), ("BANKNIFTY", f["symbol"] == "BANKNIFTY"), ("STOCK_MEDIAN", f["kind"] == "STK")):
        g = f[sel & (f["expiry"] > f["date"])].sort_values(["date", "symbol", "expiry"])
        g["rank"] = g.groupby(["date", "symbol"]).cumcount()
        a = g[g["rank"] == 0][["date", "symbol", "expiry", "close", "contracts", "value"]]
        b = g[g["rank"] == 1][["date", "symbol", "expiry", "close", "contracts", "value"]]
        m = a.merge(b, on=["date", "symbol"], suffixes=("_n", "_x"))
        m = m[(m["contracts_n"] > 0) & (m["contracts_x"] > 0) & (m["expiry_x"] > m["expiry_n"])]
        if name == "STOCK_MEDIAN":                                        # liquid stock futures: both legs >= ₹10 crore
            m = m[(m["value_n"] >= MIN_FUT_VALUE) & (m["value_x"] >= MIN_FUT_VALUE)]
        m["cs"] = np.log(m["close_x"] / m["close_n"]) * 365 / (m["expiry_x"] - m["expiry_n"]).dt.days
        if name == "STOCK_MEDIAN":
            m = m.groupby(["date", "expiry_n"], as_index=False)["cs"].median()
        sess = pd.DatetimeIndex(np.sort(f["date"].unique()))
        m["k_to_exp"] = sess.searchsorted(pd.DatetimeIndex(m["expiry_n"])) - sess.searchsorted(pd.DatetimeIndex(m["date"]))   # sessions before expiry
        cyc = []
        for e, h in m.groupby("expiry_n"):
            last = h[(h["k_to_exp"] >= 1) & (h["k_to_exp"] <= 3)]["cs"]
            base = h[(h["k_to_exp"] >= 4) & (h["k_to_exp"] <= 10)]["cs"]
            if len(last) >= 2 and len(base) >= 4:
                cyc.append({"expiry": e, "diff": float(last.mean() - base.mean()), "last": float(last.mean()), "base": float(base.mean()),
                            "roll_cost_bp_change": float((last.mean() - base.mean()) * 28 / 365 * 1e4)})
        c = pd.DataFrame(cyc)
        mu, t, p, n = nwt(c["diff"].to_numpy(), 3, two_sided=True)
        res[name] = {"mean_diff_ann": mu, "t": t, "p": p, "cycles": n, "mean_last": float(np.nanmean(c["last"])), "mean_base": float(np.nanmean(c["base"])),
                     "roll_bp_per_cycle": float(c["roll_cost_bp_change"].mean()), "share_positive": float((c["diff"] > 0).mean())}
    return res


# ------------------------------------------------------------------------------------------------ audit
def audit(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    f = futures()
    cs = cash()
    sess_cash = set(cs["date"][(cs["date"] >= START) & (cs["date"] <= END)].unique())
    sess_fo = set(f["date"].unique())
    p = stock_panel()
    ip = index_panel()
    stk = f[f["kind"] == "STK"]
    near = contracts(stk)
    near = near[near["rank"] == 0]
    joined = near.merge(cs[["date", "symbol"]], on=["date", "symbol"], how="left", indicator=True)
    exp_days = f[f["kind"] == "IDX"]["expiry"].drop_duplicates()
    _, adj = actions()
    pp = p[p["fval"] >= MIN_FUT_VALUE].copy()
    pp["carry_ann"] = (pp["F"] / pp["S"] - 1) * 365 / pp["tau"].clip(lower=1)
    pp["repo"] = rate_on(pp["date"])
    by_year = pp[pp["tau"] >= 7].groupby(pp["date"].dt.year).agg(median_carry=("carry_ann", "median"), repo=("repo", "median"),
                                                                  p10=("carry_ann", lambda v: v.quantile(.1)), p90=("carry_ann", lambda v: v.quantile(.9)), n=("carry_ann", "size"))
    ia = ip.dropna(subset=["I", "F"]).copy()
    ia["basis_ann"] = (np.log(ia["F"] / ia["I"]) + ia["dy"].fillna(0) / 100 * ia["tau"] / 365) * 365 / ia["tau"].clip(lower=1)
    a = {"sessions_cash": len(sess_cash), "sessions_fo": len(sess_fo), "cash_sessions_without_fo": sorted(str(d)[:10] for d in sess_cash - sess_fo)[:20],
         "fo_rows": int(len(f)), "stock_futures_symbols": int(stk["symbol"].nunique()),
         "zero_contract_share_all_rows": float((f["contracts"] == 0).mean()),
         "zero_contract_share_near_stock": float((near["contracts"] == 0).mean()),
         "near_stock_join_to_cash": float((joined["_merge"] == "both").mean()),
         "unjoined_examples": joined.loc[joined["_merge"] != "both", "symbol"].value_counts().head(10).to_dict(),
         "expiries_per_year": f[f["symbol"] == "NIFTY"]["expiry"].drop_duplicates().dt.year.value_counts().sort_index().to_dict(),
         "expiry_not_a_session": int((~exp_days.isin(list(sess_fo))).sum() - (exp_days > pd.Timestamp(END)).sum()),
         "stock_carry_by_year": {int(k): {kk: round(float(vv), 4) for kk, vv in v.items()} for k, v in by_year.to_dict("index").items()},
         "index_basis_ann_by_fut": ia.groupby("fut")["basis_ann"].describe(percentiles=[.1, .5, .9])[["mean", "10%", "50%", "90%"]].round(4).to_dict("index"),
         "index_rows_missing_index_close": int(ip["I"].isna().sum()), "index_rows_missing_etf": int(ip["E"].isna().sum()),
         "stock_days_with_adj_event_in_life_share": None}
    sample = pp.sample(min(20000, len(pp)), random_state=1)
    sample["adj"] = _in_window(adj.assign(n=1), sample, "date", "expiry", None)
    a["stock_days_with_adj_event_in_life_share"] = float((sample["adj"] > 0).mean())
    # settlement check: final settlement price vs cash close on expiry day
    st = settle_on_expiry(f, "STK").merge(cs[["date", "symbol", "close"]].rename(columns={"date": "expiry"}), on=["expiry", "symbol"], how="inner")
    dev = (st["F_T"] / st["close"] - 1) * 1e4
    a["expiry_settle_vs_cash_close_bp"] = {"median": float(dev.median()), "p1": float(dev.quantile(.01)), "p99": float(dev.quantile(.99)),
                                           "abs_gt_50bp_share": float((dev.abs() > 50).mean()), "n": int(len(dev))}
    (out / "audit.json").write_text(json.dumps(a, indent=1, default=str))
    return a


# ------------------------------------------------------------------------------------------------ run
def nifty_monthly() -> pd.Series:
    ix = load_index()
    ix = ix[ix["index"] == "NIFTY 50"]
    s = ix.set_index(pd.to_datetime(ix["date"]))["close"].sort_index()
    return s


def beta(cyc: pd.DataFrame, ret_col: str = "ret") -> float:
    s = nifty_monthly()
    nr = []
    for a, b in zip(cyc["d0"], cyc["expiry"]):
        x0, x1 = s[:a], s[:b]
        nr.append(float(x1.iloc[-1] / x0.iloc[-1] - 1) if len(x0) and len(x1) else np.nan)
    y, x = cyc[ret_col].to_numpy(float), np.array(nr)
    m = ~np.isnan(x) & ~np.isnan(y)
    return float(np.polyfit(x[m], y[m], 1)[0]) if m.sum() > 10 else float("nan")


def span_days(d_start: pd.Series, d_end: pd.Series) -> float:
    return float((pd.Timestamp(d_end.max()) - pd.Timestamp(d_start.min())).days) if len(d_start) else float("nan")


def summarise_cycles(cyc: pd.DataFrame, trips: pd.DataFrame | None, mode: str = "prereg") -> dict:
    """Book-level results. Annualised on calendar time (Amendment 1): the days between one expiry and the next
    cycle's execution count as idle capital (excess 0)."""
    days = span_days(cyc["d0"], cyc["expiry"])
    ann_ex = float(cyc["excess"].sum() / days * 365)
    subs = {}
    for k, (a, b) in SUBS.items():
        g = cyc[(cyc["d0"] >= a) & (cyc["d0"] <= b)]
        subs[k] = float(g["excess"].sum() / span_days(g["d0"], g["expiry"]) * 365) if len(g) else None
    tax = after_tax_excess(trips, None, mode=mode) if trips is not None and len(trips) else {}
    cyc = cyc.copy()
    cyc["FY"] = cyc["expiry"].map(fy)
    ex_after = 0.0
    for y, g in cyc.groupby("FY"):
        trip_pre = tax.get(int(y), {}).get("pre_tax", 0.0)
        trip_tax = tax.get(int(y), {}).get("tax", 0.0)
        idle_int = float(g["ret"].sum() - trip_pre)                          # the rest of the book's return is cash interest
        ex_after += (trip_pre - trip_tax + idle_int * (1 - TAX_CASH)) - float(g["cash"].sum()) * (1 - TAX_CASH)
    return {"cycles": int(len(cyc)), "calendar_days": days, "ann_excess": ann_ex, "ann_return_while_in_cycles": float(cyc["ret"].sum() / cyc["days"].sum() * 365),
            "ann_cash_while_in_cycles": float(cyc["cash"].sum() / cyc["days"].sum() * 365), "after_tax_ann_excess": float(ex_after / days * 365),
            "subperiods_ann_excess": subs, "beta_to_nifty": beta(cyc), "share_cycles_invested": float((cyc.get("n", cyc.get("w", 0)) > 0).mean()),
            "worst_cycle_excess": float(cyc["excess"].min())}


def hb1_summary(p: pd.DataFrame, include_div: bool = False) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    h = hb1(p, include_div=include_div)
    cyc, tr = h["cycles"], h["trips"]
    m, t, pv, n = nwt(cyc["excess"].to_numpy())
    r = {"mean_cycle_excess": m, "t": t, "p": pv, "n": n, **summarise_cycles(cyc, tr), "trips": int(len(tr)),
         "mean_locked_at_entry_bp": float(tr["locked_at_entry"].mean() * 1e4), "mean_pnl_bp": float(tr["pnl"].mean() * 1e4),
         "mean_cost_bp": float((tr["cost_cash"] + tr["cost_fut"]).mean() * 1e4), "mean_days": float(tr["days"].mean()),
         "trips_with_dividend": int((tr["D"] > 0).sum()), "capacity_share_trips_capped": float((tr["cap_frac"] < 1).mean())}
    h2 = hb1(p, mult=2.0, include_div=include_div)
    r["cost2x_ann_excess"] = summarise_cycles(h2["cycles"], h2["trips"])["ann_excess"]
    ex = ((tr["ret_cap"] - tr["r"] * tr["days"] / 365) * tr["cap_frac"] / TOPK).groupby(tr["date"]).sum()
    cc = cyc.copy()
    cc["excess"] = cc["d0"].map(ex).fillna(0.0)
    r["capacity_ann_excess"] = float(cc["excess"].sum() / span_days(cc["d0"], cc["expiry"]) * 365)
    r["tax_variants_after_tax_ann_excess"] = {m_: summarise_cycles(cyc, tr, m_)["after_tax_ann_excess"] for m_ in ("carry_forward", "business_cf")}
    return r, cyc, tr


def nifty_month_returns(months: pd.PeriodIndex) -> pd.Series:
    s = nifty_monthly()
    m = s.groupby(s.index.to_period("M")).last().pct_change()
    return m.reindex(months)


def hb2_summary(p: pd.DataFrame, include_div: bool = False) -> tuple[dict, pd.DataFrame]:
    t2 = hb2(p, include_div=include_div)["trips"]
    t2["month"] = t2["d1"].dt.to_period("M")
    mm = t2.groupby("month")["excess_cap"].mean()
    m, t, pv, n = nwt(mm.to_numpy())
    ann = float(t2["excess_cap"].sum() / t2["days"].sum() * 365)
    t2c = hb2(p, mult=2.0, include_div=include_div)["trips"]
    ann2 = float(t2c["excess_cap"].sum() / t2c["days"].sum() * 365)
    t2["w"] = 1.0 * (1 + CAP_MARGIN)                         # after_tax_excess divides by (1+m): per unit of capital
    cash_int = float((t2["r"] * t2["days"] / 365).sum())
    after = {}
    for mode in ("prereg", "carry_forward", "business_cf"):
        tx = after_tax_excess(t2, None, mode=mode)
        pre = sum(v["pre_tax"] for v in tx.values())
        taxs = sum(v["tax"] for v in tx.values())
        after[mode] = float(((pre - taxs) - cash_int * (1 - TAX_CASH)) / t2["days"].sum() * 365)
    subs = {}
    for k, (a, b) in SUBS.items():
        g = t2[(t2["d1"] >= a) & (t2["d1"] <= b)]
        subs[k] = float(g["excess_cap"].sum() / g["days"].sum() * 365) if len(g) else None
    mret = t2.groupby("month")["ret_cap"].mean()
    nm = nifty_month_returns(mret.index)
    ok = nm.notna().to_numpy()
    b = float(np.polyfit(nm.to_numpy()[ok], mret.to_numpy()[ok], 1)[0]) if ok.sum() > 10 else float("nan")
    cap_ann = float((t2["excess_cap"] * t2["cap_frac"]).sum() / (t2["days"] * t2["cap_frac"]).sum() * 365)
    r = {"mean_month_excess_per_trip": m, "t": t, "p": pv, "months": n, "trips": int(len(t2)), "ann_excess_on_deployed_capital": ann,
         "after_tax_ann_excess": after["prereg"], "tax_variants_after_tax_ann_excess": {k: v for k, v in after.items() if k != "prereg"},
         "cost2x_ann_excess": ann2, "subperiods_ann_excess": subs, "beta_to_nifty": b, "capacity_ann_excess": cap_ann,
         "mean_signal_e": float(t2["e"].mean()), "median_decay_locked_over_signalled": float(t2["decay"].median()),
         "mean_pnl_bp": float(t2["pnl"].mean() * 1e4), "mean_days": float(t2["days"].mean()), "trips_with_dividend": int((t2["D"] > 0).sum()),
         "capacity_share_trips_capped": float((t2["cap_frac"] < 1).mean())}
    return r, t2


def gate_for(r: dict, holm_p: float, mean_key: str) -> dict:
    ae = r.get("ann_excess", r.get("ann_excess_on_deployed_capital"))
    b = r.get("beta_to_nifty")
    steps = [("statistical", holm_p < 0.05 and r[mean_key] > 0), ("excess_ge_1.5pct", ae >= 0.015),
             ("after_tax_excess_ge_1pct", r["after_tax_ann_excess"] >= 0.01), ("cost2x_excess_ge_0.75pct", r["cost2x_ann_excess"] >= 0.0075),
             ("beta_abs_lt_0.10", b is not None and not np.isnan(b) and abs(b) < 0.10), ("capacity", r.get("capacity_ann_excess", ae) >= 0.015),
             ("subperiods_2_of_3", sum(1 for v in r["subperiods_ann_excess"].values() if v is not None and v > 0) >= 2)]
    failed = next((k for k, ok in steps if not ok), None)
    return {"holm_p": holm_p, "steps": {k: bool(v) for k, v in steps}, "first_failed": failed,
            "verdict": "PASS -> forward paper test" if failed is None else f"REJECTED at {failed}"}


def run(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    p = stock_panel()
    ip = index_panel()
    res = {"HB1": {}, "HB2": {}, "HB3": {}, "HB4": {}, "as_preregistered_signal": {}}
    # ---- HB1 (amended signal: no dividends in g)
    res["HB1"], cyc, tr = hb1_summary(p)
    res["HB1"]["descriptive"] = {"m0_pledged": summarise_cycles(hb1(p, margin=0.0)["cycles"], None)["ann_excess"],
                                 "top5": summarise_cycles(hb1(p, topk=5)["cycles"], None)["ann_excess"],
                                 "top20": summarise_cycles(hb1(p, topk=20)["cycles"], None)["ann_excess"],
                                 "capital_based_selection": summarise_cycles(hb1(p, capital_select=True)["cycles"], None)["ann_excess"]}
    CASH_SENSITIVITY["on"] = True
    try:
        res["HB1"]["descriptive"]["liquid_fund_rate_2020_22"] = summarise_cycles(hb1(p)["cycles"], None)["ann_excess"]
    finally:
        CASH_SENSITIVITY["on"] = False
    cyc.to_csv(out / "hb1_cycles.csv", index=False)
    tr.to_csv(out / "hb1_trips.csv", index=False)
    # ---- HB2
    res["HB2"], t2 = hb2_summary(p)
    res["HB2"]["descriptive"] = {}
    for th in (0.05, 0.15):
        z = hb2(p, th=th)["trips"]
        res["HB2"]["descriptive"][f"threshold_{th}"] = {"trips": int(len(z)), "ann_excess": float(z["excess_cap"].sum() / z["days"].sum() * 365)}
    t2.drop(columns=["month"]).to_csv(out / "hb2_trips.csv", index=False)
    # ---- the pre-registered signal (dividends in g; look-ahead on announcement), reported for disclosure
    a1, _, tra = hb1_summary(p, include_div=True)
    a2, t2a = hb2_summary(p, include_div=True)
    res["as_preregistered_signal"] = {"HB1": a1, "HB2": a2,
                                      "HB2_dividend_vs_no_dividend_ann_excess": {
                                          "with_dividend": float(t2a.loc[t2a["D"] > 0, "excess_cap"].sum() / t2a.loc[t2a["D"] > 0, "days"].sum() * 365) if (t2a["D"] > 0).any() else None,
                                          "without_dividend": float(t2a.loc[t2a["D"] == 0, "excess_cap"].sum() / t2a.loc[t2a["D"] == 0, "days"].sum() * 365)}}
    # ---- HB3
    for fut in ("NIFTY", "BANKNIFTY"):
        c3 = hb3(ip, fut)
        m, t, pv, n = nwt(c3["excess"].to_numpy())
        tr3 = c3[c3["in"]].copy()
        s3 = summarise_cycles(c3.assign(n=c3["w"]), tr3 if len(tr3) else None)
        c32 = hb3(ip, fut, mult=2.0)
        cap = c3.copy()
        cap["excess"] = np.where(cap["in"], cap["excess"] * cap.get("cap_frac", 1.0).fillna(1.0), 0.0)
        res["HB3"][fut] = {"mean_cycle_excess": m, "t": t, "p": pv, "n": n, **s3, "cycles_in_position": int(c3["in"].sum()),
                           "cycles_data_missing": int((~c3["data_ok"]).sum()),
                           "cost2x_ann_excess": float(c32["excess"].sum() / span_days(c32["d0"], c32["expiry"]) * 365),
                           "capacity_ann_excess": float(cap["excess"].sum() / span_days(cap["d0"], cap["expiry"]) * 365),
                           "mean_signal_e_ann": float(c3["e"].mean()), "mean_g_ann": float(c3["g_ann"].mean()),
                           "mean_excess_when_in_bp": float(tr3["excess"].mean() * 1e4) if len(tr3) else None}
        c3.to_csv(out / f"hb3_{fut}.csv", index=False)
    # ---- HB4
    res["HB4"] = hb4()
    # ---- descriptive (never gating)
    res["descriptive"] = {"index_replication": {f: index_replication(ip, f) for f in ("NIFTY", "BANKNIFTY")},
                          "reverse_arbitrage_frequency": reverse_frequency(p)}
    # ---- Holm + gate
    ps = {"HB1": res["HB1"]["p"], "HB2": res["HB2"]["p"], "HB3-N": res["HB3"]["NIFTY"]["p"], "HB3-BN": res["HB3"]["BANKNIFTY"]["p"], "HB4": res["HB4"]["NIFTY"]["p"]}
    hp = holm(ps)
    res["holm"] = hp
    gate = {}
    pa = holm({**ps, "HB1": res["as_preregistered_signal"]["HB1"]["p"], "HB2": res["as_preregistered_signal"]["HB2"]["p"]})
    res["as_preregistered_signal"]["holm"] = pa
    res["as_preregistered_signal"]["gate"] = {"HB1": gate_for(res["as_preregistered_signal"]["HB1"], pa["HB1"], "mean_cycle_excess"),
                                              "HB2": gate_for(res["as_preregistered_signal"]["HB2"], pa["HB2"], "mean_month_excess_per_trip")}
    for key, r, mean_key in (("HB1", res["HB1"], "mean_cycle_excess"), ("HB2", res["HB2"], "mean_month_excess_per_trip"),
                             ("HB3-N", res["HB3"]["NIFTY"], "mean_cycle_excess"), ("HB3-BN", res["HB3"]["BANKNIFTY"], "mean_cycle_excess")):
        gate[key] = gate_for(r, hp[key], mean_key)
    res["gate"] = gate
    (out / "v11.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["audit", "run"])
    ap.add_argument("--out", default="reports/v1.1_dev")
    a = ap.parse_args()
    r = audit(Path(a.out)) if a.cmd == "audit" else run(Path(a.out))
    print(json.dumps(r, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
