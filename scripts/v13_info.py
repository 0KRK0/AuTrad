"""v1.3 Information Advantage Lab, Part A (research/preregistration_v1.3.md): bad-news diffusion under short-sale
constraints, tested on F&O stocks that were NOT NIFTY 50 members at the event date (out-of-universe test of v0.5).

    python -m scripts.v13_info audit|run --out reports/v1.3_<ts>
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from scripts import v11_basis as B
from scripts import v12_events as E

START_R0, END_R0 = pd.Timestamp("2016-04-01"), pd.Timestamp("2026-08-31")
BURN_IN = 300
STALE_LAG_DAYS = 60
HOLD = 20
SUBS = {"2016-18": ("2016-01-01", "2018-12-31"), "2019-22": ("2019-01-01", "2022-12-31"), "2023-26": ("2023-01-01", "2026-12-31")}
FIELDS = {"economic_mechanism": "bad results news is incorporated slowly: pessimists cannot easily short Indian cash equities, long-only "
                                "institutions sell over days, analyst downgrades lag",
          "economic_counterparty": "early dip-buyers and slow sellers (long-only holders trimming over several sessions, investors anchored to the pre-results price)",
          "mechanism_constraint": "short-sale constraints in cash; institutional participation limits; analyst and attention lag outside the most-covered names",
          "our_advantage": "short side via stock futures; timestamped processing of every results filing across less-covered F&O names; AEM-priced entry",
          "failure_condition": "no drift outside NIFTY 50, or futures already discount it, or costs/tax/capacity remove it, or it lives only in distressed names"}


# ------------------------------------------------------------------------------------------------ PIT NIFTY 50
@lru_cache(maxsize=1)
def n50_membership() -> pd.DataFrame:
    """Membership intervals from the semi-annual snapshots plus press-release changes between snapshots."""
    s = pd.read_csv("marketdata/reference/nifty50_pit_snapshots.csv", parse_dates=["effective_date"])
    ch = pd.read_csv("data/events/index_changes.csv", parse_dates=["announce_date", "effective_date"])
    ch = ch[(ch["index_key"] == "NIFTY50") & ch["effective_date"].notna()].sort_values("effective_date")
    dates = sorted(s["effective_date"].unique())
    rows = []
    for k, d in enumerate(dates):
        nxt = dates[k + 1] if k + 1 < len(dates) else pd.Timestamp("2027-01-01")
        cur = set(s[s["effective_date"] == d]["symbol"])
        t = d
        for e in ch[(ch["effective_date"] > d) & (ch["effective_date"] < nxt)].itertuples(index=False):
            rows += [(x, t, e.effective_date) for x in cur]
            t = e.effective_date
            cur = (cur - {e.symbol}) if e.action == "exclude" else (cur | {e.symbol})
        rows += [(x, t, nxt) for x in cur]
    return pd.DataFrame(rows, columns=["symbol", "start", "end"])


def in_n50_fast(sym: str, d: pd.Timestamp, m: pd.DataFrame) -> bool:
    g = _n50_by_symbol().get(sym)
    return bool(g is not None and ((g[0] <= d) & (g[1] > d)).any())


@lru_cache(maxsize=1)
def _n50_by_symbol() -> dict:
    m = n50_membership()
    return {s: (g["start"].to_numpy(), g["end"].to_numpy()) for s, g in m.groupby("symbol")}


def in_n50(sym: str, d: pd.Timestamp) -> bool:
    m = n50_membership()
    g = m[m["symbol"] == sym]
    return bool(((g["start"] <= d) & (g["end"] > d)).any())


# ------------------------------------------------------------------------------------------------ events
@lru_cache(maxsize=1)
def announcements() -> pd.DataFrame:
    p = Path("/tmp/autrad_pit/index_all.pkl")
    if p.exists():
        idx = pd.read_pickle(p)
    else:
        from scripts.v13_data import load_index
        idx = load_index()
    # Amendment 1: the announcement time is the EARLIEST of broadcast and dissemination time (the first public moment);
    # index entries more than 60 days after quarter end are late uploads, not announcements (v0.5's stale rule; SEBI
    # requires results within 45 days) and are dropped; one event per (symbol, r0) is kept later.
    idx = idx.copy()
    idx["ann_time"] = idx[["published_at", "disseminated_at"]].min(axis=1)
    idx = idx[idx["ann_time"].notna()]
    first = idx.sort_values("ann_time").groupby(["symbol", "period_end"], as_index=False).first()
    pe = pd.to_datetime(first["period_end"]).dt.tz_localize("Asia/Kolkata")
    first["lag_days"] = (first["ann_time"] - pe).dt.days
    first = first[first["lag_days"] <= STALE_LAG_DAYS]
    first["available_at"] = first["ann_time"]
    return first[["symbol", "period_end", "available_at", "consolidated", "xbrl", "filing_id", "lag_days"]]


def r0_of(ts: pd.Timestamp) -> pd.Timestamp | None:
    return E.first_close_after(ts)        # the 15:00 IST rule (closing price = 15:00-15:30 weighted average)


@lru_cache(maxsize=1)
def events() -> pd.DataFrame:
    a = announcements().copy()
    a["r0"] = [r0_of(t) for t in a["available_at"]]
    a = a.dropna(subset=["r0"])
    a = a[(a["r0"] >= START_R0) & (a["r0"] <= END_R0)]
    f = B.futures()
    stk = set(zip(f.loc[f["kind"] == "STK", "symbol"], f.loc[f["kind"] == "STK", "date"]))
    a = a[[(s, d) in stk for s, d in zip(a["symbol"], a["r0"])]]
    a["in_n50"] = [in_n50(s, d) for s, d in zip(a["symbol"], a["r0"])]
    r, m = E.price_returns()
    a["reaction"] = [float(r.at[d, s] - m.get(d, np.nan)) if (s in r.columns and d in r.index) else np.nan for s, d in zip(a["symbol"], a["r0"])]
    a = a.dropna(subset=["reaction"]).sort_values(["r0", "available_at"]).drop_duplicates(["symbol", "r0"]).reset_index(drop=True)
    # expanding, past-only lower-third threshold within the out-of-N50 universe
    u = a[~a["in_n50"]].copy()
    thr, prior = [], []
    dates = u["r0"].to_numpy()
    vals = u["reaction"].to_numpy()
    for i in range(len(u)):
        k = np.searchsorted(dates, dates[i], side="left")                 # strictly earlier dates
        thr.append(np.quantile(vals[:k], 1 / 3) if k >= BURN_IN else np.nan)
        prior.append(k)
    u["threshold"], u["n_prior"] = thr, prior
    u["bad"] = u["reaction"] < u["threshold"]
    s = E.sessions()
    u["entry"] = [E.next_session(d, 1) for d in u["r0"]]
    u["exit"] = [E.next_session(d, 1 + HOLD) for d in u["r0"]]
    return u.dropna(subset=["entry", "exit"])


# ------------------------------------------------------------------------------------------------ tests
def car20(u: pd.DataFrame) -> pd.Series:
    return pd.Series([E.car(s, a, b) for s, a, b in zip(u["symbol"], u["entry"], u["exit"])], index=u.index, dtype=float)


def date_cluster_test(v: pd.Series, dates: pd.Series, sign: int = 1):
    g = (sign * v).groupby(dates).mean()
    return B.nwt(g.to_numpy())


def trades(u: pd.DataFrame, mult: float = 1.0) -> pd.DataFrame:
    f = B.futures()
    k = f.drop_duplicates(["date", "symbol", "expiry"]).set_index(["date", "symbol", "expiry"])
    by_day = {d: g for d, g in f[f["date"].isin(set(u["entry"]))].groupby("date")}
    banned = E.ban_days()
    _, adj = B.actions()
    div = B.actions()[0]
    u = u.copy()
    u["adj"] = B._in_window(adj, u.rename(columns={"r0": "a", "exit": "b"}), "a", "b", None) > 0
    cs = B.cash().drop_duplicates(["date", "symbol"]).set_index(["date", "symbol"])["close"]
    rows = []
    n_by_day = u.groupby("entry").size()
    for e in u.itertuples(index=False):
        if e.adj or (e.symbol, e.entry) in banned:
            continue
        # extraordinary dividend (>= 2 % of price) inside the window: the contract may be adjusted -> excluded
        dd = div[(div["symbol"] == e.symbol) & (div["ex_date"] > e.r0) & (div["ex_date"] <= e.exit)]["dividend"].sum()
        px = cs.get((e.entry, e.symbol), np.nan)
        if px == px and dd / px >= 0.02:
            continue
        day = by_day.get(e.entry)
        if day is None:
            continue

        def leg(sym, kind):
            q = day[(day["symbol"] == sym) & (day["kind"] == kind) & (day["expiry"] >= e.exit)].sort_values("expiry")
            if q.empty or (e.exit, sym, q.iloc[0]["expiry"]) not in k.index:
                return None
            a, b = q.iloc[0], k.loc[(e.exit, sym, q.iloc[0]["expiry"])]
            return float(a["close"]), float(b["close"]), float(a["value"]), float(a["contracts"])
        s_, n_ = leg(e.symbol, "STK"), leg("NIFTY", "IDX")
        tot_val = float(day.loc[(day["symbol"] == e.symbol) & (day["kind"] == "STK"), "value"].sum())   # all contracts (Amendment 1)
        if s_ is None or n_ is None or s_[3] <= 0 or tot_val < B.MIN_FUT_VALUE:
            continue
        rs, rn = s_[1] / s_[0] - 1, n_[1] / n_[0] - 1
        notional = B.BOOK / int(n_by_day.get(e.entry, 1))
        brok = 20 * (1 + B.GST) / notional * 1e4
        hs = max(3.0, 0.5 * B.TICK / s_[0] * 1e4)
        cost = (B.fut_stt(e.entry) + 0.2 + 2 * (B.TXN_FUT + B.SEBI) * (1 + B.GST) + 2 * brok + 2 * hs) \
            + (B.fut_stt(e.exit) + 0.2 + 2 * (B.TXN_FUT + B.SEBI) * (1 + B.GST) + 2 * brok + 2 * 1.0)
        gross = -(rs - rn)
        pnl = gross - mult * cost / 1e4
        days = (e.exit - e.entry).days
        rows.append({"symbol": e.symbol, "r0": e.r0, "d1": e.entry, "entry": e.entry, "exit": e.exit, "expiry": e.exit, "days": days,
                     "reaction": e.reaction, "gross": gross, "pnl": pnl, "excess_cap": pnl, "ret_cap": pnl, "rn": rn, "r": 0.0,
                     "cap_frac": min(1.0, 0.01 * tot_val / notional), "fut_leg": gross, "cost_fut": mult * cost / 1e4, "cash_leg": 0.0, "cost_cash": 0.0})
    return pd.DataFrame(rows)


def trade_summary(t: pd.DataFrame, t2: pd.DataFrame) -> dict:
    ann = float(t["excess_cap"].sum() / t["days"].sum() * 365)
    after = {}
    for mode in ("prereg", "carry_forward"):
        tx = B.after_tax_excess(t.assign(w=1.0 * (1 + B.CAP_MARGIN)), None, mode=mode)
        after[mode] = float((sum(v["pre_tax"] for v in tx.values()) - sum(v["tax"] for v in tx.values())) / t["days"].sum() * 365)
    subs = {}
    for k, (a, b) in SUBS.items():
        g = t[(t["entry"] >= a) & (t["entry"] <= b)]
        subs[k] = float(g["excess_cap"].sum() / g["days"].sum() * 365) if len(g) else None
    loss = t.groupby("symbol")["pnl"].sum().sort_values()
    worst10 = list(loss.index[:10])
    rest = t[~t["symbol"].isin(worst10)]
    m, tt, p, n = date_cluster_test(t["pnl"], t["r0"])
    return {"trades": int(len(t)), "mean_net_bp": float(t["pnl"].mean() * 1e4), "mean_gross_bp": float(t["gross"].mean() * 1e4),
            "t_date_clustered": tt, "p": p, "dates": n, "ann_excess_on_deployed_capital": ann, "after_tax_ann_excess": after["prereg"],
            "after_tax_carry_forward": after["carry_forward"],
            "cost2x_ann_excess": float(t2["excess_cap"].sum() / t2["days"].sum() * 365) if len(t2) else None,
            "capacity_ann_excess": float((t["excess_cap"] * t["cap_frac"]).sum() / (t["days"] * t["cap_frac"]).sum() * 365),
            "beta_to_nifty": float(np.polyfit(t["rn"], t["pnl"], 1)[0]) if len(t) > 10 else float("nan"),
            "subperiods_ann_excess": subs, "mean_net_without_10_worst_names_bp": float(rest["pnl"].mean() * 1e4),
            "worst_10_names": worst10}


def hm1b(bad: pd.DataFrame) -> dict:
    from scripts.v13_data import load_facts
    fx = load_facts()
    if fx.empty:
        return {"testable": False, "coverage": 0.0}
    fx["period_end"] = pd.to_datetime(fx["period_end"], format="mixed").dt.normalize()
    fx["consolidated"] = fx["consolidated"].astype(str).str.lower().isin(["true", "1"])
    fx = fx.dropna(subset=["net_profit"])
    key = fx.sort_values("available_at").drop_duplicates(["symbol", "period_end", "consolidated"], keep="first") \
            .set_index(["symbol", "period_end", "consolidated"])["net_profit"]
    cur, prev = [], []
    for e in bad.itertuples(index=False):
        pe = pd.Timestamp(e.period_end).normalize()
        ya = (pe - pd.DateOffset(years=1)) + pd.offsets.MonthEnd(0)
        c = y = np.nan
        for basis in (True, False):                     # same statement basis for both quarters; consolidated first
            c_, y_ = key.get((e.symbol, pe, basis), np.nan), key.get((e.symbol, ya, basis), np.nan)
            if c_ == c_ and y_ == y_:
                c, y = c_, y_
                break
        cur.append(c)
        prev.append(y)
    b = bad.copy()
    b["np_cur"], b["np_prev"] = cur, prev
    ok = b.dropna(subset=["np_cur", "np_prev", "car20"])
    cov = float(len(ok) / max(len(bad), 1))
    out = {"coverage": cov, "events_with_both": int(len(ok))}
    if cov < 0.70:
        return {**out, "testable": False}
    det = ok[ok["np_cur"] < ok["np_prev"]]["car20"]
    nod = ok[ok["np_cur"] >= ok["np_prev"]]["car20"]
    w = sps.ttest_ind(det, nod, equal_var=False)
    p = float(w.pvalue / 2 if w.statistic < 0 else 1 - w.pvalue / 2)
    return {**out, "testable": True, "deteriorated_n": int(len(det)), "not_deteriorated_n": int(len(nod)),
            "car20_deteriorated": float(det.mean()), "car20_not_deteriorated": float(nod.mean()), "t": float(w.statistic), "p": p}


def needed_filings() -> pd.DataFrame:
    """XBRL filings HM1b needs: each tested bad event's filing and the same-basis filing a year earlier."""
    u = events()
    bad = u[(u["n_prior"] >= BURN_IN) & u["bad"]]
    sel = pd.read_pickle("/tmp/autrad_pit/index_all.pkl")                 # every basis, not only the selected one
    sel = sel[sel["xbrl"].fillna("-").str.lower().str.endswith(".xml")].copy()
    sel["pe"] = pd.to_datetime(sel["period_end"]).dt.tz_localize(None).dt.normalize()
    want = set()
    for e in bad.itertuples(index=False):
        pe = pd.Timestamp(e.period_end).normalize()
        ya = (pe - pd.DateOffset(years=1)) + pd.offsets.MonthEnd(0)
        want |= {(e.symbol, pe), (e.symbol, ya)}
    k = [(s_, p_) in want for s_, p_ in zip(sel["symbol"], sel["pe"])]
    need = sel[k]
    need.to_csv("data/pit_v13/needed_filings.csv", index=False)
    return need


def benchmark_sensitivity(b: pd.DataFrame) -> dict:
    """Descriptive: CAR20 of the tested bad events against other benchmarks (the pre-registered one is NIFTY 50).
    NIFTY MIDCAP 150 suits names outside NIFTY 50; the equal-weight benchmark is the out-of-NIFTY-50 F&O universe."""
    from marketdata.nse_fo import load_index
    r, _ = E.price_returns()
    ix = load_index()
    out = {}
    for name in ("NIFTY MIDCAP 150", "NIFTY MIDCAP 100", "NIFTY 500"):
        x = ix[ix["index"] == name]
        if len(x) < 1000:
            continue
        lv = x.assign(date=pd.to_datetime(x["date"])).set_index("date")["close"].sort_index()
        car = []
        for s_, a_, z_ in zip(b["symbol"], b["entry"], b["exit"]):
            if s_ not in r.columns:
                car.append(np.nan)
                continue
            rs = r.loc[(r.index > a_) & (r.index <= z_), s_]
            la, lz = lv[:a_], lv[:z_]
            car.append(float(np.prod(1 + rs.to_numpy()) - lz.iloc[-1] / la.iloc[-1]) if len(rs) and len(la) and len(lz) and not rs.isna().any() else np.nan)
        v = pd.Series(car, index=b.index)
        g = v.dropna()
        _, t, p, n = date_cluster_test(g, b.loc[g.index, "r0"], -1)
        out[name] = {"mean_car20": float(g.mean()), "t_negative": t, "events": int(len(g))}
    fo_syms = sorted(set(b["symbol"]))
    f = B.futures()
    stk = f[f["kind"] == "STK"][["date", "symbol"]].drop_duplicates()
    m50 = n50_membership()
    rr = r.stack().rename("ret").reset_index()
    rr.columns = ["date", "symbol", "ret"]
    rr = rr.merge(stk, on=["date", "symbol"], how="inner")              # point in time: futures that day
    rr = rr[[not in_n50_fast(s_, d_, m50) for s_, d_ in zip(rr["symbol"], rr["date"])]]
    ew = rr.groupby("date")["ret"].mean()
    car = []
    for s_, a_, z_ in zip(b["symbol"], b["entry"], b["exit"]):
        rs = r.loc[(r.index > a_) & (r.index <= z_), s_] if s_ in r.columns else pd.Series(dtype=float)
        rb = ew[(ew.index > a_) & (ew.index <= z_)]
        car.append(float(np.prod(1 + rs.to_numpy()) - np.prod(1 + rb.fillna(0).to_numpy())) if len(rs) and not rs.isna().any() else np.nan)
    v = pd.Series(car, index=b.index).dropna()
    _, t, p, n = date_cluster_test(v, b.loc[v.index, "r0"], -1)
    out["equal_weight_universe"] = {"mean_car20": float(v.mean()), "t_negative": t, "events": int(len(v))}
    return out


def midcap_exposure(t: pd.DataFrame) -> dict:
    """Descriptive: regress the trade's gross P&L on the (NIFTY MIDCAP 150 - NIFTY 50) return over the same window. The
    hedge is NIFTY futures, so a short in a non-NIFTY-50 name is also a short midcap-vs-large-cap bet."""
    from marketdata.nse_fo import load_index
    ix = load_index()
    lv = {n: ix[ix["index"] == n].assign(date=lambda x: pd.to_datetime(x["date"])).set_index("date")["close"].sort_index()
          for n in ("NIFTY MIDCAP 150", "NIFTY 50")}
    spread = []
    for a_, z_ in zip(t["entry"], t["exit"]):
        try:
            ra = {n: lv[n][:z_].iloc[-1] / lv[n][:a_].iloc[-1] - 1 for n in lv}
            spread.append(ra["NIFTY MIDCAP 150"] - ra["NIFTY 50"])
        except (IndexError, KeyError):
            spread.append(np.nan)
    x = pd.DataFrame({"g": t["gross"].to_numpy(), "s": spread}).dropna()
    if len(x) < 30:
        return {}
    b1, b0 = np.polyfit(x["s"], x["g"], 1)
    return {"slope_on_midcap_minus_nifty": float(b1), "intercept_gross_bp": float(b0 * 1e4), "mean_spread_bp": float(x["s"].mean() * 1e4), "n": int(len(x))}


def promise_pilot() -> dict:
    """Part E (descriptive): the promise claims in Company Memory by metric and review status. Outcomes are never
    auto-resolved (v0.7 rule: a person or a verbatim-quoting model must confirm MET/MISSED)."""
    import sqlite3
    c = sqlite3.connect("data/intel/graph.sqlite")
    cl = pd.read_sql("select entity, metric, due, status from claims where kind = 'promise'", c)
    rv = pd.read_sql("select verdict, count(*) n from promise_reviews group by verdict", c)
    due = cl[pd.to_datetime(cl["due"], errors="coerce") <= pd.Timestamp("2026-09-30")]
    return {"promises": int(len(cl)), "companies": int(cl["entity"].nunique()), "by_metric": cl["metric"].value_counts().to_dict(),
            "due_by_2026_09": int(len(due)), "status_of_due": due["status"].str.split("(").str[0].str.strip().value_counts().to_dict(),
            "human_verdicts": dict(zip(rv["verdict"], rv["n"])) if len(rv) else {},
            "auto_checkable_revenue_promises": int((cl["metric"] == "revenue_growth").sum())}


def n50_method_check() -> dict:
    """Descriptive (already-seen universe): the same machinery inside point-in-time NIFTY 50 (v0.5's universe)."""
    a = announcements().copy()
    a["r0"] = [r0_of(t) for t in a["available_at"]]
    a = a.dropna(subset=["r0"])
    a = a[(a["r0"] >= START_R0) & (a["r0"] <= END_R0)]
    a = a[[in_n50(s, d) for s, d in zip(a["symbol"], a["r0"])]]
    r, m = E.price_returns()
    a["reaction"] = [float(r.at[d, s] - m.get(d, np.nan)) if (s in r.columns and d in r.index) else np.nan for s, d in zip(a["symbol"], a["r0"])]
    a = a.dropna(subset=["reaction"]).sort_values("r0").reset_index(drop=True)
    d, v = a["r0"].to_numpy(), a["reaction"].to_numpy()
    thr = [np.quantile(v[:np.searchsorted(d, d[i])], 1 / 3) if np.searchsorted(d, d[i]) >= 100 else np.nan for i in range(len(a))]
    b = a[a["reaction"] < np.array(thr)].copy()
    b["entry"] = [E.next_session(x, 1) for x in b["r0"]]
    b["exit"] = [E.next_session(x, 1 + HOLD) for x in b["r0"]]
    b = b.dropna(subset=["entry", "exit"])
    b["car20"] = [E.car(s, x, y) for s, x, y in zip(b["symbol"], b["entry"], b["exit"])]
    rr = []
    for s_, d_ in zip(b["symbol"], b["entry"]):
        try:
            rr.append(float(r.at[d_, s_] - m.get(d_, np.nan)))
        except KeyError:
            rr.append(np.nan)
    b["day1"] = rr
    g = b.dropna(subset=["car20"])
    _, t, p, n = date_cluster_test(g["car20"], g["r0"], -1)
    return {"bad_events": int(len(g)), "mean_car20_from_r0+1_close": float(g["car20"].mean()), "t_negative": t,
            "mean_day_r0+1": float(b["day1"].mean()), "note": "v0.5 measured from the r0+1 OPEN with an equal-weight benchmark, 2018-2026"}


# ------------------------------------------------------------------------------------------------ audit / run
def audit(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    a = announcements()
    ev = events()
    after = a["available_at"].map(lambda t: (t.hour, t.minute) >= (15, 0))
    allu = ev
    a_ = {"symbols_with_index": int(a["symbol"].nunique()), "announcements_2016_2026": int(((a["period_end"] >= "2015-12-31")).sum()),
          "share_filed_after_close": float(after.mean()), "events_out_of_n50": int(len(allu)), "events_bad_after_burn_in": int(allu["bad"].sum()),
          "events_by_year": allu.groupby(allu["r0"].dt.year).size().to_dict(),
          "membership_check_2026_09_30": int(sum(in_n50(s, pd.Timestamp("2026-09-30")) for s in n50_membership()["symbol"].unique()))}
    try:
        from scripts.v13_data import load_facts
        fx = load_facts()
        a_["xbrl_filings_parsed"] = int(len(fx))
        a_["xbrl_net_profit_share"] = float(fx["net_profit"].notna().mean()) if len(fx) else 0.0
    except Exception as e:
        a_["xbrl"] = str(e)
    (out / "audit.json").write_text(json.dumps(a_, indent=1, default=str))
    return a_


def run(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    u = events()
    tested = u[u["n_prior"] >= BURN_IN].copy()
    bad = tested[tested["bad"]].copy()
    bad["car20"] = car20(bad)
    res = {"fields": FIELDS, "events_tested": int(len(tested)), "bad_events": int(len(bad))}
    b = bad.dropna(subset=["car20"])
    m, t, p, n = date_cluster_test(b["car20"], b["r0"], sign=-1)
    gm = (-b["car20"]).groupby(b["r0"].dt.to_period("M")).mean()
    _, t_m, p_m, n_m = B.nwt(gm.to_numpy())
    g20 = (-b["car20"]).groupby(b["r0"]).mean()
    _, t_20, p_20, _ = B.nwt(g20.to_numpy(), lags=20)
    res["HM1_drift_robust_inference"] = {"t_month_clustered": t_m, "p_month_clustered": p_m, "months": n_m, "t_nw20_by_date": t_20, "p_nw20": p_20,
                                         "mean_of_date_means": float(-g20.mean())}
    res["HM1_drift"] = {"mean_car20": float(b["car20"].mean()), "median_car20": float(b["car20"].median()), "t_negative": t, "p": p,
                        "dates": n, "events": int(len(b)),
                        "by_subperiod": {k: float(b[(b["r0"] >= a_) & (b["r0"] <= z)]["car20"].mean()) for k, (a_, z) in SUBS.items()}}
    # descriptive comparison: the same rule inside NIFTY 50 (v0.5 universe; already seen)
    rr, mm = E.price_returns()
    d1 = []
    for s_, d_ in zip(b["symbol"], b["entry"]):
        try:
            d1.append(float(rr.at[d_, s_] - mm.get(d_, np.nan)))
        except KeyError:
            d1.append(np.nan)
    res["HM1_drift"]["descriptive_day_r0+1_mean"] = float(np.nanmean(d1))
    res["descriptive_n50_method_check"] = n50_method_check()
    res["HM1_drift"]["descriptive_benchmark_sensitivity"] = benchmark_sensitivity(b)
    res["partE_promise_pilot"] = promise_pilot()
    tr = trades(b)
    tr2 = trades(b, mult=2.0)
    res["HM1_trade"] = trade_summary(tr, tr2)
    res["HM1b"] = hm1b(b)
    hp = B.holm({"HM1-drift": p, "HM1-trade": res["HM1_trade"]["p"], "HM1b": res["HM1b"].get("p", 1.0)})
    res["holm"] = hp
    r = res["HM1_trade"]
    steps = [("statistical", hp["HM1-trade"] < 0.05 and r["mean_net_bp"] > 0),
             ("excess_ge_1.5pct_and_mean_net_positive", r["ann_excess_on_deployed_capital"] >= 0.015 and r["mean_net_bp"] > 0),
             ("after_tax_ge_1pct", r["after_tax_ann_excess"] >= 0.01), ("cost2x_ge_0.75pct", (r["cost2x_ann_excess"] or -1) >= 0.0075),
             ("beta_abs_lt_0.10", abs(r["beta_to_nifty"]) < 0.10), ("capacity", r["capacity_ann_excess"] >= 0.015),
             ("subperiods_2_of_3", sum(1 for v in r["subperiods_ann_excess"].values() if v is not None and v > 0) >= 2),
             ("distress_check_without_10_worst_names", r["mean_net_without_10_worst_names_bp"] > 0)]
    failed = next((k for k, ok in steps if not ok), None)
    res["HM1_trade"]["descriptive_midcap_exposure"] = midcap_exposure(tr)
    res["gate"] = {"steps": {k: bool(v) for k, v in steps}, "first_failed": failed,
                   "verdict": "PASS -> forward paper candidate" if failed is None else f"REJECTED at {failed}"}
    bad.to_csv(out / "bad_events.csv", index=False)
    tr.to_csv(out / "trades.csv", index=False)
    (out / "v13.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["audit", "run"])
    ap.add_argument("--out", default="reports/v1.3_dev")
    a = ap.parse_args()
    r = audit(Path(a.out)) if a.cmd == "audit" else run(Path(a.out))
    print(json.dumps(r, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
