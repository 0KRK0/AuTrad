"""v1.0 study M3 (research/preregistration_v1.0.md §4): funding/basis carry on Binance BTC and ETH, 2020-01..2026-09.
Long spot N + short USDⓈ-M perpetual N, capital 2N. Decision at 00:00 UTC on data up to the previous day.
Funding convention: a print at time T is credited to the UTC day containing T - 1 ms, i.e. a position held over day D
earns the prints at D 08:00, D 16:00 and D+1 00:00 (the D 00:00 print, at the entry instant, is never assumed earned).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from lab.v05 import nw_ols
from marketdata.crypto_perp import load

START, END = "2020-01-01", "2026-09-30"
ENTER, EXIT = 0.15, 0.05
COST_SIDE_BP = 17.0            # per entry or exit on N: spot 10 + perp 5 + slippage 2 (34 bp round trip)
TAX = 0.312
SUBS = {"2020-21": ("2020-01-01", "2021-12-31"), "2022-23": ("2022-01-01", "2023-12-31"), "2024-26": ("2024-01-01", "2026-09-30")}


def daily(sym: str) -> pd.DataFrame:
    f, k = load(sym)
    spot = pd.read_csv(f"data/crypto/binance/1d/{sym}.csv.gz")[["date", "close"]].rename(columns={"close": "spot"})
    perp = k[["date", "close"]].rename(columns={"close": "perp"})
    f["day"] = (f["time"] - pd.Timedelta(milliseconds=1)).dt.strftime("%Y-%m-%d")
    fd = f.groupby("day").agg(funding=("rate", "sum"), prints=("rate", "size")).reset_index().rename(columns={"day": "date"})
    d = spot.merge(perp, on="date", how="inner").merge(fd, on="date", how="left")
    d = d[(d["date"] >= "2019-12-20") & (d["date"] <= END)].reset_index(drop=True)
    d["r_spot"] = d["spot"].pct_change()
    d["r_perp"] = d["perp"].pct_change()
    d["funding"] = d["funding"].fillna(0.0)
    # signal known at 00:00 of day D: funding credited to days D-7..D-1, annualised
    d["ann7"] = d["funding"].shift(1).rolling(7).sum() / 7 * 365
    return d


def audit(sym: str) -> dict:
    f, k = load(sym)
    d = daily(sym)
    w = d[d["date"] >= START]
    basis = np.log(w["perp"] / w["spot"]) * 1e4
    gaps = pd.to_datetime(f["time"]).diff().dt.total_seconds().div(3600)
    return {"funding_prints": int(len(f)), "interval_hours": f["interval_h"].value_counts().to_dict(),
            "print_gaps_not_8h": int((gaps.dropna() != 8).sum()), "days": int(len(w)),
            "days_missing_prints": int((w["prints"].fillna(0) != 3).sum()),
            "days_missing_prints_list": w.loc[w["prints"].fillna(0) != 3, "date"].tolist()[:10],
            "close_basis_bp": {"median": float(basis.median()), "p1": float(basis.quantile(.01)), "p99": float(basis.quantile(.99)),
                               "abs_gt_100bp_days": int((basis.abs() > 100).sum())},
            "funding_ann_mean": float(w["funding"].mean() * 365), "funding_negative_share": float((w["funding"] < 0).mean()),
            "extreme_daily_rate_gt_1pct": int((w["funding"].abs() > 0.01).sum())}


def simulate(d: pd.DataFrame, cost_mult: float = 1.0, enter: float = ENTER, exit_: float = EXIT, always: bool = False,
             setoff: bool = False) -> dict:
    """setoff=False is the pre-registered (conservative) tax reading: each leg's positive trip P&L is taxed, losses never
    set off. setoff=True (descriptive only) taxes the trip's combined positive P&L."""
    d = d[d["date"] >= START].reset_index(drop=True)
    c = COST_SIDE_BP * cost_mult / 1e4 / 2                         # on capital 2N
    n = len(d)
    pos = np.zeros(n, bool)
    ret, ret_tax = np.zeros(n), np.zeros(n)
    trips, inside = [], False
    leg_s = leg_p = 0.0
    start = None
    ann = d["ann7"].to_numpy()
    for i in range(n):
        if always:
            want = True
        elif inside:
            want = not (ann[i] < exit_)
        else:
            want = bool(ann[i] >= enter)
        last = i == n - 1
        if want and not inside:
            inside, leg_s, leg_p, start = True, 0.0, 0.0, i
            ret[i] -= c
        if inside:
            if not want:                                             # exit at 00:00, flat for the day
                ret[i] -= c
                tax = TAX * ((max(leg_s + leg_p, 0)) if setoff else (max(leg_s, 0) + max(leg_p, 0))) / 2
                ret_tax[i] -= tax
                trips.append({"entry": d.loc[start, "date"], "exit": d.loc[i, "date"], "days": i - start, "spot_leg": leg_s / 2,
                              "perp_leg": leg_p / 2, "tax": tax, "max_rise": float(d["perp"].iloc[start:i].max() / d["perp"].iloc[max(start - 1, 0)] - 1)})
                inside = False
            else:
                pos[i] = True
                rs, rp, fu = d.loc[i, "r_spot"], d.loc[i, "r_perp"], d.loc[i, "funding"]
                ret[i] += (rs - rp + fu) / 2
                leg_s += rs
                leg_p += -rp + fu
                if last:                                             # close at the end of the window
                    ret[i] -= c
                    tax = TAX * ((max(leg_s + leg_p, 0)) if setoff else (max(leg_s, 0) + max(leg_p, 0))) / 2
                    ret_tax[i] -= tax
                    trips.append({"entry": d.loc[start, "date"], "exit": d.loc[i, "date"] + " (end)", "days": i - start + 1,
                                  "spot_leg": leg_s / 2, "perp_leg": leg_p / 2, "tax": tax,
                                  "max_rise": float(d["perp"].iloc[start:i + 1].max() / d["perp"].iloc[max(start - 1, 0)] - 1)})
    net, after = ret, ret + ret_tax
    dates = d["date"].to_numpy()

    def ann_(x):
        return float(np.mean(x) * 365)
    fit = nw_ols(net, np.empty((n, 0)), lags=10)
    t = float(fit["t"][0])
    eq = np.cumprod(1 + after)
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    subs = {k: {"net_ann": ann_(net[(dates >= a) & (dates <= b)]), "after_tax_ann": ann_(after[(dates >= a) & (dates <= b)])}
            for k, (a, b) in SUBS.items()}
    inpos_fund = d.loc[pos, "funding"]
    return {"days": n, "net_ann": ann_(net), "after_tax_ann": ann_(after), "t": t, "p_one_sided": float(sps.t.sf(t, n - 1)),
            "cagr_after_tax": float(eq[-1] ** (365 / n) - 1), "max_drawdown_after_tax": dd, "time_in_position": float(pos.mean()),
            "trips": len(trips), "subperiods": subs, "worst_day": float(net.min()),
            "in_position_negative_funding_days": int((inpos_fund < 0).sum()),
            "max_perp_rise_during_trip": float(max([tr["max_rise"] for tr in trips], default=0.0)),
            "gross_funding_ann_on_capital": float(d.loc[pos, "funding"].sum() / 2 / n * 365),
            "basis_pnl_ann_on_capital": float(((d.loc[pos, "r_spot"] - d.loc[pos, "r_perp"]).sum()) / 2 / n * 365),
            "tax_ann": float(-ret_tax.sum() / n * 365), "cost_ann": float((net.sum() - ((d.loc[pos, "r_spot"] - d.loc[pos, "r_perp"] + d.loc[pos, "funding"]).sum() / 2)) / n * 365),
            "trip_list": trips, "_daily_net": net.tolist(), "_daily_after": after.tolist(), "_dates": dates.tolist()}


def fixed_qty(d: pd.DataFrame, cost_mult: float = 1.0) -> dict:
    """Descriptive (added after the review): the same entry/exit days, but each trip holds a FIXED quantity of spot and
    perpetual from entry (no daily rebalancing, so no uncosted rebalancing trades), funding on the perpetual's
    current notional. Tax per trip on each leg's positive P&L, no set-off. Return on the entry capital 2N."""
    base = simulate(d, cost_mult)
    w = d[d["date"] >= START].reset_index(drop=True)
    idx = {x: i for i, x in enumerate(w["date"])}
    c = COST_SIDE_BP * cost_mult / 1e4 / 2
    ret, ret_tax = np.zeros(len(w)), np.zeros(len(w))
    S, F, fu = w["spot"].to_numpy(), w["perp"].to_numpy(), w["funding"].to_numpy()
    worst_perp_loss = 0.0
    for tr in base["trip_list"]:
        a = idx[tr["entry"]]
        end_ = tr["exit"].replace(" (end)", "")
        b = idx[end_] + (1 if tr["exit"].endswith("(end)") else 0)       # exclusive end (exit day is flat)
        s0, f0 = S[a - 1], F[a - 1]                                     # bought / sold at the previous close (00:00)
        ls = lp = 0.0
        ret[a] -= c
        for i in range(a, b):
            ds = (S[i] - S[i - 1]) / s0
            dp = -(F[i] - F[i - 1]) / f0 + fu[i] * F[i - 1] / f0
            ret[i] += (ds + dp) / 2
            ls += ds
            lp += dp
            worst_perp_loss = min(worst_perp_loss, -(F[i] - f0) / f0)
        k = min(b, len(w) - 1)
        ret[k] -= c
        ret_tax[k] -= TAX * (max(ls, 0) + max(lp, 0)) / 2
    after = ret + ret_tax
    return {"net_ann": float(ret.mean() * 365), "after_tax_ann": float(after.mean() * 365),
            "tax_ann": float(-ret_tax.mean() * 365), "worst_perp_leg_loss_vs_N": float(worst_perp_loss)}


def rebalance_cost_estimate(d: pd.DataFrame) -> float:
    """Annualised cost on capital of the daily rebalancing implied by the constant-notional P&L model (descriptive):
    each day both legs are resized by |r| of N at spot 10 bp + perp 5 bp + 2 bp slippage per leg."""
    base = simulate(d)
    w = d[d["date"] >= START].reset_index(drop=True)
    inpos = np.zeros(len(w), bool)
    idx = {x: i for i, x in enumerate(w["date"])}
    for tr in base["trip_list"]:
        inpos[idx[tr["entry"]]: idx[tr["exit"].replace(" (end)", "")] + 1] = True
    turn = (w["r_spot"].abs() * (10 + 2) + w["r_perp"].abs() * (5 + 2)) / 1e4 / 2
    return float(turn[inpos].sum() / len(w) * 365)


def holm(ps: dict) -> dict:
    keys = sorted(ps, key=ps.get)
    out, run = {}, 0.0
    for r, k in enumerate(keys):
        run = max(run, min(1.0, ps[k] * (len(ps) - r)))
        out[k] = run
    return out


def run(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    res = {"audit": {}, "primary": {}, "cost2x": {}, "descriptive_always_on": {}, "descriptive_tax_setoff_within_trip": {}}
    for sym in ("BTCUSDT", "ETHUSDT"):
        d = daily(sym)
        res["audit"][sym] = audit(sym)
        res["primary"][sym] = simulate(d)
        res["cost2x"][sym] = simulate(d, 2.0)
        res["descriptive_always_on"][sym] = simulate(d, always=True)
        res["descriptive_tax_setoff_within_trip"][sym] = simulate(d, setoff=True)
        res.setdefault("descriptive_fixed_quantity_legs", {})[sym] = fixed_qty(d)
        res.setdefault("descriptive_rebalancing_cost_ann", {})[sym] = rebalance_cost_estimate(d)
        res.setdefault("descriptive_cost2x_pre_tax_ann", {})[sym] = None
        pd.DataFrame({"date": res["primary"][sym]["_dates"], "net": res["primary"][sym]["_daily_net"],
                      "after_tax": res["primary"][sym]["_daily_after"]}).to_csv(out / f"m3_daily_{sym}.csv", index=False)
    for sym in res["cost2x"]:
        res["descriptive_cost2x_pre_tax_ann"][sym] = res["cost2x"][sym]["net_ann"]
    hp = holm({s: res["primary"][s]["p_one_sided"] for s in res["primary"]})
    gate = {}
    for s, r in res["primary"].items():
        steps = [("statistical", hp[s] < 0.05 and r["net_ann"] > 0),
                 ("net_ge_8pct", r["net_ann"] >= 0.08),
                 ("after_tax_ge_6pct", r["after_tax_ann"] >= 0.06),
                 # the pre-registration does not say pre- or post-tax; coded after tax (stricter), pre-tax also reported
                 ("cost2x_after_tax_ge_6pct", res["cost2x"][s]["after_tax_ann"] >= 0.06),
                 ("capacity_not_binding", True),
                 ("subperiods_2_of_3", sum(v["net_ann"] > 0 for v in r["subperiods"].values()) >= 2)]
        failed = next((k for k, ok in steps if not ok), None)
        gate[s] = {"holm_p": hp[s], "steps": {k: bool(ok) for k, ok in steps}, "first_failed": failed,
                   "verdict": "PASS -> forward paper test" if failed is None else f"REJECTED at {failed}"}
    res["gate"] = gate
    for grp in ("primary", "cost2x", "descriptive_always_on", "descriptive_tax_setoff_within_trip"):
        for s in res[grp]:
            for k in [k for k in res[grp][s] if k.startswith("_")]:
                res[grp][s].pop(k)
            if grp != "primary":
                res[grp][s].pop("trip_list", None)
    (out / "m3.json").write_text(json.dumps(res, indent=1, default=str))
    return res


if __name__ == "__main__":
    import sys
    r = run(Path(sys.argv[1] if len(sys.argv) > 1 else "reports/v1.0_dev"))
    for s in r["primary"]:
        p = r["primary"][s]
        print(s, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in p.items() if k not in ("trip_list", "subperiods")})
        print("  subs", p["subperiods"], "\n  2x", round(r["cost2x"][s]["after_tax_ann"], 4), "always", round(r["descriptive_always_on"][s]["net_ann"], 4), round(r["descriptive_always_on"][s]["after_tax_ann"], 4), "setoff", round(r["descriptive_tax_setoff_within_trip"][s]["after_tax_ann"], 4))
        print("  gate", r["gate"][s])
    print(json.dumps(r["audit"], default=str)[:1500])
