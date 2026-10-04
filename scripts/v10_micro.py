"""v1.0 studies M1 (execution alpha) and M2 (market-making economics), research/preregistration_v1.0.md §2-§3.
Both run on the v0.9 M0 grids (100 ms) of the 7 M0 test days and use the M0 model fitted on the 13 training days
(data/micro/btcusdt/train_model.json) without refitting. Per-day results are cached under data/micro/btcusdt/v10/.

Timing conventions (shared with M0's E2): the book at grid index t is the state at the end of bucket t; trades in
bucket t+1 happen after it. An order decided at t reaches the exchange at t+l (l = latency rounded UP to buckets) and
is priced off the book at t+l; it can fill on trades from bucket t+l+1. Queue rule: we join behind the displayed size
Q; a fill needs aggressive volume at our price >= Q + size, or a trade through our price. Cancellations ahead of us
never advance our queue position (conservative).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from lab.v05 import nw_ols
from marketdata.micro_binance import ROOT
from scripts.v09_analysis import FEATS, _windows, features, load_day, sample

CACHE = ROOT / "v10"
FEES = {"retail": (2.0, 5.0), "best_case": (0.0, 1.7)}      # (maker, taker) bp
LAT = 3                                                      # 250 ms rounded up to 300 ms
SIZE = 0.01
EVERY = 3000                                                 # 5 minutes of 100 ms buckets
DEADLINE = 600                                               # 60 s
THETA = 0.5                                                  # bp, M1 P2 trigger
THETA_DESC = [0.25, 1.0, 2.0]                                # descriptive only


def model() -> dict:
    return json.loads((ROOT / "train_model.json").read_text())


def predictions(day: str, m: dict) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"pred_{day}.npy"
    if p.exists():
        return np.load(p)
    f = features(load_day(day), m["ofi_sd"], m["p99_trade_size"])
    pred = (np.column_stack([np.ones(len(f)), f[FEATS].to_numpy(float)]) @ np.array(m["beta"])).astype(np.float32)
    np.save(p, pred)
    return pred


# ------------------------------------------------------------------------------------------------ M1
def m1_orders(d: pd.DataFrame, pred: np.ndarray, theta: float = THETA, lat: int = LAT) -> pd.DataFrame:
    """One row per parent order: the three policies' execution price, whether it was a maker fill, and timing."""
    bid, ask = d["bid"].to_numpy(), d["ask"].to_numpy()
    bq, aq = d["bidq"].to_numpy(float), d["askq"].to_numpy(float)
    mid = (bid + ask) / 2
    n = len(d)
    i = np.arange(0, n, EVERY)
    i = i[i + DEADLINE + lat < n]
    rows = []
    K = DEADLINE - lat                                       # buckets the passive order can fill in: e+1 .. i+600
    for s in (1, -1):
        e = i + lat
        arr = mid[i]
        far = (lambda t: ask[t]) if s > 0 else (lambda t: bid[t])
        p0 = far(e)
        P = bid[e] if s > 0 else ask[e]
        Q = bq[e] if s > 0 else aq[e]
        if s > 0:
            px, vol = _windows(d["sell_min_px"].to_numpy(), e, K), _windows(d["sell_vol_at_min"].to_numpy(float), e, K)
            through = px < P[:, None]
        else:
            px, vol = _windows(d["buy_max_px"].to_numpy(), e, K), _windows(d["buy_vol_at_max"].to_numpy(float), e, K)
            through = px > P[:, None]
        at = np.where(px == P[:, None], np.nan_to_num(vol), 0.0)
        fill = through | (np.cumsum(at, axis=1) >= (Q + SIZE)[:, None])
        has = fill.any(axis=1)
        fb = np.where(has, e + fill.argmax(axis=1) + 1, n + 10)   # bucket of the passive fill
        late = i + DEADLINE + lat
        p1 = np.where(has, P, far(late))
        m1 = has.copy()
        # P2: checks at i, i+10, ..., i+590 while not yet filled; the cancel is effective at the check (fills after it
        # are ignored, which can only hurt P2), the cross executes at check + latency.
        checks = i[:, None] + np.arange(0, DEADLINE, 10)[None, :]
        pc = pred[checks]
        trig = (s * pc >= theta) & (checks < fb[:, None])
        trig &= ~np.isnan(pc)
        anyt = trig.any(axis=1)
        tc = np.where(anyt, checks[np.arange(len(i)), trig.argmax(axis=1)], -1)
        p2 = np.where(anyt, far(np.where(anyt, tc + lat, 0)), p1)
        m2 = np.where(anyt, False, m1)
        rows.append(pd.DataFrame({"t": i, "hour": i // 36000, "side": s, "arrival": arr, "p0": p0, "p1": p1, "maker1": m1,
                                  "p2": p2, "maker2": m2, "triggered": anyt, "trig_at_decision": anyt & (tc == i),
                                  "fill_wait_s": np.where(has, (fb - i) / 10, np.nan),
                                  "spread_bp": (ask[i] - bid[i]) / mid[i] * 1e4}))
    return pd.concat(rows, ignore_index=True)


def is_bp(o: pd.DataFrame, fees: tuple[float, float]) -> pd.DataFrame:
    mk, tk = fees
    out = pd.DataFrame({"hour": o["hour"], "side": o["side"]})
    out["IS0"] = o["side"] * np.log(o["p0"] / o["arrival"]) * 1e4 + tk
    out["IS1"] = o["side"] * np.log(o["p1"] / o["arrival"]) * 1e4 + np.where(o["maker1"], mk, tk)
    out["IS2"] = o["side"] * np.log(o["p2"] / o["arrival"]) * 1e4 + np.where(o["maker2"], mk, tk)
    return out


def m1_day(day: str, force: bool = False) -> dict:
    p = CACHE / f"m1_{day}.parquet"
    if p.exists() and not force:
        return {"day": day, "cached": True}
    m = model()
    d = load_day(day)
    pred = predictions(day, m)
    frames = []
    for th in [THETA] + THETA_DESC:
        o = m1_orders(d, pred, th)
        o["theta"] = th
        frames.append(o)
    out = pd.concat(frames, ignore_index=True)
    out["day"] = day
    out.to_parquet(p)
    return {"day": day, "orders": int((out.theta == THETA).sum())}


# ------------------------------------------------------------------------------------------------ M2
def mm_day(d: pd.DataFrame, pred: np.ndarray, lo: float, hi: float, skew: bool, lat: int = LAT, size: float = SIZE,
           qfrac: float = 1.0) -> dict:
    """Event loop over the 100 ms grid. Inventory in units of `size` (|inv| <= 1). Returns per-hour mark-to-mid P&L
    (price units per 1 BTC, i.e. multiply by size for USDT), passive fills, and taker notional (for flattening fees).
    qfrac scales the queue ahead of us at placement (1.0 = pre-registered: behind the whole displayed size; 0.5 and 0.0
    are descriptive bounds added after the review, since cancellations ahead of us are never credited)."""
    INF = 1 << 60
    bid, ask = d["bid"].tolist(), d["ask"].tolist()
    bq, aq = d["bidq"].tolist(), d["askq"].tolist()
    smin = np.nan_to_num(d["sell_min_px"].to_numpy(float), nan=np.inf).tolist()
    svat = d["sell_vol_at_min"].fillna(0).tolist()
    bmax = np.nan_to_num(d["buy_max_px"].to_numpy(float), nan=-np.inf).tolist()
    bvat = d["buy_vol_at_max"].fillna(0).tolist()
    pr = np.nan_to_num(pred.astype(float), nan=0.0).tolist()
    n = len(bid)
    # per side: [live, P, Q, cum, from, dies, pending_at]
    B = [False, 0.0, 0.0, 0.0, 0, INF, -1]
    A = [False, 0.0, 0.0, 0.0, 0, INF, -1]
    inv, inv_t, cash, flat_at = 0, -1, 0.0, -1
    nh = n // 36000 + 1
    pnl_h, fills_h, taker_h, flat_h, v_prev = [0.0] * nh, [0] * nh, [0.0] * nh, [0] * nh, 0.0
    fill_px, fill_side, fill_t = [], [], []
    for t in range(n):
        # 1) fills from trades in bucket t
        if B[0] and B[4] <= t <= B[5] and inv < 1:
            sp = smin[t]
            hit = sp < B[1]
            if not hit and sp == B[1]:
                B[3] += svat[t]
                hit = B[3] >= B[2] + size
            if hit:
                inv += 1
                cash -= B[1]
                fills_h[t // 36000] += 1
                fill_px.append(B[1]); fill_side.append(1); fill_t.append(t)
                inv_t = t if inv != 0 else -1
                B[0], B[5] = False, INF
        if A[0] and A[4] <= t <= A[5] and inv > -1:
            bp = bmax[t]
            hit = bp > A[1]
            if not hit and bp == A[1]:
                A[3] += bvat[t]
                hit = A[3] >= A[2] + size
            if hit:
                inv -= 1
                cash += A[1]
                fills_h[t // 36000] += 1
                fill_px.append(A[1]); fill_side.append(-1); fill_t.append(t)
                inv_t = t if inv != 0 else -1
                A[0], A[5] = False, INF
        # expire cancelled orders
        if B[0] and t >= B[5]:
            B[0], B[5] = False, INF
        if A[0] and t >= A[5]:
            A[0], A[5] = False, INF
        # 2) flatten executes at the far touch
        if flat_at == t:
            if inv > 0:
                cash += bid[t]
                taker_h[t // 36000] += bid[t]
            elif inv < 0:
                cash -= ask[t]
                taker_h[t // 36000] += ask[t]
            if inv != 0:
                flat_h[t // 36000] += 1
            inv, inv_t, flat_at = 0, -1, -1
        # 3) decisions at the end of bucket t
        if inv != 0 and flat_at < 0 and t - inv_t >= DEADLINE:
            flat_at = t + lat
            S = A if inv > 0 else B                              # closing side: cancel at once (pessimistic)
            S[0], S[5], S[6] = False, INF, -1
        p_ = pr[t]
        want_b = inv < 1 and flat_at < 0 and (not skew or p_ > lo)
        want_a = inv > -1 and flat_at < 0 and (not skew or p_ < hi)
        for S, want, px_, q_ in ((B, want_b, bid, bq), (A, want_a, ask, aq)):
            if S[6] == t:                                       # placement reaches the exchange now
                S[6] = -1
                if want:
                    S[0], S[1], S[2], S[3], S[4], S[5] = True, px_[t], q_[t] * qfrac, 0.0, t + 1, INF
                continue
            if S[6] > t:                                        # a placement is in flight
                continue
            if S[0] and S[5] == INF:                            # live, not cancelling
                if not want:
                    S[5] = t + lat
                elif px_[t] != S[1]:                            # touch moved: re-join at the back
                    S[5] = t + lat
                    S[6] = t + lat
            elif not S[0] and want:
                S[6] = t + lat
        # hourly mark to mid
        if (t + 1) % 36000 == 0 or t == n - 1:
            v = cash + inv * (bid[t] + ask[t]) / 2
            pnl_h[t // 36000] += v - v_prev
            v_prev = v
    mids = [(bid[min(h * 36000 + 18000, n - 1)] + ask[min(h * 36000 + 18000, n - 1)]) / 2 for h in range(nh)]
    # adverse selection: mid 5 s and 30 s after each passive fill, signed, in bp
    mid = (np.array(bid) + np.array(ask)) / 2
    ft, fs, fp = np.array(fill_t, int), np.array(fill_side), np.array(fill_px)
    adv = {}
    for k, h in (("5s", 50), ("30s", 300)):
        j = np.minimum(ft + h, n - 1)
        adv[k] = float(np.mean(fs * np.log(mid[j] / fp) * 1e4)) if len(ft) else None
    spread_cap = float(np.mean(fs * np.log(mid[ft] / fp) * 1e4)) if len(ft) else None
    return {"pnl_h": pnl_h[:24], "fills_h": fills_h[:24], "taker_h": taker_h[:24], "flat_h": flat_h[:24], "mid_h": mids[:24],
            "fills": int(sum(fills_h)), "flattens": int(sum(flat_h)), "mid_after_fill_bp": adv,
            "edge_at_fill_vs_mid_bp": spread_cap, "end_inventory": inv}


def m2_day(day: str, variant: str, force: bool = False) -> dict:
    p = CACHE / f"m2_{day}_{variant}.json"
    if p.exists() and not force:
        return json.loads(p.read_text())
    m = model()
    d = load_day(day)
    pred = predictions(day, m)
    lo, hi = m["pred_deciles"]
    qf = float(variant.split("_q")[1]) if "_q" in variant else 1.0
    r = mm_day(d, pred, lo, hi, skew=variant.startswith("skew"), qfrac=qf)
    r.update({"day": day, "variant": variant, "theta": [lo, hi]})
    p.write_text(json.dumps(r))
    return r


# ------------------------------------------------------------------------------------------------ statistics
def _nwt(v, lags: int = 3):
    v = np.array([x for x in v if x is not None and not np.isnan(x)], float)
    fit = nw_ols(v, np.empty((len(v), 0)), lags=lags)
    t = float(fit["t"][0])
    return float(v.mean()), t, float(sps.t.sf(t, len(v) - 1)), int(len(v))


def holm(ps: dict) -> dict:
    keys = sorted(ps, key=ps.get)
    out, run = {}, 0.0
    for r, k in enumerate(keys):
        run = max(run, min(1.0, ps[k] * (len(ps) - r)))
        out[k] = run
    return out


def m1_results() -> dict:
    s = sample()
    o = pd.concat([pd.read_parquet(CACHE / f"m1_{day}.parquet") for day in s["test"]], ignore_index=True)
    res = {}
    for fee, fv in FEES.items():
        res[fee] = {}
        for th in [THETA] + THETA_DESC:
            g = o[o.theta == th]
            x = is_bp(g, fv)
            x["day"] = g["day"].to_numpy()
            x["block"] = x["day"] + "_" + x["hour"].astype(int).map("{:02d}".format)     # chronological order for NW
            hb = x.groupby("block")[["IS0", "IS1", "IS2"]].mean()
            means = {k: float(x[k].mean()) for k in ("IS0", "IS1", "IS2")}
            better = "IS0" if means["IS0"] <= means["IS1"] else "IS1"
            daily = x.groupby("day")[["IS0", "IS1", "IS2"]].mean()
            r = {"orders": int(len(x)), "mean_IS_bp": means, "better_of_P0_P1": better,
                 "saving_vs_better_bp": means[better] - means["IS2"],
                 "days_positive_vs_better": int(((daily[better] - daily["IS2"]) > 0).sum()),
                 "daily_saving_vs_better": (daily[better] - daily["IS2"]).round(4).to_dict(),
                 "P1_maker_fill_rate": float(g["maker1"].mean()), "P2_maker_fill_rate": float(g["maker2"].mean()),
                 "P2_trigger_rate": float(g["triggered"].mean()), "P2_trigger_at_decision": float(g["trig_at_decision"].mean()),
                 "median_wait_s_P1_fills": float(g["fill_wait_s"].median()),
                 "by_side": {int(sd): {k: float(v) for k, v in x[x.side == sd][["IS0", "IS1", "IS2"]].mean().items()} for sd in (1, -1)}}
            ha = _nwt((hb["IS0"] - hb["IS2"]).tolist())
            hb_ = _nwt((hb["IS1"] - hb["IS2"]).tolist())
            r["HM1a"] = {"mean": ha[0], "t": ha[1], "p": ha[2], "blocks": ha[3]}
            r["HM1b"] = {"mean": hb_[0], "t": hb_[1], "p": hb_[2], "blocks": hb_[3]}
            hp = holm({"HM1a": ha[2], "HM1b": hb_[2]})
            r["HM1a"]["holm_p"], r["HM1b"]["holm_p"] = hp["HM1a"], hp["HM1b"]
            steps = [("HM1a_HM1b_significant", hp["HM1a"] < 0.05 and hp["HM1b"] < 0.05 and ha[0] > 0 and hb_[0] > 0),
                     ("saving_ge_0.5bp", r["saving_vs_better_bp"] >= 0.5),
                     ("positive_5_of_7_days", r["days_positive_vs_better"] >= 5)]
            failed = next((k for k, ok in steps if not ok), None)
            r["gate"] = {"steps": {k: bool(v) for k, v in steps}, "first_failed": failed,
                         "verdict": "PASS (execution improvement; research cost default only)" if failed is None else f"REJECTED at {failed}"}
            d01 = _nwt((hb["IS0"] - hb["IS1"]).tolist())
            r["descriptive_P0_minus_P1"] = {"mean": d01[0], "t": d01[1], "p": d01[2], "blocks": d01[3]}
            r["primary"] = bool(fee == "retail" and th == THETA)
            res[fee][str(th)] = r
    g = o[o.theta == THETA]
    mk = g[g["maker1"]]
    cal = {"crypto_btcusdt_perp": {
        "source": "v1.0 M1, 7 M0 test days, 0.01 BTC parent orders, 300 ms latency, 60 s budget",
        "passive_fill_prob_60s": float(g["maker1"].mean()),
        "passive_price_vs_arrival_bp": float((mk["side"] * np.log(mk["p1"] / mk["arrival"]) * 1e4).mean()),
        "unfilled_cross_cost_bp_ex_fee": float((g.loc[~g["maker1"], "side"] * np.log(g.loc[~g["maker1"], "p1"] / g.loc[~g["maker1"], "arrival"]) * 1e4).mean()),
        "half_spread_bp": float(g["spread_bp"].median() / 2)}}
    res["calibration"] = cal
    return res


def m2_results(fees: dict = FEES) -> dict:
    s = sample()
    out = {}
    blocks = {}
    for var in ("plain", "skew"):
        days = [m2_day(day, var) for day in s["test"]]
        out[var] = {}
        for fee, (mk, tk) in fees.items():
            bl, tot_p, tot_n = [], 0.0, 0.0
            for dd in days:
                for h in range(24):
                    f_ = dd["fills_h"][h]
                    g = dd["pnl_h"][h] - tk / 1e4 * dd["taker_h"][h]
                    tot_p += g
                    tot_n += f_ * dd["mid_h"][h]
                    bl.append(g / (f_ * dd["mid_h"][h]) * 1e4 if f_ > 0 else np.nan)
            blocks[(var, fee)] = bl
            out[var][fee] = {"break_even_maker_fee_bp": tot_p / tot_n * 1e4 if tot_n else None,
                             "mean_hour_block_bp": float(np.nanmean(bl))}
        out[var]["fills"] = int(sum(dd["fills"] for dd in days))
        out[var]["flattens"] = int(sum(dd["flattens"] for dd in days))
        out[var]["flatten_share"] = out[var]["flattens"] / max(out[var]["fills"], 1)
        out[var]["edge_at_fill_vs_mid_bp"] = float(np.average([dd["edge_at_fill_vs_mid_bp"] for dd in days], weights=[dd["fills"] for dd in days]))
        out[var]["mid_after_fill_bp"] = {k: float(np.average([dd["mid_after_fill_bp"][k] for dd in days], weights=[dd["fills"] for dd in days])) for k in ("5s", "30s")}
        out[var]["per_day_break_even_retail_taker"] = {dd["day"]: (sum(dd["pnl_h"]) - 5e-4 * sum(dd["taker_h"])) / max(sum(f * m for f, m in zip(dd["fills_h"], dd["mid_h"])), 1e-9) * 1e4 for dd in days}
    out["descriptive_queue_bounds"] = {}
    for var in ("plain_q0.5", "plain_q0"):
        days = [m2_day(day, var) for day in s["test"]]
        tp = sum(sum(dd["pnl_h"]) - 5e-4 * sum(dd["taker_h"]) for dd in days)
        tn = sum(sum(f * m for f, m in zip(dd["fills_h"], dd["mid_h"])) for dd in days)
        out["descriptive_queue_bounds"][var] = {"break_even_maker_fee_bp_retail_taker": tp / tn * 1e4, "fills": int(sum(dd["fills"] for dd in days)),
                                                "mid_after_fill_5s_bp": float(np.average([dd["mid_after_fill_bp"]["5s"] for dd in days], weights=[dd["fills"] for dd in days]))}
    ha = _nwt(blocks[("skew", "retail")])
    diff = [a - b for a, b in zip(blocks[("skew", "retail")], blocks[("plain", "retail")])]
    hb_ = _nwt(diff)
    hp = holm({"HM2a": ha[2], "HM2b": hb_[2]})
    out["HM2a"] = {"mean": ha[0], "t": ha[1], "p": ha[2], "blocks": ha[3], "holm_p": hp["HM2a"], "pass": bool(hp["HM2a"] < 0.05 and ha[0] > 0)}
    out["HM2b"] = {"mean": hb_[0], "t": hb_[1], "p": hb_[2], "blocks": hb_[3], "holm_p": hp["HM2b"], "pass": bool(hp["HM2b"] < 0.05 and hb_[0] > 0)}
    be = out["skew"]["retail"]["break_even_maker_fee_bp"]
    out["reading"] = ("viable at retail maker fee" if be >= FEES["retail"][0] else
                      "viable only with a lower maker fee" if be >= 0 else
                      "needs a rebate (within typical programmes)" if be >= -0.5 else
                      "not viable even with typical published rebates")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["m1", "m2", "results"])
    ap.add_argument("--day", type=int)
    ap.add_argument("--variant", default="plain")   # plain | skew | plain_q0.5 | plain_q0
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default="reports/v1.0_dev")
    a = ap.parse_args()
    s = sample()
    if a.cmd == "m1":
        print(m1_day(s["test"][a.day], a.force))
    elif a.cmd == "m2":
        import time
        t0 = time.time()
        r = m2_day(s["test"][a.day], a.variant, a.force)
        print(r["day"], a.variant, r["fills"], r["flattens"], round(time.time() - t0, 1), "s")
    else:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        r = {"M1": m1_results(), "M2": m2_results()}
        cal = r["M1"]["calibration"]
        cal["crypto_btcusdt_perp"]["adverse_mid_move_after_passive_fill_5s_bp"] = -r["M2"]["plain"]["mid_after_fill_bp"]["5s"]
        cal["crypto_btcusdt_perp"]["source"] += "; adverse move after a passive fill from M2 (plain quotes, 5 s, sign: + = against us)"
        Path("research/data_audits").mkdir(parents=True, exist_ok=True)
        Path("research/data_audits/v1.0_aem_calibration.json").write_text(json.dumps(cal, indent=1))
        (out / "m1_m2.json").write_text(json.dumps(r, indent=1, default=str))
        print(json.dumps(r, default=str)[:3000])


if __name__ == "__main__":
    main()
