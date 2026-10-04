"""v0.9 study M0 analysis (research/preregistration_v0.9.md): features, the 11 hypothesis tests on test-day hours,
the training-day OLS model, the queue-aware execution simulator (E1 taker, E2 passive) with a latency curve, and
the microstructure gate. Training days fit everything; test days only measure.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from lab.v05 import nw_ols
from marketdata.micro_binance import ROOT, SIZE_BINS

A = Path("research/data_audits")
H = {"1s": 10, "5s": 50, "30s": 300}
FEATS = ["OBI", "OFI_1s", "TI_1s", "QD", "LARGE", "SPREAD_W", "BASIS_DEV", "FUT_LEAD"]
LAT_MS = [0, 100, 250, 500, 1000, 2000, 5000]
FEES = {"retail": (2.0, 5.0), "best_case": (0.0, 1.7)}           # (maker, taker) in bp
TICK = 0.1


def sample() -> dict:
    return json.loads((ROOT / "sample.json").read_text())


def load_day(day: str) -> pd.DataFrame:
    return pd.read_parquet(ROOT / f"{day}.parquet")


def features(d: pd.DataFrame, ofi_sd: float = 1.0, p99: float = np.inf) -> pd.DataFrame:
    mid = (d["bid"].to_numpy() + d["ask"].to_numpy()) / 2
    lm = np.log(mid)
    bq, aq = d["bidq"].to_numpy(float), d["askq"].to_numpy(float)
    f = pd.DataFrame(index=d.index)
    f["OBI"] = (bq - aq) / (bq + aq)
    f["OFI_1s"] = pd.Series(d["ofi"].to_numpy(float)).rolling(10, min_periods=1).sum().to_numpy() / ofi_sd
    bv = pd.Series(d["buy_vol"].to_numpy(float)).rolling(10, min_periods=1).sum().to_numpy()
    sv = pd.Series(d["sell_vol"].to_numpy(float)).rolling(10, min_periods=1).sum().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        f["TI_1s"] = np.where(bv + sv > 0, (bv - sv) / (bv + sv), 0.0)
        lbq, laq = np.log(np.maximum(bq, 1e-6)), np.log(np.maximum(aq, 1e-6))
    f["QD"] = (lbq - np.r_[np.full(10, np.nan), lbq[:-10]]) - (laq - np.r_[np.full(10, np.nan), laq[:-10]])
    mb, ms = d["max_buy"].to_numpy(float), d["max_sell"].to_numpy(float)
    f["LARGE"] = np.where(mb >= p99, 1.0, 0.0) - np.where(ms >= p99, 1.0, 0.0)
    f["LARGE_ANY"] = ((mb >= p99) | (ms >= p99)).astype(float)
    f["SPREAD_W"] = ((d["ask"].to_numpy() - d["bid"].to_numpy()) > TICK * 1.5).astype(float)
    basis = lm - np.log(d["spot_last"].to_numpy(float))
    f["BASIS"] = basis
    f["BASIS_DEV"] = basis - pd.Series(basis).rolling(6000, min_periods=600).mean().to_numpy()
    f["FUT_LEAD"] = (lm - np.r_[np.full(10, np.nan), lm[:-10]]) * 1e4
    for k, h in H.items():
        f[f"r_{k}"] = (np.r_[lm[h:], np.full(h, np.nan)] - lm) * 1e4               # bp
    f["dbasis_30s"] = (np.r_[basis[300:], np.full(300, np.nan)] - basis) * 1e4
    ls = np.log(d["spot_last"].to_numpy(float))
    f["spot_r_1s"] = (np.r_[ls[10:], np.full(10, np.nan)] - ls) * 1e4
    f["hour"] = (np.arange(len(d)) // 36000).astype(int)
    f["spread_bp"] = (d["ask"].to_numpy() - d["bid"].to_numpy()) / mid * 1e4
    return f


# ------------------------------------------------------------------------------------------------ training
def train() -> dict:
    s = sample()
    ofi_all, hist = [], np.zeros(len(SIZE_BINS) - 1)
    for day in s["train"]:
        d = load_day(day)
        ofi_all.append(pd.Series(d["ofi"].to_numpy(float)).rolling(10, min_periods=1).sum().to_numpy()[::10])
        hist += np.array(json.loads((ROOT / f"{day}.json").read_text())["size_hist"])
    ofi_sd = float(np.nanstd(np.concatenate(ofi_all)))
    c = np.cumsum(hist) / hist.sum()
    p99 = float(SIZE_BINS[1:][np.searchsorted(c, 0.99)])
    k = len(FEATS) + 1
    XtX, Xty, yy, n = np.zeros((k, k)), np.zeros(k), 0.0, 0
    uni = {f: [np.zeros((2, 2)), np.zeros(2), 0.0] for f in FEATS}
    rv_hours = []
    for day in s["train"]:
        f = features(load_day(day), ofi_sd, p99)
        m = f[FEATS + ["r_5s"]].notna().all(axis=1).to_numpy()
        X = np.column_stack([np.ones(m.sum()), f.loc[m, FEATS].to_numpy(float)])
        y = f.loc[m, "r_5s"].to_numpy(float)
        XtX += X.T @ X
        Xty += X.T @ y
        yy += float(y @ y)
        n += len(y)
        for j, ft in enumerate(FEATS, 1):
            Xs = X[:, [0, j]]
            uni[ft][0] += Xs.T @ Xs
            uni[ft][1] += Xs.T @ y
            uni[ft][2] += float(y @ y)
        r1 = f["FUT_LEAD"].to_numpy()[::10]
        rv_hours += [float(np.nanstd(r1[(h - 1) * 3600:h * 3600])) for h in range(1, 24)]   # previous hour, as for test hours
    beta = np.linalg.solve(XtX, Xty)
    r2_single = {}
    betas_single = {}
    for ft, (A_, b_, y2) in uni.items():
        bs = np.linalg.solve(A_, b_)
        sse = y2 - 2 * bs @ b_ + bs @ A_ @ bs
        r2_single[ft] = float(1 - sse / y2)
        betas_single[ft] = bs.tolist()
    best = max(r2_single, key=r2_single.get)
    # prediction deciles on training days (for E2)
    preds = []
    for day in s["train"]:
        f = features(load_day(day), ofi_sd, p99)
        X = np.column_stack([np.ones(len(f)), f[FEATS].to_numpy(float)])
        preds.append((X @ beta)[::10])
    pr = np.concatenate(preds)
    pr = pr[~np.isnan(pr)]
    out = {"ofi_sd": ofi_sd, "p99_trade_size": p99, "beta": beta.tolist(), "features": FEATS, "n_train_obs": n,
           "train_r2": float(1 - (yy - 2 * beta @ Xty + beta @ XtX @ beta) / yy), "r2_single": r2_single, "best_single": best,
           "beta_best_single": betas_single[best], "pred_deciles": [float(np.quantile(pr, 0.1)), float(np.quantile(pr, 0.9))],
           "rv_terciles": [float(np.nanquantile(rv_hours, 1 / 3)), float(np.nanquantile(rv_hours, 2 / 3))]}
    (ROOT / "train_model.json").write_text(json.dumps(out, indent=1))
    return out


# ------------------------------------------------------------------------------------------------ execution simulator
def _windows(a: np.ndarray, idx: np.ndarray, k: int) -> np.ndarray:
    """a[idx + 1 .. idx + k] as an (n, k) array (NaN past the end)."""
    j = idx[:, None] + np.arange(1, k + 1)[None, :]
    out = np.full(j.shape, np.nan)
    ok = j < len(a)
    out[ok] = a[j[ok]]
    return out


def _greedy(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Indices of non-overlapping trades, taken in time order (one position at a time)."""
    keep, last = [], -1
    for i in np.argsort(start, kind="stable"):
        if start[i] >= last:
            keep.append(i)
            last = end[i]
    return np.array(keep, dtype=int)


def simulate(d: pd.DataFrame, pred: np.ndarray, model: dict, lat_ms: int, fees: tuple[float, float], size: float = 0.01) -> dict:
    maker, taker = fees
    bid, ask = d["bid"].to_numpy(), d["ask"].to_numpy()
    bq, aq = d["bidq"].to_numpy(float), d["askq"].to_numpy(float)
    mid = (bid + ask) / 2
    n = len(d)
    l = -(-lat_ms // 100)                                   # round latency UP to whole buckets (never optimistic)
    hold = H["5s"]
    spread_bp = (ask - bid) / mid * 1e4
    res = {}
    # ---- E1 taker
    theta = 2 * taker + spread_bp
    sig = np.where(pred > theta, 1, np.where(pred < -theta, -1, 0))
    i = np.flatnonzero(sig != 0)
    i = i[i + l + hold < n]
    s_ = sig[i]
    e, x = i + l, i + l + hold
    entry = np.where(s_ > 0, ask[e], bid[e])
    exitp = np.where(s_ > 0, bid[x], ask[x])
    disp = np.where(s_ > 0, aq[e], bq[e])
    filled = np.minimum(1.0, disp / size)                     # capacity: only the displayed size fills at the touch
    gross = s_ * np.log(exitp / entry) * 1e4
    net = gross - 2 * taker
    k = _greedy(i, x)
    res["E1"] = _summ(net[k], filled[k], i[k])
    # ---- E2 passive (joins behind the displayed size; fills on aggressive volume at our price or a trade-through)
    lo, hi = model["pred_deciles"]
    sig2 = np.where(pred >= hi, 1, np.where(pred <= lo, -1, 0))
    j = np.flatnonzero(sig2 != 0)
    j = j[j + l + 2 * hold < n]
    s2 = sig2[j]
    e = j + l
    P = np.where(s2 > 0, bid[e], ask[e])
    Q = np.where(s2 > 0, bq[e], aq[e])
    smin, svat = _windows(d["sell_min_px"].to_numpy(), e, hold), _windows(d["sell_vol_at_min"].to_numpy(float), e, hold)
    bmax, bvat = _windows(d["buy_max_px"].to_numpy(), e, hold), _windows(d["buy_vol_at_max"].to_numpy(float), e, hold)
    Pm = P[:, None]
    through = np.where(s2[:, None] > 0, smin < Pm, bmax > Pm)
    at = np.where(s2[:, None] > 0, np.where(smin == Pm, svat, 0.0), np.where(bmax == Pm, bvat, 0.0))
    cum = np.cumsum(np.nan_to_num(at), axis=1)
    fill_now = through | (cum >= (Q + size)[:, None])
    has = fill_now.any(axis=1)
    kf = np.where(has, fill_now.argmax(axis=1) + 1, hold)       # buckets after placement
    f_idx = e + kf
    x2 = np.minimum(f_idx + hold, n - 1)
    exit2 = np.where(s2 > 0, bid[x2], ask[x2])
    net2 = np.where(has, s2 * np.log(exit2 / P) * 1e4 - maker - taker, np.nan)
    adverse = np.where(has, s2 * (np.log(mid[np.minimum(f_idx + hold, n - 1)] / P)) * 1e4, np.nan)
    end2 = np.where(has, x2, f_idx)
    k2 = _greedy(j, end2)
    filled_k = has[k2]
    r2 = _summ(net2[k2][filled_k], np.ones(filled_k.sum()), j[k2][filled_k])
    r2.update({"orders": int(len(k2)), "fills": int(filled_k.sum()), "fill_rate": float(filled_k.mean()) if len(k2) else None,
               "mid_drift_after_fill_bp": float(np.nanmean(adverse[k2][filled_k])) if filled_k.any() else None})
    res["E2"] = r2
    return res


def _summ(net: np.ndarray, filled: np.ndarray, at: np.ndarray) -> dict:
    if len(net) == 0:
        return {"trades": 0, "mean_net_bp": None, "hour_means": {}}
    w = filled
    hours = at // 36000
    hm = pd.Series(net * w).groupby(hours).sum() / pd.Series(w).groupby(hours).sum()
    return {"trades": int(len(net)), "mean_net_bp": float(np.sum(net * w) / np.sum(w)), "mean_fill_fraction": float(np.mean(w)),
            "hour_means": {int(k): float(v) for k, v in hm.items()}}


# ------------------------------------------------------------------------------------------------ tests
def _ic(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 200 or np.std(x[m]) == 0:
        return np.nan
    return float(sps.spearmanr(x[m], y[m]).correlation)


def _nwt(v: list[float], lags: int = 3) -> tuple[float, float, float]:
    v = np.array([x for x in v if x is not None and not np.isnan(x)])
    fit = nw_ols(v, np.empty((len(v), 0)), lags=lags)
    t = float(fit["t"][0])
    return float(v.mean()), t, float(2 * sps.t.sf(abs(t), len(v) - 1))


def run(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    model = train()
    beta = np.array(model["beta"])
    s = sample()
    blocks = {k: [] for k in ["H1", "H2", "H3", "H5", "H6", "H8", "H9", "H10", "H17", "H20"]}
    desc = {"H8_orders": [], "H20_reverse_spot_leads_futures": []}
    h11 = {"high": [], "low": []}
    sims = {(fee, L, sz): {"E1": [], "E2": []} for fee in FEES for L in LAT_MS for sz in (0.01,)}
    caps = {"E1": [], "E2": []}
    day_net = {}
    t0, t1 = model["rv_terciles"]
    bfeat, bb = model["best_single"], np.array(model["beta_best_single"])
    for day in s["test"]:
        d = load_day(day)
        f = features(d, model["ofi_sd"], model["p99_trade_size"])
        X = np.column_stack([np.ones(len(f)), f[FEATS].to_numpy(float)])
        pred = X @ beta
        pred_single = bb[0] + bb[1] * f[bfeat].to_numpy(float)
        stats_day = json.loads((ROOT / f"{day}.json").read_text())
        r5, ab = f["r_5s"].to_numpy(), np.abs(f["r_5s"].to_numpy())
        rv1 = f["FUT_LEAD"].to_numpy()[::10]
        for h in range(24):
            sl = slice(h * 36000, (h + 1) * 36000)
            g = f.iloc[sl]
            y = r5[sl]
            blocks["H1"].append(_ic(g["OBI"].to_numpy(), y))
            blocks["H2"].append(_ic(g["OFI_1s"].to_numpy(), y))
            lg = g["LARGE"].to_numpy()
            blocks["H3"].append(float(np.nanmean(lg[lg != 0] * y[lg != 0])) if (lg != 0).sum() >= 5 else np.nan)
            blocks["H5"].append(_ic(g["QD"].to_numpy(), y))
            sw = g["SPREAD_W"].to_numpy() > 0
            blocks["H6"].append(float(np.nanmean(ab[sl][sw]) - np.nanmean(ab[sl][~sw])) if sw.sum() >= 20 else np.nan)
            blocks["H8"].append(stats_day["sign_autocorr_by_hour"].get(str(h)))
            la = g["LARGE_ANY"].to_numpy() > 0
            blocks["H9"].append(float(np.nanmean(ab[sl][la]) - np.nanmean(ab[sl][~la])) if la.sum() >= 5 else np.nan)
            m = ~np.isnan(y) & ~np.isnan(pred[sl]) & ~np.isnan(pred_single[sl])
            y2 = float(np.sum(y[m] ** 2))
            r2c = 1 - float(np.sum((y[m] - pred[sl][m]) ** 2)) / y2 if y2 > 0 else np.nan
            r2s = 1 - float(np.sum((y[m] - pred_single[sl][m]) ** 2)) / y2 if y2 > 0 else np.nan
            blocks["H10"].append(r2c - r2s)
            blocks["H17"].append(_ic(g["BASIS_DEV"].to_numpy(), g["dbasis_30s"].to_numpy()))
            blocks["H20"].append(_ic(g["FUT_LEAD"].to_numpy(), g["spot_r_1s"].to_numpy()))
            desc["H8_orders"].append(stats_day.get("sign_autocorr_orders_by_hour", {}).get(str(h)))
            ls_ = np.log(d["spot_last"].to_numpy(float)[sl])
            spot_past = (ls_ - np.r_[np.full(10, np.nan), ls_[:-10]]) * 1e4
            desc["H20_reverse_spot_leads_futures"].append(_ic(spot_past, g["r_1s"].to_numpy()))
            prev = rv1[max(0, (h - 1) * 3600):h * 3600] if h > 0 else rv1[:0]
            rv = float(np.nanstd(prev)) if len(prev) > 100 else np.nan
            ic2 = blocks["H2"][-1]
            if not np.isnan(rv):
                if rv >= t1:
                    h11["high"].append(ic2)
                elif rv <= t0:
                    h11["low"].append(ic2)
        for fee, fv in FEES.items():
            for L in LAT_MS:
                r = simulate(d, pred, model, L, fv)
                for e in ("E1", "E2"):
                    sims[(fee, L, 0.01)][e].append({"day": day, **r[e]})
        rc = simulate(d, pred, model, 250, FEES["retail"], size=0.1)
        caps["E1"].append(rc["E1"])
        caps["E2"].append(rc["E2"])
        print(day, "done", flush=True)
    # ---- hypothesis results + Holm
    signs = {"H1": 1, "H2": 1, "H3": 1, "H5": 1, "H6": 1, "H8": 1, "H9": 1, "H10": 1, "H17": -1, "H20": 1}
    hyp = {}
    for k, v in blocks.items():
        m_, t, p = _nwt(v)
        hyp[k] = {"mean": m_, "t": t, "p": p, "blocks": int(sum(1 for x in v if x is not None and not np.isnan(x))), "sign": signs[k]}
    a, b = np.array([x for x in h11["high"] if not np.isnan(x)]), np.array([x for x in h11["low"] if not np.isnan(x)])
    w = sps.ttest_ind(a, b, equal_var=False) if len(a) > 2 and len(b) > 2 else None
    hyp["H11"] = {"mean": float(a.mean() - b.mean()) if w else None, "t": float(w.statistic) if w else None, "p": float(w.pvalue) if w else 1.0,
                  "blocks": int(len(a) + len(b)), "sign": 0, "high_vol_ic": float(a.mean()) if len(a) else None, "low_vol_ic": float(b.mean()) if len(b) else None}
    keys = list(hyp)
    ps = np.array([hyp[k]["p"] for k in keys])
    order = np.argsort(ps)
    run_max = 0.0
    for r_, i in enumerate(order):
        run_max = max(run_max, min(1.0, ps[i] * (len(ps) - r_)))
        hyp[keys[i]]["holm_p"] = run_max
    for k in keys:
        sg = hyp[k]["sign"]
        hyp[k]["pass"] = bool(hyp[k]["holm_p"] < 0.05 and (sg == 0 or np.sign(hyp[k]["mean"]) == sg))
    # ---- economics
    def agg(fee, L, e):
        days = sims[(fee, L, 0.01)][e]
        hm = [v for dd in days for v in dd["hour_means"].values()]
        tr = sum(dd["trades"] for dd in days)
        tot = sum((dd["mean_net_bp"] or 0) * dd["trades"] for dd in days)
        mean = tot / tr if tr else None
        t = _nwt(hm)[1] if len(hm) > 3 else None
        pos_days = sum(1 for dd in days if dd["mean_net_bp"] is not None and dd["mean_net_bp"] > 0)
        o = {"trades": tr, "mean_net_bp": mean, "t_hours": t, "positive_days": pos_days}
        if e == "E2":
            orders = sum(dd.get("orders", 0) for dd in days)
            o["orders"], o["fill_rate"] = orders, (tr / orders if orders else None)       # fills / orders, pooled
            dr = [(dd["mid_drift_after_fill_bp"], dd["trades"]) for dd in days if dd.get("mid_drift_after_fill_bp") is not None]
            o["mid_drift_after_fill_bp"] = (sum(a * b for a, b in dr) / sum(b for _, b in dr)) if dr else None
        return o
    econ = {fee: {e: {L: agg(fee, L, e) for L in LAT_MS} for e in ("E1", "E2")} for fee in FEES}
    # 2x fees at 250 ms (retail): reuse the simulator with doubled fees
    two = {"E1": [], "E2": []}
    for day in s["test"]:
        d = load_day(day)
        f = features(d, model["ofi_sd"], model["p99_trade_size"])
        pred = np.column_stack([np.ones(len(f)), f[FEATS].to_numpy(float)]) @ beta
        r = simulate(d, pred, model, 250, (2 * FEES["retail"][0], 2 * FEES["retail"][1]))
        two["E1"].append(r["E1"])
        two["E2"].append(r["E2"])

    def mean_of(lst):
        tr = sum(x["trades"] for x in lst)
        return (sum((x["mean_net_bp"] or 0) * x["trades"] for x in lst) / tr) if tr else None
    gate_steps = []
    stat_ok = hyp["H2"]["pass"] or hyp["H10"]["pass"]
    gate_steps.append(("statistical", stat_ok, f"H2 Holm p {hyp['H2']['holm_p']:.3g}, H10 Holm p {hyp['H10']['holm_p']:.3g}"))
    def ok_e(e):
        v = econ["retail"][e][250]
        return v["mean_net_bp"] is not None and v["mean_net_bp"] > 0 and (v["t_hours"] or 0) > 2
    live = [e for e in ("E1", "E2") if ok_e(e)]
    gate_steps.append(("economic_retail_250ms", bool(live), "; ".join(f"{e}: {econ['retail'][e][250]['mean_net_bp']} bp/trade over "
                                                                       f"{econ['retail'][e][250]['trades']} trades, t {econ['retail'][e][250]['t_hours']}" for e in ("E1", "E2"))))
    live = [e for e in live if (mean_of(two[e]) or -1) > 0]
    gate_steps.append(("fees_2x", bool(live), "; ".join(f"{e}: {mean_of(two[e])}" for e in ("E1", "E2"))))
    live = [e for e in live if (econ["retail"][e][500]["mean_net_bp"] or -1) > 0]
    gate_steps.append(("latency_500ms", bool(live), "; ".join(f"{e}: {econ['retail'][e][500]['mean_net_bp']}" for e in ("E1", "E2"))))
    cap_ok = all((mean_of(caps[e]) or -1) > 0 for e in ("E1", "E2"))
    gate_steps.append(("capacity_0.1BTC", cap_ok, "; ".join(f"{e}: {mean_of(caps[e])}" for e in ("E1", "E2"))))
    stab = [e for e in live if econ["retail"][e][250]["positive_days"] >= 5]
    gate_steps.append(("stability", bool(stab), "; ".join(f"{e}: {econ['retail'][e][250]['positive_days']} of 7" for e in ("E1", "E2"))))
    gate_steps.append(("forward", False, "frozen paper test on the live feed: not started"))
    verdict, steps = None, []
    for name, ok, why in gate_steps:
        if verdict is None:
            res = "PASS" if ok else ("PENDING" if name == "forward" else "FAIL")
            steps.append({"step": name, "result": res, "detail": why})
            if not ok:
                verdict = f"REJECTED at {name}" if name != "forward" else "ELIGIBLE FOR A FROZEN FORWARD TEST"
        else:
            steps.append({"step": name, "result": "NOT REACHED", "detail": why})
    result = {"model": {k: v for k, v in model.items() if k != "beta"} | {"beta": dict(zip(["const"] + FEATS, model["beta"]))},
              "hypotheses": hyp, "descriptive": {k: dict(zip(("mean", "t", "p"), _nwt(v))) for k, v in desc.items()}, "economics": econ, "two_x_fees": {e: mean_of(two[e]) for e in two},
              "capacity_0.1BTC": {e: mean_of(caps[e]) for e in caps}, "gate": {"steps": steps, "verdict": verdict}}
    (out / "m0.json").write_text(json.dumps(result, indent=1, default=str))
    return result


def audit() -> dict:
    s = sample()
    rows = []
    for day in s["train"] + s["test"]:
        d = load_day(day)
        j = json.loads((ROOT / f"{day}.json").read_text())
        sp = (d["ask"] - d["bid"]) / TICK
        rows.append({"day": day, "set": "train" if day in s["train"] else "test", "buckets": len(d), "book_updates": j["book_updates"],
                     "trades": j["n_trades"], "buckets_without_update": int((d["nupd"] == 0).sum()),
                     "max_gap_s": float(_max_gap(d["nupd"].to_numpy()) / 10), "crossed_or_locked": int((d["ask"] <= d["bid"]).sum()),
                     "spread_1tick_share": float((sp.round() == 1).mean()), "nan_book": int(d["bid"].isna().sum()),
                     "spot_nan": int(d["spot_last"].isna().sum()), "raw_backward_steps": j.get("raw_backward_steps"),
                     "sorted_pipeline": "raw_backward_steps" in j})
    df = pd.DataFrame(rows)
    res = {"days": df.to_dict("records"), "accepted": bool((df["crossed_or_locked"] == 0).all() and (df["nan_book"] < 100).all()
                                                           and df["sorted_pipeline"].all() and (df["max_gap_s"] < 300).all()),
           "note": "raw files with backward time steps (interleaved streams) are sorted per hour before OFI; counts per day above",
           "made_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (A / f"micro_m0_data_{time.strftime('%Y-%m-%d')}.json").write_text(json.dumps(res, indent=1, default=str))
    print(df.to_string())
    print("accepted", res["accepted"])
    return res


def _max_gap(nupd: np.ndarray) -> int:
    z = (nupd == 0).astype(int)
    best = cur = 0
    for v in z:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


# ------------------------------------------------------------------------------------------------ descriptive diagnostics
def diagnostics(out: Path) -> dict:
    """POST-HOC and DESCRIPTIVE (not gate inputs): how large are the predictable moves relative to costs?
    D1 realised forward move by OBI decile (training-day cut-offs); D2 how often a move exceeds a round-trip cost;
    D3 the taker rule driven by the best single feature (OBI) instead of the full model, at 0 and 250 ms."""
    model = json.loads((out / "train_model.json").read_text()) if (out / "train_model.json").exists() else json.loads((ROOT / "train_model.json").read_text())
    s = sample()
    obi_tr = np.concatenate([features(load_day(day), model["ofi_sd"], model["p99_trade_size"])["OBI"].to_numpy()[::50] for day in s["train"][:5]])
    cuts = np.nanquantile(obi_tr, np.linspace(0.1, 0.9, 9))
    from marketdata.nse_depth_recorder import nse_index_futures_round_trip_bp
    costs = {"retail_taker_round_trip": 10.0, "best_case_taker_round_trip": 3.4,
             "nse_index_futures_round_trip_approx": round(nse_index_futures_round_trip_bp(), 2)}
    dec = {h: [[] for _ in range(10)] for h in H}
    absr = {h: [] for h in H}
    d3 = {}
    bb = np.array(model["beta_best_single"])
    for day in s["test"]:
        d = load_day(day)
        f = features(d, model["ofi_sd"], model["p99_trade_size"])
        q = np.searchsorted(cuts, f["OBI"].to_numpy())
        for h in H:
            r = f[f"r_{h}"].to_numpy()
            for k in range(10):
                v = r[(q == k) & ~np.isnan(r)]
                dec[h][k].append(float(v.mean()) if len(v) else np.nan)
            absr[h].append(np.abs(r[~np.isnan(r)][::10]))
        pred = bb[0] + bb[1] * f[model["best_single"]].to_numpy(float)
        for L in (0, 250):
            for fee, fv in FEES.items():
                r_ = simulate(d, pred, model, L, fv)["E1"]
                d3.setdefault(f"{fee}_{L}ms", []).append(r_)
    res = {"note": "post-hoc, descriptive; not part of the pre-registered gate",
           "D1_mean_forward_move_bp_by_OBI_decile": {h: [float(np.nanmean(x)) for x in dec[h]] for h in H},
           "D2_share_of_moves_exceeding_cost": {h: {c: float((np.concatenate(absr[h]) > v).mean()) for c, v in costs.items()} for h in H},
           "D2_mean_abs_move_bp": {h: float(np.concatenate(absr[h]).mean()) for h in H},
           "D3_taker_rule_OBI_only": {k: {"trades": sum(x["trades"] for x in v),
                                          "mean_net_bp": (sum((x["mean_net_bp"] or 0) * x["trades"] for x in v) / max(1, sum(x["trades"] for x in v)))}
                                      for k, v in d3.items()},
           "costs_bp": costs}
    (out / "m0_diagnostics.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return res
