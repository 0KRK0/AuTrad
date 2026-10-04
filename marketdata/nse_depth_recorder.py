"""Forward recorder for NSE market depth (v0.9): the free way to build the NSE order-book history that M0 needs.
(NSE's historical order data is licensed at about ₹12.5 lakh a year.)

It records full-mode ticks from a broker websocket: last trade, cumulative volume and 5 depth levels, with the
exchange timestamp (whole seconds on Kite) and the local receive time in milliseconds. Ticks are written per instrument and day to data/micro/nse/<symbol>/<date>/part-*.parquet.
`to_grid()` converts a day of ticks to the same 100 ms grid as the Binance M0 data, so the identical analysis can
run on NSE once enough days exist.

Sources
-------
* `KiteSource`: Zerodha Kite Connect (`pip install kiteconnect`). Needs YOUR API key and a daily access token. The
  recorder never logs in and never places orders; it only subscribes to market data.
* Any other broker: implement `.run(callback)` that delivers dicts in the TICK schema.

Limits (honest): a broker websocket gives snapshots at the broker's tick rate, not every exchange event. Trade
direction is inferred (Lee-Ready tick rule), not reported. So OFI and queue estimates are coarser than true TBT data,
and the M0 queue model is used conservatively. Kite's exchange timestamp has 1-second resolution and is a naive IST
time, so the grid buckets on the local receive time (ms) and defaults to a 1-second grid; sub-second latency results
are not meaningful on this data.
"""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("data/micro/nse")
LEVELS = 5
TICK_COLS = (["ts_exchange", "ts_local", "symbol", "last_price", "last_qty", "volume_cum"]
             + [f"{s}{k}_{f}" for s in ("bid", "ask") for k in range(1, LEVELS + 1) for f in ("px", "qty", "orders")])


def flatten(t: dict, symbol: str) -> dict:
    """Kite full-mode tick -> TICK row."""
    ts = pd.Timestamp(t.get("exchange_timestamp") or t.get("last_trade_time") or pd.Timestamp.now(tz="Asia/Kolkata"))
    ts = ts.tz_localize("Asia/Kolkata") if ts.tzinfo is None else ts      # kiteconnect gives naive IST datetimes
    row = {"ts_exchange": int(ts.value // 10**6),
           "ts_local": int(time.time() * 1000), "symbol": symbol, "last_price": float(t.get("last_price") or np.nan),
           "last_qty": float(t.get("last_traded_quantity") or t.get("last_quantity") or 0), "volume_cum": float(t.get("volume_traded") or t.get("volume") or 0)}
    depth = t.get("depth") or {}
    for side, key in (("bid", "buy"), ("ask", "sell")):
        lv = depth.get(key) or []
        for k in range(1, LEVELS + 1):
            e = lv[k - 1] if len(lv) >= k else {}
            row[f"{side}{k}_px"] = float(e.get("price") or np.nan)
            row[f"{side}{k}_qty"] = float(e.get("quantity") or 0)
            row[f"{side}{k}_orders"] = float(e.get("orders") or 0)
    return row


class DepthRecorder:
    def __init__(self, root: Path = ROOT, flush_every_s: float = 30.0) -> None:
        self.root, self.flush_every_s = Path(root), flush_every_s
        self.q: queue.Queue = queue.Queue()
        self.buf: list[dict] = []
        self.last_flush = time.time()
        self.parts = 0

    def on_rows(self, rows: list[dict]) -> None:
        for r in rows:
            self.q.put(r)

    def drain(self, force: bool = False) -> int:
        while not self.q.empty():
            self.buf.append(self.q.get_nowait())
        if self.buf and (force or time.time() - self.last_flush >= self.flush_every_s):
            df = pd.DataFrame(self.buf, columns=TICK_COLS)
            for (sym, day), g in df.groupby([df["symbol"], pd.to_datetime(df["ts_exchange"], unit="ms").dt.tz_localize("UTC")
                                             .dt.tz_convert("Asia/Kolkata").dt.strftime("%Y-%m-%d")]):
                d = self.root / sym / day
                d.mkdir(parents=True, exist_ok=True)
                g.to_parquet(d / f"part-{int(time.time() * 1000)}-{self.parts}.parquet", compression="zstd")
                self.parts += 1
            n = len(self.buf)
            self.buf, self.last_flush = [], time.time()
            return n
        return 0


class KiteSource:  # pragma: no cover - needs credentials and a live market
    def __init__(self, api_key: str, access_token: str, tokens: dict[int, str]) -> None:
        self.api_key, self.access_token, self.tokens = api_key, access_token, tokens

    def run(self, recorder: DepthRecorder) -> None:
        from kiteconnect import KiteTicker

        kws = KiteTicker(self.api_key, self.access_token)

        def on_ticks(ws, ticks):
            recorder.on_rows([flatten(t, self.tokens.get(t["instrument_token"], str(t["instrument_token"]))) for t in ticks])

        def on_connect(ws, response):
            ws.subscribe(list(self.tokens))
            ws.set_mode(ws.MODE_FULL, list(self.tokens))

        kws.on_ticks, kws.on_connect = on_ticks, on_connect
        kws.connect(threaded=True)
        try:
            while True:
                time.sleep(1)
                recorder.drain()
        finally:
            recorder.drain(force=True)
            kws.close()


def load_day(symbol: str, day: str, root: Path = ROOT) -> pd.DataFrame:
    parts = sorted((Path(root) / symbol / day).glob("part-*.parquet"))
    if not parts:
        return pd.DataFrame(columns=TICK_COLS)
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True).sort_values(["ts_exchange", "ts_local"]).reset_index(drop=True)


def to_grid(ticks: pd.DataFrame, day: str, start: str = "09:15", end: str = "15:30", bucket_ms: int = 1000,
            spot: pd.DataFrame | None = None, time_col: str = "ts_local") -> pd.DataFrame:
    """A day of TICK rows -> the M0 grid (columns as marketdata.micro_binance; 1 s buckets by default, on the local
    receive time). Trades come from increases of cumulative volume; their side is inferred with the quote rule
    against the previous mid (at or above the mid = buy)."""
    t0 = int(pd.Timestamp(f"{day} {start}", tz="Asia/Kolkata").value // 10**6)
    t1 = int(pd.Timestamp(f"{day} {end}", tz="Asia/Kolkata").value // 10**6)
    nb = (t1 - t0) // bucket_ms
    x = ticks[(ticks[time_col] >= t0) & (ticks[time_col] < t1)].sort_values([time_col]).reset_index(drop=True)
    b = ((x[time_col].to_numpy() - t0) // bucket_ms).astype(int)
    bp, bq, ap, aq = (x[c].to_numpy(float) for c in ("bid1_px", "bid1_qty", "ask1_px", "ask1_qty"))
    from marketdata.micro_binance import ofi_events
    e = ofi_events(bp, bq, ap, aq, (bp[0], bq[0], ap[0], aq[0])) if len(x) else np.zeros(0)
    ofi, nupd = np.zeros(nb), np.zeros(nb, dtype=np.int64)
    np.add.at(ofi, b, e)
    np.add.at(nupd, b, 1)
    last = np.full((nb, 4), np.nan)
    if len(x):
        rev = len(b) - 1 - np.unique(b[::-1], return_index=True)[1]
        last[b[rev]] = np.column_stack([bp[rev], bq[rev], ap[rev], aq[rev]])
    st = pd.DataFrame(last).ffill().to_numpy()
    dv = np.diff(np.r_[x["volume_cum"].to_numpy(float)[:1], x["volume_cum"].to_numpy(float)]) if len(x) else np.zeros(0)
    dv = np.where(dv > 0, dv, 0.0)
    pmid = np.r_[np.nan, ((bp + ap) / 2)[:-1]] if len(x) else np.zeros(0)
    lp = x["last_price"].to_numpy(float) if len(x) else np.zeros(0)
    buy = (dv > 0) & (lp >= pmid)
    sell = (dv > 0) & (lp < pmid)
    bv, sv = np.zeros(nb), np.zeros(nb)
    np.add.at(bv, b[buy], dv[buy])
    np.add.at(sv, b[sell], dv[sell])
    mxb, mxs = np.zeros(nb), np.zeros(nb)
    np.maximum.at(mxb, b[buy], dv[buy])
    np.maximum.at(mxs, b[sell], dv[sell])
    smin, svat, bmax, bvat = np.full(nb, np.nan), np.zeros(nb), np.full(nb, np.nan), np.zeros(nb)
    np.fmin.at(smin, b[sell], lp[sell])
    np.fmax.at(bmax, b[buy], lp[buy])
    at_s = sell & (lp == smin[b])
    at_b = buy & (lp == bmax[b])
    np.add.at(svat, b[at_s], dv[at_s])
    np.add.at(bvat, b[at_b], dv[at_b])
    ntr = np.zeros(nb, dtype=np.int64)
    np.add.at(ntr, b[dv > 0], 1)
    sp = np.full(nb, np.nan)
    if spot is not None and len(spot):
        sb = ((spot[time_col].to_numpy() - t0) // bucket_ms).astype(int)
        ok = (sb >= 0) & (sb < nb)
        sp[sb[ok]] = spot["last_price"].to_numpy(float)[ok]
        sp = pd.Series(sp).ffill().to_numpy()
    return pd.DataFrame({"t": t0 + np.arange(nb, dtype=np.int64) * bucket_ms, "bid": st[:, 0], "bidq": st[:, 1], "ask": st[:, 2], "askq": st[:, 3],
                         "ofi": ofi, "nupd": nupd, "buy_vol": bv, "sell_vol": sv, "ntr": ntr, "max_buy": mxb, "max_sell": mxs,
                         "sell_min_px": smin, "sell_vol_at_min": svat, "buy_max_px": bmax, "buy_vol_at_max": bvat, "spot_last": sp})


def nse_index_futures_round_trip_bp(price: float = 25_000.0, lot: int = 65, brokerage_per_order: float = 20.0,
                                    stt_rate: float = 0.0005) -> float:
    """Approximate statutory + brokerage cost of one round trip in NSE index futures, in bp of notional:
    STT 0.05 % on sells (from 1 Apr 2026; 0.02 % before), exchange 0.00173 %, SEBI ₹10/crore, stamp 0.002 % on buys,
    GST 18 % on brokerage+exchange+SEBI; NIFTY lot 65. Spread and slippage are extra."""
    notional = price * lot
    stt, txn, sebi, stamp = stt_rate * notional, 2 * 0.0000173 * notional, 2 * 1e-6 * notional, 0.00002 * notional
    brok = 2 * brokerage_per_order
    gst = 0.18 * (brok + txn + sebi)
    return (stt + txn + sebi + stamp + brok + gst) / notional * 1e4


# ------------------------------------------------------------------------------------------------ v1.1: daily archive + QC
def qc_day(ticks: pd.DataFrame, day: str, start: str = "09:15", end: str = "15:30") -> dict:
    """Quality checks for one recorded day of one instrument (no thresholds are applied here; the report states them)."""
    if not len(ticks):
        return {"rows": 0, "ok": False, "reason": "no ticks"}
    t0 = int(pd.Timestamp(f"{day} {start}", tz="Asia/Kolkata").value // 10**6)
    t1 = int(pd.Timestamp(f"{day} {end}", tz="Asia/Kolkata").value // 10**6)
    x = ticks.sort_values("ts_local")
    inside = x[(x["ts_local"] >= t0) & (x["ts_local"] < t1)]
    gaps = np.diff(np.r_[t0, inside["ts_local"].to_numpy(), t1]) / 1000 if len(inside) else np.array([(t1 - t0) / 1000])
    s_ = inside if len(inside) else x                                        # shares are measured on session rows
    bp, ap = s_["bid1_px"].to_numpy(float), s_["ask1_px"].to_numpy(float)
    lat = (s_["ts_local"] - s_["ts_exchange"]).to_numpy(float)
    ex = s_["ts_exchange"].to_numpy()
    q = {"rows": int(len(x)), "rows_in_session": int(len(inside)),
         "first_local": str(pd.Timestamp(int(x["ts_local"].min()), unit="ms", tz="Asia/Kolkata")),
         "last_local": str(pd.Timestamp(int(x["ts_local"].max()), unit="ms", tz="Asia/Kolkata")),
         "max_gap_s": float(gaps.max()), "gaps_over_60s": int((gaps > 60).sum()),
         "crossed_or_locked_share": float(np.mean((bp >= ap) & ~np.isnan(bp) & ~np.isnan(ap))),
         "missing_quote_share": float(np.mean(np.isnan(bp) | np.isnan(ap))),
         "exchange_ts_backwards_share": float(np.mean(np.diff(ex) < 0)) if len(ex) > 1 else 0.0,
         "duplicate_share": float(s_.duplicated(subset=[c for c in TICK_COLS if c != "ts_local"]).mean()),
         "volume_cum_decreases": int((np.diff(s_["volume_cum"].to_numpy(float)) < 0).sum()),
         "latency_ms_median": float(np.nanmedian(lat)), "latency_ms_p99": float(np.nanpercentile(lat, 99))}
    q["ok"] = bool(q["rows_in_session"] > 1000 and q["max_gap_s"] <= 300 and q["crossed_or_locked_share"] < 0.01)
    return q


def seal_day(symbol: str, day: str, root: Path = ROOT) -> dict:
    """Raw parts stay as written (made read-only: the immutable archive); a compressed, sorted research file
    <day>/day.parquet and a manifest (sha256 of every part and of the day file, row counts, QC) are written beside them.
    Re-sealing an already sealed day is refused, so the archive cannot be silently rewritten."""
    import hashlib
    import json
    import os
    import stat
    d = Path(root) / symbol / day
    man_p = d / "manifest.json"
    if man_p.exists():
        return {"symbol": symbol, "day": day, "status": "already sealed"}
    parts = sorted(d.glob("part-*.parquet"))
    ticks = load_day(symbol, day, root)
    q = qc_day(ticks, day)
    out = d / "day.parquet"
    ticks.to_parquet(out, compression="zstd")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    man = {"symbol": symbol, "day": day, "parts": {p.name: sha(p) for p in parts}, "day_file": sha(out), "qc": q}
    for p in parts + [out]:
        try:
            os.chmod(p, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
        except OSError:
            pass
    man_p.write_text(json.dumps(man, indent=1))
    return {"symbol": symbol, "day": day, "status": "sealed", "qc": q}


def recorded_days(root: Path = ROOT) -> list[tuple[str, str]]:
    return sorted((p.parent.name, p.name) for p in Path(root).glob("*/*") if p.is_dir() and any(p.glob("part-*.parquet")))
