"""Binance BTCUSDT microstructure data (v0.9, study M0): best bid/ask updates and trades of the USDⓈ-M perpetual,
plus spot trades, from the public archive (data.binance.vision), aggregated to a 100 ms grid.

Per day: raw zips are streamed from a scratch directory (never kept in the repository). The grid is written to
data/micro/btcusdt/<day>.parquet, with trade-level statistics to <day>.json. The best-level order-flow imbalance (OFI,
Cont-Kukanov-Stoikov 2014) is computed event by event before aggregation.

Grid columns (bucket = 100 ms of exchange transaction time; book state = last update in or before the bucket):
  t, bid, bidq, ask, askq, ofi, nupd, buy_vol, sell_vol, ntr, max_buy, max_sell,
  sell_min_px, sell_vol_at_min, buy_max_px, buy_vol_at_max, spot_last
"""
from __future__ import annotations

import io
import json
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
FILES = "https://data.binance.vision/"
ROOT = Path("data/micro/btcusdt")
SCRATCH = Path("/tmp/autrad_micro")
BUCKET_MS = 100
SIZE_BINS = np.logspace(-4, 3, 141)          # trade-size histogram (BTC), for training-day percentiles


def _keys(session, prefix: str) -> list[str]:  # pragma: no cover - network
    keys, marker = [], ""
    while True:
        t = session.get(f"{S3}?prefix={prefix}" + (f"&marker={marker}" if marker else ""), timeout=60).text
        k = re.findall(r"<Key>([^<]+)</Key>", t)
        keys += k
        if "<IsTruncated>true</IsTruncated>" not in t or not k:
            break
        marker = k[-1]
    return keys


def available_days(session) -> list[str]:  # pragma: no cover - network
    def days(prefix):
        return {m.group(1) for k in _keys(session, prefix) if (m := re.search(r"(\d{4}-\d{2}-\d{2})\.zip$", k))}
    a = days("data/futures/um/daily/bookTicker/BTCUSDT/")
    b = days("data/futures/um/daily/trades/BTCUSDT/")
    c = days("data/spot/daily/trades/BTCUSDT/")
    return sorted(a & b & c)


def sample(days: list[str], n: int = 20, seed: int = 2026, n_train: int = 13) -> dict:
    pick = sorted(np.random.default_rng(seed).choice(np.array(days), n, replace=False).tolist())
    return {"train": pick[:n_train], "test": pick[n_train:], "seed": seed, "universe_days": len(days)}


def url(kind: str, day: str) -> str:
    return {"book": f"data/futures/um/daily/bookTicker/BTCUSDT/BTCUSDT-bookTicker-{day}.zip",
            "trades": f"data/futures/um/daily/trades/BTCUSDT/BTCUSDT-trades-{day}.zip",
            "spot": f"data/spot/daily/trades/BTCUSDT/BTCUSDT-trades-{day}.zip"}[kind]


def download(session, day: str) -> dict[str, Path]:  # pragma: no cover - network
    SCRATCH.mkdir(parents=True, exist_ok=True)
    out = {}
    for kind in ("book", "trades", "spot"):
        p = SCRATCH / f"{kind}-{day}.zip"
        if not p.exists():
            part = p.with_suffix(".part")
            with session.get(FILES + url(kind, day), stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(part, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            part.replace(p)                                 # only complete files get the final name
        out[kind] = p
    return out


def ofi_events(bp, bq, ap, aq, prev) -> np.ndarray:
    """Event-level best-level OFI e_n (Cont, Kukanov & Stoikov 2014); `prev` = (bp, bq, ap, aq) before the first row."""
    pbp, pbq, pap, paq = (np.r_[prev[0], bp[:-1]], np.r_[prev[1], bq[:-1]], np.r_[prev[2], ap[:-1]], np.r_[prev[3], aq[:-1]])
    e = (bp >= pbp) * bq - (bp <= pbp) * pbq - (ap <= pap) * aq + (ap >= pap) * paq
    return e


def _book_grid(path: Path, day0_ms: int, nb: int):
    """Some archive files interleave two time streams row by row (found in review: 2023-12-29, 2024-02-12), so rows
    are first partitioned by hour on disk, then each hour is sorted by (transaction_time, update_id) before the
    event-level OFI is computed. Returns the grid arrays and raw-order statistics for the audit."""
    import shutil

    import pyarrow.csv as pacsv

    tmp = SCRATCH / f"parts-{day0_ms}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    backwards, rows, k = 0, 0, 0
    last_t = None
    with zipfile.ZipFile(path) as z, z.open(z.namelist()[0]) as fh:
        reader = pacsv.open_csv(fh, read_options=pacsv.ReadOptions(block_size=64 << 20),
                                convert_options=pacsv.ConvertOptions(include_columns=["update_id", "best_bid_price", "best_bid_qty",
                                                                                        "best_ask_price", "best_ask_qty", "transaction_time"]))
        for batch in reader:
            tt = batch.column("transaction_time").to_numpy().astype(np.int64)
            arr = np.column_stack([tt.astype(float), batch.column("update_id").to_numpy().astype(float),
                                   *(batch.column(c).to_numpy().astype(float) for c in ("best_bid_price", "best_bid_qty", "best_ask_price", "best_ask_qty"))])
            backwards += int((np.diff(np.r_[tt[:1] if last_t is None else [last_t], tt]) < 0).sum())
            last_t, rows = int(tt[-1]), rows + len(tt)
            h = np.clip((tt - day0_ms) // 3_600_000, 0, 23)
            for hh in np.unique(h):
                np.save(tmp / f"h{hh:02d}-{k:05d}.npy", arr[h == hh])
            k += 1
    last = np.full((nb, 4), np.nan)
    ofi = np.zeros(nb)
    nupd = np.zeros(nb, dtype=np.int64)
    prev = None
    for hh in range(24):
        files = sorted(tmp.glob(f"h{hh:02d}-*.npy"))
        if not files:
            continue
        a = np.concatenate([np.load(f) for f in files])
        a = a[np.lexsort((a[:, 1], a[:, 0]))]                  # by time, then update id
        tt, bp, bq, ap, aq = a[:, 0].astype(np.int64), a[:, 2], a[:, 3], a[:, 4], a[:, 5]
        if prev is None:
            prev = (bp[0], bq[0], ap[0], aq[0])
        e = ofi_events(bp, bq, ap, aq, prev)
        prev = (bp[-1], bq[-1], ap[-1], aq[-1])
        b = np.clip((tt - day0_ms) // BUCKET_MS, 0, nb - 1)
        np.add.at(ofi, b, e)
        np.add.at(nupd, b, 1)
        rev = len(b) - 1 - np.unique(b[::-1], return_index=True)[1]
        last[b[rev]] = np.column_stack([bp[rev], bq[rev], ap[rev], aq[rev]])
    shutil.rmtree(tmp, ignore_errors=True)
    st = pd.DataFrame(last, columns=["bid", "bidq", "ask", "askq"]).ffill().to_numpy()
    return st, ofi, nupd, {"raw_rows": rows, "raw_backward_steps": backwards}


def _trades_grid(path: Path, day0_ms: int, nb: int, spot: bool = False):
    import pyarrow.csv as pacsv

    with zipfile.ZipFile(path) as z, z.open(z.namelist()[0]) as fh:
        head = fh.peek(64)[:20] if hasattr(fh, "peek") else b""
    has_header = head[:2] == b"id"
    names = None if has_header else ["id", "price", "qty", "quote_qty", "time", "is_buyer_maker", "is_best_match"]
    with zipfile.ZipFile(path) as z, z.open(z.namelist()[0]) as fh:
        ro = pacsv.ReadOptions(block_size=64 << 20, column_names=names, autogenerate_column_names=False)
        tab = pacsv.read_csv(fh, read_options=ro, convert_options=pacsv.ConvertOptions(include_columns=["price", "qty", "time", "is_buyer_maker"]))
    px = tab.column("price").to_numpy().astype(float)
    q = tab.column("qty").to_numpy().astype(float)
    tt = tab.column("time").to_numpy().astype(np.int64)
    if tt.max() > 1e14:
        tt = tt // 1000                                     # microseconds
    sell = np.array([str(x).lower() == "true" for x in tab.column("is_buyer_maker").to_pylist()]) if tab.column("is_buyer_maker").type != "bool" \
        else tab.column("is_buyer_maker").to_numpy(zero_copy_only=False)
    b = np.clip((tt - day0_ms) // BUCKET_MS, 0, nb - 1)
    if spot:
        last = np.full(nb, np.nan)
        rev = len(b) - 1 - np.unique(b[::-1], return_index=True)[1]
        last[b[rev]] = px[rev]
        return pd.Series(last).ffill().to_numpy(), None
    g = {}
    for name, mask in (("buy", ~sell), ("sell", sell)):
        vb, vq, vp = b[mask], q[mask], px[mask]
        vol = np.zeros(nb)
        np.add.at(vol, vb, vq)
        mx = np.zeros(nb)
        np.maximum.at(mx, vb, vq)
        # extreme price per bucket (min for sells, max for buys) and volume at that price
        ext = np.full(nb, np.nan)
        df = pd.DataFrame({"b": vb, "p": vp, "q": vq})
        e = df.groupby("b")["p"].min() if name == "sell" else df.groupby("b")["p"].max()
        ext[e.index.to_numpy()] = e.to_numpy()
        at = df[df["p"].to_numpy() == ext[df["b"].to_numpy()]].groupby("b")["q"].sum()
        vat = np.zeros(nb)
        vat[at.index.to_numpy()] = at.to_numpy()
        g[name] = (vol, mx, ext, vat)
    ntr = np.zeros(nb, dtype=np.int64)
    np.add.at(ntr, b, 1)
    order_ix = np.argsort(tt, kind="stable")
    tt, sell, q, px, b = tt[order_ix], sell[order_ix], q[order_ix], px[order_ix], b[order_ix]
    signs = np.where(sell, -1.0, 1.0)
    hours = (tt - day0_ms) // 3_600_000
    # one taker order filling several makers appears as several same-millisecond, same-side rows: collapse them
    new_order = np.r_[True, (np.diff(tt) != 0) | (np.diff(signs) != 0)]
    o_sign, o_hour = signs[new_order], hours[new_order]
    ac, aco = {}, {}
    for h in range(24):
        s = signs[hours == h]
        ac[h] = float(np.corrcoef(s[:-1], s[1:])[0, 1]) if len(s) > 100 and s.std() > 0 else None
        so = o_sign[o_hour == h]
        aco[h] = float(np.corrcoef(so[:-1], so[1:])[0, 1]) if len(so) > 100 and so.std() > 0 else None
    hist = np.histogram(q, bins=SIZE_BINS)[0].tolist()
    stats = {"n_trades": int(len(q)), "n_orders_collapsed": int(new_order.sum()), "sign_autocorr_by_hour": ac,
             "sign_autocorr_orders_by_hour": aco, "size_hist": hist}
    return g, ntr, stats


def process_day(day: str, paths: dict[str, Path]) -> Path:
    day0 = int(pd.Timestamp(day).timestamp() * 1000)
    nb = 86_400_000 // BUCKET_MS
    st, ofi, nupd, raw = _book_grid(paths["book"], day0, nb)
    g, ntr, stats = _trades_grid(paths["trades"], day0, nb)
    spot, _ = _trades_grid(paths["spot"], day0, nb, spot=True)
    df = pd.DataFrame({"t": day0 + np.arange(nb, dtype=np.int64) * BUCKET_MS, "bid": st[:, 0], "bidq": st[:, 1].astype("float32"),
                       "ask": st[:, 2], "askq": st[:, 3].astype("float32"), "ofi": ofi.astype("float32"), "nupd": nupd.astype("int32"),
                       "buy_vol": g["buy"][0].astype("float32"), "sell_vol": g["sell"][0].astype("float32"), "ntr": ntr.astype("int32"),
                       "max_buy": g["buy"][1].astype("float32"), "max_sell": g["sell"][1].astype("float32"),
                       "sell_min_px": g["sell"][2], "sell_vol_at_min": g["sell"][3].astype("float32"),
                       "buy_max_px": g["buy"][2], "buy_vol_at_max": g["buy"][3].astype("float32"), "spot_last": spot})
    ROOT.mkdir(parents=True, exist_ok=True)
    p = ROOT / f"{day}.parquet"
    df.to_parquet(p, compression="zstd")
    stats["book_updates"] = int(nupd.sum())
    stats.update(raw)
    stats["processed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (ROOT / f"{day}.json").write_text(json.dumps(stats))
    return p
