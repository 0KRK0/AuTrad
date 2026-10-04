"""v1.3 data (research/preregistration_v1.3.md, Part A): NSE results index and XBRL facts for every stock with single
stock futures 2016-2026. Index JSON is kept in data/pit_v13/nse/index; raw XBRL goes to local scratch
(/tmp/autrad_pit, not the repository); parsed facts are appended per shard to data/pit_v13/facts_<k>of<n>.csv.

    python -m scripts.v13_data index --shard 0/3 --budget 150
    python -m scripts.v13_data xbrl  --shard 0/3 --budget 150
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from marketdata.nse_filings import Client, filings_index, parse_xbrl, select_filings

ROOT = Path("data/pit_v13")
RAW = Path("/tmp/autrad_pit/nse/xbrl")
NEEDED_ONLY = {"on": False}


def symbols() -> list[str]:
    from marketdata.nse_fo import load_fo
    f = load_fo(2016, 2026)
    return sorted(f.loc[f["kind"] == "STK", "symbol"].unique())


def run_index(shard: int, n: int, budget: float) -> dict:  # pragma: no cover - network
    t0 = time.time()
    c = Client(pause=0.3)
    c.get("https://www.nseindia.com/")
    todo = [s for k, s in enumerate(symbols()) if k % n == shard]
    done = 0
    for s in todo:
        if time.time() - t0 > budget:
            break
        if all((ROOT / "nse" / "index" / f"{s}_{kind}.json").exists() for kind in ("legacy", "integrated")):
            continue
        try:
            filings_index(ROOT, s, c)
        except Exception as e:                           # the JSON is saved before parsing; parse errors are fixed offline
            print("parse", s, type(e).__name__)
        done += 1
    left = sum(1 for s in todo if not all((ROOT / "nse" / "index" / f"{s}_{kind}.json").exists() for kind in ("legacy", "integrated")))
    return {"fetched": done, "left": left, "seconds": round(time.time() - t0, 1)}


def load_index() -> pd.DataFrame:
    out = []
    for s in symbols():
        if (ROOT / "nse" / "index" / f"{s}_legacy.json").exists():
            out.append(filings_index(ROOT, s, None))
    return pd.concat([x for x in out if len(x)], ignore_index=True)


def run_xbrl(shard: int, n: int, budget: float, workers: int = 8) -> dict:  # pragma: no cover - network
    t0 = time.time()
    RAW.mkdir(parents=True, exist_ok=True)
    out_p = ROOT / f"facts_{shard}of{n}.csv"
    have = set(pd.read_csv(out_p, usecols=["xbrl"]).astype(str)["xbrl"]) if out_p.exists() else set()
    cache = RAW.parent / "selected_filings.pkl"
    if cache.exists():
        sel = pd.read_pickle(cache)
    else:
        idx = load_index()
        sel = pd.concat([select_filings(g, "2015-01-01") for _, g in idx.groupby("symbol")], ignore_index=True)
        sel.to_pickle(cache)
    if NEEDED_ONLY["on"] and Path("data/pit_v13/needed_filings.csv").exists():
        sel = pd.read_csv("data/pit_v13/needed_filings.csv").drop_duplicates("xbrl").reset_index(drop=True)
    sel = sel[sel.index % n == shard]
    jobs = [r for _, r in sel.iterrows() if str(r["xbrl"]) not in have]
    import requests
    from marketdata.nse_bhav import UA
    sess = requests.Session()

    def get(url):
        for k in range(3):
            try:
                r = sess.get(url, headers=UA, timeout=30)
                if r.status_code == 200 and r.content:
                    return r.content
                if r.status_code == 404:
                    return None
            except Exception:
                pass
            time.sleep(1 + k)
        return None

    def fetch(i_r):
        i, r = i_r
        if time.time() - t0 > budget:
            return None
        fn = RAW / str(r["xbrl"]).rsplit("/", 1)[-1]
        if not fn.exists():
            blob = get(r["xbrl"])
            if blob is None:
                return None
            fn.write_bytes(blob)
        raw = fn.read_bytes()
        try:
            f = parse_xbrl(raw.decode("utf-8", "ignore"))
        except Exception:
            f = {}
        return {**{k: r[k] for k in ["symbol", "source", "period_end", "consolidated", "audited", "published_at", "disseminated_at",
                                     "available_at", "filing_id", "revision", "xbrl", "bank"]},
                **f, "sha256": hashlib.sha256(raw).hexdigest(), "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    with ThreadPoolExecutor(workers) as ex:
        rows = [x for x in ex.map(fetch, enumerate(jobs)) if x is not None]
    if rows:
        new = pd.DataFrame(rows)
        old = [pd.read_csv(out_p)] if out_p.exists() else []
        pd.concat(old + [new], ignore_index=True).to_csv(out_p, index=False)
    return {"parsed": len(rows), "left": len(jobs) - len(rows), "seconds": round(time.time() - t0, 1)}


def load_facts() -> pd.DataFrame:
    fs = sorted(ROOT.glob("facts_*of*.csv"))
    return pd.concat([pd.read_csv(p) for p in fs], ignore_index=True) if fs else pd.DataFrame()


if __name__ == "__main__":  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["index", "xbrl"])
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--budget", type=float, default=150)
    ap.add_argument("--needed", action="store_true", help="only the filings HM1b needs (data/pit_v13/needed_filings.csv)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    NEEDED_ONLY["on"] = a.needed
    k, n = (int(x) for x in a.shard.split("/"))
    print(run_index(k, n, a.budget) if a.cmd == "index" else run_xbrl(k, n, a.budget, a.workers))
