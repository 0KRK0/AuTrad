"""Download BSE shareholding-pattern history (promoter, pledges, institutions) for every PIT NIFTY 50 company.

    python -m scripts.fetch_bse_shp --budget 3000        # resumable; re-run until it reports todo 0
    python -m scripts.fetch_bse_shp --wide --parts promoter --shard 0/2   # v0.8.1: the TOP500 universe (bse_codes_wide.csv)
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from marketdata.bse_shp import Client, fetch_index, fetch_part, fetch_xbrl, manifest, needs_xbrl, todo

EXTRA = {"HDFC": "500010", "TATAMTRDVR": "570001"}      # delisted / merged, not in bse_codes.csv


def wide_codes() -> dict[str, str]:
    w = pd.read_csv(Path("marketdata/reference/bse_codes_wide.csv"), dtype=str).dropna(subset=["bse_code"])
    return {r.bse_code: r.symbol for r in w.itertuples(index=False) if r.bse_code}


def codes() -> dict[str, str]:
    b = pd.read_csv(Path("marketdata/reference/bse_codes.csv"), dtype=str)
    m = {r.bse_code: r.symbol for r in b.itertuples(index=False) if r.symbol not in {"NIFTYBEES", "GOLDBEES", "BANKBEES", "SILVERBEES"}}
    m.update({v: k for k, v in EXTRA.items()})
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/pit/bse")
    ap.add_argument("--budget", type=float, default=160)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--wide", action="store_true")
    ap.add_argument("--parts", default="promoter,public")
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    root = Path(a.root)
    t0 = time.time()
    cl = Client()
    jobs = []
    shard_k, shard_n = (int(x) for x in a.shard.split("/"))
    allc = sorted(set(codes()) | (set(wide_codes()) if a.wide else set()))
    mine = allc[shard_k::shard_n]
    parts = tuple(a.parts.split(","))
    clients = [Client() for _ in range(a.workers)]

    def idx(k2: int) -> list:
        got = []
        for code in mine[k2::a.workers]:
            if time.time() - t0 > a.budget * 0.6:
                break
            rows = fetch_index(root, code, clients[k2])
            if rows is None:
                print("index failed", code)
                continue
            got += [(code, part, q) for part, q in todo(root, code, rows, parts)]
        return got

    with ThreadPoolExecutor(a.workers) as ex:
        for g in ex.map(idx, range(a.workers)):
            jobs += g
    print(f"{len(jobs)} parts to fetch", flush=True)

    def work(k: int) -> int:
        n = 0
        for code, part, q in jobs[k::a.workers]:
            if time.time() - t0 > a.budget:
                break
            n += fetch_part(root, code, part, q, clients[k])
        return n

    with ThreadPoolExecutor(a.workers) as ex:
        done = sum(ex.map(work, range(a.workers)))
    left = len(jobs) - done
    print(f"fetched {done}, todo {left}", flush=True)
    if left == 0:                                         # as-filed XBRL for filings whose structured record is a later version
        ck_p = root / "shp" / f"xbrl_checked_{a.parts.replace(',', '_')}_{shard_k}of{shard_n}.json"
        checked = set(json.loads(ck_p.read_text())) if ck_p.exists() else set()
        xj, newly = [], []
        for c in mine:                                    # reading every stored record is slow on a mounted folder:
            if c in checked or not (root / "shp" / c / "index.json").exists():
                continue                                  # done incrementally, remembered per code
            if time.time() - t0 > a.budget * 0.6:
                break
            xj += [(c, r["XbrlFile"]) for r in needs_xbrl(root, c, fetch_index(root, c, cl))
                   if not (root / "shp" / c / "xbrl" / (r["XbrlFile"] + ".gz")).exists()]
            newly.append(c)
        print(f"{len(xj)} as-filed XBRL to fetch ({len(newly)} codes checked now, {len([c for c in mine if c not in checked]) - len(newly)} left)", flush=True)

        def xwork(k: int) -> int:
            n = 0
            for c, f in xj[k::a.workers]:
                if time.time() - t0 > a.budget:
                    break
                n += fetch_xbrl(root, c, f, clients[k])
            return n

        with ThreadPoolExecutor(a.workers) as ex:
            nx = sum(ex.map(xwork, range(a.workers)))
        if nx == len(xj):
            checked |= set(newly)
            ck_p.write_text(json.dumps(sorted(checked)))
        left = len(xj) - nx + len([c for c in mine if c not in checked])
        print(f"xbrl fetched {nx}, todo {left}", flush=True)
        if left == 0 and shard_n == 1:
            print(f"manifest: {len(manifest(root))} files")



if __name__ == "__main__":
    main()
