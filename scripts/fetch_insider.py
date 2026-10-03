"""Download NSE insider-trading (SEBI PIT Reg 7(2)) disclosures for every PIT NIFTY 50 symbol (run on your machine).

    python -m scripts.fetch_insider --budget 160      # repeat until it reports todo 0
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from marketdata.nse_filings import Client
from marketdata.nse_insider import fetch_year, manifest
from marketdata.nse_ids import identity_sets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/pit/nse")
    ap.add_argument("--start", type=int, default=2016)
    ap.add_argument("--end", type=int, default=2026)
    ap.add_argument("--budget", type=float, default=160)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    ids = identity_sets()
    syms = sorted({s for k, (_, ss) in ids.items() for s in ss if k not in {"NIFTYBEES", "GOLDBEES", "BANKBEES", "SILVERBEES"}})
    todo = [(s, y) for s in syms for y in range(a.start, a.end + 1) if not (Path(a.root) / "insider" / s / f"{y}.json").exists()]
    from concurrent.futures import ThreadPoolExecutor

    t0 = time.time()
    clients = [Client(pause=0.4) for _ in range(a.workers)]

    def work(k):
        done = 0
        for s, y in todo[k::a.workers]:
            if time.time() - t0 > a.budget:
                break
            fetch_year(Path(a.root), s, y, clients[k])
            done += 1
        return done

    with ThreadPoolExecutor(a.workers) as ex:
        n = sum(ex.map(work, range(a.workers)))
    left = len(todo) - n
    print(f"fetched {n}, todo {left}")
    if left == 0:
        print(f"manifest: {len(manifest(Path(a.root)))} files")


if __name__ == "__main__":
    main()
