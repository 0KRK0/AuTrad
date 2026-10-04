"""Every Binance USDT spot pair, listed or delisted, 1-day bars from the public archive (resumable).

    python -m scripts.fetch_crypto --budget 160 --shard 0/2
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from marketdata.crypto_binance import ROOT, fetch_symbol, list_symbols, manifest, research_symbol


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=160)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    t0 = time.time()
    s = requests.Session()
    lp = ROOT / "symbols.json"
    if not lp.exists():
        ROOT.mkdir(parents=True, exist_ok=True)
        lp.write_text(json.dumps(list_symbols(s)))
    syms = [x for x in json.loads(lp.read_text()) if research_symbol(x)]
    k, n = (int(x) for x in a.shard.split("/"))
    todo = [x for x in syms[k::n] if not (ROOT / "1d" / f"{x}.csv.gz").exists()]
    sessions = [requests.Session() for _ in range(a.workers)]

    def work(i: int) -> int:
        c = 0
        for sym in todo[i::a.workers]:
            if time.time() - t0 > a.budget:
                break
            try:
                fetch_symbol(sessions[i], sym)
                c += 1
            except Exception:  # noqa: BLE001 - retried on the next run
                time.sleep(2)
        return c

    with ThreadPoolExecutor(a.workers) as ex:
        done = sum(ex.map(work, range(a.workers)))
    left = len(todo) - done
    print(f"USDT research pairs {len(syms)}; this shard fetched {done}, todo {left}")
    if left == 0 and n == 1:
        print("manifest", len(manifest()))


if __name__ == "__main__":
    main()
