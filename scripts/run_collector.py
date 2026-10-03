"""Poll internet evidence feeds.  python -m scripts.run_collector --every 300"""
from __future__ import annotations

import argparse
import asyncio

from intel.collector import Collector, EvidenceStore

ALIASES = {  # extend per universe; production loads this from the symbols table
    "AAPL": ["Apple"], "MSFT": ["Microsoft"], "NVDA": ["Nvidia"], "AMZN": ["Amazon"], "GOOGL": ["Alphabet", "Google"],
}


async def main(every: int, once: bool) -> None:
    store = EvidenceStore()
    col = Collector(store, ALIASES)
    while True:
        n = await col.run_once()
        print(f"ingested {n} new items; store={len(store.items)}")
        for e in store.items[-5:]:
            print(f"  {e.sentiment:+.2f} {e.source:10} {e.symbols} {e.title[:90]}")
        if once:
            return
        await asyncio.sleep(every)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=300)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    asyncio.run(main(a.every, a.once))
