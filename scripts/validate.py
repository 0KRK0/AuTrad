"""AuTrad v0.2 real-market validation.

python -m scripts.fetch_data                     # once, or to refresh data
python -m scripts.validate                       # all groups, 100 random traders each
python -m scripts.validate --groups all --random 50 --workers 4
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

from backtest.sim import SimConfig
from marketdata.store import DataStore
from validation import report
from validation.runner import validate_group
from validation.walkforward import GROUPS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="*", default=list(GROUPS))
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--first-test-year", type=int, default=2018)
    ap.add_argument("--random", type=int, default=100)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--cash", type=float, default=10_00_000)
    ap.add_argument("--cash-rate", type=float, default=0.06)
    ap.add_argument("--resume", action="store_true", help="reuse finished groups from reports/_partial")
    a = ap.parse_args()

    t0 = time.time()
    cfg = SimConfig(cash=a.cash, cash_rate=a.cash_rate)
    partial = Path("reports/_partial")
    partial.mkdir(parents=True, exist_ok=True)
    results = {}
    for g in a.groups:
        f = partial / f"{g}.pkl"
        if a.resume and f.exists():
            results[g] = pickle.loads(f.read_bytes())
            print(f"[{g}] reused finished result")
            continue
        results[g] = validate_group(g, a.data_dir, cfg, a.first_test_year, a.random, a.workers)
        f.write_bytes(pickle.dumps(results[g]))   # checkpoint: a crash never loses a finished group
    meta = {"data_manifest": DataStore(a.data_dir).manifest_hash(), "cash": a.cash, "cash_rate": a.cash_rate,
            "first_test_year": a.first_test_year, "n_random": a.random, "runtime_s": round(time.time() - t0)}
    out = report.write(results, meta)
    print(f"\nReport: {out / 'report.md'}  ({meta['runtime_s']}s)")


if __name__ == "__main__":
    main()
