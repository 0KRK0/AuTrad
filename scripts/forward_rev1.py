"""Run the frozen REV1 forward paper test (research/preregistration_v0.5.md, Track A).

    python -m scripts.forward_rev1 --init            # once: freeze spec.json and write the ledger genesis
    python -m scripts.forward_rev1 --fetch           # every NSE session evening (after ~18:30 IST): download, decide, mark
    python -m scripts.forward_rev1                   # re-run on the data already downloaded

Data go to data/forward/ (never the research snapshot). The ledger is forward/rev1/ledger.jsonl (append-only,
hash-chained) and the human-readable state is forward/rev1/status.md.
"""
from __future__ import annotations

import argparse
import shutil
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from forward import rev1


def fetch(root: Path, research_exchange: Path, start: str) -> None:
    from lab.panel import india_equity_assets
    from marketdata.exchange import download
    from marketdata.store import DataStore
    from marketdata.universe import BENCHMARKS

    ex = root / "exchange" / "bse"
    ex.mkdir(parents=True, exist_ok=True)
    for y in range(int(start[:4]), date.today().year + 1):            # seed from the research cache once
        src, dst = research_exchange / "bse" / f"{y}.csv.gz", ex / f"{y}.csv.gz"
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
    cm = pd.read_csv("marketdata/reference/bse_codes.csv", dtype={"bse_code": str})
    print("BSE bhavcopies:", download(root / "exchange", date.today() - timedelta(days=60), date.today(), set(cm["bse_code"]), 4,
                                     log=lambda *a: None))
    assets = [*india_equity_assets(), *[b for b in BENCHMARKS if b.symbol == "NIFTY50"]]
    rep = DataStore(root / "snap").refresh(assets, start=start)
    last = pd.Series({s: q.last for s, q in rep.items()})
    print(f"vendor: {len(rep)} of {len(assets)} assets; latest bar {last.max()}; failed quality: "
          f"{[s for s, q in rep.items() if not q.passed] or 'none'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--root", default="data/forward")
    ap.add_argument("--research-exchange", default="data/exchange")
    a = ap.parse_args()
    if a.init:
        rev1.init()
        print("initialised: forward/rev1/spec.json frozen, genesis written")
        return
    root = Path(a.root)
    if a.fetch:
        spec = rev1.json.loads((rev1.HERE / "spec.json").read_text())
        fetch(root, Path(a.research_exchange), spec["panel_start"])
    rev1.run(root / "snap")


if __name__ == "__main__":
    main()
