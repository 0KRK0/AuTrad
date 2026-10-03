"""Re-derive data/pit/nse/facts.csv from the stored raw NSE files (no network), e.g. after a parser fix.

    python -m scripts.rebuild_fundamentals --root data/pit
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from marketdata.nse_filings import rebuild_facts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/pit")
    ap.add_argument("--start", default="2014-01-01")
    a = ap.parse_args()
    cm = pd.read_csv("marketdata/reference/bse_codes.csv")
    syms = [s for s in cm["symbol"] if s not in {"NIFTYBEES", "BANKBEES", "GOLDBEES", "SILVERBEES"}]
    f = rebuild_facts(a.root, syms, a.start)
    p = Path(a.root) / "nse" / "facts.csv"
    f.to_csv(p, index=False)
    man = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "symbols": int(f["symbol"].nunique()),
           "filings": len(f), "facts_sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "rebuilt_from_raw": True}
    (Path(a.root) / "nse" / "manifest.json").write_text(json.dumps(man, indent=1))
    print(man)


if __name__ == "__main__":
    main()
