"""Candidate ledger (v1.4): ideas we have seen but not tested. A candidate is NOT a strategy.

    research/candidates.jsonl   one JSON object per line, append-only; the latest line per id is its state

A candidate records where an observation came from, why its prior is low or high, and everything a future
pre-registration must fix before any test. It cannot be traded: `may_trade()` is False for every status this ledger
allows. Promotion out of the ledger happens only through a new hashed pre-registration in research/trials.jsonl,
an independent review, the full gate and a forward paper test, and it is recorded there, not here.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

PATH = Path("research/candidates.jsonl")

STATUSES = (
    "FORWARD / PRE-REGISTRATION REQUIRED",  # observed in-sample; nothing is collected or tested until a new prereg exists
    "FORWARD / LEDGER RUNNING",             # a sealed forward ledger is accumulating; no look before its threshold
    "PRE-REGISTERED",                       # a hashed prereg exists (prereg_sha256 required); the test has not run
    "REJECTED",                             # tested and failed; kept so later searches are penalised for it
)
REQUIRED = ("id", "title", "status", "origin", "observation", "prior", "not_a_strategy", "prereg_must_fix")
PREREG_MUST_FIX = (
    "exact announcement timestamp", "exact reaction-day definition", "entry price", "exit price", "futures costs",
    "market hedge", "sector hedge", "liquidity filter", "capacity", "tax", "multiple-testing correction",
    "independent review",
)


def validate(c: dict) -> list[str]:
    errs = [f"missing {k}" for k in REQUIRED if k not in c]
    if c.get("status") not in STATUSES:
        errs.append(f"status must be one of {STATUSES}")
    if c.get("not_a_strategy") is not True:
        errs.append("not_a_strategy must be true")
    if c.get("status") == "PRE-REGISTERED" and not c.get("prereg_sha256"):
        errs.append("PRE-REGISTERED needs prereg_sha256")
    if not c.get("prereg_must_fix"):
        errs.append("prereg_must_fix must list what a future pre-registration has to fix")
    return errs


def load(path: Path = PATH) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if Path(path).exists():
        for line in Path(path).open(encoding="utf-8"):
            if line.strip():
                c = json.loads(line)
                out[c["id"]] = c
    return out


def add(c: dict, path: Path = PATH) -> dict:
    errs = validate(c)
    if errs:
        raise ValueError("; ".join(errs))
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **c}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def may_trade(c: dict) -> bool:
    """Nothing in the candidate ledger may be traded, paper or live. Always False."""
    return False
