"""Append-only, hash-chained JSONL ledger. Every record stores the sha256 of the previous record, so any edit,
deletion or reordering of history breaks the chain and is detected by verify()."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

GENESIS = "0" * 64


def _digest(rec: dict) -> str:
    body = {k: v for k, v in rec.items() if k != "hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(x) for x in self.path.read_text(encoding="utf-8").splitlines() if x.strip()]

    def head(self) -> str:
        r = self.records()
        return r[-1]["hash"] if r else GENESIS

    def append(self, kind: str, body: dict, now: datetime | None = None) -> dict:
        recs = self.records()
        rec = {"seq": len(recs), "kind": kind, "ts_utc": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
               "prev": recs[-1]["hash"] if recs else GENESIS, **body}
        rec["hash"] = _digest(rec)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n")
        return rec

    def verify(self) -> list[str]:
        problems, prev = [], GENESIS
        for i, r in enumerate(self.records()):
            if r.get("seq") != i:
                problems.append(f"record {i}: sequence number {r.get('seq')}")
            if r.get("prev") != prev:
                problems.append(f"record {i}: chain broken (prev does not match the previous record's hash)")
            if r.get("hash") != _digest(r):
                problems.append(f"record {i}: content changed after it was written")
            prev = r.get("hash")
        return problems

    def of_kind(self, kind: str) -> list[dict]:
        return [r for r in self.records() if r["kind"] == kind]
