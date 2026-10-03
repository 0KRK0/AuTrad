"""Append-only, hash-chained decision log: every order can be replayed and explained."""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def _default(o: Any) -> Any:
    if is_dataclass(o):
        return asdict(o)
    if hasattr(o, "item"):
        return o.item()
    return str(o)


class AuditLog:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.records: list[dict] = []
        self._prev = "0" * 64
        self._lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, kind: str, payload: dict) -> str:
        with self._lock:
            body = json.dumps({"kind": kind, "prev": self._prev, **payload}, default=_default, sort_keys=True)
            h = hashlib.sha256(body.encode()).hexdigest()
            rec = {"hash": h, **json.loads(body)}
            self.records.append(rec)
            self._prev = h
            if self.path:
                with self.path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
            return h

    def verify(self) -> bool:
        prev = "0" * 64
        for r in self.records:
            body = {k: v for k, v in r.items() if k != "hash"}
            if body["prev"] != prev:
                return False
            if hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != r["hash"]:
                return False
            prev = r["hash"]
        return True
