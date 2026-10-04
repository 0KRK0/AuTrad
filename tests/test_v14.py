"""v1.4: ARGO freeze, candidate ledger, data clock, desk review import."""
import hashlib
import json
from pathlib import Path

import pytest

from forward import data_clock
from lab import candidates as C

# ARGO is frozen (v1.4): no edits until a candidate has passed the full gate and a forward paper test.
# ARGO's role afterwards is the risk layer, not return generation. Changing a file here requires a new
# pre-registration that says why, and an update of these hashes in the same change.
ARGO_FROZEN = {
    "__init__.py": "dca789a7252b8145bcf9984f12a1f00287b193c8aa9245e18ec801078716c361",
    "experts.py": "2647e13199e33bc2f85c749922ac4924a8e253d63b538f08df0fa5ceb28c387f",
    "features.py": "b1d48453e132a9faf6e76646305f308f6d1bf55ec6d6000d07159383b7fd582e",
    "meta.py": "25bef1ec6d6c78174622cc2489d2ac602ddbef7c34032be9470b07e250490366",
    "model.py": "fac4b63f1d80d87b33fef33341e901558da276f845c06af682647333370e9413",
}


def test_argo_is_frozen():
    got = {p.name: hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in sorted(Path("argo").glob("*.py"))}
    assert got == ARGO_FROZEN, "ARGO is frozen in v1.4; see argo/FROZEN.md"


def test_c013_is_recorded_and_not_a_strategy():
    led = C.load()
    c = led["C-013"]
    assert c["status"] == "FORWARD / PRE-REGISTRATION REQUIRED"
    assert c["not_a_strategy"] is True and c.get("do_not_trade") is True
    for item in ("exact announcement timestamp", "market hedge", "sector hedge", "tax", "multiple-testing correction", "independent review"):
        assert item in c["prereg_must_fix"]
    assert not any(C.may_trade(x) for x in led.values())


def test_candidate_validation(tmp_path):
    good = {"id": "C-X", "title": "t", "status": "FORWARD / PRE-REGISTRATION REQUIRED", "origin": "o", "observation": "x",
            "prior": "low", "not_a_strategy": True, "prereg_must_fix": list(C.PREREG_MUST_FIX)}
    C.add(good, tmp_path / "c.jsonl")
    with pytest.raises(ValueError):
        C.add({**good, "status": "LIVE"}, tmp_path / "c.jsonl")
    with pytest.raises(ValueError):
        C.add({**good, "not_a_strategy": False}, tmp_path / "c.jsonl")
    with pytest.raises(ValueError):
        C.add({**good, "status": "PRE-REGISTERED"}, tmp_path / "c.jsonl")
    assert list(C.load(tmp_path / "c.jsonl")) == ["C-X"]


def _snap(root, day, sym, n=5, cur=10.0, avg=10.0):
    d = root / day
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{sym}.json").write_text(json.dumps({"symbol": sym, "fetched_at": day, "tier": 2, "tables": {
        "earnings_estimate": [{"period": "0y", "avg": avg, "numberOfAnalysts": n, "currency": "INR"}],
        "eps_trend": [{"period": "0y", "current": cur, "currency": "INR"}]}}))


def test_data_clock_analyst_counts(tmp_path):
    for day in ("2026-10-01", "2026-11-30"):
        for i in range(100):
            _snap(tmp_path, day, f"S{i}")
    _snap(tmp_path, "2026-11-30", "FEW", n=2)
    _snap(tmp_path, "2026-11-30", "BAD", cur=12.0)
    a = data_clock.analyst(tmp_path)
    assert a["snapshot_days"] == 2 and a["latest_names"] == 102 and a["latest_valid"] == 100
    assert a["cross_sections"] == 1  # October has no earlier snapshot; November does (60 days back)
    assert not a["unsealed"]


def test_data_clock_depth(tmp_path):
    for i, ok in enumerate([True, True, False]):
        d = tmp_path / "RELIANCE" / f"2026-10-0{i + 5}"
        d.mkdir(parents=True)
        (d / "manifest.json").write_text(json.dumps({"qc": {"ok": ok}}))
    r = data_clock.depth(tmp_path)
    assert r["sealed_days_max"] == 2 and r["recorded_days_max"] == 3 and not r["unsealed"]
    assert data_clock.depth(tmp_path / "none")["blocker"]


def test_desk_import_rows():
    from scripts.import_promise_reviews import read_reviews, to_rows
    queue = [{"claim_id": "C1", "symbol": "ABC", "speaker": "s", "said": "2023-01-01", "quote": "q", "metric": "m", "target": "t",
              "due": "2024-03-31", "suggested": "MISSED", "candidates": [{}]}]
    revs = [{"claim_id": "C1", "verdict": "partial", "evidence_doc_id": "D1", "evidence_quote": "  a   b ", "reviewer_id": "u_1", "note": "n"},
            {"claim_id": "C2", "verdict": ""}]
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        Path(d, "r.json").write_text(json.dumps([{"id": "C1", "data": revs[0]}, revs[1]]))
        got = read_reviews(Path(d))
    assert len(got) == 1
    df = to_rows(got, queue)
    row = df.iloc[0]
    assert row.verdict == "PARTIAL" and row.evidence_quote == "a b" and row.reviewer == "desk:u_1" and row.symbol == "ABC"
    assert to_rows(got, queue, "Rajesh").iloc[0].reviewer == "Rajesh"
