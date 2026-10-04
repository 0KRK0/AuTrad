import json

import pandas as pd

from forward import ban_expiry
from scripts import v13_depth, v13_info


def test_n50_membership_matches_snapshots():
    s = pd.read_csv("marketdata/reference/nifty50_pit_snapshots.csv")
    for d in ["2019-03-29", "2023-09-29", "2026-09-30"]:
        snap = set(s[s["effective_date"] == d]["symbol"])
        got = {x for x in v13_info.n50_membership()["symbol"].unique() if v13_info.in_n50(x, pd.Timestamp(d))}
        assert got == snap


def test_ban_ledger_never_backfills_but_records_forward_events():
    s = pd.DatetimeIndex(pd.bdate_range("2026-09-01", "2026-11-30"))
    pos = {d: i for i, d in enumerate(s)}
    exps = [pd.Timestamp("2026-09-29"), pd.Timestamp("2026-10-27")]
    banned = {("SAIL", pd.Timestamp("2026-09-28")), ("SAIL", pd.Timestamp("2026-10-26"))}
    ev = ban_expiry.events_from(s, pos, exps, banned, pd.Timestamp("2026-11-30"), lambda *a: 0.01)
    assert ev["expiry"].tolist() == ["2026-10-27"]          # the September expiry precedes FORWARD_START


def test_depth_study_is_sealed(tmp_path):
    d = tmp_path / "NIFTY-FUT" / "2026-10-05"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"qc": {"ok": True}}))
    assert v13_depth.sealed_days("NIFTY-FUT", tmp_path) == ["2026-10-05"]
    assert v13_depth.split([str(i) for i in range(20)]) == ([str(i) for i in range(12)], [str(i) for i in range(12, 20)])
    import pytest
    with pytest.raises(SystemExit):
        v13_depth.run("NIFTY-FUT", tmp_path / "out", tmp_path)


def test_fields_complete():
    from lab.registry import Registry
    assert all(v13_info.FIELDS.get(k) for k in Registry.SIX_QUESTIONS + ("economic_mechanism",))
