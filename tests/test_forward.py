"""Forward REV1 paper test: tamper-evident ledger, frozen code, month-end timing, no look-ahead."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from forward import rev1
from forward.ledger import Ledger
from tests.test_factors import _panel


def test_ledger_detects_edit_delete_and_reorder(tmp_path):
    L = Ledger(tmp_path / "l.jsonl")
    for i in range(4):
        L.append("mark", {"date": f"2026-10-0{i + 1}", "nav": 100.0 + i})
    assert L.verify() == []
    lines = (tmp_path / "l.jsonl").read_text().splitlines()
    edited = lines.copy()
    r = json.loads(edited[1])
    r["nav"] = 999.0
    edited[1] = json.dumps(r, sort_keys=True, separators=(",", ":"))
    (tmp_path / "l.jsonl").write_text("\n".join(edited) + "\n")
    assert any("changed" in p for p in L.verify())
    (tmp_path / "l.jsonl").write_text("\n".join(lines[:1] + lines[2:]) + "\n")        # delete a record
    assert L.verify()
    (tmp_path / "l.jsonl").write_text("\n".join([lines[0], lines[2], lines[1], lines[3]]) + "\n")
    assert L.verify()


def _status(dates, exch, max_wait=5):
    return {r["month"]: r for r in rev1.month_end_status(pd.DatetimeIndex(dates), set(exch), pd.Timestamp("2026-09-01"), max_wait)}


def test_month_end_timing_and_exchange_confirmation():
    d = pd.bdate_range("2026-09-01", "2026-11-20")
    st = _status(d, d)
    assert st["2026-09"]["date"] == pd.Timestamp("2026-09-30") and st["2026-09"]["timing"] == "LATE" and st["2026-09"]["status"] == "READY"
    upto = d[d <= "2026-10-30"]
    st = _status(upto, upto)                                   # evening of the last weekday of October
    assert st["2026-10"]["timing"] == "ON_TIME" and st["2026-10"]["status"] == "READY"
    st = _status(upto, upto[:-1])                              # the exchange file for 30 Oct is not out yet
    assert st["2026-10"]["status"] == "WAIT"
    assert "2026-10" not in _status(d[d <= "2026-10-29"], d)   # month not finished
    holes = d[d != pd.Timestamp("2026-09-30")]                 # vendor lacks the real month-end, exchange has it
    st = _status(holes[holes <= "2026-10-02"], d)
    assert st["2026-09"]["date"] == pd.Timestamp("2026-09-29") and st["2026-09"]["status"] == "WAIT"
    st = _status(holes[holes <= "2026-10-12"], d)
    assert st["2026-09"]["status"] == "MISSED"


def test_decision_requires_a_truncated_panel():
    P = _panel(T=400)
    with pytest.raises(AssertionError):
        rev1.rev1_decision(P, 300)


def test_decision_is_top_third_of_minus_last_month_return():
    P = _panel(T=400, N=24)
    d = rev1.rev1_decision(P, 399)
    C = P.ffill_close()
    s = pd.Series({P.symbols[j]: -(C[399, j] / C[378, j] - 1) for j in range(len(P.symbols)) if P.symbols[j] in dict(d["ranking"])})
    assert sorted(s.nlargest(len(s) // 3).index) == d["selected"]
    assert abs(sum(d["weights"].values()) - 1) < 1e-9


def test_run_refuses_when_frozen_code_changes(tmp_path, monkeypatch):
    spec = rev1.make_spec()
    spec["code_hashes"]["lab.factors.price_scores"] = "0" * 64
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    L = Ledger(tmp_path / "ledger.jsonl")
    L.append("genesis", {"spec_sha256": rev1._sha_file(tmp_path / "spec.json")})
    with pytest.raises(SystemExit, match="VOID"):
        rev1.run(tmp_path / "nodata", tmp_path / "ledger.jsonl", tmp_path / "spec.json", tmp_path / "status.md")
    assert L.of_kind("integrity")[0]["status"] == "VOID"


def test_run_refuses_when_spec_edited(tmp_path):
    spec = rev1.make_spec()
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    Ledger(tmp_path / "ledger.jsonl").append("genesis", {"spec_sha256": rev1._sha_file(tmp_path / "spec.json")})
    spec["capital"] = 2_000_000.0
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    with pytest.raises(SystemExit, match="spec.json changed"):
        rev1.run(tmp_path / "nodata", tmp_path / "ledger.jsonl", tmp_path / "spec.json", tmp_path / "status.md")


def test_gate_shadow_reports_sector_cap_without_changing_targets():
    w = {f"S{i}": 1 / 16 for i in range(16)}
    sectors = {f"S{i}": ("it" if i < 6 else f"x{i}") for i in range(16)}       # 6 x 6.25% = 37.5% in one sector
    g = rev1.gate_shadow({"date": "2026-09-30", "weights": w}, {}, sectors)
    assert g["would_change"] >= 1 and "sector it capped" in g["summary"]
    assert w == {f"S{i}": 1 / 16 for i in range(16)}


def _store(tmp_path, n=320, shock_after=None):
    from marketdata.store import DataStore
    from marketdata.universe import Asset

    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2025-01-01", periods=n)
    st = DataStore(tmp_path)
    for sym in ("AAA", "BBB", "NIFTY50"):
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        if shock_after is not None:
            c[dates > shock_after] *= 0.1                       # a wild (fake split-like) move after the cut date
        df = pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1e6}, index=dates)
        df.index.name = "date"
        st.save(Asset(sym, sym, "equity" if sym != "NIFTY50" else "index", "x", "it", "nse_largecap"), df, "test")
    return dates


def test_asof_cuts_raw_data_before_any_correction(tmp_path):
    from lab.panel import build_panel
    from marketdata.universe import Asset

    a = [Asset("AAA", "AAA", "equity", "x", "it", "nse_largecap"), Asset("BBB", "BBB", "equity", "x", "it", "nse_largecap")]
    dates = _store(tmp_path / "x" / "snap")
    cut = dates[200]
    P1 = build_panel(str(tmp_path / "x" / "snap"), a, start="2025-01-01", end=str(cut.date()), asof=str(cut.date()))
    _store(tmp_path / "y" / "snap", shock_after=cut)
    P2 = build_panel(str(tmp_path / "y" / "snap"), a, start="2025-01-01", end=str(cut.date()), asof=str(cut.date()))
    assert np.allclose(P1.C, P2.C, equal_nan=True) and (P1.eligible == P2.eligible).all()


def test_panel_keeps_the_exchange_quality_gate(tmp_path, monkeypatch):
    """Regression: build_panel used to lose the per-day quality scores when it loaded the benchmark, so the
    'no buys on unconfirmed prices' rule was silently off."""
    from lab.panel import build_panel
    from marketdata.store import DataStore
    from marketdata.universe import Asset

    a = [Asset("AAA", "AAA", "equity", "x", "it", "nse_largecap"), Asset("BBB", "BBB", "equity", "x", "it", "nse_largecap")]
    dates = _store(tmp_path / "snap")
    orig = DataStore.load

    def load(self, symbols, *args, **kw):
        out = orig(self, symbols, *args, **kw)
        if "AAA" in self.quality:
            q = self.quality["AAA"].copy()
            q[dates[100]] = 0.3
            self.quality["AAA"] = q
        return out

    monkeypatch.setattr(DataStore, "load", load)
    P = build_panel(str(tmp_path / "snap"), a, start="2025-01-01")
    assert not P.eligible[P.dates.get_loc(dates[100]), P.idx["AAA"]]
    assert P.eligible[P.dates.get_loc(dates[101]), P.idx["AAA"]]


def test_asof_feeds_no_later_row_into_the_exchange_correction(tmp_path, monkeypatch):
    """The exchange-anchored correction fits steps and rolling medians over whole series, so later rows could move
    earlier prices. With asof, the correction must never even see a row after the cut."""
    import marketdata.corporate_actions as CA
    from lab.panel import build_panel
    from marketdata.store import DataStore
    from marketdata.universe import Asset

    rng = np.random.default_rng(8)
    dates = pd.bdate_range("2025-01-01", periods=300)
    st = DataStore(tmp_path / "snap")
    rows = []
    for sym, code in (("INFY", "500209"), ("TCS", "532540"), ("NIFTY50", None)):
        c = 1000 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
        df = pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1e6}, index=dates)
        df.index.name = "date"
        st.save(Asset(sym, sym, "equity" if code else "index", "x", "it", "nse_largecap"), df, "test")
        if code:
            rows.append(pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "code": code, "name": sym, "isin": "", "open": c, "high": c * 1.01,
                                      "low": c * 0.99, "close": c, "prev_close": np.r_[c[0], c[:-1]], "volume": 1000, "turnover": 1e6}))
    (tmp_path / "exchange" / "bse").mkdir(parents=True)
    pd.concat(rows).to_csv(tmp_path / "exchange" / "bse" / "2025.csv.gz", index=False)
    seen = []
    orig = CA.correct_with_exchange

    def spy(symbol, vendor, bse, bench, registry):
        seen.append((symbol, vendor.index.max(), bse.index.max(), None if bench is None else bench.index.max()))
        return orig(symbol, vendor, bse, bench, registry)

    monkeypatch.setattr(CA, "correct_with_exchange", spy)
    cut = dates[200]
    a = [Asset("INFY", "INFY", "equity", "x", "it", "nse_largecap"), Asset("TCS", "TCS", "equity", "x", "it", "nse_largecap")]
    build_panel(str(tmp_path / "snap"), a, start="2025-01-01", end=str(cut.date()), asof=str(cut.date()))
    assert {s for s, *_ in seen} == {"INFY", "TCS"}
    assert all(v <= cut and b <= cut and (m is None or m <= cut) for _, v, b, m in seen)
