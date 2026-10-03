"""Point-in-time fundamentals: no look-ahead, revisions, 15:30 cut-off, split-consistent per-share values."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from marketdata.fundamentals import PITFundamentals, usable_from
from marketdata.nse_filings import parse_xbrl


def _facts(rows: list[dict]) -> pd.DataFrame:
    f = pd.DataFrame(rows)
    f["period_end"] = pd.to_datetime(f["period_end"])
    f["available_at"] = pd.to_datetime(f["available_at"]).dt.tz_localize("Asia/Kolkata").dt.tz_convert("UTC")
    f["usable_from"] = usable_from(f["available_at"])
    for c in ["revenue", "net_profit", "eps_basic", "equity", "paid_up_capital", "face_value", "borrowings"]:
        if c not in f:
            f[c] = np.nan
    return f


def _quarters(n: int, eps0: float = 10.0, growth: float = 0.02, start: str = "2019-03-31") -> list[dict]:
    rows = []
    for i, pe in enumerate(pd.date_range(start, periods=n, freq="QE")):
        rows.append({"symbol": "X", "period_end": pe, "available_at": pe + pd.Timedelta(days=20, hours=17),   # 17:00 IST
                     "revenue": 100.0e9 * (1 + growth) ** i, "net_profit": eps0 * 1e9 * (1 + growth) ** i,
                     "eps_basic": eps0 * (1 + growth) ** i, "equity": 500.0e9, "paid_up_capital": 1e9, "face_value": 1.0})
    return rows


def test_filing_after_the_close_is_usable_only_from_the_next_session():
    s = pd.Series(pd.to_datetime(["2024-01-10 15:29:59", "2024-01-10 15:30:00", "2024-01-10 21:39:43"]).tz_localize("Asia/Kolkata"))
    u = usable_from(s)
    assert list(u.dt.date.astype(str)) == ["2024-01-10", "2024-01-11", "2024-01-11"]


def test_nothing_published_after_t_is_visible_at_t():
    pit = PITFundamentals(_facts(_quarters(12)))
    t = pd.Timestamp("2020-06-15")
    k = pit.known("X", t)
    assert (k["usable_from"] <= t).all()
    assert k["period_end"].max() == pd.Timestamp("2020-03-31")        # Q4 FY20 was published 20 Apr 2020
    assert pit.known("X", pd.Timestamp("2020-04-20"))["period_end"].max() == pd.Timestamp("2019-12-31")   # 17:00 -> next day
    assert pit.known("X", pd.Timestamp("2020-04-21"))["period_end"].max() == pd.Timestamp("2020-03-31")


def test_revision_replaces_a_quarter_only_from_its_own_date():
    rows = _quarters(6)
    orig = rows[-1]
    rows.append({**orig, "available_at": orig["available_at"] + pd.Timedelta(days=30), "net_profit": 999.0})  # noqa: E501
    pit = PITFundamentals(_facts(rows))
    pe = orig["period_end"]
    before = pit.known("X", orig["available_at"] + pd.Timedelta(days=10))
    after = pit.known("X", orig["available_at"] + pd.Timedelta(days=40))
    assert before.loc[before.period_end == pe, "net_profit"].iloc[0] != 999.0
    assert after.loc[after.period_end == pe, "net_profit"].iloc[0] == 999.0


def test_ttm_and_market_cap_use_exchange_price_and_shares():
    pit = PITFundamentals(_facts(_quarters(8)))
    snap = pit.snapshot("X", pd.Timestamp("2021-06-30"), raw_price=200.0)
    assert snap["ttm_np"] == pytest.approx(sum(10.0e9 * 1.02 ** i for i in range(4, 8)))
    assert snap["mcap"] == pytest.approx(1e9 * 200.0)
    assert not snap.get("shares_from_eps")


def test_bonus_between_quarters_does_not_look_like_an_earnings_collapse():
    rows = _quarters(13, growth=0.0)
    bonus = pd.Timestamp("2021-08-15")                      # 1:1 bonus: per-share EPS halves afterwards
    for r in rows:
        if r["period_end"] > bonus:
            r["eps_basic"] /= 2
            r["paid_up_capital"] *= 2
    pit = PITFundamentals(_facts(rows), split_events={"X": [(bonus, 0.5)]})
    t = rows[-1]["available_at"] + pd.Timedelta(days=2)
    snap = pit.snapshot("X", t.tz_localize(None), raw_price=100.0)
    assert snap["ttm_eps"] == pytest.approx(snap["ttm_eps_prev"])          # flat earnings stay flat
    assert abs(snap["sue"]) < 1e-6 or np.isnan(snap["sue"])
    assert snap["shares"] == pytest.approx(2e9)


def test_gap_in_quarters_disables_ttm():
    rows = [r for i, r in enumerate(_quarters(8)) if i != 5]
    snap = PITFundamentals(_facts(rows)).snapshot("X", pd.Timestamp("2021-12-31"), raw_price=100.0)
    assert "ttm_np" not in snap


def test_parser_reads_undefined_oned_context_used_by_older_nse_filings():
    t = ('<in-bse-fin:DateOfStartOfReportingPeriod contextRef="OneD">2019-07-01</in-bse-fin:DateOfStartOfReportingPeriod>'
         '<in-bse-fin:DateOfEndOfReportingPeriod contextRef="OneD">2019-09-30</in-bse-fin:DateOfEndOfReportingPeriod>'
         '<in-bse-fin:RevenueFromOperations contextRef="OneD" unitRef="INR">389770000000.00</in-bse-fin:RevenueFromOperations>'
         '<in-bse-fin:RevenueFromOperations contextRef="FourD" unitRef="INR">771490000000.00</in-bse-fin:RevenueFromOperations>')
    f = parse_xbrl(t)
    assert f["quarter_end"] == "2019-09-30" and f["revenue"] == pytest.approx(389770000000.0)


def test_parser_reads_integrated_filing_namespace():
    t = ('<xbrli:context id="OneD"><xbrli:period><xbrli:startDate>2026-04-01</xbrli:startDate>'
         '<xbrli:endDate>2026-06-30</xbrli:endDate></xbrli:period></xbrli:context>'
         '<in-capmkt:InterestEarned contextRef="OneD" decimals="-5" unitRef="INR">905753300000</in-capmkt:InterestEarned>'
         '<in-capmkt:BasicEarningsPerShareAfterExtraordinaryItems contextRef="OneD">12.5</in-capmkt:BasicEarningsPerShareAfterExtraordinaryItems>')
    f = parse_xbrl(t)
    assert f["revenue"] == pytest.approx(905753300000.0) and f["eps_basic"] == pytest.approx(12.5)


def test_filing_published_after_a_split_is_not_adjusted_twice():
    """DRREDDY pattern: the Sep-2024 quarter ended before the 1:5 split (ex 28 Oct 2024) but was filed after it, already
    stating the post-split face value and restated EPS. Adjusting from the period end would split it twice."""
    rows = _quarters(8, growth=0.0, start="2022-12-31")          # last quarter: Sep 2024, filed 20 Oct 2024 17:00
    split = pd.Timestamp("2024-10-10")                           # between that quarter's end and its filing
    for r in rows:
        if r["available_at"] > split:
            r["eps_basic"] /= 5
            r["net_profit"] = r["net_profit"]                    # company totals do not change
            r["face_value"] = 0.2
    pit = PITFundamentals(_facts(rows), split_events={"X": [(split, 0.2)]})
    snap = pit.snapshot("X", pd.Timestamp("2024-12-15"), raw_price=100.0)
    assert snap["last_period"] == pd.Timestamp("2024-09-30")
    assert snap["shares"] == pytest.approx(5e9)                  # not 25e9
    assert snap["ttm_eps"] == pytest.approx(snap["ttm_eps_prev"])


def test_mixed_statement_basis_disables_multi_quarter_metrics():
    rows = _quarters(8)
    for r in rows[:5]:
        r["consolidated"] = False
    for r in rows[5:]:
        r["consolidated"] = True
    snap = PITFundamentals(_facts(rows)).snapshot("X", pd.Timestamp("2021-12-31"), raw_price=100.0)
    assert "ttm_np" not in snap and "ttm_np_prev" not in snap


def test_cleaning_blanks_parse_errors_without_filling_them():
    from marketdata.fundamentals import clean_facts
    rows = _quarters(12)
    rows[6]["revenue"] = rows[6]["revenue"] / 100          # TRENT Mar-2022 pattern: 100x too low
    rows[8]["face_value"] = 0.0                            # DRREDDY pattern
    rows[9]["eps_basic"] = rows[9]["eps_basic"] * 2.1     # PAGEIND pattern: YTD picked instead of quarter
    f = _facts(rows)
    f["consolidated"] = True
    c, issues = clean_facts(f)
    assert np.isnan(c.loc[6, "revenue"]) and np.isnan(c.loc[8, "face_value"]) and np.isnan(c.loc[9, "eps_basic"])
    assert c["revenue"].notna().sum() == 11 and len(issues) == 3
