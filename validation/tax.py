"""Indian tax estimate on realised trades (residents, FY April-March). An ESTIMATE, not tax advice.

Rates (FY 2025-26 / 2026-27, unchanged in Budget 2026), plus 4% health & education cess:
  Listed equity & equity ETFs : STCG 20% (held <= 12 months); LTCG 12.5% above Rs1.25 lakh/yr.
                                Short-term losses offset gains within the year.
  Crypto (VDA, s.115BBH)      : 30% on each profitable trade; losses give NO relief.
  Currency futures, gold/silver ETFs: taxed at slab (assumed 30%) on the year's net profit.
Surcharge is ignored. Unrealised gains on open positions are not taxed.
"""
from __future__ import annotations

import pandas as pd

CESS = 1.04
EQUITY_CATS = {"in_largecap", "in_midsmall", "in_index_fno"}
CRYPTO_CATS = {"crypto_major", "crypto_alt", "crypto_meme"}


def fy(ts: pd.Timestamp) -> int:
    return ts.year + 1 if ts.month >= 4 else ts.year


def estimate(ledger: pd.DataFrame, slab: float = 0.30) -> pd.DataFrame:
    if ledger.empty:
        return pd.DataFrame(columns=["equity_tax", "crypto_tax", "other_tax", "total_tax"])
    led = ledger[~ledger["open"]].copy()
    led["fy"] = led["exit"].map(fy)
    rows = []
    for year, g in led.groupby("fy"):
        eq = g[g.category.isin(EQUITY_CATS)]
        st = eq[eq.days <= 365].pnl.sum()
        lt = eq[eq.days > 365].pnl.sum()
        if st < 0:                               # short-term loss can offset long-term gain
            lt, st = lt + st, 0.0
        equity_tax = (0.20 * max(st, 0) + 0.125 * max(lt - 1_25_000, 0)) * CESS
        cr = g[g.category.isin(CRYPTO_CATS)]
        crypto_tax = 0.30 * cr[cr.pnl > 0].pnl.sum() * CESS
        other = g[~g.category.isin(EQUITY_CATS | CRYPTO_CATS)]
        other_tax = slab * max(other.pnl.sum(), 0) * CESS
        rows.append({"fy": int(year), "equity_tax": equity_tax, "crypto_tax": crypto_tax, "other_tax": other_tax,
                     "total_tax": equity_tax + crypto_tax + other_tax})
    return pd.DataFrame(rows).set_index("fy")


def after_tax_equity(equity: pd.Series, taxes: pd.DataFrame) -> pd.Series:
    """Deduct each FY's tax on 31 July after the FY ends (typical filing date); capital shrinks from then on."""
    e = equity.copy()
    if taxes.empty:
        return e
    for year, row in taxes.iterrows():
        pay_date = pd.Timestamp(f"{year}-07-31")
        if row.total_tax > 0 and pay_date <= e.index[-1]:
            mask = e.index >= pay_date
            factor = 1 - row.total_tax / float(e[mask].iloc[0])
            e[mask] = e[mask] * factor
    return e
