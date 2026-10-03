"""Indian capital-gains tax on listed equity / equity ETFs (STT paid), for research P&L after tax.

Rules (by date of sale; research/preregistration_v0.6.md section 2):
  * holding > 12 months = long-term
  * before 2018-04-01: STCG 15 %, LTCG exempt (s.10(38))
  * 2018-04-01 .. 2024-07-22: STCG 15 % (s.111A), LTCG 10 % above Rs 1 lakh (s.112A)
  * from 2024-07-23: STCG 20 %, LTCG 12.5 % above Rs 1.25 lakh (FY 2024-25 exemption Rs 1.25 lakh)
  * cess 3 % before FY 2018-19, 4 % after; no surcharge; 2018 grandfathering ignored
  * set-off: short-term losses against any gain, long-term losses only against long-term gains; carry forward 8 FYs
Simplifications (stated in reports): prices are vendor total-return-adjusted, so dividends appear as capital gains;
transaction costs are not deducted from gains (conservative); tax is paid at the FY's last session.
Source of current rates: incometaxindia.gov.in "Determination of tax in certain special cases" (as amended by Finance Act 2026).
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import pandas as pd

D_2018 = pd.Timestamp("2018-04-01")
D_2024 = pd.Timestamp("2024-07-23")


def fy_of(d: pd.Timestamp) -> int:
    """Financial year label = the calendar year in which it ends (FY 2024-25 -> 2025)."""
    return d.year + 1 if d.month >= 4 else d.year


def stcg_rate(d: pd.Timestamp) -> float:
    return 0.20 if d >= D_2024 else 0.15


def ltcg_rate(d: pd.Timestamp) -> float:
    return 0.125 if d >= D_2024 else (0.10 if d >= D_2018 else 0.0)


def ltcg_exemption(fy: int) -> float:
    return 125_000.0 if fy >= 2025 else 100_000.0


def cess(fy: int) -> float:
    return 0.04 if fy >= 2019 else 0.03


@dataclass
class Realised:
    date: pd.Timestamp
    symbol: str
    gain: float
    long_term: bool


@dataclass
class LotBook:
    lots: dict[str, deque] = field(default_factory=lambda: defaultdict(deque))   # symbol -> deque[[date, qty, price]]
    realised: list[Realised] = field(default_factory=list)

    def buy(self, d: pd.Timestamp, symbol: str, value: float, price: float) -> None:
        if value > 0 and price > 0:
            self.lots[symbol].append([d, value / price, price])

    def sell(self, d: pd.Timestamp, symbol: str, value: float, price: float) -> None:
        q = value / price if price > 0 else 0.0
        book = self.lots[symbol]
        while q > 1e-12 and book:
            lot = book[0]
            take = min(q, lot[1])
            self.realised.append(Realised(d, symbol, take * (price - lot[2]), d > lot[0] + pd.DateOffset(months=12)))
            lot[1] -= take
            q -= take
            if lot[1] <= 1e-12:
                book.popleft()

    def liquidate(self, d: pd.Timestamp, prices: dict[str, float]) -> None:
        for s, book in list(self.lots.items()):
            p = prices.get(s)
            if p and p > 0:
                self.sell(d, s, sum(l[1] for l in book) * p, p)


def tax_by_fy(realised: list[Realised]) -> pd.DataFrame:
    """Tax per financial year with set-off and 8-year carry-forward."""
    rows = defaultdict(lambda: {"st": defaultdict(float), "lt": defaultdict(float)})
    for r in realised:
        rate = ltcg_rate(r.date) if r.long_term else stcg_rate(r.date)
        if r.long_term and rate == 0.0:
            continue        # exempt transfer (s.10(38), before FY 2018-19): the gain is not taxed and a loss cannot be set off or carried
        rows[fy_of(r.date)]["lt" if r.long_term else "st"][rate] += r.gain
    cf_st: list[list] = []       # [fy, amount]
    cf_lt: list[list] = []
    out = []
    for fy in sorted(rows):
        cf_st = [x for x in cf_st if fy - x[0] <= 8]
        cf_lt = [x for x in cf_lt if fy - x[0] <= 8]
        st = {k: v for k, v in rows[fy]["st"].items()}
        lt = {k: v for k, v in rows[fy]["lt"].items()}
        # losses inside the year: ST losses vs ST gains (highest rate first), then vs LT gains; LT losses vs LT gains
        st_loss = -sum(v for v in st.values() if v < 0)
        lt_loss = -sum(v for v in lt.values() if v < 0)
        st_g = {k: v for k, v in st.items() if v > 0}
        lt_g = {k: v for k, v in lt.items() if v > 0}

        def absorb(gains: dict, loss: float) -> float:
            for k in sorted(gains, reverse=True):
                use = min(gains[k], loss)
                gains[k] -= use
                loss -= use
            return loss

        st_loss = absorb(st_g, st_loss)
        lt_loss = absorb(lt_g, lt_loss)
        st_loss = absorb(lt_g, st_loss)
        # brought-forward losses (oldest first): ST b/f vs ST then LT gains; LT b/f vs LT gains only
        for x in cf_st:
            x[1] = absorb(st_g, x[1])
            x[1] = absorb(lt_g, x[1])
        for x in cf_lt:
            x[1] = absorb(lt_g, x[1])
        cf_st = [x for x in cf_st if x[1] > 1e-9]
        cf_lt = [x for x in cf_lt if x[1] > 1e-9]
        if st_loss > 1e-9:
            cf_st.append([fy, st_loss])
        if lt_loss > 1e-9:
            cf_lt.append([fy, lt_loss])
        # exemption on long-term gains (lowest-rate bucket first is irrelevant: one rate per date regime)
        ex = ltcg_exemption(fy)
        for k in sorted(lt_g, reverse=True):
            use = min(lt_g[k], ex)
            lt_g[k] -= use
            ex -= use
        base = sum(k * v for k, v in st_g.items()) + sum(k * v for k, v in lt_g.items())
        out.append({"fy": fy, "stcg_net": sum(st.values()), "ltcg_net": sum(lt.values()), "tax": base * (1 + cess(fy)),
                    "loss_cf_st": sum(x[1] for x in cf_st), "loss_cf_lt": sum(x[1] for x in cf_lt)})
    return pd.DataFrame(out).set_index("fy") if out else pd.DataFrame(columns=["stcg_net", "ltcg_net", "tax"])


def post_tax_nav(nav: pd.Series, fills: list[dict], dates: pd.DatetimeIndex, symbols_price_end: dict[str, float] | None = None,
                 start: pd.Timestamp | None = None) -> dict:
    """Pre-tax NAV + engine fills -> after-tax NAV (tax paid at each FY's last session, scaled to the after-tax capital)
    and the liquidation value at the end (open lots sold at the final prices and taxed)."""
    nav = nav[nav.index >= start] if start is not None else nav
    book = LotBook()
    for f in sorted(fills, key=lambda x: x["t"]):                 # lots from the whole history (cost basis carries over)
        (book.buy if f["delta_w"] > 0 else book.sell)(dates[f["t"]], f["symbol"], f["value"], f["price"])
    end = nav.index[-1]
    keep = (lambda r: r.date >= start) if start is not None else (lambda r: True)
    realised_running = [r for r in book.realised if keep(r)]
    if symbols_price_end:
        book.liquidate(end, symbols_price_end)
    tax_running = tax_by_fy(realised_running)
    tax_all = tax_by_fy([r for r in book.realised if keep(r)])
    # tax payment days: last session of each FY (31 March or earlier); the final partial FY is paid at liquidation
    pay = {}
    for fy, row in tax_running.iterrows():
        fy_end = pd.Timestamp(f"{fy}-03-31")
        sess = nav.index[nav.index <= fy_end]
        if fy_end <= end and len(sess):
            pay[sess[-1]] = row["tax"]
    r = nav / nav.shift(1)
    post = [nav.iloc[0]]
    scale = 1.0                                  # after-tax capital / pre-tax capital
    for d, g in r.iloc[1:].items():
        v = post[-1] * g
        if d in pay:
            v -= pay[d] * scale
        post.append(v)
        scale = v / nav.loc[d]
    post_s = pd.Series(post, index=nav.index)
    paid = sum(pay.values())
    terminal_tax = (tax_all["tax"].sum() if len(tax_all) else 0.0) - paid
    liquidation = post_s.iloc[-1] - max(terminal_tax, 0.0) * scale
    return {"post_tax_nav": post_s, "tax_by_fy": tax_all, "tax_paid_along_the_way": paid,
            "terminal_tax": terminal_tax, "liquidation_value": liquidation}
