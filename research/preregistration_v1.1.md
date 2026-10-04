# AuTrad v1.1 — pre-registration: Indian Basis & Relative-Value Lab

Written 2026-10-04 and hashed into `research/trials.jsonl` **before any F&O data was downloaded or any v1.1 result
was computed**. The cash-market prices (whole-market NSE bhavcopy), the NSE corporate-action file and the index ETFs'
prices were already in the repository from v0.8.1; no basis or futures quantity has been computed from them.

> Guiding rule: *don't search for something that predicts price; search for a market mechanism that forces someone
> to pay us.* Every hypothesis below states its `economic_mechanism`, who is on the other side, and why the edge is
> not arbitraged away. A hypothesis without a mechanism gets a lower research priority.

## 0. Data (to be downloaded and audited after hashing)

* **NSE F&O bhavcopy**, every session 2015-01-01 … 2026-09-30, **futures only** (legacy `FUTIDX/FUTSTK`; UDiFF
  `IDF/STF`): symbol, expiry, open/high/low/close, settlement price, contracts, traded value, open interest.
* **Index closes** (NSE `ind_close_all` daily files): NIFTY 50 and NIFTY BANK close and dividend yield.
* **Cash**: `data/exchange_all/returns.parquet` (close, total return, dividends, price factors) for stocks and the
  ETFs NIFTYBEES and BANKBEES.
* **Corporate actions**: `data/exchange_all/corp_actions` (ex-dates of dividends, splits, bonuses, rights, demergers).
* **Cash rate r_t**: RBI policy repo rate, step series (RBI MPC decisions; stored in
  `marketdata/reference/rbi_repo.csv` with sources). A liquid fund earns about the repo rate.
* **Audit before any test**: coverage of sessions; futures with zero contracts (close = theoretical price) flagged;
  expiry calendar reconstructed; basis distribution; joins of futures to cash by (date, symbol); the share of
  stock-days with a corporate action inside the contract life.

## 1. Common definitions

* Near contract: the earliest expiry T ≥ t. τ = calendar days from t to T.
* **Signal day t, execution day t+1** (next session), at that day's closing prices for both legs. Both NSE closes are
  last-30-minute weighted averages, so they are executable approximately by working both legs over the last half hour;
  a half-spread is charged on every leg and side on top.
* **Exit** at the expiry-day closes of both legs (futures converge to the underlying's close; for physically settled
  stock futures, delivery is cost-equivalent to selling the stock: delivery STT 0.1 %).
* **Expected locked gross carry at t** (fraction): g_t = (F_t − S_t + D_t)/S_t, where D_t = dividends per share with
  ex-date in (t, T] from the corporate-action file (dividends are announced weeks before the ex-date; this is a
  near-point-in-time proxy and is disclosed as such). Indices: g_t = ln(F_t/I_t) + dy_t·τ/365 with the index's
  published dividend yield dy_t.
* **Trips with a split, bonus, consolidation, rights, demerger or arrangement ex-date in (t, T] are excluded** (the
  contract is adjusted; announced in advance).
* **Liquidity**: futures traded value ≥ ₹10 crore on the signal day *and* contracts > 0 on the execution day.
* **Costs (retail, per side, rates in force on each date)** from `execution/costs.py`:
  * Cash delivery: STT 0.1 % buy and sell, stamp 0.015 % on buys, exchange and SEBI fees, GST, DP ₹15.34 per sell;
    half-spread 3 bp (stocks), 4 bp (ETFs; ETF STT 0.001 % on sells only).
  * Futures: STT on sells 0.01 % (to 2023-03-31), 0.0125 % (to 2024-09-30), 0.02 % (to 2026-03-31), 0.05 %
    (from 2026-04-01); stamp 0.002 % on buys; exchange fee; brokerage ₹20 per order + GST; half-spread 3 bp (stock
    futures), 1 bp (index futures).
  * **2× costs** doubles every item.
* **Capital** per trip = notional of the cash leg × (1 + m), futures margin m = 0.20 (pledging the stock as margin
  would lower this; reported descriptively with m = 0).
* **Excess return** = strategy return on capital − r_t × days/365 on the same capital. Idle capital earns r_t.
* **Tax (India, individual, highest slab incl. cess)** per financial year: futures P&L is non-speculative business
  income at 31.2 %; the cash leg is short-term capital gain at 20.8 %. A business loss is set off against capital
  gains of the same year; a capital loss is not set off against business income; no carry-forward (conservative).
  The cash alternative is taxed at 31.2 %. **After-tax excess** = after-tax strategy − after-tax cash.

## 2. Hypotheses

| ID | Strategy | economic_mechanism | Who pays us | Why it persists |
|---|---|---|---|---|
| **HB1** | Monthly stock cash-and-carry book | financing / arbitrage-capital constraint: futures longs pay carry above the cash rate to obtain leverage | leveraged futures longs (they avoid the cash outlay) | arbitrage needs capital, STT on both cash legs, physical settlement and operational effort |
| **HB2** | One-day stock basis dislocations | temporary demand shock for leverage / limits to arbitrage around events | the same longs, at moments of unusual demand | capital is finite on the day; dislocations need to persist one session to be captured |
| **HB3-N / HB3-BN** | NIFTY / BANKNIFTY carry via ETF | index financing demand (FPI and retail futures longs) | index futures longs | ETF tracking noise and capital lock-up deter arbitrage |
| **HB4** | Roll-window calendar spread | **forced flow**: the monthly roll by every holder who must keep exposure | rollers who must transact in a fixed window | the window is fixed; rolling cannot be deferred past expiry |

### HB1 — monthly stock cash-and-carry book (primary)
* On d0, the first session after each monthly stock-futures expiry, compute for every eligible stock the net
  annualised excess carry e = (g − C_rt) × 365/τ − r_t, where C_rt is the round-trip cost of both legs.
* Select the **top 10 by e among e > 0**, equal notional; execute at d1's closes; hold to expiry; idle capital at r_t.
* Unit: monthly cycles. **HB1: mean monthly excess return > 0** (Newey–West, 3 lags; one-sided).

### HB2 — one-day dislocations (stocks)
* Any session t with τ ≥ 3: a stock whose e_t ≥ 10 %/yr. Enter at t+1's closes, hold to expiry; at most one open
  trip per stock.
* **HB2: mean per-trip excess return > 0**, aggregated to calendar months (Newey–West, 3 lags; one-sided).
* Reported: signal decay from t to t+1 (the share of the signalled carry still present at execution).

### HB3 — index carry via ETF
* As HB1, but one instrument each: NIFTY futures vs NIFTYBEES, BANKNIFTY futures vs BANKBEES; in position for a
  cycle if e > 0 at d0, else cash. Realised P&L = ETF total return − futures return over [d1, T].
* **HB3-N, HB3-BN: mean monthly excess return > 0** (Newey–West, 3 lags; one-sided).

### HB4 — roll-window forced flow (measurement; two-sided)
* Calendar-spread carry cs_t = ln(F_next/F_near) × 365/(T_next − T_near) for NIFTY, BANKNIFTY and the cross-sectional
  median over liquid stock futures.
* Per expiry cycle: mean cs over the last 3 sessions before the near expiry minus mean cs over sessions 4–10 before it.
* **HB4: the difference ≠ 0**, unit = expiry cycles, computed for NIFTY (primary series); BANKNIFTY and the stock
  median are reported. Two-sided test (holders who are long futures roll by selling near/buying next, while
  arbitrage funds that are short roll the other way; the sign is not presumed).
* Economic follow-on (descriptive): rolling HB1/HB3 positions 3 sessions early vs at expiry.

**Holm** across HB1, HB2, HB3-N, HB3-BN, HB4 (5 tests). Significance 5 %.

## 3. Gate (HB1, HB2, HB3; in order)

1. **Statistical**: Holm-significant, positive.
2. **Economic**: annualised excess over cash, after costs, ≥ **1.5 %/yr** on capital.
3. **After tax**: after-tax excess ≥ **1.0 %/yr**.
4. **2× costs**: pre-tax excess ≥ **0.75 %/yr**.
5. **Market neutrality**: |β| of monthly returns to NIFTY < 0.10.
6. **Capacity**: at a ₹1 crore book, every trade ≤ 1 % of its futures contract's traded value on the execution day
   (larger trades are cut to that limit, and steps 2–4 are recomputed).
7. **Stability**: excess > 0 in ≥ 2 of 3 subperiods (2015–18, 2019–22, 2023–26).
8. **Forward paper test** (not in this version).

HB4 is a measurement; it gates nothing.

## 4. Descriptive (never gating)

* Reverse arbitrage (futures cheaper than fair) frequency and size — needs stock borrowing (SLB), not retail-feasible.
* m = 0 (stock pledged as margin); top-5 / top-20 books; thresholds 5 % / 15 % for HB2.
* Which sectors and events carry the richest basis (dividend season, results, index events).

## 5. Architecture in this version (not hypotheses)

* AEM → **Opportunity Pricing Engine**: EV = P(fill)·(E[R | fill] − costs) − P(no fill)·opportunity cost, per unit
  of capital and time; a **capital allocator** ranking opportunities by net value per capital-day subject to capacity.
* `economic_mechanism` becomes a required field of every v1.1+ experiment row in `research/trials.jsonl`.

## 6. Unchanged

* REV1 stays frozen.
* No live trading, no broker connection, no data purchase.
* There is no NSE microstructure study until the depth recorder has ≥ 20 days and its own pre-registration.

## Amendment 1 — corrections after the independent review (2026-10-04; development results had been seen)

No hypothesis, threshold, unit of test or gate step was changed. The changes below fix look-ahead, data mechanics and
cost/tax realism. **Both versions of the HB1/HB2 signal are reported**, so the effect of change 1 is fully visible.

1. **Dividends removed from the signal (look-ahead).** The pre-registered g = (F − S + D)/S used dividends whose
   ex-date falls in (t, T]. The corporate-action file has ex-dates, not announcement dates, and the review showed the
   edge it produced comes from interim dividends that were very likely unannounced at t (excess grows with the
   signal-to-ex-date gap; trips with no dividend lose money). The primary signal is now g = (F − S)/S; realised P&L
   still includes dividends actually received. The pre-registered signal is reported as `as_preregistered_signal`
   with its own gate. A test of "futures underprice *announced* dividends" needs announcement timestamps and its own
   pre-registration.
2. **Extraordinary dividends.** When NSE adjusts the futures contract for a large dividend, the futures close drops by
   D on the ex-date and the short position does not earn it. Such events (D ≥ 2 % of the price and a basis jump on the
   ex-date below 0.5·D; the ratio is bimodal) re-base the futures leg by D.
3. **Expiry calendar.** Contracts that NSE re-labelled (June 2023 holiday shift; the 2025 change of expiry weekday)
   are keyed by contract month and take their actual last trading day. HB1 cycles follow the common monthly calendar
   (expiries shared by ≥ 50 stocks); cycle start dates are unique.
4. **Costs.** Futures settle at the final settlement price: no exit half-spread. From 2019-10-01 stock futures are
   physically settled, so the stock is delivered at that price: no cash exit half-spread (delivery STT 0.1 % still
   charged). Half-spreads are floored at half a ₹0.05 tick. SEBI fee corrected to 0.01 bp.
5. **Tax.** STCG is 15.6 % (incl. cess) until 22 Jul 2024 and 20.8 % from 23 Jul 2024. Reported variants: losses
   carried forward; both legs as business income with carry-forward (how most arbitrageurs file). The primary stays
   the conservative pre-registered rule.
6. **Annualisation on calendar time**: idle days between cycles count with zero excess.
7. **HB3**: selection always at 1× costs; the 2× cost case doubles realised costs only. ETF returns are compounded on
   the ETF's own sessions (ETFs are not in returns.parquet; split days detected).
8. **Gate mechanics**: capacity (step 6) and β (step 5) are computed for HB2 and HB3 as well; HB2's decay metric is
   locked carry at execution ÷ carry at signal. HB4's stock median uses liquid contracts (both legs ≥ ₹10 crore).
9. **Descriptive additions**: selection on capital (incl. margin), a liquid-fund-rate sensitivity for May 2020 –
   Apr 2022 (reverse repo 3.35 %), index replication with futures + cash, and reverse-arbitrage frequency.
