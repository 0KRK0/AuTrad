# AuTrad v1.0 — pre-registration: Monetisation Lab (M1 execution alpha, M2 market-making economics, M3 funding/basis carry) and the Alpha Economics Model

Written 2026-10-04 and hashed into `research/trials.jsonl` **before any v1.0 result was computed**:
* M1 and M2 reuse the v0.9 M0 grids. Those data were used in v0.9 for *different* hypotheses, and M0's training and
  test split is kept unchanged.
* M3's data (funding rates, perpetual daily bars) had not been downloaded when this file was hashed.

## 0. Thesis

v0.9 showed that *prediction is not monetisation*: order flow predicts the next seconds (IC 0.34), but the edge is
below 1 bp while costs are 3–10 bp. v1.0 therefore tests three ways of getting paid that do not require a large
directional edge:
1. **cheaper execution** of trades AuTrad would make anyway (M1);
2. **liquidity provision** (M2);
3. a **structural carry** that does not depend on forecasting price (M3).

No live trading, no broker, no data purchase.

## 1. AEM — Alpha Economics Model (architecture; not a hypothesis)

Every signal must pass `lab/aem.py` before the Risk Gate:

    E[net edge] = E[gross return over the horizon]
                - spread paid           (crossing: half-spread ×2 for the round trip; posting: 0)
                - fees (+ rebates)      (venue profile: NSE delivery / NSE futures / crypto retail / crypto best case)
                - E[adverse selection | passive fill]
                - E[market impact]      (square-root model, participation)
                - latency drift          (expected signal decay × latency)
                - tax drag               (India equity STCG/LTCG; crypto 31.2 % per transfer, no set-off)
                - opportunity cost       (cash rate × horizon)

    decision = EXECUTE if E[net] > 0 and E[net] / σ(net) ≥ the venue's minimum;
               WAIT    if E[net] ≤ 0 now but a cheaper route (passive) makes it positive;
               ABANDON otherwise.

Its cost parameters come only from measured sources:
* `execution/costs.py`;
* the NSE futures cost function;
* M0/M1 measurements of fill probability and adverse selection.

The AEM is tested by unit tests and is not part of any study's statistics.

## 2. M1 — execution alpha (Binance BTCUSDT perpetual, M0 grids)

* **Parent orders (exogenous, no alpha needed).** On each M0 test day, at every 5-minute mark, one BUY and one SELL
  parent order of 0.01 BTC, each evaluated independently. Each must be completed within **60 s**. Latency 250 ms
  (rounded up to 300 ms on the 100 ms grid). The arrival price is the mid at decision time.
* **Policies.**
  * **P0 cross now:** buy at the ask (sell at the bid) after the latency; taker fee.
  * **P1 passive then cross:** post at the near touch after the latency, behind the displayed size, using the same
    queue fill rule as M0's E2. If unfilled at 60 s, cross at the far touch; taker fee then, maker fee if filled
    passively.
  * **P2 AEM-timed:** as P1, but crossing immediately at the far touch whenever the M0 model (fitted on M0 training
    days) predicts an adverse 5 s move of at least θ = 0.5 bp against our side. This is checked at decision time
    and every 1 s while unfilled.
* **Implementation shortfall (IS, bp)** = side × ln(execution price / arrival mid) × 10⁴ + fee.
* **Hypotheses** (unit: test-day hours; Newey–West, 3 lags; Holm across 2; sign +):
  * HM1a: IS(P0) − IS(P2) > 0. The AEM-timed policy is cheaper than crossing.
  * HM1b: IS(P1) − IS(P2) > 0. The signal adds value beyond plain passive posting.
* **Fees:** retail is primary (maker 2.0 bp, taker 5.0 bp); best case (maker 0, taker 1.7 bp) is descriptive.
* **Execution-improvement gate** (this is cost reduction, not alpha):
  * HM1a and HM1b both significant after Holm;
  * mean saving of P2 against the better of P0 and P1 ≥ 0.5 bp at retail fees;
  * positive on ≥ 5 of 7 test days.
  * Passing makes P2 the AEM's default execution cost for crypto *research*; it does not authorise any trading.

## 3. M2 — market-making economics (M0 grids, test days)

* **Policy.**
  * Quote 0.01 BTC on both sides at the touch, behind the displayed size.
  * Re-join at the back of the queue whenever the touch moves away.
  * Inventory is limited to ±1 unit: when long, quote only the ask, and vice versa.
  * Inventory older than 60 s is flattened at the far touch (taker).
  * Latency 300 ms on every placement.
* **Skew variant:** withhold the bid when the M0 model predicts ≤ −θ, and withhold the ask when it predicts ≥ +θ,
  where θ is the M0 training-day decile (0.33 bp).
* **Measurement.**
  * Gross P&L, excluding maker fees and including taker fees on flattening, per passive fill, in bp of notional.
    Inventory is marked to mid.
  * **Break-even maker fee** = gross P&L per passive fill. A positive value is the fee the strategy could pay; a
    negative value is the rebate it would need.
* **Hypotheses** (test-day hours; Newey–West, 3 lags; Holm across 2; sign +):
  * HM2a: gross P&L per passive fill (skew variant) > 0.
  * HM2b: skew variant − plain variant > 0.
* **Reading.**
  * If the break-even maker fee is < 0, market making needs rebates.
  * If it is < −0.5 bp, it is not viable even with typical published market-maker rebates. This threshold is
    approximate; rebates depend on the programme and on volume.

## 4. M3 — funding/basis carry (BTC and ETH; Binance; 2020-01 to 2026-09)

* **Data:**
  * USDⓈ-M perpetual funding rates (8-hourly) and perpetual daily bars, from the public archive (monthly files);
  * spot daily bars from AuTrad's crypto store.
* **Position:** long spot N plus short perpetual N; capital 2N, fully collateralised and unlevered.
  * Daily P&L = N·(r_spot − r_perp) + N·Σ(that day's funding rates); a positive rate is received by the short side.
  * Return is measured on capital 2N.
* **Rule:**
  * Decided each day at 00:00 UTC on data up to the previous day.
  * **Enter** when the trailing 7-day mean funding rate, annualised (×3×365), is ≥ 15 %.
  * **Exit** when it is < 5 %.
  * Out of position, capital earns 0 (USDT).
* **Costs** (retail, per entry or exit): spot 10 bp + perpetual 5 bp + slippage 2 bp per leg, i.e. 34 bp per round
  trip on N. **2× costs** doubles all of these.
* **Indian tax (conservative reading; the treatment of offshore perpetuals is unsettled):**
  * at each exit, 31.2 % on a positive spot-leg gain and on a positive perpetual-leg P&L including funding;
  * losses of either leg are never set off;
  * at the end of the window, open positions are closed and taxed.
* **Hypotheses** (daily returns; Newey–West, 10 lags; Holm across BTC and ETH; sign +): HC1-BTC and HC1-ETH,
  mean daily return on capital > 0.
* **Gate** (in order):
  1. statistical;
  2. net return after costs ≥ 8 %/yr on capital (an INR cash proxy of 6 % plus 2 %);
  3. after-tax net ≥ 6 %/yr;
  4. 2× costs ≥ 6 %/yr;
  5. capacity: ₹1 Cr, which is negligible against BTC/ETH perpetual depth, so this step is not binding;
  6. stability: net > 0 in ≥ 2 of 3 subperiods (2020–21, 2022–23, 2024–26);
  7. forward paper test.
* **Risks reported, not gated:** exchange counterparty risk, liquidation distance of the short perpetual, funding
  turning negative, and INR/USD.

## 5. Unchanged

* REV1 stays frozen.
* The agents stay advisory.
* Live trading stays disabled.
* There is no NSE study until the depth recorder has ≥ 20 days and its own pre-registration.

## Amendment 1 — corrections after the independent review (2026-10-04; development results had been seen)

No hypothesis, threshold, policy, cost or gate was changed. The changes are a bug fix, written-down interpretations,
an AEM calibration correction, and descriptive analyses that are labelled as such and never gate anything.

1. **M1 Newey–West block order (bug fix).** Hour blocks were sorted as text ("_10" before "_2"); they are now in time
   order. Effect: HM1a t 35.0 → 37.2, HM1b t −2.89 → −2.85; no verdict changes.
2. **Interpretations, recorded as implemented:**
   * M2 re-joins the queue on *any* change of the touch price, not only when the touch moves away (slightly
     pessimistic); the skew thresholds are the two training-day deciles (−0.329 / +0.335 bp), i.e. "0.33 bp".
   * M2 hour blocks with no passive fill are dropped from the HM2 tests; their inventory P&L stays in the pooled
     break-even figure.
   * M3 gate step 4 ("2× costs ≥ 6 %/yr") is applied after tax (stricter); the pre-tax figure is also reported.
3. **AEM calibration (architecture, not a study).** The passive adverse-selection input is now the measured mid move
   *against* us 5 s after a passive fill (M2 plain quotes, +0.90 bp), not the fill price versus the arrival mid
   (−0.02 bp, kept as `passive_price_vs_arrival_bp`). The passive route is a passive entry plus a crossing exit. Under
   s.115BBH fees are not deductible from the crypto tax base. The calibration file is read from an absolute path.
4. **Descriptive additions (not gating):**
   * M2 queue bounds: queue ahead of us = 0.5× and 0× the displayed size (the pre-registered rule never credits
     cancellations ahead of us, so it is a lower bound).
   * M3: fixed-quantity legs (no daily rebalancing), the cost of the daily rebalancing the constant-notional model
     implies, a within-trip tax set-off reading, an always-on carry, and the 2× cost pre-tax figure.
   * M1: θ ∈ {0.25, 1.0, 2.0} and P0 − P1.
