# AuTrad v0.9 — pre-registration: Market Microstructure & Execution Alpha Lab (study M0)

Written 2026-10-04 and hashed into `research/trials.jsonl` **before any microstructure data was downloaded or
processed**. Only file listings (which days exist) and one file's header and line count were inspected.

## 0. Why this branch, and what it is not

v0.3–v0.8.1 found no slow (daily/monthly) edge that survives Indian costs and tax. v0.9 asks a different question:
does order-flow / order-book information carry a short-horizon edge that survives realistic **fees, spread,
queue position, adverse selection and latency**? This is a research lab. It places no orders, connects no broker,
and changes no portfolio.

## 1. Data (the only legitimate free source with book-level history)

* **NSE:** full historical order data is licensed (NSE price list, Mar 2026): about ₹12.5 lakh a year for
  historical order and trade data, about ₹1.1 lakh a year per segment for trades only, and about ₹12 lakh a year for
  a live TBT feed. Free alternative: record forward from a broker's market-depth websocket. Not available to this
  study.
* **Used here:** Binance's public archive (data.binance.vision) for the **BTCUSDT USDⓈ-M perpetual**:
  * `bookTicker`: every change of the best bid/ask price and size, millisecond timestamps.
    2023-05-16 to 2024-03-30 is archived.
  * `trades`: every futures trade, with aggressor side.
  * **Spot** BTCUSDT `trades`: for the futures–spot basis and lead–lag.
* **Sample (fixed now):** list the days present for all three files. Draw **20 days** with
  `numpy.random.default_rng(2026).choice(days, 20, replace=False)` and sort them. The **first 13 are training days**
  (all fitting, thresholds and percentiles); the **last 7 are test days** (every reported statistic). Test data is
  never used for fitting.
* **Grid:** event data is aggregated to **100 ms** buckets by exchange transaction time. Order-flow imbalance (OFI) is
  computed at event level, then summed per bucket. Within a bucket the last book state is used. Latencies below
  100 ms are therefore **not** evaluated; they need event-level replay and HFT infrastructure.

## 2. Features (all known at the end of bucket t)

| Name | Definition |
|---|---|
| OBI | (bid size − ask size) / (bid size + ask size) at the best level |
| OFI_1s | Cont–Kukanov–Stoikov best-level order-flow imbalance, summed over the last 10 buckets, scaled by the training-day standard deviation |
| TI_1s | (aggressive buy volume − aggressive sell volume) / total, over the last 1 s |
| QD | Δlog(bid size) − Δlog(ask size) over the last 1 s (relative queue depletion) |
| LARGE | 1 if the bucket contains an aggressive trade ≥ the 99th percentile of training-day trade sizes; signed by side |
| SPREAD_W | 1 if the spread exceeds 1 tick (0.1 USDT) |
| BASIS_DEV | log(futures mid / spot last trade) minus its trailing 10-minute mean |
| FUT_LEAD | futures mid log-return over the last 1 s |

Target: the futures **mid** log-return from the end of bucket t to t + h, for h ∈ {1 s, 5 s, 30 s}. **5 s is
primary.**

## 3. Hypotheses (11 tests; Holm across all 11; two-sided p; the stated sign is required)

The unit of inference is the **test-day hour**: 7 × 24 = 168 blocks. Each statistic is computed per block; the test
is a Newey–West (3 lags) t-test of the block mean.

| ID | Hypothesis | Statistic per block | Sign |
|---|---|---|---|
| H1 | OBI predicts the next mid move | Spearman IC(OBI, r₅ₛ) | + |
| H2 | Order-flow imbalance predicts the next mid move | IC(OFI_1s, r₅ₛ) | + |
| H3 | Large aggressive trades continue | mean of side × r₅ₛ over LARGE buckets | + |
| H5 | Queue depletion predicts the move | IC(QD, r₅ₛ) | + |
| H6 | Wide spreads mean more adverse selection | mean \|r₅ₛ\| when SPREAD_W=1 minus when 0 | + |
| H8 | Trade signs are autocorrelated | lag-1 autocorrelation of aggressor signs (trade level) | + |
| H9 | Large orders precede volatility bursts | mean \|r₅ₛ\| after LARGE minus otherwise | + |
| H10 | The combined model beats the best single feature | out-of-sample R² of the training-day OLS model (all features) minus that of the best training-day single feature | + |
| H11 | Order-flow predictive power depends on the volatility regime | IC(OFI_1s) in high minus low realised-volatility hours (training-day terciles of trailing 1 h volatility; computed over hours) | ≠ 0 |
| H17 | Basis dislocations mean-revert | IC(BASIS_DEV, next 30 s change in basis) | − |
| H20 | Futures lead spot | IC(FUT_LEAD, next 1 s spot return) | + |

Not testable with this data, and not claimed: cancellation imbalance (H4) and deep-book effects (H7) need
L2/L3; open/close effects (H12/H13), corporate events (H14), index/ETF baskets (H15/H16/H18) and options (H19)
need NSE data.

## 4. Economic tests (the model is H10's OLS model, fitted on training days only)

* **E1 — taker.**
  * When the predicted 5 s return exceeds the round-trip cost (2 × taker fee + spread), buy (or, symmetrically,
    sell) at the far touch observed at t + L, where L is the latency.
  * Exit 5 s later at the far touch. One position at a time; size 0.01 BTC.
* **E2 — passive.**
  * When the prediction is in the training-day top (bottom) decile, post a buy at the best bid (a sell at the best
    ask) at t + L. The order joins **behind the displayed size**.
  * It fills only when aggressive volume at that price exceeds the queue ahead plus our size, or when the price
    trades through.
  * It is cancelled if unfilled after 5 s. After a fill, exit 5 s later as a taker. Adverse selection is reported as
    the mid move against us within 5 s of the fill.
* **Fees.**
  * Primary, retail: maker 2.0 bp, taker 5.0 bp (Binance VIP 0).
  * Descriptive, best case: maker 0 bp, taker 1.7 bp (approximate top tier). This shows what scale or infrastructure
    would buy; it is not a gate input.
* **Latency curve:** L ∈ {0, 100, 250, 500, 1000, 2000, 5000} ms.

## 5. Microstructure gate (strictly sequential; the first failure rejects)

1. **Statistical:** H2 or H10 significant after Holm, with the right sign.
2. **Economic:** E1 or E2 mean net P&L per trade > 0 at **retail fees** and **L = 250 ms**, with t > 2 across test
   hours.
3. **2× fees:** still > 0.
4. **Latency:** still > 0 at L = 500 ms (a realistic internet/broker path).
5. **Capacity:**
   * E2 at 0.1 BTC (deeper in the queue) still > 0;
   * E1 at 0.1 BTC, filled only up to the displayed size, still > 0.
6. **Stability:** net > 0 on at least 5 of the 7 test days.
7. **Forward:** a frozen paper test on the live feed.

**Jurisdiction caveat** (not a gate step): an Indian resident trading offshore crypto perpetuals raises FEMA/LRS,
tax and platform-registration questions. A pass here proves a *mechanism*. Using it on NSE needs NSE depth data
and its own study.

## 6. Unchanged

* REV1 stays frozen.
* The agents stay advisory.
* Live trading stays disabled.
* Analyst snapshots keep accumulating.
* The analyst-revision study is **not** pre-registered until its data passes an audit.

---

## Amendment 1 — corrections after the independent review (2026-10-04; development results had been seen)

A first run (`reports/v0.9_dev`, kept as SUPERSEDED) failed at the economic step. The independent review found one
critical data defect and several smaller issues. **No hypothesis, threshold, sample, split, fee or gate step
changes.** The fixes:

1. **Data (critical).** Binance's `bookTicker` files for 2023-12-29 and 2024-02-12 interleave two time streams (18.7 M
   backward steps on 2024-02-12). Event-level OFI and the last book state were therefore wrong on those days. Every
   file is now partitioned by hour and sorted by (transaction_time, update_id) before OFI is computed. All 20 days
   were reprocessed, and the audit now records the backward steps in each raw file.
2. **Latency is rounded up** to whole 100 ms buckets (250 ms was previously evaluated as 200 ms).
3. **Gate.**
   * The economic, 2×-fee, latency and stability steps consider E1 and E2 separately (either may pass).
   * The capacity step requires both, as written in §5.
4. **H11:** training-day volatility terciles use the previous hour's volatility, the same rule as for test hours.
5. **Descriptive only:**
   * H8 on *orders*: same-millisecond, same-side trade rows are collapsed, because one taker order filling several
     makers inflates the trade-level figure.
   * The reverse lead (spot → futures) for H20.
   * E2's fill rate is now pooled across days.
   * "Adverse selection" is renamed "mid drift after fill".
6. **NSE recorder.**
   * Kite's exchange timestamps are naive IST datetimes with 1 s resolution: they are now localised, and the grid uses
     the local receive time on a 1 s grid.
   * NSE index-futures costs are updated for STT at 0.05 % (from 1 April 2026) and a NIFTY lot of 65: about 5.9 bp per
     round trip.
