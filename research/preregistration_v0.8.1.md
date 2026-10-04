# AuTrad v0.8.1 — pre-registration: wider universe (Study 2 Amendment 2) and crypto studies C1/C2

Written 2026-10-03 and hashed into `research/trials.jsonl` **before any strategy, event or portfolio return** was
computed on the v0.8.1 data. Two checks were run on returns before this file, and neither tests a hypothesis:

* A data validation of the whole-market NSE price series against the vendor-corrected NIFTY 50 panel: daily stock
  returns, 99.26 % within 10 bp.
* A listing of extreme daily moves among TOP500 members, to check the price adjustment.

## A. Data built in v0.8.1 (audited before this file)

* **Whole-market NSE bhavcopy, Jan 2015 – Oct 2026.** 2,911 sessions, including Budget-day and Muhurat specials.
  3,319 securities (ISIN-linked), 5.1 M stock-days.
* **Corporate actions** for every company from NSE's corporate-actions API:
  * splits, bonuses, consolidations and rights (theoretical ex-rights factor) are priced;
  * dividends and returns of capital are added as cash, so returns are **total returns**;
  * demergers, schemes and capital reductions cannot be priced: that day's return is set to 0, assuming the holder's
    value is conserved, and flagged;
  * moves with no corporate-action record are treated as an unrecorded split only when the price ratio is a standard
    one (within 7 %). Otherwise they are kept as genuine and flagged (e.g. YES BANK 6 Mar 2020, DHFL, INFIBEAM).
* **Point-in-time liquidity universes TOP100 / TOP200 / TOP500** (`marketdata/nse_universe.py`):
  * at each month's last session, companies (ISIN prefix INE) are ranked by median daily traded value over the
    previous 126 sessions;
  * a company must have traded on at least 80 % of those sessions and been listed for at least 126 sessions;
  * 135 month-ends, Jul 2015 – Sep 2026; 1,272 distinct TOP500 members, 133 of them later delisted;
  * these are proxies for the NIFTY 100/200/500, which use free-float market cap; their member histories are not
    used.
* **BSE shareholding (promoter table) for the TOP500 members** (`marketdata/reference/bse_codes_wide.csv`):
  * 1,257 of the 1,272 members are mapped to a BSE code (1,255 by ISIN, 2 by symbol);
  * 15 are unmapped, mostly NSE-only listings such as BSE Ltd and CDSL;
  * the same version rules as v0.8 apply (as-filed XBRL whenever the structured record is a later version);
  * its own coverage audit runs before Study 2.
* **Binance spot daily bars** for every USDT pair in the public archive, listed and delisted (663 research pairs
  after removing stablecoins, fiat, gold tokens and leveraged tokens).

## B. Study 2 — Amendment 2 (universe, prices, implementation)

The hypotheses H2a and H2b, the H2a event definition (Amendment 1), Holm across H2a and H2b, and the power rule
(≥ 100 events across ≥ 30 companies) are **unchanged**. What changes:

1. **Universe.**
   * Primary: TOP500 at the previous month-end, window May 2016 – Sep 2026.
   * TOP200 is reported as a descriptive robustness check, outside the Holm family.
2. **Information timing.** A shareholding filing is usable at decision time t only if its `available_at` is before
   15:30 IST on t. Events take effect from the next session's open after `available_at`.
3. **H2a.**
   * Cumulative abnormal return over 63 sessions from the next open after `available_at`, against the equal-weight
     TOP500 (membership as of the last month-end).
   * Test: the mean CAR is computed per calendar month of event; t-test across months (two-sided p); the
     hypothesised sign is negative.
4. **H2b.**
   * Monthly Fama–MacBeth: next-month total return (from the next open) on an indicator for encumbrance ≥ 25 % of
     promoter holding (latest quarterly filing public at t), controlling for log median traded value.
   * Newey–West t with 6 lags; the hypothesised sign is negative.
5. **Implementation for the capital-efficiency gate.**
   * Long-only: equal-weight TOP500 **excluding** names with encumbrance ≥ 25 % or an H2a event in the previous
     63 sessions. Rebalanced monthly at the next open.
   * Benchmark: equal-weight TOP500, rebalanced the same way and **paying the same costs**, so that only the
     signal's effect is measured.
   * Costs (`execution/costs.py`): charges for equity delivery; half-spread 2.5 bp for ranks 1–100, 6 bp for
     101–250 and 12 bp for 251–500; square-root impact at the stated capital.
   * Tax: the v0.6 after-tax estimate.
   * Capacity at ₹1 Cr. Subperiods: the v0.5 set.
   * The v0.7 gate is applied unchanged and in order.

## C. Crypto studies (new family, India-investor view)

**Universe.** Binance USDT spot pairs: real coins only, including delisted ones. At each weekly decision (UTC close
of Sunday), coins with at least 60 days of history are ranked by median daily quote volume over the previous
30 days.

**C1 — time-series trend (BTC, ETH; two tests).**
* Rule: each week hold the coin if its Sunday close is above its 140-day simple moving average; otherwise hold cash
  (USDT, earning 0 %).
* Benchmark: buy-and-hold of the same coin.
* Test: Newey–West t (4 lags) of weekly (strategy − benchmark) returns; the hypothesised sign is positive.

**C2 — cross-sectional momentum (TOP50; one test).**
* Rule: each week hold equal weights in the 5 coins (top decile) with the highest prior 21-day return.
* Benchmark: equal-weight TOP50 (same costs).
* Test: Newey–West t (4 lags) of weekly (strategy − benchmark) returns; the hypothesised sign is positive.

**Common rules.**
* Holm across C1-BTC, C1-ETH and C2. Window: weekly decisions from 2018-01-07 to 2026-09-27.
* Subperiods: 2018–19, 2020–21, 2022–23, 2024–26.
* **Costs:** Indian exchange profile `crypto_inr`, 59 bp fees per side including GST plus a 10 bp half-spread, with
  square-root impact against daily quote volume at the stated capital (₹84 per USD for sizing).
* **Indian crypto tax (VDA, s.115BBH):**
  * every realised gain is taxed at 31.2 % (30 % + 4 % cess);
  * **losses are never set off**, not even against other crypto gains;
  * buy-and-hold is taxed once, at liquidation at the end of the window;
  * the 1 % TDS on each sale is treated as recoverable at year end, and also shown as a non-recoverable 1 % cost.
* Returns are measured in USD (USDT). INR/USD moves affect strategy and benchmark alike and are not modelled.
* **Gate:** the v0.7 capital-efficiency gate, in order, with "after-tax" meaning the Indian crypto tax above.
  Capacity is ₹1 Cr with participation at most 1 % of daily quote volume. A pass earns a frozen forward test, not
  capital.
* **Known contamination:** 140-day trend rules and short-horizon crypto momentum are widely published, so a pass is
  weaker evidence than for a new idea. The forward test is the real arbiter.

## D. Unchanged

* REV1 stays frozen.
* The agents stay advisory (SHADOW).
* Live trading stays disabled.
* No result in v0.8.1 can change a portfolio; a pass only starts a frozen forward paper test.

---

## Amendment 1 — corrections after the independent review (2026-10-04; development results had been seen)

A first run (`reports/v0.8.1_dev`, kept as SUPERSEDED) rejected every candidate at the statistical step. An
independent reviewer then found 1 critical, 1 major and 14 minor issues; none changed a verdict. The fixes below
correct data handling or bring the code in line with the text above. **No hypothesis, threshold, window, benchmark
or test is changed**, and the final run is reported in full whatever it shows.

1. **Tax.** After-tax NAVs could go negative and turn later tax into a refund; this made C2's post-tax −82.55 %
   meaningless. Both tax modules now stop at zero and record a "wiped out" date (never a refund).
2. **Cost artefact.** Study 2's net excess partly reflects the exclusion book paying fewer fixed per-scrip charges at
   ₹10 lakh. The gross (zero-cost) excess and an all-small-cap-cost sensitivity are now reported. The pre-registered
   benchmark rule (same costs) is unchanged.
3. **Prices.**
   * The Sunday 1 Feb 2026 Budget session is added.
   * A corporate-action factor is skipped when the ex-date open shows that NSE had already adjusted prev_close.
   * Unpriced-action days keep the real intraday move.
   * Dividend subjects with typos are parsed, and duplicate listings are dropped.
   * ISINs are linked across suspensions by issuer code (within 60 sessions).
4. **BSE mapping.**
   * Where several BSE codes exist, the code whose filings cover the company's NSE trading life is chosen.
   * 9 members still have no BSE filings: mostly BSE "permitted-to-trade" codes (590xxx), whose shareholding is not
     filed under that code. They are treated as unknown, as in the original design.
5. **H2b uses the latest public version** of the newest quarter (as Amendment 1 of v0.8 specifies). The first run had
   used first versions.
6. **Crypto.**
   * The gold token, wrapped BTC/ETH and leveraged tokens are excluded.
   * A coin that resumes trading after a gap of 7 days or more at a price 20× away is split into two series, so no
     return is computed across a redenomination or ticker reuse.
7. **Disclosures (no code change).**
   * Cost tiers use each company's median rank over its history (bounded by the all-small-cap sensitivity).
   * Five pre-open filings enter one session late.
   * Delisted names are frozen at their last price.
   * India's 2022 crypto tax rules are applied to the whole window.
   * The 63-session H2a windows overlap across months; this makes the plain t-test *less* conservative, which works
     against a false rejection only.
