# AuTrad v1.2 — pre-registration: Forced Flow & Events

Written 2026-10-04 and hashed into `research/trials.jsonl` **before any v1.2 data was downloaded or any v1.2 result
was computed**.

**Feasibility probes made before hashing** (single samples only, no statistics):
* NSE corporate announcements carry exchange timestamps and dividend text. Examples:
  * INFY 12-Oct-2023 16:04:50 "declared Interim Dividend of Rs. 18";
  * IDFC 01-Feb-2023 17:12:59 "declared Interim Dividend of 11". That event confirms the v1.1 look-ahead: the
    dividend was announced *after* the v1.1 signal date of 27-Jan-2023.
* NSE publishes daily F&O ban lists and market-wide position limit (MWPL) / open-interest files back to 2016.
* NSE Indices' press-release page lists dated releases back to 2000, with PDF links.

> Rule for this version: before testing an event we must name **who pays**, **why they must trade**, **why they
> cannot arbitrage us away**, **what prevents instantaneous convergence**, **what AuTrad's advantage is** and **what
> would kill the mechanism**. Every experiment row records `economic_mechanism`, `economic_counterparty`,
> `mechanism_constraint`, `our_advantage` and `failure_condition`.

## 0. Data (downloaded and audited after hashing)

### Dividend announcements (point in time)
* **Source:** NSE corporate announcements, for every stock that had single-stock futures, for the years it had them
  (2015-01 … 2026-09).
* **Kept:** rows whose subject or text mentions a dividend, with the exchange dissemination time (`exchdisstime`, else
  `an_dt`), subject and text.
* **Parsed from the text:**
  * type (interim / final / special);
  * action (declared / recommended / approved);
  * amount per share.
* **Linked to the NSE corporate-action dividend** (symbol, ex-date, amount): the **earliest** announcement in
  [ex-date − 120 days, ex-date − 1 day] whose parsed amount equals the corporate-action amount (±1 %). Events without
  an amount match are excluded from the primary tests and counted in the audit.
* **Four dates stored:**
  * `announcement_time` (IST, from the exchange);
  * `record_date` (corporate-action file);
  * `ex_date`;
  * `payment_date` (parsed from the text when present, otherwise null).

### Index changes
* **Source:** NSE Indices press releases 2015-01 … 2026-09 whose title announces changes, replacements, inclusions or
  exclusions in equity indices.
* **Parsed from the PDF:** index, included and excluded symbols, and the effective date.
* **Event dates:**
  * announcement date A = the release date, treated as after the close, so t0 = the next session;
  * effective date E;
  * rebalance close R = the last session before E.
* **Primary index:** NIFTY 50. **Secondary:** NIFTY NEXT 50.

### F&O ban / MWPL
* **Sources:** the daily `fo_secban` file (securities in ban for the trade date) and the daily `combineoi` file
  (MWPL, open interest, "No Fresh Positions" flag), 2016-01 … 2026-09.

### Already present
F&O futures and index closes (v1.1), cash prices and total returns, corporate actions, and the RBI repo rate.

### Audit (before any test)
* The share of corporate-action dividends linked to a timestamped announcement with a matching amount.
* The distribution of announcement-to-ex-date gaps.
* The share of announcements made after the close.
* Press-release parse coverage, checked against the semi-annual NIFTY 50 snapshots already in
  `marketdata/reference/nifty50_pit_snapshots.csv`.
* Ban-list coverage.

## 1. Common definitions

* **t0** = the first session whose close (15:30 IST) is after the announcement time.
* **Contract:** the earliest expiry T ≥ ex-date among the near and next stock futures, with τ ≤ 65 days.
* **Liquidity:** futures traded value ≥ ₹10 crore on t0 and contracts > 0 on the execution day.
* **Costs, capital and tax:** as in v1.1 after Amendment 1. That means:
  * dated STT; a half-spread floor of half a ₹0.05 tick;
  * no exit spread at settlement or delivery;
  * capital = 1.2 × notional for cash-and-carry; fully collateralised futures for index trades;
  * the cash rate is RBI repo;
  * tax: business income 31.2 %, STCG 15.6 % / 20.8 % by date; set-off rules as in v1.1;
  * reported variants: carry-forward, and both legs as business income.
* **Fair futures value with known dividends:** F\* = S·(1 + r·τ/365) − D, with D counted only if
  ex-date ∈ (t, T] **and** the dividend was announced before t's close.
* **Residual mispricing:** m = (F − F\*)/S.
* **Extraordinary dividends** (D/S ≥ 2 % at announcement) are excluded from the primary tests, because NSE may adjust
  the contract (a point-in-time rule). They are reported descriptively.
* **Corporate actions** that adjust contracts (split, bonus, rights, demerger, arrangement) inside the window are
  excluded.

## 2. Hypotheses — the six questions

### HD — Dividend-information incorporation (primary)
| Question | Answer |
|---|---|
| economic_mechanism | Futures should re-price by −D the moment a dividend becomes public; participants incorporate it at different speeds |
| economic_counterparty (who pays) | Futures longs and market makers quoting stale fair values (index/stock arbitrage desks that update dividend assumptions slowly) |
| why they must trade | Futures longs roll and hold leverage regardless; quoting desks must keep quoting |
| mechanism_constraint (why not instantly arbitraged) | Information processing (unstructured filings, often after the close), cash-and-carry capital, STT on both cash legs |
| our_advantage | Timestamped parsing of every filing, automated fair value with known dividends, cost-aware entry (AEM) |
| failure_condition | Futures fully discount the dividend by the first close after the announcement (m indistinguishable from controls), or the residual is smaller than costs |

* **HD1 (incorporation, measurement):**
  * Test quantity: m_event(t0) − median m_control(t0) > 0.
  * Controls: liquid stocks on the same contract on t0, with no dividend announcement in the previous 45 days and no
    dividend ex-date in (t0, T].
  * Unit: event months (Newey–West, 3 lags). One-sided: + means futures still too expensive after the announcement.
* **HD2 (economic):**
  * Trade every eligible event with net excess on capital
    e_cap = ((F − S + D)/S − C_rt)/1.2 × 365/τ − r > 0, using t0 close prices.
  * Execution: **t0+1 close**. Hold to expiry; the dividend is received.
  * Unit: per-trip excess on capital, aggregated to calendar months (Newey–West, 3 lags; one-sided).
  * **Descriptive (speed value):** the same trades executed at the t0 close.

### HI — Index inclusion/exclusion forced flow
| Question | Answer |
|---|---|
| economic_mechanism | Passive trackers must buy inclusions and sell exclusions at the rebalance close; anticipation moves prices between announcement and rebalance |
| economic_counterparty | Index funds and ETFs (and benchmarked active funds), who pay impact to minimise tracking error |
| why they must trade | Mandate: replicate the index at the effective date |
| mechanism_constraint | Trackers cannot trade early without tracking error; liquidity providers need capital and bear event risk |
| our_advantage | Systematic parsing of every press release, execution at t0 close, futures for both directions, cost/tax-aware sizing |
| failure_condition | The whole move happens before t0 close (no drift to R), or front-running is so crowded that R-close impact is zero or reverses early |

* **HI1 (anticipation drift):**
  * Test quantity: sign-adjusted, market-adjusted return (stock − NIFTY 50) from t0 close to the R close > 0.
    Inclusions count +, exclusions −.
  * Unit: announcement dates (the mean over that date's events), Newey–West 3 lags. One-sided.
  * NIFTY 50 is primary (HI1-N50); NIFTY NEXT 50 is secondary (HI1-NN50).
* **HI2 (post-rebalance reversal):**
  * Test quantity: sign-adjusted, market-adjusted return from the R close to R+10 sessions < 0.
  * Same unit. One-sided. Reported as HI2-N50 and HI2-NN50.
* **HI-econ:**
  * Trade the HI1 window with stock futures, hedged with NIFTY futures: long inclusions, short exclusions.
  * Enter at the t0 close, exit at the R close. Capital = notional.
  * Requirements: F&O-eligible stocks only; a trade must be ≤ 1 % of the contract's traded value.

### HF — F&O ban (measurement only: is anyone actually forced?)
| Question | Answer |
|---|---|
| economic_mechanism | In a ban, no new derivative positions may be opened, only reduced: arbitrage capital cannot enter, and expiring positions cannot be rolled |
| economic_counterparty | Holders of expiring positions in banned stocks (they must close, not roll); anyone needing to hedge |
| why they must trade | Exchange rule; rolling counts as a fresh position |
| mechanism_constraint | Regulatory: new arbitrage positions are prohibited during the ban |
| our_advantage | None for a retail trader during the ban (we cannot open either). Any trade must come after the ban |
| failure_condition | No basis dislocation during bans; no abnormal open-interest unwinding at ban-at-expiry |

* **HF1 (limits to arbitrage):** |m| during ban sessions minus |m| on the same stock's non-ban sessions within ±60
  sessions > 0. Unit: ban episodes. One-sided.
* **HF2 (forced unwinding):** when a stock is in ban during any of the last 3 sessions before expiry, the total
  open-interest change over those sessions, relative to the same stock's non-ban expiries, is < 0. Unit: such
  episodes. One-sided. The market-adjusted return over the same window is reported, two-sided.
* HF gates nothing.
  * If HF1 is confirmed **and** the dislocation persists for ≥ 1 session after the ban exits, a post-ban
    cash-and-carry hypothesis goes to v1.3 with its own pre-registration.
  * If not, ban periods become a **risk rule**: no new F&O exposure in names approaching 95 % of MWPL.

**Holm** across HD1, HD2, HI1-N50, HI2-N50, HI1-NN50, HI2-NN50, HF1, HF2 (8 tests); significance 5 %.

## 3. Gate (HD2 and HI-econ; in order)

1. **Statistical:** HD2 itself for HD; HI1-N50 for HI-econ (NIFTY 50 trades). Holm-significant, positive.
2. **Economic:** annualised excess over cash on deployed capital ≥ 1.5 %/yr, **and** mean net per trip > 0.
3. **After tax:** after-tax excess ≥ 1.0 %/yr (pre-registered rule; variants reported).
4. **2× costs:** excess ≥ 0.75 %/yr.
5. **Market neutrality:** |β| < 0.10 for HD2; HI-econ is hedged with NIFTY futures, and its β is reported.
6. **Capacity:** ₹1 crore book; each trade ≤ 1 % of the contract's traded value on the execution day.
7. **Stability:** excess > 0 in ≥ 2 of 3 subperiods (2015–18, 2019–22, 2023–26).
8. **Forward paper test.** Passing step 7 makes the strategy a forward paper candidate. It authorises no trading.

## 4. Architecture in this version (not hypotheses)

* **Registry:** from v1.2, experiment rows require `economic_counterparty`, `mechanism_constraint`, `our_advantage`
  and `failure_condition` in addition to `economic_mechanism`.
* **Capital allocator objective:** (E[P&L] − after-tax cash alternative)/(capital × time). Constraints:
  * capacity and liquidity;
  * a drawdown budget (a 2σ loss per position ≤ the budget);
  * margin;
  * concentration (per position and per risk group);
  * execution probability.
* **Execution rule from v1.1 HB4:** stock-futures rolls inside the last 3 sessions cost about 2.8 bp more. The
  execution layer rolls before that window unless forced. This is an AEM parameter, not a strategy.

## 5. Unchanged

* REV1 stays frozen.
* No live trading, no broker connection, no data purchase.
* There is no NSE microstructure study until the depth recorder has ≥ 20 sealed days.

## Amendment 1 — corrections after the independent review (2026-10-04; development results had been seen)

No hypothesis, threshold, unit or gate was changed. Changes are bug fixes, timing corrections, a sample-definition
clarification and robustness statistics, which are reported next to the pre-registered statistics.

1. **t0 rule.** NSE's closing price is the weighted average of 15:00–15:30, so a filing at or after 15:00 IST is
   assigned to the next session. The pre-registered rule used 15:30.
2. **Dividend linkage.** The search window starts after the stock's previous ex-date, so a repeated amount cannot
   link to the previous dividend's announcement.
3. **Calendar.** Sessions are the union of F&O and cash sessions; NSE has no F&O file for 2021-03-30.
4. **HD2.** No new short future is opened on a day the stock is in an F&O ban (3 trades).
5. **HD2 "fast" descriptive.** Filling at the same close that is used for selection profits from noise in that
   print. It is now reported against same-close placebos (selection and fill at t0+1, t0+3, t0+5). The difference
   between t0 and the placebos is the speed value.
6. **HI sample.**
   * A release superseded by a later release for the same stock/index/action before the effective date is dropped
     (YESBANK/SHREECEM, Feb → Mar 2020).
   * NEXT 50 events that mirror a NIFTY 50 change on the same release are NIFTY 50 events, and are dropped from the
     NEXT 50 sample. This was decided before the first run and is recorded here.
   * Sensitivity: announcements with an exclusion caused by a corporate event (no price 10 sessions after R, e.g. the
     HDFC merger) are removed.
   * Futures half-spreads are floored at half a tick.
   * The β of the hedged trade is estimated rather than assumed to be 0.
7. **HF inference (reported next to the pre-registered statistic):**
   * HF1: month-clustered t, the median, the mean without the 20 largest episodes, the signed residual, and a test of
     next-session persistence.
   * HF2: expiry-clustered t, and controls matched on the OI/MWPL level at T−4 (bins 0.75–0.85, 0.85–0.95, ≥ 0.95).
   * The HF "v1.3 trigger" requires HF1 to be robust *and* persistence to be significant; the robustness statistics
     decide whether it is met.
8. **Execution rule** `fo_ban_no_new_exposure` restated: a futures discount concentrated in distressed names, not a
   generic +14 bp.
