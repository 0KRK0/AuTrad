# AuTrad v1.3 — pre-registration: Information Advantage Lab

Written 2026-10-04 and hashed into `research/trials.jsonl` **before any v1.3 data was downloaded or any v1.3 result
was computed**.

**The question changes.** It is no longer "what market pattern predicts returns?" but "is there information AuTrad can
know earlier or interpret better than the market, while it can still trade after knowing it?" Every hypothesis answers
the six questions. The registry refuses rows without `economic_mechanism`, `economic_counterparty`,
`mechanism_constraint`, `our_advantage` and `failure_condition`.

**What is already known (and therefore not re-tested here):**
* v0.4: SUE (seasonal random walk) has no premium in NIFTY 50.
* v0.5 (NIFTY 50, 2018–2026): after a bad results reaction, stocks keep falling: −107 bp over 20 sessions (t −3.7). It
  is not usable long-only (the overlay fails at 2× costs). A short diagnostic in stock futures earned +75 bp per event
  after costs (t 2.4); that was not a proposal and not gated.
* Those NIFTY 50 events have been seen. **v1.3 tests the mechanism on a disjoint universe**: F&O stocks that were not
  NIFTY 50 members at the event date. That is the out-of-sample test the v0.5 finding never had.

**Feasibility facts recorded before hashing** (no statistics computed):
* NSE's results index gives exchange broadcast/dissemination times (`marketdata/nse_filings.filings_index`).
* The XBRL parser exists, with 2,749 filings for 87 symbols stored.
* The analyst consensus store holds exactly **one** snapshot day (2026-10-03, NIFTY 50 only), so no analyst-revision
  history exists.
* The NSE depth recorder has **zero** recorded days.

## Part A — historical test (primary): bad-news diffusion under short-sale constraints

### Six questions
| Question | Answer |
|---|---|
| economic_mechanism | Bad results news is incorporated slowly: pessimists cannot easily short Indian cash equities (stock borrowing is thin), long-only institutions sell over days to limit impact, and analyst downgrades follow with a lag |
| economic_counterparty | Early dip-buyers and slow sellers: long-only holders who trim over several sessions, and investors anchored to the pre-results price |
| mechanism_constraint | Short-sale constraints in cash; institutional participation limits; analyst and attention lag outside the most-covered names |
| our_advantage | Access to the short side through stock futures; systematic, timestamped processing of every results filing across about 150–200 less-covered F&O names; AEM-priced entry |
| failure_condition | No drift outside NIFTY 50; or futures already discount the drift (basis/backwardation); or costs, tax or capacity remove it; or it lives only in distressed names |

### Data (downloaded and audited after hashing)
* **Results announcements:** the NSE results index (legacy and integrated filings) for every stock with single-stock
  futures 2016–2026. Announcement time = the earliest `available_at` (max of broadcast and dissemination time) per
  (symbol, period_end).
* **XBRL** for those filings, used only for HM1b. Raw files are kept in local scratch; parsed facts go to the repository.
* **Prices:** cash total returns, NIFTY 50 index, F&O futures (v1.1); corporate actions and ban lists (v1.2).
* **Point-in-time NIFTY 50 membership:** semi-annual snapshots plus the NSE Indices press releases (v1.2).
* **Audit before tests:**
  * the share of quarters with a timestamped filing;
  * the share filed after the close;
  * coverage of XBRL net profit, current and year-ago;
  * the count of eligible events by year.

### Definitions
* **Universe:** stocks with stock futures on the reaction day that were **not** NIFTY 50 members that day (point in
  time). Sample: reaction days 2016-04-01 … 2026-08-31.
* **Reaction day r0:** the first session whose closing price is formed after the announcement. A filing before 15:00
  IST on a session day → that day; otherwise the next session.
* **Reaction:** the stock's total return on r0 minus the NIFTY 50 return on r0.
* **Bad:** the reaction is below the lower third of all *earlier* reactions in this universe (expanding, past-only).
  The burn-in is the first 300 reactions; they are classified but not tested.
* **Window:** signal at the r0 close; entry at the **r0+1 close**; exit at the r0+21 close (20 sessions).
* **CAR20:** stock total return minus NIFTY 50 return over (r0+1, r0+21].
* **Trade:**
  * short the stock future and go long NIFTY futures with equal notional;
  * each contract is the earliest expiry ≥ the exit date among the listed contracts;
  * costs as in v1.2 HI-econ: dated STT, fees, brokerage, half-spread 3 bp with a tick floor for stock futures, 1 bp
    for index futures;
  * capital = notional; the cash collateral earns repo, so excess over cash = the futures P&L.
* **Exclusions:**
  * corporate actions that adjust the contract (split, bonus, rights, demerger, arrangement, or an extraordinary
    dividend of D/S ≥ 2 %) inside the window;
  * the stock in an F&O ban on the entry day (a new short is not allowed);
  * futures traded value below ₹10 crore on the entry day.
* **Tax:** business income at 31.2 % per financial year with set-off. Variants: carry-forward.

### Hypotheses (Holm across the 3; significance 5 %)
* **HM1-drift:** mean CAR20 of bad events < 0. Unit: reaction dates (average per date), Newey–West with 3 lags.
  One-sided.
* **HM1-trade:** mean net P&L of the hedged short trade > 0. Same unit. One-sided.
* **HM1b — interpretation:** among bad events, CAR20 is more negative when the filing itself shows deterioration
  (XBRL net profit below the same quarter a year earlier, on the same statement basis) than when it does not.
  * Welch t-test on event values. One-sided.
  * Tested only if ≥ 70 % of bad events have both net-profit values; otherwise it is reported as not testable.

### Gate (HM1-trade; in order)
1. Statistical (Holm, positive).
2. Annualised excess on deployed capital ≥ 1.5 %/yr **and** mean net per trade > 0.
3. After tax ≥ 1.0 %/yr.
4. 2× costs ≥ 0.75 %/yr.
5. |β| of the hedged trade to the NIFTY return < 0.10.
6. Capacity: ₹1 crore spread over that day's events; each trade ≤ 1 % of the contract's traded value.
7. Stability: excess > 0 in ≥ 2 of 3 subperiods (2016–18, 2019–22, 2023–26).
8. **Also required:** the mean without the 10 largest-loss names (distress check) stays > 0.
9. Forward paper test.

## Part B — analyst revisions (forward only; sealed)

| Question | Answer |
|---|---|
| economic_mechanism | Consensus revisions are incorporated gradually because institutions reposition slowly and investors process estimate changes at different speeds |
| economic_counterparty | Investors who update expectations late |
| mechanism_constraint | Asynchronous information arrival; institutional execution constraints; disagreement |
| our_advantage | AuTrad's own point-in-time daily consensus history (no free historical source exists), plus Company Memory context |
| failure_condition | Prices incorporate revisions on the day they appear in the consensus, or the residual return is below costs, tax and capacity |

* **Data:** the daily snapshot (`scripts/fetch_estimates`). The universe widens from NIFTY 50 to all stocks with stock
  futures. **Collection requires the daily task to run, which is the owner's decision.**
* **Signal:** 20-session change in the current-fiscal-year EPS consensus, as observed by AuTrad (not the vendor's
  look-back columns), in % of |EPS|. Valid when ≥ 3 analysts and the quality flags are clean.
* **Hypotheses:**
  * HA1: the cross-sectional rank IC of the revision signal with the next 20-session market-adjusted return > 0
    (monthly, Newey–West).
  * HA2: the hedged futures long-short (top vs bottom quintile of revisions) passes the Part A gate.
* **Sealed until** both conditions hold: ≥ 12 months of snapshots, and ≥ 24 monthly cross-sections with ≥ 100 valid
  stocks. No look before then.

## Part C — NSE depth: execution research (forward only; sealed)

| Question | Answer |
|---|---|
| economic_mechanism | Order-book state predicts short-horizon impact and fill quality; an executor that conditions on it pays less than one that does not |
| economic_counterparty | Liquidity providers and other takers when AuTrad avoids crossing into thin or toxic books |
| mechanism_constraint | Depth data is not public historically (₹12.5 lakh/yr) |
| our_advantage | AuTrad's own recorded depth: a proprietary dataset |
| failure_condition | NSE spreads/ticks make the execution saving < 0.5 bp per order, as for BTC in v1.0 |

* **Study (the M1 design of v1.0 applied to NSE index futures):**
  * Exogenous parent orders every 5 minutes, 1 lot each.
  * Policies: P0 cross; P1 post then cross at 60 s; P2 P1 plus a depth-imbalance trigger. The θ is chosen so that the
    predicted move exceeds the fee/spread gap, i.e. **θ = the measured half-spread + 0.2 bp**, fixed now.
  * Measure: implementation shortfall.
* **Sealed until** ≥ 20 sealed recorded days. Training uses the first 60 % of days and testing the last 40 %.
* **Requires** the owner's Kite Connect key and the recorder running each session.

## Part D — forward-only candidate from v1.2

The v1.2 in-sample observation was: banned stocks rose +0.9 % (market-adjusted) over the 4 sessions before expiry.
* It is **not** a strategy, and is tracked forward only, from 2026-10-05.
* The forward ledger logs each ban-at-expiry event: the T−4 close, the T close, and the NIFTY return.
* Evaluation after ≥ 24 months **and** ≥ 300 events: mean > 0, one-sided Newey–West t > 2.
* No backfill is ever used.

## Part E — Company Memory interpretation (descriptive pilot)

* For the promises in `data/intel/graph.sqlite` with a measurable revenue/profit target and a due date: compare the
  target with the filed numbers once due.
* Report MET / MISSED / UNRESOLVABLE, and the market-adjusted return around the first filing that resolves each
  promise.
* The sample is small, so this is **descriptive only and gates nothing**. Any hypothesis it suggests needs its own
  pre-registration.

## Unchanged

* REV1 stays frozen.
* No live trading, no broker connection, no data purchase.
* ARGO is not enlarged; it becomes relevant only after a candidate clears the gate and the forward test.

## Amendment 1 — corrections after the independent review (2026-10-04; development results had been seen)

No hypothesis, unit, threshold or gate was changed.

1. **Announcement time.** Use the earliest of broadcast and dissemination time, i.e. the first public moment. Drop
   results-index entries more than 60 days after quarter end: these are late uploads, not announcements. This is the
   v0.5 stale rule; SEBI's limit is 45 days. The review found that 2016–2018 index times show no reaction spike on r0.
   Keep one event per (symbol, r0).
2. **Liquidity.** The ₹10 crore test uses the stock's total futures traded value that day (all contracts), not the
   value of the single contract traded. The pre-registration wording was ambiguous.
3. **Descriptive equal-weight benchmark.** Now point in time: stocks with futures that day and outside NIFTY 50. The
   first version used every symbol that ever had an event, which is look-ahead.
4. **Inference reported next to the registered statistic:** month-clustered t, and Newey–West with 20 lags (the
   20-session windows overlap within a results season).
5. **Hedged trade.** Its exposure to the NIFTY MIDCAP 150 − NIFTY 50 spread is reported. The registered β-to-NIFTY
   step misses this midcap-vs-large-cap bet.
6. **HM1b.** The needed-filings list now includes both statement bases. NSE throttled XBRL downloads from this
   machine (HTTP 403) during the run, so HM1b is **not testable at this run**. It is computed automatically by the same
   code once coverage reaches 70 %, with nothing else changed.
7. **Part D forward ledger.** It now reads F&O data without the research END cut-off, so forward expiries are visible.
   A synthetic test checks that events are recorded forward and never backfilled.
8. The audit's after-close share uses the 15:00 rule.
