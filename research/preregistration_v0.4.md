# AuTrad v0.4 — pre-registration of factor and event hypotheses

Written 2026-10-03, **before any factor or event result was computed**. Definitions, data rules, tests and pass
criteria below are frozen. Anything added later is marked as an amendment with its date and reason, and every
test, including the failures, is logged in `research/trials.jsonl`. The sha256 of this file is stored in the ledger.

## Why pre-register

v0.3 tried 80 configurations, and the 2024–26 test window has already been seen. More searching would mostly
produce false positives. v0.4 therefore tests a small, fixed set of factors with an economic rationale, using
definitions fixed in advance, so results cannot steer what gets tested.

## Universe and timing

- Universe: point-in-time NIFTY 50 constituents on each decision date, using the exchange-anchored price data
  (transforms `ca_2026_10_03_exchange_anchored`).
- Decision: at the close of the last trading day of each month. Trades execute at the next session's open. The
  holding period is one month.
- Information rule: a filing may be used for a decision on day *t* only if its exchange dissemination time
  (`available_at`) is strictly before 15:30 IST on day *t*. A filing disseminated after the close counts from the
  next session. Prices are used only up to the close of *t*.
- Sample: price factors run Jan 2016 to Sep 2026. Fundamental factors start where NSE's machine-readable (XBRL)
  results begin, mid-2018. They need four quarters of history, so they run from about mid-2019. Everything before
  2024 was used in earlier research and the 2024–26 window has been seen. **Results are historical evidence, not a
  sealed test.** The clean test is forward data from 1 Oct 2026.

## Factors (fixed definitions; no parameters are tuned)

Higher score = expected to outperform. Every score is cross-sectionally ranked among eligible stocks on the decision date.

| Id | Family | Definition | Rationale |
|---|---|---|---|
| MOM12 | Momentum | Return from t−252 to t−21 sessions | Under-reaction and slow information diffusion |
| MOM6 | Momentum | Return from t−126 to t−21 | As above, shorter horizon |
| RMOM | Momentum | 12-1 momentum of residuals from a rolling 252-day market-model regression on the NIFTY 50 | Stock-specific momentum, less exposed to market-beta crashes |
| REV1 | Reversal | Minus the return from t−21 to t | Short-term liquidity provision |
| LVOL | Defensive | Minus the 252-day volatility of daily returns | Low-risk anomaly (leverage constraints, lottery preference) |
| LBETA | Defensive | Minus the 252-day beta to the NIFTY 50 | Betting-against-beta |
| LDD | Defensive | Minus the maximum drawdown over 252 days | Avoid distressed names |
| EY | Value | TTM EPS / price (TTM = last four quarterly EPS, each available by t) | Cheap stocks earn a premium |
| BM | Value | Latest book equity / market cap (shares = paid-up capital / face value) | Classic value |
| ROE | Quality | TTM net profit / latest book equity | Profitable firms are under-priced |
| MARGIN | Quality | TTM net profit / TTM revenue (non-financials) | Pricing power |
| LEV | Quality | Minus borrowings / book equity (non-financials) | Balance-sheet strength |
| STAB | Quality | Minus the std of quarterly YoY EPS growth over the last 8 quarters | Earnings stability |
| GREV | Growth | TTM revenue / TTM revenue a year earlier − 1 | Fundamental momentum |
| GEPS | Growth | TTM EPS / TTM EPS a year earlier − 1 (only where both > 0) | As above |
| SUE | Event | (EPS_q − EPS_{q−4}) / std of that difference over the previous 8 quarters, from the latest filing available by t (seasonal random walk; no analyst consensus needed) | Post-earnings-announcement drift |
| EAR | Event | Market-adjusted return from the last close before dissemination to the close of the first full session after it, for the latest filing available by t (zero if older than 63 sessions) | Under-reaction to news |

Financials (banks, NBFCs and insurers, identified by sector) are excluded from MARGIN and LEV, where those
ratios are not comparable. A stock lacking the inputs for a factor is left out of that factor's ranking.

## Event study (separate from the monthly factors)

For every results filing from 2018 onward with a timestamp, compute the event-day market-adjusted reaction (as in
EAR). Then measure the cumulative market-adjusted return from the open of the session after the reaction day to
+5, +20 and +60 sessions, by reaction tercile. Hypothesis: positive drift after good reactions and negative drift
after bad ones, net of a realistic round-trip cost (~29 bps delivery for large caps).

## Tests

For each factor:

1. Monthly cross-sectional Spearman IC with the next month's return (next open to the following month-end close).
   Report the mean IC, its t-stat (Newey–West, 3 lags), the % of months positive, and the IC by year.
2. Long-only top tercile (equal weight), rebalanced monthly through the lab engine with the full Indian cost model,
   against an equal-weight universe rebalanced the same way. Report the gross and net annual excess return,
   costs, turnover, tracking error, information ratio and % of years with positive excess.
3. Long–short top-minus-bottom tercile using stock-futures costs, as a diagnostic of the pure factor.
4. Stability: the sign of the mean IC in the first and second half of each factor's sample.

## Pass criteria (all must hold)

- Holm-adjusted p-value of the mean-IC t-stat < 0.05 across all 17 factors.
- Net annual excess return of the long-only tercile > 0 after costs.
- Mean IC with the same sign in both halves of the sample.
- Positive IC in at least 60% of calendar years.

Factors that pass may be combined **only by equal-weighting their rank scores**. Weights are not fitted. The
composite is one more logged trial, evaluated the same way. Nothing is promoted to trading on historical evidence
alone: a passing factor or composite becomes a *candidate* for the forward paper test.

## Attribution

Each existing v0.3 sleeve's monthly excess returns are regressed on the long–short returns of the market and of
the MOM12, LVOL, EY and ROE factors. This shows how much of what each sleeve earned is explained by known
factors, and how much is left unexplained (alpha).

## What v0.4 will not do

No machine-learning models, no additional technical indicators, no parameter grids, no intraday work, and no
changes to these definitions after results are seen. Any of these would be a separately registered study.
