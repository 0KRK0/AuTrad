# REV1 forward paper test — status

As of **2026-10-01** · ledger head `949b1c359daf4625` · integrity **OK** · pre-registration `4c7e8ca15ca46b0f` · paper only, never live

Frozen rule: top third of point-in-time NIFTY 50 members by minus their last-month return, equal weight, rebalanced at each month-end close, filled at the next open with full Indian costs. Nothing may be changed.

| | Paper REV1 | Equal-weight NIFTY 50 (primary) | NIFTYBEES (secondary) |
|---|---|---|---|
| Value (₹) | 988,053 | 990,660 | 990,667 |
| Return since the 1 Oct 2026 open | -1.19% | -0.93% | -0.93% |

Excess vs primary benchmark **-0.26%** (log) · vs NIFTYBEES -0.26% · modelled costs ₹1,442 · 1 sessions · 0 complete months · z vs history -0.59 (REJECT below −2, only at review)

**Verdict: KEEP OBSERVING (first review after the 2027-10 month-end)**

## Decisions

| Month | Decision close | Timing | Universe | Selected | Risk Gate (shadow) |
|---|---|---|---|---|---|
| 2026-09 | 2026-09-30 | LATE (computed 2026-10-03) | 50 | ASIANPAINT, BAJAJ-AUTO, BAJAJFINSV, BAJFINANCE, EICHERMOT, GRASIM, INFY, JIOFIN, M&M, MARUTI, MAXHEALTH, NESTLEIND, SBIN, SHRIRAMFIN, TCS, TITAN | gate approves every order unchanged |

## Audit events

- 2026-10-03 runner_version: forward/ledger.py, forward/rev1.py, scripts/forward_rev1.py
- 2026-10-03 pipeline_change: lab/panel.py, marketdata/corporate_actions.py, marketdata/exchange.py, marketdata/pit.py, marketdata/reference/bse_codes.csv, marketdata/reference/corporate_actions.csv, marketdata/reference/nifty50_pit_snapshots.csv, marketdata/reference/verified_real_moves.csv, marketdata/store.py, marketdata/universe.py

A forward test this short cannot prove REV1: at its historical information ratio (~0.26), forward data would need about 59 years to reach t = 2. It can catch REV1 breaking, and it checks the data, decision and cost pipeline in real time.

