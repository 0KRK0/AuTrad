# AuTrad research dashboard

Generated from `research/trials.jsonl`: 24 pre-registration records, 114 experiment rows.

## Discovery

| Version | Experiments | Rejected | Surviving | Significant only | With mechanism |
|---|---|---|---|---|---|
| v0.4-r1 | 17 | 16 | 0 | 0 | 0 |
| v0.3.1-aborted (run stopped: data fix incomplete) | 7 | 1 | 0 | 0 | 0 |
| v0.3.1 | 14 | 7 | 0 | 0 | 0 |
| v0.4 | 17 | 16 | 0 | 0 | 0 |
| v0.5 | 10 | 3 | 0 | 3 | 0 |
| v0.6 | 4 | 0 | 0 | 0 | 0 |
| v0.7 | 2 | 2 | 0 | 0 | 0 |
| v0.8.1 | 4 | 4 | 0 | 0 | 0 |
| v0.9 | 13 | 2 | 0 | 11 | 0 |
| v1.0 | 7 | 6 | 0 | 1 | 0 |
| v1.1 | 7 | 7 | 0 | 0 | 7 |
| v1.2 | 9 | 7 | 0 | 2 | 9 |
| v1.3 | 3 | 2 | 0 | 0 | 3 |

**Surviving the full gate: 0. Proven alpha (gate + forward paper test): 0.**

## Latest experiments

| Version | Experiment | t | Holm p | Net (as logged) | Decision | Mechanism |
|---|---|---|---|---|---|---|
| v1.0 | m1_HM1b | -2.85 | 0.998 | -0.1 | not significant (wrong sign) |  |
| v1.0 | m1_gate_retail |  |  | -0.1 | REJECTED at HM1a_HM1b_significant |  |
| v1.0 | m2_HM2a | -38.7 | 1 | -1.01 | not significant (negative) |  |
| v1.0 | m2_HM2b | -4.99 | 1 | -0.039 | not significant (negative) |  |
| v1.0 | m3_HC1_BTCUSDT | 6.38 | 1.04e-10 | 0.04 | REJECTED at net_ge_8pct |  |
| v1.0 | m3_HC1_ETHUSDT | 6.67 | 3.12e-11 | 0.0523 | REJECTED at net_ge_8pct |  |
| v1.1 | v11_HB1 | -8.44 | 1 | -0.00913 | REJECTED at statistical | financing / arbitrage-capital constraint: futures longs pay carry above cash for leverage |
| v1.1 | v11_HB2 | -7.83 | 1 | -0.0702 | REJECTED at statistical | temporary leverage demand / limits to arbitrage around events |
| v1.1 | v11_HB3-N | -3.32 | 1 | -0.00357 | REJECTED at statistical | index financing demand by futures longs; ETF tracking noise deters arbitrage |
| v1.1 | v11_HB3-BN | -1.32 | 1 | -0.00173 | REJECTED at statistical | index financing demand by futures longs; ETF tracking noise deters arbitrage |
| v1.1 | v11_HB4 | -0.543 | 1 |  | not significant (measurement) | forced flow: monthly roll by holders who must keep exposure |
| v1.1 | v11_HB1_as_preregistered_signal | 3.33 | 0.00221 | 0.0103 | REJECTED at excess_ge_1.5pct (look-ahead on dividend announcements) | financing / arbitrage-capital constraint: futures longs pay carry above cash for leverage |
| v1.1 | v11_HB2_as_preregistered_signal | 4.6 | 2.45e-05 | 0.0451 | REJECTED at after_tax_excess_ge_1pct (look-ahead on dividend announcements) | temporary leverage demand / limits to arbitrage around events |
| v1.2 | v12_HD1 | -2.25 | 1 |  | not significant (wrong sign); dividends ~93% discounted by first close | futures should re-price by -D when a dividend becomes public; participants incorporate it at different speeds |
| v1.2 | v12_HD2 | 0.332 | 1 | 0.0051 | REJECTED at statistical | futures should re-price by -D when a dividend becomes public; participants incorporate it at different speeds |
| v1.2 | v12_HI1_N50 | 0.499 | 1 | 0.00391 | not significant | passive trackers must buy inclusions and sell exclusions at the rebalance close |
| v1.2 | v12_HI2_N50 | -0.431 | 1 |  | not significant | passive trackers must buy inclusions and sell exclusions at the rebalance close |
| v1.2 | v12_HI1_NN50 | 1.03 | 0.931 | 0.0116 | not significant | passive trackers must buy inclusions and sell exclusions at the rebalance close |
| v1.2 | v12_HI2_NN50 | -0.784 | 1 |  | not significant | passive trackers must buy inclusions and sell exclusions at the rebalance close |
| v1.2 | v12_HI_econ |  | 1 | -0.0585 | REJECTED at statistical | passive trackers must buy inclusions and sell exclusions at the rebalance close |
| v1.2 | v12_HF1 | 4.58 | 1.74e-05 |  | significant as registered; tail-driven (median +0.5 bp), month-clustered t 2.1 (measurement) | in an F&O ban no new derivative positions may be opened: arbitrage capital cannot enter and expiring positions cannot be rolled |
| v1.2 | v12_HF2 | 5.44 | 2.65e-07 |  | significant as registered; null when matched on OI/MWPL level (measurement) | in an F&O ban no new derivative positions may be opened: arbitrage capital cannot enter and expiring positions cannot be rolled |
| v1.3 | v13_HM1_drift | -1.84 | 1 | 0.00738 | not significant (wrong sign) | bad results news is incorporated slowly: pessimists cannot easily short Indian cash equities, long-only institutions sell over days, analyst downgrades lag |
| v1.3 | v13_HM1_trade | -2.51 | 1 | -0.117 | REJECTED at statistical | bad results news is incorporated slowly: pessimists cannot easily short Indian cash equities, long-only institutions sell over days, analyst downgrades lag |
| v1.3 | v13_HM1b |  |  |  | not testable at this run (XBRL coverage 17%; NSE throttled downloads) | bad results news is incorporated slowly: pessimists cannot easily short Indian cash equities, long-only institutions sell over days, analyst downgrades lag |

## Data clock

As of 2026-10-04. Counts only; no returns are looked at.

| Dataset | Collected | Toward the threshold | Progress | State |
|---|---|---|---|---|
| Analyst snapshots | 2 days (2026-10-03 to 2026-10-04), 213 names, 186 valid | 0.0 / 12 months; 0 / 24 cross-sections | 0 % | sealed |
| NSE depth | 0 symbols, 0 recorded days | 0 / 20 sealed days | 0 % | recorder not running: needs the owner's Kite Connect API key and access token |
| Promise verdicts | 0 of 35 queued promises reviewed | none | 0 % | human review |
| Ban at expiry (C-014) | 0 events | 0 / 24 months; 0 / 300 events | 0 % | sealed |

Candidates (not strategies, never tradeable): C-013 FORWARD / PRE-REGISTRATION REQUIRED, C-014 FORWARD / LEDGER RUNNING

