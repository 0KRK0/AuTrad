# NIFTY 50 point-in-time constituents

`nifty50_pit_snapshots.csv` (effective_date, symbol, company_name, isin, weight_pct) is extracted from
the "All snapshots" sheet of **NIFTY50_constituents_2016_to_2026.xlsx** published by Marketcalls:
https://www.marketcalls.in/llm-models/how-to-build-survivorship-bias-free-nifty-50-historical-data-using-grokbot.html

- 22 semi-annual snapshots, 1 Apr 2016 to 30 Aug 2026, 50 companies each (51 securities in 2016-17 incl. Tata Motors DVR).
- Reconstructed by the publisher from NSE's official current constituent list walked back through NSE Indices
  press releases; weights from NSE Indices monthly `mcwb` files. The workbook's own Sources / Validation sheets
  list each press release used.
- Third-party reconstruction: spot-check against NSE Indices press releases before relying on it for capital.
- Downloaded 30 Sep 2026. Symbol aliases and exclusions are applied in `marketdata/pit.py`.
