"""Round-trip trade ledger from fills (average-cost accounting, fees and slippage included)."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class _Open:
    qty: float = 0.0
    avg: float = 0.0
    pnl: float = 0.0
    fees: float = 0.0
    entry: pd.Timestamp | None = None
    max_notional: float = 0.0


def build_ledger(fills: list[tuple[pd.Timestamp, object]], last_prices: dict[str, float],
                 category_of: dict[str, str], end: pd.Timestamp) -> pd.DataFrame:
    """fills: [(timestamp, Fill)]. A trade = one episode from flat back to flat (or a flip)."""
    state: dict[str, _Open] = {}
    rows: list[dict] = []

    def close(sym: str, st: _Open, t: pd.Timestamp, is_open: bool) -> None:
        rows.append({
            "symbol": sym, "category": category_of.get(sym, "unknown"), "side": "long" if st.qty >= 0 else "short",
            "entry": st.entry, "exit": t, "days": (t - st.entry).days if st.entry is not None else 0,
            "pnl": st.pnl - st.fees, "fees": st.fees, "notional": st.max_notional,
            "ret": (st.pnl - st.fees) / st.max_notional if st.max_notional else 0.0, "open": is_open,
        })

    for t, f in fills:
        sym, q, px = f.symbol, f.qty, f.price
        st = state.setdefault(sym, _Open())
        st.fees += f.commission
        if abs(st.qty) < 1e-12:
            st.qty, st.avg, st.entry = q, px, t
        elif (st.qty > 0) == (q > 0):          # add to position
            st.avg = (st.avg * abs(st.qty) + px * abs(q)) / (abs(st.qty) + abs(q))
            st.qty += q
        else:                                   # reduce / close / flip
            closing = min(abs(q), abs(st.qty))
            st.pnl += (px - st.avg) * closing * (1 if st.qty > 0 else -1)
            rest = abs(q) - closing
            st.qty += q if rest == 0 else -st.qty
            if abs(st.qty) < 1e-9:
                st.qty = 0.0
                close(sym, st, t, False)
                state[sym] = st = _Open()
                if rest > 1e-12:
                    st.qty, st.avg, st.entry = (rest if q > 0 else -rest), px, t
        st.max_notional = max(st.max_notional, abs(st.qty * st.avg))
    for sym, st in state.items():
        if abs(st.qty) > 1e-12 and sym in last_prices:
            st.pnl += (last_prices[sym] - st.avg) * st.qty
            close(sym, st, end, True)
    return pd.DataFrame(rows)


def trade_stats(ledger: pd.DataFrame) -> dict[str, float]:
    if ledger.empty:
        return {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "avg_trade": 0.0, "avg_trade_ret": 0.0,
                "avg_win": 0.0, "avg_loss": 0.0, "avg_hold_days": 0.0}
    wins, losses = ledger[ledger.pnl > 0], ledger[ledger.pnl <= 0]
    gross_loss = -losses.pnl.sum()
    return {
        "trades": int(len(ledger)),
        "win_rate": float(len(wins) / len(ledger)),
        "profit_factor": float(wins.pnl.sum() / gross_loss) if gross_loss > 0 else float("inf"),
        "avg_trade": float(ledger.pnl.mean()),          # expected value per trade, Rs
        "avg_trade_ret": float(ledger.ret.mean()),       # expected return per trade on notional
        "avg_win": float(wins.pnl.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.pnl.mean()) if len(losses) else 0.0,
        "avg_hold_days": float(ledger.days.mean()),
    }
