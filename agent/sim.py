"""Pure strategy simulation: bars + a strategy dict -> trades + equity curve.

No file I/O, no globals. Used by evolve.py to score a strategy over a window
many times per second. Mirrors the trade mechanics in loop.py / portfolio.py.
"""
from .indicators import rsi as rsi_calc
from .strategy import entry_signal, exit_signal


def simulate(bars: list[dict], strategy: dict, goal: dict) -> dict:
    closes = [b["close"] for b in bars]
    rsi_series = rsi_calc(closes, strategy["entry"]["period"])

    equity = goal["starting_equity"]
    peak = equity
    position = None
    trades: list[dict] = []
    equity_points: list[dict] = []

    for i, bar in enumerate(bars):
        r = rsi_series[i]
        r = None if r != r else float(r)  # NaN -> None

        eq = equity
        if position:
            eq += (bar["close"] - position["entry_price"]) * position["units"]
        peak = max(peak, eq)
        dd = 0.0 if peak == 0 else (peak - eq) / peak
        equity_points.append({"date": bar["date"], "equity": round(eq, 2),
                              "drawdown": round(dd, 5)})

        if position:
            should, reason, px = exit_signal(strategy, position["entry_price"], bar, r)
            if should:
                pnl = (px - position["entry_price"]) * position["units"]
                ret = px / position["entry_price"] - 1.0
                equity += pnl
                peak = max(peak, equity)
                trades.append({
                    "entry_date": position["entry_date"], "exit_date": bar["date"],
                    "entry_price": round(position["entry_price"], 4),
                    "exit_price": round(px, 4),
                    "units": round(position["units"], 6),
                    "pnl": round(pnl, 2), "return": round(ret, 5), "reason": reason,
                    "equity_after": round(equity, 2),
                    "strategy_version": strategy["version"],
                })
                position = None
        elif entry_signal(strategy, r):
            size_usd = equity * strategy["position_size_r"]
            position = {"entry_price": bar["close"], "units": size_usd / bar["close"],
                        "entry_date": bar["date"]}

    return {
        "trades": trades,
        "equity_points": equity_points,
        "final_equity": round(equity, 2),
        "total_return": round(equity / goal["starting_equity"] - 1.0, 5),
        "n_trades": len(trades),
    }
