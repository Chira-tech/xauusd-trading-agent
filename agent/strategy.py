"""Turn a strategy.yaml + the current bar into an entry / exit decision.

Long-only RSI mean-reversion:
  entry : RSI closes below entry.threshold
  exit  : whichever comes first of hard stop, take-profit, or RSI recovering
          above exit.rsi_exit
"""


def entry_signal(strategy: dict, rsi_val) -> bool:
    e = strategy["entry"]
    if rsi_val is None or rsi_val != rsi_val:  # None or NaN
        return False
    if e.get("direction", "long") != "long":
        return False
    return rsi_val < e["threshold"]


def exit_signal(strategy: dict, entry_price: float, bar: dict, rsi_val):
    """Return (should_exit: bool, reason: str | None, exit_price: float | None)."""
    x = strategy["exit"]
    sl = strategy["stop_loss_pct"] / 100.0
    tp = x["take_profit_pct"] / 100.0

    stop_price = entry_price * (1.0 - sl)
    tp_price = entry_price * (1.0 + tp)

    # Stop is checked first: on a bar that hit both, assume the worse fill.
    if bar["low"] <= stop_price:
        return True, "stop_loss", stop_price
    if bar["high"] >= tp_price:
        return True, "take_profit", tp_price
    if rsi_val is not None and rsi_val == rsi_val and rsi_val >= x["rsi_exit"]:
        return True, "rsi_exit", bar["close"]
    return False, None, None
