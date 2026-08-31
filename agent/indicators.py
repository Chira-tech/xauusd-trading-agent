"""Pure-numpy technical indicators. No pandas dependency."""
import numpy as np


def rsi(closes, period: int = 14):
    """Wilder's RSI. Returns an array the same length as `closes`, NaN until warmed up."""
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()

    def _rsi(g, l):
        if l == 0:
            return 100.0
        rs = g / l
        return 100.0 - (100.0 / (1.0 + rs))

    out[period] = _rsi(avg_gain, avg_loss)
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        out[i] = _rsi(avg_gain, avg_loss)
    return out


def rolling_return(closes, window: int):
    closes = np.asarray(closes, dtype=float)
    out = np.full(len(closes), np.nan)
    for i in range(window, len(closes)):
        out[i] = closes[i] / closes[i - window] - 1.0
    return out
