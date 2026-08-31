"""Score a batch of closed trades against goal.yaml -> float in [-1, +1].

Composite of three parts:
  0.5 * (realised return vs target)
  0.3 * (drawdown headroom vs max_drawdown)
  0.2 * (batch Sharpe vs min_sharpe)
"""
import numpy as np


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _sharpe(returns, periods_per_year: int = 252) -> float:
    r = np.asarray(returns, dtype=float)
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float((r.mean() / r.std(ddof=1)) * np.sqrt(periods_per_year))


def score(trades: list[dict], equity_points: list[dict], goal: dict):
    """Return (final_score: float, detail: dict)."""
    if not trades:
        return 0.0, {}

    rets = [t["return"] for t in trades]
    comp = 1.0
    for r in rets:
        comp *= (1.0 + r)
    realised = comp - 1.0

    target = goal["target_return_30d"]
    floor = goal["failure_below"]
    if realised < floor:
        ret_component = -1.0 + _clamp((realised - floor) * 5.0, -0.5, 0.0)
    else:
        ret_component = _clamp(realised / target if target else 0.0, -1.5, 1.5)

    dd = max((p["drawdown"] for p in equity_points), default=0.0)
    max_dd = goal["max_drawdown"]
    dd_component = _clamp(1.0 - dd / max_dd, -1.0, 1.0) if max_dd else 0.0

    sh = _sharpe(rets)
    min_sh = goal["min_sharpe"]
    sharpe_component = _clamp(sh / min_sh, -1.0, 1.5) if min_sh else 0.0

    raw = 0.5 * ret_component + 0.3 * dd_component + 0.2 * sharpe_component
    final = round(_clamp(raw, -1.0, 1.0), 4)

    detail = {
        "realised_return": round(realised, 5),
        "max_drawdown": round(dd, 5),
        "sharpe": round(sh, 3),
        "win_rate": round(sum(1 for r in rets if r > 0) / len(rets), 3),
        "n_trades": len(trades),
        "ret_component": round(ret_component, 3),
        "dd_component": round(dd_component, 3),
        "sharpe_component": round(sharpe_component, 3),
    }
    return final, detail
