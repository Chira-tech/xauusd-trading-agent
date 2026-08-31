"""The reflection cycle -- this is the 'self-learning' part.

After every `reflection_every` closed trades the agent:
  1. scores the batch against goal.yaml
  2. grades its own previous hypothesis (did the last change help, hurt, or nothing?)
  3. picks EXACTLY ONE variable to change, by priority of which goal condition is
     most violated
  4. snapshots the old strategy to state/history/vNN.yaml, bumps the version,
     writes the new strategy, and appends a dated hypothesis with its reasoning
     and a falsifiable prediction to state/hypotheses.jsonl

It is fully deterministic: same trades in, same change out. No LLM, no network.
"""
from datetime import datetime, timezone

import yaml

from . import config, paths
from .portfolio import append_jsonl, read_jsonl, rewrite_jsonl
from .score import score

# Guard rails: the agent may only move a variable within these bounds.
BOUNDS = {
    "entry.threshold": (10.0, 45.0),
    "exit.rsi_exit": (50.0, 80.0),
    "exit.take_profit_pct": (1.0, 10.0),
    "stop_loss_pct": (0.5, 6.0),
    "position_size_r": (0.1, 1.0),
}


def _get(strategy: dict, dotted: str):
    node = strategy
    for k in dotted.split("."):
        node = node[k]
    return node


def _set(strategy: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    node = strategy
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = value


def _clamp(value: float, dotted: str) -> float:
    lo, hi = BOUNDS[dotted]
    return round(max(lo, min(hi, value)), 4)


def choose_change(strategy: dict, detail: dict, goal: dict):
    """Return (dotted_var, new_value, reasoning, prediction). Exactly one variable."""
    realised = detail["realised_return"]
    dd = detail["max_drawdown"]
    sharpe = detail["sharpe"]
    win_rate = detail["win_rate"]

    # Priority 1 -- failure condition breached: protect capital, tighten the stop.
    if dd > goal["max_drawdown"]:
        cur = _get(strategy, "stop_loss_pct")
        new = _clamp(cur - 0.3, "stop_loss_pct")
        return ("stop_loss_pct", new,
                f"Batch drawdown {dd:.2%} breached the {goal['max_drawdown']:.0%} failure limit. "
                f"Tightening stop_loss_pct {cur} -> {new} to cap the loss on each trade.",
                "Next batch drawdown stays under the limit; realised return may fall slightly.")

    # Priority 2 -- return under target.
    if realised < goal["target_return_30d"]:
        if win_rate >= 0.5:
            # Setups are working when taken -- take more of them.
            cur = _get(strategy, "entry.threshold")
            new = _clamp(cur + 2.0, "entry.threshold")
            return ("entry.threshold", new,
                    f"Realised {realised:.2%} is below the +{goal['target_return_30d']:.0%} target but "
                    f"win rate is {win_rate:.0%}. Raising RSI entry threshold {cur} -> {new} to take more setups.",
                    "Next batch has more trades and realised return moves toward target.")
        else:
            # Winning too rarely -- bank profit sooner.
            cur = _get(strategy, "exit.take_profit_pct")
            new = _clamp(cur - 0.5, "exit.take_profit_pct")
            return ("exit.take_profit_pct", new,
                    f"Realised {realised:.2%} below target with a weak {win_rate:.0%} win rate. "
                    f"Lowering take_profit_pct {cur} -> {new} to lock gains before they reverse.",
                    "Next batch win rate rises; average winning trade is smaller.")

    # Priority 3 -- return is fine but too bumpy.
    if sharpe < goal["min_sharpe"]:
        cur = _get(strategy, "position_size_r")
        new = _clamp(cur - 0.1, "position_size_r")
        return ("position_size_r", new,
                f"Return meets target but batch Sharpe {sharpe:.2f} is under {goal['min_sharpe']}. "
                f"Cutting position_size_r {cur} -> {new} to smooth the equity curve.",
                "Next batch Sharpe improves; realised return scales down modestly.")

    # Priority 4 -- everything passed: press the edge.
    cur = _get(strategy, "position_size_r")
    new = _clamp(cur + 0.1, "position_size_r")
    return ("position_size_r", new,
            f"All bars cleared (return {realised:.2%}, drawdown {dd:.2%}, Sharpe {sharpe:.2f}). "
            f"Raising position_size_r {cur} -> {new} to compound the working edge.",
            "Next batch realised return rises; drawdown must stay within the limit.")


def _grade_previous(current_score: float) -> None:
    """Mark the last still-'pending' hypothesis helped / no_change / hurt."""
    hyps = read_jsonl(paths.HYPOTHESES_FILE)
    if not hyps or hyps[-1].get("outcome") != "pending":
        return
    prev = hyps[-1]
    baseline = prev.get("batch_score", 0.0)
    if current_score > baseline + 1e-6:
        prev["outcome"] = "helped"
    elif current_score < baseline - 1e-6:
        prev["outcome"] = "hurt"
    else:
        prev["outcome"] = "no_change"
    prev["observed_next_score"] = current_score
    rewrite_jsonl(paths.HYPOTHESES_FILE, hyps)


def run_reflection(trades_batch, equity_points, goal, source="deterministic", dry_run=False):
    """Score the batch, choose one change, apply it. Returns (hypothesis, strategy, snapshot_path)."""
    strategy = config.load_strategy()
    batch_score, detail = score(trades_batch, equity_points, goal)
    var, new_value, reasoning, prediction = choose_change(strategy, detail, goal)
    old_value = _get(strategy, var)

    prev_version = strategy["version"]
    new_version = f"{int(prev_version) + 1:02d}"

    hypothesis = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "from_version": prev_version,
        "to_version": new_version,
        "source": source,
        "batch_score": batch_score,
        "metrics": detail,
        "variable": var,
        "old_value": old_value,
        "new_value": new_value,
        "reasoning": reasoning,
        "prediction": prediction,
        "outcome": "pending",
    }

    if dry_run:
        return hypothesis, strategy, None

    _grade_previous(batch_score)

    snapshot = paths.HISTORY / f"v{prev_version}.yaml"
    with open(snapshot, "w", encoding="utf-8") as f:
        yaml.safe_dump(strategy, f, sort_keys=False)

    _set(strategy, var, new_value)
    strategy["version"] = new_version
    config.save_strategy(strategy)
    append_jsonl(paths.HYPOTHESES_FILE, hypothesis)

    return hypothesis, strategy, snapshot
