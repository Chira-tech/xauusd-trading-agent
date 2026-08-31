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


def _proposals(strategy: dict, detail: dict, goal: dict):
    """Ordered list of candidate one-variable changes, highest priority first.

    Each entry is (dotted_var, new_value, reasoning, prediction). The list lets a
    caller skip a variable it has already falsified this round (see `blacklist`
    in choose_change) and move to the next-best lever.
    """
    realised = detail["realised_return"]
    dd = detail["max_drawdown"]
    sharpe = detail["sharpe"]
    win_rate = detail["win_rate"]
    out = []

    def add(var, delta, why, predict):
        cur = _get(strategy, var)
        new = _clamp(cur + delta, var)
        if new != cur:
            out.append((var, new, why.format(cur=cur, new=new), predict))

    # Priority 1 -- failure condition breached: protect capital.
    if dd > goal["max_drawdown"]:
        add("stop_loss_pct", -0.3,
            f"Drawdown {dd:.2%} breached the {goal['max_drawdown']:.0%} failure limit. "
            "Tighten stop_loss_pct {cur} -> {new} to cap each loss.",
            "Drawdown falls back under the limit; realised return may dip.")
        add("position_size_r", -0.1,
            f"Drawdown {dd:.2%} over the {goal['max_drawdown']:.0%} limit. "
            "Cut position_size_r {cur} -> {new} to shrink every loss.",
            "Drawdown drops proportionally; returns scale down.")

    # Priority 2 -- return under target.
    if realised < goal["target_return_30d"]:
        if win_rate >= 0.5:
            add("entry.threshold", +2.0,
                f"Return {realised:.2%} below the +{goal['target_return_30d']:.0%} target, "
                f"but win rate {win_rate:.0%}. Raise entry.threshold {{cur}} -> {{new}} to take more setups.",
                "More trades next batch; realised return moves toward target.")
            add("position_size_r", +0.1,
                f"Return {realised:.2%} under target with a solid {win_rate:.0%} win rate. "
                "Raise position_size_r {cur} -> {new} to compound the edge.",
                "Realised return scales up; watch drawdown.")
        else:
            add("exit.take_profit_pct", -0.5,
                f"Return {realised:.2%} under target, weak {win_rate:.0%} win rate. "
                "Lower take_profit_pct {cur} -> {new} to bank gains sooner.",
                "Win rate rises; average win shrinks.")
            add("entry.threshold", -2.0,
                f"Return {realised:.2%} under target, weak {win_rate:.0%} win rate. "
                "Lower entry.threshold {cur} -> {new} to be more selective.",
                "Fewer, higher-quality entries; win rate rises.")

    # Priority 3 -- return fine but too bumpy.
    if sharpe < goal["min_sharpe"]:
        add("position_size_r", -0.1,
            f"Return meets target but Sharpe {sharpe:.2f} is under {goal['min_sharpe']}. "
            "Cut position_size_r {cur} -> {new} to smooth the curve.",
            "Sharpe improves; realised return scales down modestly.")
        add("exit.rsi_exit", -3.0,
            f"Sharpe {sharpe:.2f} under {goal['min_sharpe']}. "
            "Lower exit.rsi_exit {cur} -> {new} to leave trades earlier and reduce variance.",
            "Shorter holds, steadier equity.")

    # Priority 4 -- everything passed: press the edge, then widen the net.
    add("position_size_r", +0.1,
        f"All bars cleared (return {realised:.2%}, DD {dd:.2%}, Sharpe {sharpe:.2f}). "
        "Raise position_size_r {cur} -> {new} to compound.",
        "Realised return rises; drawdown must stay within the limit.")
    add("entry.threshold", +1.0,
        f"All bars cleared. Raise entry.threshold {{cur}} -> {{new}} to take a wider set of setups.",
        "More trades; return and drawdown both edge up.")
    return out


def choose_change(strategy: dict, detail: dict, goal: dict, blacklist=frozenset()):
    """Return (dotted_var, new_value, reasoning, prediction). Exactly one variable.

    `blacklist`: dotted variable names to skip (already falsified this round).
    """
    for var, new, why, predict in _proposals(strategy, detail, goal):
        if var not in blacklist:
            return var, new, why, predict

    # Everything sensible is blacklisted -- nudge any remaining tunable toward mid-range.
    for var, (lo, hi) in BOUNDS.items():
        if var in blacklist:
            continue
        cur = _get(strategy, var)
        mid = (lo + hi) / 2.0
        new = _clamp(cur + (0.1 if cur < mid else -0.1) * (hi - lo), var)
        if new != cur:
            return var, new, f"Exploratory nudge of {var} {cur} -> {new} (priority levers exhausted).", \
                   "Unknown effect; measured on the validation window."
    raise RuntimeError("no tunable variable left to change")


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


def run_reflection(trades_batch, equity_points, goal, source="deterministic",
                   dry_run=False, chooser=None, fallback=True):
    """Score the batch, choose one change, apply it. Returns (hypothesis, strategy, snapshot_path).

    chooser: callable(strategy, detail, goal, trades_batch) -> (var, new_value, reasoning, prediction).
             Defaults to the deterministic rule in this module. If it raises and
             `fallback` is true, the deterministic rule is used and `source` is
             suffixed with "+fallback".
    """
    strategy = config.load_strategy()
    batch_score, detail = score(trades_batch, equity_points, goal)

    chooser = chooser or (lambda s, d, g, _tb: choose_change(s, d, g))
    try:
        var, new_value, reasoning, prediction = chooser(strategy, detail, goal, trades_batch)
    except Exception as e:
        if not fallback:
            raise
        var, new_value, reasoning, prediction = choose_change(strategy, detail, goal)
        reasoning = f"[fell back to deterministic rule -- {e}] {reasoning}"
        source = f"{source}+fallback"
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
