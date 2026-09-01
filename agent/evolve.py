"""Continuous, scientific-method strategy evolution.

The history is split into a TRAIN window and a held-out VALIDATION window. The
validation window is never shown to the change-proposing step, so a change only
"counts" if it generalises to data the agent did not learn from.

One iteration:
  1. simulate TRAIN with the current (frozen) strategy -> metrics
  2. propose ONE variable change
       - deterministic rule (skips variables already falsified this round), or
       - an LLM (--llm), which is also shown the recent experiment ledger
  3. build the candidate strategy; score current vs candidate on VALIDATION
  4. verdict from the validation-score delta:
       > +NOISE     -> SUPPORTED     keep; clear the falsified set
       0 .. +NOISE  -> INCONCLUSIVE  keep the small real gain; blacklist the var
       == 0         -> NEUTRAL       revert (no effect, don't accumulate cruft)
       < 0          -> FALSIFIED     revert; blacklist the var
     Nothing that failed to improve the held-out window is ever adopted.
  5. append the experiment to state/experiments.jsonl

Stops after --iterations, or --patience iterations with no SUPPORTED change.
"""
import copy
import json
from datetime import datetime, timezone

import yaml

from . import config, paths
from .portfolio import read_jsonl
from .reflect import BOUNDS, _get, _set, choose_change
from .score import score
from .sim import simulate

NOISE = 0.02  # |validation score delta| at or below this == inconclusive


def _split(bars, val_frac):
    cut = int(len(bars) * (1.0 - val_frac))
    return bars[:cut], bars[cut:]


def _val_score(bars, strategy, goal):
    res = simulate(bars, strategy, goal)
    s, _ = score(res["trades"], res["equity_points"], goal)
    return s


def _snapshot(strategy):
    snap = paths.HISTORY / f"v{strategy['version']}.yaml"
    with open(snap, "w", encoding="utf-8") as f:
        yaml.safe_dump(strategy, f, sort_keys=False)


def _log(row):
    paths.ensure_dirs()
    with open(paths.EXPERIMENTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _reset():
    config.restore_baseline()
    for p in (paths.EXPERIMENTS_FILE, paths.HYPOTHESES_FILE):
        if p.exists():
            p.unlink()
    if paths.HISTORY.exists():
        for f in paths.HISTORY.glob("*.yaml"):
            f.unlink()


def evolve(goal, bars, iterations=100, val_frac=0.2, patience=15,
           use_llm=False, backend=None, fresh=True):
    if fresh:
        _reset()
    paths.ensure_dirs()

    if use_llm:
        from .llm_reflect import LLMReflectError, llm_choose_change

    train, val = _split(bars, val_frac)
    if len(train) < 60 or len(val) < 30:
        raise SystemExit(f"Not enough data: train={len(train)} val={len(val)} bars.")

    strategy = config.load_strategy()
    baseline_val = _val_score(val, strategy, goal)
    print(f"start   v{strategy['version']}   train {len(train)} bars  "
          f"({train[0]['date']}..{train[-1]['date']})   "
          f"val {len(val)} bars  ({val[0]['date']}..{val[-1]['date']})")
    print(f"        baseline validation score {baseline_val:+.3f}\n")

    n_sup = n_fal = n_inc = 0
    since_sup = 0
    falsified_vars: set[str] = set()

    # Continue iteration numbering across runs, and carry forward variables that
    # were already tried and failed against the *current* strategy version so a
    # resumed run (e.g. the CI cron) doesn't just re-test known dead ends.
    prior = [] if fresh else read_jsonl(paths.EXPERIMENTS_FILE)
    start_iter = (max(e["iter"] for e in prior) + 1) if prior else 1
    for e in prior[-len(BOUNDS) * 4:]:
        if e.get("verdict") in ("FALSIFIED", "NEUTRAL") and e.get("from_version") == strategy["version"]:
            falsified_vars.add(e["variable"])

    for it in range(start_iter, start_iter + iterations):
        train_res = simulate(train, strategy, goal)
        if train_res["n_trades"] == 0:
            print(f"iter {it}: strategy takes no trades on the train window; stopping.")
            break
        train_score, detail = score(train_res["trades"], train_res["equity_points"], goal)
        recent = train_res["trades"][-goal["reflection_every"]:]

        # ---- propose one change (validation window is NOT passed in) ----
        try:
            if use_llm:
                try:
                    var, new_value, reasoning, prediction = llm_choose_change(
                        strategy, detail, goal, recent,
                        history=read_jsonl(paths.EXPERIMENTS_FILE))
                except LLMReflectError as e:
                    var, new_value, reasoning, prediction = choose_change(
                        strategy, detail, goal, blacklist=falsified_vars)
                    reasoning = f"[LLM failed: {e}; used rule] {reasoning}"
            else:
                var, new_value, reasoning, prediction = choose_change(
                    strategy, detail, goal, blacklist=falsified_vars)
        except RuntimeError as e:
            print(f"\niter {it}: {e} -- every tunable variable has been falsified. Stopping.")
            break

        old_value = _get(strategy, var)

        # ---- controlled experiment on the held-out window ----
        candidate = copy.deepcopy(strategy)
        _set(candidate, var, new_value)
        candidate["version"] = f"{int(strategy['version']) + 1:02d}"

        val_before = _val_score(val, strategy, goal)
        val_after = _val_score(val, candidate, goal)
        delta = val_after - val_before

        if delta > NOISE:
            verdict, action = "SUPPORTED", "kept"
            n_sup += 1
            since_sup = 0
            falsified_vars.clear()
            _snapshot(strategy)
            config.save_strategy(candidate)
            strategy = candidate
        elif delta > 0.0:
            verdict, action = "INCONCLUSIVE", "kept"
            n_inc += 1
            since_sup += 1
            falsified_vars.add(var)  # small real gain banked; stop retrying it
            _snapshot(strategy)
            config.save_strategy(candidate)
            strategy = candidate
        elif delta == 0.0:
            verdict, action = "NEUTRAL", "reverted"  # no measurable effect -- don't accumulate
            n_inc += 1
            since_sup += 1
            falsified_vars.add(var)
        else:
            verdict, action = "FALSIFIED", "reverted"
            n_fal += 1
            since_sup += 1
            falsified_vars.add(var)

        _log({
            "ts": datetime.now(timezone.utc).isoformat(),
            "iter": it,
            "from_version": f"{int(candidate['version']) - 1:02d}",
            "to_version": candidate["version"] if action != "reverted" else None,
            "variable": var, "old_value": old_value, "new_value": new_value,
            "train_score": round(train_score, 4),
            "val_score_before": round(val_before, 4),
            "val_score_after": round(val_after, 4),
            "val_delta": round(delta, 4),
            "verdict": verdict, "action": action,
            "source": "llm" if use_llm else "rule",
            "reasoning": reasoning, "prediction": prediction,
        })
        print(f"iter {it:>3}  {var:<20} {old_value} -> {new_value:<6}  "
              f"train {train_score:+.3f}  val {val_before:+.3f} -> {val_after:+.3f} "
              f"(d {delta:+.3f})  {verdict} [{action}]")

        if since_sup >= patience:
            print(f"\n{patience} iterations with no supported change -- converged.")
            break

    final_val = _val_score(val, strategy, goal)
    print(f"\ndone    v{strategy['version']}   validation score "
          f"{baseline_val:+.3f} -> {final_val:+.3f}")
    print(f"        supported {n_sup}   falsified {n_fal}   inconclusive {n_inc}")
    print(f"        current strategy: state/strategy.yaml   ledger: state/experiments.jsonl")
