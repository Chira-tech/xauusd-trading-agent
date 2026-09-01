"""Progress report over state/experiments.jsonl -- how the strategy is doing
against the held-out validation window as iterations accumulate (local runs and
CI runs both append here).
"""
import json
import os
import time
from datetime import datetime

from . import config, paths

BLOCKS = ".:-=+*#@"  # ASCII ramp, low -> high (encodes on any console)


def _read_jsonl_safe(path):
    """Tolerant read -- skips a half-written trailing line from a live run."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def _sparkline(values):
    if not values:
        return ""
    lo, hi = min(values), max(values)
    rng = (hi - lo) or 1.0
    return "".join(BLOCKS[min(7, int((v - lo) / rng * 7.999))] for v in values)


def _running_val(exps):
    series = []
    for e in exps:
        series.append(e["val_score_after"] if e["action"] == "kept"
                      else e["val_score_before"])
    return series


def _fmt_span(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def build_report(goal, plateau_window=20):
    exps = _read_jsonl_safe(paths.EXPERIMENTS_FILE)
    strat = config.load_strategy()
    baseline = config.BASELINE
    base = None
    try:
        import yaml
        base = yaml.safe_load(open(baseline, encoding="utf-8"))
    except Exception:
        pass

    lines = []
    add = lines.append
    add(f"EVOLUTION REPORT  --  {goal.get('asset', '?')}")
    add("-" * 62)

    if not exps:
        add("No experiments yet. Run:  python run.py evolve --llm --iterations 100")
        return "\n".join(lines)

    ts = [datetime.fromisoformat(e["ts"]) for e in exps if e.get("ts")]
    span = (ts[-1] - ts[0]).total_seconds() if len(ts) > 1 else 0
    series = _running_val(exps)
    start_val = exps[0]["val_score_before"]
    cur_val = series[-1]

    from collections import Counter
    tally = Counter(e["verdict"] for e in exps)
    recent = exps[-plateau_window:]
    recent_sup = sum(1 for e in recent if e["verdict"] == "SUPPORTED")

    best = max(exps, key=lambda e: e["val_score_after"] - e["val_score_before"])
    best_gain = best["val_score_after"] - best["val_score_before"]

    add(f"Iterations run     {len(exps)}"
        + (f"   (span {_fmt_span(span)}, last {ts[-1]:%Y-%m-%d %H:%MZ})" if ts else ""))
    add(f"Strategy version   v01 -> v{strat['version']}")
    add(f"Validation score   {start_val:+.3f} -> {cur_val:+.3f}   (delta {cur_val - start_val:+.3f})")
    add(f"Trajectory         {_sparkline(series)}   (oldest -> newest)")
    add("")
    add(f"Verdicts           " + "   ".join(f"{k} {tally[k]}" for k in
        ("SUPPORTED", "INCONCLUSIVE", "NEUTRAL", "FALSIFIED") if tally.get(k)))
    verdict_read = ("still improving" if recent_sup else "plateaued -- try --llm, "
                    "more --iterations, or widen agent/reflect.py BOUNDS")
    add(f"Last {len(recent):>2} iters      SUPPORTED {recent_sup}  --  {verdict_read}")
    if best_gain > 0:
        add(f"Best iteration     #{best['iter']}  {best['variable']} "
            f"{best['old_value']} -> {best['new_value']}   "
            f"val {best['val_score_before']:+.3f} -> {best['val_score_after']:+.3f} "
            f"({best_gain:+.3f})")

    if base:
        add("")
        add("Strategy now vs baseline v01:")
        for var in ("entry.threshold", "exit.rsi_exit", "exit.take_profit_pct",
                    "stop_loss_pct", "position_size_r"):
            b = _dig(base, var)
            c = _dig(strat, var)
            mark = "" if b == c else "  <-- changed"
            add(f"  {var:<22} {b} -> {c}{mark}" if b != c
                else f"  {var:<22} {c}   (unchanged)")

    add("")
    add("Recent experiments:")
    for e in exps[-8:]:
        add(f"  #{e['iter']:>4}  {e['variable']:<20} {e['old_value']} -> {e['new_value']:<6} "
            f"val {e['val_score_before']:+.3f} -> {e['val_score_after']:+.3f}   "
            f"{e['verdict']} [{e['action']}]")

    add("")
    add("Full ledger: state/experiments.jsonl   |   every version: state/history/")
    return "\n".join(lines)


def _dig(d, dotted):
    for k in dotted.split("."):
        d = d[k]
    return d


def watch(goal, every=5):
    try:
        while True:
            os.system("cls" if os.name == "nt" else "clear")
            print(build_report(goal))
            print(f"\n(refreshing every {every}s -- Ctrl-C to stop)")
            time.sleep(every)
    except KeyboardInterrupt:
        print()


def to_csv(path):
    exps = _read_jsonl_safe(paths.EXPERIMENTS_FILE)
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["iter", "ts", "variable", "old_value", "new_value",
                    "train_score", "val_score_before", "val_score_after",
                    "val_delta", "verdict", "action"])
        for e in exps:
            w.writerow([e.get(k) for k in
                        ("iter", "ts", "variable", "old_value", "new_value",
                         "train_score", "val_score_before", "val_score_after",
                         "val_delta", "verdict", "action")])
    return len(exps)
