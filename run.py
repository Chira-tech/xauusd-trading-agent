"""Self-learning paper-trading agent for gold/USD.

    python run.py evolve          continuous scientific-method loop: train/validation
                                  split, one change per iteration, keep or revert on
                                  the held-out validation score (this is the one you
                                  want for "keep improving the strategy")
    python run.py evolve --llm --iterations 200
    python run.py backtest        single chronological pass, strategy adapts as it goes
    python run.py backtest --llm      let an LLM pick each one-variable change
    python run.py paper           forward paper-trade the live price on an interval
    python run.py reflect         force one reflection cycle now (deterministic rule)
    python run.py reflect --llm       let an LLM propose the change
    python run.py reflect --dry-run   show the proposed change without applying it
    python run.py status          strategy, equity, reflection + experiment ledger
    python run.py refresh         re-download the gold price history

Paper mode only. Nothing here can place a real order. --llm uses the `claude`
CLI by default (your existing login, no API key); set --backend api to use the
Anthropic API via ANTHROPIC_API_KEY instead.
"""
import argparse
import json
import os
import sys

try:  # make unicode in --llm reasoning safe on legacy Windows consoles
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from agent import config, forward, paths
from agent.data import get_history
from agent.evolve import evolve
from agent.loop import backtest, paper
from agent.portfolio import read_jsonl
from agent.reflect import run_reflection


def _llm_chooser(args):
    """Return the LLM chooser callable if --llm was passed, else None."""
    if not getattr(args, "llm", False):
        return None
    if getattr(args, "backend", None):
        os.environ["LLM_BACKEND"] = args.backend
    from agent.llm_reflect import llm_choose_change
    return llm_choose_change


def cmd_backtest(args):
    goal = config.load_goal()
    brain = "LLM" if args.llm else "deterministic rule"
    print(f"gold/USD self-learning agent  |  target +{goal['target_return_30d']:.0%}/30d  "
          f"max DD {goal['max_drawdown']:.0%}  min Sharpe {goal['min_sharpe']}  "
          f"reflect every {goal['reflection_every']} trades  |  brain: {brain}\n")
    backtest(goal, fresh=not args.resume, chooser=_llm_chooser(args))


def cmd_paper(args):
    paper(config.load_goal(), interval=args.interval, chooser=_llm_chooser(args))


def cmd_evolve(args):
    goal = config.load_goal()
    if args.backend:
        os.environ["LLM_BACKEND"] = args.backend
    bars = get_history(goal)
    brain = "LLM" if args.llm else "deterministic rule"
    print(f"evolve  |  {goal['asset']}  |  brain: {brain}  |  "
          f"val {int(args.val_frac * 100)}% held out  |  patience {args.patience}\n")
    evolve(goal, bars, iterations=args.iterations, val_frac=args.val_frac,
           patience=args.patience, use_llm=args.llm, fresh=not args.resume)


def cmd_reflect(args):
    goal = config.load_goal()
    trades = read_jsonl(paths.TRADES_FILE)
    if not trades:
        raise SystemExit("No trades logged yet -- run `python run.py backtest` first.")
    batch = trades[-goal["reflection_every"]:]
    equity = read_jsonl(paths.EQUITY_FILE)
    hyp, _, _ = run_reflection(batch, equity[-len(batch) * 3:], goal,
                               source="manual-llm" if args.llm else "manual",
                               dry_run=args.dry_run, chooser=_llm_chooser(args),
                               fallback=not args.no_fallback)
    print(json.dumps(hyp, indent=2))
    if args.dry_run:
        print("\n(dry run -- strategy.yaml was not modified)")


def cmd_status(args):
    goal = config.load_goal()
    strat = config.load_strategy()
    trades = read_jsonl(paths.TRADES_FILE)
    hyps = read_jsonl(paths.HYPOTHESES_FILE)

    print(f"STRATEGY  v{strat['version']}")
    print(json.dumps(strat, indent=2))
    print(f"\nTRADES  {len(trades)} closed")
    if trades:
        eq = trades[-1]["equity_after"]
        wins = sum(1 for t in trades if t["return"] > 0)
        print(f"  equity ${eq:,.2f}   total {eq / goal['starting_equity'] - 1:+.2%}   "
              f"win rate {wins / len(trades):.0%}")
    print(f"\nREFLECTIONS  {len(hyps)}")
    for h in hyps[-8:]:
        print(f"  v{h['from_version']}->v{h['to_version']}  "
              f"{h['variable']}: {h['old_value']} -> {h['new_value']}   "
              f"score {h['batch_score']:+.3f}   outcome={h.get('outcome')}")

    exps = read_jsonl(paths.EXPERIMENTS_FILE)
    if exps:
        from collections import Counter
        tally = Counter(e["verdict"] for e in exps)
        print(f"\nEXPERIMENTS  {len(exps)} run   "
              f"SUPPORTED {tally['SUPPORTED']}  INCONCLUSIVE {tally['INCONCLUSIVE']}  "
              f"NEUTRAL {tally['NEUTRAL']}  FALSIFIED {tally['FALSIFIED']}")
        for e in exps[-8:]:
            print(f"  #{e['iter']:>3}  {e['variable']}: {e['old_value']} -> {e['new_value']}   "
                  f"val {e['val_score_before']:+.3f} -> {e['val_score_after']:+.3f}   "
                  f"{e['verdict']} [{e['action']}]")


def cmd_refresh(args):
    goal = config.load_goal()
    bars = get_history(goal, use_cache=False, refresh=True)
    print(f"Refreshed {len(bars)} bars of {goal['asset']}  "
          f"{bars[0]['date']} -> {bars[-1]['date']}")


def cmd_forward_freeze(args):
    goal = config.load_goal()
    cfg = forward.freeze(goal, interval=args.interval,
                         spread_bps=args.spread_bps, slippage_bps=args.slippage_bps)
    print(f"Froze strategy v{cfg['strategy']['version']} for a forward test.")
    print(f"  interval {cfg['interval']}   cost {args.spread_bps}+{args.slippage_bps} bps/fill")
    print(f"  validation score at freeze: {cfg['validation_expectation']}")
    print(f"  clock starts now: {cfg['frozen_at']}")
    print("Run `python run.py forward-step` on a schedule (the CI workflow does this hourly).")


def cmd_forward_step(args):
    forward.step(config.load_goal())


def cmd_forward_report(args):
    print(forward.report(config.load_goal()))


def cmd_report(args):
    from agent import report
    goal = config.load_goal()
    if args.csv:
        n = report.to_csv(args.csv)
        print(f"wrote {n} rows to {args.csv}")
        return
    if args.watch:
        report.watch(goal, every=args.watch)
        return
    print(report.build_report(goal))


def main():
    p = argparse.ArgumentParser(prog="gold-agent", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_llm_flags(sp):
        sp.add_argument("--llm", action="store_true",
                        help="let an LLM choose the one-variable change")
        sp.add_argument("--backend", choices=("auto", "cli", "api"),
                        help="LLM backend for --llm (default auto: api if "
                             "ANTHROPIC_API_KEY set, else the claude CLI)")

    e = sub.add_parser("evolve", help="continuous train/validation evolution loop")
    e.add_argument("--iterations", type=int, default=100,
                   help="max iterations (default 100)")
    e.add_argument("--val-frac", type=float, default=0.2, dest="val_frac",
                   help="fraction of history held out for validation (default 0.2)")
    e.add_argument("--patience", type=int, default=15,
                   help="stop after this many iterations with no supported change")
    e.add_argument("--resume", action="store_true",
                   help="keep the current strategy + ledger instead of resetting to v01")
    add_llm_flags(e)
    e.set_defaults(func=cmd_evolve)

    b = sub.add_parser("backtest", help="learn over historical gold data")
    b.add_argument("--resume", action="store_true",
                   help="keep existing trades/strategy instead of starting fresh")
    add_llm_flags(b)
    b.set_defaults(func=cmd_backtest)

    pp = sub.add_parser("paper", help="forward paper-trade the live price")
    pp.add_argument("--interval", type=int, default=900,
                    help="seconds between polls (default 900)")
    add_llm_flags(pp)
    pp.set_defaults(func=cmd_paper)

    r = sub.add_parser("reflect", help="force one reflection cycle now")
    r.add_argument("--dry-run", action="store_true",
                   help="show the proposed change without applying it")
    r.add_argument("--no-fallback", action="store_true",
                   help="with --llm, fail instead of falling back to the rule")
    add_llm_flags(r)
    r.set_defaults(func=cmd_reflect)

    s = sub.add_parser("status", help="print strategy, equity, reflection log")
    s.set_defaults(func=cmd_status)

    rp = sub.add_parser("report", help="progress report over the experiment ledger")
    rp.add_argument("--watch", type=int, nargs="?", const=5, default=0,
                    metavar="SEC", help="redraw every SEC seconds (default 5)")
    rp.add_argument("--csv", metavar="PATH",
                    help="write the validation-score trajectory to a CSV and exit")
    rp.set_defaults(func=cmd_report)

    rf = sub.add_parser("refresh", help="re-download the gold price history")
    rf.set_defaults(func=cmd_refresh)

    ff = sub.add_parser("forward-freeze",
                        help="lock the current strategy and start a forward test")
    ff.add_argument("--interval", default="1h", help="bar interval (default 1h)")
    ff.add_argument("--spread-bps", type=float, default=forward.DEFAULT_SPREAD_BPS,
                    dest="spread_bps", help="modelled spread per fill (default 2)")
    ff.add_argument("--slippage-bps", type=float, default=forward.DEFAULT_SLIPPAGE_BPS,
                    dest="slippage_bps", help="modelled slippage per fill (default 1)")
    ff.set_defaults(func=cmd_forward_freeze)

    fs = sub.add_parser("forward-step",
                        help="process new bars against the frozen strategy")
    fs.set_defaults(func=cmd_forward_step)

    fr = sub.add_parser("forward-report",
                        help="forward-test results vs the backtest expectation")
    fr.set_defaults(func=cmd_forward_report)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
