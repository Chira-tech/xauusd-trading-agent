"""Self-learning paper-trading agent for gold/USD.

    python run.py backtest        learn over historical gold data (start here)
    python run.py backtest --resume   continue from the current strategy/trades
    python run.py paper           forward paper-trade the live price on an interval
    python run.py reflect         force one reflection cycle now
    python run.py reflect --dry-run   show the proposed change without applying it
    python run.py status          print current strategy, equity, reflection log
    python run.py refresh         re-download the gold price history

Paper mode only. Nothing here can place a real order.
"""
import argparse
import json

from agent import config, paths
from agent.data import get_history
from agent.loop import backtest, paper
from agent.portfolio import read_jsonl
from agent.reflect import run_reflection


def cmd_backtest(args):
    goal = config.load_goal()
    print(f"gold/USD self-learning agent  |  target +{goal['target_return_30d']:.0%}/30d  "
          f"max DD {goal['max_drawdown']:.0%}  min Sharpe {goal['min_sharpe']}  "
          f"reflect every {goal['reflection_every']} trades\n")
    backtest(goal, fresh=not args.resume)


def cmd_paper(args):
    paper(config.load_goal(), interval=args.interval)


def cmd_reflect(args):
    goal = config.load_goal()
    trades = read_jsonl(paths.TRADES_FILE)
    if not trades:
        raise SystemExit("No trades logged yet -- run `python run.py backtest` first.")
    batch = trades[-goal["reflection_every"]:]
    equity = read_jsonl(paths.EQUITY_FILE)
    hyp, _, _ = run_reflection(batch, equity[-len(batch) * 3:], goal,
                               source="manual", dry_run=args.dry_run)
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


def cmd_refresh(args):
    goal = config.load_goal()
    bars = get_history(goal, use_cache=False, refresh=True)
    print(f"Refreshed {len(bars)} bars of {goal['asset']}  "
          f"{bars[0]['date']} -> {bars[-1]['date']}")


def main():
    p = argparse.ArgumentParser(prog="gold-agent", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("backtest", help="learn over historical gold data")
    b.add_argument("--resume", action="store_true",
                   help="keep existing trades/strategy instead of starting fresh")
    b.set_defaults(func=cmd_backtest)

    pp = sub.add_parser("paper", help="forward paper-trade the live price")
    pp.add_argument("--interval", type=int, default=900,
                    help="seconds between polls (default 900)")
    pp.set_defaults(func=cmd_paper)

    r = sub.add_parser("reflect", help="force one reflection cycle now")
    r.add_argument("--dry-run", action="store_true",
                   help="show the proposed change without applying it")
    r.set_defaults(func=cmd_reflect)

    s = sub.add_parser("status", help="print strategy, equity, reflection log")
    s.set_defaults(func=cmd_status)

    rf = sub.add_parser("refresh", help="re-download the gold price history")
    rf.set_defaults(func=cmd_refresh)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
