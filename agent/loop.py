"""The two run modes: `backtest` (learn over history) and `paper` (forward test live)."""
import json
import time
from datetime import datetime, timezone

from . import config, paths
from .data import get_history, latest_bar
from .indicators import rsi as rsi_calc
from .portfolio import Paper, append_jsonl, read_jsonl
from .reflect import run_reflection
from .strategy import entry_signal, exit_signal


def _heartbeat(status: str, extra: dict | None = None) -> None:
    hb = {"ts": datetime.now(timezone.utc).isoformat(), "status": status}
    if extra:
        hb.update(extra)
    paths.ensure_dirs()
    with open(paths.HEARTBEAT_FILE, "w", encoding="utf-8") as f:
        json.dump(hb, f, indent=2)


def _reset_run_state() -> None:
    for p in (paths.TRADES_FILE, paths.EQUITY_FILE, paths.HYPOTHESES_FILE):
        if p.exists():
            p.unlink()
    if paths.HISTORY.exists():
        for f in paths.HISTORY.glob("*.yaml"):
            f.unlink()
    config.restore_baseline()


def _reflect(batch, equity_pts, goal, n) -> dict:
    hyp, strategy, _ = run_reflection(batch, equity_pts, goal, source="deterministic")
    m = hyp["metrics"]
    print(f"\n  -- reflection #{n}   v{hyp['from_version']} -> v{hyp['to_version']} --")
    print(f"     batch score {hyp['batch_score']:+.3f}    "
          f"return {m['realised_return']:+.2%}   DD {m['max_drawdown']:.2%}   "
          f"Sharpe {m['sharpe']:.2f}   win {m['win_rate']:.0%}")
    print(f"     change : {hyp['variable']}  {hyp['old_value']} -> {hyp['new_value']}")
    print(f"     why    : {hyp['reasoning']}")
    print(f"     predict: {hyp['prediction']}\n")
    return strategy


def _summary(acc: Paper, n_closed: int, n_reflections: int, strategy: dict) -> None:
    tot = acc.equity / acc.start_equity - 1.0
    wins = sum(1 for t in acc.closed if t["return"] > 0)
    print("\n" + "=" * 64)
    print("BACKTEST COMPLETE")
    print(f"  final equity   ${acc.equity:,.2f}   ({tot:+.2%})")
    if n_closed:
        print(f"  trades         {n_closed}   win rate {wins / n_closed:.0%}")
    else:
        print("  trades         0  (no entry conditions fired -- try `python run.py refresh`)")
    print(f"  reflections    {n_reflections}")
    print(f"  strategy       v01 -> v{strategy['version']}")
    print(f"  written        state/trades.jsonl  state/hypotheses.jsonl  state/history/")
    print("=" * 64)


def backtest(goal: dict, fresh: bool = True) -> None:
    if fresh:
        _reset_run_state()
    paths.ensure_dirs()

    bars = get_history(goal)
    if len(bars) < 60:
        raise SystemExit(f"Only {len(bars)} bars of gold data -- need >= 60.")
    closes = [b["close"] for b in bars]
    print(f"Loaded {len(bars)} daily bars  {bars[0]['date']} -> {bars[-1]['date']}\n")

    acc = Paper(goal["starting_equity"])
    cadence = goal["reflection_every"]
    catastrophe = goal["starting_equity"] * (1.0 - 2.0 * goal["max_drawdown"])

    strategy = config.load_strategy()
    rsi_series = rsi_calc(closes, strategy["entry"]["period"])
    cur_period = strategy["entry"]["period"]

    batch, batch_equity = [], []
    n_closed = n_reflections = 0

    for i, bar in enumerate(bars):
        r = rsi_series[i]
        r = None if r != r else float(r)

        eq, dd = acc.mark(bar["date"], bar["close"])
        batch_equity.append({"date": bar["date"], "equity": eq, "drawdown": dd})
        append_jsonl(paths.EQUITY_FILE,
                     {"date": bar["date"], "equity": round(eq, 2), "drawdown": round(dd, 5)})

        if acc.position:
            should, reason, px = exit_signal(strategy, acc.position["entry_price"], bar, r)
            if should:
                trade = acc.close(px, bar["date"], reason)
                append_jsonl(paths.TRADES_FILE, trade)
                batch.append(trade)
                n_closed += 1
                print(f"  trade #{n_closed:>3}  {trade['entry_date']} -> {trade['exit_date']}  "
                      f"{reason:<11}  ret {trade['return']:+.2%}   equity ${trade['equity_after']:,.0f}")

                if len(batch) >= cadence:
                    n_reflections += 1
                    strategy = _reflect(batch, batch_equity, goal, n_reflections)
                    if strategy["entry"]["period"] != cur_period:
                        cur_period = strategy["entry"]["period"]
                        rsi_series = rsi_calc(closes, cur_period)
                    batch, batch_equity = [], []
        elif entry_signal(strategy, r):
            acc.open_long(bar["close"], bar["date"],
                          strategy["position_size_r"], strategy["version"])

        if eq <= catastrophe:
            _heartbeat("halted_catastrophic_drawdown", {"equity": round(eq, 2), "bar": bar["date"]})
            print(f"\n!! equity ${eq:,.0f} hit the catastrophic floor ${catastrophe:,.0f}. Halting.")
            _summary(acc, n_closed, n_reflections, strategy)
            return

    _heartbeat("backtest_complete", {
        "final_equity": round(acc.equity, 2),
        "total_return": round(acc.equity / acc.start_equity - 1.0, 5),
        "trades": n_closed,
        "reflections": n_reflections,
        "final_strategy_version": strategy["version"],
    })
    _summary(acc, n_closed, n_reflections, strategy)


def paper(goal: dict, interval: int = 900) -> None:
    """Forward paper-trade the live gold price. Resumes from state/ if it exists."""
    paths.ensure_dirs()
    strategy = config.load_strategy()

    acc = Paper(goal["starting_equity"])
    existing = read_jsonl(paths.TRADES_FILE)
    for t in existing:
        acc.equity = t["equity_after"]
        acc.peak_equity = max(acc.peak_equity, acc.equity)
    n_closed = len(existing)
    since_reflection = n_closed % goal["reflection_every"]
    batch: list[dict] = []

    hist = get_history(goal)
    closes = [b["close"] for b in hist]
    print(f"[paper] equity ${acc.equity:,.0f}   {n_closed} trades on record   "
          f"polling gold every {interval // 60} min   Ctrl-C to stop")

    while True:
        try:
            bar = latest_bar(goal)
            if bar is None:
                _heartbeat("paper_no_data")
                time.sleep(interval)
                continue
            if bar["date"] != hist[-1]["date"]:
                hist.append(bar)
                closes.append(bar["close"])
            else:
                closes[-1] = bar["close"]

            r = rsi_calc(closes, strategy["entry"]["period"])[-1]
            r = None if r != r else float(r)
            eq, dd = acc.mark(bar["date"], bar["close"])
            stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M")
            append_jsonl(paths.EQUITY_FILE,
                         {"date": stamp, "equity": round(eq, 2), "drawdown": round(dd, 5)})

            if acc.position:
                should, reason, px = exit_signal(strategy, acc.position["entry_price"], bar, r)
                if should:
                    trade = acc.close(px, bar["date"], reason)
                    append_jsonl(paths.TRADES_FILE, trade)
                    batch.append(trade)
                    n_closed += 1
                    since_reflection += 1
                    print(f"[paper] closed #{n_closed}  {reason}  ret {trade['return']:+.2%}  "
                          f"equity ${trade['equity_after']:,.0f}")
                    if since_reflection >= goal["reflection_every"]:
                        strategy = _reflect(batch, acc.equity_points[-max(len(batch) * 3, 10):],
                                            goal, n_closed // goal["reflection_every"])
                        batch = []
                        since_reflection = 0
            elif entry_signal(strategy, r):
                acc.open_long(bar["close"], bar["date"],
                              strategy["position_size_r"], strategy["version"])
                print(f"[paper] opened long @ {bar['close']:.2f}   RSI {r:.1f}")

            _heartbeat("paper_running", {
                "equity": round(eq, 2), "drawdown": round(dd, 5),
                "in_position": acc.position is not None,
                "trades": n_closed, "strategy_version": strategy["version"],
            })
            time.sleep(interval)

        except KeyboardInterrupt:
            _heartbeat("paper_stopped", {"equity": round(acc.equity, 2), "trades": n_closed})
            print("\n[paper] stopped. State saved to state/.")
            return
        except Exception as e:  # keep the loop alive through transient errors
            _heartbeat("paper_error", {"error": str(e)})
            print(f"[paper] error: {e} -- retrying in {interval}s")
            time.sleep(interval)
