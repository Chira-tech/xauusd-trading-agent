"""Frozen forward test.

`freeze` snapshots the current strategy and locks it. `step` then runs that
locked strategy against bars that arrive *after* the freeze -- on paper, with a
modelled spread + slippage cost on every fill. No learning happens here; this is
the honest out-of-sample check on whatever `evolve` produced.

State lives in state/forward/:
  config.yaml    the frozen strategy + cost model + the validation score it had
  trades.jsonl   forward paper trades
  equity.jsonl   forward mark-to-market equity curve
  status.json    last bar processed, open position, running equity
"""
import json
import os
from datetime import datetime, timezone

import yaml

from . import config, paths
from .data import get_history
from .indicators import rsi as rsi_calc
from .portfolio import Paper, append_jsonl, read_jsonl
from .score import score as score_batch
from .strategy import entry_signal, exit_signal

FWD = paths.STATE / "forward"
FCONFIG = FWD / "config.yaml"
FTRADES = FWD / "trades.jsonl"
FEQUITY = FWD / "equity.jsonl"
FSTATUS = FWD / "status.json"

DEFAULT_SPREAD_BPS = 2.0
DEFAULT_SLIPPAGE_BPS = 1.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _bar_key(iso_ts: str) -> str:
    """Normalise an ISO timestamp to the 'YYYY-MM-DD HH:MM:SS' shape bar dates use."""
    return iso_ts[:19].replace("T", " ")


def _read_status() -> dict:
    return json.loads(FSTATUS.read_text(encoding="utf-8"))


def _write_status(st: dict) -> None:
    FSTATUS.write_text(json.dumps(st, indent=2), encoding="utf-8")


def freeze(goal: dict, interval: str = "1h",
           spread_bps: float = DEFAULT_SPREAD_BPS,
           slippage_bps: float = DEFAULT_SLIPPAGE_BPS) -> dict:
    FWD.mkdir(parents=True, exist_ok=True)
    strat = config.load_strategy()

    # the most recent held-out validation score this strategy earned in evolve
    val_expectation = None
    for e in reversed(read_jsonl(paths.EXPERIMENTS_FILE)):
        if e.get("action") == "kept":
            val_expectation = e.get("val_score_after")
            break

    cfg = {
        "frozen_at": _now_iso(),
        "asset": goal.get("asset"),
        "interval": interval,
        "starting_equity": goal["starting_equity"],
        "costs": {"spread_bps": spread_bps, "slippage_bps": slippage_bps},
        "validation_expectation": val_expectation,
        "goal": {k: goal[k] for k in
                 ("target_return_30d", "max_drawdown", "min_sharpe", "failure_below")},
        "strategy": strat,
    }
    # Anchor the start to the newest *closed* bar available right now, not the
    # wall clock -- so the forward test only ever sees bars that close later,
    # regardless of any clock skew between here and the data provider.
    os.environ["TWELVEDATA_INTERVAL"] = interval
    os.environ.setdefault("TWELVEDATA_OUTPUTSIZE", "500")
    bars = get_history(goal, use_cache=False, refresh=True)
    anchor = bars[-1]["date"] if bars else _bar_key(cfg["frozen_at"])

    FCONFIG.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    FTRADES.write_text("", encoding="utf-8")
    FEQUITY.write_text("", encoding="utf-8")
    _write_status({
        "frozen_at": cfg["frozen_at"],
        "anchor_bar": anchor,
        "last_bar": anchor,
        "last_step": None,
        "equity": goal["starting_equity"],
        "in_position": False,
        "open_position": None,
        "trades": 0,
    })
    return cfg


def _load_cfg() -> dict:
    if not FCONFIG.exists():
        raise SystemExit("No forward test yet -- run `python run.py forward-freeze` first.")
    return yaml.safe_load(FCONFIG.read_text(encoding="utf-8"))


def step(goal: dict) -> None:
    cfg = _load_cfg()
    strat = cfg["strategy"]
    fee = (cfg["costs"]["spread_bps"] + cfg["costs"]["slippage_bps"]) / 10_000.0

    os.environ["TWELVEDATA_INTERVAL"] = cfg["interval"]
    os.environ.setdefault("TWELVEDATA_OUTPUTSIZE", "500")
    bars = get_history(goal, use_cache=False, refresh=True)
    if not bars:
        raise SystemExit("[forward] no bars returned from the data source")

    st = _read_status()
    cutoff = st.get("last_bar") or _bar_key(cfg["frozen_at"])
    new = [b for b in bars if b["date"] > cutoff]
    if not new:
        st["last_step"] = _now_iso()
        _write_status(st)
        print(f"[forward] no new bars since {cutoff}")
        return

    closes = [b["close"] for b in bars]
    rsi_all = rsi_calc(closes, strat["entry"]["period"])
    idx = {b["date"]: i for i, b in enumerate(bars)}

    acc = Paper(cfg["starting_equity"])
    for t in read_jsonl(FTRADES):
        acc.equity = t["equity_after"]
        acc.peak_equity = max(acc.peak_equity, acc.equity)
    if st.get("open_position"):
        acc.position = st["open_position"]

    for b in new:
        r = rsi_all[idx[b["date"]]]
        r = None if r != r else float(r)
        eq, dd = acc.mark(b["date"], b["close"])
        append_jsonl(FEQUITY, {"date": b["date"], "equity": round(eq, 2),
                               "drawdown": round(dd, 5)})

        if acc.position:
            should, reason, px = exit_signal(strat, acc.position["entry_price"], b, r)
            if should:
                trade = acc.close(px * (1.0 - fee), b["date"], reason)  # sell pays the cost
                append_jsonl(FTRADES, trade)
                print(f"[forward] EXIT  {b['date']}  {reason:<11} {trade['return']:+.2%}  "
                      f"equity ${trade['equity_after']:,.0f}")
        elif entry_signal(strat, r):
            acc.open_long(b["close"] * (1.0 + fee), b["date"],
                          strat["position_size_r"], strat["version"])
            print(f"[forward] ENTER {b['date']}  @ {b['close'] * (1 + fee):.2f}  RSI {r:.1f}")

    st.update({
        "last_bar": new[-1]["date"],
        "last_step": _now_iso(),
        "equity": round(acc.equity, 2),
        "in_position": acc.position is not None,
        "open_position": acc.position,
        "trades": len(read_jsonl(FTRADES)),
    })
    _write_status(st)
    print(f"[forward] processed {len(new)} bar(s) up to {new[-1]['date']}  "
          f"equity ${acc.equity:,.2f}")


def report(goal: dict) -> str:
    cfg = _load_cfg()
    trades = read_jsonl(FTRADES)
    equity = read_jsonl(FEQUITY)
    st = _read_status()

    frozen = datetime.fromisoformat(cfg["frozen_at"])
    elapsed_days = (datetime.now(timezone.utc) - frozen).total_seconds() / 86400.0

    L = [f"FORWARD TEST  --  {cfg.get('asset', '?')}", "-" * 62]
    L.append(f"Frozen at       {cfg['frozen_at']}   ({elapsed_days:.1f} days ago)")
    L.append(f"Start bar       {st.get('anchor_bar')}   (forward test sees bars after this)")
    L.append(f"Interval        {cfg['interval']}   "
             f"cost {cfg['costs']['spread_bps']}+{cfg['costs']['slippage_bps']} bps/fill")
    L.append(f"Strategy        v{cfg['strategy']['version']}  "
             f"(RSI<{cfg['strategy']['entry']['threshold']}, "
             f"stop {cfg['strategy']['stop_loss_pct']}%, "
             f"size {cfg['strategy']['position_size_r']})")
    L.append(f"Last bar        {st.get('last_bar')}   "
             f"({'in a position' if st.get('in_position') else 'flat'})")
    L.append("")

    if not trades:
        L.append("No forward trades yet -- the strategy has not fired since the freeze.")
        L.append(f"Equity holds at ${cfg['starting_equity']:,.2f}.")
        return "\n".join(L)

    fwd_score, d = score_batch(trades, equity, {**goal, **cfg["goal"]})
    wins = sum(1 for t in trades if t["return"] > 0)
    total_ret = st["equity"] / cfg["starting_equity"] - 1.0

    L.append(f"Forward trades  {len(trades)}   win rate {wins / len(trades):.0%}")
    L.append(f"Forward return  {total_ret:+.2%}   equity ${st['equity']:,.2f}")
    L.append(f"Max drawdown    {d['max_drawdown']:.2%}   (limit {cfg['goal']['max_drawdown']:.0%})")
    L.append(f"Sharpe          {d['sharpe']:.2f}   (min {cfg['goal']['min_sharpe']})")
    L.append("")
    exp = cfg.get("validation_expectation")
    L.append(f"Score at freeze (held-out validation) : "
             f"{exp:+.3f}" if exp is not None else "Score at freeze (validation) : n/a")
    L.append(f"Score live (same formula, real time)  : {fwd_score:+.3f}")
    if exp is not None:
        gap = fwd_score - exp
        verdict = ("live is tracking the backtest" if gap > -0.15
                   else "live is UNDERperforming the backtest -- likely overfit")
        L.append(f"Gap                                  : {gap:+.3f}   ({verdict})")
    L.append("")
    L.append("Trades:")
    for t in trades[-10:]:
        L.append(f"  {t['entry_date']} -> {t['exit_date']}  {t['reason']:<11} "
                 f"{t['return']:+.2%}  equity ${t['equity_after']:,.0f}")
    L.append("")
    L.append("Ledger: state/forward/trades.jsonl   ·   config: state/forward/config.yaml")
    return "\n".join(L)
