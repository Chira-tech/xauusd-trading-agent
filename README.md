# Self-learning paper-trading agent — gold/USD

A small, fully local experiment. It paper-trades gold (`GC=F`, free daily data
from Yahoo Finance's public chart endpoint), and after every few closed trades it **reflects** on the batch and
changes **exactly one** strategy variable — tightening a stop, widening an entry,
resizing positions — always snapshotting the previous strategy first.

There is **no live trading**, no cloud deploy, no external LLM, and no network
access beyond downloading gold prices. The learning rule is deterministic and
lives in `agent/reflect.py` — you can read exactly why every change was made.

## Setup (one time)

```powershell
cd "C:\Users\USER\Documents\trading agent"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
python run.py backtest        # learn over ~5 years of gold history — start here
python run.py status          # current strategy, equity, and every reflection
python run.py reflect --dry-run   # see the next proposed change without applying it
python run.py paper           # forward paper-trade the live price (Ctrl-C to stop)
python run.py refresh         # re-download the price history
```

`backtest` starts fresh each time (strategy reset to v01). Use
`python run.py backtest --resume` to keep evolving the current strategy.

## LLM brain (optional) — `--llm`

By default the "which one variable to change" decision comes from the
deterministic rule in `agent/reflect.py`. Add `--llm` to hand that decision to a
language model instead — it reads the batch of trades and the current strategy
and proposes one change, which is still validated against the same bounds and
still logged with its reasoning:

```powershell
python run.py reflect --llm --dry-run   # see the model's proposed change
python run.py backtest --llm            # LLM picks every change across the run
python run.py paper --llm               # LLM drives the live forward test
```

Backends (`--backend`, default `auto`):

- **`cli`** — shells out to the `claude` CLI, using your existing Claude Code
  login. No API key needed. This is the default when `ANTHROPIC_API_KEY` is unset.
- **`api`** — Anthropic API via the `anthropic` package (`pip install anthropic`)
  and `ANTHROPIC_API_KEY`. Model defaults to `claude-opus-5`; override with
  `LLM_MODEL`.

If the model's answer can't be parsed or breaks a guardrail (unknown variable,
value out of range, no actual change), the cycle falls back to the deterministic
rule and the hypothesis is tagged `...+fallback`.

### Pointing `--llm` at Hermes Agent

If you install [Hermes Agent](https://github.com/NousResearch/hermes-agent) and
it exposes a non-interactive CLI, set `CLAUDE_BIN` to that binary (it must accept
a prompt on stdin and print the reply) — `agent/llm_reflect.py` will call it in
place of `claude`. Nothing in this project runs Hermes in an unattended loop or
deploys anything; you invoke each reflection yourself.

## The strategy (`state/strategy.yaml`, starts at v01)

Long-only RSI mean-reversion:

| field | v01 | meaning |
|---|---|---|
| `entry.threshold` | 30 | go long when RSI(14) closes below this |
| `exit.rsi_exit` | 55 | exit when RSI recovers above this |
| `exit.take_profit_pct` | 3.0 | or exit at +3% from entry |
| `stop_loss_pct` | 2.0 | hard stop at −2% from entry |
| `position_size_r` | 0.5 | fraction of equity per trade |

## What success / failure mean (`goal.yaml`)

- **Success:** +10% compounded over a batch, at annualised Sharpe ≥ 1.2
- **Failure:** 5% peak-to-trough equity drawdown
- **Reflection cadence:** every 5 closed trades, one variable changes

## How reflection picks its one change

Priorities, highest first (see `agent/reflect.py:choose_change`):

1. Drawdown breached the 5% limit → tighten `stop_loss_pct`
2. Return under target, win rate ≥ 50% → raise `entry.threshold` (trade more)
3. Return under target, win rate < 50% → lower `exit.take_profit_pct` (bank sooner)
4. Return fine but Sharpe < 1.2 → cut `position_size_r` (smooth the curve)
5. Everything passed → raise `position_size_r` (compound the edge)

Each reflection also **grades its previous hypothesis** (`helped` / `no_change` /
`hurt`) by comparing the new batch score to the old one.

## Files the agent writes (all under `state/`)

| file | contents |
|---|---|
| `strategy.yaml` | current strategy, version bumps each cycle |
| `history/vNN.yaml` | every prior strategy version |
| `trades.jsonl` | every closed paper trade |
| `hypotheses.jsonl` | every reflection: metrics, the change, reasoning, prediction, graded outcome |
| `equity.jsonl` | the full mark-to-market equity curve |
| `heartbeat.json` | last run status |
