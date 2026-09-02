# Self-learning paper-trading agent — XAU/USD (spot gold)

A paper-trading experiment that improves its own strategy under the scientific
method. It splits price history into a **train** window and a held-out
**validation** window, proposes **one** variable change at a time, and keeps a
change only if it improves the validation score — reverting anything that
doesn't. Every experiment is logged with its reasoning and verdict.

There is **no live trading**. The change decision comes from a deterministic
rule (`agent/reflect.py`) by default, or a language model with `--llm` (still
guardrailed and logged).

## Setup (one time)

```powershell
cd "C:\Users\USER\Documents\trading agent"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env       # then paste your Twelve Data API key into .env
```

## Data source

Set by `DATA_SOURCE` in `.env` (overrides `goal.yaml`):

| value | source | notes |
|---|---|---|
| `twelvedata` | `api.twelvedata.com` | needs `TWELVEDATA_API_KEY`; real `XAU/USD` spot, daily bars back to 2008 + intraday; free tier 800 calls/day |
| `yahoo` | Yahoo chart JSON | keyless, no signup; `GC=F` futures, ~5 years daily |
| `yfinance` | `yfinance` package | `pip install yfinance` |

Extra `.env` knobs for Twelve Data: `TWELVEDATA_SYMBOL` (default `XAU/USD`),
`TWELVEDATA_INTERVAL` (`1day`, `1h`, …), `TWELVEDATA_OUTPUTSIZE` (default 5000).

## Run

```powershell
python run.py evolve          # continuous scientific-method loop — THIS is the one
python run.py evolve --llm --iterations 200
python run.py status          # strategy, equity, reflection log + experiment ledger
python run.py backtest        # single chronological pass (diagnostic; resets to v01)
python run.py reflect --dry-run   # see the next proposed change without applying it
python run.py paper           # forward paper-trade the live price (Ctrl-C to stop)
python run.py refresh         # re-download the price history
```

## `evolve` — continuous improvement under the scientific method

`evolve` splits the price history into a **train** window and a held-out
**validation** window (last 20% by default). Each iteration:

1. simulate the train window with the current, frozen strategy
2. propose **one** variable change — the deterministic rule (which skips
   variables already falsified this round) or, with `--llm`, a model that also
   sees the experiment ledger
3. score the current strategy vs the candidate **on the validation window**
4. **verdict** from the validation-score change:
   - clearly better → `SUPPORTED`, keep it, bump the version
   - small real gain → `INCONCLUSIVE`, keep, stop retrying that variable
   - no measurable effect → `NEUTRAL`, **revert** (don't accumulate cruft)
   - worse → `FALSIFIED`, **revert**, stop retrying that variable
5. append the experiment to `state/experiments.jsonl`

Nothing that fails to improve the held-out window is ever adopted, so
improvements have to generalise rather than overfit. The loop stops after
`--iterations`, or after `--patience` iterations with no supported change.
`evolve` resets to v01 each run; use `--resume` to keep evolving the current
strategy and ledger.

The deterministic rule has a small, fixed set of hypotheses and converges in a
handful of iterations. `--llm` is where "keep improving for hundreds of
iterations" actually happens — it varies its hypotheses and learns from the
falsified ones in the ledger.

`backtest` is a different, single chronological pass where the strategy adapts as
it goes; it resets `strategy.yaml` to v01 unless you pass `--resume`. Treat
`evolve` as the source of truth for the strategy.

### Hands-off continuous evolution (GitHub Actions)

`.github/workflows/evolve.yml` runs `evolve --resume` every 6 hours on GitHub's
runners and commits the updated `state/strategy.yaml` + `state/experiments.jsonl`
back to the repo — so the strategy keeps improving with no machine of yours left
on, and every change is a reviewable commit. One-time setup:

1. Repo **Settings → Secrets and variables → Actions → New repository secret**:
   `TWELVEDATA_API_KEY` = your key. (Without it the workflow falls back to the
   keyless Yahoo feed.)
2. **Actions** tab → enable workflows if prompted.
3. Trigger a first run from the **Actions → evolve → Run workflow** button, or
   wait for the schedule.

The CI run uses the deterministic rule. To use `--llm` there, add an
`ANTHROPIC_API_KEY` secret and change the `evolve` step to
`python run.py evolve --resume --llm --backend api`.

## Forward test — the honest out-of-sample check

`evolve` optimises against history, so a great score there can still be
overfit. The forward test settles it: **freeze one strategy, then only ever run
it against bars that close *after* the freeze.**

```powershell
python run.py forward-freeze          # lock the current strategy, start the clock
python run.py forward-step            # process new bars (CI does this hourly)
python run.py forward-report          # results vs the backtest expectation
```

- Runs on **hourly** `XAU/USD` bars with a modelled cost (2 bps spread + 1 bps
  slippage) on every fill, so the numbers aren't rosier than reality.
- `forward-freeze` anchors the start to the newest closed bar at that moment,
  and records the strategy's held-out validation score for comparison.
- `forward-report` prints **score at freeze vs score live** — if live is far
  below, the backtest was overfit.
- Nothing here learns. `evolve` keeps running separately on history; you decide
  when to `forward-freeze` a newer version.

**Hands-off:** `.github/workflows/forwardtest.yml` runs `forward-step` hourly and
commits `state/forward/`. It needs the `TWELVEDATA_API_KEY` repo secret (hourly
gold data) and fails loudly without it.

State: `state/forward/config.yaml` (the frozen strategy + costs),
`trades.jsonl`, `equity.jsonl`, `status.json`.

A meaningful read needs ~20+ trades; on hourly bars that is weeks, not days.

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
