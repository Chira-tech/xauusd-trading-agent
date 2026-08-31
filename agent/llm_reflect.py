"""LLM-driven reflection.

Instead of the deterministic rule in reflect.py, an LLM reads the batch of trades
and proposes the single strategy change. The SAME guardrails still apply: exactly
one variable, and its new value must sit inside the bounds in reflect.BOUNDS.
If the model's answer breaks a rule or cannot be parsed this raises
LLMReflectError, and run_reflection falls back to the deterministic rule.

Backend (env LLM_BACKEND, default "auto"):
  cli   shell out to the `claude` CLI -- uses your existing Claude Code login,
        no API key required
  api   Anthropic API via the `anthropic` package + ANTHROPIC_API_KEY
  auto  api when ANTHROPIC_API_KEY is set and `anthropic` importable, else cli
"""
import json
import os
import re
import shutil
import subprocess

from .reflect import BOUNDS, _get

MODEL = os.environ.get("LLM_MODEL", "claude-opus-5")
_CLI_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "180"))


class LLMReflectError(RuntimeError):
    """The LLM response was unusable; caller should fall back to the rule."""


def _build_prompt(strategy: dict, detail: dict, goal: dict, trades_batch: list[dict]) -> str:
    trade_lines = "\n".join(
        f'  {t["entry_date"]} -> {t["exit_date"]}  {t["reason"]:<11} '
        f'ret {t["return"]:+.4f}  equity ${t["equity_after"]:.0f}'
        for t in trades_batch
    )
    tunable = "\n".join(
        f"  {k}: currently {_get(strategy, k)}, allowed range {lo} .. {hi}"
        for k, (lo, hi) in BOUNDS.items()
    )
    allowed = ", ".join(BOUNDS)
    return f"""You are the learning loop of a PAPER-trading agent for gold/USD. You \
review a batch of closed trades and change EXACTLY ONE strategy variable so the \
next batch moves toward the goal. Paper trading only -- no real orders.

GOAL (goal.yaml):
  target return per batch : +{goal['target_return_30d']:.1%}
  max drawdown (failure)  : {goal['max_drawdown']:.1%}
  min Sharpe              : {goal['min_sharpe']}

CURRENT STRATEGY v{strategy['version']} -- RSI mean-reversion, long only:
{json.dumps(strategy, indent=2)}

TUNABLE VARIABLES (change ONE; the new value must stay in range):
{tunable}

THIS BATCH -- {detail['n_trades']} closed trades:
{trade_lines}

BATCH METRICS:
  realised return : {detail['realised_return']:+.4f}
  max drawdown    : {detail['max_drawdown']:.4f}
  Sharpe          : {detail['sharpe']}
  win rate        : {detail['win_rate']:.2f}

Pick the single highest-impact one-variable change. Respond with ONLY a JSON \
object on one line -- no prose, no code fence:
{{"variable": "<one of: {allowed}>", "new_value": <number>, "reasoning": "<why, <=300 chars>", "prediction": "<expected effect next batch, <=200 chars>"}}
"""


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise LLMReflectError(f"no JSON object in LLM reply: {text[:200]!r}")
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise LLMReflectError(f"bad JSON from LLM ({e}): {m.group(0)[:200]!r}")


def _find_claude() -> str | None:
    override = os.environ.get("CLAUDE_BIN")
    if override:
        return override
    for name in ("claude.cmd", "claude.exe", "claude"):
        found = shutil.which(name)
        if found:
            return found
    return None


def _call_cli(prompt: str) -> str:
    exe = _find_claude()
    if not exe:
        raise LLMReflectError("`claude` CLI not found on PATH -- set CLAUDE_BIN, "
                              "or use --backend api with ANTHROPIC_API_KEY")
    argv = [exe, "-p", "--output-format", "json"]
    if exe.lower().endswith((".cmd", ".bat")):
        argv = ["cmd", "/c", *argv]  # Windows batch shims need a cmd wrapper
    try:
        proc = subprocess.run(
            argv, input=prompt, capture_output=True, text=True, timeout=_CLI_TIMEOUT,
        )
    except FileNotFoundError:
        raise LLMReflectError("`claude` CLI not found on PATH")
    except subprocess.TimeoutExpired:
        raise LLMReflectError(f"`claude` CLI timed out after {_CLI_TIMEOUT}s")
    if proc.returncode != 0:
        raise LLMReflectError(f"`claude` CLI exited {proc.returncode}: {proc.stderr[:200]}")
    try:
        env = json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise LLMReflectError(f"`claude` CLI gave a non-JSON envelope: {proc.stdout[:200]!r}")
    if env.get("is_error"):
        raise LLMReflectError(f"`claude` CLI reported an error: {env.get('result')!r}")
    return env.get("result", "")


def _call_api(prompt: str) -> str:
    try:
        import anthropic
    except ImportError:
        raise LLMReflectError("anthropic package not installed (pip install anthropic)")
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=MODEL, max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")


def _pick_backend() -> str:
    choice = os.environ.get("LLM_BACKEND", "auto").lower()
    if choice in ("cli", "api"):
        return choice
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic  # noqa: F401
            return "api"
        except ImportError:
            pass
    return "cli"


def llm_choose_change(strategy: dict, detail: dict, goal: dict, trades_batch: list[dict]):
    """Drop-in replacement for reflect.choose_change, backed by an LLM."""
    backend = _pick_backend()
    prompt = _build_prompt(strategy, detail, goal, trades_batch)
    raw = _call_api(prompt) if backend == "api" else _call_cli(prompt)
    obj = _extract_json(raw)

    var = obj.get("variable")
    if var not in BOUNDS:
        raise LLMReflectError(f"LLM picked an unknown variable: {var!r}")
    try:
        new_value = float(obj["new_value"])
    except (KeyError, TypeError, ValueError):
        raise LLMReflectError(f"LLM new_value is not numeric: {obj.get('new_value')!r}")
    lo, hi = BOUNDS[var]
    if not (lo <= new_value <= hi):
        raise LLMReflectError(f"LLM value {new_value} for {var} is outside {lo}..{hi}")
    if new_value == _get(strategy, var):
        raise LLMReflectError(f"LLM proposed no actual change to {var}")

    reasoning = (str(obj.get("reasoning", "")).strip()[:400] or "(none given)")
    prediction = (str(obj.get("prediction", "")).strip()[:300] or "(none given)")
    return var, round(new_value, 4), f"[LLM/{backend}] {reasoning}", prediction
