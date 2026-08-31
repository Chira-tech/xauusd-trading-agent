"""Every path the agent touches. All writes stay inside this project directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "state"
HISTORY = STATE / "history"

GOAL_FILE = ROOT / "goal.yaml"
STRATEGY_FILE = STATE / "strategy.yaml"
TRADES_FILE = STATE / "trades.jsonl"
EQUITY_FILE = STATE / "equity.jsonl"
HYPOTHESES_FILE = STATE / "hypotheses.jsonl"
HEARTBEAT_FILE = STATE / "heartbeat.json"
PRICE_CACHE = STATE / "price_cache.csv"


def ensure_dirs() -> None:
    STATE.mkdir(exist_ok=True)
    HISTORY.mkdir(exist_ok=True)
