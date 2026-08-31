"""Load / save the two YAML files: goal.yaml (fixed) and state/strategy.yaml (evolves)."""
import shutil

import yaml

from . import paths

BASELINE = paths.ROOT / "agent" / "baseline_strategy.yaml"


def load_goal() -> dict:
    with open(paths.GOAL_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_strategy() -> None:
    """Create state/strategy.yaml from the v01 baseline if it does not exist yet."""
    paths.ensure_dirs()
    if not paths.STRATEGY_FILE.exists():
        shutil.copyfile(BASELINE, paths.STRATEGY_FILE)


def restore_baseline() -> None:
    """Reset strategy.yaml back to v01 (used at the start of a fresh backtest)."""
    paths.ensure_dirs()
    shutil.copyfile(BASELINE, paths.STRATEGY_FILE)


def load_strategy() -> dict:
    ensure_strategy()
    with open(paths.STRATEGY_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_strategy(strategy: dict) -> None:
    with open(paths.STRATEGY_FILE, "w", encoding="utf-8") as f:
        yaml.safe_dump(strategy, f, sort_keys=False)
