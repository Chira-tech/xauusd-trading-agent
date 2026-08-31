"""Paper trading account: one long position at a time, USD equity, trade + equity logs."""
import json
from datetime import datetime, timezone

from . import paths


class Paper:
    def __init__(self, starting_equity: float):
        self.start_equity = starting_equity
        self.equity = starting_equity
        self.peak_equity = starting_equity
        self.position: dict | None = None
        self.closed: list[dict] = []
        self.equity_points: list[dict] = []

    def open_long(self, price: float, date: str, size_r: float, strategy_version: str) -> None:
        size_usd = self.equity * size_r
        self.position = {
            "entry_price": price,
            "size_usd": size_usd,
            "units": size_usd / price,
            "entry_date": date,
            "strategy_version": strategy_version,
        }

    def close(self, price: float, date: str, reason: str) -> dict:
        p = self.position
        pnl = (price - p["entry_price"]) * p["units"]
        ret = price / p["entry_price"] - 1.0
        self.equity += pnl
        self.peak_equity = max(self.peak_equity, self.equity)
        trade = {
            "entry_date": p["entry_date"],
            "exit_date": date,
            "entry_price": round(p["entry_price"], 4),
            "exit_price": round(price, 4),
            "units": round(p["units"], 6),
            "size_usd": round(p["size_usd"], 2),
            "pnl": round(pnl, 2),
            "return": round(ret, 5),
            "reason": reason,
            "equity_after": round(self.equity, 2),
            "strategy_version": p["strategy_version"],
            "closed_at": datetime.now(timezone.utc).isoformat(),
        }
        self.closed.append(trade)
        self.position = None
        return trade

    def mark(self, date: str, price: float):
        """Mark-to-market. Returns (equity, drawdown_fraction)."""
        eq = self.equity
        if self.position:
            eq += (price - self.position["entry_price"]) * self.position["units"]
        self.peak_equity = max(self.peak_equity, eq)
        dd = 0.0 if self.peak_equity == 0 else (self.peak_equity - eq) / self.peak_equity
        point = {"date": date, "equity": round(eq, 2), "drawdown": round(dd, 5)}
        self.equity_points.append(point)
        return eq, dd


def append_jsonl(path, obj: dict) -> None:
    paths.ensure_dirs()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def read_jsonl(path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def rewrite_jsonl(path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
