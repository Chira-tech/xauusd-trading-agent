"""Free gold/USD price history.

Primary source: Yahoo Finance's public chart JSON endpoint, fetched with the
standard library only -- so the core agent has zero third-party data
dependencies. Symbol `GC=F` is COMEX continuous gold futures, priced in USD.

Set `data_source: yfinance` in goal.yaml to use the `yfinance` package instead
(`pip install yfinance`).

Every successful fetch is cached to state/price_cache.csv; if a later fetch
fails the agent falls back to that cache so an offline run still works.
"""
import csv
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from . import paths

_YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range={rng}&interval=1d"
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; gold-agent/1.0)"}
_FIELDS = ["date", "open", "high", "low", "close"]


def _fetch_yahoo_json(symbol: str, rng: str = "5y") -> list[dict]:
    url = _YAHOO.format(sym=urllib.parse.quote(symbol), rng=rng)
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    result = payload["chart"]["result"][0]
    stamps = result["timestamp"]
    q = result["indicators"]["quote"][0]

    bars = []
    for i, ts in enumerate(stamps):
        o, h, l, c = q["open"][i], q["high"][i], q["low"][i], q["close"][i]
        if None in (o, h, l, c):
            continue
        bars.append({
            "date": datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d"),
            "open": float(o), "high": float(h), "low": float(l), "close": float(c),
        })
    return bars


def _fetch_yfinance(symbol: str, period: str = "5y") -> list[dict]:
    import yfinance as yf

    df = yf.download(symbol, period=period, interval="1d",
                     progress=False, auto_adjust=False)
    if df is None or df.empty:
        return []
    if hasattr(df.columns, "get_level_values"):
        df.columns = df.columns.get_level_values(0)
    return [{
        "date": idx.strftime("%Y-%m-%d"),
        "open": float(row["Open"]), "high": float(row["High"]),
        "low": float(row["Low"]), "close": float(row["Close"]),
    } for idx, row in df.iterrows()]


def _write_cache(bars: list[dict]) -> None:
    paths.ensure_dirs()
    with open(paths.PRICE_CACHE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        w.writeheader()
        w.writerows(bars)


def _read_cache() -> list[dict]:
    if not paths.PRICE_CACHE.exists():
        return []
    with open(paths.PRICE_CACHE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [{k: (r[k] if k == "date" else float(r[k])) for k in _FIELDS} for r in rows]


def get_history(goal: dict, use_cache: bool = True, refresh: bool = False) -> list[dict]:
    if use_cache and not refresh:
        cached = _read_cache()
        if cached:
            return cached

    symbol = goal.get("yf_symbol", "GC=F")
    source = goal.get("data_source", "yahoo")
    try:
        bars = _fetch_yfinance(symbol) if source == "yfinance" else _fetch_yahoo_json(symbol)
    except Exception as e:
        cached = _read_cache()
        if cached:
            print(f"[data] live fetch failed ({e}); using {len(cached)} cached bars")
            return cached
        raise SystemExit(
            f"[data] could not fetch gold prices ({e}) and no cache exists.\n"
            f"       Retry in a minute, or set data_source: yfinance in goal.yaml."
        )

    if not bars:
        raise SystemExit("[data] price source returned no rows -- retry `python run.py refresh` shortly.")
    _write_cache(bars)
    return bars


def latest_bar(goal: dict) -> dict | None:
    bars = get_history(goal, use_cache=False, refresh=True)
    return bars[-1] if bars else None
