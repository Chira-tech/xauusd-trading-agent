"""Gold/USD price history.

Source is chosen by, in order: the DATA_SOURCE env var (from .env), then
`data_source` in goal.yaml, then "yahoo".

  yahoo       Yahoo Finance public chart JSON, stdlib only, no key. `GC=F`
              (COMEX continuous gold futures), ~5 years of daily bars.
  twelvedata  api.twelvedata.com, needs TWELVEDATA_API_KEY. `XAU/USD` spot
              gold, daily bars back to 2008, plus intraday. Free tier is
              800 calls/day.
  yfinance    the `yfinance` package (`pip install yfinance`).

Every successful fetch is cached to state/price_cache.csv; if a later fetch
fails the agent falls back to that cache so an offline run still works.
"""
import csv
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from . import paths

_YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range={rng}&interval=1d"
_TWELVEDATA = ("https://api.twelvedata.com/time_series"
               "?symbol={sym}&interval={interval}&outputsize={n}&apikey={key}")
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


def _fetch_twelvedata(symbol: str, interval: str, n: int) -> list[dict]:
    key = os.environ.get("TWELVEDATA_API_KEY", "").strip()
    if not key:
        raise RuntimeError("TWELVEDATA_API_KEY is not set (put it in .env)")
    url = _TWELVEDATA.format(sym=urllib.parse.quote(symbol), interval=interval,
                             n=n, key=key)
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    if payload.get("status") != "ok":
        raise RuntimeError(f"twelvedata error: {payload.get('message', payload)}")

    bars = []
    for row in reversed(payload["values"]):          # API returns newest-first
        bars.append({
            "date": row["datetime"],                 # "YYYY-MM-DD" or "... HH:MM:SS"
            "open": float(row["open"]), "high": float(row["high"]),
            "low": float(row["low"]), "close": float(row["close"]),
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

    source = (os.environ.get("DATA_SOURCE") or goal.get("data_source") or "yahoo").lower()
    try:
        if source == "twelvedata":
            bars = _fetch_twelvedata(
                os.environ.get("TWELVEDATA_SYMBOL", "XAU/USD"),
                os.environ.get("TWELVEDATA_INTERVAL", "1day"),
                int(os.environ.get("TWELVEDATA_OUTPUTSIZE", "5000")),
            )
        elif source == "yfinance":
            bars = _fetch_yfinance(goal.get("yf_symbol", "GC=F"))
        else:
            bars = _fetch_yahoo_json(goal.get("yf_symbol", "GC=F"))
    except Exception as e:
        cached = _read_cache()
        if cached:
            print(f"[data] {source} fetch failed ({e}); using {len(cached)} cached bars")
            return cached
        raise SystemExit(
            f"[data] could not fetch gold prices from {source} ({e}) and no cache exists.\n"
            f"       Check .env / your connection, or set DATA_SOURCE=yahoo."
        )

    if not bars:
        raise SystemExit("[data] price source returned no rows -- retry `python run.py refresh` shortly.")
    _write_cache(bars)
    return bars


def latest_bar(goal: dict) -> dict | None:
    bars = get_history(goal, use_cache=False, refresh=True)
    return bars[-1] if bars else None
