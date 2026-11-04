"""Kalshi API (https://api.elections.kalshi.com/trade-api/v2).

Important limitation: Kalshi's public API is anonymous. Market metadata, settlement
results, and the trade tape are public, but trades are NOT attributed to users, so other
people's track records can't be pulled from the API. This module therefore supports:

  1. Market universe + resolutions for the Economics / Financials categories (public).
  2. Anonymous large-print ("whale tape") monitoring on those markets (public).
  3. Per-trader track records you supply yourself, via CSV import (see `import_positions_csv`),
     e.g. from Kalshi profile pages you've collected, or your own account.
  4. Your own account's settlements via the authenticated portfolio API (API key + RSA key).
"""
from __future__ import annotations

import base64
import csv
import time
from typing import Iterator

from ..http import HttpClient

KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_CATEGORIES = ["Economics", "Financials"]


def _f(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _price(m: dict, key: str) -> float | None:
    """Kalshi prices: `<key>_dollars` (string dollars) or `<key>` in cents."""
    d = _f(m.get(f"{key}_dollars"))
    if d is not None:
        return d
    c = _f(m.get(key))
    return c / 100 if c is not None else None


class KalshiClient:
    def __init__(self, cache_dir=None, min_interval=0.12, api_key_id: str | None = None, private_key_path: str | None = None):
        self.http = HttpClient(KALSHI_API, min_interval=min_interval, cache_dir=cache_dir and f"{cache_dir}/kalshi", cache_ttl=1800)
        self.api_key_id = api_key_id
        self.private_key_path = private_key_path

    def _cursor(self, path: str, key: str, params: dict, max_rows: int = 50000) -> Iterator[dict]:
        cursor, n = None, 0
        while n < max_rows:
            res = self.http.get(path, {**params, "cursor": cursor})
            items = res.get(key) or []
            yield from items
            n += len(items)
            cursor = res.get("cursor")
            if not cursor or not items:
                return

    # ---------------------------------------------------------------- public
    def series(self, category: str) -> list[dict]:
        res = self.http.get("/series", {"category": category})
        return res.get("series") or []

    def markets(self, series_ticker: str, status: str | None = None) -> list[dict]:
        out = []
        for m in self._cursor("/markets", "markets", {"series_ticker": series_ticker, "status": status, "limit": 1000}):
            res = (m.get("result") or "").lower()
            out.append({
                "platform": "kalshi",
                "market_id": m.get("ticker"),
                "event_id": m.get("event_ticker"),
                "series": series_ticker,
                "title": " ".join(x for x in [m.get("title"), m.get("yes_sub_title")] if x),
                "status": m.get("status"),
                "result": res if res in ("yes", "no") else None,
                "last_price": _price(m, "last_price"),
                "volume": _f(m.get("volume")),
                "close_time": m.get("close_time"),
            })
        return out

    def finance_econ_markets(self, status: str | None = "open") -> list[dict]:
        out = []
        for cat in KALSHI_CATEGORIES:
            for s in self.series(cat):
                for m in self.markets(s["ticker"], status=status):
                    m["category"] = cat
                    out.append(m)
        return out

    def trades(self, ticker: str, min_ts: int | None = None, max_rows: int = 20000) -> list[dict]:
        out = []
        for t in self._cursor("/markets/trades", "trades", {"ticker": ticker, "min_ts": min_ts, "limit": 1000}, max_rows):
            yes = _price(t, "yes_price")
            cnt = _f(t.get("count_fp")) or _f(t.get("count"))
            side = (t.get("taker_side") or "").lower()
            px = yes if side == "yes" else (1 - yes if yes is not None else None)
            out.append({
                "market_id": ticker, "ts": t.get("created_time"), "taker_side": side,
                "yes_price": yes, "contracts": cnt,
                "notional": (px or 0) * (cnt or 0),
            })
        return out

    def whale_prints(self, tickers: list[str], min_notional: float = 5000, lookback_hours: float = 24) -> list[dict]:
        """Large anonymous prints on the given markets - a crowd-level smart-money proxy."""
        since = int(time.time() - lookback_hours * 3600)
        prints = []
        for t in tickers:
            prints += [x for x in self.trades(t, min_ts=since) if x["notional"] >= min_notional]
        return sorted(prints, key=lambda x: -x["notional"])

    # ---------------------------------------------------------------- own account (auth)
    def _auth_headers(self, method: str, path: str) -> dict:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding

        if not (self.api_key_id and self.private_key_path):
            raise RuntimeError("Set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH for portfolio endpoints")
        with open(self.private_key_path, "rb") as fh:
            key = serialization.load_pem_private_key(fh.read(), password=None)
        ts = str(int(time.time() * 1000))
        msg = (ts + method.upper() + "/trade-api/v2" + path.split("?")[0]).encode()
        sig = key.sign(msg, padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH), hashes.SHA256())
        return {"KALSHI-ACCESS-KEY": self.api_key_id, "KALSHI-ACCESS-TIMESTAMP": ts,
                "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode()}

    def my_settlements(self, trader_id: str = "me") -> list[dict]:
        out, cursor = [], None
        while True:
            res = self.http.get("/portfolio/settlements", {"limit": 200, "cursor": cursor}, use_cache=False,
                                headers=self._auth_headers("GET", "/portfolio/settlements"))
            for s in res.get("settlements") or []:
                result = (s.get("market_result") or "").lower()
                for side in ("yes", "no"):
                    cnt = _f(s.get(f"{side}_count")) or 0
                    if cnt <= 0:
                        continue
                    cost = (_f(s.get(f"{side}_total_cost")) or 0) / 100
                    won = 1 if result == side else 0
                    out.append(_position_row("kalshi", trader_id, s.get("ticker"), s.get("ticker"), side.upper(),
                                             cost / cnt, cnt, cost, won * cnt - cost, won, s.get("settled_time")))
            cursor = res.get("cursor")
            if not cursor:
                return out


def _position_row(platform, trader_id, market_id, title, outcome, avg_price, shares, cost, pnl, resolved, closed_ts):
    from .polymarket import _ts
    return {
        "platform": platform, "trader_id": trader_id, "market_id": market_id, "token_id": f"{market_id}:{outcome}",
        "title": title, "slug": market_id, "event_slug": market_id, "outcome": outcome,
        "avg_price": avg_price, "shares": shares, "cost": cost, "cur_price": resolved, "current_value": None,
        "realized_pnl": pnl, "cash_pnl": None, "resolved_value": resolved, "end_ts": _ts(closed_ts),
        "closed_ts": _ts(closed_ts), "is_closed": True,
    }


def import_positions_csv(path: str, platform: str = "kalshi") -> list[dict]:
    """Import resolved positions you collected for any trader.

    Required columns: trader_id, market_id, title, outcome, avg_price (0-1), shares, resolved (0/1), closed_at
    Optional: realized_pnl (defaults to resolved*shares - avg_price*shares)
    """
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            avg, sh, res = float(r["avg_price"]), float(r["shares"]), int(float(r["resolved"]))
            cost = avg * sh
            pnl = _f(r.get("realized_pnl"), res * sh - cost)
            rows.append(_position_row(platform, r["trader_id"], r["market_id"], r.get("title") or r["market_id"],
                                      r.get("outcome", "YES"), avg, sh, cost, pnl, res, r.get("closed_at")))
    return rows
