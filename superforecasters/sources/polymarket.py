"""Polymarket public APIs (no auth needed).

Data API  (https://data-api.polymarket.com): leaderboard, positions, closed positions, activity.
Gamma API (https://gamma-api.polymarket.com): market/event metadata and tags.
CLOB API  (https://clob.polymarket.com): historical prices per outcome token.

Field names are parsed defensively; run `sf probe polymarket` to dump raw samples and
check the schema if Polymarket changes it.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterator

from ..http import HttpClient

DATA_API = "https://data-api.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

# Leaderboard categories as used by polymarket.com/leaderboard/<category>/<period>/<metric>
LEADERBOARD_CATEGORIES = ["ECONOMICS", "FINANCE"]


def _f(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _ts(x) -> float | None:
    """Parse epoch seconds/ms or ISO-8601 into epoch seconds."""
    if x is None or x == "":
        return None
    v = _f(x)
    if v is not None:
        return v / 1000 if v > 1e12 else v
    try:
        d = datetime.fromisoformat(str(x).replace("Z", "+00:00"))
    except ValueError:
        return None
    return (d if d.tzinfo else d.replace(tzinfo=timezone.utc)).timestamp()


class PolymarketClient:
    def __init__(self, cache_dir=None, min_interval=0.25):
        self.data = HttpClient(DATA_API, min_interval=min_interval, cache_dir=cache_dir and f"{cache_dir}/pm_data", cache_ttl=900)
        self.gamma = HttpClient(GAMMA_API, min_interval=min_interval, cache_dir=cache_dir and f"{cache_dir}/pm_gamma", cache_ttl=86400)
        self.clob = HttpClient(CLOB_API, min_interval=min_interval, cache_dir=cache_dir and f"{cache_dir}/pm_clob", cache_ttl=86400)

    # ---------------------------------------------------------------- leaderboard
    def leaderboard(self, category: str = "ECONOMICS", period: str = "ALL", order_by: str = "PNL",
                    max_rows: int = 500, page: int = 50) -> list[dict]:
        rows, offset = [], 0
        while offset < max_rows:
            batch = self.data.get("/v1/leaderboard", {
                "category": category.upper(), "timePeriod": period.upper(), "orderBy": order_by.upper(),
                "limit": page, "offset": offset,
            })
            if isinstance(batch, dict):
                batch = batch.get("data") or batch.get("leaderboard") or []
            if not batch:
                break
            for r in batch:
                rows.append({
                    "platform": "polymarket",
                    "trader_id": (r.get("proxyWallet") or r.get("address") or "").lower(),
                    "name": r.get("userName") or r.get("name") or r.get("pseudonym"),
                    "x_username": r.get("xUsername"),
                    "rank": r.get("rank"),
                    "pnl": _f(r.get("pnl")),
                    "volume": _f(r.get("vol") or r.get("volume")),
                    "category": category.upper(),
                    "period": period.upper(),
                    "order_by": order_by.upper(),
                })
            if len(batch) < page:
                break
            offset += page
        return rows

    # ---------------------------------------------------------------- per-user
    def _paged(self, path: str, params: dict, page: int = 500, max_rows: int = 20000) -> Iterator[dict]:
        offset = 0
        while offset < max_rows:
            batch = self.data.get(path, {**params, "limit": page, "offset": offset})
            if isinstance(batch, dict):
                batch = batch.get("data") or []
            if not batch:
                return
            yield from batch
            if len(batch) < page:
                return
            offset += page

    def open_positions(self, wallet: str) -> list[dict]:
        out = []
        for p in self._paged("/positions", {"user": wallet, "sizeThreshold": 0.01}, page=500):
            out.append(self._norm_position(wallet, p, closed=False))
        return out

    def closed_positions(self, wallet: str, max_rows: int = 5000) -> list[dict]:
        out = []
        for p in self._paged("/closed-positions", {"user": wallet}, page=50, max_rows=max_rows):
            out.append(self._norm_position(wallet, p, closed=True))
        return out

    def activity(self, wallet: str, max_rows: int = 10000) -> list[dict]:
        out = []
        for a in self._paged("/activity", {"user": wallet, "type": "TRADE"}, page=500, max_rows=max_rows):
            out.append({
                "platform": "polymarket",
                "trader_id": wallet.lower(),
                "ts": _ts(a.get("timestamp")),
                "market_id": a.get("conditionId"),
                "token_id": a.get("asset"),
                "title": a.get("title"),
                "event_slug": a.get("eventSlug"),
                "outcome": a.get("outcome"),
                "side": (a.get("side") or "").upper(),          # BUY / SELL
                "price": _f(a.get("price")),
                "shares": _f(a.get("size")),
                "usd": _f(a.get("usdcSize")),
                "tx": a.get("transactionHash"),
            })
        return out

    @staticmethod
    def _norm_position(wallet: str, p: dict, closed: bool) -> dict:
        avg = _f(p.get("avgPrice"))
        shares = _f(p.get("totalBought")) if closed else _f(p.get("size"))
        cur = _f(p.get("curPrice"))
        resolved = None
        if closed and cur is not None and (cur <= 0.001 or cur >= 0.999):
            resolved = round(cur)
        cost = _f(p.get("initialValue"))
        if cost is None and avg is not None and shares is not None:
            cost = avg * shares
        return {
            "platform": "polymarket",
            "trader_id": wallet.lower(),
            "market_id": p.get("conditionId"),
            "token_id": p.get("asset"),
            "title": p.get("title"),
            "slug": p.get("slug"),
            "event_slug": p.get("eventSlug"),
            "outcome": p.get("outcome"),
            "avg_price": avg,
            "shares": shares,
            "cost": cost,
            "cur_price": cur,
            "current_value": _f(p.get("currentValue")),
            "realized_pnl": _f(p.get("realizedPnl")),
            "cash_pnl": _f(p.get("cashPnl")),
            "resolved_value": resolved,
            "end_ts": _ts(p.get("endDate")),
            "closed_ts": _ts(p.get("timestamp")) if closed else None,
            "is_closed": closed,
        }

    # ---------------------------------------------------------------- metadata
    def event_tags(self, event_slug: str) -> list[str]:
        if not event_slug:
            return []
        res = self.gamma.get("/events", {"slug": event_slug})
        ev = res[0] if isinstance(res, list) and res else (res if isinstance(res, dict) else {})
        tags = ev.get("tags") or []
        return [str(t.get("label") or t.get("slug") or t).lower() if isinstance(t, dict) else str(t).lower() for t in tags]

    def price_history(self, token_id: str, fidelity_min: int = 60) -> list[tuple[float, float]]:
        """[(epoch_seconds, price)] for one outcome token."""
        res = self.clob.get("/prices-history", {"market": token_id, "interval": "max", "fidelity": fidelity_min})
        hist = res.get("history", []) if isinstance(res, dict) else []
        return [(float(h["t"]), float(h["p"])) for h in hist if "t" in h and "p" in h]

    def raw_samples(self, wallet: str | None = None) -> dict:
        lb = self.data.get("/v1/leaderboard", {"category": "ECONOMICS", "timePeriod": "ALL", "orderBy": "PNL", "limit": 3}, use_cache=False)
        out = {"leaderboard": lb}
        w = wallet or (lb[0].get("proxyWallet") if isinstance(lb, list) and lb else None)
        if w:
            out["positions"] = self.data.get("/positions", {"user": w, "limit": 2}, use_cache=False)
            out["closed_positions"] = self.data.get("/closed-positions", {"user": w, "limit": 2}, use_cache=False)
            out["activity"] = self.data.get("/activity", {"user": w, "limit": 2}, use_cache=False)
        return out
