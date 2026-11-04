"""Discover -> fetch -> classify -> score."""
from __future__ import annotations

import logging

import pandas as pd

from .categorize import is_fin_econ
from .metrics import rank_traders, roi_moments, score_trader
from .sources.polymarket import LEADERBOARD_CATEGORIES, PolymarketClient
from .storage import Store

log = logging.getLogger(__name__)


def discover_polymarket(pm: PolymarketClient, store: Store, depth: int = 500,
                        categories=LEADERBOARD_CATEGORIES, periods=("ALL", "MONTH")) -> pd.DataFrame:
    """Candidate pool = union of category leaderboards ranked by PnL *and* by volume.

    Ranking only by PnL selects for lucky whales; adding volume-ranked rows brings in
    high-activity traders with modest PnL whose per-bet edge may be better."""
    rows = []
    for cat in categories:
        for period in periods:
            for order in ("PNL", "VOL"):
                try:
                    r = pm.leaderboard(cat, period, order, max_rows=depth)
                except Exception as e:  # keep going if one board fails
                    log.warning("leaderboard %s/%s/%s failed: %s", cat, period, order, e)
                    continue
                rows += r
    store.save_leaderboard(rows)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return (df.sort_values("pnl", ascending=False)
              .groupby("trader_id", as_index=False)
              .agg(name=("name", "first"), best_rank=("rank", "min"), pnl=("pnl", "max"), volume=("volume", "max"),
                   boards=("category", lambda x: ",".join(sorted(set(x))))))


def _tags(pm: PolymarketClient, store: Store, slug: str | None) -> list[str]:
    if not slug:
        return []
    t = store.get_tags(slug)
    if t is None:
        try:
            t = pm.event_tags(slug)
        except Exception:
            t = []
        store.put_tags(slug, t)
    return t


def ingest_trader(pm: PolymarketClient, store: Store, wallet: str, max_positions: int = 5000) -> pd.DataFrame:
    rows = pm.closed_positions(wallet, max_rows=max_positions)
    for r in rows:
        r["fin_econ"] = is_fin_econ(r["title"], _tags(pm, store, r["event_slug"]))
    store.save_positions(rows)
    return pd.DataFrame(rows)


def score_and_store(store: Store, platform: str, trader_id: str, name: str | None = None) -> dict:
    df = store.positions(trader_id, fin_econ_only=True)
    s = {**score_trader(df), **roi_moments(df)}
    store.save_score(platform, trader_id, name, s)
    return s


def run_polymarket(store: Store, pm: PolymarketClient, depth: int = 500, max_traders: int = 300,
                   min_resolved: int = 20) -> pd.DataFrame:
    cands = discover_polymarket(pm, store, depth=depth)
    if cands.empty:
        return cands
    cands = cands.head(max_traders)
    for i, c in enumerate(cands.itertuples(), 1):
        log.info("[%d/%d] %s %s", i, len(cands), c.trader_id, c.name or "")
        try:
            ingest_trader(pm, store, c.trader_id)
            score_and_store(store, "polymarket", c.trader_id, c.name)
        except Exception as e:
            log.warning("failed %s: %s", c.trader_id, e)
    return ranked(store, min_resolved)


def ranked(store: Store, min_resolved: int = 20) -> pd.DataFrame:
    s = store.scores()
    return rank_traders(s, min_resolved) if not s.empty else s
