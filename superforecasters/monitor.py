"""Track what the superforecasters are doing right now."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .categorize import is_fin_econ
from .storage import Store


def diff_positions(prev: pd.DataFrame, cur: pd.DataFrame, min_usd: float = 100, rel: float = 0.1) -> list[dict]:
    """NEW / ADD / TRIM / EXIT events between two snapshots of one trader."""
    key = "token_id"
    p = prev.set_index(key) if not prev.empty else pd.DataFrame(columns=["shares"])
    c = cur.set_index(key) if not cur.empty else pd.DataFrame(columns=["shares"])
    events = []
    for tok in set(p.index) | set(c.index):
        s0 = float(p.loc[tok, "shares"]) if tok in p.index else 0.0
        s1 = float(c.loc[tok, "shares"]) if tok in c.index else 0.0
        ds = s1 - s0
        row = c.loc[tok] if tok in c.index else p.loc[tok]
        price = row.get("cur_price") if pd.notna(row.get("cur_price")) else row.get("avg_price")
        usd = ds * (price or 0)
        if abs(usd) < min_usd or (s0 > 0 and abs(ds) / s0 < rel):
            continue
        kind = "NEW" if s0 == 0 else "EXIT" if s1 == 0 else "ADD" if ds > 0 else "TRIM"
        events.append({"kind": kind, "token_id": tok, "title": row.get("title"), "outcome": row.get("outcome"),
                       "delta_shares": ds, "delta_usd": usd, "price": price})
    return sorted(events, key=lambda e: -abs(e["delta_usd"]))


def snapshot(store: Store, pm, watchlist: pd.DataFrame, fin_econ_only: bool = True, min_usd: float = 100) -> pd.DataFrame:
    ts = time.time()
    all_events = []
    for w in watchlist.itertuples():
        pos = pm.open_positions(w.trader_id)
        if fin_econ_only:
            tags = store.all_tags()
            pos = [p for p in pos if is_fin_econ(p["title"], tags.get(p["event_slug"]))]
        prev = store.last_snapshot(w.trader_id)
        store.save_snapshot(ts, pos)
        if prev.empty:
            continue  # first snapshot is a baseline, not a signal
        for e in diff_positions(prev, pd.DataFrame(pos), min_usd=min_usd):
            e.update({"ts": ts, "platform": "polymarket", "trader_id": w.trader_id,
                      "name": getattr(w, "name", None)})
            all_events.append(e)
    store.save_events(all_events)
    return pd.DataFrame(all_events)


def consensus(store: Store, watchlist: pd.DataFrame, weight_col: str = "sf_score") -> pd.DataFrame:
    """Aggregate the latest snapshot of every watched trader into a per-market view.

    weight = max(sf_score, 0) + 0.1 so every watched trader counts a bit and the best count more.
    `sf_avg_entry` vs `cur_price` shows whether you'd be getting in cheaper or dearer than they did.
    """
    frames = []
    for w in watchlist.itertuples():
        s = store.last_snapshot(w.trader_id)
        if s.empty:
            continue
        sc = getattr(w, weight_col, 0)
        s["weight"] = (max(sc, 0) if pd.notna(sc) else 0) + 0.1
        s["name"] = getattr(w, "name", None)
        frames.append(s)
    if not frames:
        return pd.DataFrame()
    a = pd.concat(frames, ignore_index=True)
    a["usd"] = a["shares"] * a["avg_price"]
    a["w_usd"] = a["usd"] * a["weight"]
    g = a.groupby(["market_id", "title", "outcome"]).agg(
        n_sf=("trader_id", "nunique"), usd=("usd", "sum"), w_usd=("w_usd", "sum"), shares=("shares", "sum"),
        cur_price=("cur_price", "last"), who=("name", lambda x: ", ".join(sorted({str(n) for n in x if n}))[:120]),
    ).reset_index()
    g["sf_avg_entry"] = g["usd"] / g["shares"].replace(0, np.nan)  # share-weighted average entry
    # Flag disagreement: smart money on more than one outcome of the same market
    sides = g.groupby("market_id")["outcome"].transform("nunique")
    g["contested"] = sides > 1
    g["entry_vs_now"] = g["cur_price"] - g["sf_avg_entry"]
    return g.drop(columns="shares").sort_values(["n_sf", "w_usd"], ascending=False)
