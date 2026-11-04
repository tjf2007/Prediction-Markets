"""Decompose *how* a trader makes money, not just whether they do.

Edge decomposition (needs price history):
    y - p_entry  =  (p_close - p_entry)   +   (y - p_close)
       total          CLV: market came        resolution residual: info the
       edge           to agree with them      market never priced in (or luck)

A trader whose edge is mostly CLV is good at price discovery - reading data/news faster
than the crowd. A trader whose edge is mostly residual is either holding genuinely
non-consensus views to expiry or getting lucky; the t-stat separates the two.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .categorize import primary_topic
from .metrics import resolved_only

PRICE_BUCKETS = [(0, .2, "longshot <20c"), (.2, .45, "underdog 20-45c"), (.45, .55, "coinflip 45-55c"),
                 (.55, .8, "favourite 55-80c"), (.8, 1.0, "heavy fav >80c")]


def _spearman(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 5:
        return float("nan")
    return float(a.rank().corr(b.rank()))


def price_profile(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lo, hi, name in PRICE_BUCKETS:
        x = d[(d["p"] >= lo) & (d["p"] < hi if hi < 1 else d["p"] <= hi)]
        rows.append({"bucket": name, "n": len(x), "cost_share": x["cost"].sum() / max(d["cost"].sum(), 1e-9),
                     "hit_rate": x["y"].mean() if len(x) else np.nan,
                     "edge": x["edge"].mean() if len(x) else np.nan, "pnl": x["pnl"].sum()})
    return pd.DataFrame(rows)


def topic_profile(d: pd.DataFrame, tags_by_event: dict[str, list[str]] | None = None) -> pd.DataFrame:
    tags_by_event = tags_by_event or {}
    t = d.assign(topic=[primary_topic(ti, tags_by_event.get(ev)) for ti, ev in zip(d["title"], d["event_slug"])])
    g = t.groupby("topic").agg(n=("y", "size"), cost=("cost", "sum"), pnl=("pnl", "sum"),
                               hit_rate=("y", "mean"), edge=("edge", "mean"))
    g["cost_share"] = g["cost"] / g["cost"].sum()
    g["roi"] = g["pnl"] / g["cost"]
    return g.sort_values("cost", ascending=False).reset_index()


def entry_stats(activity: pd.DataFrame) -> pd.DataFrame:
    """Per outcome token: first buy time, number of buy fills, whether they sold before close."""
    if activity is None or activity.empty:
        return pd.DataFrame(columns=["token_id", "first_buy_ts", "n_buys", "n_sells", "buy_usd", "sell_usd"])
    a = activity.copy()
    buys, sells = a[a["side"] == "BUY"], a[a["side"] == "SELL"]
    g = buys.groupby("token_id").agg(first_buy_ts=("ts", "min"), n_buys=("ts", "size"), buy_usd=("usd", "sum"))
    s = sells.groupby("token_id").agg(n_sells=("ts", "size"), sell_usd=("usd", "sum"))
    return g.join(s, how="left").fillna({"n_sells": 0, "sell_usd": 0}).reset_index()


def price_at(hist: list[tuple[float, float]], ts: float) -> float | None:
    """Last observed price at or before ts."""
    if not hist or ts is None:
        return None
    best = None
    for t, p in hist:
        if t <= ts:
            best = p
        else:
            break
    return best


def clv_decomposition(d: pd.DataFrame, history_fn: Callable[[str], list], entries: pd.DataFrame | None = None,
                      close_offset_h: float = 24, pre_window_h: float = 72, max_markets: int = 150) -> pd.DataFrame:
    """For each resolved position (largest first), split edge into CLV + residual, and measure
    pre-entry momentum (did they buy into a rising price or fade it?)."""
    first_buy = {} if entries is None or entries.empty else dict(zip(entries["token_id"], entries["first_buy_ts"]))
    rows = []
    for _, r in d.sort_values("cost", ascending=False).head(max_markets).iterrows():
        try:
            hist = sorted(history_fn(r["token_id"]))
        except Exception:
            continue
        if not hist:
            continue
        end = r["end_ts"] or r["closed_ts"] or hist[-1][0]
        p_close = price_at(hist, end - close_offset_h * 3600) or hist[-1][1]
        t0 = first_buy.get(r["token_id"])
        p_pre = price_at(hist, t0 - pre_window_h * 3600) if t0 else None
        rows.append({
            "token_id": r["token_id"], "title": r["title"], "cost": r["cost"], "p_entry": r["p"], "y": r["y"],
            "p_close": p_close, "clv": p_close - r["p"], "residual": r["y"] - p_close,
            "pre_entry_move": (r["p"] - p_pre) if p_pre is not None else np.nan,
            "days_before_end": (end - t0) / 86400 if t0 and end else np.nan,
        })
    return pd.DataFrame(rows)


def decompose(df: pd.DataFrame, activity: pd.DataFrame | None = None, history_fn: Callable | None = None,
              tags_by_event: dict | None = None) -> dict:
    d = resolved_only(df)
    if d.empty:
        return {"summary": {}, "price_profile": pd.DataFrame(), "topics": pd.DataFrame(), "clv": pd.DataFrame()}
    yes_mask = d["outcome"].astype(str).str.lower().eq("yes")
    no_mask = d["outcome"].astype(str).str.lower().eq("no")
    topics = topic_profile(d, tags_by_event)
    summary = {
        # Sizing skill: do they bet bigger when their edge is bigger?
        "sizing_skill_spearman": _spearman(d["cost"], d["edge"]),
        "edge_equal_weight": d["edge"].mean(),
        "edge_cost_weight": (d["edge"] * d["cost"]).sum() / d["cost"].sum(),
        "median_bet_usd": d["cost"].median(),
        "bet_size_cv": d["cost"].std() / d["cost"].mean() if d["cost"].mean() else np.nan,
        "yes_share": yes_mask.mean(),
        "no_share": no_mask.mean(),
        "edge_on_yes": d.loc[yes_mask, "edge"].mean() if yes_mask.any() else np.nan,
        "edge_on_no": d.loc[no_mask, "edge"].mean() if no_mask.any() else np.nan,
        "avg_entry_price": d["p"].mean(),
        "topic_hhi": float((topics["cost_share"] ** 2).sum()) if not topics.empty else np.nan,
        "top_topic": topics.iloc[0]["topic"] if not topics.empty else None,
    }
    ent = entry_stats(activity) if activity is not None else None
    if ent is not None and not ent.empty:
        m = d.merge(ent, on="token_id", how="left")
        summary["avg_fills_per_position"] = m["n_buys"].mean()
        summary["exit_before_resolution_rate"] = (m["n_sells"] > 0).mean()
        lead = (m["end_ts"] - m["first_buy_ts"]) / 86400
        summary["median_days_before_end"] = lead[lead >= 0].median()
    clv = pd.DataFrame()
    if history_fn is not None:
        clv = clv_decomposition(d, history_fn, ent)
        if not clv.empty:
            w = clv["cost"] / clv["cost"].sum()
            tot = (clv["clv"] * w).sum() + (clv["residual"] * w).sum()
            summary.update({
                "clv_mean": clv["clv"].mean(),
                "residual_mean": clv["residual"].mean(),
                "clv_share_of_edge": (clv["clv"] * w).sum() / tot if tot else np.nan,
                "pct_positions_positive_clv": (clv["clv"] > 0).mean(),
                # >0: buys after price already rising (momentum); <0: fades moves (contrarian)
                "momentum_corr": _spearman(clv["pre_entry_move"].dropna(),
                                           clv.loc[clv["pre_entry_move"].notna(), "clv"]) if clv["pre_entry_move"].notna().sum() >= 5 else np.nan,
                "pre_entry_move_mean": clv["pre_entry_move"].mean(),
            })
    return {"summary": summary, "price_profile": price_profile(d), "topics": topics, "clv": clv}


def style_label(summary: dict, score: dict) -> str:
    """One-line human-readable archetype."""
    tags = []
    p = summary.get("avg_entry_price", 0.5)
    tags.append("longshot hunter" if p < 0.3 else "favourite grinder" if p > 0.75 else "mid-price picker")
    if summary.get("clv_share_of_edge", np.nan) == summary.get("clv_share_of_edge"):
        tags.append("price-discovery (CLV-driven)" if summary["clv_share_of_edge"] > 0.5 else "hold-to-expiry contrarian")
    mc = summary.get("momentum_corr")
    if mc == mc and mc is not None:
        if summary.get("pre_entry_move_mean", 0) < -0.02:
            tags.append("fades moves")
        elif summary.get("pre_entry_move_mean", 0) > 0.02:
            tags.append("chases momentum")
    ss = summary.get("sizing_skill_spearman")
    if ss == ss and ss is not None and ss > 0.15:
        tags.append("sizes up on conviction")
    if summary.get("topic_hhi", 0) > 0.5:
        tags.append(f"specialist: {summary.get('top_topic')}")
    if summary.get("no_share", 0) > 0.65:
        tags.append("NO-side seller of hype")
    return ", ".join(tags)
