"""Self-contained HTML report: ranked superforecasters, profiles, and current consensus."""
from __future__ import annotations

import html
import time

import pandas as pd

CSS = """
:root{--bg:#fff;--fg:#1a1a1a;--mut:#666;--line:#e5e5e5;--pos:#0a7a4b;--neg:#b42318;--acc:#1f4fd1}
@media (prefers-color-scheme:dark){:root{--bg:#121212;--fg:#eee;--mut:#999;--line:#2a2a2a;--pos:#4cc38a;--neg:#ff6b5e;--acc:#7aa2ff}}
body{background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,system-ui,Segoe UI,sans-serif;margin:0 auto;max-width:1200px;padding:24px 16px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:32px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
.mut{color:var(--mut)}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13px}
th,td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}th{color:var(--mut);font-weight:500}
td:first-child,th:first-child,td.l{text-align:left}.pos{color:var(--pos)}.neg{color:var(--neg)}.wrap{overflow-x:auto}
"""

FMT = {"pct": lambda v: f"{v:.1%}", "num": lambda v: f"{v:,.2f}", "usd": lambda v: f"${v:,.0f}", "int": lambda v: f"{v:,.0f}"}


def _cell(v, kind="num", signed=False):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "<td class=mut>–</td>"
    if isinstance(v, str):
        return f"<td class=l>{html.escape(v)}</td>"
    cls = (" class=pos" if v > 0 else " class=neg" if v < 0 else "") if signed else ""
    return f"<td{cls}>{FMT[kind](v)}</td>"


def table(df: pd.DataFrame, cols: list[tuple[str, str, str, bool]]) -> str:
    """cols: (column, header, kind, signed)."""
    h = "".join(f"<th>{html.escape(t)}</th>" for _, t, _, _ in cols)
    body = "".join("<tr>" + "".join(_cell(r.get(c), k, s) for c, _, k, s in cols) + "</tr>" for r in df.to_dict("records"))
    return f"<div class=wrap><table><tr>{h}</tr>{body}</table></div>"


def build(ranked: pd.DataFrame, consensus: pd.DataFrame | None = None, profiles: dict | None = None, top: int = 50) -> str:
    r = ranked[ranked.get("eligible", True) == True].head(top).copy()  # noqa: E712
    r["who"] = r["name"].fillna("") + " " + r["trader_id"].str[:8]
    parts = [f"<title>Superforecaster Monitor</title><style>{CSS}</style>",
             "<h1>Finance & Economics superforecasters</h1>",
             f"<p class=mut>Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}. Scores use resolved "
             "Finance/Economics positions only. Edge = hit rate minus entry price (the trader's marginal Brier "
             "improvement over the market). Deflated Sharpe corrects for having scanned every trader on the board.</p>",
             "<h2>Ranking</h2>",
             table(r, [("who", "Trader", "num", False), ("sf_score", "SF score", "num", True),
                       ("n_resolved", "Resolved", "int", False), ("edge", "Edge", "pct", True),
                       ("edge_tstat", "Edge t", "num", True), ("hit_rate", "Hit", "pct", False),
                       ("avg_entry_price", "Avg px", "num", False), ("brier_skill", "Brier skill", "pct", True),
                       ("trade_sharpe", "Sharpe/trade", "num", True), ("daily_sharpe_ann", "Sharpe ann.", "num", True),
                       ("deflated_sharpe", "DSR", "pct", False), ("roi", "ROI", "pct", True),
                       ("total_pnl", "PnL", "usd", True), ("max_drawdown", "Max DD", "usd", True)])]
    if consensus is not None and not consensus.empty:
        parts += ["<h2>What they hold now (consensus)</h2>",
                  table(consensus.head(40), [("title", "Market", "num", False), ("outcome", "Side", "num", False),
                                             ("n_sf", "# SF", "int", False), ("usd", "Cost basis", "usd", False),
                                             ("sf_avg_entry", "SF entry", "num", False), ("cur_price", "Now", "num", False),
                                             ("entry_vs_now", "Δ since entry", "num", True), ("who", "Who", "num", False)])]
    for tid, p in (profiles or {}).items():
        parts.append(f"<h2>Profile: {html.escape(p.get('name') or tid)}</h2><p>{html.escape(p.get('style', ''))}</p>")
        s = p.get("summary", {})
        kv = pd.DataFrame([{"k": k, "v": v} for k, v in s.items() if not isinstance(v, str)])
        if not kv.empty:
            parts.append(table(kv, [("k", "Metric", "num", False), ("v", "Value", "num", True)]))
        if isinstance(p.get("price_profile"), pd.DataFrame) and not p["price_profile"].empty:
            parts.append(table(p["price_profile"], [("bucket", "Entry price", "num", False), ("n", "N", "int", False),
                                                    ("cost_share", "Capital", "pct", False), ("hit_rate", "Hit", "pct", False),
                                                    ("edge", "Edge", "pct", True), ("pnl", "PnL", "usd", True)]))
        if isinstance(p.get("topics"), pd.DataFrame) and not p["topics"].empty:
            parts.append(table(p["topics"], [("topic", "Topic", "num", False), ("n", "N", "int", False),
                                             ("cost_share", "Capital", "pct", False), ("edge", "Edge", "pct", True),
                                             ("roi", "ROI", "pct", True), ("pnl", "PnL", "usd", True)]))
    return "<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>" + "".join(parts)
