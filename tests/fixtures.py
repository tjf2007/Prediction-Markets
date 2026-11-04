"""Synthetic traders with KNOWN skill, for testing that the metrics recover it."""
import numpy as np
import pandas as pd

TITLES = ["Fed decreases interest rates by 25 bps after December meeting?", "CPI above 3.0% in October?",
          "US recession in 2026?", "S&P 500 above 6000 on Dec 31?", "Nonfarm payrolls above 150k?", "WTI crude above $80?"]


def make_trader(trader_id: str, n: int, skill: float, seed: int, start=1.70e9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.08, 0.92, n)
    y = (rng.uniform(size=n) < np.clip(p + skill, 0, 1)).astype(float)
    cost = rng.lognormal(6, 1, n)
    shares = cost / p
    ts = start + np.sort(rng.uniform(0, 365 * 86400, n))
    return pd.DataFrame({
        "platform": "polymarket", "trader_id": trader_id, "market_id": [f"m{i}" for i in range(n)],
        "token_id": [f"{trader_id}-t{i}" for i in range(n)], "title": [TITLES[i % len(TITLES)] for i in range(n)],
        "event_slug": [f"e{i % len(TITLES)}" for i in range(n)], "outcome": np.where(rng.uniform(size=n) < .5, "Yes", "No"),
        "avg_price": p, "shares": shares, "cost": cost, "realized_pnl": y * shares - cost, "resolved_value": y,
        "end_ts": ts, "closed_ts": ts,
    })
