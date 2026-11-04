"""Skill metrics for a trader's resolved prediction-market positions.

Conventions: every position is a long position in one outcome token bought at average
price p (0-1) that resolved to y in {0,1}. NO bets are just longs in the NO token.

Why "edge" is the Brier metric that matters for traders
------------------------------------------------------
Traders don't publish probabilities, they reveal direction at a price. If the market says p
and the trader buys, their forecast q > p. The change in Brier score from nudging the market
forecast toward the trader by epsilon is  d/de (p + e - y)^2 = 2(p - y),  so mean(y - p) is
(minus half) the marginal Brier improvement the trader contributes over the market. We
report it as `edge`, plus a full Brier comparison using a size-implied forecast q.
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

N = NormalDist()
EULER_GAMMA = 0.5772156649


def resolved_only(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["resolved_value"].notna() & df["avg_price"].between(0.001, 0.999) & (df["cost"].fillna(0) > 0)].copy()
    d["y"] = d["resolved_value"].astype(float)
    d["p"] = d["avg_price"].astype(float)
    d["edge"] = d["y"] - d["p"]
    pnl = d["realized_pnl"].where(d["realized_pnl"].notna(), d["y"] * d["shares"] - d["cost"])
    d["pnl"] = pnl.astype(float)
    d["roi"] = d["pnl"] / d["cost"]
    return d


# ----------------------------------------------------------------------------- Brier / calibration
def implied_forecast(p: pd.Series, cost: pd.Series, conviction: float = 0.25) -> pd.Series:
    """Size-implied belief: q = p + conviction * s * (1 - p), s = position size scaled to the
    trader's own 90th-percentile bet (capped at 1). Mirrors Kelly: stake f = (q - p)/(1 - p).
    `conviction` is the Kelly fraction assumed for a top-decile bet; results are reported
    relative to the market so its exact value matters much less than its sign."""
    ref = max(cost.quantile(0.9), 1e-9)
    s = (cost / ref).clip(0, 1)
    return p + conviction * s * (1 - p)


def fit_conviction_oos(d: pd.DataFrame, grid=np.linspace(0, 1, 21)) -> tuple[float, float]:
    """Walk-forward: choose the conviction that minimises Brier on the first half of the
    trader's history (by time), then report Brier skill on the second half. Honest
    out-of-sample answer to "if we read their size as belief, do they beat the market?"."""
    d = d.sort_values("closed_ts")
    h = len(d) // 2
    if h < 10:
        return float("nan"), float("nan")
    tr, te = d.iloc[:h], d.iloc[h:]
    ref = max(tr["cost"].quantile(0.9), 1e-9)

    def b(x, lam):
        q = x["p"] + lam * (x["cost"] / ref).clip(0, 1) * (1 - x["p"])
        return float(((q - x["y"]) ** 2).mean())
    lam = min(grid, key=lambda g: b(tr, g))
    bm = float(((te["p"] - te["y"]) ** 2).mean())
    return float(lam), (1 - b(te, lam) / bm) if bm > 0 else float("nan")


def brier_block(d: pd.DataFrame, conviction: float = 0.25) -> dict:
    if d.empty:
        return {}
    lam, oos = fit_conviction_oos(d)
    q = implied_forecast(d["p"], d["cost"], conviction)
    b_mkt = float(((d["p"] - d["y"]) ** 2).mean())
    b_trd = float(((q - d["y"]) ** 2).mean())
    return {
        "brier_market_at_entry": b_mkt,
        "brier_implied": b_trd,
        "brier_skill": 1 - b_trd / b_mkt if b_mkt > 0 else float("nan"),
        "brier_skill_oos": oos,          # walk-forward, conviction fitted on first half
        "conviction_fitted": lam,
        "log_loss_market": float(-(d["y"] * np.log(d["p"]) + (1 - d["y"]) * np.log(1 - d["p"])).mean()),
    }


def calibration_table(d: pd.DataFrame, bins=(0, .1, .2, .35, .5, .65, .8, .9, 1.0)) -> pd.DataFrame:
    """Hit rate vs entry price. A skilled trader's hit rate sits ABOVE the diagonal."""
    if d.empty:
        return pd.DataFrame()
    g = d.groupby(pd.cut(d["p"], bins, include_lowest=True), observed=True)
    t = g.agg(n=("y", "size"), mean_price=("p", "mean"), hit_rate=("y", "mean"), cost=("cost", "sum"), pnl=("pnl", "sum"))
    t["edge"] = t["hit_rate"] - t["mean_price"]
    return t.reset_index().rename(columns={"p": "price_bucket"})


# ----------------------------------------------------------------------------- Sharpe family
def sharpe(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / x.std(ddof=1))


def _moments(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, float)
    s = x.std(ddof=0)
    if s == 0:
        return 0.0, 3.0
    z = (x - x.mean()) / s
    return float((z ** 3).mean()), float((z ** 4).mean())


def probabilistic_sharpe(x: np.ndarray, sr_benchmark: float = 0.0) -> float:
    """Bailey & Lopez de Prado PSR: P(true SR > benchmark) given sample length, skew and
    kurtosis. Prediction-market returns are very fat-tailed, so this matters."""
    x = np.asarray(x, float)
    n = len(x)
    sr = sharpe(x)
    if n < 3 or math.isnan(sr):
        return float("nan")
    g3, g4 = _moments(x)
    denom = 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2
    if denom <= 0:
        return float("nan")
    return float(N.cdf((sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(denom)))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected max Sharpe across n_trials skill-less traders (the bar for the Deflated Sharpe)."""
    if n_trials < 2 or sr_variance <= 0:
        return 0.0
    a = N.inv_cdf(1 - 1 / n_trials)
    b = N.inv_cdf(1 - 1 / (n_trials * math.e))
    return math.sqrt(sr_variance) * ((1 - EULER_GAMMA) * a + EULER_GAMMA * b)


def daily_pnl(d: pd.DataFrame) -> pd.Series:
    ts = d["closed_ts"].fillna(d["end_ts"])
    d = d[ts.notna()]
    if d.empty:
        return pd.Series(dtype=float)
    days = pd.to_datetime(ts[ts.notna()], unit="s").dt.floor("D")
    s = d.groupby(days.values)["pnl"].sum()
    idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
    return s.reindex(idx, fill_value=0.0)


def max_drawdown(pnl: pd.Series) -> float:
    if pnl.empty:
        return 0.0
    eq = pnl.cumsum()
    return float((eq - eq.cummax()).min())


# ----------------------------------------------------------------------------- summary
def score_trader(df: pd.DataFrame, conviction: float = 0.25, prior_n: float = 30) -> dict:
    d = resolved_only(df)
    n = len(d)
    out = {"n_resolved": n}
    if n == 0:
        return out
    cost = d["cost"].sum()
    e = d["edge"].to_numpy()
    se = e.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
    dp = daily_pnl(d)
    span_days = max((dp.index.max() - dp.index.min()).days, 1) if len(dp) else 1
    per_year = n / span_days * 365
    out.update({
        "total_cost": float(cost),
        "total_pnl": float(d["pnl"].sum()),
        "roi": float(d["pnl"].sum() / cost) if cost else float("nan"),
        "hit_rate": float(d["y"].mean()),
        "avg_entry_price": float(d["p"].mean()),
        "edge": float(e.mean()),                                     # = -1/2 marginal Brier improvement
        "edge_cost_weighted": float((d["edge"] * d["cost"]).sum() / cost),
        "edge_tstat": float(e.mean() / se) if se and se > 0 else float("nan"),
        # Shrink toward zero edge: behaves like a prior worth `prior_n` coin-flip positions.
        "edge_shrunk": float(e.mean() * n / (n + prior_n)),
        "trade_sharpe": sharpe(d["roi"].to_numpy()),
        "trade_sharpe_ann": sharpe(d["roi"].to_numpy()) * math.sqrt(per_year) if n > 1 else float("nan"),
        "daily_sharpe_ann": sharpe(dp.to_numpy()) * math.sqrt(365) if len(dp) > 1 else float("nan"),
        "psr_trade": probabilistic_sharpe(d["roi"].to_numpy()),
        "max_drawdown": max_drawdown(dp),
        "active_days": int(span_days),
        "first_ts": float(d["closed_ts"].min()) if d["closed_ts"].notna().any() else None,
        "last_ts": float(d["closed_ts"].max()) if d["closed_ts"].notna().any() else None,
    })
    out.update(brier_block(d, conviction))
    return out


def rank_traders(scores: pd.DataFrame, min_resolved: int = 20) -> pd.DataFrame:
    """Add multiple-testing-aware columns and a composite superforecaster score.

    - deflated_sharpe: PSR against the Sharpe you'd expect from the luckiest of N
      skill-less traders (N = number of traders scanned). Leaderboards are survivorship
      machines; this is the main defence.
    - sf_score: mean of cross-sectional z-scores of edge_shrunk, edge_tstat, brier_skill and
      trade_sharpe, among traders meeting min_resolved.
    """
    s = scores.copy()
    s["eligible"] = s["n_resolved"].fillna(0) >= min_resolved
    elig = s[s["eligible"]]
    n_trials = max(len(s), 2)
    sr_var = float(elig["trade_sharpe"].var()) if len(elig) > 2 else 0.0
    sr_star = expected_max_sharpe(n_trials, sr_var)
    s["sr_hurdle"] = sr_star

    def _dsr(row):
        if not row["eligible"] or pd.isna(row.get("trade_sharpe")):
            return float("nan")
        n, sr = row["n_resolved"], row["trade_sharpe"]
        g3 = row.get("roi_skew", 0.0) or 0.0
        g4 = row.get("roi_kurt", 3.0) or 3.0
        denom = 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2
        if denom <= 0 or n < 3:
            return float("nan")
        return N.cdf((sr - sr_star) * math.sqrt(n - 1) / math.sqrt(denom))

    s["deflated_sharpe"] = s.apply(_dsr, axis=1)
    cols = ["edge_shrunk", "edge_tstat", "brier_skill", "trade_sharpe"]
    z = []
    for c in cols:
        if c in s:
            v = s.loc[s["eligible"], c]
            sd = v.std()
            z.append(((s[c] - v.mean()) / sd) if sd and sd > 0 else s[c] * 0)
    s["sf_score"] = pd.concat(z, axis=1).mean(axis=1) if z else float("nan")
    s.loc[~s["eligible"], "sf_score"] = float("nan")
    return s.sort_values("sf_score", ascending=False, na_position="last")


def roi_moments(df: pd.DataFrame) -> dict:
    d = resolved_only(df)
    if len(d) < 3:
        return {"roi_skew": 0.0, "roi_kurt": 3.0}
    g3, g4 = _moments(d["roi"].to_numpy())
    return {"roi_skew": g3, "roi_kurt": g4}
