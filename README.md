# Superforecasters — Finance & Economics prediction-market traders

Find the traders with real, persistent skill on Polymarket (and, where data allows, Kalshi)
in the **Economics** and **Finance** categories; monitor what they're betting on; and
decompose *how* they make money.

```bash
pip install -e .
sf discover --depth 500 --max-traders 300   # pull Econ+Finance leaderboards, ingest, score
sf rank --top 30                             # ranked superforecasters
sf profile 0xWALLET --clv                    # process decomposition for one trader
sf watch --top 25                            # snapshot positions; run on a cron to get NEW/ADD/TRIM/EXIT
sf consensus --top 25                        # what the top 25 hold right now, skill-weighted
sf report --clv --out data/report.html       # HTML: ranking + consensus + top profiles
python -m unittest discover -s tests -t .    # tests (synthetic traders with known skill)
```

## Pipeline

1. **Discover** — union of the Polymarket `ECONOMICS` and `FINANCE` leaderboards (all-time and
   monthly), ranked by both PnL *and* volume. PnL-only boards select for lucky whales; the
   volume boards add high-activity traders whose per-bet edge may be better.
2. **Ingest** — every closed position per wallet (`data-api /closed-positions`), each tagged
   Finance/Econ via Gamma event tags + a title keyword classifier (`categorize.py`). Only
   Finance/Econ positions are scored, so a sports whale who dabbles in CPI doesn't qualify on
   sports PnL.
3. **Score** (`metrics.py`) — per trader:

| Metric | What it is | Why |
|---|---|---|
| `edge` | mean(outcome − entry price) | Exactly −½ the marginal Brier improvement the trader adds over the market. The core "superforecaster" number. |
| `edge_tstat`, `edge_shrunk` | t-stat; edge shrunk toward 0 with a 30-bet prior | Separates skill from a short lucky streak. |
| `brier_market_at_entry`, `brier_implied`, `brier_skill` | Brier of the market vs. a size-implied belief (bet size read as Kelly conviction, quarter-Kelly default) | Traders don't state probabilities; this reads them from sizing. |
| `brier_skill_oos` | Same, conviction fitted on first half of history, scored on second | Walk-forward, so it can't be gamed in-sample. |
| `trade_sharpe`, `daily_sharpe_ann` | Sharpe of per-position ROI; annualised Sharpe of daily PnL | Risk-adjusted return. |
| `psr_trade` | Probabilistic Sharpe (Bailey & López de Prado) | Adjusts for the very fat tails of binary payoffs. |
| `deflated_sharpe` | PSR vs. the expected *max* Sharpe of N skill-less traders | Corrects for scanning hundreds of names on a leaderboard. |
| `sf_score` | Mean z-score of shrunk edge, edge t, Brier skill, Sharpe | Composite rank (min 20 resolved positions). |

4. **Decompose process** (`process.py`):
   - **CLV split**: `edge = (p_close − p_entry) + (outcome − p_close)`. CLV-driven traders read data
     faster than the crowd; residual-driven traders hold non-consensus views to expiry (or got lucky).
   - **Price profile**: longshot hunter vs favourite grinder, with hit rate/edge per bucket.
   - **Timing**: days before resolution at entry; pre-entry price move (momentum vs fade).
   - **Sizing skill**: rank-correlation of bet size with edge — do they bet bigger when right?
   - **Topic specialisation**: Fed/rates, inflation, labour, growth, equities, FX, commodities, fiscal/trade, housing; HHI concentration.
   - **Side bias**: YES vs NO preference and edge on each; scaling-in (fills/position); exits before resolution.
   - `style_label` turns these into a one-line archetype.
5. **Monitor** (`monitor.py`) — position snapshots in SQLite; diffs produce NEW/ADD/TRIM/EXIT events;
   `consensus` aggregates the watchlist per market with skill weights, flags contested markets,
   and shows SF average entry vs current price.

## Kalshi: what's possible

Kalshi's public API is **anonymous** — markets, settlements, and the trade tape are public, but
trades are not attributed to users, so other people's track records can't be scraped from the
API. Supported:

- `sf kalshi-markets` — open Economics/Financials markets.
- `sf kalshi-whales --min 5000` — large anonymous prints on those markets (crowd-level smart-money proxy).
- `sf import-csv FILE` — score any track records you collect yourself (columns:
  `trader_id,market_id,title,outcome,avg_price,shares,resolved,closed_at[,realized_pnl]`).
- `KalshiClient.my_settlements()` — your own account via the signed portfolio API, useful as a benchmark.

## Caveats

- Polymarket's data-api fields are parsed defensively; run `sf probe` first to check the live schema.
- Positions sold before resolution are still scored on the resolved outcome for `edge`, but PnL uses realized PnL.
- One person can run several wallets; scores are per wallet.
