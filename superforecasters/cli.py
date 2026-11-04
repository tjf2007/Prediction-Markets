"""sf - superforecaster discovery & monitoring CLI.

  sf probe                       dump raw API samples (schema check)
  sf discover [--depth 500]      pull Econ+Finance leaderboards, ingest & score candidates
  sf rank [--top 30]             print the ranked list
  sf profile WALLET [--clv]      decompose one trader's process
  sf watch [--top 25]            snapshot open positions of the top N, print changes
  sf consensus [--top 25]        what the top N hold right now
  sf report [--out report.html]  HTML report (ranking + consensus + top profiles)
  sf kalshi-markets              list open Kalshi Economics/Financials markets
  sf kalshi-whales [--min 5000]  large anonymous prints on Kalshi econ markets
  sf import-csv FILE             score track records you collected (e.g. Kalshi traders)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import pandas as pd

from . import monitor, pipeline, process, report
from .sources.kalshi import KalshiClient, import_positions_csv
from .sources.polymarket import PolymarketClient
from .storage import Store

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

RANK_COLS = ["name", "trader_id", "sf_score", "n_resolved", "edge", "edge_tstat", "hit_rate", "brier_skill",
             "trade_sharpe", "daily_sharpe_ann", "deflated_sharpe", "roi", "total_pnl"]


def _watchlist(store: Store, top: int, min_resolved: int) -> pd.DataFrame:
    r = pipeline.ranked(store, min_resolved)
    if r.empty:
        sys.exit("No scored traders yet - run `sf discover` first.")
    return r[r["eligible"]].head(top)


def _profile(store: Store, pm: PolymarketClient, wallet: str, clv: bool, name=None) -> dict:
    df = store.positions(wallet)
    if df.empty:
        pipeline.ingest_trader(pm, store, wallet)
        df = store.positions(wallet)
    act = pd.DataFrame(pm.activity(wallet))
    dec = process.decompose(df, act, pm.price_history if clv else None, store.all_tags())
    sc = pipeline.score_and_store(store, "polymarket", wallet, name)
    dec["style"] = process.style_label(dec["summary"], sc)
    dec["name"] = name
    return dec


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sf", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=os.environ.get("SF_DB", "data/sf.db"))
    ap.add_argument("--cache", default=os.environ.get("SF_CACHE", "data/cache"))
    ap.add_argument("--min-resolved", type=int, default=20)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe"); p.add_argument("--wallet")
    p = sub.add_parser("discover"); p.add_argument("--depth", type=int, default=500); p.add_argument("--max-traders", type=int, default=300)
    p = sub.add_parser("rank"); p.add_argument("--top", type=int, default=30)
    p = sub.add_parser("profile"); p.add_argument("wallet"); p.add_argument("--clv", action="store_true")
    p = sub.add_parser("watch"); p.add_argument("--top", type=int, default=25); p.add_argument("--min-usd", type=float, default=250)
    p = sub.add_parser("consensus"); p.add_argument("--top", type=int, default=25)
    p = sub.add_parser("report"); p.add_argument("--out", default="data/report.html"); p.add_argument("--top", type=int, default=25)
    p.add_argument("--profiles", type=int, default=5); p.add_argument("--clv", action="store_true")
    sub.add_parser("kalshi-markets")
    p = sub.add_parser("kalshi-whales"); p.add_argument("--min", type=float, default=5000); p.add_argument("--hours", type=float, default=24)
    p = sub.add_parser("import-csv"); p.add_argument("file"); p.add_argument("--platform", default="kalshi")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(message)s")

    store = Store(a.db)
    pm = PolymarketClient(cache_dir=a.cache)

    if a.cmd == "probe":
        print(json.dumps(pm.raw_samples(a.wallet), indent=2)[:8000])
    elif a.cmd == "discover":
        r = pipeline.run_polymarket(store, pm, depth=a.depth, max_traders=a.max_traders, min_resolved=a.min_resolved)
        if r.empty:
            sys.exit("No leaderboard data returned - is data-api.polymarket.com reachable? (try `sf -v discover`)")
        print(r[[c for c in RANK_COLS if c in r]].head(30).to_string(index=False))
    elif a.cmd == "rank":
        r = pipeline.ranked(store, a.min_resolved)
        print(r[[c for c in RANK_COLS if c in r]].head(a.top).to_string(index=False))
    elif a.cmd == "profile":
        d = _profile(store, pm, a.wallet.lower(), a.clv)
        print("Style:", d["style"]); print(json.dumps(d["summary"], indent=2, default=float))
        print(d["price_profile"].to_string(index=False)); print(d["topics"].to_string(index=False))
        if not d["clv"].empty:
            print(d["clv"].head(25).to_string(index=False))
    elif a.cmd == "watch":
        ev = monitor.snapshot(store, pm, _watchlist(store, a.top, a.min_resolved), min_usd=a.min_usd)
        print("Baseline snapshot saved (changes appear from next run)." if ev.empty else
              ev[["name", "kind", "title", "outcome", "delta_usd", "price"]].to_string(index=False))
    elif a.cmd == "consensus":
        c = monitor.consensus(store, _watchlist(store, a.top, a.min_resolved))
        print("No snapshots yet - run `sf watch`." if c.empty else c.head(40).to_string(index=False))
    elif a.cmd == "report":
        wl = _watchlist(store, a.top, a.min_resolved)
        profiles = {t: _profile(store, pm, t, a.clv, n) for t, n in zip(wl["trader_id"].head(a.profiles), wl["name"].head(a.profiles))}
        html = report.build(pipeline.ranked(store, a.min_resolved), monitor.consensus(store, wl), profiles)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        open(a.out, "w").write(html); print("wrote", a.out)
    elif a.cmd == "kalshi-markets":
        k = KalshiClient(cache_dir=a.cache)
        print(pd.DataFrame(k.finance_econ_markets("open")).sort_values("volume", ascending=False).head(50).to_string(index=False))
    elif a.cmd == "kalshi-whales":
        k = KalshiClient(cache_dir=a.cache)
        mk = pd.DataFrame(k.finance_econ_markets("open")).sort_values("volume", ascending=False).head(100)
        print(pd.DataFrame(k.whale_prints(mk["market_id"].tolist(), a.min, a.hours)).head(50).to_string(index=False))
    elif a.cmd == "import-csv":
        rows = import_positions_csv(a.file, a.platform)
        for r in rows:
            r["fin_econ"] = True
        store.save_positions(rows)
        for tid in {r["trader_id"] for r in rows}:
            pipeline.score_and_store(store, a.platform, tid, tid)
        r = pipeline.ranked(store, a.min_resolved)
        print(r[[c for c in RANK_COLS if c in r]].to_string(index=False))


if __name__ == "__main__":
    main()
