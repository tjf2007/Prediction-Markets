"""SQLite persistence: scored traders, resolved positions, and open-position snapshots."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS leaderboard (platform TEXT, trader_id TEXT, name TEXT, category TEXT, period TEXT,
    order_by TEXT, rank INTEGER, pnl REAL, volume REAL, fetched_ts REAL);
CREATE TABLE IF NOT EXISTS scores (platform TEXT, trader_id TEXT, name TEXT, scored_ts REAL, payload TEXT,
    PRIMARY KEY (platform, trader_id));
CREATE TABLE IF NOT EXISTS positions (platform TEXT, trader_id TEXT, token_id TEXT, market_id TEXT, title TEXT,
    event_slug TEXT, outcome TEXT, avg_price REAL, shares REAL, cost REAL, realized_pnl REAL, resolved_value REAL,
    end_ts REAL, closed_ts REAL, fin_econ INTEGER, PRIMARY KEY (platform, trader_id, token_id, closed_ts));
CREATE TABLE IF NOT EXISTS snapshots (snap_ts REAL, platform TEXT, trader_id TEXT, token_id TEXT, market_id TEXT,
    title TEXT, event_slug TEXT, outcome TEXT, shares REAL, avg_price REAL, cur_price REAL, current_value REAL);
CREATE INDEX IF NOT EXISTS ix_snap ON snapshots (trader_id, snap_ts);
CREATE TABLE IF NOT EXISTS events (ts REAL, platform TEXT, trader_id TEXT, kind TEXT, token_id TEXT, title TEXT,
    outcome TEXT, delta_shares REAL, delta_usd REAL, price REAL);
CREATE TABLE IF NOT EXISTS tags (event_slug TEXT PRIMARY KEY, tags TEXT);
"""


class Store:
    def __init__(self, path: str | Path = "data/sf.db"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(path))
        self.con.executescript(SCHEMA)

    def q(self, sql: str, params=()) -> pd.DataFrame:
        return pd.read_sql_query(sql, self.con, params=params)

    # leaderboard
    def save_leaderboard(self, rows: list[dict]):
        now = time.time()
        self.con.executemany("INSERT INTO leaderboard VALUES (?,?,?,?,?,?,?,?,?,?)", [
            (r["platform"], r["trader_id"], r.get("name"), r["category"], r["period"], r["order_by"], r.get("rank"),
             r.get("pnl"), r.get("volume"), now) for r in rows])
        self.con.commit()

    # tags cache
    def get_tags(self, slug: str) -> list[str] | None:
        r = self.con.execute("SELECT tags FROM tags WHERE event_slug=?", (slug,)).fetchone()
        return json.loads(r[0]) if r else None

    def put_tags(self, slug: str, tags: list[str]):
        self.con.execute("INSERT OR REPLACE INTO tags VALUES (?,?)", (slug, json.dumps(tags)))
        self.con.commit()

    def all_tags(self) -> dict[str, list[str]]:
        return {s: json.loads(t) for s, t in self.con.execute("SELECT event_slug, tags FROM tags")}

    # positions
    def save_positions(self, rows: list[dict]):
        self.con.executemany("INSERT OR REPLACE INTO positions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [
            (r["platform"], r["trader_id"], r["token_id"], r["market_id"], r["title"], r["event_slug"], r["outcome"],
             r["avg_price"], r["shares"], r["cost"], r["realized_pnl"], r["resolved_value"], r["end_ts"],
             r["closed_ts"] or 0, int(bool(r.get("fin_econ")))) for r in rows])
        self.con.commit()

    def positions(self, trader_id: str, fin_econ_only: bool = True) -> pd.DataFrame:
        sql = "SELECT * FROM positions WHERE trader_id=?" + (" AND fin_econ=1" if fin_econ_only else "")
        return self.q(sql, (trader_id,))

    # scores
    def save_score(self, platform: str, trader_id: str, name: str | None, payload: dict):
        self.con.execute("INSERT OR REPLACE INTO scores VALUES (?,?,?,?,?)",
                         (platform, trader_id, name, time.time(), json.dumps(payload, default=float)))
        self.con.commit()

    def scores(self) -> pd.DataFrame:
        rows = []
        for platform, tid, name, ts, payload in self.con.execute("SELECT * FROM scores"):
            rows.append({"platform": platform, "trader_id": tid, "name": name, "scored_ts": ts, **json.loads(payload)})
        return pd.DataFrame(rows)

    # snapshots
    def save_snapshot(self, ts: float, rows: list[dict]):
        self.con.executemany("INSERT INTO snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", [
            (ts, r["platform"], r["trader_id"], r["token_id"], r["market_id"], r["title"], r["event_slug"], r["outcome"],
             r["shares"], r["avg_price"], r["cur_price"], r["current_value"]) for r in rows])
        self.con.commit()

    def last_snapshot(self, trader_id: str, before: float | None = None) -> pd.DataFrame:
        """Most recent snapshot for a trader, optionally strictly before `before`."""
        r = self.con.execute("SELECT MAX(snap_ts) FROM snapshots WHERE trader_id=? AND snap_ts < ?",
                             (trader_id, before if before is not None else float("inf"))).fetchone()
        if not r or r[0] is None:
            return pd.DataFrame()
        return self.q("SELECT * FROM snapshots WHERE trader_id=? AND snap_ts=?", (trader_id, r[0]))

    def save_events(self, rows: list[dict]):
        self.con.executemany("INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?)", [
            (r["ts"], r["platform"], r["trader_id"], r["kind"], r["token_id"], r["title"], r["outcome"],
             r["delta_shares"], r["delta_usd"], r["price"]) for r in rows])
        self.con.commit()
