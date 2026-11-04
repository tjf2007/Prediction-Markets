import os
import tempfile
import unittest

import pandas as pd

from superforecasters import categorize, monitor, pipeline, report
from superforecasters.sources.kalshi import import_positions_csv
from superforecasters.sources.polymarket import PolymarketClient, _ts
from superforecasters.storage import Store
from tests.fixtures import make_trader

# Shapes mirror Polymarket data-api responses (fields we rely on).
LB = [{"rank": "1", "proxyWallet": "0xAbC", "userName": "macroguy", "vol": 1e6, "pnl": 2e5},
      {"rank": "2", "proxyWallet": "0xdef", "userName": "ratesgal", "vol": 5e5, "pnl": 1e5}]
CLOSED = [{"proxyWallet": "0xabc", "asset": "tok1", "conditionId": "c1", "avgPrice": 0.4, "totalBought": 1000,
           "realizedPnl": 600, "curPrice": 1, "title": "Fed decreases interest rates by 25 bps?", "slug": "s1",
           "eventSlug": "fed-dec", "outcome": "Yes", "outcomeIndex": 0, "endDate": "2025-12-10T00:00:00Z",
           "timestamp": 1765400000}]
OPEN = [{"proxyWallet": "0xabc", "asset": "tok2", "conditionId": "c2", "size": 500, "avgPrice": 0.3, "initialValue": 150,
         "currentValue": 200, "cashPnl": 50, "curPrice": 0.4, "title": "CPI above 3%?", "eventSlug": "cpi", "outcome": "No",
         "endDate": "2026-12-01"}]


class FakeHttp:
    def __init__(self, routes):
        self.routes = routes

    def get(self, path, params=None, **kw):
        v = self.routes.get(path, [])
        off = (params or {}).get("offset", 0)
        return v if off == 0 or isinstance(v, dict) else []


def fake_pm(open_positions=OPEN):
    pm = PolymarketClient()
    pm.data = FakeHttp({"/v1/leaderboard": LB, "/closed-positions": CLOSED, "/positions": open_positions, "/activity": []})
    pm.gamma = FakeHttp({"/events": [{"tags": [{"label": "Economy"}, {"label": "Fed Rates"}]}]})
    return pm


class TestSources(unittest.TestCase):
    def test_ts(self):
        self.assertEqual(_ts("1970-01-01T00:00:10Z"), 10)
        self.assertEqual(_ts(10_000), 10_000)
        self.assertEqual(_ts(10_000_000_000_000), 10_000_000_000)
        self.assertEqual(_ts("2026-12-01"), _ts("2026-12-01T00:00:00Z"))

    def test_normalize(self):
        pm = fake_pm()
        lb = pm.leaderboard("ECONOMICS")
        self.assertEqual(lb[0]["trader_id"], "0xabc")
        c = pm.closed_positions("0xabc")[0]
        self.assertEqual(c["resolved_value"], 1)
        self.assertAlmostEqual(c["cost"], 400)
        o = pm.open_positions("0xabc")[0]
        self.assertIsNone(o["resolved_value"])
        self.assertEqual(o["cost"], 150)
        self.assertEqual(pm.event_tags("fed-dec"), ["economy", "fed rates"])

    def test_categorize(self):
        self.assertTrue(categorize.is_fin_econ("Will the Fed cut rates in March?"))
        self.assertTrue(categorize.is_fin_econ("Random title", ["Economy"]))
        self.assertFalse(categorize.is_fin_econ("Will the Lakers win the title?", ["sports"]))
        self.assertEqual(categorize.primary_topic("October CPI above 3%?"), "inflation")

    def test_kalshi_csv(self):
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as fh:
            fh.write("trader_id,market_id,title,outcome,avg_price,shares,resolved,closed_at\n"
                     "k1,KXFED-25DEC-T4.00,Fed above 4%,NO,0.3,100,1,2025-12-11T00:00:00Z\n")
        rows = import_positions_csv(fh.name)
        os.unlink(fh.name)
        self.assertAlmostEqual(rows[0]["realized_pnl"], 70)


class TestPipelineMonitor(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = Store(os.path.join(self.tmp, "t.db"))

    def test_end_to_end_offline(self):
        pm = fake_pm()
        # extra synthetic traders so ranking has a cross-section
        for i, sk in enumerate([0.0, 0.02, 0.1]):
            df = make_trader(f"0x{i}", 60, sk, 50 + i)
            rows = df.to_dict("records")
            for r in rows:
                r["fin_econ"] = True
            self.store.save_positions(rows)
            pipeline.score_and_store(self.store, "polymarket", f"0x{i}", f"syn{i}")
        r = pipeline.run_polymarket(self.store, pm, depth=50, max_traders=5, min_resolved=20)
        self.assertIn("0xabc", set(r.trader_id))
        self.assertEqual(r.iloc[0]["trader_id"], "0x2")

        wl = r[r.eligible].head(3)
        ev = monitor.snapshot(self.store, pm, wl)
        self.assertTrue(ev.empty)  # baseline
        pm.data.routes["/positions"] = [dict(OPEN[0], size=2000)]
        ev = monitor.snapshot(self.store, pm, wl)
        self.assertEqual(set(ev.kind), {"ADD"})
        pm.data.routes["/positions"] = []
        ev = monitor.snapshot(self.store, pm, wl)
        self.assertEqual(set(ev.kind), {"EXIT"})
        pm.data.routes["/positions"] = OPEN
        monitor.snapshot(self.store, pm, wl)
        c = monitor.consensus(self.store, wl)
        self.assertEqual(int(c.iloc[0]["n_sf"]), len(wl))

        html = report.build(r, c, {})
        self.assertIn("<table>", html)

    def test_diff(self):
        prev = pd.DataFrame([{"token_id": "a", "shares": 100, "avg_price": .5, "cur_price": .5, "title": "x", "outcome": "Yes"}])
        cur = pd.DataFrame([{"token_id": "b", "shares": 1000, "avg_price": .5, "cur_price": .5, "title": "y", "outcome": "No"}])
        kinds = {e["kind"] for e in monitor.diff_positions(prev, cur, min_usd=10)}
        self.assertEqual(kinds, {"NEW", "EXIT"})


if __name__ == "__main__":
    unittest.main()
