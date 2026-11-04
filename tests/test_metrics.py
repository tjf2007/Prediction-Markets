import math
import unittest

import numpy as np
import pandas as pd

from superforecasters import metrics, process
from tests.fixtures import make_trader


class TestMetrics(unittest.TestCase):
    def test_skilled_beats_unskilled(self):
        sk = metrics.score_trader(make_trader("a", 400, 0.12, 1))
        un = metrics.score_trader(make_trader("b", 400, 0.0, 2))
        self.assertAlmostEqual(sk["edge"], 0.12, delta=0.05)
        self.assertAlmostEqual(un["edge"], 0.0, delta=0.05)
        self.assertGreater(sk["brier_skill"], 0)
        self.assertLess(sk["brier_implied"], sk["brier_market_at_entry"])
        self.assertGreater(sk["edge_tstat"], 3)
        self.assertGreater(sk["trade_sharpe"], un["trade_sharpe"])
        self.assertGreater(sk["psr_trade"], 0.95)

    def test_edge_is_marginal_brier_improvement(self):
        d = metrics.resolved_only(make_trader("a", 200, 0.05, 3))
        eps = 1e-6
        b0 = ((d.p - d.y) ** 2).mean()
        b1 = ((d.p + eps - d.y) ** 2).mean()
        self.assertAlmostEqual((b0 - b1) / eps / 2, d.edge.mean(), places=4)

    def test_rank_and_deflation(self):
        traders = [make_trader(f"u{i}", 60, 0.0, 100 + i) for i in range(40)]
        traders += [make_trader("star", 300, 0.15, 999)]
        rows = []
        for t in traders:
            rows.append({"trader_id": t.trader_id.iloc[0], "name": None, **metrics.score_trader(t), **metrics.roi_moments(t)})
        r = metrics.rank_traders(pd.DataFrame(rows), min_resolved=20)
        self.assertEqual(r.iloc[0]["trader_id"], "star")
        self.assertGreater(r.iloc[0]["sr_hurdle"], 0)
        # Deflated Sharpe of skill-less traders should mostly be low
        self.assertLess(r[r.trader_id != "star"]["deflated_sharpe"].median(), 0.5)

    def test_ineligible_excluded(self):
        rows = [{"trader_id": "few", **metrics.score_trader(make_trader("few", 5, 0.3, 4))},
                {"trader_id": "many", **metrics.score_trader(make_trader("many", 100, 0.0, 5))},
                {"trader_id": "many2", **metrics.score_trader(make_trader("many2", 100, 0.05, 6))}]
        r = metrics.rank_traders(pd.DataFrame(rows), min_resolved=20)
        self.assertTrue(math.isnan(r.set_index("trader_id").loc["few", "sf_score"]))

    def test_calibration(self):
        t = metrics.calibration_table(metrics.resolved_only(make_trader("a", 500, 0.1, 7)))
        self.assertTrue((t["edge"] > -0.05).all())

    def test_empty(self):
        self.assertEqual(metrics.score_trader(make_trader("a", 0, 0, 1))["n_resolved"], 0)


class TestProcess(unittest.TestCase):
    def test_clv_identity_and_profiles(self):
        df = make_trader("a", 80, 0.1, 11)
        d = metrics.resolved_only(df)
        # market drifts halfway toward the truth by close
        def hist(tok):
            r = d[d.token_id == tok].iloc[0]
            return [(r.end_ts - 30 * 86400, r.p), (r.end_ts - 2 * 86400, (r.p + r.y) / 2)]
        act = pd.DataFrame({"token_id": d.token_id, "ts": d.end_ts - 20 * 86400, "side": "BUY", "usd": d.cost})
        out = process.decompose(df, act, hist)
        clv = out["clv"]
        np.testing.assert_allclose(clv.clv + clv.residual, clv.y - clv.p_entry)
        self.assertAlmostEqual(out["summary"]["clv_share_of_edge"], 0.5, delta=0.01)
        self.assertAlmostEqual(out["summary"]["median_days_before_end"], 20, delta=0.01)
        self.assertEqual(len(out["price_profile"]), 5)
        self.assertIn("fed_rates", set(out["topics"].topic))
        self.assertIsInstance(process.style_label(out["summary"], {}), str)


if __name__ == "__main__":
    unittest.main()
