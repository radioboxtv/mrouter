"""Tests for heuristics' live-speed-score fallback chain: metrics_store (live
traffic) -> bench_results.json (synthetic micro-bench) -> params-based estimate.
"""

import copy
import unittest

import tests.fixtures as fx
from mrouter import heuristics


class LiveSpeedScoreTestCase(fx.ConfigIsolatedTestCase):
    def setUp(self):
        super().setUp()
        self.fleet = fx.standard_fleet()
        self.config = self.make_config()

    def test_live_speed_used_when_enough_fresh_samples(self):
        local = next(m for m in self.fleet if m["is_local"] and m["key"] == "qwen/qwen3.8-27b")
        other_local = next(m for m in self.fleet if m["is_local"] and m["key"] != local["key"])
        live_speed = {
            local["key"]: {"tokens_per_sec_p90": 100.0, "samples": 10},
            other_local["key"]: {"tokens_per_sec_p90": 50.0, "samples": 10},
        }
        score = heuristics._speed_score(local, self.fleet, bench=None, fam_info={}, live_speed=live_speed)
        # 100 is the fleet-best live speed among locals -> normalized to 100
        self.assertEqual(score, 100.0)

    def test_live_speed_ignored_below_min_samples(self):
        local = next(m for m in self.fleet if m["is_local"] and m["key"] == "qwen/qwen3.8-27b")
        other_local = next(m for m in self.fleet if m["is_local"] and m["key"] != local["key"])
        live_speed = {local["key"]: {"tokens_per_sec_p90": 999.0, "samples": 1}}
        bench = {
            local["key"]: {"ts": __import__("time").time(), "tokens_per_sec": 42.0},
            other_local["key"]: {"ts": __import__("time").time(), "tokens_per_sec": 84.0},
        }
        score = heuristics._speed_score(local, self.fleet, bench=bench, fam_info={},
                                         live_speed=live_speed, live_min_samples=5)
        # thin live sample (1 < min 5) ignored -> bench path used: 42/84 * 100 = 50, not 100
        self.assertEqual(score, 50.0)

    def test_falls_back_to_bench_when_no_live_speed(self):
        local = next(m for m in self.fleet if m["is_local"] and m["key"] == "qwen/qwen3.8-27b")
        bench = {local["key"]: {"ts": __import__("time").time(), "tokens_per_sec": 42.0}}
        score_with_live_none = heuristics._speed_score(local, self.fleet, bench=bench, fam_info={}, live_speed=None)
        score_without_live_param = heuristics._speed_score(local, self.fleet, bench=bench, fam_info={})
        self.assertEqual(score_with_live_none, score_without_live_param)

    def test_score_fleet_loads_live_speed_from_metrics_store_by_default(self):
        from mrouter import metrics_store
        now = __import__("time").time()
        local = next(m for m in self.fleet if m["is_local"])
        for _ in range(6):
            metrics_store.record(self.config, {
                "ts": now, "use_case": "chat", "model_key": local["key"],
                "provider": "lmstudio", "status_code": 200, "ttft_s": 0.1,
                "duration_s": 1.1, "tokens_in": 10, "tokens_out": 100,
                "cost_usd": 0.0, "cost_is_estimate": False,
            })
        # Should not raise, and should produce a ranked fleet (live speed feed-in
        # is exercised implicitly via the default live_speed=None -> auto-load path).
        ranked = heuristics.score_fleet(self.fleet, self.config, bench={})
        self.assertIn("chat", ranked)
        self.assertTrue(ranked["chat"]["models"])


if __name__ == "__main__":
    unittest.main()
