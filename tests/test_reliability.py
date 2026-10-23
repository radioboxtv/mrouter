"""Reliability scoring tests — TDD, written before implementation."""

import json
import os
import time
import unittest

import tests.fixtures as fx

NOW = time.time()


def req(key, status, age_h=0.0):
    return {"type": "request", "ts": NOW - age_h * 3600,
            "use_case": "chat", "routed_model_key": key, "status_code": status}


class TestComputeReliability(fx.ConfigIsolatedTestCase):
    """Slice 1: per-model reliability from route-log entries."""

    def compute(self, entries):
        from mrouter import reliability
        return reliability.compute_reliability(self.make_config(), entries)

    def test_no_entries_gives_prior_score(self):
        stats = self.compute([])
        self.assertEqual(stats, {})
        from mrouter import reliability
        self.assertEqual(reliability.PRIOR, 85.0)

    def test_model_absent_from_stats_uses_prior(self):
        from mrouter import reliability
        stats = self.compute([req("m1", 200)])
        self.assertNotIn("m2", stats)
        self.assertEqual(reliability.reliability_score(stats, "m2"), 85.0)

    def test_all_ok_scores_high(self):
        stats = self.compute([req("m1", 200, age_h=1)] * 10)
        self.assertGreater(stats["m1"]["score"], 95)
        self.assertEqual(stats["m1"]["success_rate"], 1.0)
        self.assertEqual(stats["m1"]["samples"], 10)

    def test_all_errors_scores_low(self):
        stats = self.compute([req("m1", 503, age_h=1)] * 10)
        self.assertLess(stats["m1"]["score"], 25)
        self.assertEqual(stats["m1"]["success_rate"], 0.0)

    def test_shrinkage_damps_small_samples(self):
        """1 failure out of 1 request must not crater the score the way
        10 failures out of 10 do."""
        one = self.compute([req("m1", 500, age_h=1)])
        ten = self.compute([req("m1", 500, age_h=1)] * 10)
        self.assertGreater(one["m1"]["score"], 60)   # close to prior
        self.assertLess(ten["m1"]["score"], one["m1"]["score"])

    def test_recent_failures_dominate_ancient_ones(self):
        """Same error count, but recent failures must score worse than old ones."""
        fresh_fails = [req("m1", 200, age_h=200)] * 5 + [req("m1", 500, age_h=0.5)] * 5
        old_fails = [req("m1", 200, age_h=0.5)] * 5 + [req("m1", 500, age_h=200)] * 5
        s_fresh = self.compute(fresh_fails)["m1"]["score"]
        s_old = self.compute(old_fails)["m1"]["score"]
        self.assertLess(s_fresh, s_old)

    def test_4xx_and_5xx_both_count_as_failures(self):
        stats = self.compute([req("m1", 200, age_h=1), req("m1", 429, age_h=1)])
        self.assertEqual(stats["m1"]["success_rate"], 0.5)

    def test_feedback_and_none_key_entries_ignored(self):
        entries = [
            {"type": "feedback", "use_case": "chat", "routed_model_key": "m1", "rating": 1},
            req(None, 200),
        ]
        self.assertEqual(self.compute(entries), {})


class TestReliabilityFloor(fx.ConfigIsolatedTestCase):
    """Slice 1b: below-floor exclusion predicate."""

    def test_below_floor_with_enough_samples(self):
        from mrouter import reliability
        stats = {  # 30% success over 10 samples
            "m1": {"samples": 10, "success_rate": 0.3, "score": 33.0},
        }
        self.assertTrue(reliability.is_below_floor(stats, "m1", floor=0.5, min_samples=5))

    def test_below_floor_but_too_few_samples_is_kept(self):
        from mrouter import reliability
        stats = {"m1": {"samples": 2, "success_rate": 0.0, "score": 30.0}}
        self.assertFalse(reliability.is_below_floor(stats, "m1", floor=0.5, min_samples=5))

    def test_healthy_model_kept(self):
        from mrouter import reliability
        stats = {"m1": {"samples": 10, "success_rate": 0.9, "score": 92.0}}
        self.assertFalse(reliability.is_below_floor(stats, "m1", floor=0.5, min_samples=5))

    def test_unknown_model_never_excluded(self):
        from mrouter import reliability
        self.assertFalse(reliability.is_below_floor({}, "mystery", floor=0.5, min_samples=5))


class TestReliabilityWiring(fx.ConfigIsolatedTestCase):
    """Slice 2: reliability component + below-floor swap-out in score_fleet."""

    def _cfg(self):
        cfg = self.make_config()
        cfg.setdefault("routing", {})["reliability_floor"] = 0.5
        cfg["routing"]["reliability_min_samples"] = 5
        return cfg

    def test_reliability_component_injected(self):
        from mrouter import heuristics
        cfg = self._cfg()
        fleet = fx.standard_fleet()
        stats = {"qwen/qwen3.8-27b": {"samples": 10, "success_rate": 0.9, "score": 92.0}}
        ranked = heuristics.score_fleet(fleet, cfg, bench={}, reliability=stats)
        row = next(m for m in ranked["coding"]["models"] if m["key"] == "qwen/qwen3.8-27b")
        self.assertEqual(row["components"]["reliability"], 92.0)

    def test_unobserved_models_get_neutral_prior(self):
        from mrouter import heuristics
        ranked = heuristics.score_fleet(fx.standard_fleet(), self._cfg(), bench={}, reliability={})
        for m in ranked["chat"]["models"]:
            self.assertEqual(m["components"]["reliability"], 85.0)

    def test_failing_model_excluded_below_floor(self):
        from mrouter import heuristics
        cfg = self._cfg()
        fleet = fx.standard_fleet()
        stats = {"qwen/qwen3.8-27b": {"samples": 10, "success_rate": 0.1, "score": 12.0}}
        ranked = heuristics.score_fleet(fleet, cfg, bench={}, reliability=stats)
        keys = [m["key"] for m in ranked["coding"]["models"]]
        self.assertNotIn("qwen/qwen3.8-27b", keys)
        self.assertTrue(keys)  # healthy alternatives still present

    def test_below_floor_but_thin_samples_kept(self):
        from mrouter import heuristics
        cfg = self._cfg()
        fleet = fx.standard_fleet()
        stats = {"qwen/qwen3.8-27b": {"samples": 2, "success_rate": 0.0, "score": 8.0}}
        ranked = heuristics.score_fleet(fleet, cfg, bench={}, reliability=stats)
        keys = [m["key"] for m in ranked["coding"]["models"]]
        self.assertIn("qwen/qwen3.8-27b", keys)

    def test_reliability_affects_ranking_without_reaching_floor(self):
        """A capable-but-flaky model above the floor must lose to a dependable one."""
        from mrouter import heuristics
        cfg = self._cfg()
        fleet = fx.standard_fleet()
        # qwen is otherwise the coding winner; 61 reliability drops it below
        # a healthy cloud model weighted at the 85 prior.
        stats = {"qwen/qwen3.8-27b": {"samples": 10, "success_rate": 0.6, "score": 61.0}}
        ranked = heuristics.score_fleet(fleet, cfg, bench={}, reliability=stats)
        self.assertNotEqual(ranked["coding"]["models"][0]["key"], "qwen/qwen3.8-27b")


if __name__ == "__main__":
    unittest.main()