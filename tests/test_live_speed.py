"""Tests for heuristics' real-world-perf scoring: both speed (tokens/sec,
throughput) and ttft (time-to-first-token, latency) prefer live traffic from
metrics_store over guessed/curated numbers, for LOCAL AND CLOUD models alike.

Fallback chain:
  speed: live traffic -> local bench (measured) -> local params-based guess
         -> cloud: curated family speed_est (last resort, clearly labeled)
  ttft:  live traffic -> local bench (measured ttft_s) -> neutral prior
         (no invented numbers when nothing real is available)
"""

import time
import unittest

import tests.fixtures as fx
from mrouter import heuristics


class SpeedScoreTestCase(fx.ConfigIsolatedTestCase):
    def setUp(self):
        super().setUp()
        self.fleet = fx.standard_fleet()
        self.config = self.make_config()

    def test_live_speed_used_when_enough_fresh_samples(self):
        local = next(m for m in self.fleet if m["is_local"] and m["key"] == "qwen/qwen3.8-27b")
        other_local = next(m for m in self.fleet if m["is_local"] and m["key"] != local["key"])
        live_perf = {
            local["key"]: {"tokens_per_sec_p90": 100.0, "samples": 10},
            other_local["key"]: {"tokens_per_sec_p90": 50.0, "samples": 10},
        }
        score, source = heuristics._speed_score(local, self.fleet, bench=None, fam_info={}, live_perf=live_perf)
        self.assertEqual(score, 100.0)
        self.assertEqual(source, "live")

    def test_live_speed_ignored_below_min_samples(self):
        local = next(m for m in self.fleet if m["is_local"] and m["key"] == "qwen/qwen3.8-27b")
        other_local = next(m for m in self.fleet if m["is_local"] and m["key"] != local["key"])
        live_perf = {local["key"]: {"tokens_per_sec_p90": 999.0, "samples": 1}}
        bench = {
            local["key"]: {"ts": time.time(), "tokens_per_sec": 42.0},
            other_local["key"]: {"ts": time.time(), "tokens_per_sec": 84.0},
        }
        score, source = heuristics._speed_score(local, self.fleet, bench=bench, fam_info={},
                                                  live_perf=live_perf, live_min_samples=5)
        self.assertEqual(score, 50.0)
        self.assertEqual(source, "bench")

    def test_falls_back_to_bench_when_no_live_speed(self):
        local = next(m for m in self.fleet if m["is_local"] and m["key"] == "qwen/qwen3.8-27b")
        bench = {local["key"]: {"ts": time.time(), "tokens_per_sec": 42.0}}
        score_none, src_none = heuristics._speed_score(local, self.fleet, bench=bench, fam_info={}, live_perf=None)
        score_default, src_default = heuristics._speed_score(local, self.fleet, bench=bench, fam_info={})
        self.assertEqual(score_none, score_default)
        self.assertEqual(src_none, src_default)
        self.assertEqual(src_none, "bench")

    # ---- the actual gap this PR closes: cloud models must ALSO use live data

    def test_cloud_model_uses_live_speed_when_enough_samples(self):
        cloud = next(m for m in self.fleet if not m["is_local"])
        other_cloud = next(m for m in self.fleet if not m["is_local"] and m["key"] != cloud["key"])
        live_perf = {
            cloud["key"]: {"tokens_per_sec_p90": 80.0, "samples": 10},
            other_cloud["key"]: {"tokens_per_sec_p90": 40.0, "samples": 10},
        }
        score, source = heuristics._speed_score(cloud, self.fleet, bench=None,
                                                  fam_info={"speed_est": 55}, live_perf=live_perf)
        # Real traffic (80 tok/s, fleet-best among observed) must win over the
        # curated speed_est guess (55) once there's enough real data.
        self.assertEqual(score, 100.0)
        self.assertEqual(source, "live")

    def test_cloud_model_falls_back_to_curated_estimate_without_live_data(self):
        cloud = next(m for m in self.fleet if not m["is_local"])
        score, source = heuristics._speed_score(cloud, self.fleet, bench=None,
                                                  fam_info={"speed_est": 55}, live_perf=None)
        self.assertEqual(score, 55.0)
        self.assertEqual(source, "estimated")

    def test_cloud_model_live_ignored_below_min_samples_falls_back_to_estimate(self):
        cloud = next(m for m in self.fleet if not m["is_local"])
        live_perf = {cloud["key"]: {"tokens_per_sec_p90": 999.0, "samples": 1}}
        score, source = heuristics._speed_score(cloud, self.fleet, bench=None,
                                                  fam_info={"speed_est": 55},
                                                  live_perf=live_perf, live_min_samples=5)
        self.assertEqual(score, 55.0)
        self.assertEqual(source, "estimated")

    def test_thin_sample_peer_does_not_corrupt_normalization_for_others(self):
        """A single model with a thin (< live_min_samples), noisy live reading
        must NOT pollute the peer-max/peer-min used to normalize OTHER
        models' live scores. That peer itself correctly falls back to
        bench/estimated -- but the broken behavior was including its
        unqualified reading in the fleet-wide 'best' denominator anyway,
        silently crushing every properly-qualified model's score."""
        a = next(m for m in self.fleet if m["is_local"])
        b = next(m for m in self.fleet if m["is_local"] and m["key"] != a["key"])
        live_perf = {
            a["key"]: {"tokens_per_sec_p90": 40.0, "samples": 10},   # qualifies
            b["key"]: {"tokens_per_sec_p90": 10000.0, "samples": 1},  # thin -- must be excluded from normalization
        }
        score, source = heuristics._speed_score(a, self.fleet, bench=None, fam_info={},
                                                  live_perf=live_perf, live_min_samples=5)
        self.assertEqual(source, "live")
        # a is the only qualified live reading -> it must be its own normalization
        # peer (score 100), not crushed toward ~0 by b's disqualified reading.
        self.assertEqual(score, 100.0)


class TtftScoreTestCase(fx.ConfigIsolatedTestCase):
    def setUp(self):
        super().setUp()
        self.fleet = fx.standard_fleet()
        self.config = self.make_config()

    def test_live_ttft_used_when_enough_samples_lower_is_better(self):
        fast = next(m for m in self.fleet if m["is_local"])
        slow = next(m for m in self.fleet if not m["is_local"])
        live_perf = {
            fast["key"]: {"ttft_p90_s": 0.2, "samples": 10},
            slow["key"]: {"ttft_p90_s": 0.8, "samples": 10},
        }
        fast_score, fast_src = heuristics._ttft_score(fast, self.fleet, bench=None, live_perf=live_perf)
        slow_score, slow_src = heuristics._ttft_score(slow, self.fleet, bench=None, live_perf=live_perf)
        self.assertEqual(fast_score, 100.0)  # fastest observed TTFT in the fleet
        self.assertLess(slow_score, fast_score)
        self.assertEqual(fast_src, "live")
        self.assertEqual(slow_src, "live")

    def test_live_ttft_ignored_below_min_samples(self):
        local = next(m for m in self.fleet if m["is_local"])
        live_perf = {local["key"]: {"ttft_p90_s": 0.01, "samples": 1}}
        score, source = heuristics._ttft_score(local, self.fleet, bench=None,
                                                 live_perf=live_perf, live_min_samples=5)
        self.assertNotEqual(source, "live")

    def test_local_falls_back_to_bench_ttft(self):
        local = next(m for m in self.fleet if m["is_local"] and m["key"] == "qwen/qwen3.8-27b")
        other_local = next(m for m in self.fleet if m["is_local"] and m["key"] != local["key"])
        bench = {
            local["key"]: {"ts": time.time(), "ttft_s": 0.5},
            other_local["key"]: {"ts": time.time(), "ttft_s": 0.25},
        }
        score, source = heuristics._ttft_score(local, self.fleet, bench=bench, live_perf=None)
        self.assertEqual(source, "bench")
        self.assertLess(score, 100.0)  # other_local has the faster bench ttft

    def test_no_data_at_all_is_a_flat_neutral_prior_not_a_guess(self):
        """When there's no live traffic and no bench entry (e.g. a brand-new
        cloud model never routed before), the ttft score must be an honest,
        clearly-labeled neutral prior -- never an invented number derived
        from unrelated curated data."""
        cloud = next(m for m in self.fleet if not m["is_local"])
        score, source = heuristics._ttft_score(cloud, self.fleet, bench=None, live_perf=None)
        self.assertEqual(score, heuristics.NEUTRAL_PERF_SCORE)
        self.assertEqual(source, "estimated")

    def test_thin_sample_peer_does_not_corrupt_ttft_normalization_for_others(self):
        """Same corruption bug as speed, mirrored for TTFT: a disqualified
        (thin-sample) peer's freakishly-fast reading must not become the
        fleet-wide 'best' denominator for everyone else's live TTFT score."""
        a = next(m for m in self.fleet if m["is_local"])
        b = next(m for m in self.fleet if m["is_local"] and m["key"] != a["key"])
        live_perf = {
            a["key"]: {"ttft_p90_s": 0.5, "samples": 10},     # qualifies
            b["key"]: {"ttft_p90_s": 0.0001, "samples": 1},   # thin -- must be excluded
        }
        score, source = heuristics._ttft_score(a, self.fleet, bench=None,
                                                 live_perf=live_perf, live_min_samples=5)
        self.assertEqual(source, "live")
        self.assertEqual(score, 100.0)


class ComponentScoresIncludeTtftTestCase(fx.ConfigIsolatedTestCase):
    def setUp(self):
        super().setUp()
        self.fleet = fx.standard_fleet()
        self.config = self.make_config()

    def test_ttft_component_present_and_sources_reported(self):
        live_cache = ("curated", [])
        model = self.fleet[0]
        comps, notes, est_cost, sources = heuristics._component_scores(
            model, self.fleet, self.config, bench={}, live_cache=live_cache, live_perf={})
        self.assertIn("ttft", comps)
        self.assertIn("speed", sources)
        self.assertIn("ttft", sources)
        self.assertIn("capability", sources)

    def test_malformed_live_speed_min_samples_degrades_to_default_not_raise(self):
        """metrics.live_speed_min_samples is read on every single scored
        model, on every request -- a hand-edited config with a wrong-typed
        value (string, null, etc.) must degrade to the documented default,
        not take down routing by raising out of a hot path."""
        model = self.fleet[0]
        live_cache = ("curated", [])
        for bad_value in ("oops", None, [], {}):
            cfg = {**self.config, "metrics": {"live_speed_min_samples": bad_value}}
            comps, notes, est_cost, sources = heuristics._component_scores(
                model, self.fleet, cfg, bench={}, live_cache=live_cache, live_perf={})
            self.assertIn("speed", comps)
            self.assertIn("ttft", comps)

    def test_missing_metrics_block_entirely_degrades_to_default(self):
        model = self.fleet[0]
        live_cache = ("curated", [])
        for bad_metrics in (None, {}):
            cfg = {**self.config, "metrics": bad_metrics}
            comps, notes, est_cost, sources = heuristics._component_scores(
                model, self.fleet, cfg, bench={}, live_cache=live_cache, live_perf={})
            self.assertIn("speed", comps)
            self.assertIn("ttft", comps)


class ScoreFleetSurfacesPerfSourcesTestCase(fx.ConfigIsolatedTestCase):
    def setUp(self):
        super().setUp()
        self.fleet = fx.standard_fleet()
        self.config = self.make_config()

    def test_ranked_rows_report_speed_and_ttft_source(self):
        ranked = heuristics.score_fleet(self.fleet, self.config, bench={}, live_perf={})
        for uc, r in ranked.items():
            for row in r["models"]:
                self.assertIn("speed_source", row)
                self.assertIn("ttft_source", row)

    def test_score_fleet_loads_live_perf_from_metrics_store_by_default(self):
        from mrouter import metrics_store
        now = time.time()
        local = next(m for m in self.fleet if m["is_local"])
        for _ in range(6):
            metrics_store.record(self.config, {
                "ts": now, "use_case": "chat", "model_key": local["key"],
                "provider": "lmstudio", "status_code": 200, "ttft_s": 0.1,
                "duration_s": 1.1, "tokens_in": 10, "tokens_out": 100,
                "cost_usd": 0.0, "cost_is_estimate": False,
            })
        ranked = heuristics.score_fleet(self.fleet, self.config, bench={})
        row = next(r for r in ranked["chat"]["models"] if r["key"] == local["key"])
        self.assertEqual(row["speed_source"], "live")
        self.assertEqual(row["ttft_source"], "live")

    def test_cloud_model_in_score_fleet_uses_live_perf_too(self):
        """End-to-end: a cloud model with enough real traffic must rank using
        its real speed/ttft, not the curated guess, when scored via the full
        score_fleet() pipeline (not just the unit-level _speed_score call)."""
        from mrouter import metrics_store
        now = time.time()
        cloud = next(m for m in self.fleet if not m["is_local"])
        for _ in range(6):
            metrics_store.record(self.config, {
                "ts": now, "use_case": "chat", "model_key": cloud["key"],
                "provider": "aperture", "status_code": 200, "ttft_s": 0.15,
                "duration_s": 1.0, "tokens_in": 10, "tokens_out": 200,
                "cost_usd": 0.0001, "cost_is_estimate": False,
            })
        ranked = heuristics.score_fleet(self.fleet, self.config, bench={})
        row = next(r for r in ranked["chat"]["models"] if r["key"] == cloud["key"])
        self.assertEqual(row["speed_source"], "live")
        self.assertEqual(row["ttft_source"], "live")


if __name__ == "__main__":
    unittest.main()
