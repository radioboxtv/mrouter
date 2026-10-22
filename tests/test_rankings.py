"""Live rankings (Artificial Analysis) tests — TDD, written before implementation."""

import json
import os
import unittest

import tests.fixtures as fx

RAW_AA = [
    {"slug": "gpt-6-astra", "model_name": "GPT-6 Astra",
     "artificial_analysis_intelligence_index": 72,
     "artificial_analysis_coding_index": 71,
     "artificial_analysis_agentic_index": 70,
     "median_output_tokens_per_second": 120,
     "median_time_to_first_token_seconds": 0.4},
    {"slug": "claude-opus-5-5", "model_name": "Claude Opus 5.5",
     "intelligence_index": 68, "coding_index": 69, "agentic_index": 66,
     "median_output_tokens_per_second": 95},
    {"slug": "glm-5.3-fast", "model_name": "GLM 5.3 Fast",
     "intelligence_index": 41, "coding_index": 39, "agentic_index": 37,
     "median_output_tokens_per_second": 296},
]


class TestFindRanking(unittest.TestCase):
    """Slice 1: match a fleet model id to a raw AA record, tolerantly."""

    def test_matches_publisher_prefixed_id(self):
        from mrouter import rankings
        model = fx.make_cloud("aperture:openai/gpt-6-astra", "GPT-6 Astra")
        rec = rankings.find_ranking(RAW_AA, model)
        self.assertIsNotNone(rec)
        self.assertEqual(rec["slug"], "gpt-6-astra")

    def test_matches_dotted_version_variant(self):
        """'anthropic/claude-opus-5.5' must match AA slug 'claude-opus-5-5'."""
        from mrouter import rankings
        model = fx.make_cloud("aperture:anthropic/claude-opus-5.5", "Claude Opus 5.5")
        rec = rankings.find_ranking(RAW_AA, model)
        self.assertIsNotNone(rec)
        self.assertEqual(rec["slug"], "claude-opus-5-5")

    def test_matches_case_and_separator_insensitive(self):
        """'zai-org/GLM-5.3-Fast' matches slug 'glm-5.3-fast'."""
        from mrouter import rankings
        model = fx.make_cloud("aperture:zai-org/GLM-5.3-Fast", "GLM 5.3 Fast")
        rec = rankings.find_ranking(RAW_AA, model)
        self.assertEqual(rec["slug"], "glm-5.3-fast")

    def test_no_match_returns_none(self):
        from mrouter import rankings
        model = fx.make_cloud("aperture:some/obscure-model", "Obscure")
        self.assertIsNone(rankings.find_ranking(RAW_AA, model))

    def test_local_model_ids_match_too(self):
        from mrouter import rankings
        model = fx.make_local("google/gemma-4-e4b")
        rec = rankings.find_ranking(
            [{"slug": "gemma-4-e4b", "intelligence_index": 30}], model)
        self.assertIsNotNone(rec)


class TestScoresFromRecord(unittest.TestCase):
    """Slice 2: raw AA record -> mrouter capability scores, tolerantly."""

    def test_parses_long_key_names(self):
        from mrouter import rankings
        s = rankings.scores_from_record(RAW_AA[0])
        self.assertAlmostEqual(s["capability_chat"], 7.2, places=2)
        self.assertAlmostEqual(s["capability_coding"], 7.1, places=2)
        self.assertAlmostEqual(s["capability_thinking"], 7.2, places=2)

    def test_parses_short_key_names(self):
        from mrouter import rankings
        s = rankings.scores_from_record(RAW_AA[1])
        self.assertAlmostEqual(s["capability_chat"], 6.8, places=2)
        self.assertAlmostEqual(s["capability_coding"], 6.9, places=2)

    def test_agentic_backstops_missing_intelligence(self):
        from mrouter import rankings
        rec = {"slug": "x", "agentic_index": 62}
        s = rankings.scores_from_record(rec)
        self.assertAlmostEqual(s["capability_thinking"], 6.2, places=2)

    def test_clamps_out_of_scale_indices(self):
        from mrouter import rankings
        s = rankings.scores_from_record({"slug": "x", "intelligence_index": 150,
                                         "coding_index": -5})
        self.assertEqual(s["capability_chat"], 10.0)
        self.assertEqual(s["capability_coding"], 0.0)

    def test_speed_from_median_tokens_per_second(self):
        from mrouter import rankings
        s = rankings.scores_from_record(RAW_AA[0])
        self.assertAlmostEqual(s["speed_est"], 40.0, places=1)
        fast = rankings.scores_from_record(RAW_AA[2])  # 296 tps -> clamped 98.7
        self.assertAlmostEqual(fast["speed_est"], 98.7, places=1)

    def test_missing_fields_return_none(self):
        from mrouter import rankings
        s = rankings.scores_from_record({"slug": "x"})
        self.assertIsNone(s["capability_chat"])
        self.assertIsNone(s["capability_coding"])
        self.assertIsNone(s["capability_thinking"])
        self.assertIsNone(s["speed_est"])


LIVE_RAW = [
    {"slug": "deepseek-v4-flash", "model_name": "DeepSeek V4 Flash",
     "intelligence_index": 84, "coding_index": 86, "agentic_index": 82,
     "median_output_tokens_per_second": 276},
]


class TestLiveModeScoring(fx.ConfigIsolatedTestCase):
    """Slice 3: mode=live replaces curated capabilities with AA scores (cloud)."""

    def test_live_mode_replaces_cloud_capability(self):
        from mrouter import heuristics
        cfg = self.make_config()
        cfg["rankings"] = {"enabled": True, "mode": "live", "cache_seconds": 86400}
        fleet = fx.standard_fleet()
        ranked = heuristics.score_fleet(fleet, cfg, bench={}, live_records=LIVE_RAW)
        by_key = {m["key"]: m for m in ranked["coding"]["models"]}
        flash = by_key["aperture:deepseek/deepseek-v4-flash"]
        self.assertAlmostEqual(flash["components"]["capability_coding"], 86, places=1)
        self.assertEqual(flash.get("capability_source"), "artificial-analysis")

    def test_curated_mode_ignores_live_records(self):
        from mrouter import heuristics
        cfg = self.make_config()
        cfg["rankings"] = {"enabled": True, "mode": "curated", "cache_seconds": 86400}
        fleet = fx.standard_fleet()
        ranked = heuristics.score_fleet(fleet, cfg, bench={}, live_records=LIVE_RAW)
        by_key = {m["key"]: m for m in ranked["coding"]["models"]}
        flash = by_key["aperture:deepseek/deepseek-v4-flash"]
        self.assertAlmostEqual(flash["components"]["capability_coding"], 83.0, places=1)
        self.assertEqual(flash.get("capability_source"), "curated")

    def test_unknown_model_falls_back_curated_in_live_mode(self):
        from mrouter import heuristics
        cfg = self.make_config()
        cfg["rankings"] = {"enabled": True, "mode": "live", "cache_seconds": 86400}
        fleet = [fx.make_cloud("aperture:some/unmatched-x", "Unmatched X")]
        ranked = heuristics.score_fleet(fleet, cfg, bench={}, live_records=LIVE_RAW)
        row = ranked["chat"]["models"][0]
        self.assertEqual(row.get("capability_source"), "curated")

    def test_live_records_off_by_default_no_change(self):
        from mrouter import heuristics
        cfg = self.make_config()
        cfg.pop("rankings", None)
        ranked = heuristics.score_fleet(fx.standard_fleet(), cfg, bench={})
        by_key = {m["key"]: m for m in ranked["coding"]["models"]}
        self.assertAlmostEqual(
            by_key["aperture:deepseek/deepseek-v4-flash"]["components"]["capability_coding"],
            83.0, places=1)


class TestLoadRankings(fx.ConfigIsolatedTestCase):
    """Slice 4: load_rankings — cache-first, fetch-fallback, key-gated."""

    def setUp(self):
        super().setUp()
        from mrouter import rankings
        self.rankings = rankings
        rankings.CACHE_PATH = os.path.join(self.tmp, "aa_cache.json")
        self.addCleanup(setattr, rankings, "CACHE_PATH", None)

    def _cfg(self, **rcfg):
        base = {"enabled": True, "api_key_env": "AA_API_KEY", "cache_seconds": 86400}
        base.update(rcfg)
        cfg = self.make_config()
        cfg["rankings"] = base
        return cfg

    def test_disabled_returns_empty_and_never_touches_network(self):
        def boom(*a, **k):
            raise AssertionError("fetch must not be called when disabled")
        self.rankings._fetch = boom
        self.assertEqual(self.rankings.load_rankings(self._cfg(enabled=False)), [])

    def test_fresh_cache_served_without_fetch_or_key(self):
        import time as _t
        with open(self.rankings.CACHE_PATH, "w") as f:
            json.dump({"ts": _t.time(), "data": RAW_AA}, f)

        def boom(*a, **k):
            raise AssertionError("fresh cache must not refetch")
        self.rankings._fetch = boom
        os.environ.pop("AA_API_KEY", None)
        out = self.rankings.load_rankings(self._cfg())
        self.assertEqual(len(out), 3)

    def test_fetch_success_populates_cache(self):
        with open(self.rankings.CACHE_PATH, "w") as f:
            f.write("")  # touch; no valid cache yet
        self.rankings._fetch = lambda cfg, key: RAW_AA
        os.environ["AA_API_KEY"] = "test-key"
        try:
            out = self.rankings.load_rankings(self._cfg())
        finally:
            del os.environ["AA_API_KEY"]
        self.assertEqual(out, RAW_AA)
        with open(self.rankings.CACHE_PATH) as f:
            cached = json.load(f)
        self.assertEqual(cached["data"], RAW_AA)
        self.assertGreater(cached["ts"], 0)

    def test_fetch_failure_serves_stale_cache(self):
        import time as _t
        with open(self.rankings.CACHE_PATH, "w") as f:
            json.dump({"ts": _t.time() - 999999, "data": RAW_AA[:1]}, f)

        def boom(*a, **k):
            raise ConnectionError("network down")
        self.rankings._fetch = boom
        os.environ["AA_API_KEY"] = "test-key"
        try:
            out = self.rankings.load_rankings(self._cfg())
        finally:
            del os.environ["AA_API_KEY"]
        self.assertEqual(out, RAW_AA[:1])  # stale, but better than nothing

    def test_fetch_failure_no_cache_returns_empty(self):
        self.rankings._fetch = lambda cfg, key: (_ for _ in ()).throw(ConnectionError())
        os.environ["AA_API_KEY"] = "test-key"
        try:
            out = self.rankings.load_rankings(self._cfg())
        finally:
            del os.environ["AA_API_KEY"]
        self.assertEqual(out, [])

    def test_missing_key_no_cache_returns_empty_without_fetch(self):
        def boom(*a, **k):
            raise AssertionError("no key -> no fetch")
        self.rankings._fetch = boom
        os.environ.pop("AA_API_KEY", None)
        self.assertEqual(self.rankings.load_rankings(self._cfg()), [])


if __name__ == "__main__":
    unittest.main()