"""Unit tests for mrouter scoring, cost, eligibility, and learning. Stdlib only.

Run: cd ~/workspace/model-router && python3 -m unittest discover -s tests -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mrouter import discovery, heuristics, learning, providers as providers_mod  # noqa: E402
from mrouter.categorize import _local_classify  # noqa: E402


def local_model():
    return {
        "key": "qwen/qwen3.8-27b", "api_id": "qwen/qwen3.8-27b", "upstream_id": "qwen/qwen3.8-27b",
        "provider": "lmstudio", "is_local": True, "display_name": "Qwen3.8 27B",
        "params_b": 27.0, "size_gb": 15.0, "max_ctx": 262144, "vision": True,
        "multimodal": True, "tool_use": True, "arch": "qwen3_5",
        "cost_input_per_token": 0.0, "cost_output_per_token": 0.0,
        "cost_cache_read_per_token": 0.0,
    }


def cloud_model():
    return {
        "key": "aperture:deepseek/deepseek-v4-flash", "api_id": "deepseek/deepseek-v4-flash",
        "upstream_id": "deepseek/deepseek-v4-flash", "provider": "aperture", "is_local": False,
        "display_name": "DeepSeek V4 Flash", "params_b": None, "size_gb": 0.0,
        "max_ctx": 1048576, "vision": None, "multimodal": None, "tool_use": True, "arch": "cloud",
        "cost_input_per_token": 0.13e-6, "cost_output_per_token": 0.26e-6,
        "cost_cache_read_per_token": None,
    }


def pricey_model():
    m = cloud_model()
    m.update({"key": "aperture:openai/gpt-6-astra", "api_id": "openai/gpt-6-astra",
              "upstream_id": "openai/gpt-6-astra", "display_name": "GPT-6 Astra",
              "cost_input_per_token": 10e-6, "cost_output_per_token": 50e-6})
    return m


class TestCost(unittest.TestCase):
    def test_local_is_free(self):
        self.assertEqual(providers_mod.estimate_cost_usd(local_model()), 0.0)
        self.assertEqual(providers_mod.actual_cost_usd(local_model(), {"prompt_tokens": 100, "completion_tokens": 10}), 0.0)

    def test_cloud_estimate_matches_pricing(self):
        # 2000 non-cached in @.13/M + 2000 cached @.13/M (no cache price -> input) + 1000 out @.26/M
        est = providers_mod.estimate_cost_usd(cloud_model(), 4000, 1000, 2000)
        self.assertAlmostEqual(est, 2000 * 0.13e-6 + 2000 * 0.13e-6 + 1000 * 0.26e-6, places=9)

    def test_actual_cost_uses_cached_tokens(self):
        usage = {"prompt_tokens": 1000, "completion_tokens": 100,
                 "prompt_tokens_details": {"cached_tokens": 400}}
        # no cache price set -> cached billed at input rate: 1000*.13/M + 100*.26/M
        cost = providers_mod.actual_cost_usd(cloud_model(), usage)
        self.assertAlmostEqual(cost, 1000 * 0.13e-6 + 100 * 0.26e-6, places=9)


class TestEligibility(unittest.TestCase):
    def setUp(self):
        self.config = discovery.load_config()
        self.fleet = [local_model(), cloud_model(), pricey_model()]

    def test_sensitive_is_local_only(self):
        pool = heuristics.eligible_models(self.fleet, self.config, "sensitive")
        self.assertTrue(all(m["is_local"] for m in pool))
        self.assertEqual(len(pool), 1)

    def test_chat_multiplexes(self):
        pool = heuristics.eligible_models(self.fleet, self.config, "chat")
        self.assertEqual(len(pool), 3)


class TestScoring(unittest.TestCase):
    def setUp(self):
        self.config = discovery.load_config()
        self.fleet = [local_model(), cloud_model(), pricey_model()]

    def test_sensitive_never_picks_cloud(self):
        ranked = heuristics.score_fleet(self.fleet, self.config, bench={})
        best = ranked["sensitive"]["models"][0]
        self.assertTrue(best["is_local"])

    def test_cost_efficiency_orders_free_then_cheap_then_pricey(self):
        fleets = {m["key"]: m for m in self.fleet}
        s_local = heuristics._cost_efficiency_score(fleets["qwen/qwen3.8-27b"])
        s_cheap = heuristics._cost_efficiency_score(fleets["aperture:deepseek/deepseek-v4-flash"])
        s_pricey = heuristics._cost_efficiency_score(fleets["aperture:openai/gpt-6-astra"])
        self.assertEqual(s_local, 100.0)
        self.assertGreater(s_cheap, s_pricey)

    def test_coding_prefers_capable_over_pricey_when_cheap_is_capable(self):
        ranked = heuristics.score_fleet(self.fleet, self.config, bench={})
        keys = [m["key"] for m in ranked["coding"]["models"]]
        # DeepSeek V4 Flash (capable + nearly free) should beat a $50/M-output flagship on value-weighted coding
        self.assertLess(keys.index("aperture:deepseek/deepseek-v4-flash"),
                        keys.index("aperture:openai/gpt-6-astra"))

    def test_est_cost_reported(self):
        ranked = heuristics.score_fleet(self.fleet, self.config, bench={})
        flash = next(m for m in ranked["chat"]["models"] if m["key"] == "aperture:deepseek/deepseek-v4-flash")
        self.assertGreater(flash["est_cost_usd"], 0)
        self.assertLess(flash["est_cost_usd"], 0.005)


class TestClassifierFallback(unittest.TestCase):
    def test_code_prompt(self):
        r = _local_classify("Please debug this Python function, it throws a traceback on import")
        self.assertEqual(r["top"], "coding")

    def test_sensitive_prompt(self):
        r = _local_classify("Summarize this confidential patient salary spreadsheet, do not share it")
        self.assertEqual(r["top"], "sensitive")
        self.assertGreater(r["probs"]["sensitive"], 0.4)

    def test_thinking_prompt(self):
        r = _local_classify("Analyze the trade-offs and compare these strategies in depth, with a research report")
        self.assertEqual(r["top"], "thinking")

    def test_plain_chat(self):
        r = _local_classify("What's a good weeknight dinner idea?")
        self.assertEqual(r["top"], "chat")


class TestLearning(unittest.TestCase):
    def setUp(self):
        self.config = discovery.load_config()

    def test_feedback_moves_adjustment(self):
        entries = [
            {"type": "feedback", "use_case": "coding",
             "routed_model_key": "aperture:deepseek/deepseek-v4-flash", "rating": 1},
            {"type": "feedback", "use_case": "coding",
             "routed_model_key": "aperture:deepseek/deepseek-v4-flash", "rating": 1},
        ]
        learned = learning.compute_learned(self.config, entries)
        adj = learning.adjustment_for(learned, "coding", "aperture:deepseek/deepseek-v4-flash")
        self.assertGreater(adj, 0)

    def test_adjustment_bounded(self):
        entries = [{"type": "feedback", "use_case": "chat", "routed_model_key": "x", "rating": -1}] * 50
        learned = learning.compute_learned(self.config, entries)
        self.assertGreaterEqual(learning.adjustment_for(learned, "chat", "x"),
                                -self.config["learning"]["max_adjustment"])


if __name__ == "__main__":
    unittest.main()