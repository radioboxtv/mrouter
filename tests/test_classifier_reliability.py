"""Classifier (decision-provider) reliability + circuit breaker.

Mirrors tests/test_reliability.py's TDD style, but for the classify_task
layer: every Baseten/Jev attempt is logged (ok/fail, latency) to the shared
route log under type='classifier_attempt', and a provider with a proven-bad
recent track record is skipped (circuit breaker open) before paying its
network timeout on the next request.
"""

import json
import os
import time
import unittest
from unittest import mock

import tests.fixtures as fx

NOW = time.time()


def attempt(provider, ok, age_h=0.0):
    return {"type": "classifier_attempt", "ts": NOW - age_h * 3600,
            "provider": provider, "ok": ok}


class TestComputeClassifierReliability(fx.ConfigIsolatedTestCase):
    def compute(self, entries):
        from mrouter import reliability
        return reliability.compute_classifier_reliability(self.make_config(), entries)

    def test_no_entries_gives_empty_stats(self):
        self.assertEqual(self.compute([]), {})

    def test_request_type_entries_are_ignored(self):
        """classifier_attempt and request entries share one log; model-reliability
        request entries must never leak into classifier-reliability stats."""
        entries = [{"type": "request", "ts": NOW, "routed_model_key": "baseten", "status_code": 200}]
        self.assertEqual(self.compute(entries), {})

    def test_all_ok_scores_high(self):
        stats = self.compute([attempt("baseten", True, age_h=1)] * 10)
        self.assertGreater(stats["baseten"]["score"], 95)
        self.assertEqual(stats["baseten"]["success_rate"], 1.0)
        self.assertEqual(stats["baseten"]["samples"], 10)

    def test_all_failures_scores_low(self):
        stats = self.compute([attempt("baseten", False, age_h=1)] * 10)
        self.assertLess(stats["baseten"]["score"], 25)
        self.assertEqual(stats["baseten"]["success_rate"], 0.0)

    def test_providers_tracked_independently(self):
        entries = [attempt("baseten", False, age_h=1)] * 10 + [attempt("jev", True, age_h=1)] * 10
        stats = self.compute(entries)
        self.assertLess(stats["baseten"]["score"], 25)
        self.assertGreater(stats["jev"]["score"], 95)

    def test_missing_provider_key_ignored(self):
        entries = [{"type": "classifier_attempt", "ts": NOW, "ok": True}]  # no provider
        self.assertEqual(self.compute(entries), {})


class TestClassifierCircuitBreaker(fx.ConfigIsolatedTestCase):
    """_classifier_breaker_tripped and its wiring into _attempt_baseten / the
    Jev branch of classify_task."""

    def setUp(self):
        super().setUp()
        from mrouter import categorize
        self.categorize = categorize

    def _cfg_with_attempts(self, entries, floor=0.5, min_samples=5):
        cfg = self.make_config()
        cfg["decision"] = {"provider": "auto", "reliability_floor": floor,
                            "reliability_min_samples": min_samples}
        cfg["baseten"] = {"enabled": True, "api_key_env": "BASETEN_API_KEY"}
        os.environ["BASETEN_API_KEY"] = "k"
        self.addCleanup(lambda: os.environ.pop("BASETEN_API_KEY", None))
        # Write attempts straight to the log this config resolves to.
        from mrouter import learning
        for e in entries:
            with open(learning.log_path(cfg), "a") as f:
                f.write(json.dumps(e) + "\n")
        return cfg

    def test_breaker_not_tripped_with_no_history(self):
        cfg = self._cfg_with_attempts([])
        self.assertFalse(self.categorize._classifier_breaker_tripped(cfg, "baseten"))

    def test_breaker_not_tripped_with_thin_samples(self):
        entries = [attempt("baseten", False, age_h=0.1)] * 2  # below min_samples=5
        cfg = self._cfg_with_attempts(entries)
        self.assertFalse(self.categorize._classifier_breaker_tripped(cfg, "baseten"))

    def test_breaker_trips_after_proven_failures(self):
        entries = [attempt("baseten", False, age_h=0.1)] * 10
        cfg = self._cfg_with_attempts(entries)
        self.assertTrue(self.categorize._classifier_breaker_tripped(cfg, "baseten"))

    def test_breaker_zero_floor_disables_breaker(self):
        entries = [attempt("baseten", False, age_h=0.1)] * 10
        cfg = self._cfg_with_attempts(entries, floor=0.0)
        self.assertFalse(self.categorize._classifier_breaker_tripped(cfg, "baseten"))

    def test_tripped_breaker_skips_baseten_without_network_call(self):
        """The core behavior: a tripped breaker must prevent _baseten_classify
        from being called at all -- not just fail fast after calling it."""
        entries = [attempt("baseten", False, age_h=0.1)] * 10
        cfg = self._cfg_with_attempts(entries)

        def boom(text, config, key):
            raise AssertionError("must not call baseten while circuit breaker is open")

        orig = self.categorize._baseten_classify
        self.categorize._baseten_classify = boom
        try:
            result = self.categorize._attempt_baseten("hello", cfg)
        finally:
            self.categorize._baseten_classify = orig
        self.assertIsNone(result)

    def test_healthy_provider_breaker_not_tripped(self):
        entries = [attempt("baseten", True, age_h=0.1)] * 10
        cfg = self._cfg_with_attempts(entries)
        self.assertFalse(self.categorize._classifier_breaker_tripped(cfg, "baseten"))


class TestClassifierAttemptLogging(fx.ConfigIsolatedTestCase):
    """classify_task must log every real attempt (success, exception, and
    unparseable-response outcomes) so reliability has data to work with."""

    def setUp(self):
        super().setUp()
        from mrouter import categorize, learning
        self.categorize = categorize
        self.learning = learning

    def _entries(self, cfg):
        with open(self.learning.log_path(cfg)) as f:
            return [json.loads(l) for l in f]

    def test_successful_baseten_attempt_logged_ok(self):
        cfg = self.make_config()
        cfg["decision"] = {"provider": "baseten"}
        cfg["baseten"] = {"enabled": True, "api_key_env": "BASETEN_API_KEY"}
        os.environ["BASETEN_API_KEY"] = "k"
        self.addCleanup(lambda: os.environ.pop("BASETEN_API_KEY", None))

        self.categorize._baseten_classify = lambda text, config, key: {
            "probs": {}, "top": "chat", "confidence": 0.9, "source": "baseten", "model": "m"}
        r = self.categorize.classify_task("hi", cfg)
        self.assertEqual(r["source"], "baseten")
        entries = self._entries(cfg)
        attempts = [e for e in entries if e.get("type") == "classifier_attempt"]
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]["provider"], "baseten")
        self.assertTrue(attempts[0]["ok"])

    def test_exception_attempt_logged_as_failure(self):
        cfg = self.make_config()
        cfg["decision"] = {"provider": "baseten"}
        cfg["baseten"] = {"enabled": True, "api_key_env": "BASETEN_API_KEY"}
        os.environ["BASETEN_API_KEY"] = "k"
        self.addCleanup(lambda: os.environ.pop("BASETEN_API_KEY", None))

        def boom(text, config, key):
            raise ConnectionError("down")

        self.categorize._baseten_classify = boom
        self.categorize.classify_task("hi", cfg)
        attempts = [e for e in self._entries(cfg) if e.get("type") == "classifier_attempt"]
        self.assertEqual(len(attempts), 1)
        self.assertFalse(attempts[0]["ok"])
        self.assertIn("ConnectionError", attempts[0]["error"])

    def test_unparseable_response_logged_as_failure_not_silent_success(self):
        """A provider that returns garbage (None) forever must NOT look
        healthy in reliability stats just because it didn't raise."""
        cfg = self.make_config()
        cfg["decision"] = {"provider": "baseten"}
        cfg["baseten"] = {"enabled": True, "api_key_env": "BASETEN_API_KEY"}
        os.environ["BASETEN_API_KEY"] = "k"
        self.addCleanup(lambda: os.environ.pop("BASETEN_API_KEY", None))

        self.categorize._baseten_classify = lambda text, config, key: None
        r = self.categorize.classify_task("hi", cfg)
        self.assertEqual(r["source"], "local-heuristic")  # falls back
        attempts = [e for e in self._entries(cfg) if e.get("type") == "classifier_attempt"]
        self.assertEqual(len(attempts), 1)
        self.assertFalse(attempts[0]["ok"], "an unparseable response must count as a failed attempt")


if __name__ == "__main__":
    unittest.main()
