"""Autolearn: learned.json must update itself with zero manual CLI steps."""

import json
import os
import time
import unittest

import tests.fixtures as fx
from mrouter import learning


class TestAutolearnOnFeedback(fx.ConfigIsolatedTestCase):
    """Feedback is low-volume/high-signal -> must recompute immediately (unthrottled)."""

    def test_feedback_recomputes_learned_json_immediately(self):
        cfg = self.make_config()
        self.assertFalse(os.path.exists(learning.learned_path(cfg)))
        learning.append_feedback(cfg, "coding", "aperture:deepseek/deepseek-v4-flash", 1)
        self.assertTrue(os.path.exists(learning.learned_path(cfg)),
                         "learned.json should exist immediately after one feedback call, "
                         "with no separate `mrouter learn` step")
        learned = learning.load_learned(cfg)
        self.assertIn("coding|aperture:deepseek/deepseek-v4-flash", learned)

    def test_second_feedback_updates_existing_adjustment(self):
        cfg = self.make_config()
        learning.append_feedback(cfg, "chat", "m1", 1)
        first = learning.load_learned(cfg)["chat|m1"]["adj"]
        learning.append_feedback(cfg, "chat", "m1", 1)
        second = learning.load_learned(cfg)["chat|m1"]["adj"]
        self.assertGreaterEqual(second, first)


class TestAutolearnOnRequestLog(fx.ConfigIsolatedTestCase):
    """Ordinary request logging is high-volume -> throttled, but still automatic
    (no human ever runs `mrouter learn` as part of normal operation)."""

    def setUp(self):
        super().setUp()
        learning._last_autolearn.clear()

    def test_request_log_triggers_autolearn_eventually(self):
        cfg = self.make_config()
        for _ in range(learning_min_samples(cfg)):
            learning.append_log(cfg, {
                "type": "request", "use_case": "chat", "routed_model_key": "m1",
                "status_code": 200,
            })
        self.assertTrue(os.path.exists(learning.learned_path(cfg)),
                         "autolearn should fire on normal request logging, no CLI step required")

    def test_throttle_prevents_recompute_storm(self):
        cfg = self.make_config()
        learning.append_log(cfg, {"type": "request", "use_case": "chat",
                                   "routed_model_key": "m1", "status_code": 200})
        first_mtime = os.path.getmtime(learning.learned_path(cfg))
        # Immediately log more; throttle should skip recompute within the window.
        for _ in range(5):
            learning.append_log(cfg, {"type": "request", "use_case": "chat",
                                       "routed_model_key": "m1", "status_code": 200})
        second_mtime = os.path.getmtime(learning.learned_path(cfg))
        self.assertEqual(first_mtime, second_mtime,
                          "rapid-fire logging within the throttle window should not "
                          "recompute learned.json on every single request")


def learning_min_samples(cfg):
    return int((cfg.get("learning", {}) or {}).get("min_samples_for_auto_adj", 5))


if __name__ == "__main__":
    unittest.main()
