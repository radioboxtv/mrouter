"""Autolearn: learned.json must update itself with zero manual CLI steps."""

import json
import os
import threading
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

    def test_concurrent_feedback_does_not_corrupt_learned_json(self):
        """serve() is a ThreadingHTTPServer: concurrent request threads can each
        trigger append_feedback -> forced autolearn at the same time. This is a
        smoke test (no exceptions, output stays valid JSON) rather than a true
        race reproduction — CPython's GIL plus the file's small size made
        corruption rare to trigger reliably even pre-lock. The lock
        (_autolearn_lock) is still correct defensive coding for the new
        concurrent-writer surface autolearn introduces versus the old,
        single-invocation `mrouter learn` CLI command."""
        cfg = self.make_config()
        n_threads = 8
        errors = []

        def worker(i):
            try:
                learning.append_feedback(cfg, "chat", f"m{i % 3}", 1 if i % 2 else -1)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        self.assertEqual(errors, [], f"worker threads raised: {errors}")
        # The file must still be valid, parseable JSON after concurrent writers.
        with open(learning.learned_path(cfg)) as f:
            data = json.load(f)
        self.assertIn("adjustments", data)


def learning_min_samples(cfg):
    return int((cfg.get("learning", {}) or {}).get("min_samples_for_auto_adj", 5))


if __name__ == "__main__":
    unittest.main()
