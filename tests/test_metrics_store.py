"""metrics_store: SQLite-backed request metrics for the tray/dashboard.

Hermetic: every test gets its own tmp db path via ConfigIsolatedTestCase-style
patching (see _patch_metrics_path), never touches the real metrics.db.
"""

import os
import time
import unittest

import tests.fixtures as fx
from mrouter import metrics_store  # noqa: E402


class MetricsStoreTestCase(fx.ConfigIsolatedTestCase):
    """fx.ConfigIsolatedTestCase already patches metrics_store.db_path to a
    per-test tmp dir (see _patch_metrics_path); nothing extra needed here."""

    def _entry(self, **over):
        base = {
            "ts": time.time(),
            "use_case": "coding",
            "model_key": "aperture:deepseek/deepseek-v4-flash",
            "provider": "aperture",
            "status_code": 200,
            "ttft_s": 0.3,
            "duration_s": 1.2,
            "tokens_in": 100,
            "tokens_out": 50,
            "cost_usd": 0.0001,
            "cost_is_estimate": False,
        }
        base.update(over)
        return base


class TestRecordAndQuery(MetricsStoreTestCase):
    def test_record_then_query_round_trip(self):
        cfg = self.make_config()
        metrics_store.record(cfg, self._entry())
        rows = metrics_store.query_window(cfg, since_ts=0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["model_key"], "aperture:deepseek/deepseek-v4-flash")
        self.assertEqual(rows[0]["use_case"], "coding")

    def test_query_window_excludes_rows_before_since_ts(self):
        cfg = self.make_config()
        now = time.time()
        metrics_store.record(cfg, self._entry(ts=now - 7200))  # 2h ago
        metrics_store.record(cfg, self._entry(ts=now - 60))    # 1min ago
        rows = metrics_store.query_window(cfg, since_ts=now - 3600)  # last hour
        self.assertEqual(len(rows), 1)

    def test_disabled_metrics_skips_write(self):
        cfg = self.make_config()
        cfg["metrics"] = {"enabled": False}
        metrics_store.record(cfg, self._entry())
        rows = metrics_store.query_window(cfg, since_ts=0)
        self.assertEqual(len(rows), 0)


class TestPercentile(unittest.TestCase):
    def test_percentile_known_values(self):
        vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # p50 of 1..10 -> 5 or 6 depending on method; just check monotonic sanity
        p50 = metrics_store.percentile(vals, 50)
        p90 = metrics_store.percentile(vals, 90)
        self.assertTrue(4 <= p50 <= 6)
        self.assertTrue(9 <= p90 <= 10)

    def test_percentile_empty_is_none(self):
        self.assertIsNone(metrics_store.percentile([], 90))

    def test_percentile_single_value(self):
        self.assertEqual(metrics_store.percentile([42], 90), 42)


class TestWindowSummary(MetricsStoreTestCase):
    def test_counts_and_error_rate(self):
        cfg = self.make_config()
        now = time.time()
        for i in range(8):
            metrics_store.record(cfg, self._entry(ts=now - i, status_code=200))
        metrics_store.record(cfg, self._entry(ts=now, status_code=500))
        metrics_store.record(cfg, self._entry(ts=now, status_code=503))
        summary = metrics_store.window_summary(cfg, window_seconds=3600, now=now)
        self.assertEqual(summary["requests"], 10)
        self.assertEqual(summary["errors"], 2)
        self.assertAlmostEqual(summary["error_rate"], 0.2)

    def test_ttft_percentiles(self):
        cfg = self.make_config()
        now = time.time()
        for ttft in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            metrics_store.record(cfg, self._entry(ts=now, ttft_s=ttft))
        summary = metrics_store.window_summary(cfg, window_seconds=3600, now=now)
        self.assertIsNotNone(summary["ttft_p90_s"])
        self.assertGreater(summary["ttft_p90_s"], summary["ttft_p50_s"])

    def test_tokens_per_min_p90(self):
        cfg = self.make_config()
        now = time.time()
        # 50 tokens in 1s -> 50 tok/s -> 3000 tok/min
        metrics_store.record(cfg, self._entry(ts=now, tokens_out=50, duration_s=1.0, ttft_s=0.1))
        # 120 tokens in 2s -> 60 tok/s -> 3600 tok/min
        metrics_store.record(cfg, self._entry(ts=now, tokens_out=120, duration_s=2.0, ttft_s=0.1))
        summary = metrics_store.window_summary(cfg, window_seconds=3600, now=now)
        self.assertIsNotNone(summary["tokens_per_min_p90"])
        self.assertGreaterEqual(summary["tokens_per_min_p90"], 3000)

    def test_spend_split_actual_vs_estimate(self):
        cfg = self.make_config()
        now = time.time()
        metrics_store.record(cfg, self._entry(ts=now, cost_usd=0.01, cost_is_estimate=False))
        metrics_store.record(cfg, self._entry(ts=now, cost_usd=0.02, cost_is_estimate=True))
        summary = metrics_store.window_summary(cfg, window_seconds=3600, now=now)
        self.assertAlmostEqual(summary["spend_actual_usd"], 0.01)
        self.assertAlmostEqual(summary["spend_estimated_usd"], 0.02)
        self.assertAlmostEqual(summary["spend_total_usd"], 0.03)

    def test_window_excludes_old_rows(self):
        cfg = self.make_config()
        now = time.time()
        metrics_store.record(cfg, self._entry(ts=now - 90000))  # >24h ago
        metrics_store.record(cfg, self._entry(ts=now - 10))
        summary = metrics_store.window_summary(cfg, window_seconds=3600, now=now)
        self.assertEqual(summary["requests"], 1)

    def test_per_model_breakdown_and_use_cases(self):
        cfg = self.make_config()
        now = time.time()
        for _ in range(3):
            metrics_store.record(cfg, self._entry(
                ts=now, model_key="m1", use_case="coding", status_code=200))
        for _ in range(2):
            metrics_store.record(cfg, self._entry(
                ts=now, model_key="m1", use_case="chat", status_code=200))
        metrics_store.record(cfg, self._entry(ts=now, model_key="m2", use_case="chat", status_code=500))

        summary = metrics_store.window_summary(cfg, window_seconds=3600, now=now)
        models = summary["models"]
        self.assertIn("m1", models)
        self.assertIn("m2", models)
        self.assertEqual(models["m1"]["requests"], 5)
        ucs = models["m1"]["use_cases"]
        # sorted by count desc: coding(3) before chat(2)
        self.assertEqual(ucs[0]["use_case"], "coding")
        self.assertEqual(ucs[0]["count"], 3)
        self.assertEqual(ucs[1]["use_case"], "chat")
        self.assertEqual(models["m2"]["errors"], 1)

    def test_model_status_live_vs_recent(self):
        cfg = self.make_config()
        now = time.time()
        metrics_store.record(cfg, self._entry(ts=now - 5, model_key="fresh"))
        metrics_store.record(cfg, self._entry(ts=now - 1800, model_key="stale"))
        summary = metrics_store.window_summary(cfg, window_seconds=3600, now=now, live_threshold_s=60)
        self.assertEqual(summary["models"]["fresh"]["status"], "live")
        self.assertEqual(summary["models"]["stale"]["status"], "recent")

    def test_series_buckets_sum_to_total_requests(self):
        cfg = self.make_config()
        now = time.time()
        for i in range(20):
            metrics_store.record(cfg, self._entry(ts=now - i * 100))
        summary = metrics_store.window_summary(cfg, window_seconds=3600, now=now, buckets=12)
        total_in_series = sum(b["requests"] for b in summary["series"])
        self.assertEqual(total_in_series, summary["requests"])
        self.assertEqual(len(summary["series"]), 12)


class TestPrune(MetricsStoreTestCase):
    def test_prune_removes_rows_older_than_retention(self):
        cfg = self.make_config()
        cfg["metrics"] = {"enabled": True, "retention_days": 1}
        now = time.time()
        metrics_store.record(cfg, self._entry(ts=now - 2 * 86400))  # 2 days old
        metrics_store.record(cfg, self._entry(ts=now - 10))
        metrics_store.prune(cfg, now=now)
        rows = metrics_store.query_window(cfg, since_ts=0)
        self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
