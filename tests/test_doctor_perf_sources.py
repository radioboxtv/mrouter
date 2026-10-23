"""Tests for mrouter doctor's perf-source visibility check: how many
routable models are scored on real measurements (live traffic / bench) vs.
guessed/curated speed_est or the neutral TTFT prior."""

import unittest
from unittest import mock

import tests.fixtures as fx
from mrouter import doctor


class TestPerfSourcesCheck(fx.ConfigIsolatedTestCase):
    def test_all_estimated_is_a_warning(self):
        cfg = self.make_config()
        fleet = fx.standard_fleet()
        with mock.patch("mrouter.discovery.server_alive", return_value=True), \
             mock.patch("mrouter.discovery.discover_models", return_value=[]), \
             mock.patch("mrouter.doctor._check_cloud", return_value=[]), \
             mock.patch("mrouter.discovery.discover_fleet", return_value=fleet), \
             mock.patch("mrouter.heuristics.score_fleet") as mock_score:
            mock_score.return_value = {
                "chat": {"models": [
                    {"key": "m1", "speed_source": "estimated", "ttft_source": "estimated"},
                    {"key": "m2", "speed_source": "estimated", "ttft_source": "estimated"},
                ]},
            }
            report = doctor.run(cfg)
        rows = [r for r in report.rows if r[1] == "perf_sources"]
        self.assertEqual(rows[0][0], "WARN")
        self.assertIn("guessed", rows[0][2].lower())

    def test_mostly_live_is_ok(self):
        cfg = self.make_config()
        fleet = fx.standard_fleet()
        with mock.patch("mrouter.discovery.server_alive", return_value=True), \
             mock.patch("mrouter.discovery.discover_models", return_value=[]), \
             mock.patch("mrouter.doctor._check_cloud", return_value=[]), \
             mock.patch("mrouter.discovery.discover_fleet", return_value=fleet), \
             mock.patch("mrouter.heuristics.score_fleet") as mock_score:
            mock_score.return_value = {
                "chat": {"models": [
                    {"key": "m1", "speed_source": "live", "ttft_source": "live"},
                    {"key": "m2", "speed_source": "live", "ttft_source": "bench"},
                ]},
            }
            report = doctor.run(cfg)
        rows = [r for r in report.rows if r[1] == "perf_sources"]
        self.assertEqual(rows[0][0], "OK")

    def test_no_eligible_models_is_fine(self):
        cfg = self.make_config()
        with mock.patch("mrouter.discovery.server_alive", return_value=True), \
             mock.patch("mrouter.discovery.discover_models", return_value=[]), \
             mock.patch("mrouter.doctor._check_cloud", return_value=[]), \
             mock.patch("mrouter.discovery.discover_fleet", return_value=[]), \
             mock.patch("mrouter.heuristics.score_fleet", return_value={}):
            report = doctor.run(cfg)
        rows = [r for r in report.rows if r[1] == "perf_sources"]
        self.assertEqual(rows[0][0], "OK")


if __name__ == "__main__":
    unittest.main()
