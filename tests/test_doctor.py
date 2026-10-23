"""mrouter doctor: one health-check report replacing scan/suggest/stats/reliability
as the day-to-day maintenance surface."""

import json
import os
import time
import unittest
from unittest import mock

import tests.fixtures as fx
from mrouter import doctor


class TestDoctorReport(fx.ConfigIsolatedTestCase):
    def test_lm_studio_down_is_a_failure(self):
        cfg = self.make_config()
        with mock.patch("mrouter.discovery.server_alive", return_value=False), \
             mock.patch("mrouter.discovery.discover_models", return_value=[]), \
             mock.patch("mrouter.doctor._check_cloud", return_value=[]):
            report = doctor.run(cfg)
        self.assertTrue(report.has_fail)
        sections = {r[1] for r in report.rows if r[0] == "FAIL"}
        self.assertIn("lm_studio", sections)

    def test_lm_studio_up_with_models_is_ok(self):
        cfg = self.make_config()
        fake_model = fx.make_local()
        with mock.patch("mrouter.discovery.server_alive", return_value=True), \
             mock.patch("mrouter.discovery.discover_models", return_value=[fake_model]), \
             mock.patch("mrouter.doctor._check_cloud", return_value=[]):
            report = doctor.run(cfg)
        lm_rows = [r for r in report.rows if r[1] == "lm_studio"]
        self.assertEqual(lm_rows[0][0], "OK")

    def test_stale_bench_is_a_warning_not_a_failure(self):
        cfg = self.make_config()
        fake_model = fx.make_local()
        with mock.patch("mrouter.discovery.server_alive", return_value=True), \
             mock.patch("mrouter.discovery.discover_models", return_value=[fake_model]), \
             mock.patch("mrouter.heuristics.load_bench", return_value={}), \
             mock.patch("mrouter.doctor._check_cloud", return_value=[]):
            report = doctor.run(cfg)
        bench_rows = [r for r in report.rows if r[1] == "bench"]
        self.assertEqual(bench_rows[0][0], "WARN")
        self.assertFalse(report.has_fail)

    def test_reliability_floor_violation_is_a_failure(self):
        cfg = self.make_config()
        fake_stats = {"m1": {"samples": 10, "success_rate": 0.1, "score": 10.0}}
        with mock.patch("mrouter.discovery.server_alive", return_value=True), \
             mock.patch("mrouter.discovery.discover_models", return_value=[]), \
             mock.patch("mrouter.doctor._check_cloud", return_value=[]), \
             mock.patch("mrouter.reliability.compute_reliability", return_value=fake_stats):
            report = doctor.run(cfg)
        rel_rows = [r for r in report.rows if r[1] == "reliability"]
        self.assertEqual(rel_rows[0][0], "FAIL")

    def test_no_traffic_is_fine(self):
        cfg = self.make_config()
        with mock.patch("mrouter.discovery.server_alive", return_value=True), \
             mock.patch("mrouter.discovery.discover_models", return_value=[]), \
             mock.patch("mrouter.doctor._check_cloud", return_value=[]):
            report = doctor.run(cfg)
        self.assertFalse(report.has_fail)

    def test_classifier_no_key_is_warning(self):
        cfg = self.make_config()
        with mock.patch("mrouter.discovery.server_alive", return_value=True), \
             mock.patch("mrouter.discovery.discover_models", return_value=[]), \
             mock.patch("mrouter.doctor._check_cloud", return_value=[]):
            report = doctor.run(cfg)
        clf_rows = [r for r in report.rows if r[1] == "classifier"]
        # In this hermetic test env no Baseten/Jev key is set -> local fallback -> WARN
        self.assertEqual(clf_rows[0][0], "WARN")


class TestDoctorConfigSanity(fx.ConfigIsolatedTestCase):
    def test_bad_weights_flagged(self):
        cfg = self.make_config()
        cfg["use_cases"]["chat"]["weights"] = {"speed": 0.1}  # sums way under 1
        from mrouter.doctor import Report, _check_config_sanity
        report = Report()
        _check_config_sanity(cfg, report)
        cfg_rows = [r for r in report.rows if r[1] == "config"]
        self.assertTrue(any("chat" in r[2] for r in cfg_rows))

    def test_duplicate_family_kb_pattern_flagged(self):
        cfg = self.make_config()
        cfg["family_kb"] = {
            "fam-a": {"match": ["shared-pattern"]},
            "fam-b": {"match": ["shared-pattern"]},
        }
        from mrouter.doctor import Report, _check_config_sanity
        report = Report()
        _check_config_sanity(cfg, report)
        cfg_rows = [r for r in report.rows if r[1] == "config"]
        self.assertTrue(any("shared-pattern" in r[2] for r in cfg_rows))


if __name__ == "__main__":
    unittest.main()
