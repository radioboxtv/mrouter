"""Tests for mrouter/trayapp.py's pure logic: formatting, rendering of rumps
menu items against a fake rumps module. No real rumps / macOS app loop needed
(and none available in CI) -- only the pure functions and the fake-module
wiring are exercised.
"""

import unittest

from mrouter import trayapp


class TestFormatUseCases(unittest.TestCase):
    def test_empty_list(self):
        self.assertEqual(trayapp.format_use_cases([]), "\u2014")

    def test_single_use_case(self):
        ucs = [{"use_case": "sensitive", "count": 10}]
        self.assertEqual(trayapp.format_use_cases(ucs), "sensitive")

    def test_two_use_cases_shown_in_full(self):
        ucs = [{"use_case": "coding", "count": 5}, {"use_case": "chat", "count": 2}]
        self.assertEqual(trayapp.format_use_cases(ucs), "coding, chat")

    def test_more_than_max_truncates_with_count(self):
        ucs = [
            {"use_case": "coding", "count": 10}, {"use_case": "chat", "count": 5},
            {"use_case": "thinking", "count": 3}, {"use_case": "sensitive", "count": 1},
        ]
        self.assertEqual(trayapp.format_use_cases(ucs, max_shown=2), "coding, chat +2 more")

    def test_custom_max_shown(self):
        ucs = [{"use_case": "a", "count": 3}, {"use_case": "b", "count": 2}, {"use_case": "c", "count": 1}]
        self.assertEqual(trayapp.format_use_cases(ucs, max_shown=1), "a +2 more")

    def test_exactly_at_max_no_truncation(self):
        ucs = [{"use_case": "a", "count": 2}, {"use_case": "b", "count": 1}]
        self.assertEqual(trayapp.format_use_cases(ucs, max_shown=2), "a, b")


class TestFormatters(unittest.TestCase):
    def test_fmt_usd_none(self):
        self.assertEqual(trayapp.fmt_usd(None), "\u2014")

    def test_fmt_usd_small_shows_more_precision(self):
        self.assertEqual(trayapp.fmt_usd(0.0012), "$0.0012")

    def test_fmt_usd_normal(self):
        self.assertEqual(trayapp.fmt_usd(1.8), "$1.80")

    def test_fmt_ms_none(self):
        self.assertEqual(trayapp.fmt_ms(None), "\u2014")

    def test_fmt_ms_sub_second(self):
        self.assertEqual(trayapp.fmt_ms(0.81), "810ms")

    def test_fmt_ms_over_a_second(self):
        self.assertEqual(trayapp.fmt_ms(1.23), "1.23s")


class TestIconTitle(unittest.TestCase):
    def test_combines_spend_and_ttft(self):
        summary = {"spend_total_usd": 1.84, "ttft_p90_s": 0.81}
        self.assertIn("$1.84", trayapp.icon_title(summary))
        self.assertIn("810ms", trayapp.icon_title(summary))


class TestModelRows(unittest.TestCase):
    def _summary(self):
        return {
            "models": {
                "deepseek-v4-flash": {
                    "requests": 188, "errors": 2, "status": "live",
                    "use_cases": [{"use_case": "coding", "count": 150}, {"use_case": "chat", "count": 38}],
                },
                "glm-5.3": {
                    "requests": 20, "errors": 4, "status": "recent",
                    "use_cases": [{"use_case": "chat", "count": 20}],
                },
            }
        }

    def test_sorted_by_request_count_desc(self):
        rows = trayapp.model_rows(self._summary())
        self.assertEqual(rows[0]["key"], "deepseek-v4-flash")
        self.assertEqual(rows[1]["key"], "glm-5.3")

    def test_reliability_percent_computed_from_errors(self):
        rows = trayapp.model_rows(self._summary())
        glm = next(r for r in rows if r["key"] == "glm-5.3")
        self.assertEqual(glm["reliability_pct"], 80)  # 16/20

    def test_use_cases_label_present(self):
        rows = trayapp.model_rows(self._summary())
        flash = next(r for r in rows if r["key"] == "deepseek-v4-flash")
        self.assertEqual(flash["use_cases_label"], "coding, chat")

    def test_zero_requests_no_reliability_crash(self):
        summary = {"models": {"idle-model": {"requests": 0, "errors": 0, "status": "recent", "use_cases": []}}}
        rows = trayapp.model_rows(summary)
        self.assertIsNone(rows[0]["reliability_pct"])

    def test_max_rows_caps_output(self):
        models = {f"m{i}": {"requests": i, "errors": 0, "status": "recent", "use_cases": []} for i in range(20)}
        rows = trayapp.model_rows({"models": models}, max_rows=5)
        self.assertEqual(len(rows), 5)


class _FakeMenuItem:
    def __init__(self, title, callback=None):
        self.title = title
        self.callback = callback


class _FakeMenu:
    def __init__(self):
        self.items = []

    def clear(self):
        self.items = []

    def update(self, items):
        self.items = items


class _FakeApp:
    def __init__(self, name, title=""):
        self.name = name
        self.title = title
        self.menu = _FakeMenu()

    def run(self):
        pass


class _FakeTimer:
    def __init__(self, callback, interval):
        self.callback = callback
        self.interval = interval

    def start(self):
        pass


class _FakeRumps:
    App = _FakeApp
    MenuItem = _FakeMenuItem
    Timer = _FakeTimer

    @staticmethod
    def quit_application():
        pass


class TestBuildApp(unittest.TestCase):
    def setUp(self):
        self.config = {
            "proxy": {"host": "127.0.0.1", "port": 8730},
            "metrics": {"default_window": "1h", "tray_poll_seconds": 10},
        }

    def test_build_app_handles_unreachable_proxy_gracefully(self):
        # No real proxy running on this port in the test environment; build_app
        # must not raise even though the first refresh() call will fail to connect.
        app = trayapp.build_app(self.config, _FakeRumps)
        self.assertIn("unreachable", app.title)

    def test_rebuild_menu_includes_window_buttons_and_quit(self):
        app = _FakeApp("mrouter")
        state = {"window": "1h"}
        summary = {
            "spend_total_usd": 1.0, "requests": 5, "error_rate": 0.0,
            "ttft_p90_s": 0.5, "tokens_per_min_p90": 1000,
            "models": {},
        }
        trayapp._rebuild_menu(app, _FakeRumps, summary, state, lambda: None)
        titles = [i.title for i in app.menu.items if i is not None]
        self.assertTrue(any("1h" in t for t in titles))
        self.assertTrue(any("24h" in t for t in titles))
        self.assertTrue(any("Quit" in t for t in titles))
        self.assertTrue(any("dashboard" in t.lower() for t in titles))

    def test_rebuild_menu_shows_model_use_cases_truncated(self):
        app = _FakeApp("mrouter")
        state = {"window": "1h"}
        summary = {
            "spend_total_usd": 1.0, "requests": 5, "error_rate": 0.0,
            "ttft_p90_s": 0.5, "tokens_per_min_p90": 1000,
            "models": {
                "m1": {
                    "requests": 10, "errors": 0, "status": "live",
                    "use_cases": [
                        {"use_case": "coding", "count": 4}, {"use_case": "chat", "count": 3},
                        {"use_case": "thinking", "count": 2}, {"use_case": "sensitive", "count": 1},
                    ],
                }
            },
        }
        trayapp._rebuild_menu(app, _FakeRumps, summary, state, lambda: None)
        titles = [i.title for i in app.menu.items if i is not None]
        self.assertTrue(any("+2 more" in t for t in titles))


if __name__ == "__main__":
    unittest.main()
