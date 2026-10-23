"""Autobench: `serve`'s Router benches stale local models in the background,
with zero manual `mrouter bench` step."""

import time
import unittest

import tests.fixtures as fx
from mrouter import heuristics


class TestStaleLocalModels(unittest.TestCase):
    def test_model_missing_from_bench_is_stale(self):
        m = fx.make_local("qwen/qwen3.8-27b")
        stale = heuristics.stale_local_models([m], bench={})
        self.assertEqual([x["key"] for x in stale], ["qwen/qwen3.8-27b"])

    def test_model_with_fresh_bench_is_not_stale(self):
        m = fx.make_local("qwen/qwen3.8-27b")
        bench = {"qwen/qwen3.8-27b": {"ts": time.time(), "tokens_per_sec": 42.0}}
        stale = heuristics.stale_local_models([m], bench=bench)
        self.assertEqual(stale, [])

    def test_model_with_expired_bench_is_stale(self):
        m = fx.make_local("qwen/qwen3.8-27b")
        bench = {"qwen/qwen3.8-27b": {"ts": time.time() - 999999, "tokens_per_sec": 42.0}}
        stale = heuristics.stale_local_models([m], bench=bench, max_age_hours=1)
        self.assertEqual([x["key"] for x in stale], ["qwen/qwen3.8-27b"])

    def test_model_with_failed_bench_entry_is_stale(self):
        m = fx.make_local("qwen/qwen3.8-27b")
        bench = {"qwen/qwen3.8-27b": {"ts": time.time(), "error": "boom"}}
        stale = heuristics.stale_local_models([m], bench=bench)
        self.assertEqual([x["key"] for x in stale], ["qwen/qwen3.8-27b"])

    def test_cloud_models_never_benched(self):
        cloud = fx.make_cloud()
        stale = heuristics.stale_local_models([cloud], bench={})
        self.assertEqual(stale, [])


class TestRouterAutobenchDedup(unittest.TestCase):
    """The Router should only kick one bench job per stale key per process,
    not re-trigger on every fleet refresh while a bench is already running."""

    def test_autobench_dedupes_in_flight_keys(self):
        from mrouter import proxy, discovery

        cfg = discovery.load_config()
        router = proxy.Router(cfg)
        m = fx.make_local("qwen/qwen3.8-27b")
        router._fleet = [m]

        # Simulate a bench already in flight for this key.
        router._benched_keys.add(m["key"])
        calls = []
        router._run_bench = lambda models: calls.append(models)

        import mrouter.heuristics as heuristics
        orig_load_bench = heuristics.load_bench
        heuristics.load_bench = lambda: {}
        orig_stale_local_models = heuristics.stale_local_models
        try:
            router._maybe_autobench()
            self.assertEqual(calls, [], "should not re-dispatch a bench job already in flight")
        finally:
            heuristics.load_bench = orig_load_bench
            heuristics.stale_local_models = orig_stale_local_models


class TestRouterAutobenchNonBlocking(unittest.TestCase):
    """fleet() runs on every request-handling thread (FLEET_TTL_S refresh) —
    _maybe_autobench must never block that thread on a network call or the
    ~60s `lms server start`, even when LM Studio is down."""

    def test_maybe_autobench_does_not_block_on_lm_studio_liveness(self):
        from mrouter import proxy, discovery
        from unittest import mock

        cfg = discovery.load_config()
        router = proxy.Router(cfg)
        m = fx.make_local("qwen/qwen3.8-27b")
        router._fleet = [m]

        import mrouter.heuristics as heuristics
        orig_load_bench = heuristics.load_bench
        heuristics.load_bench = lambda: {}

        # Slow stand-in for the real network call; if it ran on the calling
        # thread this test would take >1s and fail the elapsed-time assertion.
        def slow_server_alive(cfg):
            time.sleep(1.0)
            return False

        try:
            with mock.patch.object(discovery, "server_alive", side_effect=slow_server_alive), \
                 mock.patch.object(discovery, "start_server", return_value=False):
                started = time.time()
                router._maybe_autobench()
                elapsed = time.time() - started
            self.assertLess(elapsed, 0.5, "_maybe_autobench must return immediately; "
                             "the LM Studio check belongs on the background thread")
        finally:
            heuristics.load_bench = orig_load_bench


if __name__ == "__main__":
    unittest.main()
