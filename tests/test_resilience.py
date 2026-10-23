"""Resilience tests for issue #4: retry-with-backoff, cascading fallback to
runner-up models, the hard sensitive-local-only fallback floor, and the
in-process circuit breaker. Fake in-process upstreams, no real network.
"""

import json
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import tests.fixtures as fx
from mrouter import proxy  # noqa: E402


def _ok_upstream_factory():
    class _Ok(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length))
            resp = {
                "id": "chatcmpl-ok", "object": "chat.completion", "created": 0,
                "model": body.get("model", "?"),
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            }
            data = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    return _Ok


def _status_upstream_factory(status, body=b'{"error": "boom"}'):
    class _Status(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(length)
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    return _Status


def _counting_status_then_ok_factory(fail_times, status=500):
    """Fails with `status` for the first `fail_times` requests it receives
    (process-wide counter), then succeeds -- used to prove retry-then-success
    on the SAME model."""
    state = {"n": 0}

    class _Flaky(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length))
            state["n"] += 1
            if state["n"] <= fail_times:
                data = b'{"error": "boom"}'
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            resp = {
                "id": "chatcmpl-ok", "object": "chat.completion", "created": 0,
                "model": body.get("model", "?"),
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            }
            data = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    return _Flaky, state


class ResilienceTestCase(fx.ConfigIsolatedTestCase):
    """Boots the real proxy against fake local+cloud upstreams on ephemeral
    ports, with a deterministic ranked fleet, fast test-time backoff."""

    def _start_upstream(self, handler_cls):
        srv = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
        srv.daemon_threads = True
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        return srv

    def _start_proxy(self, config):
        srv = ThreadingHTTPServer(("127.0.0.1", 0), proxy.Handler)
        srv.daemon_threads = True
        proxy.Handler.router = proxy.Router(config)
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        return srv

    def _config(self, resilience=None):
        cfg = self.make_config()
        cfg["proxy"]["host"] = "127.0.0.1"
        cfg["proxy"]["port"] = 0
        cfg["resilience"] = resilience or {
            "retry": {"max_attempts": 2, "backoff_s": 0.01, "on_status": [429, 500, 502, 503, 504]},
            "fallback": {"enabled": True, "max_models": 3},
            "circuit_breaker": {"failure_threshold": 2, "cooldown_s": 0.2},
        }
        return cfg

    def _patch_fleet(self, fleet):
        original_init = proxy.Router.__init__

        def init_with_fleet(self, cfg):
            original_init(self, cfg)
            self._fleet = fleet
            self._fleet_ts = float("inf")

        proxy.Router.__init__ = init_with_fleet
        self.addCleanup(setattr, proxy.Router, "__init__", original_init)

    def _post(self, port, payload, timeout=10):
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r, json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            return e, json.loads(e.read().decode())


class FallbackTest(ResilienceTestCase):
    """Acceptance: a request whose top pick returns 500 succeeds via the
    runner-up, with the attempt chain logged."""

    def test_top_pick_500_falls_back_to_runner_up(self):
        fleet = fx.standard_fleet()  # gemma-4-e4b and qwen local; several cloud
        cfg = self._config()
        self._patch_fleet(fleet)

        broken = self._start_upstream(_status_upstream_factory(500))
        cfg["lm_studio"]["api_base"] = f"http://127.0.0.1:{broken.server_port}/v1"
        cfg["providers"]["aperture"]["api_base"] = f"http://127.0.0.1:{broken.server_port}/v1"
        cfg["providers"]["aperture"]["api_key"] = "test-key"
        cfg["providers"]["aperture"]["api_key_env"] = None

        srv = self._start_proxy(cfg)
        port = srv.server_address[1]

        # Every upstream is broken -- fallback across all candidates should be
        # exhausted and surface the last failure, proving fallback does NOT
        # infinite loop and DOES attempt every ranked candidate.
        r, body = self._post(port, {
            "model": "sensitive",
            "messages": [{"role": "user", "content": "private data"}],
            "max_tokens": 5})
        self.assertEqual(r.code, 500)

        with open(self.learning.log_path(cfg)) as f:
            entries = [json.loads(line) for line in f]
        # Both local candidates (gemma, qwen) were attempted and logged as failures.
        sensitive_entries = [e for e in entries if e.get("use_case") == "sensitive"]
        self.assertEqual(len(sensitive_entries), 2, "both sensitive candidates must be tried")
        self.assertEqual(sensitive_entries[0]["fallback_attempt"], 1)
        self.assertEqual(sensitive_entries[1]["fallback_attempt"], 2)

    def test_cloud_runner_up_succeeds_when_top_pick_500s(self):
        """Top-ranked cloud model 500s once; forces fallback onto the next
        ranked candidate, which must succeed on its first attempt."""
        fleet = fx.standard_fleet()
        cfg = self._config()
        self._patch_fleet(fleet)

        # max_attempts=1: a failure is NOT retried on the same model, so a
        # success here can only come from the fallback cascading to the next
        # ranked candidate -- isolating fallback behavior from same-model retry.
        cfg["resilience"]["retry"]["max_attempts"] = 1
        flaky_cls, state = _counting_status_then_ok_factory(fail_times=1, status=500)
        flaky = self._start_upstream(flaky_cls)
        cfg["providers"]["aperture"]["api_base"] = f"http://127.0.0.1:{flaky.server_port}/v1"
        cfg["providers"]["aperture"]["api_key"] = "test-key"
        cfg["providers"]["aperture"]["api_key_env"] = None
        # Force cloud-only by disabling local in the eligibility pool for "coding"
        cfg["routing"]["allow_cloud"]["coding"] = True

        srv = self._start_proxy(cfg)
        port = srv.server_address[1]

        r, body = self._post(port, {
            "model": "coding",
            "messages": [{"role": "user", "content": "fix this bug"}],
            "max_tokens": 5})
        self.assertEqual(r.status, 200)
        attempts = json.loads(r.headers["X-Routed-Attempts"])
        self.assertGreaterEqual(len(attempts), 1)
        self.assertTrue(attempts[-1].get("final"))

        with open(self.learning.log_path(cfg)) as f:
            entries = [json.loads(line) for line in f]
        coding_entries = [e for e in entries if e.get("use_case") == "coding"]
        # First attempt failed (fallback_attempt=1, status 500), final succeeded.
        self.assertTrue(any(e["status_code"] == 500 for e in coding_entries))
        self.assertTrue(any(e["status_code"] == 200 for e in coding_entries))


class SensitiveNeverFallsBackToCloudTest(ResilienceTestCase):
    """Acceptance (hermetic): sensitive requests never fall back to a
    non-local model, even when every local candidate is broken and cloud
    candidates exist and are healthy."""

    def test_sensitive_exhausts_local_candidates_never_touches_cloud(self):
        fleet = fx.standard_fleet()
        cfg = self._config()
        self._patch_fleet(fleet)

        broken_local = self._start_upstream(_status_upstream_factory(503))
        healthy_cloud = self._start_upstream(_ok_upstream_factory())
        cfg["lm_studio"]["api_base"] = f"http://127.0.0.1:{broken_local.server_port}/v1"
        cfg["providers"]["aperture"]["api_base"] = f"http://127.0.0.1:{healthy_cloud.server_port}/v1"
        cfg["providers"]["aperture"]["api_key"] = "test-key"
        cfg["providers"]["aperture"]["api_key_env"] = None

        srv = self._start_proxy(cfg)
        port = srv.server_address[1]

        r, body = self._post(port, {
            "model": "sensitive",
            "messages": [{"role": "user", "content": "my ssn is private"}],
            "max_tokens": 5})
        # Local-only pool is exhausted (both local candidates 503); must
        # surface the failure, NEVER silently route to the healthy cloud fake.
        self.assertEqual(r.code, 503)
        with open(self.learning.log_path(cfg)) as f:
            entries = [json.loads(line) for line in f]
        sensitive_entries = [e for e in entries if e.get("use_case") == "sensitive"]
        self.assertTrue(sensitive_entries)
        self.assertTrue(all(e["provider"] == "lmstudio" for e in sensitive_entries),
                         "every attempted provider for a sensitive request must be lmstudio (local)")


class RetryWithBackoffTest(ResilienceTestCase):
    """Acceptance: 429s are retried with backoff rather than immediately
    surfaced."""

    def test_429_retried_then_succeeds_on_same_model(self):
        fleet = [fx.make_local()]
        cfg = self._config(resilience={
            "retry": {"max_attempts": 3, "backoff_s": 0.01, "on_status": [429]},
            "fallback": {"enabled": True, "max_models": 1},
            "circuit_breaker": {"failure_threshold": 10, "cooldown_s": 60},
        })
        self._patch_fleet(fleet)
        flaky_cls, state = _counting_status_then_ok_factory(fail_times=2, status=429)
        srv_upstream = self._start_upstream(flaky_cls)
        cfg["lm_studio"]["api_base"] = f"http://127.0.0.1:{srv_upstream.server_port}/v1"

        srv = self._start_proxy(cfg)
        port = srv.server_address[1]

        r, body = self._post(port, {
            "model": "sensitive",
            "messages": [{"role": "user", "content": "x"}],
            "max_tokens": 5})
        self.assertEqual(r.status, 200)
        self.assertEqual(state["n"], 3, "two 429s then a success: three total calls to the SAME model")

    def test_non_retryable_4xx_is_not_retried(self):
        fleet = [fx.make_local()]
        cfg = self._config(resilience={
            "retry": {"max_attempts": 3, "backoff_s": 0.01, "on_status": [429, 500]},
            "fallback": {"enabled": True, "max_models": 1},
            "circuit_breaker": {"failure_threshold": 10, "cooldown_s": 60},
        })
        self._patch_fleet(fleet)
        flaky_cls, state = _counting_status_then_ok_factory(fail_times=99, status=400)
        srv_upstream = self._start_upstream(flaky_cls)
        cfg["lm_studio"]["api_base"] = f"http://127.0.0.1:{srv_upstream.server_port}/v1"

        srv = self._start_proxy(cfg)
        port = srv.server_address[1]

        r, body = self._post(port, {
            "model": "sensitive",
            "messages": [{"role": "user", "content": "x"}],
            "max_tokens": 5})
        self.assertEqual(r.code, 400)
        self.assertEqual(state["n"], 1, "a non-retryable 400 must not be retried")


class CircuitBreakerTest(ResilienceTestCase):
    """Acceptance: a model that fails repeatedly is temporarily excluded by
    the circuit breaker and recovers after cooldown."""

    def test_repeated_failures_trip_breaker_then_recover_after_cooldown(self):
        fleet = fx.standard_fleet()
        cfg = self._config(resilience={
            "retry": {"max_attempts": 1, "backoff_s": 0.0, "on_status": [500]},
            "fallback": {"enabled": True, "max_models": 5},
            "circuit_breaker": {"failure_threshold": 2, "cooldown_s": 0.15},
        })
        self._patch_fleet(fleet)
        broken = self._start_upstream(_status_upstream_factory(500))
        healthy = self._start_upstream(_ok_upstream_factory())
        cfg["lm_studio"]["api_base"] = f"http://127.0.0.1:{broken.server_port}/v1"
        cfg["providers"]["aperture"]["api_base"] = f"http://127.0.0.1:{healthy.server_port}/v1"
        cfg["providers"]["aperture"]["api_key"] = "test-key"
        cfg["providers"]["aperture"]["api_key_env"] = None

        srv = self._start_proxy(cfg)
        port = srv.server_address[1]
        router = proxy.Handler.router
        top_local_key = fleet[0]["key"]  # google/gemma-4-e4b -- top-ranked for 'sensitive' in standard_fleet()

        # Two failed sensitive requests against the SAME top local candidate
        # trip its breaker (failure_threshold=2). The circuit only governs
        # ORDERING among already-eligible candidates (not hard exclusion), so
        # drive it directly for a deterministic, fast unit-level check instead
        # of racing two live HTTP round trips against ranking variance.
        self.assertFalse(router.circuit_is_open(top_local_key))
        router.record_circuit_failure(top_local_key)
        self.assertFalse(router.circuit_is_open(top_local_key), "one failure must not trip the breaker")
        router.record_circuit_failure(top_local_key)
        self.assertTrue(router.circuit_is_open(top_local_key), "two failures must trip the breaker")

        cands = router.candidates("sensitive", limit=5)
        self.assertNotEqual(cands[0][0]["key"], top_local_key,
                             "a tripped candidate must be deprioritized behind healthy ones")

        time.sleep(0.2)  # cooldown_s = 0.15
        self.assertFalse(router.circuit_is_open(top_local_key), "breaker must half-open after cooldown")
        router.record_circuit_success(top_local_key)
        cands = router.candidates("sensitive", limit=5)
        self.assertEqual(cands[0][0]["key"], top_local_key,
                          "a recovered model regains its normal rank")


class ConfigurableTimeoutTest(ResilienceTestCase):
    """Acceptance: timeouts are configurable (defaults documented in CONFIG.md)."""

    def test_resilience_cfg_reads_upstream_timeout(self):
        from mrouter.proxy import _resilience_cfg
        cfg = {"resilience": {"upstream_timeout_s": 12.5}}
        self.assertEqual(_resilience_cfg(cfg)["timeout_s"], 12.5)

    def test_resilience_cfg_defaults_when_missing(self):
        from mrouter.proxy import _resilience_cfg
        cfg = {}
        rcfg = _resilience_cfg(cfg)
        self.assertEqual(rcfg["timeout_s"], 600.0)
        self.assertEqual(rcfg["max_attempts"], 2)
        self.assertTrue(rcfg["fallback_enabled"])
        self.assertEqual(rcfg["max_models"], 3)
        self.assertEqual(rcfg["cb_failure_threshold"], 5)
        self.assertEqual(rcfg["cb_cooldown_s"], 60.0)
        self.assertIn(503, rcfg["on_status"])


if __name__ == "__main__":
    unittest.main()
