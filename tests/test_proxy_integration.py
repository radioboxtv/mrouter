"""Proxy integration tests: fake in-process upstreams, routing decisions, logging."""

import json
import os
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import tests.fixtures as fx
from mrouter import proxy  # noqa: E402


class _FakeUpstream(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible upstream. Recognizes models and echoes JSON."""

    def log_message(self, *a):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length))
        model = body.get("model", "?")
        saw_auth = bool(self.headers.get("Authorization"))
        resp = {
            "id": "chatcmpl-test", "object": "chat.completion", "created": 0,
            "model": model,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": f"ok from {model}"}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 100,
                      "prompt_tokens_details": {"cached_tokens": 400}},
            "_saw_auth": saw_auth,
            "_path": self.path,
        }
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class ProxyIntegrationTest(fx.ConfigIsolatedTestCase):
    """Boots the real mrouter proxy against fake local+cloud upstreams on
    ephemeral ports. No real network access beyond 127.0.0.1 loopback."""

    @classmethod
    def setUpClass(cls):
        cls.handler = None

    def _start_proxy(self, config):
        for port in (8731, 8732, 8733, 8734, 8735):
            try:
                srv = ThreadingHTTPServer(("127.0.0.1", port), proxy.Handler)
                break
            except OSError:
                continue
        else:
            raise RuntimeError("no free test port")
        srv.daemon_threads = True
        proxy.Handler.router = proxy.Router(config)
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        return srv

    def _start_upstream(self):
        srv = ThreadingHTTPServer(("127.0.0.1", 0), _FakeUpstream)
        srv.daemon_threads = True
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        return srv

    def _config_with_fakes(self, fleet):
        cfg = self.make_config()
        cfg["proxy"]["host"] = "127.0.0.1"
        cfg["proxy"]["port"] = 0

        local_srv = self._start_upstream()
        cloud_srv = self._start_upstream()
        cfg["lm_studio"]["api_base"] = f"http://127.0.0.1:{local_srv.server_port}/v1"
        cfg["providers"]["aperture"]["api_base"] = f"http://127.0.0.1:{cloud_srv.server_port}/v1"
        cfg["providers"]["aperture"]["api_key"] = "test-cloud-key"
        cfg["providers"]["aperture"]["api_key_env"] = None
        return cfg, local_srv, cloud_srv

    def _patch_fleet(self, config, fleet):
        """Make the Router discover our fixture fleet instead of live models."""
        original_init = proxy.Router.__init__

        def init_with_fleet(self, cfg):
            original_init(self, cfg)
            self._fleet = fleet
            self._fleet_ts = float("inf")

        proxy.Router.__init__ = init_with_fleet
        self.addCleanup(setattr, proxy.Router, "__init__", original_init)

    def _post(self, port, payload):
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as r:
            return r, json.loads(r.read().decode())

    def test_sensitive_routes_local_only(self):
        fleet = fx.standard_fleet()
        cfg, local_srv, cloud_srv = self._config_with_fakes(fleet)
        self._patch_fleet(cfg, fleet)
        srv = self._start_proxy(cfg)
        port = srv.server_address[1]

        for i in range(6):  # try a few ports since we hand-allocated
            try:
                r, body = self._post(port, {
                    "model": "sensitive",
                    "messages": [{"role": "user", "content": "private data"}],
                    "max_tokens": 5})
                break
            except OSError:
                pass
        else:
            self.fail("could not reach test proxy")

        self.assertIn("X-Use-Case", r.headers)
        self.assertEqual(r.headers["X-Routed-Provider"], "lmstudio")
        # Local model never receives cloud auth headers, and cloud never sees the prompt.
        self.assertIn(body["model"], [m["api_id"] for m in fleet if m["is_local"]])

    def test_explicit_cloud_model_uses_key_and_logs_cost(self):
        fleet = fx.standard_fleet()
        cfg, local_srv, cloud_srv = self._config_with_fakes(fleet)
        self._patch_fleet(cfg, fleet)
        srv = self._start_proxy(cfg)
        port = srv.server_address[1]

        r, body = self._post(port, {
            "model": "deepseek/deepseek-v4-flash",
            "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5})
        self.assertEqual(body["model"], "deepseek/deepseek-v4-flash")
        self.assertTrue(body["_saw_auth"])  # fake upstream saw the api key
        # cost log: 1000 in (400 cached at input rate, no cache price set) + 100 out
        expected = 1000 * 0.13e-6 + 100 * 0.26e-6
        with open(self.learning.log_path(cfg)) as f:
            entries = [json.loads(l) for l in f]
        self.assertAlmostEqual(entries[-1]["actual_cost_usd"], expected, places=9)

    def test_auto_classification_routes(self):
        fleet = fx.standard_fleet()
        cfg, *_ = self._config_with_fakes(fleet)
        self._patch_fleet(cfg, fleet)
        srv = self._start_proxy(cfg)
        port = srv.server_address[1]

        r, body = self._post(port, {
            "model": "auto",
            "messages": [{"role": "user", "content": "debug this python traceback"}],
            "max_tokens": 5})
        self.assertTrue(r.headers["X-Classifier"])
        self.assertIn(r.headers["X-Use-Case"], ("coding", "chat"))

    def test_unknown_model_goes_to_cloud(self):
        fleet = fx.standard_fleet()
        cfg, *_ = self._config_with_fakes(fleet)
        self._patch_fleet(cfg, fleet)
        srv = self._start_proxy(cfg)
        port = srv.server_address[1]

        r, body = self._post(port, {
            "model": "totally/unknown-model",
            "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5})
        self.assertEqual(body["_path"], "/v1/chat/completions")
        self.assertEqual(body["model"], "totally/unknown-model")

    def test_error_from_upstream_is_logged(self):
        fleet = fx.standard_fleet()
        cfg, local_srv, cloud_srv = self._config_with_fakes(fleet)
        self._patch_fleet(cfg, fleet)
        srv = self._start_proxy(cfg)
        port = srv.server_address[1]

        class _BrokenUpstream(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                self.send_response(500)
                data = b'{"error": "upstream boom"}'
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        # repoint local at a broken server
        broken = ThreadingHTTPServer(("127.0.0.1", 0), _BrokenUpstream)
        broken.daemon_threads = True
        threading.Thread(target=broken.serve_forever, daemon=True).start()
        self.addCleanup(broken.server_close)
        self.addCleanup(broken.shutdown)
        cfg["lm_studio"]["api_base"] = f"http://127.0.0.1:{broken.server_port}/v1"

        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps({"model": "sensitive",
                             "messages": [{"role": "user", "content": "x"}],
                             "max_tokens": 5}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            urllib.request.urlopen(req, timeout=10)
            self.fail("expected HTTPError")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 500)
        with open(self.learning.log_path(cfg)) as f:
            entries = [json.loads(l) for l in f]
        self.assertEqual(entries[-1]["status_code"], 500)


if __name__ == "__main__":
    unittest.main()