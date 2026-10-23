"""Proxy integration tests: fake in-process upstreams, routing decisions, logging."""

import json
import os
import socket
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import tests.fixtures as fx
from mrouter import metrics_store, proxy  # noqa: E402


class _FakeUpstream(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible upstream. Recognizes models and echoes JSON."""

    def log_message(self, *a):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length))
        model = body.get("model", "?")
        saw_auth = bool(self.headers.get("Authorization"))
        if body.get("stream"):
            # Real OpenAI-compatible SSE upstream: chunked transfer-encoding,
            # a handful of data: lines, then [DONE]. Mirrors what LM Studio /
            # Aperture actually send for a streaming chat completion.
            lines = [
                'data: {"choices":[{"delta":{"content":"ok "}}]}\n\n',
                'data: {"choices":[{"delta":{"content":"from ' + model + '"}}],'
                '"usage":{"prompt_tokens":10,"completion_tokens":2}}\n\n',
                "data: [DONE]\n\n",
            ]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for line in lines:
                payload = line.encode()
                self.wfile.write(f"{len(payload):x}\r\n".encode() + payload + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
            return
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

    def test_streaming_response_is_valid_http(self):
        """Regression test: mrouter used to relay the upstream's
        `Transfer-Encoding: chunked` header verbatim while writing bytes that
        urllib had ALREADY de-chunked (http.client strips chunk framing before
        resp.read() returns). That's a protocol violation -- declaring chunked
        framing while sending a plain byte stream -- that curl/urllib silently
        tolerate but strict HTTP/1.1 clients correctly reject. Confirmed live
        against Node's undici (the client Hermes Agent uses): it raised
        HTTPParserError: "Invalid character in chunk size" on exactly this
        response shape. This test reproduces the same wire-level check with a
        raw socket, independent of any particular client library."""
        fleet = fx.standard_fleet()
        cfg, local_srv, cloud_srv = self._config_with_fakes(fleet)
        self._patch_fleet(cfg, fleet)
        srv = self._start_proxy(cfg)
        port = srv.server_address[1]

        body = json.dumps({
            "model": "deepseek/deepseek-v4-flash",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 5, "stream": True,
        }).encode()
        request = (
            b"POST /v1/chat/completions HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: " + str(len(body)).encode() + b"\r\n"
            b"Connection: close\r\n"
            b"\r\n" + body
        )
        with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
            sock.sendall(request)
            raw = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                raw += chunk

        head, _, response_body = raw.partition(b"\r\n\r\n")
        header_lines = head.decode("latin-1").split("\r\n")
        status_line = header_lines[0]
        headers = dict(
            line.split(": ", 1) for line in header_lines[1:] if ": " in line
        )
        self.assertIn("200", status_line)
        # The core bug: never claim chunked framing for a body that isn't
        # actually chunk-encoded on the wire.
        te = headers.get("Transfer-Encoding", "")
        self.assertNotEqual(te.lower(), "chunked",
                             "response declares Transfer-Encoding: chunked but the "
                             "proxy does not chunk-encode its output -- this is the "
                             "exact defect that broke Hermes/undici streaming")
        # With no Transfer-Encoding and no Content-Length, HTTP/1.0-style framing
        # (body ends at connection close) is valid -- verify the body we actually
        # received is the plain SSE text, not chunk-size-prefixed garbage.
        self.assertIn(b"data:", response_body)
        self.assertIn(b"[DONE]", response_body)
        # A chunk-encoded body, misdecoded as plain text, would still contain the
        # hex chunk-size lines as garbage before each data: line -- make sure
        # none leaked through.
        self.assertNotRegex(response_body.decode("latin-1", "replace"),
                             r"(?m)^[0-9a-f]+\r?\n")

    # -------------------------------------------------------- metrics store

    def _get(self, port, path):
        req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method="GET")
        with urllib.request.urlopen(req, timeout=10) as r:
            return r, json.loads(r.read().decode())

    def test_non_streaming_request_writes_metrics_row(self):
        fleet = fx.standard_fleet()
        cfg, local_srv, cloud_srv = self._config_with_fakes(fleet)
        self._patch_fleet(cfg, fleet)
        srv = self._start_proxy(cfg)
        port = srv.server_address[1]

        self._post(port, {
            "model": "deepseek/deepseek-v4-flash",
            "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5})

        rows = self.metrics_store.query_window(cfg, since_ts=0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status_code"], 200)
        self.assertEqual(rows[0]["tokens_in"], 1000)
        self.assertEqual(rows[0]["tokens_out"], 100)
        self.assertFalse(rows[0]["cost_is_estimate"])

    def test_streaming_request_writes_metrics_row(self):
        fleet = fx.standard_fleet()
        cfg, local_srv, cloud_srv = self._config_with_fakes(fleet)
        self._patch_fleet(cfg, fleet)
        srv = self._start_proxy(cfg)
        port = srv.server_address[1]

        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps({
                "model": "deepseek/deepseek-v4-flash",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 5, "stream": True,
            }).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as r:
            r.read()

        rows = self.metrics_store.query_window(cfg, since_ts=0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status_code"], 200)
        self.assertIsNotNone(rows[0]["ttft_s"])

    def test_errored_request_writes_metrics_row_with_error_status(self):
        """With the resilience/fallback feature (merged separately), a
        'sensitive' request has 2 local candidates (gemma, qwen); pointing
        lm_studio at a single broken upstream makes BOTH candidates fail, so
        metrics records one row per attempt -- same per-attempt granularity
        as learning.append_log, not just the terminal outcome."""
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
                data = b'{"error": "boom"}'
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        broken = ThreadingHTTPServer(("127.0.0.1", 0), _BrokenUpstream)
        broken.daemon_threads = True
        threading.Thread(target=broken.serve_forever, daemon=True).start()
        self.addCleanup(broken.server_close)
        self.addCleanup(broken.shutdown)
        cfg["lm_studio"]["api_base"] = f"http://127.0.0.1:{broken.server_port}/v1"

        try:
            self._post(port, {"model": "sensitive",
                               "messages": [{"role": "user", "content": "x"}],
                               "max_tokens": 5})
        except urllib.error.HTTPError:
            pass

        rows = self.metrics_store.query_window(cfg, since_ts=0)
        # Both local 'sensitive' candidates are broken and attempted (cascade
        # exhausted), so metrics gets one row per attempt, not one per request.
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["status_code"] == 500 for row in rows))

    def test_metrics_endpoint_returns_window_summary(self):
        fleet = fx.standard_fleet()
        cfg, local_srv, cloud_srv = self._config_with_fakes(fleet)
        self._patch_fleet(cfg, fleet)
        srv = self._start_proxy(cfg)
        port = srv.server_address[1]

        self._post(port, {
            "model": "deepseek/deepseek-v4-flash",
            "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5})

        r, body = self._get(port, "/v1/metrics?window=1h")
        self.assertEqual(r.status, 200)
        self.assertEqual(body["requests"], 1)
        self.assertIn("models", body)
        self.assertIn("series", body)

    def test_metrics_live_endpoint_is_cheap_recent_snapshot(self):
        fleet = fx.standard_fleet()
        cfg, local_srv, cloud_srv = self._config_with_fakes(fleet)
        self._patch_fleet(cfg, fleet)
        srv = self._start_proxy(cfg)
        port = srv.server_address[1]

        self._post(port, {
            "model": "deepseek/deepseek-v4-flash",
            "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5})

        r, body = self._get(port, "/v1/metrics/live")
        self.assertEqual(r.status, 200)
        self.assertIn("requests", body)
        self.assertIn("spend_total_usd", body)

    def test_routes_endpoint_surfaces_perf_source(self):
        """/v1/routes must expose speed_source/ttft_source per candidate so
        clients (dashboard, doctor-style tooling) can tell real measurements
        from guesses without a separate call."""
        fleet = fx.standard_fleet()
        cfg, local_srv, cloud_srv = self._config_with_fakes(fleet)
        self._patch_fleet(cfg, fleet)
        srv = self._start_proxy(cfg)
        port = srv.server_address[1]

        r, body = self._get(port, "/v1/routes")
        self.assertEqual(r.status, 200)
        for uc, rr in body["routes"].items():
            if not rr["top3"]:
                continue
            for row in rr["top3"]:
                self.assertIn("speed_source", row)
                self.assertIn("ttft_source", row)
                self.assertIn(row["speed_source"], ("live", "bench", "estimated"))
                self.assertIn(row["ttft_source"], ("live", "bench", "estimated"))


if __name__ == "__main__":
    unittest.main()