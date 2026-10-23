"""OpenAI-compatible multiplexing routing proxy.

Fronts BOTH LM Studio (local, $0) and your cloud gateway (priced).
Request /v1/chat/completions with `model` set to a use-case name
(chat, coding, sensitive, thinking), "router/<use-case>", or "auto"
(classify the prompt first, then route) and the proxy picks the current
best model across the whole fleet, forwards to the right upstream, logs
actual usage/cost/latency for learning, and tags the response with
X-Routed-Model / X-Routed-Provider / X-Use-Case headers.

`sensitive` is hard local-only. Any explicit model id passes through:
local ids go to LM Studio, everything else to Aperture.
"""

import json
import os
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import categorize, discovery, heuristics, learning, providers as providers_mod

FLEET_TTL_S = 30
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json")
UI_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui.html")


class Router:
    def __init__(self, config):
        self.config = config
        self._fleet = None
        self._fleet_ts = 0.0
        self._benched_keys = set()  # local model keys we've already kicked a bench job for
        self._bench_lock = threading.Lock()

    def fleet(self):
        now = time.time()
        if self._fleet is None or now - self._fleet_ts > FLEET_TTL_S:
            self._fleet = discovery.discover_fleet(self.config)
            self._fleet_ts = now
            self._maybe_autobench()
        return self._fleet

    def _maybe_autobench(self):
        """Background micro-benchmark for any local model missing fresh bench
        data, so routing speed scores stay accurate with zero manual `mrouter
        bench` runs. Fully fire-and-forget: dispatched onto its own daemon
        thread immediately, including the LM-Studio liveness/start check —
        `fleet()` is called on every request-handling thread, so nothing here
        may block waiting on a network call or a 60s `lms server start`."""
        try:
            max_age_h = float((self.config.get("bench", {}) or {}).get("max_age_hours", 7 * 24))
            bench = heuristics.load_bench()
            stale = heuristics.stale_local_models(self._fleet, bench, max_age_hours=max_age_h)
        except Exception:
            return
        with self._bench_lock:
            todo = [m for m in stale if m["key"] not in self._benched_keys]
            for m in todo:
                self._benched_keys.add(m["key"])
        if not todo:
            return
        threading.Thread(target=self._bench_worker, args=(todo,), daemon=True).start()

    def _bench_worker(self, models):
        """Runs entirely off the request-handling thread: liveness check,
        best-effort LM Studio start, then the benchmark itself."""
        try:
            if not discovery.server_alive(self.config):
                if not discovery.start_server(self.config):
                    return  # can't bench without a live LM Studio server; try again next refresh
            self._run_bench(models)
        finally:
            with self._bench_lock:
                for m in models:
                    self._benched_keys.discard(m["key"])

    def _run_bench(self, models):
        from . import bench as bench_mod
        try:
            print(f"[mrouter] auto-benchmarking {len(models)} local model(s) in background "
                  f"({', '.join(m['display_name'] for m in models)})...", flush=True)
            bench_mod.run_bench(models, self.config)
        except Exception as e:
            print(f"[mrouter] auto-bench failed ({type(e).__name__}: {e})", flush=True)

    def by_key(self, key):
        return next((m for m in self.fleet() if m["key"] == key), None)

    def by_api_id(self, api_id):
        return next((m for m in self.fleet() if m.get("api_id") == api_id), None)

    def pick(self, use_case, bench=None, learned=None):
        fleet = self.fleet()
        ranked = heuristics.score_fleet(fleet, self.config, bench, learned)
        rows = ranked[use_case]["models"]
        if not rows:
            return None, None
        info = rows[0]
        return self.by_key(info["key"]), info


def _pick_for_request(router, payload, requested_route):
    """Resolve (use_case, model, info, classification). Handles 'auto'."""
    route = requested_route
    classification = None
    if route == "auto":
        text = categorize.extract_user_text(payload)
        classification = categorize.classify_task(text, router.config)
        probs = classification["probs"]
        # Privacy floor: if the prompt looks even moderately sensitive, force
        # the local-only route regardless of which use case scored highest.
        floor = float((router.config.get("routing", {}) or {}).get(
            "sensitive_probability_floor", 0.25))
        if probs.get("sensitive", 0.0) >= floor:
            model, info = router.pick("sensitive")
            return "sensitive", model, info, classification
        # Blend: score each candidate model by probability-weighted use-case scores.
        ranked = heuristics.score_fleet(router.fleet(), router.config)
        best_key, best_blend, best_uc = None, -1, classification["top"]
        union = {}
        for uc, r in ranked.items():
            for row in r["models"]:
                union[row["key"]] = union.get(row["key"], 0.0) + probs.get(uc, 0.0) * row["score"]
        if union:
            best_key = max(union, key=union.get)
            best_uc = max(probs, key=probs.get)
        model = router.by_key(best_key) if best_key else None
        info = None
        if model:
            for uc, r in ranked.items():
                for row in r["models"]:
                    if row["key"] == best_key:
                        info = {**row, "blended_score": round(union[best_key], 1)}
                        break
        return best_uc, model, info, classification
    if route in router.config["use_cases"]:
        model, info = router.pick(route)
        return route, model, info, classification
    return None, None, None, classification


class Handler(BaseHTTPRequestHandler):
    router = None  # set by serve()

    def log_message(self, fmt, *args):
        print(f"[mrouter] {self.address_string()} {fmt % args}", flush=True)

    def _send(self, code, payload, headers=None):
        body = json.dumps(payload).encode() if not isinstance(payload, bytes) else payload
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    # ---------------------------------------------------------------- GET
    def do_GET(self):
        r = self.router
        cfg = r.config
        if self.path in ("/", "/ui", "/ui/"):
            try:
                with open(UI_PATH, "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:
                self._send(500, {"error": str(e)})
            return
        if self.path in ("/health", "/healthz"):
            fleet = r.fleet()
            alive = discovery.server_alive(cfg)
            self._send(200, {
                "ok": True, "lm_studio_up": alive,
                "fleet": {"local": sum(1 for m in fleet if m["is_local"]),
                          "aperture": sum(1 for m in fleet if not m["is_local"])},
            })
            return
        if self.path == "/v1/models":
            fleet = r.fleet()
            ids = [{"id": m["api_id"], "object": "model",
                    "x_mrouter": {"provider": m["provider"], "key": m["key"],
                                  "input_per_m": round((m.get("cost_input_per_token") or 0) * 1e6, 4),
                                  "output_per_m": round((m.get("cost_output_per_token") or 0) * 1e6, 4)}}
                   for m in fleet]
            ids += [{"id": uc, "object": "model"} for uc in cfg["use_cases"]]
            ids.append({"id": "auto", "object": "model"})
            self._send(200, {"object": "list", "data": ids})
            return
        if self.path == "/v1/config":
            try:
                with open(CONFIG_PATH) as f:
                    body = json.dumps(json.load(f), indent=2).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:
                self._send(500, {"error": str(e)})
            return
        if self.path == "/v1/routes":
            ranked = heuristics.score_fleet(r.fleet(), cfg)
            out = {}
            for uc, rr in ranked.items():
                out[uc] = {
                    "best": rr["models"][0]["key"] if rr["models"] else None,
                    "score": rr["models"][0]["score"] if rr["models"] else None,
                    "est_cost_usd": rr["models"][0]["est_cost_usd"] if rr["models"] else None,
                    "top3": [{"key": m["key"], "score": m["score"], "est_cost_usd": m["est_cost_usd"]}
                             for m in rr["models"][:3]],
                }
            self._send(200, {"routes": out, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
            return
        if self.path == "/v1/stats":
            entries = learning.load_log(cfg)
            learned = learning.load_learned(cfg)
            total_cost = sum(float(e.get("actual_cost_usd") or 0) for e in entries)
            self._send(200, {
                "logged_requests": len(entries),
                "total_actual_cost_usd": round(total_cost, 6),
                "learned_adjustments": learned,
            })
            return
        self._send(404, {"error": "not found"})

    # ---------------------------------------------------------------- POST
    def do_POST(self):
        r = self.router
        cfg = r.config

        if self.path == "/v1/config":
            length = int(self.headers.get("Content-Length") or 0)
            try:
                new_cfg = json.loads(self.rfile.read(length))
            except Exception:
                self._send(400, {"error": "invalid JSON"})
                return
            try:
                with open(CONFIG_PATH, "w") as f:
                    json.dump(new_cfg, f, indent=2)
                r.config = new_cfg
                cfg = new_cfg
                self._send(200, {"ok": True})
            except Exception as e:
                self._send(500, {"error": str(e)})
            return

        if self.path.endswith("/feedback"):
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except Exception:
                self._send(400, {"error": "invalid JSON"})
                return
            uc, key, rating = body.get("use_case"), body.get("model"), body.get("rating")
            if uc not in cfg["use_cases"] or not key or rating not in (1, -1, "1", "-1"):
                self._send(400, {"error": "need use_case, model (fleet key), rating (+1|-1)"})
                return
            learning.append_feedback(cfg, uc, key, int(rating), body.get("note", ""))
            self._send(200, {"ok": True})
            return

        if self.path.endswith("/classify"):
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except Exception:
                self._send(400, {"error": "invalid JSON"})
                return
            text = body.get("text") or categorize.extract_user_text(body)
            self._send(200, categorize.classify_task(text, cfg))
            return

        if not self.path.endswith("/chat/completions"):
            self._send(404, {"error": "only /v1/chat/completions is routed"})
            return

        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length))
        except Exception:
            self._send(400, {"error": "invalid JSON"})
            return

        requested = payload.get("model", "")
        route = requested[len("router/"):] if requested.startswith("router/") else requested

        started = time.time()
        use_case, model, info, classification = None, None, None, None

        if route == "auto" or route in cfg["use_cases"]:
            use_case, model, info, classification = _pick_for_request(r, payload, route)
            if model is None:
                self._send(503, {"error": f"no eligible model for '{route}'"})
                return
        else:
            # Explicit model id: resolve to provider for cost accounting + upstream.
            model = r.by_api_id(requested) or r.by_key(requested)
            if model is None:
                # Unknown id: assume local if LM Studio lists it, else Aperture.
                local_hit = next((m for m in r.fleet() if m["is_local"] and m["api_id"] == requested), None)
                model = local_hit

        upstream_model_id = model["api_id"] if model else requested
        is_local = bool(model and model.get("is_local"))
        # Unknown explicit ids default to Aperture (the broader catalog).
        if model is None:
            upstream_base = cfg["providers"]["aperture"]["api_base"]
            provider_name = "aperture"
        elif is_local:
            upstream_base = cfg["lm_studio"]["api_base"]
            provider_name = "lmstudio"
        else:
            upstream_base = cfg["providers"]["aperture"]["api_base"]
            provider_name = "aperture"

        payload["model"] = upstream_model_id
        headers = {"Content-Type": "application/json"}
        if not is_local:
            key = providers_mod.resolve_provider_key(cfg.get("providers", {}).get("aperture", {}) or {})
            if self.headers.get("Authorization"):
                headers["Authorization"] = self.headers["Authorization"]
            elif key and key != "no-key-required":
                headers["Authorization"] = f"Bearer {key}"
        elif self.headers.get("Authorization"):
            headers["Authorization"] = self.headers["Authorization"]

        upstream_req = urllib.request.Request(
            upstream_base.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode(), headers=headers, method="POST",
        )

        extra = {}
        if use_case:
            extra["X-Use-Case"] = use_case
            extra["X-Routed-Model"] = model["key"]
            extra["X-Routed-Provider"] = provider_name
            if classification:
                extra["X-Classifier"] = classification["source"]
                extra["X-Classifier-Confidence"] = str(classification["confidence"])
            print(f"[mrouter] route '{route}' -> {model['display_name']} "
                  f"[{provider_name}] (score {info and info.get('score')}; "
                  f"est ${info and info.get('est_cost_usd')})", flush=True)

        stream = bool(payload.get("stream"))
        try:
            resp = urllib.request.urlopen(upstream_req, timeout=600)
        except urllib.error.HTTPError as e:
            body = e.read()
            learning.append_log(cfg, {
                "type": "request", "use_case": use_case, "requested": requested,
                "routed_model_key": model["key"] if model else None,
                "provider": provider_name, "status_code": e.code,
                "classifier": classification, "ttft_s": None,
                "duration_s": round(time.time() - started, 3), "usage": None,
                "actual_cost_usd": None, "stream": stream,
            })
            self._send(e.code, body)
            return
        except Exception as e:
            self._send(502, {"error": str(e)})
            return

        if stream:
            self.send_response(resp.status)
            # NOTE: do not relay the upstream's `Transfer-Encoding: chunked` header.
            # urllib already de-chunks the upstream response before resp.read()
            # returns bytes (http.client strips chunk framing internally), so the
            # bytes written below are a plain stream, not valid chunk-encoded
            # data. Declaring chunked framing while writing plain bytes is a
            # protocol violation that curl/urllib tolerate but strict HTTP/1.1
            # clients (Node's undici/fetch, used by Hermes and others) correctly
            # reject with "Invalid character in chunk size". BaseHTTPRequestHandler
            # defaults to HTTP/1.0 anyway, which has no notion of chunked transfer
            # at all — the only valid framing here is "no Content-Length, body
            # ends at connection close", so we send Content-Type only and let
            # end-of-response close the connection.
            if "Content-Type" in resp.headers:
                self.send_header("Content-Type", resp.headers["Content-Type"])
            for k, v in extra.items():
                self.send_header(k, v)
            self.close_connection = True
            self.end_headers()
            ttft = None
            usage = None
            buf = b""
            while True:
                chunk = resp.read(1024)
                if not chunk:
                    break
                if ttft is None:
                    ttft = time.time() - started
                buf += chunk
                # SSE usage rides in a late chunk; scan incrementally, cheap enough.
                try:
                    for line in chunk.decode("utf-8", "replace").splitlines():
                        if line.startswith("data:") and '"usage"' in line:
                            d = json.loads(line[5:].strip())
                            if d.get("usage"):
                                usage = d["usage"]
                except Exception:
                    pass
                self.wfile.write(chunk)
                self.wfile.flush()
            if usage is None:
                # final parse of the tail for usage
                try:
                    for line in buf.decode("utf-8", "replace").splitlines()[-40:]:
                        if line.startswith("data:") and '"usage"' in line:
                            d = json.loads(line[5:].strip())
                            if d.get("usage"):
                                usage = d["usage"]
                except Exception:
                    pass
            learning.append_log(cfg, {
                "type": "request", "use_case": use_case, "requested": requested,
                "routed_model_key": model["key"] if model else None,
                "provider": provider_name, "status_code": resp.status,
                "classifier": classification,
                "ttft_s": round(ttft, 3) if ttft else None,
                "duration_s": round(time.time() - started, 3),
                "usage": usage,
                "actual_cost_usd": providers_mod.actual_cost_usd(model, usage) if model else None,
                "stream": True,
            })
        else:
            data = resp.read()
            usage = None
            try:
                parsed = json.loads(data)
                usage = parsed.get("usage")
            except Exception:
                pass
            learning.append_log(cfg, {
                "type": "request", "use_case": use_case, "requested": requested,
                "routed_model_key": model["key"] if model else None,
                "provider": provider_name, "status_code": resp.status,
                "classifier": classification, "ttft_s": None,
                "duration_s": round(time.time() - started, 3),
                "usage": usage,
                "actual_cost_usd": providers_mod.actual_cost_usd(model, usage) if model else None,
                "stream": False,
            })
            self._send(resp.status, data, extra)


def serve(config, host=None, port=None):
    host = host or config["proxy"]["host"]
    port = port or config["proxy"]["port"]
    # Best-effort: bring LM Studio's local server up so local/sensitive routing
    # works without a human remembering `lms server start` first. This used to
    # live only in scripts/serve.sh (shell-only, launchd-only); doing it here
    # means it also happens for `python3 -m mrouter serve` run directly.
    if not discovery.server_alive(config):
        print("[mrouter] LM Studio server is down — starting it...", flush=True)
        if not discovery.start_server(config):
            print("[mrouter] could not start LM Studio (local/sensitive routing degraded "
                  "until it's up) — open LM Studio or run: lms server start", flush=True)
    Handler.router = Router(config)
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    httpd.request_queue_size = 32  # raise from default 5 — Hermes fires parallel aux/main/subagent calls
    fleetsz = len(Handler.router.fleet())
    print(f"[mrouter] multiplexing proxy on http://{host}:{port}/v1 "
          f"({fleetsz} models: LM Studio local + Aperture)", flush=True)
    print("[mrouter] model = chat|coding|sensitive|thinking|auto (or router/<name>); "
          "explicit ids pass through; sensitive is local-only", flush=True)
    httpd.serve_forever()


def serve_forever(config, host=None, port=None):
    """Resilient entry point for daemons: restart the HTTP server on crashes."""
    while True:
        try:
            serve(config, host, port)
        except KeyboardInterrupt:
            return
        except Exception as e:
            print(f"[mrouter] server crashed ({type(e).__name__}: {e}); restarting in 3s",
                  flush=True)
            time.sleep(3)