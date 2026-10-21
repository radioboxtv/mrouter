"""Baseten decision-model classifier tests (elephant-1 / mammoth-1) — TDD."""

import json
import os
import unittest

import tests.fixtures as fx

USE_CASES = ("chat", "coding", "sensitive", "thinking")

BASETEN_RESPONSE = {
    "model": "decision-models/baseten-mammoth-27b-latest",
    "answers": {"use_case": {
        "type": "choice",
        "choice": "coding",
        "confidence": 0.94,
        "probabilities": {"coding": 0.96, "chat": 0.02,
                          "thinking": 0.015, "sensitive": 0.005},
    }},
}


class TestBasetenClassify(fx.ConfigIsolatedTestCase):
    """Slice 1: _baseten_classify — request shape + typed response parsing."""

    def setUp(self):
        super().setUp()
        from mrouter import categorize
        self.categorize = categorize
        self.requests = []
        self._orig_urlopen = categorize.urllib.request.urlopen
        categorize.urllib.request.urlopen = self._fake_urlopen
        self.addCleanup(setattr, categorize.urllib.request, "urlopen", self._orig_urlopen)

    def _fake_urlopen(self, req, timeout=None):
        self.requests.append(req)
        body = json.loads(req.data.decode())

        class R:
            def __init__(self, data):
                self._data = data

            def read(self):
                return self._data

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return True

        return R(json.dumps(BASETEN_RESPONSE).encode())

    def _cfg(self, **over):
        baseten = {
            "enabled": True,
            "endpoint": "https://inference.baseten.co/v1/systemone",
            "model": "decision-models/baseten-mammoth-27b-latest",
            "api_key_env": "BASETEN_API_KEY",
            "timeout_seconds": 5,
        }
        baseten.update(over)
        cfg = self.make_config()
        cfg["baseten"] = baseten
        return cfg

    def test_request_shape_uses_systemone_criteria_map(self):
        self.categorize._baseten_classify("debug this traceback", self._cfg(), "k")
        req = self.requests[0]
        self.assertTrue(req.full_url.endswith("/v1/systemone"))
        self.assertEqual(req.headers.get("Authorization"), "Bearer k")
        body = json.loads(req.data.decode())
        self.assertEqual(body["model"], "decision-models/baseten-mammoth-27b-latest")
        self.assertEqual(body["state"], "debug this traceback")
        q = body["questions"]["use_case"]
        self.assertEqual(q["type"], "choice")
        # Baseten uses a named criteria map, NOT Jev's options array
        self.assertEqual(set(q["criteria"].keys()), set(USE_CASES))
        self.assertNotIn("options", q)

    def test_parses_choice_answer_with_probabilities(self):
        r = self.categorize._baseten_classify("debug this traceback", self._cfg(), "k")
        self.assertEqual(r["source"], "baseten")
        self.assertEqual(r["top"], "coding")
        self.assertEqual(r["model"], "decision-models/baseten-mammoth-27b-latest")
        self.assertAlmostEqual(r["confidence"], 0.94, places=2)
        self.assertAlmostEqual(sum(r["probs"].values()), 1.0, places=2)
        self.assertAlmostEqual(r["probs"]["coding"], 0.96, places=2)

    def test_invalid_answer_returns_none(self):
        payload = {"answers": {"use_case": {"choice": "nonsense"}}}

        def bad(req, timeout=None):
            class R:
                def read(self):
                    return json.dumps(payload).encode()

                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    return True

            return R()

        self.categorize.urllib.request.urlopen = bad
        self.assertIsNone(self.categorize._baseten_classify("x", self._cfg(), "k"))

    def test_elephant_slug_configurable(self):
        self.categorize._baseten_classify("x", self._cfg(model="decision-models/baseten-elephant-9b-latest"), "k")
        body = json.loads(self.requests[0].data.decode())
        self.assertEqual(body["model"], "decision-models/baseten-elephant-9b-latest")


class TestProviderSelection(fx.ConfigIsolatedTestCase):
    """Slice 2: classify_task picks the configured decision-model provider."""

    def setUp(self):
        super().setUp()
        from mrouter import categorize
        self.categorize = categorize
        self._saved = {
            "baseten": categorize._baseten_classify,
            "jev": categorize._jev_classify,
            "attempt_baseten": categorize._attempt_baseten,
        }
        saved = dict(self._saved)

        def _restore():
            self.categorize._baseten_classify = saved["baseten"]
            self.categorize._jev_classify = saved["jev"]
            self.categorize._attempt_baseten = saved["attempt_baseten"]

        self.addCleanup(_restore)

    def test_explicit_baseten_provider_uses_baseten(self):
        cfg = self.make_config()
        cfg["decision"] = {"provider": "baseten"}
        cfg["baseten"] = {"enabled": True, "model": "decision-models/baseten-mammoth-27b-latest",
                          "api_key_env": "BASETEN_API_KEY"}
        os.environ["BASETEN_API_KEY"] = "k"
        called = {}

        def fake(text, config, key):
            called["baseten"] = True
            return {"probs": {}, "top": "coding", "confidence": 0.9,
                    "source": "baseten", "model": "m"}

        self.categorize._baseten_classify = fake
        try:
            r = self.categorize.classify_task("x", cfg)
        finally:
            del os.environ["BASETEN_API_KEY"]
        self.assertTrue(called.get("baseten"))
        self.assertEqual(r["source"], "baseten")

    def test_baseten_without_key_falls_back_local(self):
        cfg = self.make_config()
        cfg["decision"] = {"provider": "baseten"}
        cfg["baseten"] = {"enabled": True, "api_key_env": "BASETEN_API_KEY"}

        def boom(text, config, key):
            raise AssertionError("must not call baseten without a key")

        self.categorize._baseten_classify = boom
        r = self.categorize.classify_task("x", cfg)
        self.assertEqual(r["source"], "local-heuristic")

    def test_auto_prefers_baseten_over_jev_when_both_keys(self):
        cfg = self.make_config()
        cfg["decision"] = {"provider": "auto"}
        cfg["baseten"] = {"enabled": True, "api_key_env": "BASETEN_API_KEY"}
        os.environ["BASETEN_API_KEY"] = "bk"
        os.environ["TYPESAFE_API_KEY"] = "jk"
        called = {}

        def fake_baseten(*a):
            called["b"] = True
            return {"probs": {}, "top": "chat", "confidence": 0.9,
                    "source": "baseten", "model": "m"}

        def fake_jev(*a):
            called["j"] = True
            return {"probs": {}, "top": "chat", "confidence": 0.9,
                    "source": "jev", "model": "m"}

        self.categorize._baseten_classify = fake_baseten
        self.categorize._jev_classify = fake_jev
        try:
            r = self.categorize.classify_task("x", cfg)
        finally:
            del os.environ["BASETEN_API_KEY"]
            del os.environ["TYPESAFE_API_KEY"]
        self.assertTrue(called.get("b"))
        self.assertFalse(called.get("j"))
        self.assertEqual(r["source"], "baseten")

    def test_explicit_jev_provider_skips_baseten(self):
        cfg = self.make_config()
        cfg["decision"] = {"provider": "jev"}
        cfg["baseten"] = {"enabled": True, "api_key_env": "BASETEN_API_KEY"}
        cfg["jev"]["enabled"] = True
        os.environ["BASETEN_API_KEY"] = "bk"
        os.environ["TYPESAFE_API_KEY"] = "jk"

        def boom(text, config, key):
            raise AssertionError("jev provider must not call baseten")

        self.categorize._baseten_classify = boom
        self.categorize._jev_classify = lambda *a: {"probs": {}, "top": "chat",
                                                    "confidence": 0.9, "source": "jev", "model": "m"}
        try:
            r = self.categorize.classify_task("x", cfg)
        finally:
            del os.environ["BASETEN_API_KEY"]
            del os.environ["TYPESAFE_API_KEY"]
        self.assertEqual(r["source"], "jev")

    def test_baseten_failure_falls_back_local(self):
        cfg = self.make_config()
        cfg["decision"] = {"provider": "baseten"}
        cfg["baseten"] = {"enabled": True, "api_key_env": "BASETEN_API_KEY"}
        os.environ["BASETEN_API_KEY"] = "k"

        def boom(text, config, key):
            raise ConnectionError("baseten down")

        self.categorize._baseten_classify = boom
        try:
            r = self.categorize.classify_task("debug this code", cfg)
        finally:
            del os.environ["BASETEN_API_KEY"]
        self.assertEqual(r["source"], "local-heuristic")


if __name__ == "__main__":
    unittest.main()