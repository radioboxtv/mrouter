"""Categorizer tests: local heuristic classifier, Jev parsing, user-text extraction."""

import json
import os
import unittest

import tests.fixtures as fx
from mrouter.categorize import _local_classify, extract_user_text  # noqa: E402


class TestLocalClassifier(unittest.TestCase):
    CASES = [
        ("Please debug this Python function, it throws a traceback on import", "coding"),
        ("Refactor this regex and add a unit test for the api client", "coding"),
        ("Summarize this confidential patient salary spreadsheet, do not share it", "sensitive"),
        ("Analyze the trade-offs and compare these strategies in depth, with a research report", "thinking"),
        ("What's a good weeknight dinner idea?", "chat"),
        ("```python\ndef f(): pass\n```\nwhat's wrong here?", "coding"),
    ]

    def test_classifications(self):
        for text, expected in self.CASES:
            r = _local_classify(text)
            self.assertEqual(r["top"], expected, f"misclassified: {text[:40]}")

    def test_probability_shape(self):
        r = _local_classify("hello there")
        self.assertEqual(r["source"], "local-heuristic")
        total = sum(r["probs"].values())
        self.assertLess(abs(total - 1.0), 0.01)
        self.assertTrue(0 < r["confidence"] <= 1)


class TestJevRequestParsing(fx.ConfigIsolatedTestCase):
    def _parse(self, answer_payload):
        """Feed fake answer data through _jev_classify's response-tolerance logic
        by monkeypatching urlopen to return the payload."""
        from mrouter import categorize

        class FakeResp:
            def __init__(self, data):
                self._data = data

            def read(self):
                return self._data

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return True

        def fake_urlopen(req, timeout=None):
            return FakeResp(json.dumps(answer_payload).encode())

        os.environ["TYPESAFE_API_KEY"] = "fake-key"
        orig_urlopen = categorize.urllib.request.urlopen
        categorize.urllib.request.urlopen = fake_urlopen
        try:
            text = "debug this python traceback"
            result = categorize._jev_classify(text, self.make_config(), "fake-key")
        finally:
            categorize.urllib.request.urlopen = orig_urlopen
        return result

    def test_documented_choice_shape(self):
        payload = {
            "model": "jev-latest",
            "answers": {"use_case": {
                "value": "coding",
                "probabilities": {"coding": 0.72, "chat": 0.15,
                                  "thinking": 0.08, "sensitive": 0.05},
                "confidence": 0.72,
            }},
        }
        r = self._parse(payload)
        self.assertEqual(r["top"], "coding")
        self.assertEqual(r["source"], "jev")
        self.assertAlmostEqual(r["confidence"], 0.72, places=2)
        self.assertAlmostEqual(sum(r["probs"].values()), 1.0, places=2)

    def test_string_answer_shape(self):
        payload = {"model": "jev-latest", "answers": {"use_case": "thinking"}}
        r = self._parse(payload)
        self.assertEqual(r["top"], "thinking")
        self.assertGreater(r["probs"]["thinking"], 0.5)

    def test_invalid_answer_returns_none(self):
        payload = {"model": "jev-latest", "answers": {"use_case": {"value": "nonsense"}}}
        r = self._parse(payload)
        self.assertIsNone(r)

    def test_fallback_when_jev_fails(self):
        from mrouter import categorize
        os.environ["TYPESAFE_API_KEY"] = "fake-key"

        def boom(req, timeout=None):
            raise ConnectionError("network down")

        orig = categorize.urllib.request.urlopen
        categorize.urllib.request.urlopen = boom
        try:
            result = categorize.classify_task("debug this python code", self.make_config())
        finally:
            categorize.urllib.request.urlopen = orig
        self.assertEqual(result["source"], "local-heuristic")


class TestExtractUserText(unittest.TestCase):
    def test_user_only_when_system_present(self):
        """Hermes-style payload: system boilerplate must not leak into classification."""
        payload = {"messages": [
            {"role": "system", "content": "You are a coding agent with tools, code, debugging..."},
            {"role": "user", "content": "What's a nice pasta recipe?"},
        ]}
        text = extract_user_text(payload)
        self.assertNotIn("coding agent", text.lower())
        self.assertIn("pasta", text)

    def test_system_fallback_when_no_user(self):
        payload = {"messages": [{"role": "system", "content": "system only"}]}
        self.assertEqual(extract_user_text(payload), "system only")

    def test_multipart_content(self):
        payload = {"messages": [{"role": "user", "content": [
            {"type": "text", "text": "hello "}, {"type": "text", "text": "world"}]}]}
        self.assertEqual(extract_user_text(payload), "hello \nworld")


if __name__ == "__main__":
    unittest.main()