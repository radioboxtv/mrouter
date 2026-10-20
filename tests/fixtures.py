"""Shared test fixtures for the mrouter test suite. No network, no live services."""

import copy
import json
import os
import shutil
import tempfile
import unittest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys_path_setup = None
if BASE not in __import__("sys").path:
    __import__("sys").path.insert(0, BASE)

from mrouter import discovery  # noqa: E402


def make_local(key="qwen/qwen3.8-27b", **over):
    m = {
        "key": key, "api_id": key, "upstream_id": key, "provider": "lmstudio",
        "is_local": True, "display_name": "Qwen3.8 27B", "publisher": "qwen",
        "arch": "qwen3_5", "params_b": 27.0, "size_gb": 15.0, "quant_bits": 4,
        "quant_name": "4bit", "max_ctx": 262144, "max_output": None,
        "vision": True, "multimodal": True, "tool_use": True, "hybrid_attention": True,
        "cost_input_per_token": 0.0, "cost_output_per_token": 0.0,
        "cost_cache_read_per_token": 0.0,
    }
    m.update(over)
    return m


def make_cloud(key="aperture:deepseek/deepseek-v4-flash", name="DeepSeek V4 Flash",
               p_in=0.13e-6, p_out=0.26e-6, p_cache=None, **over):
    api_id = key.split(":", 1)[1] if key.startswith("aperture:") else key
    m = {
        "key": key, "api_id": api_id, "upstream_id": api_id, "provider": "aperture",
        "is_local": False, "display_name": name, "publisher": "deepseek", "arch": "cloud",
        "params_b": None, "size_gb": 0.0, "quant_bits": None, "quant_name": None,
        "max_ctx": 1048576, "max_output": 262144, "vision": None, "multimodal": None,
        "tool_use": True, "hybrid_attention": False,
        "cost_input_per_token": p_in, "cost_output_per_token": p_out,
        "cost_cache_read_per_token": p_cache, "upstream": "aperture",
    }
    m.update(over)
    return m


def standard_fleet():
    """Local cheap-fast, local big-smart, cloud value, cloud flagship, cloud premium."""
    return [
        make_local("google/gemma-4-e4b", display_name="Gemma 4 E4B", params_b=4.0,
                   size_gb=6.4, max_ctx=131072, arch="gemma4", hybrid_attention=False),
        make_local(),
        make_cloud(),
        make_cloud("aperture:openai/gpt-6-luna", "GPT-6 Luna", 0.10e-6, 0.50e-6, max_ctx=922000),
        make_cloud("aperture:openai/gpt-6-astra", "GPT-6 Astra", 10e-6, 50e-6, max_ctx=922000),
    ]


class ConfigIsolatedTestCase(unittest.TestCase):
    """Bases config paths in a temp dir so tests never touch real config/log files."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mrouter_test_")
        self._saved_env = {k: os.environ.get(k) for k in ("MROUTER_CONFIG", "TYPESAFE_API_KEY")}
        os.environ.pop("MROUTER_CONFIG", None)
        os.environ.pop("TYPESAFE_API_KEY", None)
        self._patch_learning_paths()

    def _patch_learning_paths(self):
        from mrouter import learning
        self._lp = learning.log_path, learning.learned_path

        def log_path_local(config):
            return os.path.join(self.tmp, (config.get("learning", {}) or {}).get("log_file", "route_log.jsonl"))

        def learned_path_local(config):
            return os.path.join(self.tmp, (config.get("learning", {}) or {}).get("learned_file", "learned.json"))

        learning.log_path = log_path_local
        learning.learned_path = learned_path_local
        self.learning = learning

    def make_config(self):
        cfg = copy.deepcopy(discovery.load_config())
        return cfg

    def tearDown(self):
        from mrouter import learning
        learning.log_path, learning.learned_path = self._lp
        for k, v in self._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(self.tmp, ignore_errors=True)
