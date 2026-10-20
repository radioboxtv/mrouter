"""Config loading and provider resolution tests (hermetic)."""

import json
import os
import unittest

from tests.fixtures import ConfigIsolatedTestCase


class TestConfigLoading(ConfigIsolatedTestCase):
    def test_missing_config_raises_clear_error(self):
        # MROUTER_CONFIG pointing nowhere; no config.json/config.example.json in tmp
        os.environ["MROUTER_CONFIG"] = os.path.join(self.tmp, "nope.json")
        with self.assertRaises((FileNotFoundError, OSError)):
            from mrouter import discovery
            import importlib
            importlib.reload(discovery)
            discovery.load_config()

    def test_env_var_config_wins(self):
        cfg_path = os.path.join(self.tmp, "my_config.json")
        with open(cfg_path, "w") as f:
            json.dump({"proxy": {"port": 9999}, "use_cases": {}, "providers": {},
                       "hardware": {"total_ram_gb": 64}}, f)
        os.environ["MROUTER_CONFIG"] = cfg_path
        from mrouter import discovery
        import importlib
        importlib.reload(discovery)
        cfg = discovery.load_config()
        self.assertEqual(cfg["proxy"]["port"], 9999)


class TestProviderKeyResolution(unittest.TestCase):
    def test_env_var_beats_static(self):
        from mrouter.providers import resolve_provider_key
        os.environ["FAKE_PROVIDER_KEY"] = "from-env"
        try:
            key = resolve_provider_key({"api_key": "static", "api_key_env": "FAKE_PROVIDER_KEY"})
            self.assertEqual(key, "from-env")
        finally:
            del os.environ["FAKE_PROVIDER_KEY"]

    def test_static_fallback(self):
        from mrouter.providers import resolve_provider_key
        key = resolve_provider_key({"api_key": "static", "api_key_env": "UNSET_VAR_XYZ"})
        self.assertEqual(key, "static")

    def test_no_key(self):
        from mrouter.providers import resolve_provider_key
        self.assertIsNone(resolve_provider_key({}))
        self.assertEqual(resolve_provider_key({"api_key": "no-key-required"}), "no-key-required")


if __name__ == "__main__":
    unittest.main()