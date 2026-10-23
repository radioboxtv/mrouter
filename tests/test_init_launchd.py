"""mrouter init: launchd-agent install should never create a duplicate
process fighting over the proxy port with a pre-existing install."""

import os
import tempfile
import unittest
from unittest import mock

from mrouter import init as init_mod


class TestLaunchdCollisionGuard(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mrouter_launchd_test_")
        self._orig_dir = init_mod._LAUNCHD_AGENTS_DIR
        init_mod._LAUNCHD_AGENTS_DIR = self.tmp

    def tearDown(self):
        init_mod._LAUNCHD_AGENTS_DIR = self._orig_dir
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_plist(self, name, program_args):
        import plistlib
        path = os.path.join(self.tmp, name)
        with open(path, "wb") as f:
            plistlib.dump({"Label": name, "ProgramArguments": program_args}, f)
        return path

    def test_no_existing_agent_returns_none(self):
        self.assertIsNone(init_mod._find_existing_serve_agent())

    def test_finds_agent_with_different_label_pointing_at_serve_sh(self):
        # Simulates a pre-PR install: a hand-written plist under an arbitrary
        # label that still execs scripts/serve.sh.
        self._write_plist("com.ea.mrouter.plist", ["/bin/bash", init_mod.SERVE_SCRIPT])
        found = init_mod._find_existing_serve_agent()
        self.assertIsNotNone(found)
        self.assertTrue(found.endswith("com.ea.mrouter.plist"))

    def test_unrelated_plist_is_ignored(self):
        self._write_plist("com.example.other.plist", ["/bin/bash", "/opt/something/else.sh"])
        self.assertIsNone(init_mod._find_existing_serve_agent())

    def test_step_launchd_skips_install_when_conflicting_agent_exists(self):
        self._write_plist("com.ea.mrouter.plist", ["/bin/bash", init_mod.SERVE_SCRIPT])
        with mock.patch("sys.platform", "darwin"), \
             mock.patch("os.path.exists", side_effect=lambda p: p == os.path.join(self.tmp, "com.ea.mrouter.plist")), \
             mock.patch("subprocess.run") as run_mock:
            init_mod._step_launchd(skip=False, non_interactive=True)
            run_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
