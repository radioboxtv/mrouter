"""mrouter init: _step_install must use sys.executable (never a hardcoded
python path) and record it for scripts/serve.sh to pick up under launchd."""

import os
import tempfile
import unittest
from unittest import mock

from mrouter import init as init_mod


class TestStepInstall(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mrouter_init_test_")
        self._orig_record_path = init_mod.PYTHON_RECORD_PATH
        init_mod.PYTHON_RECORD_PATH = os.path.join(self.tmp, ".mrouter_python")

    def tearDown(self):
        init_mod.PYTHON_RECORD_PATH = self._orig_record_path
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_uses_sys_executable_not_a_hardcoded_path(self):
        """The core bug this closes: a hand-written wrapper hardcoded
        /usr/bin/python3, silently diverging from whatever interpreter a
        user's `pip install` actually targeted (mise, a venv, etc.) --
        `pip install Requirement already satisfied` into one interpreter,
        `mrouter tray` still failing because it execs a different one.
        _step_install must always install into sys.executable."""
        with mock.patch("subprocess.run") as run_mock:
            run_mock.return_value = mock.Mock(returncode=0)
            init_mod._step_install(non_interactive=True)
            args = run_mock.call_args[0][0]
            self.assertEqual(args[0], init_mod.sys.executable)
            self.assertIn("pip", args)
            self.assertIn("install", args)
            self.assertIn("-e", args)

    def test_records_interpreter_path_on_success(self):
        with mock.patch("subprocess.run") as run_mock:
            run_mock.return_value = mock.Mock(returncode=0)
            init_mod._step_install(non_interactive=True)
        self.assertTrue(os.path.exists(init_mod.PYTHON_RECORD_PATH))
        with open(init_mod.PYTHON_RECORD_PATH) as f:
            self.assertEqual(f.read(), init_mod.sys.executable)

    def test_pip_failure_does_not_raise(self):
        """pip can fail (no network, read-only site-packages, etc.) -- init
        must degrade to a printed hint, never crash the whole init run."""
        import subprocess as subprocess_mod
        with mock.patch("subprocess.run") as run_mock:
            run_mock.side_effect = subprocess_mod.CalledProcessError(1, ["pip"])
            try:
                init_mod._step_install(non_interactive=True)
            except Exception as e:
                self.fail(f"_step_install raised {type(e).__name__}: {e}")
        self.assertFalse(os.path.exists(init_mod.PYTHON_RECORD_PATH))

    def test_unexpected_exception_does_not_raise(self):
        with mock.patch("subprocess.run", side_effect=OSError("no pip")):
            try:
                init_mod._step_install(non_interactive=True)
            except Exception as e:
                self.fail(f"_step_install raised {type(e).__name__}: {e}")


if __name__ == "__main__":
    unittest.main()
