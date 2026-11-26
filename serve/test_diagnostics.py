"""moex: "diagnostics": false in the config - the engine (strata-x) and the DeepSeek-V4.1 image encoder send no diagnostic
line anywhere (STRATA_X_QUIET=1), and the server opens no "log".  No GPU is used: the engine and the encoder are run with
bad arguments, whose usage text is their only output."""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from serve.server import child_env, diagnostics_on  # noqa: E402


def run(cmd, quiet):
    env = {k: v for k, v in os.environ.items() if not k.startswith("STRATA_GLM_")}
    env.pop("STRATA_X_QUIET", None)
    if quiet:
        env["STRATA_X_QUIET"] = "1"
    return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300, cwd=str(ROOT))


class Switch(unittest.TestCase):
    def test_on_without_the_key(self):
        self.assertTrue(diagnostics_on({}))
        self.assertTrue(diagnostics_on({"diagnostics": True}))
        with mock.patch.dict(os.environ):
            os.environ.pop("STRATA_X_QUIET", None)
            os.environ.pop("STRATA_GLM_QUIET", None)
            env = child_env({"log": "x.log"})
            self.assertNotIn("STRATA_X_QUIET", env)
            self.assertNotIn("STRATA_GLM_QUIET", env)

    def test_off(self):
        self.assertFalse(diagnostics_on({"diagnostics": False}))
        env = child_env({"diagnostics": False, "log": "x.log"})
        self.assertEqual((env["STRATA_X_QUIET"], env["STRATA_GLM_QUIET"]), ("1", "1"))   # the second: builds before the rename


class Engine(unittest.TestCase):
    """strata-x with an unknown argument: its usage on stderr, and nothing at all when quiet (STRATA_TEST_EXE: the build)."""

    def test_quiet(self):
        exe = os.environ.get("STRATA_TEST_EXE") or next(
            (str(p) for p in (ROOT / "build-dsv41-dev/strata-x", ROOT / "build-dsv41/strata-x") if p.exists()), None)
        if not exe:
            self.skipTest("no strata-x build")
        loud, quiet = run([exe, "--no-such-flag"], False), run([exe, "--no-such-flag"], True)
        self.assertEqual(loud.returncode, 2)
        self.assertIn("usage: strata-x", loud.stderr)
        self.assertEqual((quiet.returncode, quiet.stdout, quiet.stderr), (2, "", ""))

    def test_an_old_switch_name_is_said(self):
        exe = os.environ.get("STRATA_TEST_EXE") or next(
            (str(p) for p in (ROOT / "build-dsv41-dev/strata-x", ROOT / "build-dsv41/strata-x") if p.exists()), None)
        if not exe:
            self.skipTest("no strata-x build")
        env = {k: v for k, v in os.environ.items() if not k.startswith(("STRATA_GLM_", "STRATA_X_QUIET"))}
        env["STRATA_GLM_TC3"] = "1"
        r = subprocess.run([exe, "--no-such-flag"], env=env, capture_output=True, text=True, timeout=300, cwd=str(ROOT))
        self.assertIn("STRATA_GLM_TC3 is the old name of STRATA_X_TC3 and is not read", r.stderr)


class Encoder(unittest.TestCase):
    """vision_dsv41.py without its arguments: argparse's usage error on stderr, and nothing at all when quiet (the
    config's vision python: torch is imported first, so its warnings would show too)."""

    def test_quiet(self):
        cfg = ROOT / "strata-dsv41.json"
        py = (json.loads(cfg.read_text()).get("vision") or {}).get("python") if cfg.exists() else None
        if not py or not Path(py).exists():
            self.skipTest("no DeepSeek-V4.1 vision python")
        cmd = [py, str(ROOT / "serve/vision_dsv41.py")]
        loud, quiet = run(cmd, False), run(cmd, True)
        self.assertEqual(loud.returncode, 2)
        self.assertIn("usage:", loud.stderr)
        self.assertEqual((quiet.returncode, quiet.stdout, quiet.stderr), (2, "", ""))


if __name__ == "__main__":
    unittest.main()
