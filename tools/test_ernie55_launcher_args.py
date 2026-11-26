"""Check launch-ernie55-flash.sh's argument forwarding (as tools/test_dsv41_launcher_args.py does for the DeepSeek launcher)
without starting a model or systemd: the launcher's own exec line runs with systemd-run replaced by an argument recorder.

Run: python tools/test_ernie55_launcher_args.py
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "launch-ernie55-flash.sh"


def env_python():
    """The Python moex/env.sh picks for the launchers ($STRATA_PYTHON, else ./.venv, else a Strata install beside
    the checkout, else python3)."""
    return subprocess.run(["/bin/sh", "-c", f'cd "{ROOT}" && . moex/env.sh && printf %s "$STRATA_PYTHON"'],
                          text=True, capture_output=True, check=True).stdout


class LauncherArgsTest(unittest.TestCase):
    def forwarded(self, *args):
        lines = LAUNCHER.read_text().splitlines()
        exec_lines = [line for line in lines if line.startswith("exec systemd-run ")]
        self.assertEqual(len(exec_lines), 1)
        config_lines = [line for line in lines if line.startswith("cfg=")]
        self.assertEqual(len(config_lines), 1)
        with tempfile.TemporaryDirectory() as directory:
            recorder = Path(directory) / "systemd-run"
            recorder.write_text("#!/usr/bin/env python3\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
            recorder.chmod(0o755)
            env = dict(os.environ, PATH=directory + os.pathsep + os.environ["PATH"])
            # as the launcher runs them: from the checkout, after moex/env.sh (STRATA_X_ROOT, STRATA_PYTHON)
            script = f'cd "{ROOT}" && . moex/env.sh\n' + config_lines[0] + "\n" + exec_lines[0]
            result = subprocess.run(["/bin/sh", "-c", script, "launcher", *args],
                                    env=env, text=True, capture_output=True, check=True)
            return json.loads(result.stdout)

    def test_default_command_unchanged(self):
        self.assertEqual(self.forwarded(), [
            "--user", "--scope", "--quiet", "--collect", "--",
            env_python(),
            str(ROOT / "serve/server.py"), "--engine", "strata", "--config",
            str(ROOT / "strata-ernie55.json"), "--port", "1234",
        ])

    def test_custom_host_and_key(self):
        args = ["--host", "172.17.0.1", "--api-key", "secret with spaces;$literal"]
        self.assertEqual(self.forwarded(*args), self.forwarded() + args)

    def test_equals_form_and_help(self):
        for args in (["--host=127.0.0.1"], ["--help"], ["--port", "8080"]):
            with self.subTest(args=args):
                self.assertEqual(self.forwarded(*args), self.forwarded() + args)

    def test_later_port_wins(self):
        # the server's argparse (serve/server.py: --port type=int, --host, --api-key): the last --port is the one used
        ap = argparse.ArgumentParser()
        ap.add_argument("--port", type=int, default=8095)
        ap.add_argument("--host", default=None)
        ap.add_argument("--api-key", default="")
        cmd = self.forwarded("--host", "172.17.0.1", "--api-key", "k", "--port", "8080")
        server_args = cmd[cmd.index("--port"):]
        a = ap.parse_args(server_args)
        self.assertEqual((a.port, a.host, a.api_key), (8080, "172.17.0.1", "k"))


if __name__ == "__main__":
    unittest.main()
