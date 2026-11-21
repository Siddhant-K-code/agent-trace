"""Clean-environment tests for documented setup and rollback commands."""

from __future__ import annotations

import os
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:  # keeps product telemetry off in this process and its children
    import _no_telemetry
except ImportError:  # pragma: no cover - run as a package module
    from tests import _no_telemetry


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


@unittest.skipUnless(os.name == "posix", "documented rollback commands use cp and rm")
class SetupRollbackTests(unittest.TestCase):
    def setUp(self):
        for command in ("sh", "cp", "rm"):
            if shutil.which(command) is None:
                self.skipTest(f"{command} is required to exercise the documented command")

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.home = self.root / "home"
        self.bin_dir = self.root / "bin"
        self.home.mkdir()
        self.bin_dir.mkdir()

        executable = self.bin_dir / "agent-strace"
        executable.write_text(
            "#!/bin/sh\n"
            f"exec {shlex.quote(sys.executable)} -m agent_trace.entry \"$@\"\n",
            encoding="utf-8",
        )
        executable.chmod(0o755)

        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("AGENT_TRACE_", "AGENT_STRACE_"))
            and key not in {
                "CLAUDE_CONFIG_DIR",
                "CODEX_CONFIG_DIR",
                "CODEX_HOME",
                "COPILOT_HOME",
                "CURSOR_CONFIG_DIR",
                "GEMINI_CLI_HOME",
                "GEMINI_CONFIG_DIR",
            }
        }
        env.update(
            {
                "HOME": str(self.home),
                "USERPROFILE": str(self.home),
                "PATH": os.pathsep.join((str(self.bin_dir), env.get("PATH", ""))),
                "PYTHONPATH": os.pathsep.join(
                    item for item in (str(SRC), env.get("PYTHONPATH", "")) if item
                ),
                "PYTHONIOENCODING": "utf-8",
            }
        )
        self.env = _no_telemetry.without_telemetry(env)

    def _run(self, command: str) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            ["sh", "-c", command],
            cwd=self.root,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result

    def test_codex_existing_file_is_restored_byte_for_byte(self):
        codex_home = self.home / ".codex"
        codex_home.mkdir()
        hooks = codex_home / "hooks.json"
        backup = codex_home / "hooks.json.agent-strace.bak"
        unrelated = codex_home / "config.toml"
        original = b"\x00pre-agent-strace\r\n\xff"
        unrelated_bytes = b"[features]\nhooks = true\n"
        hooks.write_bytes(original)
        hooks.chmod(0o640)
        unrelated.write_bytes(unrelated_bytes)
        original_mode = stat.S_IMODE(hooks.stat().st_mode)

        self._run("agent-strace setup --cli codex")

        self.assertNotEqual(hooks.read_bytes(), original)
        self.assertEqual(backup.read_bytes(), original)
        self.assertEqual(unrelated.read_bytes(), unrelated_bytes)

        self._run("cp ~/.codex/hooks.json.agent-strace.bak ~/.codex/hooks.json")

        self.assertEqual(hooks.read_bytes(), original)
        self.assertEqual(stat.S_IMODE(hooks.stat().st_mode), original_mode)
        self.assertEqual(unrelated.read_bytes(), unrelated_bytes)

    def test_codex_no_prior_file_is_removed_cleanly(self):
        codex_home = self.home / ".codex"
        codex_home.mkdir()
        unrelated = codex_home / "config.toml"
        unrelated_bytes = b"[features]\nhooks = true\n"
        unrelated.write_bytes(unrelated_bytes)

        self._run("agent-strace setup --cli codex")

        hooks = codex_home / "hooks.json"
        self.assertTrue(hooks.is_file())
        self.assertFalse((codex_home / "hooks.json.agent-strace.bak").exists())

        self._run("rm ~/.codex/hooks.json")

        self.assertFalse(hooks.exists())
        self.assertEqual(unrelated.read_bytes(), unrelated_bytes)
        self.assertEqual(sorted(path.name for path in codex_home.iterdir()), ["config.toml"])

    def test_copilot_dedicated_file_is_removed_without_touching_other_hooks(self):
        hooks_dir = self.home / ".copilot" / "hooks"
        hooks_dir.mkdir(parents=True)
        unrelated = hooks_dir / "team-policy.json"
        settings = self.home / ".copilot" / "config.json"
        unrelated_bytes = b'{"hooks":{"sessionStart":[]}}\n'
        settings_bytes = b'{"theme":"dark"}\n'
        unrelated.write_bytes(unrelated_bytes)
        settings.write_bytes(settings_bytes)

        self._run("agent-strace setup --cli copilot")

        generated = hooks_dir / "agent-strace.json"
        self.assertTrue(generated.is_file())
        self.assertEqual(unrelated.read_bytes(), unrelated_bytes)
        self.assertEqual(settings.read_bytes(), settings_bytes)

        self._run("rm ~/.copilot/hooks/agent-strace.json")

        self.assertFalse(generated.exists())
        self.assertEqual(unrelated.read_bytes(), unrelated_bytes)
        self.assertEqual(settings.read_bytes(), settings_bytes)
        self.assertEqual(sorted(path.name for path in hooks_dir.iterdir()), ["team-policy.json"])


if __name__ == "__main__":
    unittest.main()
