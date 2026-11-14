"""Tests for scripts/install-agent-strace.sh, the Action's CLI install step."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "install-agent-strace.sh"

# Stands in for python on PATH. Logs each pip call, fails the first
# SHIM_PIN_FAILURES exact-pin installs, and answers the version query.
PYTHON_SHIM = """#!/bin/sh
if [ "$1" = "-c" ]; then
  echo "${SHIM_VERSION:-1.2.3}"
  exit 0
fi
(IFS='|'; echo "$*") >> "$SHIM_LOG"
case "$*" in
  *agent-strace*==*)
    n=$(cat "$SHIM_COUNT" 2>/dev/null || echo 0)
    n=$((n + 1))
    echo "$n" > "$SHIM_COUNT"
    if [ "$n" -le "${SHIM_PIN_FAILURES:-0}" ]; then
      echo "ERROR: No matching distribution found" >&2
      exit 1
    fi
    ;;
esac
exit 0
"""

PIP = "-m|pip|install|--disable-pip-version-check|"


@unittest.skipUnless(shutil.which("bash"), "bash is required")
class TestInstallScript(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        bin_dir = self.tmp / "bin"
        bin_dir.mkdir()
        shim = bin_dir / "python"
        shim.write_text(PYTHON_SHIM)
        shim.chmod(0o755)
        self.log = self.tmp / "pip.log"
        self.env = {
            "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
            "SHIM_LOG": str(self.log),
            "SHIM_COUNT": str(self.tmp / "count"),
            "AGENT_STRACE_INSTALL_DELAY": "0",
        }

    def checkout(self, ref: str | None) -> Path:
        """Make an Action checkout as the runner lays it out for REF.

        None models `uses: ./`, where the Action path is the workspace.
        """
        if ref is None:
            path = self.tmp / "work" / "repo" / "repo"
        else:
            path = self.tmp / "work" / "_actions" / "Siddhant-K-code" / "agent-trace" / ref
        path.mkdir(parents=True)
        (path / "pyproject.toml").write_text('[project]\nname = "agent-strace"\n')
        return path

    def run_script(self, action_path: Path | str, **env: str) -> subprocess.CompletedProcess:
        full_env = dict(self.env, ACTION_PATH=str(action_path), **env)
        return subprocess.run(
            ["bash", str(SCRIPT)], env=full_env, capture_output=True, text=True, timeout=60,
        )

    def pip_calls(self) -> list[str]:
        if not self.log.exists():
            return []
        return self.log.read_text().splitlines()

    def test_release_tag_installs_exact_version(self):
        result = self.run_script(self.checkout("v0.108.1"), ACTION_REF="v0.108.1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.pip_calls(), [PIP + "agent-strace==0.108.1"])
        self.assertIn("installed from PyPI (Action ref v0.108.1)", result.stdout)

    def test_release_tag_with_extras(self):
        result = self.run_script(self.checkout("v0.108.1"), INSTALL_EXTRAS="openai, anthropic")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.pip_calls(), [PIP + "agent-strace[openai,anthropic]==0.108.1"])

    def test_tag_comes_from_path_not_stale_action_ref(self):
        # Inside composite steps github.action_ref can name a nested action's
        # ref (actions/runner#2473). The checkout path is what decides.
        result = self.run_script(self.checkout("v0.108.1"), ACTION_REF="v5")
        self.assertEqual(self.pip_calls(), [PIP + "agent-strace==0.108.1"])
        self.assertIn("github.action_ref: v5", result.stdout)

    def test_branch_installs_checkout(self):
        path = self.checkout("main")
        result = self.run_script(path, ACTION_REF="main")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.pip_calls(), [PIP + str(path)])
        self.assertIn("installed from the Action checkout (Action ref main)", result.stdout)

    def test_branch_with_slash_installs_checkout(self):
        path = self.checkout("feature/pin")
        result = self.run_script(path)
        self.assertEqual(self.pip_calls(), [PIP + str(path)])
        self.assertIn("Action ref from path: feature/pin", result.stdout)

    def test_sha_installs_checkout(self):
        sha = "e37e73d0c0ffee0123456789abcdef0123456789"
        path = self.checkout(sha)
        result = self.run_script(path, ACTION_REF=sha)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.pip_calls(), [PIP + str(path)])

    def test_non_release_tags_install_checkout(self):
        for ref in ("gha-v1", "v1", "v0.108", "v0.108.1-rc1"):
            with self.subTest(ref=ref):
                self.log.unlink(missing_ok=True)
                path = self.checkout(ref)
                self.run_script(path)
                self.assertEqual(self.pip_calls(), [PIP + str(path)])

    def test_empty_ref_installs_checkout(self):
        # uses: ./ leaves github.action_ref empty and points the Action path
        # at the workspace, outside _actions.
        path = self.checkout(None)
        result = self.run_script(path, ACTION_REF="")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.pip_calls(), [PIP + str(path)])
        self.assertIn("Action ref from path: <none>; github.action_ref: <empty>", result.stdout)

    def test_checkout_install_keeps_extras(self):
        path = self.checkout("main")
        self.run_script(path, INSTALL_EXTRAS="openai")
        self.assertEqual(self.pip_calls(), [PIP + f"{path}[openai]"])

    def test_override_wins_over_ref(self):
        for ref in ("v0.108.1", "main", None):
            with self.subTest(ref=ref):
                self.log.unlink(missing_ok=True)
                result = self.run_script(self.checkout(ref), AGENT_STRACE_VERSION="0.107.0")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(self.pip_calls(), [PIP + "agent-strace==0.107.0"])
                self.assertIn("installed from PyPI (agent-strace-version input)", result.stdout)

    def test_override_accepts_leading_v(self):
        self.run_script(self.checkout("main"), AGENT_STRACE_VERSION="v0.108.1")
        self.assertEqual(self.pip_calls(), [PIP + "agent-strace==0.108.1"])

    def test_override_does_not_fall_back(self):
        result = self.run_script(
            self.checkout("main"), AGENT_STRACE_VERSION="0.999.0",
            SHIM_PIN_FAILURES="99", AGENT_STRACE_INSTALL_ATTEMPTS="2",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.pip_calls(), [PIP + "agent-strace==0.999.0"] * 2)
        self.assertIn("not installable from PyPI after 2 attempts", result.stderr)

    def test_rejects_unsafe_inputs(self):
        cases = (
            {"AGENT_STRACE_VERSION": ">=0.1"},
            {"AGENT_STRACE_VERSION": "0.1; rm -rf /"},
            {"AGENT_STRACE_VERSION": "latest"},
            {"INSTALL_EXTRAS": "openai] other-pkg ["},
            {"INSTALL_EXTRAS": "$(id)"},
        )
        for env in cases:
            with self.subTest(env=env):
                self.log.unlink(missing_ok=True)
                path = self.tmp / "work" / "_actions" / "o" / "r" / "v1.0.0"
                path.mkdir(parents=True, exist_ok=True)
                result = self.run_script(path, **env)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(self.pip_calls(), [])
                self.assertIn("::error::", result.stderr)

    def test_pypi_lag_retries_then_succeeds(self):
        result = self.run_script(self.checkout("v0.108.1"), SHIM_PIN_FAILURES="2")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.pip_calls(), [PIP + "agent-strace==0.108.1"] * 3)
        self.assertIn("attempt 3/4", result.stdout)
        self.assertIn("installed from PyPI (Action ref v0.108.1)", result.stdout)

    def test_pypi_lag_falls_back_to_checkout(self):
        path = self.checkout("v0.108.1")
        result = self.run_script(path, SHIM_PIN_FAILURES="99", INSTALL_EXTRAS="openai")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self.pip_calls(),
            [PIP + "agent-strace[openai]==0.108.1"] * 4 + [PIP + f"{path}[openai]"],
        )
        self.assertIn("::warning::agent-strace==0.108.1 is not on PyPI yet", result.stdout)
        self.assertIn("installed from the Action checkout (Action ref v0.108.1)", result.stdout)

    def test_missing_action_path_fails(self):
        result = self.run_script("")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("GITHUB_ACTION_PATH is empty", result.stderr)
        self.assertEqual(self.pip_calls(), [])

    def test_never_falls_back_to_unpinned_pypi(self):
        self.run_script(self.checkout("main"))
        self.run_script(self.checkout("v0.108.1"), SHIM_PIN_FAILURES="99")
        for call in self.pip_calls():
            self.assertNotRegex(call, r"\|agent-strace(\[[^\]]*\])?$")


class TestActionWiring(unittest.TestCase):
    def test_install_step_uses_script_and_env(self):
        action = (ROOT / "action.yml").read_text(encoding="utf-8")
        self.assertIn("  agent-strace-version:\n", action)
        self.assertIn('run: bash "$GITHUB_ACTION_PATH/scripts/install-agent-strace.sh"', action)
        self.assertIn("ACTION_REF: ${{ github.action_ref }}", action)
        self.assertIn("INSTALL_EXTRAS: ${{ inputs.install-extras }}", action)
        self.assertIn("AGENT_STRACE_VERSION: ${{ inputs.agent-strace-version }}", action)
        self.assertNotRegex(action, r"pip install\s+\"?agent-strace")
        self.assertNotIn("\u2014", action)

    def test_no_expressions_inside_run_scripts(self):
        action = (ROOT / "action.yml").read_text(encoding="utf-8")
        for match in re.finditer(r"\n(\s+)run: (\|?)(.*)", action):
            indent, block, first = match.group(1), match.group(2), match.group(3)
            body = [first]
            if block:
                rest = action[match.end():].split("\n")[1:]
                for line in rest:
                    if line.strip() and not line.startswith(indent + "  "):
                        break
                    body.append(line)
            self.assertNotIn("${{", "\n".join(body))

    def test_script_is_executable(self):
        self.assertTrue(os.access(SCRIPT, os.X_OK))


if __name__ == "__main__":
    unittest.main()
