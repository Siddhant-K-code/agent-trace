"""Tests for session attribution (issue #22)."""

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace.attribution import (
    Attribution,
    _detect_agent_provider,
    _detect_os_user,
    collect_attribution,
    format_attribution,
)
from agent_trace.hooks import hook_main
from agent_trace.store import TraceStore


def _git_repo(path: Path, branch: str = "feature/demo") -> None:
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=path, check=True, capture_output=True)

    git("init", "-q")
    git("checkout", "-q", "-b", branch)
    git("remote", "add", "origin", "https://example.com/acme/demo.git")
    git("-c", "user.name=t", "-c", "user.email=t@example.com",
        "commit", "-q", "--allow-empty", "-m", "init")


class _Chdir:
    def __init__(self, path: str):
        self.path = path

    def __enter__(self):
        self.old = os.getcwd()
        os.chdir(self.path)

    def __exit__(self, *exc):
        os.chdir(self.old)


class TestAttribution(unittest.TestCase):
    def test_collect_returns_attribution(self):
        attr = collect_attribution()
        self.assertIsInstance(attr, Attribution)
        self.assertIsInstance(attr.pid, int)
        self.assertGreater(attr.pid, 0)

    def test_working_dir_set(self):
        attr = collect_attribution()
        self.assertTrue(attr.working_dir)
        self.assertTrue(os.path.isdir(attr.working_dir))

    def test_to_dict_round_trip(self):
        attr = collect_attribution()
        d = attr.to_dict()
        self.assertIsInstance(d, dict)
        restored = Attribution.from_dict(d)
        self.assertEqual(restored.pid, attr.pid)
        self.assertEqual(restored.working_dir, attr.working_dir)

    def test_format_attribution_no_crash(self):
        attr = Attribution(os_user="alice", hostname="box", agent_provider="claude-code")
        result = format_attribution(attr)
        self.assertIn("alice", result)
        self.assertIn("claude-code", result)

    def test_format_attribution_empty(self):
        attr = Attribution()
        result = format_attribution(attr)
        self.assertIsInstance(result, str)

    def test_detect_agent_provider_unknown_by_default(self):
        # In a clean test env without agent env vars, should return "unknown"
        # (unless running inside an actual agent)
        provider, version = _detect_agent_provider()
        self.assertIsInstance(provider, str)
        self.assertIsInstance(version, str)

    def test_detect_agent_provider_claude_env(self):
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test"
        try:
            provider, _ = _detect_agent_provider()
            self.assertEqual(provider, "claude-code")
        finally:
            del os.environ["ANTHROPIC_API_KEY"]

    def test_os_user_returns_string(self):
        user = _detect_os_user()
        self.assertIsInstance(user, str)


class TestPayloadWorkingDir(unittest.TestCase):
    """Hooks may run outside the workspace; the payload cwd is the workspace."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.repo = root / "repo"
        self.repo.mkdir()
        _git_repo(self.repo)
        # Stand-in for ~/.copilot/installed-plugins/agent-trace/agent-strace.
        self.plugin_dir = root / "installed-plugins" / "agent-trace" / "agent-strace"
        self.plugin_dir.mkdir(parents=True)
        self._env = patch.dict(os.environ, {}, clear=False)
        self._env.start()
        for key in ("AGENT_TRACE_DIR", "AGENT_TRACE_COPILOT_SESSION_ID",
                    "AGENT_TRACE_CLAUDE_SESSION_ID", "AGENT_STRACE_ENDPOINT"):
            os.environ.pop(key, None)

    def tearDown(self):
        self._env.stop()
        self._tmp.cleanup()

    def test_payload_cwd_drives_working_dir_and_git_context(self):
        with _Chdir(str(self.plugin_dir)):
            attr = collect_attribution(cwd=str(self.repo))
        self.assertEqual(attr.working_dir, str(self.repo))
        self.assertEqual(attr.git_branch, "feature/demo")
        self.assertEqual(attr.git_repo, "https://example.com/acme/demo.git")
        self.assertTrue(attr.git_commit)

    def test_unusable_payload_cwd_falls_back_to_process_cwd(self):
        missing = str(self.repo / "missing")
        for cwd in (None, "", "relative/path", missing, 42):
            with self.subTest(cwd=cwd), _Chdir(str(self.repo)):
                attr = collect_attribution(cwd=cwd)
                self.assertEqual(os.path.realpath(attr.working_dir),
                                 os.path.realpath(str(self.repo)))
                self.assertEqual(attr.git_branch, "feature/demo")

    def test_hook_provider_overrides_environment_guess(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            attr = collect_attribution(hook_provider="copilot")
        self.assertEqual(attr.agent_provider, "copilot")

    def test_unknown_hook_provider_keeps_environment_guess(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            attr = collect_attribution(hook_provider="nope")
        self.assertEqual(attr.agent_provider, "claude-code")

    def _hook(self, provider: str, event: str, payload: dict) -> None:
        with _Chdir(str(self.plugin_dir)), \
                patch.object(sys, "stdin", io.StringIO(json.dumps(payload))):
            hook_main(["--provider", provider, event])

    def test_copilot_plugin_hook_records_workspace_attribution(self):
        # Copilot CLI runs plugin hooks with cwd set to the plugin directory.
        session = "3aa69885-40ee-49ed-9eca-e23a7f421f39"
        base = {"sessionId": session, "cwd": str(self.repo)}
        self._hook("copilot", "sessionStart", {**base, "source": "new"})
        self._hook("copilot", "sessionEnd", {**base, "reason": "complete"})

        store = TraceStore(str(self.repo / ".agent-traces"))
        attr = store.load_meta(session[:16]).attribution
        self.assertEqual(attr["working_dir"], str(self.repo))
        self.assertEqual(attr["git_branch"], "feature/demo")
        self.assertEqual(attr["git_repo"], "https://example.com/acme/demo.git")
        self.assertTrue(attr.get("git_commit"))
        self.assertEqual(attr["agent_provider"], "copilot")
        self.assertFalse((self.plugin_dir / ".agent-traces").exists())

    def test_implicit_session_meta_uses_payload_cwd(self):
        # Copilot subagents fire prompt hooks without sessionStart.
        session = "subagentsession-0001"
        self._hook("copilot", "userPromptSubmitted",
                   {"sessionId": session, "cwd": str(self.repo), "prompt": "hi"})

        store = TraceStore(str(self.repo / ".agent-traces"))
        attr = store.load_meta(session[:16]).attribution
        self.assertEqual(attr["working_dir"], str(self.repo))
        self.assertEqual(attr["git_branch"], "feature/demo")
        self.assertEqual(attr["agent_provider"], "copilot")

    def test_claude_hook_uses_payload_cwd(self):
        session = "claude-session-0001"
        self._hook("claude", "session-start",
                   {"session_id": session, "cwd": str(self.repo), "source": "startup"})

        store = TraceStore(str(self.repo / ".agent-traces"))
        attr = store.load_meta(session[:16]).attribution
        self.assertEqual(attr["working_dir"], str(self.repo))
        self.assertEqual(attr["git_branch"], "feature/demo")
        self.assertEqual(attr["agent_provider"], "claude-code")


if __name__ == "__main__":
    unittest.main()
