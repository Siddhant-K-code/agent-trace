"""Synthetic live setup verification tests. No provider process is started."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_trace import __version__
from agent_trace.cli import _codex_hooks_config, build_parser, cmd_setup
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore


_AGENTS = {
    "claude": "claude-code",
    "codex": "openai-codex",
}
_MODES = {
    "claude": "claude-code-hooks",
    "codex": "openai-codex-hooks",
}


def _args(**updates):
    values = {
        "cli": "codex",
        "verify": True,
        "live": True,
        "session": None,
        "trace_dir": "",
        "redact": False,
        "no_redact": False,
        "global_config": False,
    }
    values.update(updates)
    return argparse.Namespace(**values)


class LiveSetupVerifyTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.codex_home = self.root / "codex"
        self.trace_dir = self.root / "traces"
        self.codex_home.mkdir()
        self.trace_dir.mkdir()
        self.hooks_path = self.codex_home / "hooks.json"
        self.hooks_path.write_text(
            json.dumps(_codex_hooks_config(_args(live=False))),
            encoding="utf-8",
        )
        env = patch.dict(
            os.environ,
            {
                "CODEX_HOME": str(self.codex_home),
                "DO_NOT_TRACK": "1",
            },
            clear=False,
        )
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("CODEX_CONFIG_DIR", None)
        os.environ.pop("AGENT_STRACE_WORKSPACE", None)

    @contextlib.contextmanager
    def _installed_cli(self):
        completed = subprocess.CompletedProcess(
            args=["/bin/agent-strace", "--version"],
            returncode=0,
            stdout=f"agent-strace {__version__}\n",
            stderr="",
        )
        with patch(
            "agent_trace.setup_verify.shutil.which",
            return_value="/bin/agent-strace",
        ), patch(
            "agent_trace.setup_verify.subprocess.run",
            return_value=completed,
        ):
            yield

    def _run(self, **updates):
        out, err = io.StringIO(), io.StringIO()
        args = _args(trace_dir=str(self.trace_dir), **updates)
        with self._installed_cli(), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            status = cmd_setup(args)
        return status, out.getvalue(), err.getvalue()

    def _write_session(
        self,
        session_id,
        provider,
        started_at,
        *,
        duplicate_start=False,
        empty=False,
    ):
        agent = _AGENTS[provider]
        store = TraceStore(self.trace_dir)
        store.create_session(
            SessionMeta(
                session_id=session_id,
                started_at=started_at,
                ended_at=started_at + 10,
                agent_name=agent,
                command=f"{agent} (startup)",
            )
        )
        if empty:
            return store
        start = {
            "mode": _MODES[provider],
            "provider": provider,
            "source": "startup",
            "redaction": "on",
        }
        events = [
            TraceEvent(EventType.SESSION_START, timestamp=started_at, data=start),
            TraceEvent(
                EventType.USER_PROMPT,
                timestamp=started_at + 1,
                data={"prompt": "synthetic verification prompt", "provider": provider},
            ),
            TraceEvent(
                EventType.ASSISTANT_RESPONSE,
                timestamp=started_at + 2,
                data={"text": "synthetic response", "hook_event": "stop", "provider": provider},
            ),
            TraceEvent(EventType.SESSION_END, timestamp=started_at + 3, data={"reason": "other"}),
        ]
        if duplicate_start:
            events.insert(
                1,
                TraceEvent(
                    EventType.SESSION_START,
                    timestamp=started_at + 0.5,
                    data=start,
                ),
            )
        for event in events:
            store.append_event(session_id, event)
        return store

    def test_live_verify_selects_newest_matching_provider_and_reports_acceptance_evidence(self):
        self._write_session("codex-old-session", "codex", 100)
        self._write_session("codex-new-session", "codex", 200)
        self._write_session("claude-newest-session", "claude", 300)

        status, out, err = self._run()

        self.assertEqual((status, err), (0, ""))
        for expected in (
            "Codex static hook setup verified.",
            f"hooks:      {self.hooks_path}",
            "executable: /bin/agent-strace",
            "Capture evidence: observed",
            "Selection: newest matching recorded session",
            "Session: codex-new-session",
            "Provider: codex",
            "Capture method: cli_hooks",
            "Evidence health: partial",
            "Observed event types: session_start 1, session_end 1, user_prompt 1, "
            "assistant_response 1",
            "Start marker: observed",
            "End marker: observed",
            "Redaction: on",
            "Capture matrix gaps (",
            'SessionEnd always reports reason "other"',
        ):
            self.assertIn(expected, out)
        self.assertNotIn("claude-newest-session", out)

    def test_explicit_session_prefix_is_used(self):
        self._write_session("codex-selected-session", "codex", 100)
        self._write_session("codex-newer-session", "codex", 200)

        status, out, err = self._run(session="codex-selected")

        self.assertEqual((status, err), (0, ""))
        self.assertIn("Selection: requested session codex-selected", out)
        self.assertIn("Session: codex-selected-session", out)
        self.assertNotIn("Session: codex-newer-session", out)

    def test_no_matching_session_distinguishes_installed_config_from_observed_events(self):
        status, out, err = self._run()

        self.assertEqual(status, 1)
        self.assertIn("Codex static hook setup verified.", out)
        self.assertIn("Live capture verification failed:", err)
        self.assertIn("No recorded OpenAI Codex session", err)
        self.assertIn("configuration is installed", err)
        self.assertIn("trust the AgentTrace commands in /hooks", err)
        self.assertIn("--session SESSION_ID", err)

    def test_stale_setup_fails_before_recorded_evidence_is_accepted(self):
        config = json.loads(self.hooks_path.read_text(encoding="utf-8"))
        config["hooks"]["Stop"][0]["hooks"][0]["command"] = (
            "agent-strace hook --provider codex stale-stop"
        )
        self.hooks_path.write_text(json.dumps(config), encoding="utf-8")
        self._write_session("codex-valid-session", "codex", 100)

        status, out, err = self._run()

        self.assertEqual(status, 1)
        self.assertEqual(out, "")
        self.assertIn("Setup verification failed:", err)
        self.assertIn("stale or unexpected AgentTrace hook command", err)
        self.assertNotIn("Capture evidence: observed", err)

    def test_empty_evidence_is_an_actionable_failure(self):
        self._write_session("codex-empty-session", "codex", 100, empty=True)

        status, out, err = self._run(session="codex-empty-session")

        self.assertEqual(status, 1)
        self.assertIn("Codex static hook setup verified.", out)
        self.assertIn("contains no captured events", err)
        self.assertIn("configuration is installed", err)
        self.assertIn("no OpenAI Codex events were observed", err)

    def test_invalid_evidence_prints_health_details_and_fails(self):
        self._write_session(
            "codex-invalid-session",
            "codex",
            100,
            duplicate_start=True,
        )

        status, out, err = self._run(session="codex-invalid-session")

        self.assertEqual(status, 1)
        self.assertIn("Evidence health: invalid", out)
        self.assertIn("duplicate_session_start", out)
        self.assertIn("has invalid evidence", err)
        self.assertIn("--session SESSION_ID", err)

    def test_unreadable_evidence_fails_instead_of_falling_back(self):
        self._write_session("codex-corrupt-session", "codex", 200)
        self._write_session("codex-older-session", "codex", 100)
        events = self.trace_dir / "codex-corrupt-session" / "events.ndjson"
        events.write_text("{not-json}\n", encoding="utf-8")

        status, out, err = self._run()

        self.assertEqual(status, 1)
        self.assertIn("Codex static hook setup verified.", out)
        self.assertIn("Newest OpenAI Codex capture candidate codex-corrupt-session", err)
        self.assertIn("could not be read", err)
        self.assertNotIn("Session: codex-older-session", out)

    def test_explicit_session_from_another_provider_fails(self):
        self._write_session("claude-session", "claude", 100)

        status, _, err = self._run(session="claude-session")

        self.assertEqual(status, 1)
        self.assertIn("does not match requested provider 'codex'", err)
        self.assertIn("contains 'claude' hook evidence", err)

    def test_live_verify_rejects_unsupported_provider_without_writes(self):
        gemini_home = self.root / "gemini"
        with patch.dict(os.environ, {"GEMINI_CLI_HOME": str(gemini_home)}):
            status, out, err = self._run(cli="gemini")

        self.assertEqual((status, out), (2, ""))
        self.assertIn("supports only --cli codex or --cli claude", err)
        self.assertFalse(gemini_home.exists())

    def test_live_flags_require_the_full_verify_flow(self):
        status, out, err = self._run(verify=False)
        self.assertEqual((status, out), (2, ""))
        self.assertEqual(err, "setup --live requires --verify.\n")

        status, out, err = self._run(live=False, session="codex")
        self.assertEqual((status, out), (2, ""))
        self.assertEqual(err, "setup --session requires --verify --live.\n")

    def test_parser_accepts_live_verification_options(self):
        parsed = build_parser().parse_args(
            [
                "--trace-dir",
                "synthetic-traces",
                "setup",
                "--cli",
                "codex",
                "--verify",
                "--live",
                "--session",
                "abc123",
            ]
        )
        self.assertTrue(parsed.verify)
        self.assertTrue(parsed.live)
        self.assertEqual(parsed.session, "abc123")
        self.assertEqual(parsed.trace_dir, "synthetic-traces")


if __name__ == "__main__":
    unittest.main()
