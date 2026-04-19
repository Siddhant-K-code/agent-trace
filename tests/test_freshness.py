"""Tests for issue #42: context freshness check."""

from __future__ import annotations

import io
import os
import tempfile
import unittest

from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore


def _make_store(tmp_dir: str) -> TraceStore:
    return TraceStore(os.path.join(tmp_dir, "traces"))


class TestFreshnessScope(unittest.TestCase):
    def test_parse_scope_no_file(self):
        from agent_trace.freshness import _parse_scope_from_agents_md
        # Should return empty list when no CLAUDE.md/AGENTS.md present
        import os
        orig = os.getcwd()
        try:
            os.chdir(tempfile.mkdtemp())
            result = _parse_scope_from_agents_md()
            self.assertIsInstance(result, list)
        finally:
            os.chdir(orig)

    def test_parse_scope_from_claude_md(self):
        from agent_trace.freshness import _parse_scope_from_agents_md
        import os
        tmp = tempfile.mkdtemp()
        orig = os.getcwd()
        try:
            os.chdir(tmp)
            with open("CLAUDE.md", "w") as f:
                f.write("# Instructions\n\nScope:\n- src/**\n- tests/**\n")
            result = _parse_scope_from_agents_md()
            self.assertIn("src/**", result)
        finally:
            os.chdir(orig)
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)


class TestFreshnessAnalysis(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_analyse_freshness_no_sessions(self):
        from agent_trace.freshness import analyse_freshness
        store = _make_store(self._tmp)
        report = analyse_freshness(store, repo=self._tmp)
        self.assertIsNone(report.last_session_ts)
        self.assertEqual(report.files_changed_total, 0)
        self.assertEqual(report.freshness_score, 100)

    def test_analyse_freshness_with_session(self):
        from agent_trace.freshness import analyse_freshness
        store = _make_store(self._tmp)
        meta = SessionMeta(agent_name="test")
        store.create_session(meta)
        report = analyse_freshness(store, repo=self._tmp)
        self.assertIsNotNone(report.last_session_ts)
        self.assertIsInstance(report.freshness_score, int)
        self.assertGreaterEqual(report.freshness_score, 0)
        self.assertLessEqual(report.freshness_score, 100)

    def test_freshness_score_100_when_no_changes(self):
        from agent_trace.freshness import analyse_freshness
        store = _make_store(self._tmp)
        # No git repo → no changes → score 100
        report = analyse_freshness(store, repo=self._tmp)
        self.assertEqual(report.freshness_score, 100)

    def test_format_freshness_output(self):
        from agent_trace.freshness import analyse_freshness, format_freshness
        store = _make_store(self._tmp)
        report = analyse_freshness(store, repo=self._tmp)
        buf = io.StringIO()
        format_freshness(report, out=buf)
        output = buf.getvalue()
        self.assertIn("Context Freshness Report", output)
        self.assertIn("Freshness score", output)

    def test_format_freshness_fully_fresh(self):
        from agent_trace.freshness import analyse_freshness, format_freshness
        store = _make_store(self._tmp)
        report = analyse_freshness(store, repo=self._tmp)
        buf = io.StringIO()
        format_freshness(report, out=buf)
        self.assertIn("fully fresh", buf.getvalue())

    def test_cli_has_freshness_command(self):
        from agent_trace.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["freshness", "--since", "2026-01-01", "--scope", "src/**"])
        self.assertEqual(args.since, "2026-01-01")
        self.assertEqual(args.scope, "src/**")


