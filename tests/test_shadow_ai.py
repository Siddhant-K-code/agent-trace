"""Tests for issue #46: shadow AI detection (audit-tools)."""

from __future__ import annotations

import io
import os
import subprocess
import tempfile
import unittest


def _init_git(path: str) -> None:
    subprocess.run(["git", "init", "-q", path], capture_output=True)


class TestShadowAIDetection(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        _init_git(self._tmp)

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_detect_claude_code_by_file(self):
        from agent_trace.shadow_ai import detect_ai_tools
        open(os.path.join(self._tmp, "CLAUDE.md"), "w").close()
        report = detect_ai_tools(repo_path=self._tmp, since="1 day ago")
        tool_names = [d.tool_name for d in report.detections]
        self.assertIn("Claude Code", tool_names)

    def test_detect_cursor_by_file(self):
        from agent_trace.shadow_ai import detect_ai_tools
        open(os.path.join(self._tmp, ".cursorrules"), "w").close()
        report = detect_ai_tools(repo_path=self._tmp, since="1 day ago")
        tool_names = [d.tool_name for d in report.detections]
        self.assertIn("Cursor", tool_names)

    def test_detect_copilot_by_file(self):
        from agent_trace.shadow_ai import detect_ai_tools
        gh = os.path.join(self._tmp, ".github")
        os.makedirs(gh, exist_ok=True)
        open(os.path.join(gh, "copilot-instructions.md"), "w").close()
        report = detect_ai_tools(repo_path=self._tmp, since="1 day ago")
        tool_names = [d.tool_name for d in report.detections]
        self.assertIn("GitHub Copilot", tool_names)

    def test_unapproved_flagged(self):
        from agent_trace.shadow_ai import detect_ai_tools
        open(os.path.join(self._tmp, "CLAUDE.md"), "w").close()
        report = detect_ai_tools(repo_path=self._tmp, since="1 day ago", approved=["Cursor"])
        self.assertTrue(any("Claude Code" in s for s in report.unapproved_signals))

    def test_approved_not_flagged(self):
        from agent_trace.shadow_ai import detect_ai_tools
        open(os.path.join(self._tmp, "CLAUDE.md"), "w").close()
        report = detect_ai_tools(repo_path=self._tmp, since="1 day ago", approved=["Claude Code"])
        self.assertFalse(any("Claude Code" in s for s in report.unapproved_signals))

    def test_format_output_contains_header(self):
        from agent_trace.shadow_ai import detect_ai_tools, format_audit_tools
        open(os.path.join(self._tmp, "CLAUDE.md"), "w").close()
        report = detect_ai_tools(repo_path=self._tmp, since="1 day ago")
        buf = io.StringIO()
        format_audit_tools(report, out=buf)
        self.assertIn("AI Tool Usage Report", buf.getvalue())
        self.assertIn("Claude Code", buf.getvalue())

    def test_no_detections_empty_repo(self):
        from agent_trace.shadow_ai import detect_ai_tools
        report = detect_ai_tools(repo_path=self._tmp, since="1 day ago")
        self.assertEqual(report.detections, [])

    def test_confidence_confirmed_for_file_signal(self):
        from agent_trace.shadow_ai import detect_ai_tools
        open(os.path.join(self._tmp, "CLAUDE.md"), "w").close()
        report = detect_ai_tools(repo_path=self._tmp, since="1 day ago")
        claude = next(d for d in report.detections if d.tool_name == "Claude Code")
        self.assertEqual(claude.confidence, "confirmed")

    def test_cli_has_audit_tools_command(self):
        from agent_trace.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["audit-tools", "--repo", ".", "--since", "30 days ago"])
        self.assertEqual(args.repo, ".")
        self.assertEqual(args.since, "30 days ago")
