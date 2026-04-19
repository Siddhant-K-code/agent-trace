"""Tests for issue #46: shadow AI detection (audit-tools)."""

from __future__ import annotations

import io
import os

import pytest


class TestShadowAI:
    def test_detect_claude_code_by_file(self, tmp_path):
        from agent_trace.shadow_ai import detect_ai_tools
        (tmp_path / "CLAUDE.md").write_text("# Claude instructions")
        os.system(f"git -C {tmp_path} init -q 2>/dev/null")
        report = detect_ai_tools(repo_path=str(tmp_path), since="1 day ago")
        tool_names = [d.tool_name for d in report.detections]
        assert "Claude Code" in tool_names

    def test_detect_cursor_by_file(self, tmp_path):
        from agent_trace.shadow_ai import detect_ai_tools
        (tmp_path / ".cursorrules").write_text("# Cursor rules")
        os.system(f"git -C {tmp_path} init -q 2>/dev/null")
        report = detect_ai_tools(repo_path=str(tmp_path), since="1 day ago")
        tool_names = [d.tool_name for d in report.detections]
        assert "Cursor" in tool_names

    def test_detect_copilot_by_file(self, tmp_path):
        from agent_trace.shadow_ai import detect_ai_tools
        gh = tmp_path / ".github"
        gh.mkdir()
        (gh / "copilot-instructions.md").write_text("# Copilot")
        os.system(f"git -C {tmp_path} init -q 2>/dev/null")
        report = detect_ai_tools(repo_path=str(tmp_path), since="1 day ago")
        tool_names = [d.tool_name for d in report.detections]
        assert "GitHub Copilot" in tool_names

    def test_unapproved_flagged(self, tmp_path):
        from agent_trace.shadow_ai import detect_ai_tools
        (tmp_path / "CLAUDE.md").write_text("# Claude instructions")
        os.system(f"git -C {tmp_path} init -q 2>/dev/null")
        report = detect_ai_tools(
            repo_path=str(tmp_path),
            since="1 day ago",
            approved=["Cursor"],
        )
        assert any("Claude Code" in s for s in report.unapproved_signals)

    def test_approved_not_flagged(self, tmp_path):
        from agent_trace.shadow_ai import detect_ai_tools
        (tmp_path / "CLAUDE.md").write_text("# Claude instructions")
        os.system(f"git -C {tmp_path} init -q 2>/dev/null")
        report = detect_ai_tools(
            repo_path=str(tmp_path),
            since="1 day ago",
            approved=["Claude Code"],
        )
        assert not any("Claude Code" in s for s in report.unapproved_signals)

    def test_format_output(self, tmp_path):
        from agent_trace.shadow_ai import detect_ai_tools, format_audit_tools
        (tmp_path / "CLAUDE.md").write_text("# Claude instructions")
        os.system(f"git -C {tmp_path} init -q 2>/dev/null")
        report = detect_ai_tools(repo_path=str(tmp_path), since="1 day ago")
        buf = io.StringIO()
        format_audit_tools(report, out=buf)
        output = buf.getvalue()
        assert "AI Tool Usage Report" in output
        assert "Claude Code" in output

    def test_no_detections_empty_repo(self, tmp_path):
        from agent_trace.shadow_ai import detect_ai_tools
        os.system(f"git -C {tmp_path} init -q 2>/dev/null")
        report = detect_ai_tools(repo_path=str(tmp_path), since="1 day ago")
        assert report.detections == []

    def test_confidence_confirmed_for_file_signal(self, tmp_path):
        from agent_trace.shadow_ai import detect_ai_tools
        (tmp_path / "CLAUDE.md").write_text("# Claude instructions")
        os.system(f"git -C {tmp_path} init -q 2>/dev/null")
        report = detect_ai_tools(repo_path=str(tmp_path), since="1 day ago")
        claude = next(d for d in report.detections if d.tool_name == "Claude Code")
        assert claude.confidence == "confirmed"

    def test_cli_has_audit_tools_command(self):
        from agent_trace.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["audit-tools", "--repo", ".", "--since", "30 days ago"])
        assert args.repo == "."
        assert args.since == "30 days ago"
