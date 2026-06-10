import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace.cli import build_parser
from agent_trace.cognitive_debt import (
    SessionDebt,
    build_cognitive_debt_report,
    modified_file_lines,
)
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore


def _git(cwd, *args):
    subprocess.run(["git", "-C", cwd, *args], check=True, capture_output=True, text=True)


def _make_store_session(store, cwd, branch="main", author="dev@example.com"):
    meta = SessionMeta(
        agent_name="codex",
        attribution={
            "working_dir": cwd,
            "git_branch": branch,
            "git_author": author,
        },
    )
    meta.started_at = time.time() - 10
    store.create_session(meta)
    return meta


class TestCognitiveDebtScoring(unittest.TestCase):
    def test_debt_score_uses_unreviewed_agent_lines(self):
        session = SessionDebt(
            session_id="s1",
            started_at=0,
            branch="main",
            author="dev@example.com",
            agent_written_lines=100,
            human_reviewed_lines=40,
            files=[],
        )

        self.assertAlmostEqual(session.debt_score, 0.60)

    def test_modified_file_lines_counts_tool_payloads(self):
        event = TraceEvent(
            event_type=EventType.TOOL_CALL,
            data={
                "tool_name": "Write",
                "arguments": {
                    "file_path": "src/app.py",
                    "content": "a\nb\nc\n",
                },
            },
        )

        self.assertEqual(modified_file_lines([event]), {"src/app.py": 3})

    def test_git_unavailable_fallback_reports_zero_review(self):
        tmp = tempfile.mkdtemp()
        store = TraceStore(Path(tmp) / "traces")
        meta = _make_store_session(store, tmp)
        store.append_event(
            meta.session_id,
            TraceEvent(
                event_type=EventType.FILE_WRITE,
                data={"path": "src/app.py", "content": "a\nb\nc"},
            ),
        )

        report = build_cognitive_debt_report(
            store,
            window_start=time.time() - 60,
            window_end=time.time() + 60,
            threshold=0.7,
        )

        self.assertFalse(report.git_available)
        self.assertEqual(report.sessions[0].agent_written_lines, 3)
        self.assertEqual(report.sessions[0].human_reviewed_lines, 0)
        self.assertEqual(len(report.zero_review_files), 1)

    def test_local_git_review_signal_adds_reviewed_lines(self):
        tmp = tempfile.mkdtemp()
        repo = Path(tmp) / "repo"
        repo.mkdir()
        _git(str(repo), "init")
        _git(str(repo), "config", "user.email", "reviewer@example.com")
        _git(str(repo), "config", "user.name", "Reviewer")

        store = TraceStore(Path(tmp) / "traces")
        meta = _make_store_session(store, str(repo), branch="feat/reviewed")
        (repo / "app.py").write_text("print('hello')\n")
        store.append_event(
            meta.session_id,
            TraceEvent(
                event_type=EventType.TOOL_CALL,
                data={
                    "tool_name": "Write",
                    "arguments": {
                        "file_path": "app.py",
                        "content": "\n".join(str(i) for i in range(10)),
                    },
                },
            ),
        )

        _git(str(repo), "add", "app.py")
        _git(str(repo), "commit", "-m", "Merge pull request #12 from feat/reviewed")

        report = build_cognitive_debt_report(
            store,
            window_start=time.time() - 60,
            window_end=time.time() + 60,
            group_by="branch",
            threshold=0.7,
        )
        session = report.sessions[0]

        self.assertTrue(report.git_available)
        self.assertGreater(session.human_reviewed_lines, 0)
        self.assertLess(session.debt_score, 1.0)
        self.assertIn("pull_request_merge", session.files[0].review_signals)
        self.assertIn("feat/reviewed", report.rows)

    def test_github_token_can_enrich_review_signals(self):
        tmp = tempfile.mkdtemp()
        repo = Path(tmp) / "repo"
        repo.mkdir()
        _git(str(repo), "init")
        _git(str(repo), "config", "user.email", "reviewer@example.com")
        _git(str(repo), "config", "user.name", "Reviewer")

        store = TraceStore(Path(tmp) / "traces")
        meta = _make_store_session(store, str(repo), branch="feat/github-reviewed")
        store.append_event(
            meta.session_id,
            TraceEvent(
                event_type=EventType.TOOL_CALL,
                data={
                    "tool_name": "Write",
                    "arguments": {
                        "file_path": "app.py",
                        "content": "\n".join(str(i) for i in range(8)),
                    },
                },
            ),
        )

        with patch(
            "agent_trace.cognitive_debt._github_review_for_file",
            return_value=(8, ["github_line_comments"]),
        ):
            report = build_cognitive_debt_report(
                store,
                window_start=time.time() - 60,
                window_end=time.time() + 60,
                github_token="token",
            )

        session = report.sessions[0]
        self.assertEqual(session.human_reviewed_lines, 8)
        self.assertEqual(session.debt_score, 0.0)
        self.assertIn("github_line_comments", session.files[0].review_signals)

    def test_parser_registers_command(self):
        parser = build_parser()
        args = parser.parse_args(["cognitive-debt", "--by", "branch", "--threshold", "0.8"])
        self.assertEqual(args.command, "cognitive-debt")
        self.assertEqual(args.by, "branch")
        self.assertAlmostEqual(args.threshold, 0.8)


if __name__ == "__main__":
    unittest.main()
