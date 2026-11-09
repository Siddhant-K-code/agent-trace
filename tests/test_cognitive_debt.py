import argparse
import json
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
    CognitiveDebtInputError,
    SessionDebt,
    build_cognitive_debt_report,
    cmd_cognitive_debt,
    format_cognitive_debt_badge,
    format_cognitive_debt_json,
    modified_file_lines,
)
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore


def _git(cwd, *args):
    subprocess.run(["git", "-C", cwd, *args], check=True, capture_output=True, text=True)


def _make_store_session(
    store,
    cwd,
    branch="main",
    author="dev@example.com",
    *,
    started_at=None,
):
    meta = SessionMeta(
        agent_name="codex",
        attribution={
            "working_dir": cwd,
            "git_branch": branch,
            "git_author": author,
        },
    )
    meta.started_at = started_at if started_at is not None else time.time() - 10
    store.create_session(meta)
    return meta


def _append_write(store, meta, path, lines):
    store.append_event(
        meta.session_id,
        TraceEvent(
            event_type=EventType.FILE_WRITE,
            data={"path": path, "content": "\n".join(str(index) for index in range(lines))},
        ),
    )


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

    def test_clean_report_has_versioned_json_and_green_badge(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = TraceStore(Path(tmp) / "traces")
            meta = _make_store_session(store, tmp)
            _append_write(store, meta, "src/app.py", 4)

            with patch(
                "agent_trace.cognitive_debt._review_for_file",
                return_value=(4, ["reviewed_by_trailer"], True),
            ):
                report = build_cognitive_debt_report(
                    store,
                    window_start=time.time() - 60,
                    window_end=time.time() + 60,
                )

        payload = json.loads(format_cognitive_debt_json(report))
        badge = json.loads(format_cognitive_debt_badge(report))

        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["status"], "clean")
        self.assertEqual(payload["coverage"]["reviewed_lines"], 4)
        self.assertEqual(payload["coverage"]["unreviewed_lines"], 0)
        self.assertEqual(payload["coverage"]["unknown_review_lines"], 0)
        self.assertEqual(payload["scope"]["aggregate"], "repository")
        self.assertFalse(payload["interpretation"]["measures_code_quality"])
        self.assertFalse(payload["interpretation"]["claims_causation"])
        self.assertFalse(payload["interpretation"]["ranks_individuals"])
        self.assertNotIn("author", payload["workflows"][0])
        self.assertEqual(badge["schemaVersion"], 1)
        self.assertEqual(badge["message"], "100% reviewed, 0% unreviewed")
        self.assertEqual(badge["color"], "brightgreen")

    def test_partial_report_expresses_reviewed_and_unreviewed_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = TraceStore(Path(tmp) / "traces")
            meta = _make_store_session(store, tmp)
            _append_write(store, meta, "src/app.py", 10)

            with patch(
                "agent_trace.cognitive_debt._review_for_file",
                return_value=(6, ["review_reference"], True),
            ):
                report = build_cognitive_debt_report(
                    store,
                    window_start=time.time() - 60,
                    window_end=time.time() + 60,
                    threshold=0.7,
                )

        payload = json.loads(format_cognitive_debt_json(report))
        badge = json.loads(format_cognitive_debt_badge(report))

        self.assertEqual(report.status, "partial")
        self.assertEqual(payload["coverage"]["reviewed_percent"], 60.0)
        self.assertEqual(payload["coverage"]["unreviewed_percent"], 40.0)
        self.assertEqual(payload["coverage"]["unreviewed_ratio"], 0.4)
        self.assertEqual(badge["message"], "60% reviewed, 40% unreviewed")
        self.assertEqual(badge["color"], "yellow")

    def test_unknown_review_and_authorship_do_not_look_clean(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = TraceStore(Path(tmp) / "traces")
            meta = _make_store_session(store, tmp)
            store.append_event(
                meta.session_id,
                TraceEvent(
                    event_type=EventType.FILE_WRITE,
                    data={"content": "a\nb\nc"},
                ),
            )

            report = build_cognitive_debt_report(
                store,
                window_start=time.time() - 60,
                window_end=time.time() + 60,
            )

        payload = json.loads(format_cognitive_debt_json(report))
        badge = json.loads(format_cognitive_debt_badge(report))

        self.assertEqual(report.status, "unknown")
        self.assertIsNone(report.debt_score)
        self.assertEqual(payload["coverage"]["authorship_state"], "unknown")
        self.assertEqual(payload["coverage"]["review_state"], "unknown")
        self.assertEqual(payload["coverage"]["unknown_authorship_lines"], 3)
        self.assertEqual(payload["coverage"]["unknown_review_lines"], 3)
        self.assertEqual(badge["color"], "lightgrey")
        self.assertIn("unknown", badge["message"])

    def test_git_unavailable_marks_review_state_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = TraceStore(Path(tmp) / "traces")
            meta = _make_store_session(store, tmp)
            _append_write(store, meta, "src/app.py", 3)

            report = build_cognitive_debt_report(
                store,
                window_start=time.time() - 60,
                window_end=time.time() + 60,
                threshold=0.7,
            )

        self.assertFalse(report.git_available)
        self.assertEqual(report.total_reviewed_lines, 0)
        self.assertEqual(report.total_unreviewed_lines, 0)
        self.assertEqual(report.total_unknown_review_lines, 3)
        self.assertEqual(report.status, "unknown")
        self.assertEqual(len(report.zero_review_files), 0)
        self.assertEqual(len(report.unknown_review_files), 1)

    def test_malformed_trace_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_dir = Path(tmp) / "traces"
            store = TraceStore(trace_dir)
            meta = _make_store_session(store, tmp)
            events_path = trace_dir / meta.session_id / "events.ndjson"
            events_path.write_text("{not-json}\n", encoding="utf-8")

            with self.assertRaisesRegex(
                CognitiveDebtInputError,
                f"session {meta.session_id} trace could not be read",
            ):
                build_cognitive_debt_report(
                    store,
                    window_start=time.time() - 60,
                    window_end=time.time() + 60,
                    session_id=meta.session_id,
                )

    def test_malformed_review_history_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            _git(str(repo), "init")
            store = TraceStore(Path(tmp) / "traces")
            meta = _make_store_session(store, str(repo))
            _append_write(store, meta, "app.py", 2)

            with patch(
                "agent_trace.cognitive_debt._review_log",
                return_value="not-a-valid-review-record",
            ):
                with self.assertRaisesRegex(
                    CognitiveDebtInputError,
                    "local review history was malformed",
                ):
                    build_cognitive_debt_report(
                        store,
                        window_start=time.time() - 60,
                        window_end=time.time() + 60,
                    )

    def test_multi_session_report_aggregates_at_repository_and_workflow_levels(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            _git(str(repo), "init")
            store = TraceStore(Path(tmp) / "traces")
            first = _make_store_session(
                store, str(repo), branch="first", started_at=time.time() - 20
            )
            second = _make_store_session(
                store, str(repo), branch="second", started_at=time.time() - 10
            )
            _append_write(store, first, "first.py", 4)
            _append_write(store, second, "second.py", 6)

            reviews = {
                "first.py": (4, ["reviewed_by_trailer"], True),
                "second.py": (3, ["review_reference"], True),
            }
            with patch(
                "agent_trace.cognitive_debt._review_for_file",
                side_effect=lambda _cwd, path, _since, _lines: reviews[path],
            ):
                report = build_cognitive_debt_report(
                    store,
                    window_start=time.time() - 60,
                    window_end=time.time() + 60,
                    threshold=0.5,
                )

        payload = json.loads(format_cognitive_debt_json(report))

        self.assertEqual(len(payload["workflows"]), 2)
        self.assertEqual(len(payload["groups"]), 1)
        self.assertEqual(payload["groups"][0]["group"], "repo")
        self.assertEqual(payload["coverage"]["agent_written_lines"], 10)
        self.assertEqual(payload["coverage"]["reviewed_lines"], 7)
        self.assertEqual(payload["coverage"]["unreviewed_lines"], 3)
        self.assertEqual(payload["status"], "partial")

    def test_github_token_can_enrich_review_signals(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            _git(str(repo), "init")
            store = TraceStore(Path(tmp) / "traces")
            meta = _make_store_session(store, str(repo), branch="feat/github-reviewed")
            _append_write(store, meta, "app.py", 8)

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

    def test_badge_output_file_is_shields_compatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_dir = Path(tmp) / "traces"
            store = TraceStore(trace_dir)
            meta = _make_store_session(store, tmp)
            _append_write(store, meta, "src/app.py", 2)
            output = Path(tmp) / "badge.json"
            args = argparse.Namespace(
                trace_dir=str(trace_dir),
                since="30d",
                until="",
                threshold=0.7,
                by="repository",
                session="",
                github_token="",
                format="badge",
                output=str(output),
            )

            self.assertEqual(cmd_cognitive_debt(args), 0)
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(set(payload), {"schemaVersion", "label", "message", "color"})
        self.assertEqual(payload["label"], "cognitive debt")

    def test_parser_registers_machine_output_options_and_legacy_alias(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "cognitive-debt",
                "--by",
                "author",
                "--threshold",
                "0.8",
                "--format",
                "badge",
                "--output",
                "badge.json",
            ]
        )
        self.assertEqual(args.command, "cognitive-debt")
        self.assertEqual(args.by, "author")
        self.assertAlmostEqual(args.threshold, 0.8)
        self.assertEqual(args.format, "badge")
        self.assertEqual(args.output, "badge.json")


if __name__ == "__main__":
    unittest.main()
