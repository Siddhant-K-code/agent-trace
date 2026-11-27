import argparse
import io
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

from agent_trace import cognitive_debt
from agent_trace.cli import build_parser
from agent_trace.cognitive_debt import (
    CognitiveDebtInputError,
    CognitiveDebtReport,
    FileDebt,
    SessionDebt,
    build_cognitive_debt_report,
    cmd_cognitive_debt,
    format_cognitive_debt_badge,
    format_cognitive_debt_json,
    format_cognitive_debt_text,
    modified_file_lines,
)
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore

try:
    import _escapes
except ImportError:
    from tests import _escapes


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


def _append_write(store, meta, path, lines, *, timestamp=None):
    store.append_event(
        meta.session_id,
        TraceEvent(
            event_type=EventType.FILE_WRITE,
            timestamp=timestamp if timestamp is not None else time.time(),
            data={"path": path, "content": "\n".join(str(index) for index in range(lines))},
        ),
    )


def _append_raw_write(store, meta, path, tenant_id):
    """Append a two-line write as a raw line, so the store cannot retag it."""
    record = json.loads(
        TraceEvent(
            event_type=EventType.FILE_WRITE,
            session_id=meta.session_id,
            data={"path": path, "content": "a\nb"},
        ).to_json()
    )
    record["tenant_id"] = tenant_id
    events_path = store._session_dir(meta.session_id) / "events.ndjson"
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


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

    def test_window_report_reads_tagged_events_under_untagged_metadata(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"AGENT_STRACE_TENANT_ID": ""}
        ):
            store = TraceStore(Path(tmp) / "traces")
            meta = _make_store_session(store, tmp)
            _append_raw_write(store, meta, "a.py", "acme")
            _append_raw_write(store, meta, "b.py", "acme")

            report = build_cognitive_debt_report(
                store,
                window_start=time.time() - 60,
                window_end=time.time() + 60,
            )

        self.assertEqual(
            [session.session_id for session in report.sessions], [meta.session_id]
        )
        self.assertEqual(report.total_agent_written_lines, 4)

    def test_window_report_still_fails_closed_on_corrupt_sessions(self):
        # Each case breaks one part of the state the test above reads.
        cases = {
            "unusable started_at": ("meta.json", {"started_at": -1}),
            "session tenant ID that is not a string": ("meta.json", {"tenant_id": 5}),
            "event from another tenant": ("meta.json", {"tenant_id": "other"}),
            "event tenant ID that is not a string": (
                "events.ndjson",
                {"tenant_id": 5},
            ),
            "event line that cannot be read": ("events.ndjson", None),
        }
        for name, (file_name, fields) in cases.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp, patch.dict(
                os.environ, {"AGENT_STRACE_TENANT_ID": ""}
            ):
                store = TraceStore(Path(tmp) / "traces")
                meta = _make_store_session(store, tmp)
                _append_raw_write(store, meta, "a.py", "acme")
                path = store._session_dir(meta.session_id) / file_name
                if fields is None:
                    path.write_text("{not-json}\n", encoding="utf-8")
                else:
                    record = json.loads(path.read_text(encoding="utf-8"))
                    record.update(fields)
                    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

                with self.assertRaisesRegex(
                    CognitiveDebtInputError,
                    "^trace input could not be read safely$",
                ):
                    build_cognitive_debt_report(
                        store,
                        window_start=time.time() - 60,
                        window_end=time.time() + 60,
                    )

    def test_malformed_review_history_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _init_repo(Path(tmp) / "repo")
            _commit(repo, "app.py", "Add app")
            write_at = _commit_timestamp(repo) - 1
            store = TraceStore(Path(tmp) / "traces")
            meta = _make_store_session(store, repo, started_at=write_at - 60)
            _append_write(store, meta, "app.py", 2, timestamp=write_at)

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
                        window_start=write_at - 120,
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
                side_effect=lambda _cwd, path, _since, _lines, **_kwargs: reviews[path],
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


def _call(tool_name, arguments):
    return TraceEvent(
        event_type=EventType.TOOL_CALL,
        data={"tool_name": tool_name, "arguments": arguments},
    )


def _result(call, **data):
    return TraceEvent(event_type=EventType.TOOL_RESULT, parent_id=call.event_id, data=data)


def _lines(count, prefix="line"):
    return "".join(f"{prefix} {index}\n" for index in range(count))


def _coverage(events):
    """Return (lines per file, lines with no path) for *events*."""
    return cognitive_debt._modified_file_coverage(events)


class TestAgentWrittenLines(unittest.TestCase):
    """Each agent's write tools count the lines they add.

    Argument shapes follow tests/fixtures/capture_matrix/ and the trace of
    agent-trace-demo PR #1 (Copilot CLI 1.0.103).
    """

    def test_copilot_create_and_edit_count_the_lines_of_a_49_line_change(self):
        init_old = 'from .words import word_count\n\n__all__ = ["word_count"]\n'
        init_new = (
            "from .sentences import sentence_count\n"
            "from .words import word_count\n\n"
            '__all__ = ["sentence_count", "word_count"]\n'
        )
        calls = [
            _call("create", {"path": "/repo/src/tinytext/sentences.py", "file_text": _lines(8)}),
            _call("create", {"path": "/repo/tests/test_sentences.py", "file_text": _lines(26)}),
            _call("edit", {"path": "/repo/src/tinytext/__init__.py", "old_str": init_old, "new_str": init_new}),
            _call("edit", {
                "path": "/repo/README.md",
                "old_str": "Run the tests:",
                "new_str": _lines(13, "readme") + "Run the tests:",
            }),
            _call("bash", {"command": "python -m unittest", "description": "Run tests"}),
        ]
        events = [event for call in calls for event in (call, _result(call, success=True))]

        files, unknown = _coverage(events)

        self.assertEqual(files, {
            "/repo/src/tinytext/sentences.py": 8,
            "/repo/tests/test_sentences.py": 26,
            "/repo/src/tinytext/__init__.py": 2,
            "/repo/README.md": 13,
        })
        self.assertEqual(sum(files.values()), 49)
        self.assertEqual(unknown, 0)

    def test_claude_write_edit_multiedit_and_notebookedit(self):
        events = [
            _call("Write", {"file_path": "a.py", "content": "a\nb\nc\n"}),
            _call("Edit", {"file_path": "b.py", "old_string": "x\ny", "new_string": "x\nY\nz", "replace_all": False}),
            _call("MultiEdit", {"file_path": "c.py", "edits": [
                {"old_string": "p", "new_string": "p\nq"},
                {"old_string": "r", "new_string": "R"},
            ]}),
            _call("NotebookEdit", {"notebook_path": "n.ipynb", "cell_id": "1", "new_source": "s\nt"}),
            _call("NotebookEdit", {"notebook_path": "gone.ipynb", "cell_id": "2", "new_source": "", "edit_mode": "delete"}),
            _call("TodoWrite", {"todos": [{"content": "write the edit", "status": "pending"}]}),
        ]

        self.assertEqual(_coverage(events), ({"a.py": 3, "b.py": 2, "c.py": 2, "n.ipynb": 2}, 0))

    def test_gemini_write_file_and_replace(self):
        events = [
            _call("write_file", {"file_path": "w.py", "content": "1\n2\n3\n4\n"}),
            _call("replace", {"file_path": "r.py", "old_string": "a", "new_string": "b", "expected_replacements": 3}),
            _call("write_todos", {"todos": [{"description": "edit r.py", "status": "pending"}]}),
        ]

        self.assertEqual(_coverage(events), ({"w.py": 4, "r.py": 3}, 0))

    def test_codex_apply_patch_counts_each_file(self):
        patch_text = (
            "*** Begin Patch\n"
            "*** Add File: new.py\n+one\n+two\n"
            "*** Update File: changed.py\n@@\n-old\n+new\n+more\n"
            "*** Delete File: gone.py\n"
            "*** Update File: old_name.py\n*** Move to: new_name.py\n@@\n-q\n+r\n"
            "*** End Patch\n"
        )
        codex = _call("apply_patch", {"command": patch_text})
        raw = _call("apply_patch", patch_text)

        expected = ({"new.py": 2, "changed.py": 2, "new_name.py": 1}, 0)
        self.assertEqual(_coverage([codex]), expected)
        self.assertEqual(_coverage([raw]), expected)

    def test_cursor_file_write_counts_its_edits(self):
        event = TraceEvent(
            event_type=EventType.FILE_WRITE,
            data={"path": "src/x.py", "edits": [
                {"old_string": "a", "new_string": "a\nb"},
                {"old_string": "", "new_string": "c\nd\ne"},
            ]},
        )

        self.assertEqual(modified_file_lines([event]), {"src/x.py": 4})

    def test_editor_and_mcp_write_tools(self):
        events = [
            _call("str_replace_based_edit_tool", {"command": "create", "path": "e.py", "file_text": "1\n2\n"}),
            _call("str_replace_based_edit_tool", {"command": "view", "path": "v.py"}),
            _call("mcp__filesystem__write_file", {"path": "m.py", "content": "1\n2\n3\n"}),
            _call("mcp__github__create_issue", {"path": "not-a-write.md", "body": "x"}),
        ]

        self.assertEqual(_coverage(events), ({"e.py": 2, "m.py": 3}, 0))

    def test_failed_write_is_not_counted(self):
        error_call = _call("Write", {"file_path": "error.py", "content": "a\n"})
        is_error_call = _call("Write", {"file_path": "is_error.py", "content": "a\n"})
        unsuccessful_call = _call("apply_patch", {})
        ok_call = _call("Write", {"file_path": "ok.py", "content": "a\n"})
        events = [
            error_call,
            TraceEvent(event_type=EventType.ERROR, parent_id=error_call.event_id, data={"error": "denied"}),
            is_error_call,
            _result(is_error_call, is_error=True),
            unsuccessful_call,
            _result(unsuccessful_call, success=False),
            ok_call,
            _result(ok_call, success=True),
        ]

        self.assertEqual(_coverage(events), ({"ok.py": 1}, 0))

    def test_write_with_unrecorded_content_counts_one_line(self):
        events = [
            _call("create", {"path": "omitted.py", "file_text": "[OMITTED]"}),
            TraceEvent(event_type=EventType.FILE_WRITE, data={"path": "cursor.py"}),
        ]

        self.assertEqual(modified_file_lines(events), {"omitted.py": 1, "cursor.py": 1})

    def test_latest_write_time_is_tracked_without_changing_line_totals(self):
        events = [
            TraceEvent(
                event_type=EventType.FILE_WRITE,
                timestamp=10,
                data={"path": "app.py", "content": "a\nb"},
            ),
            TraceEvent(
                event_type=EventType.FILE_WRITE,
                timestamp=20,
                data={"path": "app.py", "content": "c\nd\ne"},
            ),
            TraceEvent(
                event_type=EventType.FILE_WRITE,
                timestamp=30,
                data={"content": "unknown\npath"},
            ),
        ]

        files, unknown = cognitive_debt._modified_file_stats(events)

        self.assertEqual(files["app.py"].lines, 5)
        self.assertEqual(files["app.py"].latest_write_at, 20)
        self.assertEqual(unknown, 2)

    def test_malformed_write_arguments_fail_closed(self):
        for event, message in (
            (_call("Write", ["a.py"]), "write tool arguments must be an object"),
            (_call(7, {"file_path": "a.py"}), "write tool name must be a string"),
            (_call("Edit", {"file_path": 3, "new_string": "x"}), "write event file_path must be a string"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(CognitiveDebtInputError, message):
                    modified_file_lines([event])


def _commit(repo, path, message, author="Dev <dev@example.com>"):
    target = Path(repo) / path
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(f"{message}\n")
    _git(repo, "add", path)
    _git(repo, "commit", "-q", "--author", author, "-m", message)


def _commit_timestamp(repo):
    result = subprocess.run(
        ["git", "-C", repo, "show", "-s", "--format=%ct", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def _init_repo(path, remote=""):
    path.mkdir(parents=True)
    _git(str(path), "init", "-q", "-b", "main")
    _git(str(path), "config", "user.email", "dev@example.com")
    _git(str(path), "config", "user.name", "Dev")
    _git(str(path), "config", "commit.gpgsign", "false")
    if remote:
        _git(str(path), "remote", "add", "origin", remote)
    return str(path)


class TestLocalReviewSignals(unittest.TestCase):
    """Only a Reviewed-by trailer naming another person, on the session's
    branch, counts as local review evidence."""

    def review(self, repo, *, write_at=None):
        """Return (reviewed lines, review signals, review state) for app.py."""
        write_at = write_at if write_at is not None else time.time() - 7200
        with tempfile.TemporaryDirectory() as traces:
            store = TraceStore(Path(traces))
            meta = _make_store_session(store, repo, started_at=write_at - 60)
            _append_write(store, meta, "app.py", 20, timestamp=write_at)
            report = build_cognitive_debt_report(
                store,
                window_start=write_at - 120,
                window_end=time.time() + 60,
            )
        (file_debt,) = report.sessions[0].files
        return file_debt.human_reviewed_lines, file_debt.review_signals, file_debt.review_state

    def test_words_co_author_trailers_and_other_authors_are_not_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _init_repo(Path(tmp) / "repo")
            _commit(repo, "app.py", "Add pipeline outline\n\nCo-authored-by: Copilot <copilot@github.com>")
            _commit(repo, "app.py", "Address comment on line 3 (#12)", author="Other <other@example.com>")
            _commit(repo, "app.py", "Merge pull request #12 from acme/review-fixes")

            self.assertEqual(self.review(repo), (0, ["committed_after_session"], "known"))

    def test_commit_on_another_branch_is_not_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _init_repo(Path(tmp) / "repo")
            _commit(repo, "app.py", "Add app")
            _git(repo, "checkout", "-q", "-b", "other")
            _commit(repo, "app.py", "Tweak app\n\nReviewed-by: Jane Reviewer <jane@example.com>")
            _git(repo, "checkout", "-q", "main")

            self.assertEqual(self.review(repo), (0, ["committed_after_session"], "known"))

    def test_self_and_agent_reviewed_by_trailers_are_not_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _init_repo(Path(tmp) / "repo")
            _commit(repo, "app.py", "Add app\n\nReviewed-by: Dev <DEV@example.com>")
            _commit(repo, "app.py", "Fix app\n\nReviewed-by: Claude <noreply@anthropic.com>")
            _commit(repo, "app.py", "Fix app\n\nReviewed-by: Dev")
            _commit(repo, "app.py", "Fix app\n\nReviewed-by: dev@example.com")
            _commit(repo, "app.py", "Fix app\n\nReviewed-by: Dev (dev@example.com)")
            _commit(repo, "app.py", "Fix app\n\nReviewed-by: dev@example.com.")

            self.assertEqual(self.review(repo), (0, ["committed_after_session"], "known"))

    def test_reviewed_by_another_person_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _init_repo(Path(tmp) / "repo")
            _commit(repo, "app.py", "Add app\n\nReviewed-by: Jane Reviewer <jane@example.com>")

            self.assertEqual(
                self.review(repo), (15, ["committed_after_session", "reviewed_by_trailer"], "known")
            )

    def test_review_before_latest_write_does_not_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _init_repo(Path(tmp) / "repo")
            _commit(repo, "app.py", "Add app\n\nReviewed-by: Jane Reviewer <jane@example.com>")

            self.assertEqual(
                self.review(repo, write_at=_commit_timestamp(repo) + 1),
                (0, ["not_committed"], "known"),
            )

    def test_review_after_latest_write_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _init_repo(Path(tmp) / "repo")
            _commit(repo, "app.py", "Add app\n\nReviewed-by: Jane Reviewer <jane@example.com>")

            self.assertEqual(
                self.review(repo, write_at=_commit_timestamp(repo) - 1),
                (15, ["committed_after_session", "reviewed_by_trailer"], "known"),
            )

    def test_repository_without_commits_is_not_committed(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _init_repo(Path(tmp) / "repo")

            self.assertEqual(self.review(repo), (0, ["not_committed"], "known"))

    def test_writes_outside_the_repository_do_not_stop_the_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _init_repo(Path(tmp) / "repo")
            _commit(repo, "app.py", "Add app")
            other = _init_repo(Path(tmp) / "other")
            store = TraceStore(Path(tmp) / "traces")
            write_at = time.time() - 3600
            meta = _make_store_session(store, repo, started_at=write_at - 60)
            _append_write(store, meta, "app.py", 20, timestamp=write_at)
            _append_write(
                store,
                meta,
                str(Path(tmp) / "plans" / "plan.md"),
                7,
                timestamp=write_at,
            )
            _append_write(store, meta, "../scratch.txt", 3, timestamp=write_at)
            _append_write(
                store,
                meta,
                str(Path(other) / "lib.py"),
                5,
                timestamp=write_at,
            )
            with patch(
                "agent_trace.cognitive_debt._github_review_for_file",
                return_value=(0, []),
            ) as github_review:
                report = build_cognitive_debt_report(
                    store,
                    window_start=write_at - 120,
                    window_end=time.time() + 60,
                )

        files = {Path(item.path).name: item for item in report.sessions[0].files}
        self.assertEqual(github_review.call_count, 1)
        self.assertEqual(github_review.call_args.args[2], "app.py")
        self.assertEqual(sorted(files), ["app.py", "lib.py"])
        self.assertEqual(files["lib.py"].review_signals, ["outside_repository"])
        self.assertEqual(files["lib.py"].review_state, "unknown")
        self.assertEqual(report.total_unreviewed_lines, 20)
        self.assertEqual(report.total_unknown_review_lines, 5)

    def test_missing_recorded_branch_does_not_fall_back_to_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _init_repo(Path(tmp) / "repo")
            _commit(repo, "app.py", "Add app\n\nReviewed-by: Jane Reviewer <jane@example.com>")
            write_at = _commit_timestamp(repo) - 1
            store = TraceStore(Path(tmp) / "traces")
            meta = _make_store_session(
                store,
                repo,
                branch="deleted-branch",
                started_at=write_at - 60,
            )
            _append_write(store, meta, "app.py", 20, timestamp=write_at)

            report = build_cognitive_debt_report(
                store,
                window_start=write_at - 120,
                window_end=time.time() + 60,
            )

        (file_debt,) = report.sessions[0].files
        self.assertEqual(file_debt.human_reviewed_lines, 0)
        self.assertEqual(file_debt.review_signals, ["git_revision_unavailable"])
        self.assertEqual(file_debt.review_state, "unknown")


class TestReviewPaths(unittest.TestCase):
    def test_file_write_and_tool_call_file_uris_are_reviewed_as_local_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _init_repo(Path(tmp) / "repo")
            write_at = time.time() - 3600
            store = TraceStore(Path(tmp) / "traces")
            meta = _make_store_session(store, repo, started_at=write_at - 60)
            file_path = Path(repo) / "src" / "file name.py"
            tool_path = Path(repo) / "src" / "tool name.py"
            _append_write(
                store,
                meta,
                file_path.as_uri(),
                4,
                timestamp=write_at,
            )
            store.append_event(
                meta.session_id,
                TraceEvent(
                    event_type=EventType.TOOL_CALL,
                    timestamp=write_at + 1,
                    data={
                        "tool_name": "Write",
                        "arguments": {
                            "file_path": tool_path.as_uri(),
                            "content": "a\nb\nc\n",
                        },
                    },
                ),
            )
            _commit(
                repo,
                "src/file name.py",
                "Add file\n\nReviewed-by: Jane Reviewer <jane@example.com>",
            )
            _commit(
                repo,
                "src/tool name.py",
                "Add tool file\n\nReviewed-by: Jane Reviewer <jane@example.com>",
            )

            report = build_cognitive_debt_report(
                store,
                window_start=write_at - 120,
                window_end=time.time() + 60,
            )

        files = {item.path: item for item in report.sessions[0].files}
        self.assertEqual(files[file_path.as_uri()].human_reviewed_lines, 3)
        self.assertEqual(files[tool_path.as_uri()].human_reviewed_lines, 2)
        self.assertIn("reviewed_by_trailer", files[file_path.as_uri()].review_signals)
        self.assertIn("reviewed_by_trailer", files[tool_path.as_uri()].review_signals)

    def test_unsupported_uri_scheme_stays_unknown_without_github_lookup(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _init_repo(Path(tmp) / "repo")
            _commit(repo, "app.py", "Add app")
            store = TraceStore(Path(tmp) / "traces")
            meta = _make_store_session(store, repo)
            _append_write(store, meta, "https://example.com/app.py", 4)

            with patch(
                "agent_trace.cognitive_debt._github_review_for_file",
                return_value=(4, ["github_line_comments"]),
            ) as github_review:
                report = build_cognitive_debt_report(
                    store,
                    window_start=time.time() - 60,
                    window_end=time.time() + 60,
                    github_token="token",
                )

        github_review.assert_not_called()
        (file_debt,) = report.sessions[0].files
        self.assertEqual(file_debt.review_signals, ["not_in_checkout"])
        self.assertEqual(file_debt.review_state, "unknown")


class TestGitHubReviewSignals(unittest.TestCase):
    WRITE_AT = 1_700_000_000.0
    BEFORE_WRITE = "2023-11-14T00:00:00Z"
    AFTER_WRITE = "2023-11-15T00:00:00Z"

    def search(self, *numbers):
        return {
            "total_count": len(numbers),
            "incomplete_results": False,
            "items": [{"number": number} for number in numbers],
        }

    def pr(
        self,
        number,
        *,
        merged_at=None,
        changed_files=1,
        head_repo="acme/widgets",
    ):
        return {
            "number": number,
            "user": {"login": "dev"},
            "merged": True,
            "merged_at": merged_at or self.AFTER_WRITE,
            "closed_at": merged_at or self.AFTER_WRITE,
            "changed_files": changed_files,
            "head": {"ref": "feature", "repo": {"full_name": head_repo}},
            "base": {"repo": {"full_name": "acme/widgets"}},
        }

    def review(self, responses, cwd, path, *, latest_write_at=None):
        replies = iter(responses)
        with patch("agent_trace.cognitive_debt._github_repo", return_value=("acme", "widgets")), patch(
            "agent_trace.cognitive_debt._github_json", side_effect=lambda *_args: next(replies)
        ):
            return cognitive_debt._github_review_for_file(
                cwd,
                "feature",
                path,
                "token",
                20,
                latest_write_at if latest_write_at is not None else self.WRITE_AT,
            )

    def test_merged_pr_with_only_author_bot_and_pending_reviews_is_unreviewed(self):
        reviews = [
            {"user": {"login": "dev"}, "state": "APPROVED", "submitted_at": self.AFTER_WRITE},
            {
                "user": {"login": "copilot-pull-request-reviewer", "type": "Bot"},
                "state": "COMMENTED",
                "submitted_at": self.AFTER_WRITE,
            },
            {"user": {"login": "jane"}, "state": "PENDING"},
        ]
        comments = [
            {
                "user": {"login": "github-actions[bot]"},
                "path": "app.py",
                "created_at": self.AFTER_WRITE,
            },
            {"user": {"login": "Dev"}, "path": "app.py", "created_at": self.AFTER_WRITE},
        ]

        self.assertEqual(
            self.review(
                [self.search(5), self.pr(5), [{"filename": "app.py"}], reviews, comments],
                ".",
                "app.py",
            ),
            (0, ["github_merged_pr"]),
        )

    def test_line_comment_from_another_person_matches_an_absolute_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _init_repo(Path(tmp) / "repo")
            reviews = [
                {
                    "user": {"login": "jane"},
                    "state": "APPROVED",
                    "submitted_at": self.AFTER_WRITE,
                }
            ]
            comments = [
                {
                    "user": {"login": "jane"},
                    "path": "src/app.py",
                    "created_at": self.AFTER_WRITE,
                }
            ]

            result = self.review(
                [
                    self.search(5),
                    self.pr(5),
                    [{"filename": "src/app.py"}],
                    reviews,
                    comments,
                ],
                repo,
                str(Path(repo) / "src" / "app.py"),
            )

        self.assertEqual(
            result,
            (20, ["github_line_comments", "github_merged_pr", "github_review_comments", "github_reviews"]),
        )

    def test_reused_branch_ignores_old_pr_and_uses_new_matching_pr(self):
        reviews = [
            {
                "user": {"login": "jane"},
                "state": "APPROVED",
                "submitted_at": self.AFTER_WRITE,
            }
        ]

        self.assertEqual(
            self.review(
                [
                    self.search(4, 5),
                    self.pr(4, merged_at=self.BEFORE_WRITE),
                    self.pr(5),
                    [{"filename": "app.py"}],
                    reviews,
                    [],
                ],
                ".",
                "app.py",
            ),
            (15, ["github_merged_pr", "github_reviews"]),
        )

    def test_pr_without_the_file_provides_no_evidence(self):
        self.assertEqual(
            self.review(
                [
                    self.search(5),
                    self.pr(5),
                    [{"filename": "other.py"}],
                ],
                ".",
                "app.py",
            ),
            (0, []),
        )

    def test_fork_pr_with_the_same_branch_name_provides_no_evidence(self):
        self.assertEqual(
            self.review(
                [
                    self.search(5),
                    self.pr(5, head_repo="someone/widgets"),
                ],
                ".",
                "app.py",
            ),
            (0, []),
        )

    def test_review_before_latest_write_does_not_count(self):
        reviews = [
            {
                "user": {"login": "jane"},
                "state": "APPROVED",
                "submitted_at": self.BEFORE_WRITE,
            }
        ]
        comments = [
            {
                "user": {"login": "jane"},
                "path": "app.py",
                "created_at": self.BEFORE_WRITE,
            }
        ]

        self.assertEqual(
            self.review(
                [self.search(5), self.pr(5), [{"filename": "app.py"}], reviews, comments],
                ".",
                "app.py",
            ),
            (0, ["github_merged_pr"]),
        )

    def test_review_after_latest_write_counts(self):
        reviews = [
            {
                "user": {"login": "jane"},
                "state": "APPROVED",
                "submitted_at": self.AFTER_WRITE,
            }
        ]

        self.assertEqual(
            self.review(
                [self.search(5), self.pr(5), [{"filename": "app.py"}], reviews, []],
                ".",
                "app.py",
            ),
            (15, ["github_merged_pr", "github_reviews"]),
        )

    def test_ambiguous_and_paginated_searches_fail_closed(self):
        ambiguous = [
            self.search(4, 5),
            self.pr(4),
            [{"filename": "app.py"}],
            self.pr(5),
            [{"filename": "app.py"}],
        ]
        paginated = {
            "total_count": 2,
            "incomplete_results": False,
            "items": [{"number": 5}],
        }

        self.assertEqual(self.review(ambiguous, ".", "app.py"), (0, []))
        self.assertEqual(self.review([paginated], ".", "app.py"), (0, []))


class TestMissingWorkingDirectory(unittest.TestCase):
    """A session read from another clone uses that clone's history when the
    recorded directory is gone and the repository matches."""

    def setUp(self):
        cwd = os.getcwd()
        self.addCleanup(os.chdir, cwd)

    def report(self, tmp, remote, recorded_repo):
        checkout = _init_repo(Path(tmp) / "checkout", remote)
        _commit(checkout, "src/app.py", "Add app")
        recorded = str(Path(tmp) / "gone" / "project")
        store = TraceStore(Path(tmp) / "traces")
        write_at = time.time() - 3600
        meta = _make_store_session(store, recorded, started_at=write_at - 60)
        meta.attribution["git_repo"] = recorded_repo
        store.update_meta(meta)
        _append_write(
            store,
            meta,
            os.path.join(recorded, "src", "app.py"),
            4,
            timestamp=write_at,
        )
        _append_write(store, meta, "/elsewhere/other.py", 2, timestamp=write_at)
        (Path(tmp) / "sibling.py").write_text("x\n", encoding="utf-8")
        _append_write(store, meta, "../sibling.py", 3, timestamp=write_at)
        os.chdir(Path(checkout) / "src")
        return build_cognitive_debt_report(
            store,
            window_start=write_at - 120,
            window_end=time.time() + 60,
        )

    def test_matching_repository_reads_the_current_checkout(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
            "agent_trace.cognitive_debt._github_review_for_file",
            return_value=(0, []),
        ) as github_review:
            report = self.report(
                tmp, "https://github.com/Acme/Widgets", "git@github.com:acme/widgets.git"
            )

        files = {Path(item.path).name: item for item in report.sessions[0].files}
        self.assertEqual(github_review.call_count, 1)
        self.assertTrue(github_review.call_args.args[2].endswith(os.path.join("src", "app.py")))
        self.assertTrue(report.git_available)
        self.assertTrue(files["app.py"].path.startswith(str(Path(tmp) / "gone")))
        self.assertEqual(files["app.py"].review_state, "known")
        self.assertEqual(files["app.py"].review_signals, ["committed_after_session"])
        self.assertEqual(files["other.py"].review_state, "unknown")
        self.assertEqual(files["other.py"].review_signals, ["not_in_checkout"])
        self.assertEqual(files["sibling.py"].review_signals, ["not_in_checkout"])
        self.assertEqual(report.total_unreviewed_lines, 4)
        self.assertEqual(report.total_unknown_review_lines, 5)

    def test_other_repository_stays_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = self.report(
                tmp, "https://github.com/acme/other", "https://github.com/acme/widgets.git"
            )

        self.assertFalse(report.git_available)
        self.assertEqual(report.status, "unknown")
        self.assertEqual(report.total_unknown_review_lines, 9)

    def subdirectory_report(self, tmp, remove_pkg):
        """Report a session recorded in ``project/pkg`` whose top-level README was reviewed."""
        checkout = _init_repo(Path(tmp) / "checkout", "https://github.com/acme/widgets")
        _commit(checkout, "README.md", "Add readme\n\nReviewed-by: Jane Reviewer <jane@example.com>")
        _commit(checkout, "pkg/README.md", "Add pkg readme")
        start = subprocess.run(
            ["git", "-C", checkout, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        if remove_pkg:
            _git(checkout, "rm", "-q", "-r", "pkg")
            _git(checkout, "commit", "-q", "-m", "Remove pkg")
        recorded = str(Path(tmp) / "gone" / "project" / "pkg")
        store = TraceStore(Path(tmp) / "traces")
        meta = _make_store_session(store, recorded, started_at=time.time() - 3600)
        meta.attribution.update(git_repo="https://github.com/acme/widgets.git", git_commit=start)
        store.update_meta(meta)
        _append_write(store, meta, os.path.join(recorded, "README.md"), 20)
        os.chdir(checkout)
        return build_cognitive_debt_report(
            store,
            window_start=time.time() - 7200,
            window_end=time.time() + 60,
        )

    def test_session_started_in_a_subdirectory_stays_unknown(self):
        for remove_pkg in (False, True):
            with self.subTest(remove_pkg=remove_pkg), tempfile.TemporaryDirectory() as tmp:
                report = self.subdirectory_report(tmp, remove_pkg)

                self.assertFalse(report.git_available)
                self.assertEqual(report.total_reviewed_lines, 0)
                self.assertEqual(report.total_unknown_review_lines, 20)


def _debt_report(branch="plain", group="plain", path="plain", review_state="known"):
    """Return a report with one workflow that wrote one file."""
    file_debt = FileDebt(path=path, agent_written_lines=4, review_state=review_state)
    session = SessionDebt(session_id="s1", started_at=0.0, branch=branch, author="plain",
                          agent_written_lines=4, human_reviewed_lines=0, files=[file_debt])
    row = {"group": group, "reviewed_lines": 0, "unreviewed_lines": 4,
           "unknown_review_lines": 0, "status": "above_threshold"}
    return CognitiveDebtReport(window_start=0.0, window_end=0.0, group_by="repository",
                               threshold=0.7, sessions=[session], rows={group: row},
                               git_available=True)


def _render(report):
    buf = io.StringIO()
    format_cognitive_debt_text(report, out=buf)
    return buf.getvalue()


# A payload that fits in the 20 characters of the branch column, and how it shows.
BRANCH = "A\x1b[2K\x07\rB\nC\x9bD\u202eE"
BRANCH_TEXT = r"A\x1b[2K\x07\rB\nC\x9bD\u202eE"

# (label, payload, how it shows, build): build(value) returns a report with
# value in the place the label names.
FORMAT_TEXT_CASES = [
    ("branch column", BRANCH, BRANCH_TEXT, lambda v: _debt_report(branch=v)),
    ("group", _escapes.PAYLOAD, _escapes.PAYLOAD_TEXT, lambda v: _debt_report(group=v)),
    ("zero-review path", _escapes.PAYLOAD, _escapes.PAYLOAD_TEXT,
     lambda v: _debt_report(path=v)),
    ("unknown-review path", _escapes.PAYLOAD, _escapes.PAYLOAD_TEXT,
     lambda v: _debt_report(path=v, review_state="unknown")),
]

# (label, cut, build): the text report keeps cut characters of the value the
# label names.
FORMAT_TEXT_CUTS = [
    ("branch column", 20, lambda v: _debt_report(branch=v)),
]


class TestTraceTextShowsAsText(unittest.TestCase):
    """A branch, a repository or a path in a trace can hold escape codes and
    line breaks. The text report shows them as text."""

    def test_each_value_shows_escapes_as_text(self):
        for label, payload, expected, build in FORMAT_TEXT_CASES:
            with self.subTest(label):
                _escapes.assert_shows_as_text(
                    self, _render(build(payload)), expected, _render(build("plain"))
                )

    def test_plain_text_prints_unchanged(self):
        for label, _, _, build in FORMAT_TEXT_CASES:
            with self.subTest(label):
                self.assertIn(_escapes.PLAIN, _render(build(_escapes.PLAIN)))

    def test_value_as_long_as_the_cut_shows_in_full(self):
        _escapes.check_each_cut(self, FORMAT_TEXT_CUTS, _render)

    def test_sgr_code_shows_as_text(self):
        output = _render(_debt_report(branch="a\x1b[8mhidden"))
        self.assertIn(r"a\x1b[8mhidden", output)
        self.assertNotIn("\x1b", output)


# The repository and the path in the payload trace, as stored.
REPOSITORY = "o/A\x1b]52;c;aWQK\x07B\x1b[2K\rC\nD\x9bE\u202eF"
PATH = "src/A\x1b]52;c;aWQK\x07B\x1b[2K\rC\nD\x9bE\u202eF"

BADGE = """{
  "color": "orange",
  "label": "cognitive debt",
  "message": "0% reviewed, 100% unreviewed",
  "schemaVersion": 1
}
"""


class TestPayloadTrace(unittest.TestCase):
    """A trace whose branch, repository and path hold a payload. The text
    report shows them as text. The JSON and the badge print as before."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.trace_dir = str(Path(tmp.name) / "traces")
        store = TraceStore(self.trace_dir)
        meta = SessionMeta(
            agent_name="codex",
            attribution={
                "working_dir": tmp.name,
                "git_branch": _escapes.PAYLOAD,
                "git_repo": REPOSITORY,
                "git_author": "dev@example.com",
            },
        )
        meta.started_at = time.time() - 10
        store.create_session(meta)
        _append_write(store, meta, PATH, 4)
        review = patch(
            "agent_trace.cognitive_debt._review_for_file", return_value=(0, [], True)
        )
        review.start()
        self.addCleanup(review.stop)

    def run_command(self, output_format, by="repository", output=""):
        args = argparse.Namespace(
            trace_dir=self.trace_dir,
            since="2000-01-01",
            until="2099-01-01",
            threshold=0.7,
            by=by,
            session="",
            github_token="",
            format=output_format,
            output=output,
        )
        with patch("sys.stdout", new_callable=io.StringIO) as out:
            self.assertEqual(cmd_cognitive_debt(args), 0)
        return out.getvalue()

    def test_text_report_shows_escapes_in_stdout_and_output_file(self):
        stdout = self.run_command("text", by="branch")
        report = Path(self.tmp) / "report.txt"
        self.assertEqual(self.run_command("text", by="branch", output=str(report)), "")
        self.assertEqual(report.read_text(encoding="utf-8"), stdout)
        for line in stdout.split("\n"):
            self.assertTrue(line.isprintable(), ascii(line))
        self.assertIn(f"\n  {_escapes.PAYLOAD_TEXT}: 0 reviewed", stdout)
        self.assertIn(f"\n  src/{_escapes.PAYLOAD_TEXT} (4 lines", stdout)

    def test_json_keeps_the_raw_text(self):
        for by, group in (("repository", REPOSITORY), ("branch", _escapes.PAYLOAD)):
            with self.subTest(by):
                payload = json.loads(self.run_command("json", by=by))
                workflow = payload["workflows"][0]
                session = payload["sessions"][0]
                self.assertEqual(workflow["branch"], _escapes.PAYLOAD)
                self.assertEqual(workflow["repository"], REPOSITORY)
                self.assertEqual(workflow["files"][0]["path"], PATH)
                self.assertEqual(session["branch"], _escapes.PAYLOAD)
                self.assertEqual(session["files"][0]["path"], PATH)
                self.assertEqual([row["group"] for row in payload["groups"]], [group])

    def test_badge_prints_as_before(self):
        self.assertEqual(self.run_command("badge"), BADGE)


if __name__ == "__main__":
    unittest.main()
