"""meta.json text fields that are not strings.

A hand-edited or foreign meta.json can hold a list, an object, a number, a
boolean or null where agent_name, command or team should be a string. Each
reader below treats such a value as an empty string: it does not crash, and
the raw value does not reach the output.
"""

from __future__ import annotations

import io
import json
import os
import re
import tempfile
import unittest
from unittest import mock

from agent_trace import cognitive_debt, team_report
from agent_trace.budget_report import (
    build_report,
    format_report_json,
    format_report_markdown,
    format_report_text,
    team_summary,
)
from agent_trace.cognitive_debt import build_cognitive_debt_report
from agent_trace.curve import DEFAULT_TASK_TYPE, analyse_curve, classify_session
from agent_trace.dashboard import build_dashboard
from agent_trace.freeze import freeze_session
from agent_trace.health import ProviderEvidence, detect_provider, redaction_state_for
from agent_trace.hooks import IMPLICIT_SESSION_SOURCE
from agent_trace.live_verify import inspect_live_capture
from agent_trace.store import TraceStore
from agent_trace.team_report import build_team_report

START = 1767225600.0  # 2026-01-01T00:00:00Z
WINDOW = (START - 3600, START + 86400)
IMPLICIT = f"claude-code ({IMPLICIT_SESSION_SOURCE})"
FIRST = "a1b2c3d4e5f60001"
SECOND = "b1b2c3d4e5f60002"
THIRD = "c1b2c3d4e5f60003"


def not_text(words: str) -> dict:
    """JSON values that are not strings, by label.

    The list and the object hold ``words``, so a reader that turns the value
    into text with str() shows them.
    """
    return {
        "list": [words],
        "object": {"value": words},
        "number": 5,
        "boolean": True,
        "null": None,
    }


class MetaTextTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for name in ("AGENT_STRACE_WORKSPACE", "AGENT_STRACE_TENANT_ID", "AGENT_TRACE_DIR"):
            os.environ.pop(name, None)

    def new_store(self) -> TraceStore:
        return TraceStore(os.path.join(tempfile.mkdtemp(dir=self.root), ".agent-traces"))

    def write_session(self, store: TraceStore, sid: str, started_at: float = START, **fields) -> None:
        meta = {
            "session_id": sid,
            "started_at": started_at,
            "ended_at": started_at + 1.5,
            "agent_name": "claude-code",
            "command": "claude-code (hook)",
            "tool_calls": 1,
            "total_tokens": 100,
            "total_duration_ms": 1500.0,
            **fields,
        }
        event = {
            "event_type": "tool_call",
            "timestamp": started_at + 0.5,
            "event_id": "e1" + sid[:6],
            "session_id": sid,
            "data": {"tool_name": "Bash", "arguments": {"command": "ls"}},
        }
        session_dir = store.base_dir / sid
        session_dir.mkdir(parents=True)
        (session_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        (session_dir / "events.ndjson").write_text(json.dumps(event) + "\n", encoding="utf-8")


class TestCurve(MetaTextTest):
    def test_value_that_is_not_text_classifies_as_general(self):
        for label, value in not_text("fix the bug").items():
            with self.subTest(label):
                self.assertEqual(classify_session(value, ""), DEFAULT_TASK_TYPE)
                self.assertEqual(classify_session("", value), DEFAULT_TASK_TYPE)

    def test_the_other_field_still_classifies(self):
        for label, value in not_text("fix the bug").items():
            with self.subTest(label):
                self.assertEqual(classify_session(value, "pytest tests/"), "Unit test writing")
                self.assertEqual(classify_session("pytest", value), "Unit test writing")

    def test_curve_counts_such_sessions(self):
        for label, value in not_text("fix the bug").items():
            with self.subTest(label):
                store = self.new_store()
                self.write_session(store, FIRST)
                self.write_session(store, SECOND, started_at=START + 60, agent_name=value)
                self.write_session(store, THIRD, started_at=START + 120, command=value)
                report = analyse_curve(store, min_sessions=1)
                self.assertEqual(
                    [(stat.task_type, stat.session_count) for stat in report.stats],
                    [(DEFAULT_TASK_TYPE, 3)],
                )


class TestDashboard(MetaTextTest):
    def test_agent_filter_passes_over_a_name_that_is_not_text(self):
        for label, value in not_text("claude-code").items():
            with self.subTest(label):
                store = self.new_store()
                self.write_session(store, FIRST)
                self.write_session(store, SECOND, started_at=START + 60, agent_name=value)
                filtered = build_dashboard(store, agent_filter="claude")
                self.assertEqual([s.session_id for s in filtered.summaries], [FIRST])
                everything = build_dashboard(store)
                self.assertEqual(
                    sorted(s.session_id for s in everything.summaries), [FIRST, SECOND]
                )


class TestBudgetReport(MetaTextTest):
    def render(self, report) -> tuple[str, str, dict]:
        text = io.StringIO()
        format_report_text(report, out=text)
        markdown = io.StringIO()
        format_report_markdown(report, out=markdown)
        top = json.loads(format_report_json(report))["top_sessions"]
        return text.getvalue(), markdown.getvalue(), {s["session_id"]: s["agent_name"] for s in top}

    def test_agent_name_that_is_not_text_falls_back_to_the_command(self):
        for label, value in not_text("fix the bug").items():
            with self.subTest(label):
                store = self.new_store()
                self.write_session(store, FIRST, agent_name=value)
                self.write_session(
                    store, SECOND, started_at=START + 60, agent_name=value, command=value
                )
                report = build_report(store, *WINDOW, include_prior=False)
                names = {FIRST: "claude-code (hook)", SECOND: ""}
                self.assertEqual({s.session_id: s.agent_name for s in report.sessions}, names)
                text, markdown, top = self.render(report)
                self.assertEqual(top, names)
                # A session without a name shows its short ID in the name column.
                for sid, shown in ((FIRST, "claude-code (hook)"), (SECOND, SECOND[:12])):
                    name = re.escape(shown)
                    self.assertRegex(
                        text, re.compile(rf"^  \d+\. {sid[:12]}  \$\d+\.\d\d  {name}  ", re.M)
                    )
                    self.assertRegex(
                        markdown,
                        re.compile(rf"^\| \d+ \| `{sid[:12]}` \| \$\d+\.\d\d \| {name} \| ", re.M),
                    )

    def test_team_that_is_not_text_reads_as_unassigned(self):
        for label, value in not_text("platform").items():
            with self.subTest(label):
                store = self.new_store()
                self.write_session(store, FIRST, team="platform")
                self.write_session(store, SECOND, started_at=START + 60, team=value)
                report = build_report(store, *WINDOW, include_prior=False)
                self.assertEqual(
                    {s.session_id: s.team for s in report.sessions}, {FIRST: "platform", SECOND: ""}
                )
                self.assertEqual(sorted(team_summary(report.sessions)), ["(unassigned)", "platform"])
                text, _, _ = self.render(report)
                self.assertRegex(text, re.compile(r"^  \(unassigned\) +\$", re.M))


class TestFreeze(MetaTextTest):
    def test_command_that_is_not_text_falls_back_to_the_agent_name(self):
        for label, value in not_text("fix the bug").items():
            with self.subTest(label):
                store = self.new_store()
                self.write_session(store, FIRST, command=value)
                self.write_session(
                    store, SECOND, started_at=START + 60, agent_name=value, command=value
                )
                self.assertEqual(freeze_session(store, FIRST).task, "claude-code")
                self.assertEqual(freeze_session(store, SECOND).task, "")


class TestBranchFromCommand(MetaTextTest):
    COMMAND = "claude (branch: feat/x)"

    def store_with(self, value) -> TraceStore:
        store = self.new_store()
        self.write_session(store, FIRST, command=self.COMMAND)
        self.write_session(store, SECOND, started_at=START + 60, command=value)
        return store

    def test_team_report(self):
        for label, value in not_text(self.COMMAND).items():
            with self.subTest(label):
                store = self.store_with(value)
                with mock.patch.object(team_report, "_git", return_value=""):
                    report = build_team_report(store, *WINDOW, group_by="branch")
                self.assertEqual(
                    {s.session_id: s.branch for s in report.sessions},
                    {FIRST: "feat/x", SECOND: "(unknown)"},
                )
                self.assertEqual(sorted(report.rows), ["(unknown)", "feat/x"])

    def test_cognitive_debt(self):
        for label, value in not_text(self.COMMAND).items():
            with self.subTest(label):
                store = self.store_with(value)
                with mock.patch.object(cognitive_debt, "_git", return_value=""):
                    report = build_cognitive_debt_report(store, *WINDOW, group_by="branch")
                self.assertEqual(
                    {s.session_id: s.branch for s in report.sessions},
                    {FIRST: "feat/x", SECOND: "(unknown)"},
                )
                self.assertEqual(sorted(report.rows), ["(unknown)", "feat/x"])


class TestHealth(MetaTextTest):
    def test_hook_metadata_names_the_provider(self):
        store = self.new_store()
        self.write_session(store, FIRST, command=IMPLICIT)
        meta = store.load_meta(FIRST)
        self.assertEqual(detect_provider(meta, []), ProviderEvidence("claude", "hook_metadata"))
        self.assertEqual(len(redaction_state_for(meta, []).session_start_modes), 1)

    def test_agent_name_that_is_not_text_is_not_hook_metadata(self):
        for label, value in not_text("claude-code").items():
            with self.subTest(label):
                store = self.new_store()
                self.write_session(store, FIRST, agent_name=value, command=IMPLICIT)
                self.write_session(store, SECOND, started_at=START + 60, agent_name="", command=IMPLICIT)
                meta = store.load_meta(FIRST)
                self.assertEqual(detect_provider(meta, []), ProviderEvidence())
                self.assertEqual(
                    redaction_state_for(meta, []),
                    redaction_state_for(store.load_meta(SECOND), []),
                )
                self.assertEqual(redaction_state_for(meta, []).session_start_modes, ())


class TestLiveVerify(MetaTextTest):
    def assert_selects_older(self, **newer_fields) -> None:
        store = self.new_store()
        self.write_session(store, FIRST, started_at=START - 600, command=IMPLICIT)
        self.write_session(store, SECOND, **newer_fields)
        result = inspect_live_capture(store.base_dir, "claude")
        self.assertEqual(result.report.session_id, FIRST)
        self.assertEqual(result.selection, "newest matching recorded session")

    def test_newer_session_with_an_agent_name_that_is_not_text_is_passed_over(self):
        for label, value in not_text("claude-code").items():
            with self.subTest(label):
                self.assert_selects_older(agent_name=value, command=IMPLICIT)

    def test_newer_session_with_a_command_that_is_not_text_is_passed_over(self):
        for label, value in not_text("claude-code (hook)").items():
            with self.subTest(label):
                self.assert_selects_older(command=value)


if __name__ == "__main__":
    unittest.main()
