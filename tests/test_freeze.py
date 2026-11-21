"""Tests for freeze/regression fixtures."""

from __future__ import annotations

import argparse
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_trace.cli import build_parser
from agent_trace.freeze import (
    RegressionChange,
    RegressionFixture,
    RegressionReport,
    cmd_freeze,
    cmd_regression,
    compare_fixtures,
    extract_steps,
    freeze_session,
    print_regression_report,
)
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore

try:
    import _escapes
except ImportError:
    from tests import _escapes


def _make_store() -> TraceStore:
    return TraceStore(tempfile.mkdtemp())


def _add_session(
    store: TraceStore,
    session_id: str,
    tools: list[tuple[str, dict]],
    prompt: str = "fix auth",
) -> str:
    meta = SessionMeta(session_id=session_id, command=prompt)
    store.create_session(meta)
    store.append_event(
        session_id,
        TraceEvent(
            event_type=EventType.USER_PROMPT,
            session_id=session_id,
            data={"prompt": prompt},
        ),
    )
    for tool, arguments in tools:
        store.append_event(
            session_id,
            TraceEvent(
                event_type=EventType.TOOL_CALL,
                session_id=session_id,
                data={"tool_name": tool, "arguments": arguments},
            ),
        )
    return session_id


class TestFreezeHelpers(unittest.TestCase):
    def test_extract_steps_hashes_inputs(self):
        events = [
            TraceEvent(
                event_type=EventType.TOOL_CALL,
                data={"tool_name": "Bash", "arguments": {"command": "pytest"}},
            )
        ]
        steps = extract_steps(events)
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].tool, "Bash")
        self.assertEqual(len(steps[0].input_hash), 64)

    def test_freeze_does_not_store_raw_arguments(self):
        store = _make_store()
        sid = _add_session(
            store,
            "abc123",
            [("Bash", {"command": "echo secret-token"})],
        )
        fixture = freeze_session(store, sid)
        text = fixture.to_json()
        self.assertIn("input_hash", text)
        self.assertNotIn("secret-token", text)

    def test_fixture_round_trip(self):
        store = _make_store()
        sid = _add_session(store, "abc123", [("Read", {"file_path": "a.py"})])
        fixture = freeze_session(store, sid, task="custom task")
        loaded = RegressionFixture.from_json(fixture.to_json())
        self.assertEqual(loaded.session, sid)
        self.assertEqual(loaded.task, "custom task")
        self.assertEqual(loaded.steps[0].tool, "Read")


class TestRegressionCompare(unittest.TestCase):
    def _fixture(self, store: TraceStore, sid: str, tools: list[tuple[str, dict]]) -> RegressionFixture:
        _add_session(store, sid, tools)
        return freeze_session(store, sid)

    def test_identical_sequence_passes(self):
        store = _make_store()
        expected = self._fixture(store, "a", [("Read", {"file_path": "a.py"})])
        actual = self._fixture(store, "b", [("Read", {"file_path": "a.py"})])
        report = compare_fixtures(expected, actual)
        self.assertFalse(report.exceeded)
        self.assertEqual(report.divergence_score, 0.0)
        self.assertEqual(report.changes, [])

    def test_changed_input_fails(self):
        store = _make_store()
        expected = self._fixture(store, "a", [("Read", {"file_path": "a.py"})])
        actual = self._fixture(store, "b", [("Read", {"file_path": "b.py"})])
        report = compare_fixtures(expected, actual)
        self.assertTrue(report.exceeded)
        self.assertEqual(report.changes[0].kind, "changed_input")

    def test_added_step_reported(self):
        store = _make_store()
        expected = self._fixture(store, "a", [("Read", {"file_path": "a.py"})])
        actual = self._fixture(
            store,
            "b",
            [("Read", {"file_path": "a.py"}), ("Bash", {"command": "pytest"})],
        )
        report = compare_fixtures(expected, actual)
        self.assertTrue(report.exceeded)
        self.assertIn("added", {change.kind for change in report.changes})

    def test_reordered_step_reported(self):
        store = _make_store()
        expected = self._fixture(
            store,
            "a",
            [("Read", {"file_path": "a.py"}), ("Bash", {"command": "pytest"})],
        )
        actual = self._fixture(
            store,
            "b",
            [("Bash", {"command": "pytest"}), ("Read", {"file_path": "a.py"})],
        )
        report = compare_fixtures(expected, actual)
        self.assertTrue(report.exceeded)
        self.assertIn("reordered", {change.kind for change in report.changes})


class TestFreezeCommands(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = TraceStore(self.tmp)
        _add_session(self.store, "abc123", [("Read", {"file_path": "a.py"})])

    def _freeze_args(self, **kwargs):
        defaults = dict(
            trace_dir=self.tmp,
            session_id="abc",
            output=None,
            task="",
            format="text",
        )
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def _regression_args(self, **kwargs):
        defaults = dict(
            trace_dir=self.tmp,
            fixture_file="",
            session_id="abc",
            threshold=0.0,
            format="text",
        )
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_cmd_freeze_writes_output(self):
        output = Path(self.tmp) / "fixture.json"
        result = cmd_freeze(self._freeze_args(output=str(output), task="stored task"))
        self.assertEqual(result, 0)
        data = json.loads(output.read_text())
        self.assertEqual(data["task"], "stored task")
        self.assertEqual(data["steps"][0]["tool"], "Read")

    def test_cmd_freeze_json_stdout_parseable_with_output(self):
        output = Path(self.tmp) / "fixture.json"
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            result = cmd_freeze(self._freeze_args(output=str(output), format="json"))
        self.assertEqual(result, 0)
        data = json.loads(captured.getvalue())
        self.assertEqual(data["session"], "abc123")

    def test_cmd_regression_json_fails_on_divergence(self):
        fixture_path = Path(self.tmp) / "fixture.json"
        fixture_path.write_text(freeze_session(self.store, "abc123").to_json())
        _add_session(self.store, "def456", [("Bash", {"command": "pytest"})])
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            result = cmd_regression(
                self._regression_args(
                    fixture_file=str(fixture_path),
                    session_id="def",
                    format="json",
                )
            )
        self.assertEqual(result, 1)
        data = json.loads(captured.getvalue())
        self.assertTrue(data["exceeded"])

    def _add_unreadable_session(self) -> str:
        """Add session bad789 with two unreadable lines. Return the store's reason."""
        _add_session(self.store, "bad789", [("Read", {"file_path": "a.py"})])
        with (Path(self.tmp) / "bad789" / "events.ndjson").open("a", encoding="utf-8") as handle:
            handle.write("[]\n{\n")
        return "line 3 of events.ndjson is an array, not an object; 1 more line is unreadable"

    def test_cmd_freeze_names_an_unreadable_session_once(self):
        reason = self._add_unreadable_session()
        output = Path(self.tmp) / "fixture.json"
        out, err = io.StringIO(), io.StringIO()
        with patch("sys.stdout", out), patch("sys.stderr", err):
            result = cmd_freeze(self._freeze_args(session_id="bad789", output=str(output)))
        self.assertEqual(result, 1)
        self.assertEqual(out.getvalue(), "")
        self.assertEqual(err.getvalue(), f"Could not freeze session bad789: {reason}\n")
        self.assertFalse(output.exists())

    def test_cmd_regression_names_an_unreadable_session_once(self):
        reason = self._add_unreadable_session()
        fixture_path = Path(self.tmp) / "fixture.json"
        fixture_path.write_text(freeze_session(self.store, "abc123").to_json())
        out, err = io.StringIO(), io.StringIO()
        with patch("sys.stdout", out), patch("sys.stderr", err):
            result = cmd_regression(
                self._regression_args(fixture_file=str(fixture_path), session_id="bad789")
            )
        self.assertEqual(result, 1)
        self.assertEqual(out.getvalue(), "")
        self.assertEqual(err.getvalue(), f"Could not load session bad789: {reason}\n")

    def test_parser_registers_commands(self):
        parser = build_parser()
        freeze_args = parser.parse_args(["freeze", "abc", "--output", "fixture.json"])
        self.assertEqual(freeze_args.command, "freeze")
        self.assertEqual(freeze_args.output, "fixture.json")

        regression_args = parser.parse_args(["regression", "fixture.json", "abc", "--threshold", "0.25"])
        self.assertEqual(regression_args.command, "regression")
        self.assertEqual(regression_args.threshold, 0.25)


def _report(fixture="plain", reordered="plain", changed_input="plain", removed="plain", added="plain"):
    """Return a report with one change of each kind, each named by its tool."""
    return RegressionReport(
        fixture=fixture, actual="actual", expected_steps=3, actual_steps=3,
        edit_distance=3, divergence_score=1.0, threshold=0.0, exceeded=True,
        changes=[
            RegressionChange(kind="reordered", expected_seq=1, actual_seq=2, tool=reordered),
            RegressionChange(kind="changed_input", expected_seq=2, actual_seq=1, tool=changed_input),
            RegressionChange(kind="removed", expected_seq=3, tool=removed),
            RegressionChange(kind="added", actual_seq=3, tool=added),
        ],
    )


def _render(report):
    buf = io.StringIO()
    print_regression_report(report, out=buf)
    return buf.getvalue()


# (label, kind, build): build(value) returns a report with value in the place
# the label names. The kinds are in tests/_escapes.py.
REGRESSION_REPORT_CASES = [
    ("Fixture", "text", lambda v: _report(fixture=v)),
    ("reordered", "text", lambda v: _report(reordered=v)),
    ("changed input", "text", lambda v: _report(changed_input=v)),
    ("removed", "text", lambda v: _report(removed=v)),
    ("added", "text", lambda v: _report(added=v)),
]


class TestRegressionReportShowsEscapesAsText(unittest.TestCase):
    """A fixture file sets the session and each tool name, and an MCP server
    names its own tools. regression shows escape codes in them as text."""

    def test_each_value_shows_escapes_as_text(self):
        _escapes.check_each_value(self, REGRESSION_REPORT_CASES, _render)

    def test_plain_text_prints_unchanged(self):
        _escapes.check_plain_text_unchanged(self, REGRESSION_REPORT_CASES, _render)

    def test_sgr_code_shows_as_text(self):
        sgr = "a\x1b[8mhidden"
        output = _render(_report(fixture=sgr, reordered=sgr, changed_input=sgr, removed=sgr, added=sgr))
        self.assertEqual(output.count(r"a\x1b[8mhidden"), 5)
        self.assertNotIn("\x1b", output)

    def test_json_output_keeps_the_raw_text(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        payload = _escapes.PAYLOAD
        _add_session(TraceStore(tmp.name), "esc-actual", [("B" + payload, {"q": "x"})])
        fixture = Path(tmp.name) / "fixture.json"
        fixture.write_text(json.dumps({
            "session": payload,
            "steps": [{"seq": 1, "tool": payload, "input_hash": "abc"}],
        }))
        args = argparse.Namespace(trace_dir=tmp.name, fixture_file=str(fixture),
                                  session_id="esc-actual", threshold=0.0, format="json")
        with patch("sys.stdout", new_callable=io.StringIO) as out:
            self.assertEqual(cmd_regression(args), 1)
        data = json.loads(out.getvalue())
        self.assertEqual(data["fixture"], payload)
        self.assertEqual({change["tool"] for change in data["changes"]}, {payload, "B" + payload})


if __name__ == "__main__":
    unittest.main()
