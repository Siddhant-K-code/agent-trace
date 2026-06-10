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
    RegressionFixture,
    cmd_freeze,
    cmd_regression,
    compare_fixtures,
    extract_steps,
    freeze_session,
)
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore


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

    def test_parser_registers_commands(self):
        parser = build_parser()
        freeze_args = parser.parse_args(["freeze", "abc", "--output", "fixture.json"])
        self.assertEqual(freeze_args.command, "freeze")
        self.assertEqual(freeze_args.output, "fixture.json")

        regression_args = parser.parse_args(["regression", "fixture.json", "abc", "--threshold", "0.25"])
        self.assertEqual(regression_args.command, "regression")
        self.assertEqual(regression_args.threshold, 0.25)


if __name__ == "__main__":
    unittest.main()
