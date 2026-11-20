"""Tests for issue #41: agent standup report."""

from __future__ import annotations

import io
import os
import tempfile
import time
import unittest

from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.standup import (
    ApproachAttempt,
    ReviewItem,
    StandupReport,
    UncertaintySignal,
    analyse_standup,
    format_standup,
)
from agent_trace.store import TraceStore

try:
    import _escapes
except ImportError:
    from tests import _escapes


def _make_store(tmp_dir: str) -> TraceStore:
    return TraceStore(os.path.join(tmp_dir, "traces"))


def _make_session(store: TraceStore, events_data: list[dict]) -> str:
    meta = SessionMeta(agent_name="test-agent")
    store.create_session(meta)
    sid = meta.session_id
    for ed in events_data:
        store.append_event(sid, TraceEvent(
            event_type=ed["event_type"],
            session_id=sid,
            data=ed.get("data", {}),
            **{key: ed[key] for key in ("event_id", "parent_id") if key in ed},
        ))
    return sid


class TestStandupExtraction(unittest.TestCase):
    def test_extract_new_deps_npm(self):
        from agent_trace.standup import _extract_new_deps
        events = [TraceEvent(
            event_type=EventType.TOOL_CALL,
            data={"tool_name": "bash", "arguments": {"command": "npm install jsonwebtoken"}},
        )]
        deps = _extract_new_deps(events)
        self.assertIn("jsonwebtoken", deps)

    def test_extract_new_deps_pip(self):
        from agent_trace.standup import _extract_new_deps
        events = [TraceEvent(
            event_type=EventType.TOOL_CALL,
            data={"tool_name": "bash", "arguments": {"command": "pip install requests"}},
        )]
        deps = _extract_new_deps(events)
        self.assertIn("requests", deps)

    def test_extract_uncertainties_todo(self):
        from agent_trace.standup import _extract_uncertainties
        events = [TraceEvent(
            event_type=EventType.TOOL_CALL,
            data={"tool_name": "write", "arguments": {
                "file_path": "src/auth.py",
                "new_str": "# TODO: handle edge case X\ndef login(): pass",
            }},
        )]
        signals = _extract_uncertainties(events)
        self.assertTrue(any("TODO" in s.description for s in signals))

    def test_extract_approaches_no_retries(self):
        from agent_trace.standup import _extract_approaches
        events = [
            TraceEvent(event_type=EventType.TOOL_RESULT,
                       data={"content": "success", "is_error": False}),
        ]
        approaches = _extract_approaches(events)
        self.assertEqual(len(approaches), 1)
        self.assertFalse(approaches[0].abandoned)

    def test_extract_approaches_with_retries(self):
        from agent_trace.standup import _extract_approaches
        events = [
            TraceEvent(event_type=EventType.TOOL_CALL,
                       data={"tool_name": "bash", "arguments": {}}),
            TraceEvent(event_type=EventType.TOOL_RESULT,
                       data={"content": "error: build failed", "is_error": True}),
            TraceEvent(event_type=EventType.TOOL_RESULT,
                       data={"content": "error: still failing", "is_error": True}),
            TraceEvent(event_type=EventType.TOOL_RESULT,
                       data={"content": "success", "is_error": False}),
        ]
        approaches = _extract_approaches(events)
        abandoned = [a for a in approaches if a.abandoned]
        self.assertGreater(len(abandoned), 0)

    def test_stream_ending_while_failing_is_still_failing(self):
        from agent_trace.standup import _extract_approaches
        events = [TraceEvent(**_tool("bash")), TraceEvent(**_result(True)),
                  TraceEvent(**_result(True))]
        self.assertEqual(_extract_approaches(events), [ApproachAttempt(
            "Approach using bash", abandoned=False, failure_reason="Failed 2 times",
            retries=2, still_failing=True)])

    def test_abandoned_then_still_failing(self):
        from agent_trace.standup import _extract_approaches
        events = [TraceEvent(**e) for e in [
            _tool("bash"), _result(True), _result(True), _result(False),
            _tool("edit"), _result(True), _result(True),
        ]]
        approaches = _extract_approaches(events)
        self.assertEqual([(a.description, a.abandoned, a.still_failing) for a in approaches],
                         [("Approach using bash", True, False),
                          ("Approach using edit", False, True)])

    def test_failing_tool_name_on_the_result_wins(self):
        from agent_trace.standup import _extract_approaches
        failed = {"event_type": EventType.TOOL_RESULT,
                  "data": {"tool_name": "bash", "content": "done", "is_error": True}}
        events = [TraceEvent(**e) for e in [
            _tool("bash"), _tool("read"), failed, failed, _result(False),
        ]]
        approaches = _extract_approaches(events)
        self.assertEqual([a.description for a in approaches], ["Approach using bash"])


class TestStandupReport(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_analyse_standup_returns_report(self):
        from agent_trace.standup import analyse_standup
        store = _make_store(self._tmp)
        sid = _make_session(store, [
            {"event_type": EventType.SESSION_START},
            {"event_type": EventType.TOOL_CALL,
             "data": {"tool_name": "read", "arguments": {"file_path": "src/auth.py"}}},
            {"event_type": EventType.TOOL_CALL,
             "data": {"tool_name": "write", "arguments": {"file_path": "src/auth.py",
                                                           "new_str": "def login(): pass"}}},
            {"event_type": EventType.SESSION_END},
        ])
        report = analyse_standup(store, sid)
        self.assertEqual(report.session_id, sid)
        self.assertGreaterEqual(report.files_read, 0)
        self.assertGreaterEqual(report.files_modified, 0)
        self.assertIsInstance(report.cost_usd, float)

    def test_files_modified_tracked(self):
        from agent_trace.standup import analyse_standup
        store = _make_store(self._tmp)
        sid = _make_session(store, [
            {"event_type": EventType.TOOL_CALL,
             "data": {"tool_name": "write", "arguments": {"file_path": "src/a.py", "new_str": "x"}}},
            {"event_type": EventType.TOOL_CALL,
             "data": {"tool_name": "write", "arguments": {"file_path": "src/b.py", "new_str": "y"}}},
        ])
        report = analyse_standup(store, sid)
        self.assertEqual(report.files_modified, 2)
        self.assertIn("src/a.py", report.files_modified_list)

    def test_format_standup_output(self):
        from agent_trace.standup import analyse_standup, format_standup
        store = _make_store(self._tmp)
        sid = _make_session(store, [
            {"event_type": EventType.SESSION_START},
            {"event_type": EventType.SESSION_END},
        ])
        report = analyse_standup(store, sid)
        buf = io.StringIO()
        format_standup(report, out=buf)
        output = buf.getvalue()
        self.assertIn("Session:", output)
        self.assertIn("What the agent did", output)
        self.assertIn("Stats:", output)

    def test_cli_has_standup_command(self):
        from agent_trace.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["standup", "--no-llm"])
        self.assertTrue(args.no_llm)


def _report(**changes):
    """Return a report with plain text in each value, changed by *changes*."""
    fields = dict(
        session_id="s1", duration_seconds=1.0, cost_usd=0.0, files_read=0,
        files_modified=1, files_modified_list=["plain"], new_dependencies=["plain"],
        approaches=[ApproachAttempt("plain", abandoned=False)],
        uncertainties=[UncertaintySignal("plain", location="plain")],
        review_items=[ReviewItem("plain")],
        tool_calls=0, llm_requests=0, context_resets=0, retries=0, errors=0,
    )
    fields.update(changes)
    return StandupReport(**fields)


def _render(report):
    buf = io.StringIO()
    format_standup(report, out=buf)
    return buf.getvalue()


def _approach(description="plain", **fields):
    return _report(approaches=[ApproachAttempt(description, **fields)])


def _uncertainty(description="plain", location="plain"):
    return _report(uncertainties=[UncertaintySignal(description, location=location)])


# (label, kind, build): build(value) returns a report with value in the place
# the label names. The kinds are in tests/_escapes.py.
FORMAT_STANDUP_CASES = [
    ("modified path", "text", lambda v: _report(files_modified_list=["plain", v])),
    ("approach", "text", lambda v: _approach(v, abandoned=False)),
    ("retried approach", "text", lambda v: _approach(v, abandoned=False, retries=2)),
    ("abandoned approach", "text",
     lambda v: _approach(v, abandoned=True, failure_reason="plain")),
    ("failure reason", "text", lambda v: _approach(abandoned=True, failure_reason=v)),
    ("still-failing approach", "text",
     lambda v: _approach(v, abandoned=False, failure_reason="plain", still_failing=True)),
    ("new dependency", "text", lambda v: _report(new_dependencies=["plain", v])),
    ("uncertainty", "free", lambda v: _uncertainty(description=v)),
    ("uncertainty location", "text", lambda v: _uncertainty(location=v)),
    ("review item", "text", lambda v: _report(review_items=[ReviewItem(v)])),
]

# (label, cut, build): standup keeps cut characters of the value the label
# names, after it shows each run of whitespace as one space.
FORMAT_STANDUP_CUTS = [
    ("uncertainty", 80, lambda v: _uncertainty(description=v)),
]


class TestTraceTextShowsAsText(unittest.TestCase):
    """A path, a package name or a line the agent wrote can hold escape codes
    and line breaks. standup shows them as text."""

    def test_each_value_shows_escapes_as_text(self):
        _escapes.check_each_value(self, FORMAT_STANDUP_CASES, _render)

    def test_plain_text_prints_unchanged(self):
        _escapes.check_plain_text_unchanged(self, FORMAT_STANDUP_CASES, _render)

    def test_whitespace_in_free_text_shows_as_one_space(self):
        _escapes.check_whitespace_in_free_text(self, FORMAT_STANDUP_CASES, _render)

    def test_value_as_long_as_the_cut_shows_in_full(self):
        _escapes.check_each_cut(self, FORMAT_STANDUP_CUTS, _render)

    def test_sgr_code_shows_as_text(self):
        output = _render(_report(files_modified_list=["a\x1b[8mhidden"]))
        self.assertIn(r"a\x1b[8mhidden", output)
        self.assertNotIn("\x1b", output)


def _tool(name, **arguments):
    return {"event_type": EventType.TOOL_CALL,
            "data": {"tool_name": name, "arguments": arguments}}


def _result(is_error):
    return {"event_type": EventType.TOOL_RESULT,
            "data": {"content": "done", "is_error": is_error}}


def _call(name, call_id):
    return {**_tool(name), "event_id": call_id}


def _outcome(data, parent="", event_type=EventType.TOOL_RESULT):
    return {"event_type": event_type, "data": data, "parent_id": parent}


END = {"event_type": EventType.SESSION_END}
STILL_FAILING_AT_END = [
    "  - Still failing when the session ended: Approach using bash (Failed 2 times)"]
NO_RETRIES = ["  - Single approach (no retries detected)"]


# A package name with no whitespace, as an install command holds it, and how it shows.
PACKAGE = "pkg\x1b]52;c;aWQK\x07\x1b[2K\x9b\u202e"
PACKAGE_TEXT = r"pkg\x1b]52;c;aWQK\x07\x1b[2K\x9b\u202e"


class TestPayloadTrace(unittest.TestCase):
    """Text that standup takes from a trace shows as text."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = _make_store(tmp.name)

    def standup(self, events_data):
        sid = _make_session(self.store, events_data)
        output = _render(analyse_standup(self.store, sid))
        for line in output.split("\n"):
            self.assertTrue(line.isprintable(), ascii(line))
        return output

    def test_written_line_shows_one_space_for_each_run_and_escapes(self):
        content = "x = 1  # TODO\tA\x1b]52;c;aWQK\x07B\x9bC\u202eD"
        output = self.standup([_tool("write", file_path="src/a.py", content=content)])
        self.assertIn(r"  - x = 1 # TODO A\x1b]52;c;aWQK\x07B\x9bC\u202eD  [src/a.py:1]" + "\n",
                      output)

    def test_installed_package_shows_as_text(self):
        output = self.standup([_tool("bash", command=f"pip install {PACKAGE}")])
        self.assertIn(f"new dependency(ies): {PACKAGE_TEXT}\n", output)
        self.assertIn(f"New dependency added: {PACKAGE_TEXT}\n", output)

    def test_large_change_path_shows_as_text(self):
        output = self.standup([_tool("write", file_path=_escapes.PAYLOAD, content="line\n" * 101)])
        self.assertIn(f"\n      {_escapes.PAYLOAD_TEXT}\n", output)
        self.assertIn(f"Large change: {_escapes.PAYLOAD_TEXT} (101 lines)\n", output)

    def test_abandoned_tool_shows_as_text(self):
        output = self.standup([_tool(_escapes.PAYLOAD), _result(True), _result(True),
                               _result(False)])
        self.assertIn(f"Tried and abandoned: Approach using {_escapes.PAYLOAD_TEXT} "
                      "(Failed 2 times)\n", output)


class TestApproachLines(unittest.TestCase):
    """The approach lines standup prints for a session."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = _make_store(tmp.name)

    def approach_lines(self, events_data, ended_at=None):
        sid = _make_session(self.store, events_data)
        if ended_at is not None:
            meta = self.store.load_meta(sid)
            meta.ended_at = ended_at
            self.store.update_meta(meta)
        output = _render(analyse_standup(self.store, sid))
        lines = output.split("\n")
        start = lines.index("What the agent did:") + 2
        return lines[start:lines.index("", start)]

    def test_session_ending_while_failing(self):
        self.assertEqual(
            self.approach_lines([_tool("bash"), _result(True), _result(True), END]),
            STILL_FAILING_AT_END)

    def test_meta_end_time_after_the_failures_ends_the_session(self):
        events = [_tool("bash"), _result(True), _result(True)]
        self.assertEqual(self.approach_lines(events, ended_at=time.time() + 1),
                         STILL_FAILING_AT_END)

    def test_trace_ending_while_failing(self):
        trace_ends = ["  - Still failing when the trace ends: Approach using bash (Failed 2 times)"]
        cases = {
            "no session_end": [_tool("bash"), _result(True), _result(True)],
            "failures after a session_end": [_tool("bash"), _result(False), END,
                                             _tool("bash"), _result(True), _result(True)],
        }
        for label, events in cases.items():
            with self.subTest(label):
                self.assertEqual(self.approach_lines(events), trace_ends)

    def test_each_recorded_failure_starts_a_streak(self):
        cases = {
            "is_error true": {"is_error": True},
            "success false": {"success": False},
            "nonzero exit_code": {"exit_code": 1},
            "failed status": {"status": "failed"},
            "error status with no failure words": {"status": " Error ", "content": "done"},
            "failure status": {"status": "FAILURE"},
            "timeout status": {"status": "timeout"},
            "failure words with no outcome field": {"content": "Traceback (most recent call)"},
        }
        for label, data in cases.items():
            with self.subTest(label):
                self.assertEqual(
                    self.approach_lines([_tool("bash"), _outcome(data), _outcome(data), END]),
                    STILL_FAILING_AT_END)

    def test_error_event_for_a_tool_call_is_a_failure(self):
        events = [_call("bash", "c1"), _outcome({"error": "x"}, "c1", EventType.ERROR),
                  _call("bash", "c2"), _outcome({"error": "x"}, "c2", EventType.ERROR), END]
        self.assertEqual(self.approach_lines(events), STILL_FAILING_AT_END)

    def test_recorded_success_is_a_success_whatever_the_text(self):
        for data in ({"is_error": False}, {"success": True}, {"exit_code": 0},
                     {"status": "success"}, {"status": " Succeeded "}, {"status": "OK"},
                     {"status": "completed"}):
            data = {**data, "content": "ran: 0 errors"}
            with self.subTest(data):
                self.assertEqual(
                    self.approach_lines([_tool("bash"), _outcome(data), _outcome(data)]),
                    NO_RETRIES)
                self.assertEqual(
                    self.approach_lines([_tool("bash"), _result(True), _result(True),
                                         _outcome(data)]),
                    ["  - Tried and abandoned: Approach using bash (Failed 2 times)"])

    def test_events_that_are_not_tool_failures(self):
        cancelled = {"status": "cancelled", "content": "error: cancelled"}
        canceled = {"status": " Canceled ", "content": "error: cancelled"}
        cases = {
            "cancelled result": [_tool("bash"), _outcome(cancelled), _outcome(cancelled), END],
            "canceled status in any case": [_tool("bash"), _outcome(canceled),
                                            _outcome(canceled), END],
            "error with no tool call": [_tool("bash"), _outcome({}, "", EventType.ERROR),
                                        _outcome({}, "", EventType.ERROR), END],
        }
        for label, events in cases.items():
            with self.subTest(label):
                self.assertEqual(self.approach_lines(events), NO_RETRIES)

    def test_one_failure_for_each_call(self):
        events = [_call("bash", "c1"), _outcome({"is_error": True}, "c1"),
                  _outcome({"error": "x"}, "c1", EventType.ERROR), END]
        self.assertEqual(self.approach_lines(events), ["  - Single approach (1 retries)"])

    def test_parallel_calls_name_the_failing_call(self):
        failed = {"content": "done", "is_error": True}
        events = [_call("bash", "c1"), _call("read", "c2"), _outcome(failed, "c1"),
                  _call("bash", "c3"), _call("read", "c4"), _outcome(failed, "c3"),
                  _outcome({"is_error": False}, "c2")]
        self.assertEqual(self.approach_lines(events),
                         ["  - Tried and abandoned: Approach using bash (Failed 2 times)"])

    def test_isolated_failure(self):
        self.assertEqual(
            self.approach_lines([_tool("bash"), _result(True), _tool("bash"), _result(False)]),
            ["  - Single approach (1 retries)"])

    def test_no_failure(self):
        self.assertEqual(self.approach_lines([_tool("bash"), _result(False)]),
                         ["  - Single approach (no retries detected)"])

    def test_abandoned_approach(self):
        self.assertEqual(
            self.approach_lines([_tool("bash"), _result(True), _tool("bash"), _result(True),
                                 _tool("write"), _result(False)]),
            ["  - Tried and abandoned: Approach using bash (Failed 2 times)"])
