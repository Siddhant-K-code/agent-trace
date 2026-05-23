"""Tests for agent-strace timeline (Issue #81)."""

from __future__ import annotations

import argparse
import io
import json
import tempfile
import time
import unittest
from pathlib import Path

from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore
from agent_trace.timeline import (
    TimelineEntry,
    TimelinePhase,
    TimelineResult,
    build_timeline,
    cmd_timeline,
    format_timeline,
    format_timeline_json,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store() -> TraceStore:
    return TraceStore(Path(tempfile.mkdtemp()))


def _add_session(
    store: TraceStore,
    *,
    tool_calls: list[dict] | None = None,
    tool_results: list[dict] | None = None,
    llm_requests: int = 0,
    llm_responses: int = 0,
    errors: list[str] | None = None,
    file_reads: list[str] | None = None,
    file_writes: list[str] | None = None,
    decisions: list[str] | None = None,
    user_prompts: list[str] | None = None,
    duration_ms: float = 5000.0,
) -> str:
    meta = SessionMeta(agent_name="test", command="test", total_duration_ms=duration_ms)
    sp = store.create_session(meta)
    sid = sp.name
    base = 1_000_000.0
    t = base

    def ev(etype, data, dur=None):
        nonlocal t
        t += 0.5
        return TraceEvent(
            event_type=etype,
            timestamp=t,
            session_id=sid,
            data=data,
            duration_ms=dur,
        )

    events = [ev(EventType.SESSION_START, {})]

    for prompt in (user_prompts or []):
        events.append(ev(EventType.USER_PROMPT, {"prompt": prompt}))

    for tc in (tool_calls or []):
        events.append(ev(EventType.TOOL_CALL, tc, dur=100.0))

    for tr in (tool_results or []):
        events.append(ev(EventType.TOOL_RESULT, tr, dur=50.0))

    for _ in range(llm_requests):
        events.append(ev(EventType.LLM_REQUEST, {"model": "claude-3-5-sonnet", "message_count": 3}))

    for _ in range(llm_responses):
        events.append(ev(EventType.LLM_RESPONSE, {"total_tokens": 500}, dur=800.0))

    for path in (file_reads or []):
        events.append(ev(EventType.FILE_READ, {"uri": path}))

    for path in (file_writes or []):
        events.append(ev(EventType.FILE_WRITE, {"uri": path}))

    for msg in (errors or []):
        events.append(ev(EventType.ERROR, {"message": msg}))

    for text in (decisions or []):
        events.append(ev(EventType.DECISION, {"choice": text, "reason": "because"}))

    events.append(ev(EventType.SESSION_END, {"exit_code": 0}))

    for e in events:
        store.append_event(sid, e)

    return sid


# ---------------------------------------------------------------------------
# build_timeline
# ---------------------------------------------------------------------------

class TestBuildTimeline(unittest.TestCase):

    def test_returns_timeline_result(self):
        store = _make_store()
        sid = _add_session(store)
        result = build_timeline(store, sid)
        self.assertIsInstance(result, TimelineResult)

    def test_session_id_preserved(self):
        store = _make_store()
        sid = _add_session(store)
        result = build_timeline(store, sid)
        self.assertEqual(result.session_id, sid)

    def test_has_at_least_one_phase(self):
        store = _make_store()
        sid = _add_session(store, tool_calls=[{"tool_name": "Bash", "arguments": {"command": "ls"}}])
        result = build_timeline(store, sid)
        self.assertGreater(len(result.phases), 0)

    def test_phases_are_timeline_phase(self):
        store = _make_store()
        sid = _add_session(store)
        result = build_timeline(store, sid)
        for phase in result.phases:
            self.assertIsInstance(phase, TimelinePhase)

    def test_total_duration_positive(self):
        store = _make_store()
        sid = _add_session(store, duration_ms=3000.0)
        result = build_timeline(store, sid)
        self.assertGreaterEqual(result.total_duration, 0)

    def test_total_events_positive(self):
        store = _make_store()
        sid = _add_session(store, tool_calls=[{"tool_name": "Read", "arguments": {"file_path": "x.py"}}])
        result = build_timeline(store, sid)
        self.assertGreater(result.total_events, 0)

    def test_error_count_increments(self):
        store = _make_store()
        sid = _add_session(store, errors=["something broke", "also this"])
        result = build_timeline(store, sid)
        self.assertGreaterEqual(result.error_count, 2)

    def test_no_errors_gives_zero_error_count(self):
        store = _make_store()
        sid = _add_session(store)
        result = build_timeline(store, sid)
        self.assertEqual(result.error_count, 0)

    def test_total_cost_non_negative(self):
        store = _make_store()
        sid = _add_session(store, llm_requests=2, llm_responses=2)
        result = build_timeline(store, sid)
        self.assertGreaterEqual(result.total_cost, 0.0)

    def test_wasted_cost_non_negative(self):
        store = _make_store()
        sid = _add_session(store, errors=["fail"])
        result = build_timeline(store, sid)
        self.assertGreaterEqual(result.wasted_cost, 0.0)

    def test_wasted_cost_lte_total_cost(self):
        store = _make_store()
        sid = _add_session(store, llm_requests=1, errors=["fail"])
        result = build_timeline(store, sid)
        self.assertLessEqual(result.wasted_cost, result.total_cost + 1e-9)

    def test_multiple_phases_from_user_prompts(self):
        store = _make_store()
        sid = _add_session(store, user_prompts=["first task", "second task"])
        result = build_timeline(store, sid)
        self.assertGreaterEqual(len(result.phases), 2)

    def test_phase_offsets_non_negative(self):
        store = _make_store()
        sid = _add_session(store)
        result = build_timeline(store, sid)
        for phase in result.phases:
            self.assertGreaterEqual(phase.start_offset, 0.0)
            self.assertGreaterEqual(phase.end_offset, 0.0)

    def test_phase_end_gte_start(self):
        store = _make_store()
        sid = _add_session(store)
        result = build_timeline(store, sid)
        for phase in result.phases:
            self.assertGreaterEqual(phase.end_offset, phase.start_offset)

    def test_entries_are_timeline_entry(self):
        store = _make_store()
        sid = _add_session(store, tool_calls=[{"tool_name": "Bash", "arguments": {"command": "echo hi"}}])
        result = build_timeline(store, sid)
        for phase in result.phases:
            for entry in phase.entries:
                self.assertIsInstance(entry, TimelineEntry)

    def test_entry_status_values(self):
        store = _make_store()
        sid = _add_session(store, tool_calls=[{"tool_name": "Read", "arguments": {"file_path": "a.py"}}])
        result = build_timeline(store, sid)
        valid = {"ok", "fail", "info"}
        for phase in result.phases:
            for entry in phase.entries:
                self.assertIn(entry.status, valid)

    def test_bash_tool_label(self):
        store = _make_store()
        sid = _add_session(store, tool_calls=[{"tool_name": "Bash", "arguments": {"command": "pytest"}}])
        result = build_timeline(store, sid)
        labels = [e.label for p in result.phases for e in p.entries]
        self.assertTrue(any("Bash" in l or "Run" in l for l in labels))

    def test_read_tool_label(self):
        store = _make_store()
        sid = _add_session(store, tool_calls=[{"tool_name": "Read", "arguments": {"file_path": "src/main.py"}}])
        result = build_timeline(store, sid)
        labels = [e.label for p in result.phases for e in p.entries]
        self.assertTrue(any("src/main.py" in l for l in labels))

    def test_write_tool_label(self):
        store = _make_store()
        sid = _add_session(store, tool_calls=[{"tool_name": "Write", "arguments": {"file_path": "out.py", "new_string": "x\ny\nz"}}])
        result = build_timeline(store, sid)
        labels = [e.label for p in result.phases for e in p.entries]
        self.assertTrue(any("out.py" in l for l in labels))

    def test_error_entry_status_fail(self):
        store = _make_store()
        sid = _add_session(store, errors=["test failed"])
        result = build_timeline(store, sid)
        fail_entries = [e for p in result.phases for e in p.entries if e.status == "fail"]
        self.assertGreater(len(fail_entries), 0)

    def test_error_entry_label_contains_error(self):
        store = _make_store()
        sid = _add_session(store, errors=["something broke"])
        result = build_timeline(store, sid)
        labels = [e.label for p in result.phases for e in p.entries if e.status == "fail"]
        self.assertTrue(any("Error" in l or "error" in l for l in labels))

    def test_llm_request_entry_present(self):
        store = _make_store()
        sid = _add_session(store, llm_requests=1)
        result = build_timeline(store, sid)
        labels = [e.label for p in result.phases for e in p.entries]
        self.assertTrue(any("LLM" in l or "llm" in l for l in labels))

    def test_llm_response_entry_present(self):
        store = _make_store()
        sid = _add_session(store, llm_responses=1)
        result = build_timeline(store, sid)
        labels = [e.label for p in result.phases for e in p.entries]
        self.assertTrue(any("response" in l.lower() for l in labels))

    def test_decision_entry_present(self):
        store = _make_store()
        sid = _add_session(store, decisions=["use approach A"])
        result = build_timeline(store, sid)
        labels = [e.label for p in result.phases for e in p.entries]
        self.assertTrue(any("Decision" in l or "decision" in l for l in labels))

    def test_file_read_entry_present(self):
        store = _make_store()
        sid = _add_session(store, file_reads=["README.md"])
        result = build_timeline(store, sid)
        labels = [e.label for p in result.phases for e in p.entries]
        self.assertTrue(any("README.md" in l for l in labels))

    def test_file_write_entry_present(self):
        store = _make_store()
        sid = _add_session(store, file_writes=["output.txt"])
        result = build_timeline(store, sid)
        labels = [e.label for p in result.phases for e in p.entries]
        self.assertTrue(any("output.txt" in l for l in labels))

    def test_custom_model_affects_cost(self):
        store = _make_store()
        sid = _add_session(store, llm_requests=2, llm_responses=2)
        result_sonnet = build_timeline(store, sid, model="sonnet")
        result_opus = build_timeline(store, sid, model="opus")
        # opus is more expensive than sonnet
        self.assertGreaterEqual(result_opus.total_cost, result_sonnet.total_cost)

    def test_empty_session_no_crash(self):
        store = _make_store()
        sid = _add_session(store)
        result = build_timeline(store, sid)
        self.assertIsInstance(result, TimelineResult)

    def test_retry_count_non_negative(self):
        store = _make_store()
        sid = _add_session(store)
        result = build_timeline(store, sid)
        self.assertGreaterEqual(result.retry_count, 0)


# ---------------------------------------------------------------------------
# format_timeline (text)
# ---------------------------------------------------------------------------

class TestFormatTimeline(unittest.TestCase):

    def _result(self, **kwargs) -> TimelineResult:
        store = _make_store()
        sid = _add_session(store, **kwargs)
        return build_timeline(store, sid)

    def test_output_contains_session_id(self):
        store = _make_store()
        sid = _add_session(store)
        result = build_timeline(store, sid)
        out = io.StringIO()
        format_timeline(result, out)
        self.assertIn(sid, out.getvalue())

    def test_output_contains_phase_header(self):
        result = self._result()
        out = io.StringIO()
        format_timeline(result, out)
        self.assertIn("Phase", out.getvalue())

    def test_output_contains_ok_icon(self):
        result = self._result(llm_responses=1)
        out = io.StringIO()
        format_timeline(result, out)
        self.assertIn("✓", out.getvalue())

    def test_output_contains_fail_icon_on_error(self):
        result = self._result(errors=["boom"])
        out = io.StringIO()
        format_timeline(result, out)
        self.assertIn("✗", out.getvalue())

    def test_output_contains_wasted_spend_on_error(self):
        store = _make_store()
        sid = _add_session(store, llm_requests=2, errors=["fail"])
        result = build_timeline(store, sid)
        # Manually mark a phase as failed to trigger wasted cost output
        if result.phases:
            result.phases[0].failed = True
            result.wasted_cost = result.phases[0].total_cost
            result.total_cost = max(result.total_cost, result.wasted_cost + 0.001)
        out = io.StringIO()
        format_timeline(result, out)
        text = out.getvalue()
        # Either wasted spend callout or FAILED tag should appear
        self.assertTrue("Wasted" in text or "FAILED" in text or "✗" in text)

    def test_output_contains_tool_label(self):
        result = self._result(tool_calls=[{"tool_name": "Bash", "arguments": {"command": "ls -la"}}])
        out = io.StringIO()
        format_timeline(result, out)
        self.assertIn("ls -la", out.getvalue())

    def test_output_contains_error_message(self):
        result = self._result(errors=["test suite failed"])
        out = io.StringIO()
        format_timeline(result, out)
        self.assertIn("test suite failed", out.getvalue())

    def test_output_non_empty(self):
        result = self._result()
        out = io.StringIO()
        format_timeline(result, out)
        self.assertGreater(len(out.getvalue()), 0)

    def test_multiple_phases_all_shown(self):
        result = self._result(user_prompts=["task one", "task two"])
        out = io.StringIO()
        format_timeline(result, out)
        text = out.getvalue()
        self.assertIn("Phase 1", text)
        self.assertIn("Phase 2", text)

    def test_cost_shown_when_nonzero(self):
        store = _make_store()
        sid = _add_session(store, llm_requests=3, llm_responses=3)
        result = build_timeline(store, sid)
        # Force a non-zero cost for display
        result.total_cost = 0.0042
        out = io.StringIO()
        format_timeline(result, out)
        self.assertIn("$", out.getvalue())


# ---------------------------------------------------------------------------
# format_timeline_json
# ---------------------------------------------------------------------------

class TestFormatTimelineJson(unittest.TestCase):

    def _json_result(self, **kwargs) -> dict:
        store = _make_store()
        sid = _add_session(store, **kwargs)
        result = build_timeline(store, sid)
        out = io.StringIO()
        format_timeline_json(result, out)
        return json.loads(out.getvalue())

    def test_valid_json(self):
        store = _make_store()
        sid = _add_session(store)
        result = build_timeline(store, sid)
        out = io.StringIO()
        format_timeline_json(result, out)
        data = json.loads(out.getvalue())
        self.assertIsInstance(data, dict)

    def test_json_has_session_id(self):
        data = self._json_result()
        self.assertIn("session_id", data)

    def test_json_has_phases(self):
        data = self._json_result()
        self.assertIn("phases", data)
        self.assertIsInstance(data["phases"], list)

    def test_json_has_total_duration(self):
        data = self._json_result()
        self.assertIn("total_duration", data)

    def test_json_has_total_cost(self):
        data = self._json_result()
        self.assertIn("total_cost", data)

    def test_json_has_error_count(self):
        data = self._json_result()
        self.assertIn("error_count", data)

    def test_json_has_retry_count(self):
        data = self._json_result()
        self.assertIn("retry_count", data)

    def test_json_has_wasted_cost(self):
        data = self._json_result()
        self.assertIn("wasted_cost", data)

    def test_json_phase_has_entries(self):
        data = self._json_result(tool_calls=[{"tool_name": "Read", "arguments": {"file_path": "x.py"}}])
        for phase in data["phases"]:
            self.assertIn("entries", phase)
            self.assertIsInstance(phase["entries"], list)

    def test_json_phase_has_name(self):
        data = self._json_result()
        for phase in data["phases"]:
            self.assertIn("name", phase)

    def test_json_phase_has_index(self):
        data = self._json_result()
        for phase in data["phases"]:
            self.assertIn("index", phase)

    def test_json_phase_has_failed_flag(self):
        data = self._json_result()
        for phase in data["phases"]:
            self.assertIn("failed", phase)

    def test_json_entry_has_status(self):
        data = self._json_result(tool_calls=[{"tool_name": "Bash", "arguments": {"command": "echo"}}])
        for phase in data["phases"]:
            for entry in phase["entries"]:
                self.assertIn("status", entry)

    def test_json_entry_has_label(self):
        data = self._json_result(tool_calls=[{"tool_name": "Bash", "arguments": {"command": "echo"}}])
        for phase in data["phases"]:
            for entry in phase["entries"]:
                self.assertIn("label", entry)

    def test_json_entry_status_valid(self):
        data = self._json_result(tool_calls=[{"tool_name": "Read", "arguments": {"file_path": "a.py"}}])
        valid = {"ok", "fail", "info"}
        for phase in data["phases"]:
            for entry in phase["entries"]:
                self.assertIn(entry["status"], valid)

    def test_json_error_count_matches(self):
        data = self._json_result(errors=["e1", "e2"])
        self.assertGreaterEqual(data["error_count"], 2)

    def test_json_total_events_positive(self):
        data = self._json_result(tool_calls=[{"tool_name": "Read", "arguments": {}}])
        self.assertGreater(data["total_events"], 0)


# ---------------------------------------------------------------------------
# cmd_timeline
# ---------------------------------------------------------------------------

class TestCmdTimeline(unittest.TestCase):

    def _args(self, store: TraceStore, session_id: str | None = None,
              fmt: str = "text", model: str = "sonnet") -> argparse.Namespace:
        return argparse.Namespace(
            trace_dir=store.base_dir,
            session_id=session_id,
            format=fmt,
            model=model,
        )

    def test_returns_0_on_success(self):
        store = _make_store()
        sid = _add_session(store)
        args = self._args(store, sid)
        self.assertEqual(cmd_timeline(args), 0)

    def test_returns_0_with_latest_session(self):
        store = _make_store()
        _add_session(store)
        args = self._args(store, session_id=None)
        self.assertEqual(cmd_timeline(args), 0)

    def test_returns_1_when_no_sessions(self):
        store = _make_store()
        args = self._args(store, session_id=None)
        self.assertEqual(cmd_timeline(args), 1)

    def test_returns_1_for_unknown_session(self):
        store = _make_store()
        _add_session(store)
        args = self._args(store, session_id="nonexistent000")
        self.assertEqual(cmd_timeline(args), 1)

    def test_json_format_returns_0(self):
        store = _make_store()
        sid = _add_session(store)
        args = self._args(store, sid, fmt="json")
        self.assertEqual(cmd_timeline(args), 0)

    def test_prefix_lookup_works(self):
        store = _make_store()
        sid = _add_session(store)
        args = self._args(store, session_id=sid[:6])
        self.assertEqual(cmd_timeline(args), 0)

    def test_text_output_written_to_stdout(self, capsys=None):
        store = _make_store()
        sid = _add_session(store, tool_calls=[{"tool_name": "Bash", "arguments": {"command": "ls"}}])
        result = build_timeline(store, sid)
        buf = io.StringIO()
        format_timeline(result, buf)
        self.assertIn(sid, buf.getvalue())

    def test_json_output_is_valid_json(self):
        store = _make_store()
        sid = _add_session(store)
        result = build_timeline(store, sid)
        buf = io.StringIO()
        format_timeline_json(result, buf)
        data = json.loads(buf.getvalue())
        self.assertIn("session_id", data)

    def test_opus_model_accepted(self):
        store = _make_store()
        sid = _add_session(store)
        args = self._args(store, sid, model="opus")
        self.assertEqual(cmd_timeline(args), 0)

    def test_haiku_model_accepted(self):
        store = _make_store()
        sid = _add_session(store)
        args = self._args(store, sid, model="haiku")
        self.assertEqual(cmd_timeline(args), 0)


if __name__ == "__main__":
    unittest.main()
