"""Tests for agent-strace compare (Issue #114)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent_trace.compare import (
    cmd_compare,
    decision_divergence,
    _decision_texts,
    _edit_distance,
    _sessions_by_tag,
    _get_user_prompt,
    _report_to_dict,
)
from agent_trace.diff import compare_sessions
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store() -> TraceStore:
    tmp = tempfile.mkdtemp()
    return TraceStore(Path(tmp))


def _add_session(store: TraceStore, agent_name: str = "test",
                 tool_calls: list[str] | None = None,
                 decisions: list[str] | None = None,
                 user_prompt: str | None = None,
                 write_files: list[str] | None = None) -> str:
    meta = SessionMeta(agent_name=agent_name, command=agent_name)
    sp = store.create_session(meta)
    sid = sp.name

    events: list[TraceEvent] = [
        TraceEvent(event_type=EventType.SESSION_START, timestamp=1.0,
                   session_id=sid, data={}),
    ]
    if user_prompt:
        events.append(TraceEvent(event_type=EventType.USER_PROMPT, timestamp=1.1,
                                 session_id=sid, data={"content": user_prompt}))
    for tool in (tool_calls or ["Read"]):
        events.append(TraceEvent(event_type=EventType.TOOL_CALL, timestamp=2.0,
                                 session_id=sid, data={"tool_name": tool}))
    for path in (write_files or []):
        events.append(TraceEvent(event_type=EventType.TOOL_CALL, timestamp=2.5,
                                 session_id=sid,
                                 data={"tool_name": "Write",
                                       "arguments": {"file_path": path}}))
    for text in (decisions or []):
        events.append(TraceEvent(event_type=EventType.DECISION, timestamp=3.0,
                                 session_id=sid, data={"text": text}))
    events.append(TraceEvent(event_type=EventType.SESSION_END, timestamp=4.0,
                             session_id=sid, data={}))

    for e in events:
        store.append_event(sid, e)
    return sid


# ---------------------------------------------------------------------------
# _edit_distance
# ---------------------------------------------------------------------------

class TestEditDistance(unittest.TestCase):
    def test_identical_lists(self):
        self.assertEqual(_edit_distance(["a", "b"], ["a", "b"]), 0)

    def test_empty_lists(self):
        self.assertEqual(_edit_distance([], []), 0)

    def test_one_empty(self):
        self.assertEqual(_edit_distance(["a", "b"], []), 2)
        self.assertEqual(_edit_distance([], ["a", "b"]), 2)

    def test_single_insertion(self):
        self.assertEqual(_edit_distance(["a"], ["a", "b"]), 1)

    def test_single_deletion(self):
        self.assertEqual(_edit_distance(["a", "b"], ["a"]), 1)

    def test_single_substitution(self):
        self.assertEqual(_edit_distance(["a"], ["b"]), 1)

    def test_completely_different(self):
        self.assertEqual(_edit_distance(["a", "b", "c"], ["x", "y", "z"]), 3)


# ---------------------------------------------------------------------------
# _decision_texts
# ---------------------------------------------------------------------------

class TestDecisionTexts(unittest.TestCase):
    def test_extracts_decision_events(self):
        store = _make_store()
        sid = _add_session(store, decisions=["chose A", "chose B"])
        texts = _decision_texts(store, sid)
        self.assertEqual(texts, ["chose A", "chose B"])

    def test_empty_when_no_decisions(self):
        store = _make_store()
        sid = _add_session(store)
        texts = _decision_texts(store, sid)
        self.assertEqual(texts, [])

    def test_missing_session_returns_empty(self):
        store = _make_store()
        texts = _decision_texts(store, "nonexistent")
        self.assertEqual(texts, [])


# ---------------------------------------------------------------------------
# decision_divergence
# ---------------------------------------------------------------------------

class TestDecisionDivergence(unittest.TestCase):
    def test_identical_decisions_zero_divergence(self):
        store = _make_store()
        sid_a = _add_session(store, decisions=["chose A", "chose B"])
        sid_b = _add_session(store, decisions=["chose A", "chose B"])
        self.assertEqual(decision_divergence(store, sid_a, sid_b), 0)

    def test_different_decisions_nonzero(self):
        store = _make_store()
        sid_a = _add_session(store, decisions=["chose A", "chose B"])
        sid_b = _add_session(store, decisions=["chose X", "chose Y"])
        self.assertGreater(decision_divergence(store, sid_a, sid_b), 0)

    def test_extra_decision_in_b(self):
        store = _make_store()
        sid_a = _add_session(store, decisions=["chose A"])
        sid_b = _add_session(store, decisions=["chose A", "chose B"])
        self.assertEqual(decision_divergence(store, sid_a, sid_b), 1)

    def test_no_decisions_both_zero(self):
        store = _make_store()
        sid_a = _add_session(store)
        sid_b = _add_session(store)
        self.assertEqual(decision_divergence(store, sid_a, sid_b), 0)


# ---------------------------------------------------------------------------
# _sessions_by_tag
# ---------------------------------------------------------------------------

class TestSessionsByTag(unittest.TestCase):
    def test_finds_sessions_by_agent_name(self):
        store = _make_store()
        _add_session(store, agent_name="refactor-auth")
        _add_session(store, agent_name="refactor-auth")
        _add_session(store, agent_name="add-tests")
        ids = _sessions_by_tag(store, "refactor-auth", last=2)
        self.assertEqual(len(ids), 2)

    def test_returns_at_most_last_n(self):
        store = _make_store()
        for _ in range(5):
            _add_session(store, agent_name="my-task")
        ids = _sessions_by_tag(store, "my-task", last=2)
        self.assertEqual(len(ids), 2)

    def test_returns_empty_when_no_match(self):
        store = _make_store()
        _add_session(store, agent_name="other-task")
        ids = _sessions_by_tag(store, "nonexistent", last=2)
        self.assertEqual(ids, [])

    def test_case_insensitive_match(self):
        store = _make_store()
        _add_session(store, agent_name="Refactor-Auth")
        _add_session(store, agent_name="Refactor-Auth")
        ids = _sessions_by_tag(store, "refactor-auth", last=2)
        self.assertEqual(len(ids), 2)


# ---------------------------------------------------------------------------
# _get_user_prompt
# ---------------------------------------------------------------------------

class TestGetUserPrompt(unittest.TestCase):
    def test_returns_prompt_when_present(self):
        store = _make_store()
        sid = _add_session(store, user_prompt="Fix the login bug")
        prompt = _get_user_prompt(store, sid)
        self.assertEqual(prompt, "Fix the login bug")

    def test_returns_none_when_absent(self):
        store = _make_store()
        sid = _add_session(store)
        prompt = _get_user_prompt(store, sid)
        self.assertIsNone(prompt)


# ---------------------------------------------------------------------------
# compare_sessions + _report_to_dict
# ---------------------------------------------------------------------------

class TestCompareSessionsIntegration(unittest.TestCase):
    def test_produces_report(self):
        store = _make_store()
        sid_a = _add_session(store, tool_calls=["Read", "Write"],
                             write_files=["a.py"])
        sid_b = _add_session(store, tool_calls=["Read"],
                             write_files=["a.py"])
        report = compare_sessions(store, sid_a, sid_b)
        self.assertEqual(report.session_a, sid_a)
        self.assertEqual(report.session_b, sid_b)
        self.assertIsInstance(report.verdict, str)

    def test_report_to_dict_structure(self):
        store = _make_store()
        sid_a = _add_session(store)
        sid_b = _add_session(store)
        report = compare_sessions(store, sid_a, sid_b)
        d = _report_to_dict(report, divergence=3)
        self.assertIn("session_a", d)
        self.assertIn("session_b", d)
        self.assertIn("verdict", d)
        self.assertIn("decision_divergence", d)
        self.assertEqual(d["decision_divergence"], 3)
        self.assertIn("divergence_points", d)

    def test_json_serialisable(self):
        store = _make_store()
        sid_a = _add_session(store)
        sid_b = _add_session(store)
        report = compare_sessions(store, sid_a, sid_b)
        d = _report_to_dict(report, divergence=0)
        # Must not raise
        json.dumps(d)


# ---------------------------------------------------------------------------
# cmd_compare CLI
# ---------------------------------------------------------------------------

class TestCmdCompare(unittest.TestCase):
    def _args(self, store_dir, sid_a=None, sid_b=None, fmt="text",
              tag=None, last=2, rerun=False, model=None):
        import argparse
        args = argparse.Namespace()
        args.trace_dir = store_dir
        args.session_id_a = sid_a
        args.session_id_b = sid_b
        args.format = fmt
        args.tag = tag
        args.last = last
        args.rerun = rerun
        args.model = model
        return args

    def test_compare_two_sessions_text(self):
        store = _make_store()
        sid_a = _add_session(store)
        sid_b = _add_session(store)
        args = self._args(store.base_dir, sid_a=sid_a, sid_b=sid_b)
        result = cmd_compare(args)
        self.assertEqual(result, 0)

    def test_compare_two_sessions_json(self):
        import io
        from unittest.mock import patch
        store = _make_store()
        sid_a = _add_session(store)
        sid_b = _add_session(store)
        args = self._args(store.base_dir, sid_a=sid_a, sid_b=sid_b, fmt="json")
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            result = cmd_compare(args)
        self.assertEqual(result, 0)
        data = json.loads(captured.getvalue())
        self.assertIn("verdict", data)
        self.assertIn("decision_divergence", data)

    def test_missing_session_a_returns_1(self):
        store = _make_store()
        args = self._args(store.base_dir, sid_a="nonexistent", sid_b="also-missing")
        result = cmd_compare(args)
        self.assertEqual(result, 1)

    def test_missing_session_b_returns_1(self):
        store = _make_store()
        sid_a = _add_session(store)
        args = self._args(store.base_dir, sid_a=sid_a, sid_b="nonexistent")
        result = cmd_compare(args)
        self.assertEqual(result, 1)

    def test_no_args_returns_1(self):
        store = _make_store()
        args = self._args(store.base_dir)
        result = cmd_compare(args)
        self.assertEqual(result, 1)

    def test_tag_compare(self):
        store = _make_store()
        _add_session(store, agent_name="refactor-auth")
        _add_session(store, agent_name="refactor-auth")
        args = self._args(store.base_dir, tag="refactor-auth", last=2)
        result = cmd_compare(args)
        self.assertEqual(result, 0)

    def test_tag_not_enough_sessions_returns_1(self):
        store = _make_store()
        _add_session(store, agent_name="refactor-auth")  # only 1
        args = self._args(store.base_dir, tag="refactor-auth", last=2)
        result = cmd_compare(args)
        self.assertEqual(result, 1)

    def test_rerun_without_prompt_returns_1(self):
        store = _make_store()
        sid_a = _add_session(store)  # no user_prompt
        args = self._args(store.base_dir, sid_a=sid_a, rerun=True)
        result = cmd_compare(args)
        self.assertEqual(result, 1)

    def test_rerun_with_prompt_returns_1_with_message(self):
        """--rerun is not yet automated; returns 1 with instructions."""
        import io
        from unittest.mock import patch
        store = _make_store()
        sid_a = _add_session(store, user_prompt="Fix the login bug")
        args = self._args(store.base_dir, sid_a=sid_a, rerun=True)
        captured = io.StringIO()
        with patch("sys.stderr", captured):
            result = cmd_compare(args)
        self.assertEqual(result, 1)
        self.assertIn("Fix the login bug", captured.getvalue())

    def test_json_output_includes_divergence(self):
        import io
        from unittest.mock import patch
        store = _make_store()
        sid_a = _add_session(store, decisions=["chose A"])
        sid_b = _add_session(store, decisions=["chose B"])
        args = self._args(store.base_dir, sid_a=sid_a, sid_b=sid_b, fmt="json")
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            cmd_compare(args)
        data = json.loads(captured.getvalue())
        self.assertIn("decision_divergence", data)
        self.assertGreaterEqual(data["decision_divergence"], 0)

    def test_text_output_includes_divergence_line(self):
        import io
        from unittest.mock import patch
        store = _make_store()
        sid_a = _add_session(store)
        sid_b = _add_session(store)
        args = self._args(store.base_dir, sid_a=sid_a, sid_b=sid_b, fmt="text")
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            cmd_compare(args)
        self.assertIn("Decision divergence", captured.getvalue())


if __name__ == "__main__":
    unittest.main()
