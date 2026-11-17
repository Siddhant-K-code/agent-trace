"""Tests for agent-strace compare (Issue #114)."""

from __future__ import annotations

import argparse
import ast
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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

try:
    import _escapes
except ImportError:
    from tests import _escapes


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


def _compare_args(store_dir, sid_a, sid_b=None, fmt="text", rerun=False):
    return argparse.Namespace(
        trace_dir=store_dir, session_id_a=sid_a, session_id_b=sid_b, format=fmt,
        tag=None, last=2, rerun=rerun, model=None,
    )


def _compare_stdout(agent_name, fmt="text"):
    """Return what compare writes for two sessions of agent *agent_name*."""
    with tempfile.TemporaryDirectory() as tmp:
        store = TraceStore(Path(tmp))
        sid_a = _add_session(store, agent_name=agent_name)
        sid_b = _add_session(store, agent_name=agent_name)
        out = io.StringIO()
        with patch("sys.stdout", out):
            cmd_compare(_compare_args(store.base_dir, sid_a, sid_b, fmt=fmt))
        return out.getvalue()


def _rerun_stderr(prompt):
    """Return what compare --rerun writes to stderr for a session with *prompt*."""
    with tempfile.TemporaryDirectory() as tmp:
        store = TraceStore(Path(tmp))
        sid = _add_session(store, user_prompt=prompt)
        err = io.StringIO()
        with patch("sys.stderr", err):
            cmd_compare(_compare_args(store.base_dir, sid, rerun=True))
        return err.getvalue()


def _rerun_prompt_block(prompt):
    """Return the lines of the prompt that compare --rerun prints. Each one
    starts with two spaces, so a blank line ends the block."""
    _header, block, _footer = _rerun_stderr(prompt).split("\n\n")
    return block.split("\n")


def _read_rerun_line(line):
    """Read a printed prompt line, without its indent, as a Python string literal."""
    return ast.literal_eval('"' + line.replace('"', '\\"') + '"')


# Prompts that must read back exactly from what --rerun prints. Without the
# doubled backslash, a tab and a typed backslash-t would print the same, and
# a typed backslash-N escape would read back as the character it names.
RERUN_PROMPTS = [
    ("tab beside a typed backslash-t", "a\tb\\tc"),
    ("typed double backslash", "a\\\\b"),
    ("trailing backslash", "a\\"),
    ("backslash before a line break", "a\\\nb"),
    ("backslash before a quote", 'a\\"b'),
    ("ESC beside a typed backslash-x1b", "\x1b[2J\\x1b[2J"),
    ("typed backslash-N escape", "\\N{BULLET}"),
    ("CR", "a\rb"),
    ("CRLF", "a\r\nb"),
    ("line separator", "a\u2028b"),
    ("right-to-left override", "a\u202eb"),
    ("NUL", "a\x00b"),
    ("DEL", "a\x7fb"),
    ("next line", "a\x85b"),
    ("C1 CSI", "a\x9bb"),
    ("accent", "café"),
    ("CJK", "日本語"),
    ("emoji", "😀"),
    ("language tag", "a\U000e0001b"),
    ("lone surrogate", "a\ud800b"),
    ("empty first and last line", "\nmiddle\n"),
    ("indentation", "def f():\n    return 1"),
    ("both quotes", "say \"hi\" and 'bye'"),
    ("escape payload", _escapes.PAYLOAD),
]


def _failure_stderr(message):
    """Return what compare writes to stderr when the comparison raises *message*."""
    with tempfile.TemporaryDirectory() as tmp:
        store = TraceStore(Path(tmp))
        sid_a = _add_session(store)
        sid_b = _add_session(store)
        err = io.StringIO()
        with patch("agent_trace.compare.compare_sessions", side_effect=RuntimeError(message)):
            with patch("sys.stderr", err):
                cmd_compare(_compare_args(store.base_dir, sid_a, sid_b))
        return err.getvalue()


class TestTraceTextShowsAsText(unittest.TestCase):
    """An agent name, a stored prompt or an error can hold escape codes and
    line breaks. compare shows them as text."""

    # (name, cases, render). Each case is (label, kind, build), and
    # build(value) returns what render takes. The kinds are in
    # tests/_escapes.py. A label shows cut at 10 characters, so the agent
    # name takes the short kind. The --rerun prompt keeps its lines, so its
    # tests are below.
    VIEWS = [
        ("text output", [("agent_name", "short", lambda v: v)], _compare_stdout),
        ("failed comparison", [("exception text", "free", lambda v: v)], _failure_stderr),
    ]

    def test_each_value_shows_escapes_as_text(self):
        for name, cases, render in self.VIEWS:
            with self.subTest(name):
                _escapes.check_each_value(self, cases, render)

    def test_plain_text_prints_unchanged(self):
        for name, cases, render in self.VIEWS:
            with self.subTest(name):
                _escapes.check_plain_text_unchanged(self, cases, render)

    def test_whitespace_in_free_text_shows_as_one_space(self):
        for name, cases, render in self.VIEWS:
            with self.subTest(name):
                _escapes.check_whitespace_in_free_text(self, cases, render)

    def test_sgr_code_shows_as_text(self):
        output = _rerun_stderr("a\x1b[8mhidden")
        self.assertIn(r"a\x1b[8mhidden", output)
        self.assertNotIn("\x1b", output)

    def test_rerun_prompt_shows_escapes_as_text_and_keeps_its_lines(self):
        # Each line of the prompt goes through inert() on its own and starts
        # with two spaces, so the LF in it still starts a line.
        output = _rerun_stderr(_escapes.PAYLOAD)
        for line in output.split("\n"):
            self.assertTrue(line.isprintable(), ascii(line))
        self.assertIn("\n  " + r"A\x1b]52;c;aWQK\x07B\x1b[2K\rC" + "\n  " + r"D\x9bE\u202eF" + "\n", output)
        self.assertEqual(output.count("\n"), _rerun_stderr("plain").count("\n") + 1)

    def test_rerun_prompt_keeps_indentation_and_spaces(self):
        prompt = "Fix the bug:\n    def f():\n        return 1\n\n  - " + _escapes.SPACES + "\xa0end"
        self.assertIn(
            "\n\n"
            "  Fix the bug:\n"
            "      def f():\n"
            "          return 1\n"
            "  \n"
            r"    - one\ttwo  three\u3000four\xa0end" "\n\n",
            _rerun_stderr(prompt),
        )

    def test_rerun_prompt_starts_a_line_only_at_an_lf(self):
        # str.splitlines() would also start a line at a CR, a form feed or a
        # Unicode line separator, and drop which one it was.
        output = _rerun_stderr("a\r\nb\rc\x0bd\x0ce\x1cf\x85g\u2028h\u2029i")
        self.assertIn("\n  " + r"a\r" + "\n  " + r"b\rc\x0bd\x0ce\x1cf\x85g\u2028h\u2029i" + "\n", output)

    def test_rerun_prompt_in_plain_text_shows_as_it_is(self):
        self.assertIn(_escapes.PLAIN, _rerun_stderr(_escapes.PLAIN))

    def test_rerun_prompt_cannot_pass_for_a_line_of_the_message(self):
        output = _rerun_stderr(
            "x\n[compare] Done.\n\nRun the agent with this prompt, then compare the two sessions manually."
        )
        unindented = [line for line in output.split("\n") if line and not line.startswith("  ")]
        self.assertEqual(len(unindented), 2, unindented)
        self.assertTrue(unindented[0].startswith("[compare] --rerun is not yet automated."))
        self.assertEqual(unindented[1], "Run the agent with this prompt, then compare the two sessions manually.")

    def test_rerun_prompt_reads_back_to_the_exact_prompt(self):
        # Each backslash shows as two, so every backslash on a printed line
        # starts an escape. Read as a Python string literal, each printed
        # line gives back its line of the prompt.
        for label, prompt in RERUN_PROMPTS:
            with self.subTest(label):
                lines = _rerun_prompt_block(prompt)
                for line in lines:
                    self.assertTrue(line.isprintable(), ascii(line))
                    self.assertTrue(line.startswith("  "), ascii(line))
                read_back = "\n".join(_read_rerun_line(line[2:]) for line in lines)
                self.assertEqual(read_back, prompt)

    def test_rerun_prompt_with_a_tab_and_with_a_typed_backslash_t_print_differently(self):
        self.assertEqual(_rerun_prompt_block("a\tb"), [r"  a\tb"])
        self.assertEqual(_rerun_prompt_block("a\\tb"), [r"  a\\tb"])

    def test_json_keeps_the_raw_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = TraceStore(Path(tmp))
            sid_a = _add_session(store, agent_name=_escapes.MID, write_files=[_escapes.PAYLOAD])
            sid_b = _add_session(store, agent_name=_escapes.MID, write_files=["plain"])
            out = io.StringIO()
            with patch("sys.stdout", out):
                cmd_compare(_compare_args(store.base_dir, sid_a, sid_b, fmt="json"))
        data = json.loads(out.getvalue())
        self.assertEqual((data["label_a"], data["label_b"]), (_escapes.MID, _escapes.MID))
        self.assertIn(_escapes.PAYLOAD, data["divergence_points"][0]["description_a"])


if __name__ == "__main__":
    unittest.main()
