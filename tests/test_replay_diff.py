"""Tests for replay --diff dual-session HTML viewer (issue #139)."""

import io
import os
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace.cli import _run_handler, build_parser, cmd_replay
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.replay import replay_to_html_diff
from agent_trace.store import TraceStore

try:
    import _no_telemetry  # noqa: F401
    from test_html_replay import (
        LEGACY_ID, PAYLOAD, SKIP_LEGACY_ON_WINDOWS, check_script_json, write_legacy_session,
    )
    from test_store import NOT_A_SESSION_STATES, make_run1_not_a_session
except ImportError:  # pragma: no cover - package-style test discovery
    from tests import _no_telemetry  # noqa: F401
    from tests.test_html_replay import (
        LEGACY_ID, PAYLOAD, SKIP_LEGACY_ON_WINDOWS, check_script_json, write_legacy_session,
    )
    from tests.test_store import NOT_A_SESSION_STATES, make_run1_not_a_session


def _make_session(store: TraceStore, tool_names: list[str]) -> str:
    meta = SessionMeta(agent_name="test")
    meta.started_at = time.time() - 60
    meta.ended_at = time.time()
    store.create_session(meta)
    for name in tool_names:
        store.append_event(meta.session_id, TraceEvent(
            event_type=EventType.TOOL_CALL,
            session_id=meta.session_id,
            data={"tool_name": name},
        ))
    store.update_meta(meta)
    return meta.session_id


class TestReplayDiff(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = TraceStore(self.tmpdir)

    def test_returns_html_string(self):
        sid_a = _make_session(self.store, ["Bash", "Read"])
        sid_b = _make_session(self.store, ["Bash", "Write"])
        html = replay_to_html_diff(self.store, sid_a, sid_b)
        self.assertIsInstance(html, str)
        self.assertIn("<!DOCTYPE html>", html)

    def test_both_session_ids_in_output(self):
        sid_a = _make_session(self.store, ["Bash"])
        sid_b = _make_session(self.store, ["Read"])
        html = replay_to_html_diff(self.store, sid_a, sid_b)
        self.assertIn(sid_a[:16], html)
        self.assertIn(sid_b[:16], html)

    def test_event_counts_in_output(self):
        sid_a = _make_session(self.store, ["Bash", "Read", "Write"])
        sid_b = _make_session(self.store, ["Bash"])
        html = replay_to_html_diff(self.store, sid_a, sid_b)
        # A has 3 events — the count should appear in the stats section
        self.assertIn("<b>3</b>", html)

    def test_writes_to_output_path(self):
        sid_a = _make_session(self.store, ["Bash"])
        sid_b = _make_session(self.store, ["Read"])
        out = os.path.join(self.tmpdir, "diff.html")
        replay_to_html_diff(self.store, sid_a, sid_b, output_path=out)
        self.assertTrue(os.path.exists(out))
        content = open(out).read()
        self.assertIn("<!DOCTYPE html>", content)

    def test_only_a_ids_in_output(self):
        sid_a = _make_session(self.store, ["UniqueToolA"])
        sid_b = _make_session(self.store, ["UniqueToolB"])
        html = replay_to_html_diff(self.store, sid_a, sid_b)
        # JS variables onlyA / onlyB must be present
        self.assertIn("onlyA", html)
        self.assertIn("onlyB", html)

    def test_empty_sessions_dont_crash(self):
        sid_a = _make_session(self.store, [])
        sid_b = _make_session(self.store, [])
        html = replay_to_html_diff(self.store, sid_a, sid_b)
        self.assertIn("<!DOCTYPE html>", html)


class TestReplayDiffShowsTextAsText(unittest.TestCase):
    """Event text, event ids and session ids reach the diff page as text."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = TraceStore(tmp.name)

    def _payload_session(self, label: str) -> tuple[str, list]:
        """Store events whose text, tool name and id hold PAYLOAD.

        Returns the session id and the events value the diff page embeds.
        """
        meta = SessionMeta(agent_name="payload")
        self.store.create_session(meta)
        tool = f"{label} {PAYLOAD}"
        rows = [
            (EventType.TOOL_CALL, 1000.0,
             {"tool_name": tool, "arguments": {"command": PAYLOAD}}),
            (EventType.TOOL_RESULT, 1000.5, {"tool_name": tool, "content": PAYLOAD}),
            (EventType.USER_PROMPT, 1001.25, {"content": PAYLOAD}),
        ]
        embedded = []
        for n, (event_type, ts, data) in enumerate(rows, 1):
            event_id = f"{label}{n} {PAYLOAD}"
            self.store.append_event(meta.session_id, TraceEvent(
                event_type=event_type, session_id=meta.session_id, timestamp=ts,
                event_id=event_id, data=dict(data),
            ))
            embedded.append({"event_type": event_type.value, "event_id": event_id,
                             "_offset": ts - 1000.0, "duration_ms": None, "data": data})
        return meta.session_id, embedded

    def test_embedded_values_are_inert_and_unchanged(self):
        sid_a, events_a = self._payload_session("A")
        sid_b, events_b = self._payload_session("B")

        page = replay_to_html_diff(self.store, sid_a, sid_b)

        # The user prompts share a key, so only the tool events are one-sided.
        check_script_json(self, page, {
            r"^const eventsA = (.*);$": events_a,
            r"^const eventsB = (.*);$": events_b,
            r"^const onlyA = new Set\((.*)\);$": [f"A1 {PAYLOAD}", f"A2 {PAYLOAD}"],
            r"^const onlyB = new Set\((.*)\);$": [f"B1 {PAYLOAD}", f"B2 {PAYLOAD}"],
        })

    @SKIP_LEGACY_ON_WINDOWS
    def test_legacy_session_ids_show_as_text(self):
        other = "M<\"&b #?%21&>'c&tail"
        for sid in (LEGACY_ID, other):
            write_legacy_session(self.store, sid, [TraceEvent(
                event_type=EventType.TOOL_CALL, session_id=sid, data={"tool_name": "Bash"},
            )])

        page = replay_to_html_diff(self.store, LEGACY_ID, other)

        a = "L&lt;&quot;&amp;a #?%20&amp;&gt;&#x27;b&amp;"
        b = "M&lt;&quot;&amp;b #?%21&amp;&gt;&#x27;c&amp;"
        self.assertIn(f"<title>agent-strace diff: {a} vs {b}</title>", page)
        self.assertIn(f'<div class="subtitle">{a} &nbsp;vs&nbsp; {b}</div>', page)
        self.assertIn(f"{a}</h2>", page)
        self.assertIn(f"{b}</h2>", page)
        self.assertNotIn('L<"', page)
        self.assertNotIn('M<"', page)


class TestReplayDiffNamesTheExactSession(unittest.TestCase):
    """replay other --diff run1 never diffs against run10 in run1's place."""

    def setUp(self):
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("AGENT_STRACE_WORKSPACE", None)
        os.environ.pop("AGENT_TRACE_DIR", None)

    def test_run1_that_is_not_a_session_is_not_found(self):
        # Before, the diff used run10.
        for state in NOT_A_SESSION_STATES:
            with self.subTest(state=state):
                tmp = tempfile.TemporaryDirectory()
                self.addCleanup(tmp.cleanup)
                trace_dir = Path(tmp.name) / "traces"
                store = TraceStore(str(trace_dir), use_workspace_env=False)
                for session_id in ("other", "run1", "run10"):
                    store.create_session(SessionMeta(session_id=session_id, started_at=1.0))
                    store.append_event(session_id, TraceEvent(
                        EventType.TOOL_CALL, timestamp=1.0, data={"tool_name": "Bash"},
                    ))
                make_run1_not_a_session(trace_dir, Path(tmp.name), state)
                output = Path(tmp.name) / "diff.html"
                args = build_parser().parse_args([
                    "--trace-dir", str(trace_dir),
                    "replay", "other", "--diff", "run1", "-o", str(output),
                ])
                out, err = io.StringIO(), io.StringIO()
                with redirect_stdout(out), redirect_stderr(err):
                    code = _run_handler(args, cmd_replay)
                self.assertEqual(
                    (code, out.getvalue(), err.getvalue()),
                    (1, "", "Session not found: run1\n"),
                )
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
