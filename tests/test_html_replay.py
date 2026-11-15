"""Tests for issue #40: web-based HTML session replay viewer."""

from __future__ import annotations

import json
import os
import re
import tempfile
import unittest
from unittest.mock import patch

from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore

try:
    from test_dashboard import _codex_session
except ImportError:  # pragma: no cover - package-style test discovery
    from tests.test_dashboard import _codex_session


def _make_store(tmp_dir: str) -> TraceStore:
    return TraceStore(os.path.join(tmp_dir, "traces"))


def _make_session(store: TraceStore) -> str:
    meta = SessionMeta(agent_name="test-agent")
    store.create_session(meta)
    sid = meta.session_id
    for event in [
        TraceEvent(event_type=EventType.SESSION_START, session_id=sid, data={}),
        TraceEvent(
            event_type=EventType.TOOL_CALL, session_id=sid,
            data={"tool_name": "read_file", "arguments": {"file_path": "src/auth.py"}},
        ),
        TraceEvent(
            event_type=EventType.LLM_REQUEST, session_id=sid,
            data={"model": "claude-opus-4-6", "messages": [{"role": "user", "content": "fix it"}]},
        ),
        TraceEvent(
            event_type=EventType.TOOL_CALL, session_id=sid,
            data={"tool_name": "write", "arguments": {"file_path": "src/auth.py", "new_str": "fixed"}},
        ),
        TraceEvent(event_type=EventType.SESSION_END, session_id=sid, data={}),
    ]:
        store.append_event(sid, event)
    return sid


class TestHTMLReplay(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_replay_to_html_returns_string(self):
        from agent_trace.replay import replay_to_html
        store = _make_store(self._tmp)
        sid = _make_session(store)
        html = replay_to_html(store, sid)
        self.assertIsInstance(html, str)
        self.assertGreater(len(html), 100)

    def test_html_is_valid_document(self):
        from agent_trace.replay import replay_to_html
        store = _make_store(self._tmp)
        sid = _make_session(store)
        html = replay_to_html(store, sid)
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("<html", html)
        self.assertIn("</html>", html)

    def test_html_contains_session_id(self):
        from agent_trace.replay import replay_to_html
        store = _make_store(self._tmp)
        sid = _make_session(store)
        html = replay_to_html(store, sid)
        self.assertIn(sid[:12], html)

    def test_html_contains_events_json(self):
        from agent_trace.replay import replay_to_html
        store = _make_store(self._tmp)
        sid = _make_session(store)
        html = replay_to_html(store, sid)
        self.assertIn("const EVENTS =", html)
        self.assertIn("tool_call", html)

    def test_html_contains_cost_counter(self):
        from agent_trace.replay import replay_to_html
        store = _make_store(self._tmp)
        sid = _make_session(store)
        html = replay_to_html(store, sid)
        self.assertIn("cost", html.lower())

    def test_html_contains_play_controls(self):
        from agent_trace.replay import replay_to_html
        store = _make_store(self._tmp)
        sid = _make_session(store)
        html = replay_to_html(store, sid)
        self.assertIn("togglePlay", html)
        self.assertIn("scrubber", html)

    def test_html_written_to_file(self):
        from agent_trace.replay import replay_to_html
        store = _make_store(self._tmp)
        sid = _make_session(store)
        output_path = os.path.join(self._tmp, "replay.html")
        replay_to_html(store, sid, output_path=output_path)
        self.assertTrue(os.path.exists(output_path))
        content = open(output_path).read()
        self.assertIn("<!DOCTYPE html>", content)

    def test_html_no_events_graceful(self):
        from agent_trace.replay import replay_to_html
        store = _make_store(self._tmp)
        meta = SessionMeta(agent_name="empty")
        store.create_session(meta)
        html = replay_to_html(store, meta.session_id)
        self.assertIsInstance(html, str)

    def test_cli_has_html_format_flag(self):
        from agent_trace.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["replay", "--format", "html", "--output", "out.html"])
        self.assertEqual(args.format, "html")
        self.assertEqual(args.output, "out.html")


class TestHTMLReplayCost(unittest.TestCase):
    """The cost counter ends at the `agent-strace cost` total, or shows n/a."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = _make_store(tmp.name)

    @staticmethod
    def _costs(page: str) -> list:
        return json.loads(re.search(r"const COSTS = (\[.*?\]);", page).group(1))

    def test_counter_ends_at_session_cost(self):
        from agent_trace.cost import session_cost
        from agent_trace.replay import replay_to_html
        sid = _codex_session(self.store)
        page = replay_to_html(self.store, sid)
        costs = self._costs(page)
        self.assertEqual(costs[-1], session_cost(self.store.load_events(sid)))
        self.assertEqual(costs, sorted(costs))
        self.assertIn('<span id="cost">$0.0000</span>', page)

    def test_unpriceable_session_shows_na(self):
        from agent_trace.replay import replay_to_html
        sid = _codex_session(self.store, prompt_data=[])
        page = replay_to_html(self.store, sid)
        costs = self._costs(page)
        self.assertEqual(costs, [None] * len(self.store.load_events(sid)))
        self.assertIn('<span id="cost">n/a</span>', page)
        self.assertIn("cost === null ? 'n/a'", page)

    def test_pricing_bugs_are_not_hidden(self):
        from agent_trace.replay import replay_to_html
        sid = _codex_session(self.store)
        with patch("agent_trace.cost.session_cost", side_effect=TypeError("bug")):
            with self.assertRaises(TypeError):
                replay_to_html(self.store, sid)
