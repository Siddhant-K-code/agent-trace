"""Tests for the shared integration helpers in integrations/_base.py."""

from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_trace.integrations import _base
from agent_trace.models import EventType
from agent_trace.store import TraceStore

try:
    from _concurrency import run_concurrently
except ImportError:  # pragma: no cover - run as a package module
    from tests._concurrency import run_concurrently


class TestGetOrCreateSession(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.trace_dir = tmp.name
        self.store = TraceStore(self.trace_dir)

    def test_reuses_the_active_session(self):
        first = _base._get_or_create_session(self.store, "agent")
        second = _base._get_or_create_session(self.store, "agent")
        self.assertEqual(first, second)
        self.assertEqual(self.store.list_sessions()[0].session_id, first)

    def test_agent_names_keep_separate_sessions(self):
        a = _base._get_or_create_session(self.store, "agent-a")
        b = _base._get_or_create_session(self.store, "agent-b")
        self.assertNotEqual(a, b)
        marker = Path(self.trace_dir) / ".active-session.agent-a"
        self.assertEqual(marker.read_text(encoding="utf-8"), a)

    def test_concurrent_threads_share_one_session(self):
        threads = 8
        barrier = threading.Barrier(threads)
        results: list[str] = []
        errors: list[BaseException] = []
        original = TraceStore.create_session

        # Widen the gap between the marker check and the marker write so a
        # missing lock reliably lets several threads create a session.
        def slow_create(store, meta, **kwargs):
            time.sleep(0.05)
            return original(store, meta, **kwargs)

        def worker():
            try:
                store = TraceStore(self.trace_dir)
                barrier.wait()
                results.append(_base._get_or_create_session(store, "agent"))
            except BaseException as exc:  # pragma: no cover - reported below
                errors.append(exc)

        with patch.object(TraceStore, "create_session", slow_create):
            pool = [threading.Thread(target=worker) for _ in range(threads)]
            for thread in pool:
                thread.start()
            for thread in pool:
                thread.join(timeout=60)

        self.assertEqual(errors, [])
        self.assertEqual(len(results), threads)
        self.assertEqual(len(set(results)), 1)
        self.assertEqual(len(self.store.list_sessions()), 1)

    def test_concurrent_processes_share_one_session(self):
        script = (
            "from agent_trace.integrations import _base\n"
            "from agent_trace.store import TraceStore\n"
            "store = TraceStore(sys.argv[2])\n"
            "wait_for_start()\n"
            "print(_base._get_or_create_session(store, 'agent'))\n"
        )
        outputs = run_concurrently(script, [[self.trace_dir]] * 4)

        self.assertEqual(len({out.strip() for out in outputs}), 1)
        self.assertEqual(len(self.store.list_sessions()), 1)

    def test_lock_file_is_not_an_active_session_marker(self):
        _base._get_or_create_session(self.store, "agent")
        markers = sorted(p.name for p in Path(self.trace_dir).glob(".active-session*"))
        self.assertEqual(markers, [".active-session.agent"])


class Interrupted(BaseException):
    pass


class TestRunSession(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.trace_dir = tmp.name
        self.store = TraceStore(self.trace_dir)
        env = {k: v for k, v in os.environ.items() if k != "AGENT_STRACE_ENDPOINT"}
        environ = patch.dict(os.environ, env, clear=True)
        environ.start()
        self.addCleanup(environ.stop)

    def markers(self):
        return sorted(p.name for p in Path(self.trace_dir).glob(".active-session*"))

    def kinds(self, session_id):
        return [e.event_type for e in self.store.load_events(session_id)]

    def test_run_is_closed_and_its_marker_cleared(self):
        with _base.run_session(self.store, "agent", agent_name="crew") as sid:
            self.assertEqual(_base._get_or_create_session(self.store, "agent"), sid)
            self.assertEqual(self.markers(), [".active-session.agent"])
            _base.emit(EventType.TOOL_CALL, sid, self.store, tool_name="t")
            _base.emit(EventType.LLM_REQUEST, sid, self.store, model="m")
        meta = self.store.load_meta(sid)
        self.assertIsNotNone(meta.ended_at)
        self.assertEqual(meta.total_duration_ms, (meta.ended_at - meta.started_at) * 1000)
        self.assertEqual((meta.tool_calls, meta.llm_requests, meta.errors), (1, 1, 0))
        self.assertEqual(self.markers(), [])
        self.assertEqual(self.store.load_events(sid)[0].data, {"agent_name": "crew"})
        self.assertNotEqual(_base._get_or_create_session(self.store, "agent"), sid)

    def test_interrupted_run_records_its_error_and_end(self):
        with self.assertRaises(Interrupted):
            with _base.run_session(self.store, "agent") as sid:
                raise Interrupted()
        self.assertEqual(self.kinds(sid), [EventType.SESSION_START, EventType.ERROR,
                                           EventType.SESSION_END])
        self.assertEqual(self.store.load_events(sid)[1].data["error_type"], "Interrupted")
        meta = self.store.load_meta(sid)
        self.assertIsNotNone(meta.ended_at)
        self.assertEqual(meta.errors, 1)
        self.assertEqual(self.markers(), [])

    def test_nested_run_records_into_the_outer_session(self):
        with _base.run_session(self.store, "agent") as outer:
            with _base.run_session(self.store, "agent") as inner:
                self.assertEqual(inner, outer)
            with _base.run_session(self.store, "other") as other:
                self.assertNotEqual(other, outer)
            self.assertIsNone(self.store.load_meta(outer).ended_at)
        self.assertEqual(self.kinds(outer), [EventType.SESSION_START, EventType.SESSION_END])
        self.assertEqual(len(self.store.list_sessions()), 2)

    def test_overlapping_runs_keep_the_marker_on_a_running_session(self):
        first = _base.run_session(self.store, "agent")
        first_sid = first.__enter__()
        # A second run in another thread starts after the first and ends
        # before it, while the first is still running.
        result = {}

        def second_run():
            with _base.run_session(self.store, "agent") as sid:
                result["sid"] = sid

        thread = threading.Thread(target=second_run)
        thread.start()
        thread.join(timeout=60)
        marker = Path(self.trace_dir) / ".active-session.agent"
        self.assertNotEqual(result["sid"], first_sid)
        self.assertEqual(marker.read_text(encoding="utf-8"), first_sid)
        first.__exit__(None, None, None)
        self.assertEqual(self.markers(), [])

    def test_failure_to_record_the_end_never_reaches_the_caller(self):
        with patch.object(TraceStore, "mutate_meta", side_effect=OSError("disk full")):
            with _base.run_session(self.store, "agent") as sid:
                pass
            with self.assertRaises(RuntimeError):
                with _base.run_session(self.store, "agent"):
                    raise RuntimeError("boom")
        self.assertEqual(self.kinds(sid)[-1], EventType.SESSION_END)
        self.assertEqual(self.markers(), [])


if __name__ == "__main__":
    unittest.main()
