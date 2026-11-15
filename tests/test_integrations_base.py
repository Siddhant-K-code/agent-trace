"""Tests for the shared integration helpers in integrations/_base.py."""

from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_trace.integrations import _base
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


if __name__ == "__main__":
    unittest.main()
