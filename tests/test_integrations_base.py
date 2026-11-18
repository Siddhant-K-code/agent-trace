"""Tests for the shared integration helpers in integrations/_base.py."""

from __future__ import annotations

import contextvars
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_trace.integrations import _base
from agent_trace.models import EventType
from agent_trace.store import TraceStore

try:  # keeps product telemetry off in this process and its children
    import _no_telemetry  # noqa: F401
except ImportError:  # pragma: no cover - run as a package module
    from tests import _no_telemetry  # noqa: F401

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

    def marker(self):
        path = Path(self.trace_dir) / ".active-session.agent"
        return path.read_text(encoding="utf-8") if path.exists() else None

    def start_run_in_new_context(self):
        """Start a run in a context of its own; return its ID and a function that ends it."""
        context = contextvars.Context()
        run = _base.run_session(self.store, "agent")
        sid = context.run(run.__enter__)
        return sid, lambda: context.run(run.__exit__, None, None, None)

    def session_in_fresh_thread(self):
        """The session an event from a thread outside any run goes to."""
        found = []
        thread = threading.Thread(
            target=lambda: found.append(_base._get_or_create_session(self.store, "agent")))
        thread.start()
        thread.join(timeout=60)
        return found[0]

    def test_three_overlapping_runs_finishing_out_of_order_restore_oldest_marker(self):
        first, end_first = self.start_run_in_new_context()
        second, end_second = self.start_run_in_new_context()
        third, end_third = self.start_run_in_new_context()
        self.assertEqual(self.marker(), third)
        end_second()
        self.assertEqual(self.marker(), third)
        end_third()
        self.assertEqual(self.marker(), first)
        self.assertEqual(self.session_in_fresh_thread(), first)
        end_first()
        self.assertEqual(self.marker(), None)
        self.assertEqual(list(Path(self.trace_dir).glob(".integration-runs*")), [])
        for sid in (first, second, third):
            self.assertIsNotNone(self.store.load_meta(sid).ended_at)

    def run_in_child(self, script):
        env = dict(os.environ)
        src = str(Path(__file__).resolve().parents[1] / "src")
        env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
        child = subprocess.Popen(
            [sys.executable, "-c",
             "import sys\n"
             "from agent_trace.integrations import _base\n"
             "from agent_trace.store import TraceStore\n"
             "run = _base.run_session(TraceStore(sys.argv[1]), 'agent')\n"
             "print(run.__enter__(), flush=True)\n" + script,
             self.trace_dir],
            env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        )

        def stop():
            if child.poll() is None:
                child.kill()
            child.wait(timeout=60)
            for pipe in (child.stdin, child.stdout):
                if not pipe.closed:
                    pipe.close()

        self.addCleanup(stop)
        sid = child.stdout.readline().strip()
        self.assertTrue(sid)
        return child, sid

    def test_run_in_another_process_keeps_the_marker_while_it_runs(self):
        child, child_sid = self.run_in_child(
            "sys.stdin.readline()\nrun.__exit__(None, None, None)\n")
        with _base.run_session(self.store, "agent") as sid:
            self.assertEqual(self.marker(), sid)
        self.assertEqual(self.marker(), child_sid)
        self.assertIsNone(self.store.load_meta(child_sid).ended_at)
        child.communicate("\n", timeout=60)
        self.assertEqual(child.returncode, 0)
        self.assertEqual(self.marker(), None)
        self.assertIsNotNone(self.store.load_meta(child_sid).ended_at)

    def test_subprocess_run_restores_parent_process_marker(self):
        with _base.run_session(self.store, "agent") as sid:
            child, child_sid = self.run_in_child("run.__exit__(None, None, None)\n")
            child.communicate(timeout=60)
            self.assertEqual(child.returncode, 0)
            self.assertIsNotNone(self.store.load_meta(child_sid).ended_at)
            self.assertEqual(self.marker(), sid)
            self.assertEqual(self.session_in_fresh_thread(), sid)
        self.assertEqual(self.marker(), None)

    def test_run_of_a_process_that_died_is_dropped(self):
        child, child_sid = self.run_in_child("import os\nos._exit(0)\n")
        child.wait(timeout=60)
        self.assertEqual(self.marker(), child_sid)
        with _base.run_session(self.store, "agent"):
            pass
        self.assertEqual(self.marker(), None)

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
