"""Tests for the shared integration helpers in integrations/_base.py."""

from __future__ import annotations

import contextvars
import json
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
from agent_trace.models import EventType, TraceEvent
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

    def test_events_another_process_records_into_the_run_are_counted(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
        with _base.run_session(self.store, "agent") as sid:
            _base.emit(EventType.TOOL_CALL, sid, self.store, tool_name="here")
            subprocess.run(
                [sys.executable, "-c",
                 "import sys\n"
                 "from agent_trace.integrations import _base\n"
                 "from agent_trace.models import EventType\n"
                 "from agent_trace.store import TraceStore\n"
                 "store = TraceStore(sys.argv[1])\n"
                 "sid = _base._get_or_create_session(store, 'agent')\n"
                 "assert sid == sys.argv[2], sid\n"
                 "for kind in (EventType.TOOL_CALL, EventType.LLM_REQUEST, EventType.ERROR):\n"
                 "    _base.emit(kind, sid, store)\n",
                 self.trace_dir, sid],
                env=env, check=True, timeout=60,
            )
        meta = self.store.load_meta(sid)
        kinds = self.kinds(sid)
        self.assertEqual((meta.tool_calls, meta.llm_requests, meta.errors), (
            kinds.count(EventType.TOOL_CALL), kinds.count(EventType.LLM_REQUEST),
            kinds.count(EventType.ERROR)))
        self.assertEqual((meta.tool_calls, meta.llm_requests, meta.errors), (2, 1, 1))

    LATE_CHILD = (
        "import json, sys\n"
        "from agent_trace import server\n"
        "from agent_trace.integrations import _base\n"
        "from agent_trace.models import EventType\n"
        "from agent_trace.store import TraceStore\n"
        "def send(event, endpoint):\n"
        "    with open(sys.argv[2], 'a', encoding='utf-8') as f:\n"
        "        f.write(event.to_json() + '\\n')\n"
        "    return True\n"
        "server.send_event_to_endpoint = send\n"
        "store = TraceStore(sys.argv[1])\n"
        "sid = _base._get_or_create_session(store, 'agent')\n"
        "call = _base.emit(EventType.TOOL_CALL, sid, store, tool_name='slow')\n"
        "print(sid, flush=True)\n"
        "sys.stdin.readline()\n"
        "late = [_base.emit(EventType.TOOL_RESULT, sid, store, parent=call)]\n"
        "for kind in (EventType.TOOL_CALL, EventType.LLM_REQUEST, EventType.ERROR):\n"
        "    late.append(_base.emit(kind, sid, store))\n"
        "print(json.dumps([call.event_id] + [e.session_id for e in late]), flush=True)\n"
    )

    def run_with_late_child(self, endpoint):
        """Run a session while a child process looks it up and records a
        tool_call into it, then, after the run ended, its tool_result and
        three more events. Returns the run's session ID, the tool_call's ID
        and the events, by session, in the order they were written."""
        log = Path(self.trace_dir) / "endpoint.ndjson"
        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
        env.pop("AGENT_STRACE_ENDPOINT", None)
        if endpoint:
            env["AGENT_STRACE_ENDPOINT"] = "http://collector.invalid"

        def send(event, url):
            with open(log, "a", encoding="utf-8") as f:
                f.write(event.to_json() + "\n")
            return True

        from agent_trace import server
        with patch.dict(os.environ, env), patch.object(server, "send_event_to_endpoint", send):
            with _base.run_session(self.store, "agent") as sid:
                child = subprocess.Popen(
                    [sys.executable, "-c", self.LATE_CHILD, self.trace_dir, str(log)],
                    env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                )
                self.addCleanup(child.wait, 60)
                self.addCleanup(child.stdout.close)
                self.assertEqual(child.stdout.readline().strip(), sid)
            child.stdin.write("\n")
            child.stdin.close()
            call_id, *late = json.loads(child.stdout.read())
            self.assertEqual(child.wait(timeout=60), 0)

        self.assertEqual(len(set(late)), 1)
        self.assertNotEqual(late[0], sid)
        if endpoint:
            events = {}
            for line in log.read_text(encoding="utf-8").splitlines():
                event = TraceEvent.from_json(line)
                events.setdefault(event.session_id, []).append(event)
            self.assertEqual(self.store.load_events(sid), [])
        else:
            events = {s: self.store.load_events(s) for s in (sid, late[0])}
        return sid, call_id, events[sid], events[late[0]]

    def check_late_events(self, endpoint):
        sid, call_id, run_events, late_events = self.run_with_late_child(endpoint)
        self.assertEqual([e.event_type for e in run_events], [
            EventType.SESSION_START, EventType.TOOL_CALL, EventType.SESSION_END])
        self.assertEqual(run_events[1].event_id, call_id)
        # The late events are kept, in another session of the agent.
        self.assertEqual([e.event_type for e in late_events], [
            EventType.TOOL_RESULT, EventType.TOOL_CALL, EventType.LLM_REQUEST,
            EventType.ERROR])
        for event in late_events:
            self.assertEqual(event.data["original_session_id"], sid)
        # The result does not name a request in another session.
        result = late_events[0]
        self.assertEqual((result.parent_id, result.duration_ms), ("", None))
        self.assertEqual(result.data["original_parent_id"], call_id)
        self.assertNotIn("original_parent_id", late_events[1].data)
        return sid

    def test_event_another_process_records_after_the_run_ends_is_not_appended_to_it(self):
        sid = self.check_late_events(endpoint=False)
        meta = self.store.load_meta(sid)
        self.assertEqual((meta.tool_calls, meta.llm_requests, meta.errors), (1, 0, 0))

    def test_event_another_process_sends_after_the_run_ends_is_not_sent_for_it(self):
        self.check_late_events(endpoint=True)

    def test_event_recorded_while_the_run_closes_waits_and_is_not_appended_to_it(self):
        real = self.store.mutate_meta
        late = []
        blocked = []
        thread = None

        def mutate_meta(sid, mutate, **kwargs):
            nonlocal thread
            if getattr(mutate, "__name__", "") == "close":
                thread = threading.Thread(target=lambda: late.append(
                    _base.emit(EventType.TOOL_CALL, sid, self.store).session_id))
                thread.start()
                thread.join(0.5)
                blocked.append(thread.is_alive())
            return real(sid, mutate, **kwargs)

        with patch.object(self.store, "mutate_meta", mutate_meta):
            with _base.run_session(self.store, "agent") as sid:
                pass
        thread.join(60)
        self.assertEqual(blocked, [True])
        self.assertEqual(self.kinds(sid), [EventType.SESSION_START, EventType.SESSION_END])
        self.assertNotEqual(late[0], sid)
        self.assertEqual(self.kinds(late[0]), [EventType.TOOL_CALL])

    def test_run_of_a_process_that_died_is_dropped(self):
        child, child_sid = self.run_in_child("import os\nos._exit(0)\n")
        child.wait(timeout=60)
        self.assertEqual(self.marker(), child_sid)
        with _base.run_session(self.store, "agent"):
            pass
        self.assertEqual(self.marker(), None)

    def registry(self):
        return Path(self.trace_dir) / ".integration-runs.agent"

    def test_dead_process_marker_is_not_reused_without_starting_another_run(self):
        child, child_sid = self.run_in_child("import os\nos._exit(0)\n")
        child.wait(timeout=60)
        self.assertEqual(self.marker(), child_sid)
        sid = _base._get_or_create_session(self.store, "agent")
        self.assertNotEqual(sid, child_sid)
        self.assertEqual(self.marker(), sid)
        self.assertFalse(self.registry().exists())

    def test_run_whose_pid_was_reused_is_dropped(self):
        child, child_sid = self.run_in_child("sys.stdin.readline()\n")
        sid, pid, recorded = _base._read_runs(self.registry())[0]
        self.assertEqual((sid, pid), (child_sid, child.pid))
        if not recorded:
            self.skipTest("no process start identity on this platform")
        self.assertEqual(_base._get_or_create_session(self.store, "agent"), child_sid)

        # The process now holding the PID started at another time.
        real = _base._process_identity

        def identity(p):
            return recorded + "0" if p == child.pid else real(p)

        with patch.object(_base, "_process_identity", identity):
            sid = _base._get_or_create_session(self.store, "agent")
        self.assertNotEqual(sid, child_sid)
        self.assertEqual(self.marker(), sid)
        self.assertFalse(self.registry().exists())

    def test_run_whose_process_dies_right_after_a_lookup_is_not_reused(self):
        child, child_sid = self.run_in_child("sys.stdin.readline()\n")
        self.assertEqual(_base._get_or_create_session(self.store, "agent"), child_sid)
        child.kill()
        child.wait(timeout=60)
        self.assertNotEqual(_base._get_or_create_session(self.store, "agent"), child_sid)

    def start_run_stopping_in_child(self, stop):
        """Start a run in a child process that stops at ``stop(path, text)``.

        ``stop`` replaces ``_base._replace_file`` for the marker or for the
        registry; it prints the run's session ID first. Returns the child and
        that ID.
        """
        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
        child = subprocess.Popen(
            [sys.executable, "-c",
             "import os, sys\n"
             "from agent_trace.integrations import _base\n"
             "from agent_trace.store import TraceStore\n"
             "replace_file = _base._replace_file\n" + stop +
             "_base.run_session(TraceStore(sys.argv[1]), 'agent').__enter__()\n",
             self.trace_dir],
            env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        )

        def cleanup():
            if child.poll() is None:
                child.kill()
            child.wait(timeout=60)
            for pipe in (child.stdin, child.stdout):
                pipe.close()

        self.addCleanup(cleanup)
        sid = child.stdout.readline().strip()
        self.assertTrue(sid)
        return child, sid

    def test_process_dying_before_its_run_is_registered_leaves_no_marker(self):
        child, child_sid = self.start_run_stopping_in_child(
            "def stop(path, text):\n"
            "    print(text.splitlines()[-1].split()[0], flush=True)\n"
            "    os._exit(0)\n"
            "_base._replace_file = stop\n")
        child.wait(timeout=60)
        self.assertTrue(self.store.session_exists(child_sid))
        self.assertNotEqual(_base._get_or_create_session(self.store, "agent"), child_sid)

    marker_stop = (
        "def stop(path, text):\n"
        "    if not path.name.startswith('.active-session'):\n"
        "        return replace_file(path, text)\n"
        "    print(text, flush=True)\n"
        "    {}\n"
        "_base._replace_file = stop\n"
    )

    def test_registered_run_without_its_marker_is_found(self):
        # The state a run leaves when its process registered it but did not
        # write the marker: the marker still names an older session.
        standalone = _base._get_or_create_session(self.store, "agent")
        child = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"],
                                 stdin=subprocess.PIPE)
        self.addCleanup(lambda: (child.kill(), child.wait(timeout=60), child.stdin.close()))
        run = _base.SessionMeta(agent_name="agent")
        self.store.create_session(run)
        identity = _base._process_identity(child.pid) or ""
        _base._write_runs(self.registry(), [(run.session_id, child.pid, identity)])
        self.assertEqual(self.marker(), standalone)
        self.assertEqual(_base._get_or_create_session(self.store, "agent"), run.session_id)
        self.assertEqual(self.marker(), run.session_id)

    def test_failure_to_write_the_marker_unregisters_the_run(self):
        standalone = _base._get_or_create_session(self.store, "agent")
        real = _base._replace_file

        def fail_marker(path, text):
            if path.name.startswith(".active-session"):
                raise OSError("disk full")
            real(path, text)

        with patch.object(_base, "_replace_file", fail_marker):
            with self.assertRaises(OSError):
                with _base.run_session(self.store, "agent"):
                    pass
        self.assertFalse(self.registry().exists())
        self.assertEqual(_base._get_or_create_session(self.store, "agent"), standalone)
        failed, = [m for m in self.store.list_sessions() if m.session_id != standalone]
        self.assertIsNotNone(failed.ended_at)

    def test_dead_registered_run_without_its_marker_is_not_used(self):
        standalone = _base._get_or_create_session(self.store, "agent")
        child, child_sid = self.start_run_stopping_in_child(
            self.marker_stop.format("os._exit(0)"))
        child.wait(timeout=60)
        sid = _base._get_or_create_session(self.store, "agent")
        self.assertNotEqual(sid, child_sid)
        self.assertEqual(sid, standalone)
        self.assertFalse(self.registry().exists())

    def test_symlinked_registry_is_refused(self):
        outside = Path(self.trace_dir) / "outside"
        outside.write_text("", encoding="utf-8")
        try:
            self.registry().symlink_to(outside)
        except OSError as exc:
            self.skipTest(f"cannot create a symlink: {exc}")
        with self.assertRaises(ValueError):
            with _base.run_session(self.store, "agent"):
                pass
        self.assertEqual(outside.read_text(encoding="utf-8"), "")
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


class FakeKernel32:
    """kernel32 functions as _base._windows_process calls them, answering
    for one process."""

    def __init__(self, handle=7, open_error=0, exit_code=259, created=None):
        self.closed = []
        self.opened = []

        def open_process(access, inherit, pid):
            self.opened.append((access, inherit, pid))
            return handle

        def get_exit_code(h, code):
            code._obj.value = exit_code
            return 1

        def get_times(h, *times):
            if created is None:
                return 0
            times[0]._obj.dwHighDateTime = created >> 32
            times[0]._obj.dwLowDateTime = created & 0xFFFFFFFF
            return 1

        self.OpenProcess = open_process
        self.GetExitCodeProcess = get_exit_code
        self.GetProcessTimes = get_times

        def close_handle(h):
            self.closed.append(h)
            return 1

        self.CloseHandle = close_handle
        self.open_error = open_error


class TestWindowsProcess(unittest.TestCase):
    """The Windows process checks behind the run registry."""

    def windows_process(self, kernel32):
        return self.windows_process_call(kernel32, _base._windows_process)

    def test_running_process_has_its_creation_time_identity(self):
        kernel32 = FakeKernel32(created=(5 << 32) | 9)
        self.assertEqual(self.windows_process(kernel32), (True, f"win:{(5 << 32) | 9}"))
        self.assertEqual(kernel32.opened, [(0x1000, False, 1234)])
        self.assertEqual(kernel32.closed, [7])

    def test_exited_process_is_not_alive(self):
        kernel32 = FakeKernel32(exit_code=0, created=1)
        self.assertEqual(self.windows_process(kernel32), (False, None))
        self.assertEqual(kernel32.closed, [7])

    def test_process_that_cannot_be_opened(self):
        # Access denied means a process has the PID; any other error means none does.
        self.assertEqual(self.windows_process(FakeKernel32(handle=0, open_error=5)), (True, None))
        self.assertEqual(self.windows_process(FakeKernel32(handle=0, open_error=87)), (False, None))

    def test_unknown_creation_time_keeps_the_process_alive_without_identity(self):
        kernel32 = FakeKernel32(created=None)
        self.assertEqual(self.windows_process(kernel32), (True, None))
        self.assertEqual(kernel32.closed, [7])

    def test_identity_and_liveness_use_it_on_windows(self):
        kernel32 = FakeKernel32(created=42)
        with patch.object(_base.os, "name", "nt"):
            self.assertEqual(self.windows_process_call(kernel32, _base._process_identity), "win:42")
            self.assertTrue(self.windows_process_call(kernel32, _base._pid_alive))
        with patch.object(_base.os, "name", "nt"):
            self.assertFalse(self.windows_process_call(
                FakeKernel32(exit_code=1, created=42), _base._pid_alive))

    def windows_process_call(self, kernel32, function):
        import ctypes
        with patch.object(ctypes, "WinDLL", lambda *a, **k: kernel32, create=True), \
                patch.object(ctypes, "get_last_error", lambda: kernel32.open_error, create=True):
            return function(1234)

    @unittest.skipUnless(os.name == "nt", "real Windows processes")
    def test_real_processes(self):
        alive, identity = _base._windows_process(os.getpid())
        self.assertTrue(alive)
        self.assertRegex(identity, r"^win:\d+$")
        self.assertEqual(_base._process_identity(os.getpid()), identity)

        child = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.readline()"],
                                 stdin=subprocess.PIPE)
        self.addCleanup(child.wait, 60)
        child_alive, child_identity = _base._windows_process(child.pid)
        self.assertTrue(child_alive)
        self.assertRegex(child_identity, r"^win:\d+$")
        self.assertNotEqual(child_identity, identity)
        child.stdin.close()
        child.wait(timeout=60)
        # Popen still holds the process handle, so the PID cannot be reused yet.
        self.assertEqual(_base._windows_process(child.pid), (False, None))
        self.assertFalse(_base._pid_alive(child.pid))


if __name__ == "__main__":
    unittest.main()
