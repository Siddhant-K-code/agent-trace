"""Tests for the Python decorator API."""

import asyncio
import contextlib
import enum
import functools
import inspect
import io
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace.decorator import (
    end_session,
    log_decision,
    start_session,
    trace_llm_call,
    trace_tool,
)
from agent_trace.models import EventType
from agent_trace.store import TraceStore


class TestDecorator(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_trace_tool_basic(self):
        session_id = start_session(name="test", trace_dir=self.tmpdir)

        @trace_tool
        def add(a: int, b: int) -> int:
            return a + b

        result = add(2, 3)
        self.assertEqual(result, 5)

        meta = end_session()
        self.assertIsNotNone(meta)
        self.assertEqual(meta.tool_calls, 1)

        store = TraceStore(self.tmpdir)
        events = store.load_events(session_id)
        tool_calls = [e for e in events if e.event_type == EventType.TOOL_CALL]
        tool_results = [e for e in events if e.event_type == EventType.TOOL_RESULT]
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(len(tool_results), 1)
        self.assertEqual(tool_calls[0].data["tool_name"], "add")
        self.assertEqual(tool_results[0].parent_id, tool_calls[0].event_id)

    def test_trace_tool_with_custom_name(self):
        session_id = start_session(name="test", trace_dir=self.tmpdir)

        @trace_tool(name="my_search")
        def search(query: str) -> str:
            return f"results for {query}"

        search("hello")
        end_session()

        store = TraceStore(self.tmpdir)
        events = store.load_events(session_id)
        tool_calls = [e for e in events if e.event_type == EventType.TOOL_CALL]
        self.assertEqual(tool_calls[0].data["tool_name"], "my_search")

    def test_trace_tool_captures_errors(self):
        session_id = start_session(name="test", trace_dir=self.tmpdir)

        @trace_tool
        def failing_tool() -> None:
            raise ValueError("something broke")

        with self.assertRaises(ValueError):
            failing_tool()

        meta = end_session()
        self.assertEqual(meta.errors, 1)

        store = TraceStore(self.tmpdir)
        events = store.load_events(session_id)
        errors = [e for e in events if e.event_type == EventType.ERROR]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].data["message"], "something broke")
        self.assertEqual(errors[0].data["exception_type"], "ValueError")

    def test_trace_llm_call(self):
        session_id = start_session(name="test", trace_dir=self.tmpdir)

        @trace_llm_call
        def call_llm(messages: list, model: str = "gpt-4") -> str:
            return "Hello, world!"

        result = call_llm([{"role": "user", "content": "hi"}], model="gpt-4")
        self.assertEqual(result, "Hello, world!")

        meta = end_session()
        self.assertEqual(meta.llm_requests, 1)

        store = TraceStore(self.tmpdir)
        events = store.load_events(session_id)
        llm_reqs = [e for e in events if e.event_type == EventType.LLM_REQUEST]
        llm_resps = [e for e in events if e.event_type == EventType.LLM_RESPONSE]
        self.assertEqual(len(llm_reqs), 1)
        self.assertEqual(len(llm_resps), 1)
        self.assertEqual(llm_reqs[0].data["model"], "gpt-4")
        self.assertEqual(llm_reqs[0].data["message_count"], 1)

    def test_trace_llm_call_failure_exports_as_failed_model_call(self):
        from agent_trace.otlp import _to_span_id, session_to_otlp_genai

        session_id = start_session(name="test", trace_dir=self.tmpdir)

        @trace_llm_call
        def call_llm(messages: list, model: str = "gpt-4") -> str:
            raise TimeoutError("model timed out")

        with self.assertRaises(TimeoutError):
            call_llm([{"role": "user", "content": "hi"}], model="gpt-4")
        meta = end_session()

        store = TraceStore(self.tmpdir)
        events = store.load_events(session_id)
        request = next(e for e in events if e.event_type == EventType.LLM_REQUEST)
        error = next(e for e in events if e.event_type == EventType.ERROR)
        self.assertEqual(error.parent_id, request.event_id)

        spans = session_to_otlp_genai(meta, events)["resourceSpans"][0][
            "scopeSpans"][0]["spans"]
        llm = [s for s in spans if s["name"] == "chat gpt-4"]
        self.assertEqual(len(llm), 1)
        self.assertEqual(llm[0]["spanId"], _to_span_id(request.event_id))
        self.assertEqual(
            llm[0]["status"], {"code": 2, "message": "model timed out"})
        attrs = {a["key"]: a["value"] for a in llm[0]["attributes"]}
        self.assertEqual(attrs["error.type"], {"stringValue": "TimeoutError"})
        self.assertFalse(
            [s for s in spans if s["name"] == "execute_tool error"])

    def test_log_decision(self):
        session_id = start_session(name="test", trace_dir=self.tmpdir)

        log_decision(
            choice="use_cache",
            reason="data is fresh",
            alternatives=["fetch_new", "use_cache"],
        )

        end_session()

        store = TraceStore(self.tmpdir)
        events = store.load_events(session_id)
        decisions = [e for e in events if e.event_type == EventType.DECISION]
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].data["choice"], "use_cache")
        self.assertEqual(decisions[0].data["reason"], "data is fresh")

    def test_multiple_tool_calls(self):
        session_id = start_session(name="test", trace_dir=self.tmpdir)

        @trace_tool
        def tool_a() -> str:
            return "a"

        @trace_tool
        def tool_b() -> str:
            return "b"

        tool_a()
        tool_b()
        tool_a()

        meta = end_session()
        self.assertEqual(meta.tool_calls, 3)

    def test_no_session_doesnt_crash(self):
        """Calling traced functions without a session should not raise."""

        @trace_tool
        def safe_tool() -> str:
            return "ok"

        result = safe_tool()
        self.assertEqual(result, "ok")


class TestDecoratorAsync(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def _events(self, session_id, event_type):
        events = TraceStore(self.tmpdir).load_events(session_id)
        return [e for e in events if e.event_type == event_type]

    def test_async_tool_records_awaited_result_and_duration(self):
        session_id = start_session(name="test", trace_dir=self.tmpdir)

        @trace_tool
        async def fetch(url: str) -> str:
            await asyncio.sleep(0.05)
            return f"body of {url}"

        self.assertTrue(inspect.iscoroutinefunction(fetch))
        self.assertEqual(asyncio.run(fetch("x")), "body of x")
        meta = end_session()
        self.assertEqual(meta.tool_calls, 1)

        call = self._events(session_id, EventType.TOOL_CALL)[0]
        results = self._events(session_id, EventType.TOOL_RESULT)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].parent_id, call.event_id)
        self.assertEqual(results[0].data["content_preview"], "'body of x'")
        self.assertGreaterEqual(results[0].duration_ms, 40)

    def test_async_tool_error_is_recorded(self):
        session_id = start_session(name="test", trace_dir=self.tmpdir)

        @trace_tool
        async def fetch() -> str:
            await asyncio.sleep(0)
            raise RuntimeError("upstream down")

        with self.assertRaises(RuntimeError):
            asyncio.run(fetch())
        meta = end_session()
        self.assertEqual(meta.errors, 1)

        call = self._events(session_id, EventType.TOOL_CALL)[0]
        self.assertEqual(self._events(session_id, EventType.TOOL_RESULT), [])
        errors = self._events(session_id, EventType.ERROR)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].parent_id, call.event_id)
        self.assertEqual(errors[0].data["message"], "upstream down")
        self.assertEqual(errors[0].data["exception_type"], "RuntimeError")

    def test_async_llm_call_records_awaited_response_and_error(self):
        session_id = start_session(name="test", trace_dir=self.tmpdir)

        @trace_llm_call
        async def call_llm(messages: list, model: str = "gpt-4") -> str:
            await asyncio.sleep(0)
            if messages[0]["content"] == "fail":
                raise TimeoutError("model timed out")
            return "Hello"

        self.assertEqual(
            asyncio.run(call_llm([{"role": "user", "content": "hi"}])), "Hello")
        with self.assertRaises(TimeoutError):
            asyncio.run(call_llm([{"role": "user", "content": "fail"}]))
        meta = end_session()
        self.assertEqual(meta.llm_requests, 2)
        self.assertEqual(meta.errors, 1)

        requests = self._events(session_id, EventType.LLM_REQUEST)
        responses = self._events(session_id, EventType.LLM_RESPONSE)
        errors = self._events(session_id, EventType.ERROR)
        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0].data["content_preview"], "'Hello'")
        self.assertEqual(responses[0].parent_id, requests[0].event_id)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].parent_id, requests[1].event_id)


class TestStartSessionWhileActive(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_second_start_ends_the_first_session(self):
        first = start_session(name="first", trace_dir=self.tmpdir)

        @trace_tool
        def work() -> str:
            return "ok"

        work()
        second = start_session(name="second", trace_dir=self.tmpdir)
        work()
        meta = end_session()
        self.assertEqual(meta.session_id, second)
        self.assertEqual(meta.tool_calls, 1)

        store = TraceStore(self.tmpdir)
        first_meta = store.load_meta(first)
        self.assertIsNotNone(first_meta.ended_at)
        self.assertEqual(first_meta.tool_calls, 1)
        first_types = [e.event_type for e in store.load_events(first)]
        self.assertEqual(first_types[-1], EventType.SESSION_END)
        self.assertEqual(first_types.count(EventType.TOOL_CALL), 1)
        self.assertIsNotNone(store.load_meta(second).ended_at)
        self.assertIsNone(end_session())


class TestDecoratorThreadSafety(unittest.TestCase):
    """Verify concurrent agents in the same process use isolated sessions."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_concurrent_sessions_are_isolated(self):
        """Two threads each running start_session/end_session must not share state."""
        import threading

        results = {}
        errors = []

        def run_agent(agent_name):
            try:
                sid = start_session(name=agent_name, trace_dir=self.tmpdir)

                @trace_tool
                def work() -> str:
                    return agent_name

                work()
                work()
                meta = end_session()
                results[agent_name] = (sid, meta.tool_calls)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=run_agent, args=("agent-alpha",))
        t2 = threading.Thread(target=run_agent, args=("agent-beta",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertEqual(errors, [])
        self.assertIn("agent-alpha", results)
        self.assertIn("agent-beta", results)

        sid_alpha, calls_alpha = results["agent-alpha"]
        sid_beta, calls_beta = results["agent-beta"]

        # Sessions must be distinct
        self.assertNotEqual(sid_alpha, sid_beta)
        # Each agent must see exactly its own 2 tool calls
        self.assertEqual(calls_alpha, 2)
        self.assertEqual(calls_beta, 2)

        # Verify events on disk are also isolated
        store = TraceStore(self.tmpdir)
        events_alpha = store.load_events(sid_alpha)
        events_beta = store.load_events(sid_beta)
        calls_in_alpha = [e for e in events_alpha if e.event_type == EventType.TOOL_CALL]
        calls_in_beta = [e for e in events_beta if e.event_type == EventType.TOOL_CALL]
        self.assertEqual(len(calls_in_alpha), 2)
        self.assertEqual(len(calls_in_beta), 2)


class Model(enum.Enum):
    BIG = "gpt-4o"


class StrModel(str, enum.Enum):
    BIG = "gpt-4o"


class NoStr(Exception):
    def __str__(self):
        raise RuntimeError("no str")


_BIG = "<Model.BIG: 'gpt-4o'>"
_MODES = ("sync", "async")
_WRITE_ERRORS = (OSError, RuntimeError)
_KINDS = {
    "tool": (trace_tool, EventType.TOOL_CALL, EventType.TOOL_RESULT, "tool_calls"),
    "llm": (
        trace_llm_call, EventType.LLM_REQUEST, EventType.LLM_RESPONSE, "llm_requests"),
}


def _failing_write(event_type, error=OSError):
    """Make every write of one event type raise ``error``."""
    real_append = TraceStore.append_event

    def append_or_fail(self, session_id, event):
        if event.event_type == event_type:
            raise error("write failed")
        return real_append(self, session_id, event)

    return mock.patch.object(TraceStore, "append_event", append_or_fail)


def _sync_and_async(func):
    """Return the function, and an async def twin that calls it."""

    @functools.wraps(func)
    async def twin(*args, **kwargs):
        await asyncio.sleep(0)
        return func(*args, **kwargs)

    return {"sync": func, "async": twin}


def _call(func, *args, **kwargs):
    """Call the function. An async def function is run to its end."""
    if inspect.iscoroutinefunction(func):
        return asyncio.run(func(*args, **kwargs))
    return func(*args, **kwargs)


def _end_quietly():
    try:
        end_session()
    except Exception:
        pass


class _SessionTestCase(unittest.TestCase):
    def setUp(self):
        _end_quietly()
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        self.addCleanup(_end_quietly)

    def start(self, **kwargs):
        return start_session(name="test", trace_dir=self.tmpdir, **kwargs)

    def store(self):
        return TraceStore(self.tmpdir)

    def events(self, session_id, event_type=None):
        events = self.store().load_events(session_id)
        if event_type is None:
            return events
        return [e for e in events if e.event_type == event_type]

    def event_types(self, session_id):
        return [e.event_type.value for e in self.events(session_id)]

    def meta(self, session_id):
        return self.store().load_meta(session_id)

    def work(self, decorate=trace_tool, mode="sync", fail=False):
        """Return a traced function that doubles x, and the list of its runs."""
        runs = []

        def work(x):
            runs.append(x)
            if fail:
                raise ValueError(f"bad input {x}")
            return x * 2

        return decorate(_sync_and_async(work)[mode]), runs


class TestRecordingFailureNeverReachesTheCaller(_SessionTestCase):
    def test_failed_call_write_runs_the_function_and_records_nothing_more(self):
        for kind, (decorate, call_type, _, counter) in _KINDS.items():
            for error in _WRITE_ERRORS:
                for mode in _MODES:
                    with self.subTest(kind=kind, error=error.__name__, mode=mode):
                        sid = self.start()
                        traced, runs = self.work(decorate, mode)
                        stderr = io.StringIO()
                        with _failing_write(call_type, error), \
                                contextlib.redirect_stderr(stderr):
                            self.assertEqual(_call(traced, 2), 4)
                        self.assertEqual(runs, [2])
                        self.assertEqual(stderr.getvalue(), "")
                        end_session()
                        self.assertEqual(
                            self.event_types(sid), ["session_start", "session_end"])
                        self.assertEqual(getattr(self.meta(sid), counter), 0)

    def test_failed_call_write_keeps_the_functions_own_error(self):
        for kind, (decorate, call_type, _, counter) in _KINDS.items():
            for error in _WRITE_ERRORS:
                for mode in _MODES:
                    with self.subTest(kind=kind, error=error.__name__, mode=mode):
                        sid = self.start()
                        traced, runs = self.work(decorate, mode, fail=True)
                        with _failing_write(call_type, error):
                            with self.assertRaises(ValueError) as caught:
                                _call(traced, 2)
                        self.assertEqual(str(caught.exception), "bad input 2")
                        self.assertIsNone(caught.exception.__context__)
                        self.assertEqual(runs, [2])
                        end_session()
                        self.assertEqual(
                            self.event_types(sid), ["session_start", "session_end"])
                        meta = self.meta(sid)
                        self.assertEqual((getattr(meta, counter), meta.errors), (0, 0))

    def test_failed_result_write_returns_the_result(self):
        for kind, (decorate, call_type, result_type, counter) in _KINDS.items():
            for error in _WRITE_ERRORS:
                for mode in _MODES:
                    with self.subTest(kind=kind, error=error.__name__, mode=mode):
                        sid = self.start()
                        traced, runs = self.work(decorate, mode)
                        with _failing_write(result_type, error):
                            self.assertEqual(_call(traced, 2), 4)
                        self.assertEqual(runs, [2])
                        end_session()
                        self.assertEqual(
                            self.event_types(sid),
                            ["session_start", call_type.value, "session_end"])
                        self.assertEqual(getattr(self.meta(sid), counter), 1)

    def test_failed_error_write_raises_the_functions_own_error(self):
        for kind, (decorate, call_type, _, counter) in _KINDS.items():
            for error in _WRITE_ERRORS:
                for mode in _MODES:
                    with self.subTest(kind=kind, error=error.__name__, mode=mode):
                        sid = self.start()
                        traced, runs = self.work(decorate, mode, fail=True)
                        with _failing_write(EventType.ERROR, error):
                            with self.assertRaises(ValueError) as caught:
                                _call(traced, 2)
                        self.assertEqual(str(caught.exception), "bad input 2")
                        self.assertIsNone(caught.exception.__context__)
                        self.assertEqual(runs, [2])
                        end_session()
                        self.assertEqual(
                            self.event_types(sid),
                            ["session_start", call_type.value, "session_end"])
                        meta = self.meta(sid)
                        self.assertEqual((getattr(meta, counter), meta.errors), (1, 0))

    def test_failed_recording_step_before_the_write_is_dropped(self):
        for mode in _MODES:
            with self.subTest(mode=mode):
                sid = self.start(redact=True)
                traced, runs = self.work(trace_tool, mode)
                with mock.patch(
                    "agent_trace.decorator.redact_data",
                    side_effect=RuntimeError("redaction failed"),
                ):
                    self.assertEqual(_call(traced, 2), 4)
                self.assertEqual(runs, [2])
                end_session()
                self.assertEqual(
                    self.event_types(sid), ["session_start", "session_end"])

    def test_base_exception_from_a_write_propagates(self):
        for kind, (decorate, call_type, result_type, counter) in _KINDS.items():
            for mode in _MODES:
                with self.subTest(kind=kind, mode=mode):
                    sid = self.start()
                    traced, runs = self.work(decorate, mode)
                    with _failing_write(call_type, KeyboardInterrupt):
                        with self.assertRaises(KeyboardInterrupt):
                            _call(traced, 2)
                    self.assertEqual(runs, [])
                    with _failing_write(result_type, KeyboardInterrupt):
                        with self.assertRaises(KeyboardInterrupt):
                            _call(traced, 3)
                    self.assertEqual(runs, [3])
                    end_session()
                    self.assertEqual(
                        self.event_types(sid),
                        ["session_start", call_type.value, "session_end"])
                    self.assertEqual(getattr(self.meta(sid), counter), 1)

    def test_result_is_not_written_into_a_session_the_call_started(self):
        for kind, (decorate, call_type, _, _) in _KINDS.items():
            for fail in (False, True):
                for mode in _MODES:
                    with self.subTest(kind=kind, fail=fail, mode=mode):
                        outer = self.start()
                        inner = []

                        def body(x, fail=fail):
                            inner.append(
                                start_session(name="inner", trace_dir=self.tmpdir))
                            if fail:
                                raise ValueError("bad")
                            return x * 2

                        traced = decorate(_sync_and_async(body)[mode])
                        if fail:
                            with self.assertRaises(ValueError):
                                _call(traced, 2)
                        else:
                            self.assertEqual(_call(traced, 2), 4)
                        self.assertEqual(end_session().session_id, inner[0])
                        self.assertEqual(
                            self.event_types(inner[0]),
                            ["session_start", "session_end"])
                        self.assertEqual(
                            self.event_types(outer),
                            ["session_start", call_type.value, "session_end"])


class TestCallThatCannotBeBound(_SessionTestCase):
    def test_wrong_arguments_raise_pythons_own_type_error(self):
        for kind, (decorate, call_type, _, counter) in _KINDS.items():
            for mode in _MODES:
                with self.subTest(kind=kind, mode=mode):
                    runs = []

                    def tool(x):
                        runs.append(x)

                    raw = _sync_and_async(tool)[mode]
                    with self.assertRaises(TypeError) as plain:
                        _call(raw)
                    message = str(plain.exception)
                    self.assertIn(
                        "missing 1 required positional argument: 'x'", message)

                    sid = self.start()
                    with self.assertRaises(TypeError) as caught:
                        _call(decorate(raw))
                    self.assertEqual(str(caught.exception), message)
                    self.assertIsNone(caught.exception.__context__)
                    self.assertEqual(runs, [])
                    end_session()
                    self.assertEqual(
                        self.event_types(sid),
                        ["session_start", call_type.value, "error", "session_end"])
                    call = self.events(sid, call_type)[0]
                    error = self.events(sid, EventType.ERROR)[0]
                    self.assertEqual(error.parent_id, call.event_id)
                    self.assertEqual(
                        error.data, {"message": message, "exception_type": "TypeError"})
                    meta = self.meta(sid)
                    self.assertEqual((getattr(meta, counter), meta.errors), (1, 1))

    def test_builtin_without_a_signature_runs(self):
        for mode in _MODES:
            with self.subTest(mode=mode):
                traced = trace_tool(name="max")(_sync_and_async(max)[mode])
                self.assertEqual(_call(traced, [1, 5, 3], default=0), 5)
                sid = self.start()
                self.assertEqual(_call(traced, [1, 5, 3], default=0), 5)
                end_session()
                self.assertEqual(
                    self.event_types(sid),
                    ["session_start", "tool_call", "tool_result", "session_end"])
                call = self.events(sid, EventType.TOOL_CALL)[0]
                self.assertEqual(call.data, {
                    "tool_name": "max",
                    "arguments": {"args": "([1, 5, 3],)", "kwargs": {"default": "0"}},
                })

    def test_unreadable_signature_stores_args_and_kwargs(self):
        cases = ((True, "[REDACTED:sensitive]"), (False, "'t'"))
        for mode in _MODES:
            for redact, token in cases:
                with self.subTest(mode=mode, redact=redact):
                    runs = []

                    def search(*args, **kwargs):
                        runs.append((args, kwargs))
                        return "found"

                    search.__signature__ = 42
                    traced = trace_tool(_sync_and_async(search)[mode])
                    sid = self.start(redact=redact)
                    self.assertEqual(_call(traced, "q", token="t"), "found")
                    self.assertEqual(runs, [(("q",), {"token": "t"})])
                    end_session()
                    call = self.events(sid, EventType.TOOL_CALL)[0]
                    self.assertEqual(
                        call.data["arguments"],
                        {"args": "('q',)", "kwargs": {"token": token}})

    def test_unreadable_signature_llm_call_reads_keywords_only(self):
        for mode in _MODES:
            with self.subTest(mode=mode):
                def chat(*args, **kwargs):
                    return "ok"

                chat.__signature__ = 42
                traced = trace_llm_call(_sync_and_async(chat)[mode])
                sid = self.start()
                self.assertEqual(_call(traced, [1, 2, 3], model="m"), "ok")
                self.assertEqual(_call(traced, messages=[1, 2]), "ok")
                end_session()
                requests = self.events(sid, EventType.LLM_REQUEST)
                self.assertEqual(
                    [(r.data["message_count"], r.data["model"]) for r in requests],
                    [(0, "m"), (2, "")])
                self.assertEqual(self.meta(sid).llm_requests, 2)

    def test_wrapper_that_adds_an_argument_runs(self):
        def search(client, query):
            return f"{client}:{query}"

        @functools.wraps(search)
        def with_client(query):
            return search("client", query)

        for mode in _MODES:
            with self.subTest(mode=mode):
                traced = trace_tool(_sync_and_async(with_client)[mode])
                self.assertEqual(_call(traced, "q"), "client:q")
                sid = self.start()
                self.assertEqual(_call(traced, "q"), "client:q")
                end_session()
                call = self.events(sid, EventType.TOOL_CALL)[0]
                self.assertEqual(
                    call.data["arguments"], {"args": "('q',)", "kwargs": {}})


class TestVarKeywordRedaction(_SessionTestCase):
    def test_var_keyword_values_are_stored_as_an_object(self):
        def connect(host, *items, **options):
            return "ok"

        cases = (
            (True, "[REDACTED:sensitive]", "[REDACTED:sensitive]"),
            (False, "'hunter2'", "'sk-1'"),
        )
        for mode in _MODES:
            for redact, password, api_key in cases:
                with self.subTest(mode=mode, redact=redact):
                    traced = trace_tool(_sync_and_async(connect)[mode])
                    sid = self.start(redact=redact)
                    self.assertEqual(
                        _call(
                            traced, "db", 1, 2,
                            password="hunter2", api_key="sk-1", port=5432),
                        "ok")
                    end_session()
                    call = self.events(sid, EventType.TOOL_CALL)[0]
                    self.assertEqual(call.data["arguments"], {
                        "host": "'db'",
                        "items": "(1, 2)",
                        "options": {
                            "password": password, "api_key": api_key, "port": "5432"},
                    })


class TestDefaultsAreStored(_SessionTestCase):
    def test_defaults_the_call_leaves_out_are_stored(self):
        def search(query, limit=10):
            return []

        def chat(messages, model="gpt-4"):
            return "ok"

        for mode in _MODES:
            with self.subTest(mode=mode):
                sid = self.start()
                _call(trace_tool(_sync_and_async(search)[mode]), "q")
                _call(trace_llm_call(_sync_and_async(chat)[mode]), [{"role": "user"}])
                end_session()
                call = self.events(sid, EventType.TOOL_CALL)[0]
                self.assertEqual(
                    call.data["arguments"], {"query": "'q'", "limit": "10"})
                request = self.events(sid, EventType.LLM_REQUEST)[0]
                self.assertEqual(request.data["model"], "gpt-4")


class TestValuesThatAreNotJson(_SessionTestCase):
    def test_model_that_is_not_json_is_stored_as_its_repr(self):
        sentinel = object()
        cases = (
            (sentinel, repr(sentinel)),
            (Model.BIG, _BIG),
            (None, None),
            (StrModel.BIG, "gpt-4o"),
        )
        for mode in _MODES:
            for model, stored in cases:
                with self.subTest(mode=mode, stored=stored):
                    runs = []

                    def chat(messages, model=""):
                        runs.append(model)
                        return "ok"

                    traced = trace_llm_call(_sync_and_async(chat)[mode])
                    sid = self.start()
                    self.assertEqual(
                        _call(traced, [{"role": "user"}], model=model), "ok")
                    self.assertEqual(runs, [model])
                    end_session()
                    request = self.events(sid, EventType.LLM_REQUEST)[0]
                    self.assertEqual(request.data["model"], stored)
                    self.assertEqual(request.data["message_count"], 1)
                    self.assertEqual(self.meta(sid).llm_requests, 1)

    def test_decision_values_that_are_not_json_are_stored_as_their_repr(self):
        cases = (
            ([Model.BIG, "b"], [_BIG, "b"]),
            ((Model.BIG, "b"), [_BIG, "b"]),
            (["a", 1], ["a", 1]),
            ({"a"}, "{'a'}"),
            (None, []),
        )
        for alternatives, stored in cases:
            with self.subTest(alternatives=alternatives):
                sid = self.start()
                self.assertIsNone(log_decision(Model.BIG, Model.BIG, alternatives))
                end_session()
                decision = self.events(sid, EventType.DECISION)[0]
                self.assertEqual(
                    decision.data,
                    {"choice": _BIG, "reason": _BIG, "alternatives": stored})

    def test_failed_decision_write_is_dropped(self):
        for error in _WRITE_ERRORS:
            with self.subTest(error=error.__name__):
                sid = self.start()
                with _failing_write(EventType.DECISION, error):
                    self.assertIsNone(log_decision("a", "b", ["a", "c"]))
                end_session()
                self.assertEqual(
                    self.event_types(sid), ["session_start", "session_end"])


class TestUnprintableException(_SessionTestCase):
    def test_exception_whose_str_raises_reaches_the_caller(self):
        for mode in _MODES:
            with self.subTest(mode=mode):
                raised = []

                def explode():
                    raised.append(NoStr())
                    raise raised[-1]

                traced = trace_tool(_sync_and_async(explode)[mode])
                with self.assertRaises(NoStr):
                    _call(traced)
                sid = self.start()
                with self.assertRaises(NoStr) as caught:
                    _call(traced)
                self.assertIs(caught.exception, raised[-1])
                self.assertIsNone(caught.exception.__context__)
                end_session()
                error = self.events(sid, EventType.ERROR)[0]
                self.assertEqual(
                    error.data,
                    {"message": "<exception str() failed>", "exception_type": "NoStr"})
                self.assertEqual(self.meta(sid).errors, 1)


class TestSessionStateAfterFailure(_SessionTestCase):
    def test_start_session_starts_when_ending_the_old_one_fails(self):
        for error in _WRITE_ERRORS:
            with self.subTest(error=error.__name__):
                old = self.start()
                with _failing_write(EventType.SESSION_END, error):
                    new = self.start()
                self.assertNotEqual(new, old)
                traced, runs = self.work()
                self.assertEqual(traced(2), 4)
                self.assertEqual(end_session().session_id, new)
                self.assertEqual(self.event_types(old), ["session_start"])
                self.assertIsNone(self.meta(old).ended_at)
                self.assertEqual(
                    self.event_types(new),
                    ["session_start", "tool_call", "tool_result", "session_end"])

    def test_keyboard_interrupt_while_ending_the_old_session_starts_no_new_one(self):
        old = self.start()
        with _failing_write(EventType.SESSION_END, KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.start()
        self.assertIsNone(end_session())
        self.assertEqual(
            [m.session_id for m in self.store().list_sessions()], [old])

    def test_start_session_starts_when_the_old_directory_is_gone(self):
        old = self.start()
        shutil.rmtree(self.store().base_dir / old)
        new = self.start()
        self.assertNotEqual(new, old)
        self.assertEqual(end_session().session_id, new)
        self.assertEqual(self.event_types(new), ["session_start", "session_end"])

    def test_failed_session_end_write_leaves_no_active_session(self):
        for error in (*_WRITE_ERRORS, KeyboardInterrupt):
            with self.subTest(error=error.__name__):
                sid = self.start()
                with _failing_write(EventType.SESSION_END, error):
                    with self.assertRaises(error):
                        end_session()
                self.assertIsNone(end_session())
                self.assertIsNone(self.meta(sid).ended_at)
                traced, runs = self.work()
                self.assertEqual(traced(2), 4)
                self.assertEqual(self.event_types(sid), ["session_start"])

    def test_failed_meta_write_leaves_no_active_session(self):
        sid = self.start()
        with mock.patch.object(
            TraceStore, "update_meta", side_effect=RuntimeError("meta failed")
        ):
            with self.assertRaises(RuntimeError):
                end_session()
        self.assertIsNone(end_session())
        self.assertIsNone(self.meta(sid).ended_at)
        self.assertEqual(self.event_types(sid), ["session_start", "session_end"])

    def test_start_into_a_file_leaves_no_active_session(self):
        old = self.start()
        not_a_dir = os.path.join(self.tmpdir, "not-a-dir")
        Path(not_a_dir).write_text("x")
        with self.assertRaises(OSError):
            start_session(name="bad", trace_dir=not_a_dir)
        self.assertIsNotNone(self.meta(old).ended_at)
        self.assertIsNone(end_session())
        traced, runs = self.work()
        self.assertEqual(traced(2), 4)
        self.assertEqual(runs, [2])
        new = self.start()
        self.assertEqual(end_session().session_id, new)

    def test_failed_session_start_write_leaves_no_active_session(self):
        for error in _WRITE_ERRORS:
            with self.subTest(error=error.__name__):
                trace_dir = os.path.join(self.tmpdir, error.__name__)
                with _failing_write(EventType.SESSION_START, error):
                    with self.assertRaises(error):
                        start_session(name="bad", trace_dir=trace_dir)
                self.assertIsNone(end_session())
                traced, runs = self.work()
                self.assertEqual(traced(2), 4)
                self.assertEqual(runs, [2])
                # The failed start keeps the directory it made.
                base_dir = TraceStore(trace_dir).base_dir
                sessions = [p for p in base_dir.iterdir() if p.is_dir()]
                self.assertEqual(len(sessions), 1)
                self.assertTrue((sessions[0] / "meta.json").is_file())

    def test_failed_create_session_leaves_no_active_session(self):
        old = self.start()
        with mock.patch.object(
            TraceStore, "create_session", side_effect=RuntimeError("no dir")
        ):
            with self.assertRaises(RuntimeError):
                self.start()
        self.assertIsNotNone(self.meta(old).ended_at)
        self.assertIsNone(end_session())


if __name__ == "__main__":
    unittest.main()
