"""Tests for the Python decorator API."""

import os
import sys
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
