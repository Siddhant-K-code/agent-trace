"""Tests for the MCP server (agent-strace mcp)."""

import json
import tempfile
import time
import unittest

from agent_trace.mcp_server import (
    _handle,
    _tool_diff_sessions,
    _tool_get_session,
    _tool_get_session_summary,
    _tool_list_sessions,
    _tool_search_events,
)
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store() -> TraceStore:
    return TraceStore(tempfile.mkdtemp())


def _add_session(
    store: TraceStore,
    session_id: str,
    events: list[TraceEvent] | None = None,
    agent_name: str = "",
    total_tokens: int = 1000,
    total_duration_ms: float = 60_000,
) -> SessionMeta:
    ts = time.time()
    meta = SessionMeta(
        session_id=session_id,
        started_at=ts,
        ended_at=ts + 60,
        agent_name=agent_name,
        total_tokens=total_tokens,
        total_duration_ms=total_duration_ms,
        tool_calls=sum(1 for e in (events or []) if e.event_type == EventType.TOOL_CALL),
        errors=sum(1 for e in (events or []) if e.event_type == EventType.ERROR),
    )
    store.create_session(meta)
    for ev in (events or []):
        store.append_event(session_id, ev)
    return meta


def _tool_call(name: str, ts: float = 0.0) -> TraceEvent:
    return TraceEvent(event_type=EventType.TOOL_CALL, timestamp=ts,
                      data={"tool_name": name, "arguments": {}})


def _file_write(path: str, ts: float = 0.0) -> TraceEvent:
    return TraceEvent(event_type=EventType.FILE_WRITE, timestamp=ts, data={"path": path})


def _file_read(path: str, ts: float = 0.0) -> TraceEvent:
    return TraceEvent(event_type=EventType.FILE_READ, timestamp=ts, data={"path": path})


def _error(msg: str = "fail", ts: float = 0.0) -> TraceEvent:
    return TraceEvent(event_type=EventType.ERROR, timestamp=ts, data={"message": msg})


def _tool_result(exit_code: int = 0, ts: float = 0.0) -> TraceEvent:
    return TraceEvent(event_type=EventType.TOOL_RESULT, timestamp=ts,
                      data={"exit_code": exit_code})


def _rpc(method: str, params: dict, req_id: int = 1) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}


# ---------------------------------------------------------------------------
# JSON-RPC lifecycle
# ---------------------------------------------------------------------------

class TestMcpLifecycle(unittest.TestCase):
    def setUp(self):
        self.store = _make_store()

    def test_initialize_returns_server_info(self):
        r = _handle(self.store, _rpc("initialize", {}))
        self.assertEqual(r["result"]["serverInfo"]["name"], "agent-trace")
        self.assertIn("protocolVersion", r["result"])

    def test_tools_list_returns_five_tools(self):
        r = _handle(self.store, _rpc("tools/list", {}))
        names = [t["name"] for t in r["result"]["tools"]]
        self.assertIn("list_sessions", names)
        self.assertIn("get_session", names)
        self.assertIn("search_events", names)
        self.assertIn("get_session_summary", names)
        self.assertIn("diff_sessions", names)
        self.assertEqual(len(names), 5)

    def test_unknown_method_returns_error(self):
        r = _handle(self.store, _rpc("unknown/method", {}))
        self.assertIn("error", r)
        self.assertEqual(r["error"]["code"], -32601)

    def test_notification_returns_none(self):
        req = {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
        r = _handle(self.store, req)
        self.assertIsNone(r)

    def test_ping_returns_empty_result(self):
        r = _handle(self.store, _rpc("ping", {}))
        self.assertEqual(r["result"], {})

    def test_unknown_tool_returns_error(self):
        r = _handle(self.store, _rpc("tools/call", {"name": "nonexistent", "arguments": {}}))
        self.assertIn("error", r)

    def test_malformed_json_handled_gracefully(self):
        # _handle itself doesn't parse JSON — that's run_stdio's job
        # but we can verify a missing method returns an error
        r = _handle(self.store, {"jsonrpc": "2.0", "id": 1})
        self.assertIn("error", r)


# ---------------------------------------------------------------------------
# list_sessions
# ---------------------------------------------------------------------------

class TestListSessions(unittest.TestCase):
    def setUp(self):
        self.store = _make_store()

    def test_empty_store(self):
        result = json.loads(_tool_list_sessions(self.store, {}))
        self.assertEqual(result["count"], 0)
        self.assertEqual(result["sessions"], [])

    def test_returns_sessions(self):
        _add_session(self.store, "sess1", agent_name="claude")
        _add_session(self.store, "sess2", agent_name="cursor")
        result = json.loads(_tool_list_sessions(self.store, {}))
        self.assertEqual(result["count"], 2)

    def test_limit_respected(self):
        for i in range(5):
            _add_session(self.store, f"sess{i}")
        result = json.loads(_tool_list_sessions(self.store, {"limit": 2}))
        self.assertEqual(result["count"], 2)

    def test_agent_filter(self):
        _add_session(self.store, "s1", agent_name="claude")
        _add_session(self.store, "s2", agent_name="cursor")
        result = json.loads(_tool_list_sessions(self.store, {"agent": "claude"}))
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["sessions"][0]["agent_name"], "claude")

    def test_session_has_cost_field(self):
        _add_session(self.store, "s1", total_tokens=1000)
        result = json.loads(_tool_list_sessions(self.store, {}))
        self.assertIn("estimated_cost_usd", result["sessions"][0])


# ---------------------------------------------------------------------------
# get_session
# ---------------------------------------------------------------------------

class TestGetSession(unittest.TestCase):
    def setUp(self):
        self.store = _make_store()
        _add_session(self.store, "abc123", events=[
            _tool_call("Bash"), _file_write("src/main.py"), _error("oops"),
        ])

    def test_returns_events(self):
        result = json.loads(_tool_get_session(self.store, {"session_id": "abc123"}))
        self.assertEqual(result["event_count"], 3)

    def test_prefix_match(self):
        result = json.loads(_tool_get_session(self.store, {"session_id": "abc"}))
        self.assertEqual(result["event_count"], 3)

    def test_event_type_filter(self):
        result = json.loads(_tool_get_session(self.store, {
            "session_id": "abc123",
            "event_types": ["tool_call"],
        }))
        self.assertEqual(result["event_count"], 1)
        self.assertEqual(result["events"][0]["event_type"], "tool_call")

    def test_not_found_returns_error(self):
        result = json.loads(_tool_get_session(self.store, {"session_id": "zzz"}))
        self.assertIn("error", result)

    def test_missing_session_id_returns_error(self):
        result = json.loads(_tool_get_session(self.store, {}))
        self.assertIn("error", result)

    def test_session_metadata_included(self):
        result = json.loads(_tool_get_session(self.store, {"session_id": "abc123"}))
        self.assertIn("session", result)
        self.assertEqual(result["session"]["session_id"], "abc123")


# ---------------------------------------------------------------------------
# search_events
# ---------------------------------------------------------------------------

class TestSearchEvents(unittest.TestCase):
    def setUp(self):
        self.store = _make_store()
        _add_session(self.store, "s1", events=[
            _tool_call("Bash"), _tool_call("Read"),
            _file_write("src/app.py"), _file_read("README.md"),
            _error("something failed"),
            _tool_result(exit_code=1),
        ])
        _add_session(self.store, "s2", events=[
            _tool_call("Write"), _file_write("tests/test_foo.py"),
        ])

    def test_filter_by_tool_name(self):
        result = json.loads(_tool_search_events(self.store, {"tool_name": "bash"}))
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["events"][0]["data"]["tool_name"], "Bash")

    def test_filter_by_file_path(self):
        result = json.loads(_tool_search_events(self.store, {"file_path": "src/"}))
        self.assertEqual(result["count"], 1)

    def test_filter_has_error(self):
        result = json.loads(_tool_search_events(self.store, {"has_error": True}))
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["events"][0]["event_type"], "error")

    def test_filter_by_exit_code(self):
        result = json.loads(_tool_search_events(self.store, {"exit_code": 1}))
        self.assertEqual(result["count"], 1)

    def test_scoped_to_session(self):
        result = json.loads(_tool_search_events(self.store, {
            "session_id": "s1", "tool_name": "bash",
        }))
        self.assertEqual(result["count"], 1)

    def test_cross_session_search(self):
        # src/app.py (file_write s1) + tests/test_foo.py (file_write s2) = 2
        result = json.loads(_tool_search_events(self.store, {"file_path": ".py"}))
        self.assertEqual(result["count"], 2)

    def test_limit_respected(self):
        result = json.loads(_tool_search_events(self.store, {"has_error": True, "limit": 1}))
        self.assertLessEqual(result["count"], 1)

    def test_session_not_found_returns_error(self):
        result = json.loads(_tool_search_events(self.store, {"session_id": "zzz"}))
        self.assertIn("error", result)


# ---------------------------------------------------------------------------
# get_session_summary
# ---------------------------------------------------------------------------

class TestGetSessionSummary(unittest.TestCase):
    def setUp(self):
        self.store = _make_store()
        _add_session(self.store, "sum1", events=[
            _tool_call("Bash", ts=time.time()),
            _file_write("src/main.py", ts=time.time() + 1),
        ])

    def test_returns_text_summary(self):
        result = _tool_get_session_summary(self.store, {"session_id": "sum1"})
        self.assertIsInstance(result, str)
        self.assertIn("sum1", result)

    def test_not_found_returns_error_json(self):
        result = json.loads(_tool_get_session_summary(self.store, {"session_id": "zzz"}))
        self.assertIn("error", result)

    def test_missing_session_id_returns_error(self):
        result = json.loads(_tool_get_session_summary(self.store, {}))
        self.assertIn("error", result)


# ---------------------------------------------------------------------------
# diff_sessions
# ---------------------------------------------------------------------------

class TestDiffSessions(unittest.TestCase):
    def setUp(self):
        self.store = _make_store()
        _add_session(self.store, "before", events=[
            _tool_call("Bash"), _tool_call("Bash"),
            _file_write("src/a.py"), _file_read("README.md"),
            _error("fail"),
        ], total_tokens=1000)
        _add_session(self.store, "after", events=[
            _tool_call("Bash"), _tool_call("Write"),
            _file_write("src/b.py"), _file_read("README.md"),
        ], total_tokens=800)

    def test_returns_diff_structure(self):
        result = json.loads(_tool_diff_sessions(self.store, {
            "session_a": "before", "session_b": "after",
        }))
        self.assertIn("tool_call_diff", result)
        self.assertIn("files_only_in_a", result)
        self.assertIn("files_only_in_b", result)
        self.assertIn("files_in_both", result)
        self.assertIn("token_delta", result)
        self.assertIn("cost_delta_usd", result)

    def test_token_delta(self):
        result = json.loads(_tool_diff_sessions(self.store, {
            "session_a": "before", "session_b": "after",
        }))
        self.assertEqual(result["token_delta"], -200)

    def test_files_only_in_a(self):
        result = json.loads(_tool_diff_sessions(self.store, {
            "session_a": "before", "session_b": "after",
        }))
        self.assertIn("src/a.py", result["files_only_in_a"])

    def test_files_only_in_b(self):
        result = json.loads(_tool_diff_sessions(self.store, {
            "session_a": "before", "session_b": "after",
        }))
        self.assertIn("src/b.py", result["files_only_in_b"])

    def test_files_in_both(self):
        result = json.loads(_tool_diff_sessions(self.store, {
            "session_a": "before", "session_b": "after",
        }))
        self.assertIn("README.md", result["files_in_both"])

    def test_error_counts(self):
        result = json.loads(_tool_diff_sessions(self.store, {
            "session_a": "before", "session_b": "after",
        }))
        self.assertEqual(result["error_count_a"], 1)
        self.assertEqual(result["error_count_b"], 0)

    def test_missing_session_returns_error(self):
        result = json.loads(_tool_diff_sessions(self.store, {
            "session_a": "before", "session_b": "zzz",
        }))
        self.assertIn("error", result)

    def test_missing_args_returns_error(self):
        result = json.loads(_tool_diff_sessions(self.store, {}))
        self.assertIn("error", result)


# ---------------------------------------------------------------------------
# tools/call dispatch via _handle
# ---------------------------------------------------------------------------

class TestHandleToolsCall(unittest.TestCase):
    def setUp(self):
        self.store = _make_store()
        _add_session(self.store, "t1", events=[_tool_call("Bash")])

    def test_list_sessions_via_handle(self):
        r = _handle(self.store, _rpc("tools/call", {
            "name": "list_sessions", "arguments": {},
        }))
        result = json.loads(r["result"]["content"][0]["text"])
        self.assertEqual(result["count"], 1)

    def test_get_session_via_handle(self):
        r = _handle(self.store, _rpc("tools/call", {
            "name": "get_session", "arguments": {"session_id": "t1"},
        }))
        result = json.loads(r["result"]["content"][0]["text"])
        self.assertEqual(result["event_count"], 1)

    def test_is_error_false_on_success(self):
        r = _handle(self.store, _rpc("tools/call", {
            "name": "list_sessions", "arguments": {},
        }))
        self.assertFalse(r["result"]["isError"])


if __name__ == "__main__":
    unittest.main()
