"""Tests for the MCP server (agent-strace mcp)."""

import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from agent_trace.mcp_server import (
    _handle,
    _tool_diff_sessions,
    _tool_get_session,
    _tool_get_session_summary,
    _tool_health,
    _tool_list_sessions,
    _tool_search_events,
    _tool_skills,
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

    def test_tools_list_returns_all_tools(self):
        r = _handle(self.store, _rpc("tools/list", {}))
        names = [t["name"] for t in r["result"]["tools"]]
        self.assertIn("list_sessions", names)
        self.assertIn("get_session", names)
        self.assertIn("search_events", names)
        self.assertIn("get_session_summary", names)
        self.assertIn("diff_sessions", names)
        self.assertIn("health", names)
        self.assertIn("skills", names)
        self.assertEqual(len(names), 7)

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


# ---------------------------------------------------------------------------
# health and skills
# ---------------------------------------------------------------------------

class TestServerVersion(unittest.TestCase):
    def test_initialize_reports_package_version(self):
        from agent_trace import __version__
        r = _handle(_make_store(), _rpc("initialize", {}))
        self.assertEqual(r["result"]["serverInfo"]["version"], __version__)


class TestHealthTool(unittest.TestCase):
    def setUp(self):
        self.store = _make_store()

    def test_empty_store_returns_error(self):
        result = json.loads(_tool_health(self.store, {}))
        self.assertIn("error", result)

    def test_not_found_returns_error(self):
        result = json.loads(_tool_health(self.store, {"session_id": "nope"}))
        self.assertIn("session not found", result["error"])

    def test_reports_named_session(self):
        _add_session(self.store, "health-1", [_tool_call("bash", 1.0), _tool_result(0, 2.0)])
        result = json.loads(_tool_health(self.store, {"session_id": "health"}))
        self.assertEqual(result["session_id"], "health-1")
        self.assertIn("status", result)
        self.assertIn("capture_matrix", result)
        self.assertIn("redaction", result)

    def test_defaults_to_latest_session(self):
        _add_session(self.store, "only-one", [_tool_call("bash", 1.0)])
        result = json.loads(_tool_health(self.store, {}))
        self.assertEqual(result["session_id"], "only-one")

    def test_via_handle(self):
        _add_session(self.store, "handle-health", [])
        r = _handle(self.store, _rpc("tools/call", {"name": "health", "arguments": {}}))
        self.assertFalse(r["result"]["isError"])
        self.assertEqual(json.loads(r["result"]["content"][0]["text"])["session_id"], "handle-health")


class TestSkillsTool(unittest.TestCase):
    def setUp(self):
        self.store = _make_store()
        self.repo = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.repo, True)
        skill_dir = self.repo / ".github" / "skills" / "deploy-check"
        skill_dir.mkdir(parents=True)
        self.skill_file = skill_dir / "SKILL.md"
        self.skill_file.write_text("---\nname: deploy-check\ndescription: test\n---\nSECRET_BODY\n")

    def _session(self, session_id: str, load: bool) -> None:
        events = [TraceEvent(event_type=EventType.SESSION_START, timestamp=1.0,
                             data={"cwd": str(self.repo)})]
        if load:
            events.append(_file_read(str(self.skill_file), 2.0))
        _add_session(self.store, session_id, events)

    def test_session_report_lists_loaded_skill(self):
        self._session("skills-1", load=True)
        text = _tool_skills(self.store, {"session_id": "skills-1"})
        self.assertNotIn("SECRET_BODY", text)
        result = json.loads(text)
        self.assertIn("deploy-check", [s["name"] for s in result["loaded"]])

    def test_aggregate_report_lists_never_loaded(self):
        for i in range(2):
            self._session(f"agg-{i}", load=False)
        result = json.loads(_tool_skills(self.store, {"since": "7d", "min_sessions": 1}))
        self.assertIn("deploy-check", json.dumps(result["never_loaded"]))

    def test_not_found_returns_error(self):
        result = json.loads(_tool_skills(self.store, {"session_id": "missing"}))
        self.assertIn("session not found", result["error"])

    def test_bad_since_returns_error(self):
        result = json.loads(_tool_skills(self.store, {"since": "not-a-date"}))
        self.assertIn("error", result)

    def test_bad_min_sessions_returns_error(self):
        result = json.loads(_tool_skills(self.store, {"min_sessions": 0}))
        self.assertIn("error", result)

    def test_listed_in_tools(self):
        r = _handle(self.store, _rpc("tools/list", {}))
        names = [t["name"] for t in r["result"]["tools"]]
        self.assertIn("health", names)
        self.assertIn("skills", names)
