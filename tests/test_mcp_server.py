"""Tests for the MCP server (agent-strace mcp)."""

import io
import json
import os
import shutil
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from agent_trace.mcp_server import (
    _handle,
    _tool_diff_sessions,
    _tool_get_session,
    _tool_get_session_summary,
    _tool_health,
    _tool_list_sessions,
    _tool_review,
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
        self.assertIn("review", names)
        self.assertEqual(len(names), 8)

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


class TestSkillsCommandMatchRedaction(unittest.TestCase):
    def setUp(self):
        self.store = _make_store()
        self.repo = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.repo, True)
        skill_dir = self.repo / ".github" / "skills" / "deploy-check"
        skill_dir.mkdir(parents=True)
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(
            "---\nname: deploy-check\ndescription: test\n---\n"
            "```bash\nzzdeployctl rollout --fast\n```\n"
        )
        events = [
            TraceEvent(event_type=EventType.SESSION_START, timestamp=1.0, data={"cwd": str(self.repo)}),
            _file_read(str(skill_file), 2.0),
            TraceEvent(event_type=EventType.TOOL_CALL, timestamp=3.0, data={
                "tool_name": "bash", "arguments": {"command": "zzdeployctl rollout --fast"},
            }),
        ]
        _add_session(self.store, "cmd-1", events)

    def test_library_report_contains_signature(self):
        from agent_trace.skills import build_session_report
        report = build_session_report(self.store, "cmd-1")
        self.assertIn("zzdeployctl rollout", json.dumps(report))

    def test_mcp_session_report_omits_signature(self):
        text = _tool_skills(self.store, {"session_id": "cmd-1"})
        self.assertNotIn("zzdeployctl", text)
        loaded = json.loads(text)["loaded"][0]
        match = loaded["follow"]["signals"]["command_match"]
        self.assertEqual(match, {"count": 1, "distinct_signatures": 1})

    def test_mcp_aggregate_report_omits_signature(self):
        text = _tool_skills(self.store, {"since": "7d", "min_sessions": 1})
        self.assertNotIn("zzdeployctl", text)


class TestReviewTool(unittest.TestCase):
    BRANCH = "feature/mcp-review"
    SECRET_PROMPT = "Refactor auth for the zebra-quartz-77 rollout"
    SECRET_COMMAND = "python -m pytest tests -q --zz-secret-flag"

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        cwd = os.getcwd()
        os.chdir(self.tmp)
        self.addCleanup(os.chdir, cwd)
        self.trace_dir = self.tmp / ".agent-traces"
        writer = TraceStore(str(self.trace_dir))
        meta = SessionMeta(session_id="rev0001aaaa", started_at=1000.0, ended_at=1500.0,
                           agent_name="claude-code", attribution={"git_branch": self.BRANCH})
        writer.create_session(meta)
        ts = [1000.0]

        def add(kind, parent="", **data):
            ts[0] += 1
            ev = TraceEvent(event_type=kind, timestamp=ts[0], session_id="rev0001aaaa",
                            parent_id=parent, data=data)
            writer.append_event("rev0001aaaa", ev)
            return ev

        add(EventType.SESSION_START, mode="claude-code-hooks", cwd="/work/repo", redaction="on")
        add(EventType.USER_PROMPT, prompt=self.SECRET_PROMPT)
        add(EventType.TOOL_CALL, tool_name="Read", arguments={"file_path": "/work/repo/src/auth.py"})
        add(EventType.TOOL_CALL, tool_name="Edit", arguments={"file_path": "/work/repo/src/auth.py"})
        call = add(EventType.TOOL_CALL, tool_name="Bash", arguments={"command": self.SECRET_COMMAND})
        add(EventType.TOOL_RESULT, parent=call.event_id, tool_name="Bash", result="exit code 0")
        add(EventType.ASSISTANT_RESPONSE, text="done", hook_event="stop", provider="claude")
        add(EventType.SESSION_END)
        self.store = TraceStore(str(self.trace_dir), read_only=True)
        branch = patch("agent_trace.mcp_server.current_branch", return_value=self.BRANCH)
        branch.start()
        self.addCleanup(branch.stop)

    def _call(self, args: dict) -> dict:
        r = _handle(self.store, _rpc("tools/call", {"name": "review", "arguments": args}))
        return r["result"]

    def _snapshot(self) -> list[str]:
        return sorted(str(p.relative_to(self.tmp)) for p in self.tmp.rglob("*"))

    def test_matches_cli_json(self):
        from agent_trace.cli import build_parser
        from agent_trace.reviewability import cmd_review
        args = build_parser().parse_args(
            ["--trace-dir", str(self.trace_dir), "review", "--format", "json"])
        out = io.StringIO()
        with redirect_stdout(out), \
                patch("agent_trace.reviewability.current_branch", return_value=self.BRANCH):
            cmd_review(args)
        self.assertEqual(json.loads(_tool_review(self.store, {})), json.loads(out.getvalue()))

    def test_links_by_branch_and_explicit_ids(self):
        result = json.loads(_tool_review(self.store, {}))
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["branch"], self.BRANCH)
        self.assertEqual(result["sessions"], [{"session_id": "rev0001aaaa", "link": "branch"}])
        result = json.loads(_tool_review(self.store, {"session_ids": ["rev0001"],
                                                      "branch": "other"}))
        self.assertEqual(result["sessions"], [{"session_id": "rev0001aaaa", "link": "explicit"}])
        self.assertEqual(result["branch"], "other")

    def test_never_returns_prompt_or_command(self):
        for fmt in ("json", "markdown"):
            text = self._call({"format": fmt})["content"][0]["text"]
            self.assertNotIn("zebra-quartz-77", text)
            self.assertNotIn("zz-secret-flag", text)

    def test_markdown_format(self):
        result = self._call({"format": "markdown"})
        self.assertFalse(result["isError"])
        self.assertTrue(result["content"][0]["text"].startswith("## agent-trace reviewability"))

    def test_threshold_options(self):
        result = json.loads(_tool_review(self.store, {
            "fail_on": "partial,missing,inconclusive", "min_confidence": "low",
        }))
        self.assertEqual(result["result"], "fail")
        self.assertTrue(result["failing"])
        self.assertEqual(result["threshold"]["source"], "flags")
        result = json.loads(_tool_review(self.store, {"fail_on": "none"}))
        self.assertEqual(result["result"], "pass")
        result = json.loads(_tool_review(self.store, {"findings": "validation"}))
        self.assertEqual(result["threshold"]["findings"], ["validation"])

    def test_invalid_inputs_return_errors(self):
        cases = [
            {"fail_on": "wrong"},
            {"min_confidence": "very"},
            {"findings": "nope"},
            {"commits": "--output=/tmp/x"},
            {"commits": "-n1"},
            {"session_ids": "rev0001"},
            {"session_ids": [1]},
            {"session_ids": ["missing"]},
            {"format": "text"},
            {"otel_logs": "logs\x00.json"},
            {"branch": 5},
            {"config": "missing.yaml"},
        ]
        for args in cases:
            with self.subTest(args=args):
                result = self._call(args)
                self.assertTrue(result["isError"])
                self.assertIn("error", json.loads(result["content"][0]["text"]))

    def test_otel_logs_reported_as_external_evidence(self):
        logs = self.tmp / "otel.jsonl"
        logs.write_text(json.dumps({
            "attributes": {"event.name": "claude_code.safety_event", "session.id": "s1",
                           "category": "cyber", "decision": "reject"},
            "body": "claude_code.safety_event",
        }) + "\n")
        result = json.loads(_tool_review(self.store, {"otel_logs": "otel.jsonl"}))
        evidence = next(f for f in result["findings"] if f["name"] == "evidence")
        external = evidence["facts"]["external"]
        self.assertEqual(external["otel_logs"], "otel.jsonl")
        self.assertFalse(external["graded"])
        self.assertEqual(external["signals"][0]["status"], "present")
        self.assertNotIn(str(self.tmp), json.dumps(external))

    def test_read_only(self):
        before = self._snapshot()
        with patch("agent_trace.pr_comment.post_or_update_comment") as post:
            self._call({"format": "markdown"})
            self._call({"fail_on": "partial"})
            post.assert_not_called()
        self.assertEqual(self._snapshot(), before)


class TestJsonRpcValidation(unittest.TestCase):
    def setUp(self):
        self.store = _make_store()

    def _code(self, request):
        r = _handle(self.store, request)
        return r["error"]["code"]

    def test_non_object_request(self):
        for request in ([], "x", 3, None):
            self.assertEqual(self._code(request), -32600)

    def test_wrong_jsonrpc_version(self):
        self.assertEqual(self._code({"jsonrpc": "1.0", "id": 1, "method": "ping"}), -32600)

    def test_non_string_method(self):
        self.assertEqual(self._code({"jsonrpc": "2.0", "id": 1, "method": 5}), -32600)

    def test_invalid_id_type(self):
        r = _handle(self.store, {"jsonrpc": "2.0", "id": {"a": 1}, "method": "ping"})
        self.assertEqual(r["error"]["code"], -32600)
        self.assertIsNone(r["id"])

    def test_params_not_object(self):
        self.assertEqual(self._code({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": [1]}), -32602)

    def test_tool_name_not_string(self):
        self.assertEqual(self._code(_rpc("tools/call", {"name": 7})), -32602)

    def test_arguments_not_object(self):
        self.assertEqual(self._code(_rpc("tools/call", {"name": "list_sessions", "arguments": "x"})), -32602)

    def test_unknown_tool_is_invalid_params(self):
        self.assertEqual(self._code(_rpc("tools/call", {"name": "nope", "arguments": {}})), -32602)

    def test_notification_with_bad_params_gets_no_response(self):
        self.assertIsNone(_handle(self.store, {"jsonrpc": "2.0", "method": "x", "params": 3}))

    def test_tool_exception_sets_is_error(self):
        r = _handle(self.store, _rpc("tools/call", {"name": "list_sessions", "arguments": {"limit": "many"}}))
        self.assertTrue(r["result"]["isError"])
        self.assertIn("list_sessions failed", r["result"]["content"][0]["text"])

    def test_tool_error_payload_sets_is_error(self):
        r = _handle(self.store, _rpc("tools/call", {"name": "get_session", "arguments": {"session_id": "nope"}}))
        self.assertTrue(r["result"]["isError"])

    def test_stdio_survives_bad_requests(self):
        import io
        import sys
        from agent_trace.mcp_server import run_stdio
        lines = [
            "not json",
            "[]",
            json.dumps([{"jsonrpc": "2.0", "id": 1, "method": "ping"}]),
            json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "skills", "arguments": []}}),
            json.dumps({"jsonrpc": "2.0", "id": 3, "method": "ping"}),
        ]
        stdin, stdout = sys.stdin, sys.stdout
        sys.stdin = io.StringIO("\n".join(lines) + "\n")
        sys.stdout = io.StringIO()
        try:
            run_stdio(self.store)
            out = sys.stdout.getvalue()
        finally:
            sys.stdin, sys.stdout = stdin, stdout
        responses = [json.loads(line) for line in out.splitlines()]
        self.assertEqual([r.get("error", {}).get("code") for r in responses], [-32700, -32600, -32600, -32602, None])
        self.assertEqual(responses[-1]["id"], 3)


class TestReadOnlyStore(unittest.TestCase):
    def test_mcp_store_does_not_write(self):
        import argparse
        from unittest.mock import patch
        from agent_trace import mcp_server
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        trace_dir = root / ".agent-traces"
        seen = {}

        def fake_run(store):
            seen["read_only"] = store.read_only
            _handle(store, _rpc("tools/call", {"name": "list_sessions", "arguments": {}}))
            _handle(store, _rpc("tools/call", {"name": "skills", "arguments": {}}))

        import io
        with patch.object(mcp_server, "run_stdio", fake_run), patch("sys.stderr", io.StringIO()):
            mcp_server.cmd_mcp(argparse.Namespace(trace_dir=str(trace_dir)))
        self.assertTrue(seen["read_only"])
        self.assertFalse(trace_dir.exists())

    def test_read_only_skips_gitignore_and_blocks_writes(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        TraceStore(root)
        (root / ".gitignore").unlink()
        store = TraceStore(root, read_only=True)
        self.assertFalse((root / ".gitignore").exists())
        with self.assertRaises(PermissionError):
            store.create_session(SessionMeta(session_id="ro-1"))
        self.assertEqual(list(root.iterdir()), [])
