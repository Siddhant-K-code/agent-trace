"""Tests for permission audit trail."""

import io
import json
import tempfile
import unittest
from pathlib import Path

from agent_trace.audit import (
    AuditReport,
    Policy,
    _cmd_matches,
    _glob_match,
    _is_sensitive,
    audit_session,
    format_audit,
)
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore


def _make_event(event_type: EventType, ts: float, session_id: str, **data) -> TraceEvent:
    return TraceEvent(event_type=event_type, timestamp=ts, session_id=session_id, data=data)


def _make_store(events: list[TraceEvent], session_id: str = "sess1") -> tuple[TraceStore, tempfile.TemporaryDirectory]:
    tmp = tempfile.TemporaryDirectory()
    store = TraceStore(tmp.name)
    meta = SessionMeta(session_id=session_id, started_at=0.0,
                       total_duration_ms=5000)
    store.create_session(meta)
    for e in events:
        store.append_event(session_id, e)
    store.update_meta(meta)
    return store, tmp


def _write_policy(d: dict, tmp_dir: str) -> str:
    path = str(Path(tmp_dir) / ".agent-scope.json")
    Path(path).write_text(json.dumps(d))
    return path


class TestGlobMatch(unittest.TestCase):
    def test_exact_match(self):
        self.assertTrue(_glob_match(".env", [".env"]))

    def test_glob_pattern(self):
        self.assertTrue(_glob_match("src/auth.py", ["src/**"]))

    def test_no_match(self):
        self.assertFalse(_glob_match("README.md", ["src/**", "tests/**"]))

    def test_filename_match(self):
        self.assertTrue(_glob_match("config/.env", [".env"]))


class TestIsSensitive(unittest.TestCase):
    def test_dotenv(self):
        self.assertTrue(_is_sensitive(".env"))

    def test_dotenv_local(self):
        self.assertTrue(_is_sensitive(".env.local"))

    def test_pem_file(self):
        self.assertTrue(_is_sensitive("certs/server.pem"))

    def test_github_workflow(self):
        self.assertTrue(_is_sensitive(".github/workflows/deploy.yml"))

    def test_normal_file(self):
        self.assertFalse(_is_sensitive("src/auth.py"))

    def test_readme(self):
        self.assertFalse(_is_sensitive("README.md"))


class TestCmdMatches(unittest.TestCase):
    def test_exact(self):
        self.assertTrue(_cmd_matches("pytest", ["pytest"]))

    def test_prefix(self):
        self.assertTrue(_cmd_matches("uv run pytest", ["uv run"]))

    def test_no_match(self):
        self.assertFalse(_cmd_matches("curl https://example.com", ["pytest", "cat"]))

    def test_case_insensitive(self):
        self.assertTrue(_cmd_matches("PYTEST", ["pytest"]))

    def test_prefix_no_false_positive(self):
        # "curl" pattern must not match "curling" command
        self.assertFalse(_cmd_matches("curling --help", ["curl"]))

    def test_prefix_with_args_matches(self):
        # "curl" pattern should match "curl https://example.com"
        self.assertTrue(_cmd_matches("curl https://example.com", ["curl"]))


class TestNoPolicyFile(unittest.TestCase):
    def test_all_entries_no_policy(self):
        events = [
            _make_event(EventType.TOOL_CALL, 1.0, "sess1",
                        tool_name="Read", arguments={"file_path": "src/auth.py"}),
            _make_event(EventType.TOOL_CALL, 2.0, "sess1",
                        tool_name="Bash", arguments={"command": "pytest"}),
        ]
        store, tmp = _make_store(events)
        report = audit_session(store, "sess1", policy_path="/nonexistent/.agent-scope.json")
        self.assertFalse(report.policy_loaded)
        self.assertEqual(len(report.denied), 0)
        self.assertTrue(all(e.verdict == "no_policy" for e in report.entries))
        tmp.cleanup()


class TestFileReadPolicy(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.policy_path = _write_policy({
            "files": {
                "read": {"allow": ["src/**", "tests/**"], "deny": [".env"]}
            }
        }, self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_allowed_read(self):
        events = [_make_event(EventType.TOOL_CALL, 1.0, "sess1",
                              tool_name="Read", arguments={"file_path": "src/auth.py"})]
        store, tmp2 = _make_store(events)
        report = audit_session(store, "sess1", policy_path=self.policy_path)
        self.assertEqual(report.entries[0].verdict, "allowed")
        tmp2.cleanup()

    def test_denied_read(self):
        events = [_make_event(EventType.TOOL_CALL, 1.0, "sess1",
                              tool_name="Read", arguments={"file_path": ".env"})]
        store, tmp2 = _make_store(events)
        report = audit_session(store, "sess1", policy_path=self.policy_path)
        self.assertEqual(report.entries[0].verdict, "denied")
        tmp2.cleanup()

    def test_not_in_allow_list(self):
        events = [_make_event(EventType.TOOL_CALL, 1.0, "sess1",
                              tool_name="Read", arguments={"file_path": "README.md"})]
        store, tmp2 = _make_store(events)
        report = audit_session(store, "sess1", policy_path=self.policy_path)
        self.assertEqual(report.entries[0].verdict, "denied")
        tmp2.cleanup()


class TestCommandPolicy(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.policy_path = _write_policy({
            "commands": {
                "allow": ["pytest", "uv run", "cat"],
                "deny": ["rm -rf", "curl", "wget"]
            }
        }, self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_allowed_command(self):
        events = [_make_event(EventType.TOOL_CALL, 1.0, "sess1",
                              tool_name="Bash", arguments={"command": "pytest tests/"})]
        store, tmp2 = _make_store(events)
        report = audit_session(store, "sess1", policy_path=self.policy_path)
        cmd_entries = [e for e in report.entries if e.action.startswith("Ran:")]
        self.assertEqual(cmd_entries[0].verdict, "allowed")
        tmp2.cleanup()

    def test_denied_command(self):
        events = [_make_event(EventType.TOOL_CALL, 1.0, "sess1",
                              tool_name="Bash", arguments={"command": "curl https://evil.com"})]
        store, tmp2 = _make_store(events)
        report = audit_session(store, "sess1", policy_path=self.policy_path)
        cmd_entries = [e for e in report.entries if e.action.startswith("Ran:")]
        self.assertEqual(cmd_entries[0].verdict, "denied")
        tmp2.cleanup()


class TestSensitiveFileDetection(unittest.TestCase):
    def test_sensitive_flagged_without_policy(self):
        events = [_make_event(EventType.TOOL_CALL, 1.0, "sess1",
                              tool_name="Read", arguments={"file_path": ".env"})]
        store, tmp = _make_store(events)
        report = audit_session(store, "sess1", policy_path="/nonexistent")
        self.assertTrue(report.entries[0].sensitive)
        tmp.cleanup()

    def test_normal_file_not_sensitive(self):
        events = [_make_event(EventType.TOOL_CALL, 1.0, "sess1",
                              tool_name="Read", arguments={"file_path": "src/auth.py"})]
        store, tmp = _make_store(events)
        report = audit_session(store, "sess1", policy_path="/nonexistent")
        self.assertFalse(report.entries[0].sensitive)
        tmp.cleanup()


class TestNetworkAccessCheck(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.policy_path = _write_policy({
            "network": {"deny_all": True, "allow": ["localhost", "127.0.0.1"]}
        }, self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_denied_external_url(self):
        events = [_make_event(EventType.TOOL_CALL, 1.0, "sess1",
                              tool_name="Bash",
                              arguments={"command": "curl https://example.com/data"})]
        store, tmp2 = _make_store(events)
        report = audit_session(store, "sess1", policy_path=self.policy_path)
        net_entries = [e for e in report.entries if "Network access" in e.action]
        self.assertTrue(len(net_entries) > 0)
        self.assertEqual(net_entries[0].verdict, "denied")
        tmp2.cleanup()

    def test_allowed_localhost(self):
        events = [_make_event(EventType.TOOL_CALL, 1.0, "sess1",
                              tool_name="Bash",
                              arguments={"command": "curl http://localhost:8080/health"})]
        store, tmp2 = _make_store(events)
        report = audit_session(store, "sess1", policy_path=self.policy_path)
        net_entries = [e for e in report.entries if "Network access" in e.action]
        self.assertTrue(len(net_entries) > 0)
        self.assertEqual(net_entries[0].verdict, "allowed")
        tmp2.cleanup()


class TestFullAuditReport(unittest.TestCase):
    def test_report_structure(self):
        events = [
            _make_event(EventType.USER_PROMPT, 0.0, "sess1", prompt="do it"),
            _make_event(EventType.TOOL_CALL, 1.0, "sess1",
                        tool_name="Read", arguments={"file_path": "src/auth.py"}),
            _make_event(EventType.TOOL_CALL, 2.0, "sess1",
                        tool_name="Bash", arguments={"command": "pytest"}),
            _make_event(EventType.TOOL_CALL, 3.0, "sess1",
                        tool_name="Read", arguments={"file_path": ".env"}),
            _make_event(EventType.SESSION_END, 4.0, "sess1"),
        ]
        tmp = tempfile.TemporaryDirectory()
        policy_path = _write_policy({
            "files": {"read": {"allow": ["src/**"], "deny": [".env"]}},
            "commands": {"allow": ["pytest"]},
        }, tmp.name)
        store, tmp2 = _make_store(events)
        report = audit_session(store, "sess1", policy_path=policy_path)

        self.assertEqual(report.total_tool_calls, 3)
        self.assertTrue(len(report.denied) > 0)
        self.assertTrue(len(report.sensitive_accesses) > 0)
        tmp.cleanup()
        tmp2.cleanup()

    def test_format_output_contains_violations(self):
        events = [
            _make_event(EventType.TOOL_CALL, 1.0, "sess1",
                        tool_name="Read", arguments={"file_path": ".env"}),
        ]
        tmp = tempfile.TemporaryDirectory()
        policy_path = _write_policy({
            "files": {"read": {"deny": [".env"]}}
        }, tmp.name)
        store, tmp2 = _make_store(events)
        report = audit_session(store, "sess1", policy_path=policy_path)
        buf = io.StringIO()
        format_audit(report, out=buf)
        output = buf.getvalue()
        self.assertIn("Violations", output)
        self.assertIn(".env", output)
        tmp.cleanup()
        tmp2.cleanup()


if __name__ == "__main__":
    unittest.main()
