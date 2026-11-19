"""Tests for compliance export (issue #144)."""

import copy
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import _escapes
    import _no_telemetry  # noqa: F401
    from test_hash_chain import (
        _CliCase,
        _edit_hole,
        _edit_strip_rest,
        _make_probe_session,
        _read_lines,
        _strip,
        _upgrade_prefix,
        _write_lines,
    )
    from test_store import raw_separator_session
except ImportError:
    from tests import _escapes
    from tests import _no_telemetry  # noqa: F401
    from tests.test_hash_chain import (
        _CliCase,
        _edit_hole,
        _edit_strip_rest,
        _make_probe_session,
        _read_lines,
        _strip,
        _upgrade_prefix,
        _write_lines,
    )
    from tests.test_store import raw_separator_session

from agent_trace.audit import verify_chain
from agent_trace.cli import build_parser
from agent_trace.compliance import (
    build_audit_readiness,
    export_compliance,
    export_compliance_bulk,
    export_eu_ai_act,
    format_audit_readiness,
    select_sessions,
    verify_eu_ai_act_export,
)
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore


def _make_session(store: TraceStore, tool_names: list[str],
                  errors: int = 0) -> str:
    meta = SessionMeta(agent_name="test-agent")
    meta.started_at = time.time() - 120
    meta.ended_at = time.time()
    meta.total_duration_ms = 120_000.0
    meta.tool_calls = len(tool_names)
    store.create_session(meta)
    for name in tool_names:
        store.append_event(meta.session_id, TraceEvent(
            event_type=EventType.TOOL_CALL,
            session_id=meta.session_id,
            data={"tool_name": name},
        ))
    for i in range(errors):
        store.append_event(meta.session_id, TraceEvent(
            event_type=EventType.ERROR,
            session_id=meta.session_id,
            data={"error": f"error-{i}", "error_type": "RuntimeError"},
        ))
    store.update_meta(meta)
    return meta.session_id


class TestEuAiActReport(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = TraceStore(self.tmpdir)

    def test_has_required_keys(self):
        sid = _make_session(self.store, ["Bash", "Read"])
        report = export_compliance(self.store, sid, "eu-ai-act")
        fw = report["frameworks"]["eu-ai-act"]
        self.assertIn("article_13_transparency", fw)
        self.assertIn("article_9_risk_management", fw)

    def test_tools_used_populated(self):
        sid = _make_session(self.store, ["Bash", "Read", "Bash"])
        report = export_compliance(self.store, sid, "eu-ai-act")
        tools = report["frameworks"]["eu-ai-act"]["article_13_transparency"]["tools_used"]
        self.assertIn("Bash", tools)
        self.assertIn("Read", tools)

    def test_error_count(self):
        sid = _make_session(self.store, ["Bash"], errors=2)
        report = export_compliance(self.store, sid, "eu-ai-act")
        self.assertEqual(
            report["frameworks"]["eu-ai-act"]["article_9_risk_management"]["error_count"], 2
        )


class TestSoc2Report(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = TraceStore(self.tmpdir)

    def test_has_cc6_and_cc7(self):
        sid = _make_session(self.store, ["Bash"])
        report = export_compliance(self.store, sid, "soc2")
        fw = report["frameworks"]["soc2"]
        self.assertIn("cc6_logical_access", fw)
        self.assertIn("cc7_system_operations", fw)

    def test_resources_accessed(self):
        sid = _make_session(self.store, ["Bash", "Write"])
        report = export_compliance(self.store, sid, "soc2")
        resources = report["frameworks"]["soc2"]["cc6_logical_access"]["resources_accessed"]
        self.assertIn("Bash", resources)
        self.assertIn("Write", resources)

    def test_error_details_populated(self):
        sid = _make_session(self.store, [], errors=1)
        report = export_compliance(self.store, sid, "soc2")
        details = report["frameworks"]["soc2"]["cc7_system_operations"]["error_details"]
        self.assertEqual(len(details), 1)


class TestHipaaReport(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = TraceStore(self.tmpdir)

    def test_has_audit_controls(self):
        sid = _make_session(self.store, ["Bash"])
        report = export_compliance(self.store, sid, "hipaa")
        fw = report["frameworks"]["hipaa"]
        self.assertIn("section_164_312_audit_controls", fw)
        self.assertIn("section_164_312_integrity", fw)

    def test_access_log_entries(self):
        sid = _make_session(self.store, ["Bash", "Read"])
        report = export_compliance(self.store, sid, "hipaa")
        log = report["frameworks"]["hipaa"]["section_164_312_audit_controls"]["access_log"]
        self.assertEqual(len(log), 2)
        actions = [e["action"] for e in log]
        self.assertIn("Bash", actions)

    def test_failure_counted(self):
        sid = _make_session(self.store, [], errors=2)
        report = export_compliance(self.store, sid, "hipaa")
        self.assertEqual(
            report["frameworks"]["hipaa"]["section_164_312_audit_controls"]["failures"], 2
        )


class TestAllFrameworks(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = TraceStore(self.tmpdir)

    def test_all_includes_three_frameworks(self):
        sid = _make_session(self.store, ["Bash"])
        report = export_compliance(self.store, sid, "all")
        self.assertIn("eu-ai-act", report["frameworks"])
        self.assertIn("soc2", report["frameworks"])
        self.assertIn("hipaa", report["frameworks"])

    def test_report_is_json_serialisable(self):
        sid = _make_session(self.store, ["Bash", "Read"], errors=1)
        report = export_compliance(self.store, sid, "all")
        serialised = json.dumps(report)
        self.assertIsInstance(serialised, str)

    def test_missing_session_returns_error(self):
        report = export_compliance(self.store, "nonexistent-session", "all")
        self.assertIn("error", report)


class TestBulkExport(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = TraceStore(self.tmpdir)

    def test_bulk_returns_list(self):
        _make_session(self.store, ["Bash"])
        _make_session(self.store, ["Read"])
        reports = export_compliance_bulk(self.store, "soc2", since_days=1)
        self.assertIsInstance(reports, list)
        self.assertEqual(len(reports), 2)

    def test_bulk_excludes_old_sessions(self):
        # Create a session 60 days ago
        meta = SessionMeta(agent_name="old")
        meta.started_at = time.time() - 86400 * 60
        meta.ended_at = meta.started_at + 60
        self.store.create_session(meta)
        self.store.update_meta(meta)

        reports = export_compliance_bulk(self.store, "soc2", since_days=30)
        self.assertEqual(len(reports), 0)


class TestEuAiActArticleExport(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = TraceStore(self.tmpdir)

    def test_article_12_and_13_blocks_present(self):
        sid = _make_session(self.store, ["Bash", "Read"], errors=1)
        report = export_eu_ai_act(self.store, [sid])

        self.assertIn("compliance_metadata", report)
        self.assertIn("article_12", report)
        self.assertIn("article_13", report)
        self.assertEqual(report["compliance_metadata"]["articles_covered"], ["Article 12", "Article 13"])
        self.assertEqual(report["compliance_metadata"]["session_count"], 1)

    def test_article_12_events_include_hashes(self):
        sid = _make_session(self.store, ["Bash", "Read"])
        report = export_eu_ai_act(self.store, [sid])
        events = report["sessions"][0]["article_12"]["events"]

        self.assertGreaterEqual(len(events), 2)
        self.assertIn("prev_hash", events[1])
        self.assertIn("line_hash", events[1])
        self.assertTrue(events[1]["line_hash"])

    def test_line_hashes_cover_a_line_with_a_raw_line_separator(self):
        sid, lines = raw_separator_session(self.store, "\u2028")
        report = export_eu_ai_act(self.store, [sid])
        article_12 = report["sessions"][0]["article_12"]

        self.assertEqual(
            [event["line_hash"] for event in article_12["events"]],
            [hashlib.sha256(line.encode("utf-8")).hexdigest() for line in lines],
        )
        self.assertTrue(article_12["trace_integrity"]["hash_chain_valid"])
        path = Path(self.tmpdir) / "export.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        self.assertTrue(verify_eu_ai_act_export(str(path))["ok"])

    def test_article_13_lists_tools_and_oversight(self):
        sid = _make_session(self.store, ["Bash"], errors=1)
        report = export_eu_ai_act(self.store, [sid])
        article_13 = report["sessions"][0]["article_13"]

        self.assertIn("Bash", article_13["capabilities_summary"]["tools_used"])
        self.assertEqual(len(article_13["human_oversight_points"]), 1)

    def test_select_sessions_honours_since_until(self):
        recent = _make_session(self.store, ["Bash"])
        old_meta = SessionMeta(agent_name="old")
        old_meta.started_at = time.time() - 86400 * 60
        old_meta.ended_at = old_meta.started_at + 10
        self.store.create_session(old_meta)
        self.store.update_meta(old_meta)

        selected = select_sessions(self.store, since="30d")

        self.assertIn(recent, selected)
        self.assertNotIn(old_meta.session_id, selected)

    def test_verify_export_detects_tampered_hash_link(self):
        sid = _make_session(self.store, ["Bash", "Read"])
        report = export_eu_ai_act(self.store, [sid])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(report, f)
            ok_path = f.name

        self.assertTrue(verify_eu_ai_act_export(ok_path)["ok"])

        report["sessions"][0]["article_12"]["events"][1]["prev_hash"] = "bad"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(report, f)
            bad_path = f.name

        result = verify_eu_ai_act_export(bad_path)
        self.assertFalse(result["ok"])
        self.assertEqual(len(result["failures"]), 1)


class TestAuditReadiness(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = TraceStore(self.tmpdir)

    def test_readiness_report_has_checks(self):
        _make_session(self.store, ["Bash"])

        report = build_audit_readiness(self.store, retention_days=0)

        self.assertIn("hash_chain_integrity", report["checks"])
        self.assertIn("retention_coverage", report["checks"])
        self.assertIn("timestamp_continuity", report["checks"])
        self.assertIn("compliance_score", report)

    def test_empty_store_is_not_ready(self):
        report = build_audit_readiness(self.store)
        self.assertFalse(report["ready"])
        self.assertEqual(report["compliance_score"], 0)


class TestComplianceCliParser(unittest.TestCase):
    def test_export_accepts_eu_ai_act_format_and_all(self):
        parser = build_parser()
        args = parser.parse_args(["export", "--format", "eu-ai-act", "--all", "--since", "30d"])
        self.assertEqual(args.format, "eu-ai-act")
        self.assertTrue(args.all)

    def test_audit_readiness_command_registered(self):
        parser = build_parser()
        args = parser.parse_args(["audit-readiness", "--format", "json"])
        self.assertEqual(args.command, "audit-readiness")
        self.assertEqual(args.format, "json")

    def test_verify_from_export_registered(self):
        parser = build_parser()
        args = parser.parse_args(["verify", "--from-export", "audit.json"])
        self.assertEqual(args.command, "verify")
        self.assertEqual(args.from_export, "audit.json")



# ---------------------------------------------------------------------------
# Hash chain state in the EU AI Act export, verify --from-export and
# audit-readiness. The sessions are the probe sessions from test_hash_chain.
# ---------------------------------------------------------------------------

def _change_stream(store: TraceStore, sid: str, change=None, count: int = 7) -> str:
    path = _make_probe_session(store, sid, count)
    if change is not None:
        lines = _read_lines(path)
        change(lines)
        _write_lines(path, lines)
    return sid


def _strip_all(lines):
    _strip(lines, range(1, len(lines)))


def _add_garbage_line(lines):
    lines.insert(4, "not json")


class TestEuAiActChainState(unittest.TestCase):
    def setUp(self):
        root = Path(tempfile.mkdtemp(prefix="compliance-chain-"))
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        self.store = TraceStore(str(root), use_workspace_env=False)

    def test_flags_follow_the_chain_state(self):
        for sid, change, count, state in (
            ("f-intact", None, 7, "unanchored"),
            ("f-edit-hole", _edit_hole, 7, "broken"),
            ("f-partial", _edit_strip_rest, 7, "partial"),
            ("f-legacy", _strip_all, 7, "legacy"),
            ("f-empty", None, 0, "empty"),
        ):
            with self.subTest(sid):
                _change_stream(self.store, sid, change, count)
                report = export_eu_ai_act(self.store, [sid])
                integrity = report["sessions"][0]["article_12"]["trace_integrity"]
                valid = state == "unanchored"
                self.assertIs(integrity["hash_chain_valid"], valid)
                self.assertEqual(integrity["hash_chain_state"], state)
                self.assertIs(report["compliance_metadata"]["tamper_evident"], valid)
                obligations = report["article_12"]["logging_obligations"]
                self.assertEqual(obligations[0]["trace_integrity"], integrity)

    def test_tamper_evident_needs_every_session_valid(self):
        for sid, change in (("m-intact-a", None), ("m-intact-b", None),
                            ("m-partial", _edit_strip_rest)):
            _change_stream(self.store, sid, change)
        both = export_eu_ai_act(self.store, ["m-intact-a", "m-intact-b"])
        self.assertIs(both["compliance_metadata"]["tamper_evident"], True)
        mixed = export_eu_ai_act(self.store, ["m-intact-a", "m-partial"])
        self.assertIs(mixed["compliance_metadata"]["tamper_evident"], False)
        # A session with missing links is still exported.
        self.assertEqual(
            [(s["session_id"], s["article_12"]["trace_integrity"]["hash_chain_valid"])
             for s in mixed["sessions"]],
            [("m-intact-a", True), ("m-partial", False)],
        )
        nothing = export_eu_ai_act(self.store, [])
        self.assertIs(nothing["compliance_metadata"]["tamper_evident"], False)
        self.assertEqual(nothing["compliance_metadata"]["session_count"], 0)

    def test_root_hash_and_event_count_match_verify(self):
        for sid, change in (("r-intact", None), ("r-partial", _edit_strip_rest)):
            with self.subTest(sid):
                _change_stream(self.store, sid, change)
                report = export_eu_ai_act(self.store, [sid])
                integrity = report["sessions"][0]["article_12"]["trace_integrity"]
                chain = verify_chain(self.store, sid)
                self.assertEqual(integrity["root_hash"], chain.root_hash)
                self.assertEqual(integrity["event_count"], chain.total_events)


EXPORT_SID = "x-c01"
LEGACY_SID = "x-c05"
S = EXPORT_SID

SEQ = "seq is not the event's position"
FIRST_PREV = "first event has a prev_hash"
PREV_MISSING = "prev_hash is missing"
PREV_NOT_HEX = "prev_hash is not a SHA-256 hex digest"
PREV_MISMATCH = "prev_hash does not match the previous event's line_hash"
LINE_NOT_HEX = "line_hash is not a SHA-256 hex digest"
NOT_OBJECT = "event is not an object"
NO_EVENTS = "session has no events"
VALID = "hash_chain_valid is not true"
STATE = "hash_chain_state is not unanchored"
ROOT = "root_hash does not match the last event's line_hash"
COUNT = "event_count does not match the number of events"
SESSION_COUNT = "session_count does not match the number of sessions"
TAMPER = "tamper_evident is not true"
OBLIGATIONS = "article_12.logging_obligations does not match the sessions"
METADATA = "compliance_metadata is missing or not an object"
ARTICLE_12 = "article_12 is missing or not an object"
INTEGRITY = "trace_integrity is missing or not an object"


def _session(doc):
    return doc["sessions"][0]


def _events(doc):
    return doc["sessions"][0]["article_12"]["events"]


def _integrity(doc):
    return doc["sessions"][0]["article_12"]["trace_integrity"]


def _sync(doc):
    """Rebuild logging_obligations from the sessions, as export_eu_ai_act does."""
    doc["article_12"]["logging_obligations"] = [
        {"session_id": session["session_id"], **session["article_12"]}
        for session in doc["sessions"]
    ]


def _both_copies(change):
    """Make *change* to the sessions, then copy them to logging_obligations."""
    def edit(doc):
        change(doc)
        _sync(doc)
    return edit


def _delete_event(blank=False, copy_hash=False, count=False, renumber=False):
    def change(doc):
        events = _events(doc)
        del events[4]
        if blank:
            events[4]["prev_hash"] = ""
        if copy_hash:
            events[4]["prev_hash"] = events[3]["line_hash"]
        if count:
            _integrity(doc)["event_count"] = len(events)
            _integrity(doc)["root_hash"] = events[-1]["line_hash"]
        if renumber:
            for position, event in enumerate(events, start=1):
                event["seq"] = position
    return _both_copies(change)


def _edit_summary(doc):
    _events(doc)[4]["input_summary"] = '{"arguments": {"command": "echo ok"}, "tool_name": "Bash"}'


def _blank_hashes(doc):
    for event in _events(doc):
        event["prev_hash"] = ""
        event["line_hash"] = ""


def _junk_hashes(doc):
    for position, event in enumerate(_events(doc)):
        event["line_hash"] = "x"
        if position:
            event["prev_hash"] = "x"
    _integrity(doc)["root_hash"] = "x"


def _set(where, key, value):
    return _both_copies(lambda doc: where(doc).__setitem__(key, value))


def _drop(where, key):
    return _both_copies(lambda doc: where(doc).pop(key))


def _replace(value):
    return lambda doc: copy.deepcopy(value)


def _metadata(doc):
    return doc["compliance_metadata"]


def _event(position):
    return lambda doc: _events(doc)[position - 1]


def _obligation_integrity(doc):
    return doc["article_12"]["logging_obligations"][0]["trace_integrity"]


# Each edit changes a copy of an export of the intact probe session. An edit
# that returns a value replaces the whole file. Each expected failure is
# (session_id, seq, reason).
EXPORT_EDITS = [
    ("unchanged", lambda doc: None, []),
    ("delete+blank", _delete_event(blank=True),
     [(S, 5, SEQ), (S, 5, PREV_MISSING), (S, 6, SEQ), (S, None, COUNT)]),
    ("delete+copy", _delete_event(copy_hash=True), [(S, 5, SEQ), (S, 6, SEQ), (S, None, COUNT)]),
    ("delete+copy+count", _delete_event(copy_hash=True, count=True), [(S, 5, SEQ), (S, 6, SEQ)]),
    ("summary-edit", _both_copies(_edit_summary), []),
    ("valid-flag-false", _set(_integrity, "hash_chain_valid", False), [(S, None, VALID)]),
    ("root-hash-zeroed", _set(_integrity, "root_hash", "0" * 64), [(S, None, ROOT)]),
    ("event-count-99", _set(_integrity, "event_count", 99), [(S, None, COUNT)]),
    ("events-emptied", _set(lambda doc: _session(doc)["article_12"], "events", []),
     [(S, None, NO_EVENTS), (S, None, COUNT)]),
    ("all-hashes-blank", _both_copies(_blank_hashes),
     [(S, 1, LINE_NOT_HEX)]
     + [failure for seq in range(2, 8)
        for failure in ((S, seq, PREV_MISSING), (S, seq, LINE_NOT_HEX))]
     + [(S, None, ROOT)]),
    ("one-copy-edit", _edit_summary, [(None, None, OBLIGATIONS)]),
    # == treats true as equal to 1, so the two copies are compared as JSON text.
    ("one-copy-true-as-1",
     lambda doc: _obligation_integrity(doc).update(hash_chain_valid=1),
     [(None, None, OBLIGATIONS)]),
    ("session-count-2", _set(_metadata, "session_count", 2), [(None, None, SESSION_COUNT)]),
    ("tamper-evident-false", _set(_metadata, "tamper_evident", False), [(None, None, TAMPER)]),
    ("delete+renumber", _delete_event(copy_hash=True, count=True, renumber=True), []),
    ("hashes-junk", _both_copies(_junk_hashes),
     [(S, 1, LINE_NOT_HEX)]
     + [failure for seq in range(2, 8)
        for failure in ((S, seq, PREV_NOT_HEX), (S, seq, LINE_NOT_HEX))]),
    ("first-prev-hash-set", _set(_event(1), "prev_hash", "a" * 64), [(S, 1, FIRST_PREV)]),
    ("empty-object", _replace({}),
     [(None, None, METADATA), (None, None, "sessions is missing"), (None, None, ARTICLE_12)]),
    ("no-sessions", _replace({"sessions": []}),
     [(None, None, METADATA), (None, None, "export has no sessions"), (None, None, ARTICLE_12)]),
    ("top-level-list", _replace([]), [(None, None, "export is not a JSON object")]),
    ("sessions-object", _replace({"sessions": {"a": 1}}),
     [(None, None, METADATA), (None, None, "sessions is not a list"), (None, None, ARTICLE_12)]),
]

# Shapes that are not a well-formed export.
MALFORMED_EXPORTS = [
    ("not-json", _replace("not json"), [(None, None, "export can't be parsed as JSON")]),
    ("session-not-object", lambda doc: doc.update(sessions=[1]),
     [(None, None, "session 1 is not an object")]),
    ("event-not-object", _set(_events, 2, "x"), [(S, 3, NOT_OBJECT)]),
    ("session-article-12-not-object", lambda doc: _session(doc).update(article_12=[]),
     [(S, None, ARTICLE_12)]),
    ("trace-integrity-not-object",
     _set(lambda doc: _session(doc)["article_12"], "trace_integrity", "x"),
     [(S, None, INTEGRITY)]),
    ("metadata-not-object", lambda doc: doc.update(compliance_metadata=[]),
     [(None, None, METADATA)]),
    ("package-article-12-not-object", lambda doc: doc.update(article_12="x"),
     [(None, None, ARTICLE_12)]),
    ("events-not-list", _set(lambda doc: _session(doc)["article_12"], "events", {"a": 1}),
     [(S, None, "events is not a list")]),
    ("events-missing", _drop(lambda doc: _session(doc)["article_12"], "events"),
     [(S, None, NO_EVENTS), (S, None, COUNT)]),
    ("first-prev-hash-null", _set(_event(1), "prev_hash", None), [(S, 1, FIRST_PREV)]),
    ("line-hash-missing", _drop(_event(4), "line_hash"), [(S, 4, LINE_NOT_HEX)]),
    ("prev-hash-uppercase",
     _both_copies(lambda doc: _event(4)(doc).update(prev_hash=_event(4)(doc)["prev_hash"].upper())),
     [(S, 4, PREV_NOT_HEX)]),
    ("prev-hash-not-string", _set(_event(4), "prev_hash", 5), [(S, 4, PREV_NOT_HEX)]),
    ("prev-hash-mismatch", _set(_event(4), "prev_hash", "b" * 64), [(S, 4, PREV_MISMATCH)]),
    ("seq-bool", _set(_event(1), "seq", True), [(S, 1, SEQ)]),
    ("seq-string", _set(_event(2), "seq", "2"), [(S, 2, SEQ)]),
    ("valid-flag-string", _set(_integrity, "hash_chain_valid", "true"), [(S, None, VALID)]),
    ("valid-flag-missing", _drop(_integrity, "hash_chain_valid"), [(S, None, VALID)]),
    ("state-partial", _set(_integrity, "hash_chain_state", "partial"), [(S, None, STATE)]),
    ("event-count-string", _set(_integrity, "event_count", "7"), [(S, None, COUNT)]),
    ("event-count-float", _set(_integrity, "event_count", 7.0), [(S, None, COUNT)]),
    ("session-count-string", _set(_metadata, "session_count", "1"),
     [(None, None, SESSION_COUNT)]),
    ("session-count-true", _set(_metadata, "session_count", True),
     [(None, None, SESSION_COUNT)]),
    ("tamper-evident-string", _set(_metadata, "tamper_evident", "true"), [(None, None, TAMPER)]),
    # An honest export that holds NaN still passes.
    ("nan-in-both-copies", _set(_event(5), "cost_usd", float("nan")), []),
]


class TestVerifyExport(_CliCase):
    def setUp(self):
        super().setUp()
        _make_probe_session(self.store, EXPORT_SID)
        self.base = json.loads(json.dumps(export_eu_ai_act(self.store, [EXPORT_SID])))
        self.exports = self.root / "exports"
        self.exports.mkdir()

    def write(self, name, doc):
        path = self.exports / f"{name}.json"
        path.write_text(doc if isinstance(doc, str) else json.dumps(doc), encoding="utf-8")
        return path

    def build(self, name, edit):
        doc = copy.deepcopy(self.base)
        replaced = edit(doc)
        return self.write(name, doc if replaced is None else replaced)

    def check(self, cases):
        """Check each (name, path, expected failures) in process and through the CLI."""
        commands = []
        for _, path, _ in cases:
            commands += [
                ("verify", "--from-export", str(path)),
                ("verify", "--from-export", str(path), "--format", "json"),
            ]
        runs = iter(self.cli_many(commands))
        for name, path, expected in cases:
            text, as_json = next(runs), next(runs)
            with self.subTest(name):
                result = verify_eu_ai_act_export(path)
                self.assertEqual(
                    [(f["session_id"], f["seq"], f["reason"]) for f in result["failures"]],
                    expected,
                )
                self.assertIs(result["ok"], not expected)
                for failure in result["failures"]:
                    self.assertEqual(set(failure), {"session_id", "seq", "event_id", "reason"})
                for run in (text, as_json):
                    self.assertEqual(run.stderr, "")
                    self.assertEqual(run.returncode, 1 if expected else 0, run.stdout)
                self.assertEqual(json.loads(as_json.stdout), result)
                lines = text.stdout.splitlines()
                if expected:
                    noun = "problem" if len(expected) == 1 else "problems"
                    self.assertEqual(lines[0], f"Export hash chain failed - {len(expected)} {noun}")
                    self.assertEqual(
                        [line.rsplit(": ", 1)[1] for line in lines[1:]],
                        [reason for _, _, reason in expected],
                    )
                else:
                    self.assertEqual(lines, [
                        f"Export hash chain intact - {result['checked_events']} events checked",
                    ])

    def test_export_edits(self):
        self.check([(name, self.build(name, edit), expected)
                    for name, edit, expected in EXPORT_EDITS])

    def test_malformed_exports(self):
        cases = [(name, self.build(name, edit), expected)
                 for name, edit, expected in MALFORMED_EXPORTS]
        cases.append(("missing-file", self.exports / "missing.json",
                      [(None, None, "export file can't be read")]))
        self.check(cases)

    def test_old_exports(self):
        """Exports made before hash_chain_state have no state and a constant tamper_evident."""
        _change_stream(self.store, LEGACY_SID, _strip_all)
        legacy = json.loads(json.dumps(export_eu_ai_act(self.store, [LEGACY_SID])))

        def old_shape(doc):
            _integrity(doc).pop("hash_chain_state", None)
            _integrity(doc)["hash_chain_valid"] = True
            _metadata(doc)["tamper_evident"] = True
            _sync(doc)

        linked = copy.deepcopy(self.base)
        old_shape(linked)
        old_shape(legacy)
        self.check([
            ("old-linked", self.write("old-linked", linked), []),
            ("old-legacy", self.write("old-legacy", legacy),
             [(LEGACY_SID, seq, PREV_MISSING) for seq in range(2, 8)]),
        ])

    def test_text_and_json_output(self):
        paths = [
            self.build("ok", lambda doc: None),
            self.build("one", _set(_event(4), "prev_hash", "b" * 64)),
            self.build("many", _delete_event(blank=True)),
            self.build("package", _set(_metadata, "tamper_evident", False)),
        ]
        commands = [("verify", "--from-export", str(path)) for path in paths]
        commands.append(("verify", "--from-export", str(paths[1]), "--format", "json"))
        ok, one, many, package, one_json = self.cli_many(commands)
        self.assertEqual(ok.stdout, "Export hash chain intact - 7 events checked\n")
        self.assertEqual(
            one.stdout,
            "Export hash chain failed - 1 problem\n"
            f"  session {S}, event 4 (id=ev0000000003): {PREV_MISMATCH}\n",
        )
        self.assertEqual(
            many.stdout,
            "Export hash chain failed - 4 problems\n"
            f"  session {S}, event 5 (id=ev0000000005): {SEQ}\n"
            f"  session {S}, event 5 (id=ev0000000005): {PREV_MISSING}\n"
            f"  session {S}, event 6 (id=ev0000000006): {SEQ}\n"
            f"  session {S}: {COUNT}\n",
        )
        self.assertEqual(package.stdout, f"Export hash chain failed - 1 problem\n  export: {TAMPER}\n")
        self.assertEqual(json.loads(one_json.stdout), {
            "ok": False,
            "checked_sessions": 1,
            "checked_events": 7,
            "failures": [{
                "session_id": S,
                "seq": 4,
                "event_id": "ev0000000003",
                "reason": PREV_MISMATCH,
            }],
        })

    def test_values_from_the_file_show_as_escapes(self):
        def tampered(name, value):
            def change(doc):
                _session(doc)["session_id"] = value
                _event(4)(doc)["event_id"] = value
                _event(4)(doc)["prev_hash"] = "b" * 64
                _integrity(doc)["hash_chain_valid"] = False
            return self.build(name, _both_copies(change))

        esc, plain = tampered("esc", _escapes.PAYLOAD), tampered("plain", "plain")
        text, as_json, plain_text = self.cli_many([
            ("verify", "--from-export", str(esc)),
            ("verify", "--from-export", str(esc), "--format", "json"),
            ("verify", "--from-export", str(plain)),
        ])
        shown = _escapes.PAYLOAD_TEXT
        self.assertEqual(
            text.stdout,
            "Export hash chain failed - 2 problems\n"
            f"  session {shown}, event 4 (id={shown}): {PREV_MISMATCH}\n"
            f"  session {shown}: {VALID}\n",
        )
        _escapes.assert_shows_as_text(self, text.stdout, shown, plain_text.stdout)
        self.assertTrue(as_json.stdout.isascii())
        self.assertEqual(
            [(f["session_id"], f["event_id"]) for f in json.loads(as_json.stdout)["failures"]],
            [(_escapes.PAYLOAD, _escapes.PAYLOAD), (_escapes.PAYLOAD, None)],
        )


def _readiness_checks(sid, *, broken=False, missing_links=False):
    return {
        "hash_chain_integrity": {
            "ok": not broken,
            "checked_sessions": 1,
            "broken_sessions": [sid] if broken else [],
        },
        "hash_chain_presence": {
            "ok": not missing_links,
            "sessions_missing_prev_hashes": [sid] if missing_links else [],
        },
    }


class TestAuditReadinessChain(_CliCase):
    """One session per store. A new store has no retention, which costs 10 points."""

    def readiness(self, sid, change=None, count=7, after=None, age_days=None):
        store = TraceStore(str(self.root / sid), use_workspace_env=False)
        path = _make_probe_session(store, sid, count)
        if change is not None:
            lines = _read_lines(path)
            change(lines)
            _write_lines(path, lines)
        if after is not None:
            after(path)
        if age_days is not None:
            started_at = time.time() - age_days * 86400
            store.mutate_meta(sid, lambda meta: setattr(meta, "started_at", started_at))
        report = build_audit_readiness(store)
        chain_checks = {
            key: report["checks"][key]
            for key in ("hash_chain_integrity", "hash_chain_presence")
        }
        buf = io.StringIO()
        format_audit_readiness(report, out=buf)
        return report, chain_checks, buf.getvalue()

    def test_missing_links_fail_presence(self):
        intact, intact_checks, _ = self.readiness("a-intact")
        self.assertEqual(intact_checks, _readiness_checks("a-intact"))
        self.assertEqual((intact["compliance_score"], intact["ready"]), (90, True))
        for sid, change, count in (
            ("a-partial", _edit_strip_rest, 7),
            ("a-upgrade", _upgrade_prefix, 7),
            ("a-empty", None, 0),
        ):
            with self.subTest(sid):
                report, checks, text = self.readiness(sid, change, count)
                self.assertEqual(checks, _readiness_checks(sid, missing_links=True))
                self.assertEqual((report["compliance_score"], report["ready"]), (70, False))
                self.assertIn("OK Hash chain integrity: 1 sessions checked\n", text)
                self.assertIn("WARN Hash chain presence: 1 sessions with missing links\n", text)

    def test_legacy_store_report_as_on_main(self):
        report, checks, _ = self.readiness("a-legacy", _strip_all)
        self.assertEqual(checks, _readiness_checks("a-legacy", missing_links=True))
        self.assertEqual((report["compliance_score"], report["ready"]), (70, False))

    def test_missing_links_are_not_ready_with_full_retention(self):
        # A session that started 91 days ago passes retention and continuity,
        # so missing links alone leave a score of 80.
        report, _, _ = self.readiness("b-intact", age_days=91)
        self.assertEqual((report["compliance_score"], report["ready"]), (100, True))
        intact_run = self.cli("audit-readiness", trace_dir=self.root / "b-intact")
        self.assertEqual(intact_run.returncode, 0, intact_run.stdout)
        for sid, change, count in (
            ("b-partial", _edit_strip_rest, 7),
            ("b-legacy", _strip_all, 7),
            ("b-empty", None, 0),
        ):
            with self.subTest(sid):
                report, checks, text = self.readiness(sid, change, count, age_days=91)
                self.assertEqual(checks, _readiness_checks(sid, missing_links=True))
                self.assertTrue(report["checks"]["retention_coverage"]["ok"])
                self.assertTrue(report["checks"]["timestamp_continuity"]["ok"])
                self.assertEqual((report["compliance_score"], report["ready"]), (80, False))
                self.assertIn("WARN Hash chain presence: 1 sessions with missing links\n", text)
                trace_dir = self.root / sid
                text_run = self.cli("audit-readiness", trace_dir=trace_dir)
                json_run = self.cli("audit-readiness", "--format", "json", trace_dir=trace_dir)
                for run in (text_run, json_run):
                    self.assertEqual(run.returncode, 1, run.stdout)
                    self.assertEqual(run.stderr, "")
                self.assertIs(json.loads(json_run.stdout)["ready"], False)

    def test_broken_session_fails_integrity(self):
        report, checks, _ = self.readiness("a-edit-hole", _edit_hole)
        self.assertEqual(checks, _readiness_checks("a-edit-hole", broken=True))
        self.assertEqual((report["compliance_score"], report["ready"]), (55, False))

    def test_unreadable_stream_fails_integrity_without_a_traceback(self):
        for sid, change, after in (
            ("a-garbage", _add_garbage_line, None),
            ("a-missing", None, lambda path: path.unlink()),
        ):
            with self.subTest(sid):
                report, checks, _ = self.readiness(sid, change, after=after)
                self.assertEqual(checks, _readiness_checks(sid, broken=True))
                self.assertEqual((report["compliance_score"], report["ready"]), (55, False))
                for args in (("audit-readiness",), ("audit-readiness", "--format", "json")):
                    run = self.cli(*args, trace_dir=self.root / sid)
                    self.assertEqual(run.returncode, 1, run.stdout)
                    self.assertEqual(run.stderr, "")


if __name__ == "__main__":
    unittest.main()
