"""Tests for compliance export (issue #144)."""

import json
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace.compliance import export_compliance, export_compliance_bulk
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


if __name__ == "__main__":
    unittest.main()
