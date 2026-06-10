"""Tests for compliance export (issue #144)."""

import json
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace.cli import build_parser
from agent_trace.compliance import (
    build_audit_readiness,
    export_compliance,
    export_compliance_bulk,
    export_eu_ai_act,
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


if __name__ == "__main__":
    unittest.main()
