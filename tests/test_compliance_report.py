"""Tests for the privacy-minimized compliance evidence crosswalk."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent_trace.cli import build_parser
from agent_trace.compliance_report import (
    ComplianceReportError,
    _atomic_write_private,
    _pdf_safe_text,
    build_compliance_report,
    cmd_compliance_report,
    compliance_report_to_sarif,
    format_compliance_pdf,
    load_framework_manifest,
    resolve_report_window,
)
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore


NOW = datetime(2026, 8, 16, tzinfo=timezone.utc)
SINCE = "1970-01-01T00:00:05Z"
UNTIL = "1970-01-01T00:01:00Z"


def _event(
    event_type: EventType,
    timestamp: float,
    event_id: str,
    *,
    parent_id: str = "",
    data: dict | None = None,
) -> TraceEvent:
    return TraceEvent(
        event_type=event_type,
        timestamp=timestamp,
        event_id=event_id,
        session_id="session",
        parent_id=parent_id,
        data=data or {},
    )


def _store(directory: str, *, started_at: float = 1, redact: bool = False) -> TraceStore:
    store = TraceStore(directory, use_workspace_env=False, redact=redact)
    store.create_session(SessionMeta(
        session_id="session",
        started_at=started_at,
        attribution={"actor_id": "private-person"},
    ))
    return store


def _append_call(
    store: TraceStore,
    timestamp: float,
    event_id: str,
    *,
    tool_name: str = "bash",
    arguments: dict | None = None,
    extra: dict | None = None,
) -> None:
    data = {"tool_name": tool_name, "arguments": arguments or {"command": "true"}}
    data.update(extra or {})
    store.append_event("session", _event(
        EventType.TOOL_CALL, timestamp, event_id, data=data,
    ))


def _report(store: TraceStore, framework: str = "owasp-agentic", **kwargs):
    return build_compliance_report(
        store, framework, since=SINCE, until=UNTIL,
        generated_at=NOW, **kwargs,
    )


class TestFrameworkManifests(unittest.TestCase):
    def test_bundled_manifests_have_current_identity_and_exact_owasp_titles(self):
        aicpa, _ = load_framework_manifest("aicpa-tsc")
        owasp, _ = load_framework_manifest("owasp-agentic")
        eu, _ = load_framework_manifest("eu-ai-act")
        self.assertEqual(aicpa["framework"]["id"], "aicpa-tsc")
        self.assertNotIn("cli_aliases", aicpa["framework"])
        self.assertEqual(
            [(item["id"], item["title"]) for item in owasp["controls"]],
            [
                ("ASI01", "Agent Goal Hijack"),
                ("ASI02", "Tool Misuse and Exploitation"),
                ("ASI03", "Identity and Privilege Abuse"),
                ("ASI04", "Agentic Supply Chain Vulnerabilities"),
                ("ASI05", "Unexpected Code Execution (RCE)"),
                ("ASI06", "Memory & Context Poisoning"),
                ("ASI07", "Insecure Inter-Agent Communication"),
                ("ASI08", "Cascading Failures"),
                ("ASI09", "Human-Agent Trust Exploitation"),
                ("ASI10", "Rogue Agents"),
            ],
        )
        self.assertEqual(owasp["framework"]["license"], "CC-BY-SA-4.0")
        self.assertEqual(eu["framework"]["source_as_of"], "2026-07-27")
        self.assertEqual(eu["framework"]["applicability_policy"], "not_assessed")

    def test_index_digests_match_all_packaged_manifests(self):
        root = Path(__file__).parents[1] / "src" / "agent_trace" / "frameworks"
        index = json.loads((root / "index.json").read_text())
        for entry in index["manifests"]:
            digest = hashlib.sha256((root / entry["file"]).read_bytes()).hexdigest()
            self.assertEqual(digest, entry["sha256"])
        self.assertTrue((root / "THIRD_PARTY_NOTICES.md").is_file())

    def test_local_manifest_rejects_bidi_and_symlink(self):
        manifest, _ = load_framework_manifest("owasp-agentic")
        manifest["controls"][0]["title"] = "safe\u202edanger"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ComplianceReportError, "Unicode"):
                load_framework_manifest("owasp-agentic", str(path))
            target = Path(directory) / "target.json"
            target.write_text("{}")
            link = Path(directory) / "link.json"
            link.symlink_to(target)
            with self.assertRaises(ComplianceReportError):
                load_framework_manifest("owasp-agentic", str(link))

    def test_local_manifest_rejects_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text('{"schema":"x","schema":"y"}')
            with self.assertRaisesRegex(ComplianceReportError, "duplicate"):
                load_framework_manifest("owasp-agentic", str(path))

    def test_index_validates_unselected_rows(self):
        valid_manifest = (
            Path(__file__).parents[1]
            / "src" / "agent_trace" / "frameworks" / "owasp-agentic.json"
        ).read_bytes()
        bad_index = {
            "schema": "agent-strace-compliance-manifest-index/v1",
            "index_version": "2026-08-16",
            "manifests": [
                {
                    "framework": "owasp-agentic",
                    "file": "owasp-agentic.json",
                    "mapping_version": "1.0.0",
                    "sha256": hashlib.sha256(valid_manifest).hexdigest(),
                },
                {
                    "framework": "bad",
                    "file": "../bad.json",
                    "mapping_version": "1.0.0",
                    "sha256": "0" * 64,
                },
            ],
        }

        def package(name):
            return json.dumps(bad_index).encode() if name == "index.json" else valid_manifest

        with patch("agent_trace.compliance_report._package_bytes", side_effect=package):
            with self.assertRaisesRegex(ComplianceReportError, "unsafe"):
                load_framework_manifest("owasp-agentic")


class TestWindowSnapshotAndCorrelation(unittest.TestCase):
    def test_window_requires_ordered_utc_and_bounds_duration(self):
        with self.assertRaisesRegex(ComplianceReportError, "UTC offset"):
            resolve_report_window("2026-01-01T00:00:00", None, now=NOW)
        with self.assertRaisesRegex(ComplianceReportError, "earlier"):
            resolve_report_window("2026-01-02", "2026-01-01", now=NOW)
        with self.assertRaisesRegex(ComplianceReportError, "range"):
            resolve_report_window("9" * 100 + "d", None, now=NOW)

    def test_event_time_selects_old_session_and_excludes_out_of_window_call(self):
        with tempfile.TemporaryDirectory() as directory:
            store = _store(directory, started_at=0)
            _append_call(store, 10, "inside")
            _append_call(store, 100, "outside")
            report = _report(store)
            self.assertEqual(report["coverage"]["sessions"]["selected"], 1)
            self.assertEqual(report["coverage"]["tool_results"]["tool_calls"], 1)

    def test_cross_window_result_is_boundary_context_not_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            store = _store(directory)
            _append_call(store, 10, "call")
            store.append_event("session", _event(
                EventType.TOOL_RESULT, 70, "result", parent_id="call",
            ))
            report = _report(store)
            coverage = report["coverage"]["tool_results"]
            self.assertEqual(coverage["correlated_outcomes"], 0)
            self.assertEqual(coverage["boundary_context_outcomes"], 1)
            self.assertEqual(
                report["authorization_timeline"][0]["result_status"],
                "boundary_context_result",
            )

    def test_multiple_results_and_duplicate_ids_are_ambiguous(self):
        with tempfile.TemporaryDirectory() as directory:
            store = _store(directory)
            _append_call(store, 10, "same")
            store.append_event("session", _event(
                EventType.TOOL_RESULT, 11, "r1", parent_id="same",
            ))
            store.append_event("session", _event(
                EventType.TOOL_RESULT, 12, "r2", parent_id="same",
            ))
            _append_call(store, 70, "same")
            report = _report(store)
            quality = report["coverage"]["tool_results"]["correlation_quality"]
            self.assertGreater(quality["ambiguous_results"], 0)
            self.assertGreater(quality["duplicate_call_ids"], 0)

    def test_result_timestamp_before_call_is_not_correlated_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            store = _store(directory)
            _append_call(store, 20, "call")
            store.append_event("session", _event(
                EventType.TOOL_RESULT, 10, "result", parent_id="call",
            ))
            report = _report(store)
            coverage = report["coverage"]["tool_results"]
            self.assertEqual(coverage["correlated_outcomes"], 0)
            self.assertEqual(coverage["unobserved_outcomes"], 1)
            self.assertEqual(
                coverage["correlation_quality"]["before_call_results"], 1
            )

    def test_malformed_or_blank_stream_fails_complete_report(self):
        for suffix in (b"{bad}\n", b"\n"):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as directory:
                store = _store(directory)
                _append_call(store, 10, "call")
                path = Path(directory) / "session" / "events.ndjson"
                path.write_bytes(path.read_bytes() + suffix)
                with self.assertRaises(ComplianceReportError):
                    _report(store)

    def test_integrity_empty_and_nonempty_first_hash_are_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            empty = _store(directory, started_at=10)
            report = _report(empty)
            self.assertEqual(report["coverage"]["integrity"]["empty"], 1)
        with tempfile.TemporaryDirectory() as directory:
            store = _store(directory)
            path = Path(directory) / "session" / "events.ndjson"
            payload = json.loads(_event(EventType.USER_PROMPT, 10, "one").to_json())
            payload["prev_hash"] = "f" * 64
            path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
            report = _report(store)
            self.assertEqual(report["coverage"]["integrity"]["broken"], 1)

    def test_symlinked_event_stream_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            store = _store(directory)
            event_path = Path(directory) / "session" / "events.ndjson"
            target = Path(directory) / "outside"
            target.write_text("")
            event_path.unlink()
            event_path.symlink_to(target)
            with self.assertRaises(ComplianceReportError):
                _report(store)


class TestAuthorizationPrivacyAndDetectors(unittest.TestCase):
    def test_namespaced_authorization_digest_survives_store_redaction(self):
        with tempfile.TemporaryDirectory() as directory:
            store = _store(directory, redact=True)
            authorization = {
                "schema": "agent-strace-authorization/v1",
                "decision": "allowed",
                "mode": "pre_action",
                "evaluated_at": 9.0,
                "policy_sha256": "sha256:" + "a" * 64,
                "rule_id": "allow-shell",
            }
            _append_call(store, 10, "call", extra={
                "agent_trace_authorization": authorization,
            })
            loaded = store.load_events("session")[0]
            self.assertEqual(
                loaded.data["agent_trace_authorization"]["policy_sha256"],
                authorization["policy_sha256"],
            )
            report = _report(store)
            evidence = report["authorization_timeline"][0]["authorization"]
            self.assertEqual(evidence["basis"], "recorded_event")
            self.assertTrue(evidence["pre_action_authorization_evidence"])
            self.assertEqual(evidence["policy_sha256"], authorization["policy_sha256"])
            self.assertNotIn("allow-shell", json.dumps(report))

    def test_retrospective_policy_has_safe_provenance_and_would_outcome(self):
        with tempfile.TemporaryDirectory() as directory:
            store = _store(directory)
            _append_call(store, 10, "call", arguments={"command": "danger"})
            policy = Path(directory) / "scope.json"
            policy.write_text(json.dumps({
                "schema": "agent-scope/v1",
                "commands": {"deny": ["danger"]},
            }))
            report = _report(store, policy_path=str(policy))
            context = report["authorization_timeline"][0]["authorization"]
            self.assertEqual(context["decision"], "would_deny")
            provenance = report["retrospective_policy"]["provenance"]
            self.assertEqual(provenance["schema"], "agent-scope/v1")
            self.assertNotIn(str(policy), json.dumps(report))
            self.assertNotIn("danger", json.dumps(provenance))

    def test_policy_complexity_is_bounded_for_rules_and_observed_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            store = _store(directory)
            _append_call(
                store, 10, "call", tool_name="read",
                arguments={"path": "/".join(["x"] * 2000 + [".env"])},
            )
            policy = Path(directory) / "scope.json"
            policy.write_text(json.dumps({"files": {"read": {"allow": ["**/z"]}}}))
            report = _report(store, policy_path=str(policy))
            auth = report["authorization_timeline"][0]["authorization"]
            self.assertEqual(auth["decision"], "not_covered")
            policy.write_text(json.dumps({
                "files": {"read": {"allow": ["**/**/**/z"]}},
            }))
            with self.assertRaisesRegex(ComplianceReportError, "recursive wildcard"):
                _report(store, policy_path=str(policy))

    def test_retrospective_policy_normalizes_untrusted_tool_shapes(self):
        for tool_name, arguments in ((7, {}), ("bash", [])):
            with self.subTest(tool_name=tool_name), tempfile.TemporaryDirectory() as directory:
                store = _store(directory)
                store.append_event("session", _event(
                    EventType.TOOL_CALL, 10, "call",
                    data={"tool_name": tool_name, "arguments": arguments},
                ))
                policy = Path(directory) / "scope.json"
                policy.write_text(json.dumps({"commands": {"deny": ["danger"]}}))
                report = _report(store, policy_path=str(policy))
                self.assertEqual(report["coverage"]["sessions"]["selected"], 1)
                self.assertEqual(
                    report["coverage"]["authorization"]["retrospective_policy"], 1
                )

    def test_approval_missing_required_fields_is_not_synthesized(self):
        with tempfile.TemporaryDirectory() as directory:
            store = _store(directory)
            _append_call(store, 10, "call")
            approvals = Path(directory) / ".approvals"
            approvals.mkdir()
            (approvals / "bad.json").write_text(json.dumps({
                "session_id": "session", "event_id": "call",
            }))
            report = _report(store)
            auth = report["coverage"]["authorization"]
            self.assertEqual(auth["linked_sensitive_approval_calls"], 0)
            self.assertEqual(auth["malformed_approval_records"], 1)

    def test_duplicate_approval_and_cross_session_duplicate_call_are_ambiguous(self):
        with tempfile.TemporaryDirectory() as directory:
            store = _store(directory)
            _append_call(store, 10, "same")
            _append_call(store, 70, "same")
            approvals = Path(directory) / ".approvals"
            approvals.mkdir()
            payload = {
                "request_id": "r1", "session_id": "session", "event_id": "same",
                "state": "approved", "created_at": 11.0, "decided_at": 12.0,
            }
            (approvals / "one.json").write_text(json.dumps(payload))
            report = _report(store)
            self.assertEqual(
                report["authorization_timeline"][0]["authorization"]["basis"],
                "ambiguous_record",
            )
            self.assertEqual(
                report["coverage"]["authorization"]["linked_sensitive_approval_calls"], 0
            )

    def test_denial_effective_time_not_sidecar_presence_controls_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            store = _store(directory)
            _append_call(store, 10, "one")
            _append_call(store, 20, "two")
            _append_call(store, 40, "three")
            approvals = Path(directory) / ".approvals"
            approvals.mkdir()
            (approvals / "deny.json").write_text(json.dumps({
                "request_id": "deny", "session_id": "session", "event_id": "one",
                "state": "denied", "created_at": 11.0, "decided_at": 30.0,
            }))
            report = _report(store)
            asi10 = next(item for item in report["controls"] if item["control_id"] == "ASI10")
            self.assertEqual(asi10["gap_count"], 1)

    def test_after_denial_uses_timestamps_not_append_order(self):
        with tempfile.TemporaryDirectory() as directory:
            store = _store(directory)
            _append_call(store, 40, "later-operation")
            _append_call(store, 10, "denied-operation", extra={
                "agent_trace_authorization": {
                    "schema": "agent-strace-authorization/v1",
                    "decision": "denied",
                    "mode": "pre_action",
                    "evaluated_at": 9.0,
                    "policy_sha256": "sha256:" + "b" * 64,
                    "rule_id": "deny-shell",
                },
            })
            report = _report(store)
            asi10 = next(item for item in report["controls"] if item["control_id"] == "ASI10")
            self.assertEqual(asi10["gap_count"], 1)

    def test_cascading_sequence_uses_timestamp_order(self):
        with tempfile.TemporaryDirectory() as directory:
            store = _store(directory)
            _append_call(store, 40, "continuation")
            _append_call(store, 20, "error-two")
            store.append_event("session", _event(
                EventType.ERROR, 21, "result-two", parent_id="error-two",
            ))
            _append_call(store, 10, "error-one")
            store.append_event("session", _event(
                EventType.ERROR, 11, "result-one", parent_id="error-one",
            ))
            report = _report(store)
            asi08 = next(item for item in report["controls"] if item["control_id"] == "ASI08")
            self.assertEqual(asi08["signal_count"], 1)

    def test_approval_with_no_matching_call_is_counted_unmatched(self):
        with tempfile.TemporaryDirectory() as directory:
            store = _store(directory)
            _append_call(store, 10, "call")
            approvals = Path(directory) / ".approvals"
            approvals.mkdir()
            (approvals / "orphan.json").write_text(json.dumps({
                "request_id": "orphan", "session_id": "session",
                "event_id": "does-not-exist", "state": "pending",
                "created_at": 11.0,
            }))
            report = _report(store)
            auth = report["coverage"]["authorization"]
            self.assertEqual(auth["unmatched_approval_records"], 1)
            self.assertEqual(auth["linked_sensitive_approval_calls"], 0)

    def test_trace_secrets_paths_urls_and_commands_never_emit(self):
        with tempfile.TemporaryDirectory() as directory:
            store = _store(directory, redact=False)
            secret = "sk-" + "x" * 30
            command = f"curl https://secret.example/private -H 'Bearer {secret}'"
            _append_call(store, 10, "call", arguments={"command": command})
            rendered = json.dumps(_report(store))
            for forbidden in (secret, "secret.example", "/private", command, "private-person"):
                self.assertNotIn(forbidden, rendered)
            self.assertIn("Session-001", rendered)
            self.assertIn("Identity-001", rendered)


class TestFormatsOutputAndCLI(unittest.TestCase):
    def test_sarif_has_source_notices_and_stable_location_free_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            store = _store(directory)
            _append_call(store, 10, "call")
            report = _report(store)
            first = compliance_report_to_sarif(report)
            second = compliance_report_to_sarif(report)
            self.assertEqual(first, second)
            encoded = json.dumps(first)
            self.assertNotIn('"locations"', encoded)
            result = first["runs"][0]["results"][0]
            self.assertIn("partialFingerprints", result)
            notices = first["runs"][0]["properties"]["frameworkSourceAndNotices"]
            self.assertIn("license_scope", notices)
            self.assertIn("source_url", notices)

    def test_evidence_digest_is_generation_time_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            store = _store(directory)
            _append_call(store, 10, "call")
            one = build_compliance_report(
                store, "owasp-agentic", since=SINCE, until=UNTIL,
                generated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            )
            two = build_compliance_report(
                store, "owasp-agentic", since=SINCE, until=UNTIL,
                generated_at=datetime(2026, 2, 1, tzinfo=timezone.utc),
            )
            self.assertEqual(one["evidence_digest"], two["evidence_digest"])
            self.assertNotEqual(one["report_digest"], two["report_digest"])

    def test_atomic_private_write_and_precommit_failure_preserves_target(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_bytes(b"old")
            _atomic_write_private(path, b"new")
            self.assertEqual(path.read_bytes(), b"new")
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            path.write_bytes(b"old-again")
            with patch("agent_trace.compliance_report.os.chmod", side_effect=OSError("boom")):
                with self.assertRaises(OSError):
                    _atomic_write_private(path, b"replacement")
            self.assertEqual(path.read_bytes(), b"old-again")

    def test_atomic_write_refuses_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "target"
            target.write_bytes(b"safe")
            link = Path(directory) / "report"
            link.symlink_to(target)
            with self.assertRaises(ComplianceReportError):
                _atomic_write_private(link, b"unsafe")
            self.assertEqual(target.read_bytes(), b"safe")

    def test_atomic_write_refuses_existing_child_below_symlink_ancestor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real"
            child = real / "child"
            child.mkdir(parents=True)
            link = root / "link"
            link.symlink_to(real, target_is_directory=True)
            with self.assertRaises(ComplianceReportError):
                _atomic_write_private(link / "child" / "report.json", b"unsafe")
            self.assertFalse((child / "report.json").exists())

    def test_pdf_dependency_error_is_actionable(self):
        real_import = __import__

        def blocked(name, *args, **kwargs):
            if name.startswith("reportlab"):
                raise ImportError("missing")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=blocked):
            with self.assertRaisesRegex(ComplianceReportError, r"agent-strace\[pdf\]"):
                format_compliance_pdf({})

    def test_pdf_safe_text_preserves_punctuation_and_escapes_japanese(self):
        self.assertEqual(_pdf_safe_text("AICPA – evidence"), "AICPA – evidence")
        self.assertEqual(_pdf_safe_text("証拠"), "\\u8A3C\\u62E0")

    def test_pdf_renders_with_supported_reportlab(self):
        try:
            import reportlab  # noqa: F401
        except ImportError:
            self.skipTest("optional ReportLab dependency is not installed")
        with tempfile.TemporaryDirectory() as directory:
            store = _store(directory, started_at=10)
            report = _report(store)
            rendered = format_compliance_pdf(report)
            self.assertTrue(rendered.startswith(b"%PDF-"))
            self.assertGreater(len(rendered), 1000)

    def test_cli_parser_has_bare_rescan_and_no_licensing_unsafe_alias(self):
        parser = build_parser()
        args = parser.parse_args([
            "compliance-report", "--framework", "aicpa-tsc", "--rescan",
        ])
        self.assertEqual(args.rescan, "installed")
        with self.assertRaises(SystemExit):
            parser.parse_args([
                "compliance-report", "--framework", "soc2",
            ])

    def test_cli_writes_authoritative_json(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            args = SimpleNamespace(
                trace_dir=str(Path(directory) / "traces"),
                framework="aicpa-tsc", since="1d", until=None,
                format="json", output=str(output), policy=None, rescan=None,
            )
            self.assertEqual(cmd_compliance_report(args), 0)
            self.assertEqual(json.loads(output.read_text())["schema"],
                             "agent-strace-compliance-report/v1")


if __name__ == "__main__":
    unittest.main()
