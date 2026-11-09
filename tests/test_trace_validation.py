from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import agent_trace
from agent_trace.assignment import MAX_JSON_BYTES
from agent_trace.cli import build_parser
from agent_trace.freeze import RegressionFixture, extract_steps
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.validation import (
    CORPUS_SCHEMA,
    TRACE_SCHEMA_VERSION,
    _json_type_matches,
    _schema_errors,
    cmd_validate,
    load_trace_schema,
    validate_packaged_corpus,
    validate_trace_directory,
)


CORPUS_ROOT = Path(agent_trace.__file__).parent / "corpus" / "v1"


class TestPublicTraceSchemas(unittest.TestCase):
    def test_schemas_use_json_schema_2020_12(self):
        for kind in ("meta", "event"):
            with self.subTest(kind=kind):
                schema = load_trace_schema(kind)
                self.assertEqual(
                    schema["$schema"],
                    "https://json-schema.org/draft/2020-12/schema",
                )
                self.assertIn(f"/trace/v1/{kind}.schema.json", schema["$id"])

    def test_meta_schema_tracks_session_meta_fields(self):
        schema = load_trace_schema("meta")
        self.assertEqual(
            set(schema["properties"]),
            {item.name for item in fields(SessionMeta)},
        )
        self.assertEqual(set(schema["required"]), {"session_id", "started_at"})
        self.assertFalse(schema["additionalProperties"])

    def test_event_schema_tracks_event_model_and_enum(self):
        schema = load_trace_schema("event")
        self.assertEqual(
            set(schema["properties"]),
            {item.name for item in fields(TraceEvent)},
        )
        self.assertEqual(
            set(schema["properties"]["event_type"]["enum"]),
            {item.value for item in EventType},
        )
        self.assertEqual(
            set(schema["required"]),
            {"event_type", "timestamp", "event_id", "data"},
        )
        self.assertFalse(schema["additionalProperties"])

    def test_integer_type_accepts_integral_json_numbers(self):
        self.assertTrue(_json_type_matches(1, "integer"))
        self.assertTrue(_json_type_matches(1.0, "integer"))
        self.assertTrue(_json_type_matches(-0.0, "integer"))
        self.assertFalse(_json_type_matches(1.5, "integer"))
        self.assertFalse(_json_type_matches(True, "integer"))

    def test_pattern_uses_json_schema_search_semantics(self):
        schema = {"type": "string", "pattern": "[0-9a-f]{4}"}
        self.assertEqual(_schema_errors("prefix-dead-suffix", schema), [])
        self.assertTrue(_schema_errors("prefix-nope-suffix", schema))

    def test_exact_hash_pattern_is_anchored_in_public_schema(self):
        schema = load_trace_schema("event")["properties"]["prev_hash"]
        self.assertEqual(_schema_errors("a" * 64, schema), [])
        self.assertTrue(_schema_errors("prefix" + "a" * 64, schema))
        self.assertTrue(_schema_errors("a" * 64 + "suffix", schema))

    def test_schema_subset_matches_independent_validator_when_available(self):
        try:
            from jsonschema import Draft202012Validator
        except ImportError:
            self.skipTest("jsonschema is not installed")
        cases = (
            (1.0, {"type": "integer"}),
            (1.5, {"type": "integer"}),
            ("prefix-dead-suffix", {"type": "string", "pattern": "[0-9a-f]{4}"}),
            ("prefix", {"type": "string", "pattern": "^[0-9a-f]{4}$"}),
        )
        for value, schema in cases:
            with self.subTest(value=value, schema=schema):
                ours = bool(_schema_errors(value, schema))
                independent = bool(
                    list(Draft202012Validator(schema).iter_errors(value))
                )
                self.assertEqual(ours, independent)


class TestAgentEvidenceCorpus(unittest.TestCase):
    def test_manifest_is_synthetic_mit_data(self):
        manifest = json.loads(
            (CORPUS_ROOT / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["schema"], CORPUS_SCHEMA)
        self.assertEqual(manifest["version"], TRACE_SCHEMA_VERSION)
        self.assertEqual(manifest["license"], "MIT")
        self.assertTrue(manifest["synthetic"])
        self.assertFalse(manifest["contains_private_traces"])
        self.assertFalse(manifest["contains_real_secrets"])

    def test_all_expected_labels_match_validator_output(self):
        result = validate_packaged_corpus()
        self.assertTrue(result["matched"])
        self.assertEqual(result["case_count"], 12)
        self.assertTrue(all(case["matched"] for case in result["cases"]))

    def test_corpus_covers_required_evidence_states(self):
        result = validate_packaged_corpus()
        labels = {
            label
            for case in result["cases"]
            for label in case["actual"]["labels"]
        }
        for required in (
            "complete",
            "incomplete",
            "recovered",
            "tampered",
            "redacted",
            "summary_contradiction",
            "usage_missing",
            "usage_zero",
            "duplicate_ids",
            "out_of_order",
            "replay_drift",
            "missing_outcome",
            "terminal_absent",
            "schema_invalid",
        ):
            self.assertIn(required, labels)

    def test_validation_does_not_rewrite_trace_files(self):
        case = CORPUS_ROOT / "complete-current"
        before = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (case / "meta.json", case / "events.ndjson")
        }
        report = validate_trace_directory(case)
        after = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (case / "meta.json", case / "events.ndjson")
        }
        self.assertEqual(report.status, "valid")
        self.assertEqual(before, after)

    def test_legacy_and_current_hash_states_are_valid(self):
        current = validate_trace_directory(CORPUS_ROOT / "complete-current")
        legacy = validate_trace_directory(CORPUS_ROOT / "usage-missing")
        self.assertEqual(current.status, "valid")
        self.assertIn("current", current.labels)
        self.assertEqual(current.hash_chain["state"], "unanchored")
        self.assertEqual(legacy.status, "valid")
        self.assertIn("legacy", legacy.labels)
        self.assertEqual(legacy.hash_chain["state"], "legacy")

    def test_replay_equivalence_uses_existing_regression_projection(self):
        case = CORPUS_ROOT / "replay-drift"
        events = [
            TraceEvent.from_json(line)
            for line in (case / "events.ndjson").read_text(encoding="utf-8").splitlines()
        ]
        fixture = RegressionFixture(
            session="equivalent",
            task="synthetic",
            steps=extract_steps(events),
        )
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            baseline = Path(tmp) / "baseline.json"
            baseline.write_text(fixture.to_json(), encoding="utf-8")
            report = validate_trace_directory(case, replay_baseline=baseline)
        self.assertEqual(report.status, "valid")
        self.assertIn("replay_equivalent", report.labels)
        self.assertNotIn("replay_drift", report.labels)

    def test_malformed_replay_steps_return_invalid_finding(self):
        case = CORPUS_ROOT / "usage-zero"
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            baseline = Path(tmp) / "baseline.json"
            baseline.write_text('{"steps":"not-an-array"}', encoding="utf-8")
            report = validate_trace_directory(case, replay_baseline=baseline)
        self.assertEqual(report.status, "invalid")
        self.assertIn("replay_baseline_invalid", report.labels)
        self.assertEqual(
            [finding.code for finding in report.findings],
            ["replay_baseline_invalid"],
        )

    def test_replay_baseline_rejects_symlinked_ancestor(self):
        case = CORPUS_ROOT / "usage-zero"
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            real = root / "real"
            real.mkdir()
            baseline = real / "baseline.json"
            baseline.write_text('{"steps":[]}', encoding="utf-8")
            link = root / "link"
            link.symlink_to(real, target_is_directory=True)
            report = validate_trace_directory(
                case,
                replay_baseline=link / "baseline.json",
            )
        self.assertEqual(report.status, "invalid")
        self.assertIn("replay_baseline_invalid", report.labels)

    def test_replay_baseline_is_bounded_before_parsing(self):
        case = CORPUS_ROOT / "usage-zero"
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            baseline = Path(tmp) / "baseline.json"
            baseline.write_bytes(b" " * (MAX_JSON_BYTES + 1))
            report = validate_trace_directory(case, replay_baseline=baseline)
        self.assertEqual(report.status, "invalid")
        self.assertIn("replay_baseline_invalid", report.labels)


class TestValidationCli(unittest.TestCase):
    def test_parser_registers_validate_command(self):
        args = build_parser().parse_args(
            [
                "validate",
                "trace-dir",
                "--format",
                "json",
                "--allow-active",
                "--replay-baseline",
                "baseline.json",
            ]
        )
        self.assertEqual(args.command, "validate")
        self.assertEqual(args.target, "trace-dir")
        self.assertEqual(args.format, "json")
        self.assertTrue(args.allow_active)
        self.assertEqual(args.replay_baseline, "baseline.json")

    def test_corpus_cli_returns_success(self):
        args = SimpleNamespace(
            corpus=True,
            format="json",
            trace_dir=".agent-traces",
            target=None,
            allow_active=False,
            replay_baseline=None,
        )
        stdout = io.StringIO()
        with patch.dict(os.environ, {"AGENT_STRACE_WORKSPACE": ""}, clear=False):
            with patch("sys.stdout", stdout):
                result = cmd_validate(args)
        self.assertEqual(result, 0)
        self.assertTrue(json.loads(stdout.getvalue())["matched"])

    def test_session_id_resolution_does_not_write_to_store(self):
        source = CORPUS_ROOT / "usage-zero"
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            case = root / "usage-zero"
            case.mkdir()
            for name in ("meta.json", "events.ndjson"):
                (case / name).write_bytes((source / name).read_bytes())
            args = SimpleNamespace(
                corpus=False,
                format="json",
                trace_dir=str(root),
                target="usage-zero",
                allow_active=False,
                replay_baseline=None,
            )
            stdout = io.StringIO()
            with patch("sys.stdout", stdout):
                result = cmd_validate(args)
            self.assertEqual(result, 0)
            self.assertFalse((root / ".gitignore").exists())
            self.assertEqual(json.loads(stdout.getvalue())["status"], "valid")


if __name__ == "__main__":
    unittest.main()
