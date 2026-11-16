from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import agent_trace
from agent_trace.assignment import MAX_JSON_BYTES
from agent_trace.cli import build_parser
from agent_trace.freeze import RegressionFixture, extract_steps
from agent_trace.health import load_capture_matrix
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore
from agent_trace.validation import (
    CORPUS_SCHEMA,
    TRACE_SCHEMA_VERSION,
    _json_type_matches,
    _read_trace_files,
    _schema_errors,
    cmd_validate,
    load_trace_schema,
    validate_packaged_corpus,
    validate_trace_directory,
)


CORPUS_ROOT = Path(agent_trace.__file__).parent / "corpus" / "v1"
SRC_ROOT = Path(agent_trace.__file__).resolve().parent.parent

# Windows cannot open directories or use dir_fd, so it only reads by path.
READ_MODES = (
    ("dir_fd", "by_path") if os.open in os.supports_dir_fd else ("by_path",)
)


def read_mode(mode):
    return patch("agent_trace.validation._READ_BY_DIR_FD", mode == "dir_fd")


def _comparable(report):
    """Return a report dict without the paths, which differ per copy."""
    data = report.to_dict()
    data.pop("target")
    if data.get("replay"):
        data["replay"].pop("baseline")
    return data


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
            try:
                link.symlink_to(real, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("symlinks unavailable")
            for mode in READ_MODES:
                with self.subTest(mode=mode), read_mode(mode):
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


class TestTraceFileReads(unittest.TestCase):
    """validate reads the same bytes and reports the same results on every OS.

    POSIX reads through open directory fds. Windows reads by path. Both run
    here where the OS allows it.
    """

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        # Resolve: neither reader follows symlinks such as macOS's /var.
        self.root = Path(tmp.name).resolve()

    def trace_dir(self, name="session", meta=b"{}", events=b""):
        path = self.root / name
        path.mkdir()
        (path / "meta.json").write_bytes(meta)
        (path / "events.ndjson").write_bytes(events)
        return path

    def read_errors(self, path):
        errors = {}
        for mode in READ_MODES:
            with read_mode(mode):
                report = validate_trace_directory(path)
            self.assertEqual(
                [finding.code for finding in report.findings], ["trace_read_failed"]
            )
            errors[mode] = report.findings[0].message
        return errors

    def assert_read_error(self, path, message):
        self.assertEqual(self.read_errors(path), {mode: message for mode in READ_MODES})

    def test_files_are_read_byte_for_byte(self):
        # A text-mode read on Windows would turn CRLF into LF and stop at ^Z.
        meta = b'{"a":1}\r\n\x1a tail\r\n'
        events = b"line one\r\nline two\n\x1a\r\n"
        path = self.trace_dir(meta=meta, events=events)
        for mode in READ_MODES:
            with self.subTest(mode=mode), read_mode(mode):
                self.assertEqual(_read_trace_files(path), (meta, events))

    def test_corpus_results_match_for_lf_and_crlf_copies(self):
        manifest = json.loads((CORPUS_ROOT / "manifest.json").read_bytes())
        for case in manifest["cases"]:
            source = CORPUS_ROOT / case["path"]
            results = {}
            for newline in (b"\n", b"\r\n"):
                copy = self.root / f"crlf{len(newline) - 1}" / case["path"]
                copy.mkdir(parents=True)
                for item in source.iterdir():
                    lf = item.read_bytes().replace(b"\r\n", b"\n")
                    (copy / item.name).write_bytes(lf.replace(b"\n", newline))
                baseline = case.get("replay_baseline")
                for mode in READ_MODES:
                    with read_mode(mode):
                        report = validate_trace_directory(
                            copy,
                            replay_baseline=copy / baseline if baseline else None,
                        )
                    results[(newline, mode)] = _comparable(report)
            with self.subTest(case=case["id"]):
                first = next(iter(results.values()))
                self.assertEqual(first["status"], case["expected_status"])
                self.assertEqual(first["labels"], sorted(case["expected_labels"]))
                for key, result in results.items():
                    self.assertEqual(result, first, key)

    def test_missing_directory(self):
        self.assert_read_error(self.root / "missing" / "session", "directory does not exist")

    def test_file_as_directory_component(self):
        (self.root / "file").write_bytes(b"")
        self.assert_read_error(self.root / "file" / "session", "directory has an unsafe component")

    def test_missing_trace_file(self):
        path = self.trace_dir()
        (path / "meta.json").unlink()
        self.assert_read_error(path, "session metadata could not be opened safely")

    def test_trace_file_that_is_a_directory(self):
        path = self.trace_dir()
        (path / "events.ndjson").unlink()
        (path / "events.ndjson").mkdir()
        self.assert_read_error(path, "session event stream size or type is invalid")

    def test_oversized_trace_file(self):
        path = self.trace_dir(events=b"x" * 11)
        with patch("agent_trace.validation.MAX_SOURCE_TRACE_BYTES", 10):
            self.assert_read_error(path, "session event stream size or type is invalid")

    def _symlink(self, link, target, *, directory):
        try:
            link.symlink_to(target, target_is_directory=directory)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable")

    def test_symlinked_trace_directory(self):
        real = self.trace_dir()
        self._symlink(self.root / "link", real, directory=True)
        self.assert_read_error(self.root / "link", "directory has an unsafe component")

    def test_symlinked_trace_file(self):
        path = self.trace_dir()
        (self.root / "real.json").write_bytes(b"{}")
        (path / "meta.json").unlink()
        self._symlink(path / "meta.json", self.root / "real.json", directory=False)
        self.assert_read_error(path, "session metadata could not be opened safely")

    def test_by_path_refuses_a_file_swapped_after_lstat(self):
        path = self.trace_dir()
        other = self.trace_dir("other")
        real_open = os.open

        def swapped_open(name, flags, *args, **kwargs):
            if Path(name) == path / "meta.json":
                name = other / "meta.json"
            return real_open(name, flags, *args, **kwargs)

        with read_mode("by_path"), patch("os.open", swapped_open):
            report = validate_trace_directory(path)
        self.assertEqual(
            [finding.message for finding in report.findings],
            ["session metadata changed while it was read"],
        )

    @unittest.skipUnless(os.name == "nt", "Windows junctions")
    def test_junction_is_refused(self):
        import _winapi

        real = self.trace_dir()
        (real / "baseline.json").write_bytes(b'{"steps":[]}')
        junction = self.root / "junction"
        _winapi.CreateJunction(str(real), str(junction))
        self.assert_read_error(junction, "directory has an unsafe component")
        report = validate_trace_directory(
            CORPUS_ROOT / "usage-zero", replay_baseline=junction / "baseline.json"
        )
        self.assertIn("replay_baseline_invalid", report.labels)
        self.assertEqual(report.replay["state"], "invalid")

    def test_cli_validates_a_recorded_session(self):
        session = "cli-0001"
        store = TraceStore(self.root, use_workspace_env=False, redact=False)
        store.create_session(SessionMeta(session_id=session, agent_name="test"))
        for event_type in (EventType.SESSION_START, EventType.SESSION_END):
            store.append_event(session, TraceEvent(event_type=event_type, session_id=session))
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("AGENT_STRACE_") and key != "AGENT_TRACE_DIR"
        }
        env["PYTHONPATH"] = os.pathsep.join(
            item for item in (str(SRC_ROOT), env.get("PYTHONPATH", "")) if item
        )
        command = [
            sys.executable, "-c", "from agent_trace.entry import main; main()",
            "--trace-dir", str(self.root),
        ]
        outputs = {}
        for args in (["validate", session, "--format", "json"], ["validate", "--corpus"]):
            result = subprocess.run(
                command + args,
                capture_output=True,
                text=True,
                encoding="utf-8",
                env=env,
                timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            outputs[args[1]] = result.stdout
        report = json.loads(outputs[session])
        self.assertEqual(report["status"], "valid")
        self.assertEqual(report["session_id"], session)
        self.assertIn("Agent Evidence Corpus: matched", outputs["--corpus"])


class TestDeclaredEventsBeforeStart(unittest.TestCase):
    """validate reads ``events_before_start`` for the hook provider, as health does."""

    SESSION = "pre-start-0001"

    def validate(self, pre_start, *, provider="copilot", agent_name="github-copilot"):
        """Write ``pre_start`` events, a session start, then an end; validate it."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        # Resolve the temp dir: validate refuses symlinked components such as
        # macOS's /var.
        root = Path(tmp.name).resolve()
        store = TraceStore(root, use_workspace_env=False, redact=False)
        store.create_session(SessionMeta(session_id=self.SESSION, agent_name=agent_name))
        start_data = (
            {"mode": f"{agent_name}-hooks", "provider": provider, "source": "new"}
            if provider
            else {}
        )
        for event_type in pre_start:
            store.append_event(
                self.SESSION,
                TraceEvent(event_type=event_type, session_id=self.SESSION),
            )
        for event_type, data in (
            (EventType.SESSION_START, start_data),
            (EventType.SESSION_END, {}),
        ):
            store.append_event(
                self.SESSION,
                TraceEvent(event_type=event_type, session_id=self.SESSION, data=data),
            )
        return validate_trace_directory(root / self.SESSION)

    @staticmethod
    def findings(report):
        return [(finding.code, finding.severity) for finding in report.findings]

    def assert_undeclared(self, report):
        self.assertEqual(report.status, "invalid")
        self.assertEqual(report.exit_code, 1)
        self.assertEqual(self.findings(report), [("events_before_session_start", "error")])
        self.assertNotIn("declared_events_before_session_start", report.labels)

    def test_declared_pre_start_type_is_an_info_finding(self):
        self.assertEqual(
            load_capture_matrix().entry("copilot")["events_before_start"], ["user_prompt"]
        )
        report = self.validate([EventType.USER_PROMPT])

        self.assertEqual(report.status, "valid")
        self.assertEqual(report.exit_code, 0)
        self.assertEqual(
            self.findings(report), [("declared_events_before_session_start", "info")]
        )
        self.assertIn("user_prompt events were recorded", report.findings[0].message)
        self.assertIn("declared_events_before_session_start", report.labels)
        self.assertIn("conformant", report.labels)
        self.assertIn("complete", report.labels)
        self.assertNotEqual(report.evidence_health["status"], "invalid")

    def test_undeclared_pre_start_type_stays_invalid(self):
        for pre_start in (
            [EventType.FILE_READ],
            [EventType.USER_PROMPT, EventType.FILE_READ],
        ):
            with self.subTest(pre_start=[t.value for t in pre_start]):
                self.assert_undeclared(self.validate(pre_start))

    def test_provider_without_the_declaration_stays_invalid(self):
        self.assertNotIn("events_before_start", load_capture_matrix().entry("claude"))
        self.assert_undeclared(
            self.validate([EventType.USER_PROMPT], provider="claude", agent_name="claude-code")
        )

    def test_any_provider_entry_can_declare_the_order(self):
        matrix = load_capture_matrix()
        declared = replace(
            matrix,
            providers=tuple(
                {**entry, "events_before_start": ["user_prompt"]}
                if entry["id"] == "claude"
                else entry
                for entry in matrix.providers
            ),
        )
        with patch("agent_trace.health.load_capture_matrix", return_value=declared):
            report = self.validate(
                [EventType.USER_PROMPT], provider="claude", agent_name="claude-code"
            )

        self.assertEqual(report.status, "valid")
        self.assertEqual(
            self.findings(report), [("declared_events_before_session_start", "info")]
        )

    def test_trace_without_hook_provider_is_unchanged(self):
        # The agent name alone, as set by record --name, the SDK or an
        # importer, does not identify a hook provider.
        for agent_name in ("sdk", "github-copilot"):
            with self.subTest(agent_name=agent_name):
                self.assert_undeclared(
                    self.validate([EventType.USER_PROMPT], provider="", agent_name=agent_name)
                )

    def test_matrix_load_failure_declares_nothing(self):
        for error in (OSError("unreadable"), ValueError("malformed")):
            with self.subTest(error=type(error).__name__):
                with patch("agent_trace.health.load_capture_matrix", side_effect=error):
                    self.assert_undeclared(self.validate([EventType.USER_PROMPT]))


class TestErrorSummary(unittest.TestCase):
    """metadata ``errors`` counts error events and failed model requests."""

    SESSION = "errors-0001"

    def validate(self, recorded_errors, *, linked_error=False):
        """Write a session with a failed tool call and a failed model request.

        With *linked_error*, an error event is also linked to the request.
        """
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name).resolve()
        store = TraceStore(root, use_workspace_env=False, redact=False)
        store.create_session(
            SessionMeta(session_id=self.SESSION, agent_name="test", errors=recorded_errors)
        )
        request = TraceEvent(
            event_type=EventType.LLM_REQUEST, session_id=self.SESSION, data={"model": "m"}
        )
        call = TraceEvent(
            event_type=EventType.TOOL_CALL, session_id=self.SESSION, data={"tool_name": "Bash"}
        )
        events = [
            TraceEvent(event_type=EventType.SESSION_START, session_id=self.SESSION),
            call,
            TraceEvent(
                event_type=EventType.ERROR,
                session_id=self.SESSION,
                parent_id=call.event_id,
                data={"error": "tool failed"},
            ),
            request,
            TraceEvent(
                event_type=EventType.LLM_RESPONSE,
                session_id=self.SESSION,
                parent_id=request.event_id,
                data={"status": "failed", "error": "stream disconnected"},
            ),
        ]
        if linked_error:
            events.append(TraceEvent(
                event_type=EventType.ERROR,
                session_id=self.SESSION,
                parent_id=request.event_id,
                data={"error": "stream disconnected"},
            ))
        events.append(TraceEvent(event_type=EventType.SESSION_END, session_id=self.SESSION))
        for index, event in enumerate(events):
            event.timestamp = 1000.0 + index
            store.append_event(self.SESSION, event)
        return validate_trace_directory(root / self.SESSION)

    @staticmethod
    def findings(report):
        return [(finding.code, finding.severity) for finding in report.findings]

    def test_failed_request_is_counted(self):
        report = self.validate(2)
        self.assertEqual(report.status, "valid")
        self.assertEqual(report.findings, ())
        self.assertEqual(report.summary["state"], "consistent")
        self.assertEqual(report.summary["compared"]["errors"], {"recorded": 2, "observed": 2})

    def test_count_written_before_failed_requests_is_accepted(self):
        report = self.validate(1)
        self.assertEqual(report.status, "valid")
        self.assertEqual(self.findings(report), [("legacy_error_summary", "info")])
        self.assertIn("leaves out 1 failed model request", report.findings[0].message)
        self.assertEqual(report.summary["state"], "consistent")
        self.assertNotIn("summary_contradiction", report.labels)

    def test_other_counts_contradict(self):
        for recorded in (3, 4):
            with self.subTest(recorded=recorded):
                report = self.validate(recorded)
                self.assertEqual(report.status, "invalid")
                self.assertEqual(self.findings(report), [("summary_contradiction", "error")])
                self.assertEqual(report.summary["state"], "contradictory")

    def test_linked_error_and_failed_response_count_once(self):
        # Both record the request's outcome, which health reports, but the
        # request is one error.
        report = self.validate(2, linked_error=True)
        self.assertEqual(report.summary["state"], "consistent")
        self.assertEqual(report.summary["compared"]["errors"], {"recorded": 2, "observed": 2})
        self.assertNotIn("summary_contradiction", [code for code, _ in self.findings(report)])


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
