"""Assignment bundle privacy, hostile-input, rubric, and scoring tests."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import stat
import struct
import tempfile
import unittest
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from agent_trace.assignment import (
    AssignmentError,
    AssignmentCriterion,
    AssignmentRubric,
    LoadedAssignment,
    ZIP_MEMBERS,
    ZIP_TIMESTAMP,
    _ASSIGNMENT_CSS_V1,
    _ASSIGNMENT_JS_V1,
    _atomic_write_private,
    _canonical_json_bytes,
    _inflate_exact_member,
    _parse_trace,
    _read_regular_nofollow,
    _strict_json_bytes,
    build_assignment_bundle,
    build_score_report,
    cmd_score,
    cmd_share_assignment,
    format_score_json,
    format_score_text,
    load_assignment_bundle,
    load_assignment_rubric,
)
from agent_trace.cli import build_parser
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore


def _event(kind: EventType, timestamp: float, event_id: str, **data) -> TraceEvent:
    return TraceEvent(
        event_type=kind,
        timestamp=timestamp,
        event_id=event_id,
        session_id="assignment-session",
        data=data,
    )


def _sample_events() -> list[TraceEvent]:
    return [
        _event(EventType.SESSION_START, 1_000.0, "start", environment={"TOKEN": "ENV_SECRET"}),
        _event(
            EventType.USER_PROMPT,
            1_001.0,
            "prompt",
            prompt="PROMPT_SECRET\nwith\ta tab candidate@example.test",
        ),
        _event(
            EventType.LLM_REQUEST,
            1_002.0,
            "request",
            model="private-model-identifier",
            prompt="MODEL_PROMPT_SECRET",
            input_tokens=100,
        ),
        _event(
            EventType.TOOL_CALL,
            1_003.0,
            "read-one",
            tool_name="Read",
            arguments={"file_path": "/private/CANDIDATE_PATH.txt"},
        ),
        _event(
            EventType.TOOL_CALL,
            1_004.0,
            "read-two",
            tool_name="open_file",
            arguments={"file_path": "/private/CANDIDATE_PATH.txt"},
        ),
        _event(
            EventType.TOOL_RESULT,
            1_005.0,
            "result",
            result="FILE_RESULT_SECRET",
        ),
        _event(
            EventType.LLM_RESPONSE,
            1_006.0,
            "response",
            model="private-model-identifier",
            result="MODEL_RESULT_SECRET",
            output_tokens=25,
        ),
        _event(EventType.SESSION_END, 1_007.0, "end", exit_code=0),
    ]


def _bundle(events: list[TraceEvent] | None = None) -> bytes:
    with tempfile.TemporaryDirectory() as directory:
        # Keep sentinels in the source snapshot so this suite exercises
        # assignment minimization rather than the store's ordinary redactor.
        store = TraceStore(directory, redact=False)
        meta = SessionMeta(
            session_id="assignment-session",
            started_at=1_000.0,
            agent_name="Candidate Name",
            command="run --token META_SECRET",
            attribution={"email": "candidate@example.test"},
        )
        store.create_session(meta)
        for item in events if events is not None else _sample_events():
            store.append_event(meta.session_id, item)
        return build_assignment_bundle(store, meta.session_id)


def _write_rubric(directory: str, text: str) -> Path:
    path = Path(directory) / "rubric.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _valid_rubric() -> str:
    return """task: "Implement rate limiting middleware"
max_cost_usd: 0.50
max_duration_minutes: 30
criteria:
  - name: task-completed
    scorer: session_status
    expected: completed
    weight: 3
  - name: cost-efficiency
    scorer: cost_usd
    threshold: 0.50
    fail_on: above
    weight: 2
  - name: no-tool-loops
    scorer: lint_violations
    rule: tool-loop
    threshold: 0
    weight: 2
"""


def _rewrite_members(members: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in ZIP_MEMBERS:
            info = zipfile.ZipInfo(name, ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            archive.writestr(info, members[name], compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return output.getvalue()


class AssignmentBundleTests(unittest.TestCase):
    def test_checked_in_golden_submission_remains_loadable(self):
        fixture = Path(__file__).resolve().parents[1] / "examples" / "hiring" / "example-submission"
        members = {name: (fixture / name).read_bytes() for name in ZIP_MEMBERS}
        loaded = load_assignment_bundle(_rewrite_members(members))
        self.assertEqual(
            loaded.bundle_digest,
            "sha256:54f280974d4015665bff0ca662dbf6bc9c043610c65e728afe498121f810a907",
        )
        self.assertEqual(loaded.stats["session_status"], "completed")

    def test_bundle_is_deterministic_private_and_exact(self):
        first = _bundle()
        second = _bundle()
        self.assertEqual(first, second)
        loaded = load_assignment_bundle(first)
        self.assertEqual(loaded.stats["event_count"], 8)
        self.assertEqual(loaded.stats["session_status"], "completed")
        with zipfile.ZipFile(io.BytesIO(first)) as archive:
            self.assertEqual(archive.namelist(), list(ZIP_MEMBERS))
            for info in archive.infolist():
                self.assertEqual(info.date_time, ZIP_TIMESTAMP)
                self.assertEqual(stat.S_IMODE(info.external_attr >> 16), 0o600)

    def test_every_uncompressed_member_omits_source_secrets_and_identity(self):
        raw = _bundle()
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            combined = b"\n".join(archive.read(name) for name in archive.namelist())
        for secret in (
            b"ENV_SECRET",
            b"PROMPT_SECRET",
            b"candidate@example.test",
            b"private-model-identifier",
            b"CANDIDATE_PATH",
            b"FILE_RESULT_SECRET",
            b"MODEL_RESULT_SECRET",
            b"META_SECRET",
            b"Candidate Name",
        ):
            self.assertNotIn(secret, combined)
        self.assertIn(b"Content-Security-Policy", combined)
        self.assertIn(b"Model-001", combined)
        self.assertIn(b"Resource-001", combined)

    def test_empty_session_bundle_round_trips(self):
        loaded = load_assignment_bundle(_bundle([]))
        self.assertEqual(loaded.events, [])
        self.assertEqual(loaded.stats["duration_seconds"], 0.0)

    def test_distinct_read_tools_do_not_create_false_tool_loop(self):
        loaded = load_assignment_bundle(_bundle())
        self.assertEqual(loaded.lint["rule_counts"]["tool-loop"], 0)
        self.assertEqual(loaded.stats["redundant_read_ratio"], 0.5)

    def test_v1_assets_and_pricing_are_frozen_from_shared_product_defaults(self):
        self.assertEqual(
            hashlib.sha256(_ASSIGNMENT_CSS_V1.encode()).hexdigest(),
            "ffd47a829e949b906e6249d8ea01c2e598777bfe8cb097689486e124bc630790",
        )
        self.assertEqual(
            hashlib.sha256(_ASSIGNMENT_JS_V1.encode()).hexdigest(),
            "5c80d7b4e02a34de16642793d1a9828a8a18cc5e2f8d01e35e0b52f36499c884",
        )
        with patch.dict("agent_trace.cost.PRICING", {"sonnet": {"input": 999.0, "output": 999.0}}):
            loaded = load_assignment_bundle(_bundle())
        self.assertEqual(loaded.cost["pricing_profile"], "bundled-default-v1")
        self.assertEqual(loaded.cost["input_rate_per_million"], 3.0)
        self.assertEqual(loaded.cost["output_rate_per_million"], 15.0)

    def test_atomic_output_is_private_and_rejects_symlink_target(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "submission.zip"
            _atomic_write_private(output, b"first")
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
            output.unlink()
            target = Path(directory) / "target"
            target.write_bytes(b"untouched")
            output.symlink_to(target)
            with self.assertRaises(AssignmentError):
                _atomic_write_private(output, b"replacement")
            self.assertEqual(target.read_bytes(), b"untouched")

    def test_atomic_output_never_uses_post_replace_chmod(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "submission.zip"
            with patch("agent_trace.assignment.os.chmod") as chmod:
                _atomic_write_private(output, b"private")
            chmod.assert_not_called()
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)

    def test_source_and_output_component_symlinks_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            real_store = Path(directory) / "real-store"
            store = TraceStore(real_store)
            meta = SessionMeta(session_id="assignment-session")
            store.create_session(meta)
            linked_store = Path(directory) / "linked-store"
            linked_store.symlink_to(real_store, target_is_directory=True)
            with self.assertRaises(AssignmentError):
                build_assignment_bundle(TraceStore(linked_store), meta.session_id)

            real_output = Path(directory) / "real-output"
            real_output.mkdir()
            linked_output = Path(directory) / "linked-output"
            linked_output.symlink_to(real_output, target_is_directory=True)
            with self.assertRaises(AssignmentError):
                _atomic_write_private(linked_output / "submission.zip", b"private")

    def test_manifest_boolean_is_not_equal_to_integer_version(self):
        raw = _bundle()
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            members = {name: archive.read(name) for name in ZIP_MEMBERS}
        manifest = json.loads(members["manifest.json"])
        manifest["bundle_version"] = True
        members["manifest.json"] = (
            json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
        ).encode()
        with self.assertRaises(AssignmentError):
            load_assignment_bundle(_rewrite_members(members))

    def test_trailing_bytes_inside_declared_deflate_stream_are_rejected(self):
        original = _bundle()
        with zipfile.ZipFile(io.BytesIO(original)) as archive:
            last = archive.infolist()[-1]
        central_offset = struct.unpack_from("<L", original, len(original) - 6)[0]
        trailer = b"COMPRESSED_TRAILER_SECRET"
        data_end = central_offset
        mutated = bytearray(original[:data_end] + trailer + original[data_end:])
        struct.pack_into("<L", mutated, last.header_offset + 18, last.compress_size + len(trailer))
        central = central_offset + len(trailer)
        cursor = central
        while cursor < len(mutated) - 22:
            name_length, extra_length, comment_length = struct.unpack_from("<3H", mutated, cursor + 28)
            name = bytes(mutated[cursor + 46:cursor + 46 + name_length]).decode("ascii")
            if name == last.filename:
                struct.pack_into("<L", mutated, cursor + 20, last.compress_size + len(trailer))
                break
            cursor += 46 + name_length + extra_length + comment_length
        struct.pack_into("<L", mutated, len(mutated) - 6, central)
        with self.assertRaises(AssignmentError):
            load_assignment_bundle(bytes(mutated))

    def test_hidden_gap_before_central_directory_is_rejected(self):
        original = _bundle()
        central = struct.unpack_from("<L", original, len(original) - 6)[0]
        mutated = bytearray(original[:central] + b"X" + original[central:])
        struct.pack_into("<L", mutated, len(mutated) - 6, central + 1)
        with self.assertRaises(AssignmentError):
            load_assignment_bundle(bytes(mutated))

    def test_altered_local_version_or_timestamp_is_rejected(self):
        original = _bundle()
        for offset, value in ((4, 10), (10, 1), (12, 34)):
            with self.subTest(offset=offset):
                mutated = bytearray(original)
                struct.pack_into("<H", mutated, offset, value)
                with self.assertRaises(AssignmentError):
                    load_assignment_bundle(bytes(mutated))

    def test_small_declared_size_cannot_trigger_unbounded_inflate(self):
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("large", b"A" * 100_000)
        raw = output.getvalue()
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            forged = archive.infolist()[0]
            forged.file_size = 1
            with self.assertRaises(AssignmentError):
                _inflate_exact_member(raw, forged)


class StrictInputTests(unittest.TestCase):
    def test_deep_json_and_oversized_integer_are_normalized(self):
        deep = ("[" * 2_000 + "0" + "]" * 2_000).encode()
        huge = ("1" * 400).encode()
        for raw in (deep, huge):
            with self.assertRaises(AssignmentError):
                _strict_json_bytes(raw, label="hostile", max_bytes=10_000)

    def test_read_stability_detects_same_size_rewrite_with_restored_mtime(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_bytes(b"original")
            before = path.stat()
            real_read = os.read
            changed = False

            def mutate_after_read(descriptor: int, length: int) -> bytes:
                nonlocal changed
                value = real_read(descriptor, length)
                if value and not changed:
                    changed = True
                    path.write_bytes(b"modified")
                    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
                return value

            with patch("agent_trace.assignment.os.read", side_effect=mutate_after_read):
                with self.assertRaises(AssignmentError):
                    _read_regular_nofollow(path, max_bytes=100, label="test input")

    def test_trace_requires_canonical_first_use_aliases_and_encoding(self):
        item = {
            "schema": "agent-strace-assignment-trace-event/v1",
            "sequence": 1,
            "event_ref": "Event-000001",
            "event_type": "tool_call",
            "offset_ms": 0,
            "data": {
                "arguments_omitted": True,
                "content_omitted": True,
                "tool_category": "read",
                "tool_ref": "Tool-999",
                "resource_ref": "Resource-999",
            },
        }
        with self.assertRaises(AssignmentError):
            _parse_trace(_canonical_json_bytes(item) + b"\n")
        item["data"]["tool_ref"] = "Tool-001"
        item["data"]["resource_ref"] = "Resource-001"
        with self.assertRaises(AssignmentError):
            _parse_trace(json.dumps(item).encode() + b"\n")

    def test_rubric_example_defaults_and_rejects_hostile_scalars(self):
        with tempfile.TemporaryDirectory() as directory:
            rubric = load_assignment_rubric(_write_rubric(directory, _valid_rubric()))
            self.assertEqual(rubric.criteria[0].fail_on, "not_equal")
            self.assertEqual(rubric.max_cost_usd, 0.5)
            hostile = _valid_rubric().replace("weight: 3", "weight: " + "9" * 400)
            _write_rubric(directory, hostile)
            with self.assertRaises(AssignmentError):
                load_assignment_rubric(Path(directory) / "rubric.yaml")

    def test_rubric_rejects_surrogate_and_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_rubric(directory, _valid_rubric().replace(
                '"Implement rate limiting middleware"', '"Review \\ud800 payload"'
            ))
            with self.assertRaises(AssignmentError):
                load_assignment_rubric(path)
            target = Path(directory) / "target.yaml"
            target.write_text(_valid_rubric(), encoding="utf-8")
            path.unlink()
            path.symlink_to(target)
            with self.assertRaises(AssignmentError):
                load_assignment_rubric(path)

    def test_rubric_rejects_fractional_weight_duplicate_and_unknown_scorer(self):
        variants = (
            _valid_rubric().replace("weight: 3", "weight: 3.0"),
            _valid_rubric().replace("scorer: session_status", "scorer: mystery"),
            _valid_rubric().replace("    weight: 3", "    weight: 3\n    weight: 4"),
            _valid_rubric().replace("rule: tool-loop", "rule: budget-proximity"),
        )
        with tempfile.TemporaryDirectory() as directory:
            for value in variants:
                with self.subTest(value=value):
                    with self.assertRaises(AssignmentError):
                        load_assignment_rubric(_write_rubric(directory, value))

    def test_rubric_rejects_rules_unavailable_after_privacy_minimization(self):
        for rule in ("budget-proximity", "post-compaction-regression"):
            with self.subTest(rule=rule), tempfile.TemporaryDirectory() as directory:
                value = _valid_rubric().replace("rule: tool-loop", f"rule: {rule}")
                with self.assertRaisesRegex(AssignmentError, "unavailable"):
                    load_assignment_rubric(_write_rubric(directory, value))


class AssignmentScoringTests(unittest.TestCase):
    def test_display_percentage_uses_decimal_half_up_but_rank_uses_exact_fraction(self):
        loaded = load_assignment_bundle(_bundle([]))
        rubric = AssignmentRubric(
            task="Rounding contract",
            criteria=(
                AssignmentCriterion(
                    name="one point",
                    scorer="error_count",
                    weight=1,
                    fail_on="above",
                    threshold=0,
                ),
                AssignmentCriterion(
                    name="thirty-one missed",
                    scorer="error_count",
                    weight=31,
                    fail_on="below",
                    threshold=1,
                ),
            ),
        )
        report = build_score_report([loaded], rubric, compare=False)
        self.assertEqual(report["submissions"][0]["score_percent"], 3.13)

    def test_failed_criterion_scores_and_ties_without_float_fraction_crash(self):
        loaded = load_assignment_bundle(_bundle())
        other = LoadedAssignment(
            bundle_digest="sha256:" + "f" * 64,
            archive_sha256=loaded.archive_sha256,
            events=loaded.events,
            stats={**loaded.stats, "session_status": "terminated"},
            cost=loaded.cost,
            lint=loaded.lint,
        )
        loaded = replace(loaded, bundle_digest="sha256:" + "0" * 64)
        with tempfile.TemporaryDirectory() as directory:
            rubric = load_assignment_rubric(_write_rubric(directory, _valid_rubric()))
        report = build_score_report([loaded, other], rubric, compare=True)
        self.assertEqual(report["submission_count"], 2)
        self.assertEqual([item["rubric_rank"] for item in report["submissions"]], [1, 2])
        failed = report["submissions"][1]
        self.assertIsInstance(failed["points_awarded"], int)
        self.assertTrue(any(not criterion["criterion_met"] for criterion in failed["criteria"]))
        self.assertIn("not a hiring decision", format_score_text(report))
        self.assertEqual(json.loads(format_score_json(report))["schema"], report["schema"])

    def test_equal_scores_share_rank_with_digest_stable_order(self):
        loaded = load_assignment_bundle(_bundle())
        first = replace(loaded, bundle_digest="sha256:" + "1" * 64)
        second = replace(loaded, bundle_digest="sha256:" + "2" * 64)
        with tempfile.TemporaryDirectory() as directory:
            rubric = load_assignment_rubric(_write_rubric(directory, _valid_rubric()))
        report = build_score_report([second, first], rubric, compare=True)
        self.assertEqual([item["rubric_rank"] for item in report["submissions"]], [1, 1])
        self.assertEqual(report["submissions"][0]["bundle_digest"], first.bundle_digest)
        self.assertEqual(
            [item["submission_ref"] for item in report["submissions"]],
            ["Submission-001", "Submission-002"],
        )

    def test_cli_parser_and_handlers(self):
        parser = build_parser()
        share = parser.parse_args(["share", "abc", "--assignment", "-o", "result.zip"])
        self.assertTrue(share.assignment)
        score = parser.parse_args(["score", "result.zip", "--rubric", "rubric.yaml", "--format", "json"])
        self.assertEqual(score.command, "score")
        self.assertEqual(score.format, "json")

        with tempfile.TemporaryDirectory() as directory:
            raw = _bundle()
            submission = Path(directory) / "submission.zip"
            submission.write_bytes(raw)
            rubric = _write_rubric(directory, _valid_rubric())
            args = argparse.Namespace(
                submission=str(submission), rubric=str(rubric), compare=False, format="json"
            )
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                self.assertEqual(cmd_score(args), 0)
            self.assertEqual(stderr.getvalue(), "")
            self.assertEqual(json.loads(stdout.getvalue())["submission_count"], 1)

    def test_share_assignment_writes_private_zip(self):
        with tempfile.TemporaryDirectory() as directory:
            store_dir = Path(directory) / "traces"
            store = TraceStore(store_dir)
            meta = SessionMeta(session_id="assignment-session")
            store.create_session(meta)
            output = Path(directory) / "submission.zip"
            args = argparse.Namespace(
                trace_dir=str(store_dir), session_id=meta.session_id, output=str(output),
                stdout=False, open=False, postmortem=False, assignment=True,
            )
            with redirect_stderr(io.StringIO()):
                self.assertEqual(cmd_share_assignment(args), 0)
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
            load_assignment_bundle(output.read_bytes())


if __name__ == "__main__":
    unittest.main()
