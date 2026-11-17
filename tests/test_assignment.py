"""Assignment bundle privacy, hostile-input, rubric, and scoring tests."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import stat
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
import zipfile
import zlib
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

try:
    from _no_telemetry import without_telemetry
except ImportError:  # pragma: no cover - run as a package module
    from tests._no_telemetry import without_telemetry

import agent_trace
from agent_trace import fileutil
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
    _load_submission_inputs,
    _open_directory_chain,
    _parse_trace,
    _read_regular_nofollow,
    _strict_json_bytes,
    _validate_no_symlink_ancestors,
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
from agent_trace.fileutil import MAX_TRUSTED_SYMLINKS, trusted_symlink_target
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore, _store_write_lock


SRC_ROOT = Path(agent_trace.__file__).resolve().parent.parent
# Windows has no dir_fd, so it only has the by-path helpers.
FS_MODES = ("dir_fd", "by_path") if os.open in os.supports_dir_fd else ("by_path",)


def fs_mode(mode: str):
    return patch("agent_trace.assignment._USE_DIR_FD", mode == "dir_fd")


def _symlink_or_skip(test: unittest.TestCase, link: Path, target: Path, *, directory=False):
    try:
        link.symlink_to(target, target_is_directory=directory)
    except (OSError, NotImplementedError) as exc:
        test.skipTest(f"cannot create symlinks here: {exc}")


def _assert_private_mode(test: unittest.TestCase, path: Path) -> None:
    # Windows maps a mode only to the read-only flag; its ACL decides access.
    if os.name != "nt":
        test.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)


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
            _assert_private_mode(self, output)
            output.unlink()
            target = Path(directory) / "target"
            target.write_bytes(b"untouched")
            _symlink_or_skip(self, output, target)
            with self.assertRaises(AssignmentError):
                _atomic_write_private(output, b"replacement")
            self.assertEqual(target.read_bytes(), b"untouched")

    def test_atomic_output_never_uses_post_replace_chmod(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "submission.zip"
            with patch("agent_trace.assignment.os.chmod") as chmod:
                _atomic_write_private(output, b"private")
            chmod.assert_not_called()
            _assert_private_mode(self, output)

    def test_source_and_output_component_symlinks_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            real_store = Path(directory) / "real-store"
            store = TraceStore(real_store)
            meta = SessionMeta(session_id="assignment-session")
            store.create_session(meta)
            linked_store = Path(directory) / "linked-store"
            _symlink_or_skip(self, linked_store, real_store, directory=True)
            with self.assertRaises(AssignmentError):
                build_assignment_bundle(TraceStore(linked_store), meta.session_id)

            real_output = Path(directory) / "real-output"
            real_output.mkdir()
            linked_output = Path(directory) / "linked-output"
            _symlink_or_skip(self, linked_output, real_output, directory=True)
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

    @unittest.skipIf(os.name == "nt", "Windows has no change time, only a creation time")
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
            _symlink_or_skip(self, path, target)
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
            _assert_private_mode(self, output)
            load_assignment_bundle(output.read_bytes())


# The bundle of the session ``ByPathParityTests.make_store`` records, with the
# producer version pinned to GOLDEN_VERSION so a release does not change it.
GOLDEN_VERSION = "0.0.0"
GOLDEN_BUNDLE_DIGEST = "sha256:56d90e7f6735f89e215ce75e3c0cfab1848ee964a3a6c32c3143a16a63a8720a"
GOLDEN_ARCHIVE_SHA256 = "b176d75c233b1f0cc76cae5088d2ce64f5c80486003397ed75dc2d3c0efb42c9"


class ByPathParityTests(unittest.TestCase):
    """The by-path helpers that Windows uses must act like the dir_fd ones."""

    SESSION = "assignment-session"

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        # Resolve the temp dir: the by-path helpers refuse links such as macOS /var.
        self.root = Path(tmp.name).resolve()

    def make_store(self, name: str = "traces") -> Path:
        store_dir = self.root / name
        store = TraceStore(store_dir, redact=False, use_workspace_env=False)
        store.create_session(SessionMeta(
            session_id=self.SESSION,
            started_at=1_000.0,
            agent_name="Candidate Name",
            command="run --token META_SECRET",
            attribution={"email": "candidate@example.test"},
        ))
        for item in _sample_events():
            store.append_event(self.SESSION, item)
        return store_dir

    def share(self, store_dir: Path, output: Path) -> None:
        args = argparse.Namespace(
            trace_dir=str(store_dir), session_id=self.SESSION, output=str(output),
            stdout=False, open=False, postmortem=False, assignment=True,
        )
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            self.assertEqual(cmd_share_assignment(args), 0, stderr.getvalue())

    def score(self, submission: Path, rubric: Path, *, compare: bool) -> str:
        args = argparse.Namespace(
            submission=str(submission), rubric=str(rubric), compare=compare, format="json"
        )
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            self.assertEqual(cmd_score(args), 0, stderr.getvalue())
        return stdout.getvalue()

    def error(self, call) -> str:
        with self.assertRaises(AssignmentError) as caught:
            call()
        return str(caught.exception)

    def test_share_and_score_outputs_match_across_modes(self):
        store_dir = self.make_store()
        rubric = _write_rubric(str(self.root), _valid_rubric())
        results = {}
        for mode in FS_MODES:
            with self.subTest(mode=mode), fs_mode(mode), patch(
                "agent_trace.assignment.__version__", GOLDEN_VERSION
            ):
                output = self.root / mode / "bundles" / "submission.zip"
                self.share(store_dir, output)
                self.assertEqual([item.name for item in output.parent.iterdir()], ["submission.zip"])
                _assert_private_mode(self, output)
                results[mode] = (
                    output.read_bytes(),
                    self.score(output, rubric, compare=False),
                    self.score(output.parent, rubric, compare=True),
                )
        self.assertEqual(len(set(results.values())), 1)
        raw, single, compared = results[FS_MODES[0]]
        self.assertEqual(load_assignment_bundle(raw).bundle_digest, GOLDEN_BUNDLE_DIGEST)
        self.assertEqual(
            json.loads(single)["submissions"][0]["bundle_digest"], GOLDEN_BUNDLE_DIGEST
        )
        self.assertEqual(json.loads(compared)["submission_count"], 1)
        # Compressed bytes depend on the zlib build; zlib-ng deflates differently.
        if "ng" not in zlib.ZLIB_RUNTIME_VERSION:
            self.assertEqual(hashlib.sha256(raw).hexdigest(), GOLDEN_ARCHIVE_SHA256)

    def test_errors_match_across_modes(self):
        rubric = _write_rubric(str(self.root), _valid_rubric())
        gone_dir = self.make_store("gone")
        gone = TraceStore(gone_dir, use_workspace_env=False)
        shutil.rmtree(gone_dir)
        file_session = TraceStore(self.make_store("file-session"), use_workspace_env=False)
        shutil.rmtree(file_session.base_dir / self.SESSION)
        (file_session.base_dir / self.SESSION).write_bytes(b"x")
        no_meta = TraceStore(self.make_store("no-meta"), use_workspace_env=False)
        (no_meta.base_dir / self.SESSION / "meta.json").unlink()
        plain_file = self.root / "plain-file"
        plain_file.write_bytes(b"x")
        taken = self.root / "out" / "taken.zip"
        taken.mkdir(parents=True)
        rubric_dir = self.root / "rubric-dir.yaml"
        rubric_dir.mkdir()
        empty_rubric = self.root / "empty.yaml"
        empty_rubric.write_bytes(b"")
        folder_zip = self.root / "folder-zip"
        (folder_zip / "folder.zip").mkdir(parents=True)
        no_zip = self.root / "no-zip"
        no_zip.mkdir()

        def score(submission, *, compare=True, rubric_path=rubric):
            def call():
                bundles = _load_submission_inputs(submission, compare=compare)
                load_assignment_rubric(rubric_path)
                return bundles
            return call

        cases = {
            "missing store": (
                lambda: build_assignment_bundle(gone, self.SESSION),
                "directory does not exist",
            ),
            "session is a file": (
                lambda: build_assignment_bundle(file_session, self.SESSION),
                "session directory could not be opened safely",
            ),
            "missing session": (
                lambda: build_assignment_bundle(no_meta, "other-session"),
                "session directory could not be opened safely",
            ),
            "missing meta.json": (
                lambda: build_assignment_bundle(no_meta, self.SESSION),
                "session metadata could not be opened safely",
            ),
            "output parent is a file": (
                lambda: _atomic_write_private(plain_file / "out.zip", b"x"),
                "directory has an unsafe component",
            ),
            "output target is a directory": (
                lambda: _atomic_write_private(taken, b"x"),
                "output must be a non-symlinked regular file",
            ),
            "missing rubric": (
                lambda: load_assignment_rubric(self.root / "missing.yaml"),
                "rubric file could not be opened safely",
            ),
            "rubric is a directory": (
                lambda: load_assignment_rubric(rubric_dir),
                "rubric file size or type is invalid",
            ),
            "empty rubric": (
                lambda: load_assignment_rubric(empty_rubric),
                "rubric file size or type is invalid",
            ),
            "missing submission": (
                score(self.root / "missing.zip", compare=False),
                "submission archive could not be opened safely",
            ),
            "missing compare directory": (
                score(self.root / "missing"),
                "directory does not exist",
            ),
            "compare directory has no ZIP": (
                score(no_zip),
                "submission directory ZIP count is invalid",
            ),
            "directory named .zip": (
                score(folder_zip),
                "submission directory contains an unsafe ZIP entry",
            ),
        }
        if self._can_symlink():
            cases.update(self._symlink_cases(rubric, score))
        for mode in FS_MODES:
            for name, (call, message) in cases.items():
                with self.subTest(mode=mode, case=name), fs_mode(mode):
                    self.assertEqual(self.error(call), message)
        self.assertEqual(plain_file.read_bytes(), b"x")

    def _can_symlink(self) -> bool:
        probe = self.root / "probe-link"
        try:
            probe.symlink_to(self.root / "plain-file")
        except (OSError, NotImplementedError):
            return False
        probe.unlink()
        return True

    def _symlink_cases(self, rubric: Path, score) -> dict:
        linked_meta = TraceStore(self.make_store("linked-meta"), use_workspace_env=False)
        meta = linked_meta.base_dir / self.SESSION / "meta.json"
        real_meta = self.root / "real-meta.json"
        real_meta.write_bytes(meta.read_bytes())
        meta.unlink()
        meta.symlink_to(real_meta)

        linked_session = TraceStore(self.make_store("linked-session"), use_workspace_env=False)
        real_session = self.root / "real-session"
        (linked_session.base_dir / self.SESSION).rename(real_session)
        (linked_session.base_dir / self.SESSION).symlink_to(real_session, target_is_directory=True)

        linked_lock = TraceStore(self.make_store("linked-lock"), use_workspace_env=False)
        lock_target = self.root / "lock-target"
        lock_target.write_bytes(b"")
        (linked_lock.base_dir / ".store.lock").unlink()
        (linked_lock.base_dir / ".store.lock").symlink_to(lock_target)

        linked_rubric = self.root / "linked-rubric.yaml"
        linked_rubric.symlink_to(rubric)
        output_target = self.root / "output-target"
        output_target.write_bytes(b"untouched")
        linked_output = self.root / "linked-output.zip"
        linked_output.symlink_to(output_target)
        linked_zip = self.root / "linked-zip"
        linked_zip.mkdir()
        (linked_zip / "a.zip").symlink_to(self.root / "plain-file")

        return {
            "meta.json is a symlink": (
                lambda: build_assignment_bundle(linked_meta, self.SESSION),
                "session metadata could not be opened safely",
            ),
            "session is a symlink": (
                lambda: build_assignment_bundle(linked_session, self.SESSION),
                "session directory could not be opened safely",
            ),
            ".store.lock is a symlink": (
                lambda: build_assignment_bundle(linked_lock, self.SESSION),
                "trace store lock could not be opened safely",
            ),
            "rubric is a symlink": (
                lambda: load_assignment_rubric(linked_rubric),
                "rubric file could not be opened safely",
            ),
            "output is a symlink": (
                lambda: _atomic_write_private(linked_output, b"x"),
                "output must be a non-symlinked regular file",
            ),
            "symlinked .zip": (
                score(linked_zip),
                "submission directory contains an unsafe ZIP entry",
            ),
        }

    def test_snapshot_waits_for_the_store_write_lock(self):
        store_dir = self.make_store()
        for mode in FS_MODES:
            with self.subTest(mode=mode), fs_mode(mode):
                store = TraceStore(store_dir, use_workspace_env=False)
                done = threading.Event()
                result = {}

                def build():
                    try:
                        result["raw"] = build_assignment_bundle(store, self.SESSION)
                    except BaseException as exc:  # reported below
                        result["error"] = exc
                    finally:
                        done.set()

                with _store_write_lock(store_dir):
                    thread = threading.Thread(target=build)
                    thread.start()
                    self.assertFalse(done.wait(0.3))
                thread.join(30)
                self.assertTrue(done.is_set())
                self.assertNotIn("error", result)
                load_assignment_bundle(result["raw"])

    def test_output_write_replaces_exact_bytes_and_cleans_up(self):
        payload = b"second\r\n\x1a\n"
        for mode in FS_MODES:
            with self.subTest(mode=mode), fs_mode(mode):
                output = self.root / mode / "made" / "here" / "submission.zip"
                _atomic_write_private(output, b"first")
                _atomic_write_private(output, payload)
                self.assertEqual(output.read_bytes(), payload)
                with patch("agent_trace.assignment.os.write", side_effect=OSError("disk full")):
                    with self.assertRaises(OSError):
                        _atomic_write_private(output, b"third")
                self.assertEqual(output.read_bytes(), payload)
                self.assertEqual([item.name for item in output.parent.iterdir()], ["submission.zip"])
                _assert_private_mode(self, output)

    def test_rubric_accepts_crlf_but_not_a_lone_cr(self):
        lf = _valid_rubric().encode()
        path = self.root / "rubric.yaml"
        path.write_bytes(lf)
        expected = load_assignment_rubric(path)
        path.write_bytes(lf.replace(b"\n", b"\r\n"))
        self.assertEqual(load_assignment_rubric(path), expected)
        for hostile in (lf.replace(b"\n", b"\r"), lf.replace(b"\n", b"\r\r\n", 1)):
            with self.subTest(hostile=hostile[:40]):
                path.write_bytes(hostile)
                with self.assertRaises(AssignmentError):
                    load_assignment_rubric(path)

    @unittest.skipUnless(os.name == "nt", "Windows junctions")
    def test_junctions_are_refused(self):
        import _winapi

        store_dir = self.make_store()
        rubric = _write_rubric(str(self.root), _valid_rubric())
        bundles = self.root / "bundles"
        self.share(store_dir, bundles / "submission.zip")

        def junction(target: Path, name: str) -> Path:
            path = self.root / name
            _winapi.CreateJunction(str(target), str(path))
            return path

        store_junction = junction(store_dir, "store-junction")
        output_junction = junction(bundles, "output-junction")
        compare_junction = junction(bundles, "compare-junction")
        rubric_junction = junction(self.root, "rubric-junction")
        session_junction = store_dir / "junction-session"
        _winapi.CreateJunction(str(store_dir / self.SESSION), str(session_junction))
        cases = {
            "store": lambda: build_assignment_bundle(
                TraceStore(store_junction, use_workspace_env=False), self.SESSION
            ),
            "output": lambda: _atomic_write_private(output_junction / "new.zip", b"x"),
            "compare": lambda: _load_submission_inputs(compare_junction, compare=True),
            "submission": lambda: _load_submission_inputs(
                compare_junction / "submission.zip", compare=False
            ),
            "rubric": lambda: load_assignment_rubric(rubric_junction / rubric.name),
        }
        for name, call in cases.items():
            with self.subTest(case=name):
                self.assertEqual(self.error(call), "directory has an unsafe component")
        self.assertEqual(
            self.error(lambda: build_assignment_bundle(
                TraceStore(store_dir, use_workspace_env=False), "junction-session"
            )),
            "session directory could not be opened safely",
        )
        self.assertEqual([item.name for item in bundles.iterdir()], ["submission.zip"])

    def test_cli_shares_and_scores_end_to_end(self):
        store_dir = self.make_store()
        rubric = _write_rubric(str(self.root), _valid_rubric())
        output = self.root / "bundles" / "submission.zip"
        env = without_telemetry({
            key: value
            for key, value in os.environ.items()
            if not key.startswith("AGENT_STRACE_") and key != "AGENT_TRACE_DIR"
        })
        env["PYTHONPATH"] = os.pathsep.join(
            item for item in (str(SRC_ROOT), env.get("PYTHONPATH", "")) if item
        )
        # The text report has characters that cp1252 stdout on Windows lacks.
        env["PYTHONIOENCODING"] = "utf-8"
        command = [
            sys.executable, "-c", "from agent_trace.entry import main; main()",
            "--trace-dir", str(store_dir),
        ]

        def run(*args: str) -> subprocess.CompletedProcess:
            result = subprocess.run(
                command + list(args),
                capture_output=True,
                text=True,
                encoding="utf-8",
                env=env,
                timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return result

        shared = run("share", self.SESSION, "--assignment", "-o", str(output))
        self.assertIn("Created assignment bundle", shared.stderr)
        in_process = self.root / "in-process" / "submission.zip"
        self.share(store_dir, in_process)
        self.assertEqual(output.read_bytes(), in_process.read_bytes())
        digest = load_assignment_bundle(output.read_bytes()).bundle_digest
        _assert_private_mode(self, output)
        single = json.loads(
            run("score", str(output), "--rubric", str(rubric), "--format", "json").stdout
        )
        compared = json.loads(run(
            "score", str(output.parent), "--compare", "--rubric", str(rubric), "--format", "json"
        ).stdout)
        text = run("score", str(output), "--rubric", str(rubric)).stdout
        self.assertEqual(single["submission_count"], 1)
        self.assertEqual(single["submissions"][0]["bundle_digest"], digest)
        self.assertEqual(compared["submission_count"], 1)
        self.assertEqual(compared["submissions"], single["submissions"])
        self.assertIn("not a hiring decision", text)


def _same_directory(test: unittest.TestCase, fd: int, expected: str | Path) -> None:
    opened = os.fstat(fd)
    target = os.stat(expected)
    test.assertEqual((opened.st_dev, opened.st_ino), (target.st_dev, target.st_ino))


class TrustedParentSymlinkTests(unittest.TestCase):
    """Root-owned OS links such as macOS /tmp and /var, with the real policy."""

    def test_directory_chain_follows_root_owned_system_link(self):
        link = next(
            (
                candidate
                for candidate in ("/tmp", "/var", "/etc", "/bin", "/sbin", "/lib")
                if trusted_symlink_target(candidate) is not None
            ),
            None,
        )
        if link is None:
            self.skipTest("no trusted root-owned symlink in /")
        fd = _open_directory_chain(link)
        try:
            _same_directory(self, fd, os.path.realpath(link))
        finally:
            os.close(fd)

    def test_share_and_score_accept_absolute_paths_through_tmp(self):
        if sys.platform != "darwin" or trusted_symlink_target("/tmp") is None:
            self.skipTest("needs the root-owned /tmp symlink of macOS")
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            self.assertTrue(directory.startswith("/tmp/"))
            store_dir = Path(directory) / "traces"
            store = TraceStore(store_dir)
            meta = SessionMeta(session_id="assignment-session")
            store.create_session(meta)
            output = Path(directory) / "bundles" / "submission.zip"
            args = argparse.Namespace(
                trace_dir=str(store_dir), session_id=meta.session_id, output=str(output),
                stdout=False, open=False, postmortem=False, assignment=True,
            )
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                self.assertEqual(cmd_share_assignment(args), 0, stderr.getvalue())
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
            rubric = _write_rubric(directory, _valid_rubric())
            for submission, compare in ((output, False), (output.parent, True)):
                with self.subTest(compare=compare):
                    score_args = argparse.Namespace(
                        submission=str(submission), rubric=str(rubric),
                        compare=compare, format="json",
                    )
                    stdout, stderr = io.StringIO(), io.StringIO()
                    with redirect_stdout(stdout), redirect_stderr(stderr):
                        self.assertEqual(cmd_score(score_args), 0, stderr.getvalue())
                    self.assertEqual(json.loads(stdout.getvalue())["submission_count"], 1)


@unittest.skipIf(os.name == "nt" or not hasattr(os, "symlink"), "POSIX directory descriptors")
class DirectoryChainSymlinkTests(unittest.TestCase):
    """Anchored directory walks that trust only the links each test marks."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        # Resolve first so the walk meets only the links each test makes.
        self.root = Path(os.path.realpath(tmp.name))
        (self.root / "real" / "sub").mkdir(parents=True)
        (self.root / "other").mkdir()
        self.trusted: set[tuple[int, int]] = set()
        for patcher in (
            patch.object(fileutil, "_trusts_root_owned_symlinks", return_value=True),
            patch.object(fileutil, "_symlink_is_trusted", side_effect=self._is_trusted),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def _is_trusted(self, link_stat, dir_stat):
        return (link_stat.st_dev, link_stat.st_ino) in self.trusted

    def _link(self, name: str, target: str, *, trusted: bool) -> Path:
        path = self.root / name
        path.symlink_to(target, target_is_directory=True)
        if trusted:
            info = os.lstat(path)
            self.trusted.add((info.st_dev, info.st_ino))
        return path

    def test_follows_relative_absolute_and_parent_targets(self):
        (self.root / "nested").mkdir()
        self._link("relative", "real", trusted=True)
        self._link("absolute", str(self.root / "real"), trusted=True)
        self._link("nested/up", "../real", trusted=True)
        for name in ("relative", "absolute", "nested/up"):
            with self.subTest(name=name):
                fd = _open_directory_chain(self.root / name / "sub")
                try:
                    _same_directory(self, fd, self.root / "real" / "sub")
                finally:
                    os.close(fd)

    def test_refuses_untrusted_link_inside_trusted_target(self):
        self._link("real/inner", "../other", trusted=False)
        self._link("outer", "real", trusted=True)
        self._link("direct", "real/inner", trusted=True)
        for path in (self.root / "outer" / "inner", self.root / "direct"):
            with self.subTest(path=path):
                with self.assertRaisesRegex(AssignmentError, "directory has an unsafe component"):
                    _open_directory_chain(path)
                with self.assertRaisesRegex(
                    AssignmentError, "refusing a path with a symlink component"
                ):
                    _validate_no_symlink_ancestors(path / "store")
        _validate_no_symlink_ancestors(self.root / "outer" / "sub")

    def test_creates_only_components_named_by_the_caller(self):
        self._link("relative", "real", trusted=True)
        self._link("dangling", "missing", trusted=True)
        fd = _open_directory_chain(self.root / "relative" / "made", create=True)
        os.close(fd)
        self.assertTrue((self.root / "real" / "made").is_dir())
        with self.assertRaisesRegex(AssignmentError, "directory does not exist"):
            _open_directory_chain(self.root / "dangling" / "made", create=True)
        self.assertFalse((self.root / "missing").exists())

    def test_refuses_trusted_link_loop_and_long_chain(self):
        self._link("loop", "loop", trusted=True)
        with self.assertRaisesRegex(AssignmentError, "directory has an unsafe component"):
            _open_directory_chain(self.root / "loop")
        previous = "real"
        for index in range(MAX_TRUSTED_SYMLINKS):
            previous = self._link(f"hop{index}", previous, trusted=True).name
        os.close(_open_directory_chain(self.root / previous))
        self._link("one-more", previous, trusted=True)
        with self.assertRaisesRegex(AssignmentError, "directory has an unsafe component"):
            _open_directory_chain(self.root / "one-more")


if __name__ == "__main__":
    unittest.main()
