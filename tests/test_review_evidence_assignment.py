"""Exercise the native exporter, offline converter and evidence assembler together."""

from __future__ import annotations

import hashlib
import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    from . import _no_telemetry  # noqa: F401
    from .test_assignment import _bundle
    from .test_review_evidence import Fixture
except ImportError:  # unittest discover -s tests
    import _no_telemetry  # noqa: F401
    from test_assignment import _bundle
    from test_review_evidence import Fixture

from agent_trace import cli
from agent_trace.assignment import load_assignment_bundle
from agent_trace.review_evidence_assignment import SUMMARY_FILENAME


def prepare_assignment_fixture(root: Path) -> None:
    """Generate a ZIP through the real exporter with only synthetic trace data."""
    fixture = Fixture(root / "input")
    fixture.receipt()
    fixture.intent["expected_steps"] = [{"id": "inspect", "text": "Inspect greeting.", "required": True}]
    fixture.manifest["association"] = {"by": "fixture operator", "session_id": "s1", "attempt_id": "a1"}
    fixture.write()
    archive = _bundle()
    (root / "assignment.zip").write_bytes(archive)
    (root / "binding.json").write_text(json.dumps({
        "archive_sha256": hashlib.sha256(archive).hexdigest(),
        "target": fixture.manifest["target"], "association": fixture.manifest["association"],
    }), encoding="utf-8")


def admit_assignment_summary(bundle: Path, summary: Path) -> None:
    """Explicitly admit the generated record into the synthetic evidence manifest."""
    data = json.loads(bundle.read_bytes())
    raw = summary.read_bytes()
    data["sources"].append({
        "id": "assignment-summary", "path": summary.relative_to(bundle.parent).as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(), "type": "session-summary", "version": 0,
        "origin": "agent-strace assignment v1 adapter", "admission": "SYNTHETIC",
        "lines": [1, raw.count(b"\n")],
    })
    bundle.write_text(json.dumps(data), encoding="utf-8")


class AssignmentSummaryTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        prepare_assignment_fixture(self.root)
        self.archive = self.root / "assignment.zip"
        self.bundle = self.root / "input" / "bundle.json"
        self.binding = self.root / "binding.json"
        self.output = self.bundle.parent / "converted"

    def cli(self, *args: str) -> tuple[int, str, str]:
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            patch.object(sys, "argv", ["agent-strace", *args]),
            patch.object(cli._product_telemetry, "maybe_show_telemetry_notice", side_effect=AssertionError),
            patch.object(cli._product_telemetry, "capture", side_effect=AssertionError),
            patch.object(cli, "TraceStore", side_effect=AssertionError),
            redirect_stdout(stdout), redirect_stderr(stderr),
            self.assertRaises(SystemExit) as ctx,
        ):
            cli.main()
        return ctx.exception.code, stdout.getvalue(), stderr.getvalue()

    def convert(self, output: Path | None = None) -> tuple[int, str, str]:
        return self.cli(
            "review-evidence-import-assignment", str(self.archive), str(self.bundle),
            "--binding", str(self.binding), "--output", str(output or self.output),
        )

    def update_binding(self, **updates) -> None:
        binding = json.loads(self.binding.read_bytes())
        self.binding.write_text(json.dumps({**binding, **updates}), encoding="utf-8")

    def assert_refused(self, message: str) -> None:
        status, stdout, stderr = self.convert()
        self.assertEqual((status, stdout), (1, ""))
        self.assertIn(message, stderr)
        self.assertFalse(self.output.exists())

    def test_real_minimized_archive_produces_only_partial_summary_consumed_by_assembler(self):
        originals = {path: path.read_bytes() for path in (self.archive, self.bundle, self.binding)}
        exported = load_assignment_bundle(originals[self.archive])
        self.assertGreater(exported.stats["tool_call_count"], 0)
        self.assertGreater(exported.cost["total_cost_usd"], 0)
        self.assertEqual(self.convert(), (0, f"Wrote {SUMMARY_FILENAME} (PARTIAL; no named-step observations)\n", ""))
        summary = self.output / SUMMARY_FILENAME
        raw = summary.read_bytes()
        self.assertEqual(list(self.output.iterdir()), [summary])
        self.assertEqual(self.convert(self.bundle.parent / "repeat")[0], 0)
        self.assertEqual((self.bundle.parent / "repeat" / SUMMARY_FILENAME).read_bytes(), raw)
        for path, data in originals.items():
            self.assertEqual(path.read_bytes(), data)
        data = json.loads(raw)
        self.assertEqual(data["target"], json.loads(originals[self.bundle])["target"])
        self.assertEqual((data["session_id"], data["attempt_id"]), ("s1", "a1"))
        self.assertEqual(data["capture"]["completeness"], "PARTIAL")
        self.assertEqual(data["observations"], [])
        self.assertIn(hashlib.sha256(originals[self.archive]).hexdigest(), data["capture"]["limits"][0])
        for omitted in (
            b"assignment-session", b"Session-001", b"PROMPT_SECRET", b"FILE_RESULT_SECRET",
            b"CANDIDATE_PATH", b"private-model-identifier", b"META_SECRET", b"candidate@example.test",
            b"estimated_cost", b"Tool-001", b"exit_code",
        ):
            self.assertNotIn(omitted, raw)

        admit_assignment_summary(self.bundle, summary)
        packet_dir = self.root / "packet"
        self.assertEqual(self.cli("review-evidence", str(self.bundle), "--output", str(packet_dir)),
                         (2, "VALID / INCONCLUSIVE / PASS\n", ""))
        packet = json.loads((packet_dir / "packet.json").read_bytes())
        self.assertEqual(packet["steps"][0]["status"], "UNOBSERVED")
        self.assertIsNone(packet["steps"][0]["observation_source"])
        self.assertEqual(packet["gaps"][0]["code"], "STEP_UNOBSERVED")
        self.assertEqual(packet["capture"]["completeness"], "PARTIAL")
        self.assertEqual(packet["attempt_inventory"]["aggregate_cost"]["completeness"], "UNKNOWN")
        self.assertEqual(packet["association"]["matching_summary"], {"source_id": "assignment-summary", "pointer": ""})
        self.assertEqual((packet_dir / "sources" / "assignment-summary.txt").read_bytes(), raw)

    def test_rejects_other_formats_and_corrupt_zip_even_when_pinned(self):
        for other in (b'{"schema":"review-evidence-session-summary/v0"}', self.archive.read_bytes()[:-10]):
            with self.subTest(prefix=other[:4]):
                self.archive.write_bytes(other)
                self.update_binding(archive_sha256=hashlib.sha256(other).hexdigest())
                self.assert_refused("invalid or unsupported agent-strace-assignment-bundle/v1 ZIP")

    def test_rejects_digest_target_or_association_mismatches(self):
        original = json.loads(self.binding.read_bytes())
        for key, value, message in [
            ("archive_sha256", "0" * 64, "ZIP SHA-256 does not match"),
            ("target", {**original["target"], "head": "c" * 40}, "target does not match"),
            ("association", {**original["association"], "session_id": "other"}, "association does not match"),
            ("association", {**original["association"], "attempt_id": "other"}, "association does not match"),
        ]:
            with self.subTest(key=key, value=value):
                self.update_binding(**{**original, key: value})
                self.assert_refused(message)
        self.update_binding(**original)
        manifest = json.loads(self.bundle.read_bytes())
        del manifest["association"]
        self.bundle.write_text(json.dumps(manifest), encoding="utf-8")
        self.assert_refused("bundle has missing or unsupported fields")

    def test_rejects_oversize_archive_symlink_and_existing_output(self):
        with patch("agent_trace.review_evidence_assignment.MAX_ARCHIVE_BYTES", self.archive.stat().st_size - 1):
            self.assert_refused("assignment ZIP size or type is invalid")
        self.output.mkdir()
        old = self.output / SUMMARY_FILENAME
        old.write_bytes(b"existing summary")
        self.assertEqual(self.convert(), (1, "", "review-evidence-import-assignment: output already exists\n"))
        self.assertEqual(old.read_bytes(), b"existing summary")
        original = self.archive
        self.archive = self.root / "link.zip"
        try:
            self.archive.symlink_to(original)
        except OSError as exc:
            self.skipTest(f"symlinks unavailable: {exc}")
        self.output = self.bundle.parent / "not-written"
        self.assert_refused("assignment ZIP could not be opened safely")
