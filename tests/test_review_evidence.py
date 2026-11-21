"""Tests for the offline assembler using only synthetic retained evidence."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    from . import _no_telemetry  # noqa: F401
except ImportError:
    import _no_telemetry  # noqa: F401

from agent_trace import cli, review_evidence as evidence

TARGET = {"repo": "demo/greetings", "pr": 7, "task": "friendlier-label", "base": "a" * 40, "head": "b" * 40}
DIFF = (
    b"diff --git a/greet.py b/greet.py\n"
    b"index 0123456..fedcba9 100644\n"
    b"--- a/greet.py\n+++ b/greet.py\n"
    b"@@ -1 +1 @@\n-print('Hi')\n+print('Hello')\n"
)


class Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir()
        self.manifest = {"schema": evidence.INPUT_SCHEMA, "target": TARGET.copy(), "sources": []}
        self.intent = {
            "schema": evidence.RECORD_SCHEMAS["intent"], "target": TARGET.copy(),
            "by": "fixture author", "summary": "Use a friendlier greeting.",
            "criteria": [{"id": "hello", "text": "Greeting is Hello."}],
        }
        self.add("change", evidence.DIFF_TYPE, DIFF)

    def add(self, sid: str, kind: str, value: bytes | dict) -> dict:
        raw = value if isinstance(value, bytes) else json.dumps(value, indent=2).encode() + b"\n"
        name = sid + ".input"
        (self.root / name).write_bytes(raw)
        entry = {
            "id": sid, "type": kind, "version": 0, "path": name,
            "origin": "fixture author", "admission": "SYNTHETIC",
            "sha256": hashlib.sha256(raw).hexdigest(), "lines": [1, len(raw.splitlines())],
        }
        sources = self.manifest["sources"]
        sources[:] = [old for old in sources if old["id"] != sid]
        sources.append(entry)
        return entry

    def receipt(
        self, sid: str = "unit", *, check_id: str | None = None, head: str = TARGET["head"],
        result: str = "PASS", relevant: bool = True, target: dict | None = None,
    ) -> dict:
        return self.add(sid, "validation-receipt", {
            "schema": evidence.RECORD_SCHEMAS["validation-receipt"],
            "target": {**TARGET, "head": head} if target is None else target,
            "check_id": check_id or sid, "reported_by": "fixture CI", "result": result,
            "assessment": {
                "by": "fixture maintainer", "relevance": "RELEVANT" if relevant else "IRRELEVANT",
                "criterion_ids": ["hello"] if relevant else [],
            },
        })

    def summary(self, *, observed: bool = True, session_id: str = "s1") -> None:
        self.manifest["association"] = {"by": "fixture operator", "session_id": "s1", "attempt_id": "a1"}
        value = {
            "schema": evidence.RECORD_SCHEMAS["session-summary"], "target": TARGET.copy(),
            "session_id": session_id, "attempt_id": "a1", "reported_by": "fixture recorder",
            "capture": {"completeness": "PARTIAL", "limits": ["Only selected fixture events were retained."]},
            "observations": [{"step_id": "inspect", "note": "Inspected the greeting."}] if observed else [],
        }
        self.add("summary", "session-summary", value)

    def attempts(self, rows: list, completeness: str = "COMPLETE") -> None:
        self.add("attempts", "attempt-inventory", {
            "schema": evidence.RECORD_SCHEMAS["attempt-inventory"], "target": TARGET.copy(),
            "by": "fixture operator", "completeness": completeness, "attempts": rows,
        })

    def write(self) -> Path:
        self.add("intent", "intent", self.intent)
        path = self.root / "bundle.json"
        path.write_text(json.dumps(self.manifest), encoding="utf-8")
        return path


class ReviewEvidenceTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        # macOS's TMPDIR normally contains /var, itself a symlink.
        self.root = Path(temp.name).resolve()
        self.fixture = Fixture(self.root / "input")

    def run_cli(self, *args: str) -> tuple[int, str, str]:
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            patch.object(sys, "argv", ["agent-strace", *args]),
            patch.object(cli._product_telemetry, "maybe_show_telemetry_notice", side_effect=AssertionError),
            patch.object(cli._product_telemetry, "capture", side_effect=AssertionError),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
            self.assertRaises(SystemExit) as ctx,
        ):
            cli.main()
        return ctx.exception.code, stdout.getvalue(), stderr.getvalue()

    def cli(self, path: Path, output: Path) -> tuple[int, str, str]:
        return self.run_cli("review-evidence", str(path), "--output", str(output))

    def packet(self, name: str, code: int, validation: str) -> tuple[Path, dict]:
        output = self.root / name
        status = "COMPLETE" if code == 0 else "INCONCLUSIVE"
        self.assertEqual(self.cli(self.fixture.write(), output), (code, f"VALID / {status} / {validation}\n", ""))
        packet = json.loads((output / "packet.json").read_bytes())
        self.assertEqual((packet["integrity"], packet["coverage"], packet["observed_validation"]),
                         ("VALID", status, validation))
        return output, packet

    def test_checked_in_greeting_produces_pinned_packet(self):
        bundle = Path(__file__).resolve().parents[1] / "examples/review-evidence/greeting/bundle.json"
        output = self.root / "greeting"
        self.assertEqual(self.cli(bundle, output), (0, "VALID / COMPLETE / PASS\n", ""))
        self.assertEqual(
            hashlib.sha256((output / "packet.json").read_bytes()).hexdigest(),
            "92c1e896d9ab6b38831709867bdf87f8c5623fa432b73d8e95e0423039ec3019",
        )

    def test_supported_change_retains_exact_sources_and_resolves_locators(self):
        self.fixture.receipt()
        self.fixture.intent["expected_steps"] = [{"id": "inspect", "text": "Inspect greeting.", "required": True}]
        self.fixture.summary()
        bait = "[text](https://example.invalid/) <b>plain text</b>\n## fake heading\u202e"
        self.fixture.intent["summary"] = bait
        (self.fixture.root / "unlisted.txt").write_text("not admitted", encoding="utf-8")
        output, packet = self.packet("output", 0, "PASS")
        self.assertEqual(packet["schema"], "review-evidence-packet/v0")
        self.assertEqual(packet["diff"], {"binding": "DECLARED", "source": {"source_id": "change", "lines": [1, 7]}})
        self.assertEqual(packet["criteria"][0]["status"], "COVERED")
        self.assertEqual(packet["criteria"][0]["check_source_ids"], ["unit"])
        self.assertEqual(packet["steps"][0]["status"], "OBSERVED")
        self.assertEqual(packet["steps"][0]["observation_source"], {"source_id": "summary", "pointer": "/observations/0"})
        self.assertEqual(packet["association"]["matching_summary"], {"source_id": "summary", "pointer": ""})
        self.assertEqual(packet["attempt_inventory"]["aggregate_cost"]["completeness"], "UNKNOWN")
        self.assertIsNone(packet["attempt_inventory"]["aggregate_cost"]["amount_minor"])
        self.assertEqual(packet["gaps"], [])
        self.assertEqual({f.name for f in (output / "sources").iterdir()},
                         {"change.txt", "unit.txt", "intent.txt", "summary.txt"})
        for source in packet["sources"]:
            data = (output / source["retained_path"]).read_bytes()
            self.assertEqual(data, (self.fixture.root / source["path"]).read_bytes())
            self.assertEqual(hashlib.sha256(data).hexdigest(), source["sha256"])
        review = (output / "review.md").read_text(encoding="utf-8")
        self.assertIn("    [text](https://example.invalid/) <b>plain text</b>\\n## fake heading\\u202e", review)
        self.assertNotIn("\n## fake heading", review)
        self.assertIn("intent#/criteria/0", review)

    def test_irrelevant_pass_stays_visible_without_coverage(self):
        self.fixture.receipt("formatting", relevant=False)
        _, packet = self.packet("irrelevant", 2, "PASS")
        self.assertEqual(packet["criteria"][0]["status"], "MISSING")
        self.assertEqual(packet["criteria"][0]["check_source_ids"], [])
        self.assertEqual(packet["checks"][0]["reported_result"], "PASS")
        self.assertFalse(packet["checks"][0]["coverage_eligible"])
        self.assertEqual([gap["code"] for gap in packet["gaps"]], ["CRITERION_UNCOVERED"])

    def test_required_unobserved_step_with_partial_capture_is_not_skipped(self):
        self.fixture.receipt()
        self.fixture.summary(observed=False)
        self.fixture.attempts([{"id": "a1", "billing": {"completeness": "PARTIAL"}}])
        self.fixture.intent["expected_steps"] = [
            {"id": "inspect", "text": "Inspect greeting.", "required": True},
            {"id": "refactor", "text": "Optional refactor.", "required": False},
        ]
        output, packet = self.packet("partial", 2, "PASS")
        self.assertEqual(packet["capture"]["completeness"], "PARTIAL")
        billing = packet["attempt_inventory"]["attempts"][0]["billing"]
        self.assertIsNone(billing["amount_minor"])
        self.assertIsNone(billing["currency"])
        self.assertEqual([step["status"] for step in packet["steps"]], ["UNOBSERVED", "UNOBSERVED"])
        self.assertEqual(packet["gaps"], [{
            "code": "STEP_UNOBSERVED", "subject_id": "inspect", "sources": [
                {"source_id": "intent", "pointer": "/expected_steps/0"},
                {"source_id": "summary", "pointer": "/capture"},
            ],
        }])
        self.assertIn("not proof the step was skipped", (output / "review.md").read_text(encoding="utf-8"))

    def test_cheaper_failed_attempt_can_have_complete_coverage(self):
        self.fixture.receipt(result="FAIL")
        self.fixture.summary()
        rows = [{
            "id": "a1", "reported_outcome": "FAIL",
            "billing": {"completeness": "COMPLETE", "currency": "USD", "amount_minor": 25},
        }, {
            "id": "a2", "reported_outcome": "PASS",
            "billing": {"completeness": "COMPLETE", "currency": "USD", "amount_minor": 80},
        }, {"id": "a3"}]
        self.fixture.attempts(rows)
        output, failed = self.packet("failed", 0, "FAIL")
        inventory = failed["attempt_inventory"]
        self.assertEqual([row["id"] for row in inventory["attempts"]], ["a1", "a2", "a3"])
        cheap, expensive, unbilled = inventory["attempts"]
        self.assertEqual((cheap["reported_outcome"], expensive["reported_outcome"]), ("FAIL", "PASS"))
        self.assertLess(cheap["billing"]["amount_minor"], expensive["billing"]["amount_minor"])
        self.assertIsNone(unbilled["billing"]["amount_minor"])
        total = inventory["aggregate_cost"]
        self.assertEqual(total, {
            "completeness": "UNKNOWN", "amount_minor": None, "currency": None, "reason": "BILLING_NOT_COMPLETE",
        })
        self.assertTrue(failed["checks"][0]["coverage_eligible"])
        review = (output / "review.md").read_text(encoding="utf-8")
        self.assertIn("Coverage is not acceptance", review)
        self.assertIn("a1: reported outcome FAIL (attempts#/attempts/0); billing 25 minor units USD", review)
        self.assertIn("Aggregate cost: unknown (BILLING_NOT_COMPLETE)", review)
        rows.pop()
        self.fixture.attempts(rows)
        _, billed = self.packet("billed", 0, "FAIL")
        self.assertEqual(billed["attempt_inventory"]["aggregate_cost"], {
            "completeness": "COMPLETE", "amount_minor": 105, "currency": "USD", "reason": None,
        })
        self.fixture.receipt("formatting", relevant=False)
        self.packet("mixed", 0, "MIXED")

    def test_aggregate_cost_stays_unknown_for_partial_roster_and_mixed_currencies(self):
        rows = [
            {"id": "a1", "billing": {"completeness": "COMPLETE", "currency": "USD", "amount_minor": 0}},
            {"id": "a2", "billing": {"completeness": "COMPLETE", "currency": "CAD", "amount_minor": 10}},
        ]
        self.fixture.attempts(rows, "PARTIAL")
        _, partial = self.packet("partial-roster", 2, "UNKNOWN")
        self.assertEqual(partial["attempt_inventory"]["aggregate_cost"]["reason"], "INVENTORY_NOT_COMPLETE")
        self.fixture.attempts(rows)
        _, mixed = self.packet("mixed-currencies", 2, "UNKNOWN")
        self.assertEqual(mixed["attempt_inventory"]["aggregate_cost"]["reason"], "MIXED_CURRENCY")
        self.assertEqual(mixed["attempt_inventory"]["attempts"][0]["billing"]["amount_minor"], 0)

    def test_stale_and_missing_receipts_do_not_cover_or_count_as_current(self):
        self.fixture.intent["criteria"].append({"id": "scope", "text": "No unrelated changes."})
        self.fixture.receipt("previous", check_id="unit", head="c" * 40)
        self.fixture.manifest["association"] = {"by": "operator", "session_id": "declared-only"}
        output, packet = self.packet("stale", 2, "UNKNOWN")
        self.assertEqual(packet["checks"][0]["binding"], "STALE")
        self.assertEqual(packet["checks"][0]["reported_result"], "PASS")
        self.assertFalse(packet["checks"][0]["coverage_eligible"])
        self.assertEqual(packet["criteria"][0]["stale_check_source_ids"], ["previous"])
        self.assertEqual(packet["criteria"][1]["stale_check_source_ids"], [])
        self.assertIsNone(packet["association"]["matching_summary"])
        self.assertIn("reported PASS by fixture CI; STALE", (output / "review.md").read_text(encoding="utf-8"))
        self.fixture.receipt()
        _, current = self.packet("later", 2, "PASS")
        self.assertEqual([row["status"] for row in current["criteria"]], ["COVERED", "MISSING"])
        self.fixture.intent["criteria"].pop()
        self.packet("superseded", 0, "PASS")

    def invalid(self, path: Path, error: str):
        output = self.root / "invalid"
        status, stdout, stderr = self.cli(path, output)
        self.assertEqual((status, stdout), (1, ""))
        self.assertIn(error, stderr)
        self.assertFalse(output.exists())

    def test_usage_errors_are_not_inconclusive_packets(self):
        status, stdout, stderr = self.run_cli("review-evidence", "bundle.json")
        self.assertEqual((status, stdout), (1, ""))
        self.assertIn("--output", stderr)

    def test_hash_mismatch_and_malformed_valid_hash_records(self):
        path = self.fixture.write()
        diff = self.fixture.root / "change.input"
        diff.write_bytes(DIFF.replace(b"Hello", b"Howdy"))
        self.invalid(path, "SHA-256 mismatch")
        self.fixture.add("change", evidence.DIFF_TYPE, DIFF.replace(b"@@ -1 +1 @@", b"@@ -2,2 +1 @@"))
        self.invalid(self.fixture.write(), "diff hunk body")
        self.fixture.add("change", evidence.DIFF_TYPE, DIFF)
        self.fixture.add("unit", "validation-receipt", b'{"schema":"duplicate","schema":"same key"}')
        self.invalid(self.fixture.write(), "duplicate JSON object key")

    def test_strict_source_path_locator_version_and_admission(self):
        for field, values in {
            "path": ["/absolute", "../escape", "a/../b", "a//b", "a\\b", "a:b", "bundle.json"],
            "lines": [[0, 1], [1, 8], [5, 1], [True, 1], [1]],
            "version": [True, 1],
            "type": ["raw-trace"],
            "admission": ["UNREVIEWED"],
        }.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    change = self.fixture.add("change", evidence.DIFF_TYPE, DIFF)
                    change[field] = value
                    self.invalid(self.fixture.write(), "review-evidence:")

    def test_duplicate_sources_paths_checks_criteria_and_bindings(self):
        self.fixture.write()
        change = self.fixture.manifest["sources"][0]
        self.fixture.manifest["sources"].append(dict(change))
        self.invalid(self.fixture.write(), "duplicate or unsafe source ID")
        self.fixture.manifest["sources"][-2] = {**change, "id": "other"}
        self.invalid(self.fixture.write(), "duplicate source path")
        self.fixture.manifest["sources"] = [change]
        self.fixture.receipt("first", check_id="unit")
        self.fixture.receipt("second", check_id="unit")
        self.invalid(self.fixture.write(), "duplicate check ID for the same head")
        self.fixture.receipt("second", target={**TARGET, "pr": 8})
        self.invalid(self.fixture.write(), "does not match bundle target")
        self.fixture.intent["criteria"] *= 2
        self.invalid(self.fixture.write(), "duplicate criterion ID")
        self.fixture.intent["criteria"].pop()
        self.fixture.intent["criteria"][0]["id"] = "other"
        self.invalid(self.fixture.write(), "criterion mappings are inconsistent")

    def test_association_conflicts_and_incomplete_billing_total_are_invalid(self):
        self.fixture.summary(session_id="another-session")
        self.invalid(self.fixture.write(), "do not match the declared association")
        self.fixture.summary()
        del self.fixture.manifest["association"]
        self.invalid(self.fixture.write(), "requires an explicit association")
        self.fixture.summary()
        self.fixture.attempts([{"id": "a2"}])
        self.invalid(self.fixture.write(), "complete inventory does not include the declared attempt")
        self.fixture.attempts([{"id": "a1"}, {"id": "a1"}])
        self.invalid(self.fixture.write(), "duplicate attempt ID")
        self.fixture.attempts([{"id": "a1", "billing": {"completeness": "UNKNOWN", "currency": "USD", "amount_minor": 0}}])
        self.invalid(self.fixture.write(), "incomplete billing")

    def test_links_and_nonregular_input_are_refused_without_mutation(self):
        path = self.fixture.write()
        diff = self.fixture.root / "change.input"
        original = self.root / "original"
        diff.rename(original)
        try:
            diff.symlink_to(original)
        except OSError as exc:
            self.skipTest(f"symlinks unavailable: {exc}")
        self.invalid(path, "could not be opened safely")
        self.assertEqual(original.read_bytes(), DIFF)
        diff.unlink()
        os.link(original, diff)
        self.invalid(path, "multiply linked")
        diff.unlink()
        if hasattr(os, "mkfifo"):
            os.mkfifo(diff)
            self.invalid(path, "size or type is invalid")
        else:
            diff.mkdir()
            self.invalid(path, "review-evidence:")

    def test_linked_ancestors_existing_output_and_size_limits(self):
        path = self.fixture.write()
        alias = self.root / "alias"
        try:
            alias.symlink_to(path.parent, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"symlinks unavailable: {exc}")
        self.invalid(alias / path.name, "unsafe component")
        self.assertEqual(self.cli(path, alias / "new")[0], 1)
        self.assertFalse((path.parent / "new").exists())
        empty = self.root / "empty"
        empty.mkdir()
        self.assertEqual(self.cli(path, empty), (1, "", "review-evidence: output already exists\n"))
        self.assertEqual(list(empty.iterdir()), [])
        too_big = DIFF + b" " * evidence.MAX_DIFF_BYTES
        self.fixture.add("change", evidence.DIFF_TYPE, too_big)
        self.invalid(self.fixture.write(), "size or type is invalid")

    def test_hash_parse_and_copy_use_the_same_admitted_snapshot(self):
        path = self.fixture.write()
        real_read = evidence._read

        def change_after_read(source, label, bound):
            data = real_read(source, label, bound)
            if label == "change":
                source.write_bytes(b"replacement after the one admitted read")
            return data

        output = self.root / "snapshot"
        with patch.object(evidence, "_read", side_effect=change_after_read):
            self.assertEqual(self.cli(path, output)[0], 2)
        self.assertEqual((output / "sources" / "change.txt").read_bytes(), DIFF)

    def test_deterministic_output_no_clobber_and_interrupted_write(self):
        self.fixture.receipt()
        path = self.fixture.write()
        first, second, failed, interrupted = (self.root / item for item in ("first", "second", "failed", "interrupted"))
        self.assertEqual(self.cli(path, first)[0], 0)
        self.assertEqual(self.cli(path, second)[0], 0)
        files = [file.relative_to(first) for file in first.rglob("*") if file.is_file()]
        for file in files:
            self.assertEqual((first / file).read_bytes(), (second / file).read_bytes())
        self.assertEqual(self.cli(path, first)[0], 1)
        for file in files:
            self.assertEqual((first / file).read_bytes(), (second / file).read_bytes())
        with patch.object(evidence, "write_atomic", side_effect=OSError(5, "fixture write failed")):
            code, stdout, stderr = self.cli(path, failed)
        self.assertEqual((code, stdout), (1, ""))
        self.assertIn("fixture write failed", stderr)
        self.assertFalse(failed.exists())
        abrupt_exit = """\
import os, sys
from agent_trace import review_evidence as e
write_atomic = e.write_atomic
def interrupt(path, *args, **kwargs):
    if path.name == 'packet.json':
        os._exit(88)
    write_atomic(path, *args, **kwargs)
e.write_atomic = interrupt
e.assemble_review_evidence(*sys.argv[1:])
"""
        child = subprocess.run([sys.executable, "-c", abrupt_exit, str(path), str(interrupted)], timeout=10)
        self.assertEqual(child.returncode, 88)
        self.assertTrue((interrupted / "review.md").is_file())
        self.assertFalse((interrupted / "packet.json").exists())

    def test_failed_parent_sync_never_exposes_packet(self):
        input_path = self.fixture.write()
        output = self.root / "output"

        def fail_on_parent(directory):
            if directory == output.parent:
                self.assertFalse((output / "packet.json").exists())
                raise OSError(5, "fixture parent sync failed")

        with patch.object(evidence, "fsync_directory", side_effect=fail_on_parent):
            status, stdout, stderr = self.cli(input_path, output)
        self.assertEqual((status, stdout), (1, ""))
        self.assertIn("fixture parent sync failed", stderr)
        self.assertFalse(output.exists())
