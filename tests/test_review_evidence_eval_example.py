import hashlib
import json
import os
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests import _no_telemetry  # noqa: F401


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "review-evidence-eval"


class ReviewEvidenceEvalExampleTests(unittest.TestCase):
    def load_observations(self):
        return json.loads((EXAMPLE / "observations.json").read_text(encoding="utf-8"))

    def run_scorer(self, observations=None):
        argv = [sys.executable, str(EXAMPLE / "score.py")]
        if observations is not None:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "observations.json"
                path.write_text(json.dumps(observations), encoding="utf-8")
                return subprocess.run([*argv, str(path)], capture_output=True, text=True, timeout=10)
        return subprocess.run(argv, capture_output=True, text=True, timeout=10)

    def bundled_report(self):
        result = self.run_scorer()
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
        report = json.loads(result.stdout)
        self.assertEqual(report["regression_mismatches"], [])
        return report["cases"]

    def test_irrelevant_green_lint_does_not_rescue_wrong_outcome_or_missing_review(self):
        case = self.bundled_report()["wrong-summary-green-lint"]
        self.assertEqual(case["outcome"]["label"], "fail")
        self.assertEqual(case["workflow"]["skill:release-notes"]["label"], "pass")
        self.assertEqual(case["workflow"]["review:fix-summary"], {
            "expectation": "required", "evidence": "absent", "label": "fail",
        })

    def test_partial_capture_keeps_absence_unknown_without_hiding_positive_evidence(self):
        cases = self.bundled_report()
        partial = cases["summary-partial-capture"]
        self.assertEqual(partial["outcome"]["label"], "pass")
        self.assertEqual(partial["workflow"]["skill:release-notes"], {
            "expectation": "required", "evidence": "unknown", "label": "unknown",
        })
        self.assertEqual(partial["workflow"]["review:fix-summary"]["label"], "pass")
        negative = cases["count-partial-negative"]
        self.assertEqual(negative["outcome"]["label"], "unknown")
        self.assertEqual(negative["workflow"]["skill:release-notes"]["label"], "unknown")

    def test_should_not_trigger_distinguishes_wrong_activation_from_supported_absence(self):
        cases = self.bundled_report()
        self.assertEqual(cases["count-wrong-activation"]["workflow"]["skill:release-notes"], {
            "expectation": "forbidden", "evidence": "observed", "label": "fail",
        })
        self.assertEqual(cases["count-clean-negative"]["workflow"]["skill:release-notes"], {
            "expectation": "forbidden", "evidence": "absent", "label": "pass",
        })

    def test_wrong_candidate_artifact_regresses_even_with_all_workflow_steps_observed(self):
        observations = self.load_observations()
        observations["reviewed-summary"]["artifact"]["fix_ids"] = ["FIX-12", "DOC-04"]
        result = self.run_scorer(observations)
        self.assertEqual(result.returncode, 1, result.stderr or result.stdout)
        report = json.loads(result.stdout)
        self.assertEqual(report["regression_mismatches"], [{
            "case": "reviewed-summary", "dimension": "outcome", "step": None,
            "expected": "pass", "actual": "fail",
        }])
        case = report["cases"]["reviewed-summary"]
        self.assertEqual(case["outcome"]["label"], "fail")
        self.assertEqual(
            [step["label"] for step in case["workflow"].values()], ["pass", "pass"],
        )

    def test_claimed_pass_without_artifact_is_still_unknown(self):
        observations = self.load_observations()
        observations["reviewed-summary"].update(artifact=None, agent_claim="PASS")
        result = self.run_scorer(observations)
        self.assertEqual(result.returncode, 1, result.stderr or result.stdout)
        case = json.loads(result.stdout)["cases"]["reviewed-summary"]
        self.assertEqual(case["outcome"]["label"], "unknown")
        self.assertEqual(case["workflow"]["review:fix-summary"]["label"], "pass")

    def test_dropping_capture_coverage_regresses_clean_negative_to_unknown(self):
        observations = self.load_observations()
        observations["count-clean-negative"]["sufficient_capture_for"] = []
        result = self.run_scorer(observations)
        self.assertEqual(result.returncode, 1, result.stderr or result.stdout)
        report = json.loads(result.stdout)
        self.assertEqual(report["regression_mismatches"], [{
            "case": "count-clean-negative", "dimension": "workflow", "step": "skill:release-notes",
            "expected": "pass", "actual": "unknown",
        }])
        self.assertEqual(report["cases"]["count-clean-negative"]["outcome"]["label"], "pass")

    def test_missing_observation_is_invalid_not_a_clean_negative(self):
        observations = self.load_observations()
        del observations["count-clean-negative"]
        result = self.run_scorer(observations)
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("missing=['count-clean-negative']", result.stderr)


class ReviewEvidencePacketExampleTests(unittest.TestCase):
    fixture = EXAMPLE / "packet-fixture"

    def load_input(self, name):
        return json.loads((self.fixture / f"{name}.json").read_text(encoding="utf-8"))

    def run_packet(self, *, evidence=None, binding=None, packet=None, expected_exit=0):
        with tempfile.TemporaryDirectory() as directory:
            files = []
            for name, data in (("packet", packet), ("binding", binding), ("evidence", evidence)):
                path = self.fixture / f"{name}.json"
                if data is not None:
                    path = Path(directory) / path.name
                    path.write_bytes(data if isinstance(data, bytes) else json.dumps(data).encode("utf-8"))
                files.append(str(path))
            result = subprocess.run(
                [sys.executable, "-B", str(EXAMPLE / "score_packet.py"), *files],
                capture_output=True, text=True, timeout=10,
            )
        self.assertEqual(result.returncode, expected_exit, result.stderr or result.stdout)
        if expected_exit == 2:
            self.assertFalse(result.stdout)
            return result.stderr
        return json.loads(result.stdout)

    def test_produced_packet_with_independent_artifact_and_corroborated_inspection(self):
        report = self.run_packet()
        case = report["cases"]["packet-greeting"]
        self.assertEqual(report["regression_mismatches"], [])
        self.assertEqual(case["outcome"]["label"], "pass")
        self.assertEqual(case["workflow"]["review:greeting"]["evidence"], "observed")
        self.assertEqual(case["workflow"]["skill:release-notes"]["label"], "unknown")

    def test_current_pass_does_not_rescue_missing_or_wrong_outcome(self):
        missing = self.load_input("missing-evidence")
        wrong = self.load_input("evidence")
        wrong["artifact"] = {"greeting": "Hi"}
        for evidence, outcome, inspection in ((missing, "unknown", "unknown"), (wrong, "fail", "pass")):
            with self.subTest(outcome=outcome):
                case = self.run_packet(evidence=evidence, expected_exit=1)["cases"]["packet-greeting"]
                self.assertEqual(case["outcome"]["label"], outcome)
                self.assertEqual(case["workflow"]["review:greeting"]["label"], inspection)

    def test_wrong_locator_or_untyped_evidence_leaves_inspection_unknown(self):
        for delta in (
            {"corroborates": {"source_id": "summary", "pointer": "/observations/1"}},
            {"kind": None},
            {"kind": "skill"},
        ):
            with self.subTest(delta=delta):
                evidence = self.load_input("evidence")
                evidence["inspection"].update(delta)
                case = self.run_packet(evidence=evidence, expected_exit=1)["cases"]["packet-greeting"]
                self.assertEqual(case["outcome"]["label"], "pass")
                self.assertEqual(case["workflow"]["review:greeting"]["label"], "unknown")
                self.assertEqual(case["workflow"]["skill:release-notes"]["label"], "unknown")

    def test_negative_needs_explicit_absence_and_sufficient_step_capture(self):
        # An in-memory variant tests the negative path; the producer fixture remains untouched.
        packet = self.load_input("packet")
        packet["steps"][0].update(status="UNOBSERVED", observation_source=None)
        packet["observations"] = []
        packet["capture"]["completeness"] = "COMPLETE"
        raw = json.dumps(packet).encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()
        binding = self.load_input("binding")
        binding["packet_sha256"] = digest
        binding["row"]["observation_source"] = None
        evidence = self.load_input("evidence")
        evidence["packet_sha256"] = digest
        for state, capture, expected in (
            (None, "sufficient", "unknown"),
            ("absent", "partial", "unknown"),
            ("absent", "sufficient", "absent"),
        ):
            with self.subTest(state=state, capture=capture):
                evidence["inspection"] = {
                    "kind": "greeting-inspection", "state": state,
                    "source": "fixture-observer:inspection-window", "capture": capture,
                }
                report = self.run_packet(packet=raw, binding=binding, evidence=evidence, expected_exit=1)
                review = report["cases"]["packet-greeting"]["workflow"]["review:greeting"]
                self.assertEqual(review["evidence"], expected)
                self.assertEqual(review["label"], "fail" if expected == "absent" else "unknown")

    def test_rejects_changed_bytes_and_target_or_locator_binding_mismatches(self):
        packet = (self.fixture / "packet.json").read_bytes()
        self.assertIn("SHA-256", self.run_packet(packet=packet + b"\n", expected_exit=2))
        for field, key, replacement in (
            ("target", "head", "c" * 40),
            ("row", "id", "unrelated"),
            ("row", "observation_source", {"source_id": "summary", "pointer": "/observations/1"}),
        ):
            with self.subTest(field=field, key=key):
                binding = self.load_input("binding")
                binding[field][key] = replacement
                self.assertIn("does not match the supplied binding", self.run_packet(
                    binding=binding, expected_exit=2,
                ))

    def test_missing_bound_row_is_rejected_and_evidence_cannot_cross_packets(self):
        packet = self.load_input("packet")
        packet["steps"] = []
        raw = json.dumps(packet).encode("utf-8")
        binding = self.load_input("binding")
        binding["packet_sha256"] = hashlib.sha256(raw).hexdigest()
        self.assertIn("inspect row or locator", self.run_packet(packet=raw, binding=binding, expected_exit=2))
        evidence = self.load_input("evidence")
        evidence["packet_sha256"] = "0" * 64
        self.assertIn("reference the bound packet", self.run_packet(evidence=evidence, expected_exit=2))


class ReviewerExerciseTests(unittest.TestCase):
    def test_evidence_parity_rejects_a_changed_arm_or_packet_copy(self):
        with patch.object(sys, "path", [str(EXAMPLE), *sys.path]):
            verify_parity = runpy.run_path(str(EXAMPLE / "build_reviewer_exercise.py"))["verify_parity"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline, assisted, packet = (root / name for name in ("baseline", "assisted", "packet"))
            for folder in (baseline, assisted, packet):
                folder.mkdir()
            manifest = b'{"sources":[]}\n'
            evidence = b"identical retained evidence\n"
            for arm in (baseline, assisted):
                (arm / "bundle.json").write_bytes(manifest)
                (arm / "source.input").write_bytes(evidence)
            retained = packet / "source.txt"
            retained.write_bytes(evidence)
            (packet / "packet.json").write_text(json.dumps({
                "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
                "sources": [{
                    "path": "source.input", "retained_path": retained.name,
                    "sha256": hashlib.sha256(evidence).hexdigest(),
                }],
            }), encoding="utf-8")
            verify_parity(baseline, assisted, packet)
            retained.write_bytes(b"changed retained bytes\n")
            with self.assertRaisesRegex(ValueError, "retained bytes differ"):
                verify_parity(baseline, assisted, packet)
            retained.write_bytes(evidence)
            (assisted / "source.input").write_bytes(b"changed arm\n")
            with self.assertRaisesRegex(ValueError, "arms differ"):
                verify_parity(baseline, assisted, packet)

    @unittest.skipUnless(os.environ.get("REVIEW_EVIDENCE_PRODUCER"), "requires a local Phase A checkout")
    def test_both_views_replay_the_same_evidence_and_footer_pass_has_gaps(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "exercise"
            result = subprocess.run(
                [sys.executable, "-B", str(EXAMPLE / "build_reviewer_exercise.py"),
                 "--producer", os.environ["REVIEW_EVIDENCE_PRODUCER"], "--output", str(output)],
                text=True, capture_output=True, timeout=45,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["cases"]["greeting"]["producer_status"], "VALID / COMPLETE / PASS")
            self.assertEqual(report["cases"]["footer"]["producer_status"], "VALID / INCONCLUSIVE / PASS")
            self.assertEqual(report["cases"]["footer"]["producer_exit"], 2)
            for case in ("greeting", "footer"):
                plain, assisted = (output / case / name for name in ("baseline", "packet-assisted"))
                self.assertEqual({path.name for path in plain.iterdir()}, {"worksheet.md", "evidence"})
                self.assertEqual(
                    {path.name for path in assisted.iterdir()}, {"worksheet.md", "evidence", "packet"},
                )
                self.assertEqual((plain / "worksheet.md").read_bytes(), (assisted / "worksheet.md").read_bytes())
                for name in report["cases"][case]["identical_evidence_sha256"]:
                    self.assertEqual((plain / "evidence" / name).read_bytes(), (assisted / "evidence" / name).read_bytes())
            footer = json.loads((output / "footer/packet-assisted/packet/packet.json").read_bytes())
            self.assertEqual(
                {gap["code"] for gap in footer["gaps"]}, {"CRITERION_UNCOVERED", "STEP_UNOBSERVED"},
            )
            self.assertEqual(footer["steps"][0]["status"], "UNOBSERVED")
            self.assertEqual(footer["capture"]["completeness"], "PARTIAL")
            self.assertFalse(footer["checks"][0]["coverage_eligible"])
