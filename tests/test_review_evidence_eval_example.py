import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


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
