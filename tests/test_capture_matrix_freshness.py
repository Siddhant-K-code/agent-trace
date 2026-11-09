"""Tests for scripts/check_capture_matrix_freshness.py.

Every check date is fixed so the suite never depends on the current date.
"""

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_capture_matrix_freshness.py"
MATRIX_PATH = ROOT / "src" / "agent_trace" / "capture_matrix.json"


def _load_script():
    spec = importlib.util.spec_from_file_location("check_capture_matrix_freshness", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


freshness = _load_script()

TODAY = date(2026, 11, 9)


def verified(provider="copilot", last_verified="2026-10-22", tested_version="1.0.103-0"):
    return {
        "id": provider,
        "name": provider.title(),
        "verification_status": "verified",
        "tested_version": tested_version,
        "last_verified": last_verified,
    }


def unverified(provider="claude"):
    return {
        "id": provider,
        "name": provider.title(),
        "verification_status": "unverified",
        "tested_version": None,
        "last_verified": None,
    }


class TestAssessEntry(unittest.TestCase):
    def assess(self, entry, today=TODAY, max_age_days=30):
        return freshness.assess_entry(entry, today, max_age_days)

    def test_recent_verification_is_fresh(self):
        result = self.assess(verified(last_verified="2026-10-22"))

        self.assertEqual(result.status, "fresh")
        self.assertEqual(result.age_days, 18)
        self.assertEqual(result.stale_from, "2026-11-22")
        self.assertIn("1.0.103-0", result.message)
        self.assertIn("18 days ago", result.message)

    def test_verified_today_is_fresh_with_zero_age(self):
        result = self.assess(verified(last_verified="2026-11-09"))

        self.assertEqual(result.status, "fresh")
        self.assertEqual(result.age_days, 0)

    def test_age_equal_to_threshold_is_still_fresh(self):
        result = self.assess(verified(last_verified="2026-10-10"))

        self.assertEqual(result.age_days, 30)
        self.assertEqual(result.status, "fresh")

    def test_age_one_day_over_threshold_is_stale(self):
        result = self.assess(verified(last_verified="2026-10-09"))

        self.assertEqual(result.status, "stale")
        self.assertEqual(result.age_days, 31)
        self.assertEqual(result.stale_from, "2026-11-09")

    def test_stale_message_names_age_and_tested_version(self):
        result = self.assess(
            verified(last_verified="2026-08-01", tested_version="0.163.0-alpha.11"),
        )

        self.assertEqual(result.status, "stale")
        self.assertEqual(result.age_days, 100)
        self.assertEqual(result.tested_version, "0.163.0-alpha.11")
        self.assertIn("0.163.0-alpha.11", result.message)
        self.assertIn("100 days ago", result.message)
        self.assertIn("more than 30 days", result.message)

    def test_threshold_is_injectable(self):
        entry = verified(last_verified="2026-10-22")

        self.assertEqual(self.assess(entry, max_age_days=17).status, "stale")
        self.assertEqual(self.assess(entry, max_age_days=18).status, "fresh")
        self.assertEqual(self.assess(entry, max_age_days=0).status, "stale")

    def test_today_is_injectable(self):
        entry = verified(last_verified="2026-10-22")

        self.assertEqual(self.assess(entry, today=date(2026, 11, 21)).status, "fresh")
        self.assertEqual(self.assess(entry, today=date(2026, 11, 22)).status, "stale")

    def test_unverified_entry_is_informational(self):
        result = self.assess(unverified())

        self.assertEqual(result.status, "unverified")
        self.assertIsNone(result.age_days)
        self.assertIn("informational", result.message)

    def test_unverified_entry_ignores_its_date_fields(self):
        entry = unverified()
        entry["last_verified"] = "2020-01-01"

        self.assertEqual(self.assess(entry).status, "unverified")

    def test_malformed_last_verified_values(self):
        for value in (
            None,
            "",
            20261022,
            "20261022",
            "2026/10/22",
            "2026-10-32",
            "2026-02-30",
            "2026-1-5",
            " 2026-10-22",
            "2026-10-22T00:00:00",
        ):
            with self.subTest(value=value):
                result = self.assess(verified(last_verified=value))
                self.assertEqual(result.status, "malformed")
                self.assertIn("YYYY-MM-DD", result.message)

    def test_missing_last_verified_is_malformed(self):
        entry = verified()
        del entry["last_verified"]

        self.assertEqual(self.assess(entry).status, "malformed")

    def test_one_day_in_the_future_is_fresh_with_zero_age(self):
        result = self.assess(verified(last_verified="2026-11-10"))

        self.assertEqual(result.status, "fresh")
        self.assertEqual(result.age_days, 0)

    def test_more_than_one_day_in_the_future_is_malformed(self):
        result = self.assess(verified(last_verified="2026-11-11"))

        self.assertEqual(result.status, "malformed")
        self.assertIn("after the check date 2026-11-09", result.message)

    def test_unknown_verification_status_is_malformed(self):
        entry = verified()
        entry["verification_status"] = "partially_verified"

        result = self.assess(entry)

        self.assertEqual(result.status, "malformed")
        self.assertIn("partially_verified", result.message)


class TestAssessMatrix(unittest.TestCase):
    def test_rejects_matrix_without_provider_list(self):
        for matrix in ([], {}, {"providers": {}}, {"providers": ["codex"]}):
            with self.subTest(matrix=matrix):
                with self.assertRaises(ValueError):
                    freshness.assess_matrix(matrix, TODAY, 30)

    def test_packaged_matrix_dates_are_well_formed(self):
        # The check date comes from the matrix itself, not the clock.
        matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
        dates = [
            freshness.parse_iso_date(entry.get("last_verified"))
            for entry in matrix["providers"]
            if entry.get("verification_status") == "verified"
        ]
        self.assertTrue(dates)
        self.assertNotIn(None, dates)

        results = freshness.assess_matrix(matrix, max(dates), freshness.DEFAULT_MAX_AGE_DAYS)

        self.assertEqual(
            [r.provider for r in results],
            [entry["id"] for entry in matrix["providers"]],
        )
        for entry, result in zip(matrix["providers"], results):
            with self.subTest(provider=entry["id"]):
                expected = (
                    {"fresh", "stale"}
                    if entry["verification_status"] == "verified"
                    else {"unverified"}
                )
                self.assertIn(result.status, expected)


class TestMain(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def write_matrix(self, *providers, raw=None):
        path = self.tmp / "capture_matrix.json"
        if raw is None:
            raw = json.dumps({"schema_version": 1, "providers": list(providers)})
        path.write_text(raw, encoding="utf-8")
        return path

    def run_main(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        code = freshness.main(list(args), stdout=stdout, stderr=stderr)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_report_without_check_exits_zero_even_when_stale(self):
        path = self.write_matrix(verified(last_verified="2026-08-01"), unverified())

        code, out, _ = self.run_main("--matrix", str(path), "--today", "2026-11-09")

        self.assertEqual(code, 0)
        self.assertIn("copilot  stale", out)
        self.assertIn("claude   unverified", out)
        self.assertIn("Summary: 0 fresh, 1 stale, 1 unverified, 0 malformed", out)

    def test_check_fails_on_stale_entry(self):
        path = self.write_matrix(verified(last_verified="2026-08-01"))

        code, out, _ = self.run_main(
            "--matrix", str(path), "--today", "2026-11-09", "--check"
        )

        self.assertEqual(code, 1)
        self.assertIn("100 days ago", out)

    def test_check_fails_on_malformed_date(self):
        path = self.write_matrix(verified(last_verified="2026/10/22"))

        code, out, _ = self.run_main(
            "--matrix", str(path), "--today", "2026-11-09", "--check"
        )

        self.assertEqual(code, 1)
        self.assertIn("malformed", out)

    def test_check_passes_with_fresh_and_unverified_entries(self):
        path = self.write_matrix(
            unverified(),
            verified("codex", "2026-11-09", "0.167.0-alpha.3.2"),
            verified("copilot", "2026-10-22", "1.0.103-0"),
        )

        code, out, _ = self.run_main(
            "--matrix", str(path), "--today", "2026-11-09", "--check"
        )

        self.assertEqual(code, 0)
        self.assertEqual(
            out,
            "Capture matrix freshness on 2026-11-09 "
            "(verified entries are stale after 30 days)\n"
            "claude   unverified  no verified release recorded (informational)\n"
            "codex    fresh       0.167.0-alpha.3.2, last verified 2026-11-09 "
            "(0 days ago; stale from 2026-12-10)\n"
            "copilot  fresh       1.0.103-0, last verified 2026-10-22 "
            "(18 days ago; stale from 2026-11-22)\n"
            "Summary: 2 fresh, 0 stale, 1 unverified, 0 malformed\n",
        )

    def test_check_passes_with_only_unverified_entries(self):
        path = self.write_matrix(unverified("claude"), unverified("gemini"))

        code, _, _ = self.run_main(
            "--matrix", str(path), "--today", "2026-11-09", "--check"
        )

        self.assertEqual(code, 0)

    def test_max_age_days_option(self):
        path = self.write_matrix(verified(last_verified="2026-10-22"))

        code, out, _ = self.run_main(
            "--matrix", str(path), "--today", "2026-11-09",
            "--max-age-days", "7", "--check",
        )

        self.assertEqual(code, 1)
        self.assertIn("stale after 7 days", out)

    def test_json_output(self):
        path = self.write_matrix(
            unverified(),
            verified("codex", "2026-11-09", "0.167.0-alpha.3.2"),
            verified("copilot", "2026-08-01", "1.0.94-3"),
        )

        code, out, _ = self.run_main(
            "--matrix", str(path), "--today", "2026-11-09", "--format", "json"
        )

        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(payload["today"], "2026-11-09")
        self.assertEqual(payload["max_age_days"], 30)
        self.assertEqual(payload["matrix"], str(path))
        self.assertTrue(payload["failing"])
        self.assertEqual(
            payload["counts"],
            {"fresh": 1, "stale": 1, "unverified": 1, "malformed": 0},
        )
        stale = payload["entries"][2]
        self.assertEqual(
            {key: stale[key] for key in (
                "provider", "status", "verification_status", "tested_version",
                "last_verified", "age_days", "stale_from",
            )},
            {
                "provider": "copilot",
                "status": "stale",
                "verification_status": "verified",
                "tested_version": "1.0.94-3",
                "last_verified": "2026-08-01",
                "age_days": 100,
                "stale_from": "2026-09-01",
            },
        )
        self.assertIsNone(payload["entries"][0]["age_days"])

    def test_invalid_arguments_exit_two(self):
        path = self.write_matrix(verified())
        for args in (
            ("--today", "2026-13-01"),
            ("--today", "20261109"),
            ("--max-age-days", "-1"),
            ("--max-age-days", "thirty"),
            ("--format", "yaml"),
        ):
            with self.subTest(args=args):
                with mock.patch("sys.stderr", io.StringIO()):
                    code, _, _ = self.run_main("--matrix", str(path), *args)
                self.assertEqual(code, 2)

    def test_unreadable_matrix_exits_two_even_without_check(self):
        cases = {
            "missing": self.tmp / "missing.json",
            "invalid json": self.write_matrix(raw="{"),
        }
        for label, path in cases.items():
            with self.subTest(case=label):
                code, out, err = self.run_main(
                    "--matrix", str(path), "--today", "2026-11-09"
                )
                self.assertEqual(code, 2)
                self.assertEqual(out, "")
                self.assertIn("cannot read capture matrix", err)

    def test_matrix_without_providers_list_exits_two(self):
        path = self.write_matrix(raw=json.dumps({"providers": "codex"}))

        code, _, err = self.run_main("--matrix", str(path), "--today", "2026-11-09")

        self.assertEqual(code, 2)
        self.assertIn("providers list", err)

    def test_default_matrix_is_the_packaged_source(self):
        self.assertEqual(freshness.DEFAULT_MATRIX, MATRIX_PATH)


class TestFreshnessDocs(unittest.TestCase):
    def test_capture_matrix_docs_state_threshold_and_command(self):
        docs = (ROOT / "docs" / "capture-matrix.md").read_text(encoding="utf-8")

        self.assertIn("## Freshness", docs)
        self.assertIn(f"more than {freshness.DEFAULT_MAX_AGE_DAYS} days", docs)
        self.assertIn("python scripts/check_capture_matrix_freshness.py", docs)
        self.assertIn("capture-matrix-freshness.yml", docs)

    def test_release_runbook_includes_freshness_review(self):
        runbook = (ROOT / "docs" / "releases.md").read_text(encoding="utf-8")

        self.assertIn("python scripts/check_capture_matrix_freshness.py", runbook)


if __name__ == "__main__":
    unittest.main()
