import hashlib
import json
import unittest
from pathlib import Path

from agent_trace.health import load_capture_matrix


ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "src" / "agent_trace" / "capture_matrix.json"
DOC_PATH = ROOT / "docs" / "capture-matrix.md"
REQUIRED_COVERAGE = {
    "session_start",
    "session_end",
    "prompt",
    "response",
    "tool",
    "file",
    "command",
    "error",
    "stop",
}
ALLOWED_COVERAGE = {"captured", "provider_defined", "unavailable"}
ALLOWED_VERIFICATION_STATUS = {"verified", "unverified"}


class TestCaptureMatrix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
        cls.docs = DOC_PATH.read_text(encoding="utf-8")

    def test_provider_ids_are_unique(self):
        providers = self.matrix["providers"]
        ids = [provider["id"] for provider in providers]
        self.assertEqual(len(ids), len(set(ids)))

    def test_entries_have_required_contract_fields(self):
        for provider in self.matrix["providers"]:
            with self.subTest(provider=provider["id"]):
                for key in (
                    "name",
                    "capture_method",
                    "setup_command",
                    "config_location",
                    "rollback",
                    "verification_status",
                    "coverage",
                    "privacy",
                    "known_gaps",
                ):
                    self.assertIn(key, provider)
                self.assertEqual(set(provider["coverage"]), REQUIRED_COVERAGE)
                self.assertTrue(set(provider["coverage"].values()) <= ALLOWED_COVERAGE)
                self.assertIn(provider["verification_status"], ALLOWED_VERIFICATION_STATUS)

    def test_verified_entries_require_reproducibility_metadata(self):
        for provider in self.matrix["providers"]:
            if provider["verification_status"] != "verified":
                continue
            with self.subTest(provider=provider["id"]):
                self.assertTrue(provider["tested_version"])
                self.assertTrue(provider["last_verified"])
                self.assertTrue(provider["fixture_test_reference"])

    def test_human_coverage_table_matches_json(self):
        rows = {}
        for line in self.docs.splitlines():
            if not line.startswith("| "):
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) != len(REQUIRED_COVERAGE) + 2:
                continue
            rows[cells[0]] = cells

        fields = [
            "session_start",
            "session_end",
            "prompt",
            "response",
            "tool",
            "file",
            "command",
            "error",
            "stop",
        ]
        header = ["Provider", *fields, "verification_status"]
        self.assertEqual(rows.pop("Provider", None), header)
        self.assertEqual(set(rows), {p["id"] for p in self.matrix["providers"]})
        for provider in self.matrix["providers"]:
            with self.subTest(provider=provider["id"]):
                self.assertEqual(
                    rows[provider["id"]],
                    [provider["id"], *(provider["coverage"][key] for key in fields),
                     provider["verification_status"]],
                )

    def test_unverified_entries_may_omit_reproducibility_metadata(self):
        for provider in self.matrix["providers"]:
            with self.subTest(provider=provider["id"]):
                if provider["verification_status"] == "unverified":
                    self.assertIsNotNone(provider.get("coverage"))

    def test_human_docs_cover_each_machine_readable_provider(self):
        for provider in self.matrix["providers"]:
            with self.subTest(provider=provider["id"]):
                self.assertIn(provider["name"], self.docs)
                self.assertIn(provider["setup_command"], self.docs)
                self.assertIn(provider["config_location"], self.docs)

    def test_packaged_loader_reads_the_single_matrix_source(self):
        loaded = load_capture_matrix()
        raw = MATRIX_PATH.read_bytes()

        self.assertEqual(loaded.schema_version, self.matrix["schema_version"])
        self.assertEqual(list(loaded.providers), self.matrix["providers"])
        self.assertEqual(loaded.version, "sha256:" + hashlib.sha256(raw).hexdigest())
        self.assertFalse((ROOT / "docs" / "capture-matrix.json").exists())
        self.assertIn("../src/agent_trace/capture_matrix.json", self.docs)


if __name__ == "__main__":
    unittest.main()
