"""Manifest and privacy-boundary tests for VS Code extension telemetry."""

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXTENSION = ROOT / "vscode-extension"


class TestVSCodeTelemetryManifest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.package = json.loads((EXTENSION / "package.json").read_text())

    def test_extension_minor_version_is_bumped(self):
        self.assertEqual(self.package["version"], "0.3.0")

    def test_default_on_application_setting_is_registered(self):
        setting = self.package["contributes"]["configuration"]["properties"][
            "agentTrace.telemetry.enabled"
        ]
        self.assertIs(setting["default"], True)
        self.assertEqual(setting["scope"], "application")

    def test_enable_and_disable_commands_are_registered(self):
        commands = {
            item["command"] for item in self.package["contributes"]["commands"]
        }
        self.assertIn("agentTrace.enableTelemetry", commands)
        self.assertIn("agentTrace.disableTelemetry", commands)


class TestVSCodeTelemetryPrivacyBoundary(unittest.TestCase):
    def test_event_schema_has_no_sensitive_fields(self):
        source = (EXTENSION / "src" / "telemetryCore.ts").read_text()
        schema = source.split("const EVENT_FIELDS", 1)[1].split(
            "export function isExtensionTelemetryEvent", 1
        )[0]
        for forbidden in (
            "arguments",
            "endpoint",
            "hostname",
            "path",
            "prompt",
            "repository",
            "response",
            "session_id",
            "trace",
            "username",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, schema)

    def test_posthog_profiles_and_geoip_are_disabled(self):
        source = (EXTENSION / "src" / "telemetry.ts").read_text()
        self.assertIn("$process_person_profile: false", source)
        self.assertIn("$geoip_disable: true", source)


if __name__ == "__main__":
    unittest.main()
