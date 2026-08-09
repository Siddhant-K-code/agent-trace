"""Tests for privacy-preserving product telemetry."""

import argparse
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_trace import telemetry
from agent_trace.cli import (
    _run_with_product_telemetry,
    _telemetry_command_properties,
    build_parser,
)
from agent_trace.hooks import handle_session_end, handle_session_start


class _Response:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _TTY(io.StringIO):
    def isatty(self):
        return True


class TelemetryTestCase(unittest.TestCase):
    _ENV_NAMES = (
        "AGENT_STRACE_TELEMETRY_CONFIG",
        "AGENT_STRACE_TELEMETRY",
        "AGENT_STRACE_TELEMETRY_TOKEN",
        "AGENT_STRACE_TELEMETRY_HOST",
        "AGENT_STRACE_TELEMETRY_TIMEOUT",
        "DO_NOT_TRACK",
        "CI",
        "GITHUB_ACTIONS",
        "GITLAB_CI",
        "BUILDKITE",
        "JENKINS_URL",
        "TF_BUILD",
        "PYTEST_CURRENT_TEST",
        "AGENT_TRACE_DIR",
        "AGENT_TRACE_CLAUDE_SESSION_ID",
    )

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._old_env = {name: os.environ.get(name) for name in self._ENV_NAMES}
        for name in self._ENV_NAMES:
            os.environ.pop(name, None)
        self.config_path = Path(self.tmp.name) / "telemetry.json"
        os.environ["AGENT_STRACE_TELEMETRY_CONFIG"] = str(self.config_path)

    def tearDown(self):
        for name in self._ENV_NAMES:
            os.environ.pop(name, None)
        for name, value in self._old_env.items():
            if value is not None:
                os.environ[name] = value
        self.tmp.cleanup()


class TestTelemetryConsent(TelemetryTestCase):
    def test_default_is_disabled_without_creating_an_identifier(self):
        self.assertEqual(telemetry.consent_state(), "unset")
        self.assertFalse(telemetry.telemetry_enabled())
        self.assertFalse(self.config_path.exists())

    def test_enable_and_disable_manage_anonymous_identifier(self):
        self.assertTrue(telemetry.set_telemetry_enabled(True))
        enabled = json.loads(self.config_path.read_text())
        self.assertTrue(enabled["enabled"])
        self.assertRegex(enabled["anonymous_id"], r"^[0-9a-f]{32}$")
        self.assertTrue(telemetry.telemetry_enabled())

        self.assertTrue(telemetry.set_telemetry_enabled(False))
        disabled = json.loads(self.config_path.read_text())
        self.assertFalse(disabled["enabled"])
        self.assertNotIn("anonymous_id", disabled)
        self.assertFalse(telemetry.telemetry_enabled())

    def test_do_not_track_overrides_stored_consent(self):
        telemetry.set_telemetry_enabled(True)
        os.environ["DO_NOT_TRACK"] = "1"
        self.assertFalse(telemetry.telemetry_enabled())

    def test_environment_can_explicitly_enable_ci(self):
        os.environ["CI"] = "true"
        self.assertFalse(telemetry.telemetry_enabled())
        os.environ["AGENT_STRACE_TELEMETRY"] = "1"
        self.assertTrue(telemetry.telemetry_enabled())

    def test_environment_override_does_not_persist_consent(self):
        os.environ["AGENT_STRACE_TELEMETRY"] = "1"
        os.environ["AGENT_STRACE_TELEMETRY_TOKEN"] = "phc_public_project_token"
        with patch.object(telemetry.request, "urlopen", return_value=_Response()):
            telemetry.capture(
                telemetry.CLI_COMMAND_COMPLETED,
                {"command": "list", "success": True},
            )
        self.assertEqual(telemetry.consent_state(), "unset")

    def test_interactive_prompt_records_explicit_opt_in(self):
        os.environ["AGENT_STRACE_TELEMETRY_TOKEN"] = "phc_public_project_token"
        input_stream = _TTY("yes\n")
        output_stream = _TTY()
        with patch.object(telemetry, "capture", return_value=True) as capture:
            prompted = telemetry.maybe_prompt_for_consent(input_stream, output_stream)

        self.assertTrue(prompted)
        self.assertEqual(telemetry.consent_state(), "enabled")
        self.assertIn("No prompts, arguments, paths", output_stream.getvalue())
        capture.assert_called_once_with(
            telemetry.TELEMETRY_ENABLED,
            {"source": "first_run_prompt"},
        )


class TestTelemetryPayload(TelemetryTestCase):
    def setUp(self):
        super().setUp()
        os.environ["AGENT_STRACE_TELEMETRY_TOKEN"] = "phc_public_project_token"

    def test_payload_drops_unknown_and_free_form_properties(self):
        payload = telemetry.build_payload(
            telemetry.CLI_COMMAND_COMPLETED,
            {
                "command": "replay",
                "success": True,
                "duration_ms": 25,
                "error_type": "ValueError",
                "prompt": "secret prompt",
                "path": "/private/repository",
                "arguments": "--token secret",
            },
            "a" * 32,
        )

        self.assertIsNotNone(payload)
        properties = payload["properties"]
        self.assertEqual(properties["command"], "replay")
        self.assertTrue(properties["success"])
        self.assertNotIn("prompt", properties)
        self.assertNotIn("path", properties)
        self.assertNotIn("arguments", properties)
        self.assertFalse(properties["$process_person_profile"])
        self.assertTrue(properties["$geoip_disable"])

    def test_payload_rejects_unknown_events_and_unsafe_text(self):
        self.assertIsNone(telemetry.build_payload("unknown", {}, "a" * 32))
        payload = telemetry.build_payload(
            telemetry.CLI_COMMAND_COMPLETED,
            {"command": "replay /private/path", "success": True},
            "a" * 32,
        )
        self.assertNotIn("command", payload["properties"])

    def test_capture_posts_to_posthog_without_raising(self):
        os.environ["AGENT_STRACE_TELEMETRY"] = "1"
        os.environ["AGENT_STRACE_TELEMETRY_HOST"] = "https://eu.i.posthog.com"
        with patch.object(telemetry.request, "urlopen", return_value=_Response()) as urlopen:
            sent = telemetry.capture(
                telemetry.CLI_COMMAND_COMPLETED,
                {"command": "list", "success": True, "duration_ms": 3},
            )

        self.assertTrue(sent)
        request_arg = urlopen.call_args.args[0]
        self.assertEqual(request_arg.full_url, "https://eu.i.posthog.com/i/v0/e/")
        body = json.loads(request_arg.data)
        self.assertEqual(body["event"], telemetry.CLI_COMMAND_COMPLETED)
        self.assertEqual(body["properties"]["command"], "list")

    def test_network_failure_is_silent(self):
        os.environ["AGENT_STRACE_TELEMETRY"] = "1"
        with patch.object(telemetry.request, "urlopen", side_effect=OSError("offline")):
            self.assertFalse(
                telemetry.capture(
                    telemetry.CLI_COMMAND_COMPLETED,
                    {"command": "list", "success": True},
                )
            )


class TestCliTelemetry(TelemetryTestCase):
    def test_parser_registers_telemetry_commands(self):
        parser = build_parser()
        for action in ("status", "enable", "disable"):
            args = parser.parse_args(["telemetry", action])
            self.assertEqual(args.command, "telemetry")
            self.assertEqual(args.telemetry_command, action)

    def test_command_properties_never_include_user_arguments(self):
        args = argparse.Namespace(
            command="export",
            format="otlp-genai",
            backend="otlp",
            session_id="private-session-id",
            endpoint="https://secret.example",
        )
        properties = _telemetry_command_properties(args)
        self.assertEqual(properties["export_format"], "otlp-genai")
        self.assertEqual(properties["backend"], "otlp")
        self.assertNotIn("session_id", properties)
        self.assertNotIn("endpoint", properties)

    def test_wrapper_captures_success_and_preserves_return_code(self):
        args = argparse.Namespace(command="list")
        with patch("agent_trace.cli._product_telemetry.capture") as capture:
            result = _run_with_product_telemetry(args, lambda _args: 2)

        self.assertEqual(result, 2)
        properties = capture.call_args.args[1]
        self.assertEqual(properties["command"], "list")
        self.assertFalse(properties["success"])
        self.assertEqual(properties["exit_code"], 2)

    def test_wrapper_reports_exception_type_not_message(self):
        args = argparse.Namespace(command="list")

        def fail(_args):
            raise RuntimeError("secret path and token")

        with patch("agent_trace.cli._product_telemetry.capture") as capture:
            with self.assertRaises(RuntimeError):
                _run_with_product_telemetry(args, fail)

        properties = capture.call_args.args[1]
        self.assertEqual(properties["error_type"], "RuntimeError")
        self.assertNotIn("secret", json.dumps(properties))

    def test_status_explains_collection_boundaries(self):
        args = argparse.Namespace(telemetry_command="status")
        output = io.StringIO()
        with patch("sys.stdout", output):
            self.assertEqual(telemetry.cmd_telemetry(args), 0)
        self.assertIn("Never collected: prompts", output.getvalue())


class TestSessionTelemetry(TelemetryTestCase):
    def test_session_end_emits_one_sanitised_lifecycle_event(self):
        trace_dir = Path(self.tmp.name) / "traces"
        os.environ["AGENT_TRACE_DIR"] = str(trace_dir)
        with patch("agent_trace.hooks._product_telemetry.capture") as capture:
            handle_session_start(
                {"session_id": "telemetry-session-123", "source": "startup"},
                provider="claude",
            )
            handle_session_end(
                {"session_id": "telemetry-session-123", "cwd": "/private/repo"},
                provider="claude",
            )

        capture.assert_called_once()
        event, properties = capture.call_args.args
        self.assertEqual(event, telemetry.SESSION_COMPLETED)
        self.assertEqual(properties["provider"], "claude")
        self.assertEqual(properties["capture_method"], "hooks")
        self.assertNotIn("session_id", properties)
        self.assertNotIn("cwd", properties)


if __name__ == "__main__":
    unittest.main()
