"""Tests for privacy-preserving product telemetry."""

import argparse
import io
import json
import os
import sys
import tempfile
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

try:  # keeps product telemetry off in this process and its children
    import _no_telemetry  # noqa: F401
except ImportError:  # pragma: no cover - run as a package module
    from tests import _no_telemetry  # noqa: F401

from agent_trace import telemetry
from agent_trace.cli import (
    _run_with_product_telemetry,
    _telemetry_command_properties,
    build_parser,
    main as cli_main,
)
from agent_trace.entry import main as entry_main
from agent_trace.hooks import handle_session_end, handle_session_start

try:
    from _concurrency import run_concurrently
except ImportError:
    from tests._concurrency import run_concurrently


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
        "AGENT_TRACE_CODEX_SESSION_ID",
        "AGENT_TRACE_GEMINI_SESSION_ID",
        "AGENT_TRACE_CURSOR_SESSION_ID",
        "AGENT_TRACE_COPILOT_SESSION_ID",
        # Set by the agents that run hooks; the notice relies on them.
        "CLAUDE_CODE_ENTRYPOINT",
        "CLAUDE_CODE_SESSION_ID",
        "GEMINI_SESSION_ID",
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

    def mark_notice_shown(self):
        self.config_path.write_text(json.dumps({"notice_shown": True}))


class TestTelemetryPreference(TelemetryTestCase):
    def test_default_is_enabled_without_eagerly_creating_an_identifier(self):
        self.assertEqual(telemetry.consent_state(), "unset")
        self.assertTrue(telemetry.telemetry_enabled())
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

    def test_environment_can_disable_the_default(self):
        os.environ["AGENT_STRACE_TELEMETRY"] = "0"
        self.assertFalse(telemetry.telemetry_enabled())
        self.assertEqual(telemetry.consent_state(), "unset")

    def test_stored_opt_out_prevents_capture(self):
        telemetry.set_telemetry_enabled(False)
        with patch.object(telemetry.request, "urlopen") as urlopen:
            self.assertFalse(
                telemetry.capture(
                    telemetry.CLI_COMMAND_COMPLETED,
                    {"command": "list", "success": True},
                )
            )
        urlopen.assert_not_called()

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

    def test_interactive_notice_discloses_default_once_without_prompting(self):
        os.environ["AGENT_STRACE_TELEMETRY_TOKEN"] = "phc_public_project_token"
        output_stream = _TTY()
        shown = telemetry.maybe_show_telemetry_notice(output_stream)

        self.assertTrue(shown)
        self.assertEqual(telemetry.consent_state(), "unset")
        self.assertTrue(telemetry.telemetry_enabled())
        self.assertTrue(json.loads(self.config_path.read_text())["notice_shown"])
        self.assertIn("enabled by default", output_stream.getvalue())
        self.assertIn("No prompts, arguments, paths", output_stream.getvalue())

        second_output = _TTY()
        self.assertFalse(telemetry.maybe_show_telemetry_notice(second_output))
        self.assertEqual(second_output.getvalue(), "")


_ANONYMOUS_ID_SCRIPT = """
import os
from agent_trace import telemetry
os.environ["AGENT_STRACE_TELEMETRY_CONFIG"] = sys.argv[2]
wait_for_start()
print(telemetry._anonymous_id())
"""


class TestTelemetryConfigLocking(TelemetryTestCase):
    def test_concurrent_processes_agree_on_one_anonymous_id(self):
        env = {"AGENT_STRACE_TELEMETRY": "1"}
        for name in self._ENV_NAMES:
            env.setdefault(name, "")
        with patch.dict(os.environ, env):
            outputs = run_concurrently(
                _ANONYMOUS_ID_SCRIPT, [[str(self.config_path)] for _ in range(8)]
            )
        ids = {out.strip() for out in outputs}
        stored = json.loads(self.config_path.read_text())["anonymous_id"]
        self.assertEqual(ids, {stored})

    def test_notice_does_not_overwrite_a_concurrent_opt_out(self):
        class _RacingTTY(_TTY):
            def write(self, text):
                # Another process opts out while the notice is printed.
                telemetry.set_telemetry_enabled(False)
                return super().write(text)

        with patch.object(telemetry, "telemetry_configured", return_value=True):
            self.assertTrue(telemetry.maybe_show_telemetry_notice(_RacingTTY()))
        data = json.loads(self.config_path.read_text())
        self.assertIs(data["enabled"], False)
        self.assertIs(data["notice_shown"], True)

    def test_writes_do_not_use_a_fixed_temp_name(self):
        # A leftover or concurrent fixed temp path must not block writes.
        Path(str(self.config_path) + ".tmp").mkdir()
        self.assertTrue(telemetry.set_telemetry_enabled(True))
        self.assertEqual(telemetry.consent_state(), "enabled")
        leftovers = [
            p.name for p in self.config_path.parent.iterdir()
            if p.name.startswith(".telemetry.json.")
        ]
        self.assertEqual(leftovers, [])

    @unittest.skipIf(os.name == "nt", "POSIX file modes")
    def test_config_file_is_private(self):
        self.assertTrue(telemetry.set_telemetry_enabled(True))
        self.assertEqual(self.config_path.stat().st_mode & 0o777, 0o600)

    def test_lock_failure_is_silent(self):
        os.environ["AGENT_STRACE_TELEMETRY"] = "1"
        with patch.object(telemetry, "file_lock", side_effect=TimeoutError("busy")):
            self.assertFalse(telemetry.set_telemetry_enabled(True))
            self.assertEqual(telemetry._anonymous_id(), "")
            with patch.object(telemetry, "telemetry_configured", return_value=True):
                self.assertFalse(telemetry.capture(telemetry.TELEMETRY_ENABLED, {}))
        self.assertFalse(self.config_path.exists())

    def test_on_disk_format_is_unchanged(self):
        self.assertTrue(telemetry.set_telemetry_enabled(True))
        text = self.config_path.read_text()
        data = json.loads(text)
        self.assertEqual(text, json.dumps(data, indent=2, sort_keys=True) + "\n")
        self.assertEqual(set(data), {"anonymous_id", "enabled", "updated_at"})


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
        os.environ["AGENT_STRACE_TELEMETRY_HOST"] = "https://eu.i.posthog.com"
        self.mark_notice_shown()
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
        self.assertRegex(body["distinct_id"], r"^[0-9a-f]{32}$")
        self.assertRegex(
            json.loads(self.config_path.read_text())["anonymous_id"],
            r"^[0-9a-f]{32}$",
        )

    def test_network_failure_is_silent(self):
        os.environ["AGENT_STRACE_TELEMETRY"] = "1"
        with patch.object(telemetry.request, "urlopen", side_effect=OSError("offline")):
            self.assertFalse(
                telemetry.capture(
                    telemetry.CLI_COMMAND_COMPLETED,
                    {"command": "list", "success": True},
                )
            )


_SESSION_ID_SCRIPT = """
import os
from agent_trace import telemetry
os.environ["AGENT_STRACE_TELEMETRY_CONFIG"] = sys.argv[2]
wait_for_start()
print(telemetry._analytics_session_id()[0])
"""


class TestAnalyticsSession(TelemetryTestCase):
    """Events carry a PostHog $session_id that follows PostHog's session rules."""

    def setUp(self):
        super().setUp()
        os.environ["AGENT_STRACE_TELEMETRY_TOKEN"] = "phc_public_project_token"
        os.environ["AGENT_STRACE_TELEMETRY"] = "1"
        self.now = 1_800_000_000.0

    def _session_at(self, minutes):
        with patch.object(telemetry.time, "time", return_value=self.now + minutes * 60):
            return telemetry._analytics_session_id()[0]

    def _capture(self):
        with patch.object(telemetry.request, "urlopen", return_value=_Response()) as urlopen:
            self.assertTrue(telemetry.capture(telemetry.SESSION_COMPLETED, {}))
        return json.loads(urlopen.call_args.args[0].data)

    def test_events_share_a_uuid7_session_that_starts_before_them(self):
        first, second = self._capture(), self._capture()
        session_id = first["properties"]["$session_id"]
        self.assertEqual(second["properties"]["$session_id"], session_id)
        self.assertEqual(uuid.UUID(session_id).version, 7)
        start = datetime.fromtimestamp(
            int(session_id[:8] + session_id[9:13], 16) / 1000, timezone.utc
        )
        for body in (first, second):
            self.assertLessEqual(start, datetime.fromisoformat(body["timestamp"]))

    def test_uuid7_clamps_times_outside_its_range(self):
        for unix_ms, start in ((-1, 0), (-86_400_000, 0), (1 << 48, (1 << 48) - 1)):
            with self.subTest(unix_ms=unix_ms):
                session_id = telemetry._uuid7(unix_ms)
                self.assertEqual(uuid.UUID(session_id).version, 7)
                self.assertEqual(telemetry._session_start_ms(session_id), start)

    def test_pre_epoch_clock_never_starts_a_session_after_its_event(self):
        self._capture()
        with patch.object(telemetry.time, "time", return_value=-86_400.5):
            body = self._capture()
        start = telemetry._session_start_ms(body["properties"]["$session_id"])
        self.assertEqual(start, 0)
        event_ms = datetime.fromisoformat(body["timestamp"]).timestamp() * 1000
        self.assertLessEqual(start, event_ms)

    def test_session_rotates_after_idle_time_and_maximum_length(self):
        telemetry.set_telemetry_enabled(True)
        first = self._session_at(0)
        self.assertEqual(self._session_at(29), first)
        idle = self._session_at(59)
        self.assertNotEqual(idle, first)

        # Kept active every 20 minutes, a session still ends after 24 hours.
        sessions = {self._session_at(59 + 20 * step) for step in range(73)}
        self.assertEqual(len(sessions), 2)

        # A clock set back before the session's start starts a new one.
        latest = self._session_at(59 + 20 * 72)
        self.assertNotIn(self._session_at(30), {first, idle, latest})

    def test_disable_deletes_the_session_and_enable_keeps_it(self):
        telemetry.set_telemetry_enabled(True)
        session_id = self._session_at(0)
        telemetry.set_telemetry_enabled(True)
        self.assertEqual(self._session_at(1), session_id)

        telemetry.set_telemetry_enabled(False)
        self.assertNotIn(session_id, self.config_path.read_text())
        self.assertEqual(self._session_at(2), "")

        telemetry.set_telemetry_enabled(True)
        self.assertNotIn(self._session_at(3), ("", session_id))

    def test_payload_never_takes_a_session_id_from_elsewhere(self):
        trace_session = telemetry.build_payload(
            telemetry.SESSION_COMPLETED, {}, "a" * 32, "telemetry-session-123"
        )
        from_properties = telemetry.build_payload(
            telemetry.SESSION_COMPLETED,
            {"$session_id": telemetry._uuid7(0), "session_id": "private"},
            "a" * 32,
        )
        for payload in (trace_session, from_properties):
            self.assertNotIn("$session_id", payload["properties"])
            self.assertNotIn("session_id", payload["properties"])

    def test_concurrent_processes_agree_on_one_session(self):
        telemetry.set_telemetry_enabled(True)
        env = {name: os.environ.get(name, "") for name in self._ENV_NAMES}
        with patch.dict(os.environ, env):
            outputs = run_concurrently(
                _SESSION_ID_SCRIPT, [[str(self.config_path)] for _ in range(8)]
            )
        stored = json.loads(self.config_path.read_text())["analytics_session_id"]
        self.assertEqual({out.strip() for out in outputs}, {stored})


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
        self.assertIn("enabled (enabled by default)", output.getvalue())
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


class TestDisclosureGate(TelemetryTestCase):
    """Default-on telemetry sends nothing until the notice or an explicit opt-in."""

    def setUp(self):
        super().setUp()
        os.environ["AGENT_STRACE_TELEMETRY_TOKEN"] = "phc_public_project_token"
        self.trace_dir = Path(self.tmp.name) / "traces"
        os.environ["AGENT_TRACE_DIR"] = str(self.trace_dir)

    def _run_cli(self, argv, stdin="", stderr=None):
        stdout = io.StringIO()
        stderr = stderr if stderr is not None else io.StringIO()
        with patch.object(sys, "argv", ["agent-strace", *argv]), \
                patch.object(sys, "stdin", io.StringIO(stdin)), \
                patch.object(sys, "stdout", stdout), \
                patch.object(sys, "stderr", stderr), \
                patch("agent_trace.cli.cmd_list", return_value=0):
            with self.assertRaises(SystemExit):
                cli_main()
        return stdout.getvalue(), stderr.getvalue()

    def _hook_session(self, session_id="gate-session-0001"):
        """Run session-start and session-end through the CLI hook entry point."""
        payload = json.dumps({"session_id": session_id, "source": "startup"})
        output = ""
        with patch.object(telemetry.request, "urlopen", return_value=_Response()) as urlopen:
            for event in ("session-start", "session-end"):
                out, err = self._run_cli(["hook", event], stdin=payload)
                output += out + err
        return urlopen, output

    def _sent_events(self, urlopen):
        return [json.loads(c.args[0].data)["event"] for c in urlopen.call_args_list]

    def test_hook_sends_nothing_and_creates_no_id_before_the_notice(self):
        urlopen, output = self._hook_session()
        urlopen.assert_not_called()
        self.assertEqual(output, "")
        self.assertFalse(self.config_path.exists())

    def test_hook_sends_once_the_notice_was_shown(self):
        self.mark_notice_shown()
        urlopen, output = self._hook_session()
        self.assertEqual(self._sent_events(urlopen), [telemetry.SESSION_COMPLETED])
        self.assertEqual(output, "")

    def test_hook_sends_after_an_explicit_choice(self):
        with self.subTest("telemetry enable"):
            telemetry.set_telemetry_enabled(True)
            urlopen, _ = self._hook_session()
            self.assertEqual(self._sent_events(urlopen), [telemetry.SESSION_COMPLETED])
        self.config_path.unlink()
        with self.subTest("AGENT_STRACE_TELEMETRY=1"):
            os.environ["AGENT_STRACE_TELEMETRY"] = "1"
            urlopen, _ = self._hook_session("gate-session-0002")
            self.assertEqual(self._sent_events(urlopen), [telemetry.SESSION_COMPLETED])

    def test_opt_outs_still_block_after_the_notice(self):
        opt_outs = {
            "DO_NOT_TRACK=1": {"DO_NOT_TRACK": "1"},
            "AGENT_STRACE_TELEMETRY=0": {"AGENT_STRACE_TELEMETRY": "0"},
            "CI default": {"CI": "true"},
        }
        for label, env in opt_outs.items():
            with self.subTest(label), patch.dict(os.environ, env):
                self.mark_notice_shown()
                with patch.object(telemetry.request, "urlopen") as urlopen:
                    self.assertFalse(telemetry.capture(telemetry.SESSION_COMPLETED, {}))
                urlopen.assert_not_called()
        with self.subTest("telemetry disable"):
            self.mark_notice_shown()
            telemetry.set_telemetry_enabled(False)
            with patch.object(telemetry.request, "urlopen") as urlopen:
                self.assertFalse(telemetry.capture(telemetry.SESSION_COMPLETED, {}))
            urlopen.assert_not_called()

    def test_non_interactive_cli_run_sends_nothing_before_the_notice(self):
        with patch.object(telemetry.request, "urlopen", return_value=_Response()) as urlopen:
            _, stderr = self._run_cli(["list"])
        urlopen.assert_not_called()
        self.assertNotIn("telemetry", stderr)
        self.assertFalse(self.config_path.exists())

    def test_first_interactive_cli_run_shows_the_notice_then_sends(self):
        stderr = _TTY()
        with patch.object(telemetry.request, "urlopen", return_value=_Response()) as urlopen:
            self._run_cli(["list"], stderr=stderr)
        self.assertIn("enabled by default", stderr.getvalue())
        self.assertEqual(self._sent_events(urlopen), [telemetry.CLI_COMMAND_COMPLETED])

    def test_notice_still_shows_when_the_environment_value_is_not_a_choice(self):
        os.environ["AGENT_STRACE_TELEMETRY"] = "maybe"
        self.assertTrue(telemetry.maybe_show_telemetry_notice(_TTY()))

    def test_status_explains_the_pending_notice(self):
        args = argparse.Namespace(telemetry_command="status")
        output = io.StringIO()
        with patch("sys.stdout", output):
            telemetry.cmd_telemetry(args)
        self.assertIn("Waiting for the one-time notice", output.getvalue())

        self.mark_notice_shown()
        output = io.StringIO()
        with patch("sys.stdout", output):
            telemetry.cmd_telemetry(args)
        self.assertNotIn("Waiting", output.getvalue())


_SID = "notice-session-0001"

# Each worker is a separate hook process at session start of its own
# interactive Claude Code session, racing on one telemetry config.
_HOOK_NOTICE_SCRIPT = """
import io, os
from agent_trace.entry import main
session_id = sys.argv[2]
wait_for_start()
os.environ["CLAUDE_CODE_ENTRYPOINT"] = "cli"
os.environ["CLAUDE_CODE_SESSION_ID"] = session_id
sys.argv = ["agent-strace", "hook", "session-start"]
sys.stdin = io.StringIO('{"session_id": "%s", "source": "startup"}' % session_id)
try:
    main()
except SystemExit as exc:
    if exc.code not in (0, None):
        raise
"""


class _BrokenStdout(io.StringIO):
    def write(self, text):
        raise BrokenPipeError("agent closed the hook's stdout")


class TestHookTelemetryNotice(TelemetryTestCase):
    """Hooks show the notice once, only where the agent displays it."""

    def setUp(self):
        super().setUp()
        os.environ["AGENT_STRACE_TELEMETRY_TOKEN"] = "phc_public_project_token"
        # Unreachable on purpose; every hook run below also patches urlopen.
        os.environ["AGENT_STRACE_TELEMETRY_HOST"] = "http://127.0.0.1:9"
        os.environ["AGENT_TRACE_DIR"] = str(Path(self.tmp.name) / "traces")

    def _hook(self, provider, event, payload, stdout=None):
        """Run one hook through the console-script entry point."""
        stdout = stdout if stdout is not None else io.StringIO()
        argv = ["agent-strace", "hook", "--provider", provider, event]
        with patch.object(sys, "argv", argv), \
                patch.object(sys, "stdin", io.StringIO(json.dumps(payload))), \
                patch.object(sys, "stdout", stdout), \
                patch.object(sys, "stderr", io.StringIO()), \
                patch.object(telemetry.request, "urlopen", return_value=_Response()) as urlopen:
            with self.assertRaises(SystemExit) as exited:
                entry_main()
        self.assertIn(exited.exception.code, (0, None))
        return stdout.getvalue(), [
            json.loads(c.args[0].data)["event"] for c in urlopen.call_args_list
        ]

    def _rollout(self, source="cli", session_id=_SID, first_line=None):
        if first_line is None:
            first_line = json.dumps({
                "timestamp": "2026-01-01T00:00:00.000Z",
                "type": "session_meta",
                "payload": {
                    "session_id": session_id,
                    "id": session_id,
                    "source": source,
                    "originator": "codex-tui" if source == "cli" else "codex_exec",
                    "base_instructions": {"text": "x" * 30000},
                },
            })
        path = Path(self.tmp.name) / f"rollout-{source}.jsonl"
        path.write_text(first_line + "\n" + json.dumps({"type": "turn_context"}) + "\n")
        return str(path)

    def _displayed(self, provider):
        """Env and payload of a session start whose message the agent shows."""
        payload = {"session_id": _SID, "source": "startup", "hook_event_name": "SessionStart"}
        if provider == "claude":
            return {"CLAUDE_CODE_ENTRYPOINT": "cli", "CLAUDE_CODE_SESSION_ID": _SID}, payload
        if provider == "gemini":
            return {"GEMINI_SESSION_ID": _SID}, payload
        if provider == "codex":
            return {}, {**payload, "transcript_path": self._rollout()}
        raise AssertionError(provider)

    def _config(self):
        if not self.config_path.exists():
            return {}
        return json.loads(self.config_path.read_text())

    def _assert_notice_shown_once(self, provider):
        env, payload = self._displayed(provider)
        with patch.dict(os.environ, env):
            out, sent = self._hook(provider, "session-start", payload)
            self.assertEqual(out.count("\n"), 1)
            self.assertEqual(json.loads(out), {"systemMessage": telemetry.HOOK_NOTICE})
            self.assertIn("agent-strace telemetry disable", telemetry.HOOK_NOTICE)
            self.assertEqual(sent, [])
            config = self._config()
            self.assertIs(config["notice_shown"], True)
            self.assertNotIn("anonymous_id", config)

            again, _ = self._hook(provider, "session-start", payload)
            self.assertEqual(again, "")
        return env, payload

    def _assert_session_end_sends(self, provider, env, payload):
        with patch.dict(os.environ, env):
            out, sent = self._hook(provider, "session-end", payload)
        self.assertEqual(out, "")
        self.assertEqual(sent, [telemetry.SESSION_COMPLETED])

    def test_claude_code_interactive_session_shows_the_notice_then_sends(self):
        env, payload = self._assert_notice_shown_once("claude")
        self._assert_session_end_sends("claude", env, payload)

    def test_gemini_cli_session_shows_the_notice_then_sends(self):
        env, payload = self._assert_notice_shown_once("gemini")
        self._assert_session_end_sends("gemini", env, payload)

    def test_codex_tui_session_shows_the_notice(self):
        self._assert_notice_shown_once("codex")

    def test_cursor_and_copilot_print_nothing_and_keep_the_gate(self):
        env = {
            "CLAUDE_CODE_ENTRYPOINT": "cli",
            "CLAUDE_CODE_SESSION_ID": _SID,
            "GEMINI_SESSION_ID": _SID,
        }
        payload = {
            "session_id": _SID,
            "source": "startup",
            "transcript_path": self._rollout(),
        }
        for provider in ("cursor", "copilot"):
            with self.subTest(provider), patch.dict(os.environ, env):
                out, sent = self._hook(provider, "session-start", payload)
                self.assertEqual(out, "")
                out, sent_at_end = self._hook(provider, "session-end", payload)
                self.assertEqual(out, "")
                self.assertEqual(sent + sent_at_end, [])
                self.assertFalse(self.config_path.exists())

    def test_no_notice_where_the_agent_does_not_display_it(self):
        startup = {"session_id": _SID, "source": "startup"}
        cases = {
            "claude -p": ("claude", {"CLAUDE_CODE_ENTRYPOINT": "sdk-cli",
                                     "CLAUDE_CODE_SESSION_ID": _SID}, startup),
            "claude env from a parent session": (
                "claude", {"CLAUDE_CODE_ENTRYPOINT": "cli",
                           "CLAUDE_CODE_SESSION_ID": "parent-session"}, startup),
            "claude-format hook in another agent": ("claude", {}, startup),
            "codex exec": ("codex", {}, {**startup, "transcript_path": self._rollout("exec")}),
            "codex resume": ("codex", {}, {**startup, "source": "resume",
                                           "transcript_path": self._rollout()}),
            "codex other session's rollout": (
                "codex", {}, {**startup,
                              "transcript_path": self._rollout(session_id="other")}),
            "codex rollout missing": (
                "codex", {}, {**startup,
                              "transcript_path": str(Path(self.tmp.name) / "none.jsonl")}),
            "codex rollout not session_meta": (
                "codex", {}, {**startup,
                              "transcript_path": self._rollout(first_line="{not json")}),
            "codex no rollout": ("codex", {}, startup),
            "gemini other session": ("gemini", {"GEMINI_SESSION_ID": "other"}, startup),
            "gemini env missing": ("gemini", {}, startup),
        }
        for label, (provider, env, payload) in cases.items():
            with self.subTest(label), patch.dict(os.environ, env):
                out, sent = self._hook(provider, "session-start", payload)
                self.assertEqual(out, "")
                self.assertEqual(sent, [])
                self.assertFalse(self.config_path.exists())

    def test_no_notice_when_telemetry_is_off_or_already_disclosed(self):
        cases = {
            "DO_NOT_TRACK=1": ({"DO_NOT_TRACK": "1"}, None),
            "AGENT_STRACE_TELEMETRY=0": ({"AGENT_STRACE_TELEMETRY": "0"}, None),
            "telemetry disable": ({}, {"enabled": False}),
            "CI": ({"CI": "true"}, None),
            "GITHUB_ACTIONS": ({"GITHUB_ACTIONS": "true"}, None),
            "notice already shown": ({}, {"notice_shown": True}),
            "telemetry enable": ({}, {"enabled": True}),
            "AGENT_STRACE_TELEMETRY=1": ({"AGENT_STRACE_TELEMETRY": "1"}, None),
        }
        for provider in ("claude", "codex", "gemini"):
            agent_env, payload = self._displayed(provider)
            for label, (env, config) in cases.items():
                with self.subTest(provider=provider, case=label), \
                        patch.dict(os.environ, {**agent_env, **env}):
                    if config is None:
                        self.config_path.unlink(missing_ok=True)
                    else:
                        self.config_path.write_text(json.dumps(config))
                    out, sent = self._hook(provider, "session-start", payload)
                    self.assertEqual(out, "")
                    self.assertEqual(sent, [])
                    self.assertEqual(self._config(), config or {})

    def test_failed_stdout_write_does_not_mark_the_notice(self):
        env, payload = self._displayed("claude")
        with patch.dict(os.environ, env):
            out, _ = self._hook("claude", "session-start", payload, stdout=_BrokenStdout())
        self.assertNotIn("notice_shown", self._config())
        self.assertFalse(telemetry.disclosure_satisfied())

    def test_config_failures_never_fail_the_hook(self):
        env, payload = self._displayed("claude")
        blocker = Path(self.tmp.name) / "not-a-directory"
        blocker.write_text("")
        with self.subTest("config directory cannot be created"), \
                patch.dict(os.environ, {
                    **env,
                    "AGENT_STRACE_TELEMETRY_CONFIG": str(blocker / "telemetry.json"),
                }):
            out, _ = self._hook("claude", "session-start", payload)
            self.assertEqual(out, "")
        with self.subTest("lock busy"), patch.dict(os.environ, env), \
                patch.object(telemetry, "file_lock", side_effect=TimeoutError("busy")):
            out, _ = self._hook("claude", "session-start", payload)
            self.assertEqual(out, "")
            self.assertFalse(self.config_path.exists())
        with self.subTest("write fails after the notice"), patch.dict(os.environ, env), \
                patch.object(telemetry, "_write_config", return_value=False):
            out, _ = self._hook("claude", "session-start", payload)
            # Shown but not recorded: it shows again rather than claim a
            # notice that might not have been seen.
            self.assertEqual(json.loads(out), {"systemMessage": telemetry.HOOK_NOTICE})
            self.assertFalse(telemetry.disclosure_satisfied())

    def test_notice_rechecks_the_config_under_the_lock(self):
        # Another process recorded a choice or the notice after notice_due().
        for config in ({"enabled": False}, {"enabled": True}, {"notice_shown": True}):
            with self.subTest(config), \
                    patch.object(telemetry, "notice_due", return_value=True):
                self.config_path.write_text(json.dumps(config))
                written = []
                self.assertFalse(telemetry.show_hook_notice(written.append))
                self.assertEqual(written, [])
                self.assertEqual(self._config(), config)

    def test_concurrent_session_starts_show_the_notice_once(self):
        env = {"HOME": self.tmp.name}
        for name in self._ENV_NAMES:
            env.setdefault(name, os.environ.get(name, ""))
        with patch.dict(os.environ, env):
            outputs = run_concurrently(
                _HOOK_NOTICE_SCRIPT, [[f"race-session-{i:04d}"] for i in range(8)]
            )
        notices = [out for out in outputs if out]
        self.assertEqual(len(notices), 1, outputs)
        self.assertEqual(json.loads(notices[0]), {"systemMessage": telemetry.HOOK_NOTICE})
        config = self._config()
        self.assertIs(config["notice_shown"], True)
        self.assertNotIn("anonymous_id", config)


if __name__ == "__main__":
    unittest.main()
