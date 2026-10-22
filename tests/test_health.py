"""Tests for agent-strace health."""

import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from agent_trace import health
from agent_trace.cli import _telemetry_command_properties, build_parser, main
from agent_trace.evidence_health import EvidenceHealthStatus
from agent_trace.health import (
    CaptureMatrix,
    assess_session,
    blind_spots_for,
    cmd_health,
    detect_provider,
    format_health_text,
    load_capture_matrix,
    session_finalized_for,
)
from agent_trace.hooks import (
    _PROVIDER_ENV,
    handle_post_tool,
    handle_pre_tool,
    handle_session_end,
    handle_session_start,
    handle_stop,
    handle_user_prompt,
    hook_main,
)
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore
from agent_trace.telemetry import CLI_COMMAND_COMPLETED, _sanitise_properties

T0 = 1_700_000_000.0
MATRIX_FILE = Path(health.__file__).with_name("capture_matrix.json")
RAW_MATRIX = json.loads(MATRIX_FILE.read_text(encoding="utf-8"))
TOP_LEVEL_KEYS = {
    "session_id",
    "status",
    "reasons",
    "schema_version",
    "provider",
    "capture_method",
    "capture_matrix_version",
    "observed_start",
    "observed_end",
    "event_count",
    "session_finalized",
    "provider_source",
    "capture_matrix",
}
ENTRY_KEYS = {
    "id",
    "name",
    "capture_method",
    "verification_status",
    "tested_version",
    "last_verified",
    "coverage",
}


def raw_entry(provider):
    return next(entry for entry in RAW_MATRIX["providers"] if entry["id"] == provider)


def unavailable(provider):
    return [cap for cap, value in raw_entry(provider)["coverage"].items() if value == "unavailable"]


def event(event_type, offset, data=None):
    return TraceEvent(event_type, timestamp=T0 + offset, data=data or {}, tenant_id="")


def write_session(trace_dir, session_id, events, *, agent_name="", command="",
                  ended=True, started_at=T0):
    store = TraceStore(trace_dir)
    store.create_session(SessionMeta(
        session_id=session_id,
        started_at=started_at,
        agent_name=agent_name,
        command=command,
    ))
    for item in events:
        item.session_id = session_id
        store.append_event(session_id, item)
    if ended:
        meta = store.load_meta(session_id)
        meta.ended_at = started_at + 60
        store.update_meta(meta)
    return store


def hook_session(session_id, provider, *, end=True, source="startup", tool_use_id="tool-1"):
    """Drive the real hook handlers through one tool call."""
    handle_session_start({"session_id": session_id, "source": source}, provider=provider)
    handle_user_prompt({"session_id": session_id, "prompt": "list files"}, provider=provider)
    handle_pre_tool({
        "session_id": session_id,
        "tool_name": "Bash",
        "tool_use_id": tool_use_id,
        "tool_input": {"command": "ls"},
    }, provider=provider)
    handle_post_tool({
        "session_id": session_id,
        "tool_name": "Bash",
        "tool_use_id": tool_use_id,
        "tool_response": {"output": "ok"},
    }, provider=provider)
    handle_stop({"session_id": session_id, "last_assistant_message": "done"}, provider=provider)
    if end:
        handle_session_end({"session_id": session_id}, provider=provider)


class _HealthBase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.trace_dir = str(Path(tmp.name) / ".agent-traces")
        env = patch.dict(os.environ, {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        for name in (
            *_PROVIDER_ENV.values(),
            "AGENT_TRACE_REDACT",
            "AGENT_STRACE_TENANT_ID",
            "AGENT_STRACE_WORKSPACE",
            "AGENT_STRACE_ENDPOINT",
        ):
            os.environ.pop(name, None)
        os.environ["AGENT_TRACE_DIR"] = self.trace_dir

    def run_health(self, *argv):
        args = build_parser().parse_args(["--trace-dir", self.trace_dir, "health", *argv])
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cmd_health(args)
        return code, out.getvalue(), err.getvalue()

    def health_json(self, *argv):
        code, out, err = self.run_health(*argv, "--format", "json")
        self.assertEqual(err, "")
        return code, json.loads(out)

    def markers_naming(self, session_id):
        base = TraceStore(self.trace_dir).base_dir
        return [
            path for path in base.glob(".active-session*")
            if path.read_text(encoding="utf-8").strip() == session_id
        ]

    @staticmethod
    def messages(payload, kind):
        return [r["message"] for r in payload["reasons"] if r["kind"] == kind]

    @staticmethod
    def codes(payload, kind):
        return [r["code"] for r in payload["reasons"] if r["kind"] == kind]


class TestHookSessions(_HealthBase):
    def test_ended_claude_session_reports_only_declared_limitations(self):
        hook_session("claudeend0000001", "claude")

        code, payload = self.health_json("claudeend0000001")

        gaps = raw_entry("claude")["known_gaps"]
        self.assertEqual(unavailable("claude"), [])
        self.assertEqual(payload["status"], "partial" if gaps else "healthy")
        self.assertEqual(code, 0)
        self.assertEqual(self.codes(payload, "observed"), [])
        self.assertEqual(self.messages(payload, "provider_limitation"), [g.strip() for g in gaps])
        self.assertEqual(payload["provider"], "claude")
        self.assertEqual(payload["provider_source"], "hook_events")
        self.assertEqual(payload["capture_method"], raw_entry("claude")["capture_method"])
        self.assertEqual(payload["capture_matrix_version"], load_capture_matrix().version)
        self.assertIs(payload["session_finalized"], True)
        self.assertEqual(payload["capture_matrix"]["entry"]["id"], "claude")
        self.assertIsNone(payload["capture_matrix"]["note"])

    def test_codex_session_is_partial_with_known_gaps_and_unavailable_coverage(self):
        hook_session("codexsession0001", "codex", end=False)
        self.assertTrue(self.markers_naming("codexsession0001"))

        code, payload = self.health_json("codexsession0001")

        gaps = [g.strip() for g in raw_entry("codex")["known_gaps"]]
        missing = unavailable("codex")
        self.assertIn("session_end", missing)
        limitations = self.messages(payload, "provider_limitation")
        self.assertEqual(limitations[: len(gaps)], gaps)
        self.assertEqual(len(limitations), len(gaps) + len(missing))
        for capability, message in zip(missing, limitations[len(gaps):]):
            self.assertIn(capability, message)
        spots = payload["capture_matrix"]["blind_spots"]
        self.assertEqual([s["text"] for s in spots], limitations)
        self.assertEqual([s.get("capability") for s in spots if s["source"] == "coverage"], missing)

        # Codex declares no session-end hook, so its marker never proves activity.
        self.assertIsNone(payload["session_finalized"])
        self.assertEqual(self.codes(payload, "observed"), ["missing_session_end"])
        self.assertEqual(payload["status"], "partial")
        self.assertEqual(code, 0)

    def test_active_claude_session_is_reported_active_not_unfinished(self):
        hook_session("claudelive000001", "claude", end=False)
        self.assertTrue(self.markers_naming("claudelive000001"))

        code, payload = self.health_json("claudelive000001")

        self.assertIs(payload["session_finalized"], False)
        self.assertEqual(self.codes(payload, "observed"), ["session_active"])
        self.assertEqual(payload["status"], "partial")
        self.assertEqual(code, 0)

    def test_claude_session_without_end_or_marker_is_not_assumed_active(self):
        hook_session("claudegone000001", "claude", end=False)
        for marker in self.markers_naming("claudegone000001"):
            marker.unlink()

        _, payload = self.health_json("claudegone000001")

        self.assertIsNone(payload["session_finalized"])
        self.assertIn("missing_session_end", self.codes(payload, "observed"))
        self.assertNotIn("session_active", self.codes(payload, "observed"))

    def test_provider_without_matrix_entry_says_so(self):
        hook_session("geminisession001", "gemini")

        code, payload = self.health_json("geminisession001")

        self.assertEqual(payload["provider"], "gemini")
        self.assertEqual(payload["provider_source"], "hook_events")
        self.assertEqual(payload["capture_method"], "")
        self.assertEqual(payload["capture_matrix_version"], "")
        self.assertEqual(self.messages(payload, "provider_limitation"), [])
        self.assertIsNone(payload["capture_matrix"]["entry"])
        self.assertEqual(payload["capture_matrix"]["blind_spots"], [])
        self.assertIn("no entry for provider 'gemini'", payload["capture_matrix"]["note"])
        self.assertEqual(payload["capture_matrix"]["version"], load_capture_matrix().version)
        self.assertEqual(payload["status"], "healthy")
        self.assertEqual(code, 0)

    def test_hook_metadata_identifies_a_session_without_session_start(self):
        handle_pre_tool({
            "session_id": "codeximplicit001",
            "tool_name": "Bash",
            "tool_use_id": "tool-1",
            "tool_input": {"command": "ls"},
        }, provider="codex")

        _, payload = self.health_json("codeximplicit001")

        self.assertEqual(payload["provider"], "codex")
        self.assertEqual(payload["provider_source"], "hook_metadata")
        self.assertEqual(payload["capture_matrix"]["entry"]["id"], "codex")
        self.assertFalse(payload["observed_start"])


class TestCopilotSessions(_HealthBase):
    """GitHub Copilot CLI hook payloads, in the order Copilot sends them."""

    SID = "c0ffee00-0000-4000-8000-000000000001"
    CHILD = "c0ffee01-0000-4000-8000-000000000002"

    def setUp(self):
        super().setUp()
        os.environ["DO_NOT_TRACK"] = "1"

    def hook(self, subcommand, payload):
        err = io.StringIO()
        with patch.object(sys, "stdin", io.StringIO(json.dumps(payload))), redirect_stderr(err):
            hook_main(["--provider", "copilot", subcommand])
        self.assertEqual(err.getvalue(), "")

    def main_session(self):
        self.hook("user-prompt", {"sessionId": self.SID, "prompt": "list files"})
        self.hook("session-start", {"sessionId": self.SID, "source": "new", "initialPrompt": "list files"})
        self.hook("pre-tool", {"sessionId": self.SID, "toolName": "view", "toolArgs": {"path": "a"}})
        self.hook("post-tool", {
            "sessionId": self.SID,
            "toolName": "view",
            "toolArgs": {"path": "a"},
            "toolResult": {"resultType": "success", "textResultForLlm": "ok"},
        })
        self.hook("stop", {"sessionId": self.SID, "stopReason": "end_turn", "stop_hook_active": False})
        self.hook("session-end", {"sessionId": self.SID, "reason": "complete"})
        return self.SID[:16]

    def test_main_session_maps_to_the_copilot_entry(self):
        session_id = self.main_session()

        code, payload = self.health_json(session_id)

        gaps = [g.strip() for g in raw_entry("copilot")["known_gaps"]]
        missing = unavailable("copilot")
        self.assertEqual(missing, ["response"])
        self.assertEqual(payload["provider"], "copilot")
        self.assertEqual(payload["provider_source"], "hook_events")
        self.assertEqual(payload["capture_method"], "cli_hooks")
        self.assertEqual(payload["capture_matrix"]["entry"]["id"], "copilot")
        self.assertEqual(payload["capture_matrix"]["entry"]["verification_status"], "verified")
        self.assertIsNone(payload["capture_matrix"]["note"])
        self.assertIs(payload["session_finalized"], True)
        self.assertEqual(self.codes(payload, "observed"), [])
        self.assertEqual(
            self.codes(payload, "provider_limitation"),
            ["declared_events_before_session_start"]
            + ["provider_blind_spot"] * (len(gaps) + len(missing)),
        )
        self.assertEqual(self.messages(payload, "provider_limitation")[1:len(gaps) + 1], gaps)
        self.assertEqual(payload["status"], "partial")
        self.assertEqual(code, 0)

    def test_subagent_session_without_session_start_maps_from_hook_metadata(self):
        self.hook("user-prompt", {"sessionId": self.CHILD, "prompt": "report a line"})
        self.hook("pre-tool", {"sessionId": self.CHILD, "toolName": "view", "toolArgs": {"path": "a"}})
        self.hook("post-tool", {
            "sessionId": self.CHILD,
            "toolName": "view",
            "toolArgs": {"path": "a"},
            "toolResult": {"resultType": "success", "textResultForLlm": "ok"},
        })
        session_id = self.CHILD[:16]
        meta = TraceStore(self.trace_dir).load_meta(session_id)
        self.assertEqual(meta.command, "github-copilot (no session-start)")

        code, payload = self.health_json(session_id)

        self.assertEqual(payload["provider"], "copilot")
        self.assertEqual(payload["provider_source"], "hook_metadata")
        self.assertEqual(payload["capture_matrix"]["entry"]["id"], "copilot")
        self.assertFalse(payload["observed_start"])
        self.assertIsNone(payload["session_finalized"])
        self.assertEqual(
            self.codes(payload, "observed"),
            ["missing_session_start", "missing_session_end"],
        )
        self.assertEqual(payload["status"], "partial")
        self.assertEqual(code, 0)

    def test_subagent_session_with_a_stop_maps_from_hook_events(self):
        self.hook("user-prompt", {"sessionId": self.CHILD, "prompt": "report a line"})
        self.hook("stop", {"sessionId": self.CHILD, "stopReason": "end_turn", "stop_hook_active": False})

        code, payload = self.health_json(self.CHILD[:16])

        self.assertEqual(payload["provider"], "copilot")
        self.assertEqual(payload["provider_source"], "hook_events")
        self.assertEqual(payload["status"], "partial")
        self.assertEqual(code, 0)

    def test_backfilled_subagent_session_maps_from_its_stop_event(self):
        write_session(self.trace_dir, "backfilledchild1", [
            event(EventType.USER_PROMPT, 1, {"prompt": "report a line"}),
            event(EventType.ASSISTANT_RESPONSE, 2,
                  {"text": "", "hook_event": "stop", "provider": "copilot"}),
        ], agent_name="github-copilot", command="github-copilot (backfilled)", ended=False)

        code, payload = self.health_json("backfilledchild1")

        self.assertEqual(payload["provider"], "copilot")
        self.assertEqual(payload["provider_source"], "hook_events")
        self.assertEqual(payload["status"], "partial")
        self.assertEqual(code, 0)

    def test_prompt_before_start_stays_invalid_for_providers_that_do_not_declare_it(self):
        handle_user_prompt({"session_id": "claudeearly00001", "prompt": "hi"}, provider="claude")
        handle_session_start({"session_id": "claudeearly00001", "source": "startup"}, provider="claude")
        handle_session_end({"session_id": "claudeearly00001"}, provider="claude")
        self.assertNotIn("events_before_start", raw_entry("claude"))

        code, payload = self.health_json("claudeearly00001")

        self.assertEqual(payload["provider"], "claude")
        self.assertIn("events_before_session_start", self.codes(payload, "observed"))
        self.assertEqual(payload["status"], "invalid")
        self.assertEqual(code, 1)


class TestResumedAndCompactedSessions(_HealthBase):
    """Real hook handlers: resume and compact continue one recorded session."""

    def assert_one_session(self, session_id):
        code, payload = self.health_json(session_id)
        gaps = raw_entry("claude")["known_gaps"]
        self.assertEqual(self.codes(payload, "observed"), [])
        self.assertEqual(payload["status"], "partial" if gaps else "healthy")
        self.assertEqual(code, 0)
        return payload

    def start_sources(self, session_id):
        return [
            item.data.get("source")
            for item in TraceStore(self.trace_dir).load_events(session_id)
            if item.event_type == EventType.SESSION_START
        ]

    def test_plain_session(self):
        hook_session("claudeplain00001", "claude")

        payload = self.assert_one_session("claudeplain00001")
        self.assertIs(payload["session_finalized"], True)

    def test_compacted_session(self):
        hook_session("claudecompact001", "claude", end=False)
        started = TraceStore(self.trace_dir).load_meta("claudecompact001")
        hook_session("claudecompact001", "claude", source="compact", tool_use_id="tool-2")

        self.assert_one_session("claudecompact001")
        meta = TraceStore(self.trace_dir).load_meta("claudecompact001")
        self.assertEqual(meta.started_at, started.started_at)
        self.assertEqual(meta.command, "claude-code (startup)")
        self.assertEqual(self.start_sources("claudecompact001"), ["startup", "compact"])

    def test_resumed_session(self):
        hook_session("clauderesume0001", "claude")
        hook_session("clauderesume0001", "claude", source="resume", tool_use_id="tool-2")

        payload = self.assert_one_session("clauderesume0001")
        self.assertIs(payload["session_finalized"], True)
        self.assertEqual(self.start_sources("clauderesume0001"), ["startup", "resume"])

    def test_resumed_then_compacted_session(self):
        hook_session("clauderescomp001", "claude")
        hook_session("clauderescomp001", "claude", source="resume", end=False, tool_use_id="tool-2")
        hook_session("clauderescomp001", "claude", source="compact", tool_use_id="tool-3")

        self.assert_one_session("clauderescomp001")
        self.assertEqual(
            self.start_sources("clauderescomp001"),
            ["startup", "resume", "compact"],
        )

    def test_active_resumed_session_is_reported_active(self):
        hook_session("claudereslive001", "claude")
        hook_session("claudereslive001", "claude", source="resume", end=False, tool_use_id="tool-2")

        code, payload = self.health_json("claudereslive001")

        self.assertIs(payload["session_finalized"], False)
        self.assertEqual(self.codes(payload, "observed"), ["session_active"])
        self.assertEqual(payload["status"], "partial")
        self.assertEqual(code, 0)

    def test_codex_resume_then_compact_without_session_end(self):
        hook_session("codexrescomp0001", "codex", end=False)
        hook_session("codexrescomp0001", "codex", source="resume", end=False, tool_use_id="tool-2")
        hook_session("codexrescomp0001", "codex", source="compact", end=False, tool_use_id="tool-3")

        code, payload = self.health_json("codexrescomp0001")

        self.assertIsNone(payload["session_finalized"])
        self.assertEqual(
            self.start_sources("codexrescomp0001"),
            ["startup", "resume", "compact"],
        )
        self.assertEqual(self.codes(payload, "observed"), ["missing_session_end"])
        self.assertEqual(payload["status"], "partial")
        self.assertEqual(code, 0)

    def test_repeated_startup_start_stays_invalid(self):
        hook_session("claudedupstart01", "claude")
        hook_session("claudedupstart01", "claude", tool_use_id="tool-2")

        code, payload = self.health_json("claudedupstart01")

        self.assertEqual(payload["status"], "invalid")
        self.assertIn("duplicate_session_start", self.codes(payload, "observed"))
        self.assertEqual(code, 1)

    def test_event_name_fallback_source_is_not_a_continuation(self):
        # Cursor sends no source, so the hooks record the hook event name.
        for provider, session_id, event_name in (
            ("cursor", "cursorfallback01", "sessionStart"),
            ("gemini", "geminifallback01", "SessionStart"),
        ):
            with self.subTest(provider=provider):
                payload = json.dumps({"session_id": session_id, "hook_event_name": event_name})
                for _ in range(2):
                    with patch.object(sys, "stdin", io.StringIO(payload)):
                        hook_main(["--provider", provider, "session-start"])

                self.assertEqual(self.start_sources(session_id), [event_name, event_name])
                code, result = self.health_json(session_id)
                self.assertEqual(result["status"], "invalid")
                self.assertIn("duplicate_session_start", self.codes(result, "observed"))
                self.assertEqual(code, 1)


class TestUnknownProvider(_HealthBase):
    def test_sdk_session_named_like_a_hook_agent_gets_no_limitations(self):
        write_session(
            self.trace_dir,
            "a1b2c3d4e5f60718",
            [
                event(EventType.SESSION_START, 0, {"agent_name": "claude-code"}),
                event(EventType.SESSION_END, 10),
            ],
            agent_name="claude-code",
            command="python agent.py",
        )

        code, payload = self.health_json("a1b2c3d4e5f60718")

        self.assertEqual(payload["provider"], "")
        self.assertIsNone(payload["provider_source"])
        self.assertEqual(payload["capture_method"], "")
        self.assertEqual(payload["capture_matrix_version"], "")
        self.assertEqual(payload["reasons"], [])
        self.assertIsNone(payload["capture_matrix"]["entry"])
        self.assertIn("Provider not identified", payload["capture_matrix"]["note"])
        self.assertEqual(payload["status"], "healthy")
        self.assertEqual(code, 0)

        _, out, _ = self.run_health("a1b2c3d4e5f60718")
        self.assertIn("Provider:  unknown\n", out)
        self.assertIn("Provider limitations (0):\n  not assessed\n", out)
        self.assertIn("Capture matrix: no entry applied\nNote: Provider not identified", out)

    def test_conflicting_hook_providers_apply_no_entry(self):
        write_session(
            self.trace_dir,
            "b1b2c3d4e5f60718",
            [
                event(EventType.SESSION_START, 0, {"mode": "claude-code-hooks", "provider": "claude"}),
                event(EventType.ASSISTANT_RESPONSE, 5, {"hook_event": "stop", "provider": "codex"}),
                event(EventType.SESSION_END, 10),
            ],
            agent_name="claude-code",
        )

        _, payload = self.health_json("b1b2c3d4e5f60718")

        self.assertEqual(payload["provider"], "")
        self.assertIsNone(payload["capture_matrix"]["entry"])
        self.assertIn("(claude, codex)", payload["capture_matrix"]["note"])
        self.assertEqual(self.messages(payload, "provider_limitation"), [])


class TestDetectProvider(unittest.TestCase):
    def detect(self, events=(), agent_name="", command=""):
        return detect_provider(SessionMeta(agent_name=agent_name, command=command), list(events))

    def test_legacy_session_start_without_provider_key(self):
        found = self.detect([event(EventType.SESSION_START, 0, {"mode": "claude-code-hooks"})])
        self.assertEqual((found.provider, found.source), ("claude", "hook_events"))

    def test_stop_event_names_the_provider(self):
        found = self.detect([
            event(EventType.ASSISTANT_RESPONSE, 0, {"hook_event": "stop", "provider": "codex"}),
        ])
        self.assertEqual((found.provider, found.source), ("codex", "hook_events"))

    def test_llm_provider_fields_are_not_hook_evidence(self):
        found = self.detect([event(EventType.LLM_REQUEST, 0, {"provider": "codex"})])
        self.assertEqual(found.provider, "")

    def test_conflict_is_reported(self):
        found = self.detect([
            event(EventType.SESSION_START, 0, {"mode": "claude-code-hooks", "provider": "claude"}),
            event(EventType.ASSISTANT_RESPONSE, 1, {"hook_event": "stop", "provider": "codex"}),
        ])
        self.assertEqual(found.provider, "")
        self.assertEqual(found.conflicting, ("claude", "codex"))

    def test_only_implicit_hook_metadata_is_trusted(self):
        found = self.detect(agent_name="openai-codex", command="openai-codex (no session-start)")
        self.assertEqual((found.provider, found.source), ("codex", "hook_metadata"))
        found = self.detect(agent_name="github-copilot", command="github-copilot (no session-start)")
        self.assertEqual((found.provider, found.source), ("copilot", "hook_metadata"))
        for agent_name, command in (
            ("claude-code", "imported from session.jsonl"),
            ("claude-code", "claude-code (backfilled)"),
            ("github-copilot", "github-copilot (backfilled)"),
            ("claude-code", "python agent.py"),
            ("my-agent", "my-agent (no session-start)"),
        ):
            with self.subTest(command=command):
                self.assertEqual(self.detect(agent_name=agent_name, command=command).provider, "")


class TestBlindSpots(unittest.TestCase):
    def test_known_gaps_then_unavailable_coverage_in_order(self):
        entry = {
            "id": "x",
            "name": "X Agent",
            "capture_method": "cli_hooks",
            "coverage": {
                "a": "captured",
                "b": "unavailable",
                "c": "provider_defined",
                "d": "unavailable",
            },
            "known_gaps": ["  first gap  ", "", "   ", "second gap"],
        }

        spots = blind_spots_for(entry)

        self.assertEqual([s.text for s in spots], [
            "first gap",
            "second gap",
            "Capture matrix: b is unavailable for X Agent.",
            "Capture matrix: d is unavailable for X Agent.",
        ])
        self.assertEqual([s.source for s in spots], ["known_gaps", "known_gaps", "coverage", "coverage"])
        self.assertEqual([s.capability for s in spots], ["", "", "b", "d"])

    def test_entry_without_gaps_or_unavailable_coverage_declares_nothing(self):
        entry = {"id": "x", "capture_method": "m", "coverage": {"a": "provider_defined"}, "known_gaps": []}
        self.assertEqual(blind_spots_for(entry), ())

    def test_packaged_entries_only_yield_declared_limitations(self):
        for entry in load_capture_matrix().providers:
            gaps = {gap.strip() for gap in entry["known_gaps"]}
            missing = {cap for cap, value in entry["coverage"].items() if value == "unavailable"}
            for spot in blind_spots_for(entry):
                with self.subTest(provider=entry["id"], spot=spot.text):
                    if spot.source == "known_gaps":
                        self.assertIn(spot.text, gaps)
                    else:
                        self.assertIn(spot.capability, missing)


class TestCaptureMatrixLoader(unittest.TestCase):
    class _Resource:
        def __init__(self, raw):
            self.raw = raw

        def joinpath(self, _name):
            return self

        def read_bytes(self):
            return self.raw

    def load(self, raw):
        with patch.object(health.resources, "files", return_value=self._Resource(raw)):
            return load_capture_matrix()

    def test_packaged_matrix_is_versioned_by_content(self):
        matrix = load_capture_matrix()
        digest = hashlib.sha256(MATRIX_FILE.read_bytes()).hexdigest()
        self.assertEqual(matrix.version, f"sha256:{digest}")
        self.assertEqual(matrix.schema_version, RAW_MATRIX["schema_version"])
        self.assertEqual(matrix.entry("codex"), raw_entry("codex"))
        self.assertIsNone(matrix.entry("gemini"))

    def test_malformed_matrices_are_rejected(self):
        for raw in (
            b"not json",
            b"[]",
            b'{"schema_version": 2, "providers": []}',
            b'{"schema_version": 1, "providers": {}}',
            b'{"schema_version": 1, "providers": [{"id": "x"}]}',
            b'{"schema_version": 1, "providers": [{"id": "x", "capture_method": "m",'
            b' "coverage": {}, "known_gaps": [1]}]}',
            b'{"schema_version": 1, "providers": [{"id": "x", "capture_method": "m",'
            b' "coverage": {}, "known_gaps": [], "events_before_start": "user_prompt"}]}',
            b'{"schema_version": 1, "providers": [{"id": "x", "capture_method": "m",'
            b' "coverage": {}, "known_gaps": [], "events_before_start": ["not_an_event"]}]}',
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                self.load(raw)

    def test_events_before_start_is_optional(self):
        matrix = self.load(
            b'{"schema_version": 1, "providers": [{"id": "x", "capture_method": "m",'
            b' "coverage": {}, "known_gaps": [], "events_before_start": ["user_prompt"]},'
            b' {"id": "y", "capture_method": "m", "coverage": {}, "known_gaps": []}]}'
        )
        self.assertEqual(matrix.entry("x")["events_before_start"], ["user_prompt"])
        self.assertNotIn("events_before_start", matrix.entry("y"))


class TestSessionFinalized(_HealthBase):
    def entry(self, session_end):
        return {"id": "claude", "capture_method": "m", "coverage": {"session_end": session_end},
                "known_gaps": []}

    def test_rules(self):
        store = TraceStore(self.trace_dir)
        store.base_dir.mkdir(parents=True, exist_ok=True)
        (store.base_dir / ".active-session").write_text("c1c2c3d4e5f60718\n", encoding="utf-8")
        open_meta = SessionMeta(session_id="c1c2c3d4e5f60718")
        ended_meta = SessionMeta(session_id="c1c2c3d4e5f60718", ended_at=T0)

        self.assertIs(session_finalized_for(store, "c1c2c3d4e5f60718", ended_meta, None), True)
        self.assertIs(
            session_finalized_for(store, "c1c2c3d4e5f60718", open_meta, self.entry("captured")),
            False,
        )
        self.assertIsNone(
            session_finalized_for(store, "c1c2c3d4e5f60718", open_meta, self.entry("unavailable")))
        self.assertIsNone(session_finalized_for(store, "c1c2c3d4e5f60718", open_meta, None))
        self.assertIsNone(
            session_finalized_for(store, "d1d2d3d4e5f60718", open_meta, self.entry("captured")))

    def test_symlinked_marker_is_ignored(self):
        store = TraceStore(self.trace_dir)
        store.base_dir.mkdir(parents=True, exist_ok=True)
        target = Path(self.trace_dir).parent / "elsewhere"
        target.write_text("c1c2c3d4e5f60718", encoding="utf-8")
        try:
            os.symlink(target, store.base_dir / ".active-session")
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable")
        meta = SessionMeta(session_id="c1c2c3d4e5f60718")

        self.assertIsNone(
            session_finalized_for(store, "c1c2c3d4e5f60718", meta, self.entry("captured")))


class TestStatusAndExitCodes(_HealthBase):
    def test_empty_session_is_unknown_and_fails(self):
        write_session(self.trace_dir, "e1e2e3d4e5f60718", [], ended=False)

        code, payload = self.health_json("e1e2e3d4e5f60718")

        self.assertEqual(payload["status"], "unknown")
        self.assertEqual(payload["event_count"], 0)
        self.assertEqual(code, 1)

    def test_structurally_broken_session_is_invalid_and_fails(self):
        write_session(self.trace_dir, "f1f2f3d4e5f60718", [
            event(EventType.SESSION_START, 0, {"mode": "claude-code-hooks", "provider": "claude"}),
            event(EventType.SESSION_START, 1, {"mode": "claude-code-hooks", "provider": "claude"}),
            event(EventType.SESSION_END, 2),
        ])

        code, out, err = self.run_health("f1f2f3d4e5f60718")

        self.assertEqual(err, "")
        self.assertTrue(out.startswith("Evidence health: invalid\n"))
        self.assertIn("duplicate_session_start", out)
        self.assertEqual(code, 1)

    def test_gap_free_entry_prints_none_declared(self):
        hook_session("claudeend0000002", "claude")
        store = TraceStore(self.trace_dir)
        matrix = CaptureMatrix(schema_version=1, version="sha256:test", providers=({
            "id": "claude",
            "name": "Claude Code",
            "capture_method": "cli_hooks",
            "verification_status": "verified",
            "coverage": {"session_end": "captured"},
            "known_gaps": [],
        },))

        report = assess_session(
            store,
            "claudeend0000002",
            store.load_meta("claudeend0000002"),
            store.load_events("claudeend0000002"),
            matrix,
        )
        out = io.StringIO()
        format_health_text(report, out)

        self.assertEqual(report.result.status, EvidenceHealthStatus.HEALTHY)
        self.assertEqual(report.result.capture_matrix_version, "sha256:test")
        self.assertEqual(report.exit_code, 0)
        self.assertIn("Provider limitations (0):\n  none declared\n", out.getvalue())
        self.assertIn("verification_status verified", out.getvalue())


class TestOutput(_HealthBase):
    def test_json_shape(self):
        hook_session("codexsession0002", "codex", end=False)

        _, payload = self.health_json("codexsession0002")

        self.assertEqual(set(payload), TOP_LEVEL_KEYS)
        self.assertEqual(payload["session_id"], "codexsession0002")
        self.assertEqual(
            set(payload["capture_matrix"]),
            {"schema_version", "version", "entry", "blind_spots", "note"},
        )
        self.assertEqual(set(payload["capture_matrix"]["entry"]), ENTRY_KEYS)
        for spot in payload["capture_matrix"]["blind_spots"]:
            expected = {"source", "text", "capability"} if spot["source"] == "coverage" else {"source", "text"}
            self.assertEqual(set(spot), expected)
        for reason in payload["reasons"]:
            self.assertEqual(set(reason), {"code", "message", "kind", "event_id"})

    def test_text_lists_status_reasons_then_matrix(self):
        hook_session("codexsession0003", "codex", end=False)

        code, out, err = self.run_health("codexsession0003")

        self.assertEqual(err, "")
        self.assertEqual(code, 0)
        self.assertEqual(out.splitlines()[0], "Evidence health: partial")
        positions = [
            out.index(marker)
            for marker in ("Observed (1):", "Provider limitations (", "Capture matrix: OpenAI Codex (codex)",
                           "Matrix version: sha256:")
        ]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("Provider:  codex (from hook events)\n", out)
        self.assertIn("Finalized: unknown\n", out)
        self.assertIn("  - missing_session_end: ", out)
        self.assertIn(f"verification_status {raw_entry('codex')['verification_status']}", out)


class TestSessionResolution(_HealthBase):
    def test_defaults_to_latest_session(self):
        write_session(self.trace_dir, "a0a0a0a0a0a0a0a1", [event(EventType.SESSION_START, 0)])
        write_session(self.trace_dir, "a0a0a0a0a0a0a0a2", [event(EventType.SESSION_START, 0)],
                      started_at=T0 + 100)

        _, payload = self.health_json()

        self.assertEqual(payload["session_id"], "a0a0a0a0a0a0a0a2")

    def test_prefix_resolves(self):
        write_session(self.trace_dir, "a0a0a0a0a0a0a0a1", [event(EventType.SESSION_START, 0)])

        _, payload = self.health_json("a0a0a0")

        self.assertEqual(payload["session_id"], "a0a0a0a0a0a0a0a1")

    def test_missing_and_invalid_sessions_fail_cleanly(self):
        write_session(self.trace_dir, "a0a0a0a0a0a0a0a1", [event(EventType.SESSION_START, 0)])
        for requested in ("zzzz", "../x"):
            with self.subTest(requested=requested):
                code, out, err = self.run_health(requested)
                self.assertEqual(code, 1)
                self.assertEqual(out, "")
                self.assertEqual(err, f"Session not found: {requested}\n")

    def test_empty_store_fails(self):
        code, out, err = self.run_health()
        self.assertEqual((code, out, err), (1, "", "No sessions found.\n"))

    def test_matrix_load_failure_fails(self):
        write_session(self.trace_dir, "a0a0a0a0a0a0a0a1", [event(EventType.SESSION_START, 0)])
        with patch.object(health, "load_capture_matrix", side_effect=ValueError("bad matrix")):
            code, out, err = self.run_health("a0a0a0a0a0a0a0a1")
        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        self.assertEqual(err, "Could not load the packaged capture matrix: bad matrix\n")


class TestCliRegistration(_HealthBase):
    def test_parser(self):
        parser = build_parser()
        args = parser.parse_args(["health"])
        self.assertEqual((args.command, args.session_id, args.format), ("health", None, "text"))
        args = parser.parse_args(["health", "abc", "--format", "json"])
        self.assertEqual((args.session_id, args.format), ("abc", "json"))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["health", "--format", "yaml"])

    def test_telemetry_reports_only_the_command_name(self):
        args = build_parser().parse_args(["health", "private-session", "--format", "json"])
        properties = _telemetry_command_properties(args)
        self.assertEqual(_sanitise_properties(CLI_COMMAND_COMPLETED, properties), {"command": "health"})

    def test_main_dispatches_and_exits_with_the_health_code(self):
        hook_session("claudeend0000003", "claude")
        argv = ["agent-strace", "--trace-dir", self.trace_dir, "health", "claudeend0000003"]
        out = io.StringIO()
        with patch.object(sys, "argv", argv), \
                patch("agent_trace.cli._product_telemetry.capture") as capture, \
                patch("agent_trace.cli._product_telemetry.maybe_show_telemetry_notice"), \
                redirect_stdout(out), \
                self.assertRaises(SystemExit) as raised:
            main()

        self.assertEqual(raised.exception.code, 0)
        self.assertTrue(out.getvalue().startswith("Evidence health: "))
        properties = capture.call_args.args[1]
        self.assertEqual(properties["command"], "health")
        self.assertNotIn("session_id", properties)


if __name__ == "__main__":
    unittest.main()
