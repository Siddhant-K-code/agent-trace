"""Replay recorded provider hook payloads and check them against the capture matrix.

Each fixture in ``tests/fixtures/capture_matrix/`` is a sanitized recording of
a real headless CLI run with agent-strace hooks installed. Hook names, event
order, payload keys and value types are as the provider emitted them; ids,
paths, prompts, tool output, model names and timestamps are synthetic.

Only the hooks that ``agent-strace setup`` registers are replayed through
``hook_main``. The rest show what the provider exposes that agent-strace does
not record. A verified capture-matrix entry names this file as its
``fixture_test_reference``. Older recordings stay as legacy fixtures so the
hooks keep handling payloads from earlier CLI releases.
"""

import argparse
import io
import json
import os
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from agent_trace.cli import _codex_hooks_config, _copilot_hooks_config
from agent_trace.health import assess_session, load_capture_matrix
from agent_trace.hooks import _PROVIDER_ENV, IMPLICIT_SESSION_SOURCE, hook_main
from agent_trace.models import EventType
from agent_trace.store import TraceStore
from agent_trace.subagent import build_tree

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "capture_matrix"
REFERENCE = "tests/test_capture_matrix_fixtures.py"
FIXTURES = {
    "copilot": "copilot-1.0.103-0.json",
    "codex": "codex-0.163.0-alpha.11.json",
}
LEGACY_COPILOT_FIXTURE = "copilot-1.0.94-3.json"
SYNTHETIC_ID = re.compile(r"^[cd]\d{7}-0000-4000-8000-\d{12}$")


def load_fixture(name):
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def registered_hooks(provider):
    """Return ``{provider hook event: (matcher, hook subcommand)}`` from setup."""
    args = argparse.Namespace(redact=False, no_redact=False)
    registered = {}
    if provider == "copilot":
        for event, entries in _copilot_hooks_config(args)["hooks"].items():
            registered[event] = ("", entries[0]["command"].split()[-1])
    elif provider == "codex":
        for event, groups in _codex_hooks_config(args)["hooks"].items():
            group = groups[0]
            registered[event] = (group.get("matcher", ""), group["hooks"][0]["command"].split()[-1])
    else:
        raise AssertionError(f"no hook config for {provider}")
    return registered


def matcher_applies(matcher, event, payload):
    if matcher in ("", "*"):
        return True
    value = payload.get("source") if event == "SessionStart" else payload.get("tool_name", "")
    return re.fullmatch(matcher, str(value or "")) is not None


def observed_capabilities(events):
    """Map recorded events to capture-matrix capabilities."""
    observed = set()
    for item in events:
        kind = item.event_type
        if kind == EventType.SESSION_START:
            observed.add("session_start")
        elif kind == EventType.SESSION_END:
            observed.add("session_end")
        elif kind == EventType.USER_PROMPT and item.data.get("prompt"):
            observed.add("prompt")
        elif kind == EventType.ASSISTANT_RESPONSE:
            if item.data.get("hook_event") == "stop":
                observed.add("stop")
            if item.data.get("text"):
                observed.add("response")
        elif kind in (EventType.TOOL_CALL, EventType.TOOL_RESULT):
            observed.add("tool")
        elif kind == EventType.ERROR:
            observed.add("error")
        elif kind == EventType.FILE_WRITE:
            observed.add("file")
    return observed


class _ReplayChecks:
    """Checks shared by every provider fixture; mixed into a TestCase."""

    provider = ""
    fixture_name = ""

    @classmethod
    def setUpClass(cls):
        cls.fixture_name = cls.fixture_name or FIXTURES[cls.provider]
        cls.fixture = load_fixture(cls.fixture_name)
        cls.matrix = load_capture_matrix()
        cls.entry = cls.matrix.entry(cls.provider)

    def setUp(self):
        env = patch.dict(os.environ, {"DO_NOT_TRACK": "1"}, clear=False)
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

    def replay(self, scenario):
        """Replay one scenario into a fresh store; return (store, skipped events)."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        trace_dir = str(Path(tmp.name) / ".agent-traces")
        os.environ["AGENT_TRACE_DIR"] = trace_dir
        registered = registered_hooks(self.provider)
        skipped = []
        for hook in self.fixture["scenarios"][scenario]["hooks"]:
            event, payload = hook["event"], hook["payload"]
            if event not in registered or not matcher_applies(registered[event][0], event, payload):
                skipped.append(event)
                continue
            err = io.StringIO()
            with patch.object(sys, "stdin", io.StringIO(json.dumps(payload))), redirect_stderr(err):
                hook_main(["--provider", self.provider, registered[event][1]])
            self.assertEqual(err.getvalue(), "", f"{scenario}: {event}")
        return TraceStore(trace_dir), skipped

    def sessions(self, store):
        return {meta.session_id: meta for meta in store.list_sessions()}

    def health(self, store, session_id):
        return assess_session(
            store,
            session_id,
            store.load_meta(session_id),
            store.load_events(session_id),
            self.matrix,
        )

    @staticmethod
    def codes(report, kind):
        return [r.code for r in report.result.reasons if r.kind.value == kind]

    def test_entry_is_verified_by_this_fixture(self):
        self.assertIsNotNone(self.entry)
        self.assertEqual(self.entry["verification_status"], "verified")
        self.assertEqual(self.entry["tested_version"], self.fixture["tested_version"])
        self.assertEqual(self.entry["last_verified"], self.fixture["recorded"])
        self.assertEqual(self.entry["fixture_test_reference"], REFERENCE)
        self.assertEqual(self.fixture["provider"], self.provider)

    def test_fixture_values_are_synthetic(self):
        raw = (FIXTURE_DIR / self.fixture_name).read_text(encoding="utf-8")
        for marker in ("/Users/", "/home/", "/private/", "/var/folders/", "/tmp/", "\\Users\\"):
            self.assertNotIn(marker, raw)
        for scenario in self.fixture["scenarios"].values():
            for hook in scenario["hooks"]:
                payload = hook["payload"]
                for key in (
                    "sessionId", "session_id", "agentId", "agent_id",
                    "parentAgentId", "parent_agent_id",
                ):
                    if key in payload:
                        self.assertRegex(payload[key], SYNTHETIC_ID)
                self.assertEqual(payload.get("cwd"), "/workspace/project")

    def test_every_registered_hook_fired_in_the_recording(self):
        fired = {
            hook["event"]
            for scenario in self.fixture["scenarios"].values()
            for hook in scenario["hooks"]
        }
        self.assertLessEqual(set(registered_hooks(self.provider)), fired)

    def test_observed_coverage_matches_the_entry(self):
        observed = set()
        for scenario in self.fixture["scenarios"]:
            store, _ = self.replay(scenario)
            for session_id in self.sessions(store):
                observed |= observed_capabilities(store.load_events(session_id))
        for capability, value in self.entry["coverage"].items():
            with self.subTest(capability=capability, value=value):
                if value == "captured":
                    self.assertIn(capability, observed)
                elif value == "unavailable":
                    self.assertNotIn(capability, observed)

    def test_no_session_is_invalid_or_unmapped(self):
        for scenario in self.fixture["scenarios"]:
            store, _ = self.replay(scenario)
            for session_id in self.sessions(store):
                with self.subTest(scenario=scenario, session=session_id):
                    report = self.health(store, session_id)
                    self.assertEqual(report.result.status.value, "partial")
                    self.assertEqual(report.provider.provider, self.provider)
                    self.assertIs(report.entry, self.entry)
                    self.assertEqual(report.result.capture_matrix_version, self.matrix.version)


class TestCopilotFixture(_ReplayChecks, unittest.TestCase):
    provider = "copilot"

    def main_and_child(self, store):
        sessions = self.sessions(store)
        main = [m for m in sessions.values() if IMPLICIT_SESSION_SOURCE not in m.command]
        child = [m for m in sessions.values() if IMPLICIT_SESSION_SOURCE in m.command]
        return main, child

    def test_session_start_precedes_prompt_in_this_recording(self):
        # Copilot does not fire the two in a fixed order: 1.0.94-3 always sent
        # userPromptSubmitted first, 1.0.103-0 did so in some runs, so the
        # entry still declares it (see TestCopilotLegacyFixture).
        self.assertEqual(self.entry["events_before_start"], ["user_prompt"])
        for scenario in ("tool", "tool_failure", "permission_denied", "subagent"):
            store, _ = self.replay(scenario)
            (main,), _ = self.main_and_child(store)
            events = store.load_events(main.session_id)
            with self.subTest(scenario=scenario):
                self.assertEqual(
                    [e.event_type for e in events[:2]],
                    [EventType.SESSION_START, EventType.USER_PROMPT],
                )
                self.assertEqual(events[-1].event_type, EventType.SESSION_END)
                self.assertEqual(main.agent_name, "github-copilot")
                self.assertIsNotNone(main.ended_at)
                report = self.health(store, main.session_id)
                self.assertNotIn("events_before_session_start", self.codes(report, "observed"))
                self.assertNotIn(
                    "declared_events_before_session_start",
                    self.codes(report, "provider_limitation"),
                )
                self.assertIs(report.session_finalized, True)

    def test_tool_result_pairs_by_name_without_a_call_id(self):
        store, skipped = self.replay("tool")
        self.assertEqual(skipped, [])
        (main,), child = self.main_and_child(store)
        self.assertEqual(child, [])
        events = store.load_events(main.session_id)
        call = next(e for e in events if e.event_type == EventType.TOOL_CALL)
        result = next(e for e in events if e.event_type == EventType.TOOL_RESULT)
        self.assertNotIn("tool_use_id", call.data)
        self.assertEqual(result.parent_id, call.event_id)
        self.assertEqual(self.codes(self.health(store, main.session_id), "observed"), [])

    def test_failed_tool_is_an_error_event(self):
        store, _ = self.replay("tool_failure")
        (main,), _ = self.main_and_child(store)
        errors = [e for e in store.load_events(main.session_id) if e.event_type == EventType.ERROR]
        self.assertEqual(len(errors), 1)
        self.assertEqual(main.errors, 1)

    def test_stop_has_no_response_text_model_or_tokens(self):
        store, _ = self.replay("tool")
        (main,), _ = self.main_and_child(store)
        events = store.load_events(main.session_id)
        stops = [e for e in events if e.event_type == EventType.ASSISTANT_RESPONSE]
        self.assertEqual([s.data["text"] for s in stops], [""])
        self.assertEqual(stops[0].data["stop_reason"], "end_turn")
        start = next(e for e in events if e.event_type == EventType.SESSION_START)
        self.assertEqual(start.data["model"], "")
        self.assertEqual(main.total_tokens, 0)

    def test_denied_tool_leaves_an_unpaired_call(self):
        # In this recording the model retried the denied tool once.
        store, skipped = self.replay("permission_denied")
        self.assertEqual(
            skipped,
            ["permissionRequest", "notification", "permissionRequest", "notification"],
        )
        (main,), _ = self.main_and_child(store)
        kinds = [e.event_type for e in store.load_events(main.session_id)]
        self.assertEqual(kinds.count(EventType.TOOL_CALL), 2)
        self.assertNotIn(EventType.TOOL_RESULT, kinds)
        self.assertNotIn(EventType.ERROR, kinds)
        self.assertIn("unpaired_tool_call", self.codes(self.health(store, main.session_id), "observed"))

    def test_subagent_hooks_link_the_child_to_the_parent(self):
        store, skipped = self.replay("subagent")
        self.assertEqual(skipped, [])
        (main,), (child,) = self.main_and_child(store)

        hooks = self.fixture["scenarios"]["subagent"]["hooks"]
        for event in ("subagentStart", "subagentStop"):
            payload = next(h["payload"] for h in hooks if h["event"] == event)
            self.assertTrue(payload["agentId"].startswith(child.session_id))
            self.assertTrue(payload["sessionId"].startswith(main.session_id))

        main_events = store.load_events(main.session_id)
        (task_call,) = [e for e in main_events if e.event_type == EventType.TOOL_CALL]
        self.assertEqual(task_call.data["tool_name"], "task")
        (start,) = [e for e in main_events if e.event_type == EventType.SUBAGENT_START]
        (stop,) = [e for e in main_events if e.event_type == EventType.SUBAGENT_STOP]
        for event in (start, stop):
            self.assertEqual(event.data["child_session_id"], child.session_id)
            self.assertEqual(event.data["agent_type"], "explore")
            self.assertEqual(event.parent_id, task_call.event_id)
            self.assertNotIn("response", event.data)
        # The child has no metadata yet when subagentStart fires.
        self.assertEqual(start.data["child_link"], "missing_child")
        self.assertEqual(stop.data["child_link"], "linked")
        self.assertEqual(stop.data["stop_reason"], "end_turn")

        self.assertEqual(child.command, f"github-copilot ({IMPLICIT_SESSION_SOURCE})")
        self.assertEqual(child.parent_session_id, main.session_id)
        self.assertEqual(child.parent_event_id, task_call.event_id)
        self.assertEqual(child.depth, 1)
        self.assertIsNone(child.ended_at)
        kinds = [e.event_type for e in store.load_events(child.session_id)]
        self.assertNotIn(EventType.SESSION_START, kinds)
        self.assertNotIn(EventType.SESSION_END, kinds)
        self.assertIn(EventType.USER_PROMPT, kinds)
        self.assertIn(EventType.TOOL_CALL, kinds)

        tree = build_tree(store, main.session_id)
        self.assertEqual([c.meta.session_id for c in tree.children], [child.session_id])

        report = self.health(store, child.session_id)
        self.assertEqual(report.provider.source, "hook_events")
        self.assertIsNone(report.session_finalized)
        self.assertEqual(
            self.codes(report, "observed"),
            ["missing_session_start", "missing_session_end"],
        )


class TestCopilotLegacyFixture(TestCopilotFixture):
    """Copilot CLI 1.0.94-3: prompt before sessionStart, and a subagentStart
    payload that does not name the child. Some 1.0.103-0 runs still send this
    shape, so the hooks must keep handling it."""

    fixture_name = LEGACY_COPILOT_FIXTURE

    def test_entry_is_verified_by_this_fixture(self):
        self.assertEqual(self.fixture["provider"], self.provider)
        self.assertEqual(self.fixture["tested_version"], "1.0.94-3")
        self.assertNotEqual(self.entry["tested_version"], self.fixture["tested_version"])

    def test_session_start_precedes_prompt_in_this_recording(self):
        self.skipTest("1.0.94-3 fired the prompt first; see the declared-order test")

    def test_prompt_precedes_session_start_and_health_accepts_the_declared_order(self):
        for scenario in ("tool", "tool_failure", "permission_denied", "subagent"):
            store, _ = self.replay(scenario)
            (main,), _ = self.main_and_child(store)
            events = store.load_events(main.session_id)
            with self.subTest(scenario=scenario):
                self.assertEqual(
                    [e.event_type for e in events[:2]],
                    [EventType.USER_PROMPT, EventType.SESSION_START],
                )
                report = self.health(store, main.session_id)
                self.assertNotIn("events_before_session_start", self.codes(report, "observed"))
                self.assertIn(
                    "declared_events_before_session_start",
                    self.codes(report, "provider_limitation"),
                )

    def test_denied_tool_leaves_an_unpaired_call(self):
        store, skipped = self.replay("permission_denied")
        self.assertEqual(skipped, ["permissionRequest", "notification"])
        (main,), _ = self.main_and_child(store)
        kinds = [e.event_type for e in store.load_events(main.session_id)]
        self.assertEqual(kinds.count(EventType.TOOL_CALL), 1)
        self.assertIn("unpaired_tool_call", self.codes(self.health(store, main.session_id), "observed"))

    def test_subagent_hooks_link_the_child_to_the_parent(self):
        store, skipped = self.replay("subagent")
        self.assertEqual(skipped, [])
        (main,), (child,) = self.main_and_child(store)
        main_events = store.load_events(main.session_id)
        (task_call,) = [e for e in main_events if e.event_type == EventType.TOOL_CALL]
        (start,) = [e for e in main_events if e.event_type == EventType.SUBAGENT_START]
        (stop,) = [e for e in main_events if e.event_type == EventType.SUBAGENT_STOP]
        self.assertNotIn("child_session_id", start.data)
        self.assertNotIn("child_link", start.data)
        self.assertEqual(start.parent_id, task_call.event_id)
        self.assertEqual(stop.data["child_session_id"], child.session_id)
        self.assertEqual(stop.data["child_link"], "linked")
        self.assertEqual(stop.parent_id, task_call.event_id)
        self.assertEqual(child.parent_session_id, main.session_id)
        self.assertEqual(child.parent_event_id, task_call.event_id)
        self.assertEqual(child.depth, 1)


class TestCodexFixture(_ReplayChecks, unittest.TestCase):
    provider = "codex"

    def only_session(self, store):
        (meta,) = self.sessions(store).values()
        return meta

    def test_session_end_finishes_every_session(self):
        for scenario in self.fixture["scenarios"]:
            store, skipped = self.replay(scenario)
            meta = self.only_session(store)
            events = store.load_events(meta.session_id)
            with self.subTest(scenario=scenario):
                self.assertEqual(skipped, [])
                self.assertEqual(events[-1].event_type, EventType.SESSION_END)
                self.assertEqual(events[-1].data["reason"], "other")
                self.assertEqual(
                    [e.event_type for e in events].count(EventType.SESSION_END), 1
                )
                self.assertIsNotNone(meta.ended_at)
                report = self.health(store, meta.session_id)
                self.assertIs(report.session_finalized, True)
                self.assertEqual(self.codes(report, "observed"), [])

    def test_tool_session_records_model_but_no_tokens(self):
        store, skipped = self.replay("tool")
        self.assertEqual(skipped, [])
        meta = self.only_session(store)
        events = store.load_events(meta.session_id)
        self.assertEqual(
            [e.event_type for e in events],
            [
                EventType.SESSION_START,
                EventType.USER_PROMPT,
                EventType.TOOL_CALL,
                EventType.TOOL_RESULT,
                EventType.ASSISTANT_RESPONSE,
                EventType.SESSION_END,
            ],
        )
        self.assertEqual(events[0].data["model"], "synthetic-model")
        self.assertEqual(events[2].data["tool_name"], "Bash")
        self.assertIn("command", events[2].data["arguments"])
        self.assertTrue(events[-2].data["text"])
        self.assertEqual(meta.total_tokens, 0)

    def test_failed_command_is_a_tool_result_not_an_error(self):
        store, _ = self.replay("tool_failure")
        meta = self.only_session(store)
        kinds = [e.event_type for e in store.load_events(meta.session_id)]
        self.assertNotIn(EventType.ERROR, kinds)
        self.assertEqual(kinds.count(EventType.TOOL_RESULT), 1)
        self.assertEqual(meta.errors, 0)

    def test_subagent_events_are_recorded_in_the_parent_session(self):
        for scenario in ("subagent", "subagent_start_before_wait"):
            with self.subTest(scenario=scenario):
                store, skipped = self.replay(scenario)
                self.assertEqual(skipped, [])
                meta = self.only_session(store)
                hooks = self.fixture["scenarios"][scenario]["hooks"]
                (agent_id,) = {
                    h["payload"]["agent_id"] for h in hooks if "agent_id" in h["payload"]
                }
                self.assertTrue(
                    all(h["payload"]["session_id"].startswith(meta.session_id) for h in hooks)
                )
                events = store.load_events(meta.session_id)
                tools = [e.data["tool_name"] for e in events if e.event_type == EventType.TOOL_CALL]
                self.assertEqual(len(tools), 3)
                self.assertIn("Bash", tools)

                kinds = [e.event_type for e in events]
                start = kinds.index(EventType.SUBAGENT_START)
                stop = kinds.index(EventType.SUBAGENT_STOP)
                self.assertEqual(kinds.count(EventType.SUBAGENT_START), 1)
                self.assertEqual(kinds.count(EventType.SUBAGENT_STOP), 1)
                bash = next(
                    i for i, e in enumerate(events)
                    if e.event_type == EventType.TOOL_CALL and e.data["tool_name"] == "Bash"
                )
                self.assertLess(start, bash)
                self.assertLess(bash, stop)
                for event in (events[start], events[stop]):
                    self.assertEqual(event.data["agent_id"], agent_id)
                    self.assertEqual(event.data["agent_type"], "default")
                    self.assertEqual(event.data["provider"], "codex")
                    self.assertNotIn("child_session_id", event.data)
                    self.assertNotIn("child_link", event.data)
                    self.assertEqual(event.parent_id, "")
                self.assertNotIn("last_assistant_message", events[stop].data)
                self.assertEqual(events[stop].data["text"], "line-1")
                self.assertEqual(
                    events[stop].data["agent_transcript_path"],
                    f"/codex-home/sessions/rollout-{agent_id}.jsonl",
                )
                self.assertNotIn("text", events[start].data)
                self.assertEqual(meta.parent_session_id, "")

                (bash_result,) = [
                    e for e in events
                    if e.event_type == EventType.TOOL_RESULT and e.data["tool_name"] == "Bash"
                ]
                self.assertEqual(bash_result.parent_id, events[bash].event_id)
                for event in (events[bash], bash_result):
                    self.assertEqual(event.data["agent_id"], agent_id)
                    self.assertEqual(event.data["agent_type"], "default")
                for event in events:
                    if (
                        event.event_type in (EventType.TOOL_CALL, EventType.TOOL_RESULT)
                        and event.data["tool_name"] != "Bash"
                    ):
                        self.assertNotIn("agent_id", event.data)
                        self.assertNotIn("agent_type", event.data)

    def test_subagent_start_order_differs_between_recordings(self):
        def order(scenario):
            return [
                (h["event"], h["payload"].get("tool_name", ""))
                for h in self.fixture["scenarios"][scenario]["hooks"]
            ]

        spawn = ("PostToolUse", "collaborationspawn_agent")
        wait = ("PreToolUse", "collaborationwait_agent")
        start = ("SubagentStart", "")
        first, second = order("subagent"), order("subagent_start_before_wait")
        self.assertEqual(first.index(spawn) + 1, first.index(wait))
        self.assertEqual(first.index(wait) + 1, first.index(start))
        self.assertEqual(second.index(spawn) + 1, second.index(start))
        self.assertEqual(second.index(start) + 1, second.index(wait))
        self.assertEqual(sorted(first), sorted(second))


if __name__ == "__main__":
    unittest.main()
