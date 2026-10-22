"""Copilot subagentStart/subagentStop hooks: parent events and child linking."""

import argparse
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

from agent_trace.cli import _copilot_hooks_config
from agent_trace.hooks import _PROVIDER_ENV, IMPLICIT_SESSION_SOURCE, hook_main
from agent_trace.models import EventType, SessionMeta
from agent_trace.store import TraceStore
from agent_trace.subagent import build_tree, cmd_tree, format_session_tree

ROOT = Path(__file__).resolve().parents[1]
PARENT = "a0000001-0000-4000-8000-000000000001"
CHILD = "a0000002-0000-4000-8000-000000000002"
OTHER = "a0000003-0000-4000-8000-000000000003"
CWD = "/workspace/project"
TASK_ARGS = {
    "agent_type": "explore",
    "name": "synthetic-task",
    "description": "Synthetic subagent task",
    "prompt": "Synthetic subagent prompt.",
    "mode": "sync",
}


def local(session_id):
    return session_id[:16]


class _SubagentHookTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / ".agent-traces"
        env = patch.dict(
            os.environ,
            {"AGENT_TRACE_DIR": str(self.root), "DO_NOT_TRACK": "1"},
            clear=False,
        )
        env.start()
        self.addCleanup(env.stop)
        for name in (
            *_PROVIDER_ENV.values(),
            "AGENT_TRACE_REDACT",
            "AGENT_STRACE_TENANT_ID",
            "AGENT_STRACE_WORKSPACE",
            "AGENT_STRACE_ENDPOINT",
            "AGENT_STRACE_PARENT_SESSION",
        ):
            os.environ.pop(name, None)

    def dispatch(self, event, payload, provider="copilot", quiet=True):
        payload = {"cwd": CWD, **payload}
        err = io.StringIO()
        with patch.object(sys, "stdin", io.StringIO(json.dumps(payload))), redirect_stderr(err):
            hook_main(["--provider", provider, event])
        if quiet:
            self.assertEqual(err.getvalue(), "", event)
        return err.getvalue()

    def store(self):
        return TraceStore(self.root)

    def meta_bytes(self, session_id):
        return (self.root / local(session_id) / "meta.json").read_bytes()

    def events_bytes(self, session_id):
        return (self.root / local(session_id) / "events.ndjson").read_bytes()

    def events(self, session_id, event_type=None):
        events = self.store().load_events(local(session_id))
        if event_type is None:
            return events
        return [e for e in events if e.event_type == event_type]

    def start_parent(self, task_args=TASK_ARGS):
        self.dispatch("sessionStart", {"sessionId": PARENT, "source": "new"})
        self.dispatch("userPromptSubmitted", {"sessionId": PARENT, "prompt": "Synthetic prompt."})
        self.dispatch("preToolUse", {"sessionId": PARENT, "toolName": "task", "toolArgs": task_args})

    def subagent_start(self, child=CHILD, **extra):
        payload = {
            "sessionId": PARENT,
            "transcriptPath": "/copilot-home/session-state/parent/events.jsonl",
            "agentName": "explore",
            "agentType": "explore",
            "parentAgentId": PARENT,
        }
        if child:
            payload["agentId"] = child
        payload.update(extra)
        self.dispatch("subagentStart", payload)

    def child_runs(self, child=CHILD):
        ids = {"sessionId": child, "agentId": child, "agentType": "explore", "parentAgentId": PARENT}
        self.dispatch("userPromptSubmitted", {**ids, "prompt": "Synthetic subagent prompt."})
        view = {"path": f"{CWD}/notes.txt", "view_range": [1, 1]}
        self.dispatch("preToolUse", {**ids, "toolName": "view", "toolArgs": view})
        self.dispatch("postToolUse", {
            **ids,
            "toolName": "view",
            "toolArgs": view,
            "toolResult": {"resultType": "success", "textResultForLlm": "line-1"},
        })
        self.dispatch("agentStop", {**ids, "stopReason": "end_turn", "stop_hook_active": False})

    def subagent_stop(self, child=CHILD, **extra):
        payload = {
            "sessionId": PARENT,
            "agentId": child,
            "agentType": "explore",
            "agentName": "explore",
            "response": "Synthetic subagent response.",
            "stopReason": "end_turn",
            "stop_hook_active": False,
            "parentAgentId": PARENT,
        }
        payload.update(extra)
        self.dispatch("subagentStop", payload)

    def finish_parent(self):
        self.dispatch("postToolUse", {
            "sessionId": PARENT,
            "toolName": "task",
            "toolArgs": TASK_ARGS,
            "toolResult": {"resultType": "success", "textResultForLlm": "line-1"},
        })
        self.dispatch("agentStop", {"sessionId": PARENT, "stopReason": "end_turn"})
        self.dispatch("sessionEnd", {"sessionId": PARENT, "reason": "complete"})

    def task_call(self):
        (call,) = [
            e for e in self.events(PARENT, EventType.TOOL_CALL)
            if e.data["tool_name"] == "task"
        ]
        return call

    def last_stop(self):
        return self.events(PARENT, EventType.SUBAGENT_STOP)[-1]


class TestSubagentHookRegistration(unittest.TestCase):
    def test_setup_config_and_plugin_register_subagent_hooks(self):
        args = argparse.Namespace(redact=False, no_redact=False)
        configs = {
            "setup": _copilot_hooks_config(args),
            "plugin": json.loads((ROOT / "hooks.json").read_text(encoding="utf-8")),
        }
        for source, config in configs.items():
            hooks = config["hooks"]
            with self.subTest(source=source):
                self.assertEqual(
                    hooks["subagentStart"][0]["command"],
                    "agent-strace hook --provider copilot subagent-start",
                )
                self.assertEqual(
                    hooks["subagentStop"][0]["command"],
                    "agent-strace hook --provider copilot subagent-stop",
                )
                for name in ("subagentStart", "subagentStop"):
                    self.assertEqual(hooks[name][0]["type"], "command")
                    self.assertLessEqual(hooks[name][0]["timeoutSec"], 10)


class TestCopilotSubagentLink(_SubagentHookTest):
    def run_subagent(self):
        self.start_parent()
        self.subagent_start()
        self.child_runs()
        self.subagent_stop()
        self.finish_parent()

    def test_parent_records_start_and_stop_with_child_session(self):
        self.run_subagent()
        call = self.task_call()
        (start,) = self.events(PARENT, EventType.SUBAGENT_START)
        (stop,) = self.events(PARENT, EventType.SUBAGENT_STOP)
        for event in (start, stop):
            self.assertEqual(event.parent_id, call.event_id)
            self.assertEqual(event.data["child_session_id"], local(CHILD))
            self.assertEqual(event.data["provider"], "copilot")
            self.assertEqual(event.data["agent_type"], "explore")
            self.assertEqual(event.data["agent_name"], "explore")
            self.assertNotIn("response", event.data)
        self.assertEqual(start.data["hook_event"], "subagent_start")
        self.assertEqual(start.data["child_link"], "missing_child")
        self.assertNotIn("stop_reason", start.data)
        self.assertEqual(stop.data["hook_event"], "subagent_stop")
        self.assertEqual(stop.data["child_link"], "linked")
        self.assertEqual(stop.data["stop_reason"], "end_turn")

        kinds = [e.event_type for e in self.events(PARENT)]
        self.assertEqual(kinds.count(EventType.TOOL_CALL), 1)
        self.assertEqual(kinds.count(EventType.TOOL_RESULT), 1)
        self.assertLess(kinds.index(EventType.SUBAGENT_START), kinds.index(EventType.SUBAGENT_STOP))

    def test_child_meta_is_linked_to_parent_and_task_call(self):
        self.run_subagent()
        child = self.store().load_meta(local(CHILD))
        self.assertEqual(child.parent_session_id, local(PARENT))
        self.assertEqual(child.parent_event_id, self.task_call().event_id)
        self.assertEqual(child.depth, 1)
        self.assertEqual(child.command, f"github-copilot ({IMPLICIT_SESSION_SOURCE})")
        self.assertEqual(child.tool_calls, 1)
        parent = self.store().load_meta(local(PARENT))
        self.assertEqual(parent.parent_session_id, "")
        self.assertEqual(parent.depth, 0)

    def test_depth_is_parent_depth_plus_one(self):
        self.dispatch("sessionStart", {"sessionId": OTHER, "source": "new"})
        self.start_parent()
        store = self.store()
        parent = store.load_meta(local(PARENT))
        parent.parent_session_id = local(OTHER)
        parent.depth = 1
        store.update_meta(parent)
        self.subagent_start()
        self.child_runs()
        self.subagent_stop()
        self.assertEqual(self.store().load_meta(local(CHILD)).depth, 2)

    def test_child_events_are_never_rewritten(self):
        self.start_parent()
        self.subagent_start()
        self.child_runs()
        before = self.events_bytes(CHILD)
        self.subagent_stop()
        self.subagent_stop()
        self.assertEqual(self.events_bytes(CHILD), before)
        self.assertEqual(self.store().load_meta(local(CHILD)).parent_session_id, local(PARENT))

    def test_relinking_is_idempotent(self):
        self.run_subagent()
        before = self.meta_bytes(CHILD)
        self.subagent_stop()
        self.assertEqual(self.last_stop().data["child_link"], "already_linked")
        self.assertEqual(self.meta_bytes(CHILD), before)

    def test_tree_shows_the_copilot_child(self):
        self.run_subagent()
        tree = build_tree(self.store(), local(PARENT))
        (node,) = tree.children
        self.assertEqual(node.meta.session_id, local(CHILD))
        self.assertEqual(node.depth, 1)

        out = io.StringIO()
        args = argparse.Namespace(trace_dir=str(self.root), session_id=local(PARENT), format="json")
        with redirect_stdout(out):
            self.assertEqual(cmd_tree(args), 0)
        data = json.loads(out.getvalue())
        self.assertEqual(data["total_sessions"], 2)
        (child,) = data["tree"]["children"]
        self.assertEqual(child["session_id"], local(CHILD))
        self.assertEqual(child["parent_event_id"], self.task_call().event_id)

        out = io.StringIO()
        format_session_tree(tree, out=out)
        lines = out.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith(local(PARENT)[:12]))
        self.assertIn(local(CHILD)[:12], lines[1])


class TestLinkSurvivesLaterMetaWrites(_SubagentHookTest):
    def setUp(self):
        super().setUp()
        self.start_parent()
        self.subagent_start()
        self.child_runs()
        self.subagent_stop()
        self.linked = self.store().load_meta(local(CHILD))
        self.assertEqual(self.linked.parent_session_id, local(PARENT))

    def assert_link_kept(self):
        meta = self.store().load_meta(local(CHILD))
        self.assertEqual(meta.parent_session_id, self.linked.parent_session_id)
        self.assertEqual(meta.parent_event_id, self.linked.parent_event_id)
        self.assertEqual(meta.depth, self.linked.depth)
        return meta

    def test_later_child_hooks_keep_the_link(self):
        ids = {"sessionId": CHILD}
        self.dispatch("preToolUse", {**ids, "toolName": "bash", "toolArgs": {"command": "true"}})
        self.dispatch("postToolUseFailure", {
            **ids, "toolName": "bash", "toolArgs": {"command": "true"}, "error": "exit 1",
        })
        self.dispatch("agentStop", {**ids, "stopReason": "end_turn"})
        meta = self.assert_link_kept()
        self.assertEqual(meta.tool_calls, 2)
        self.assertEqual(meta.errors, 1)

    def test_real_session_start_and_end_keep_the_link(self):
        self.dispatch("sessionStart", {"sessionId": CHILD, "source": "new"})
        meta = self.assert_link_kept()
        self.assertEqual(meta.command, "github-copilot (new)")
        self.dispatch("sessionEnd", {"sessionId": CHILD, "reason": "complete"})
        self.assertIsNotNone(self.assert_link_kept().ended_at)

    def test_store_update_meta_preserves_only_when_asked(self):
        store = self.store()
        stale = SessionMeta(**{**self.linked.__dict__, "parent_session_id": "", "parent_event_id": "", "depth": 0})
        store.update_meta(stale, preserve_parent_link=True)
        self.assert_link_kept()


class TestParentEventIdInference(_SubagentHookTest):
    def test_payload_without_child_id_is_linked_at_stop(self):
        # Copilot CLI 1.0.94-3: subagentStart does not name the child.
        self.start_parent()
        self.subagent_start(child="")
        self.child_runs()
        self.subagent_stop()
        call = self.task_call()
        (start,) = self.events(PARENT, EventType.SUBAGENT_START)
        self.assertNotIn("child_session_id", start.data)
        self.assertNotIn("child_link", start.data)
        self.assertEqual(start.parent_id, call.event_id)
        self.assertEqual(self.last_stop().data["child_link"], "linked")
        child = self.store().load_meta(local(CHILD))
        self.assertEqual(child.parent_event_id, call.event_id)

    def test_ambiguous_pending_task_calls_leave_parent_event_empty(self):
        self.start_parent()
        self.dispatch("preToolUse", {"sessionId": PARENT, "toolName": "task", "toolArgs": TASK_ARGS})
        self.subagent_start()
        self.child_runs()
        self.subagent_stop()
        self.assertEqual(self.last_stop().parent_id, "")
        self.assertEqual(self.last_stop().data["child_link"], "linked")
        child = self.store().load_meta(local(CHILD))
        self.assertEqual(child.parent_session_id, local(PARENT))
        self.assertEqual(child.parent_event_id, "")

    def test_contradicting_agent_type_leaves_parent_event_empty(self):
        self.start_parent(task_args={**TASK_ARGS, "agent_type": "general-purpose"})
        self.subagent_start()
        self.child_runs()
        self.subagent_stop()
        self.assertEqual(self.store().load_meta(local(CHILD)).parent_event_id, "")

    def test_pending_call_missing_from_parent_trace_is_not_used(self):
        self.dispatch("sessionStart", {"sessionId": PARENT, "source": "new"})
        self.child_runs()
        stale = {"0123456789abcdef": {"tool_name": "task", "timestamp": 0}}
        with patch("agent_trace.hooks._read_pending_calls", return_value=stale):
            self.subagent_stop()
        self.assertEqual(self.last_stop().parent_id, "")
        child = self.store().load_meta(local(CHILD))
        self.assertEqual(child.parent_session_id, local(PARENT))
        self.assertEqual(child.parent_event_id, "")

    def test_no_pending_task_call_leaves_parent_event_empty(self):
        self.dispatch("sessionStart", {"sessionId": PARENT, "source": "new"})
        self.child_runs()
        self.subagent_stop()
        child = self.store().load_meta(local(CHILD))
        self.assertEqual(child.parent_session_id, local(PARENT))
        self.assertEqual(child.parent_event_id, "")

    def test_later_stop_fills_an_empty_parent_event_id(self):
        self.dispatch("sessionStart", {"sessionId": PARENT, "source": "new"})
        self.child_runs()
        self.subagent_stop()
        self.dispatch("preToolUse", {"sessionId": PARENT, "toolName": "task", "toolArgs": TASK_ARGS})
        self.subagent_stop()
        self.assertEqual(
            self.store().load_meta(local(CHILD)).parent_event_id, self.task_call().event_id
        )


class TestLinkRefusals(_SubagentHookTest):
    def test_missing_child_is_not_created(self):
        self.start_parent()
        self.subagent_stop()
        stop = self.last_stop()
        self.assertEqual(stop.data["child_session_id"], local(CHILD))
        self.assertEqual(stop.data["child_link"], "missing_child")
        self.assertEqual(stop.parent_id, self.task_call().event_id)
        self.assertFalse((self.root / local(CHILD)).exists())
        self.assertEqual(
            [m.session_id for m in self.store().list_sessions()], [local(PARENT)]
        )

    def test_existing_different_parent_is_kept(self):
        self.dispatch("sessionStart", {"sessionId": OTHER, "source": "new"})
        self.start_parent()
        self.child_runs()
        store = self.store()
        child = store.load_meta(local(CHILD))
        child.parent_session_id = local(OTHER)
        child.parent_event_id = "synthetic-event"
        child.depth = 1
        store.update_meta(child)
        before = self.meta_bytes(CHILD)
        self.subagent_stop()
        self.assertEqual(self.last_stop().data["child_link"], "different_parent")
        self.assertEqual(self.meta_bytes(CHILD), before)

    def test_erased_child_is_not_linked(self):
        self.start_parent()
        self.child_runs()
        digest = hashlib.sha256(local(CHILD).encode()).hexdigest()
        (self.root / "tenant-deletions.ndjson").write_text(
            json.dumps({"tenant_id": "acme", "session_id_sha256": [digest]}) + "\n"
        )
        before = self.meta_bytes(CHILD)
        self.subagent_stop()
        self.assertEqual(self.last_stop().data["child_link"], "erased")
        self.assertEqual(self.meta_bytes(CHILD), before)

    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "root ignores file modes")
    def test_unreadable_erasure_audit_fails_closed(self):
        self.start_parent()
        self.child_runs()
        audit = self.root / "tenant-deletions.ndjson"
        audit.write_text("")
        audit.chmod(0)
        self.addCleanup(audit.chmod, 0o644)
        before = self.meta_bytes(CHILD)
        self.subagent_stop()
        self.assertEqual(self.last_stop().data["child_link"], "unreadable_erasure_audit")
        self.assertEqual(self.meta_bytes(CHILD), before)

    def _journal(self, name, record):
        journal = self.root / ".tenant-journal"
        journal.mkdir(exist_ok=True)
        (journal / name).write_text(json.dumps(record))

    def test_pending_journals_block_the_link(self):
        cases = (
            ("delete-op1.json", {"operation": "x", "sessions": [{"session_id": local(CHILD)}]},
             "pending_erasure"),
            ("tag-op1.json", {"operation": "x", "session_id": local(CHILD)}, "pending_tag"),
            ("delete-op1.json", "{truncated", "unreadable_journal"),
        )
        self.start_parent()
        self.child_runs()
        before = self.meta_bytes(CHILD)
        for name, record, reason in cases:
            with self.subTest(reason=reason):
                journal = self.root / ".tenant-journal"
                journal.mkdir(exist_ok=True)
                for path in journal.iterdir():
                    path.unlink()
                text = record if isinstance(record, str) else json.dumps(record)
                (journal / name).write_text(text)
                self.subagent_stop()
                self.assertEqual(self.last_stop().data["child_link"], reason)
                self.assertEqual(self.meta_bytes(CHILD), before)

    def test_tenant_mismatch_is_not_linked(self):
        with patch.dict(os.environ, {"AGENT_STRACE_TENANT_ID": "tenant-a"}):
            self.start_parent()
        with patch.dict(os.environ, {"AGENT_STRACE_TENANT_ID": "tenant-b"}):
            self.child_runs()
        before = self.meta_bytes(CHILD)
        with patch.dict(os.environ, {"AGENT_STRACE_TENANT_ID": "tenant-a"}):
            self.subagent_stop()
        stop = self.events(PARENT, EventType.SUBAGENT_STOP)[-1]
        self.assertEqual(stop.data["child_link"], "tenant_mismatch")
        self.assertEqual(self.meta_bytes(CHILD), before)

    def test_ancestor_child_is_not_linked(self):
        self.dispatch("sessionStart", {"sessionId": CHILD, "source": "new"})
        self.start_parent()
        store = self.store()
        parent = store.load_meta(local(PARENT))
        parent.parent_session_id = local(CHILD)
        parent.depth = 1
        store.update_meta(parent)
        before = self.meta_bytes(CHILD)
        self.subagent_stop()
        self.assertEqual(self.last_stop().data["child_link"], "cycle")
        self.assertEqual(self.meta_bytes(CHILD), before)

    def test_child_id_equal_to_parent_is_ignored(self):
        self.start_parent()
        self.subagent_stop(child=PARENT)
        stop = self.last_stop()
        self.assertNotIn("child_session_id", stop.data)
        self.assertNotIn("child_link", stop.data)
        self.assertEqual(self.store().load_meta(local(PARENT)).parent_session_id, "")

    def test_other_providers_do_not_link_by_agent_id(self):
        self.start_parent()
        self.child_runs()
        before = self.meta_bytes(CHILD)
        self.dispatch(
            "SubagentStop",
            {"session_id": PARENT, "agent_id": CHILD, "agent_type": "explore"},
            provider="claude",
        )
        self.assertEqual(self.meta_bytes(CHILD), before)


class TestSubagentHookFailures(_SubagentHookTest):
    def test_failed_meta_write_still_records_the_parent_event(self):
        self.start_parent()
        self.child_runs()
        before = self.meta_bytes(CHILD)
        with patch("agent_trace.hooks._write_atomic", side_effect=OSError("disk full")):
            self.subagent_stop()
        self.assertEqual(self.last_stop().data["child_link"], "link_failed")
        self.assertEqual(self.meta_bytes(CHILD), before)

    def test_handler_errors_never_fail_the_agent(self):
        self.start_parent()
        err = io.StringIO()
        payload = json.dumps({"sessionId": PARENT, "agentId": CHILD, "cwd": CWD})
        with patch("agent_trace.hooks._link_subagent_session", side_effect=RuntimeError("boom")), \
                patch.object(sys, "stdin", io.StringIO(payload)), redirect_stderr(err):
            with self.assertRaises(SystemExit) as raised:
                hook_main(["--provider", "copilot", "subagentStop"])
        self.assertEqual(raised.exception.code, 0)
        self.assertIn("agent-strace hook error", err.getvalue())

    def test_redaction_applies_to_subagent_events(self):
        os.environ["AGENT_TRACE_REDACT"] = "1"
        self.start_parent()
        self.dispatch("subagentStart", {
            "sessionId": PARENT, "agentId": CHILD, "agentType": "explore",
            "agentName": "ghp_" + "a" * 36,
        }, quiet=False)
        (start,) = self.events(PARENT, EventType.SUBAGENT_START)
        self.assertNotIn("ghp_" + "a" * 36, json.dumps(start.data))
        self.assertEqual(start.data["child_session_id"], local(CHILD))


if __name__ == "__main__":
    unittest.main()
