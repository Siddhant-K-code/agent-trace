"""Tests for agent-strace backfill-meta."""

import argparse
import hashlib
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from agent_trace.audit import verify_chain
from agent_trace.backfill import (
    BACKFILL_SOURCE,
    DEFAULT_AGENT_NAME,
    backfill_trace_dir,
    cmd_backfill_meta,
    derive_meta,
    infer_agent_name,
)
from agent_trace.cli import build_parser
from agent_trace.hooks import IMPLICIT_SESSION_SOURCE
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore
from agent_trace.tenancy import delete_tenant_data

T0 = 1_700_000_000.0


def _event(event_type, offset, data=None, **kwargs):
    kwargs.setdefault("tenant_id", "")
    return TraceEvent(event_type, timestamp=T0 + offset, data=data or {}, **kwargs)


def subagent_events(session_id, tenant_id="", provider="copilot"):
    """Events the hooks write for a Copilot CLI subagent: no session_start."""
    return [
        _event(EventType.USER_PROMPT, 0, {"prompt": "review"},
               session_id=session_id, tenant_id=tenant_id),
        _event(EventType.TOOL_CALL, 5, {"tool_name": "skill", "arguments": {"skill": "lint"}},
               session_id=session_id, tenant_id=tenant_id),
        _event(EventType.TOOL_RESULT, 6, {"tool_name": "skill"},
               session_id=session_id, tenant_id=tenant_id),
        _event(EventType.TOOL_CALL, 10, {"tool_name": "bash", "arguments": {"command": "x"}},
               session_id=session_id, tenant_id=tenant_id),
        _event(EventType.ERROR, 12, {"tool_name": "bash", "error": "exit 1"},
               session_id=session_id, tenant_id=tenant_id),
        _event(EventType.LLM_REQUEST, 13, {"provider": "gemini"},
               session_id=session_id, tenant_id=tenant_id),
        _event(EventType.ASSISTANT_RESPONSE, 20,
               {"text": "done", "hook_event": "stop", "provider": provider},
               session_id=session_id, tenant_id=tenant_id),
    ]


def write_metaless(base: Path, session_id: str, events) -> Path:
    """Write events.ndjson with a valid hash chain and no meta.json."""
    directory = base / session_id
    directory.mkdir(parents=True)
    lines = []
    previous = ""
    for event in events:
        event.prev_hash = hashlib.sha256(previous.encode()).hexdigest() if previous else ""
        previous = event.to_json()
        lines.append(previous)
    (directory / "events.ndjson").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return directory


class _Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / ".agent-traces"
        self.root.mkdir()
        env = patch.dict(os.environ, {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        for name in ("AGENT_STRACE_TENANT_ID", "AGENT_STRACE_WORKSPACE", "AGENT_STRACE_REDACT"):
            os.environ.pop(name, None)

    def tearDown(self):
        self._tmp.cleanup()

    def item(self, report, session_id):
        matches = [item for item in report.items if item.session_id == session_id]
        self.assertEqual(len(matches), 1, [i.to_dict() for i in report.items])
        return matches[0]


class TestDeriveMeta(unittest.TestCase):
    def test_fields_match_hook_semantics(self):
        events = subagent_events("sub1")
        meta, source = derive_meta("sub1", events)
        self.assertEqual(source, "events")
        self.assertEqual(meta.agent_name, "github-copilot")
        self.assertEqual(meta.command, f"github-copilot ({BACKFILL_SOURCE})")
        self.assertNotIn(IMPLICIT_SESSION_SOURCE, meta.command)
        self.assertEqual(meta.started_at, T0)
        self.assertEqual(meta.ended_at, T0 + 20)
        self.assertEqual(meta.total_duration_ms, 20_000)
        self.assertEqual(meta.tool_calls, 2)
        self.assertEqual(meta.errors, 1)
        self.assertEqual(meta.llm_requests, 1)
        self.assertEqual(meta.attribution, {})
        self.assertEqual(meta.tenant_id, "")

    def test_out_of_order_timestamps_use_earliest_and_latest(self):
        events = subagent_events("sub1")
        events[0].timestamp = T0 + 30
        meta, _ = derive_meta("sub1", events)
        self.assertEqual(meta.started_at, T0 + 5)
        self.assertEqual(meta.ended_at, T0 + 30)
        self.assertGreaterEqual(meta.total_duration_ms, 0)

    def test_agent_name_from_session_start_mode(self):
        events = [_event(EventType.SESSION_START, 0, {"mode": "claude-code-hooks"})]
        self.assertEqual(infer_agent_name(events), "claude-code")

    def test_llm_provider_is_not_mistaken_for_agent(self):
        events = [_event(EventType.LLM_REQUEST, 0, {"provider": "gemini"}),
                  _event(EventType.TOOL_CALL, 1, {"provider": "codex"})]
        self.assertEqual(infer_agent_name(events), "")

    def test_conflicting_providers_are_not_inferred(self):
        events = [
            _event(EventType.ASSISTANT_RESPONSE, 0, {"hook_event": "stop", "provider": "copilot"}),
            _event(EventType.ASSISTANT_RESPONSE, 1, {"hook_event": "stop", "provider": "codex"}),
        ]
        self.assertEqual(infer_agent_name(events), "")

    def test_flag_then_default(self):
        events = [_event(EventType.TOOL_CALL, 0, {"tool_name": "bash"})]
        meta, source = derive_meta("s1", events, agent_name="github-copilot")
        self.assertEqual((meta.agent_name, source), ("github-copilot", "flag"))
        meta, source = derive_meta("s1", events)
        self.assertEqual((meta.agent_name, source), (DEFAULT_AGENT_NAME, "default"))
        self.assertEqual(meta.command, f"{DEFAULT_AGENT_NAME} ({BACKFILL_SOURCE})")

    def test_events_win_over_flag(self):
        meta, source = derive_meta("sub1", subagent_events("sub1"), agent_name="other")
        self.assertEqual((meta.agent_name, source), ("github-copilot", "events"))


class TestDryRunAndApply(_Base):
    def test_dry_run_writes_nothing(self):
        directory = write_metaless(self.root, "sub1", subagent_events("sub1"))
        report = backfill_trace_dir(self.root)
        item = self.item(report, "sub1")
        self.assertEqual(item.status, "would_write")
        self.assertEqual(item.meta["command"], f"github-copilot ({BACKFILL_SOURCE})")
        self.assertFalse((directory / "meta.json").exists())
        self.assertEqual(report.to_dict()["mode"], "dry_run")

    def test_dry_run_does_not_take_the_lock(self):
        write_metaless(self.root, "sub1", subagent_events("sub1"))
        with patch.object(TraceStore, "write_lock", side_effect=AssertionError("locked")):
            report = backfill_trace_dir(self.root)
        self.assertEqual(report.count("would_write"), 1)

    def test_apply_writes_meta_and_leaves_events_untouched(self):
        directory = write_metaless(self.root, "sub1", subagent_events("sub1"))
        events_path = directory / "events.ndjson"
        before = events_path.read_bytes()
        mtime = events_path.stat().st_mtime_ns

        report = backfill_trace_dir(self.root, apply=True)

        self.assertEqual(self.item(report, "sub1").status, "written")
        self.assertEqual(events_path.read_bytes(), before)
        self.assertEqual(events_path.stat().st_mtime_ns, mtime)
        meta = json.loads((directory / "meta.json").read_text())
        self.assertEqual(meta["session_id"], "sub1")
        self.assertEqual(meta["agent_name"], "github-copilot")
        self.assertEqual(meta["command"], f"github-copilot ({BACKFILL_SOURCE})")
        self.assertEqual(meta["started_at"], T0)
        self.assertEqual(meta["ended_at"], T0 + 20)
        self.assertEqual(meta["tool_calls"], 2)
        self.assertEqual(meta["errors"], 1)
        self.assertEqual(meta["attribution"], {})
        self.assertNotIn("tenant_id", meta)

        store = TraceStore(self.root, redact=False, use_workspace_env=False)
        self.assertEqual([m.session_id for m in store.list_sessions()], ["sub1"])
        self.assertEqual(len(store.load_events("sub1")), 7)
        self.assertTrue(verify_chain(store, "sub1").ok)

    def test_apply_holds_store_write_lock(self):
        write_metaless(self.root, "sub1", subagent_events("sub1"))
        held = []
        original = TraceStore.write_lock

        def recording_lock(store):
            manager = original(store)

            class Recorder:
                def __enter__(self_inner):
                    manager.__enter__()
                    held.append(True)
                    return self_inner

                def __exit__(self_inner, *exc):
                    held.append(False)
                    return manager.__exit__(*exc)

            return Recorder()

        written_while_locked = []
        from agent_trace import backfill as module
        original_write = module._write_meta_exclusive

        def recording_write(*args, **kwargs):
            written_while_locked.append(bool(held) and held[-1])
            return original_write(*args, **kwargs)

        with patch.object(TraceStore, "write_lock", recording_lock), \
                patch.object(module, "_write_meta_exclusive", recording_write):
            backfill_trace_dir(self.root, apply=True)
        self.assertEqual(written_while_locked, [True])
        self.assertTrue((self.root / ".store.lock").exists())

    def test_existing_meta_is_never_overwritten(self):
        store = TraceStore(self.root, redact=False, use_workspace_env=False)
        store.create_session(SessionMeta(session_id="main1", agent_name="x", command="captured"))
        meta_path = self.root / "main1" / "meta.json"
        before = meta_path.read_bytes()
        report = backfill_trace_dir(self.root, apply=True)
        self.assertEqual(report.sessions_with_meta, 1)
        self.assertEqual(report.items, [])
        self.assertEqual(meta_path.read_bytes(), before)

    def test_meta_appearing_before_write_is_kept(self):
        directory = write_metaless(self.root, "sub1", subagent_events("sub1"))
        from agent_trace import backfill as module
        original_examine = module._examine

        def racing_examine(*args, **kwargs):
            result = original_examine(*args, **kwargs)
            (directory / "meta.json").write_text('{"session_id": "sub1"}')
            return result

        with patch.object(module, "_examine", racing_examine):
            report = backfill_trace_dir(self.root, apply=True)
        item = self.item(report, "sub1")
        self.assertEqual((item.status, item.reason), ("skipped", "meta_exists"))
        self.assertEqual((directory / "meta.json").read_text(), '{"session_id": "sub1"}')

    def test_second_apply_is_idempotent(self):
        write_metaless(self.root, "sub1", subagent_events("sub1"))
        backfill_trace_dir(self.root, apply=True)
        report = backfill_trace_dir(self.root, apply=True)
        self.assertEqual(report.items, [])
        self.assertEqual(report.sessions_with_meta, 1)

    def test_tenant_env_is_not_applied(self):
        directory = write_metaless(self.root, "sub1", subagent_events("sub1"))
        with patch.dict(os.environ, {"AGENT_STRACE_TENANT_ID": "acme"}):
            backfill_trace_dir(self.root, apply=True)
        meta = json.loads((directory / "meta.json").read_text())
        self.assertNotIn("tenant_id", meta)

    def test_agent_name_flag_used_when_events_are_silent(self):
        events = [_event(EventType.TOOL_CALL, 0, {"tool_name": "bash"}, session_id="sub1")]
        directory = write_metaless(self.root, "sub1", events)
        report = backfill_trace_dir(self.root, apply=True, agent_name="github-copilot")
        self.assertEqual(self.item(report, "sub1").agent_name_source, "flag")
        meta = json.loads((directory / "meta.json").read_text())
        self.assertEqual(meta["command"], f"github-copilot ({BACKFILL_SOURCE})")


class TestSkips(_Base):
    def test_unsafe_and_invalid_entries_are_reported(self):
        good = write_metaless(self.root, "sub1", subagent_events("sub1"))
        outside = Path(self._tmp.name) / "outside"
        write_metaless(outside, "sub2", subagent_events("sub2"))
        os.symlink(outside / "sub2", self.root / "sub2")
        (self.root / "notes.txt").write_text("not a session")
        bad_name = self.root / "bad name"
        bad_name.mkdir()
        (bad_name / "events.ndjson").write_text(subagent_events("x")[0].to_json() + "\n")
        (self.root / "empty1").mkdir()
        (self.root / "empty1" / "events.ndjson").write_text("")
        (self.root / "blank1").mkdir()
        (self.root / "blank1" / "events.ndjson").write_text("\n  \n")
        (self.root / "garbage1").mkdir()
        (self.root / "garbage1" / "events.ndjson").write_text("{not json\n")
        (self.root / "binary1").mkdir()
        (self.root / "binary1" / "events.ndjson").write_bytes(b"\xff\xfe\x00")
        (self.root / "nan1").mkdir()
        (self.root / "nan1" / "events.ndjson").write_text(
            '{"event_type": "tool_call", "timestamp": NaN, "session_id": "nan1"}\n'
        )
        (self.root / "other1").mkdir()
        (self.root / "other1" / "events.ndjson").write_text(
            subagent_events("someone-else")[0].to_json() + "\n"
        )
        (self.root / "evdir1" / "events.ndjson").mkdir(parents=True)
        events_link = self.root / "evlink1"
        events_link.mkdir()
        os.symlink(good / "events.ndjson", events_link / "events.ndjson")
        (self.root / "noevents1").mkdir()

        report = backfill_trace_dir(self.root, apply=True)

        reasons = {item.session_id: item.reason for item in report.items
                   if item.status == "skipped"}
        self.assertEqual(reasons, {
            "sub2": "symlink",
            "bad name": "invalid_session_id",
            "empty1": "empty_events",
            "blank1": "empty_events",
            "garbage1": "unparseable_events",
            "binary1": "unparseable_events",
            "nan1": "unparseable_events",
            "other1": "session_id_mismatch",
            "evdir1": "events_not_file",
            "evlink1": "events_symlink",
        })
        self.assertEqual(self.item(report, "sub1").status, "written")
        self.assertEqual(report.ignored_entries, 2)  # notes.txt, noevents1
        for name in reasons:
            self.assertFalse(os.path.lexists(self.root / name / "meta.json"), name)
        self.assertFalse((outside / "sub2" / "meta.json").exists())
        self.assertEqual(report.errors, [])

    def test_symlinked_meta_is_reported(self):
        directory = write_metaless(self.root, "sub1", subagent_events("sub1"))
        target = Path(self._tmp.name) / "elsewhere.json"
        os.symlink(target, directory / "meta.json")
        report = backfill_trace_dir(self.root, apply=True)
        self.assertEqual(self.item(report, "sub1").reason, "unsafe_meta")
        self.assertFalse(target.exists())

    def test_signed_session_without_meta_is_skipped(self):
        directory = write_metaless(self.root, "sub1", subagent_events("sub1"))
        (directory / "identity.json").write_text("{}")
        report = backfill_trace_dir(self.root, apply=True)
        self.assertEqual(self.item(report, "sub1").reason, "signed_without_meta")
        self.assertFalse((directory / "meta.json").exists())

    def test_symlinked_trace_dir_is_refused(self):
        link = Path(self._tmp.name) / "link"
        os.symlink(self.root, link)
        report = backfill_trace_dir(link, apply=True)
        self.assertTrue(report.errors)

    def test_missing_trace_dir_is_empty(self):
        report = backfill_trace_dir(Path(self._tmp.name) / "missing")
        self.assertEqual((report.items, report.errors), ([], []))
        self.assertFalse((Path(self._tmp.name) / "missing").exists())


class TestErasureAndTenancy(_Base):
    def _erase_then_resurrect(self):
        store = TraceStore(self.root, redact=False, use_workspace_env=False)
        store.create_session(SessionMeta(session_id="gone1", agent_name="x", tenant_id="acme"))
        store.append_event("gone1", _event(EventType.TOOL_CALL, 0, {"tool_name": "bash"},
                                           tenant_id="acme"))
        delete_tenant_data(store, "acme")
        self.assertFalse((self.root / "gone1").exists())
        # A late writer leaves an events-only directory behind.
        return write_metaless(self.root, "gone1", subagent_events("gone1", tenant_id="acme"))

    def test_erased_session_is_not_resurrected(self):
        directory = self._erase_then_resurrect()
        report = backfill_trace_dir(self.root, apply=True)
        self.assertEqual(self.item(report, "gone1").reason, "erased")
        self.assertFalse((directory / "meta.json").exists())

    def test_unreadable_erasure_audit_fails_closed(self):
        directory = self._erase_then_resurrect()
        audit = self.root / "tenant-deletions.ndjson"
        real = Path(self._tmp.name) / "audit-copy"
        real.write_bytes(audit.read_bytes())
        audit.unlink()
        os.symlink(real, audit)
        report = backfill_trace_dir(self.root, apply=True)
        self.assertEqual(self.item(report, "gone1").reason, "unreadable_erasure_audit")
        self.assertFalse((directory / "meta.json").exists())

    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "root ignores file modes")
    def test_permission_denied_erasure_audit_fails_closed(self):
        directory = self._erase_then_resurrect()
        audit = self.root / "tenant-deletions.ndjson"
        audit.chmod(0)
        # The store treats an unreadable audit as "not erased"; backfill must not.
        store = TraceStore(self.root, redact=False, use_workspace_env=False)
        self.assertFalse(store._session_was_erased("gone1"))
        report = backfill_trace_dir(self.root, apply=True)
        self.assertEqual(self.item(report, "gone1").reason, "unreadable_erasure_audit")
        self.assertFalse((directory / "meta.json").exists())

    def _journal(self, name, record):
        journal_dir = self.root / ".tenant-journal"
        journal_dir.mkdir(exist_ok=True)
        (journal_dir / name).write_text(json.dumps(record))

    def test_pending_delete_journal_blocks_session(self):
        directory = write_metaless(self.root, "sub1", subagent_events("sub1", tenant_id="acme"))
        self._journal("delete-op1.json", {
            "operation": "not-recoverable-here",
            "sessions": [{"session_id": "sub1", "event_count": 7}],
        })
        report = backfill_trace_dir(self.root, apply=True)
        self.assertEqual(self.item(report, "sub1").reason, "pending_erasure")
        self.assertFalse((directory / "meta.json").exists())

    def test_pending_tag_journal_blocks_session(self):
        directory = write_metaless(self.root, "sub1", subagent_events("sub1"))
        self._journal("tag-op1.json", {"operation": "not-recoverable-here",
                                       "session_id": "sub1"})
        report = backfill_trace_dir(self.root, apply=True)
        self.assertEqual(self.item(report, "sub1").reason, "pending_tag")
        self.assertFalse((directory / "meta.json").exists())

    def test_unreadable_journal_blocks_the_store(self):
        directory = write_metaless(self.root, "sub1", subagent_events("sub1"))
        journal_dir = self.root / ".tenant-journal"
        journal_dir.mkdir()
        (journal_dir / "delete-op1.json").write_text("{truncated")
        report = backfill_trace_dir(self.root, apply=True)
        self.assertEqual(self.item(report, "sub1").reason, "unreadable_journal")
        self.assertFalse((directory / "meta.json").exists())

    def test_tenant_tagged_events_get_matching_meta(self):
        write_metaless(self.root, "sub1", subagent_events("sub1", tenant_id="acme"))
        store = TraceStore(self.root, redact=False, use_workspace_env=False)
        with self.assertRaises(ValueError):
            store.list_sessions_strict()

        with patch.dict(os.environ, {"AGENT_STRACE_TENANT_ID": "other"}):
            report = backfill_trace_dir(self.root, apply=True)

        self.assertEqual(self.item(report, "sub1").meta["tenant_id"], "acme")
        store = TraceStore(self.root, redact=False, use_workspace_env=False)
        self.assertEqual(store.load_meta("sub1").tenant_id, "acme")
        self.assertEqual(len(store.load_events("sub1")), 7)
        self.assertEqual([m.session_id for m in store.list_sessions_strict(tenant_id="acme")],
                         ["sub1"])

    def test_backfilled_tenant_session_can_be_erased(self):
        write_metaless(self.root, "sub1", subagent_events("sub1", tenant_id="acme"))
        backfill_trace_dir(self.root, apply=True)
        store = TraceStore(self.root, redact=False, use_workspace_env=False)
        result = delete_tenant_data(store, "acme")
        self.assertEqual(result.deleted_sessions, 1)
        self.assertFalse((self.root / "sub1").exists())

    def test_mixed_tenants_are_skipped(self):
        events = subagent_events("sub1", tenant_id="acme")
        events[-1].tenant_id = "globex"
        directory = write_metaless(self.root, "sub1", events)
        report = backfill_trace_dir(self.root, apply=True)
        self.assertEqual(self.item(report, "sub1").reason, "mixed_tenants")
        self.assertFalse((directory / "meta.json").exists())

    def test_partially_tagged_events_take_the_single_tenant(self):
        events = subagent_events("sub1")
        events[0].tenant_id = "acme"
        directory = write_metaless(self.root, "sub1", events)
        report = backfill_trace_dir(self.root, apply=True)
        self.assertEqual(self.item(report, "sub1").status, "written")
        self.assertEqual(json.loads((directory / "meta.json").read_text())["tenant_id"], "acme")

    def test_cross_tenant_child_blocks_parent(self):
        write_metaless(self.root, "parent1", subagent_events("parent1", tenant_id="acme"))
        child = SessionMeta(session_id="child1", agent_name="x", tenant_id="globex",
                            parent_session_id="parent1")
        (self.root / "child1").mkdir()
        (self.root / "child1" / "meta.json").write_text(child.to_json())
        (self.root / "child1" / "events.ndjson").write_text("")
        report = backfill_trace_dir(self.root, apply=True)
        self.assertEqual(self.item(report, "parent1").reason, "cross_tenant_child")
        self.assertFalse((self.root / "parent1" / "meta.json").exists())


class TestWorkspaces(_Base):
    def test_workspace_stores_are_scanned(self):
        ws_base = self.root / "workspaces" / "team-a"
        directory = write_metaless(ws_base, "sub1", subagent_events("sub1"))
        write_metaless(self.root, "flat1", subagent_events("flat1"))

        with patch.dict(os.environ, {"AGENT_STRACE_WORKSPACE": "team-b"}):
            report = backfill_trace_dir(self.root, apply=True)

        item = self.item(report, "sub1")
        self.assertEqual((item.status, item.workspace_id), ("written", "team-a"))
        self.assertEqual(item.path, "workspaces/team-a/sub1")
        self.assertEqual(self.item(report, "flat1").workspace_id, "")
        self.assertEqual(report.stores, [".", "workspaces/team-a"])
        self.assertFalse((self.root / "workspaces" / "team-b").exists())
        meta = json.loads((directory / "meta.json").read_text())
        self.assertEqual(meta["workspace_id"], "team-a")
        store = TraceStore(self.root, workspace_id="team-a", redact=False)
        self.assertEqual([m.session_id for m in store.list_sessions()], ["sub1"])

    def test_symlinked_workspace_is_reported_not_followed(self):
        outside = Path(self._tmp.name) / "outside-ws"
        write_metaless(outside, "sub1", subagent_events("sub1"))
        (self.root / "workspaces").mkdir()
        os.symlink(outside, self.root / "workspaces" / "team-x")
        write_metaless(self.root, "flat1", subagent_events("flat1"))
        report = backfill_trace_dir(self.root, apply=True)
        reasons = [item.reason for item in report.items if item.status == "skipped"]
        self.assertEqual(reasons, ["workspace_symlink"])
        self.assertEqual(self.item(report, "flat1").status, "written")
        self.assertFalse((outside / "sub1" / "meta.json").exists())


class TestCli(_Base):
    def test_parser_registers_command(self):
        args = build_parser().parse_args(["backfill-meta", "--apply", "--agent-name", "x",
                                          "--format", "json"])
        self.assertTrue(args.apply)
        self.assertEqual(args.agent_name, "x")
        self.assertEqual(args.format, "json")
        args = build_parser().parse_args(["backfill-meta"])
        self.assertFalse(args.apply)

    def _run(self, *extra):
        args = argparse.Namespace(trace_dir=str(self.root), apply=False, agent_name="",
                                  format="text")
        for key, value in extra:
            setattr(args, key, value)
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cmd_backfill_meta(args)
        return code, out.getvalue(), err.getvalue()

    def test_text_summary(self):
        write_metaless(self.root, "sub1", subagent_events("sub1"))
        (self.root / "empty1").mkdir()
        (self.root / "empty1" / "events.ndjson").write_text("")
        code, out, _ = self._run()
        self.assertEqual(code, 0)
        self.assertIn("dry run", out)
        self.assertIn("metaless dirs found:  2", out)
        self.assertIn("would write:          1", out)
        self.assertIn("empty_events", out)
        self.assertIn("retention status", out)
        self.assertFalse((self.root / "sub1" / "meta.json").exists())

        code, out, _ = self._run(("apply", True))
        self.assertEqual(code, 0)
        self.assertIn("written:              1", out)
        self.assertTrue((self.root / "sub1" / "meta.json").exists())

    def test_json_output(self):
        write_metaless(self.root, "sub1", subagent_events("sub1"))
        code, out, _ = self._run(("format", "json"))
        self.assertEqual(code, 0)
        data = json.loads(out)
        self.assertEqual(data["mode"], "dry_run")
        self.assertEqual((data["found"], data["would_write"], data["written"]), (1, 1, 0))
        session = data["sessions"][0]
        self.assertEqual(session["status"], "would_write")
        self.assertEqual(session["meta"]["command"], f"github-copilot ({BACKFILL_SOURCE})")

    def test_invalid_agent_name_is_rejected(self):
        code, _, err = self._run(("agent_name", "bad\nname"))
        self.assertEqual(code, 1)
        self.assertIn("control characters", err)

    def test_store_error_returns_nonzero(self):
        link = Path(self._tmp.name) / "link"
        os.symlink(self.root, link)
        args = argparse.Namespace(trace_dir=str(link), apply=True, agent_name="",
                                  format="text")
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(cmd_backfill_meta(args), 1)
        self.assertIn("refusing symlinked trace directory", out.getvalue())


if __name__ == "__main__":
    unittest.main()
