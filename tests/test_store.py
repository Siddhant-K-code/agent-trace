"""Tests for trace storage."""

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore, ensure_trace_dir_gitignore


class TestTraceStore(unittest.TestCase):
    def setUp(self):
        os.environ.pop("AGENT_TRACE_NO_REDACT", None)
        self.tmpdir = tempfile.mkdtemp()
        self.store = TraceStore(self.tmpdir)

    def tearDown(self):
        os.environ.pop("AGENT_TRACE_NO_REDACT", None)

    def test_create_and_load_session(self):
        meta = SessionMeta(agent_name="test-agent")
        self.store.create_session(meta)

        loaded = self.store.load_meta(meta.session_id)
        self.assertEqual(loaded.agent_name, "test-agent")
        self.assertEqual(loaded.session_id, meta.session_id)

    def test_append_and_load_events(self):
        meta = SessionMeta()
        self.store.create_session(meta)

        e1 = TraceEvent(
            event_type=EventType.TOOL_CALL,
            session_id=meta.session_id,
            data={"tool_name": "read_file"},
        )
        e2 = TraceEvent(
            event_type=EventType.TOOL_RESULT,
            session_id=meta.session_id,
            parent_id=e1.event_id,
            duration_ms=42.5,
            data={"content_preview": "hello world"},
        )

        self.store.append_event(meta.session_id, e1)
        self.store.append_event(meta.session_id, e2)

        events = self.store.load_events(meta.session_id)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].event_type, EventType.TOOL_CALL)
        self.assertEqual(events[1].parent_id, e1.event_id)
        self.assertAlmostEqual(events[1].duration_ms, 42.5)

    def test_list_sessions(self):
        m1 = SessionMeta(agent_name="first")
        m2 = SessionMeta(agent_name="second")
        self.store.create_session(m1)
        self.store.create_session(m2)

        sessions = self.store.list_sessions()
        self.assertEqual(len(sessions), 2)

    def test_list_sessions_sorted_newest_first_by_started_at(self):
        old = SessionMeta(session_id="zz-old", started_at=1.0)
        new = SessionMeta(session_id="aa-new", started_at=2.0)
        self.store.create_session(old)
        self.store.create_session(new)

        sessions = self.store.list_sessions()
        self.assertEqual(
            [(m.session_id, m.started_at) for m in sessions],
            [("aa-new", 2.0), ("zz-old", 1.0)],
        )

    def test_list_sessions_uses_session_id_tiebreaker(self):
        lower = SessionMeta(session_id="aa-same", started_at=1.0)
        higher = SessionMeta(session_id="zz-same", started_at=1.0)
        self.store.create_session(lower)
        self.store.create_session(higher)

        sessions = self.store.list_sessions()
        self.assertEqual([m.session_id for m in sessions], ["zz-same", "aa-same"])

    def test_list_sessions_skips_malformed_metadata(self):
        valid = SessionMeta(session_id="valid", started_at=1.0)
        self.store.create_session(valid)

        malformed_dir = os.path.join(self.tmpdir, "malformed")
        os.makedirs(malformed_dir)
        with open(os.path.join(malformed_dir, "meta.json"), "w") as f:
            f.write("{not json")
        with open(os.path.join(self.tmpdir, "loose-file"), "w") as f:
            f.write("ignored")

        sessions = self.store.list_sessions()
        self.assertEqual([m.session_id for m in sessions], ["valid"])

    def test_get_latest_session_returns_newest_meta(self):
        old = SessionMeta(session_id="zz-old", started_at=1.0)
        new = SessionMeta(session_id="aa-new", started_at=2.0)
        self.store.create_session(old)
        self.store.create_session(new)

        latest = self.store.get_latest_session()
        self.assertIsNotNone(latest)
        self.assertEqual(latest.session_id, "aa-new")
        self.assertEqual(latest.started_at, 2.0)

    def test_get_latest_session_id_uses_started_at_not_session_id(self):
        old = SessionMeta(session_id="zz-old", started_at=1.0)
        new = SessionMeta(session_id="aa-new", started_at=2.0)
        self.store.create_session(old)
        self.store.create_session(new)

        self.assertEqual(self.store.get_latest_session_id(), "aa-new")

    def test_find_session_by_prefix(self):
        meta = SessionMeta()
        self.store.create_session(meta)

        prefix = meta.session_id[:6]
        found = self.store.find_session(prefix)
        self.assertEqual(found, meta.session_id)

    def test_find_session_not_found(self):
        found = self.store.find_session("nonexistent")
        self.assertIsNone(found)

    def test_session_exists(self):
        meta = SessionMeta()
        self.store.create_session(meta)

        self.assertTrue(self.store.session_exists(meta.session_id))
        self.assertFalse(self.store.session_exists("fake"))

    def test_update_meta(self):
        meta = SessionMeta(agent_name="test")
        self.store.create_session(meta)

        meta.tool_calls = 10
        meta.errors = 2
        self.store.update_meta(meta)

        loaded = self.store.load_meta(meta.session_id)
        self.assertEqual(loaded.tool_calls, 10)
        self.assertEqual(loaded.errors, 2)

    def test_empty_events(self):
        meta = SessionMeta()
        self.store.create_session(meta)

        events = self.store.load_events(meta.session_id)
        self.assertEqual(events, [])

    def test_append_event_redacts_secrets_by_default(self):
        meta = SessionMeta()
        self.store.create_session(meta)

        event = TraceEvent(
            event_type=EventType.TOOL_CALL,
            session_id=meta.session_id,
            data={
                "tool_name": "http_request",
                "arguments": {
                    "headers": {
                        "Authorization": "Bearer sk-abc123def456ghi789jkl012mno345pqr678",
                    },
                },
            },
        )

        self.store.append_event(meta.session_id, event)

        events = self.store.load_events(meta.session_id)
        self.assertTrue(events[0].redacted)
        self.assertNotIn("sk-abc123", str(events[0].data))
        self.assertIn("[REDACTED:", str(events[0].data))

    def test_append_event_allows_redaction_opt_out(self):
        store = TraceStore(self.tmpdir, redact=False)
        meta = SessionMeta()
        store.create_session(meta)

        event = TraceEvent(
            event_type=EventType.USER_PROMPT,
            session_id=meta.session_id,
            data={"prompt": "use sk-abc123def456ghi789jkl012mno345pqr678"},
        )
        store.append_event(meta.session_id, event)

        events = store.load_events(meta.session_id)
        self.assertFalse(events[0].redacted)
        self.assertIn("sk-abc123", str(events[0].data))

    def test_append_event_respects_no_redact_env(self):
        os.environ["AGENT_TRACE_NO_REDACT"] = "1"
        store = TraceStore(self.tmpdir)
        meta = SessionMeta()
        store.create_session(meta)

        event = TraceEvent(
            event_type=EventType.USER_PROMPT,
            session_id=meta.session_id,
            data={"prompt": "use sk-abc123def456ghi789jkl012mno345pqr678"},
        )
        store.append_event(meta.session_id, event)

        events = store.load_events(meta.session_id)
        self.assertFalse(events[0].redacted)
        self.assertIn("sk-abc123", str(events[0].data))

    def test_create_session_redacts_metadata_by_default(self):
        meta = SessionMeta(
            agent_name="test-agent",
            command="run --api-key sk-abc123def456ghi789jkl012mno345pqr678",
        )
        self.store.create_session(meta)

        loaded = self.store.load_meta(meta.session_id)
        self.assertTrue(loaded.redacted)
        self.assertNotIn("sk-abc123", loaded.command)


_GITIGNORE_BYTES = (
    b"# Created by agent-strace: traces may contain prompts and tool output.\n"
    b"*\n"
)


class TestTraceDirGitignore(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.trace_dir = self.root / "traces"
        self._saved_env = {
            key: os.environ.pop(key, None)
            for key in ("AGENT_STRACE_WORKSPACE", "AGENT_TRACE_DIR",
                        "AGENT_TRACE_CLAUDE_SESSION_ID")
        }

    def tearDown(self):
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._tmp.cleanup()

    def _new_session(self, store):
        meta = SessionMeta(agent_name="test-agent")
        store.create_session(meta)
        store.append_event(meta.session_id, TraceEvent(
            event_type=EventType.TOOL_CALL,
            session_id=meta.session_id,
            data={"tool_name": "read_file"},
        ))
        return meta.session_id

    def test_new_directory_gets_gitignore_without_premature_creation(self):
        store = TraceStore(self.trace_dir)
        self.assertFalse(self.trace_dir.exists())

        self._new_session(store)

        gitignore = self.trace_dir / ".gitignore"
        self.assertEqual(gitignore.read_bytes(), _GITIGNORE_BYTES)
        self.assertIn("*", gitignore.read_text().splitlines())

    def test_existing_directory_without_gitignore_gets_one(self):
        self.trace_dir.mkdir()
        TraceStore(self.trace_dir)
        self.assertEqual(
            (self.trace_dir / ".gitignore").read_bytes(), _GITIGNORE_BYTES
        )

    def test_helper_never_creates_missing_directory(self):
        ensure_trace_dir_gitignore(self.trace_dir)
        self.assertFalse(self.trace_dir.exists())

    def test_existing_gitignore_is_left_byte_for_byte_unchanged(self):
        self.trace_dir.mkdir()
        original = b"# mine\r\n!keep-me.ndjson\nno-trailing-newline"
        (self.trace_dir / ".gitignore").write_bytes(original)

        store = TraceStore(self.trace_dir)
        self._new_session(store)

        self.assertEqual((self.trace_dir / ".gitignore").read_bytes(), original)

    def test_symlinked_gitignore_is_not_followed_or_replaced(self):
        self.trace_dir.mkdir()
        outside = self.root / "outside.txt"
        outside.write_text("untouched")
        (self.trace_dir / ".gitignore").symlink_to(outside)

        self._new_session(TraceStore(self.trace_dir))

        self.assertTrue((self.trace_dir / ".gitignore").is_symlink())
        self.assertEqual(outside.read_text(), "untouched")

    def test_open_failure_does_not_raise(self):
        self.trace_dir.mkdir()
        with patch("agent_trace.store.os.open", side_effect=PermissionError("ro")):
            store = TraceStore(self.trace_dir)
        self.assertFalse((self.trace_dir / ".gitignore").exists())
        self.assertEqual(store.list_sessions(), [])

    def test_write_failure_does_not_raise_or_leave_empty_file(self):
        self.trace_dir.mkdir()
        with patch("agent_trace.store.os.fdopen", side_effect=OSError("disk full")):
            TraceStore(self.trace_dir)
        self.assertFalse((self.trace_dir / ".gitignore").exists())

    @unittest.skipIf(
        os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
        "directory permissions are not enforced for this user/platform",
    )
    def test_read_only_store_does_not_raise(self):
        self.trace_dir.mkdir()
        os.chmod(self.trace_dir, 0o555)
        try:
            store = TraceStore(self.trace_dir)
            self.assertEqual(store.list_sessions(), [])
            self.assertFalse((self.trace_dir / ".gitignore").exists())
        finally:
            os.chmod(self.trace_dir, 0o755)

    def test_session_listing_ignores_gitignore(self):
        store = TraceStore(self.trace_dir)
        first = self._new_session(store)
        second = self._new_session(store)
        self.assertTrue((self.trace_dir / ".gitignore").is_file())

        listed = {meta.session_id for meta in store.list_sessions()}
        strict = {meta.session_id for meta in store.list_sessions_strict()}
        self.assertEqual(listed, {first, second})
        self.assertEqual(strict, {first, second})
        self.assertIn(store.get_latest_session_id(), {first, second})
        self.assertEqual(store.find_session(first[:8]), first)

    def test_workspace_store_ignores_from_storage_root(self):
        from agent_trace.workspace import create_workspace, list_workspaces

        store = TraceStore(self.trace_dir, workspace_id="team-a")
        session_id = self._new_session(store)

        self.assertTrue((self.trace_dir / ".gitignore").is_file())
        self.assertFalse((store.base_dir / ".gitignore").exists())
        self.assertEqual(
            [meta.session_id for meta in store.list_sessions()], [session_id]
        )

        other_root = self.root / "other-traces"
        create_workspace("team-b", str(other_root))
        self.assertTrue((other_root / ".gitignore").is_file())
        self.assertEqual(list_workspaces(str(other_root)), ["team-b"])

    def test_repository_root_or_working_directory_is_never_ignored(self):
        repo_like = self.root / "repo"
        (repo_like / ".git").mkdir(parents=True)
        TraceStore(repo_like)
        self.assertFalse((repo_like / ".gitignore").exists())

        working = self.root / "work"
        (working / "nested").mkdir(parents=True)
        original_cwd = os.getcwd()
        os.chdir(working / "nested")
        try:
            TraceStore(working)
            TraceStore(working / "nested")
        finally:
            os.chdir(original_cwd)
        self.assertFalse((working / ".gitignore").exists())
        self.assertFalse((working / "nested" / ".gitignore").exists())

    def test_hook_capture_path_writes_gitignore(self):
        from agent_trace.hooks import hook_main

        project = self.root / "project"
        project.mkdir()
        payload = {"session_id": "hook-gitignore-01", "cwd": str(project),
                   "source": "startup"}
        with patch.object(sys, "stdin", io.StringIO(json.dumps(payload))):
            hook_main(["session-start"])

        trace_dir = project / ".agent-traces"
        self.assertEqual((trace_dir / ".gitignore").read_bytes(), _GITIGNORE_BYTES)
        self.assertEqual(
            [meta.session_id for meta in TraceStore(trace_dir).list_sessions()],
            ["hook-gitignore-0"],
        )

    @unittest.skipUnless(shutil.which("git"), "git binary not available")
    def test_git_check_ignore_confirms_trace_files_are_ignored(self):
        repo = self.root / "repo"
        repo.mkdir()
        env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")

        def git(*args):
            return subprocess.run(
                ["git", "-C", str(repo), *args],
                env=env, capture_output=True, text=True,
            )

        self.assertEqual(git("init", "-q").returncode, 0)
        (repo / "notes.txt").write_text("tracked")
        session_id = self._new_session(TraceStore(repo / ".agent-traces"))
        trace_file = f".agent-traces/{session_id}/events.ndjson"

        self.assertEqual(git("check-ignore", "-q", trace_file).returncode, 0)
        self.assertEqual(git("check-ignore", "-q", "notes.txt").returncode, 1)
        status = git("status", "--porcelain", "--untracked-files=all").stdout
        self.assertEqual(status.splitlines(), ["?? notes.txt"])
        self.assertEqual(git("add", "-f", trace_file).returncode, 0)
        staged = git("diff", "--cached", "--name-only").stdout.splitlines()
        self.assertEqual(staged, [trace_file])


if __name__ == "__main__":
    unittest.main()
