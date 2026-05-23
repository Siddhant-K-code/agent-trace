"""Tests for agent-strace config-watch (Issue #86)."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from agent_trace.config_watch import (
    ConfigSnapshot,
    FileDiff,
    FileSnapshot,
    SnapshotDiff,
    _hash_file,
    _load_snapshots,
    _load_watch_paths,
    _save_snapshots,
    _snapshot_file,
    cmd_config_watch,
    diff_snapshots,
    find_affected_sessions,
    format_affected,
    format_check,
    format_history,
    take_snapshot,
    DEFAULT_WATCH_PATHS,
)
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_workspace() -> tuple[Path, TraceStore]:
    tmp = Path(tempfile.mkdtemp())
    trace_dir = tmp / ".agent-traces"
    trace_dir.mkdir()
    store = TraceStore(trace_dir)
    return tmp, store


def _add_session(store: TraceStore, started_at: float) -> str:
    meta = SessionMeta(agent_name="test", command="test")
    sp = store.create_session(meta)
    sid = sp.name
    meta2 = store.load_meta(sid)
    meta2.started_at = started_at
    store.update_meta(meta2)
    e = TraceEvent(event_type=EventType.SESSION_END, timestamp=started_at + 60,
                   session_id=sid, data={})
    store.append_event(sid, e)
    return sid


# ---------------------------------------------------------------------------
# _hash_file
# ---------------------------------------------------------------------------

class TestHashFile(unittest.TestCase):
    def test_returns_hex_string(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# AGENTS\nDo stuff.")
            path = Path(f.name)
        h = _hash_file(path)
        self.assertEqual(len(h), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in h))

    def test_returns_empty_for_missing_file(self):
        h = _hash_file(Path("/nonexistent/path.md"))
        self.assertEqual(h, "")

    def test_different_content_different_hash(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("version 1")
            p = Path(f.name)
        h1 = _hash_file(p)
        p.write_text("version 2")
        h2 = _hash_file(p)
        self.assertNotEqual(h1, h2)

    def test_same_content_same_hash(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("same content")
            p = Path(f.name)
        self.assertEqual(_hash_file(p), _hash_file(p))


# ---------------------------------------------------------------------------
# _snapshot_file
# ---------------------------------------------------------------------------

class TestSnapshotFile(unittest.TestCase):
    def test_existing_file(self):
        root = Path(tempfile.mkdtemp())
        (root / "AGENTS.md").write_text("# AGENTS")
        snap = _snapshot_file(root, "AGENTS.md")
        self.assertTrue(snap.exists)
        self.assertEqual(len(snap.sha256), 64)
        self.assertGreater(snap.mtime, 0)

    def test_missing_file(self):
        root = Path(tempfile.mkdtemp())
        snap = _snapshot_file(root, "AGENTS.md")
        self.assertFalse(snap.exists)
        self.assertEqual(snap.sha256, "")
        self.assertEqual(snap.mtime, 0.0)


# ---------------------------------------------------------------------------
# take_snapshot / _load_snapshots / _save_snapshots
# ---------------------------------------------------------------------------

class TestSnapshotPersistence(unittest.TestCase):
    def test_snapshot_saved_and_loaded(self):
        root = Path(tempfile.mkdtemp())
        (root / ".agent-traces").mkdir()
        (root / "AGENTS.md").write_text("# AGENTS")
        snap = take_snapshot(root, ["AGENTS.md"], label="test")
        loaded = _load_snapshots(root)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].snapshot_id, snap.snapshot_id)
        self.assertEqual(loaded[0].label, "test")

    def test_multiple_snapshots_appended(self):
        root = Path(tempfile.mkdtemp())
        (root / ".agent-traces").mkdir()
        take_snapshot(root, ["AGENTS.md"])
        take_snapshot(root, ["AGENTS.md"])
        loaded = _load_snapshots(root)
        self.assertEqual(len(loaded), 2)

    def test_empty_store_returns_empty_list(self):
        root = Path(tempfile.mkdtemp())
        self.assertEqual(_load_snapshots(root), [])

    def test_snapshot_with_session_id(self):
        root = Path(tempfile.mkdtemp())
        (root / ".agent-traces").mkdir()
        snap = take_snapshot(root, [], session_id="abc123")
        loaded = _load_snapshots(root)
        self.assertEqual(loaded[0].session_id, "abc123")

    def test_serialisation_roundtrip(self):
        root = Path(tempfile.mkdtemp())
        (root / ".agent-traces").mkdir()
        (root / "AGENTS.md").write_text("hello")
        snap = take_snapshot(root, ["AGENTS.md"])
        loaded = _load_snapshots(root)
        self.assertEqual(loaded[0].files[0].path, "AGENTS.md")
        self.assertTrue(loaded[0].files[0].exists)


# ---------------------------------------------------------------------------
# diff_snapshots
# ---------------------------------------------------------------------------

class TestDiffSnapshots(unittest.TestCase):
    def _make_snap(self, files: dict[str, str]) -> ConfigSnapshot:
        """files: {path: sha256 or '' for absent}"""
        file_snaps = [
            FileSnapshot(path=p, sha256=h, mtime=1.0 if h else 0.0, exists=bool(h))
            for p, h in files.items()
        ]
        return ConfigSnapshot(
            snapshot_id="test",
            timestamp=time.time(),
            files=file_snaps,
        )

    def test_no_changes(self):
        a = self._make_snap({"AGENTS.md": "abc123"})
        b = self._make_snap({"AGENTS.md": "abc123"})
        diff = diff_snapshots(a, b)
        self.assertFalse(diff.has_changes)

    def test_modified_file(self):
        a = self._make_snap({"AGENTS.md": "abc123"})
        b = self._make_snap({"AGENTS.md": "def456"})
        diff = diff_snapshots(a, b)
        self.assertTrue(diff.has_changes)
        self.assertEqual(diff.changed_paths, ["AGENTS.md"])
        self.assertEqual(diff.changes[0].change, "modified")

    def test_added_file(self):
        a = self._make_snap({})
        b = self._make_snap({"AGENTS.md": "abc123"})
        diff = diff_snapshots(a, b)
        self.assertTrue(diff.has_changes)
        self.assertEqual(diff.changes[0].change, "added")

    def test_removed_file(self):
        a = self._make_snap({"AGENTS.md": "abc123"})
        b = self._make_snap({"AGENTS.md": ""})
        diff = diff_snapshots(a, b)
        self.assertTrue(diff.has_changes)
        self.assertEqual(diff.changes[0].change, "removed")

    def test_multiple_changes(self):
        a = self._make_snap({"AGENTS.md": "aaa", "policy.json": "bbb"})
        b = self._make_snap({"AGENTS.md": "ccc", "policy.json": "bbb"})
        diff = diff_snapshots(a, b)
        self.assertEqual(len(diff.changed_paths), 1)
        self.assertIn("AGENTS.md", diff.changed_paths)

    def test_both_absent_not_reported(self):
        a = self._make_snap({"AGENTS.md": ""})
        b = self._make_snap({"AGENTS.md": ""})
        diff = diff_snapshots(a, b)
        self.assertFalse(diff.has_changes)


# ---------------------------------------------------------------------------
# _load_watch_paths
# ---------------------------------------------------------------------------

class TestLoadWatchPaths(unittest.TestCase):
    def test_defaults_returned(self):
        root = Path(tempfile.mkdtemp())
        paths = _load_watch_paths(root)
        self.assertIn("AGENTS.md", paths)

    def test_extra_paths_added(self):
        root = Path(tempfile.mkdtemp())
        paths = _load_watch_paths(root, extra=["custom/prompt.txt"])
        self.assertIn("custom/prompt.txt", paths)

    def test_config_file_merged(self):
        root = Path(tempfile.mkdtemp())
        (root / ".agent-strace-watch.json").write_text(
            json.dumps({"watch": ["my_prompt.md"]})
        )
        paths = _load_watch_paths(root)
        self.assertIn("my_prompt.md", paths)

    def test_no_duplicates(self):
        root = Path(tempfile.mkdtemp())
        paths = _load_watch_paths(root, extra=["AGENTS.md"])
        self.assertEqual(paths.count("AGENTS.md"), 1)


# ---------------------------------------------------------------------------
# find_affected_sessions
# ---------------------------------------------------------------------------

class TestFindAffectedSessions(unittest.TestCase):
    def test_no_snapshots_returns_empty(self):
        ws, store = _make_workspace()
        result = find_affected_sessions(store, ws)
        self.assertEqual(result, [])

    def test_one_snapshot_returns_empty(self):
        ws, store = _make_workspace()
        take_snapshot(ws, [])
        result = find_affected_sessions(store, ws)
        self.assertEqual(result, [])

    def test_session_after_change_flagged(self):
        ws, store = _make_workspace()
        agents_md = ws / "AGENTS.md"

        # Snapshot 1: AGENTS.md = "v1"
        agents_md.write_text("v1")
        snap1 = take_snapshot(ws, ["AGENTS.md"])

        # Change AGENTS.md
        agents_md.write_text("v2")
        snap2 = take_snapshot(ws, ["AGENTS.md"])

        # Session runs after the change
        now = time.time()
        sid = _add_session(store, started_at=snap2.timestamp + 10)

        affected = find_affected_sessions(store, ws)
        session_ids = [a[0] for a in affected]
        self.assertIn(sid, session_ids)

    def test_session_before_change_not_flagged(self):
        ws, store = _make_workspace()
        agents_md = ws / "AGENTS.md"

        # Session runs first
        now = time.time()
        sid = _add_session(store, started_at=now - 100)

        # Then config changes
        agents_md.write_text("v1")
        take_snapshot(ws, ["AGENTS.md"])
        agents_md.write_text("v2")
        take_snapshot(ws, ["AGENTS.md"])

        affected = find_affected_sessions(store, ws)
        session_ids = [a[0] for a in affected]
        self.assertNotIn(sid, session_ids)

    def test_no_change_between_snapshots_no_affected(self):
        ws, store = _make_workspace()
        agents_md = ws / "AGENTS.md"
        agents_md.write_text("same")
        take_snapshot(ws, ["AGENTS.md"])
        take_snapshot(ws, ["AGENTS.md"])  # no change

        sid = _add_session(store, started_at=time.time())
        affected = find_affected_sessions(store, ws)
        self.assertEqual(affected, [])


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

class TestFormatting(unittest.TestCase):
    def _make_diff(self, has_changes: bool) -> SnapshotDiff:
        changes = [
            FileDiff(path="AGENTS.md", change="modified" if has_changes else "unchanged",
                     old_sha="aaa", new_sha="bbb" if has_changes else "aaa")
        ]
        return SnapshotDiff(
            snapshot_a_id="snap1",
            snapshot_b_id="snap2",
            timestamp_a=time.time() - 3600,
            timestamp_b=time.time(),
            changes=changes,
        )

    def test_format_check_no_changes(self):
        import io
        diff = self._make_diff(has_changes=False)
        out = io.StringIO()
        format_check(diff, out)
        self.assertIn("No config changes", out.getvalue())

    def test_format_check_with_changes(self):
        import io
        diff = self._make_diff(has_changes=True)
        out = io.StringIO()
        format_check(diff, out)
        self.assertIn("AGENTS.md", out.getvalue())
        self.assertIn("modified", out.getvalue())

    def test_format_history_empty(self):
        import io
        out = io.StringIO()
        format_history([], out)
        self.assertIn("No snapshots", out.getvalue())

    def test_format_history_with_snapshots(self):
        import io
        root = Path(tempfile.mkdtemp())
        (root / ".agent-traces").mkdir()
        (root / "AGENTS.md").write_text("v1")
        take_snapshot(root, ["AGENTS.md"])
        (root / "AGENTS.md").write_text("v2")
        take_snapshot(root, ["AGENTS.md"])
        snaps = _load_snapshots(root)
        out = io.StringIO()
        format_history(snaps, out)
        text = out.getvalue()
        self.assertIn("snapshot(s)", text)

    def test_format_affected_empty(self):
        import io
        out = io.StringIO()
        format_affected([], out)
        self.assertIn("No sessions", out.getvalue())

    def test_format_affected_with_results(self):
        import io
        out = io.StringIO()
        format_affected([("abc123def456", "2026-05-23 10:00", ["AGENTS.md"])], out)
        self.assertIn("abc123def456", out.getvalue())
        self.assertIn("AGENTS.md", out.getvalue())


# ---------------------------------------------------------------------------
# cmd_config_watch CLI
# ---------------------------------------------------------------------------

class TestCmdConfigWatch(unittest.TestCase):
    def _args(self, trace_dir, subcommand=None, label=None, watch=None,
              fmt="text", since=None):
        import argparse
        args = argparse.Namespace()
        args.trace_dir = str(trace_dir)
        args.config_watch_command = subcommand
        args.label = label
        args.watch = watch
        args.format = fmt
        args.since = since
        return args

    def test_snapshot_returns_0(self):
        ws, store = _make_workspace()
        args = self._args(ws / ".agent-traces", subcommand="snapshot")
        result = cmd_config_watch(args)
        self.assertEqual(result, 0)

    def test_snapshot_creates_file(self):
        ws, store = _make_workspace()
        args = self._args(ws / ".agent-traces", subcommand="snapshot")
        cmd_config_watch(args)
        snaps = _load_snapshots(ws)
        self.assertEqual(len(snaps), 1)

    def test_check_no_snapshot_returns_1(self):
        ws, store = _make_workspace()
        args = self._args(ws / ".agent-traces", subcommand="check")
        result = cmd_config_watch(args)
        self.assertEqual(result, 1)

    def test_check_no_changes_returns_0(self):
        ws, store = _make_workspace()
        (ws / "AGENTS.md").write_text("v1")
        take_snapshot(ws, ["AGENTS.md"])
        args = self._args(ws / ".agent-traces", subcommand="check",
                          watch=["AGENTS.md"])
        result = cmd_config_watch(args)
        self.assertEqual(result, 0)

    def test_check_with_changes_returns_1(self):
        ws, store = _make_workspace()
        (ws / "AGENTS.md").write_text("v1")
        take_snapshot(ws, ["AGENTS.md"])
        (ws / "AGENTS.md").write_text("v2 — changed!")
        args = self._args(ws / ".agent-traces", subcommand="check",
                          watch=["AGENTS.md"])
        result = cmd_config_watch(args)
        self.assertEqual(result, 1)

    def test_check_json_format(self):
        import io
        from unittest.mock import patch
        ws, store = _make_workspace()
        (ws / "AGENTS.md").write_text("v1")
        take_snapshot(ws, ["AGENTS.md"])
        (ws / "AGENTS.md").write_text("v2")
        args = self._args(ws / ".agent-traces", subcommand="check",
                          watch=["AGENTS.md"], fmt="json")
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            cmd_config_watch(args)
        data = json.loads(captured.getvalue())
        self.assertIn("has_changes", data)
        self.assertTrue(data["has_changes"])

    def test_history_returns_0(self):
        ws, store = _make_workspace()
        take_snapshot(ws, [])
        args = self._args(ws / ".agent-traces", subcommand="history")
        result = cmd_config_watch(args)
        self.assertEqual(result, 0)

    def test_affected_returns_0(self):
        ws, store = _make_workspace()
        args = self._args(ws / ".agent-traces", subcommand="affected")
        result = cmd_config_watch(args)
        self.assertEqual(result, 0)

    def test_snapshot_with_label(self):
        ws, store = _make_workspace()
        args = self._args(ws / ".agent-traces", subcommand="snapshot",
                          label="before-deploy")
        cmd_config_watch(args)
        snaps = _load_snapshots(ws)
        self.assertEqual(snaps[0].label, "before-deploy")

    def test_none_subcommand_defaults_to_snapshot(self):
        ws, store = _make_workspace()
        args = self._args(ws / ".agent-traces", subcommand=None)
        result = cmd_config_watch(args)
        self.assertEqual(result, 0)
        snaps = _load_snapshots(ws)
        self.assertEqual(len(snaps), 1)


if __name__ == "__main__":
    unittest.main()
