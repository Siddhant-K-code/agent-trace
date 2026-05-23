"""Tests for session data retention management."""

import io
import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.retention import (
    RetentionConfig,
    RetentionStatus,
    _parse_simple_yaml,
    _session_size_bytes,
    _store_size_bytes,
    compute_sessions_to_delete,
    delete_sessions,
    get_retention_status,
    cmd_retention_status,
    cmd_retention_clean,
)
from agent_trace.store import TraceStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store() -> tuple[TraceStore, str]:
    tmpdir = tempfile.mkdtemp()
    return TraceStore(tmpdir), tmpdir


def _add_session(store: TraceStore, age_days: float = 0.0, name: str = "") -> SessionMeta:
    """Add a session with started_at set to `age_days` days ago."""
    meta = SessionMeta(agent_name=name)
    meta.started_at = time.time() - age_days * 86400
    store.create_session(meta)
    # Write a small event so the session has non-zero size
    ev = TraceEvent(
        event_type=EventType.TOOL_CALL,
        session_id=meta.session_id,
        data={"tool_name": "Bash", "arguments": {"command": "echo hi"}},
    )
    store.append_event(meta.session_id, ev)
    return meta


# ---------------------------------------------------------------------------
# _parse_simple_yaml
# ---------------------------------------------------------------------------

class TestParseSimpleYaml(unittest.TestCase):
    def test_flat_keys(self):
        text = "max_age_days: 30\nmax_sessions: 100\n"
        result = _parse_simple_yaml(text)
        self.assertEqual(result["max_age_days"], 30)
        self.assertEqual(result["max_sessions"], 100)

    def test_nested_retention_section(self):
        text = "retention:\n  max_age_days: 14\n  max_sessions: 500\n"
        result = _parse_simple_yaml(text)
        self.assertEqual(result["retention"]["max_age_days"], 14)
        self.assertEqual(result["retention"]["max_sessions"], 500)

    def test_comments_ignored(self):
        text = "# comment\nmax_age_days: 7\n"
        result = _parse_simple_yaml(text)
        self.assertEqual(result["max_age_days"], 7)

    def test_null_value(self):
        result = _parse_simple_yaml("max_age_days: null\n")
        self.assertIsNone(result["max_age_days"])

    def test_float_value(self):
        result = _parse_simple_yaml("max_size_mb: 500.5\n")
        self.assertAlmostEqual(result["max_size_mb"], 500.5)


# ---------------------------------------------------------------------------
# RetentionConfig
# ---------------------------------------------------------------------------

class TestRetentionConfig(unittest.TestCase):
    def test_defaults(self):
        config = RetentionConfig()
        self.assertIsNone(config.max_age_days)
        self.assertIsNone(config.max_sessions)
        self.assertIsNone(config.max_size_mb)
        self.assertEqual(config.on_delete, "log")

    def test_from_dict_nested(self):
        config = RetentionConfig.from_dict({
            "retention": {"max_age_days": 30, "max_sessions": 1000, "on_delete": "silent"}
        })
        self.assertEqual(config.max_age_days, 30)
        self.assertEqual(config.max_sessions, 1000)
        self.assertEqual(config.on_delete, "silent")

    def test_from_dict_flat(self):
        config = RetentionConfig.from_dict({"max_age_days": 7})
        self.assertEqual(config.max_age_days, 7)

    def test_load_missing_file_returns_defaults(self):
        config = RetentionConfig.load("/nonexistent/path.yaml")
        self.assertIsInstance(config, RetentionConfig)
        self.assertIsNone(config.max_age_days)

    def test_load_from_yaml_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("retention:\n  max_age_days: 14\n  max_sessions: 200\n")
            path = f.name
        try:
            config = RetentionConfig.load(path)
            self.assertEqual(config.max_age_days, 14)
            self.assertEqual(config.max_sessions, 200)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# compute_sessions_to_delete
# ---------------------------------------------------------------------------

class TestComputeSessionsToDelete(unittest.TestCase):
    def test_no_policy_deletes_nothing(self):
        store, _ = _make_store()
        _add_session(store, age_days=100)
        config = RetentionConfig()
        result = compute_sessions_to_delete(store, config)
        self.assertEqual(result, [])

    def test_age_policy_deletes_old_sessions(self):
        store, _ = _make_store()
        old = _add_session(store, age_days=40)
        _add_session(store, age_days=5)
        config = RetentionConfig(max_age_days=30)
        result = compute_sessions_to_delete(store, config)
        self.assertIn(old.session_id, result)
        self.assertEqual(len(result), 1)

    def test_age_policy_keeps_recent_sessions(self):
        store, _ = _make_store()
        _add_session(store, age_days=5)
        _add_session(store, age_days=10)
        config = RetentionConfig(max_age_days=30)
        result = compute_sessions_to_delete(store, config)
        self.assertEqual(result, [])

    def test_count_policy_keeps_most_recent(self):
        store, _ = _make_store()
        old1 = _add_session(store, age_days=10)
        old2 = _add_session(store, age_days=5)
        _add_session(store, age_days=1)  # newest — should be kept
        config = RetentionConfig(max_sessions=1)
        result = compute_sessions_to_delete(store, config)
        self.assertIn(old1.session_id, result)
        self.assertIn(old2.session_id, result)
        self.assertEqual(len(result), 2)

    def test_count_policy_no_excess(self):
        store, _ = _make_store()
        _add_session(store, age_days=5)
        _add_session(store, age_days=1)
        config = RetentionConfig(max_sessions=5)
        result = compute_sessions_to_delete(store, config)
        self.assertEqual(result, [])

    def test_empty_store_returns_empty(self):
        store, _ = _make_store()
        config = RetentionConfig(max_age_days=30, max_sessions=10)
        result = compute_sessions_to_delete(store, config)
        self.assertEqual(result, [])

    def test_age_and_count_combined(self):
        store, _ = _make_store()
        very_old = _add_session(store, age_days=60)
        old = _add_session(store, age_days=20)
        _add_session(store, age_days=1)
        config = RetentionConfig(max_age_days=30, max_sessions=1)
        result = compute_sessions_to_delete(store, config)
        # very_old deleted by age; old deleted by count (only 1 kept)
        self.assertIn(very_old.session_id, result)
        self.assertIn(old.session_id, result)

    def test_no_duplicates_in_result(self):
        store, _ = _make_store()
        old = _add_session(store, age_days=60)
        _add_session(store, age_days=1)
        # Both age and count would select the old session
        config = RetentionConfig(max_age_days=30, max_sessions=1)
        result = compute_sessions_to_delete(store, config)
        self.assertEqual(len(result), len(set(result)))


# ---------------------------------------------------------------------------
# delete_sessions
# ---------------------------------------------------------------------------

class TestDeleteSessions(unittest.TestCase):
    def test_deletes_session_directory(self):
        store, _ = _make_store()
        meta = _add_session(store, age_days=40)
        config = RetentionConfig(on_delete="silent")
        session_dir = store._session_dir(meta.session_id)
        self.assertTrue(session_dir.exists())
        deleted = delete_sessions(store, [meta.session_id], config)
        self.assertEqual(deleted, 1)
        self.assertFalse(session_dir.exists())

    def test_logs_deletion_when_on_delete_log(self):
        store, tmpdir = _make_store()
        meta = _add_session(store, age_days=40)
        log_path = os.path.join(tmpdir, "retention.log")
        config = RetentionConfig(on_delete="log", log_path=log_path)
        delete_sessions(store, [meta.session_id], config, log_path=log_path)
        self.assertTrue(Path(log_path).exists())
        log_content = Path(log_path).read_text()
        self.assertIn(meta.session_id, log_content)

    def test_no_log_when_silent(self):
        store, tmpdir = _make_store()
        meta = _add_session(store, age_days=40)
        log_path = os.path.join(tmpdir, "retention.log")
        config = RetentionConfig(on_delete="silent", log_path=log_path)
        delete_sessions(store, [meta.session_id], config, log_path=log_path)
        self.assertFalse(Path(log_path).exists())

    def test_nonexistent_session_skipped(self):
        store, _ = _make_store()
        config = RetentionConfig(on_delete="silent")
        deleted = delete_sessions(store, ["nonexistent-id"], config)
        self.assertEqual(deleted, 0)


# ---------------------------------------------------------------------------
# get_retention_status
# ---------------------------------------------------------------------------

class TestGetRetentionStatus(unittest.TestCase):
    def test_empty_store(self):
        store, _ = _make_store()
        config = RetentionConfig()
        status = get_retention_status(store, config)
        self.assertEqual(status.session_count, 0)

    def test_reports_session_count(self):
        store, _ = _make_store()
        _add_session(store, age_days=5)
        _add_session(store, age_days=10)
        config = RetentionConfig()
        status = get_retention_status(store, config)
        self.assertEqual(status.session_count, 2)

    def test_reports_sessions_to_delete(self):
        store, _ = _make_store()
        _add_session(store, age_days=40)
        _add_session(store, age_days=5)
        config = RetentionConfig(max_age_days=30)
        status = get_retention_status(store, config)
        self.assertEqual(len(status.sessions_to_delete), 1)

    def test_bytes_to_free_positive(self):
        store, _ = _make_store()
        _add_session(store, age_days=40)
        config = RetentionConfig(max_age_days=30)
        status = get_retention_status(store, config)
        self.assertGreater(status.bytes_to_free, 0)


# ---------------------------------------------------------------------------
# cmd_retention_status / cmd_retention_clean
# ---------------------------------------------------------------------------

class TestCmdRetentionStatus(unittest.TestCase):
    def _make_args(self, store: TraceStore) -> object:
        import argparse
        args = argparse.Namespace()
        args.trace_dir = str(store.base_dir)
        args.config = None
        return args

    def test_status_empty_store(self):
        store, _ = _make_store()
        args = self._make_args(store)
        out = io.StringIO()
        rc = cmd_retention_status(args, out=out)
        self.assertEqual(rc, 0)
        self.assertIn("No sessions", out.getvalue())

    def test_status_shows_count(self):
        store, _ = _make_store()
        _add_session(store, age_days=5)
        args = self._make_args(store)
        out = io.StringIO()
        rc = cmd_retention_status(args, out=out)
        self.assertEqual(rc, 0)
        self.assertIn("1", out.getvalue())


class TestCmdRetentionClean(unittest.TestCase):
    def _make_args(self, store: TraceStore, dry_run: bool = False,
                   max_age_days: int | None = None) -> object:
        import argparse
        args = argparse.Namespace()
        args.trace_dir = str(store.base_dir)
        args.config = None
        args.dry_run = dry_run
        args.max_age_days = max_age_days
        args.max_sessions = None
        args.max_size_mb = None
        return args

    def test_clean_dry_run_does_not_delete(self):
        store, _ = _make_store()
        meta = _add_session(store, age_days=40)
        args = self._make_args(store, dry_run=True, max_age_days=30)
        out = io.StringIO()
        rc = cmd_retention_clean(args, out=out)
        self.assertEqual(rc, 0)
        self.assertIn("Would delete", out.getvalue())
        # Session still exists
        self.assertTrue(store._session_dir(meta.session_id).exists())

    def test_clean_deletes_old_sessions(self):
        store, _ = _make_store()
        meta = _add_session(store, age_days=40)
        args = self._make_args(store, dry_run=False, max_age_days=30)
        out = io.StringIO()
        rc = cmd_retention_clean(args, out=out)
        self.assertEqual(rc, 0)
        self.assertIn("Deleted", out.getvalue())
        self.assertFalse(store._session_dir(meta.session_id).exists())

    def test_clean_nothing_to_delete(self):
        store, _ = _make_store()
        _add_session(store, age_days=5)
        args = self._make_args(store, dry_run=False, max_age_days=30)
        out = io.StringIO()
        rc = cmd_retention_clean(args, out=out)
        self.assertEqual(rc, 0)
        self.assertIn("Nothing to delete", out.getvalue())


# ---------------------------------------------------------------------------
# CLI registration
# ---------------------------------------------------------------------------

class TestRetentionCLIRegistered(unittest.TestCase):
    def test_retention_command_in_help(self):
        import sys
        from agent_trace.cli import main
        old_argv = sys.argv
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.argv = ["agent-strace", "--help"]
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            try:
                main()
            except SystemExit:
                pass
            output = sys.stdout.getvalue() + sys.stderr.getvalue()
        finally:
            sys.argv = old_argv
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        self.assertIn("retention", output)


if __name__ == "__main__":
    unittest.main()
