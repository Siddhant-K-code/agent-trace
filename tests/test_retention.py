"""Tests for session data retention management."""

import io
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

try:  # keeps product telemetry off in this process and its children
    import _no_telemetry  # noqa: F401
except ImportError:  # pragma: no cover - run as a package module
    from tests import _no_telemetry  # noqa: F401

from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.retention import (
    OPEN_SESSION_IDLE_DAYS,
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


def _add_session(
    store: TraceStore, age_days: float = 0.0, name: str = "", ended: bool = True,
) -> SessionMeta:
    """Add a session with started_at set to `age_days` days ago."""
    meta = SessionMeta(agent_name=name)
    meta.started_at = time.time() - age_days * 86400
    if ended:
        meta.ended_at = meta.started_at + 60
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

    def test_size_budget_accounts_for_age_and_count_deletions(self):
        policies = (
            {"max_age_days": 30},
            {"max_sessions": 1},
            {"max_age_days": 30, "max_sessions": 1},
        )
        for policy in policies:
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as tmpdir:
                store = TraceStore(tmpdir)
                old = _add_session(store, age_days=60)
                recent = _add_session(store, age_days=1)
                checkpoint = store.checkpoint_path(old.session_id)
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_text("old recovery context" * 100)
                remaining_bytes = _session_size_bytes(store._session_dir(recent.session_id))
                config = RetentionConfig(
                    **policy, max_size_mb=remaining_bytes / (1024 * 1024)
                )

                self.assertEqual(compute_sessions_to_delete(store, config), [old.session_id])
                status = get_retention_status(store, config)
                self.assertEqual(status.sessions_to_delete, [old.session_id])
                self.assertEqual(status.total_size_bytes - status.bytes_to_free, remaining_bytes)
                self.assertTrue(store._session_dir(recent.session_id).exists())

    def test_size_policy_only_deletes_additional_sessions_until_budget_is_met(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = TraceStore(tmpdir)
            old = _add_session(store, age_days=60)
            middle = _add_session(store, age_days=5)
            recent = _add_session(store, age_days=1)
            remaining_bytes = _session_size_bytes(store._session_dir(recent.session_id))
            config = RetentionConfig(
                max_age_days=30, max_size_mb=remaining_bytes / (1024 * 1024)
            )

            self.assertEqual(
                compute_sessions_to_delete(store, config),
                [old.session_id, middle.session_id],
            )

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

    def test_deletes_compaction_checkpoint_with_session(self):
        store, _ = _make_store()
        meta = _add_session(store, age_days=40)
        checkpoint = store.base_dir / "checkpoints" / f"{meta.session_id}.md"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_text("sensitive recovery context")

        deleted = delete_sessions(
            store, [meta.session_id], RetentionConfig(on_delete="silent")
        )

        self.assertEqual(deleted, 1)
        self.assertFalse(checkpoint.exists())

    def test_deletes_orphaned_compaction_checkpoint(self):
        store, _ = _make_store()
        checkpoint = store.base_dir / "checkpoints" / "missing-session.md"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_text("orphaned recovery context")

        deleted = delete_sessions(
            store, ["missing-session"], RetentionConfig(on_delete="silent")
        )

        self.assertEqual(deleted, 0)
        self.assertFalse(checkpoint.exists())

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
# Open sessions and the size measure
# ---------------------------------------------------------------------------

def _claude_marker(store: TraceStore, session_id: str) -> Path:
    """Write a provider-scoped hook marker the way hooks.py names it."""
    import hashlib
    raw = session_id + "-conversation"
    digest = hashlib.sha256(f"claude\0{raw}".encode()).hexdigest()
    suffix = f".v2.claude.{raw[:16].encode().hex()}.{digest}"
    marker = store.base_dir / f".active-session{suffix}"
    marker.write_text(session_id, encoding="utf-8")
    (store.base_dir / f".pending-calls{suffix}.json").write_text("{}")
    return marker


def _make_idle(store: TraceStore, session_id: str, days: float) -> None:
    stamp = time.time() - days * 86400
    for f in store._session_dir(session_id).iterdir():
        os.utime(f, (stamp, stamp))


def _clean_args(store: TraceStore, **overrides) -> object:
    import argparse
    args = argparse.Namespace(
        trace_dir=str(store.base_dir), config=None, dry_run=False,
        max_age_days=None, max_sessions=None, max_size_mb=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class TestOpenSessionsAreKept(unittest.TestCase):
    def test_clean_keeps_running_session_usable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = TraceStore(tmpdir)
            running = _add_session(store, age_days=2, ended=False)
            finished = _add_session(store, age_days=1)
            out = io.StringIO()

            cmd_retention_clean(_clean_args(store, max_sessions=1), out=out)

            self.assertIn("Nothing to delete", out.getvalue())
            self.assertIn("Kept: 1 open session(s)", out.getvalue())
            store.append_event(running.session_id, TraceEvent(
                event_type=EventType.TOOL_CALL, session_id=running.session_id,
                data={"tool_name": "Read"},
            ))
            meta = store.load_meta(running.session_id)
            meta.ended_at = time.time()
            store.update_meta(meta)
            listed = {m.session_id for m in store.list_sessions()}
            self.assertEqual(listed, {running.session_id, finished.session_id})
            self.assertEqual(len(store.load_events(running.session_id)), 2)

    def test_ended_session_named_by_hook_marker_is_kept(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = TraceStore(tmpdir)
            resumed = _add_session(store, age_days=40)
            marker = _claude_marker(store, resumed.session_id)

            self.assertEqual(
                compute_sessions_to_delete(store, RetentionConfig(max_age_days=30)), [],
            )
            self.assertTrue(marker.exists())

    def test_abandoned_open_session_is_deleted_with_its_hook_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = TraceStore(tmpdir)
            crashed = _add_session(store, age_days=40, ended=False)
            marker = _claude_marker(store, crashed.session_id)
            pending = Path(str(marker).replace(".active-session", ".pending-calls") + ".json")
            _make_idle(store, crashed.session_id, days=OPEN_SESSION_IDLE_DAYS + 1)
            out = io.StringIO()

            cmd_retention_clean(_clean_args(store, max_age_days=30), out=out)

            self.assertIn("Deleted: 1 session(s)", out.getvalue())
            self.assertFalse(store._session_dir(crashed.session_id).exists())
            self.assertFalse(marker.exists())
            self.assertFalse(pending.exists())

    def test_session_resumed_after_selection_is_not_deleted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = TraceStore(tmpdir)
            old = _add_session(store, age_days=40)
            selected = compute_sessions_to_delete(store, RetentionConfig(max_age_days=30))
            self.assertEqual(selected, [old.session_id])
            meta = store.load_meta(old.session_id)
            meta.ended_at = None
            store.update_meta(meta)

            deleted = delete_sessions(store, selected, RetentionConfig(on_delete="silent"))

            self.assertEqual(deleted, 0)
            self.assertTrue(store._session_dir(old.session_id).exists())

    def test_delete_removes_session_approval_and_dataset_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = TraceStore(tmpdir)
            old = _add_session(store, age_days=40)
            kept = _add_session(store, age_days=1)
            approvals = store.base_dir / ".approvals"
            approvals.mkdir()
            (approvals / "a1.json").write_text(json.dumps({"session_id": old.session_id}))
            dataset = store.base_dir / "datasets" / "golden.jsonl"
            dataset.parent.mkdir()
            dataset.write_text(
                json.dumps({"session_id": old.session_id}) + "\n"
                + json.dumps({"session_id": kept.session_id}) + "\n"
            )

            delete_sessions(store, [old.session_id], RetentionConfig(on_delete="silent"))

            self.assertFalse((approvals / "a1.json").exists())
            self.assertEqual(
                [json.loads(line)["session_id"] for line in dataset.read_text().splitlines()],
                [kept.session_id],
            )


class TestSizePolicyMeasure(unittest.TestCase):
    def test_non_session_data_does_not_trigger_size_deletions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = TraceStore(tmpdir)
            for age in (3, 2, 1):
                _add_session(store, age_days=age)
            dataset = store.base_dir / "datasets" / "golden.jsonl"
            dataset.parent.mkdir()
            dataset.write_text("x" * (2 * 1024 * 1024))
            out = io.StringIO()

            cmd_retention_clean(_clean_args(store, max_size_mb=1), out=out)

            self.assertIn("Nothing to delete", out.getvalue())
            self.assertEqual(len(store.list_sessions()), 3)
            self.assertLess(_store_size_bytes(store), 1024 * 1024)

    def test_open_session_size_does_not_trigger_size_deletions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = TraceStore(tmpdir)
            running = _add_session(store, age_days=2, ended=False)
            (store._session_dir(running.session_id) / "events.ndjson").write_text(
                "x" * (2 * 1024 * 1024)
            )
            finished = _add_session(store, age_days=1)

            self.assertEqual(
                compute_sessions_to_delete(store, RetentionConfig(max_size_mb=1)), [],
            )
            self.assertTrue(store._session_dir(finished.session_id).exists())


# ---------------------------------------------------------------------------
# Stored times that are not usable
# ---------------------------------------------------------------------------

# meta.json can hold any JSON value. usable_time rejects each of these.
_UNUSABLE_TIMES = [
    float("nan"), float("inf"), float("-inf"), 1e308, -1e308, 1e12,
    True, "x", None, [], {},
]


def _add_raw_session(
    store: TraceStore, file_age_days: float = 0.0, ended: bool = True, **fields,
) -> str:
    """Add a session, write raw JSON values into its meta.json, and age its files."""
    session_id = _add_session(store, age_days=file_age_days, ended=ended).session_id
    meta_path = store._session_dir(session_id) / "meta.json"
    data = json.loads(meta_path.read_text())
    data.update(fields)
    meta_path.write_text(json.dumps(data))
    _make_idle(store, session_id, file_age_days)
    return session_id


def _left(store: TraceStore) -> set[str]:
    return {meta.session_id for meta in store.list_sessions()}


def _unlistable(session_dir: Path):
    """Make listing ``session_dir`` raise OSError. Other directories still list."""
    real_iterdir = Path.iterdir

    def iterdir(self):
        if self == session_dir:
            raise OSError("cannot list")
        return real_iterdir(self)

    return mock.patch.object(Path, "iterdir", iterdir)


class TestUnusableStoredTimes(unittest.TestCase):
    def _store(self) -> TraceStore:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return TraceStore(tmp.name)

    def test_running_session_with_nan_start_survives_clean(self):
        for marker in (True, False):
            for policy in ({"max_sessions": 2}, {"max_size_mb": 0}):
                with self.subTest(marker=marker, policy=policy):
                    store = self._store()
                    _add_session(store, age_days=100)
                    _add_session(store, age_days=1)
                    running = _add_raw_session(
                        store, ended=False, started_at=float("nan"),
                    )
                    if marker:
                        _claude_marker(store, running)
                    out = io.StringIO()

                    cmd_retention_clean(_clean_args(store, **policy), out=out)

                    self.assertIn(running, _left(store))
                    self.assertIn("Kept: 1 open session(s)", out.getvalue())

    def test_recheck_under_lock_keeps_session_with_unusable_times(self):
        cases = [
            {"started_at": float("nan")},
            {"started_at": "x"},
            {"ended_at": "x"},
            {"ended_at": float("nan")},
        ]
        for fields in cases:
            with self.subTest(fields=fields):
                store = self._store()
                running = _add_raw_session(
                    store, ended="ended_at" in fields, **fields,
                )

                deleted = delete_sessions(
                    store, [running], RetentionConfig(on_delete="silent"),
                )

                self.assertEqual(deleted, 0)
                self.assertIn(running, _left(store))

    def test_start_that_is_not_a_number_does_not_crash(self):
        policies = ({"max_age_days": 30}, {"max_sessions": 2}, {"max_size_mb": 0})
        for value in ("x", None, [], {}):
            for ended in (True, False):
                with self.subTest(start=value, ended=ended, command="status"):
                    store = self._store()
                    _add_session(store, age_days=100)
                    _add_session(store, age_days=1)
                    _add_raw_session(store, ended=ended, started_at=value)
                    out = io.StringIO()

                    self.assertEqual(cmd_retention_status(_clean_args(store), out=out), 0)
                    self.assertIn("Sessions:    3", out.getvalue())
                for policy in policies:
                    for dry_run in (True, False):
                        with self.subTest(
                            start=value, ended=ended, policy=policy, dry_run=dry_run,
                        ):
                            store = self._store()
                            _add_session(store, age_days=100)
                            _add_session(store, age_days=1)
                            session_id = _add_raw_session(
                                store, ended=ended, started_at=value,
                            )
                            args = _clean_args(store, dry_run=dry_run, **policy)

                            self.assertEqual(
                                cmd_retention_clean(args, out=io.StringIO()), 0,
                            )
                            if not ended:
                                self.assertIn(session_id, _left(store))

    def test_count_and_size_limits_order_by_newest_write(self):
        for value in _UNUSABLE_TIMES:
            for limit in ("count", "size"):
                with self.subTest(start=value, limit=limit):
                    store = self._store()
                    _add_session(store, age_days=100)
                    recent = _add_session(store, age_days=1).session_id
                    unusable = _add_raw_session(store, file_age_days=1, started_at=value)
                    if limit == "count":
                        policy = {"max_sessions": 2}
                    else:
                        # One byte over the limit: the oldest session must go.
                        policy = {"max_size_mb": (_store_size_bytes(store) - 1) / (1024 * 1024)}

                    cmd_retention_clean(_clean_args(store, **policy), out=io.StringIO())

                    self.assertEqual(_left(store), {recent, unusable})

    def test_count_limit_breaks_a_start_tie_by_session_id(self):
        store = self._store()
        start = time.time() - 10 * 86400
        ids = [
            _add_raw_session(store, file_age_days=10, started_at=start)
            for _ in range(2)
        ]

        cmd_retention_clean(_clean_args(store, max_sessions=1), out=io.StringIO())

        self.assertEqual(_left(store), {max(ids)})

    def test_age_limit_and_oldest_line_use_newest_write(self):
        for value in _UNUSABLE_TIMES:
            with self.subTest(start=value, files="200 days old"):
                store = self._store()
                _add_session(store, age_days=100)
                recent = _add_session(store, age_days=1).session_id
                unusable = _add_raw_session(store, file_age_days=200, started_at=value)
                out = io.StringIO()

                cmd_retention_status(_clean_args(store), out=out)
                status = get_retention_status(store, RetentionConfig())
                cmd_retention_clean(_clean_args(store, max_age_days=30), out=io.StringIO())

                self.assertIn(f"Oldest:      {unusable} (200.0 days ago)", out.getvalue())
                self.assertEqual(status.oldest_session_id, unusable)
                self.assertAlmostEqual(status.oldest_session_age_days, 200, places=2)
                self.assertEqual(_left(store), {recent})
            with self.subTest(start=value, files="fresh"):
                store = self._store()
                _add_session(store, age_days=100)
                recent = _add_session(store, age_days=1).session_id
                unusable = _add_raw_session(store, started_at=value)

                cmd_retention_clean(_clean_args(store, max_age_days=30), out=io.StringIO())

                self.assertEqual(_left(store), {recent, unusable})

    def test_newest_file_write_counts_not_the_oldest(self):
        # `agent-strace record` writes meta.json when it starts and stops,
        # and events.ndjson for each event in between.
        for ended, start in ((False, time.time() - 100 * 86400), (True, "x")):
            with self.subTest(ended=ended, start=start):
                store = self._store()
                session_id = _add_raw_session(
                    store, file_age_days=100, ended=ended, started_at=start,
                )
                stamp = time.time() - 86400
                events = store._session_dir(session_id) / "events.ndjson"
                os.utime(events, (stamp, stamp))

                cmd_retention_clean(_clean_args(store, max_age_days=30), out=io.StringIO())

                self.assertEqual(_left(store), {session_id})

    def test_end_that_is_not_a_usable_time_counts_as_no_end(self):
        for value in _UNUSABLE_TIMES:
            if value is None:
                continue  # null is no end time already
            with self.subTest(end=value, files="fresh"):
                store = self._store()
                running = _add_raw_session(store, ended_at=value)
                out = io.StringIO()

                cmd_retention_clean(_clean_args(store, max_size_mb=0), out=out)

                self.assertEqual(_left(store), {running})
                self.assertIn("Kept: 1 open session(s)", out.getvalue())
            with self.subTest(end=value, files="idle"):
                store = self._store()
                _add_raw_session(
                    store, file_age_days=OPEN_SESSION_IDLE_DAYS + 3, ended_at=value,
                )

                cmd_retention_clean(_clean_args(store, max_size_mb=0), out=io.StringIO())

                self.assertEqual(_left(store), set())

    def test_session_with_no_usable_time_is_kept(self):
        for ended in (False, True):
            with self.subTest(ended=ended):
                store = self._store()
                _add_session(store, age_days=100)
                undated = _add_raw_session(store, ended=ended, started_at=float("nan"))
                clean_out = io.StringIO()
                status_out = io.StringIO()

                with _unlistable(store._session_dir(undated)):
                    cmd_retention_clean(_clean_args(store, max_sessions=0), out=clean_out)
                    rc = cmd_retention_status(_clean_args(store), out=status_out)

                self.assertEqual(_left(store), {undated})
                self.assertEqual(rc, 0)
                self.assertIn("Sessions:    1", status_out.getvalue())
                self.assertNotIn("Oldest:", status_out.getvalue())
                if not ended:
                    self.assertIn("Kept: 1 open session(s)", clean_out.getvalue())
                    self.assertIn("Open:        1 session(s)", status_out.getvalue())


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
