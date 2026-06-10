"""Tests for crashed-session postmortem detection."""

from __future__ import annotations

import io
import os
import tempfile
import time
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.postmortem import (
    CRASH_POSTMORTEM_FILE,
    analyze_session,
    clear_heartbeat,
    cmd_postmortem,
    detect_crash,
    find_crashed_sessions,
    heartbeat_path,
    write_crash_postmortem,
    write_heartbeat,
)
from agent_trace.store import TraceStore


def _store_with_session() -> tuple[TraceStore, SessionMeta, tempfile.TemporaryDirectory]:
    tmp = tempfile.TemporaryDirectory()
    store = TraceStore(Path(tmp.name))
    meta = SessionMeta(agent_name="agent", command="test")
    store.create_session(meta)
    store.append_event(
        meta.session_id,
        TraceEvent(
            event_type=EventType.SESSION_START,
            timestamp=meta.started_at,
            session_id=meta.session_id,
            data={},
        ),
    )
    return store, meta, tmp


class TestHeartbeatCrashDetection(unittest.TestCase):
    def test_stale_heartbeat_marks_session_crashed(self) -> None:
        store, meta, tmp = _store_with_session()
        try:
            store.append_event(
                meta.session_id,
                TraceEvent(
                    event_type=EventType.TOOL_CALL,
                    timestamp=meta.started_at + 5,
                    session_id=meta.session_id,
                    data={"tool_name": "Bash", "arguments": {"command": "pytest"}},
                ),
            )
            write_heartbeat(store, meta.session_id)
            old = time.time() - 120
            os.utime(heartbeat_path(store, meta.session_id), (old, old))

            crash = detect_crash(store, meta.session_id, stale_after_seconds=30)

            self.assertIsNotNone(crash)
            self.assertEqual(crash.reason, "unknown")
            self.assertGreaterEqual(crash.stale_seconds, 30)
        finally:
            tmp.cleanup()

    def test_clean_session_end_is_not_crashed(self) -> None:
        store, meta, tmp = _store_with_session()
        try:
            store.append_event(
                meta.session_id,
                TraceEvent(
                    event_type=EventType.SESSION_END,
                    timestamp=meta.started_at + 10,
                    session_id=meta.session_id,
                    data={"exit_code": 0},
                ),
            )
            meta.ended_at = meta.started_at + 10
            store.update_meta(meta)
            write_heartbeat(store, meta.session_id)

            self.assertIsNone(detect_crash(store, meta.session_id, stale_after_seconds=0))
        finally:
            tmp.cleanup()

    def test_nonzero_session_end_is_classified(self) -> None:
        store, meta, tmp = _store_with_session()
        try:
            store.append_event(
                meta.session_id,
                TraceEvent(
                    event_type=EventType.SESSION_END,
                    timestamp=meta.started_at + 10,
                    session_id=meta.session_id,
                    data={"exit_code": 137},
                ),
            )
            meta.ended_at = meta.started_at + 10
            store.update_meta(meta)

            crash = detect_crash(store, meta.session_id)
            report = analyze_session(store, meta.session_id)

            self.assertIsNotNone(crash)
            self.assertEqual(crash.reason, "SIGKILL")
            self.assertTrue(report.failed)
            self.assertEqual(report.crash_reason, "SIGKILL")
        finally:
            tmp.cleanup()

    def test_write_and_clear_heartbeat(self) -> None:
        store, meta, tmp = _store_with_session()
        try:
            write_heartbeat(store, meta.session_id)
            self.assertTrue(heartbeat_path(store, meta.session_id).exists())
            clear_heartbeat(store, meta.session_id)
            self.assertFalse(heartbeat_path(store, meta.session_id).exists())
        finally:
            tmp.cleanup()


class TestCrashPostmortemOutput(unittest.TestCase):
    def test_persistent_markdown_contains_recovery_context(self) -> None:
        store, meta, tmp = _store_with_session()
        try:
            store.append_event(
                meta.session_id,
                TraceEvent(
                    event_type=EventType.TOOL_CALL,
                    timestamp=meta.started_at + 5,
                    session_id=meta.session_id,
                    data={"tool_name": "Write", "arguments": {"file_path": "src/app.py"}},
                ),
            )
            write_heartbeat(store, meta.session_id)
            old = time.time() - 120
            os.utime(heartbeat_path(store, meta.session_id), (old, old))

            report = analyze_session(store, meta.session_id, stale_after_seconds=30)
            path = write_crash_postmortem(store, report)

            self.assertEqual(path.name, CRASH_POSTMORTEM_FILE)
            text = path.read_text()
            self.assertIn("Recovery Context", text)
            self.assertIn("src/app.py", text)
        finally:
            tmp.cleanup()

    def test_find_crashed_sessions_returns_stale_sessions(self) -> None:
        store, meta, tmp = _store_with_session()
        try:
            write_heartbeat(store, meta.session_id)
            old = time.time() - 120
            os.utime(heartbeat_path(store, meta.session_id), (old, old))

            crashes = find_crashed_sessions(store, stale_after_seconds=30)

            self.assertEqual([c.session_id for c in crashes], [meta.session_id])
        finally:
            tmp.cleanup()

    def test_cmd_postmortem_list_writes_markdown(self) -> None:
        store, meta, tmp = _store_with_session()
        try:
            write_heartbeat(store, meta.session_id)
            old = time.time() - 120
            os.utime(heartbeat_path(store, meta.session_id), (old, old))
            args = Namespace(
                trace_dir=str(store.base_dir),
                list=True,
                session_id=None,
                agents_md="AGENTS.md",
                stale_after=30.0,
            )
            out = io.StringIO()

            with patch("sys.stdout", out):
                code = cmd_postmortem(args)

            self.assertEqual(code, 1)
            self.assertIn(meta.session_id, out.getvalue())
            self.assertTrue((store._session_dir(meta.session_id) / CRASH_POSTMORTEM_FILE).exists())
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
