"""Tests for watchdog mode: --timeout, --budget, --on-death, post-mortem JSON."""

import json
import os
import tempfile
import time
import unittest

from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore
from agent_trace.watch import (
    WatcherConfig,
    WatchState,
    _parse_duration,
    _write_watchdog_postmortem,
    _invoke_on_death,
)


# ---------------------------------------------------------------------------
# _parse_duration
# ---------------------------------------------------------------------------

class TestParseDuration(unittest.TestCase):
    def test_bare_number_is_seconds(self):
        self.assertAlmostEqual(_parse_duration("90"), 90.0)

    def test_seconds_suffix(self):
        self.assertAlmostEqual(_parse_duration("30s"), 30.0)

    def test_minutes_suffix(self):
        self.assertAlmostEqual(_parse_duration("5m"), 300.0)

    def test_hours_suffix(self):
        self.assertAlmostEqual(_parse_duration("2h"), 7200.0)

    def test_days_suffix(self):
        self.assertAlmostEqual(_parse_duration("1d"), 86400.0)

    def test_compound_hours_minutes(self):
        self.assertAlmostEqual(_parse_duration("1h30m"), 5400.0)

    def test_float_minutes(self):
        self.assertAlmostEqual(_parse_duration("1.5m"), 90.0)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            _parse_duration("")

    def test_invalid_raises(self):
        with self.assertRaises(ValueError):
            _parse_duration("xyz")


# ---------------------------------------------------------------------------
# _write_watchdog_postmortem
# ---------------------------------------------------------------------------

class TestWriteWatchdogPostmortem(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = TraceStore(self.tmpdir)
        self.meta = SessionMeta()
        self.store.create_session(self.meta)

    def _add_event(self, event_type: EventType, **data):
        ev = TraceEvent(
            event_type=event_type,
            session_id=self.meta.session_id,
            data=data,
        )
        self.store.append_event(self.meta.session_id, ev)
        return ev

    def test_writes_json_file(self):
        self._add_event(EventType.TOOL_CALL, tool_name="Bash", arguments={"command": "ls"})
        state = WatchState(start_time=time.time() - 60)
        state.estimated_cost = 1.23

        pm_path = _write_watchdog_postmortem(
            self.store, self.meta.session_id, state, reason="DurationWatcher: 60s elapsed"
        )

        self.assertIsNotNone(pm_path)
        self.assertTrue(pm_path.exists())
        data = json.loads(pm_path.read_text())
        self.assertEqual(data["session_id"], self.meta.session_id)
        self.assertIn("reason", data)
        self.assertIn("cost_at_death", data)
        self.assertIn("recovery_context", data)
        self.assertAlmostEqual(data["cost_at_death"], 1.23, places=4)

    def test_last_tool_call_captured(self):
        self._add_event(EventType.TOOL_CALL, tool_name="Bash", arguments={"command": "pytest"})
        state = WatchState(start_time=time.time())
        pm_path = _write_watchdog_postmortem(
            self.store, self.meta.session_id, state, reason="test"
        )
        data = json.loads(pm_path.read_text())
        self.assertIsNotNone(data["last_tool_call"])
        self.assertEqual(data["last_tool_call"]["tool_name"], "Bash")

    def test_last_llm_response_captured(self):
        self._add_event(EventType.LLM_RESPONSE, model="claude-3-5-sonnet", content="done")
        state = WatchState(start_time=time.time())
        pm_path = _write_watchdog_postmortem(
            self.store, self.meta.session_id, state, reason="test"
        )
        data = json.loads(pm_path.read_text())
        self.assertIsNotNone(data["last_llm_response"])

    def test_empty_session_still_writes(self):
        state = WatchState(start_time=time.time())
        pm_path = _write_watchdog_postmortem(
            self.store, self.meta.session_id, state, reason="budget exceeded"
        )
        self.assertIsNotNone(pm_path)
        data = json.loads(pm_path.read_text())
        self.assertIsNone(data["last_tool_call"])

    def test_recovery_context_mentions_reason(self):
        state = WatchState(start_time=time.time() - 120)
        pm_path = _write_watchdog_postmortem(
            self.store, self.meta.session_id, state, reason="CostWatcher: $5.00"
        )
        data = json.loads(pm_path.read_text())
        self.assertIn("CostWatcher", data["recovery_context"])

    def test_invalid_session_returns_none(self):
        state = WatchState(start_time=time.time())
        result = _write_watchdog_postmortem(
            self.store, "nonexistent-session-id", state, reason="test"
        )
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# WatcherConfig: on_death_cmd field
# ---------------------------------------------------------------------------

class TestWatcherConfigOnDeath(unittest.TestCase):
    def test_default_on_death_is_empty(self):
        config = WatcherConfig()
        self.assertEqual(config.on_death_cmd, "")

    def test_on_death_cmd_set(self):
        config = WatcherConfig(on_death_cmd="python recover.py --pm {post_mortem_path}")
        self.assertIn("{post_mortem_path}", config.on_death_cmd)


# ---------------------------------------------------------------------------
# _invoke_on_death: substitution
# ---------------------------------------------------------------------------

class TestInvokeOnDeath(unittest.TestCase):
    def test_no_crash_on_none_path(self):
        # Should not raise even with None path
        _invoke_on_death("echo done", None)

    def test_no_crash_on_empty_cmd(self):
        _invoke_on_death("", None)


# ---------------------------------------------------------------------------
# CLI: --timeout and --budget flags parsed correctly
# ---------------------------------------------------------------------------

class TestCLIWatchdogFlags(unittest.TestCase):
    def test_timeout_flag_registered(self):
        import argparse
        from agent_trace.cli import main
        import sys
        from io import StringIO

        # Just verify the flags exist by checking help output
        old_argv = sys.argv
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.argv = ["agent-strace", "watch", "--help"]
            sys.stdout = StringIO()
            sys.stderr = StringIO()
            try:
                main()
            except SystemExit:
                pass
            output = sys.stdout.getvalue() + sys.stderr.getvalue()
        finally:
            sys.argv = old_argv
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        self.assertIn("--timeout", output)
        self.assertIn("--budget", output)
        self.assertIn("--on-death", output)


if __name__ == "__main__":
    unittest.main()
