"""Tests for live session monitoring and circuit breakers."""

import io
import json
import os
import tempfile
import time
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import patch, MagicMock

from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore
from agent_trace.watch import (
    WatcherConfig,
    WatchState,
    check_event,
    _alert_terminal,
    _alert_file,
    _detect_loop,
    _event_key,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(event_type: EventType, ts: float, session_id: str = "s1", **data) -> TraceEvent:
    return TraceEvent(event_type=event_type, timestamp=ts, session_id=session_id, data=data)


def _bash_event(cmd: str, ts: float = 0.0) -> TraceEvent:
    return _make_event(EventType.TOOL_CALL, ts, tool_name="Bash", arguments={"command": cmd})


def _default_config(**kwargs) -> WatcherConfig:
    defaults = dict(
        max_retries=3,
        max_cost_dollars=10.0,
        max_duration_seconds=3600.0,
        loop_sequence_length=3,
        loop_max_repeats=3,
        on_violation="terminal",
    )
    defaults.update(kwargs)
    return WatcherConfig(**defaults)


# ---------------------------------------------------------------------------
# Retry detection
# ---------------------------------------------------------------------------

class TestRetryDetection(unittest.TestCase):
    def test_no_violation_below_threshold(self):
        config = _default_config(max_retries=3)
        state = WatchState()
        cmd = "pytest"
        for i in range(3):
            violations = check_event(_bash_event(cmd, float(i)), config, state)
        self.assertEqual(violations, [])

    def test_violation_at_threshold_plus_one(self):
        config = _default_config(max_retries=3)
        state = WatchState()
        cmd = "pytest"
        all_violations = []
        for i in range(5):
            all_violations.extend(check_event(_bash_event(cmd, float(i)), config, state))
        self.assertTrue(any("RetryWatcher" in v for v in all_violations))

    def test_different_commands_no_violation(self):
        config = _default_config(max_retries=2)
        state = WatchState()
        violations = []
        for i, cmd in enumerate(["ls", "pwd", "echo hi"]):
            violations = check_event(_bash_event(cmd, float(i)), config, state)
        self.assertEqual(violations, [])

    def test_violation_fires_only_once(self):
        config = _default_config(max_retries=2)
        state = WatchState()
        cmd = "failing-cmd"
        all_violations = []
        for i in range(6):
            v = check_event(_bash_event(cmd, float(i)), config, state)
            all_violations.extend(v)
        retry_violations = [v for v in all_violations if "RetryWatcher" in v]
        self.assertEqual(len(retry_violations), 1)


# ---------------------------------------------------------------------------
# Cost threshold
# ---------------------------------------------------------------------------

class TestCostThreshold(unittest.TestCase):
    def test_no_violation_below_threshold(self):
        config = _default_config(max_cost_dollars=100.0)
        state = WatchState()
        event = _make_event(EventType.ASSISTANT_RESPONSE, 0.0, text="hello")
        violations = check_event(event, config, state)
        self.assertEqual(violations, [])

    def test_violation_when_cost_exceeds_threshold(self):
        config = _default_config(max_cost_dollars=0.0)
        state = WatchState()
        # Force cost above threshold by setting it directly
        state.estimated_cost = 0.01
        event = _make_event(EventType.ASSISTANT_RESPONSE, 0.0, text="x" * 10000)
        violations = check_event(event, config, state)
        self.assertTrue(any("CostWatcher" in v for v in violations))


# ---------------------------------------------------------------------------
# Duration limit
# ---------------------------------------------------------------------------

class TestDurationLimit(unittest.TestCase):
    def test_no_violation_within_limit(self):
        config = _default_config(max_duration_seconds=3600.0)
        state = WatchState(start_time=time.time())
        event = _make_event(EventType.TOOL_CALL, 0.0, tool_name="Bash",
                            arguments={"command": "ls"})
        violations = check_event(event, config, state)
        self.assertEqual(violations, [])

    def test_violation_when_duration_exceeded(self):
        config = _default_config(max_duration_seconds=0.0)
        state = WatchState(start_time=time.time() - 10)  # 10s ago
        event = _make_event(EventType.USER_PROMPT, 0.0, prompt="hi")
        violations = check_event(event, config, state)
        self.assertTrue(any("DurationWatcher" in v for v in violations))


# ---------------------------------------------------------------------------
# Loop detection
# ---------------------------------------------------------------------------

class TestLoopDetection(unittest.TestCase):
    def test_no_loop_with_varied_events(self):
        recent = deque(["a", "b", "c", "d", "e", "f"], maxlen=30)
        result = _detect_loop(recent, seq_len=3, max_repeats=3)
        self.assertIsNone(result)

    def test_detects_repeating_sequence(self):
        # Sequence [a, b, c] repeated 3 times
        seq = ["a", "b", "c"] * 3
        recent = deque(seq, maxlen=30)
        result = _detect_loop(recent, seq_len=3, max_repeats=3)
        self.assertIsNotNone(result)
        self.assertIn("loop", result)

    def test_no_loop_below_repeat_threshold(self):
        seq = ["a", "b", "c"] * 2
        recent = deque(seq, maxlen=30)
        result = _detect_loop(recent, seq_len=3, max_repeats=3)
        self.assertIsNone(result)

    def test_insufficient_events_no_loop(self):
        recent = deque(["a", "b"], maxlen=30)
        result = _detect_loop(recent, seq_len=3, max_repeats=3)
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# Alert actions
# ---------------------------------------------------------------------------

class TestAlertTerminal(unittest.TestCase):
    def test_writes_to_stderr(self):
        buf = io.StringIO()
        _alert_terminal("test alert", out=buf)
        self.assertIn("test alert", buf.getvalue())
        self.assertIn("[watch]", buf.getvalue())


class TestAlertFile(unittest.TestCase):
    def test_writes_to_log_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = os.path.join(tmp, "alerts.log")
            _alert_file("file alert message", log_path=log_path)
            content = Path(log_path).read_text()
            self.assertIn("file alert message", content)

    def test_creates_parent_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = os.path.join(tmp, "subdir", "alerts.log")
            _alert_file("msg", log_path=log_path)
            self.assertTrue(Path(log_path).exists())


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

class TestWatcherConfigLoad(unittest.TestCase):
    def test_load_missing_file_returns_defaults(self):
        config = WatcherConfig.load("/nonexistent/path/.agent-watch.json")
        self.assertIsInstance(config, WatcherConfig)
        self.assertEqual(config.max_retries, 5)

    def test_load_valid_config(self):
        data = {
            "watchers": {
                "retry": {"max": 2, "alert": "terminal"},
                "cost": {"max_dollars": 5.0},
                "duration": {"max_minutes": 10},
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name
        try:
            config = WatcherConfig.load(path)
            self.assertEqual(config.max_retries, 2)
            self.assertAlmostEqual(config.max_cost_dollars, 5.0)
            self.assertAlmostEqual(config.max_duration_seconds, 600.0)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Event key
# ---------------------------------------------------------------------------

class TestEventKey(unittest.TestCase):
    def test_bash_event_key_includes_command(self):
        event = _bash_event("pytest tests/")
        key = _event_key(event)
        self.assertIn("Bash", key)
        self.assertIn("pytest", key)

    def test_non_tool_event_key_is_event_type(self):
        event = _make_event(EventType.USER_PROMPT, 0.0, prompt="hi")
        key = _event_key(event)
        self.assertEqual(key, "user_prompt")


if __name__ == "__main__":
    unittest.main()
