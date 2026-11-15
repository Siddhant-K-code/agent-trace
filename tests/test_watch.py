"""Tests for live session monitoring and circuit breakers."""

import argparse
import errno
import io
import json
import os
import socket
import sys
import tempfile
import threading
import time
import unittest
from collections import deque
from contextlib import redirect_stderr, redirect_stdout
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import call, patch, MagicMock

from agent_trace.cli import build_parser
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.project_budget import ProjectBudgetConfig
from agent_trace.store import TraceStore
from agent_trace.watch import (
    NannyRule,
    WatcherConfig,
    WatchState,
    check_event,
    _alert_terminal,
    _alert_file,
    _alert_webhook,
    _apply_builtin_rules_spec,
    _detect_loop,
    _detect_spin_loop,
    _dispatch_alert,
    _dispatch_nanny_rule,
    _event_key,
    cmd_watch,
    watch_session,
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
        loop_threshold=99,
        loop_window=10,
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
        state.estimated_cost = 0.01
        event = _make_event(EventType.ASSISTANT_RESPONSE, 0.0, text="x" * 10000)
        violations = check_event(event, config, state)
        self.assertTrue(any("CostWatcher" in v for v in violations))

    def test_cost_violation_fires_only_once(self):
        config = _default_config(max_cost_dollars=0.0)
        state = WatchState()
        state.estimated_cost = 0.01
        all_violations = []
        for _ in range(5):
            all_violations.extend(check_event(
                _make_event(EventType.ASSISTANT_RESPONSE, 0.0, text="x"),
                config, state,
            ))
        cost_violations = [v for v in all_violations if "CostWatcher" in v]
        self.assertEqual(len(cost_violations), 1)


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
    def test_detects_identical_tool_call_spin_loop(self):
        recent = deque([
            _event_key(_bash_event("pytest")),
            "user_prompt",
            _event_key(_bash_event("pytest")),
            _event_key(_bash_event("pytest")),
        ], maxlen=30)
        result = _detect_spin_loop(recent, threshold=3, window=10)
        self.assertIsNotNone(result)
        self.assertIn("identical tool call repeated 3 times", result)

    def test_spin_loop_respects_window(self):
        repeated = _event_key(_bash_event("pytest"))
        recent = deque([repeated, repeated, "user_prompt", "assistant_response", repeated], maxlen=30)
        result = _detect_spin_loop(recent, threshold=3, window=3)
        self.assertIsNone(result)

    def test_check_event_reports_spin_loop(self):
        config = _default_config(loop_threshold=3, loop_window=10)
        state = WatchState()
        violations = []
        for _ in range(3):
            violations.extend(check_event(_bash_event("pytest"), config, state))
        self.assertTrue(any("LoopWatcher" in v for v in violations))

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

    def test_dispatch_always_writes_terminal(self):
        from agent_trace.watch import _dispatch_alert, WatcherConfig, WatchState
        import unittest.mock as mock
        config = WatcherConfig(on_violation="file")
        state = WatchState()
        with mock.patch("agent_trace.watch._alert_terminal") as mock_terminal, \
             mock.patch("agent_trace.watch._alert_file"):
            _dispatch_alert("msg", config, state)
            mock_terminal.assert_called_once()


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
    def test_load_missing_file_is_refused(self):
        # A missing file used to mean "use the defaults", which silently
        # dropped every limit the file was meant to set.
        with self.assertRaises(ValueError) as ctx:
            WatcherConfig.load("/nonexistent/path/.agent-watch.json")
        self.assertEqual(str(ctx.exception), f"cannot read it ({os.strerror(errno.ENOENT)})")

    def test_load_valid_config(self):
        data = {
            "watchers": {
                "retry": {"max": 2, "alert": "terminal"},
                "cost": {"max_dollars": 5.0},
                "duration": {"max_minutes": 10},
                "loop": {"identical_calls": 4, "window": 12},
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
            self.assertEqual(config.loop_threshold, 4)
            self.assertEqual(config.loop_window, 12)
        finally:
            os.unlink(path)

    def test_from_dict_reads_every_known_key(self):
        config = WatcherConfig.from_dict({
            "watchers": {
                # A whole number may be written as 2.0.
                "retry": {"max": 2.0, "alert": "require_approval"},
                "cost": {"max_dollars": 0},
                "duration": {"max_minutes": 0.5},
                "loop": {"threshold": 4, "window": 12, "sequence_length": 2, "max_repeats": 5},
                "scope": {"policy": "scope.json"},
            },
            "webhook": {"url": ""},
            "operation_rules": [
                {"tool": "bash", "pattern": "rm -rf *", "action": "block", "reason": "no deletes"},
                {},
            ],
            "max_context_pct": 100,
            "built_in_rules": ["mcp-poisoning", "loop", "cognitive-debt", "loop"],
            "cognitive_debt_threshold": 0.5,
        })
        self.assertEqual(
            (
                config.max_retries, config.on_violation, config.max_cost_dollars,
                config.max_duration_seconds, config.loop_threshold, config.loop_window,
                config.loop_sequence_length, config.loop_max_repeats, config.scope_policy,
                config.webhook_url, config.max_context_pct, config.built_in_rules,
                config.cognitive_debt_threshold,
            ),
            (
                2, "require_approval", 0.0, 30.0, 4, 12, 2, 5, "scope.json", "", 100,
                {"mcp-poisoning", "loop", "cognitive-debt"}, 0.5,
            ),
        )
        self.assertEqual(
            [(r.tool_name, r.pattern, r.action, r.reason) for r in config.operation_rules],
            [("bash", "rm -rf *", "block", "no deletes"), ("*", "*", "alert", "")],
        )
        # identical_calls wins over threshold, which wins over max_repeats.
        loops = (
            ({"identical_calls": 6, "threshold": 4, "max_repeats": 5}, 6),
            ({"threshold": 4, "max_repeats": 5}, 4),
            ({"max_repeats": 5}, 5),
        )
        for loop, threshold in loops:
            with self.subTest(loop=loop):
                config = WatcherConfig.from_dict({"watchers": {"loop": loop}})
                self.assertEqual(config.loop_threshold, threshold)
        for action in ("terminal", "file", "kill", "pause", "require_approval"):
            with self.subTest(action=action):
                config = WatcherConfig.from_dict({"watchers": {"retry": {"alert": action}}})
                self.assertEqual(config.on_violation, action)

    def test_absent_keys_keep_the_defaults(self):
        documents = (
            {},
            {"watchers": {}},
            {
                "watchers": {"retry": {}, "cost": {}, "duration": {}, "loop": {}, "scope": {}},
                "webhook": {},
                "operation_rules": [],
                "built_in_rules": [],
                "cognitive_debt_threshold": None,
            },
        )
        for document in documents:
            with self.subTest(document=document):
                self.assertEqual(WatcherConfig.from_dict(document), WatcherConfig())


class TestBuiltInRules(unittest.TestCase):
    def test_builtin_rules_spec_enables_mcp_poisoning_and_limits(self):
        config = WatcherConfig()

        # Raises WatchRulesError for an item it cannot use.
        _apply_builtin_rules_spec(
            "mcp-poisoning,loop:4/12,budget:$5,timeout:30m,cognitive-debt:0.8",
            config,
        )

        self.assertIn("mcp-poisoning", config.built_in_rules)
        self.assertIn("loop", config.built_in_rules)
        self.assertEqual(config.loop_threshold, 4)
        self.assertEqual(config.loop_window, 12)
        self.assertEqual(config.max_cost_dollars, 5.0)
        self.assertEqual(config.max_duration_seconds, 1800.0)
        self.assertIn("cognitive-debt", config.built_in_rules)
        self.assertAlmostEqual(config.cognitive_debt_threshold, 0.8)

    def test_builtin_rules_spec_refuses_items_before_changing_config(self):
        import copy
        from agent_trace.watch import WatchRulesError
        built_ins = "the built-in rules are mcp-poisoning, loop, budget, timeout and cognitive-debt"
        cases = (
            ("loop:4/12,budget:2x", "'budget:2x': N in budget:N must be a finite number, 0 or more"),
            ("budget:3,cognitive-debt:2",
             "'cognitive-debt:2': N in cognitive-debt:N must be a number from 0 to 1"),
            ("mcp-poisoning,loop:0",
             "'loop:0': N and W in loop:N and loop:N/W must be whole numbers, 1 or more"),
            ("loop,timeout:0",
             "'timeout:0': D in timeout:D must be a duration more than 0, such as 30m or 2h"),
            ("mcp-poisoning,kill", f"'kill' is not a built-in rule or an existing file; {built_ins}"),
            ("budget:3,budget:4", "'budget:4' lists budget again; list each built-in rule once"),
            ("timeout:30m,,loop", "item 2 is empty"),
            (",", "it lists no rules"),
        )
        for spec, problem in cases:
            with self.subTest(spec=spec):
                config = WatcherConfig()
                before = copy.deepcopy(vars(config))
                with self.assertRaises(WatchRulesError) as ctx:
                    _apply_builtin_rules_spec(spec, config)
                self.assertEqual(str(ctx.exception), problem)
                self.assertEqual(vars(config), before)

    def test_watch_parser_registers_loop_flags(self):
        parser = build_parser()
        args = parser.parse_args(["watch", "--loop-threshold", "4", "--loop-window", "12"])
        self.assertEqual(args.command, "watch")
        self.assertEqual(args.loop_threshold, 4)
        self.assertEqual(args.loop_window, 12)

    def test_compaction_cli_and_watch_flags_are_registered(self):
        parser = build_parser()
        compaction = parser.parse_args([
            "compaction", "abc123", "--diff", "--behavior-diff",
            "--compaction-threshold", "0.65",
        ])
        self.assertEqual(compaction.command, "compaction")
        self.assertEqual(compaction.session_id, "abc123")
        self.assertTrue(compaction.diff)
        self.assertTrue(compaction.behavior_diff)
        self.assertEqual(compaction.compaction_threshold, 0.65)

        watch = parser.parse_args([
            "watch", "abc123", "--compaction-checkpoint", "--checkpoint-at", "0.75",
        ])
        self.assertTrue(watch.compaction_checkpoint)
        self.assertEqual(watch.checkpoint_at, 0.75)

    def test_cmd_watch_wires_compaction_checkpoint_watcher(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = TraceStore(tmp, redact=False)
            meta = SessionMeta(session_id="checkpoint-session", started_at=1.0)
            store.create_session(meta)
            args = build_parser().parse_args([
                "--trace-dir", tmp, "watch", meta.session_id,
                "--compaction-checkpoint", "--checkpoint-at", "0.75",
            ])

            with patch("agent_trace.watch.watch_session") as watch_session_mock:
                result = cmd_watch(args)

            self.assertEqual(result, 0)
            watcher = watch_session_mock.call_args.kwargs["checkpoint_watcher"]
            self.assertIsNotNone(watcher)
            self.assertEqual(watcher.checkpoint_at, 0.75)

    def test_watch_loop_checks_existing_and_new_checkpoint_state(self):
        class FakeCheckpointWatcher:
            def __init__(self):
                self.updated = []

            def checkpoint_current(self):
                return Path("initial-checkpoint.md")

            def update(self, event):
                self.updated.append(event)
                return Path("updated-checkpoint.md") if len(self.updated) == 1 else None

        with tempfile.TemporaryDirectory() as tmp:
            store = TraceStore(tmp, redact=False)
            meta = SessionMeta(session_id="checkpoint-loop", started_at=1.0)
            store.create_session(meta)
            event = _make_event(EventType.LLM_REQUEST, 2.0, input_tokens=160_000)
            end = _make_event(EventType.SESSION_END, 3.0, exit_code=0)
            watcher = FakeCheckpointWatcher()
            output = io.StringIO()

            with patch("agent_trace.watch._tail_events", return_value=iter([event, end])):
                watch_session(
                    store,
                    meta.session_id,
                    _default_config(),
                    out=output,
                    checkpoint_watcher=watcher,
                )

            self.assertEqual(watcher.updated, [event, end])
            self.assertIn("initial-checkpoint.md", output.getvalue())
            self.assertIn("updated-checkpoint.md", output.getvalue())

    def test_mcp_poisoning_rule_alerts_on_exfil_sequence(self):
        config = WatcherConfig(built_in_rules={"mcp-poisoning"})
        state = WatchState()
        read_event = _make_event(
            EventType.TOOL_CALL,
            0.0,
            tool_name="read_file",
            arguments={"file_path": "/home/me/.ssh/id_rsa"},
        )
        http_event = _make_event(
            EventType.TOOL_CALL,
            1.0,
            tool_name="http_request",
            arguments={"url": "https://example.com/collect"},
        )

        self.assertEqual(check_event(read_event, config, state), [])
        violations = check_event(http_event, config, state)

        self.assertTrue(any("McpPoisoningWatcher" in v for v in violations))

    def test_cognitive_debt_rule_alerts_on_live_write(self):
        config = WatcherConfig(built_in_rules={"cognitive-debt"}, cognitive_debt_threshold=0.8)
        state = WatchState()
        event = _make_event(
            EventType.FILE_WRITE,
            0.0,
            path="src/app.py",
        )

        violations = check_event(event, config, state)

        self.assertTrue(any("CognitiveDebtWatcher" in v for v in violations))


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


# ---------------------------------------------------------------------------
# --stream-to and --stream-flush-interval checks
# ---------------------------------------------------------------------------

class TestWatchStreamFlagChecks(unittest.TestCase):
    """cmd_watch refuses a bad --stream-to or --stream-flush-interval with one
    line and status 1 before it looks up the session or writes anything."""

    _URL = "http://127.0.0.1:9/SECRET-PATH?token=SECRET-QUERY"

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.trace_dir = self.root / "traces"
        store = TraceStore(self.trace_dir, redact=False)
        store.create_session(SessionMeta(session_id="stream-flags", started_at=1.0))
        store.append_event(
            "stream-flags",
            _make_event(EventType.TOOL_CALL, 2.0, session_id="stream-flags", tool_name="Bash"),
        )
        # A writable TraceStore writes this file back, so a refusal that
        # opened the store first changes the snapshot.
        (self.trace_dir / ".gitignore").unlink(missing_ok=True)
        self.config = self.root / "watch.json"
        self.config.write_text("{}")
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("AGENT_STRACE_TENANT_ID", None)

    def _snapshot(self):
        return {
            path.relative_to(self.root).as_posix(): path.read_bytes() if path.is_file() else None
            for path in sorted(self.root.rglob("*"))
        }

    def _run(self, *flags, session="stream-flags", config=False, tenant=False):
        argv = ["--trace-dir", str(self.trace_dir), "watch", session, *flags]
        if config:
            argv += ["--config", str(self.config)]
        if tenant:
            argv += ["--tenant-id", "acme"]
        args = build_parser().parse_args(argv)
        out, err = io.StringIO(), io.StringIO()
        with patch("agent_trace.watch.watch_session") as watch_session_mock, \
                redirect_stdout(out), redirect_stderr(err):
            status = cmd_watch(args)
        return status, out.getvalue(), err.getvalue(), watch_session_mock

    def _assert_refused(self, flags, line):
        for session in ("stream-flags", "missing-session"):
            for config in (False, True):
                for tenant in (False, True):
                    with self.subTest(flags=flags, session=session, config=config, tenant=tenant):
                        before = self._snapshot()
                        status, out, err, watch_session_mock = self._run(
                            *flags, session=session, config=config, tenant=tenant,
                        )
                        self.assertEqual((status, out, err), (1, "", line))
                        for marker in ("SECRET", "token"):
                            self.assertNotIn(marker, err)
                        watch_session_mock.assert_not_called()
                        self.assertEqual(self._snapshot(), before)
        with self.subTest(flags=flags, check="events stay untagged"):
            store = TraceStore(self.trace_dir, read_only=True)
            self.assertEqual(store.load_meta("stream-flags").tenant_id, "")
            self.assertEqual(
                [event.tenant_id for event in store.load_events("stream-flags")], [""],
            )

    def test_bad_stream_to_values_are_refused(self):
        c1_controls = ("\x85", "\x9b")
        controls = (" ", "\t", "\n", "\r", "\x00", "\x1f", "\x7f", *c1_controls)
        reasons = {
            "it must start with http:// or https://": (
                "",
                "notaurl/SECRET-PATH?token=SECRET-QUERY",
                "ftp//SECRET-HOST/SECRET-PATH?token=SECRET-QUERY",
                "ftp://SECRET-HOST/SECRET-PATH?token=SECRET-QUERY",
                "file:///SECRET-PATH?token=SECRET-QUERY",
            ),
            "it is not a valid URL": ("http://[::1/SECRET-PATH?token=SECRET-QUERY",),
            "it has no host": (
                "http:///SECRET-PATH?token=SECRET-QUERY",
                "https://?token=SECRET-QUERY",
                "http://:4318/SECRET-PATH?token=SECRET-QUERY",
            ),
            "it must not contain a user name or password": (
                "http://SECRET-USER:SECRET-PASS@SECRET-HOST/SECRET-PATH?token=SECRET-QUERY",
                "http://SECRET-USER@SECRET-HOST/SECRET-PATH?token=SECRET-QUERY",
            ),
            "its port must be a number from 0 to 65535": (
                "http://SECRET-HOST:99999/SECRET-PATH?token=SECRET-QUERY",
                "http://SECRET-HOST:SECRET-PORT/SECRET-PATH?token=SECRET-QUERY",
            ),
            "it must not contain spaces or control characters": (
                *(f"http://SECRET-HOST/SECRET{char}PATH?token=SECRET-QUERY" for char in controls),
                *(f"http://SECRET{char}HOST/SECRET-PATH?token=SECRET-QUERY" for char in c1_controls),
            ),
        }
        for reason, urls in reasons.items():
            for url in urls:
                self._assert_refused(
                    [f"--stream-to={url}"], f"[watch] invalid --stream-to value: {reason}\n",
                )

    def test_bad_stream_flush_intervals_are_refused(self):
        cases = {
            "nan": "it must be a finite number of seconds",
            "inf": "it must be a finite number of seconds",
            "-inf": "it must be a finite number of seconds",
            "1e999": "it must be a finite number of seconds",
            "-1": "it must not be negative",
            "-0.5": "it must not be negative",
        }
        for value, reason in cases.items():
            self._assert_refused(
                [f"--stream-to={self._URL}", f"--stream-flush-interval={value}"],
                f"[watch] invalid --stream-flush-interval value: {reason}\n",
            )

    def test_good_stream_values_are_accepted(self):
        cases = (
            ("https://collector.example/api/public/otel?token=x", "0"),
            ("http://[::1]:4318", "1e12"),
            ("HTTP://collector.example", "2"),
            (self._URL, "0.25"),
            # Non-ASCII characters that are not control characters
            ("http://collector.example/caf\u00e9", "1"),
            ("http://a\u202eb.example/x", "1"),
        )
        for url, interval in cases:
            for config in (False, True):
                with self.subTest(url=url, interval=interval, config=config):
                    status, _, err, watch_session_mock = self._run(
                        f"--stream-to={url}", f"--stream-flush-interval={interval}",
                        "--stream-otlp", config=config,
                    )
                    self.assertEqual((status, err), (0, ""))
                    stream_config = watch_session_mock.call_args.kwargs["stream_config"]
                    self.assertEqual(stream_config.url, url)
                    self.assertEqual(stream_config.flush_interval, float(interval))
                    self.assertTrue(stream_config.otlp)


# ---------------------------------------------------------------------------
# watch --config FILE checks and flag precedence
# ---------------------------------------------------------------------------

class TestWatchConfigFile(unittest.TestCase):
    """cmd_watch refuses a --config file it cannot use with one line and
    status 1 before it checks --agent-pid, looks up the session or writes
    anything. Flags given on the command line override the file."""

    # A file that sets every value a flag can override.
    _FILE = {
        "watchers": {
            "retry": {"max": 2, "alert": "pause"},
            "cost": {"max_dollars": 9},
            "duration": {"max_minutes": 5},
            "loop": {"identical_calls": 6, "window": 20},
            "scope": {"policy": "file-scope.json"},
        },
        "webhook": {"url": "http://file.example/hook"},
        "max_context_pct": 80,
    }
    _FILE_VALUES = {
        "max_retries": 2,
        "on_violation": "pause",
        "max_cost_dollars": 9.0,
        "max_duration_seconds": 300.0,
        "loop_threshold": 6,
        "loop_window": 20,
        "scope_policy": "file-scope.json",
        "webhook_url": "http://file.example/hook",
        "on_death_cmd": "",
        "max_context_pct": 80,
    }
    # Flags that must not change which problem is reported.
    _OTHER_FLAGS = (
        "--agent-pid", "1", "--max-retries", "2", "--on-violation", "kill",
        "--webhook", "http://127.0.0.1:9/hook", "--max-context-pct", "50", "--timeout", "1m",
    )

    @classmethod
    def setUpClass(cls):
        cls.parser = build_parser()

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.trace_dir = self.root / "traces"
        store = TraceStore(self.trace_dir, redact=False)
        store.create_session(SessionMeta(session_id="config-file", started_at=1.0))
        store.append_event(
            "config-file",
            _make_event(EventType.TOOL_CALL, 2.0, session_id="config-file", tool_name="Bash"),
        )
        # A writable TraceStore writes this file back, so a refusal that
        # opened the store first changes the snapshot.
        (self.trace_dir / ".gitignore").unlink(missing_ok=True)
        self.config = self.root / "watch.json"
        self._write({})
        # cmd_watch also reads project budget files from the working directory.
        cwd = os.getcwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, cwd)
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("AGENT_STRACE_TENANT_ID", None)

    def _write(self, data):
        """Write *data* to the config file: bytes as given, anything else as
        JSON (NaN and Infinity included)."""
        if not isinstance(data, bytes):
            data = json.dumps(data).encode("utf-8")
        self.config.write_bytes(data)

    def _snapshot(self):
        return {
            path.relative_to(self.root).as_posix(): path.read_bytes() if path.is_file() else None
            for path in sorted(self.root.rglob("*"))
        }

    def _run(self, *flags, session="config-file", config=None):
        argv = ["--trace-dir", str(self.trace_dir), "watch", session, *flags]
        if config is not None:
            argv += ["--config", config]
        args = self.parser.parse_args(argv)
        out, err = io.StringIO(), io.StringIO()
        with patch("agent_trace.watch.watch_session") as watch_session_mock, \
                redirect_stdout(out), redirect_stderr(err):
            status = cmd_watch(args)
        return status, out.getvalue(), err.getvalue(), watch_session_mock

    def _config_for(self, *flags, config=None):
        """Return the WatcherConfig that cmd_watch passes to watch_session."""
        status, out, err, watch_session_mock = self._run(*flags, config=config)
        self.assertEqual((status, out, err), (0, "", ""))
        return watch_session_mock.call_args.args[2]

    def _assert_status_1(self, flags, line, config=None):
        """Assert cmd_watch prints *line*, exits 1 and writes nothing, with
        and without a session that exists and an unusable --agent-pid."""
        for session, more_flags in (("config-file", ()), ("missing-session", ("--agent-pid", "1"))):
            with self.subTest(session=session, flags=(*flags, *more_flags)):
                before = self._snapshot()
                status, out, err, watch_session_mock = self._run(
                    *flags, *more_flags, session=session, config=config,
                )
                self.assertEqual((status, out, err), (1, "", line))
                self.assertNotIn("SECRET", err)
                watch_session_mock.assert_not_called()
                self.assertEqual(self._snapshot(), before)

    def _assert_refused(self, problem, config=None, shown=None):
        """Assert cmd_watch refuses the --config file with *problem*."""
        if config is None:
            config = str(self.config)
        line = f"[watch] invalid --config file {config if shown is None else shown}: {problem}\n"
        self._assert_status_1((), line, config=config)
        with self.subTest(check="other flags do not hide the problem"):
            before = self._snapshot()
            status, out, err, watch_session_mock = self._run(
                *self._OTHER_FLAGS, session="missing-session", config=config,
            )
            self.assertEqual((status, out, err), (1, "", line))
            watch_session_mock.assert_not_called()
            self.assertEqual(self._snapshot(), before)

    def test_files_it_cannot_read_are_refused(self):
        missing = self.root / "missing.json"
        folder = self.root / "folder.json"
        folder.mkdir()
        for path in (missing, folder):
            with self.subTest(path=path.name):
                with self.assertRaises(OSError) as ctx:
                    path.read_bytes()
                self._assert_refused(f"cannot read it ({ctx.exception.strerror})", config=str(path))

    @unittest.skipUnless(hasattr(os, "geteuid"), "needs POSIX file permissions")
    def test_file_without_read_permission_is_refused(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        locked = Path(tmp.name) / "locked.json"
        locked.write_text("{}")
        locked.chmod(0)
        self.addCleanup(locked.chmod, 0o600)
        try:
            locked.read_bytes()
        except PermissionError as exc:
            reason = exc.strerror
        else:
            self.skipTest("this user can read a file without read permission")
        self._assert_refused(f"cannot read it ({reason})", config=str(locked))

    def test_empty_and_unprintable_paths_are_shown_quoted(self):
        # An empty --config used to mean "no config file".
        self._assert_refused("the path is empty", config="", shown="''")
        odd = str(self.root / "watch\n.json")
        self._assert_refused(
            f"cannot read it ({os.strerror(errno.ENOENT)})", config=odd, shown=repr(odd),
        )

    def test_files_that_are_not_utf8_json_are_refused(self):
        cases = [
            (b'{"watchers": {"retry": {"alert": "SECRET\xff"}}}', "it is not valid UTF-8"),
            ('{"webhook": {"url": "SECRET-caf\u00e9"}}'.encode("latin-1"), "it is not valid UTF-8"),
            ('{"watchers": {}}'.encode("utf-16"), "it is not valid UTF-8"),
            (
                b'\xef\xbb\xbf{"watchers": {"retry": {"max": 1, "alert": "kill"}}}',
                "it starts with a byte order mark; save it without one",
            ),
            (b'{"watchers": ' + b"[" * 1_000_000, "it is nested too deeply"),
        ]
        if hasattr(sys, "get_int_max_str_digits"):
            cases.append(
                (b'{"watchers": {"retry": {"max": ' + b"1" * 5000 + b"}}}",
                 "it has a number with too many digits"),
            )
        texts = (
            # Case p from the report: a trailing comma.
            '{"watchers": {"retry": {"max": 1, "alert": "kill"}},}',
            "",
            "   \n",
            # YAML is not read as a watch config.
            "watchers:\n  retry:\n    max: 1\n",
            "{'SECRET': 1}",
            '{"watchers": {"retry": {"alert": "SECRET',
            '{"webhook": {"url": "http://SECRET"}} trailing SECRET',
        )
        for text in texts:
            # The message from json differs between Python versions.
            with self.assertRaises(json.JSONDecodeError) as ctx:
                json.loads(text)
            exc = ctx.exception
            cases.append((
                text.encode("utf-8"),
                f"it is not valid JSON ({exc.msg} at line {exc.lineno} column {exc.colno})",
            ))
        for data, problem in cases:
            with self.subTest(data=data[:60], problem=problem):
                self._write(data)
                self._assert_refused(problem)

    def test_sections_with_wrong_types_are_refused(self):
        cases = []
        for value in ([], "SECRET", None, 1, 0.5, True):
            cases += [
                (value, "the top level must be a JSON object"),
                ({"watchers": value}, "watchers must be a JSON object"),
                *(
                    ({"watchers": {section: value}}, f"watchers.{section} must be a JSON object")
                    for section in ("retry", "cost", "duration", "loop", "scope")
                ),
                ({"webhook": value}, "webhook must be a JSON object"),
                ({"operation_rules": [{"tool": "bash"}, value]},
                 "operation_rules[1] must be a JSON object"),
            ]
        for value in ({}, {"tool": "bash"}, "SECRET", None, 1, True):
            cases += [
                ({"operation_rules": value}, "operation_rules must be a JSON array"),
                ({"built_in_rules": value}, "built_in_rules must be a JSON array"),
            ]
        cases += [
            # Case o from the report: the webhook URL given as a string.
            (
                {
                    "watchers": {"retry": {"max": 1, "alert": "kill"}},
                    "webhook": "http://127.0.0.1:9/SECRET-PATH?token=SECRET-QUERY",
                },
                "webhook must be a JSON object",
            ),
            ({"built_in_rules": "loop"}, "built_in_rules must be a JSON array"),
        ]
        for document, problem in cases:
            with self.subTest(document=document):
                self._write(document)
                self._assert_refused(problem)

    def test_known_fields_with_wrong_types_or_values_are_refused(self):
        nan, inf = float("nan"), float("inf")
        not_whole = (1.5, "5", "SECRET", True, False, None, nan, inf, [], {})
        not_number = ("10", "SECRET", True, False, None, nan, inf, -inf, 10**400, [], {})
        not_string = (1, 0.5, True, None, ["SECRET"], {"SECRET": 1})

        def watchers(section, key, value):
            return {"watchers": {section: {key: value}}}

        cases = []
        for value in (-1, *not_whole):
            cases.append((watchers("retry", "max", value),
                          "watchers.retry.max must be a whole number, 0 or more"))
        for value in ("SECRET-ACTION", "Kill", "block", "", 1, True, None, ["kill"]):
            cases.append((
                watchers("retry", "alert", value),
                "watchers.retry.alert must be one of terminal, file, kill, pause, require_approval",
            ))
        for value in (-0.01, -1, *not_number):
            cases.append((watchers("cost", "max_dollars", value),
                          "watchers.cost.max_dollars must be a finite number, 0 or more"))
            cases.append((watchers("duration", "max_minutes", value),
                          "watchers.duration.max_minutes must be a finite number more than 0"))
        for key in ("identical_calls", "threshold", "max_repeats", "window", "sequence_length"):
            for value in (0, -1, *not_whole):
                cases.append((watchers("loop", key, value),
                              f"watchers.loop.{key} must be a whole number, 1 or more"))
        for value in ("", *not_string):
            cases.append((watchers("scope", "policy", value),
                          "watchers.scope.policy must be a non-empty string"))
        for value in not_string:
            cases.append(({"webhook": {"url": value}}, "webhook.url must be a string"))
        for key in ("tool", "pattern", "reason"):
            for value in not_string:
                cases.append(({"operation_rules": [{}, {key: value}]},
                              f"operation_rules[1].{key} must be a string"))
        for value in ("SECRET-ACTION", "kill", "Block", "", 1, True, None):
            cases.append(({"operation_rules": [{"action": value}]},
                          "operation_rules[0].action must be alert or block"))
        for value in (0, 101, -5, 50.5, "90", True, None, nan, inf, []):
            cases.append(({"max_context_pct": value},
                          "max_context_pct must be a whole number from 1 to 100"))
        for value in ("SECRET", "loop:5", "Loop", "budget", 1, None, True, []):
            cases.append((
                {"built_in_rules": ["loop", value]},
                "built_in_rules[1] must be one of mcp-poisoning, loop, cognitive-debt",
            ))
        for value in (-0.1, 1.5, "0.5", "SECRET", True, False, nan, inf, []):
            cases.append(({"cognitive_debt_threshold": value},
                          "cognitive_debt_threshold must be a number from 0 to 1"))
        for document, problem in cases:
            with self.subTest(document=document):
                self._write(document)
                self._assert_refused(problem)

    def test_a_duration_limit_of_0_is_refused(self):
        # It would fire at the first event: an alert at once, or a kill at
        # once when the action is kill. A cost limit of 0 is kept.
        for value in (0, 0.0, -0.0):
            with self.subTest(value=value):
                self._write({"watchers": {"duration": {"max_minutes": value}}})
                self._assert_refused(
                    "watchers.duration.max_minutes must be a finite number more than 0",
                )
        self._write({"watchers": {"cost": {"max_dollars": 0}}})
        self.assertEqual(self._config_for(config=str(self.config)).max_cost_dollars, 0.0)

    def test_unknown_keys_and_budget_keys_are_not_refused(self):
        documents = (
            {
                "watchers": {"retry": {"max": 2, "note": "x"}, "custom": [1, None]},
                "extra": {"anything": [1, "two", None]},
                "budget": {"weekly": 50, "warn_at": 0.5, "stop_at": 0.9, "per_session_max": 4},
            },
            {"watchers": {"retry": {"max": 2}}, "weekly": 50, "warn_at": 0.5,
             "stop_at": 0.9, "per_session_max": 4},
        )
        for document in documents:
            # per_session_max still wins over --max-cost, as it did before,
            # because the file's action is not kill.
            for flags in ((), ("--max-cost", "3")):
                with self.subTest(document=document, flags=flags):
                    self._write(document)
                    status, out, err, watch_session_mock = self._run(
                        *flags, config=str(self.config),
                    )
                    self.assertEqual((status, out, err), (0, "", ""))
                    config = watch_session_mock.call_args.args[2]
                    budget = watch_session_mock.call_args.kwargs["project_budget_config"]
                    self.assertEqual(config.max_retries, 2)
                    self.assertEqual(config.max_cost_dollars, 4.0)
                    self.assertEqual(
                        (budget.weekly, budget.warn_at, budget.stop_at, budget.per_session_max),
                        (50.0, 0.5, 0.9, 4.0),
                    )
        # Watch does not check the budget keys; project_budget.py ignores bad ones.
        self._write({"budget": "not an object", "weekly": "lots"})
        self.assertEqual(self._config_for(config=str(self.config)), WatcherConfig())

    def test_file_values_are_kept_when_no_flag_is_given(self):
        self._write(self._FILE)
        config = self._config_for(config=str(self.config))
        self.assertEqual({name: getattr(config, name) for name in self._FILE_VALUES},
                         self._FILE_VALUES)

    def test_given_flags_override_the_file(self):
        self._write(self._FILE)
        cases = (
            (("--max-retries", "7"), {"max_retries": 7}),
            (("--max-retries", "0"), {"max_retries": 0}),
            (("--max-cost", "3.5"), {"max_cost_dollars": 3.5}),
            (("--max-cost", "0"), {"max_cost_dollars": 0.0}),
            (("--budget", "4"), {"max_cost_dollars": 4.0}),
            # --budget wins over --max-cost and --timeout over
            # --max-duration, in either order.
            (("--max-cost", "3.5", "--budget", "4"), {"max_cost_dollars": 4.0}),
            (("--budget", "4", "--max-cost", "3.5"), {"max_cost_dollars": 4.0}),
            (("--max-duration", "120"), {"max_duration_seconds": 120}),
            (("--timeout", "90s"), {"max_duration_seconds": 90.0}),
            (("--max-duration", "120", "--timeout", "3m"), {"max_duration_seconds": 180.0}),
            (("--timeout", "3m", "--max-duration", "120"), {"max_duration_seconds": 180.0}),
            (("--on-violation", "kill"), {"on_violation": "kill"}),
            # A flag set to its default still overrides the file.
            (("--on-violation", "terminal"), {"on_violation": "terminal"}),
            (("--webhook", "http://flag.example/hook"), {"webhook_url": "http://flag.example/hook"}),
            (("--webhook", ""), {"webhook_url": ""}),
            (("--on-death", "echo {post_mortem_path}"), {"on_death_cmd": "echo {post_mortem_path}"}),
            (("--policy", "flag-scope.json"), {"scope_policy": "flag-scope.json"}),
            (("--policy", ""), {"scope_policy": ".agent-scope.json"}),
            (("--max-context-pct", "50"), {"max_context_pct": 50}),
            (("--max-context-pct", "90"), {"max_context_pct": 90}),
            (("--loop-threshold", "4", "--loop-window", "15"), {"loop_threshold": 4, "loop_window": 15}),
            # Case q from the report.
            (
                ("--webhook", "http://flag.example/hook", "--on-violation", "kill"),
                {"webhook_url": "http://flag.example/hook", "on_violation": "kill"},
            ),
        )
        for flags, changed in cases:
            with self.subTest(flags=flags):
                config = self._config_for(*flags, config=str(self.config))
                self.assertEqual(
                    {name: getattr(config, name) for name in self._FILE_VALUES},
                    {**self._FILE_VALUES, **changed},
                )

    def test_max_context_pct_flag_is_applied(self):
        self.assertEqual(self._config_for().max_context_pct, 90)
        for value in ("1", "50", "100"):
            with self.subTest(value=value):
                self.assertEqual(self._config_for("--max-context-pct", value).max_context_pct,
                                 int(value))
        # 120,000 of claude-sonnet-4's 200,000 tokens is 60%.
        request = _make_event(
            EventType.LLM_REQUEST, 2.0, model="claude-sonnet-4", input_tokens=120_000,
        )
        cases = (
            ((), None, False),
            (("--max-context-pct", "50"), None, True),
            ((), {"max_context_pct": 50}, True),
            (("--max-context-pct", "50"), {"max_context_pct": 80}, True),
            (("--max-context-pct", "70"), {"max_context_pct": 50}, False),
        )
        for flags, document, fires in cases:
            with self.subTest(flags=flags, document=document):
                config_path = None
                if document is not None:
                    self._write(document)
                    config_path = str(self.config)
                config = self._config_for(*flags, config=config_path)
                violations = check_event(request, config, WatchState())
                self.assertEqual(
                    any(v.startswith("TokenBudgetWatcher:") for v in violations), fires,
                )

    def test_bad_max_context_pct_is_refused(self):
        line = "[watch] invalid --max-context-pct value: it must be a whole number from 1 to 100\n"
        for value in ("0", "101", "-5"):
            for config in (None, str(self.config)):
                self._assert_status_1((f"--max-context-pct={value}",), line, config=config)

    def test_bad_timeout_is_refused_with_or_without_a_config_file(self):
        line = "[watch] invalid --timeout value: cannot parse duration: 'soon'\n"
        for config in (None, str(self.config)):
            self._assert_status_1(("--timeout", "soon"), line, config=config)

    def test_help_still_shows_the_defaults(self):
        out = io.StringIO()
        with redirect_stdout(out), self.assertRaises(SystemExit) as ctx:
            build_parser().parse_args(["watch", "--help"])
        self.assertEqual(ctx.exception.code, 0)
        subparsers = next(
            action for action in self.parser._actions
            if isinstance(action, argparse._SubParsersAction)
        )
        helps = {action.dest: action.help for action in subparsers.choices["watch"]._actions}
        defaults = (
            ("max_retries", "5"), ("max_cost", "10"), ("max_duration", "1800"),
            ("on_violation", "terminal"), ("max_context_pct", "90"),
        )
        for dest, default in defaults:
            with self.subTest(dest=dest):
                self.assertIn(f"(default: {default})", helps[dest])


# ---------------------------------------------------------------------------
# Alert channels: the alert log, --webhook, and Slack/Teams through notify.py
# ---------------------------------------------------------------------------

# No alert line may show these parts of a URL.
_MARKERS = ("SECRET", "token")
_USER_INFO = "SECRET-USER:SECRET-PASS@"
_NOT_FOLLOWED = " (redirects are not followed; give the final URL)"
_NOTIFY_ENV = ("AGENT_STRACE_SLACK_WEBHOOK", "AGENT_STRACE_TEAMS_WEBHOOK")


def _closed_port() -> int:
    """Return a local port that nothing listens on."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _clear_notify_env(test: unittest.TestCase) -> None:
    """Unset the Slack and Teams URLs for the rest of *test*."""
    env = patch.dict(os.environ)
    env.start()
    test.addCleanup(env.stop)
    for name in _NOTIFY_ENV:
        os.environ.pop(name, None)


class _AlertServer(ThreadingHTTPServer):
    """Answer each POST with the next scripted (status, hold) reply.

    A reply waits *hold* seconds, or until ``release`` is set, before it is
    sent. A 3xx reply points Location at a URL with markers in it, and every
    GET gets 200. ``on_post`` is called when a POST arrives.
    """

    daemon_threads = True
    block_on_close = False

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _AlertHandler)
        self.lock = threading.Lock()
        self.requests: list[tuple[str, str, bytes]] = []
        self.replies: list[tuple[int, float]] = []
        self.release = threading.Event()
        self.on_post = None

    def handle_error(self, request, client_address) -> None:
        pass  # the client timed out and closed the connection

    def script(self, *replies: tuple[int, float]) -> None:
        with self.lock:
            self.requests.clear()
            self.replies[:] = replies
        self.release.clear()

    def methods(self) -> list[str]:
        with self.lock:
            return [method for method, _, _ in self.requests]


class _AlertHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        server = self.server
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        with server.lock:
            server.requests.append(("POST", self.path, body))
            status, hold = server.replies.pop(0) if server.replies else (200, 0.0)
        if server.on_post is not None:
            server.on_post()
        if hold:
            server.release.wait(hold)
        self.send_response(status)
        if 300 <= status < 400:
            self.send_header("Location", "/SECRET-LOCATION?token=SECRET-QUERY")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        with self.server.lock:
            self.server.requests.append(("GET", self.path, b""))
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args: object) -> None:
        pass


class _AlertServerTestCase(unittest.TestCase):
    """Run a scripted local webhook receiver."""

    def setUp(self) -> None:
        self.server = _AlertServer()
        self.port = self.server.server_address[1]
        threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True,
        ).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.addCleanup(self.server.release.set)
        _clear_notify_env(self)
        os.environ["no_proxy"] = "*"
        self.url = f"http://127.0.0.1:{self.port}/SECRET-PATH?token=SECRET-QUERY"
        self.endpoint = f"http://127.0.0.1:{self.port}"

    def _assert_no_markers(self, text: str) -> None:
        for marker in _MARKERS:
            self.assertNotIn(marker, text)


class TestAlertWebhookFailures(_AlertServerTestCase):
    """A --webhook POST that fails prints one line naming only
    scheme://host[:port] and never raises."""

    def _post(self, url: str | None = None) -> str:
        err = io.StringIO()
        with redirect_stderr(err):
            _alert_webhook("alert message", self.url if url is None else url)
        self._assert_no_markers(err.getvalue())
        return err.getvalue()

    def test_2xx_answers_print_nothing(self):
        # Guard: unchanged from main.
        body = json.dumps({"text": "alert message", "source": "agent-strace"}).encode()
        for status in (200, 201, 202, 204):
            with self.subTest(status=status):
                self.server.script((status, 0.0))
                self.assertEqual(self._post(), "")
                self.assertEqual(
                    self.server.requests,
                    [("POST", "/SECRET-PATH?token=SECRET-QUERY", body)],
                )

    def test_error_answers_print_one_line(self):
        for status in (400, 401, 404, 500, 503):
            with self.subTest(status=status):
                self.server.script((status, 0.0))
                self.assertEqual(
                    self._post(),
                    f"[watch] Webhook POST to {self.endpoint} failed: "
                    f"HTTP Error {status}: {HTTPStatus(status).phrase}\n",
                )
                self.assertEqual(self.server.methods(), ["POST"])

    def test_redirects_are_not_followed(self):
        for status in (301, 302, 303, 307, 308):
            with self.subTest(status=status):
                self.server.script((status, 0.0))
                self.assertEqual(
                    self._post(),
                    f"[watch] Webhook POST to {self.endpoint} failed: "
                    f"HTTP Error {status}: {HTTPStatus(status).phrase}{_NOT_FOLLOWED}\n",
                )
                self.assertEqual(self.server.methods(), ["POST"])

    def test_refused_connection_prints_one_line(self):
        port = _closed_port()
        self.assertRegex(
            self._post(f"http://127.0.0.1:{port}/SECRET-PATH?token=SECRET-QUERY"),
            rf"\A\[watch\] Webhook POST to http://127\.0\.0\.1:{port} failed: "
            r"\[Errno \d+\] Connection refused\n\Z",
        )

    def test_timeout_prints_one_line(self):
        self.server.script((200, 30.0))
        with patch("agent_trace.watch._WEBHOOK_TIMEOUT", 0.2):
            started = time.monotonic()
            line = self._post()
            elapsed = time.monotonic() - started
        self.assertEqual(line, f"[watch] Webhook POST to {self.endpoint} failed: TimeoutError\n")
        self.assertLess(elapsed, 5.0)

    def test_unusable_urls_are_not_posted(self):
        cases = {
            "it must start with http:// or https://": (
                "hooks.example.com/services/T0/B0/SECRET-PATH?token=SECRET-QUERY",
                f"127.0.0.1:{self.port}/SECRET-PATH?token=SECRET-QUERY",
                f"localhost:{self.port}/SECRET-PATH?token=SECRET-QUERY",
            ),
            "it must not contain a user name or password": (
                f"http://{_USER_INFO}127.0.0.1:{self.port}/SECRET-PATH?token=SECRET-QUERY",
            ),
            "it must not contain spaces or control characters": (
                f"http://127.0.0.1:{self.port}/SECRET PATH?token=SECRET-QUERY",
            ),
        }
        for reason, urls in cases.items():
            for url in urls:
                with self.subTest(url=url):
                    self.server.script()
                    self.assertEqual(self._post(url), f"[watch] Webhook URL not used: {reason}\n")
                    self.assertEqual(self.server.methods(), [])


class TestAlertLogFailures(unittest.TestCase):
    """An alert log that cannot be written prints one line and never raises,
    so the alert goes on to its pause or kill."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        blocker = Path(tmp.name) / "not-a-directory"
        blocker.write_text("")
        self.blocked_log = str(blocker / "alerts.log")
        self.log = str(Path(tmp.name) / "alerts.log")
        _clear_notify_env(self)

    def _write(self, message: str, log_path: str) -> str:
        err = io.StringIO()
        with redirect_stderr(err):
            _alert_file(message, log_path)
        return err.getvalue()

    def _assert_one_line(self, text: str) -> None:
        self.assertRegex(text, r"\A\[watch\] Could not write alert log: [^\n]+\n\Z")

    def test_unwritable_log_prints_one_line(self):
        self._assert_one_line(self._write("alert message", self.blocked_log))

    def test_unencodable_message_prints_one_line(self):
        # A JSON "\ud800" escape in a tool argument reaches the message as a
        # lone surrogate, which UTF-8 cannot encode.
        self._assert_one_line(self._write("RetryWatcher: 'echo \ud800' repeated", self.log))

    def test_pause_and_kill_run_when_the_alert_log_fails(self):
        for action, target in (("kill", "_kill_process"), ("pause", "_pause_process")):
            with self.subTest(action=action):
                config = WatcherConfig(on_violation=action, alert_log=self.blocked_log)
                err = io.StringIO()
                with patch(f"agent_trace.watch.{target}") as signal_mock, \
                        patch("agent_trace.watch.pid_reused", return_value=False), \
                        patch("agent_trace.watch._alert_terminal"), \
                        redirect_stderr(err):
                    _dispatch_alert("RetryWatcher: retried", config, WatchState(agent_pid=4242))
                self.assertEqual(signal_mock.call_args.args[0], 4242)
                self._assert_one_line(err.getvalue())


class TestNetworkAlertsRunLast(unittest.TestCase):
    """The --webhook POST and Slack/Teams run after the pause or kill, the
    post-mortem and --on-death. The terminal line and the alert log come
    first."""

    _MSG = "RetryWatcher: Bash retried"
    _URL = "http://hooks.example/x"

    def setUp(self):
        self.calls: list[tuple] = []
        record = self.calls.append
        _clear_notify_env(self)
        os.environ["AGENT_STRACE_SLACK_WEBHOOK"] = "http://127.0.0.1:9/x"
        steps = {
            "_alert_terminal": lambda msg, *a, **k: record(("terminal", msg)),
            "_alert_file": lambda msg, *a, **k: record(("alert-log", msg)),
            "_pause_process": lambda pid: record(("pause", pid)),
            # True: SIGTERM was sent, so the post-mortem and --on-death follow.
            "_kill_process": lambda pid, *a: record(("kill", pid)) or True,
            "_write_watchdog_postmortem": (
                lambda *a, **k: (record(("post-mortem",)), Path("pm.json"))[1]
            ),
            "_invoke_on_death": lambda cmd, path: record(("on-death", path)),
            "_alert_webhook": lambda msg, url: record(("webhook", msg)),
            "pid_reused": lambda *a: False,
        }
        targets = {f"agent_trace.watch.{name}": step for name, step in steps.items()}
        targets["agent_trace.notify.notify_violation"] = (
            lambda **k: record(("slack/teams", k["action"]))
        )
        for target, step in targets.items():
            patcher = patch(target, side_effect=step)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _steps(self) -> list[str]:
        return [step for step, *_ in self.calls]

    def test_watcher_kill(self):
        config = WatcherConfig(
            on_violation="kill", webhook_url=self._URL, on_death_cmd="echo {post_mortem_path}",
        )
        _dispatch_alert(
            self._MSG, config, WatchState(agent_pid=4242), store=MagicMock(), session_id="s1",
        )
        self.assertEqual(self.calls, [
            ("terminal", self._MSG),
            ("alert-log", self._MSG),
            ("terminal", "Killing agent process 4242"),
            ("kill", 4242),
            ("post-mortem",),
            ("terminal", "Post-mortem written to pm.json"),
            ("on-death", Path("pm.json")),
            ("webhook", self._MSG),
            ("slack/teams", "kill"),
        ])

    def test_nanny_kill(self):
        # A rule kill takes the watcher's path: one post-mortem and one
        # --on-death, both after SIGTERM.
        rule = NannyRule(name="cost cap", condition="cost_usd > 0.5", action="kill")
        msg = "NannyRule 'cost cap': cost_usd > 0.5 → kill"
        config = WatcherConfig(webhook_url=self._URL, on_death_cmd="echo {post_mortem_path}")
        _dispatch_nanny_rule(
            rule, _bash_event("ls"), MagicMock(), "s1", WatchState(agent_pid=4242), config,
        )
        self.assertEqual(self.calls, [
            ("terminal", msg),
            ("alert-log", msg),
            ("terminal", "Killing agent process 4242"),
            ("kill", 4242),
            ("post-mortem",),
            ("terminal", "Post-mortem written to pm.json"),
            ("on-death", Path("pm.json")),
            ("webhook", msg),
            ("slack/teams", "kill"),
        ])

    def test_pause(self):
        config = WatcherConfig(on_violation="pause", webhook_url=self._URL)
        _dispatch_alert(self._MSG, config, WatchState(agent_pid=4242))
        self.assertEqual(self.calls, [
            ("terminal", self._MSG),
            ("alert-log", self._MSG),
            ("terminal", "Pausing agent process 4242 (SIGSTOP)"),
            ("pause", 4242),
            ("webhook", self._MSG),
            ("slack/teams", "pause"),
        ])

    def test_dry_run_sends_after_its_line(self):
        config = WatcherConfig(on_violation="kill", webhook_url=self._URL)
        _dispatch_alert(self._MSG, config, WatchState(agent_pid=4242), dry_run=True)
        self.assertEqual(self.calls, [
            ("terminal", self._MSG),
            ("alert-log", self._MSG),
            ("terminal", "[dry-run] would kill agent process"),
            ("webhook", self._MSG),
            ("slack/teams", "kill"),
        ])

    def test_legacy_slack_hint_posts_after_the_pause(self):
        del os.environ["AGENT_STRACE_SLACK_WEBHOOK"]
        config = WatcherConfig(on_violation="pause", webhook_url=self._URL)
        _dispatch_alert(self._MSG, config, WatchState(agent_pid=4242), notify="slack:#ops")
        self.assertEqual(self.calls, [
            ("terminal", self._MSG),
            ("alert-log", self._MSG),
            ("terminal", "Pausing agent process 4242 (SIGSTOP)"),
            ("pause", 4242),
            ("webhook", self._MSG),
            ("webhook", f"[slack:#ops] {self._MSG}"),
        ])

    def test_require_approval_sends_nothing_over_the_network(self):
        # Guard: unchanged from main.
        config = WatcherConfig(webhook_url=self._URL)
        rule = NannyRule(name="review", condition="cost_usd > 0.5", action="require_approval")
        with tempfile.TemporaryDirectory() as tmp:
            store = TraceStore(tmp, redact=False)
            store.create_session(SessionMeta(session_id="s1", started_at=1.0))
            for dry_run in (False, True):
                with self.subTest(path="watcher", dry_run=dry_run):
                    self.calls.clear()
                    _dispatch_alert(
                        self._MSG, config, WatchState(agent_pid=4242),
                        action="require_approval", notify="review", dry_run=dry_run,
                    )
                    self.assertEqual(self._steps()[0], "terminal")
                    self.assertNotIn("webhook", self._steps())
                    self.assertNotIn("slack/teams", self._steps())
                with self.subTest(path="nanny", dry_run=dry_run):
                    self.calls.clear()
                    _dispatch_nanny_rule(
                        rule, _bash_event("ls"), store, "s1", WatchState(agent_pid=4242),
                        config, dry_run=dry_run,
                    )
                    self.assertEqual(self._steps()[0], "terminal")
                    self.assertNotIn("webhook", self._steps())
                    self.assertNotIn("slack/teams", self._steps())


class TestNannyPostMortemOnlyOnKill(unittest.TestCase):
    """Only a kill nanny rule writes the post-mortem and runs --on-death."""

    def test_alert_pause_and_file_rules_write_no_post_mortem(self):
        # Guard: unchanged from main.
        _clear_notify_env(self)
        with tempfile.TemporaryDirectory() as tmp:
            store = TraceStore(tmp, redact=False)
            store.create_session(SessionMeta(session_id="s1", started_at=1.0))
            post_mortem = store._session_dir("s1") / "watchdog-postmortem.json"
            config = WatcherConfig(
                alert_log=str(Path(tmp) / "alerts.log"), on_death_cmd="true {post_mortem_path}",
            )
            # kill is the control: it shows the checks below can see a post-mortem.
            for action in ("alert", "pause", "file", "kill"):
                with self.subTest(action=action):
                    post_mortem.unlink(missing_ok=True)
                    rule = NannyRule(name="cost cap", condition="cost_usd > 0.5", action=action)
                    with patch("agent_trace.watch.subprocess.Popen") as popen, \
                            patch("agent_trace.watch._pause_process"), \
                            patch("agent_trace.watch._kill_process"), \
                            patch("agent_trace.watch.pid_reused", return_value=False), \
                            patch("agent_trace.watch._alert_terminal"):
                        _dispatch_nanny_rule(
                            rule, _bash_event("ls"), store, "s1", WatchState(agent_pid=4242),
                            config,
                        )
                    if action == "kill":
                        self.assertTrue(post_mortem.exists())
                        self.assertIn(
                            call(f"true {post_mortem}", shell=True), popen.call_args_list,
                        )
                    else:
                        self.assertFalse(post_mortem.exists())
                        popen.assert_not_called()


class TestAlertChannelsDoNotHoldTheKill(_AlertServerTestCase):
    """A slow, failing or unusable channel neither stops nor delays the kill."""

    def _dispatch(self, config: WatcherConfig, events: list[str]) -> tuple[str, float]:
        """Dispatch one kill alert; return stderr and seconds until the kill."""
        started = time.monotonic()
        killed_after: list[float] = []

        def kill(*args):
            killed_after.append(time.monotonic() - started)
            events.append("kill")
            self.server.release.set()

        err = io.StringIO()
        with patch("agent_trace.watch._kill_process", side_effect=kill), \
                patch("agent_trace.watch.pid_reused", return_value=False), \
                patch("agent_trace.watch._alert_file"), \
                patch("agent_trace.watch._alert_terminal"), \
                redirect_stderr(err):
            _dispatch_alert("RetryWatcher: retried", config, WatchState(agent_pid=4242))
        self.assertEqual(len(killed_after), 1)
        self._assert_no_markers(err.getvalue())
        return err.getvalue(), killed_after[0]

    def test_kill_is_sent_before_a_slow_webhook_post(self):
        events: list[str] = []
        self.server.on_post = lambda: events.append("webhook-received")
        self.server.script((200, 3.0))
        config = WatcherConfig(on_violation="kill", webhook_url=self.url)
        err, killed_after = self._dispatch(config, events)
        self.assertEqual(err, "")
        self.assertEqual(events, ["kill", "webhook-received"])
        self.assertLess(killed_after, 1.0)

    def test_kill_runs_when_the_webhook_url_is_unusable(self):
        config = WatcherConfig(
            on_violation="kill",
            webhook_url="hooks.example.com/services/T0/B0/SECRET-PATH?token=SECRET-QUERY",
        )
        events: list[str] = []
        err, _ = self._dispatch(config, events)
        self.assertEqual(events, ["kill"])
        self.assertEqual(
            err, "[watch] Webhook URL not used: it must start with http:// or https://\n",
        )

    def test_kill_runs_when_slack_or_teams_raises(self):
        os.environ["AGENT_STRACE_SLACK_WEBHOOK"] = self.url
        config = WatcherConfig(on_violation="kill")
        events: list[str] = []
        with patch(
            "agent_trace.notify.notify_violation",
            side_effect=RuntimeError("SECRET-PATH?token=SECRET-QUERY"),
        ):
            err, _ = self._dispatch(config, events)
        self.assertEqual(events, ["kill"])
        self.assertEqual(err, "[watch] Slack/Teams notification failed: RuntimeError\n")


class TestProjectBudgetAlertFailures(unittest.TestCase):
    """A project budget warning whose alert log or webhook fails prints one
    line for each, and watch goes on."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        blocker = root / "not-a-directory"
        blocker.write_text("")
        self.store = TraceStore(root / "traces", redact=False)
        self.session_id = "budget-session"
        self.store.create_session(SessionMeta(session_id=self.session_id, started_at=time.time()))
        self.port = _closed_port()
        self.config = WatcherConfig(
            alert_log=str(blocker / "alerts.log"),
            webhook_url=f"http://127.0.0.1:{self.port}/SECRET-PATH?token=SECRET-QUERY",
        )
        _clear_notify_env(self)
        os.environ["no_proxy"] = "*"

    def _watch(self, budget: ProjectBudgetConfig, events: list[TraceEvent]) -> tuple[str, str]:
        def tail(_path, poll_interval=0.5):
            yield from events

        out, err = io.StringIO(), io.StringIO()
        with patch("agent_trace.watch._tail_events", tail), redirect_stderr(err):
            watch_session(
                self.store, self.session_id, self.config, out=out,
                project_budget_config=budget,
            )
        for marker in _MARKERS:
            self.assertNotIn(marker, err.getvalue())
        return out.getvalue(), err.getvalue()

    def _end(self) -> TraceEvent:
        return _make_event(EventType.SESSION_END, time.time(), session_id=self.session_id)

    def test_startup_warning(self):
        out, err = self._watch(ProjectBudgetConfig(weekly=1.0, warn_at=0.0), [self._end()])
        self.assertIn("ProjectBudget", out)
        self.assertIn("[watch] Session ended", out)
        self.assertRegex(err, r"\A\[watch\] Could not write alert log: [^\n]+\n\Z")

    def test_warning_while_watching(self):
        events = [
            _make_event(
                EventType.ASSISTANT_RESPONSE, time.time(), session_id=self.session_id,
                text="x" * 10000,
            ),
            self._end(),
        ]
        out, err = self._watch(ProjectBudgetConfig(weekly=0.01, warn_at=0.5), events)
        self.assertEqual(out.count("ProjectBudget"), 1)
        self.assertIn("[watch] Session ended", out)
        self.assertRegex(
            err,
            r"\A\[watch\] Could not write alert log: [^\n]+\n"
            rf"\[watch\] Webhook POST to http://127\.0\.0\.1:{self.port} failed: "
            r"\[Errno \d+\] Connection refused\n\Z",
        )


# ---------------------------------------------------------------------------
# --webhook and webhook.url checks
# ---------------------------------------------------------------------------

_BAD_WEBHOOK_URLS = {
    "it must start with http:// or https://": (
        "hooks.example.com/services/T0/B0/SECRET-PATH?token=SECRET-QUERY",
        "127.0.0.1:9/SECRET-PATH?token=SECRET-QUERY",
        "ftp://SECRET-HOST/SECRET-PATH?token=SECRET-QUERY",
    ),
    "it is not a valid URL": ("http://[::1/SECRET-PATH?token=SECRET-QUERY",),
    "it has no host": ("http:///SECRET-PATH?token=SECRET-QUERY",),
    "it must not contain a user name or password": (
        f"http://{_USER_INFO}SECRET-HOST/SECRET-PATH?token=SECRET-QUERY",
    ),
    "its port must be a number from 0 to 65535": (
        "http://SECRET-HOST:99999/SECRET-PATH?token=SECRET-QUERY",
    ),
    "it must not contain spaces or control characters": (
        "http://SECRET-HOST/SECRET PATH?token=SECRET-QUERY",
        "http://SECRET-HOST/SECRET\x00PATH?token=SECRET-QUERY",
    ),
}


class TestWatchWebhookCheck(unittest.TestCase):
    """cmd_watch refuses a bad --webhook or config file webhook.url with one
    line and status 1 before it checks --agent-pid, looks up the session or
    writes anything."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.trace_dir = self.root / "traces"
        store = TraceStore(self.trace_dir, redact=False)
        store.create_session(SessionMeta(session_id="webhook-check", started_at=1.0))
        store.append_event(
            "webhook-check",
            _make_event(EventType.TOOL_CALL, 2.0, session_id="webhook-check", tool_name="Bash"),
        )
        # A writable TraceStore writes this file back, so a refusal that
        # opened the store first changes the snapshot.
        (self.trace_dir / ".gitignore").unlink(missing_ok=True)
        self.config = self.root / "watch.json"
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("AGENT_STRACE_TENANT_ID", None)

    def _snapshot(self):
        return {
            path.relative_to(self.root).as_posix(): path.read_bytes() if path.is_file() else None
            for path in sorted(self.root.rglob("*"))
        }

    def _run(self, *flags, session="webhook-check", config=False, tenant=False, agent_pid=False):
        argv = ["--trace-dir", str(self.trace_dir), "watch", session, *flags]
        if config:
            argv += ["--config", str(self.config)]
        if tenant:
            argv += ["--tenant-id", "acme"]
        if agent_pid:
            # Refused by the --agent-pid check, so the webhook check must come first.
            argv += ["--agent-pid", "1"]
        args = build_parser().parse_args(argv)
        out, err = io.StringIO(), io.StringIO()
        with patch("agent_trace.watch.watch_session") as watch_session_mock, \
                redirect_stdout(out), redirect_stderr(err):
            status = cmd_watch(args)
        return status, out.getvalue(), err.getvalue(), watch_session_mock

    def _write_config(self, url: str) -> None:
        self.config.write_text(json.dumps({"webhook": {"url": url}}))

    def _assert_refused(self, flags, line, config=False):
        for session in ("webhook-check", "missing-session"):
            for tenant in (False, True):
                for agent_pid in (False, True):
                    with self.subTest(
                        flags=flags, config=config, session=session, tenant=tenant,
                        agent_pid=agent_pid,
                    ):
                        before = self._snapshot()
                        status, out, err, watch_session_mock = self._run(
                            *flags, session=session, config=config, tenant=tenant,
                            agent_pid=agent_pid,
                        )
                        self.assertEqual((status, out, err), (1, "", line))
                        for marker in _MARKERS:
                            self.assertNotIn(marker, err)
                        watch_session_mock.assert_not_called()
                        self.assertEqual(self._snapshot(), before)
        with self.subTest(flags=flags, config=config, check="events stay untagged"):
            store = TraceStore(self.trace_dir, read_only=True)
            self.assertEqual(store.load_meta("webhook-check").tenant_id, "")
            self.assertEqual(
                [event.tenant_id for event in store.load_events("webhook-check")], [""],
            )

    def test_bad_webhook_flag_is_refused(self):
        for reason, urls in _BAD_WEBHOOK_URLS.items():
            for url in urls:
                self._assert_refused(
                    [f"--webhook={url}"], f"[watch] invalid --webhook value: {reason}\n",
                )

    def test_bad_config_webhook_url_is_refused(self):
        for reason, urls in _BAD_WEBHOOK_URLS.items():
            for url in urls:
                self._write_config(url)
                self._assert_refused(
                    [], f"[watch] invalid webhook.url value in {self.config}: {reason}\n",
                    config=True,
                )

    def test_bad_flag_is_named_even_with_a_config_file(self):
        # A given --webhook overrides webhook.url, so the line names the flag.
        for file_url in ("https://hooks.example.com/x", "hooks.example.com/SECRET"):
            self._write_config(file_url)
            self._assert_refused(
                ["--webhook=hooks.example.com/services/SECRET"],
                "[watch] invalid --webhook value: it must start with http:// or https://\n",
                config=True,
            )

    def test_flag_replaces_a_bad_config_url_before_the_check(self):
        # Guard: unchanged from main. Watch checks only the URL it POSTs to.
        self._write_config("hooks.example.com/services/SECRET")
        for flags, expected in (
            (["--webhook=https://hooks.example.com/x"], "https://hooks.example.com/x"),
            (["--webhook="], ""),
        ):
            with self.subTest(flags=flags):
                status, _, err, watch_session_mock = self._run(*flags, config=True)
                self.assertEqual((status, err), (0, ""))
                self.assertEqual(watch_session_mock.call_args.args[2].webhook_url, expected)

    def test_unprintable_config_path_is_shown_quoted(self):
        self.config = self.root / "watch\n.json"
        try:
            self._write_config("hooks.example.com/services/SECRET")
        except OSError:
            self.skipTest("this file system refuses a newline in a file name")
        self._assert_refused(
            [],
            f"[watch] invalid webhook.url value in {str(self.config)!r}: "
            "it must start with http:// or https://\n",
            config=True,
        )

    def test_empty_webhook_means_no_webhook(self):
        # Guard: unchanged from main.
        self._write_config("")
        for flags, config in ((["--webhook="], False), ([], True)):
            with self.subTest(flags=flags, config=config):
                status, _, err, watch_session_mock = self._run(*flags, config=config)
                self.assertEqual((status, err), (0, ""))
                self.assertEqual(watch_session_mock.call_args.args[2].webhook_url, "")

    def test_good_webhook_urls_are_accepted(self):
        # Guard: unchanged from main.
        urls = (
            "https://hooks.example.com/services/T0/B0/x?token=y",
            "http://[::1]:8080/hook",
            "HTTP://hooks.example/x",
            "http://127.0.0.1:9/SECRET-PATH?token=SECRET-QUERY",
        )
        for url in urls:
            self._write_config(url)
            for flags, config in (([f"--webhook={url}"], False), ([], True)):
                with self.subTest(url=url, config=config):
                    status, _, err, watch_session_mock = self._run(*flags, config=config)
                    self.assertEqual((status, err), (0, ""))
                    self.assertEqual(watch_session_mock.call_args.args[2].webhook_url, url)


if __name__ == "__main__":
    unittest.main()
