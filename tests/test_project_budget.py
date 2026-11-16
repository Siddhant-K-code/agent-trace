"""Tests for per-project rolling budget guardrails."""

from __future__ import annotations

import io
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import agent_trace.project_budget as project_budget
from agent_trace.cli import build_parser, cmd_record, cmd_record_http
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.project_budget import (
    ProjectBudgetConfig,
    enforce_new_session_budget,
    load_project_budget_config,
    project_budget_status,
    rolling_project_spend,
)
from agent_trace.store import TraceStore
from agent_trace.watch import WatcherConfig, watch_session


def _add_session(store: TraceStore, started_at: float, payload_size: int = 4000) -> str:
    meta = SessionMeta(agent_name="agent", command="test")
    meta.started_at = started_at
    session_path = store.create_session(meta)
    session_id = session_path.name
    store.append_event(
        session_id,
        TraceEvent(
            event_type=EventType.SESSION_START,
            timestamp=started_at,
            session_id=session_id,
            data={},
        ),
    )
    store.append_event(
        session_id,
        TraceEvent(
            event_type=EventType.ASSISTANT_RESPONSE,
            timestamp=started_at + 1,
            session_id=session_id,
            data={"text": "x" * payload_size},
        ),
    )
    store.append_event(
        session_id,
        TraceEvent(
            event_type=EventType.SESSION_END,
            timestamp=started_at + 2,
            session_id=session_id,
            data={},
        ),
    )
    return session_id


class TestProjectBudgetConfig(unittest.TestCase):
    def test_loads_budget_block_from_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".agent-strace.yaml"
            path.write_text(
                "\n".join([
                    "budget:",
                    "  weekly: 20.00",
                    "  warn_at: 0.75",
                    "  stop_at: 1.00",
                    "  per_session_max: 5.50",
                ])
            )

            cfg = load_project_budget_config(path)

        self.assertEqual(cfg.weekly, 20.0)
        self.assertEqual(cfg.warn_at, 0.75)
        self.assertEqual(cfg.stop_at, 1.0)
        self.assertEqual(cfg.per_session_max, 5.5)

    def test_missing_config_disables_budget(self) -> None:
        cfg = load_project_budget_config("/no/such/.agent-strace.yaml")
        self.assertFalse(cfg.enabled)


class TestRollingProjectSpend(unittest.TestCase):
    def test_rolling_spend_excludes_old_sessions(self) -> None:
        now = time.time()
        with tempfile.TemporaryDirectory() as tmp:
            store = TraceStore(Path(tmp))
            recent_id = _add_session(store, now - 60)
            _add_session(store, now - 8 * 24 * 60 * 60)

            total = rolling_project_spend(store, now=now)
            without_recent = rolling_project_spend(
                store, now=now, exclude_session_id=recent_id
            )

        self.assertGreater(total, 0.0)
        self.assertEqual(without_recent, 0.0)

    def test_status_warns_and_stops_at_thresholds(self) -> None:
        now = time.time()
        with tempfile.TemporaryDirectory() as tmp:
            store = TraceStore(Path(tmp))
            _add_session(store, now - 60, payload_size=20000)
            spent = rolling_project_spend(store, now=now)
            cfg = ProjectBudgetConfig(
                weekly=spent,
                warn_at=0.50,
                stop_at=1.00,
            )

            status = project_budget_status(store, cfg, now=now)

        self.assertTrue(status.should_warn)
        self.assertTrue(status.should_stop)

    def test_enforce_new_session_budget_refuses_over_stop(self) -> None:
        now = time.time()
        with tempfile.TemporaryDirectory() as tmp:
            store = TraceStore(Path(tmp))
            _add_session(store, now - 60, payload_size=20000)
            spent = rolling_project_spend(store, now=now)
            cfg = ProjectBudgetConfig(weekly=spent / 2, stop_at=1.00)
            out = io.StringIO()

            allowed = enforce_new_session_budget(store, cfg, out, now=now)

        self.assertFalse(allowed)
        self.assertIn("refusing new session", out.getvalue())


class TestWatchProjectBudget(unittest.TestCase):
    def _fake_tail(self, events: list[TraceEvent]):
        def _gen(_path, poll_interval=0.5):
            yield from events
        return _gen

    def test_watch_session_writes_project_budget_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TraceStore(Path(tmp) / "traces")
            session_path = store.create_session(SessionMeta(agent_name="agent"))
            session_id = session_path.name
            out = io.StringIO()
            alert_log = Path(tmp) / "alerts.log"
            cfg = WatcherConfig(alert_log=str(alert_log))
            budget = ProjectBudgetConfig(weekly=1.0, warn_at=0.0)
            events = [
                TraceEvent(
                    event_type=EventType.SESSION_END,
                    timestamp=time.time(),
                    session_id=session_id,
                    data={},
                )
            ]

            with patch("agent_trace.watch._tail_events", self._fake_tail(events)):
                watch_session(
                    store,
                    session_id,
                    cfg,
                    out=out,
                    project_budget_config=budget,
                )

            self.assertIn("ProjectBudget", out.getvalue())
            self.assertIn("ProjectBudget", alert_log.read_text())

    def test_per_session_max_cost_violation_uses_kill_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TraceStore(Path(tmp))
            session_path = store.create_session(SessionMeta(agent_name="agent"))
            session_id = session_path.name
            cfg = WatcherConfig(max_cost_dollars=0.0)
            budget = ProjectBudgetConfig(
                weekly=10.0,
                warn_at=2.0,
                per_session_max=1.0,
            )
            events = [
                TraceEvent(
                    event_type=EventType.ASSISTANT_RESPONSE,
                    timestamp=time.time(),
                    session_id=session_id,
                    data={"text": "x" * 10000},
                ),
                TraceEvent(
                    event_type=EventType.SESSION_END,
                    timestamp=time.time(),
                    session_id=session_id,
                    data={},
                ),
            ]

            with patch("agent_trace.watch._tail_events", self._fake_tail(events)), \
                 patch("agent_trace.watch._dispatch_alert") as dispatch:
                watch_session(
                    store,
                    session_id,
                    cfg,
                    out=io.StringIO(),
                    project_budget_config=budget,
                )

        self.assertEqual(dispatch.call_args.kwargs["action"], "kill")


_UNKNOWN = "; use weekly, warn_at, stop_at or per_session_max"
_NO_BUDGET = (0.0, 0.8, 1.0, None)
# The int digit limit, 0 when this Python has none.
_INT_DIGITS = getattr(sys, "get_int_max_str_digits", lambda: 0)()


def _json_problem(text: str) -> str:
    """Return how a refusal describes what json.loads finds wrong in *text*."""
    try:
        json.loads(text)
    except json.JSONDecodeError as exc:
        return f"it is not valid JSON ({exc.msg} at line {exc.lineno} column {exc.colno})"
    except RecursionError:
        return "it is nested too deeply"
    raise AssertionError(f"{text[:40]!r} is valid JSON")


def _dollars(name: str, shown: str) -> str:
    return f"{name} must be a finite number, 0 or more, not {shown}"


def _fraction(name: str, shown: str) -> str:
    return (
        f"{name} must be a number from 0 to 10, a fraction of weekly such as 0.9 "
        f"for 90%, not {shown}"
    )


class TestProjectBudgetConfigFile(unittest.TestCase):
    """load_project_budget_config refuses a budget config it cannot use.

    On main it returned a disabled budget for each file refused here, so
    watch, record and record-http ran without the budget and said nothing.
    """

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        cwd = os.getcwd()
        os.chdir(tmp.name)
        self.addCleanup(os.chdir, cwd)

    def _write(self, data, name: str = ".agent-strace.yaml") -> Path:
        """Write *data* to *name*: bytes as given, a str as UTF-8, else as JSON."""
        path = Path(name)
        if isinstance(data, bytes):
            path.write_bytes(data)
        elif isinstance(data, str):
            path.write_bytes(data.encode("utf-8"))
        else:
            path.write_bytes(json.dumps(data).encode("utf-8"))
        return path

    def _load(self, data, name: str = ".agent-strace.yaml") -> tuple:
        """Return (weekly, warn_at, stop_at, per_session_max) for *data* in *name*."""
        path = self._write(data, name)
        try:
            cfg = load_project_budget_config()
        finally:
            path.unlink()
        return (cfg.weekly, cfg.warn_at, cfg.stop_at, cfg.per_session_max)

    def _assert_refused(self, data, problem: str, name: str = ".agent-strace.yaml") -> None:
        path = self._write(data, name)
        try:
            self._assert_load_refused(f"invalid project budget in {name}: {problem}")
        finally:
            path.unlink()

    def _assert_load_refused(self, message: str, *args) -> None:
        with self.assertRaises(ValueError) as caught:
            load_project_budget_config(*args)
        self.assertIs(type(caught.exception), project_budget.ProjectBudgetConfigError)
        self.assertEqual(str(caught.exception), message)

    def test_dollar_amounts_it_cannot_use_are_refused(self):
        many_digits = "1" + "0" * 5000
        for value, shown in [
            ("-5", "-5"),
            ("-0.01", "-0.01"),
            ("$-5", "'$-5'"),
            ("$$20", "'$$20'"),
            ("$", "'$'"),
            ("twenty", "'twenty'"),
            ("20 USD", "'20 USD'"),
            ("20#x", "'20#x'"),
            ("0x10", "'0x10'"),
            ("nan", "'nan'"),
            (".nan", "'.nan'"),
            ("inf", "'inf'"),
            ("-.inf", "'-.inf'"),
            ("1e400", "inf"),
            ("true", "true"),
            ("yes", "'yes'"),
            ('"-5"', "'-5'"),
            ("''", "''"),
            ('"null"', "'null'"),
            # int() refuses so many digits when the digit limit is on.
            (many_digits, "a string of 5001 characters"
             if 0 < _INT_DIGITS < 5001 else "a number too long to show"),
        ]:
            for key in ("weekly", "per_session_max"):
                with self.subTest(value=value[:20], key=key):
                    self._assert_refused(
                        f"budget:\n  {key}: {value}\n", _dollars(f"budget.{key}", shown),
                    )
                    self._assert_refused(f"{key}: {value}\n", _dollars(key, shown))

    def test_json_dollar_amounts_it_cannot_use_are_refused(self):
        for value, shown in [
            (-5, "-5"),
            (-0.5, "-0.5"),
            (float("nan"), "nan"),
            (float("inf"), "inf"),
            (float("-inf"), "-inf"),
            (True, "true"),
            (False, "false"),
            ("twenty", "'twenty'"),
            ("$-5", "'$-5'"),
            ("x" * 41, "a string of 41 characters"),
            (-10 ** 50, "a number too long to show"),
            ([20], "a list"),
            ({"amount": 20}, "a JSON object"),
        ]:
            for key in ("weekly", "per_session_max"):
                with self.subTest(value=value, key=key):
                    self._assert_refused(
                        {"budget": {key: value}}, _dollars(f"budget.{key}", shown),
                        ".agent-strace.json",
                    )
                    self._assert_refused(
                        {key: value}, _dollars(key, shown), ".agent-strace.json",
                    )

    def test_fractions_it_cannot_use_are_refused(self):
        """A percent means that many times weekly: stop_at 90 stopped a $20 budget at $1800."""
        for value, shown in [
            ("90", "90"),
            ("10.5", "10.5"),
            ("-0.1", "-0.1"),
            ("90%", "'90%'"),
            ("$0.9", "'$0.9'"),
            ("nan", "'nan'"),
            ("1e400", "inf"),
            ("true", "true"),
            ('"90"', "'90'"),
        ]:
            for key in ("warn_at", "stop_at"):
                with self.subTest(value=value, key=key):
                    self._assert_refused(
                        f"budget:\n  weekly: 20\n  {key}: {value}\n",
                        _fraction(f"budget.{key}", shown),
                    )
                    self._assert_refused(
                        f"weekly: 20\n{key}: {value}\n", _fraction(key, shown),
                    )
        for data, problem in [
            ({"budget": {"weekly": 20, "stop_at": 90}}, _fraction("budget.stop_at", "90")),
            ({"budget": {"warn_at": float("nan")}}, _fraction("budget.warn_at", "nan")),
            ({"weekly": 20, "warn_at": [0.5]}, _fraction("warn_at", "a list")),
        ]:
            with self.subTest(data=data):
                self._assert_refused(data, problem, ".agent-strace.json")

    def test_yaml_budget_blocks_it_cannot_use_are_refused(self):
        flow = (
            "line 1: budget uses flow style; write each key on its own line, "
            "indented under budget:"
        )
        backslash = (
            'line 1: in double quotes a backslash may only escape " or \\; '
            "use single quotes for other backslashes"
        )
        for text, problem in [
            ("budget: 20\n", "budget must be a mapping, not 20"),
            ("budget: weekly\n", "budget must be a mapping, not 'weekly'"),
            ("budget: true\n", "budget must be a mapping, not true"),
            ("budget:\n  - weekly: 20\n", "line 2: budget must be a mapping, not a list"),
            ("budget:\n- weekly: 20\n", "line 2: budget must be a mapping, not a list"),
            ("budget: {weekly: 20, per_session_max: 5}\n", flow),
            ("budget: [20]\n", flow),
            (
                "review:\n  max_lines: 400\n{weekly: 20}\n",
                "line 3: flow style is not supported; write one key per line",
            ),
            (
                "budget:\n  weekly: 20\nper_session_max: 5\n",
                "it has both budget and a top-level per_session_max; "
                "move per_session_max under budget",
            ),
            (
                "weekly: 20\nbudget:\n  per_session_max: 5\n",
                "it has both budget and a top-level weekly; move weekly under budget",
            ),
            (
                "budget:\n  weekly: 20\n  per_session: 5\n",
                f"line 3: budget has an unknown key 'per_session'{_UNKNOWN}",
            ),
            ("budget:\n  weekley: 20\n", f"line 2: budget has an unknown key 'weekley'{_UNKNOWN}"),
            ("budget:\n  Weekly: 20\n", f"line 2: budget has an unknown key 'Weekly'{_UNKNOWN}"),
            ("budget:\n  monthly: 80\n", f"line 2: budget has an unknown key 'monthly'{_UNKNOWN}"),
            ("budget:\n  weekly: 20\n  weekly: 30\n", "line 3: budget.weekly is given twice"),
            ("budget:\n  weekly: 20\nbudget:\n  weekly: 30\n", "line 3: budget is given twice"),
            ("weekly: 20\nweekly: 30\n", "line 2: weekly is given twice"),
            ("budget:\n  weekly: 20\n    per_session_max: 5\n", "line 3: unexpected indentation"),
            ("budget:\n    weekly: 20\n  per_session_max: 5\n", "line 3: unexpected indentation"),
            ("budget:\n  weekly:\n    amount: 20\n", "line 3: unexpected indentation"),
            ("weekly: 20\n  per_session_max: 5\n", "line 2: unexpected indentation"),
            ("per_session_max:\n  5\n", "line 2: unexpected indentation"),
            ("  budget:\n    weekly: 20\n", "line 1: unexpected indentation"),
            ("- budget:\n    weekly: 20\n", "line 1: the top level must be a mapping, not a list"),
            ("weekly 20\n", "line 1: expected key: value"),
            ("budget:\n  weekly 20\n", "line 2: expected key: value"),
            ("{budget: {weekly: 20}}\n", _json_problem("{budget: {weekly: 20}}\n")),
            ('[{"budget": {"weekly": 20}}]\n', "the top level must be a JSON object"),
            ('weekly: "20\n', "line 1: the closing quote is missing"),
            ('weekly: "20" USD\n', "line 1: only a comment may follow the closing quote"),
            ('weekly: "\\x20"\n', backslash),
            ("weekly: &a 20\n", "line 1: a value cannot start with '&' unless it is quoted"),
            ("weekly: *a\n", "line 1: a value cannot start with '*' unless it is quoted"),
            ("weekly: !!float 20\n", "line 1: a value cannot start with '!' unless it is quoted"),
            ("weekly: |\n  20\n", "line 1: a value cannot start with '|' unless it is quoted"),
            ("weekly: - 20\n", "line 1: a value cannot start with '-' unless it is quoted"),
            (
                "weekly: amount: 20\n",
                "line 1: a value cannot contain ': ' or end with ':' unless it is quoted",
            ),
        ]:
            with self.subTest(text=text):
                self._assert_refused(text, problem)

    def test_json_budget_blocks_it_cannot_use_are_refused(self):
        for data, problem in [
            ({"budget": "20"}, "budget must be a JSON object, not '20'"),
            ({"budget": 20}, "budget must be a JSON object, not 20"),
            ({"budget": [{"weekly": 20}]}, "budget must be a JSON object, not a list"),
            (
                {"budget": {"weekly": 20}, "weekly": 20},
                "it has both budget and a top-level weekly; move weekly under budget",
            ),
            ({"budget": {"per_session": 5}}, f"budget has an unknown key 'per_session'{_UNKNOWN}"),
            (
                {"budget": {"x" * 41: 5}},
                f"budget has an unknown key that is too long to show{_UNKNOWN}",
            ),
            ('{"budget": {"weekly": 20, "weekly": 30}}', "budget.weekly is given twice"),
            ('{"budget": {"weekly": 20}, "budget": {}}', "budget is given twice"),
            ('{"weekly": 20, "weekly": 30}', "weekly is given twice"),
        ]:
            with self.subTest(data=data):
                self._assert_refused(data, problem, ".agent-strace.json")

    def test_json_files_it_cannot_parse_are_refused(self):
        """A .json file is never read as YAML. Main did, and lost the budget."""
        for text in [
            '{"budget": {"weekly": 20, "per_session_max": 5,}}',
            "budget:\n  weekly: 20\n",
            # Deep enough for a RecursionError on Python 3.14 too.
            '{"budget": ' + "[" * 1_000_000,
        ]:
            with self.subTest(text=text[:30]):
                self._assert_refused(text, _json_problem(text), ".agent-strace.json")
        for text in ['[{"budget": {"weekly": 20}}]', '"budget"', "20", "null"]:
            with self.subTest(text=text):
                self._assert_refused(
                    text, "the top level must be a JSON object", ".agent-strace.json",
                )
        self._assert_refused(
            '{"budget": {"weekly": 1' + "0" * 5000 + "}}",
            "it has a number with too many digits" if 0 < _INT_DIGITS < 5001
            else _dollars("budget.weekly", "a number too long to show"),
            ".agent-strace.json",
        )

    def test_files_that_are_not_utf8_are_refused(self):
        """Even with no budget keys in sight: UTF-16 would hide them."""
        for data in [
            b"budget:\n  weekly: \xff20\n",
            "budget:\n  weekly: 20\n".encode("utf-16"),
            "review:\n  owner: caf\xe9\n".encode("latin-1"),
        ]:
            with self.subTest(data=data[:12]):
                self._assert_refused(data, "it is not valid UTF-8")

    def test_a_directory_is_refused(self):
        path = Path(".agent-strace.yaml")
        path.mkdir()
        self.addCleanup(path.rmdir)
        with self.assertRaises(OSError) as caught:
            path.read_bytes()
        self._assert_load_refused(
            "invalid project budget in .agent-strace.yaml: "
            f"cannot read it ({caught.exception.strerror})"
        )

    def test_a_dangling_symlink_is_refused(self):
        """Main skipped it, and read the next file or none."""
        self._write({"budget": {"weekly": 20}}, ".agent-strace.json")
        try:
            os.symlink("missing.yaml", ".agent-strace.yaml")
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"cannot create a symlink: {exc}")
        self.addCleanup(os.unlink, ".agent-strace.yaml")
        with self.assertRaises(OSError) as caught:
            Path(".agent-strace.yaml").read_bytes()
        self._assert_load_refused(
            "invalid project budget in .agent-strace.yaml: "
            f"cannot read it ({caught.exception.strerror})"
        )

    @unittest.skipUnless(hasattr(os, "geteuid"), "needs POSIX file permissions")
    def test_a_file_it_may_not_read_is_refused(self):
        path = self._write("budget:\n  weekly: 20\n")
        path.chmod(0)
        self.addCleanup(path.chmod, 0o600)
        try:
            path.read_bytes()
        except PermissionError as exc:
            reason = exc.strerror
        else:
            self.skipTest("this user can read a file without read permission")
        self._assert_load_refused(
            f"invalid project budget in .agent-strace.yaml: cannot read it ({reason})"
        )

    @unittest.skipIf(os.name == "nt", "a file name cannot hold a newline")
    def test_a_path_that_cannot_be_printed_is_quoted(self):
        """The refusal stays on one line."""
        path = self._write("weekly: -5\n", "a\nb.yaml")
        self._assert_load_refused(
            "invalid project budget in 'a\\nb.yaml': " + _dollars("weekly", "-5"), path,
        )

    def test_budgets_main_dropped_are_used(self):
        """A $, a byte order mark or quoted keys made main drop the budget."""
        for name, data in [
            (".agent-strace.yaml", "budget:\n  weekly: $20\n  per_session_max: $5\n"),
            (".agent-strace.yml", "weekly: \"$20\"\nper_session_max: ' $5 '\n"),
            (".agent-strace.json", {"budget": {"weekly": "$20", "per_session_max": " $5"}}),
            (".agent-strace.yaml", "\ufeffbudget:\n  weekly: 20\n  per_session_max: 5\n"),
            (
                ".agent-strace.json",
                "\ufeff" + json.dumps({"budget": {"weekly": 20, "per_session_max": 5}}),
            ),
            (".agent-strace.yaml", "\"budget\":\n  'weekly': 20\n  \"per_session_max\": 5\n"),
        ]:
            with self.subTest(name=name, data=data):
                self.assertEqual(self._load(data, name), (20.0, 0.8, 1.0, 5.0))

    def test_budgets_that_worked_still_work(self):
        """Guard: passes on main too."""
        full = (20.0, 0.5, 1.2, 5.0)
        weekly = (20.0, 0.8, 1.0, None)
        for name, data, expected in [
            (
                ".agent-strace.yaml",
                "budget:\n  weekly: 20\n  warn_at: 0.5\n  stop_at: 1.2\n  per_session_max: 5\n",
                full,
            ),
            (".agent-strace.yml", "weekly: 20\nwarn_at: 0.5\nstop_at: 1.2\nper_session_max: 5\n", full),
            (
                ".agent-strace.json",
                {"budget": {"weekly": 20, "warn_at": 0.5, "stop_at": 1.2, "per_session_max": 5}},
                full,
            ),
            (
                ".agent-strace.json",
                {"weekly": "20", "warn_at": "0.5", "stop_at": "1.2", "per_session_max": "5"},
                full,
            ),
            (".agent-strace.yaml", '{"budget": {"weekly": 20}}', weekly),
            (".agent-strace.yaml", '\n{"budget": {"weekly": 20}}', weekly),
            (".agent-strace.yaml", "budget:\n  weekly: \"20\"\n  stop_at: '1'\n", weekly),
            (".agent-strace.yaml", "budget:  # dollars\n  weekly: 20  # a week\n", weekly),
            (".agent-strace.yaml", "budget:\r\n  weekly: 20\r\n", weekly),
            (".agent-strace.yaml", "budget:\n  weekly:20\n", weekly),
            (".agent-strace.yaml", "budget:\n  weekly: 2e1\n", weekly),
            (".agent-strace.yaml", "budget:\n  weekly: 20\nreview:\n  max_lines: 400\n", weekly),
            (".agent-strace.yaml", "my budget: 5\nweekly: 20\n", weekly),
            (
                ".agent-strace.yaml",
                "retention.days: 30\n  note: weekly cleanup\nweekly: 20\n",
                weekly,
            ),
            (".agent-strace.yaml", "budget:\n  weekly: 20\n  per_session_max:\n", weekly),
            (".agent-strace.yaml", "budget:\n  weekly: 20\n  per_session_max: null\n", weekly),
            (".agent-strace.yaml", "budget:\n  weekly: 20\n  per_session_max: ~\n", weekly),
            (".agent-strace.yaml", "budget:\n  weekly: 20\n  per_session_max: 0\n", weekly),
            (".agent-strace.json", {"budget": {"weekly": 20, "per_session_max": None}}, weekly),
            (".agent-strace.json", {"budget": {"weekly": 20, "per_session_max": 0}}, weekly),
            (".agent-strace.yaml", "budget:\n  weekly: 0\n", _NO_BUDGET),
            (".agent-strace.yaml", "budget:\n  # weekly: 20\n  # per_session_max: 5\n", _NO_BUDGET),
            (".agent-strace.yaml", "budget: {}\n", _NO_BUDGET),
            (".agent-strace.json", {"budget": {}}, _NO_BUDGET),
            (".agent-strace.json", {"budget": None}, _NO_BUDGET),
        ]:
            with self.subTest(name=name, data=data):
                self.assertEqual(self._load(data, name), expected)

    def test_yaml_forms_that_worked_still_work(self):
        """Guard: passes on main too."""
        weekly = (20.0, 0.8, 1.0, None)
        for data in [
            "budget:\n\tweekly: 20\n",
            "---\nbudget:\n  weekly: 20\n...\n",
            "notes: |\n  weekly: 5 is too low\nbudget:\n  weekly: 20\n",
            "steps:\n- weekly: 5\nbudget:\n  weekly: 20\n",
        ]:
            with self.subTest(data=data):
                self.assertEqual(self._load(data), weekly)

    def test_files_without_budget_keys_set_no_budget(self):
        """Guard: passes on main too."""
        iac = (
            "workspaces:\n  - name: production\n\n"
            "identities:\n  - name: billing-agent\n    team: backend\n\n"
            "team_budgets:\n  - team: backend\n    monthly: 500.0\n    alert_threshold: 0.8\n"
        )
        for name, data in [
            (".agent-strace.yaml", iac),
            (".agent-strace.yaml", "review:\n  max_lines: 400\nretention:\n  days: 30\n"),
            (".agent-strace.yml", "# budget:\n#   weekly: 20\n"),
            (".agent-strace.yaml", "weekley: 20\nmonthly_budget: 80\n"),
            (".agent-strace.yaml", "[1, 2,]\n"),
            (".agent-strace.yaml", "{review: {team_budgets: 3}}\n"),
            (".agent-strace.yaml", ""),
            (".agent-strace.json", "  \n"),
            (".agent-strace.json", {"review": {"max_lines": 400}, "retention": {"days": 30}}),
        ]:
            with self.subTest(name=name, data=data):
                self.assertEqual(self._load(data, name), _NO_BUDGET)

    def test_only_the_first_file_is_read(self):
        """Guard: passes on main too."""
        self._write("weekly: 20\n", ".agent-strace.yml")
        self._write('{"weekly": -5,}', ".agent-strace.json")
        self.assertEqual(self._load("weekly: 10\n")[0], 10.0)
        self.assertEqual(load_project_budget_config().weekly, 20.0)


class TestRecordProjectBudgetConfig(unittest.TestCase):
    """record and record-http refuse a budget config they cannot use.

    One line on stderr and status 1, before the store is opened: no session,
    and not even the trace directory's .gitignore. On main they started the
    session without the budget.
    """

    STOP = (
        "agent-strace: weekly project budget exceeded; refusing new session "
        "($30.00 spent, stop threshold $20.00)\n"
    )

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        cwd = os.getcwd()
        os.chdir(tmp.name)
        self.addCleanup(os.chdir, cwd)
        self.trace_dir = Path(tmp.name) / "traces"
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for name in ("AGENT_STRACE_PARENT_SESSION", "AGENT_STRACE_PARENT_EVENT", "AGENT_TRACE_DIR"):
            os.environ.pop(name, None)
        self.proxies = []
        for target in ("agent_trace.cli.MCPProxy", "agent_trace.cli.HTTPProxyServer"):
            patcher = patch(target)
            proxy = patcher.start()
            self.addCleanup(patcher.stop)
            proxy.return_value.run.return_value = 0
            self.proxies.append(proxy)

    def _run(self, command: str, name: str, data: str) -> tuple[int, str, str]:
        path = Path(name)
        path.write_bytes(data.encode("utf-8"))
        argv = ["--trace-dir", str(self.trace_dir), command]
        if command == "record":
            argv += ["--", "true"]
        else:
            argv += ["--url", "http://127.0.0.1:9/mcp"]
        args = build_parser().parse_args(argv)
        handler = cmd_record if command == "record" else cmd_record_http
        out, err = io.StringIO(), io.StringIO()
        try:
            with redirect_stdout(out), redirect_stderr(err):
                status = handler(args)
        finally:
            path.unlink()
        return status, out.getvalue(), err.getvalue()

    def _spend_thirty_dollars(self) -> None:
        store = TraceStore(self.trace_dir)
        meta = SessionMeta(agent_name="agent", command="test")
        store.create_session(meta)
        for _ in range(10):
            store.append_event(
                meta.session_id,
                TraceEvent(
                    event_type=EventType.LLM_RESPONSE,
                    timestamp=time.time(),
                    session_id=meta.session_id,
                    data={"input_tokens": 1_000_000, "output_tokens": 0},
                ),
            )

    def _sessions(self) -> int:
        return len(TraceStore(self.trace_dir).list_sessions())

    def _assert_no_proxy(self) -> None:
        for proxy in self.proxies:
            proxy.assert_not_called()

    def test_a_budget_config_it_cannot_use_is_refused(self):
        trailing_comma = '{"budget": {"weekly": 20,}}'
        for name, data, problem in [
            (
                ".agent-strace.yaml",
                "budget:\n  weekly: 20\n  weekley: 20\n",
                f"line 3: budget has an unknown key 'weekley'{_UNKNOWN}",
            ),
            (".agent-strace.yml", "budget:\n  stop_at: 90\n", _fraction("budget.stop_at", "90")),
            (".agent-strace.json", trailing_comma, _json_problem(trailing_comma)),
        ]:
            for command in ("record", "record-http"):
                with self.subTest(name=name, command=command):
                    # Opening the store would write a .gitignore here.
                    shutil.rmtree(self.trace_dir, ignore_errors=True)
                    self.trace_dir.mkdir()
                    self.assertEqual(
                        self._run(command, name, data),
                        (1, "", f"agent-strace: invalid project budget in {name}: {problem}\n"),
                    )
                    self.assertEqual(list(self.trace_dir.iterdir()), [])
                    self._assert_no_proxy()

    def test_a_weekly_budget_main_dropped_stops_a_new_session(self):
        self._spend_thirty_dollars()
        for name, data in [
            (".agent-strace.yaml", "budget:\n  weekly: $20\n"),
            (".agent-strace.yaml", "\ufeffbudget:\n  weekly: 20\n"),
            (".agent-strace.json", '{"budget": {"weekly": "$20"}}'),
        ]:
            for command in ("record", "record-http"):
                with self.subTest(name=name, data=data, command=command):
                    self.assertEqual(self._run(command, name, data), (1, "", self.STOP))
                    self.assertEqual(self._sessions(), 1)
                    self._assert_no_proxy()

    def test_a_weekly_budget_stops_a_new_session(self):
        """Guard: passes on main too."""
        self._spend_thirty_dollars()
        for command in ("record", "record-http"):
            with self.subTest(command=command):
                self.assertEqual(
                    self._run(command, ".agent-strace.yaml", "budget:\n  weekly: 20\n"),
                    (1, "", self.STOP),
                )
                self.assertEqual(self._sessions(), 1)
                self._assert_no_proxy()

    def test_a_file_without_budget_keys_starts_a_session(self):
        """Guard: passes on main too."""
        for command in ("record", "record-http"):
            with self.subTest(command=command):
                status, _out, _err = self._run(
                    command, ".agent-strace.yaml", "review:\n  max_lines: 400\n",
                )
                self.assertEqual(status, 0)
        self.assertEqual(self._sessions(), 2)


if __name__ == "__main__":
    unittest.main()
