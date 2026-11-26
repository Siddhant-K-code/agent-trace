"""compliance export, dashboard --trend, drift and eval dataset auto read --since.

Each reads it as N or Nd days, as export --scores does. Any other value exits 1
with one line on stderr, before the command reads the store or a baseline, or
writes a file.
"""

import json
import os
import re
import subprocess
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:  # keeps product telemetry off in this process and its children
    import _no_telemetry  # noqa: F401
    from test_hash_chain import _CliCase
except ImportError:  # pragma: no cover - run as a package module
    from tests import _no_telemetry  # noqa: F401
    from tests.test_hash_chain import _CliCase

from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore

ONE = "a1b2c3d4e5f60101"
TEN = "a1b2c3d4e5f60110"
FORTY = "a1b2c3d4e5f60140"
AGES = {ONE: 1, TEN: 10, FORTY: 40}

REFUSAL = "invalid --since value: use a number of days as N or Nd, such as 7d\n"
REFUSED = ["", "abc", "7d ", "7dd", "1e2d", "\u0661\u0660", "nan", "-5d", "1" + "0" * 400]

# The sessions each command selects for a value it reads.
READ = {
    "7d": {"compliance": {ONE}, "trend": 1, "drift": (1, 0), "dataset": (1, {ONE})},
    "7": {"compliance": {ONE}, "trend": 1, "drift": (1, 0), "dataset": (1, {ONE})},
    "0.5d": {"compliance": set(), "trend": 0, "drift": (0, 0), "dataset": (0, set())},
    "30d": {"compliance": {ONE, TEN}, "trend": 2, "drift": (1, 1), "dataset": (2, {ONE, TEN})},
    "0d": {"compliance": set(), "trend": 0, "drift": (0, 0), "dataset": (0, set())},
}

# Each command's output file. drift and dataset make its parent directory.
OUTPUT = {
    "compliance": ("-o", "compliance.json"),
    "trend": ("--html", "trend.html"),
    "drift": ("--save-baseline", "sub/fingerprint.json"),
    "dataset": ("--dataset", "sub/d.jsonl"),
}

SPECIFY = "Specify --since Nd or --current YYYY-MM-DD:YYYY-MM-DD\n"
NO_BASELINE = "No baseline available. Use --baseline <file> or --since Nd.\n"


def seed(store: TraceStore) -> None:
    now = time.time()
    for session_id, age in AGES.items():
        start = now - age * 86400
        meta = SessionMeta(session_id=session_id, agent_name="a", command="c", started_at=start)
        meta.ended_at = start + 60
        meta.total_duration_ms = 60000
        store.create_session(meta)
        for event_type, offset, data in (
            (EventType.SESSION_START, 0, {}),
            (EventType.TOOL_CALL, 1, {"tool_name": "Bash", "arguments": {"command": "ls"}}),
            (EventType.SESSION_END, 60, {}),
        ):
            store.append_event(session_id, TraceEvent(
                event_type=event_type, timestamp=start + offset,
                session_id=session_id, data=data,
            ))
        store.update_meta(meta)


def command_argv(command: str, value: str, out: Path | None = None) -> list[str]:
    """Return the argv for one command, with its output file when out is set."""
    since = f"--since={value}"
    if command == "compliance":
        argv = ["compliance", "export", since]
    elif command == "trend":
        argv = ["dashboard", "--trend", since]
    elif command == "drift":
        argv = ["drift", since, "--format", "json"]
    else:
        argv = ["eval", "dataset", "auto", since, "--filter", "long-duration:30s"]
    if out is not None:
        argv += [OUTPUT[command][0], str(out)]
    return argv


def compliance_ids(run: subprocess.CompletedProcess) -> set[str]:
    return set(re.findall(r'"session_id": "([0-9a-f]+)"', run.stdout))


def trend_count(run: subprocess.CompletedProcess) -> int | None:
    # The header has a dash between the words, so \S+ stands for it.
    match = re.search(r"Trend report \S+ (\d+) session", run.stdout)
    if match:
        return int(match.group(1))
    return 0 if "No sessions found." in run.stdout else None


def drift_counts(run: subprocess.CompletedProcess) -> tuple | None:
    try:
        data = json.loads(run.stdout)
    except ValueError:
        return None
    return data["baseline_sessions"], data["current_sessions"]


def dataset_added(run: subprocess.CompletedProcess, path: Path) -> tuple:
    match = re.match(r"Added (\d+) session", run.stdout)
    ids = set()
    if path.exists():
        ids = {
            json.loads(line)["session_id"]
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    return (int(match.group(1)) if match else None), ids


def label(value: str) -> str:
    return value if len(value) < 20 else f"1 and {len(value) - 1} zeros"


class _SinceCase(_CliCase):
    def setUp(self):
        super().setUp()
        seed(self.store)

    def assertRefused(self, run: subprocess.CompletedProcess) -> None:
        self.assertEqual((run.returncode, run.stdout, run.stderr), (1, "", REFUSAL))


class TestRefusedValues(_SinceCase):
    """Each refused value exits 1 with one line and writes no file."""

    def check_refused(self, command: str) -> None:
        cases = []
        for n, value in enumerate(REFUSED):
            case_dir = self.root / "out" / command / str(n)
            case_dir.mkdir(parents=True)
            cases.append((value, case_dir, case_dir / OUTPUT[command][1]))
        runs = self.cli_many([command_argv(command, value, out) for value, _, out in cases])
        for (value, case_dir, _), run in zip(cases, runs):
            with self.subTest(value=label(value)):
                self.assertRefused(run)
                self.assertEqual(list(case_dir.iterdir()), [])

    def test_compliance_export(self):
        self.check_refused("compliance")

    def test_dashboard_trend(self):
        self.check_refused("trend")

    def test_drift(self):
        self.check_refused("drift")

    def test_eval_dataset_auto(self):
        self.check_refused("dataset")


class TestReadValues(_SinceCase):
    """N and Nd are days, and 0d is a window of no days."""

    def test_compliance_export(self):
        runs = self.cli_many([command_argv("compliance", value) for value in READ])
        for value, run in zip(READ, runs):
            with self.subTest(value=value):
                self.assertEqual((run.returncode, run.stderr), (0, ""))
                self.assertEqual(compliance_ids(run), READ[value]["compliance"])

    def test_dashboard_trend(self):
        runs = self.cli_many([command_argv("trend", value) for value in READ])
        for value, run in zip(READ, runs):
            with self.subTest(value=value):
                self.assertEqual((run.returncode, run.stderr), (0, ""))
                self.assertEqual(trend_count(run), READ[value]["trend"])

    def test_drift(self):
        runs = self.cli_many([command_argv("drift", value) for value in READ])
        for value, run in zip(READ, runs):
            with self.subTest(value=value):
                self.assertEqual(run.stderr, "")
                self.assertEqual(drift_counts(run), READ[value]["drift"])

    def test_eval_dataset_auto(self):
        paths = {value: self.root / "out" / str(n) / "d.jsonl" for n, value in enumerate(READ)}
        runs = self.cli_many([command_argv("dataset", value, paths[value]) for value in READ])
        for value, run in zip(READ, runs):
            with self.subTest(value=value):
                self.assertEqual((run.returncode, run.stderr), (0, ""))
                self.assertEqual(dataset_added(run, paths[value]), READ[value]["dataset"])


class TestCheckOrder(_SinceCase):
    """The check comes before the store or a baseline is read."""

    def test_compliance_export_refuses_before_it_reads_the_store(self):
        bad = self.root / "bad"
        seed(TraceStore(str(bad), use_workspace_env=False))
        (bad / TEN / "meta.json").write_text("{", encoding="utf-8")
        self.assertRefused(self.cli("compliance", "export", "--since=abc", trace_dir=bad))
        run = self.cli("compliance", "export", "--since=7d", trace_dir=bad)
        self.assertEqual((run.returncode, run.stdout), (1, ""))
        self.assertTrue(
            run.stderr.startswith(f"Error: session {TEN} has a corrupt meta.json:"), run.stderr,
        )

    def test_drift_refuses_before_it_reads_the_baseline(self):
        baseline = self.root / "baseline.json"
        baseline.write_text("{", encoding="utf-8")
        self.assertRefused(self.cli("drift", "--since=abc", "--baseline", str(baseline)))


class TestComplianceExportSessionId(_SinceCase):
    """With a session ID, --since still takes the window of sessions."""

    def test_since_with_a_session_id(self):
        empty, week, plain = self.cli_many([
            ("compliance", "export", FORTY, "--since="),
            ("compliance", "export", FORTY, "--since=7d"),
            ("compliance", "export", FORTY),
        ])
        with self.subTest(since=""):
            self.assertRefused(empty)
        with self.subTest(since="7d"):
            self.assertEqual((week.returncode, week.stderr), (0, ""))
            self.assertEqual(compliance_ids(week), {ONE})
        with self.subTest(since=None):
            self.assertEqual((plain.returncode, plain.stderr), (0, ""))
            self.assertEqual(compliance_ids(plain), {FORTY})


class TestUnchangedPaths(_SinceCase):
    """No --since, drift --current and plain dashboard work as before."""

    def test_compliance_export_takes_30_days(self):
        run = self.cli("compliance", "export")
        self.assertEqual((run.returncode, run.stderr), (0, ""))
        self.assertEqual(compliance_ids(run), {ONE, TEN})

    def test_dashboard_trend_takes_every_session(self):
        run = self.cli("dashboard", "--trend")
        self.assertEqual((run.returncode, run.stderr), (0, ""))
        self.assertEqual(trend_count(run), 3)

    def test_drift_asks_for_since_or_current(self):
        run = self.cli("drift", "--format", "json")
        self.assertEqual((run.returncode, run.stdout, run.stderr), (1, "", SPECIFY))

    def test_eval_dataset_auto_takes_7_days(self):
        path = self.root / "out" / "d.jsonl"
        run = self.cli("eval", "dataset", "auto", "--filter", "long-duration:30s",
                       "--dataset", str(path))
        self.assertEqual((run.returncode, run.stderr), (0, ""))
        self.assertEqual(dataset_added(run, path), (1, {ONE}))

    def test_drift_with_current_does_not_read_since(self):
        today = time.strftime("%Y-%m-%d", time.gmtime())
        run = self.cli("drift", f"--current={today}:{today}", "--since=abc", "--format", "json")
        self.assertEqual((run.returncode, run.stdout, run.stderr), (1, "", NO_BASELINE))

    def test_dashboard_without_trend_does_not_read_since(self):
        run = self.cli("dashboard", "--since=abc")
        self.assertEqual((run.returncode, run.stderr), (0, ""))
        self.assertNotEqual(run.stdout, "")


if __name__ == "__main__":
    unittest.main()
