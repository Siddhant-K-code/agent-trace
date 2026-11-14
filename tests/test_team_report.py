"""Tests for team cost attribution reports."""

from __future__ import annotations

import argparse
import csv
import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_trace.cli import build_parser
from agent_trace.cost import estimate_cost
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore
from agent_trace.team_report import (
    build_team_report,
    cmd_team_report,
    format_team_report_csv,
    format_team_report_json,
    format_team_report_text,
)


try:
    from test_cost import _codex_events
except ImportError:  # pragma: no cover - package-style test discovery
    from tests.test_cost import _codex_events


def _make_store() -> tuple[TraceStore, str]:
    tmp = tempfile.mkdtemp()
    return TraceStore(Path(tmp)), tmp


def _add_session(
    store: TraceStore,
    started_at: float,
    branch: str = "main",
    files: list[str] | None = None,
    prompt_size: int = 2000,
    fallback_user: str = "local-user",
) -> str:
    meta = SessionMeta(agent_name="agent", command="agent")
    meta.started_at = started_at
    meta.attribution = {
        "git_branch": branch,
        "working_dir": str(store.base_dir),
        "os_user": fallback_user,
    }
    store.create_session(meta)
    store.update_meta(meta)
    sid = meta.session_id

    events = [
        TraceEvent(
            event_type=EventType.SESSION_START,
            timestamp=started_at,
            session_id=sid,
            data={},
        ),
        TraceEvent(
            event_type=EventType.USER_PROMPT,
            timestamp=started_at + 1,
            session_id=sid,
            data={"prompt": "x" * prompt_size},
        ),
    ]
    for i, path in enumerate(files or []):
        events.append(TraceEvent(
            event_type=EventType.TOOL_CALL,
            timestamp=started_at + 2 + i,
            session_id=sid,
            data={"tool_name": "Write", "arguments": {"file_path": path}},
        ))
    events.append(TraceEvent(
        event_type=EventType.ASSISTANT_RESPONSE,
        timestamp=started_at + 20,
        session_id=sid,
        data={"text": "y" * prompt_size},
    ))
    events.append(TraceEvent(
        event_type=EventType.SESSION_END,
        timestamp=started_at + 30,
        session_id=sid,
        data={},
    ))
    for ev in events:
        store.append_event(sid, ev)
    return sid


def _fake_git(author_by_file: dict[str, str], fallback_email: str = "fallback@example.com"):
    def fake(cwd, *args):
        if args == ("config", "user.email"):
            return fallback_email
        if len(args) >= 5 and args[:3] == ("log", "-1", "--format=%ae"):
            return author_by_file.get(args[-1], "")
        return ""
    return fake


class TestBuildTeamReport(unittest.TestCase):
    def test_groups_cost_by_git_author_with_file_shares(self):
        store, _ = _make_store()
        now = time.time()
        _add_session(store, now - 3600, files=["a.py", "b.py", "c.py"])

        with patch("agent_trace.team_report._git", _fake_git({
            "a.py": "alice@example.com",
            "b.py": "bob@example.com",
            "c.py": "alice@example.com",
        })):
            report = build_team_report(store, now - 86400, now)

        self.assertIn("alice@example.com", report.rows)
        self.assertIn("bob@example.com", report.rows)
        self.assertAlmostEqual(report.rows["alice@example.com"]["sessions"], 2 / 3)
        self.assertAlmostEqual(report.rows["bob@example.com"]["sessions"], 1 / 3)
        self.assertGreater(report.rows["alice@example.com"]["cost"], report.rows["bob@example.com"]["cost"])

    def test_groups_by_branch(self):
        store, _ = _make_store()
        now = time.time()
        _add_session(store, now - 3600, branch="feat/payments", files=["a.py"])

        with patch("agent_trace.team_report._git", _fake_git({"a.py": "alice@example.com"})):
            report = build_team_report(store, now - 86400, now, group_by="branch")

        self.assertIn("feat/payments", report.rows)

    def test_groups_by_pr_from_branch_name(self):
        store, _ = _make_store()
        now = time.time()
        _add_session(store, now - 3600, branch="feature/pr-412-payments", files=["a.py"])

        with patch("agent_trace.team_report._git", _fake_git({"a.py": "alice@example.com"})):
            report = build_team_report(store, now - 86400, now, group_by="pr")

        self.assertIn("#412", report.rows)

    def test_date_filter_excludes_old_sessions(self):
        store, _ = _make_store()
        now = time.time()
        _add_session(store, now - 3600, files=["new.py"])
        _add_session(store, now - 10 * 86400, files=["old.py"])

        with patch("agent_trace.team_report._git", _fake_git({
            "new.py": "new@example.com",
            "old.py": "old@example.com",
        })):
            report = build_team_report(store, now - 86400, now)

        self.assertIn("new@example.com", report.rows)
        self.assertNotIn("old@example.com", report.rows)

    def test_falls_back_when_git_has_no_author(self):
        store, _ = _make_store()
        now = time.time()
        _add_session(store, now - 3600, files=["a.py"], fallback_user="local-user")

        with patch("agent_trace.team_report._git", _fake_git({}, fallback_email="")):
            report = build_team_report(store, now - 86400, now)

        self.assertIn("local-user", report.rows)

    def test_outliers_use_threshold(self):
        store, _ = _make_store()
        now = time.time()
        _add_session(store, now - 3600, files=["small.py"], prompt_size=200)
        _add_session(store, now - 1800, files=["large.py"], prompt_size=20000)

        with patch("agent_trace.team_report._git", _fake_git({
            "small.py": "alice@example.com",
            "large.py": "alice@example.com",
        })):
            report = build_team_report(store, now - 86400, now, outlier_threshold=1.2)

        self.assertEqual(len(report.outliers), 1)


class TestFormatTeamReport(unittest.TestCase):
    def _report(self):
        store, _ = _make_store()
        now = time.time()
        _add_session(store, now - 3600, branch="feat/pr-99", files=["a.py"], prompt_size=20000)
        _add_session(store, now - 1800, branch="main", files=["b.py"], prompt_size=200)
        with patch("agent_trace.team_report._git", _fake_git({
            "a.py": "alice@example.com",
            "b.py": "bob@example.com",
        })):
            return build_team_report(store, now - 86400, now, outlier_threshold=1.2)

    def test_text_contains_table_and_outlier_hint(self):
        report = self._report()
        out = io.StringIO()
        format_team_report_text(report, out)
        text = out.getvalue()
        self.assertIn("Team Agent Cost Report", text)
        self.assertIn("alice@example.com", text)
        self.assertIn("agent-strace lint", text)

    def test_csv_output_has_rows(self):
        report = self._report()
        rows = list(csv.DictReader(io.StringIO(format_team_report_csv(report))))
        self.assertEqual(rows[0]["group_by"], "author")
        self.assertIn(rows[0]["group"], {"alice@example.com", "bob@example.com"})

    def test_json_output_is_valid(self):
        report = self._report()
        data = json.loads(format_team_report_json(report))
        self.assertEqual(data["group_by"], "author")
        self.assertIn("rows", data)


class TestTeamReportCLI(unittest.TestCase):
    def _args(self, trace_dir, by="author", export="text", since=None, until=None):
        return argparse.Namespace(
            trace_dir=str(trace_dir),
            by=by,
            export=export,
            since=since,
            until=until,
            outlier_threshold=2.0,
        )

    def test_cmd_team_report_csv(self):
        store, tmp = _make_store()
        now = time.time()
        _add_session(store, now - 3600, files=["a.py"])
        args = self._args(tmp, export="csv")

        captured = io.StringIO()
        with patch("agent_trace.team_report._git", _fake_git({"a.py": "alice@example.com"})):
            with patch("sys.stdout", captured):
                result = cmd_team_report(args)

        self.assertEqual(result, 0)
        self.assertIn("group_by,group,sessions", captured.getvalue())

    def test_parser_registers_team_report(self):
        parser = build_parser()
        args = parser.parse_args(["team-report", "--by", "branch", "--export", "csv"])
        self.assertEqual(args.command, "team-report")
        self.assertEqual(args.by, "branch")
        self.assertEqual(args.export, "csv")


def _add_codex_session(
    store: TraceStore, started_at: float, branch: str, prompt_data=None
) -> str:
    """A session with recorded token usage; list prompt data makes `cost` refuse it."""
    meta = SessionMeta(agent_name="codex", command="codex")
    meta.started_at = started_at
    meta.attribution = {"git_branch": branch, "working_dir": str(store.base_dir)}
    store.create_session(meta)
    events = _codex_events(meta.session_id)
    if prompt_data is not None:
        events[0].data = prompt_data
    for event in events:
        store.append_event(meta.session_id, event)
    return meta.session_id


class TestTeamReportUnpricedSessions(unittest.TestCase):
    """Costs are the `agent-strace cost` totals; refused sessions are not $0."""

    def setUp(self):
        self.store, tmp = _make_store()
        self.now = time.time()

    def _report(self):
        with patch("agent_trace.team_report._git", _fake_git({})):
            return build_team_report(
                self.store, self.now - 86400, self.now, group_by="branch"
            )

    def test_unpriceable_sessions_are_left_out_of_totals(self):
        priced = _add_codex_session(self.store, self.now - 300, "main")
        _add_codex_session(self.store, self.now - 200, "feat/x", prompt_data=[])
        broken = _add_codex_session(self.store, self.now - 100, "feat/x")
        with open(self.store.base_dir / broken / "events.ndjson", "a") as f:
            f.write("not json\n")
        want = estimate_cost(self.store, priced).total_cost

        report = self._report()
        costs = {s.session_id: s.cost for s in report.sessions}
        self.assertEqual(costs.pop(priced), want)
        self.assertEqual(list(costs.values()), [None, None])
        self.assertEqual(report.unpriced_sessions, 2)
        self.assertEqual(report.total_cost, want)
        self.assertEqual(report.avg_cost, want)
        self.assertEqual(report.rows["main"]["cost"], want)
        self.assertEqual(report.rows["main"]["avg_session"], want)
        self.assertIsNone(report.rows["feat/x"]["cost"])
        self.assertIsNone(report.rows["feat/x"]["avg_session"])
        self.assertEqual(report.rows["feat/x"]["unpriced_sessions"], 2.0)
        self.assertEqual(list(report.rows), ["main", "feat/x"])

        out = io.StringIO()
        format_team_report_text(report, out)
        lines = out.getvalue().splitlines()
        row = next(line for line in lines if line.startswith("feat/x"))
        self.assertEqual(row.split()[-2:], ["n/a", "n/a"])
        total = next(line for line in lines if line.startswith("Total"))
        self.assertTrue(total.endswith(f"${want:.2f} (2 not priced)"), total)

        csv_rows = {
            r["group"]: r
            for r in csv.DictReader(io.StringIO(format_team_report_csv(report)))
        }
        csv_row = csv_rows["feat/x"]
        self.assertEqual((csv_row["cost"], csv_row["avg_session"]), ("", ""))
        self.assertEqual(csv_row["unpriced_sessions"], "2.000")

        data = json.loads(format_team_report_json(report))
        self.assertEqual(data["unpriced_sessions"], 2)
        self.assertEqual(data["total_cost"], want)
        by_group = {r["group"]: r for r in data["rows"]}
        self.assertIsNone(by_group["feat/x"]["cost"])
        self.assertIsNone(by_group["feat/x"]["avg_session"])

    def test_no_priced_session_has_unknown_total_and_average(self):
        _add_codex_session(self.store, self.now - 100, "main", prompt_data=[])
        report = self._report()
        self.assertIsNone(report.total_cost)
        self.assertIsNone(report.avg_cost)
        self.assertEqual(report.outliers, [])
        out = io.StringIO()
        format_team_report_text(report, out)
        total = next(line for line in out.getvalue().splitlines() if line.startswith("Total"))
        self.assertEqual(total.split()[-5:], ["n/a", "n/a", "(1", "not", "priced)"])
        self.assertNotIn("$", total)
        self.assertIsNone(json.loads(format_team_report_json(report))["total_cost"])

    def test_pricing_bugs_are_not_hidden(self):
        _add_codex_session(self.store, self.now - 100, "main")
        with patch("agent_trace.team_report.session_cost", side_effect=TypeError("bug")):
            with self.assertRaises(TypeError):
                self._report()


if __name__ == "__main__":
    unittest.main()
