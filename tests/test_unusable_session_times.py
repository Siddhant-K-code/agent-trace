"""A session time that is not a usable time reads as missing.

A session's meta.json can hold any JSON value in started_at and ended_at.
usable_time() refuses text, a bool, null, a list, an object, NaN, infinity, a
negative number and a time after 9999-12-30T00:00:00Z. These tests store each
such value and check that a command leaves the session out of its time
window, shows the start as missing, reads the end as no end, or refuses the
session.
"""

from __future__ import annotations

import contextlib
import hashlib
import http.client
import json
import os
import re
import shutil
import sys
import threading
import unittest
from datetime import datetime, timezone
from http.server import HTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import _no_telemetry  # noqa: F401
    from test_non_object_event_data import (
        BAD,
        CHILD,
        GOOD,
        NOW,
        CliTestCase,
        _chdir,
        _isolated_env,
        _session_events,
        _write_session,
    )
except ImportError:
    from tests import _no_telemetry  # noqa: F401
    from tests.test_non_object_event_data import (
        BAD,
        CHILD,
        GOOD,
        NOW,
        CliTestCase,
        _chdir,
        _isolated_env,
        _session_events,
        _write_session,
    )

from agent_trace import mcp_server, oncall, postmortem, pr_comment
from agent_trace.policy_backtest import _select_sessions
from agent_trace.project_budget import rolling_project_spend
from agent_trace.server import _make_handler
from agent_trace.store import TraceStore

# Each start that usable_time() refuses. json.dumps writes NaN and Infinity
# as JSON text, and 10**400 as an integer too large for a float.
UNUSABLE = [
    ("text", "abc"),
    ("numeric text", "1700000000"),
    ("true", True),
    ("null", None),
    ("NaN", float("nan")),
    ("Infinity", float("inf")),
    ("negative", -5),
    ("1e20", 1e20),
    ("list", []),
    ("object", {}),
    ("milliseconds", 1700000000000),
    ("too large for a float", 10**400),
]

# A usable start that lies in no time window.
OUTSIDE = 1.0

# BAD's start in the fixture. Its events run from here to 26 seconds later.
BAD_START = NOW - 3600

STARTED = "Error: session started_at must be a Unix time in seconds\n"
ENDED = "Error: session ended_at must be a Unix time in seconds\n"

SUBTITLE = re.compile(r'<div class="subtitle">.*?</div>')
ANNOTATION = re.compile(r'<line [^>]*stroke="#e3b341"[^>]*/><text [^>]*>[^<]*</text>')

# The random IDs and clock times that eval dataset and baseline write.
VOLATILE = re.compile(r'"(entry_id|added_at|created_at)": ?("[^"]*"|[0-9.eE+-]+)')

BASELINE_UPDATE = ["baseline", "update", "--since", "30", "--output", "b.json"]


def traces(root: Path) -> Path:
    return root / "work" / ".agent-traces"


def set_meta(session_dir: Path, **fields) -> None:
    path = session_dir / "meta.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta.update(fields)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def rewrite_events(session_dir: Path, rewrite) -> None:
    """Replace the event records with *rewrite* of them and rebuild the hash chain."""
    path = session_dir / "events.ndjson"
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        record.pop("prev_hash", None)
        records.append(record)
    lines: list[str] = []
    for record in rewrite(records):
        if lines:
            record["prev_hash"] = hashlib.sha256(lines[-1].encode()).hexdigest()
        lines.append(json.dumps(record, separators=(",", ":")))
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def edit_events(session_dir: Path, edit) -> None:
    """Apply *edit* to each event record and rebuild the hash chain."""
    def apply(records: list[dict]) -> list[dict]:
        for record in records:
            edit(record)
        return records

    rewrite_events(session_dir, apply)


def drop_session_end(session_dir: Path) -> None:
    rewrite_events(
        session_dir, lambda records: [r for r in records if r["event_type"] != "session_end"],
    )


def add_skill_load(session_dir: Path) -> None:
    """Record a load of the deploy-check skill 1.5 seconds after the first event."""
    load = {
        "event_type": "tool_call",
        "timestamp": first_event_time(session_dir) + 1.5,
        "event_id": "skill-load",
        "session_id": session_dir.name,
        "data": {
            "tool_name": "skill",
            "arguments": {"skill": "deploy-check"},
            "tool_use_id": "tu-skill",
        },
    }
    rewrite_events(session_dir, lambda records: records[:2] + [load] + records[2:])


def scrub(outcome: tuple) -> tuple:
    """*outcome* without the random IDs and clock times in its files."""
    files = {name: VOLATILE.sub(r'"\1": "?"', text) for name, text in outcome[4].items()}
    return outcome[:4] + (files,)


def write_policies(root: Path) -> None:
    """deny.json blocks the fixture's pytest runs and allow.json allows them."""
    for name, rule in (("deny.json", "deny"), ("allow.json", "allow")):
        (root / "work" / name).write_text(
            json.dumps({"commands": {rule: ["pytest *"]}}), encoding="utf-8",
        )


def first_event_time(session_dir: Path) -> float:
    with open(session_dir / "events.ndjson", encoding="utf-8") as f:
        return json.loads(f.readline())["timestamp"]


def utc_day(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def local_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts).isoformat()


def write_config_snapshots(root: Path) -> None:
    """Record a change to AGENTS.md three days ago, before every fixture start."""
    workspace = os.path.realpath(root / "work")
    entries = [
        {
            "snapshot_id": f"snap{index}",
            "timestamp": timestamp,
            "session_id": "",
            "label": "",
            "workspace": workspace,
            "files": [{"path": "AGENTS.md", "sha256": digest, "mtime": 0.0, "exists": True}],
        }
        for index, (timestamp, digest) in enumerate(
            [(NOW - 5 * 86400, "a" * 64), (NOW - 3 * 86400, "b" * 64)]
        )
    ]
    (traces(root) / ".config-snapshots.json").write_text(
        json.dumps(entries, indent=2), encoding="utf-8",
    )


def stat_snapshot(root: Path) -> dict[str, tuple[int, int, int]]:
    """Map every path under *root* to its mtime, size and mode, without reading it."""
    snapshot = {}
    for path in root.rglob("*"):
        st = path.stat()
        snapshot[str(path.relative_to(root))] = (st.st_mtime_ns, st.st_size, st.st_mode)
    return snapshot


class SessionTimeCase(CliTestCase):
    def ws(self, meta: dict[str, dict] | None = None, *, tenant: bool = False) -> Path:
        """The fixture, with the meta.json fields in *meta* set per session ID."""
        root = self.make_workspace(tenant=tenant)
        for session_id, fields in (meta or {}).items():
            set_meta(traces(root) / session_id, **fields)
        return root

    @contextlib.contextmanager
    def direct(self, root: Path):
        """A store for a library call, with the cwd and environment run_cli uses."""
        with _chdir(root / "work"), _isolated_env(root / "home"):
            yield TraceStore(str(traces(root)), read_only=True)

    def outcome(self, root: Path, argv: list[str], files: str | None = None) -> tuple:
        """Exit code, stdout, stderr, network and browser calls, and the files *files* matches."""
        result = self.run_cli(root, *argv)
        written = {}
        if files:
            written = {
                path.name: path.read_text(encoding="utf-8")
                for path in sorted((root / "work").glob(files))
            }
        return (
            result.code,
            result.out.replace(str(root), "ROOT"),
            result.err.replace(str(root), "ROOT"),
            len(result.network) + len(result.browser),
            written,
        )


class TimeWindowTest(SessionTimeCase):
    """A session whose start is not usable is in no time window."""

    def check_window(
        self,
        argv: list[str],
        *,
        snapshots: bool = False,
        files: str | None = None,
        tenant: bool = False,
        prepare=None,
    ) -> None:
        def run(fields: dict | None = None) -> tuple:
            root = self.ws({BAD: fields} if fields is not None else None, tenant=tenant)
            if snapshots:
                write_config_snapshots(root)
            if prepare:
                prepare(root)
            return scrub(self.outcome(root, argv, files))

        outside = run({"started_at": OUTSIDE})
        self.assertNotEqual(run(), outside, "the fixture start is in the window")
        for name, value in UNUSABLE:
            with self.subTest(value=name):
                self.assertEqual(run({"started_at": value}), outside)

    def test_dashboard_trend_since(self):
        self.check_window(["dashboard", "--trend", "--since", "30d"])

    def test_budget_report(self):
        self.check_window([
            "budget-report", "--since", local_iso(NOW - 86400),
            "--until", local_iso(NOW + 86400), "--format", "json",
        ])

    def test_team_report(self):
        self.check_window([
            "team-report", "--since", local_iso(NOW - 86400),
            "--until", local_iso(NOW + 86400), "--export", "json",
        ])

    def test_context_score(self):
        self.check_window(["context-score", "--file", "../aux/AGENTS.md", "--min-sessions", "1"])

    def test_lint_since(self):
        self.check_window(["lint", "--all", "--since", "30d"])

    def test_mcp_scan_since(self):
        self.check_window(["mcp-scan", "--since", "30d"])

    def test_oncall(self):
        # oncall prints each file's age in whole days from the clock. At NOW,
        # GOOD's write is 11.5 seconds short of two days old. So a real clock
        # can print 1d in one run and 2d in the next.
        with mock.patch.object(oncall, "time", SimpleNamespace(time=lambda: NOW)):
            self.check_window(["oncall", "--rotation-start", "2000-01-01"])

    def test_drift_since(self):
        self.check_window(["drift", "--since", "30d"])

    def test_drift_current(self):
        self.check_window([
            "drift", "--current", f"{utc_day(NOW - 86400)}:{utc_day(NOW)}",
            "--baseline-range", f"{utc_day(NOW - 5 * 86400)}:{utc_day(NOW - 2 * 86400)}",
        ])

    def test_config_watch_affected(self):
        self.check_window(["config-watch", "affected"], snapshots=True)

    def test_config_watch_affected_since(self):
        self.check_window(
            ["config-watch", "affected", "--since", "30d", "--format", "json"], snapshots=True,
        )

    def test_cost_by_provider(self):
        self.check_window(["cost", "--breakdown", "provider"])

    def test_cost_by_provider_csv(self):
        self.check_window(["cost", "--breakdown", "provider", "--csv"])

    def test_cost_tenant(self):
        self.check_window(["cost", "--tenant", "acme"], tenant=True)

    def test_policy_backtest(self):
        self.check_window(
            ["policy", "backtest", "--policy", "deny.json", "--days", "30"], prepare=write_policies,
        )

    def test_policy_diff(self):
        self.check_window(
            ["policy", "diff", "deny.json", "allow.json", "--days", "30"], prepare=write_policies,
        )

    def test_policy_coverage(self):
        self.check_window(
            ["policy", "coverage", "--policy", "deny.json", "--days", "30"], prepare=write_policies,
        )

    def test_eval_dataset_auto(self):
        self.check_window(
            [
                "eval", "dataset", "auto", "--filter", "cost-above:0",
                "--since", "30d", "--dataset", "ds.jsonl",
            ],
            files="ds.jsonl",
        )

    def test_baseline_update(self):
        self.check_window(BASELINE_UPDATE, files="b.json")

    def test_baseline_update_end(self):
        def run(fields: dict) -> tuple:
            return scrub(self.outcome(self.ws({BAD: fields}), BASELINE_UPDATE, "b.json"))

        null = run({"ended_at": None})
        self.assertNotEqual(run({}), null, "the fixture end is usable")
        self.assertEqual(run({"ended_at": 0}), null)
        for name, value in UNUSABLE:
            with self.subTest(value=name):
                self.assertEqual(run({"ended_at": value}), null)

    def test_policy_without_days(self):
        def selected(fields: dict) -> list[str]:
            with self.direct(self.ws({BAD: fields})) as store:
                sessions, _start, _end = _select_sessions(store, None, NOW)
            return [meta.session_id for meta in sessions]

        later = selected({"started_at": NOW + 3600})
        self.assertNotIn(BAD, later)
        self.assertIn(BAD, selected({"started_at": 0}))
        for name, value in UNUSABLE:
            with self.subTest(value=name):
                self.assertEqual(selected({"started_at": value}), later)

    def test_rolling_project_spend(self):
        def spend(fields: dict) -> float:
            with self.direct(self.ws({BAD: fields})) as store:
                return rolling_project_spend(store, now=NOW)

        outside = spend({"started_at": OUTSIDE})
        self.assertNotEqual(spend({}), outside)
        for name, value in UNUSABLE:
            with self.subTest(value=name):
                self.assertEqual(spend({"started_at": value}), outside)


class TimelineTest(SessionTimeCase):
    """timeline measures from the first event when the start is not usable."""

    def check_timeline(self, argv: list[str]) -> None:
        root = self.ws()
        set_meta(traces(root) / BAD, started_at=first_event_time(traces(root) / BAD))
        expected = self.outcome(root, argv)
        self.assertEqual(expected[0], 0)
        for name, value in UNUSABLE:
            with self.subTest(value=name):
                self.assertEqual(self.outcome(self.ws({BAD: {"started_at": value}}), argv), expected)

    def test_text(self):
        self.check_timeline(["timeline", BAD])

    def test_json(self):
        self.check_timeline(["timeline", BAD, "--format", "json"])


class DashboardTest(SessionTimeCase):
    """dashboard shows "?" for the start, and the trend uses the usable starts."""

    TREND = ["dashboard", "--trend", "--html", "trend.html"]

    def check_row(self, argv: list[str], files: str | None, old: str, new: str) -> None:
        outside = self.outcome(self.ws({BAD: {"started_at": OUTSIDE}}), argv, files)
        text = outside[4][files] if files else outside[1]
        self.assertEqual(text.count(old), 1)
        if files:
            expected = outside[:4] + ({files: text.replace(old, new)},)
        else:
            expected = (outside[0], text.replace(old, new)) + outside[2:]
        for name, value in UNUSABLE:
            with self.subTest(value=name):
                self.assertEqual(
                    self.outcome(self.ws({BAD: {"started_at": value}}), argv, files), expected,
                )

    def test_text(self):
        self.check_row(
            ["dashboard"], None,
            f"  {BAD[:12]:<14}  {'01-01 00:00':<12}", f"  {BAD[:12]:<14}  {'?':<12}",
        )

    def test_html(self):
        self.check_row(
            ["dashboard", "-o", "dash.html"], "dash.html",
            f"<td>{BAD[:12]}</td><td>01-01 00:00</td>", f"<td>{BAD[:12]}</td><td>?</td>",
        )

    def trend_ws(self, meta: dict[str, dict] | None = None) -> Path:
        root = self.ws(meta)
        (traces(root) / "annotations.jsonl").write_text(
            json.dumps({"date": utc_day(NOW - 86400), "note": "new prompt"}) + "\n",
            encoding="utf-8",
        )
        return root

    def test_trend_html(self):
        keep = self.outcome(self.trend_ws(), self.TREND, "trend.html")
        keep_html = keep[4]["trend.html"]
        subtitle = SUBTITLE.findall(keep_html)
        self.assertEqual(subtitle, [
            f'<div class="subtitle">3 sessions &nbsp;·&nbsp; '
            f'{utc_day(NOW - 2 * 86400)} to {utc_day(NOW - 3600 + 14.5)}</div>'
        ])
        annotations = ANNOTATION.findall(keep_html)
        self.assertTrue(annotations)
        for name, value in UNUSABLE:
            with self.subTest(value=name):
                result = self.outcome(
                    self.trend_ws({BAD: {"started_at": value}}), self.TREND, "trend.html",
                )
                page = result[4]["trend.html"]
                self.assertEqual(result[:4], keep[:4])
                self.assertEqual(SUBTITLE.findall(page), subtitle)
                self.assertEqual(ANNOTATION.findall(page), annotations)
                self.assertIn(f"<tr><td>{BAD[:12]}</td><td>?</td>", page)

    def test_trend_html_without_a_usable_start(self):
        for name, value in UNUSABLE:
            with self.subTest(value=name):
                root = self.trend_ws({sid: {"started_at": value} for sid in (GOOD, BAD, CHILD)})
                result = self.outcome(root, self.TREND, "trend.html")
                self.assertEqual(result[0], 0)
                page = result[4]["trend.html"]
                self.assertEqual(
                    SUBTITLE.findall(page), ['<div class="subtitle">3 sessions &nbsp;·&nbsp; </div>'],
                )
                self.assertNotIn('stroke="#e3b341"', page)
                self.assertEqual(page.count("</td><td>?</td>"), 3)

    def test_a_start_of_zero_counts(self):
        result = self.outcome(self.trend_ws({GOOD: {"started_at": 0}}), self.TREND, "trend.html")
        self.assertEqual(SUBTITLE.findall(result[4]["trend.html"]), [
            f'<div class="subtitle">3 sessions &nbsp;·&nbsp; '
            f'1970-01-01 to {utc_day(NOW - 3600 + 14.5)}</div>'
        ])


class FingerprintTest(SessionTimeCase):
    """fingerprint takes its period from the usable starts."""

    def test_period_from_usable_starts(self):
        for argv in (["fingerprint"], ["fingerprint", "--format", "json"]):
            keep = self.outcome(self.ws(), argv)
            self.assertIn(f"fp_{utc_day(NOW - 2 * 86400)}", keep[1])
            for name, value in UNUSABLE:
                with self.subTest(argv=argv, value=name):
                    self.assertEqual(self.outcome(self.ws({BAD: {"started_at": value}}), argv), keep)

    def test_no_usable_start(self):
        for name, value in UNUSABLE:
            with self.subTest(value=name):
                root = self.ws({sid: {"started_at": value} for sid in (GOOD, BAD, CHILD)})
                text = self.run_cli(root, "fingerprint")
                self.assertEqual((text.code, text.err), (0, ""))
                self.assertIn("ID:       (none)\n", text.out)
                self.assertIn("Period:   - to -\n", text.out)
                self.assertIn("Sessions: 3\n", text.out)
                data = self.run_cli(root, "fingerprint", "--format", "json")
                self.assertEqual(data.code, 0)
                fp = json.loads(data.out)
                self.assertEqual((fp["fingerprint_id"], fp["period"]), ("", {"start": "", "end": ""}))

    def test_a_start_of_zero_counts(self):
        result = self.run_cli(self.ws({GOOD: {"started_at": 0}}), "fingerprint")
        self.assertIn("ID:       fp_1970-01-01\n", result.out)
        self.assertIn(f"Period:   1970-01-01 to {utc_day(NOW - 3600 + 14.5)}\n", result.out)


class CurveTest(SessionTimeCase):
    """curve counts the session but measures the days from the usable starts."""

    def test_days_from_usable_starts(self):
        early = {"started_at": NOW - 10 * 86400}
        keep = self.outcome(self.ws({GOOD: early}), ["curve"])
        self.assertIn("(3 sessions, 9 days)", keep[1])
        for name, value in UNUSABLE:
            with self.subTest(value=name):
                root = self.ws({GOOD: early, BAD: {"started_at": value}})
                self.assertEqual(self.outcome(root, ["curve"]), keep)


class McpScanTest(SessionTimeCase):
    """mcp-scan compares descriptions only with sessions that start earlier."""

    ARGV = ["mcp-scan", "--session", BAD]
    CHANGED = f"lookup: tool description changed since session {GOOD}"

    def tools_ws(self, meta: dict[str, dict] | None = None) -> Path:
        root = self.ws(meta)
        for session_id, description in (
            (GOOD, "Look up an issue by number."),
            (BAD, "Look up an issue or a pull request by number."),
        ):
            def add_tools(record, description=description):
                if record["event_type"] == "session_start":
                    record["data"]["tools_available"] = [
                        {"name": "lookup", "description": description},
                    ]
            edit_events(traces(root) / session_id, add_tools)
        return root

    def test_unusable_start_is_not_earlier(self):
        self.assertIn(self.CHANGED, self.outcome(self.tools_ws(), self.ARGV)[1])
        later = self.outcome(self.tools_ws({GOOD: {"started_at": NOW - 60}}), self.ARGV)
        self.assertNotIn(self.CHANGED, later[1])
        for name, value in UNUSABLE:
            with self.subTest(value=name):
                root = self.tools_ws({GOOD: {"started_at": value}})
                self.assertEqual(self.outcome(root, self.ARGV), later)

    def test_unusable_start_has_no_earlier_sessions(self):
        first = self.outcome(self.tools_ws({BAD: {"started_at": OUTSIDE}}), self.ARGV)
        self.assertNotIn(self.CHANGED, first[1])
        for name, value in UNUSABLE:
            with self.subTest(value=name):
                root = self.tools_ws({BAD: {"started_at": value}})
                self.assertEqual(self.outcome(root, self.ARGV), first)


class SampleTest(SessionTimeCase):
    """sample writes null for the start and treats the session as the oldest."""

    ARGV = ["sample", "--strategy", "recent", "--n", "3", "--output", "s.jsonl"]

    def test_null_and_oldest(self):
        outside = self.outcome(self.ws({BAD: {"started_at": OUTSIDE}}), self.ARGV, "s.jsonl")
        jsonl = outside[4]["s.jsonl"]
        self.assertIn(f'"session_id":"{BAD}"', jsonl.splitlines()[-1])
        self.assertEqual(jsonl.count('"started_at":1.0'), 1)
        expected = outside[:4] + ({"s.jsonl": jsonl.replace('"started_at":1.0', '"started_at":null')},)
        for name, value in UNUSABLE:
            with self.subTest(value=name):
                root = self.ws({BAD: {"started_at": value}})
                self.assertEqual(self.outcome(root, self.ARGV, "s.jsonl"), expected)


class ReviewTest(SessionTimeCase):
    """review sorts a start that is not usable as 0."""

    ARGV = ["review", BAD, GOOD, "--format", "json", "--branch", "fix/add"]

    def review_ws(self, bad_start) -> Path:
        root = self.make_workspace()
        for session_id in (GOOD, BAD, CHILD):
            shutil.rmtree(traces(root) / session_id)
        start = NOW - 3600
        _write_session(traces(root), GOOD, start, _session_events(GOOD, start, "good"))
        _write_session(traces(root), BAD, start, _session_events(BAD, start, "bad"))
        set_meta(traces(root) / GOOD, started_at=0.5)
        set_meta(traces(root) / BAD, started_at=bad_start)
        return root

    def test_sorts_as_zero(self):
        zero = self.outcome(self.review_ws(0), self.ARGV)
        self.assertNotEqual(self.outcome(self.review_ws(OUTSIDE), self.ARGV), zero)
        for name, value in UNUSABLE:
            with self.subTest(value=name):
                self.assertEqual(self.outcome(self.review_ws(value), self.ARGV), zero)


SHARE_ARGVS = [
    ["share", BAD, "--stdout"],
    ["share", BAD, "--output", "packet.html"],
    ["share", BAD, "--output", "open.html", "--open"],
    ["share", BAD, "--dry-run"],
]


class ShareTest(SessionTimeCase):
    """share refuses a session whose start, or end when set, is not usable."""

    def test_unusable_start_is_refused(self):
        for name, value in UNUSABLE:
            for argv in SHARE_ARGVS:
                with self.subTest(value=name, argv=argv):
                    self.assert_refused(self.ws({BAD: {"started_at": value}}), argv, STARTED)

    def test_unusable_end_is_refused(self):
        for name, value in UNUSABLE:
            if value is None:
                continue
            for argv in SHARE_ARGVS:
                with self.subTest(value=name, argv=argv):
                    self.assert_refused(self.ws({BAD: {"ended_at": value}}), argv, ENDED)

    def test_open_session_shares(self):
        root = self.ws({BAD: {"ended_at": None}})
        result = self.run_cli(root, "share", BAD, "--stdout")
        self.assertEqual(result.code, 0)
        self.assertIn("<html", result.out)
        result = self.run_cli(root, "share", BAD, "--dry-run")
        self.assertEqual(result.code, 0)

    def check_corrupt(self, content: str | None, expected: str) -> None:
        for argv in (["share", BAD, "--stdout"], ["share", BAD, "--dry-run"]):
            with self.subTest(argv=argv):
                root = self.ws()
                meta = traces(root) / BAD / "meta.json"
                if content is None:
                    meta.chmod(0)
                    self.addCleanup(meta.chmod, 0o644)
                else:
                    meta.write_text(content, encoding="utf-8")
                before = stat_snapshot(root)
                result = self.run_cli(root, *argv)
                self.assertEqual((result.code, result.out, result.err), (1, "", expected))
                self.assertEqual((result.network, result.browser), ([], []))
                self.assertEqual(stat_snapshot(root), before)

    def test_corrupt_meta(self):
        self.check_corrupt(
            "{",
            f"Error: session {BAD} has a corrupt meta.json: Expecting property name "
            "enclosed in double quotes: line 1 column 2 (char 1)\n",
        )

    def test_meta_not_an_object(self):
        self.check_corrupt(
            "[]", f"Error: session {BAD} has a corrupt meta.json: metadata is not a JSON object\n",
        )

    @unittest.skipIf(os.name == "nt", "mode 000 does not block reads on Windows")
    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "root reads a mode 000 file")
    def test_unreadable_meta(self):
        self.check_corrupt(None, f"Session not found: {BAD}\n")


class InRangeTest(SessionTimeCase):
    """An int start is written as the int, as main writes it."""

    START = int(NOW - 3600)

    def test_timeline(self):
        for argv in (["timeline", BAD], ["timeline", BAD, "--format", "json"]):
            with self.subTest(argv=argv):
                as_float = self.outcome(self.ws({BAD: {"started_at": float(self.START)}}), argv)
                as_int = self.outcome(self.ws({BAD: {"started_at": self.START}}), argv)
                if "json" not in argv:
                    self.assertEqual(as_int, as_float)
                    continue
                old = f'"started_at": {float(self.START)!r},'
                self.assertEqual(as_float[1].count(old), 1)
                new = as_float[1].replace(old, f'"started_at": {self.START},')
                self.assertEqual(as_int, (as_float[0], new) + as_float[2:])

    def test_sample(self):
        argv = SampleTest.ARGV
        as_float = self.outcome(self.ws({BAD: {"started_at": float(self.START)}}), argv, "s.jsonl")
        as_int = self.outcome(self.ws({BAD: {"started_at": self.START}}), argv, "s.jsonl")
        jsonl = as_float[4]["s.jsonl"]
        old = f'"started_at":{float(self.START)!r}'
        self.assertEqual(jsonl.count(old), 1)
        new = jsonl.replace(old, f'"started_at":{self.START}')
        self.assertEqual(as_int, as_float[:4] + ({"s.jsonl": new},))


class SkillsTest(SessionTimeCase):
    """skills counts such a session as unreadable or malformed, and ignores such a start."""

    def aggregate(self, fields: dict) -> tuple:
        """skills --since 30d as text, as JSON and through the MCP tool, without the clock time."""
        root = self.ws({BAD: fields})
        text = self.outcome(root, ["skills", "--since", "30d"])
        data = self.outcome(root, ["skills", "--since", "30d", "--format", "json"])
        with self.direct(root) as store:
            tool = mcp_server._tool_skills(store, {"since": "30d"})
        reports = [json.loads(data[1]), json.loads(tool.replace(str(root), "ROOT"))]
        for report in reports:
            report.pop("since")
        return (text, data[:1] + data[2:], *reports)

    def test_unusable_start_is_unreadable(self):
        null = self.aggregate({"started_at": None})
        self.assertIn("(1 unreadable or malformed)", null[0][1])
        for report in null[2:]:
            self.assertEqual(
                (report["sessions_total"], report["sessions_skipped_unreadable"]), (3, 1),
            )
        for name, value in UNUSABLE:
            with self.subTest(value=name):
                self.assertEqual(self.aggregate({"started_at": value}), null)

    def test_a_start_of_zero_is_before_the_window(self):
        outside = self.aggregate({"started_at": OUTSIDE})
        self.assertIn("(1 started before the window)", outside[0][1])
        self.assertEqual(self.aggregate({"started_at": 0}), outside)

    def session(self, fields: dict) -> tuple:
        """skills BAD as text, as JSON and through the MCP tool, after a skill load."""
        root = self.ws({BAD: fields})
        add_skill_load(traces(root) / BAD)
        text = self.outcome(root, ["skills", BAD])
        data = self.outcome(root, ["skills", BAD, "--format", "json"])
        with self.direct(root) as store:
            tool = mcp_server._tool_skills(store, {"session_id": BAD})
        return text, data, tool.replace(str(root), "ROOT")

    def test_load_timing_ignores_unusable_start(self):
        null = self.session({"started_at": None})
        self.assertIn("(skill) at +2s", null[0][1])
        self.assertNotEqual(self.session({"started_at": OUTSIDE}), null)
        self.assertEqual(self.session({"started_at": 0}), null)
        for name, value in UNUSABLE:
            with self.subTest(value=name):
                self.assertEqual(self.session({"started_at": value}), null)


class PrCommentTest(SessionTimeCase):
    """pr-comment takes the duration from event times, and reads such an end as no end."""

    ARGV = ["pr-comment", BAD, "--dry-run"]

    def comment(self, fields: dict) -> tuple:
        """BAD with no session_end event and ten minutes between its stored times, then *fields*."""
        root = self.ws({BAD: {"total_duration_ms": 0, "ended_at": BAD_START + 600, **fields}})
        drop_session_end(traces(root) / BAD)
        return self.outcome(root, self.ARGV)

    @staticmethod
    def rows(outcome: tuple) -> list[str]:
        return [
            line for line in outcome[1].splitlines()
            if line.startswith(("| **Session** |", "| **Status** |"))
        ]

    @staticmethod
    def expect(duration: str, status: str) -> list[str]:
        return [f"| **Session** | `{BAD}` · {duration} |", f"| **Status** | {status} |"]

    def test_unusable_start(self):
        self.assertEqual(self.rows(self.comment({})), self.expect("10m 00s", "completed"))
        null = self.comment({"started_at": None})
        self.assertEqual(self.rows(null), self.expect("25s", "completed"))
        for name, value in UNUSABLE:
            with self.subTest(value=name):
                self.assertEqual(self.comment({"started_at": value}), null)

    def test_unusable_end(self):
        null = self.comment({"ended_at": None})
        self.assertEqual(self.rows(null), self.expect("25s", "in progress"))
        for name, value in UNUSABLE:
            with self.subTest(value=name):
                self.assertEqual(self.comment({"ended_at": value}), null)

    def test_times_of_zero(self):
        self.assertEqual(self.rows(self.comment({"ended_at": 0})), self.expect("25s", "completed"))
        self.assertEqual(
            self.rows(self.comment({"started_at": 0, "ended_at": 12})),
            self.expect("12s", "completed"),
        )

    def test_int_times(self):
        start = int(BAD_START)
        as_int = self.comment({"started_at": start, "ended_at": start + 600})
        as_float = self.comment({"started_at": float(start), "ended_at": float(start + 600)})
        self.assertEqual(as_int, as_float)
        self.assertEqual(self.rows(as_int), self.expect("10m 00s", "completed"))


class LinkTest(SessionTimeCase):
    """The commit-time link matches only a usable start, and with none it links the latest."""

    def linked(self, meta: dict[str, dict] | None = None) -> tuple:
        """pr-comment, review and the MCP review tool, with HEAD committed at BAD's start."""
        root = self.ws(meta)
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(pr_comment, "head_commit_timestamp", lambda cwd=".": BAD_START),
            )
            stack.enter_context(mock.patch.object(pr_comment, "current_branch", lambda cwd=".": ""))
            comment = self.outcome(root, ["pr-comment", "--dry-run", "--review"])
            review = self.outcome(root, ["review", "--format", "json", "--branch", "fix/add"])
            with self.direct(root) as store:
                tool = mcp_server._tool_review(store, {"branch": "fix/add", "format": "json"})
        return comment, review, tool

    @staticmethod
    def links(result: tuple) -> list[tuple[str, str]]:
        """The linked session and how it was linked, for each of the three."""
        comment, review, tool = result
        session = re.search(r"\| \*\*Session\*\* \| `([0-9a-f]+)`", comment[1]).group(1)
        label = re.search(r"Linked by: (commit time|latest)\.", comment[1]).group(1)
        found = [(session, label.replace(" ", "_"))]
        for text in (review[1], tool):
            found += [(item["session_id"], item["link"]) for item in json.loads(text)["sessions"]]
        return found

    def test_unusable_start_is_never_matched(self):
        self.assertEqual(self.links(self.linked()), [(BAD, "commit_time")] * 3)
        outside = self.linked({BAD: {"started_at": OUTSIDE}})
        self.assertEqual(self.links(outside), [(CHILD, "commit_time")] * 3)
        for name, value in UNUSABLE:
            with self.subTest(value=name):
                self.assertEqual(self.linked({BAD: {"started_at": value}}), outside)

    def test_no_usable_start_links_the_latest(self):
        def everywhere(value) -> dict[str, dict]:
            return {session_id: {"started_at": value} for session_id in (GOOD, BAD, CHILD)}

        null = self.linked(everywhere(None))
        self.assertEqual(self.links(null), [(CHILD, "latest")] * 3)
        for name, value in UNUSABLE:
            with self.subTest(value=name):
                self.assertEqual(self.linked(everywhere(value)), null)

    def test_a_start_of_zero_is_matched(self):
        # -5 is not usable. main reads it as a time further from HEAD than 0.
        result = self.linked({
            GOOD: {"started_at": 0}, BAD: {"started_at": -5}, CHILD: {"started_at": -5},
        })
        self.assertEqual(self.links(result), [(GOOD, "commit_time")] * 3)


class DashboardApiTest(SessionTimeCase):
    """server --dashboard drops such a time from /api/sessions and reads it as missing."""

    PATHS = [
        "/api/sessions",
        f"/api/sessions/{BAD}/review",
        f"/api/sessions/{BAD}/share?mode=minimized",
    ]

    def pages(self, fields: dict) -> list[tuple[int, str]]:
        """The status and body of each path in PATHS, from the server --dashboard handler."""
        root = self.ws({BAD: fields})
        with _chdir(root / "work"), _isolated_env(root / "home"):
            handler = _make_handler(TraceStore(str(traces(root))), threading.Lock(), dashboard=True)
            quiet = type("QuietHandler", (handler,), {"log_message": lambda self, *args: None})
            server = HTTPServer(("127.0.0.1", 0), quiet)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                return [self.get(server.server_address[1], path) for path in self.PATHS]
            finally:
                server.shutdown()
                server.server_close()
                thread.join()

    @staticmethod
    def get(port: int, path: str) -> tuple[int, str]:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        try:
            connection.request("GET", path)
            response = connection.getresponse()
            return response.status, response.read().decode("utf-8")
        finally:
            connection.close()

    @staticmethod
    def listed(pages: list[tuple[int, str]]) -> dict:
        """BAD's entry in /api/sessions."""
        return next(item for item in json.loads(pages[0][1]) if item["session_id"] == BAD)

    def test_unusable_start(self):
        null = self.pages({"started_at": None})
        self.assertEqual([status for status, _ in null], [200, 200, 200])
        self.assertNotIn("started_at", self.listed(null))
        outside = self.pages({"started_at": OUTSIDE})
        for path, page, other in zip(self.PATHS, null, outside):
            with self.subTest(path=path):
                self.assertNotEqual(page, other)
        self.assertEqual(self.pages({"started_at": 0}), null)
        for name, value in UNUSABLE:
            with self.subTest(value=name):
                self.assertEqual(self.pages({"started_at": value}), null)

    def test_unusable_end(self):
        null = self.pages({"ended_at": None})
        self.assertEqual([status for status, _ in null], [200, 200, 200])
        self.assertNotIn("ended_at", self.listed(null))
        later = self.pages({"ended_at": BAD_START + 600})
        for path, page, other in zip(self.PATHS, null, later):
            with self.subTest(path=path):
                self.assertNotEqual(page, other)
        for name, value in UNUSABLE:
            with self.subTest(value=name):
                self.assertEqual(self.pages({"ended_at": value}), null)

    def test_an_end_of_zero(self):
        null = self.pages({"ended_at": None})
        zero = self.pages({"ended_at": 0})
        self.assertEqual((zero[0], zero[2]), (null[0], null[2]))
        # The review page reads the recorded end. Its evidence health reads
        # 0 as an end, as health does.
        self.assertEqual(json.loads(zero[1][1])["session"], json.loads(null[1][1])["session"])

    def test_int_times(self):
        start, end = int(BAD_START), int(BAD_START) + 26
        as_int = self.pages({"started_at": start, "ended_at": end})
        as_float = self.pages({"started_at": float(start), "ended_at": float(end)})
        self.assertEqual(as_int[2], as_float[2])
        for index in (0, 1):
            self.assertEqual(json.loads(as_int[index][1]), json.loads(as_float[index][1]))
        for times in (self.listed(as_int), json.loads(as_int[1][1])["session"]):
            self.assertEqual(
                [(type(times[key]), times[key]) for key in ("started_at", "ended_at")],
                [(int, start), (int, end)],
            )


class PostmortemTest(SessionTimeCase):
    """postmortem reads such an end as no end, and such a start as missing."""

    ARGVS = [["postmortem", BAD], ["postmortem", "--list"]]

    def report(self, fields: dict, argv: list[str]) -> tuple:
        """*argv* on BAD with no events, no end and a heartbeat two minutes old, then *fields*."""
        root = self.ws({BAD: {"ended_at": None, **fields}})
        (traces(root) / BAD / "events.ndjson").write_text("", encoding="utf-8")
        clock = mock.patch.object(postmortem, "time", SimpleNamespace(time=lambda: NOW))
        store = TraceStore(str(traces(root)))
        with clock:
            postmortem.write_heartbeat(store, BAD)
        os.utime(postmortem.heartbeat_path(store, BAD), (NOW - 120, NOW - 120))
        with clock:
            return self.outcome(root, argv, ".agent-traces/*/postmortem.md")

    def test_unusable_start(self):
        for argv in self.ARGVS:
            null = self.report({"started_at": None}, argv)
            self.assertEqual(null[0], 1)
            self.assertEqual(list(null[4]), ["postmortem.md"])
            for name, value in UNUSABLE:
                with self.subTest(argv=argv, value=name):
                    self.assertEqual(self.report({"started_at": value}, argv), null)

    def test_unusable_end(self):
        crashed = self.report({}, ["postmortem", BAD])
        self.assertEqual(crashed[0], 1)
        self.assertIn("CRASHED (unknown after 26s)", crashed[1])
        listed = self.report({}, ["postmortem", "--list"])
        self.assertEqual(listed[0], 1)
        self.assertIn("Crashed sessions", listed[1])
        ended = self.report({"ended_at": BAD_START + 26}, ["postmortem", "--list"])
        self.assertEqual(ended[:2], (0, "No crashed sessions found.\n"))
        for argv in self.ARGVS:
            null = self.report({}, argv)
            self.assertEqual(self.report({"ended_at": 0}, argv), null)
            for name, value in UNUSABLE:
                with self.subTest(argv=argv, value=name):
                    self.assertEqual(self.report({"ended_at": value}, argv), null)

    def test_int_times(self):
        start = int(BAD_START)
        for argv in self.ARGVS:
            for fields in ({"started_at": start}, {"ended_at": start + 26}):
                with self.subTest(argv=argv, fields=fields):
                    as_float = {key: float(value) for key, value in fields.items()}
                    self.assertEqual(self.report(fields, argv), self.report(as_float, argv))


class HealthTest(SessionTimeCase):
    """health reads such an end as no end."""

    def health(self, fields: dict, *, marker: bool = False) -> tuple:
        root = self.ws({BAD: fields})
        if marker:
            (traces(root) / ".active-session").write_text(BAD, encoding="utf-8")
        return (
            self.outcome(root, ["health", BAD]),
            self.outcome(root, ["health", BAD, "--format", "json"]),
        )

    @staticmethod
    def finalized(result: tuple) -> tuple:
        line = next(line for line in result[0][1].splitlines() if "Finalized:" in line)
        return line.split("Finalized:")[1].strip(), json.loads(result[1][1])["session_finalized"]

    def test_unusable_end(self):
        null = self.health({"ended_at": None})
        self.assertEqual(self.finalized(null), ("unknown", None))
        self.assertEqual(self.finalized(self.health({})), ("yes", True))
        for name, value in UNUSABLE:
            with self.subTest(value=name):
                self.assertEqual(self.health({"ended_at": value}), null)

    def test_an_end_of_zero_is_finalized(self):
        self.assertEqual(self.finalized(self.health({"ended_at": 0})), ("yes", True))

    def test_unusable_end_with_an_active_marker(self):
        # Metadata the Codex hooks write when the first hook was not SessionStart.
        hook = {"agent_name": "openai-codex", "command": "openai-codex (no session-start)"}
        null = self.health({**hook, "ended_at": None}, marker=True)
        self.assertEqual(self.finalized(null), ("no (active-session marker present)", False))
        self.assertEqual(self.finalized(self.health(hook, marker=True)), ("yes", True))
        for name, value in UNUSABLE:
            with self.subTest(value=name):
                self.assertEqual(self.health({**hook, "ended_at": value}, marker=True), null)


class EvalGateTest(SessionTimeCase):
    """eval gate reads such an end as no end, and takes the duration from event times."""

    STATUS = (
        "evals:\n  - name: status\n    scorer: session_status\n"
        "    expected: killed\n    fail_on: not_equal\n"
    )
    DURATION = (
        "evals:\n  - name: dur\n    scorer: duration_seconds\n"
        "    threshold: 100\n    fail_on: above\n"
    )
    ARGVS = [
        ["eval", "gate", BAD, "--config", "gate.yaml"],
        ["eval", "gate", BAD, "--config", "gate.yaml", "--format", "json"],
    ]

    def gate(self, config: str, fields: dict, prepare=None) -> list[tuple]:
        root = self.ws({BAD: fields})
        if prepare:
            prepare(traces(root) / BAD)
        (root / "work" / "gate.yaml").write_text(config, encoding="utf-8")
        return [self.outcome(root, argv) for argv in self.ARGVS]

    @staticmethod
    def score(result: list[tuple]) -> tuple:
        """The exit code and the score in the JSON output."""
        return result[1][0], json.loads(result[1][1])["results"][0]["score"]

    def test_status(self):
        def status(fields: dict) -> list[tuple]:
            return self.gate(self.STATUS, fields, drop_session_end)

        self.assertEqual(self.score(status({})), (1, "completed"))
        null = status({"ended_at": None})
        self.assertEqual(self.score(null), (0, "killed"))
        self.assertEqual(status({"ended_at": 0}), null)
        for name, value in UNUSABLE:
            with self.subTest(value=name):
                self.assertEqual(status({"ended_at": value}), null)

    def test_duration(self):
        def duration(fields: dict) -> list[tuple]:
            return self.gate(
                self.DURATION, {"total_duration_ms": 0, "ended_at": BAD_START + 600, **fields},
            )

        self.assertEqual(self.score(duration({})), (1, 600))
        self.assertEqual(self.score(duration({"started_at": 0, "ended_at": 12})), (0, 12))
        for field in ("started_at", "ended_at"):
            null = duration({field: None})
            self.assertEqual(self.score(null), (0, 26))
            for name, value in UNUSABLE:
                with self.subTest(field=field, value=name):
                    self.assertEqual(duration({field: value}), null)

    def test_event_time_check(self):
        def poison(records: list[dict]) -> list[dict]:
            records[2]["timestamp"] = "abc"
            return records

        def checked(fields: dict) -> list[tuple]:
            return self.gate(
                self.DURATION, {"total_duration_ms": 0, **fields},
                lambda session_dir: rewrite_events(session_dir, poison),
            )

        self.assertEqual(self.score(checked({})), (0, 26))
        refusal = (
            f"Error: session {BAD} event timestamp must be a Unix time in seconds: "
            "event 3 of 25 (llm_request) has a string timestamp\n"
        )
        for field in ("started_at", "ended_at"):
            null = checked({field: None})
            for result in null:
                self.assertEqual(result[:3], (1, "", refusal))
            for name, value in UNUSABLE:
                with self.subTest(field=field, value=name):
                    self.assertEqual(checked({field: value}), null)


class BaselineCheckTest(SessionTimeCase):
    """baseline check reads such an end as an open session."""

    def check(self, fields: dict) -> tuple:
        root = self.ws()
        self.assertEqual(self.run_cli(root, *BASELINE_UPDATE).code, 0)
        set_meta(traces(root) / BAD, **fields)
        return self.outcome(root, ["baseline", "check", BAD, "--baseline", "b.json"])

    def test_unusable_end(self):
        self.assertNotIn("open session", self.check({})[1])
        null = self.check({"ended_at": None})
        self.assertIn("[OK]  open session", null[1])
        self.assertEqual(self.check({"ended_at": 0}), null)
        for name, value in UNUSABLE:
            with self.subTest(value=name):
                self.assertEqual(self.check({"ended_at": value}), null)


class CognitiveDebtTest(SessionTimeCase):
    """cognitive-debt --session refuses a session whose start is not usable."""

    ARGVS = [
        ["cognitive-debt", "--session", BAD],
        ["cognitive-debt", "--session", BAD, "--format", "json"],
    ]

    def test_unusable_start_is_refused(self):
        for name, value in UNUSABLE:
            for argv in self.ARGVS:
                with self.subTest(value=name, argv=argv):
                    self.assert_refused(
                        self.ws({BAD: {"started_at": value}}), argv,
                        "Error: trace input could not be read safely\n",
                    )

    def test_a_start_of_zero_is_accepted(self):
        root = self.ws({BAD: {"started_at": 0}})
        for argv in self.ARGVS:
            with self.subTest(argv=argv):
                result = self.run_cli(root, *argv)
                self.assertEqual((result.code, result.err), (0, ""))


if __name__ == "__main__":
    unittest.main()
