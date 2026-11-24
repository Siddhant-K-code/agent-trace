"""A session start that is not a usable time reads as missing.

A session's meta.json can hold any JSON value in started_at and ended_at.
usable_time() refuses text, a bool, null, a list, an object, NaN, infinity, a
negative number and a time after 9999-12-30T00:00:00Z. These tests store each
such value and check that a command leaves the session out of its time
window, shows the start as missing, or, for share, refuses the session.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import unittest
from datetime import datetime, timezone
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
        _session_events,
        _write_session,
    )

from agent_trace import oncall

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

STARTED = "Error: session started_at must be a Unix time in seconds\n"
ENDED = "Error: session ended_at must be a Unix time in seconds\n"

SUBTITLE = re.compile(r'<div class="subtitle">.*?</div>')
ANNOTATION = re.compile(r'<line [^>]*stroke="#e3b341"[^>]*/><text [^>]*>[^<]*</text>')


def traces(root: Path) -> Path:
    return root / "work" / ".agent-traces"


def set_meta(session_dir: Path, **fields) -> None:
    path = session_dir / "meta.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta.update(fields)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def edit_events(session_dir: Path, edit) -> None:
    """Apply *edit* to each event record and rebuild the hash chain."""
    path = session_dir / "events.ndjson"
    lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        record.pop("prev_hash", None)
        edit(record)
        if lines:
            record["prev_hash"] = hashlib.sha256(lines[-1].encode()).hexdigest()
        lines.append(json.dumps(record, separators=(",", ":")))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    def ws(self, meta: dict[str, dict] | None = None) -> Path:
        """The fixture, with the meta.json fields in *meta* set per session ID."""
        root = self.make_workspace()
        for session_id, fields in (meta or {}).items():
            set_meta(traces(root) / session_id, **fields)
        return root

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

    def check_window(self, argv: list[str], *, snapshots: bool = False) -> None:
        def ws(fields: dict | None = None) -> Path:
            root = self.ws({BAD: fields} if fields is not None else None)
            if snapshots:
                write_config_snapshots(root)
            return root

        outside = self.outcome(ws({"started_at": OUTSIDE}), argv)
        self.assertNotEqual(self.outcome(ws(), argv), outside, "the fixture start is in the window")
        for name, value in UNUSABLE:
            with self.subTest(value=name):
                self.assertEqual(self.outcome(ws({"started_at": value}), argv), outside)

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


if __name__ == "__main__":
    unittest.main()
