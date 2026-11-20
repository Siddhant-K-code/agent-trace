import argparse
import io
import json
import os
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace.cli import build_parser
from agent_trace.config_watch import ConfigSnapshot, FileSnapshot
from agent_trace.context_score import (
    CAVEATS,
    TEXT_CAVEAT,
    build_context_score_report,
    cmd_context_score,
    format_context_score_json,
    format_context_score_text,
)
from agent_trace.cost import estimate_cost
from agent_trace.lint import lint_session
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore

try:
    from test_dashboard import _codex_session
except ImportError:  # pragma: no cover - package-style test discovery
    from tests.test_dashboard import _codex_session


def _write_snapshots(root, snapshots):
    path = root / ".agent-traces" / ".config-snapshots.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([snapshot.to_dict() for snapshot in snapshots]))


def _snapshot(snapshot_id, ts, sha, label):
    return ConfigSnapshot(
        snapshot_id=snapshot_id,
        timestamp=ts,
        label=label,
        files=[FileSnapshot(path="AGENTS.md", sha256=sha, mtime=ts, exists=True)],
    )


def _session(store, ts, sid_suffix, events):
    meta = SessionMeta(agent_name="codex")
    meta.session_id = f"session{sid_suffix:02d}"
    meta.started_at = ts
    meta.ended_at = ts + 10
    meta.tool_calls = sum(1 for event in events if event.event_type == EventType.TOOL_CALL)
    store.create_session(meta)
    for event in events:
        event.timestamp = ts
        store.append_event(meta.session_id, event)
    return meta.session_id


def _tool(name="Read", **arguments):
    return TraceEvent(
        event_type=EventType.TOOL_CALL,
        data={"tool_name": name, "arguments": arguments},
    )


def _end():
    return TraceEvent(event_type=EventType.SESSION_END, data={})


class TestContextScore(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.trace_dir = self.tmpdir / ".agent-traces"
        self.store = TraceStore(self.trace_dir)
        (self.tmpdir / "AGENTS.md").write_text(
            "Scope:\n- src/**\nNever use rm\n",
        )

    def test_no_history_reports_current_stats(self):
        now = time.time()
        _session(self.store, now - 10, 1, [_tool("Read", file_path="src/app.py"), _end()])

        report = build_context_score_report(
            self.store,
            self.tmpdir,
            context_file="AGENTS.md",
            history_days=1,
            min_sessions=1,
        )

        self.assertTrue(report.no_history)
        self.assertIsNotNone(report.current)
        self.assertEqual(report.current.session_count, 1)
        self.assertIsNone(report.baseline)

    def test_compare_versions_scores_dimensions(self):
        now = time.time()
        _write_snapshots(self.tmpdir, [
            _snapshot("old111", now - 1000, "oldsha", "old"),
            _snapshot("new222", now - 500, "newsha", "new"),
        ])
        for i in range(5):
            _session(self.store, now - 900 + i, i, [
                _tool("Read", file_path="src/app.py"),
                _tool("Read", file_path="src/app.py"),
                _tool("Read", file_path="src/app.py"),
                _tool("Bash", command="rm -rf tmp"),
                _end(),
            ])
        for i in range(5, 10):
            _session(self.store, now - 400 + i, i, [
                _tool("Read", file_path="src/app.py"),
                _end(),
            ])

        report = build_context_score_report(
            self.store,
            self.tmpdir,
            context_file="AGENTS.md",
            history_days=1,
            compare=True,
            min_sessions=5,
        )

        self.assertFalse(report.insufficient_data)
        self.assertEqual(report.current.label, "new")
        self.assertEqual(report.baseline.label, "old")
        self.assertEqual(len(report.dimensions), 4)
        self.assertIsNotNone(report.overall_score)
        self.assertGreater(report.baseline.instruction_violation_rate, 0.0)

    def test_insufficient_data_below_min_sessions(self):
        now = time.time()
        _write_snapshots(self.tmpdir, [
            _snapshot("old111", now - 1000, "oldsha", "old"),
            _snapshot("new222", now - 500, "newsha", "new"),
        ])
        _session(self.store, now - 900, 1, [_tool("Read", file_path="src/app.py"), _end()])
        _session(self.store, now - 400, 2, [_tool("Read", file_path="src/app.py"), _end()])

        report = build_context_score_report(
            self.store,
            self.tmpdir,
            context_file="AGENTS.md",
            history_days=1,
            compare=True,
            min_sessions=5,
        )

        self.assertTrue(report.insufficient_data)
        self.assertTrue(all(d.score is None for d in report.dimensions))

    def test_suggestions_cover_scope_and_redundant_reads(self):
        now = time.time()
        for i in range(3):
            _session(self.store, now - 100 + i, i, [
                _tool("Read", file_path="outside/secret.py"),
                _tool("Read", file_path="outside/secret.py"),
                _tool("Read", file_path="outside/secret.py"),
                _end(),
            ])

        report = build_context_score_report(
            self.store,
            self.tmpdir,
            context_file="AGENTS.md",
            history_days=1,
            min_sessions=1,
        )

        text = " ".join(report.suggestions)
        self.assertIn("redundant-read", text)
        self.assertIn("Out-of-scope", text)

    def test_parser_registers_context_score(self):
        parser = build_parser()
        args = parser.parse_args(["context-score", "--file", "CLAUDE.md", "--history", "14", "--compare"])
        self.assertEqual(args.command, "context-score")
        self.assertEqual(args.file, "CLAUDE.md")
        self.assertEqual(args.history, 14)
        self.assertTrue(args.compare)

    def _two_version_report(self, baseline_calls=3, current_calls=1):
        now = time.time()
        _write_snapshots(self.tmpdir, [
            _snapshot("old111", now - 1000, "oldsha", "old"),
            _snapshot("new222", now - 500, "newsha", "new"),
        ])
        for i in range(5):
            _session(self.store, now - 900 + i, i, [
                *[_tool("Read", file_path=f"src/a{n}.py") for n in range(baseline_calls)],
                _end(),
            ])
        for i in range(5, 10):
            _session(self.store, now - 400 + i, i, [
                *[_tool("Read", file_path=f"src/a{n}.py") for n in range(current_calls)],
                _end(),
            ])
        return build_context_score_report(
            self.store,
            self.tmpdir,
            context_file="AGENTS.md",
            history_days=1,
            compare=True,
            min_sessions=5,
        )

    def test_no_change_scores_50_and_grades_f(self):
        report = self._two_version_report(baseline_calls=2, current_calls=2)
        self.assertEqual(report.overall_score, 50.0)
        self.assertEqual(report.grade, "F")

    def test_text_output_has_caveat_under_overall_score(self):
        report = self._two_version_report()
        buf = io.StringIO()
        format_context_score_text(report, buf)
        lines = buf.getvalue().splitlines()
        overall = next(i for i, line in enumerate(lines) if line.startswith("Overall score:"))
        self.assertEqual(
            lines[overall + 1],
            "50 = no change vs baseline. Efficiency proxies, not outcomes. Not causal.",
        )
        self.assertEqual(lines[overall + 1], TEXT_CAVEAT)

    def test_json_output_adds_caveats_and_keeps_fields(self):
        report = self._two_version_report()
        data = json.loads(format_context_score_json(report))
        self.assertEqual(data["caveats"], list(CAVEATS))
        self.assertTrue(all(isinstance(c, str) and c for c in data["caveats"]))
        joined = " ".join(data["caveats"])
        self.assertIn("not outcomes", joined)
        self.assertIn("50 = no change", joined)
        self.assertIn(
            "Sessions that start before the first config-watch snapshot count as the current version.",
            data["caveats"],
        )
        self.assertEqual(len(data["caveats"]), 8)
        for key in (
            "context_file", "window_start", "window_end", "min_sessions",
            "insufficient_data", "no_history", "overall_score", "grade",
            "current", "baseline", "versions", "dimensions", "suggestions",
        ):
            self.assertIn(key, data)
        self.assertEqual(
            [d["name"] for d in data["dimensions"]],
            ["Cost efficiency", "Tool efficiency", "Lint violations", "Scope adherence"],
        )

    def test_json_output_has_caveats_without_sessions(self):
        report = build_context_score_report(
            self.store,
            self.tmpdir,
            context_file="AGENTS.md",
            history_days=1,
        )
        data = json.loads(format_context_score_json(report))
        self.assertIsNone(data["current"])
        self.assertEqual(data["caveats"], list(CAVEATS))

    def test_help_does_not_claim_outcomes(self):
        parser = build_parser()
        sub_action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
        choice = next(c for c in sub_action._choices_actions if c.dest == "context-score")
        self.assertNotIn("outcomes", choice.help)
        self.assertIn("not an outcome score", choice.help)
        description = sub_action.choices["context-score"].description
        self.assertIn("Not an outcome score", description)
        self.assertIn("50 = no change vs baseline", description)

    def _run_command(self, *argv):
        """Run cmd_context_score with these CLI arguments. Return (exit code, stdout, stderr)."""
        args = build_parser().parse_args(list(argv))
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cmd_context_score(args)
        return code, out.getvalue(), err.getvalue()

    def _chdir(self, path):
        old = os.getcwd()
        os.chdir(path)
        self.addCleanup(os.chdir, old)

    def test_command_reads_the_current_directory_and_the_trace_dir(self):
        """The command read the context file and the snapshots next to the
        trace dir, so with a trace dir outside the repository it found
        neither. With AGENT_STRACE_WORKSPACE set, the sessions are in a
        workspace of the trace dir and the snapshots stay at its root."""
        self._chdir(self.tmpdir)
        now = time.time()
        for workspace in ("", "team1"):
            with self.subTest(workspace=workspace), \
                    patch.dict(os.environ, {"AGENT_STRACE_WORKSPACE": workspace}):
                traces = Path(tempfile.mkdtemp()) / "traces"
                store = TraceStore(traces)
                snapshots = [
                    _snapshot("old111", now - 1000, "oldsha", "old"),
                    _snapshot("new222", now - 500, "newsha", "new"),
                ]
                for snapshot in snapshots:
                    snapshot.workspace = os.path.realpath(self.tmpdir)
                traces.mkdir()
                (traces / ".config-snapshots.json").write_text(
                    json.dumps([snapshot.to_dict() for snapshot in snapshots])
                )
                _session(store, now - 900, 1, [_tool("Read", file_path="src/app.py"), _end()])
                _session(store, now - 400, 2, [_tool("Read", file_path="src/app.py"), _end()])

                code, out, _ = self._run_command(
                    "--trace-dir", str(traces), "context-score",
                    "--format", "json", "--min-sessions", "1", "--history", "1",
                )

                self.assertEqual(code, 0)
                versions = json.loads(out)["versions"]
                self.assertEqual(
                    [(v["version_hash"], v["label"], v["session_count"]) for v in versions],
                    [("oldsha", "old", 1), ("newsha", "new", 1)],
                )

    @unittest.skipIf(os.name == "nt", "Windows cannot remove the current directory")
    def test_deleted_current_directory_is_an_error(self):
        gone = self.tmpdir / "gone"
        gone.mkdir()
        self._chdir(gone)
        gone.rmdir()
        code, out, err = self._run_command("--trace-dir", str(self.trace_dir), "context-score")
        self.assertEqual((code, out), (1, ""))
        self.assertEqual(err, "error: cannot read the current directory: FileNotFoundError\n")


class TestContextScoreUnpricedSessions(unittest.TestCase):
    """Cost is the `agent-strace cost` total; sessions it refuses are not $0."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.store = TraceStore(self.tmpdir / ".agent-traces")
        (self.tmpdir / "AGENTS.md").write_text("Scope:\n- src/**\n")

    def _report(self, **kwargs):
        kwargs.setdefault("min_sessions", 1)
        return build_context_score_report(
            self.store, self.tmpdir, context_file="AGENTS.md", history_days=1, **kwargs
        )

    def test_unpriceable_sessions_are_left_out_of_avg_cost(self):
        now = time.time()
        priced = _codex_session(self.store, started_at=now - 30)
        _codex_session(self.store, prompt_data=[], started_at=now - 20)
        broken = _codex_session(self.store, started_at=now - 10)
        with open(self.store.base_dir / broken / "events.ndjson", "a") as f:
            f.write("not json\n")
        want = estimate_cost(self.store, priced).total_cost

        report = self._report()
        version = report.current
        costs = {s.session_id: s.cost for s in version.sessions}
        self.assertEqual(costs.pop(priced), want)
        self.assertEqual(list(costs.values()), [None, None])
        self.assertEqual(version.unpriced_sessions, 2)
        self.assertEqual(version.avg_cost, want)

        buf = io.StringIO()
        format_context_score_text(report, buf)
        self.assertIn(f"  cost/session: ${want:.4f} (2 not priced)\n", buf.getvalue())
        self.assertNotIn("$0.0000", buf.getvalue())
        data = json.loads(format_context_score_json(report))
        self.assertEqual(data["current"]["avg_cost"], want)
        self.assertEqual(data["current"]["unpriced_sessions"], 2)

    def test_no_priced_session_has_unknown_avg_cost(self):
        _codex_session(self.store, prompt_data=[], started_at=time.time() - 10)
        report = self._report()
        self.assertIsNone(report.current.avg_cost)
        buf = io.StringIO()
        format_context_score_text(report, buf)
        self.assertIn("  cost/session: n/a (1 not priced)\n", buf.getvalue())
        data = json.loads(format_context_score_json(report))
        self.assertIsNone(data["current"]["avg_cost"])

    def test_unpriced_baseline_does_not_score_cost(self):
        now = time.time()
        _write_snapshots(self.tmpdir, [
            _snapshot("old111", now - 1000, "oldsha", "old"),
            _snapshot("new222", now - 500, "newsha", "new"),
        ])
        for i in range(2):
            _codex_session(self.store, prompt_data=[], started_at=now - 900 + i)
        for i in range(2):
            _codex_session(self.store, started_at=now - 400 + i)

        report = self._report(compare=True, min_sessions=2)
        self.assertEqual(report.baseline.unpriced_sessions, 2)
        cost = next(d for d in report.dimensions if d.name == "Cost efficiency")
        self.assertIsNone(cost.baseline_value)
        self.assertIsNone(cost.score)
        self.assertFalse(cost.sufficient_data)
        tools = next(d for d in report.dimensions if d.name == "Tool efficiency")
        self.assertTrue(tools.sufficient_data)
        buf = io.StringIO()
        format_context_score_text(report, buf)
        self.assertIn("  cost/session: n/a (2 not priced)\n", buf.getvalue())

    def test_pricing_bugs_are_not_hidden(self):
        _codex_session(self.store, started_at=time.time() - 10)
        with patch("agent_trace.context_score.session_cost", side_effect=TypeError("bug")):
            with self.assertRaises(TypeError):
                self._report()


class TestContextScoreUnlintedSessions(unittest.TestCase):
    """Sessions lint could not fully check are not counted as 0 findings."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.store = TraceStore(self.tmpdir / ".agent-traces")
        (self.tmpdir / "AGENTS.md").write_text("Scope:\n- src/**\n")

    def _report(self):
        report = build_context_score_report(
            self.store, self.tmpdir, context_file="AGENTS.md", history_days=1, min_sessions=1
        )
        buf = io.StringIO()
        format_context_score_text(report, buf)
        return report, buf.getvalue(), json.loads(format_context_score_json(report))

    def test_refused_session_is_left_out_of_lint_averages(self):
        now = time.time()
        linted = _codex_session(self.store, started_at=now - 20)
        _codex_session(self.store, prompt_data=[], started_at=now - 10)
        want = float(len(lint_session(self.store, linted).findings))

        report, text, data = self._report()
        self.assertEqual(report.current.linted_session_count, 1)
        self.assertEqual(report.current.lint_per_session, want)
        self.assertIn(f"  lint findings/session: {want:.1f} (1 not linted)\n", text)
        self.assertEqual(data["current"]["unlinted_sessions"], 1)

    def test_rule_that_could_not_run_makes_lint_unavailable(self):
        _codex_session(self.store, started_at=time.time() - 10)

        def broken(events, cfg):
            raise RuntimeError("boom")

        with patch.dict("agent_trace.lint._RULES", {"no-output": broken}):
            report, text, data = self._report()
        self.assertIsNone(report.current.lint_per_session)
        self.assertIsNone(report.current.redundant_reads_per_session)
        self.assertIsNone(report.current.context_saturation_rate)
        self.assertIn("  lint findings/session: n/a (1 not linted)\n", text)
        self.assertIsNone(data["current"]["lint_per_session"])
        self.assertEqual(data["current"]["unlinted_sessions"], 1)


if __name__ == "__main__":
    unittest.main()
