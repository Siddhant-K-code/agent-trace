import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace.cli import build_parser
from agent_trace.config_watch import ConfigSnapshot, FileSnapshot
from agent_trace.context_score import build_context_score_report
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore


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


if __name__ == "__main__":
    unittest.main()
