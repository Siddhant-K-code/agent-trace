"""Tests for issue #44: token inflation calculator."""

from __future__ import annotations

import io
import os
import tempfile
import unittest

from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore


def _make_store(tmp_dir: str) -> TraceStore:
    return TraceStore(os.path.join(tmp_dir, "traces"))


def _populate(store: TraceStore, n: int = 5) -> None:
    for _ in range(n):
        meta = SessionMeta()
        store.create_session(meta)
        sid = meta.session_id
        for event in [
            TraceEvent(
                event_type=EventType.LLM_REQUEST,
                session_id=sid,
                data={
                    "model": "claude-opus-4-6",
                    "system": "You are a helpful assistant.",
                    "tools": [{"name": "read_file"}],
                    "messages": [{"role": "user", "content": "x" * 200}],
                },
            ),
            TraceEvent(
                event_type=EventType.USER_PROMPT,
                session_id=sid,
                data={"content": "x" * 400},
            ),
        ]:
            store.append_event(sid, event)


class TestInflationFactors(unittest.TestCase):
    def test_resolve_factor_baseline(self):
        from agent_trace.inflation import _resolve_factor
        self.assertEqual(_resolve_factor("claude-opus-4-6"), 1.0)

    def test_resolve_factor_inflated(self):
        from agent_trace.inflation import _resolve_factor
        self.assertEqual(_resolve_factor("claude-opus-4-7"), 1.38)

    def test_resolve_factor_prefix_match(self):
        from agent_trace.inflation import _resolve_factor
        self.assertEqual(_resolve_factor("claude-opus-4-7-20260101"), 1.38)

    def test_resolve_factor_unknown_defaults_to_one(self):
        from agent_trace.inflation import _resolve_factor
        self.assertEqual(_resolve_factor("unknown-model-xyz"), 1.0)


class TestInflationAnalysis(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_analyse_returns_report(self):
        from agent_trace.inflation import analyse_inflation
        store = _make_store(self._tmp)
        _populate(store, 5)
        report = analyse_inflation(
            store,
            model_baseline="claude-opus-4-6",
            model_inflated="claude-opus-4-7",
        )
        self.assertGreaterEqual(report.session_count, 1)
        self.assertGreater(report.factor_inflated, report.factor_baseline)
        self.assertGreaterEqual(report.avg_tokens_inflated, report.avg_tokens_baseline)

    def test_inflation_factor_ratio_correct(self):
        from agent_trace.inflation import analyse_inflation, _resolve_factor
        store = _make_store(self._tmp)
        _populate(store, 3)
        report = analyse_inflation(store)
        if report.avg_tokens_baseline > 0:
            ratio = report.avg_tokens_inflated / report.avg_tokens_baseline
            expected = _resolve_factor("claude-opus-4-7") / _resolve_factor("claude-opus-4-6")
            self.assertAlmostEqual(ratio, expected, places=2)

    def test_by_content_type_all_present(self):
        from agent_trace.inflation import analyse_inflation, CONTENT_TYPES
        store = _make_store(self._tmp)
        _populate(store, 3)
        report = analyse_inflation(store)
        ct_names = {ct.content_type for ct in report.by_content_type}
        for expected in CONTENT_TYPES:
            self.assertIn(expected, ct_names)

    def test_system_prompt_bucket_populated(self):
        from agent_trace.inflation import analyse_inflation
        store = _make_store(self._tmp)
        _populate(store, 3)
        report = analyse_inflation(store)
        sp = next(ct for ct in report.by_content_type if ct.content_type == "system_prompt")
        self.assertGreater(sp.tokens_baseline, 0)

    def test_tool_definitions_bucket_populated(self):
        from agent_trace.inflation import analyse_inflation
        store = _make_store(self._tmp)
        _populate(store, 3)
        report = analyse_inflation(store)
        td = next(ct for ct in report.by_content_type if ct.content_type == "tool_definitions")
        self.assertGreater(td.tokens_baseline, 0)

    def test_format_output(self):
        from agent_trace.inflation import analyse_inflation, format_inflation
        store = _make_store(self._tmp)
        _populate(store, 3)
        report = analyse_inflation(store)
        buf = io.StringIO()
        format_inflation(report, out=buf)
        output = buf.getvalue()
        self.assertIn("Token Inflation Report", output)
        self.assertIn("Monthly", output)

    def test_empty_store(self):
        from agent_trace.inflation import analyse_inflation
        store = _make_store(self._tmp)
        report = analyse_inflation(store)
        self.assertEqual(report.session_count, 0)
        self.assertEqual(report.avg_tokens_baseline, 0)

    def test_cli_has_inflation_command(self):
        from agent_trace.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "inflation",
            "--compare", "claude-opus-4-6,claude-opus-4-7",
            "--sessions", "20",
        ])
        self.assertEqual(args.compare, "claude-opus-4-6,claude-opus-4-7")
        self.assertEqual(args.sessions, 20)
