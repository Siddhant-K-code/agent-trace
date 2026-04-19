"""Tests for issue #44: token inflation calculator."""

from __future__ import annotations

import io

import pytest

from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore


def _make_store(tmp_path) -> TraceStore:
    return TraceStore(str(tmp_path / "traces"))


def _populate(store: TraceStore, n: int = 5) -> None:
    for _ in range(n):
        meta = SessionMeta()
        store.create_session(meta)
        sid = meta.session_id
        for event in [
            TraceEvent(
                event_type=EventType.LLM_REQUEST,
                session_id=sid,
                data={"model": "claude-opus-4-6", "input_tokens": 1000,
                      "messages": [{"role": "user", "content": "x" * 200}]},
            ),
            TraceEvent(
                event_type=EventType.USER_PROMPT,
                session_id=sid,
                data={"content": "x" * 400},
            ),
        ]:
            store.append_event(sid, event)


class TestInflation:
    def test_analyse_inflation_returns_report(self, tmp_path):
        from agent_trace.inflation import analyse_inflation
        store = _make_store(tmp_path)
        _populate(store, 5)
        report = analyse_inflation(
            store,
            model_baseline="claude-opus-4-6",
            model_inflated="claude-opus-4-7",
        )
        assert report.session_count >= 1
        assert report.factor_inflated > report.factor_baseline
        assert report.avg_tokens_inflated >= report.avg_tokens_baseline

    def test_inflation_factor_applied(self, tmp_path):
        from agent_trace.inflation import analyse_inflation, _resolve_factor
        store = _make_store(tmp_path)
        _populate(store, 3)
        report = analyse_inflation(store)
        factor_b = _resolve_factor("claude-opus-4-6")
        factor_i = _resolve_factor("claude-opus-4-7")
        assert factor_i > factor_b
        if report.avg_tokens_baseline > 0:
            ratio = report.avg_tokens_inflated / report.avg_tokens_baseline
            assert abs(ratio - (factor_i / factor_b)) < 0.01

    def test_format_inflation_output(self, tmp_path):
        from agent_trace.inflation import analyse_inflation, format_inflation
        store = _make_store(tmp_path)
        _populate(store, 3)
        report = analyse_inflation(store)
        buf = io.StringIO()
        format_inflation(report, out=buf)
        output = buf.getvalue()
        assert "Token Inflation Report" in output
        assert "Monthly" in output

    def test_resolve_factor_known_model(self):
        from agent_trace.inflation import _resolve_factor
        assert _resolve_factor("claude-opus-4-6") == 1.0
        assert _resolve_factor("claude-opus-4-7") == 1.38

    def test_resolve_factor_prefix_match(self):
        from agent_trace.inflation import _resolve_factor
        assert _resolve_factor("claude-opus-4-7-20260101") == 1.38

    def test_resolve_factor_unknown(self):
        from agent_trace.inflation import _resolve_factor
        assert _resolve_factor("unknown-model-xyz") == 1.0

    def test_empty_store(self, tmp_path):
        from agent_trace.inflation import analyse_inflation
        store = _make_store(tmp_path)
        report = analyse_inflation(store)
        assert report.session_count == 0
        assert report.avg_tokens_baseline == 0

    def test_by_content_type_breakdown(self, tmp_path):
        from agent_trace.inflation import analyse_inflation, CONTENT_TYPES
        store = _make_store(tmp_path)
        _populate(store, 3)
        report = analyse_inflation(store)
        assert len(report.by_content_type) == len(CONTENT_TYPES)
        for ct in report.by_content_type:
            assert ct.content_type in CONTENT_TYPES

    def test_cli_has_inflation_command(self):
        from agent_trace.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "inflation",
            "--compare", "claude-opus-4-6,claude-opus-4-7",
            "--sessions", "20",
        ])
        assert args.compare == "claude-opus-4-6,claude-opus-4-7"
        assert args.sessions == 20
