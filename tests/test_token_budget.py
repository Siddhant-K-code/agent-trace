"""Tests for token budget tracking (issue #27)."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore
from agent_trace.token_budget import (
    TokenBudgetWatcher,
    _resolve_limit,
    analyse_token_budget,
)


def _make_store_with_llm_events(input_tokens_per_req, model="claude-sonnet-4"):
    tmpdir = tempfile.mkdtemp()
    store = TraceStore(tmpdir)
    meta = SessionMeta(agent_name="test")
    store.create_session(meta)
    for tok in input_tokens_per_req:
        e = TraceEvent(
            event_type=EventType.LLM_REQUEST,
            session_id=meta.session_id,
            data={"model": model, "input_tokens": tok},
        )
        store.append_event(meta.session_id, e)
    return store, meta.session_id


class TestResolveLimit(unittest.TestCase):
    def test_known_model(self):
        self.assertEqual(_resolve_limit("claude-sonnet-4"), 200_000)

    def test_prefix_match(self):
        limit = _resolve_limit("claude-sonnet-4-20251022")
        self.assertEqual(limit, 200_000)

    def test_unknown_model(self):
        self.assertIsNone(_resolve_limit("unknown-model-xyz"))

    def test_empty_model(self):
        self.assertIsNone(_resolve_limit(""))

    def test_gpt4o(self):
        self.assertEqual(_resolve_limit("gpt-4o"), 128_000)


class TestAnalyseTokenBudget(unittest.TestCase):
    def test_basic_accumulation(self):
        store, sid = _make_store_with_llm_events([1000, 2000, 3000])
        report = analyse_token_budget(store, sid)
        self.assertEqual(len(report.requests), 3)
        self.assertEqual(report.final_cumulative, 6000)

    def test_pct_computed(self):
        store, sid = _make_store_with_llm_events([100_000])
        report = analyse_token_budget(store, sid)
        self.assertIsNotNone(report.final_pct)
        self.assertAlmostEqual(report.final_pct, 0.5, places=2)

    def test_no_events(self):
        tmpdir = tempfile.mkdtemp()
        store = TraceStore(tmpdir)
        meta = SessionMeta(agent_name="test")
        store.create_session(meta)
        report = analyse_token_budget(store, meta.session_id)
        self.assertEqual(report.final_cumulative, 0)
        self.assertEqual(len(report.requests), 0)

    def test_warning_threshold_flag(self):
        # 180k tokens out of 200k = 90%
        store, sid = _make_store_with_llm_events([180_000])
        report = analyse_token_budget(store, sid, warning_threshold=0.85)
        self.assertIsNotNone(report.final_pct)
        self.assertGreater(report.final_pct, report.warning_threshold)


class TestTokenBudgetWatcher(unittest.TestCase):
    def test_no_fire_below_threshold(self):
        watcher = TokenBudgetWatcher(threshold=0.9)
        event = TraceEvent(
            event_type=EventType.LLM_REQUEST,
            data={"model": "claude-sonnet-4", "input_tokens": 10_000},
        )
        result = watcher.update(event)
        self.assertIsNone(result)

    def test_fires_at_threshold(self):
        watcher = TokenBudgetWatcher(threshold=0.9)
        event = TraceEvent(
            event_type=EventType.LLM_REQUEST,
            data={"model": "claude-sonnet-4", "input_tokens": 185_000},
        )
        result = watcher.update(event)
        self.assertIsNotNone(result)
        self.assertIn("TokenBudgetWatcher", result)
        self.assertIn("185,000", result)

    def test_fires_only_once(self):
        watcher = TokenBudgetWatcher(threshold=0.5)
        event = TraceEvent(
            event_type=EventType.LLM_REQUEST,
            data={"model": "claude-sonnet-4", "input_tokens": 120_000},
        )
        first = watcher.update(event)
        second = watcher.update(event)
        self.assertIsNotNone(first)
        self.assertIsNone(second)

    def test_unknown_model_no_fire(self):
        watcher = TokenBudgetWatcher(threshold=0.9)
        event = TraceEvent(
            event_type=EventType.LLM_REQUEST,
            data={"model": "unknown-model", "input_tokens": 999_999},
        )
        result = watcher.update(event)
        self.assertIsNone(result)

    def test_non_llm_event_ignored(self):
        watcher = TokenBudgetWatcher(threshold=0.5)
        event = TraceEvent(
            event_type=EventType.TOOL_CALL,
            data={"tool_name": "bash"},
        )
        result = watcher.update(event)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
