"""Tests for token budget tracking (issue #27)."""

import io
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore
from agent_trace.token_budget import (
    TokenBudgetReport,
    TokenBudgetWatcher,
    _resolve_limit,
    analyse_token_budget,
    format_token_budget,
)

try:
    import _escapes
except ImportError:
    from tests import _escapes


def _make_store_with_llm_events(input_tokens_per_req, model="claude-sonnet-4"):
    return _make_store_with_request_data(
        [{"model": model, "input_tokens": tok} for tok in input_tokens_per_req]
    )


def _make_store_with_request_data(data_per_req):
    tmpdir = tempfile.mkdtemp()
    store = TraceStore(tmpdir)
    meta = SessionMeta(agent_name="test")
    store.create_session(meta)
    for data in data_per_req:
        e = TraceEvent(
            event_type=EventType.LLM_REQUEST,
            session_id=meta.session_id,
            data=data,
        )
        store.append_event(meta.session_id, e)
    return store, meta.session_id


# Each request's input tokens are its whole context. Request 4 follows a
# compaction. Of claude-sonnet-4's 200,000 tokens, request 3 fills 40% and
# request 6 fills 92.5%; the running sum passes 90% at request 3.
COMPACTED_SESSION = [40_000, 60_000, 80_000, 20_000, 120_000, 185_000]


def _request(tokens, model="claude-sonnet-4"):
    return TraceEvent(
        event_type=EventType.LLM_REQUEST,
        data={"model": model, "input_tokens": tokens},
    )


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


class TestContextFill(unittest.TestCase):
    """The fill is the current request's input tokens, not a running sum."""

    def _report(self, data_per_req, **kwargs):
        store, sid = _make_store_with_request_data(data_per_req)
        self.addCleanup(shutil.rmtree, store.base_dir, True)
        return analyse_token_budget(store, sid, **kwargs)

    def test_fill_is_per_request(self):
        report = self._report(
            [{"model": "claude-sonnet-4", "input_tokens": t} for t in COMPACTED_SESSION]
        )
        self.assertEqual(
            [r.pct_used for r in report.requests],
            [t / 200_000 for t in COMPACTED_SESSION],
        )
        self.assertAlmostEqual(report.final_pct, 0.925)
        self.assertEqual(report.final_context_tokens, 185_000)
        self.assertEqual(report.final_cumulative, sum(COMPACTED_SESSION))

    def test_peak_and_final_after_compaction(self):
        report = self._report([
            {"model": "claude-sonnet-4", "input_tokens": 180_000},
            {"model": "claude-sonnet-4", "input_tokens": 20_000},
        ])
        self.assertAlmostEqual(report.final_pct, 0.1)
        self.assertEqual(report.peak_context_tokens, 180_000)
        self.assertAlmostEqual(report.peak_pct, 0.9)
        self.assertEqual(report.peak_request_index, 1)

    def test_cache_and_nested_usage_tokens_count(self):
        report = self._report([
            {"model": "claude-sonnet-4", "input_tokens": 1_000,
             "cache_read_input_tokens": 149_000},
            {"model": "claude-sonnet-4", "usage": {"prompt_tokens": 60_000}},
        ])
        self.assertEqual([r.input_tokens for r in report.requests], [150_000, 60_000])
        self.assertEqual([r.estimated for r in report.requests], [False, False])
        self.assertAlmostEqual(report.peak_pct, 0.75)

    def test_estimate_covers_only_the_request_itself(self):
        data = {"model": "claude-sonnet-4", "messages": ["x" * 4_000]}
        report = self._report([data, data])
        first, second = report.requests
        self.assertTrue(first.estimated and second.estimated)
        self.assertEqual(first.input_tokens, second.input_tokens)
        self.assertEqual(second.pct_used, second.input_tokens / 200_000)

    def test_text_shows_fill_peak_and_estimate_marker(self):
        report = self._report([
            {"model": "claude-sonnet-4", "input_tokens": 185_000},
            {"model": "claude-sonnet-4", "messages": ["x" * 400]},
        ])
        output = _render_budget(report)
        self.assertIn("Peak:    185,000 tokens  (92.5% of limit) at request 1", output)
        self.assertIn(f"~{report.requests[1].input_tokens:,}", output)
        self.assertIn("~ estimated", output)
        self.assertNotIn("Cumulative", output)


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

    def test_fires_on_current_fill_not_running_sum(self):
        watcher = TokenBudgetWatcher(threshold=0.9)
        fired = [watcher.update(_request(t)) is not None for t in COMPACTED_SESSION]
        self.assertEqual(fired, [False] * 5 + [True])
        self.assertEqual(watcher.context_tokens, 185_000)

    def test_rearms_after_fill_drops_below_threshold(self):
        watcher = TokenBudgetWatcher(threshold=0.9)
        results = [watcher.update(_request(t))
                   for t in (185_000, 190_000, 20_000, 182_000)]
        self.assertEqual([r is not None for r in results], [True, False, False, True])
        self.assertIn("182,000 / 200,000 tokens (91.0%)", results[3])

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


def _budget_report(model="claude-sonnet-4"):
    return TokenBudgetReport(session_id="s1", model=model, context_limit=None, requests=[],
                             final_cumulative=0, final_pct=None, warning_threshold=0.9)


def _render_budget(report):
    out = io.StringIO()
    format_token_budget(report, out)
    return out.getvalue()


# (label, kind, build): build(value) returns a report with value in the place
# the label names. The kinds are in tests/_escapes.py.
TOKEN_BUDGET_CASES = [
    ("model", "text", lambda v: _budget_report(model=v)),
]


class TestTraceTextShowsAsText(unittest.TestCase):
    """A model name in a trace can hold escape codes and line breaks.
    token-budget shows them as text."""

    def test_each_value_shows_escapes_as_text(self):
        _escapes.check_each_value(self, TOKEN_BUDGET_CASES, _render_budget)

    def test_plain_text_prints_unchanged(self):
        _escapes.check_plain_text_unchanged(self, TOKEN_BUDGET_CASES, _render_budget)

    def test_sgr_code_shows_as_text(self):
        output = _render_budget(_budget_report(model="a\x1b[8mhidden"))
        self.assertIn(r"Model: a\x1b[8mhidden", output)
        self.assertNotIn("\x1b", output)

    def test_model_in_trace_shows_as_text(self):
        store, sid = _make_store_with_llm_events([1000], model=_escapes.PAYLOAD)
        self.addCleanup(shutil.rmtree, store.base_dir, True)
        output = _render_budget(analyse_token_budget(store, sid))
        self.assertIn(f"Model: {_escapes.PAYLOAD_TEXT}  ", output)
        for line in output.split("\n"):
            self.assertTrue(line.isprintable(), ascii(line))


if __name__ == "__main__":
    unittest.main()
