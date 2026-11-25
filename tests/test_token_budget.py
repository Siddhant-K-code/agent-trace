"""Tests for token budget tracking (issue #27)."""

import io
import os
import shutil
import subprocess
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

try:  # keeps product telemetry off in this process and its children
    import _no_telemetry
except ImportError:  # pragma: no cover - run as a package module
    from tests import _no_telemetry

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))


def _run_cli(trace_dir, *argv):
    """Run agent-strace in a child process, as a user runs it."""
    env = _no_telemetry.without_telemetry({
        key: value
        for key, value in os.environ.items()
        if not key.startswith("AGENT_STRACE_") and key != "AGENT_TRACE_DIR"
    })
    env["PYTHONPATH"] = os.pathsep.join(item for item in (SRC, env.get("PYTHONPATH", "")) if item)
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, "-m", "agent_trace.entry", "--trace-dir", trace_dir, *argv],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        timeout=120,
    )


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


US_SONNET_45 = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"


class TestResolveLimitByCostNameRule(unittest.TestCase):
    """A model name finds its context limit as cost finds its price."""

    def test_bedrock_and_prefixed_names_get_their_base_model_limit(self):
        cases = {
            US_SONNET_45: 200_000,
            "global.anthropic.claude-sonnet-4-5-20250929-v1:0": 200_000,
            "jp.anthropic.claude-sonnet-4-5-20250929-v1:0": 200_000,
            "anthropic.claude-3-haiku-20240307-v1:0": 200_000,
            "bedrock/" + US_SONNET_45: 200_000,
            "aws/bedrock/anthropic.claude-3-opus-20240229-v1:0": 200_000,
            "arn:aws:bedrock:us-east-1:123456789012:inference-profile/" + US_SONNET_45: 200_000,
            "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-haiku-20240307-v1:0":
                200_000,
            "anthropic/claude-sonnet-4-5": 200_000,
            "openai/gpt-4o": 128_000,
            "models/gemini-2.0-flash-001": 1_048_576,
        }
        for model, limit in cases.items():
            with self.subTest(model=model):
                self.assertEqual(_resolve_limit(model), limit)

    def test_dated_name_uses_the_longest_key(self):
        self.assertEqual(_resolve_limit("o1-mini-2024-09-12"), 128_000)

    def test_name_with_no_key_of_its_own_gets_no_limit(self):
        for model in (
            "gpt-4.1", "gpt-4.1-2025-04-14", "claude", "claude-3", "claude-opus-47",
            "gpt", "o", "gemini",
            "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/a1b2c3d4e5f6",
        ):
            with self.subTest(model=model):
                self.assertIsNone(_resolve_limit(model))

    def test_claude_name_with_a_dot_gets_the_limit_of_its_dash_form(self):
        for model in (
            "claude-sonnet-4.5", "claude-opus-4.1", "claude-opus-4.7",
            "anthropic/claude-opus-4.7", "us.anthropic.claude-sonnet-4.5-v1:0",
        ):
            with self.subTest(model=model):
                self.assertEqual(_resolve_limit(model), 200_000)

    def test_names_that_keep_their_limit(self):
        cases = {
            "gpt-4-turbo-2024-04-09": 128_000,
            "gpt-4o-mini-2024-07-18": 128_000,
            "gpt-4-0613": 8_192,
            "o3-mini": 200_000,
            "o1-preview": 200_000,
            "gemini-2.0-flash-lite": 1_048_576,
            "llama-3.1-70b-instruct": 131_072,
            "claude-opus-4-5-20251101": 200_000,
            "claude-sonnet-4-6": 200_000,
        }
        for model, limit in cases.items():
            with self.subTest(model=model):
                self.assertEqual(_resolve_limit(model), limit)


class TestReportLimitByCostNameRule(unittest.TestCase):
    """Requests of 10,000 then 150,000 input tokens, as in the bug report."""

    def _report(self, model):
        store, sid = _make_store_with_llm_events([10_000, 150_000], model=model)
        self.addCleanup(shutil.rmtree, store.base_dir, True)
        report = analyse_token_budget(store, sid)
        return report, _render_budget(report)

    def test_bedrock_session_gets_its_limit(self):
        report, text = self._report(US_SONNET_45)
        self.assertEqual(report.context_limit, 200_000)
        self.assertIn("Context limit: 200,000 tokens", text)
        self.assertIn("Current: 150,000 tokens  (75.0% of limit)", text)

    def test_session_of_a_claude_name_with_a_dot_gets_its_limit(self):
        for model in ("claude-sonnet-4.5", "claude-opus-4.7"):
            with self.subTest(model=model):
                report, text = self._report(model)
                self.assertEqual(report.context_limit, 200_000)
                self.assertIn("Context limit: 200,000 tokens", text)
                self.assertIn("Current: 150,000 tokens  (75.0% of limit)", text)

    def test_ca_and_in_sessions_get_their_limit_from_the_cli(self):
        for geography in ("ca", "in"):
            with self.subTest(geography=geography):
                store, sid = _make_store_with_llm_events(
                    [10_000, 150_000],
                    model=f"{geography}.anthropic.claude-sonnet-4-5-20250929-v1:0",
                )
                self.addCleanup(shutil.rmtree, store.base_dir, True)
                result = _run_cli(store.base_dir, "token-budget", sid)
                self.assertEqual((result.returncode, result.stderr), (0, ""))
                self.assertIn("Context limit: 200,000 tokens", result.stdout)
                self.assertIn("Current: 150,000 tokens  (75.0% of limit)", result.stdout)

    def test_session_with_no_known_limit_prints_unknown(self):
        report, text = self._report("gpt-4.1-2025-04-14")
        self.assertIsNone(report.context_limit)
        self.assertEqual([r.pct_used for r in report.requests], [None, None])
        self.assertIn("Context limit: unknown tokens", text)
        self.assertIn("Current: 150,000 tokens  (n/a of limit)", text)
        self.assertNotIn("\u2190 warning", text)
        self.assertNotIn("Headroom", text)

    def test_dated_o1_mini_session_uses_the_o1_mini_limit(self):
        report, text = self._report("o1-mini-2024-09-12")
        self.assertEqual(report.context_limit, 128_000)
        self.assertIn("Context limit: 128,000 tokens", text)
        self.assertIn("117.2%  \u2190 warning", text)
        self.assertEqual(text.count("\u2190 warning"), 1)


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

    def test_no_fire_for_a_model_with_no_known_limit(self):
        # gpt-4.1 has no limit in the table. gpt-4's 8,192 is not its limit.
        watcher = TokenBudgetWatcher(threshold=0.9)
        self.assertIsNone(watcher.update(_request(10_000, "gpt-4.1-2025-04-14")))
        self.assertIsNone(watcher.context_limit)

    def test_fires_for_a_bedrock_inference_profile(self):
        for geography in ("us", "ca", "in"):
            with self.subTest(geography=geography):
                watcher = TokenBudgetWatcher(threshold=0.9)
                model = f"{geography}.anthropic.claude-sonnet-4-5-20250929-v1:0"
                result = watcher.update(_request(190_000, model))
                self.assertIsNotNone(result)
                self.assertIn("190,000 / 200,000 tokens (95.0%)", result)

    def test_fires_for_a_claude_name_with_a_dot_in_its_version(self):
        for model in ("claude-sonnet-4.5", "claude-opus-4.1", "anthropic/claude-opus-4.7"):
            with self.subTest(model=model):
                watcher = TokenBudgetWatcher(threshold=0.9)
                result = watcher.update(_request(190_000, model))
                self.assertEqual(watcher.context_limit, 200_000)
                self.assertIsNotNone(result)
                self.assertIn("190,000 / 200,000 tokens (95.0%)", result)

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
