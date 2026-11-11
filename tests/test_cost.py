"""Tests for session cost estimation."""

import io
import json
import re
import tempfile
import unittest

from agent_trace.cost import (
    CostResult,
    DEFAULT_MODEL,
    PhaseCost,
    PRICING,
    TokenCounter,
    TokenUsage,
    _dollars,
    _estimate_tokens,
    _event_tokens,
    _usage_dollars,
    cost_inputs_captured,
    estimate_cost,
    format_cost,
    session_event_tokens,
    session_cost,
    session_event_usage,
    session_records_usage,
)
from agent_trace.event_data import NonObjectEventDataError
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore


def _make_event(event_type: EventType, ts: float, session_id: str = "s1", **data) -> TraceEvent:
    return TraceEvent(event_type=event_type, timestamp=ts, session_id=session_id, data=data)


def _make_store(events: list[TraceEvent], session_id: str = "s1") -> tuple[TraceStore, tempfile.TemporaryDirectory]:
    tmp = tempfile.TemporaryDirectory()
    store = TraceStore(tmp.name)
    meta = SessionMeta(
        session_id=session_id,
        started_at=events[0].timestamp,
        total_duration_ms=(events[-1].timestamp - events[0].timestamp) * 1000,
    )
    store.create_session(meta)
    for e in events:
        store.append_event(session_id, e)
    store.update_meta(meta)
    return store, tmp


class TestEstimateTokens(unittest.TestCase):
    def test_empty_string(self):
        self.assertEqual(_estimate_tokens(""), 1)  # min 1

    def test_four_chars_one_token(self):
        self.assertEqual(_estimate_tokens("abcd"), 1)

    def test_hundred_chars(self):
        self.assertEqual(_estimate_tokens("a" * 100), 25)

    def test_longer_text(self):
        result = _estimate_tokens("x" * 400)
        self.assertEqual(result, 100)


class TestEventTokens(unittest.TestCase):
    def test_user_prompt_is_input(self):
        e = _make_event(EventType.USER_PROMPT, 0.0, prompt="hello world")
        inp, out = _event_tokens(e)
        self.assertGreater(inp, 0)
        self.assertEqual(out, 0)

    def test_assistant_response_is_output(self):
        e = _make_event(EventType.ASSISTANT_RESPONSE, 0.0, text="here is my answer")
        inp, out = _event_tokens(e)
        self.assertEqual(inp, 0)
        self.assertGreater(out, 0)

    def test_tool_call_is_input(self):
        e = _make_event(EventType.TOOL_CALL, 0.0, tool_name="Bash",
                        arguments={"command": "ls -la"})
        inp, out = _event_tokens(e)
        self.assertGreater(inp, 0)
        self.assertEqual(out, 0)

    def test_tool_result_is_output(self):
        e = _make_event(EventType.TOOL_RESULT, 0.0, result="file1.py\nfile2.py")
        inp, out = _event_tokens(e)
        self.assertEqual(inp, 0)
        self.assertGreater(out, 0)

    def test_session_end_zero_tokens(self):
        e = _make_event(EventType.SESSION_END, 0.0)
        inp, out = _event_tokens(e)
        self.assertEqual(inp, 0)
        self.assertEqual(out, 0)


class TestDollars(unittest.TestCase):
    def test_zero_tokens_zero_cost(self):
        self.assertEqual(_dollars(0, 0, "sonnet"), 0.0)

    def test_sonnet_pricing(self):
        # 1M input tokens at $3.00/M = $3.00
        cost = _dollars(1_000_000, 0, "sonnet")
        self.assertAlmostEqual(cost, 3.00, places=4)

    def test_opus_more_expensive_than_sonnet(self):
        cost_sonnet = _dollars(100_000, 100_000, "sonnet")
        cost_opus = _dollars(100_000, 100_000, "opus")
        self.assertGreater(cost_opus, cost_sonnet)

    def test_haiku_cheapest(self):
        cost_haiku = _dollars(100_000, 100_000, "haiku")
        cost_sonnet = _dollars(100_000, 100_000, "sonnet")
        self.assertLess(cost_haiku, cost_sonnet)

    def test_unknown_model_falls_back_to_sonnet(self):
        cost_unknown = _dollars(100_000, 100_000, "nonexistent")
        cost_sonnet = _dollars(100_000, 100_000, "sonnet")
        self.assertAlmostEqual(cost_unknown, cost_sonnet, places=6)


class TestEstimateCost(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._store = TraceStore(self._tmp.name)
        events = [
            _make_event(EventType.USER_PROMPT, 0.0, session_id="sess1",
                        prompt="run the tests"),
            _make_event(EventType.TOOL_CALL, 1.0, session_id="sess1",
                        tool_name="Bash", arguments={"command": "pytest"}),
            _make_event(EventType.TOOL_RESULT, 2.0, session_id="sess1",
                        result="5 passed"),
            _make_event(EventType.ASSISTANT_RESPONSE, 3.0, session_id="sess1",
                        text="All tests passed."),
            _make_event(EventType.SESSION_END, 4.0, session_id="sess1"),
        ]
        meta = SessionMeta(session_id="sess1", started_at=0.0, total_duration_ms=4000)
        self._store.create_session(meta)
        for e in events:
            self._store.append_event("sess1", e)
        self._store.update_meta(meta)

    def tearDown(self):
        self._tmp.cleanup()

    def test_returns_cost_result(self):
        result = estimate_cost(self._store, "sess1")
        self.assertIsInstance(result, CostResult)

    def test_total_cost_positive(self):
        result = estimate_cost(self._store, "sess1")
        self.assertGreater(result.total_cost, 0)

    def test_phase_costs_sum_to_total(self):
        result = estimate_cost(self._store, "sess1")
        phase_sum = sum(pc.cost_dollars for pc in result.phase_costs)
        self.assertAlmostEqual(phase_sum, result.total_cost, places=8)

    def test_custom_pricing(self):
        result = estimate_cost(self._store, "sess1", input_price=1.0, output_price=2.0)
        self.assertEqual(result.model, "custom")
        self.assertGreater(result.total_cost, 0)

    def test_wasted_cost_from_failed_phase(self):
        tmp2 = tempfile.TemporaryDirectory()
        store2 = TraceStore(tmp2.name)
        events = [
            _make_event(EventType.USER_PROMPT, 0.0, session_id="sess2", prompt="run tests"),
            _make_event(EventType.TOOL_CALL, 1.0, session_id="sess2",
                        tool_name="Bash", arguments={"command": "pytest"}),
            _make_event(EventType.ERROR, 2.0, session_id="sess2", message="exit 1"),
            _make_event(EventType.SESSION_END, 3.0, session_id="sess2"),
        ]
        meta = SessionMeta(session_id="sess2", started_at=0.0, total_duration_ms=3000)
        store2.create_session(meta)
        for e in events:
            store2.append_event("sess2", e)
        store2.update_meta(meta)
        result = estimate_cost(store2, "sess2")
        tmp2.cleanup()
        self.assertGreater(result.wasted_cost, 0)

    def test_model_selection(self):
        result_haiku = estimate_cost(self._store, "sess1", model="haiku")
        result_opus = estimate_cost(self._store, "sess1", model="opus")
        self.assertLess(result_haiku.total_cost, result_opus.total_cost)


def _model_request(ts: float, input_tokens: int | None, output_tokens: int | None,
                   model: str | None = "gpt-5-codex", session_id: str = "s1") -> TraceEvent:
    """A Codex PostModelRequest event: an llm_response with no llm_request."""
    data = {"hook_event": "post_model_request", "provider": "codex"}
    if model:
        data["model"] = model
    if input_tokens is None:
        data.update(status="failed", error="stream disconnected", token_usage=None)
    else:
        usage = {
            "input_tokens": input_tokens,
            "cached_input_tokens": input_tokens // 2,
            "cache_write_input_tokens": 0,
            "output_tokens": output_tokens,
            "reasoning_output_tokens": output_tokens // 2,
            "total_tokens": input_tokens + output_tokens,
        }
        data.update(
            status="completed",
            token_usage=usage,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )
    return TraceEvent(event_type=EventType.LLM_RESPONSE, timestamp=ts,
                      session_id=session_id, data=data)


def _codex_events(session_id: str = "s1") -> list[TraceEvent]:
    return [
        _make_event(EventType.USER_PROMPT, 0.0, session_id, prompt="list the files"),
        _model_request(1.0, 1200, 10, session_id=session_id),
        _make_event(EventType.TOOL_CALL, 2.0, session_id,
                    tool_name="Bash", arguments={"command": "ls"}),
        _make_event(EventType.TOOL_RESULT, 3.0, session_id, result="README.md src tests"),
        _model_request(4.0, 2300, 20, session_id=session_id),
        _make_event(EventType.ASSISTANT_RESPONSE, 5.0, session_id, text="Three entries."),
        _make_event(EventType.USER_PROMPT, 6.0, session_id, prompt="now count them"),
        _model_request(7.0, None, None, session_id=session_id),
        _model_request(8.0, 2500, 30, model=None, session_id=session_id),
        _make_event(EventType.ASSISTANT_RESPONSE, 9.0, session_id, text="Three."),
    ]


class TestSessionCost(unittest.TestCase):
    """session_cost prices loaded events exactly as `agent-strace cost` does."""

    def _check_matches(self, events):
        store, tmp = _make_store(events)
        self.addCleanup(tmp.cleanup)
        loaded = store.load_events("s1")
        for model in (DEFAULT_MODEL, "opus", "gpt4o", "no-such-model"):
            self.assertEqual(
                session_cost(loaded, model),
                estimate_cost(store, "s1", model=model).total_cost,
            )
        return estimate_cost(store, "s1")

    def test_matches_recorded_usage_with_cache_reads(self):
        result = self._check_matches(_codex_events())
        self.assertGreater(result.cache_read_tokens, 0)

    def test_matches_size_estimate(self):
        result = self._check_matches([
            _make_event(EventType.USER_PROMPT, 0.0, prompt="fix the bug"),
            _make_event(EventType.TOOL_CALL, 1.0, tool_name="Read",
                        arguments={"file_path": "a.py"}),
            _make_event(EventType.TOOL_RESULT, 2.0, result="x" * 400),
            _make_event(EventType.USER_PROMPT, 3.0, prompt="and test it"),
            _make_event(EventType.ASSISTANT_RESPONSE, 4.0, text="Done."),
        ])
        self.assertGreater(result.total_cost, 0)
        self.assertEqual(result.cache_read_tokens, 0)

    def test_empty_session_costs_nothing(self):
        self.assertEqual(session_cost([]), 0.0)

    def test_refuses_what_the_cost_command_refuses(self):
        for index in (0, 2):  # a user_prompt and a tool_call
            events = _codex_events()
            events[index].data = ["not", "an", "object"]
            with self.assertRaises(NonObjectEventDataError) as caught:
                session_cost(events, session_id="s1")
            self.assertIsInstance(caught.exception, ValueError)
            self.assertIn("s1", str(caught.exception))

    def test_prices_non_object_data_the_cost_command_does_not_read(self):
        events = _codex_events()
        events[3].data = ["README.md"]  # a tool_result
        self._check_matches(events)


class TestCostInputsCaptured(unittest.TestCase):
    def _ev(self, kind: EventType, **data) -> TraceEvent:
        return TraceEvent(event_type=kind, timestamp=1.0, session_id="s", data=data)

    def test_hook_session_without_model_or_usage(self):
        events = [
            self._ev(EventType.SESSION_START, provider="copilot", model=""),
            self._ev(EventType.TOOL_CALL, tool_name="bash", arguments={"command": "ls"}),
            self._ev(EventType.TOOL_RESULT, tool_name="bash", result="ok"),
        ]
        self.assertFalse(cost_inputs_captured(events))

    def test_recorded_model(self):
        events = [self._ev(EventType.SESSION_START, model="claude-sonnet-4-5")]
        self.assertTrue(cost_inputs_captured(events))

    def test_recorded_token_usage(self):
        events = [self._ev(
            EventType.LLM_RESPONSE, usage={"input_tokens": 10, "output_tokens": 2},
        )]
        self.assertTrue(cost_inputs_captured(events))


class TestRecordedUsage(unittest.TestCase):
    """An llm_response with no llm_request carries its request's exact usage."""

    def test_unpaired_responses_count_their_recorded_usage_only(self):
        events = _codex_events()
        self.assertTrue(session_records_usage(events))
        self.assertEqual(session_event_tokens(events), [
            (0, 0), (1200, 10), (0, 0), (0, 0), (2300, 20),
            (0, 0), (0, 0), (0, 0), (2500, 30), (0, 0),
        ])

    def test_usage_nested_in_token_usage_is_read(self):
        event = _model_request(0.0, 700, 9)
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            del event.data[key]
        self.assertEqual(session_event_tokens([event]), [(700, 9)])

    def test_paired_trace_keeps_size_estimates(self):
        request = _make_event(EventType.LLM_REQUEST, 0.0, model="gpt-4o",
                              messages=[{"role": "user", "content": "hi"}])
        response = TraceEvent(
            event_type=EventType.LLM_RESPONSE, timestamp=1.0, session_id="s1",
            parent_id=request.event_id,
            data={"model": "gpt-4o", "input_tokens": 900, "output_tokens": 40},
        )
        events = [
            _make_event(EventType.USER_PROMPT, 0.0, prompt="hi"),
            request,
            response,
            _make_event(EventType.ASSISTANT_RESPONSE, 2.0, text="Hello."),
        ]
        self.assertFalse(session_records_usage(events))
        self.assertEqual(session_event_tokens(events), [_event_tokens(e) for e in events])

    def test_unpaired_responses_without_counts_keep_size_estimates(self):
        events = [
            _make_event(EventType.USER_PROMPT, 0.0, prompt="hi"),
            _model_request(1.0, None, None),
            _make_event(EventType.ASSISTANT_RESPONSE, 2.0, text="Hello."),
        ]
        self.assertFalse(session_records_usage(events))
        self.assertEqual(session_event_tokens(events), [_event_tokens(e) for e in events])

    def test_streaming_counter_switches_at_first_usage_response(self):
        events = _codex_events()
        counter = TokenCounter()
        counts = [counter.count(event) for event in events]
        self.assertEqual(counts[0], _event_tokens(events[0]))
        self.assertEqual(counts[1:], session_event_tokens(events)[1:])

    def test_estimate_cost_prices_recorded_usage_once(self):
        store, tmp = _make_store(_codex_events())
        self.addCleanup(tmp.cleanup)
        result = estimate_cost(store, "s1")
        self.assertEqual((result.input_tokens, result.output_tokens), (6000, 60))
        self.assertEqual(
            [(p.input_tokens, p.output_tokens) for p in result.phase_costs],
            [(3500, 30), (2500, 30)],
        )
        # Half of each request's input was a cache read.
        self.assertEqual((result.cache_read_tokens, result.cache_write_tokens), (3000, 0))
        self.assertEqual(
            [(p.cache_read_tokens, p.cache_write_tokens) for p in result.phase_costs],
            [(1750, 0), (1250, 0)],
        )
        usage = TokenUsage(6000, 60, cache_read_tokens=3000)
        self.assertAlmostEqual(result.total_cost, _usage_dollars(usage, DEFAULT_MODEL), places=10)
        self.assertLess(result.total_cost, _dollars(6000, 60, DEFAULT_MODEL))

    def test_other_commands_agree_on_recorded_usage(self):
        from agent_trace.compliance import _event_costs
        from agent_trace.replay import replay_to_html
        from agent_trace.timeline import build_timeline
        from agent_trace.watch import WatcherConfig, WatchState, check_event

        events = _codex_events()
        store, tmp = _make_store(events)
        self.addCleanup(tmp.cleanup)
        expected = _usage_dollars(TokenUsage(6000, 60, cache_read_tokens=3000), "sonnet")

        self.assertAlmostEqual(build_timeline(store, "s1").total_cost, expected, places=10)
        self.assertAlmostEqual(sum(_event_costs(events)), expected, places=10)
        html = replay_to_html(store, "s1")
        costs = json.loads(re.search(r"const COSTS = (\[.*?\]);", html).group(1))
        self.assertAlmostEqual(costs[-1], expected, places=6)

        # watch streams, so events before the first usage response keep
        # their size estimate.
        state, config = WatchState(), WatcherConfig()
        for event in events:
            check_event(event, config, state)
        prompt = _dollars(*_event_tokens(events[0]), "sonnet")
        self.assertAlmostEqual(state.estimated_cost, expected + prompt, places=10)


class TestFormatCost(unittest.TestCase):
    def test_output_contains_session_id(self):
        result = CostResult(
            session_id="abc123",
            model="sonnet",
            total_cost=1.2345,
            input_tokens=10000,
            output_tokens=5000,
            phase_costs=[
                PhaseCost(phase_index=1, phase_name="Setup",
                          input_tokens=10000, output_tokens=5000,
                          cost_dollars=1.2345, failed=False)
            ],
            wasted_cost=0.0,
        )
        buf = io.StringIO()
        format_cost(result, out=buf)
        output = buf.getvalue()
        self.assertIn("abc123", output)
        self.assertIn("1.2345", output)
        self.assertIn("sonnet", output)

    def test_wasted_shown_when_nonzero(self):
        result = CostResult(
            session_id="xyz",
            model="sonnet",
            total_cost=2.0,
            input_tokens=50000,
            output_tokens=20000,
            phase_costs=[
                PhaseCost(phase_index=1, phase_name="Failed run",
                          input_tokens=50000, output_tokens=20000,
                          cost_dollars=2.0, failed=True)
            ],
            wasted_cost=2.0,
        )
        buf = io.StringIO()
        format_cost(result, out=buf)
        output = buf.getvalue()
        self.assertIn("Wasted", output)

    def test_wasted_not_shown_when_zero(self):
        result = CostResult(
            session_id="xyz",
            model="sonnet",
            total_cost=1.0,
            input_tokens=10000,
            output_tokens=5000,
            phase_costs=[],
            wasted_cost=0.0,
        )
        buf = io.StringIO()
        format_cost(result, out=buf)
        output = buf.getvalue()
        self.assertNotIn("Wasted", output)


def _unpaired_response(ts: float, session_id: str = "s1", **data) -> TraceEvent:
    return TraceEvent(event_type=EventType.LLM_RESPONSE, timestamp=ts,
                      session_id=session_id, data=data)


class TestCachedTokenCost(unittest.TestCase):
    """estimate_cost and its consumers bill cache reads and writes apart."""

    def test_uncached_usage_costs_exactly_what_input_and_output_did(self):
        for model in ("sonnet", "haiku", "claude-sonnet-4-6", "gpt-4o"):
            with self.subTest(model=model):
                rates = PRICING[model]
                self.assertEqual(
                    _dollars(1234, 567, model),
                    1234 / 1_000_000 * rates["input"] + 567 / 1_000_000 * rates["output"],
                )

    def test_token_usage_keeps_cache_parts_inside_input(self):
        usage = TokenUsage(1000, 10, cache_read_tokens=600, cache_write_tokens=300,
                           cache_write_long_tokens=100)
        self.assertEqual(usage.uncached_input_tokens, 100)
        self.assertEqual(
            usage + usage,
            TokenUsage(2000, 20, cache_read_tokens=1200, cache_write_tokens=600,
                       cache_write_long_tokens=200),
        )

    def test_anthropic_session_includes_cache_tokens_in_input(self):
        store, tmp = _make_store([
            _make_event(EventType.USER_PROMPT, 0.0, prompt="hi"),
            _unpaired_response(1.0, model="claude-sonnet-4-6", input_tokens=100,
                               output_tokens=50, cache_read_input_tokens=8000,
                               cache_creation_input_tokens=1000),
        ])
        self.addCleanup(tmp.cleanup)
        result = estimate_cost(store, "s1", model="sonnet")
        self.assertEqual(
            (result.input_tokens, result.output_tokens,
             result.cache_read_tokens, result.cache_write_tokens),
            (9100, 50, 8000, 1000),
        )
        # sonnet: input 3, cache read 0.30, 5m cache write 3.75, output 15.
        self.assertAlmostEqual(
            result.total_cost, (100 * 3 + 8000 * 0.30 + 1000 * 3.75 + 50 * 15) / 1e6, places=12,
        )
        self.assertEqual(session_event_usage(_codex_events())[1],
                         TokenUsage(1200, 10, cache_read_tokens=600))

    def test_model_without_cache_rates_bills_cache_reads_as_input(self):
        store, tmp = _make_store(_codex_events())
        self.addCleanup(tmp.cleanup)
        result = estimate_cost(store, "s1", model="haiku")
        self.assertEqual(result.cache_read_tokens, 3000)
        self.assertAlmostEqual(result.total_cost, _dollars(6000, 60, "haiku"), places=12)

    def test_custom_prices_bill_cache_reads_at_the_custom_input_price(self):
        store, tmp = _make_store(_codex_events())
        self.addCleanup(tmp.cleanup)
        result = estimate_cost(store, "s1", input_price=1.0, output_price=2.0)
        self.assertEqual(result.cache_read_tokens, 3000)
        self.assertAlmostEqual(result.total_cost, (6000 * 1.0 + 60 * 2.0) / 1e6, places=12)

    def test_format_cost_names_cached_input_only_when_present(self):
        result = CostResult(
            session_id="abc", model="sonnet", total_cost=0.01,
            input_tokens=2030, output_tokens=13, phase_costs=[], wasted_cost=0.0,
            cache_read_tokens=1024, cache_write_tokens=5,
        )
        buf = io.StringIO()
        format_cost(result, out=buf)
        self.assertIn("2,030 input tokens (1,024 cache read, 5 cache write), 13 output tokens",
                      buf.getvalue())

        result.cache_read_tokens = result.cache_write_tokens = 0
        buf = io.StringIO()
        format_cost(result, out=buf)
        self.assertIn("2,030 input tokens, 13 output tokens", buf.getvalue())

    def test_tenant_report_sums_cache_tokens(self):
        from agent_trace.cost import build_tenant_cost_report, format_tenant_cost

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = TraceStore(tmp.name)
        for session_id in ("t1", "t2"):
            store.create_session(SessionMeta(session_id=session_id, tenant_id="acme",
                                             started_at=1000.0))
            for event in _codex_events(session_id):
                event.timestamp += 1000.0
                store.append_event(session_id, event)
        report = build_tenant_cost_report(store, "acme", since="1d", now=2000.0)
        self.assertEqual((report.sessions, report.input_tokens, report.cache_read_tokens),
                         (2, 12_000, 6000))
        buf = io.StringIO()
        format_tenant_cost(report, out=buf)
        self.assertIn("Tokens: 12,000 input (6,000 cache read), 120 output", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
