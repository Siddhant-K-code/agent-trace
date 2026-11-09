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
    _dollars,
    _estimate_tokens,
    _event_tokens,
    estimate_cost,
    format_cost,
    session_event_tokens,
    session_records_usage,
)
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
        self.assertAlmostEqual(result.total_cost, _dollars(6000, 60, DEFAULT_MODEL), places=10)

    def test_other_commands_agree_on_recorded_usage(self):
        from agent_trace.compliance import _event_costs
        from agent_trace.replay import replay_to_html
        from agent_trace.timeline import build_timeline
        from agent_trace.watch import WatcherConfig, WatchState, check_event

        events = _codex_events()
        store, tmp = _make_store(events)
        self.addCleanup(tmp.cleanup)
        expected = _dollars(6000, 60, "sonnet")

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


if __name__ == "__main__":
    unittest.main()
