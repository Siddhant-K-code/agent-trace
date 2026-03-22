"""Tests for session cost estimation."""

import io
import tempfile
import unittest

from agent_trace.cost import (
    CostResult,
    PhaseCost,
    PRICING,
    _dollars,
    _estimate_tokens,
    _event_tokens,
    estimate_cost,
    format_cost,
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
