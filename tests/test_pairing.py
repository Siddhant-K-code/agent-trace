"""Tests for the shared request/response pairing rules."""

import random
import unittest

from agent_trace.cost import _session_model_usage
from agent_trace.langfuse_export import (
    _events_to_langfuse_observations,
    _iso,
    langfuse_data_indexes,
)
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.otlp import (
    _to_span_id,
    _ts_to_nanos,
    otlp_data_indexes,
    otlp_genai_data_indexes,
    session_to_otlp,
    session_to_otlp_genai,
)
from agent_trace.pairing import (
    FALLBACK_POLICIES,
    LLM_FALLBACK,
    NEWEST_SAME_MODEL,
    NO_FALLBACK,
    TOOL_FALLBACK,
    Pairer,
    event_model,
    llm_pairer,
    pair_events,
    pair_llm_events,
    pair_tool_events,
    tool_pairer,
)


def _event(event_type, event_id, parent_id="", timestamp=0.0, **data):
    return TraceEvent(
        event_type=event_type,
        event_id=event_id,
        parent_id=parent_id,
        timestamp=timestamp,
        session_id="s",
        data=data,
        tenant_id="",
    )


def _req(event_id, model="", **data):
    if model:
        data["model"] = model
    return _event(EventType.LLM_REQUEST, event_id, **data)


def _resp(event_id, parent_id="", model="", **data):
    if model:
        data["model"] = model
    return _event(EventType.LLM_RESPONSE, event_id, parent_id, **data)


class EventModelTest(unittest.TestCase):
    def test_reads_model_keys_in_order(self):
        self.assertEqual(event_model({"model": " gpt-4o "}), "gpt-4o")
        self.assertEqual(event_model({"model_id": "m1"}), "m1")
        self.assertEqual(event_model({"modelId": "m2"}), "m2")
        self.assertEqual(event_model({"model": "", "model_id": "m1"}), "m1")

    def test_missing_or_non_object_data_names_no_model(self):
        self.assertEqual(event_model({}), "")
        self.assertEqual(event_model([]), "")
        self.assertEqual(event_model("text"), "")
        self.assertEqual(event_model(None), "")


class PairerTest(unittest.TestCase):
    def test_channel_policies(self):
        self.assertEqual(TOOL_FALLBACK, NO_FALLBACK)
        self.assertEqual(LLM_FALLBACK, NEWEST_SAME_MODEL)
        self.assertEqual(tool_pairer().fallback, NO_FALLBACK)
        self.assertEqual(llm_pairer().fallback, NEWEST_SAME_MODEL)
        self.assertEqual(FALLBACK_POLICIES, {NO_FALLBACK, NEWEST_SAME_MODEL})

    def test_unknown_policy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown fallback policy"):
            Pairer("oldest")

    def test_item_defaults_to_the_request(self):
        pairer = Pairer()
        request = _req("r1")
        pairer.add(request)
        self.assertIs(pairer.match(_resp("x", "r1")), request)

    def test_parent_id_wins_over_fallback(self):
        for policy in FALLBACK_POLICIES:
            with self.subTest(policy=policy):
                pairer = Pairer(policy)
                pairer.add(_req("r1", "gpt-4o"), "first")
                pairer.add(_req("r2", "gpt-4o"), "second")
                self.assertEqual(pairer.match(_resp("x", "r1", "gpt-4o")), "first")
                self.assertEqual(pairer.unmatched(), ["second"])

    def test_parent_id_ignores_model(self):
        pairer = llm_pairer()
        pairer.add(_req("r1", "gpt-4o"), "first")
        self.assertEqual(pairer.match(_resp("x", "r1", "claude-opus-4-6")), "first")

    def test_no_fallback_leaves_parentless_response_unpaired(self):
        pairer = Pairer(NO_FALLBACK)
        pairer.add(_req("r1"), "first")
        self.assertIsNone(pairer.match(_resp("x")))
        self.assertIsNone(pairer.match(_resp("y", "unknown")))
        self.assertEqual(pairer.unmatched(), ["first"])

    def test_newest_same_model_takes_newest_compatible_request(self):
        pairer = llm_pairer()
        pairer.add(_req("r1", "gpt-4o"), "gpt-old")
        pairer.add(_req("r2", "gpt-4o"), "gpt-new")
        pairer.add(_req("r3", "claude-opus-4-6"), "claude")
        self.assertEqual(pairer.match(_resp("x", model="gpt-4o")), "gpt-new")
        self.assertEqual(pairer.match(_resp("y", model="gpt-4o")), "gpt-old")
        self.assertIsNone(pairer.match(_resp("z", model="gpt-4o")))
        self.assertEqual(pairer.unmatched(), ["claude"])

    def test_missing_model_on_either_side_is_compatible(self):
        pairer = llm_pairer()
        pairer.add(_req("r1", "gpt-4o"), "gpt")
        pairer.add(_req("r2"), "none")
        self.assertEqual(pairer.match(_resp("x", model="claude-opus-4-6")), "none")
        self.assertEqual(pairer.match(_resp("y")), "gpt")

    def test_unknown_parent_id_uses_fallback(self):
        pairer = llm_pairer()
        pairer.add(_req("r1"), "first")
        self.assertEqual(pairer.match(_resp("x", "not-a-request")), "first")

    def test_duplicate_response_does_not_take_another_request(self):
        pairer = llm_pairer()
        pairer.add(_req("r1"), "first")
        pairer.add(_req("r2"), "second")
        self.assertEqual(pairer.match(_resp("x", "r1")), "first")
        self.assertIsNone(pairer.match(_resp("y", "r1")))
        self.assertEqual(pairer.unmatched(), ["second"])

    def test_failed_call_without_response_does_not_shift_later_pairs(self):
        # An SDK integration records a failed call as an error and no
        # response, then a successful call with a parentless response.
        pairer = llm_pairer()
        pairer.add(_req("r1", "gpt-4o"), "failed")
        pairer.add(_req("r2", "gpt-4o"), "ok-1")
        self.assertEqual(pairer.match(_resp("x", model="gpt-4o")), "ok-1")
        pairer.add(_req("r3", "gpt-4o"), "ok-2")
        self.assertEqual(pairer.match(_resp("y", model="gpt-4o")), "ok-2")
        self.assertEqual(pairer.unmatched(), ["failed"])

    def test_same_id_replaces_pending_request_and_keeps_its_place(self):
        pairer = llm_pairer()
        pairer.add(_req("a"), "a-1")
        pairer.add(_req("b"), "b")
        pairer.add(_req("a"), "a-2")
        self.assertEqual(pairer.unmatched(), ["a-2", "b"])
        self.assertEqual(pairer.match(_resp("x")), "b")
        self.assertEqual(pairer.match(_resp("y", "a")), "a-2")

    def test_same_id_after_pairing_is_a_new_request(self):
        pairer = llm_pairer()
        pairer.add(_req("a"), "a-1")
        self.assertEqual(pairer.match(_resp("x", "a")), "a-1")
        pairer.add(_req("a"), "a-2")
        self.assertEqual(pairer.match(_resp("y", "a")), "a-2")

    def test_empty_ids_never_replace_each_other(self):
        pairer = llm_pairer()
        pairer.add(_req(""), "one")
        pairer.add(_req(""), "two")
        self.assertEqual(pairer.unmatched(), ["one", "two"])
        self.assertIsNone(tool_pairer().match(_resp("x", "")))
        self.assertEqual(pairer.match(_resp("x")), "two")

    def test_unmatched_is_in_arrival_order(self):
        pairer = tool_pairer()
        for event_id in ("c1", "c2", "c3"):
            pairer.add(_event(EventType.TOOL_CALL, event_id), event_id)
        pairer.match(_event(EventType.TOOL_RESULT, "x", "c2"))
        self.assertEqual(pairer.unmatched(), ["c1", "c3"])


class PairEventsTest(unittest.TestCase):
    def test_tool_errors_close_calls_and_results_need_parent(self):
        events = [
            _event(EventType.TOOL_CALL, "c1"),
            _event(EventType.TOOL_CALL, "c2"),
            _event(EventType.ERROR, "e1", "c1"),
            _event(EventType.TOOL_RESULT, "t1"),
            _event(EventType.TOOL_RESULT, "t2", "c1"),
            _event(EventType.TOOL_CALL, "c3"),
        ]
        pairing = pair_tool_events(events)
        self.assertEqual(pairing.pairs, {2: 0})
        self.assertEqual(pairing.unpaired_responses, [3, 4])
        self.assertEqual(pairing.unmatched_requests, [1, 5])

    def test_llm_channel_ignores_tool_events(self):
        events = [
            _event(EventType.TOOL_CALL, "c1"),
            _req("r1"),
            _resp("x", "c1"),
            _event(EventType.TOOL_RESULT, "t1", "r1"),
        ]
        pairing = pair_llm_events(events)
        self.assertEqual(pairing.pairs, {2: 1})
        self.assertEqual(pairing.unmatched_requests, [])
        self.assertEqual(pair_tool_events(events).pairs, {})

    def test_custom_types_and_policy(self):
        events = [
            _req("r1", "gpt-4o"),
            _event(EventType.ASSISTANT_RESPONSE, "a1", model="gpt-4o"),
        ]
        pairing = pair_events(
            events,
            {EventType.LLM_REQUEST},
            {EventType.ASSISTANT_RESPONSE},
            NEWEST_SAME_MODEL,
        )
        self.assertEqual(pairing.pairs, {1: 0})


# ---------------------------------------------------------------------------
# Exporters agree with each other and with parent_id links
# ---------------------------------------------------------------------------

_META = SessionMeta(session_id="s", started_at=1000.0, ended_at=2000.0, agent_name="claude")
_MODELS = ("gpt-4o", "claude-opus-4-6")


def _well_formed_trace(seed):
    """Return a random trace where every response names a pending request."""
    rng = random.Random(seed)
    events = []
    calls, requests = [], []
    clock = [1000.0]

    def add(event_type, parent_id="", **data):
        clock[0] += 1
        event = _event(event_type, f"e{len(events) + 1}", parent_id, clock[0], **data)
        events.append(event)
        return event

    for _ in range(rng.randint(2, 16)):
        roll = rng.random()
        if roll < 0.25:
            calls.append(add(EventType.TOOL_CALL, tool_name="Bash", arguments={"n": len(events)}))
        elif roll < 0.45 and calls:
            call = calls.pop(rng.randrange(len(calls)))
            kind = rng.choice([EventType.TOOL_RESULT, EventType.ERROR])
            add(kind, call.event_id, result=f"r{len(events)}", error="bad")
        elif roll < 0.7:
            requests.append(add(
                EventType.LLM_REQUEST,
                model=rng.choice(_MODELS),
                input_tokens=rng.randint(1, 50),
                prompt=f"p{len(events)}",
            ))
        elif requests:
            request = requests.pop(rng.randrange(len(requests)))
            data = {"output_tokens": rng.randint(1, 50), "text": "t"}
            if rng.random() < 0.5:
                data["model"] = request.data["model"]
            if rng.random() < 0.5:
                data["input_tokens"] = rng.randint(1, 50)
            add(EventType.LLM_RESPONSE, request.event_id, **data)
    return events


def _spans(payload):
    return payload["resourceSpans"][0]["scopeSpans"][0]["spans"]


class WellFormedTracesFollowParentIdTest(unittest.TestCase):
    """When every response names a pending request, every exporter pairs by parent_id."""

    SEEDS = range(300)

    def test_otlp_genai_spans_start_at_the_named_request(self):
        for seed in self.SEEDS:
            events = _well_formed_trace(seed)
            by_id = {event.event_id: event for event in events}
            spans = {span["spanId"]: span for span in _spans(session_to_otlp_genai(_META, events))[1:]}
            for event in events:
                if event.event_type in (EventType.LLM_RESPONSE, EventType.TOOL_RESULT, EventType.ERROR):
                    with self.subTest(seed=seed, event=event.event_id):
                        span = spans[_to_span_id(event.parent_id)]
                        start = _ts_to_nanos(by_id[event.parent_id].timestamp)
                        self.assertEqual(span["startTimeUnixNano"], start)

    def test_plain_otlp_tool_spans_start_at_the_named_call(self):
        for seed in self.SEEDS:
            events = _well_formed_trace(seed)
            by_id = {event.event_id: event for event in events}
            spans = {span["spanId"]: span for span in _spans(session_to_otlp(_META, events))[1:]}
            for event in events:
                if event.event_type in (EventType.TOOL_RESULT, EventType.ERROR):
                    with self.subTest(seed=seed, event=event.event_id):
                        span = spans[_to_span_id(event.parent_id)]
                        start = _ts_to_nanos(by_id[event.parent_id].timestamp)
                        self.assertEqual(span["startTimeUnixNano"], start)

    def test_langfuse_observations_use_the_named_request(self):
        for seed in self.SEEDS:
            events = _well_formed_trace(seed)
            by_id = {event.event_id: event for event in events}
            observations = {o["id"]: o for o in _events_to_langfuse_observations("s", events)}
            for event in events:
                if event.event_type in (EventType.LLM_RESPONSE, EventType.TOOL_RESULT):
                    with self.subTest(seed=seed, event=event.event_id):
                        parent = by_id[event.parent_id]
                        observation = observations[event.event_id]
                        self.assertEqual(observation["startTime"], _iso(parent.timestamp))
                        if event.event_type == EventType.LLM_RESPONSE:
                            self.assertEqual(observation["model"], parent.data["model"])
                            self.assertEqual(observation["input"], parent.data["prompt"])

    def test_cost_replaces_request_usage_with_the_named_response(self):
        for seed in self.SEEDS:
            events = _well_formed_trace(seed)
            expected = {}
            usage = {}
            for event in events:
                if event.event_type == EventType.LLM_REQUEST:
                    usage[event.event_id] = [event.data["model"], event.data["input_tokens"], 0]
                elif event.event_type == EventType.LLM_RESPONSE:
                    sample = usage[event.parent_id]
                    sample[1] = event.data.get("input_tokens", sample[1])
                    sample[2] = event.data["output_tokens"]
            for model, inp, out in usage.values():
                totals = expected.setdefault(model, [0, 0])
                totals[0] += inp
                totals[1] += out
            with self.subTest(seed=seed):
                self.assertEqual(
                    _session_model_usage(events),
                    {model: tuple(values) for model, values in expected.items()},
                )

    def test_data_indexes_match_parent_links(self):
        for seed in self.SEEDS:
            events = _well_formed_trace(seed)
            index_of = {event.event_id: index for index, event in enumerate(events)}
            paired = {
                index_of[event.parent_id]
                for event in events
                if event.event_type in (EventType.LLM_RESPONSE, EventType.TOOL_RESULT)
            }
            responses = {
                index
                for index, event in enumerate(events)
                if event.event_type in (EventType.LLM_RESPONSE, EventType.TOOL_RESULT)
            }
            with self.subTest(seed=seed):
                self.assertEqual(set(langfuse_data_indexes(events)), paired | responses)
                genai = set(otlp_genai_data_indexes(events))
                plain = set(otlp_data_indexes(events))
                self.assertTrue(paired <= genai)
                self.assertTrue(paired <= plain)


class ExportersAgreeOnFallbackTest(unittest.TestCase):
    """A parentless response pairs with the same request in every exporter."""

    def setUp(self):
        self.events = [
            _event(EventType.LLM_REQUEST, "failed", "", 1.0, model="gpt-4o", input_tokens=10, prompt="a"),
            _event(EventType.ERROR, "err", "", 2.0, message="rate limited"),
            _event(EventType.LLM_REQUEST, "ok", "", 3.0, model="gpt-4o", input_tokens=20, prompt="b"),
            _event(EventType.LLM_RESPONSE, "resp", "", 4.0, model="gpt-4o", output_tokens=5, input_tokens=30),
        ]

    def test_otlp_genai(self):
        spans = {s["spanId"]: s for s in _spans(session_to_otlp_genai(_META, self.events))[1:]}
        self.assertEqual(spans[_to_span_id("ok")]["startTimeUnixNano"], _ts_to_nanos(3.0))
        self.assertEqual(spans[_to_span_id("failed")]["status"], {"code": 0})

    def test_langfuse(self):
        (observation,) = _events_to_langfuse_observations("s", self.events)
        self.assertEqual(observation["startTime"], _iso(3.0))
        self.assertEqual(observation["input"], "b")
        self.assertEqual(observation["model"], "gpt-4o")
        self.assertEqual(sorted(langfuse_data_indexes(self.events)), [2, 3])

    def test_cost(self):
        self.assertEqual(_session_model_usage(self.events), {"gpt-4o": (10 + 30, 5)})


if __name__ == "__main__":
    unittest.main()
