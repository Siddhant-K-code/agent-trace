"""Tests for the shared request/response pairing rules."""

import io
import json
import random
import tempfile
import threading
import unittest

from agent_trace.cost import _session_model_usage
from agent_trace.langfuse_export import (
    _events_to_langfuse_observations,
    _iso,
    langfuse_data_indexes,
)
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.proxy import MCPProxy
from agent_trace.store import TraceStore
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
    KeyedPairer,
    Pairer,
    event_model,
    hook_call_timestamp,
    link_response,
    llm_pairer,
    match_hook_tool_call,
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

    def test_same_model_beats_newer_request_without_model(self):
        pairer = llm_pairer()
        pairer.add(_req("r1", "gpt-4o"), "gpt")
        pairer.add(_req("r2"), "task")
        self.assertEqual(pairer.match(_resp("x", model="gpt-4o")), "gpt")
        self.assertEqual(pairer.unmatched(), ["task"])

    def test_response_without_model_prefers_request_without_model(self):
        pairer = llm_pairer()
        pairer.add(_req("r1"), "task")
        pairer.add(_req("r2", "gpt-4o"), "failed-gpt")
        self.assertEqual(pairer.match(_resp("x")), "task")
        self.assertEqual(pairer.unmatched(), ["failed-gpt"])

    def test_request_without_model_is_a_last_resort(self):
        pairer = llm_pairer()
        pairer.add(_req("r1"), "task")
        pairer.add(_req("r2", "claude-opus-4-6"), "claude")
        self.assertEqual(pairer.match(_resp("x", model="gpt-4o")), "task")
        self.assertEqual(pairer.unmatched(), ["claude"])

    def test_different_models_never_pair(self):
        pairer = llm_pairer()
        pairer.add(_req("r1", "gpt-4o"), "gpt")
        self.assertIsNone(pairer.match(_resp("x", model="claude-opus-4-6")))
        self.assertEqual(pairer.unmatched(), ["gpt"])

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


class ExportersAgreeOnModelPrecedenceTest(unittest.TestCase):
    """A newer request without a model does not take a same-model response."""

    def setUp(self):
        self.events = [
            _event(EventType.LLM_REQUEST, "gpt", "", 1.0, model="gpt-4o", input_tokens=10, prompt="a"),
            _event(EventType.LLM_REQUEST, "task", "", 2.0, prompt="b"),
            _event(EventType.LLM_RESPONSE, "resp", "", 3.0, model="gpt-4o", output_tokens=5, input_tokens=30),
        ]

    def test_pair_llm_events(self):
        pairing = pair_llm_events(self.events)
        self.assertEqual(pairing.pairs, {2: 0})
        self.assertEqual(pairing.unmatched_requests, [1])

    def test_otlp_genai(self):
        spans = {s["spanId"]: s for s in _spans(session_to_otlp_genai(_META, self.events))[1:]}
        self.assertEqual(spans[_to_span_id("gpt")]["startTimeUnixNano"], _ts_to_nanos(1.0))
        self.assertEqual(spans[_to_span_id("gpt")]["endTimeUnixNano"], _ts_to_nanos(3.0))
        self.assertEqual(spans[_to_span_id("task")]["status"], {"code": 0})

    def test_langfuse(self):
        (observation,) = _events_to_langfuse_observations("s", self.events)
        self.assertEqual(observation["startTime"], _iso(1.0))
        self.assertEqual(observation["input"], "a")
        self.assertEqual(sorted(langfuse_data_indexes(self.events)), [0, 2])

    def test_cost(self):
        self.assertEqual(_session_model_usage(self.events), {"gpt-4o": (30, 5)})


# ---------------------------------------------------------------------------
# Live pairing: proxies and hooks
# ---------------------------------------------------------------------------

class LinkResponseTest(unittest.TestCase):
    def test_sets_parent_and_duration(self):
        response = _event(EventType.TOOL_RESULT, "r", timestamp=12.5)
        link_response(response, "c", 10.0)
        self.assertEqual(response.parent_id, "c")
        self.assertEqual(response.duration_ms, 2500.0)


class KeyedPairerTest(unittest.TestCase):
    def test_pairs_only_the_same_key(self):
        pairer = KeyedPairer()
        pairer.add(1, "a")
        pairer.add("1", "b")
        self.assertIsNone(pairer.match(2))
        self.assertEqual(pairer.match("1"), "b")
        self.assertEqual(pairer.match(1), "a")
        self.assertIsNone(pairer.match(1))
        self.assertEqual(len(pairer), 0)

    def test_none_key_is_never_pending_and_never_pairs(self):
        pairer = KeyedPairer()
        pairer.add(None, "a")
        self.assertEqual(len(pairer), 0)
        self.assertIsNone(pairer.match(None))

    def test_same_key_replaces_and_becomes_newest(self):
        pairer = KeyedPairer()
        pairer.add(1, "old")
        pairer.add(2, "b")
        pairer.add(1, "new")
        self.assertEqual(pairer.pending(), {2: "b", 1: "new"})
        self.assertEqual(pairer.match(1), "new")

    def test_oldest_is_dropped_past_the_cap(self):
        pairer = KeyedPairer(max_pending=2)
        for key in range(4):
            pairer.add(key, key)
        self.assertEqual(list(pairer.pending()), [2, 3])

    def test_cap_must_be_positive(self):
        with self.assertRaises(ValueError):
            KeyedPairer(max_pending=0)

    def test_clear_returns_count(self):
        pairer = KeyedPairer()
        pairer.add(1, "a")
        pairer.add(2, "b")
        self.assertEqual(pairer.clear(), 2)
        self.assertEqual(pairer.pending(), {})

    def test_pending_is_a_copy(self):
        pairer = KeyedPairer()
        pairer.add(1, "a")
        pairer.pending().clear()
        self.assertEqual(pairer.match(1), "a")

    def test_concurrent_adds_and_matches_pair_every_key_once(self):
        pairer = KeyedPairer()
        matched = []
        lock = threading.Lock()

        def worker(base):
            for key in range(base, base + 200):
                pairer.add(key, key)
            for key in range(base, base + 200):
                item = pairer.match(key)
                with lock:
                    matched.append(item)

        threads = [threading.Thread(target=worker, args=(i * 200,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(sorted(matched), list(range(1600)))
        self.assertEqual(len(pairer), 0)


def _hook(tool_name, timestamp, tool_use_id="", agent_id=None):
    info = {"session_id": "s", "tool_name": tool_name, "tool_use_id": tool_use_id,
            "timestamp": timestamp}
    if agent_id:
        info["agent_id"] = agent_id
    return info


class MatchHookToolCallTest(unittest.TestCase):
    def test_tool_use_id_matches_only_its_call(self):
        pending = {
            "a": _hook("Bash", 1.0, "tu-1"),
            "b": _hook("Bash", 2.0, "tu-2"),
        }
        self.assertEqual(match_hook_tool_call(pending, "Bash", "tu-2"), "b")
        self.assertEqual(match_hook_tool_call(pending, "Read", "tu-1"), "a")
        self.assertIsNone(match_hook_tool_call(pending, "Bash", "tu-3"))

    def test_without_tool_use_id_takes_oldest_call_of_the_tool(self):
        pending = {
            "new": _hook("Bash", 3.0),
            "read": _hook("Read", 1.0),
            "old": _hook("Bash", 2.0),
        }
        self.assertEqual(match_hook_tool_call(pending, "Bash"), "old")
        self.assertEqual(match_hook_tool_call(pending, "Read"), "read")
        self.assertIsNone(match_hook_tool_call(pending, "Edit"))

    def test_equal_timestamps_keep_file_order(self):
        pending = {"first": _hook("Bash", 1.0), "second": _hook("Bash", 1.0)}
        self.assertEqual(match_hook_tool_call(pending, "Bash"), "first")

    def test_same_agent_beats_an_older_call_from_another_agent(self):
        pending = {
            "parent": _hook("Bash", 1.0),
            "sub": _hook("Bash", 2.0, agent_id="agent-1"),
        }
        self.assertEqual(match_hook_tool_call(pending, "Bash", agent_id="agent-1"), "sub")
        self.assertEqual(match_hook_tool_call(pending, "Bash"), "parent")
        self.assertEqual(match_hook_tool_call(pending, "Bash", agent_id="agent-2"), "parent")

    def test_malformed_entries_never_match(self):
        pending = {
            "list": ["Bash"],
            "no_ts": {"tool_name": "Bash", "tool_use_id": "tu"},
            "bool_ts": _hook("Bash", True, "tu"),
            "str_ts": _hook("Bash", "1", "tu"),
            "nan_ts": _hook("Bash", float("nan"), "tu"),
            "": _hook("Bash", 0.5, "tu"),
            "ok": _hook("Bash", 9.0, "tu"),
        }
        self.assertEqual(match_hook_tool_call(pending, "Bash"), "ok")
        self.assertEqual(match_hook_tool_call(pending, "Bash", "tu"), "ok")

    def test_hook_call_timestamp(self):
        self.assertEqual(hook_call_timestamp({"timestamp": 2}), 2.0)
        for info in ({"timestamp": True}, {"timestamp": None}, {}, ["x"], None):
            with self.subTest(info=info):
                self.assertIsNone(hook_call_timestamp(info))


class StdioProxyTraceFollowsSharedRulesTest(unittest.TestCase):
    """What the stdio proxy records pairs the same way in every exporter."""

    def test_interleaved_results_and_errors(self):
        store = TraceStore(tempfile.mkdtemp(), redact=False)
        meta = SessionMeta()
        store.create_session(meta)
        proxy = MCPProxy(["true"], store, meta)
        messages = [
            ({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
              "params": {"name": "a", "arguments": {}}}, "agent_to_server"),
            ({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
              "params": {"name": "b", "arguments": {}}}, "agent_to_server"),
            ({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
              "params": {"name": "c", "arguments": {}}}, "agent_to_server"),
            ({"jsonrpc": "2.0", "id": 2,
              "error": {"code": -32602, "message": "bad"}}, "server_to_agent"),
            ({"jsonrpc": "2.0", "id": 1,
              "result": {"content": [{"type": "text", "text": "ok"}]}}, "server_to_agent"),
            ({"jsonrpc": "2.0", "id": 99,
              "result": {"content": [{"type": "text", "text": "?"}]}}, "server_to_agent"),
        ]
        for message, direction in messages:
            source = io.BytesIO((json.dumps(message) + "\n").encode("utf-8"))
            proxy._forward_and_trace(source, io.BytesIO(), direction, threading.Event())

        events = store.load_events(meta.session_id)
        by_rpc_id = {
            (e.event_type, e.data.get("request_id")): i for i, e in enumerate(events)
        }
        call_1 = by_rpc_id[(EventType.TOOL_CALL, 1)]
        call_2 = by_rpc_id[(EventType.TOOL_CALL, 2)]
        call_3 = by_rpc_id[(EventType.TOOL_CALL, 3)]
        error_2 = by_rpc_id[(EventType.ERROR, 2)]
        result_1 = by_rpc_id[(EventType.TOOL_RESULT, 1)]
        stray = by_rpc_id[(EventType.TOOL_RESULT, 99)]

        pairing = pair_tool_events(events)
        self.assertEqual(pairing.pairs, {error_2: call_2, result_1: call_1})
        self.assertEqual(pairing.unpaired_responses, [stray])
        self.assertEqual(pairing.unmatched_requests, [call_3])
        self.assertEqual(proxy._calls.pending(), {3: events[call_3]})


if __name__ == "__main__":
    unittest.main()
