"""Tests for OTLP exporter."""

import hashlib
import json
import math
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import _escapes
except ImportError:
    from tests import _escapes

try:  # keeps product telemetry off in this process and its children
    import _no_telemetry
except ImportError:  # pragma: no cover - run as a package module
    from tests import _no_telemetry

from agent_trace.event_data import UnusableTimestampError
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.otlp import (
    _make_attributes,
    _to_span_id,
    _to_trace_id,
    _ts_to_nanos,
    session_to_otlp,
    session_to_otlp_genai,
    tree_to_otlp,
    tree_to_otlp_genai,
)
from agent_trace.store import TraceStore

try:
    from test_unusable_event_timestamps import (
        BAD,
        CHILD,
        ENDPOINT,
        TimestampCliTestCase,
        restamp,
        set_meta,
    )
except ImportError:
    from tests.test_unusable_event_timestamps import (
        BAD,
        CHILD,
        ENDPOINT,
        TimestampCliTestCase,
        restamp,
        set_meta,
    )

LONE = _escapes.LONE_SURROGATE
SHOWN = _escapes.LONE_SURROGATE_SHOWN
PLAIN = _escapes.PLAIN

# OTLP carries each time as an unsigned 64-bit count of nanoseconds.
UINT64_MAX = 2**64 - 1
PAST_LIMIT = "is past the latest time OTLP can carry (2554-07-21T23:34:33.709551615Z)"
YEAR_3000 = 32503680000.0
# The largest float whose nanoseconds fit, and the next float up. The second
# is (2**64 - 1) / 1e9 as a float, and its nanoseconds are 2**64.
LAST_FLOAT = 18446744073.70955
PAST_FLOAT = 18446744073.709553
LAST_NANOS = "18446744073709549568"
_NANOS = re.compile(r"0|[1-9][0-9]*")


def unix_nanos_values(node) -> list:
    """Return the value of every key ending in ``UnixNano`` in *node*."""
    values = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key.endswith("UnixNano"):
                values.append(value)
            else:
                values.extend(unix_nanos_values(value))
    elif isinstance(node, list):
        for item in node:
            values.extend(unix_nanos_values(item))
    return values


def assert_unix_nanos_in_range(test: unittest.TestCase, payload) -> None:
    """Assert each ``UnixNano`` value in *payload* is a decimal string from 0 to 2**64 - 1."""
    values = unix_nanos_values(payload)
    test.assertTrue(values)
    bad = [
        value for value in values
        if not (isinstance(value, str) and _NANOS.fullmatch(value) and int(value) <= UINT64_MAX)
    ]
    test.assertEqual(bad, [])


def set_event_fields(session_dir: Path, event_id: str, **fields) -> None:
    """Set *fields* on the stored event *event_id* and rebuild the hash chain."""
    path = session_dir / "events.ndjson"
    lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        record.pop("prev_hash", None)
        if record.get("event_id") == event_id:
            record.update(fields)
        if lines:
            record["prev_hash"] = hashlib.sha256(lines[-1].encode()).hexdigest()
        lines.append(json.dumps(record, separators=(",", ":")))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def all_spans(payload: dict) -> list[dict]:
    return [
        span
        for resource in payload["resourceSpans"]
        for scope in resource["scopeSpans"]
        for span in scope["spans"]
    ]


def span_times(span: dict) -> tuple[str, str]:
    return span["startTimeUnixNano"], span["endTimeUnixNano"]


class TestHelpers(unittest.TestCase):
    def test_trace_id_is_32_hex(self):
        tid = _to_trace_id("abc123")
        self.assertEqual(len(tid), 32)
        int(tid, 16)  # should not raise

    def test_span_id_is_16_hex(self):
        sid = _to_span_id("event-1")
        self.assertEqual(len(sid), 16)
        int(sid, 16)

    def test_trace_id_deterministic(self):
        self.assertEqual(_to_trace_id("session-1"), _to_trace_id("session-1"))

    def test_different_sessions_different_ids(self):
        self.assertNotEqual(_to_trace_id("a"), _to_trace_id("b"))

    def test_ts_to_nanos(self):
        result = _ts_to_nanos(1.5)
        self.assertEqual(result, "1500000000")

    def test_make_attributes_string(self):
        attrs = _make_attributes({"key": "value"})
        self.assertEqual(len(attrs), 1)
        self.assertEqual(attrs[0]["key"], "key")
        self.assertEqual(attrs[0]["value"]["stringValue"], "value")

    def test_make_attributes_int(self):
        attrs = _make_attributes({"count": 42})
        self.assertEqual(attrs[0]["value"]["intValue"], "42")

    def test_make_attributes_float(self):
        attrs = _make_attributes({"ratio": 3.14})
        self.assertAlmostEqual(attrs[0]["value"]["doubleValue"], 3.14)

    def test_make_attributes_bool(self):
        attrs = _make_attributes({"enabled": True})
        self.assertEqual(attrs[0]["value"]["boolValue"], True)

    def test_make_attributes_dict(self):
        attrs = _make_attributes({"config": {"a": 1}})
        self.assertIn("a", attrs[0]["value"]["stringValue"])

    def test_make_attributes_list(self):
        attrs = _make_attributes({"items": [1, 2, 3]})
        self.assertIn("1", attrs[0]["value"]["stringValue"])


class TestIdHashes(unittest.TestCase):
    """A span ID or trace ID is the first hex digits of the SHA-256 of the ID's UTF-8 bytes.

    A stored ID can hold a lone surrogate, which strict UTF-8 refuses. The
    helpers encode it with surrogatepass, which gives bytes no valid ID has.
    """

    def test_valid_ids_keep_their_hashes(self):
        # A session exported again must keep its span and trace IDs.
        pinned = (
            ("event-1", "ce36863f51b6baf9", "ce36863f51b6baf9d16397ffb3e9af50"),
            ("call-1", "5d7963c4f471e142", "5d7963c4f471e142f5a72214a9666fb1"),
            ("root-abc123", "83c60204acb6bb9d", "83c60204acb6bb9d649341d666972e86"),
            (PLAIN, "81713abb8fb5e6a2", "81713abb8fb5e6a2e3d353b9961d4ad5"),
            ("call-" + SHOWN, "fcb5dd94c5e02840", "fcb5dd94c5e0284082ab0aba1c1a9a77"),
        )
        for value, span_id, trace_id in pinned:
            with self.subTest(value=value):
                self.assertEqual((_to_span_id(value), _to_trace_id(value)), (span_id, trace_id))

    def test_span_id_of_lone_surrogate(self):
        span_id = _to_span_id("call-" + LONE)
        self.assertEqual(span_id, "00888d4bf0f69018")
        # The six characters of its escape are another ID.
        self.assertNotEqual(span_id, _to_span_id("call-" + SHOWN))

    def test_trace_id_of_lone_surrogate(self):
        trace_id = _to_trace_id("sess-" + LONE)
        self.assertEqual(trace_id, "f802fac41222ee5c70813f000aaf3df9")
        self.assertNotEqual(trace_id, _to_trace_id("sess-" + SHOWN))

    def test_converters_take_lone_surrogates(self):
        meta = SessionMeta(session_id="sess-" + LONE, agent_name="probe", started_at=1.0, ended_at=3.0)
        events = [
            TraceEvent(
                event_type=EventType.TOOL_CALL, event_id="call-" + LONE, parent_id=PLAIN + " " + LONE,
                timestamp=1.5, data={"tool_name": "Bash", "arguments": {"command": "ls"}},
            ),
            TraceEvent(
                event_type=EventType.TOOL_RESULT, event_id="result-" + LONE, parent_id="call-" + LONE,
                timestamp=2.0, duration_ms=250, data={"tool_name": "Bash", "result": "ok"},
            ),
        ]
        for convert in (session_to_otlp, session_to_otlp_genai):
            with self.subTest(convert=convert.__name__):
                payload = convert(meta, events)
                root, call = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]
                self.assertEqual(root["traceId"], _to_trace_id("sess-" + LONE))
                self.assertEqual(root["spanId"], _to_span_id("root-sess-" + LONE))
                self.assertEqual(call["spanId"], _to_span_id("call-" + LONE))
                self.assertEqual(call["parentSpanId"], root["spanId"])
                # The result, found by its parent ID, ends the span 250 ms after the call.
                self.assertEqual(call["endTimeUnixNano"], "1750000000")
                # json.dumps writes the surrogate in an attribute as its escape.
                self.assertIn('"call-' + SHOWN + '"', json.dumps(payload))


class TestSessionToOTLP(unittest.TestCase):
    def _make_session(self):
        meta = SessionMeta(
            agent_name="claude-code",
            command="test",
            tool_calls=3,
            errors=1,
        )
        events = [
            TraceEvent(
                event_type=EventType.SESSION_START,
                session_id=meta.session_id,
                data={"mode": "claude-code-hooks"},
            ),
            TraceEvent(
                event_type=EventType.USER_PROMPT,
                session_id=meta.session_id,
                data={"prompt": "Fix the bug"},
            ),
            TraceEvent(
                event_type=EventType.TOOL_CALL,
                session_id=meta.session_id,
                data={"tool_name": "Bash", "arguments": {"command": "npm test"}},
            ),
        ]
        # tool_result linked to tool_call
        call_id = events[2].event_id
        events.append(TraceEvent(
            event_type=EventType.TOOL_RESULT,
            session_id=meta.session_id,
            data={"tool_name": "Bash", "result": "All tests passed"},
            parent_id=call_id,
            duration_ms=150,
        ))
        events.append(TraceEvent(
            event_type=EventType.TOOL_CALL,
            session_id=meta.session_id,
            data={"tool_name": "Read", "arguments": {"file_path": "/src/main.py"}},
        ))
        # error linked to tool_call
        read_id = events[4].event_id
        events.append(TraceEvent(
            event_type=EventType.ERROR,
            session_id=meta.session_id,
            data={"tool_name": "Read", "error": "File not found"},
            parent_id=read_id,
            duration_ms=5,
        ))
        events.append(TraceEvent(
            event_type=EventType.ASSISTANT_RESPONSE,
            session_id=meta.session_id,
            data={"text": "I fixed the bug."},
        ))
        events.append(TraceEvent(
            event_type=EventType.SESSION_END,
            session_id=meta.session_id,
            data={},
        ))
        return meta, events

    def test_returns_valid_structure(self):
        meta, events = self._make_session()
        payload = session_to_otlp(meta, events)

        self.assertIn("resourceSpans", payload)
        self.assertEqual(len(payload["resourceSpans"]), 1)

        rs = payload["resourceSpans"][0]
        self.assertIn("resource", rs)
        self.assertIn("scopeSpans", rs)
        self.assertEqual(len(rs["scopeSpans"]), 1)

        ss = rs["scopeSpans"][0]
        self.assertIn("spans", ss)
        self.assertGreater(len(ss["spans"]), 0)

    def test_root_span_exists(self):
        meta, events = self._make_session()
        payload = session_to_otlp(meta, events)
        spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]

        root = spans[0]
        self.assertIn("agent-session", root["name"])
        self.assertNotIn("parentSpanId", root)

    def test_tool_call_becomes_child_span(self):
        meta, events = self._make_session()
        payload = session_to_otlp(meta, events)
        spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]

        root_id = spans[0]["spanId"]
        child_spans = [s for s in spans[1:] if s.get("parentSpanId") == root_id]
        self.assertGreater(len(child_spans), 0)

        bash_span = [s for s in child_spans if "bash" in s["name"].lower()]
        self.assertEqual(len(bash_span), 1)

    def test_error_span_has_error_status(self):
        meta, events = self._make_session()
        payload = session_to_otlp(meta, events)
        spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]

        # Root span also has error status; filter to child spans only
        root_id = spans[0]["spanId"]
        error_children = [s for s in spans[1:] if s.get("status", {}).get("code") == 2]
        self.assertEqual(len(error_children), 1)
        self.assertEqual(error_children[0]["name"], "tool/Read")
        self.assertEqual(error_children[0]["kind"], 3)

    def test_error_span_has_exception_event(self):
        meta, events = self._make_session()
        payload = session_to_otlp(meta, events)
        spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]

        error_children = [s for s in spans[1:] if s.get("status", {}).get("code") == 2]
        self.assertEqual(len(error_children[0].get("events", [])), 1)
        self.assertEqual(error_children[0]["events"][0]["name"], "exception")

    def test_user_prompt_is_root_event(self):
        meta, events = self._make_session()
        payload = session_to_otlp(meta, events)
        spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]

        root = spans[0]
        prompt_events = [e for e in root.get("events", []) if e["name"] == "user_prompt"]
        self.assertEqual(len(prompt_events), 1)

    def test_assistant_response_is_root_event(self):
        meta, events = self._make_session()
        payload = session_to_otlp(meta, events)
        spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]

        root = spans[0]
        resp_events = [e for e in root.get("events", []) if e["name"] == "assistant_response"]
        self.assertEqual(len(resp_events), 1)

    def test_trace_id_consistent(self):
        meta, events = self._make_session()
        payload = session_to_otlp(meta, events)
        spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]

        trace_ids = {s["traceId"] for s in spans}
        self.assertEqual(len(trace_ids), 1)

    def test_service_name_in_resource(self):
        meta, events = self._make_session()
        payload = session_to_otlp(meta, events, service_name="my-agent")

        attrs = payload["resourceSpans"][0]["resource"]["attributes"]
        service_attr = [a for a in attrs if a["key"] == "service.name"]
        self.assertEqual(len(service_attr), 1)
        self.assertEqual(service_attr[0]["value"]["stringValue"], "my-agent")

    def test_tool_input_in_span_attributes(self):
        meta, events = self._make_session()
        payload = session_to_otlp(meta, events)
        spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]

        bash_span = [s for s in spans if "bash" in s["name"].lower()][0]
        attr_keys = [a["key"] for a in bash_span["attributes"]]
        self.assertIn("tool.input.command", attr_keys)

    def test_span_has_duration(self):
        meta, events = self._make_session()
        payload = session_to_otlp(meta, events)
        spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]

        bash_span = [s for s in spans if "bash" in s["name"].lower()][0]
        start = int(bash_span["startTimeUnixNano"])
        end = int(bash_span["endTimeUnixNano"])
        self.assertGreater(end, start)

    def test_empty_session(self):
        meta = SessionMeta(agent_name="test")
        payload = session_to_otlp(meta, [])
        spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]
        # Should still have root span
        self.assertEqual(len(spans), 1)


class TestSpanStatus(unittest.TestCase):
    """Successful spans leave status UNSET; only failures set ERROR.

    OTel semantic conventions v1.45.0 docs/general/recording-errors.md:
    span status MUST be left unset when the operation ended without errors.
    """

    _make_session = TestSessionToOTLP._make_session

    def _spans(self, meta, events):
        return session_to_otlp(meta, events)["resourceSpans"][0]["scopeSpans"][0]["spans"]

    def test_successful_tool_span_and_root_leave_status_unset(self):
        meta = SessionMeta(agent_name="claude-code", errors=0)
        call = TraceEvent(event_type=EventType.TOOL_CALL, event_id="call1",
                          data={"tool_name": "Bash", "arguments": {"command": "ls"}})
        result = TraceEvent(event_type=EventType.TOOL_RESULT, parent_id="call1",
                            duration_ms=10, data={"tool_name": "Bash", "result": "ok"})
        root, tool = self._spans(meta, [call, result])
        self.assertEqual(tool["name"], "tool/Bash")
        self.assertEqual(tool["status"], {"code": 0})
        self.assertEqual(root["status"], {"code": 0})

    def test_error_statuses_are_unchanged(self):
        meta, events = self._make_session()
        spans = self._spans(meta, events)
        self.assertEqual(spans[0]["status"], {"code": 2})
        (error,) = [s for s in spans[1:] if s["name"] == "tool/Read"]
        self.assertEqual(error["status"], {"code": 2, "message": "File not found"})
        (bash,) = [s for s in spans if s["name"] == "tool/Bash"]
        self.assertEqual(bash["status"], {"code": 0})

    def test_no_span_sets_status_ok(self):
        meta, events = self._make_session()
        events.append(TraceEvent(event_type=EventType.LLM_REQUEST, event_id="r1",
                                 data={"model": "gpt-4o", "input_tokens": 3}))
        events.append(TraceEvent(event_type=EventType.LLM_RESPONSE, parent_id="r1",
                                 data={"model": "gpt-4o", "output_tokens": 2}))
        for errors in (0, 1):
            meta.errors = errors
            with self.subTest(errors=errors):
                codes = [s["status"]["code"] for s in self._spans(meta, events)]
                self.assertNotIn(1, codes)


class TestUnrequestedLLMResponse(unittest.TestCase):
    """Codex records each model request only as an llm_response."""

    @staticmethod
    def _response_attrs(events):
        payload = session_to_otlp(SessionMeta(agent_name="openai-codex"), events)
        root = payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        return [
            {a["key"]: a["value"] for a in e["attributes"]}
            for e in root["events"] if e["name"] == "gen_ai.assistant.message"
        ]

    def test_unrequested_response_carries_its_model_and_input(self):
        events = [
            TraceEvent(event_type=EventType.LLM_RESPONSE, data={
                "model": "gpt-5-codex", "input_tokens": 1200, "output_tokens": 10,
            }),
            TraceEvent(event_type=EventType.LLM_RESPONSE, data={
                "input_tokens": 2500, "output_tokens": 30,
            }),
        ]
        attrs = self._response_attrs(events)
        self.assertEqual(attrs[0]["gen_ai.request.model"], {"stringValue": "gpt-5-codex"})
        self.assertEqual(attrs[0]["gen_ai.usage.input_tokens"], {"intValue": "1200"})
        self.assertEqual(attrs[0]["gen_ai.usage.output_tokens"], {"intValue": "10"})
        self.assertNotIn("gen_ai.request.model", attrs[1])
        self.assertEqual(attrs[1]["gen_ai.usage.input_tokens"], {"intValue": "2500"})

    def test_paired_response_is_unchanged(self):
        request = TraceEvent(event_type=EventType.LLM_REQUEST, data={
            "model": "gpt-4o", "input_tokens": 100,
        })
        response = TraceEvent(event_type=EventType.LLM_RESPONSE, parent_id=request.event_id, data={
            "model": "gpt-4o", "input_tokens": 100, "output_tokens": 5,
        })
        attrs = self._response_attrs([request, response])
        self.assertEqual(sorted(attrs[0]), ["gen_ai.operation.name", "gen_ai.usage.output_tokens"])


class TestFailedLLMResponse(unittest.TestCase):
    """A failed or cancelled model request is marked on its response event."""

    META = SessionMeta(session_id="s1", agent_name="openai-codex", started_at=1000.0, ended_at=1100.0)

    def _payload(self, events):
        return session_to_otlp(self.META, events)

    def _response_attrs(self, *responses):
        events = [
            TraceEvent(event_type=EventType.LLM_RESPONSE, timestamp=1001.0 + i, data=data)
            for i, data in enumerate(responses)
        ]
        root = self._payload(events)["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        return [
            {a["key"]: a["value"] for a in e["attributes"]}
            for e in root["events"] if e["name"] == "gen_ai.assistant.message"
        ]

    def test_failed_with_error_text(self):
        (attrs,) = self._response_attrs({
            "model": "gpt-5-codex", "status": "failed", "error": "synthetic failure",
            "token_usage": None,
        })
        self.assertEqual(attrs["error.type"], {"stringValue": "_OTHER"})
        self.assertEqual(attrs["error.message"], {"stringValue": "synthetic failure"})
        self.assertEqual(attrs["gen_ai.response.finish_reasons"],
                         {"arrayValue": {"values": [{"stringValue": "error"}]}})

    def test_failed_without_error_text(self):
        (attrs,) = self._response_attrs({"model": "gpt-5-codex", "status": "failed"})
        self.assertEqual(attrs["error.type"], {"stringValue": "_OTHER"})
        self.assertNotIn("error.message", attrs)

    def test_cancelled(self):
        (attrs,) = self._response_attrs({"model": "gpt-5-codex", "status": "cancelled"})
        self.assertEqual(attrs["error.type"], {"stringValue": "cancelled"})
        self.assertEqual(attrs["gen_ai.response.finish_reasons"],
                         {"arrayValue": {"values": [{"stringValue": "error"}]}})
        self.assertNotIn("error.message", attrs)

    def test_completed_is_unchanged(self):
        (attrs,) = self._response_attrs({
            "model": "gpt-5-codex", "status": "completed", "output_tokens": 3,
        })
        self.assertEqual(sorted(attrs), [
            "gen_ai.operation.name", "gen_ai.request.model", "gen_ai.usage.output_tokens",
        ])

    def test_root_status_follows_the_last_request(self):
        completed = [TraceEvent(event_type=EventType.LLM_RESPONSE, timestamp=1001.0,
                                data={"status": "completed"})]
        failed = [TraceEvent(event_type=EventType.LLM_RESPONSE, timestamp=1001.0,
                             data={"status": "failed", "error": "boom"})]
        def root(events):
            return self._payload(events)["resourceSpans"][0]["scopeSpans"][0]["spans"][0]

        self.assertEqual(root(completed)["status"], {"code": 0})
        self.assertEqual(root(failed)["status"], {"code": 2})
        self.assertIn(
            {"key": "error.type", "value": {"stringValue": "_OTHER"}},
            root(failed)["attributes"],
        )


def _classic_spans(events):
    meta = SessionMeta(agent_name="claude-code", started_at=1000.0, ended_at=1100.0)
    return session_to_otlp(meta, events)["resourceSpans"][0]["scopeSpans"][0]["spans"]


def _attrs(span):
    return {a["key"]: a["value"].get("stringValue") for a in span["attributes"]}


class TestToolSpanIdentity(unittest.TestCase):
    """A tool call that failed or got no result names its tool the way a
    tool call with a result does (#552). An error with no call and no
    tool_name is not a tool call."""

    IDENTITY = ("gen_ai.tool.name", "gen_ai.operation.name", "gen_ai.tool.call.id")

    def _call(self, event_id="call1", **data):
        return TraceEvent(event_type=EventType.TOOL_CALL, event_id=event_id, timestamp=1001.0,
                          data={"tool_name": "Bash", "arguments": {"command": "false"}, **data})

    def test_failed_call_has_the_identity_of_a_call_with_a_result(self):
        result = TraceEvent(event_type=EventType.TOOL_RESULT, parent_id="ok1", timestamp=1002.0,
                            data={"result": "done"})
        error = TraceEvent(event_type=EventType.ERROR, parent_id="call1", timestamp=1003.0,
                           data={"tool_name": "Bash", "error": "exit 1"})
        _, ok, failed = _classic_spans([self._call("ok1"), result, self._call(), error])
        self.assertEqual((ok["name"], ok["kind"]), ("tool/Bash", 3))
        self.assertEqual((failed["name"], failed["kind"]), ("tool/Bash", 3))
        attrs = _attrs(failed)
        self.assertEqual(
            {key: attrs.get(key) for key in self.IDENTITY},
            {"gen_ai.tool.name": "Bash", "gen_ai.operation.name": "tool.call",
             "gen_ai.tool.call.id": "call1"},
        )
        self.assertEqual(
            {key: _attrs(ok).get(key) for key in self.IDENTITY},
            {**{key: attrs[key] for key in self.IDENTITY}, "gen_ai.tool.call.id": "ok1"},
        )
        # The keys the failed span carried before stay.
        self.assertEqual(attrs["tool.name"], "Bash")
        self.assertEqual(attrs["tool.input.command"], "false")
        self.assertEqual(attrs["error.message"], "exit 1")
        self.assertEqual(failed["status"], {"code": 2, "message": "exit 1"})

    def test_failed_call_takes_its_name_from_the_call(self):
        # The @trace_tool decorator records the tool name only on the call.
        error = TraceEvent(event_type=EventType.ERROR, parent_id="call1", timestamp=1003.0,
                           data={"error": "exit 1"})
        _, failed = _classic_spans([self._call(), error])
        self.assertEqual(failed["name"], "tool/Bash")
        self.assertEqual(_attrs(failed)["gen_ai.tool.name"], "Bash")
        self.assertEqual(_attrs(failed)["tool.name"], "Bash")

    def test_error_with_no_call_and_no_tool_name_is_not_a_tool_call(self):
        request = TraceEvent(event_type=EventType.LLM_REQUEST, event_id="req1", timestamp=1001.0,
                             data={"model": "gpt-4o"})
        cases = {
            # proxy.py, when the MCP server command fails.
            "could not start the MCP server: [Errno 2] No such file or directory: 'srv'": [],
            # http_proxy.py sets no parent_id, so the error does not end the request.
            "Proxy error: [Errno 61] Connection refused": [request],
        }
        for message, before in cases.items():
            with self.subTest(message=message):
                error = TraceEvent(event_type=EventType.ERROR, event_id="err1", timestamp=1003.0,
                                   data={"message": message})
                _, span = _classic_spans([*before, error])
                self.assertEqual((span["name"], span["kind"]), ("error", 1))
                self.assertEqual(_attrs(span), {"tool.name": "error", "error.message": message})
                self.assertEqual(span["status"], {"code": 2, "message": message})
                self.assertEqual([event["name"] for event in span["events"]], ["exception"])

    def test_error_with_an_empty_tool_name_and_no_call_is_named_error(self):
        for tool_name in ("", None):
            with self.subTest(tool_name=tool_name):
                error = TraceEvent(event_type=EventType.ERROR, timestamp=1003.0,
                                   data={"tool_name": tool_name, "error": "boom"})
                _, span = _classic_spans([error])
                self.assertEqual((span["name"], span["kind"]), ("error", 1))
                self.assertEqual(_attrs(span), {"tool.name": "error", "error.message": "boom"})

    def test_error_with_a_call_or_a_tool_name_is_a_tool_call(self):
        nameless_call = TraceEvent(event_type=EventType.TOOL_CALL, event_id="call1",
                                   timestamp=1001.0, data={"arguments": {"x": 1}})
        cases = [
            # A call and a tool_name, a call that names no tool, a tool_name and no call.
            ([self._call()], "call1", {"tool_name": "Bash", "error": "exit 1"}, "Bash", "call1"),
            ([nameless_call], "call1", {"error": "exit 1"}, "error", "call1"),
            ([], "", {"tool_name": "Read", "error": "exit 1"}, "Read", "err1"),
        ]
        for before, parent_id, data, tool, call_id in cases:
            with self.subTest(tool=tool):
                error = TraceEvent(event_type=EventType.ERROR, event_id="err1",
                                   parent_id=parent_id, timestamp=1003.0, data=data)
                _, span = _classic_spans([*before, error])
                self.assertEqual((span["name"], span["kind"]), (f"tool/{tool}", 3))
                attrs = _attrs(span)
                self.assertEqual(
                    {key: attrs.get(key) for key in self.IDENTITY},
                    {"gen_ai.tool.name": tool, "gen_ai.operation.name": "tool.call",
                     "gen_ai.tool.call.id": call_id},
                )

    def test_error_that_ends_a_model_request_keeps_its_own_tool_name(self):
        request = TraceEvent(event_type=EventType.LLM_REQUEST, event_id="req1", timestamp=1001.0,
                             data={"model": "gpt-4o"})
        error = TraceEvent(event_type=EventType.ERROR, parent_id="req1", timestamp=1002.0,
                           data={"tool_name": "web_search", "error": "rate limited"})
        _, span = _classic_spans([request, error])
        self.assertEqual((span["name"], span["kind"]), ("web_search", 1))
        self.assertEqual(_attrs(span), {"tool.name": "web_search", "error.message": "rate limited"})

    def test_call_without_a_result_has_the_identity_of_a_call_with_one(self):
        _, span = _classic_spans([self._call()])
        self.assertEqual((span["name"], span["kind"]), ("tool/Bash", 3))
        attrs = _attrs(span)
        self.assertEqual(
            {key: attrs.get(key) for key in self.IDENTITY},
            {"gen_ai.tool.name": "Bash", "gen_ai.operation.name": "tool.call",
             "gen_ai.tool.call.id": "call1"},
        )
        self.assertEqual(attrs["tool.name"], "Bash")
        self.assertEqual(attrs["tool.input.command"], "false")

    def test_error_that_ends_a_model_request_is_not_a_tool_call(self):
        request = TraceEvent(event_type=EventType.LLM_REQUEST, event_id="req1", timestamp=1001.0,
                             data={"model": "gpt-4o"})
        error = TraceEvent(event_type=EventType.ERROR, parent_id="req1", timestamp=1002.0,
                           data={"error": "rate limited"})
        _, span = _classic_spans([request, error])
        self.assertEqual((span["name"], span["kind"]), ("error", 1))
        self.assertEqual(_attrs(span), {"tool.name": "error", "error.message": "rate limited"})


class TestRetryLinks(unittest.TestCase):
    """A tool call whose ``retry_of`` names an earlier call links to its span (#552)."""

    def _session(self, retry_of="call1"):
        return [
            TraceEvent(event_type=EventType.TOOL_CALL, event_id="call1", timestamp=1001.0,
                       data={"tool_name": "Bash", "arguments": {"command": "make"}}),
            TraceEvent(event_type=EventType.ERROR, parent_id="call1", timestamp=1002.0,
                       data={"tool_name": "Bash", "error": "exit 2"}),
            TraceEvent(event_type=EventType.TOOL_CALL, event_id="call2", timestamp=1003.0,
                       data={"tool_name": "Bash", "arguments": {"command": "make"},
                             "retry_of": retry_of}),
            TraceEvent(event_type=EventType.TOOL_RESULT, parent_id="call2", timestamp=1004.0,
                       data={"result": "ok"}),
        ]

    def test_retry_links_to_the_attempt_it_retries(self):
        _, first, retry = _classic_spans(self._session())
        self.assertEqual(retry["links"], [{
            "traceId": first["traceId"],
            "spanId": first["spanId"],
            "attributes": [{"key": "agent.strace.link.type", "value": {"stringValue": "retry_of"}}],
        }])
        self.assertNotIn("links", first)

    def test_no_link_to_an_unknown_event_or_to_itself(self):
        for retry_of in ("missing", "call2", "", 7, ["call1"]):
            with self.subTest(retry_of=retry_of):
                spans = _classic_spans(self._session(retry_of))
                self.assertFalse(any("links" in span for span in spans))

    def test_session_without_retries_has_no_links(self):
        meta, events = TestSessionToOTLP()._make_session()
        spans = session_to_otlp(meta, events)["resourceSpans"][0]["scopeSpans"][0]["spans"]
        self.assertFalse(any("links" in span for span in spans))


def _call(event_id, ts, command="make"):
    return TraceEvent(event_type=EventType.TOOL_CALL, event_id=event_id, timestamp=ts,
                      data={"tool_name": "Bash", "arguments": {"command": command}})


def _tool_error(call_id, ts, **data):
    return TraceEvent(event_type=EventType.ERROR, parent_id=call_id, timestamp=ts,
                      data={"tool_name": "Bash", "error": "exit 2", **data})


def _tool_ok(call_id, ts):
    return TraceEvent(event_type=EventType.TOOL_RESULT, parent_id=call_id, timestamp=ts,
                      data={"tool_name": "Bash", "result": "ok"})


def _model_request(event_id, ts):
    return TraceEvent(event_type=EventType.LLM_REQUEST, event_id=event_id, timestamp=ts,
                      data={"model": "gpt-4o"})


def _model_response(request_id, ts, **data):
    return TraceEvent(event_type=EventType.LLM_RESPONSE, parent_id=request_id, timestamp=ts,
                      data={"model": "gpt-4o", **data})


def _session_end(**data):
    return TraceEvent(event_type=EventType.SESSION_END, timestamp=1090.0, data=data)


_RECOVERED = [
    _call("c1", 1001.0), _tool_error("c1", 1002.0, error_type="ShellError"),
    _call("c2", 1003.0), _tool_ok("c2", 1004.0),
    _session_end(reason="completed"),
]
_OUTCOME_KEYS = (
    "agent.strace.behavior.failed_operation_count",
    "agent.strace.behavior.retry_count",
    "agent.strace.behavior.recovered_operation_count",
    "agent.strace.behavior.unresolved_failed_operation_count",
    "agent.strace.session_status",
    "agent.strace.last_model_request_failed",
)


def _attribute_values(span):
    values = {}
    for attribute in span["attributes"]:
        value = attribute["value"]
        if "intValue" in value:
            values[attribute["key"]] = int(value["intValue"])
        else:
            (values[attribute["key"]],) = value.values()
    return values


class TestRootOutcome(unittest.TestCase):
    """The session root's status follows how the session ended (#561)."""

    EXPORTERS = (("otlp", session_to_otlp), ("otlp-genai", session_to_otlp_genai))
    SCENARIOS = {
        "recovered": (_RECOVERED, None),
        "failed again after recovering": (_RECOVERED[:-1] + [
            _call("c3", 1005.0), _tool_error("c3", 1006.0, error_type="AgainError"),
            _session_end(reason="completed"),
        ], "AgainError"),
        "unresolved": ([
            _call("c1", 1001.0), _tool_error("c1", 1002.0, error_type="ShellError"),
            _call("c2", 1003.0, command="make -k"), _tool_ok("c2", 1004.0),
        ], "ShellError"),
        "unresolved without a type": ([_call("c1", 1001.0), _tool_error("c1", 1002.0)], "_OTHER"),
        "unresolved exception": (
            [_call("c1", 1001.0), _tool_error("c1", 1002.0, exception_type="OSError")],
            "OSError",
        ),
        "timeout reason": ([_session_end(reason="Timeout after 600s")], "timeout"),
        "timeout status": (_RECOVERED[:-1] + [_session_end(status="timed-out")], "timeout"),
        "timeout exit code": ([_session_end(exit_code=124)], "timeout"),
        "terminated reason": ([_session_end(reason="killed by user")], "terminated"),
        "terminated exit code": ([_session_end(exit_code=137)], "terminated"),
        "huge exit code": ([_session_end(exit_code=10**400)], "terminated"),
        "last model request failed": ([
            _model_request("r1", 1001.0),
            _model_response("r1", 1002.0, status="failed", error_type="rate_limit_error"),
        ], "rate_limit_error"),
        "last model request failed untyped": ([
            _model_request("r1", 1001.0),
            TraceEvent(event_type=EventType.ERROR, parent_id="r1", timestamp=1002.0,
                       data={"error": "connection reset"}),
        ], "_OTHER"),
        "earlier model request failed": ([
            _model_request("r1", 1001.0), _model_response("r1", 1002.0, status="failed"),
            _model_request("r2", 1003.0), _model_response("r2", 1004.0, output_tokens=2),
        ], None),
        "last model request cancelled": ([
            _model_request("r1", 1001.0), _model_response("r1", 1002.0, status="cancelled"),
        ], None),
        "session status wins": (
            [_call("c1", 1001.0), _tool_error("c1", 1002.0, error_type="ShellError"),
             _session_end(exit_code=124)],
            "timeout",
        ),
        "model failure wins over tool failure": ([
            _call("c1", 1001.0), _tool_error("c1", 1002.0, error_type="ShellError"),
            _model_request("r1", 1003.0),
            _model_response("r1", 1004.0, status="failed", error_type="overloaded"),
        ], "overloaded"),
        "last unresolved failure names the type": ([
            _call("c1", 1001.0), _tool_error("c1", 1002.0, error_type="First"),
            _call("c2", 1003.0, command="test"), _tool_error("c2", 1004.0, error_type="Last"),
        ], "Last"),
        "no session end": ([], None),
    }

    def _root(self, convert, events, **kwargs):
        meta = SessionMeta(session_id="s1", agent_name="claude-code", started_at=1000.0,
                           ended_at=1100.0, errors=1)
        return convert(meta, events, **kwargs)["resourceSpans"][0]["scopeSpans"][0]["spans"][0]

    def test_root_status_and_error_type(self):
        for name, convert in self.EXPORTERS:
            for scenario, (events, error_type) in self.SCENARIOS.items():
                with self.subTest(exporter=name, scenario=scenario):
                    root = self._root(convert, events)
                    attrs = _attribute_values(root)
                    if error_type is None:
                        self.assertEqual(root["status"], {"code": 0})
                        self.assertNotIn("error.type", attrs)
                    else:
                        self.assertEqual(root["status"], {"code": 2})
                        self.assertEqual(attrs["error.type"], error_type)

    def test_every_error_root_carries_error_type(self):
        # Recording Errors: error.type MUST be set when the operation fails.
        for name, convert in self.EXPORTERS:
            for scenario, (events, _) in self.SCENARIOS.items():
                with self.subTest(exporter=name, scenario=scenario):
                    root = self._root(convert, events)
                    attrs = _attribute_values(root)
                    self.assertEqual(root["status"]["code"] == 2, "error.type" in attrs)
                    self.assertNotEqual(attrs.get("error.type"), "")

    def test_copilot_session_end_reasons(self):
        # Copilot's sessionEnd reason is complete, error, abort, timeout or user_exit.
        expected = {
            "error": ("terminated", {"code": 2}, "terminated"),
            "abort": ("terminated", {"code": 2}, "terminated"),
            "complete": ("completed", {"code": 0}, None),
            "user_exit": ("completed", {"code": 0}, None),
        }
        for name, convert in self.EXPORTERS:
            for reason, (session_status, status, error_type) in expected.items():
                with self.subTest(exporter=name, reason=reason):
                    root = self._root(convert, [_session_end(reason=reason)])
                    attrs = _attribute_values(root)
                    self.assertEqual(attrs["agent.strace.session_status"], session_status)
                    self.assertEqual(root["status"], status)
                    self.assertEqual(attrs.get("error.type"), error_type)

    def test_outcome_attributes(self):
        for name, convert in self.EXPORTERS:
            with self.subTest(exporter=name):
                attrs = _attribute_values(self._root(convert, _RECOVERED))
                self.assertEqual({key: attrs[key] for key in _OUTCOME_KEYS}, {
                    "agent.strace.behavior.failed_operation_count": 1,
                    "agent.strace.behavior.retry_count": 1,
                    "agent.strace.behavior.recovered_operation_count": 1,
                    "agent.strace.behavior.unresolved_failed_operation_count": 0,
                    "agent.strace.session_status": "completed",
                    "agent.strace.last_model_request_failed": False,
                })
                self.assertEqual(attrs["agent.errors"], 1)

        attrs = _attribute_values(self._root(session_to_otlp_genai, self.SCENARIOS[
            "last model request failed"][0]))
        self.assertTrue(attrs["agent.strace.last_model_request_failed"])
        self.assertEqual(attrs["agent.strace.session_status"], "not_recorded")
        metadata = json.loads(attrs["metadata"])["agent_strace"]
        self.assertTrue(metadata["last_model_request_failed"])
        self.assertEqual(metadata["behavior.unresolved_failed_operation_count"], 0)

    def test_failed_child_spans_keep_their_status(self):
        for name, convert in self.EXPORTERS:
            with self.subTest(exporter=name):
                meta = SessionMeta(session_id="s1", agent_name="claude-code",
                                   started_at=1000.0, ended_at=1100.0)
                root, *children = convert(meta, _RECOVERED)["resourceSpans"][0][
                    "scopeSpans"][0]["spans"]
                self.assertEqual(root["status"], {"code": 0})
                self.assertEqual(
                    sorted(child["status"]["code"] for child in children), [0, 2],
                )

    def test_root_outcome_false_leaves_the_root_unset_without_outcome(self):
        for name, convert in self.EXPORTERS:
            with self.subTest(exporter=name):
                root = self._root(convert, self.SCENARIOS["session status wins"][0],
                                  root_outcome=False)
                attrs = _attribute_values(root)
                self.assertEqual(root["status"], {"code": 0})
                self.assertFalse(set(_OUTCOME_KEYS + ("error.type",)) & set(attrs))

    def test_subagent_roots(self):
        from agent_trace.otlp import node_tree_to_otlp
        from agent_trace.subagent import SessionNode

        parent = SessionNode(
            meta=SessionMeta(session_id="parent", agent_name="claude-code",
                             started_at=1000.0, ended_at=1100.0),
            events=[_call("task", 1001.0), _tool_ok("task", 1050.0)],
        )
        parent.children.append(SessionNode(
            meta=SessionMeta(session_id="child", agent_name="worker", started_at=1001.0,
                             ended_at=1049.0, parent_session_id="parent",
                             parent_event_id="task", depth=1),
            events=[_call("c1", 1002.0), _tool_error("c1", 1003.0, error_type="ShellError")],
        ))
        for genai in (False, True):
            with self.subTest(genai=genai):
                spans = node_tree_to_otlp(parent, genai=genai)["resourceSpans"][0][
                    "scopeSpans"][0]["spans"]
                roots = {span["spanId"]: span for span in spans}
                parent_root = roots[_to_span_id("root-parent")]
                child_root = roots[_to_span_id("root-child")]
                self.assertEqual(parent_root["status"], {"code": 0})
                self.assertNotIn("error.type", _attribute_values(parent_root))
                attrs = _attribute_values(child_root)
                self.assertEqual(child_root["status"], {"code": 2})
                self.assertEqual(attrs["error.type"], "ShellError")
                self.assertEqual(
                    attrs["agent.strace.behavior.unresolved_failed_operation_count"], 1,
                )
                self.assertEqual(attrs["agent.strace.session_status"], "not_recorded")

    def test_unusable_event_fields_do_not_stop_the_export(self):
        arguments = {"x": math.nan, "y": LONE}
        events = [
            # The converter never reads this replaced call or the session_end.
            TraceEvent(event_type=EventType.TOOL_CALL, event_id="c0", timestamp=1000.5,
                       data=["not", "an", "object"]),
            TraceEvent(event_type=EventType.TOOL_CALL, event_id="c0", timestamp=1000.6,
                       data={"tool_name": "Read"}),
            TraceEvent(event_type=EventType.TOOL_CALL, event_id="c1", timestamp=1001.0,
                       data={"tool_name": "Bash", "arguments": arguments}),
            _tool_error("c1", 1002.0),
            TraceEvent(event_type=EventType.TOOL_CALL, event_id="c2", timestamp=1003.0,
                       data={"tool_name": "Bash", "arguments": arguments}),
            _tool_ok("c2", 1004.0),
            TraceEvent(event_type=EventType.SESSION_END, timestamp=1006.0, data="done"),
        ]
        for name, convert in self.EXPORTERS:
            with self.subTest(exporter=name):
                root = self._root(convert, events)
                attrs = _attribute_values(root)
                self.assertEqual(attrs["agent.strace.session_status"], "completed")
                self.assertEqual(attrs["agent.strace.behavior.failed_operation_count"], 1)
                self.assertEqual(attrs["agent.strace.behavior.recovered_operation_count"], 1)
                self.assertEqual(root["status"], {"code": 0})


def library_meta(**fields) -> SessionMeta:
    return SessionMeta(**{"session_id": "s1", "agent_name": "claude-code",
                          "started_at": 1000.0, "ended_at": 1100.0, **fields})


def tool_events(duration_ms=None) -> list[TraceEvent]:
    """A tool call with a result, and a lone error, each with *duration_ms*."""
    return [
        TraceEvent(event_type=EventType.TOOL_CALL, event_id="call1", timestamp=1001.0,
                   data={"tool_name": "Bash", "arguments": {"command": "make"}}),
        TraceEvent(event_type=EventType.TOOL_RESULT, event_id="res1", parent_id="call1",
                   timestamp=1002.0, duration_ms=duration_ms, data={"result": "ok"}),
        TraceEvent(event_type=EventType.ERROR, event_id="err1", timestamp=1003.0,
                   duration_ms=duration_ms, data={"error": "boom"}),
    ]


def near_limit_session() -> tuple[SessionMeta, list[TraceEvent]]:
    """A session at LAST_FLOAT. Each span but the transcript response would end
    1 ms or more later."""
    def event(event_type, event_id, parent_id="", duration_ms=None, **data):
        return TraceEvent(event_type=event_type, event_id=event_id, parent_id=parent_id,
                          timestamp=LAST_FLOAT, duration_ms=duration_ms, data=data)

    events = [
        event(EventType.LLM_REQUEST, "req1", model="m"),
        event(EventType.LLM_RESPONSE, "resp1", "req1", model="m"),
        event(EventType.LLM_REQUEST, "req2", model="m"),
        event(EventType.TOOL_CALL, "call1", tool_name="Bash"),
        event(EventType.TOOL_CALL, "call2", tool_name="Bash"),
        event(EventType.TOOL_RESULT, "res2", "call2", result="ok"),
        event(EventType.ERROR, "err1", duration_ms=1e14, error="boom"),
        # A model response read from a transcript. Its span may have zero length.
        event(EventType.ASSISTANT_RESPONSE, "msg1", model="claude-sonnet-4-5",
              context_stream="main", message_id="msg_1",
              usage={"input_tokens": 1, "output_tokens": 1}),
    ]
    return library_meta(started_at=LAST_FLOAT, ended_at=LAST_FLOAT), events


class TestOtlpTimeLimit(unittest.TestCase):
    """OTLP carries each time as an unsigned 64-bit count of nanoseconds, so
    every time the classic converter writes is from 0 to 2**64 - 1."""

    def test_ts_to_nanos_refuses_a_time_outside_the_range(self):
        self.assertEqual(_ts_to_nanos(0), "0")
        self.assertEqual(_ts_to_nanos(LAST_FLOAT), LAST_NANOS)
        cases = [
            (YEAR_3000, f"timestamp {PAST_LIMIT}"),
            (PAST_FLOAT, f"timestamp {PAST_LIMIT}"),
            (-5.0, "timestamp must be a Unix time in seconds"),
            (float("nan"), "timestamp must be a Unix time in seconds"),
            (float("inf"), "timestamp must be a Unix time in seconds"),
        ]
        for value, message in cases:
            with self.subTest(value=value):
                with self.assertRaises(UnusableTimestampError) as caught:
                    _ts_to_nanos(value)
                self.assertEqual(str(caught.exception), message)

    def test_duration_past_the_limit_reads_as_missing(self):
        payload = session_to_otlp(library_meta(), tool_events(1e14))
        self.assertEqual(payload, session_to_otlp(library_meta(), tool_events()))
        _, tool, error = all_spans(payload)
        self.assertEqual(span_times(tool), ("1001000000000", "1001001000000"))
        self.assertEqual(span_times(error), ("1003000000000", "1003001000000"))
        assert_unix_nanos_in_range(self, payload)

    def test_failed_tool_result_duration_past_the_limit_reads_as_missing(self):
        def failed_events(duration_ms=None):
            events = tool_events(duration_ms)[:2]
            events[1].data["is_error"] = True
            return events

        # A failed tool_result exports as an error span (#573).
        for name, convert in (("otlp", session_to_otlp), ("otlp-genai", session_to_otlp_genai)):
            with self.subTest(format=name):
                payload = convert(library_meta(), failed_events(1e14))
                self.assertEqual(payload, convert(library_meta(), failed_events()))
                _, tool = all_spans(payload)
                self.assertEqual(tool["status"]["code"], 2)
                self.assertEqual(span_times(tool), ("1001000000000", "1001001000000"))
                assert_unix_nanos_in_range(self, payload)

    def test_root_total_past_the_limit_ends_at_the_start(self):
        without_total = session_to_otlp(library_meta(ended_at=None), tool_events())
        for total in (1e14, 1e308):
            with self.subTest(total=total):
                payload = session_to_otlp(
                    library_meta(ended_at=None, total_duration_ms=total), tool_events(),
                )
                self.assertEqual(payload, without_total)
                self.assertEqual(
                    span_times(all_spans(payload)[0]), ("1000000000000", "1000000000000"),
                )
                assert_unix_nanos_in_range(self, payload)

    def test_spans_near_the_limit_end_at_the_limit(self):
        payload = session_to_otlp(*near_limit_session())
        root, *children = all_spans(payload)
        self.assertEqual(span_times(root), (LAST_NANOS, LAST_NANOS))
        limit = (LAST_NANOS, str(UINT64_MAX))
        self.assertEqual(sorted((span["name"], span_times(span)) for span in children), [
            ("chat claude-sonnet-4-5", (LAST_NANOS, LAST_NANOS)),
            ("error", limit),
            ("tool/Bash", limit),
            ("tool/Bash", limit),
        ])
        assert_unix_nanos_in_range(self, payload)

    def test_tree_export_refuses_a_time_past_the_limit(self):
        session_id = "d4e5f6a7b8c90004"
        with tempfile.TemporaryDirectory() as tmp:
            store = TraceStore(tmp)
            store.create_session(SessionMeta(session_id=session_id, started_at=1000.0,
                                             ended_at=1100.0))
            store.append_event(session_id, TraceEvent(
                event_type=EventType.TOOL_CALL, timestamp=YEAR_3000, session_id=session_id,
                event_id="call1", data={"tool_name": "Bash"},
            ))
            for convert in (tree_to_otlp, tree_to_otlp_genai):
                with self.subTest(convert=convert.__name__):
                    with self.assertRaises(UnusableTimestampError) as caught:
                        convert(store, session_id)
                    self.assertEqual(str(caught.exception), f"timestamp {PAST_LIMIT}")


class TestOtlpExportTimeLimit(TimestampCliTestCase):
    """export --format otlp and otlp-genai refuse a time OTLP cannot carry
    before they write or send anything, and keep every time they write in
    range."""

    FORMATS = ("otlp", "otlp-genai")

    def _argvs(self, fmt: str, *flags: str) -> list[list[str]]:
        return [
            ["export", BAD, "--format", fmt, *flags, "--output", "out.json"],
            ["export", BAD, "--format", fmt, *flags, "--endpoint", ENDPOINT],
        ]

    def _export(self, root: Path, fmt: str, *flags: str) -> dict:
        code, out, err, network, written = self.outcome(
            root, ["export", BAD, "--format", fmt, *flags, "--output", "out.json"],
            files="out.json", read=json.loads,
        )
        self.assertEqual((code, out, network), (0, "", 0), err)
        return written["out.json"]

    def test_refuses_session_times_past_the_limit(self):
        for field in ("started_at", "ended_at"):
            for value in (YEAR_3000, PAST_FLOAT):
                for fmt in self.FORMATS:
                    for argv in self._argvs(fmt):
                        with self.subTest(field=field, value=value, argv=argv):
                            root = self.make_ts_workspace()
                            set_meta(root / "work" / ".agent-traces" / BAD, **{field: value})
                            self.assert_refused(root, argv, f"Error: session {field} {PAST_LIMIT}\n")

    def test_refuses_event_timestamps_past_the_limit(self):
        def response2(record):
            return record["event_id"] == "bad-llmresp2"

        cases = [
            ("tool_call", None, YEAR_3000, self.FORMATS,
             "event 6 of 25 (tool_call); 5 more events have timestamps past that time"),
            ("llm_request", None, YEAR_3000, self.FORMATS,
             "event 3 of 25 (llm_request); 1 more event has a timestamp past that time"),
            ("decision", None, YEAR_3000, ("otlp",), "event 10 of 25 (decision)"),
            ("llm_response", response2, PAST_FLOAT, self.FORMATS, "event 23 of 25 (llm_response)"),
        ]
        for event_type, where, value, formats, tail in cases:
            for fmt in formats:
                for argv in self._argvs(fmt):
                    with self.subTest(event_type=event_type, argv=argv):
                        root = self.make_ts_workspace(BAD, event_type, value, where=where)
                        self.assert_refused(
                            root, argv, f"Error: session event timestamp {PAST_LIMIT}: {tail}\n",
                        )

    def test_accepts_the_last_time_that_fits(self):
        for fmt in self.FORMATS:
            with self.subTest(fmt=fmt):
                root = self.make_ts_workspace(
                    BAD, "tool_call", LAST_FLOAT,
                    where=lambda record: record["event_id"] == "bad-call-test2",
                )
                set_meta(root / "work" / ".agent-traces" / BAD,
                         started_at=LAST_FLOAT, ended_at=LAST_FLOAT)
                payload = self._export(root, fmt)
                root_span, *children = all_spans(payload)
                self.assertEqual(span_times(root_span), (LAST_NANOS, LAST_NANOS))
                # The call's result lasts 1 s, which would end past the limit.
                self.assertEqual(
                    [span_times(span) for span in children
                     if span["startTimeUnixNano"] == LAST_NANOS],
                    [(LAST_NANOS, str(UINT64_MAX))],
                )
                assert_unix_nanos_in_range(self, payload)

    def test_root_total_past_the_limit_ends_at_the_start(self):
        for total in (1e14, 1e308):
            for fmt in self.FORMATS:
                with self.subTest(total=total, fmt=fmt):
                    root = self.make_ts_workspace()
                    set_meta(root / "work" / ".agent-traces" / BAD,
                             ended_at=None, total_duration_ms=total)
                    payload = self._export(root, fmt)
                    start, end = span_times(all_spans(payload)[0])
                    self.assertEqual(end, start)
                    assert_unix_nanos_in_range(self, payload)

    def test_subagent_tree_refuses_a_time_past_the_limit(self):
        cases = [
            ("started_at", f"Error: session {CHILD} started_at {PAST_LIMIT}\n"),
            ("ended_at", f"Error: session {CHILD} ended_at {PAST_LIMIT}\n"),
            ("tool_call", f"Error: session {CHILD} event timestamp {PAST_LIMIT}: "
                          "event 3 of 7 (tool_call)\n"),
        ]
        for what, expected in cases:
            for fmt in self.FORMATS:
                for argv in self._argvs(fmt, "--include-subagents"):
                    with self.subTest(what=what, argv=argv):
                        root = self.make_ts_workspace()
                        child = root / "work" / ".agent-traces" / CHILD
                        if what == "tool_call":
                            restamp(child, what, YEAR_3000)
                        else:
                            set_meta(child, **{what: YEAR_3000})
                        self.assert_refused(root, argv, expected)

    def test_subagent_tree_keeps_every_time_in_range(self):
        def tree(root: Path, fmt: str) -> dict:
            payload = self._export(root, fmt, "--include-subagents")
            assert_unix_nanos_in_range(self, payload)
            return payload

        for fmt in self.FORMATS:
            with self.subTest(fmt=fmt):
                times = []
                for duration in (1e14, None):
                    root = self.make_ts_workspace()
                    child = root / "work" / ".agent-traces" / CHILD
                    set_meta(child, ended_at=None, total_duration_ms=1e308)
                    set_event_fields(child, "chld-res", duration_ms=duration)
                    spans = all_spans(tree(root, fmt))
                    child_root = [span for span in spans
                                  if span["spanId"] == _to_span_id(f"root-{CHILD}")]
                    self.assertEqual(len(child_root), 1)
                    start, end = span_times(child_root[0])
                    self.assertEqual(end, start)
                    times.append([span_times(span) for span in spans])
                # A result duration past the limit reads as a missing one.
                self.assertEqual(times[0], times[1])


if __name__ == "__main__":
    unittest.main()
