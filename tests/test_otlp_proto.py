"""Tests for the OTLP/HTTP protocol choice and the protobuf encoder."""

from __future__ import annotations

import base64
import copy
import json
import unittest
from pathlib import Path

from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.otlp import node_tree_to_otlp, session_to_otlp, session_to_otlp_genai
from agent_trace.otlp_proto import (
    EncodeError,
    encode_body,
    encode_trace_request,
    resolve_protocol,
)
from agent_trace.subagent import SessionNode
from agent_trace.temporal import session_to_temporal_otlp

try:
    from google.protobuf.json_format import ParseDict
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
        ExportTraceServiceRequest,
    )
except ImportError:  # the official package is not a dependency
    ParseDict = ExportTraceServiceRequest = None

SOURCE_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "otlp_genai" / "session.json"
SECRET = "SECRET-SENTINEL"
TRACE = "0102030405060708090a0b0c0d0e0f10"
TRACES_VAR = "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL"
GENERIC_VAR = "OTEL_EXPORTER_OTLP_PROTOCOL"

# One span that uses every kind of field the exporters write and some they
# do not: an AnyValue of each kind, a dropped count, an event, a link with
# flags and an attribute, a status, span flags, and fields left at their
# defaults.
GOLDEN = {
    "resourceSpans": [{
        "resource": {"attributes": [{"key": "service.name", "value": {"stringValue": "agent-trace"}}]},
        "scopeSpans": [{
            "scope": {"name": "agent-trace", "version": "1.0"},
            "spans": [{
                "traceId": TRACE,
                "spanId": "1112131415161718",
                "parentSpanId": "2122232425262728",
                "traceState": "",
                "name": "chat m",
                "kind": 3,
                "startTimeUnixNano": "1700000000000000000",
                "endTimeUnixNano": 1700000001000000000,
                "attributes": [
                    {"key": "s", "value": {"stringValue": "\u00e9"}},
                    {"key": "b", "value": {"boolValue": False}},
                    {"key": "i", "value": {"intValue": "-5"}},
                    {"key": "d", "value": {"doubleValue": 0.25}},
                    {"key": "a", "value": {"arrayValue": {"values": [
                        {"intValue": "0"}, {"stringValue": ""},
                    ]}}},
                    {"key": "k", "value": {"kvlistValue": {"values": [
                        {"key": "raw", "value": {"bytesValue": "+/8="}},
                    ]}}},
                ],
                "droppedAttributesCount": 2,
                "events": [{
                    "timeUnixNano": "1700000000500000000",
                    "name": "exception",
                    "attributes": [{"key": "exception.type", "value": {"stringValue": "E"}}],
                }],
                "droppedEventsCount": 0,
                "links": [{
                    "traceId": TRACE,
                    "spanId": "3132333435363738",
                    "attributes": [{"key": "agent.strace.link.type", "value": {"stringValue": "retry_of"}}],
                    "flags": 257,
                }],
                "status": {"code": "STATUS_CODE_ERROR", "message": "boom"},
                "flags": 256,
            }],
        }],
    }],
}
# GOLDEN as serialized by opentelemetry-proto 1.45.2 (protobuf 7.36.2):
# ParseDict, with the hex ids as base64, then SerializeToString.
GOLDEN_HEX = (
    "0ae2020a1f0a1d0a0c736572766963652e6e616d65120d0a0b6167656e742d747261636512be020a"
    "120a0b6167656e742d74726163651203312e3012a7020a100102030405060708090a0b0c0d0e0f10"
    "12081112131415161718220821222324252627282a0663686174206d30033900002a36fe9c971741"
    "00cac471fe9c97174a090a017312040a02c3a94a070a0162120210004a100a0169120b18fbffffff"
    "ffffffffff014a0e0a0164120921000000000000d03f4a0f0a0161120a2a080a0218000a020a004a"
    "140a016b120f320d0a0b0a0372617712043a02fbff50025a2b090065f753fe9c9717120965786365"
    "7074696f6e1a150a0e657863657074696f6e2e7479706512030a01456a470a100102030405060708"
    "090a0b0c0d0e0f101208313233343536373822240a166167656e742e7374726163652e6c696e6b2e"
    "74797065120a0a0872657472795f6f6635010100007a081204626f6f6d1802850100010000"
)


def _span(payload: dict) -> dict:
    return payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]


def _span_list(payload: dict) -> list[dict]:
    return [span for rs in payload["resourceSpans"] for ss in rs["scopeSpans"] for span in ss["spans"]]


def _with_span(**fields: object) -> dict:
    return {"resourceSpans": [{"scopeSpans": [{"spans": [fields]}]}]}


def _with_value(value: object) -> dict:
    return _with_span(attributes=[{"key": "k", "value": value}])


class TestEncodeTraceRequest(unittest.TestCase):
    def test_golden_bytes(self):
        self.assertEqual(encode_trace_request(GOLDEN).hex(), GOLDEN_HEX)

    def test_other_spellings_encode_the_same(self):
        # Proto field names, enum names for numbers and numbers for names,
        # 64-bit integers as numbers or strings, nulls for defaults,
        # upper-case hex and unpadded URL-safe base64.
        variant = copy.deepcopy(GOLDEN)
        resource_spans = variant.pop("resourceSpans")
        variant["resource_spans"] = resource_spans
        resource_spans[0]["scope_spans"] = resource_spans[0].pop("scopeSpans")
        resource_spans[0]["schemaUrl"] = None
        span = resource_spans[0]["scope_spans"][0]["spans"][0]
        span["trace_id"] = span.pop("traceId").upper()
        span["kind"] = "SPAN_KIND_CLIENT"
        span["startTimeUnixNano"] = 1700000000000000000
        span["end_time_unix_nano"] = str(span.pop("endTimeUnixNano"))
        span["traceState"] = None
        span["status"] = {"code": 2, "message": "boom"}
        span["flags"] = "256"
        attributes = span["attributes"]
        attributes[2]["value"] = {"int_value": -5}
        attributes[3]["value"] = {"doubleValue": "0.25"}
        attributes[5]["value"]["kvlistValue"]["values"][0]["value"] = {"bytesValue": "-_8"}
        span["events"][0]["time_unix_nano"] = float(span["events"][0].pop("timeUnixNano"))
        self.assertEqual(encode_trace_request(variant).hex(), GOLDEN_HEX)

    def test_defaults(self):
        # Scalars at their defaults are left out. A message that is present
        # is written even when it is empty, and an AnyValue keeps a default
        # value so the receiver sees its kind. Checked against
        # opentelemetry-proto like GOLDEN_HEX.
        cases = (
            ({}, ""),
            ({"resourceSpans": [{}]}, "0a00"),
            (_with_span(name="", kind=0, flags=0, status={}), "0a06120412027a00"),
            (_with_value({"boolValue": False}), "0a0d120b12094a070a016b12021000"),
        )
        for payload, expected in cases:
            with self.subTest(payload=payload):
                self.assertEqual(encode_trace_request(payload).hex(), expected)

    def test_text_that_is_not_utf8(self):
        # A lone surrogate is sent as U+FFFD, as a JSON receiver reads the
        # escape json.dumps writes; a split pair is joined.
        cases = (
            ("a\udc80b", "a\ufffdb"),
            ("\ud83d\ude00", "\U0001F600"),
        )
        for text, sent in cases:
            with self.subTest(text=text):
                self.assertEqual(
                    encode_trace_request(_with_value({"stringValue": text})),
                    encode_trace_request(_with_value({"stringValue": sent})),
                )

    def test_refusals_name_the_field_and_never_the_value(self):
        span = "resourceSpans[0].scopeSpans[0].spans[0]"
        value = f"{span}.attributes[0].value"
        cases = (
            ([SECRET], "expected an object"),
            ({"resourceSpans": SECRET}, "resourceSpans: expected an array"),
            ({"resourceSpans": [SECRET]}, "resourceSpans[0]: expected an object"),
            ({"resourceSpans": [{"bogus": SECRET}]},
             "resourceSpans[0]: ResourceSpans has no field 'bogus'"),
            ({"resourceSpans": [], "resource_spans": [SECRET]},
             "field resource_spans is given twice"),
            (_with_span(traceId=SECRET), f"{span}.traceId: expected an even number of hex digits"),
            (_with_span(spanId="abc"), f"{span}.spanId: expected an even number of hex digits"),
            (_with_span(kind=SECRET), f"{span}.kind: expected an integer"),
            (_with_span(startTimeUnixNano=SECRET), f"{span}.startTimeUnixNano: expected an integer"),
            (_with_span(startTimeUnixNano=1.5), f"{span}.startTimeUnixNano: expected an integer"),
            (_with_span(startTimeUnixNano=-1),
             f"{span}.startTimeUnixNano: integer out of range for fixed64"),
            (_with_span(droppedAttributesCount=True),
             f"{span}.droppedAttributesCount: expected an integer, got a boolean"),
            (_with_span(name=7), f"{span}.name: expected a string"),
            (_with_value({"intValue": SECRET}), f"{value}.intValue: expected an integer"),
            (_with_value({"intValue": str(1 << 63)}),
             f"{value}.intValue: integer out of range for int64"),
            (_with_value({"doubleValue": SECRET}), f"{value}.doubleValue: expected a number"),
            (_with_value({"doubleValue": True}),
             f"{value}.doubleValue: expected a number, got a boolean"),
            (_with_value({"doubleValue": 10 ** 400}),
             f"{value}.doubleValue: number out of range for double"),
            (_with_value({"boolValue": SECRET}), f"{value}.boolValue: expected a boolean"),
            (_with_value({"bytesValue": SECRET + "!"}), f"{value}.bytesValue: expected base64"),
            (_with_value({"bytesValue": "QUJDR"}), f"{value}.bytesValue: expected base64"),
            (_with_value({"stringValue": SECRET, "intValue": "1"}),
             f"{value}: more than one value is set"),
        )
        for payload, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(EncodeError) as caught:
                    encode_trace_request(payload)
                self.assertEqual(str(caught.exception), message)
                self.assertNotIn(SECRET, str(caught.exception))

    def test_encode_body(self):
        self.assertEqual(encode_body(GOLDEN, "http/json"), json.dumps(GOLDEN).encode("utf-8"))
        self.assertEqual(encode_body(GOLDEN, "http/protobuf").hex(), GOLDEN_HEX)


class TestResolveProtocol(unittest.TestCase):
    def test_precedence(self):
        cases = (
            (None, {}, "http/json"),
            (None, {GENERIC_VAR: "http/protobuf"}, "http/protobuf"),
            (None, {TRACES_VAR: "http/protobuf", GENERIC_VAR: "http/json"}, "http/protobuf"),
            (None, {TRACES_VAR: "http/json", GENERIC_VAR: "http/protobuf"}, "http/json"),
            # An empty variable counts as unset.
            (None, {TRACES_VAR: " ", GENERIC_VAR: "http/protobuf"}, "http/protobuf"),
            (None, {TRACES_VAR: "", GENERIC_VAR: ""}, "http/json"),
            ("http/json", {TRACES_VAR: "http/protobuf"}, "http/json"),
            ("http/protobuf", {GENERIC_VAR: "http/json"}, "http/protobuf"),
            # Only the value that wins is checked.
            ("http/json", {TRACES_VAR: "grpc", GENERIC_VAR: SECRET}, "http/json"),
            (None, {TRACES_VAR: "http/protobuf", GENERIC_VAR: "grpc"}, "http/protobuf"),
            # Case and surrounding spaces do not matter.
            (" HTTP/Protobuf ", {}, "http/protobuf"),
            (None, {TRACES_VAR: "HTTP/JSON\n"}, "http/json"),
        )
        for flag, environ, expected in cases:
            with self.subTest(flag=flag, environ=environ):
                self.assertEqual(resolve_protocol(flag, environ), expected)

    def test_refusals_never_repeat_the_value(self):
        grpc = "agent-strace sends OTLP over HTTP only; use http/protobuf or http/json"
        override = "; or pass --protocol to override it"
        cases = (
            ("grpc", {}, f"--protocol grpc is not supported: {grpc}"),
            (" GRPC", {}, f"--protocol grpc is not supported: {grpc}"),
            (SECRET, {}, "invalid --protocol value: use http/json or http/protobuf"),
            ("", {}, "invalid --protocol value: use http/json or http/protobuf"),
            (None, {TRACES_VAR: "grpc"},
             f"{TRACES_VAR} grpc is not supported: {grpc}{override}"),
            (None, {GENERIC_VAR: "grpc"},
             f"{GENERIC_VAR} grpc is not supported: {grpc}{override}"),
            (None, {TRACES_VAR: SECRET, GENERIC_VAR: "http/json"},
             f"invalid {TRACES_VAR} value: use http/json or http/protobuf{override}"),
            (None, {GENERIC_VAR: "http/protobuf;" + SECRET},
             f"invalid {GENERIC_VAR} value: use http/json or http/protobuf{override}"),
        )
        for flag, environ, message in cases:
            with self.subTest(flag=flag, environ=environ):
                with self.assertRaises(ValueError) as caught:
                    resolve_protocol(flag, environ)
                self.assertEqual(str(caught.exception), message)


_HEX_KEYS = {"traceId", "trace_id", "spanId", "span_id", "parentSpanId", "parent_span_id"}


def _for_parse_dict(value: object) -> object:
    """Return *value* with hex ids as base64, the form ParseDict reads for bytes."""
    if isinstance(value, dict):
        return {
            key: base64.b64encode(bytes.fromhex(item)).decode("ascii")
            if key in _HEX_KEYS and isinstance(item, str) else _for_parse_dict(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_for_parse_dict(item) for item in value]
    return value


def _retry_events() -> list[TraceEvent]:
    """A tool call and a model request that each fail once and are retried."""
    def event(event_type, timestamp, event_id="", parent_id="", **data):
        return TraceEvent(event_type=event_type, event_id=event_id or f"e{timestamp}",
                          parent_id=parent_id, timestamp=timestamp, data=data)

    return [
        event(EventType.TOOL_CALL, 1001.0, "call1", tool_name="Bash", arguments={"command": "make"}),
        event(EventType.ERROR, 1002.0, parent_id="call1", error="exit 2"),
        event(EventType.TOOL_CALL, 1003.0, "call2", tool_name="Bash",
              arguments={"command": "make"}, retry_of="call1"),
        event(EventType.TOOL_RESULT, 1004.0, parent_id="call2", result="ok"),
        event(EventType.LLM_REQUEST, 1005.0, "req1", model="gpt-4o"),
        event(EventType.ERROR, 1006.0, parent_id="req1", error="overloaded"),
        event(EventType.LLM_REQUEST, 1007.0, "req2", model="gpt-4o", retry_of="req1"),
        event(EventType.LLM_RESPONSE, 1008.0, parent_id="req2", model="gpt-4o"),
    ]


def _fixture_payloads() -> dict[str, dict]:
    source = json.loads(SOURCE_FIXTURE.read_text(encoding="utf-8"))
    meta = SessionMeta.from_json(json.dumps(source["meta"]))
    events = [TraceEvent.from_json(json.dumps(item)) for item in source["events"]]
    temporal_meta = copy.deepcopy(meta)
    temporal_meta.trace_id = "4bf92f3577b34da6a3ce929d0e0e4736"
    temporal_meta.parent_span_id = "00f067aa0ba902b7"
    temporal_meta.trace_flags = "01"
    retry_meta = SessionMeta(session_id="retry", started_at=1000.0, ended_at=1010.0)
    spawn = next(event for event in events if event.event_type == EventType.TOOL_CALL)
    child = SessionNode(
        meta=SessionMeta(session_id="child", parent_session_id=meta.session_id,
                         parent_event_id=spawn.event_id, started_at=1000.0, ended_at=1010.0),
        events=_retry_events(),
    )
    tree = SessionNode(meta=meta, events=events, children=[child])
    return {
        "otlp": session_to_otlp(meta, events),
        "otlp-genai": session_to_otlp_genai(meta, events),
        "temporal": session_to_temporal_otlp(temporal_meta, events),
        "otlp retry links": session_to_otlp(retry_meta, _retry_events()),
        "otlp-genai retry links": session_to_otlp_genai(retry_meta, _retry_events()),
        "otlp --include-subagents": node_tree_to_otlp(tree),
        "otlp-genai --include-subagents": node_tree_to_otlp(tree, genai=True),
    }


@unittest.skipIf(ParseDict is None, "opentelemetry-proto is not installed")
class TestAgainstOfficialSchema(unittest.TestCase):
    """The bytes equal what the official opentelemetry-proto package makes."""

    def assertOfficial(self, payload: dict) -> None:
        official = ParseDict(_for_parse_dict(payload), ExportTraceServiceRequest())
        self.assertEqual(encode_trace_request(payload), official.SerializeToString())

    def test_golden(self):
        self.assertOfficial(GOLDEN)
        self.assertEqual(
            ParseDict(_for_parse_dict(GOLDEN), ExportTraceServiceRequest()).SerializeToString().hex(),
            GOLDEN_HEX,
        )

    def test_exporter_payloads(self):
        payloads = _fixture_payloads()
        for name, payload in payloads.items():
            with self.subTest(format=name):
                self.assertTrue(_span(payload))
                self.assertOfficial(payload)
        for name in ("otlp retry links", "otlp-genai retry links"):
            with self.subTest(format=name):
                spans = _span_list(payloads[name])
                links = [link for span in spans for link in span.get("links", [])]
                self.assertTrue(links)
                self.assertTrue(all(link["attributes"] for link in links))
        for name in ("otlp --include-subagents", "otlp-genai --include-subagents"):
            with self.subTest(format=name):
                self.assertEqual(len({span["traceId"] for span in _span_list(payloads[name])}), 1)


if __name__ == "__main__":
    unittest.main()
