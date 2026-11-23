"""Tests for OTLP GenAI semantic conventions export (issue #100)."""

import json
import os
import sys
import tempfile
import time
import unittest

from agent_trace import __version__
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.otlp import (
    _canonical_provider,
    _to_span_id,
    _ts_to_nanos,
    session_to_otlp,
    session_to_otlp_genai,
)
from agent_trace.store import TraceStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_meta(agent_name: str = "claude-code") -> SessionMeta:
    meta = SessionMeta(agent_name=agent_name)
    meta.started_at = time.time() - 60
    meta.ended_at = time.time()
    return meta


def _make_event(event_type: EventType, **data) -> TraceEvent:
    return TraceEvent(event_type=event_type, data=data)


def _get_spans(payload: dict) -> list[dict]:
    return payload["resourceSpans"][0]["scopeSpans"][0]["spans"]


def _get_attr(span: dict, key: str):
    for attr in span.get("attributes", []):
        if attr["key"] == key:
            v = attr["value"]
            if "arrayValue" in v:
                return [item["stringValue"] for item in v["arrayValue"]["values"]]
            return (
                v.get("stringValue")
                or v.get("intValue")
                or v.get("doubleValue")
                or v.get("boolValue")
            )
    return None


# ---------------------------------------------------------------------------
# Root span attributes
# ---------------------------------------------------------------------------

class TestGenAIRootSpan(unittest.TestCase):
    def test_root_span_is_invoke_agent_without_agent_id(self):
        meta = _make_meta()
        payload = session_to_otlp_genai(meta, [])
        root = _get_spans(payload)[0]
        self.assertEqual(root["name"], "invoke_agent claude-code")
        self.assertEqual(root["kind"], 1)
        self.assertEqual(_get_attr(root, "gen_ai.operation.name"), "invoke_agent")
        # gen_ai.agent.id names a hosted agent resource, not a session.
        self.assertIsNone(_get_attr(root, "gen_ai.agent.id"))
        self.assertEqual(_get_attr(root, "agent.session_id"), meta.session_id)

    def test_root_span_has_gen_ai_agent_name(self):
        meta = _make_meta("my-agent")
        payload = session_to_otlp_genai(meta, [])
        spans = _get_spans(payload)
        root = spans[0]
        self.assertEqual(_get_attr(root, "gen_ai.agent.name"), "my-agent")

    def test_unnamed_agent_has_no_agent_name(self):
        payload = session_to_otlp_genai(_make_meta(""), [])
        root = _get_spans(payload)[0]
        self.assertEqual(root["name"], "invoke_agent")
        self.assertIsNone(_get_attr(root, "gen_ai.agent.name"))

    def test_root_span_has_provider_name_and_deprecated_system(self):
        meta = _make_meta("claude-code")
        payload = session_to_otlp_genai(meta, [])
        spans = _get_spans(payload)
        root = spans[0]
        self.assertEqual(_get_attr(root, "gen_ai.provider.name"), "anthropic")
        self.assertEqual(_get_attr(root, "gen_ai.system"), "anthropic")

    def test_scope_version_is_agent_trace_version(self):
        meta = _make_meta()
        payload = session_to_otlp_genai(meta, [])
        scope = payload["resourceSpans"][0]["scopeSpans"][0]["scope"]
        self.assertEqual(scope, {"name": "agent-trace", "version": __version__})


# ---------------------------------------------------------------------------
# LLM request/response → chat {model} child span
# ---------------------------------------------------------------------------

class TestGenAILLMSpan(unittest.TestCase):
    def _make_llm_pair(self, model: str = "claude-3-5-sonnet") -> list[TraceEvent]:
        req = TraceEvent(
            event_type=EventType.LLM_REQUEST,
            data={"model": model, "input_tokens": 100, "max_tokens": 4096},
        )
        resp = TraceEvent(
            event_type=EventType.LLM_RESPONSE,
            parent_id=req.event_id,
            data={"model": model, "output_tokens": 50, "stop_reason": "end_turn"},
        )
        return [req, resp]

    def test_llm_pair_produces_child_span(self):
        meta = _make_meta()
        events = self._make_llm_pair()
        payload = session_to_otlp_genai(meta, events)
        spans = _get_spans(payload)
        # root + 1 LLM span
        self.assertEqual(len(spans), 2)
        llm_span = spans[1]
        self.assertEqual(llm_span["name"], "chat claude-3-5-sonnet")
        self.assertEqual(llm_span["kind"], 3)
        self.assertEqual(_get_attr(llm_span, "gen_ai.operation.name"), "chat")

    def test_llm_span_without_model_is_named_by_operation(self):
        req = TraceEvent(event_type=EventType.LLM_REQUEST, data={"input_tokens": 1})
        resp = TraceEvent(
            event_type=EventType.LLM_RESPONSE,
            parent_id=req.event_id,
            data={"output_tokens": 1},
        )
        unmatched = TraceEvent(event_type=EventType.LLM_REQUEST, data={})
        spans = _get_spans(session_to_otlp_genai(_make_meta(), [req, resp, unmatched]))
        self.assertEqual([span["name"] for span in spans[1:]], ["chat", "chat"])

    def test_llm_span_has_model_attribute(self):
        meta = _make_meta()
        events = self._make_llm_pair("claude-3-5-sonnet")
        payload = session_to_otlp_genai(meta, events)
        llm_span = _get_spans(payload)[1]
        self.assertEqual(_get_attr(llm_span, "gen_ai.request.model"), "claude-3-5-sonnet")

    def test_llm_span_has_token_attributes(self):
        meta = _make_meta()
        events = self._make_llm_pair()
        payload = session_to_otlp_genai(meta, events)
        llm_span = _get_spans(payload)[1]
        self.assertIsNotNone(_get_attr(llm_span, "gen_ai.usage.input_tokens"))
        self.assertIsNotNone(_get_attr(llm_span, "gen_ai.usage.output_tokens"))

    def test_llm_span_has_finish_reason(self):
        meta = _make_meta()
        events = self._make_llm_pair()
        payload = session_to_otlp_genai(meta, events)
        llm_span = _get_spans(payload)[1]
        self.assertEqual(_get_attr(llm_span, "gen_ai.response.finish_reasons"), ["end_turn"])

    def test_finish_reasons_is_a_string_array_in_recorded_order(self):
        req, resp = self._make_llm_pair()
        resp.data["stop_reason"] = ["stop", "length", 3]
        llm_span = _get_spans(session_to_otlp_genai(_make_meta(), [req, resp]))[1]
        (attr,) = [
            a for a in llm_span["attributes"] if a["key"] == "gen_ai.response.finish_reasons"
        ]
        self.assertEqual(attr["value"], {"arrayValue": {"values": [
            {"stringValue": "stop"}, {"stringValue": "length"}, {"stringValue": "3"},
        ]}})
        self.assertEqual(
            _get_attr(llm_span, "agent.strace.finish_reason"), '["stop", "length", 3]'
        )

    def test_openinference_finish_reason_is_one_string(self):
        llm_span = _get_spans(session_to_otlp_genai(_make_meta(), self._make_llm_pair()))[1]
        (attr,) = [a for a in llm_span["attributes"] if a["key"] == "llm.finish_reason"]
        self.assertEqual(attr["value"], {"stringValue": "end_turn"})
        self.assertEqual(_get_attr(llm_span, "gen_ai.response.finish_reasons"), ["end_turn"])
        self.assertEqual(_get_attr(llm_span, "agent.strace.finish_reason"), "end_turn")

    def test_openinference_finish_reason_is_the_first_recorded_reason(self):
        for recorded, expected in (
            (["length", "stop"], "length"),
            ([3, "stop"], "3"),
        ):
            with self.subTest(recorded=recorded):
                req, resp = self._make_llm_pair()
                del resp.data["stop_reason"]
                resp.data["finish_reason"] = recorded
                llm_span = _get_spans(session_to_otlp_genai(_make_meta(), [req, resp]))[1]
                self.assertEqual(_get_attr(llm_span, "llm.finish_reason"), expected)
                self.assertEqual(
                    _get_attr(llm_span, "gen_ai.response.finish_reasons"),
                    [str(item) for item in recorded],
                )
                self.assertEqual(
                    _get_attr(llm_span, "agent.strace.finish_reason"), json.dumps(recorded)
                )

    def test_openinference_finish_reason_only_when_recorded(self):
        meta = SessionMeta(session_id="s1", agent_name="codex", started_at=1000.0, ended_at=1100.0)
        events = [
            _codex_response("a", 1001.0, status="completed", stop_reason="stop"),
            _codex_response("b", 1002.0, status="completed", output_tokens=2),
            _codex_response("c", 1003.0, status="failed", error="boom"),
            _codex_response("d", 1004.0, status="completed", stop_reason=[""]),
            TraceEvent(
                event_type=EventType.LLM_REQUEST, event_id="e", timestamp=1005.0,
                data={"model": "gpt-5-codex", "input_tokens": 3},
            ),
        ]
        spans = [
            span for span in _get_spans(session_to_otlp_genai(meta, events))
            if _get_attr(span, "openinference.span.kind") == "LLM"
        ]
        self.assertEqual(
            [_get_attr(span, "llm.finish_reason") for span in spans],
            ["stop", None, None, None, None],
        )
        self.assertEqual(
            [_get_attr(span, "gen_ai.response.finish_reasons") for span in spans],
            [["stop"], None, ["error"], [""], None],
        )

    def test_llm_span_parented_to_root(self):
        meta = _make_meta()
        events = self._make_llm_pair()
        payload = session_to_otlp_genai(meta, events)
        spans = _get_spans(payload)
        root_id = spans[0]["spanId"]
        llm_span = spans[1]
        self.assertEqual(llm_span["parentSpanId"], root_id)

    def test_gen_ai_system_derived_from_model(self):
        meta = _make_meta("unknown-agent")
        req = TraceEvent(event_type=EventType.LLM_REQUEST, data={"model": "gpt-4o"})
        resp = TraceEvent(
            event_type=EventType.LLM_RESPONSE,
            parent_id=req.event_id,
            data={"model": "gpt-4o", "output_tokens": 10},
        )
        payload = session_to_otlp_genai(meta, [req, resp])
        llm_span = _get_spans(payload)[1]
        self.assertEqual(_get_attr(llm_span, "gen_ai.provider.name"), "openai")
        self.assertEqual(_get_attr(llm_span, "gen_ai.system"), "openai")


class TestGenAIProviderName(unittest.TestCase):
    def test_well_known_provider_values(self):
        cases = {
            "gemini": "gcp.gen_ai",
            "gemini-2.5-pro": "gcp.gen_ai",
            "google": "gcp.gen_ai",
            "vertex_ai/gemini-2.5-pro": "gcp.vertex_ai",
            "gemini/gemini-2.5-flash": "gcp.gemini",
            "gcp.gemini": "gcp.gemini",
            "bedrock/anthropic.claude-3-5-sonnet": "aws.bedrock",
            "amazon.nova-pro-v1:0": "aws.bedrock",
            "azure/gpt-4o": "azure.ai.openai",
            "azure": "azure.ai.openai",
            "azure_ai/mistral-large": "azure.ai.inference",
            "claude-sonnet-4": "anthropic",
            "gpt-4o": "openai",
            "o3-mini": "openai",
            "mistral-large-latest": "mistral",
            "codestral-latest": "mistral",
            "deepseek-chat": "deepseek",
            "grok-4": "x_ai",
            "xai/grok-2": "x_ai",
            "groq/llama-3.1-70b": "groq",
            "ollama/llama3": "ollama",
            "kimi-k2": "moonshot_ai",
            "copilot": "unknown",
            "": "unknown",
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(_canonical_provider(value), expected)

    def test_gemini_session_uses_gcp_gen_ai(self):
        req = TraceEvent(event_type=EventType.LLM_REQUEST, data={"model": "gemini-2.5-pro"})
        resp = TraceEvent(
            event_type=EventType.LLM_RESPONSE,
            parent_id=req.event_id,
            data={"model": "gemini-2.5-pro", "output_tokens": 1},
        )
        spans = _get_spans(session_to_otlp_genai(_make_meta("gemini"), [req, resp]))
        for span in spans:
            with self.subTest(span=span["name"]):
                self.assertEqual(_get_attr(span, "gen_ai.provider.name"), "gcp.gen_ai")
                self.assertEqual(_get_attr(span, "gen_ai.system"), "gcp.gen_ai")
        self.assertEqual(_get_attr(spans[1], "llm.provider"), "gcp.gen_ai")


class TestGenAIAgentProductProvider(unittest.TestCase):
    """A multi-model agent product names no provider; the model decides."""

    def _llm_span(self, meta: SessionMeta, **data) -> dict:
        resp = TraceEvent(event_type=EventType.LLM_RESPONSE, data=data)
        return _get_spans(session_to_otlp_genai(meta, [resp]))[1]

    def _provider(self, span: dict) -> tuple:
        return tuple(
            _get_attr(span, key)
            for key in (
                "gen_ai.provider.name", "gen_ai.system", "llm.provider",
                "agent.strace.provider",
            )
        )

    def test_product_provider_uses_the_model(self):
        cases = [
            ("openai-codex", "codex", "gpt-5.3-codex", ("openai", "openai", "openai", "codex")),
            ("github-copilot", "copilot", "claude-sonnet-4.5",
             ("anthropic", "anthropic", "anthropic", "copilot")),
            ("github-copilot", "copilot", "gpt-5", ("openai", "openai", "openai", "copilot")),
            ("cursor-agent", "cursor", "gemini-2.5-pro",
             ("gcp.gen_ai", "gcp.gen_ai", "gcp.gen_ai", "cursor")),
        ]
        for agent, provider, model, expected in cases:
            with self.subTest(provider=provider, model=model):
                span = self._llm_span(_make_meta(agent), provider=provider, model=model)
                self.assertEqual(self._provider(span), expected)

    def test_product_provider_without_a_known_model_is_unknown(self):
        for provider, model in (("codex", "synthetic-model"), ("codex", ""), ("copilot", "")):
            with self.subTest(provider=provider, model=model):
                span = self._llm_span(_make_meta(), provider=provider, model=model)
                self.assertEqual(
                    self._provider(span), ("unknown", "unknown", provider, provider)
                )

    def test_recorded_model_provider_still_wins(self):
        span = self._llm_span(_make_meta("openai-codex"), provider="azure", model="gpt-4o")
        self.assertEqual(_get_attr(span, "gen_ai.provider.name"), "azure.ai.openai")

    def test_product_root_is_unknown(self):
        hook = {"agent_provider": "codex"}
        for agent, attribution in (
            ("openai-codex", {}), ("openai-codex", hook), ("github-copilot", {}),
            ("github-copilot", {"agent_provider": "copilot"}), ("cursor-agent", {}),
        ):
            with self.subTest(agent=agent, attribution=attribution):
                meta = _make_meta(agent)
                meta.attribution = attribution
                root = _get_spans(session_to_otlp_genai(meta, []))[0]
                self.assertEqual(_get_attr(root, "gen_ai.provider.name"), "unknown")
        claude = _make_meta("claude-code")
        claude.attribution = {"agent_provider": "claude-code"}
        root = _get_spans(session_to_otlp_genai(claude, []))[0]
        self.assertEqual(_get_attr(root, "gen_ai.provider.name"), "anthropic")


class TestGenAILLMPairingFallback(unittest.TestCase):
    """Parentless responses follow the pairing.py LLM fallback."""

    def _req(self, event_id, ts, model="gpt-4o"):
        return TraceEvent(
            event_type=EventType.LLM_REQUEST, event_id=event_id, timestamp=ts,
            data={"model": model, "input_tokens": 1},
        )

    def _resp(self, ts, parent_id="", model="gpt-4o"):
        return TraceEvent(
            event_type=EventType.LLM_RESPONSE, parent_id=parent_id, timestamp=ts,
            data={"model": model, "output_tokens": 1},
        )

    def _span_starts(self, events):
        # A paired request carries its response's output tokens; an
        # unmatched one does not. Both leave status unset.
        spans = _get_spans(session_to_otlp_genai(_make_meta(), events))[1:]
        return {
            s["spanId"]: (
                s["startTimeUnixNano"],
                _get_attr(s, "gen_ai.usage.output_tokens") is not None,
            )
            for s in spans
        }

    def test_parentless_response_pairs_with_newest_open_request(self):
        events = [self._req("failed", 1.0), self._req("ok", 2.0), self._resp(3.0)]
        spans = self._span_starts(events)
        self.assertEqual(spans[_to_span_id("ok")], (_ts_to_nanos(2.0), True))
        self.assertEqual(spans[_to_span_id("failed")], (_ts_to_nanos(1.0), False))

    def test_parentless_response_skips_requests_for_other_models(self):
        events = [
            self._req("gpt", 1.0, "gpt-4o"),
            self._req("claude", 2.0, "claude-opus-4-6"),
            self._resp(3.0, model="gpt-4o"),
        ]
        spans = self._span_starts(events)
        self.assertEqual(spans[_to_span_id("gpt")], (_ts_to_nanos(1.0), True))
        self.assertEqual(spans[_to_span_id("claude")], (_ts_to_nanos(2.0), False))

    def test_duplicate_response_does_not_take_another_request(self):
        events = [
            self._req("a", 1.0),
            self._req("b", 2.0),
            self._resp(3.0, parent_id="a"),
            self._resp(4.0, parent_id="a"),
        ]
        spans = self._span_starts(events)
        self.assertEqual(spans[_to_span_id("b")], (_ts_to_nanos(2.0), False))


class TestGenAIUnrequestedResponses(unittest.TestCase):
    """Codex records each model request only as an llm_response."""

    def _events(self, mode="openai-codex-hooks"):
        return [
            TraceEvent(event_type=EventType.SESSION_START, timestamp=1.0, data={"mode": mode}),
            TraceEvent(event_type=EventType.LLM_RESPONSE, timestamp=2.0, data={
                "provider": "codex", "model": "gpt-5-codex",
                "input_tokens": 1200, "output_tokens": 10,
            }),
            TraceEvent(event_type=EventType.LLM_RESPONSE, timestamp=3.0, data={
                "provider": "codex", "status": "failed", "token_usage": None,
            }),
            TraceEvent(event_type=EventType.SESSION_END, timestamp=4.0, data={}),
        ]

    def test_response_span_carries_its_own_usage(self):
        spans = _get_spans(session_to_otlp_genai(_make_meta("openai-codex"), self._events()))
        llm = spans[1]
        self.assertEqual(_get_attr(llm, "gen_ai.request.model"), "gpt-5-codex")
        self.assertEqual(_get_attr(llm, "gen_ai.usage.input_tokens"), "1200")
        self.assertEqual(_get_attr(llm, "gen_ai.usage.output_tokens"), "10")

    def test_declared_responses_are_a_provider_limitation_not_orphans(self):
        root = _get_spans(session_to_otlp_genai(_make_meta("openai-codex"), self._events()))[0]
        self.assertEqual(
            json.loads(_get_attr(root, "agent.strace.evidence.gap_codes")),
            ["declared_results_without_requests"],
        )
        gaps = json.loads(_get_attr(root, "agent.strace.evidence.gaps"))
        self.assertEqual(gaps[0]["kind"], "provider_limitation")

    def test_undeclared_responses_stay_orphans(self):
        events = self._events(mode="sdk")
        root = _get_spans(session_to_otlp_genai(_make_meta("my-agent"), events))[0]
        self.assertEqual(
            json.loads(_get_attr(root, "agent.strace.evidence.gap_codes")),
            ["orphan_llm_response", "orphan_llm_response"],
        )


class TestGenAIEventsBeforeStart(unittest.TestCase):
    """Copilot can send its first user_prompt before sessionStart."""

    def _gap_codes(self, pre_start, mode="github-copilot-hooks", agent="github-copilot"):
        events = [
            TraceEvent(event_type=event_type, timestamp=1.0, data={})
            for event_type in pre_start
        ] + [
            TraceEvent(event_type=EventType.SESSION_START, timestamp=2.0, data={"mode": mode}),
            TraceEvent(event_type=EventType.SESSION_END, timestamp=3.0, data={}),
        ]
        root = _get_spans(session_to_otlp_genai(_make_meta(agent), events))[0]
        return json.loads(_get_attr(root, "agent.strace.evidence.gap_codes"))

    def test_declared_order_is_a_provider_limitation(self):
        self.assertEqual(
            self._gap_codes([EventType.USER_PROMPT]),
            ["declared_events_before_session_start"],
        )

    def test_undeclared_type_before_start_is_still_a_gap(self):
        self.assertEqual(
            self._gap_codes([EventType.USER_PROMPT, EventType.FILE_READ]),
            ["events_before_session_start"],
        )

    def test_trace_without_hook_provider_is_unchanged(self):
        self.assertEqual(
            self._gap_codes([EventType.USER_PROMPT], mode="sdk", agent="my-agent"),
            ["events_before_session_start"],
        )


# ---------------------------------------------------------------------------
# Tool call/result → execute_tool {name} child span
# ---------------------------------------------------------------------------

class TestGenAIToolSpan(unittest.TestCase):
    def _make_tool_pair(self, tool_name: str = "Bash") -> list[TraceEvent]:
        call = TraceEvent(
            event_type=EventType.TOOL_CALL,
            data={"tool_name": tool_name, "arguments": {"command": "ls -la"}},
        )
        result = TraceEvent(
            event_type=EventType.TOOL_RESULT,
            parent_id=call.event_id,
            duration_ms=50.0,
            data={"tool_name": tool_name, "result": "file1.py\nfile2.py"},
        )
        return [call, result]

    def test_tool_pair_produces_child_span(self):
        meta = _make_meta()
        events = self._make_tool_pair()
        payload = session_to_otlp_genai(meta, events)
        spans = _get_spans(payload)
        self.assertEqual(len(spans), 2)
        tool_span = spans[1]
        self.assertEqual(tool_span["name"], "execute_tool Bash")
        self.assertEqual(tool_span["kind"], 1)
        self.assertEqual(_get_attr(tool_span, "gen_ai.operation.name"), "execute_tool")
        self.assertIsNone(_get_attr(tool_span, "error.type"))

    def test_unmatched_tool_call_is_execute_tool(self):
        call = TraceEvent(event_type=EventType.TOOL_CALL, data={"tool_name": "Open"})
        span = _get_spans(session_to_otlp_genai(_make_meta(), [call]))[1]
        self.assertEqual(span["name"], "execute_tool Open")
        self.assertEqual(span["kind"], 1)
        self.assertEqual(_get_attr(span, "gen_ai.operation.name"), "execute_tool")

    def test_tool_span_has_gen_ai_tool_name(self):
        meta = _make_meta()
        events = self._make_tool_pair("Bash")
        payload = session_to_otlp_genai(meta, events)
        tool_span = _get_spans(payload)[1]
        self.assertEqual(_get_attr(tool_span, "gen_ai.tool.name"), "Bash")

    def test_tool_span_has_call_id(self):
        meta = _make_meta()
        events = self._make_tool_pair()
        payload = session_to_otlp_genai(meta, events)
        tool_span = _get_spans(payload)[1]
        self.assertIsNotNone(_get_attr(tool_span, "gen_ai.tool.call.id"))

    def test_tool_span_has_input_attributes(self):
        meta = _make_meta()
        events = self._make_tool_pair()
        payload = session_to_otlp_genai(meta, events)
        tool_span = _get_spans(payload)[1]
        self.assertIsNotNone(_get_attr(tool_span, "gen_ai.tool.input.command"))

    def test_tool_span_has_output(self):
        meta = _make_meta()
        events = self._make_tool_pair()
        payload = session_to_otlp_genai(meta, events)
        tool_span = _get_spans(payload)[1]
        self.assertIsNotNone(_get_attr(tool_span, "gen_ai.tool.output"))


class TestGenAISpanStatus(unittest.TestCase):
    """Successful spans leave status UNSET; only failures set ERROR.

    The GenAI span conventions defer span status to OTel semantic
    conventions v1.45.0 docs/general/recording-errors.md, which says span
    status MUST be left unset when the operation ended without errors.
    """

    def _session(self):
        call = TraceEvent(event_type=EventType.TOOL_CALL, event_id="call1",
                          data={"tool_name": "Bash", "arguments": {"command": "ls"}})
        result = TraceEvent(event_type=EventType.TOOL_RESULT, parent_id="call1",
                            duration_ms=10.0, data={"tool_name": "Bash", "result": "ok"})
        req = TraceEvent(event_type=EventType.LLM_REQUEST, event_id="req1",
                         data={"model": "gpt-4o", "input_tokens": 3})
        resp = TraceEvent(event_type=EventType.LLM_RESPONSE, parent_id="req1",
                          data={"model": "gpt-4o", "output_tokens": 2})
        return [call, result, req, resp]

    def _statuses(self, meta, events):
        return {s["name"]: s["status"] for s in _get_spans(session_to_otlp_genai(meta, events))}

    def test_successful_tool_llm_and_root_spans_leave_status_unset(self):
        meta = SessionMeta(session_id="s1", agent_name="claude-code", errors=0)
        self.assertEqual(self._statuses(meta, self._session()), {
            "invoke_agent claude-code": {"code": 0},
            "execute_tool Bash": {"code": 0},
            "chat gpt-4o": {"code": 0},
        })

    def test_unlinked_error_leaves_root_unset_and_error_spans_unchanged(self):
        # An error linked to no tool call is not an unresolved operation,
        # so it does not end the root in ERROR; its own span still is.
        meta = SessionMeta(session_id="s1", agent_name="claude-code", errors=1)
        events = self._session() + [TraceEvent(
            event_type=EventType.ERROR, data={"tool_name": "Read", "error": "missing"},
        )]
        statuses = self._statuses(meta, events)
        self.assertEqual(statuses["invoke_agent claude-code"], {"code": 0})
        self.assertEqual(statuses["execute_tool Read"], {"code": 2, "message": "missing"})
        self.assertEqual(statuses["execute_tool Bash"], {"code": 0})
        self.assertEqual(statuses["chat gpt-4o"], {"code": 0})

    def test_no_span_sets_status_ok(self):
        events = self._session() + [
            _codex_response("c1", 1001.0, status="completed"),
            _codex_response("c2", 1002.0, status="failed", error="boom"),
            TraceEvent(event_type=EventType.TOOL_CALL, data={"tool_name": "Open"}),
            TraceEvent(event_type=EventType.LLM_REQUEST, data={"model": "gpt-4o"}),
        ]
        for errors in (0, 1):
            meta = SessionMeta(session_id="s1", agent_name="codex", errors=errors)
            with self.subTest(errors=errors):
                spans = _get_spans(session_to_otlp_genai(meta, events))
                self.assertNotIn(1, [s["status"]["code"] for s in spans])


# ---------------------------------------------------------------------------
# Error events → exception event format
# ---------------------------------------------------------------------------

class TestGenAIErrorSpan(unittest.TestCase):
    def test_error_produces_exception_event(self):
        meta = _make_meta()
        call = TraceEvent(
            event_type=EventType.TOOL_CALL,
            data={"tool_name": "Bash", "arguments": {"command": "bad-cmd"}},
        )
        error = TraceEvent(
            event_type=EventType.ERROR,
            parent_id=call.event_id,
            data={"tool_name": "Bash", "error": "command not found", "error_type": "ShellError"},
        )
        payload = session_to_otlp_genai(meta, [call, error])
        spans = _get_spans(payload)
        error_span = spans[1]
        # Should have an exception event
        events = error_span.get("events", [])
        self.assertTrue(any(e["name"] == "exception" for e in events))
        self.assertEqual(error_span["name"], "execute_tool Bash")
        self.assertEqual(error_span["kind"], 1)
        self.assertEqual(_get_attr(error_span, "error.type"), "ShellError")

    def test_decorator_exception_type_is_the_error_type(self):
        error = TraceEvent(
            event_type=EventType.ERROR,
            data={"message": "boom", "exception_type": "ValueError"},
        )
        span = _get_spans(session_to_otlp_genai(_make_meta(), [error]))[1]
        self.assertEqual(_get_attr(span, "error.type"), "ValueError")

    def test_error_without_type_uses_other(self):
        for error_type in (None, "", "  ", 3):
            data = {"tool_name": "Bash", "error": "fail"}
            if error_type is not None:
                data["error_type"] = error_type
            with self.subTest(error_type=error_type):
                error = TraceEvent(event_type=EventType.ERROR, data=data)
                span = _get_spans(session_to_otlp_genai(_make_meta(), [error]))[1]
                self.assertEqual(_get_attr(span, "error.type"), "_OTHER")

    def test_error_span_has_error_status(self):
        meta = _make_meta()
        error = TraceEvent(
            event_type=EventType.ERROR,
            data={"tool_name": "Bash", "error": "fail"},
        )
        payload = session_to_otlp_genai(meta, [error])
        spans = _get_spans(payload)
        error_span = spans[1]
        self.assertEqual(error_span["status"]["code"], 2)  # STATUS_CODE_ERROR

    def test_exception_event_has_message(self):
        meta = _make_meta()
        error = TraceEvent(
            event_type=EventType.ERROR,
            data={"error": "something went wrong"},
        )
        payload = session_to_otlp_genai(meta, [error])
        spans = _get_spans(payload)
        error_span = spans[1]
        exc_events = [e for e in error_span.get("events", []) if e["name"] == "exception"]
        self.assertTrue(exc_events)
        exc_attrs = {a["key"]: a["value"] for a in exc_events[0].get("attributes", [])}
        self.assertIn("exception.message", exc_attrs)


# ---------------------------------------------------------------------------
# User prompt / assistant response → events on root span
# ---------------------------------------------------------------------------

class TestGenAIMessageEvents(unittest.TestCase):
    def test_user_prompt_becomes_root_event(self):
        meta = _make_meta()
        ev = TraceEvent(
            event_type=EventType.USER_PROMPT,
            data={"prompt": "Hello agent"},
        )
        payload = session_to_otlp_genai(meta, [ev])
        root = _get_spans(payload)[0]
        event_names = [e["name"] for e in root.get("events", [])]
        self.assertIn("gen_ai.user.message", event_names)

    def test_assistant_response_becomes_root_event(self):
        meta = _make_meta()
        ev = TraceEvent(
            event_type=EventType.ASSISTANT_RESPONSE,
            data={"text": "I will help you."},
        )
        payload = session_to_otlp_genai(meta, [ev])
        root = _get_spans(payload)[0]
        event_names = [e["name"] for e in root.get("events", [])]
        self.assertIn("gen_ai.assistant.message", event_names)


# ---------------------------------------------------------------------------
# Backwards compatibility: --format otlp unchanged
# ---------------------------------------------------------------------------

class TestConversationId(unittest.TestCase):
    """gen_ai.conversation.id carries the agent's own session ID when known."""

    NATIVE_ID = "1f0c2b9e-7d3a-4e6b-9c1d-2a3b4c5d6e7f"

    def _events(self) -> list[TraceEvent]:
        req = TraceEvent(event_type=EventType.LLM_REQUEST, data={"model": "claude-3-5-sonnet"})
        resp = TraceEvent(
            event_type=EventType.LLM_RESPONSE,
            parent_id=req.event_id,
            data={"model": "claude-3-5-sonnet", "output_tokens": 5},
        )
        call = TraceEvent(event_type=EventType.TOOL_CALL, data={"tool_name": "Read"})
        result = TraceEvent(
            event_type=EventType.TOOL_RESULT,
            parent_id=call.event_id,
            data={"tool_name": "Read", "result": "ok"},
        )
        failing = TraceEvent(event_type=EventType.TOOL_CALL, data={"tool_name": "Bash"})
        error = TraceEvent(
            event_type=EventType.ERROR,
            parent_id=failing.event_id,
            data={"tool_name": "Bash", "error": "exit 1"},
        )
        open_req = TraceEvent(event_type=EventType.LLM_REQUEST, data={"model": "gpt-4o"})
        open_call = TraceEvent(event_type=EventType.TOOL_CALL, data={"tool_name": "Grep"})
        return [req, resp, call, result, failing, error, open_req, open_call]

    def test_genai_spans_carry_native_conversation_id(self):
        meta = _make_meta()
        meta.conversation_id = self.NATIVE_ID
        spans = _get_spans(session_to_otlp_genai(meta, self._events()))
        # root, paired LLM, paired tool, error, unmatched LLM, unmatched tool
        self.assertEqual(len(spans), 6)
        for span in spans:
            with self.subTest(span=span["name"]):
                self.assertEqual(_get_attr(span, "gen_ai.conversation.id"), self.NATIVE_ID)
        # The local session ID stays on agent.session_id and session.id.
        self.assertEqual(_get_attr(spans[0], "agent.session_id"), meta.session_id)
        self.assertEqual(_get_attr(spans[0], "session.id"), meta.session_id)

    def test_genai_omits_conversation_id_without_native_id(self):
        meta = _make_meta()
        spans = _get_spans(session_to_otlp_genai(meta, self._events()))
        for span in spans:
            with self.subTest(span=span["name"]):
                self.assertIsNone(_get_attr(span, "gen_ai.conversation.id"))

    def test_plain_otlp_root_carries_conversation_id(self):
        meta = _make_meta()
        meta.conversation_id = self.NATIVE_ID
        spans = _get_spans(session_to_otlp(meta, self._events()))
        self.assertEqual(_get_attr(spans[0], "gen_ai.conversation.id"), self.NATIVE_ID)
        for span in spans[1:]:
            self.assertIsNone(_get_attr(span, "gen_ai.conversation.id"))

    def test_plain_otlp_omits_conversation_id_without_native_id(self):
        spans = _get_spans(session_to_otlp(_make_meta(), self._events()))
        for span in spans:
            self.assertIsNone(_get_attr(span, "gen_ai.conversation.id"))

    def test_meta_round_trip_keeps_conversation_id(self):
        meta = _make_meta()
        meta.conversation_id = self.NATIVE_ID
        loaded = SessionMeta.from_json(meta.to_json())
        self.assertEqual(loaded.conversation_id, self.NATIVE_ID)

    def test_meta_without_conversation_id_omits_key(self):
        meta = _make_meta()
        self.assertNotIn("conversation_id", json.loads(meta.to_json()))
        self.assertEqual(SessionMeta.from_json(meta.to_json()).conversation_id, "")


class TestOTLPBackwardsCompat(unittest.TestCase):
    def test_legacy_otlp_unchanged(self):
        """session_to_otlp() output must not change."""
        meta = _make_meta()
        call = TraceEvent(
            event_type=EventType.TOOL_CALL,
            data={"tool_name": "Bash", "arguments": {"command": "ls"}},
        )
        result = TraceEvent(
            event_type=EventType.TOOL_RESULT,
            parent_id=call.event_id,
            data={"tool_name": "Bash", "result": "ok"},
        )
        payload = session_to_otlp(meta, [call, result])
        spans = _get_spans(payload)
        # Legacy: tool spans use tool/<name>, not execute_tool <name>
        tool_span = spans[1]
        self.assertEqual(tool_span["name"], "tool/Bash")

    def test_genai_and_legacy_produce_different_span_names(self):
        meta = _make_meta()
        call = TraceEvent(
            event_type=EventType.TOOL_CALL,
            data={"tool_name": "Bash", "arguments": {"command": "ls"}},
        )
        result = TraceEvent(
            event_type=EventType.TOOL_RESULT,
            parent_id=call.event_id,
            data={"tool_name": "Bash", "result": "ok"},
        )
        legacy = session_to_otlp(meta, [call, result])
        genai = session_to_otlp_genai(meta, [call, result])
        legacy_names = {s["name"] for s in _get_spans(legacy)}
        genai_names = {s["name"] for s in _get_spans(genai)}
        self.assertNotEqual(legacy_names, genai_names)


# ---------------------------------------------------------------------------
# CLI: --format otlp-genai registered
# ---------------------------------------------------------------------------

class TestOTLPGenAICLIFlag(unittest.TestCase):
    def test_otlp_genai_in_format_choices(self):
        import sys, io
        from agent_trace.cli import main
        old_argv = sys.argv
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.argv = ["agent-strace", "export", "--help"]
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            try:
                main()
            except SystemExit:
                pass
            output = sys.stdout.getvalue() + sys.stderr.getvalue()
        finally:
            sys.argv = old_argv
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        self.assertIn("otlp-genai", output)


# ---------------------------------------------------------------------------
# Failed and cancelled model requests
# ---------------------------------------------------------------------------

def _codex_response(event_id: str = "resp1", timestamp: float = 1010.0, **data) -> TraceEvent:
    """An llm_response as the Codex PostModelRequest hook records it."""
    base = {
        "hook_event": "post_model_request",
        "provider": "codex",
        "model": "gpt-5-codex",
        "turn_id": "turn-1",
    }
    base.update(data)
    return TraceEvent(
        event_type=EventType.LLM_RESPONSE,
        event_id=event_id,
        timestamp=timestamp,
        data=base,
    )


class TestGenAIFailedModelRequests(unittest.TestCase):
    """A failed or cancelled model request exports with ERROR status."""

    def _llm_spans(self, events: list[TraceEvent]) -> list[dict]:
        meta = SessionMeta(
            session_id="s1", agent_name="codex", started_at=1000.0, ended_at=1100.0,
        )
        spans = _get_spans(session_to_otlp_genai(meta, events))
        return [s for s in spans if _get_attr(s, "gen_ai.operation.name") == "chat"]

    def _span(self, **data) -> dict:
        (span,) = self._llm_spans([_codex_response(**data)])
        return span

    def test_failed_with_error_text(self):
        span = self._span(status="failed", error="synthetic failure", token_usage=None)
        self.assertEqual(span["status"], {"code": 2, "message": "synthetic failure"})
        self.assertEqual(_get_attr(span, "error.type"), "_OTHER")
        self.assertEqual(_get_attr(span, "gen_ai.response.finish_reasons"), ["error"])
        self.assertIsNone(_get_attr(span, "agent.strace.finish_reason"))
        self.assertNotIn("events", span)

    def test_failed_without_error_text(self):
        span = self._span(status="failed", token_usage=None)
        self.assertEqual(span["status"], {"code": 2})
        self.assertEqual(_get_attr(span, "error.type"), "_OTHER")
        self.assertEqual(_get_attr(span, "gen_ai.response.finish_reasons"), ["error"])

    def test_cancelled(self):
        span = self._span(status="cancelled", token_usage=None)
        self.assertEqual(span["status"], {"code": 2})
        self.assertEqual(_get_attr(span, "error.type"), "cancelled")
        self.assertEqual(_get_attr(span, "gen_ai.response.finish_reasons"), ["error"])

    def test_status_is_case_insensitive_and_accepts_canceled(self):
        self.assertEqual(_get_attr(self._span(status="Canceled"), "error.type"), "cancelled")
        self.assertEqual(_get_attr(self._span(status=" FAILED "), "error.type"), "_OTHER")

    def test_error_without_status_fails(self):
        span = self._span(error="upstream 500")
        self.assertEqual(span["status"], {"code": 2, "message": "upstream 500"})
        self.assertEqual(_get_attr(span, "error.type"), "_OTHER")

    def test_error_object_supplies_type_and_message(self):
        span = self._span(
            status="failed",
            error={"message": "rate limited", "code": "rate_limit_exceeded"},
        )
        self.assertEqual(span["status"], {"code": 2, "message": "rate limited"})
        self.assertEqual(_get_attr(span, "error.type"), "rate_limit_exceeded")

    def test_recorded_error_type_wins(self):
        span = self._span(status="cancelled", error_type="ClientDisconnected")
        self.assertEqual(_get_attr(span, "error.type"), "ClientDisconnected")

    def test_status_message_is_bounded(self):
        span = self._span(status="failed", error="x" * 5000)
        self.assertEqual(span["status"]["message"], "x" * 200)

    def test_recorded_finish_reason_is_kept(self):
        span = self._span(status="failed", finish_reason="length")
        self.assertEqual(_get_attr(span, "gen_ai.response.finish_reasons"), ["length"])
        self.assertEqual(_get_attr(span, "agent.strace.finish_reason"), "length")
        self.assertEqual(_get_attr(span, "llm.finish_reason"), "length")
        self.assertEqual(span["status"]["code"], 2)

    def test_completed_and_unknown_statuses_leave_status_unset(self):
        for data in (
            {"status": "completed", "input_tokens": 10, "output_tokens": 2},
            {"input_tokens": 10, "output_tokens": 2},
            {"status": "incomplete"},
            {"status": "completed", "error": ""},
            {"status": "completed", "error": None},
        ):
            with self.subTest(data=data):
                span = self._span(**data)
                self.assertEqual(span["status"], {"code": 0})
                self.assertIsNone(_get_attr(span, "error.type"))
                self.assertIsNone(_get_attr(span, "gen_ai.response.finish_reasons"))

    def test_only_the_failed_request_changes(self):
        events = [
            _codex_response("a", 1001.0, status="completed", input_tokens=10, output_tokens=2),
            _codex_response("b", 1002.0, status="failed", error="boom", token_usage=None),
            _codex_response("c", 1003.0, status="cancelled", token_usage=None),
        ]
        spans = self._llm_spans(events)
        self.assertEqual([s["status"]["code"] for s in spans], [0, 2, 2])
        root = _get_spans(session_to_otlp_genai(
            SessionMeta(session_id="s1", agent_name="codex", started_at=1000.0, ended_at=1100.0),
            events,
        ))[0]
        self.assertEqual(root["status"], {"code": 0})

    def _linked_error_spans(self, error_data: dict) -> list[dict]:
        request = TraceEvent(
            event_type=EventType.LLM_REQUEST, event_id="req1", timestamp=1001.0,
            data={"model": "gpt-4o", "max_tokens": 100, "message_count": 1},
        )
        error = TraceEvent(
            event_type=EventType.ERROR, event_id="err1", parent_id="req1",
            timestamp=1001.5, duration_ms=500.0, data=error_data,
        )
        meta = SessionMeta(session_id="s1", agent_name="openai", started_at=1000.0, ended_at=1100.0)
        return _get_spans(session_to_otlp_genai(meta, [request, error]))[1:]

    def test_sdk_error_linked_to_request_is_one_failed_llm_span(self):
        spans = self._linked_error_spans({
            "error": "Rate limit reached", "error_type": "RateLimitError",
        })
        self.assertEqual([s["name"] for s in spans], ["chat gpt-4o"])
        (span,) = spans
        self.assertEqual(span["spanId"], _to_span_id("req1"))
        self.assertEqual(span["startTimeUnixNano"], _ts_to_nanos(1001.0))
        self.assertEqual(span["endTimeUnixNano"], _ts_to_nanos(1001.5))
        self.assertEqual(span["status"], {"code": 2, "message": "Rate limit reached"})
        self.assertEqual(_get_attr(span, "error.type"), "RateLimitError")
        self.assertEqual(_get_attr(span, "gen_ai.request.model"), "gpt-4o")
        self.assertEqual(_get_attr(span, "gen_ai.response.finish_reasons"), ["error"])
        (exc,) = span["events"]
        exc_attrs = {a["key"]: a["value"] for a in exc["attributes"]}
        self.assertEqual(exc["name"], "exception")
        self.assertEqual(exc_attrs["exception.type"], {"stringValue": "RateLimitError"})
        self.assertEqual(exc_attrs["exception.message"], {"stringValue": "Rate limit reached"})

    def test_decorator_error_linked_to_request_uses_exception_type(self):
        (span,) = self._linked_error_spans({
            "message": "boom", "exception_type": "ValueError",
        })
        self.assertEqual(span["status"], {"code": 2, "message": "boom"})
        self.assertEqual(_get_attr(span, "error.type"), "ValueError")

    def test_unlinked_error_after_request_is_unchanged(self):
        request = TraceEvent(
            event_type=EventType.LLM_REQUEST, event_id="req1", timestamp=1001.0,
            data={"model": "gpt-4o"},
        )
        error = TraceEvent(
            event_type=EventType.ERROR, event_id="err1", timestamp=1002.0,
            data={"error": "boom", "error_type": "APIError"},
        )
        meta = SessionMeta(session_id="s1", agent_name="openai", started_at=1000.0, ended_at=1100.0)
        spans = _get_spans(session_to_otlp_genai(meta, [request, error]))[1:]
        names = sorted(s["name"] for s in spans)
        self.assertEqual(names, ["chat gpt-4o", "execute_tool error"])
        llm = next(s for s in spans if s["name"] == "chat gpt-4o")
        self.assertEqual(llm["status"], {"code": 0})
        self.assertIsNone(_get_attr(llm, "error.type"))

    def test_error_linked_to_tool_call_is_still_a_tool_error(self):
        call = TraceEvent(
            event_type=EventType.TOOL_CALL, event_id="call1", timestamp=1001.0,
            data={"tool_name": "Bash", "arguments": {"command": "false"}},
        )
        error = TraceEvent(
            event_type=EventType.ERROR, event_id="err1", parent_id="call1",
            timestamp=1002.0, data={"tool_name": "Bash", "error": "exit 1"},
        )
        meta = SessionMeta(session_id="s1", agent_name="claude-code", started_at=1000.0, ended_at=1100.0)
        (span,) = _get_spans(session_to_otlp_genai(meta, [call, error]))[1:]
        self.assertEqual(span["name"], "execute_tool Bash")
        self.assertEqual(_get_attr(span, "gen_ai.operation.name"), "execute_tool")
        self.assertEqual(_get_attr(span, "error.type"), "_OTHER")


# ---------------------------------------------------------------------------
# Content attributes, error tool names and retry links (#552)
# ---------------------------------------------------------------------------

def _genai_spans(events):
    meta = SessionMeta(session_id="s1", agent_name="claude-code", started_at=1000.0, ended_at=1100.0)
    return _get_spans(session_to_otlp_genai(meta, events))


def _event(event_type, timestamp, event_id="", parent_id="", **data):
    return TraceEvent(event_type=event_type, event_id=event_id or f"e{timestamp}",
                      parent_id=parent_id, timestamp=timestamp, data=data)


def _text(role, *texts):
    return {"role": role, "parts": [{"type": "text", "content": text} for text in texts]}


class TestGenAIRootMessages(unittest.TestCase):
    """The root carries gen_ai.input.messages and gen_ai.output.messages as
    semconv message JSON, with the text the message events carry."""

    def _root(self, *turns):
        events = [
            _event(EventType.USER_PROMPT if kind == "user" else EventType.ASSISTANT_RESPONSE,
                   1001.0 + i, **({"prompt": text} if kind == "user" else {"text": text}))
            for i, (kind, text) in enumerate(turns)
        ]
        return _genai_spans(events)[0]

    def _messages(self, root, key):
        value = _get_attr(root, key)
        return None if value is None else json.loads(value)

    def test_input_is_the_conversation_up_to_the_last_prompt(self):
        root = self._root(
            ("user", "Fix the build"), ("assistant", "Looking at it."),
            ("user", "Then run the tests"), ("assistant", "Done."), ("assistant", "All green."),
        )
        self.assertEqual(self._messages(root, "gen_ai.input.messages"), [
            _text("user", "Fix the build"),
            _text("assistant", "Looking at it."),
            _text("user", "Then run the tests"),
        ])
        self.assertEqual(self._messages(root, "gen_ai.output.messages"), [
            _text("assistant", "Done.", "All green."),
        ])
        # Compact JSON text, the way the OTel GenAI instrumentations write it.
        self.assertNotIn(", ", _get_attr(root, "gen_ai.input.messages"))

    def test_text_is_the_message_event_text(self):
        prompt = "[REDACTED:api-key] " + "p" * 2000
        reply = "r" * 1500
        root = self._root(("user", prompt), ("assistant", reply))
        events = {e["name"]: {a["key"]: a["value"].get("stringValue") for a in e["attributes"]}
                  for e in root["events"]}
        [user] = self._messages(root, "gen_ai.input.messages")
        [assistant] = self._messages(root, "gen_ai.output.messages")
        self.assertEqual(user["parts"][0]["content"], events["gen_ai.user.message"]["gen_ai.prompt"])
        self.assertEqual(user["parts"][0]["content"], prompt[:1000])
        self.assertEqual(
            assistant["parts"][0]["content"], events["gen_ai.assistant.message"]["gen_ai.completion"],
        )
        self.assertEqual(len(assistant["parts"][0]["content"]), 1000)

    def test_no_prompt_puts_every_reply_in_the_output(self):
        root = self._root(("assistant", "One."), ("assistant", "Two."))
        self.assertIsNone(_get_attr(root, "gen_ai.input.messages"))
        self.assertEqual(self._messages(root, "gen_ai.output.messages"), [_text("assistant", "One.", "Two.")])

    def test_prompt_without_a_reply_has_no_output(self):
        root = self._root(("user", "Hello"))
        self.assertEqual(self._messages(root, "gen_ai.input.messages"), [_text("user", "Hello")])
        self.assertIsNone(_get_attr(root, "gen_ai.output.messages"))

    def test_empty_and_wrong_typed_text_is_skipped(self):
        root = self._root(("user", ""), ("assistant", 5), ("user", ["x"]), ("assistant", None))
        self.assertIsNone(_get_attr(root, "gen_ai.input.messages"))
        self.assertIsNone(_get_attr(root, "gen_ai.output.messages"))
        root = self._root(("user", "Go"), ("assistant", {"a": 1}), ("assistant", "Went."))
        self.assertEqual(self._messages(root, "gen_ai.input.messages"), [_text("user", "Go")])
        self.assertEqual(self._messages(root, "gen_ai.output.messages"), [_text("assistant", "Went.")])

    def test_empty_session_has_no_messages(self):
        root = _genai_spans([])[0]
        self.assertIsNone(_get_attr(root, "gen_ai.input.messages"))
        self.assertIsNone(_get_attr(root, "gen_ai.output.messages"))


class TestGenAIToolCallContent(unittest.TestCase):
    """execute_tool spans carry gen_ai.tool.call.arguments and
    gen_ai.tool.call.result: the text input.value and output.value carry,
    except that JSON stays valid JSON within the same limits."""

    def _span(self, arguments, *rest):
        call = _event(EventType.TOOL_CALL, 1001.0, "call1", tool_name="Bash", arguments=arguments)
        spans = _genai_spans([call, *rest])
        (span,) = spans[1:]
        return span

    def _result(self, result):
        return _event(EventType.TOOL_RESULT, 1002.0, parent_id="call1", result=result)

    def test_arguments_and_result(self):
        span = self._span({"command": "ls -la", "timeout": 5}, self._result("a.txt\nb.txt"))
        arguments = _get_attr(span, "gen_ai.tool.call.arguments")
        self.assertEqual(json.loads(arguments), {"command": "ls -la", "timeout": 5})
        self.assertEqual(arguments, _get_attr(span, "input.value"))
        self.assertEqual(_get_attr(span, "gen_ai.tool.call.result"), "a.txt\nb.txt")
        self.assertEqual(_get_attr(span, "gen_ai.tool.call.result"), _get_attr(span, "output.value"))
        # The older keys stay.
        self.assertEqual(_get_attr(span, "gen_ai.tool.input.command"), "ls -la")
        self.assertEqual(_get_attr(span, "gen_ai.tool.output"), "a.txt\nb.txt")

    def test_structured_result_is_json(self):
        span = self._span({"path": "a"}, self._result({"files": ["a", "b"]}))
        self.assertEqual(json.loads(_get_attr(span, "gen_ai.tool.call.result")), {"files": ["a", "b"]})

    def test_result_is_cut_like_output_value(self):
        span = self._span({"path": "a"}, self._result("x" * 2000))
        self.assertEqual(_get_attr(span, "gen_ai.tool.call.result"), "x" * 500)

    def test_long_arguments_stay_a_json_object_of_the_cut_values(self):
        span = self._span({"path": "a.py", "content": "y" * 5000}, self._result("ok"))
        self.assertEqual(
            json.loads(_get_attr(span, "gen_ai.tool.call.arguments")),
            {"path": "a.py", "content": "y" * 200},
        )
        # input.value is the cut JSON text, which no longer parses.
        self.assertEqual(_get_attr(span, "input.mime_type"), "text/plain")
        self.assertEqual(len(_get_attr(span, "input.value")), 1000)

    def test_arguments_that_are_not_an_object_are_carried_as_text(self):
        span = self._span("ls -la", self._result("ok"))
        self.assertEqual(_get_attr(span, "gen_ai.tool.call.arguments"), "ls -la")

    def _content(self, span, key, limit):
        value = _get_attr(span, key)
        self.assertLessEqual(len(value), limit)
        return json.loads(value)

    def test_long_structured_result_stays_valid_json(self):
        result = {"path": "/", "files": [f"file-{i}.txt" for i in range(200)]}
        span = self._span({"path": "a"}, self._result(result))
        parsed = self._content(span, "gen_ai.tool.call.result", 500)
        # Keys in sorted order, as output.value writes them; the list is cut.
        self.assertEqual(list(parsed), ["files"])
        files = parsed["files"]
        self.assertEqual(files[:-1], result["files"][:len(files) - 1])
        self.assertTrue(result["files"][len(files) - 1].startswith(files[-1]))
        self.assertGreater(len(files), 20)
        self.assertEqual(len(_get_attr(span, "output.value")), 500)

    def test_long_json_text_result_stays_valid_json(self):
        result = json.dumps({"stdout": "z" * 2000, "exit_code": 0})
        span = self._span({"path": "a"}, self._result(result))
        parsed = self._content(span, "gen_ai.tool.call.result", 500)
        # The recorded key order stays.
        self.assertEqual(list(parsed), ["stdout"])
        self.assertGreater(len(parsed["stdout"]), 400)
        self.assertEqual(parsed["stdout"], "z" * len(parsed["stdout"]))
        self.assertEqual(_get_attr(span, "output.value"), result[:500])

    def test_json_text_result_that_fits_is_kept(self):
        result = '{"b": 1, "a": [true, null]}'
        span = self._span({"path": "a"}, self._result(result))
        self.assertEqual(_get_attr(span, "gen_ai.tool.call.result"), result)

    def test_result_that_only_looks_like_json_becomes_a_json_string(self):
        python_text = "{'filePath': '/a.py', 'success': True}"
        cut_json = '{"content": "' + "c" * 1200 + "... (truncated)"
        for result in (python_text, cut_json):
            with self.subTest(result=result[:20]):
                span = self._span({"path": "a"}, self._result(result))
                parsed = self._content(span, "gen_ai.tool.call.result", 500)
                self.assertIsInstance(parsed, str)
                self.assertTrue(result.startswith(parsed))
                self.assertEqual(_get_attr(span, "output.value"), result[:500])

    def test_long_list_arguments_stay_valid_json(self):
        span = self._span(["x" * 600, "y" * 600], self._result("ok"))
        parsed = self._content(span, "gen_ai.tool.call.arguments", 1000)
        self.assertEqual(parsed[0], "x" * 600)
        self.assertTrue(("y" * 600).startswith(parsed[1]))
        self.assertEqual(len(_get_attr(span, "input.value")), 1000)

    def test_no_arguments_and_no_result_give_no_content(self):
        span = self._span({}, self._result(""))
        self.assertIsNone(_get_attr(span, "gen_ai.tool.call.arguments"))
        self.assertIsNone(_get_attr(span, "gen_ai.tool.call.result"))

    def test_failed_call_and_call_without_a_result_carry_their_arguments(self):
        error = _event(EventType.ERROR, 1002.0, parent_id="call1", error="exit 1")
        for rest in ((error,), ()):
            with self.subTest(error=bool(rest)):
                span = self._span({"command": "false"}, *rest)
                self.assertEqual(_get_attr(span, "gen_ai.tool.call.arguments"), '{"command":"false"}')
                self.assertIsNone(_get_attr(span, "gen_ai.tool.call.result"))

    def test_chat_spans_carry_no_content(self):
        request = _event(EventType.LLM_REQUEST, 1001.0, "req1", model="gpt-4o", messages=[{"role": "user"}])
        response = _event(EventType.LLM_RESPONSE, 1002.0, parent_id="req1", model="gpt-4o", text="hi")
        (chat,) = _genai_spans([request, response])[1:]
        self.assertIsNone(_get_attr(chat, "gen_ai.input.messages"))
        self.assertIsNone(_get_attr(chat, "gen_ai.output.messages"))


class TestGenAIResponseAndServer(unittest.TestCase):
    """Chat spans carry gen_ai.response.id, gen_ai.response.model,
    server.address and server.port only where a capture path recorded them."""

    KEYS = ("gen_ai.response.id", "gen_ai.response.model", "server.address", "server.port")

    def _chat(self, request_data, response_data=None):
        events = [_event(EventType.LLM_REQUEST, 1001.0, "req1", model="gpt-4o", **request_data)]
        if response_data is not None:
            events.append(_event(EventType.LLM_RESPONSE, 1002.0, parent_id="req1",
                                 model="gpt-4o", **response_data))
        (chat,) = _genai_spans(events)[1:]
        return {key: _get_attr(chat, key) for key in self.KEYS}

    def test_recorded_values(self):
        attrs = self._chat(
            {"server_address": "api.openai.com", "server_port": 443},
            {"response_id": "chatcmpl-1", "response_model": "gpt-4o-2024-08-06"},
        )
        self.assertEqual(attrs, {
            "gen_ai.response.id": "chatcmpl-1", "gen_ai.response.model": "gpt-4o-2024-08-06",
            "server.address": "api.openai.com", "server.port": "443",
        })
        request = _event(EventType.LLM_REQUEST, 1001.0, "req1", server_address="h", server_port=1)
        (chat,) = _genai_spans([request])[1:]
        (port,) = [a["value"] for a in chat["attributes"] if a["key"] == "server.port"]
        self.assertEqual(port, {"intValue": "1"})

    def test_nothing_recorded_sets_nothing(self):
        # The request model is not the response model.
        self.assertEqual(self._chat({}, {}), dict.fromkeys(self.KEYS))

    def test_unusable_values_are_left_out(self):
        for port in (True, 0, 65536, "443", 443.0):
            with self.subTest(port=port):
                attrs = self._chat({"server_address": "localhost", "server_port": port}, {})
                self.assertEqual(attrs["server.address"], "localhost")
                self.assertIsNone(attrs["server.port"])
        attrs = self._chat({"server_address": "", "server_port": 443},
                           {"response_id": 5, "response_model": ""})
        self.assertEqual(attrs, dict.fromkeys(self.KEYS))

    def test_request_without_a_response_or_with_an_error_keeps_its_server(self):
        expected = {**dict.fromkeys(self.KEYS), "server.address": "localhost", "server.port": "8080"}
        self.assertEqual(self._chat({"server_address": "localhost", "server_port": 8080}), expected)
        events = [
            _event(EventType.LLM_REQUEST, 1001.0, "req1", model="gpt-4o",
                   server_address="localhost", server_port=8080),
            _event(EventType.ERROR, 1002.0, parent_id="req1", error="Rate limit reached"),
        ]
        (chat,) = _genai_spans(events)[1:]
        self.assertEqual({key: _get_attr(chat, key) for key in self.KEYS}, expected)


class TestGenAIErrorToolName(unittest.TestCase):
    def test_error_without_a_tool_name_takes_the_calls(self):
        # The @trace_tool decorator records the tool name only on the call.
        call = _event(EventType.TOOL_CALL, 1001.0, "call1", tool_name="Read", arguments={"path": "a"})
        error = _event(EventType.ERROR, 1002.0, parent_id="call1", error="not found")
        (span,) = _genai_spans([call, error])[1:]
        self.assertEqual(span["name"], "execute_tool Read")
        self.assertEqual(_get_attr(span, "gen_ai.tool.name"), "Read")

    def test_lone_error_is_still_named_error(self):
        (span,) = _genai_spans([_event(EventType.ERROR, 1002.0, error="boom")])[1:]
        self.assertEqual(span["name"], "execute_tool error")


class TestGenAIRetryLinks(unittest.TestCase):
    LINK_ATTRS = [{"key": "agent.strace.link.type", "value": {"stringValue": "retry_of"}}]

    def test_tool_retry_links_to_the_failed_attempt(self):
        spans = _genai_spans([
            _event(EventType.TOOL_CALL, 1001.0, "call1", tool_name="Bash", arguments={"command": "make"}),
            _event(EventType.ERROR, 1002.0, parent_id="call1", error="exit 2"),
            _event(EventType.TOOL_CALL, 1003.0, "call2", tool_name="Bash",
                   arguments={"command": "make"}, retry_of="call1"),
            _event(EventType.TOOL_RESULT, 1004.0, parent_id="call2", result="ok"),
        ])
        _, first, retry = spans
        self.assertEqual(retry["links"], [
            {"traceId": first["traceId"], "spanId": first["spanId"], "attributes": self.LINK_ATTRS},
        ])
        self.assertNotIn("links", first)

    def test_model_request_retry_links_to_the_failed_request(self):
        spans = _genai_spans([
            _event(EventType.LLM_REQUEST, 1001.0, "req1", model="gpt-4o"),
            _event(EventType.ERROR, 1002.0, parent_id="req1", error="overloaded"),
            _event(EventType.LLM_REQUEST, 1003.0, "req2", model="gpt-4o", retry_of="req1"),
            _event(EventType.LLM_RESPONSE, 1004.0, parent_id="req2", model="gpt-4o"),
        ])
        _, first, retry = spans
        self.assertEqual((first["name"], retry["name"]), ("chat gpt-4o", "chat gpt-4o"))
        self.assertEqual(first["status"]["code"], 2)
        self.assertEqual(retry["links"], [
            {"traceId": first["traceId"], "spanId": first["spanId"], "attributes": self.LINK_ATTRS},
        ])

    def test_retry_of_a_result_event_links_to_its_call_span(self):
        spans = _genai_spans([
            _event(EventType.TOOL_CALL, 1001.0, "call1", tool_name="Bash"),
            _event(EventType.TOOL_RESULT, 1002.0, "res1", parent_id="call1", result="flaky"),
            _event(EventType.TOOL_CALL, 1003.0, "call2", tool_name="Bash", retry_of="res1"),
        ])
        _, first, retry = spans
        self.assertEqual(retry["links"][0]["spanId"], first["spanId"])

    def test_no_link_to_an_unknown_event_or_to_itself(self):
        for retry_of in ("missing", "call2", "", None, {"id": "call1"}):
            with self.subTest(retry_of=retry_of):
                spans = _genai_spans([
                    _event(EventType.TOOL_CALL, 1001.0, "call1", tool_name="Bash"),
                    _event(EventType.TOOL_CALL, 1003.0, "call2", tool_name="Bash", retry_of=retry_of),
                ])
                self.assertFalse(any("links" in span for span in spans))


def _imported(timestamp, usage=None, stream="main", **data):
    """An assistant_response as agent-strace import writes it."""
    data = {"model": "claude-sonnet-4-5", "context_stream": stream, **data}
    if usage is not None:
        data["usage"] = usage
    return _event(EventType.ASSISTANT_RESPONSE, timestamp, **data)


class TestImportedModelResponses(unittest.TestCase):
    """Each model response read from a transcript is a chat span with an
    inferred start, in both formats (#564)."""

    CONVERTERS = (session_to_otlp, session_to_otlp_genai)

    def _spans(self, convert, events):
        meta = SessionMeta(session_id="s1", agent_name="claude-code",
                           started_at=1000.0, ended_at=1100.0)
        return _get_spans(convert(meta, events))

    def _chats(self, convert, events):
        return [s for s in self._spans(convert, events)[1:] if s["name"].startswith("chat")]

    @staticmethod
    def _window(span):
        return (span["startTimeUnixNano"], span["endTimeUnixNano"])

    def test_span_without_earlier_input_has_zero_length(self):
        events = [_imported(1002.0, {"input_tokens": 25, "output_tokens": 6},
                            message_id="msg_1", stop_reason="end_turn")]
        for convert in self.CONVERTERS:
            with self.subTest(convert=convert.__name__):
                [chat] = self._chats(convert, events)
                self.assertEqual(self._window(chat), (_ts_to_nanos(1002.0),) * 2)
                self.assertEqual(_get_attr(chat, "agent.strace.timing"), "inferred")
                self.assertEqual(_get_attr(chat, "gen_ai.usage.input_tokens"), "25")
                self.assertEqual(_get_attr(chat, "gen_ai.response.finish_reasons"), ["end_turn"])
                self.assertEqual(chat["status"], {"code": 0})

    def test_span_starts_at_the_latest_input_of_its_own_stream(self):
        usage = {"input_tokens": 10, "output_tokens": 1}
        events = [
            _event(EventType.USER_PROMPT, 1001.0, prompt="go", context_stream="main"),
            _event(EventType.TOOL_RESULT, 1003.0, result="ok", context_stream="sidechain:a"),
            _imported(1004.0, usage, stream="sidechain:a", message_id="side_1"),
            _imported(1005.0, usage, message_id="msg_1"),
            _event(EventType.TOOL_CALL, 1006.0, tool_name="Bash", context_stream="main"),
            _imported(1007.0, usage, message_id="msg_2"),
            _event(EventType.TOOL_RESULT, 1008.0, result="ok", context_stream="main"),
            _imported(1010.0, usage, message_id="msg_3"),
        ]
        expected = [(1003.0, 1004.0), (1001.0, 1005.0), (1005.0, 1007.0), (1008.0, 1010.0)]
        for convert in self.CONVERTERS:
            with self.subTest(convert=convert.__name__):
                self.assertEqual(
                    [self._window(span) for span in self._chats(convert, events)],
                    [(_ts_to_nanos(start), _ts_to_nanos(end)) for start, end in expected],
                )

    def test_a_message_recorded_twice_is_one_span(self):
        usage = {"input_tokens": 10, "output_tokens": 1}
        events = [
            _event(EventType.USER_PROMPT, 1001.0, prompt="go", context_stream="main"),
            _imported(1002.0, text="Looking.", message_id="msg_1"),
            _imported(1003.0, usage, message_id="msg_1"),
            _imported(1004.0, usage, message_id="msg_1"),
        ]
        for convert in self.CONVERTERS:
            with self.subTest(convert=convert.__name__):
                root, *_ = self._spans(convert, events)
                [chat] = self._chats(convert, events)
                self.assertEqual(self._window(chat), (_ts_to_nanos(1001.0), _ts_to_nanos(1003.0)))
                # Every response keeps its root span event.
                self.assertEqual(
                    sum(e["name"] in ("assistant_response", "gen_ai.assistant.message")
                        for e in root["events"]),
                    3,
                )

    def test_response_without_a_message_id_has_no_response_id(self):
        events = [_imported(1002.0, {"input_tokens": 1, "output_tokens": 1})]
        for convert in self.CONVERTERS:
            with self.subTest(convert=convert.__name__):
                [chat] = self._chats(convert, events)
                self.assertIsNone(_get_attr(chat, "gen_ai.response.id"))
                self.assertIsNone(_get_attr(chat, "gen_ai.response.finish_reasons"))
                self.assertEqual(_get_attr(chat, "gen_ai.response.model"), "claude-sonnet-4-5")

    def test_hook_session_has_no_chat_span(self):
        events = [
            _event(EventType.USER_PROMPT, 1001.0, prompt="go"),
            _event(EventType.ASSISTANT_RESPONSE, 1002.0, text="Done.", hook_event="stop",
                   provider="claude", stop_reason="end_turn"),
        ]
        for convert in self.CONVERTERS:
            with self.subTest(convert=convert.__name__):
                self.assertEqual(len(self._spans(convert, events)), 1)


if __name__ == "__main__":
    unittest.main()
