"""Tests for OTLP GenAI semantic conventions export (issue #100)."""

import json
import os
import sys
import tempfile
import time
import unittest

from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.otlp import _to_span_id, _ts_to_nanos, session_to_otlp, session_to_otlp_genai
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
    def test_root_span_has_gen_ai_agent_id(self):
        meta = _make_meta()
        payload = session_to_otlp_genai(meta, [])
        spans = _get_spans(payload)
        root = spans[0]
        self.assertEqual(_get_attr(root, "gen_ai.agent.id"), meta.session_id)

    def test_root_span_has_gen_ai_agent_name(self):
        meta = _make_meta("my-agent")
        payload = session_to_otlp_genai(meta, [])
        spans = _get_spans(payload)
        root = spans[0]
        self.assertEqual(_get_attr(root, "gen_ai.agent.name"), "my-agent")

    def test_root_span_has_gen_ai_system(self):
        meta = _make_meta("claude-code")
        payload = session_to_otlp_genai(meta, [])
        spans = _get_spans(payload)
        root = spans[0]
        self.assertEqual(_get_attr(root, "gen_ai.system"), "anthropic")

    def test_root_span_name_contains_agent_session(self):
        meta = _make_meta()
        payload = session_to_otlp_genai(meta, [])
        spans = _get_spans(payload)
        self.assertIn("gen_ai.agent.session", spans[0]["name"])

    def test_scope_name_includes_semconv_version(self):
        meta = _make_meta()
        payload = session_to_otlp_genai(meta, [])
        scope = payload["resourceSpans"][0]["scopeSpans"][0]["scope"]
        self.assertIn("genai-semconv", scope.get("version", ""))


# ---------------------------------------------------------------------------
# LLM request/response → gen_ai.client.operation child span
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
        self.assertEqual(llm_span["name"], "gen_ai.client.operation")

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
        self.assertEqual(_get_attr(llm_span, "gen_ai.response.finish_reasons"), "end_turn")

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
        self.assertEqual(_get_attr(llm_span, "gen_ai.system"), "openai")


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
        spans = _get_spans(session_to_otlp_genai(_make_meta(), events))[1:]
        return {s["spanId"]: (s["startTimeUnixNano"], s["status"]["code"]) for s in spans}

    def test_parentless_response_pairs_with_newest_open_request(self):
        events = [self._req("failed", 1.0), self._req("ok", 2.0), self._resp(3.0)]
        spans = self._span_starts(events)
        self.assertEqual(spans[_to_span_id("ok")], (_ts_to_nanos(2.0), 1))
        self.assertEqual(spans[_to_span_id("failed")], (_ts_to_nanos(1.0), 0))

    def test_parentless_response_skips_requests_for_other_models(self):
        events = [
            self._req("gpt", 1.0, "gpt-4o"),
            self._req("claude", 2.0, "claude-opus-4-6"),
            self._resp(3.0, model="gpt-4o"),
        ]
        spans = self._span_starts(events)
        self.assertEqual(spans[_to_span_id("gpt")], (_ts_to_nanos(1.0), 1))
        self.assertEqual(spans[_to_span_id("claude")], (_ts_to_nanos(2.0), 0))

    def test_duplicate_response_does_not_take_another_request(self):
        events = [
            self._req("a", 1.0),
            self._req("b", 2.0),
            self._resp(3.0, parent_id="a"),
            self._resp(4.0, parent_id="a"),
        ]
        spans = self._span_starts(events)
        self.assertEqual(spans[_to_span_id("b")], (_ts_to_nanos(2.0), 0))


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
# Tool call/result → gen_ai.tool.call/<name> child span
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
        self.assertIn("gen_ai.tool.call", tool_span["name"])
        self.assertIn("Bash", tool_span["name"])

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
        # The local session ID stays on gen_ai.agent.id and session.id.
        self.assertEqual(_get_attr(spans[0], "gen_ai.agent.id"), meta.session_id)
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
        # Legacy: tool spans use plain tool name, not gen_ai.tool.call/<name>
        tool_span = spans[1]
        self.assertNotIn("gen_ai.tool.call", tool_span["name"])

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


if __name__ == "__main__":
    unittest.main()
