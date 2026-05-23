"""Tests for OTLP GenAI semantic conventions export (issue #100)."""

import json
import os
import sys
import tempfile
import time
import unittest

from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.otlp import session_to_otlp, session_to_otlp_genai
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
