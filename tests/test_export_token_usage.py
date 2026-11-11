"""Token usage, cache reads and writes included, in the OTLP and Langfuse exports.

Each provider shape is checked in the three exports: OTLP with the GenAI
semantic conventions (``--genai-semconv``), classic OTLP and Langfuse.
"""

import unittest

from agent_trace.cost import TokenUsage, usage_detail
from agent_trace.langfuse_export import _events_to_langfuse_observations
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.otlp import model_request_usage, session_to_otlp, session_to_otlp_genai


INPUT = "gen_ai.usage.input_tokens"
OUTPUT = "gen_ai.usage.output_tokens"
CACHE_READ = "gen_ai.usage.cache_read.input_tokens"
CACHE_WRITE = "gen_ai.usage.cache_write.input_tokens"
OI_PROMPT = "llm.token_count.prompt"
OI_COMPLETION = "llm.token_count.completion"
OI_TOTAL = "llm.token_count.total"
OI_CACHE_READ = "llm.token_count.prompt_details.cache_read"
OI_CACHE_WRITE = "llm.token_count.prompt_details.cache_write"
TOKEN_KEYS = (
    INPUT, OUTPUT, CACHE_READ, CACHE_WRITE,
    OI_PROMPT, OI_COMPLETION, OI_TOTAL, OI_CACHE_READ, OI_CACHE_WRITE,
)


# name: (request data or None for a response with no request, response data,
#        (whole input, output, cache read, cache write), Langfuse usageDetails)
CASES = {
    "anthropic_sdk": (
        {"model": "claude-sonnet-4-5", "max_tokens": 64},
        {
            "model": "claude-sonnet-4-5",
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_read_input_tokens": 200,
            "cache_creation_input_tokens": 50,
            "cache_creation": {
                "ephemeral_5m_input_tokens": 20,
                "ephemeral_1h_input_tokens": 30,
            },
        },
        (260, 5, 200, 50),
        {
            "input": 10,
            "input_cached_tokens": 200,
            "input_cache_creation": 20,
            "input_cache_creation_1h": 30,
            "output": 5,
            "total": 265,
        },
    ),
    "anthropic_nested": (
        None,
        {
            "model": "claude-sonnet-4-5",
            "usage": {
                "input_tokens": 3,
                "output_tokens": 9,
                "cache_read_input_tokens": 1000,
                "cache_creation_input_tokens": 400,
            },
        },
        (1403, 9, 1000, 400),
        {
            "input": 3,
            "input_cached_tokens": 1000,
            "input_cache_creation": 400,
            "output": 9,
            "total": 1412,
        },
    ),
    "bedrock_converse": (
        {"model": "us.anthropic.claude-sonnet-4-5"},
        {
            "model": "us.anthropic.claude-sonnet-4-5",
            "usage": {
                "inputTokens": 7,
                "outputTokens": 11,
                "totalTokens": 618,
                "cacheReadInputTokens": 500,
                "cacheWriteInputTokens": 100,
                "cacheDetails": [
                    {"ttl": "5m", "inputTokens": 40},
                    {"ttl": "1h", "inputTokens": 60},
                ],
            },
        },
        (607, 11, 500, 100),
        {
            "input": 7,
            "input_cached_tokens": 500,
            "input_cache_creation": 40,
            "input_cache_creation_1h": 60,
            "output": 11,
            "total": 618,
        },
    ),
    "gemini": (
        {"model": "gemini-2.5-pro"},
        {
            "model": "gemini-2.5-pro",
            "usageMetadata": {
                "promptTokenCount": 900,
                "candidatesTokenCount": 30,
                "cachedContentTokenCount": 600,
                "thoughtsTokenCount": 70,
                "toolUsePromptTokenCount": 25,
                "totalTokenCount": 1025,
            },
        },
        (925, 100, 600, 0),
        {"input": 325, "input_cached_tokens": 600, "output": 100, "total": 1025},
    ),
    "openai_chat": (
        {"model": "gpt-4o"},
        {
            "model": "gpt-4o",
            "usage": {
                "prompt_tokens": 2006,
                "completion_tokens": 300,
                "total_tokens": 2306,
                "prompt_tokens_details": {"cached_tokens": 1920},
            },
        },
        (2006, 300, 1920, 0),
        {"input": 86, "input_cached_tokens": 1920, "output": 300, "total": 2306},
    ),
    "openai_responses": (
        None,
        {
            "model": "gpt-5.6-sol",
            "usage": {
                "input_tokens": 5000,
                "output_tokens": 800,
                "total_tokens": 5800,
                "input_tokens_details": {
                    "cached_tokens": 3000,
                    "cache_write_tokens": 1500,
                    "cache_write_tokens_details": {
                        "ttl_30m_tokens": 500,
                        "ttl_12h_tokens": 1000,
                    },
                },
            },
        },
        (5000, 800, 3000, 1500),
        # Langfuse has no 12-hour key, so every write is input_cache_creation.
        {
            "input": 500,
            "input_cached_tokens": 3000,
            "input_cache_creation": 1500,
            "output": 800,
            "total": 5800,
        },
    ),
    "openai_sdk": (
        {"model": "gpt-4o", "max_tokens": 64},
        {
            "model": "gpt-4o",
            "input_tokens": 2006,
            "output_tokens": 300,
            "cached_input_tokens": 1920,
            "cache_write_input_tokens": 0,
        },
        (2006, 300, 1920, 0),
        {"input": 86, "input_cached_tokens": 1920, "output": 300, "total": 2306},
    ),
    "codex": (
        None,
        {
            "model": "gpt-5.3-codex",
            "input_tokens": 1000,
            "output_tokens": 100,
            "total_tokens": 1100,
            "token_usage": {
                "input_tokens": 1000,
                "cached_input_tokens": 800,
                "cache_write_input_tokens": 100,
                "output_tokens": 100,
                "reasoning_output_tokens": 20,
                "total_tokens": 1100,
            },
        },
        (1000, 100, 800, 100),
        {
            "input": 100,
            "input_cached_tokens": 800,
            "input_cache_creation": 100,
            "output": 100,
            "total": 1100,
        },
    ),
}


def _meta() -> SessionMeta:
    return SessionMeta(session_id="s1", agent_name="test", started_at=1000.0, ended_at=1100.0)


def _events(request: dict | None, response: dict) -> list[TraceEvent]:
    events = []
    if request is not None:
        events.append(TraceEvent(
            event_type=EventType.LLM_REQUEST,
            timestamp=1001.0,
            event_id="req1",
            data=dict(request),
        ))
    events.append(TraceEvent(
        event_type=EventType.LLM_RESPONSE,
        timestamp=1002.0,
        event_id="resp1",
        parent_id="req1" if request is not None else "",
        data=dict(response),
    ))
    return events


def _attrs(item: dict) -> dict:
    out = {}
    for attr in item.get("attributes", []):
        value = attr["value"]
        if "intValue" in value:
            out[attr["key"]] = int(value["intValue"])
        else:
            out[attr["key"]] = next(iter(value.values()))
    return out


def _spans(payload: dict) -> list[dict]:
    return payload["resourceSpans"][0]["scopeSpans"][0]["spans"]


def _genai_llm_spans(events: list[TraceEvent]) -> list[dict]:
    spans = _spans(session_to_otlp_genai(_meta(), events))
    return [_attrs(s) for s in spans if s["name"] == "gen_ai.client.operation"]


def _classic_llm_events(events: list[TraceEvent]) -> list[dict]:
    root = next(s for s in _spans(session_to_otlp(_meta(), events)) if not s.get("parentSpanId"))
    return [
        _attrs(e) for e in root.get("events", [])
        if e["name"] in ("gen_ai.system.message", "gen_ai.assistant.message")
    ]


def _classic_tokens(events: list[TraceEvent]) -> dict:
    """Merge the token attributes of the classic request and response events."""
    merged: dict = {}
    for attrs in _classic_llm_events(events):
        for key in TOKEN_KEYS:
            if key in attrs:
                assert key not in merged, f"{key} reported twice"
                merged[key] = attrs[key]
    return merged


def _generations(events: list[TraceEvent]) -> list[dict]:
    return [
        o for o in _events_to_langfuse_observations("s1", events)
        if o["type"] == "GENERATION"
    ]


def _expected_semconv(tokens: tuple[int, int, int, int]) -> dict:
    input_tokens, output_tokens, read, write = tokens
    expected = {INPUT: input_tokens, OUTPUT: output_tokens}
    if read:
        expected[CACHE_READ] = read
    if write:
        expected[CACHE_WRITE] = write
    return expected


class _ShapeTests:
    """One test per provider shape; subclasses check one export."""

    def check(self, name: str) -> None:
        raise NotImplementedError

    def test_anthropic_sdk(self):
        self.check("anthropic_sdk")

    def test_anthropic_nested_usage(self):
        self.check("anthropic_nested")

    def test_bedrock_converse(self):
        self.check("bedrock_converse")

    def test_gemini(self):
        self.check("gemini")

    def test_openai_chat_completions(self):
        self.check("openai_chat")

    def test_openai_responses_with_cache_write_lifetimes(self):
        self.check("openai_responses")

    def test_openai_sdk_integration(self):
        self.check("openai_sdk")

    def test_codex_token_usage(self):
        self.check("codex")


class TestGenAISemconvCacheUsage(_ShapeTests, unittest.TestCase):
    def check(self, name: str) -> None:
        request, response, tokens, _ = CASES[name]
        (span,) = _genai_llm_spans(_events(request, response))
        input_tokens, output_tokens, read, write = tokens
        expected = _expected_semconv(tokens)
        expected.update({
            OI_PROMPT: input_tokens,
            OI_COMPLETION: output_tokens,
            OI_TOTAL: input_tokens + output_tokens,
        })
        if read:
            expected[OI_CACHE_READ] = read
        if write:
            expected[OI_CACHE_WRITE] = write
        self.assertEqual({k: v for k, v in span.items() if k in TOKEN_KEYS}, expected)


class TestClassicOtlpCacheUsage(_ShapeTests, unittest.TestCase):
    def check(self, name: str) -> None:
        request, response, tokens, _ = CASES[name]
        self.assertEqual(_classic_tokens(_events(request, response)), _expected_semconv(tokens))


class TestLangfuseCacheUsage(_ShapeTests, unittest.TestCase):
    def check(self, name: str) -> None:
        request, response, tokens, details = CASES[name]
        (generation,) = _generations(_events(request, response))
        input_tokens, output_tokens, _, _ = tokens
        self.assertEqual(generation["usage"], {
            "input": input_tokens,
            "output": output_tokens,
            "total": input_tokens + output_tokens,
        })
        self.assertEqual(generation["usageDetails"], details)
        buckets = sum(v for k, v in details.items() if k != "total")
        self.assertEqual(buckets, details["total"])


class TestCacheUsageEdges(unittest.TestCase):
    def test_unanswered_request_with_usage_reports_cache_in_genai(self):
        request = {
            "model": "claude-sonnet-4-5",
            "input_tokens": 10,
            "cache_read_input_tokens": 90,
        }
        (span,) = _genai_llm_spans(_events(request, {})[:1])
        self.assertEqual(span[INPUT], 100)
        self.assertEqual(span[CACHE_READ], 90)
        self.assertEqual(span[OI_CACHE_READ], 90)
        self.assertEqual(span[OI_TOTAL], 100)
        self.assertNotIn(CACHE_WRITE, span)

    def test_classic_request_event_carries_its_own_cache_usage(self):
        request = {
            "model": "claude-sonnet-4-5",
            "input_tokens": 10,
            "cache_creation_input_tokens": 40,
        }
        response = {"model": "claude-sonnet-4-5", "output_tokens": 3}
        request_attrs, response_attrs = _classic_llm_events(_events(request, response))
        self.assertEqual(request_attrs[INPUT], 50)
        self.assertEqual(request_attrs[CACHE_WRITE], 40)
        self.assertNotIn(INPUT, response_attrs)
        self.assertNotIn(CACHE_WRITE, response_attrs)
        self.assertEqual(response_attrs[OUTPUT], 3)

    def test_sdk_response_usage_reports_input_without_cache(self):
        # The SDK integrations record usage on the response only.
        events = _events(
            {"model": "gpt-4o", "max_tokens": 64},
            {"model": "gpt-4o", "input_tokens": 120, "output_tokens": 30},
        )
        self.assertEqual(_classic_tokens(events), {INPUT: 120, OUTPUT: 30})
        (generation,) = _generations(events)
        self.assertEqual(generation["usage"], {"input": 120, "output": 30, "total": 150})
        self.assertNotIn("usageDetails", generation)

    def test_langfuse_long_writes_of_unknown_provider_are_not_labelled_one_hour(self):
        response = {
            "model": "custom-model",
            "input_tokens": 1,
            "output_tokens": 2,
            "cache_creation_input_tokens": 30,
            "cache_creation": {"ephemeral_1h_input_tokens": 30},
        }
        (generation,) = _generations(_events(None, response))
        self.assertEqual(generation["usageDetails"], {
            "input": 1,
            "input_cache_creation": 30,
            "output": 2,
            "total": 33,
        })

    def test_failed_request_linked_error_keeps_request_usage(self):
        request = {
            "model": "claude-sonnet-4-5",
            "input_tokens": 5,
            "cache_read_input_tokens": 15,
        }
        events = _events(request, {})[:1] + [TraceEvent(
            event_type=EventType.ERROR,
            timestamp=1002.0,
            event_id="err1",
            parent_id="req1",
            data={"error": "boom"},
        )]
        (span,) = _genai_llm_spans(events)
        self.assertEqual((span[INPUT], span[CACHE_READ]), (20, 15))
        (generation,) = _generations(events)
        self.assertEqual(generation["usage"], {"input": 20, "output": 0, "total": 20})
        self.assertEqual(generation["usageDetails"]["input_cached_tokens"], 15)


class TestUnchangedWithoutCache(unittest.TestCase):
    """Requests without cache usage export exactly as before."""

    REQUEST_INPUT = (
        {"model": "gpt-4o", "input_tokens": 50},
        {"model": "gpt-4o", "output_tokens": 7, "stop_reason": "stop"},
    )
    CODEX = (
        None,
        {
            "model": "gpt-5.3-codex",
            "input_tokens": 1000,
            "output_tokens": 100,
            "total_tokens": 1100,
            "token_usage": {
                "input_tokens": 1000,
                "cached_input_tokens": 0,
                "cache_write_input_tokens": 0,
                "output_tokens": 100,
                "reasoning_output_tokens": 20,
                "total_tokens": 1100,
            },
        },
    )

    def test_genai_request_input(self):
        (span,) = _genai_llm_spans(_events(*self.REQUEST_INPUT))
        self.assertEqual({k: v for k, v in span.items() if k in TOKEN_KEYS}, {
            INPUT: 50, OUTPUT: 7, OI_PROMPT: 50, OI_COMPLETION: 7, OI_TOTAL: 57,
        })

    def test_classic_request_input(self):
        request_attrs, response_attrs = _classic_llm_events(_events(*self.REQUEST_INPUT))
        self.assertEqual(request_attrs, {
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": "gpt-4o",
            INPUT: 50,
        })
        self.assertEqual(response_attrs, {
            "gen_ai.operation.name": "chat",
            OUTPUT: 7,
            "gen_ai.response.finish_reasons": "stop",
        })

    def test_langfuse_request_input(self):
        (generation,) = _generations(_events(*self.REQUEST_INPUT))
        self.assertEqual(generation["usage"], {"input": 50, "output": 7, "total": 57})
        self.assertNotIn("usageDetails", generation)

    def test_codex_zero_cache_counts(self):
        events = _events(*self.CODEX)
        (span,) = _genai_llm_spans(events)
        self.assertEqual({k: v for k, v in span.items() if k in TOKEN_KEYS}, {
            INPUT: 1000, OUTPUT: 100, OI_PROMPT: 1000, OI_COMPLETION: 100, OI_TOTAL: 1100,
        })
        self.assertEqual(_classic_tokens(events), {INPUT: 1000, OUTPUT: 100})
        (generation,) = _generations(events)
        self.assertEqual(generation["usage"], {"input": 1000, "output": 100, "total": 1100})
        self.assertNotIn("usageDetails", generation)

    def test_no_usage_reports_no_tokens(self):
        events = _events({"model": "gpt-4o"}, {"model": "gpt-4o", "text": "hi"})
        (span,) = _genai_llm_spans(events)
        self.assertFalse(set(span) & set(TOKEN_KEYS))
        self.assertEqual(_classic_tokens(events), {})
        (generation,) = _generations(events)
        self.assertEqual(generation["usage"], {"input": 0, "output": 0, "total": 0})
        self.assertNotIn("usageDetails", generation)


class TestUsageHelpers(unittest.TestCase):
    def test_usage_detail_ignores_non_objects(self):
        self.assertEqual(usage_detail(None), (TokenUsage(), False, False))
        self.assertEqual(usage_detail(["input_tokens", 3]), (TokenUsage(), False, False))

    def test_model_request_usage_takes_input_from_request_first(self):
        usage, has_input, has_output = model_request_usage(
            {"input_tokens": 10, "cache_read_input_tokens": 5},
            {"input_tokens": 99, "output_tokens": 4},
        )
        self.assertEqual(
            usage,
            TokenUsage(input_tokens=15, output_tokens=4, cache_read_tokens=5),
        )
        self.assertEqual((has_input, has_output), (True, True))

    def test_model_request_usage_falls_back_to_response_input(self):
        usage, has_input, has_output = model_request_usage(
            {"model": "gpt-4o"},
            {"prompt_tokens": 8, "completion_tokens": 2},
        )
        self.assertEqual(usage, TokenUsage(input_tokens=8, output_tokens=2))
        self.assertEqual((has_input, has_output), (True, True))


if __name__ == "__main__":
    unittest.main()
