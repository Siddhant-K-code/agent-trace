"""The SDK integrations record async calls, the Responses API, streams and cache usage.

The SDKs are replaced by stubs laid out like the real packages (see
``_sdk_stubs``), so no SDK, network or API key is needed.
"""

from __future__ import annotations

import asyncio
import csv
import gc
import importlib
import io
import os
import sys
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from unittest.mock import patch

try:
    import _sdk_stubs as stubs
except ImportError:  # run as a package module
    from tests import _sdk_stubs as stubs

from agent_trace.cost import _session_model_token_usage, cmd_cost
from agent_trace.models import EventType
from agent_trace.store import TraceStore

ns = stubs.ns
HI = [{"role": "user", "content": "hi"}]
# claude-sonnet-4-5 rates in USD per 1M tokens: input, output, cache read,
# 5-minute cache write, 1-hour cache write.
SONNET = {"input": 3.00, "output": 15.00, "read": 0.30, "write": 3.75, "long": 6.00}


class RateLimitError(Exception):
    pass


def _raise(kwargs):
    raise RateLimitError("Rate limit reached")


def _run(coroutine):
    return asyncio.run(coroutine)


class SDKCase(unittest.TestCase):
    """Instruments stub SDKs into a fresh trace directory."""

    def setUp(self):
        self.trace_dir = tempfile.mkdtemp()
        env = {k: v for k, v in os.environ.items() if k != "AGENT_STRACE_ENDPOINT"}
        env["AGENT_TRACE_DIR"] = self.trace_dir
        environ = patch.dict(os.environ, env, clear=True)
        environ.start()
        self.addCleanup(environ.stop)

    def instrument(self, name, modules):
        integration = importlib.import_module(f"agent_trace.integrations.{name}")
        modules_patch = patch.dict(sys.modules, modules)
        modules_patch.start()
        self.addCleanup(modules_patch.stop)
        integration._PATCHED = False
        getattr(integration, f"instrument_{name}")(agent_name=f"{name}-test")
        self.addCleanup(getattr(integration, f"uninstrument_{name}"))
        return integration

    def events(self):
        """Events of every recorded session; each integration records its own."""
        store = TraceStore(self.trace_dir)
        events = []
        for meta in store.list_sessions():
            events.extend(store.load_events(meta.session_id))
        return sorted(events, key=lambda event: event.timestamp)

    def model_usage(self):
        """Usage by model, as ``agent-strace cost --breakdown provider`` counts it."""
        return _session_model_token_usage(self.events())

    def provider_costs(self):
        """Rows of ``agent-strace cost --breakdown provider --csv``."""
        out = io.StringIO()
        args = Namespace(trace_dir=self.trace_dir, breakdown="provider", since="30d",
                         csv=True, live_pricing=False, tenant=None)
        with redirect_stdout(out):
            self.assertEqual(cmd_cost(args), 0)
        return list(csv.DictReader(io.StringIO(out.getvalue())))

    def assertProviderCost(self, model, input_tokens, read, write, cost):
        (row,) = self.provider_costs()
        self.assertEqual(
            (row["model"], row["input_tokens"], row["cache_read_tokens"],
             row["cache_write_tokens"], row["cost_usd"]),
            (model, str(input_tokens), str(read), str(write), f"{cost:.8f}"),
        )

    def pair(self):
        """The recorded request and response or error of the only model call."""
        events = self.events()
        self.assertEqual(len(events), 2, [e.event_type for e in events])
        request, response = events
        self.assertEqual(request.event_type, EventType.LLM_REQUEST)
        return request, response

    def assertLinked(self, request, response):
        self.assertEqual(response.parent_id, request.event_id)
        self.assertIsNotNone(response.duration_ms)

    def assertResponse(self, response, expected):
        self.assertEqual(response.event_type, EventType.LLM_RESPONSE)
        data = dict(response.data)
        self.assertIsInstance(data.pop("duration_ms"), float)
        self.assertEqual(data, expected)

    def assertError(self, request, error):
        self.assertEqual(error.event_type, EventType.ERROR)
        self.assertLinked(request, error)
        self.assertEqual(error.data["error"], "Rate limit reached")
        self.assertEqual(error.data["error_type"], "RateLimitError")

    def assertCancelled(self, request, response, model):
        self.assertResponse(response, {"model": model, "status": "cancelled"})
        self.assertLinked(request, response)


def _cost(uncached=0, read=0, write=0, long_write=0, output=0, rates=SONNET):
    return (uncached * rates["input"] + read * rates["read"] + write * rates["write"]
            + long_write * rates["long"] + output * rates["output"]) / 1_000_000


# ---------------------------------------------------------------------------
# OpenAI Chat Completions
# ---------------------------------------------------------------------------

def _completion(usage=None, finish_reason="stop"):
    return ns(usage=usage, choices=[ns(index=0, finish_reason=finish_reason)])


def _chunk(finish_reason=None, usage=None, choices=True):
    return ns(usage=usage,
              choices=[ns(index=0, finish_reason=finish_reason, delta=ns(content="x"))]
              if choices else [])


CHAT_USAGE = ns(prompt_tokens=1000, completion_tokens=50,
                prompt_tokens_details=ns(cached_tokens=800, cache_write_tokens=100))


class TestOpenAIChat(SDKCase):
    def instrument_chat(self, handler):
        modules = stubs.openai_modules(chat=handler)
        self.instrument("openai", modules)
        module = modules["openai.resources.chat.completions"]
        return module.Completions(), module.AsyncCompletions()

    def test_sync_call_records_the_same_fields(self):
        sync, _ = self.instrument_chat(lambda kw: _completion(
            ns(prompt_tokens=10, completion_tokens=5, prompt_tokens_details=None)))
        sync.create(model="gpt-4o", messages=HI, max_tokens=100)
        request, response = self.pair()
        self.assertEqual(request.data, {"model": "gpt-4o", "max_tokens": 100, "message_count": 1})
        self.assertResponse(response, {"model": "gpt-4o", "input_tokens": 10,
                                       "output_tokens": 5, "finish_reason": "stop"})
        self.assertLinked(request, response)

    def test_sync_call_records_cache_counts(self):
        sync, _ = self.instrument_chat(lambda kw: _completion(CHAT_USAGE))
        sync.create(model="gpt-4o", messages=HI)
        _, response = self.pair()
        self.assertResponse(response, {
            "model": "gpt-4o", "input_tokens": 1000, "output_tokens": 50,
            "cached_input_tokens": 800, "cache_write_input_tokens": 100,
            "finish_reason": "stop",
        })

    def test_async_call(self):
        _, client = self.instrument_chat(lambda kw: _completion(CHAT_USAGE, "length"))
        result = _run(client.create(model="gpt-4o", messages=HI, max_tokens=7))
        self.assertEqual(result.choices[0].finish_reason, "length")
        request, response = self.pair()
        self.assertEqual(request.data, {"model": "gpt-4o", "max_tokens": 7, "message_count": 1})
        self.assertResponse(response, {
            "model": "gpt-4o", "input_tokens": 1000, "output_tokens": 50,
            "cached_input_tokens": 800, "cache_write_input_tokens": 100,
            "finish_reason": "length",
        })
        self.assertLinked(request, response)

    def test_async_failure_is_linked(self):
        _, client = self.instrument_chat(_raise)
        with self.assertRaises(RateLimitError):
            _run(client.create(model="gpt-4o", messages=HI))
        self.assertError(*self.pair())

    def test_async_cancellation_is_recorded(self):
        _, client = self.instrument_chat(lambda kw: asyncio.sleep(60))

        async def cancel():
            task = asyncio.ensure_future(client.create(model="gpt-4o", messages=HI))
            await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        _run(cancel())
        self.assertCancelled(*self.pair(), "gpt-4o")

    def test_arguments_left_out_are_recorded_as_absent(self):
        sync, _ = self.instrument_chat(lambda kw: _completion())
        sync.create(model="gpt-4o", messages=HI, max_tokens=stubs.NotGiven(),
                    stream=stubs.Omit())
        request, response = self.pair()
        self.assertEqual(request.data, {"model": "gpt-4o", "max_tokens": 0, "message_count": 1})
        self.assertEqual(response.event_type, EventType.LLM_RESPONSE)

    def test_messages_generator_is_not_consumed(self):
        seen = []
        sync, _ = self.instrument_chat(lambda kw: seen.extend(kw["messages"]) or _completion())
        sync.create(model="gpt-4o", messages=(m for m in HI))
        self.assertEqual(seen, HI)
        request, _ = self.pair()
        self.assertEqual(request.data["message_count"], 0)

    def test_raw_response_is_recorded_when_parsed(self):
        sync, _ = self.instrument_chat(lambda kw: _completion(CHAT_USAGE))
        raw = sync.with_raw_response.create(model="gpt-4o", messages=HI)
        self.assertIsInstance(raw, stubs.RawResponse)
        self.assertEqual(len(self.events()), 1)
        body = raw.parse()
        self.assertIs(raw.parse(), body)
        self.assertIs(body.usage, CHAT_USAGE)
        request, response = self.pair()
        self.assertEqual(request.data, {"model": "gpt-4o", "max_tokens": 0, "message_count": 1})
        self.assertResponse(response, {
            "model": "gpt-4o", "input_tokens": 1000, "output_tokens": 50,
            "cached_input_tokens": 800, "cache_write_input_tokens": 100,
            "finish_reason": "stop",
        })
        self.assertLinked(request, response)

    def test_raw_response_never_parsed_records_only_the_request(self):
        sync, _ = self.instrument_chat(lambda kw: _completion(CHAT_USAGE))
        sync.with_raw_response.create(model="gpt-4o", messages=HI)
        (request,) = self.events()
        self.assertEqual(request.event_type, EventType.LLM_REQUEST)

    def test_raw_stream_is_recorded_at_its_end(self):
        chunks = [_chunk("stop"), _chunk(usage=CHAT_USAGE, choices=False)]
        stream = stubs.Stream(chunks)
        sync, _ = self.instrument_chat(lambda kw: stream)
        raw = sync.with_raw_response.create(model="gpt-4o", messages=HI, stream=True)
        self.assertIs(raw.parse(), stream)
        self.assertEqual(len(self.events()), 1)
        self.assertEqual(list(stream), chunks)
        request, response = self.pair()
        self.assertEqual(response.data["input_tokens"], 1000)
        self.assertLinked(request, response)

    def test_stream_is_passed_through_and_usage_recorded_at_its_end(self):
        chunks = [_chunk(), _chunk("stop"),
                  _chunk(usage=CHAT_USAGE, choices=False)]
        stream = stubs.Stream(chunks)
        sync, _ = self.instrument_chat(lambda kw: stream)
        result = sync.create(model="gpt-4o", messages=HI, stream=True,
                             stream_options={"include_usage": True})
        self.assertIs(result, stream)
        (request,) = self.events()
        items = list(result)
        self.assertEqual(len(items), len(chunks))
        for item, chunk in zip(items, chunks):
            self.assertIs(item, chunk)
        request, response = self.pair()
        self.assertResponse(response, {
            "model": "gpt-4o", "input_tokens": 1000, "output_tokens": 50,
            "cached_input_tokens": 800, "cache_write_input_tokens": 100,
            "finish_reason": "stop",
        })
        self.assertLinked(request, response)

    def test_stream_read_with_next(self):
        chunks = [_chunk("stop"), _chunk(usage=CHAT_USAGE, choices=False)]
        sync, _ = self.instrument_chat(lambda kw: stubs.Stream(chunks))
        stream = sync.create(model="gpt-4o", messages=HI, stream=True)
        self.assertIs(next(stream), chunks[0])
        self.assertIs(next(stream), chunks[1])
        with self.assertRaises(StopIteration):
            next(stream)
        _, response = self.pair()
        self.assertEqual(response.data["input_tokens"], 1000)

    def test_stream_without_usage_records_no_token_counts(self):
        sync, _ = self.instrument_chat(lambda kw: stubs.Stream([_chunk(), _chunk("stop")]))
        list(sync.create(model="gpt-4o", messages=HI, stream=True))
        _, response = self.pair()
        self.assertResponse(response, {"model": "gpt-4o", "finish_reason": "stop"})

    def test_stream_failure_is_linked_and_raised(self):
        chunks = [_chunk()]
        sync, _ = self.instrument_chat(
            lambda kw: stubs.Stream(chunks, error=RateLimitError("Rate limit reached")))
        stream = sync.create(model="gpt-4o", messages=HI, stream=True)
        received = []
        with self.assertRaises(RateLimitError):
            for chunk in stream:
                received.append(chunk)
        self.assertEqual(received, chunks)
        self.assertError(*self.pair())

    def test_dropped_stream_records_only_the_request(self):
        sync, _ = self.instrument_chat(lambda kw: stubs.Stream([_chunk(), _chunk("stop")]))
        stream = sync.create(model="gpt-4o", messages=HI, stream=True)
        next(iter(stream))
        del stream
        gc.collect()
        (request,) = self.events()
        self.assertEqual(request.event_type, EventType.LLM_REQUEST)

    def test_stream_it_cannot_see_inside_is_recorded_when_returned(self):
        opaque = stubs.Opaque([_chunk()])
        sync, _ = self.instrument_chat(lambda kw: opaque)
        self.assertIs(sync.create(model="gpt-4o", messages=HI, stream=True), opaque)
        _, response = self.pair()
        self.assertResponse(response, {"model": "gpt-4o", "input_tokens": 0,
                                       "output_tokens": 0, "finish_reason": ""})

    def test_async_stream(self):
        chunks = [_chunk("stop"), _chunk(usage=CHAT_USAGE, choices=False)]
        stream = stubs.AsyncStream(chunks)
        _, client = self.instrument_chat(lambda kw: stream)

        async def read():
            result = await client.create(model="gpt-4o", messages=HI, stream=True)
            self.assertIs(result, stream)
            return [chunk async for chunk in result]

        self.assertEqual(_run(read()), chunks)
        request, response = self.pair()
        self.assertEqual(response.data["cached_input_tokens"], 800)
        self.assertEqual(response.data["finish_reason"], "stop")
        self.assertLinked(request, response)

    def test_async_stream_read_with_anext(self):
        chunks = [_chunk("stop")]
        _, client = self.instrument_chat(lambda kw: stubs.AsyncStream(chunks))

        async def read():
            stream = await client.create(model="gpt-4o", messages=HI, stream=True)
            first = await stream.__anext__()
            with self.assertRaises(StopAsyncIteration):
                await stream.__anext__()
            return first

        self.assertIs(_run(read()), chunks[0])
        self.assertResponse(self.pair()[1], {"model": "gpt-4o", "finish_reason": "stop"})

    def test_cancelled_async_stream_is_recorded(self):
        _, client = self.instrument_chat(lambda kw: stubs.AsyncStream(
            [_chunk()], wait=asyncio.sleep(60)))

        async def read():
            stream = await client.create(model="gpt-4o", messages=HI, stream=True)
            task = asyncio.ensure_future(_drain(stream))
            await asyncio.sleep(0.01)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        _run(read())
        self.assertCancelled(*self.pair(), "gpt-4o")


async def _drain(stream):
    return [item async for item in stream]


# ---------------------------------------------------------------------------
# OpenAI Responses API
# ---------------------------------------------------------------------------

RESPONSES_USAGE = ns(input_tokens=1000, output_tokens=50,
                     input_tokens_details=ns(cached_tokens=800, cache_write_tokens=100))


def _response(usage=RESPONSES_USAGE, status="completed", model="gpt-5-2025-08-07",
              incomplete=None, error=None):
    return ns(model=model, status=status, usage=usage, error=error,
              incomplete_details=ns(reason=incomplete) if incomplete else None)


class TestOpenAIResponses(SDKCase):
    def instrument_responses(self, handler):
        modules = stubs.openai_modules(responses=handler)
        self.instrument("openai", modules)
        module = modules["openai.resources.responses"]
        return module.Responses(), module.AsyncResponses()

    def test_sync_call(self):
        sync, _ = self.instrument_responses(lambda kw: _response())
        sync.create(model="gpt-5", input="hello", max_output_tokens=500)
        request, response = self.pair()
        self.assertEqual(request.data, {"model": "gpt-5", "max_tokens": 500, "message_count": 1})
        self.assertResponse(response, {
            "model": "gpt-5", "input_tokens": 1000, "output_tokens": 50,
            "cached_input_tokens": 800, "cache_write_input_tokens": 100,
            "finish_reason": "completed", "status": "completed",
        })
        self.assertLinked(request, response)

    def test_input_items_are_counted(self):
        sync, _ = self.instrument_responses(lambda kw: _response())
        sync.create(model="gpt-5", input=HI * 3)
        request, _ = self.pair()
        self.assertEqual(request.data["message_count"], 3)
        self.assertEqual(request.data["max_tokens"], 0)

    def test_incomplete_response_records_its_reason(self):
        sync, _ = self.instrument_responses(
            lambda kw: _response(status="incomplete", incomplete="max_output_tokens"))
        sync.create(model="gpt-5", input="hello")
        _, response = self.pair()
        self.assertEqual(response.data["finish_reason"], "max_output_tokens")
        self.assertEqual(response.data["status"], "incomplete")

    def test_failed_response_records_its_error(self):
        sync, _ = self.instrument_responses(lambda kw: _response(
            usage=None, status="failed",
            error=ns(code="server_error", message="The server had an error")))
        sync.create(model="gpt-5", input="hello")
        _, response = self.pair()
        self.assertResponse(response, {
            "model": "gpt-5", "input_tokens": 0, "output_tokens": 0,
            "finish_reason": "failed", "status": "failed",
            "error": "The server had an error", "error_type": "server_error",
        })

    def test_model_comes_from_the_response_when_the_request_names_none(self):
        sync, _ = self.instrument_responses(lambda kw: _response())
        sync.create(prompt={"id": "pmpt_1"}, input="hello")
        request, response = self.pair()
        self.assertEqual(request.data["model"], "")
        self.assertEqual(response.data["model"], "gpt-5-2025-08-07")

    def test_stream_helper_without_a_model(self):
        # responses.stream() passes omit for every argument the caller left out.
        events = self._events(_response())
        sync, _ = self.instrument_responses(lambda kw: stubs.Stream(events))
        list(sync.create(model=stubs.Omit(), input="hello", max_output_tokens=stubs.Omit(),
                         prompt={"id": "pmpt_1"}, stream=True))
        request, response = self.pair()
        self.assertEqual(request.data, {"model": "", "max_tokens": 0, "message_count": 1})
        self.assertEqual(response.data["model"], "gpt-5-2025-08-07")
        self.assertEqual(response.data["input_tokens"], 1000)

    def test_async_call(self):
        _, client = self.instrument_responses(lambda kw: _response())
        _run(client.create(model="gpt-5", input="hello"))
        request, response = self.pair()
        self.assertEqual(response.data["cached_input_tokens"], 800)
        self.assertLinked(request, response)

    def test_failures_are_linked(self):
        sync, client = self.instrument_responses(_raise)
        with self.assertRaises(RateLimitError):
            sync.create(model="gpt-5", input="hello")
        self.assertError(*self.pair())

    def test_async_failure_is_linked(self):
        _, client = self.instrument_responses(_raise)
        with self.assertRaises(RateLimitError):
            _run(client.create(model="gpt-5", input="hello"))
        self.assertError(*self.pair())

    def _events(self, final):
        return [
            ns(type="response.created", response=_response(usage=None, status="in_progress")),
            ns(type="response.output_text.delta", delta="hi"),
            ns(type="response.completed", response=final),
        ]

    def test_stream(self):
        events = self._events(_response())
        stream = stubs.Stream(events)
        sync, _ = self.instrument_responses(lambda kw: stream)
        result = sync.create(model="gpt-5", input="hello", stream=True)
        self.assertIs(result, stream)
        self.assertEqual(list(result), events)
        request, response = self.pair()
        self.assertResponse(response, {
            "model": "gpt-5", "input_tokens": 1000, "output_tokens": 50,
            "cached_input_tokens": 800, "cache_write_input_tokens": 100,
            "finish_reason": "completed", "status": "completed",
        })
        self.assertLinked(request, response)

    def test_async_stream(self):
        events = self._events(_response(status="incomplete", incomplete="max_output_tokens"))
        _, client = self.instrument_responses(lambda kw: stubs.AsyncStream(events))

        async def read():
            return await _drain(await client.create(model="gpt-5", input="x", stream=True))

        self.assertEqual(_run(read()), events)
        _, response = self.pair()
        self.assertEqual(response.data["finish_reason"], "max_output_tokens")
        self.assertEqual(response.data["input_tokens"], 1000)


# ---------------------------------------------------------------------------
# Anthropic Messages
# ---------------------------------------------------------------------------

def _message(usage, stop_reason="end_turn"):
    return ns(usage=usage, stop_reason=stop_reason)


def _message_events(start_usage, delta_usage, stop_reason="end_turn"):
    return [
        ns(type="message_start", message=ns(usage=start_usage)),
        ns(type="content_block_start", index=0),
        ns(type="content_block_delta", index=0, delta=ns(text="hi")),
        ns(type="content_block_stop", index=0),
        ns(type="message_delta", delta=ns(stop_reason=stop_reason), usage=delta_usage),
        ns(type="message_stop"),
    ]


START_USAGE = ns(input_tokens=10, output_tokens=1, cache_read_input_tokens=100,
                 cache_creation_input_tokens=50,
                 cache_creation=ns(ephemeral_5m_input_tokens=20, ephemeral_1h_input_tokens=30))
DELTA_USAGE = ns(output_tokens=42, input_tokens=None, cache_read_input_tokens=None,
                 cache_creation_input_tokens=None)
STREAM_FIELDS = {
    "model": "claude-sonnet-4-5", "input_tokens": 10, "output_tokens": 42,
    "cache_read_input_tokens": 100, "cache_creation_input_tokens": 50,
    "cache_creation": {"ephemeral_5m_input_tokens": 20, "ephemeral_1h_input_tokens": 30},
    "stop_reason": "end_turn",
}


class TestAnthropicMessages(SDKCase):
    def instrument_messages(self, handler):
        modules = stubs.anthropic_modules(handler)
        self.instrument("anthropic", modules)
        module = modules["anthropic.resources.messages"]
        return module.Messages(), module.AsyncMessages()

    def test_sync_call_records_the_same_fields(self):
        sync, _ = self.instrument_messages(
            lambda kw: _message(ns(input_tokens=10, output_tokens=5)))
        sync.create(model="claude-sonnet-4-5", messages=HI, max_tokens=1024)
        request, response = self.pair()
        self.assertEqual(request.data, {"model": "claude-sonnet-4-5", "max_tokens": 1024,
                                        "message_count": 1})
        self.assertResponse(response, {"model": "claude-sonnet-4-5", "input_tokens": 10,
                                       "output_tokens": 5, "stop_reason": "end_turn"})
        self.assertLinked(request, response)

    def test_async_call(self):
        _, client = self.instrument_messages(lambda kw: _message(START_USAGE, "max_tokens"))
        _run(client.create(model="claude-sonnet-4-5", messages=HI, max_tokens=1))
        request, response = self.pair()
        self.assertResponse(response, {**STREAM_FIELDS, "output_tokens": 1,
                                       "stop_reason": "max_tokens"})
        self.assertLinked(request, response)

    def test_async_failure_is_linked(self):
        _, client = self.instrument_messages(_raise)
        with self.assertRaises(RateLimitError):
            _run(client.create(model="claude-sonnet-4-5", messages=HI, max_tokens=1))
        self.assertError(*self.pair())

    def test_stream_is_passed_through_and_usage_recorded_at_its_end(self):
        events = _message_events(START_USAGE, DELTA_USAGE)
        stream = stubs.CachedStream(events)
        sync, _ = self.instrument_messages(lambda kw: stream)
        result = sync.create(model="claude-sonnet-4-5", messages=HI, max_tokens=1024,
                             stream=True)
        self.assertIs(result, stream)
        self.assertEqual(len(self.events()), 1)
        self.assertEqual(list(result), events)
        request, response = self.pair()
        self.assertResponse(response, STREAM_FIELDS)
        self.assertLinked(request, response)

    def test_message_delta_usage_replaces_the_start_counts(self):
        delta = ns(output_tokens=42, input_tokens=12, cache_read_input_tokens=200,
                   cache_creation_input_tokens=0)
        sync, _ = self.instrument_messages(
            lambda kw: stubs.CachedStream(_message_events(START_USAGE, delta)))
        list(sync.create(model="claude-sonnet-4-5", messages=HI, max_tokens=1, stream=True))
        _, response = self.pair()
        self.assertEqual(response.data["input_tokens"], 12)
        self.assertEqual(response.data["cache_read_input_tokens"], 200)
        self.assertEqual(response.data["cache_creation_input_tokens"], 0)

    def test_stream_failure_is_linked_and_raised(self):
        events = _message_events(START_USAGE, DELTA_USAGE)[:3]
        sync, _ = self.instrument_messages(lambda kw: stubs.CachedStream(
            events, error=RateLimitError("Rate limit reached")))
        stream = sync.create(model="claude-sonnet-4-5", messages=HI, max_tokens=1, stream=True)
        with self.assertRaises(RateLimitError):
            list(stream)
        self.assertError(*self.pair())

    def test_async_stream(self):
        events = _message_events(START_USAGE, DELTA_USAGE)
        stream = stubs.AsyncCachedStream(events)
        _, client = self.instrument_messages(lambda kw: stream)

        async def read():
            result = await client.create(model="claude-sonnet-4-5", messages=HI,
                                         max_tokens=1, stream=True)
            self.assertIs(result, stream)
            return await _drain(result)

        self.assertEqual(_run(read()), events)
        request, response = self.pair()
        self.assertResponse(response, STREAM_FIELDS)
        self.assertLinked(request, response)

    def test_raw_response_is_recorded_when_parsed(self):
        sync, _ = self.instrument_messages(
            lambda kw: _message(ns(input_tokens=10, output_tokens=5)))
        raw = sync.with_raw_response.create(model="claude-sonnet-4-5", messages=HI,
                                            max_tokens=1024)
        self.assertIsInstance(raw.parse(to=str), str)
        self.assertEqual(len(self.events()), 1)
        raw.parse()
        request, response = self.pair()
        self.assertResponse(response, {"model": "claude-sonnet-4-5", "input_tokens": 10,
                                       "output_tokens": 5, "stop_reason": "end_turn"})
        self.assertLinked(request, response)

    def test_async_raw_stream_is_recorded_at_its_end(self):
        """LangChain's ChatAnthropic streams through ``with_raw_response``."""
        events = _message_events(START_USAGE, DELTA_USAGE)
        stream = stubs.AsyncCachedStream(events)
        _, client = self.instrument_messages(lambda kw: stream)

        async def read():
            raw = await client.with_raw_response.create(
                model="claude-sonnet-4-5", messages=HI, max_tokens=1, stream=True)
            body = await raw.parse()
            self.assertIs(body, stream)
            return await _drain(body)

        self.assertEqual(_run(read()), events)
        request, response = self.pair()
        self.assertResponse(response, STREAM_FIELDS)
        self.assertLinked(request, response)

    def test_stream_cost(self):
        sync, _ = self.instrument_messages(lambda kw: stubs.CachedStream(
            _message_events(START_USAGE, DELTA_USAGE)))
        list(sync.create(model="claude-sonnet-4-5", messages=HI, max_tokens=1, stream=True))
        usage = self.model_usage()["claude-sonnet-4-5"]
        self.assertEqual((usage.input_tokens, usage.cache_read_tokens,
                          usage.cache_write_tokens, usage.cache_write_long_tokens),
                         (160, 100, 50, 30))
        self.assertProviderCost("claude-sonnet-4-5", 160, 100, 50, _cost(
            uncached=10, read=100, write=20, long_write=30, output=42))


# ---------------------------------------------------------------------------
# LiteLLM
# ---------------------------------------------------------------------------

def _litellm_response(usage):
    return ns(usage=usage, choices=[ns(finish_reason="stop")])


# Anthropic through LiteLLM: prompt_tokens includes cache reads and writes.
LITELLM_USAGE = ns(prompt_tokens=1000, completion_tokens=50,
                   prompt_tokens_details=ns(cached_tokens=800, cache_write_tokens=100),
                   cache_read_input_tokens=800, cache_creation_input_tokens=100)


class TestLiteLLM(SDKCase):
    def instrument_litellm(self, handler):
        modules = stubs.litellm_module(handler)
        self.instrument("litellm", modules)
        return modules["litellm"]

    def test_call_without_cache_records_the_same_fields(self):
        litellm = self.instrument_litellm(lambda kw: _litellm_response(
            ns(prompt_tokens=10, completion_tokens=5, prompt_tokens_details=None)))
        litellm.completion(model="gpt-4o", messages=HI)
        request, response = self.pair()
        self.assertEqual(request.data, {"model": "gpt-4o", "message_count": 1})
        self.assertResponse(response, {"model": "gpt-4o", "input_tokens": 10,
                                       "output_tokens": 5})
        self.assertLinked(request, response)

    def test_cache_counts(self):
        litellm = self.instrument_litellm(lambda kw: _litellm_response(LITELLM_USAGE))
        litellm.completion("claude-sonnet-4-5", HI)
        request, response = self.pair()
        self.assertEqual(request.data, {"model": "claude-sonnet-4-5", "message_count": 1})
        self.assertResponse(response, {
            "model": "claude-sonnet-4-5", "input_tokens": 1000, "output_tokens": 50,
            "cached_input_tokens": 800, "cache_write_input_tokens": 100,
        })

    def test_cache_counts_reported_only_beside_the_prompt_count(self):
        usage = ns(prompt_tokens=1000, completion_tokens=50,
                   prompt_tokens_details=ns(cached_tokens=None),
                   cache_read_input_tokens=800, cache_creation_input_tokens=100)
        litellm = self.instrument_litellm(lambda kw: _litellm_response(usage))
        litellm.completion(model="claude-sonnet-4-5", messages=HI)
        _, response = self.pair()
        self.assertEqual(response.data["cached_input_tokens"], 800)
        self.assertEqual(response.data["cache_write_input_tokens"], 100)

    def test_acompletion(self):
        litellm = self.instrument_litellm(lambda kw: _litellm_response(LITELLM_USAGE))
        _run(litellm.acompletion(model="claude-sonnet-4-5", messages=HI))
        request, response = self.pair()
        self.assertEqual(response.data["cached_input_tokens"], 800)
        self.assertLinked(request, response)

    def test_acompletion_failure_is_linked(self):
        litellm = self.instrument_litellm(_raise)
        with self.assertRaises(RateLimitError):
            _run(litellm.acompletion(model="gpt-4o", messages=HI))
        self.assertError(*self.pair())

    def test_stream_records_no_token_counts(self):
        stream = iter(["chunk"])
        litellm = self.instrument_litellm(lambda kw: stream)
        self.assertIs(litellm.completion(model="gpt-4o", messages=HI, stream=True), stream)
        _, response = self.pair()
        self.assertResponse(response, {"model": "gpt-4o"})

    def test_cost_counts_cache_tokens_once(self):
        litellm = self.instrument_litellm(lambda kw: _litellm_response(LITELLM_USAGE))
        litellm.completion(model="claude-sonnet-4-5", messages=HI)
        usage = self.model_usage()["claude-sonnet-4-5"]
        self.assertEqual((usage.input_tokens, usage.cache_read_tokens, usage.cache_write_tokens),
                         (1000, 800, 100))
        self.assertProviderCost("claude-sonnet-4-5", 1000, 800, 100, _cost(
            uncached=100, read=800, write=100, output=50))


# ---------------------------------------------------------------------------
# LangChain
# ---------------------------------------------------------------------------

OPENAI_METADATA = {"input_tokens": 1000, "output_tokens": 50, "total_tokens": 1050,
                   "input_token_details": {"cache_read": 800, "cache_creation": 100}}


class TestLangChain(SDKCase):
    def instrument_langchain(self):
        modules, base = stubs.langchain_modules()
        self.instrument("langchain", modules)
        return base

    def chat_model(self, base, usage_metadata=None, name="claude-sonnet-4-5"):
        calls = []

        class FakeChatModel(base):
            model_name = name

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                calls.append(run_manager)
                return stubs.chat_result(usage_metadata)

        return FakeChatModel(), calls

    def test_invoke_records_usage_and_cache_counts(self):
        base = self.instrument_langchain()
        model, calls = self.chat_model(base, OPENAI_METADATA)
        model.invoke(["hi", "there"])
        self.assertEqual(calls, ["manager"])
        request, response = self.pair()
        self.assertEqual(request.data, {"model": "claude-sonnet-4-5", "message_count": 2})
        self.assertResponse(response, {
            "model": "claude-sonnet-4-5", "input_tokens": 1000, "output_tokens": 50,
            "cached_input_tokens": 800, "cache_write_input_tokens": 100,
        })
        self.assertLinked(request, response)

    def test_cache_writes_split_by_ttl(self):
        base = self.instrument_langchain()
        metadata = {"input_tokens": 1000, "output_tokens": 50, "input_token_details": {
            "cache_read": 800, "cache_creation": 0,
            "ephemeral_5m_input_tokens": 60, "ephemeral_1h_input_tokens": 40}}
        model, _ = self.chat_model(base, metadata)
        model.invoke(["hi"])
        _, response = self.pair()
        self.assertEqual(response.data["cached_input_tokens"], 800)
        self.assertEqual(response.data["cache_write_input_tokens"], 100)

    def test_service_tier_cache_counts(self):
        base = self.instrument_langchain()
        metadata = {"input_tokens": 1000, "output_tokens": 50, "input_token_details": {
            "priority_cache_read": 800, "priority": 200}}
        model, _ = self.chat_model(base, metadata)
        model.invoke(["hi"])
        _, response = self.pair()
        self.assertEqual(response.data["cached_input_tokens"], 800)
        self.assertNotIn("cache_write_input_tokens", response.data)

    def test_model_without_usage_records_the_same_fields(self):
        base = self.instrument_langchain()
        model, _ = self.chat_model(base, None, name="local-model")
        model.invoke(["hi"])
        request, response = self.pair()
        self.assertEqual(request.data, {"model": "local-model", "message_count": 1})
        self.assertResponse(response, {"model": "local-model"})

    def test_ainvoke_with_the_default_agenerate(self):
        base = self.instrument_langchain()
        model, calls = self.chat_model(base, OPENAI_METADATA)
        _run(model.ainvoke(["hi"]))
        self.assertEqual(calls, ["manager"])
        request, response = self.pair()
        self.assertEqual(response.data["cached_input_tokens"], 800)
        self.assertLinked(request, response)

    def test_ainvoke_with_its_own_agenerate(self):
        base = self.instrument_langchain()

        class AsyncChatModel(base):
            model_name = "gpt-4o"

            async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
                return stubs.chat_result(OPENAI_METADATA)

        _run(AsyncChatModel().ainvoke(["hi"]))
        request, response = self.pair()
        self.assertEqual(response.data["input_tokens"], 1000)
        self.assertLinked(request, response)

    def test_failure_is_linked(self):
        base = self.instrument_langchain()

        class FailingChatModel(base):
            model_name = "gpt-4o"

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                raise RateLimitError("Rate limit reached")

        with self.assertRaises(RateLimitError):
            FailingChatModel().invoke(["hi"])
        self.assertError(*self.pair())

    def test_uninstrument_restores_the_model_class(self):
        base = self.instrument_langchain()
        model, _ = self.chat_model(base, OPENAI_METADATA)
        original = type(model)._generate
        model.invoke(["hi"])
        self.assertIsNot(type(model)._generate, original)
        from agent_trace.integrations.langchain import uninstrument_langchain
        uninstrument_langchain()
        self.assertIs(type(model)._generate, original)
        model.invoke(["hi"])
        self.assertEqual(len(self.events()), 2)

    def test_cost_counts_cache_tokens_once(self):
        base = self.instrument_langchain()
        model, _ = self.chat_model(base, OPENAI_METADATA)
        model.invoke(["hi"])
        usage = self.model_usage()["claude-sonnet-4-5"]
        self.assertEqual((usage.input_tokens, usage.cache_read_tokens, usage.cache_write_tokens),
                         (1000, 800, 100))
        self.assertProviderCost("claude-sonnet-4-5", 1000, 800, 100, _cost(
            uncached=100, read=800, write=100, output=50))


# ---------------------------------------------------------------------------
# One model call is recorded once
# ---------------------------------------------------------------------------

class TestNestedIntegrations(SDKCase):
    def test_litellm_calling_openai_records_one_call(self):
        openai_modules = stubs.openai_modules(chat=lambda kw: _completion(CHAT_USAGE))
        self.instrument("openai", openai_modules)
        completions = openai_modules["openai.resources.chat.completions"].Completions()

        def through_openai(kwargs):
            completions.create(model=kwargs["model"], messages=kwargs["messages"])
            return _litellm_response(LITELLM_USAGE)

        litellm = stubs.litellm_module(through_openai)
        self.instrument("litellm", litellm)
        litellm["litellm"].completion(model="claude-sonnet-4-5", messages=HI)
        request, response = self.pair()
        self.assertEqual(response.data["model"], "claude-sonnet-4-5")
        self.assertEqual([meta.agent_name for meta in TraceStore(self.trace_dir).list_sessions()],
                         ["litellm-test"])
        completions.create(model="gpt-4o", messages=HI)
        self.assertEqual(len(self.events()), 4)

    def test_langchain_calling_openai_records_one_call(self):
        openai_modules = stubs.openai_modules(chat=lambda kw: _completion(CHAT_USAGE))
        self.instrument("openai", openai_modules)
        client = openai_modules["openai.resources.chat.completions"].AsyncCompletions()
        modules, base = stubs.langchain_modules()
        self.instrument("langchain", modules)

        class ChatOpenAI(base):
            model_name = "gpt-4o"

            async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
                await client.create(model="gpt-4o", messages=HI)
                return stubs.chat_result(OPENAI_METADATA)

        _run(ChatOpenAI().ainvoke(["hi"]))
        request, response = self.pair()
        self.assertEqual(response.data["input_tokens"], 1000)
        self.assertEqual([meta.agent_name for meta in TraceStore(self.trace_dir).list_sessions()],
                         ["langchain-test"])


# ---------------------------------------------------------------------------
# Patching
# ---------------------------------------------------------------------------

class TestPatching(SDKCase):
    def test_instrumenting_twice_records_each_call_once(self):
        modules = stubs.openai_modules(chat=lambda kw: _completion(), responses=lambda kw: _response())
        integration = self.instrument("openai", modules)
        integration._PATCHED = False
        integration.instrument_openai(agent_name="openai-test")
        modules["openai.resources.chat.completions"].Completions().create(model="m", messages=HI)
        self.assertEqual(len(self.events()), 2)
        _run(modules["openai.resources.responses"].AsyncResponses().create(model="m", input="x"))
        self.assertEqual(len(self.events()), 4)

    def test_uninstrument_restores_every_method(self):
        modules = stubs.openai_modules(chat=lambda kw: _completion(), responses=lambda kw: _response())
        modules.update(stubs.anthropic_modules(lambda kw: _message(None)))
        targets = [
            (modules["openai.resources.chat.completions"], "Completions"),
            (modules["openai.resources.chat.completions"], "AsyncCompletions"),
            (modules["openai.resources.responses"], "Responses"),
            (modules["openai.resources.responses"], "AsyncResponses"),
            (modules["anthropic.resources.messages"], "Messages"),
            (modules["anthropic.resources.messages"], "AsyncMessages"),
        ]
        originals = [vars(getattr(module, name))["create"] for module, name in targets]
        openai = self.instrument("openai", modules)
        anthropic = self.instrument("anthropic", {})
        patched = [vars(getattr(module, name))["create"] for module, name in targets]
        for before, after in zip(originals, patched):
            self.assertIsNot(before, after)
        openai.uninstrument_openai()
        anthropic.uninstrument_anthropic()
        restored = [vars(getattr(module, name))["create"] for module, name in targets]
        self.assertEqual(restored, originals)

    def test_failure_to_record_never_reaches_the_caller(self):
        original = TraceStore.append_event
        completion = _completion(CHAT_USAGE)
        calls = {"ok": lambda kw: completion, "fail": _raise}
        modules = stubs.openai_modules(chat=lambda kw: calls[kw["model"]](kw))
        self.instrument("openai", modules)
        module = modules["openai.resources.chat.completions"]
        sync, client = module.Completions(), module.AsyncCompletions()

        for failing in ("every write", "response write"):
            def append(store, session_id, event):
                if failing == "every write" or event.event_type != EventType.LLM_REQUEST:
                    raise OSError("disk full")
                return original(store, session_id, event)

            with self.subTest(failing=failing), patch.object(TraceStore, "append_event", append):
                self.assertIs(sync.create(model="ok", messages=HI), completion)
                self.assertIs(_run(client.create(model="ok", messages=HI)), completion)
                raw = sync.with_raw_response.create(model="ok", messages=HI)
                self.assertIs(raw.parse(), completion)
                with self.assertRaises(RateLimitError):
                    sync.create(model="fail", messages=HI)
                with self.assertRaises(RateLimitError):
                    _run(client.create(model="fail", messages=HI))
        self.assertEqual([e.event_type for e in self.events()], [EventType.LLM_REQUEST] * 5)

    def test_raw_response_keeps_its_post_parser(self):
        from agent_trace.integrations._base import record_when_parsed

        recorded = []
        raw = stubs.RawResponse(ns(usage=None),
                                post_parser=lambda body: ns(usage=body.usage, parsed=True))
        self.assertTrue(record_when_parsed(raw, recorded.append))
        body = raw.parse()
        self.assertTrue(body.parsed)
        self.assertEqual(recorded, [body])
        raw.parse(to=lambda body: ns(usage=None))
        self.assertEqual(recorded, [body])

        failing = stubs.RawResponse(ns(usage=None))
        self.assertTrue(record_when_parsed(failing, lambda body: 1 / 0))
        self.assertIsNotNone(failing.parse())
        self.assertFalse(record_when_parsed(ns(usage=None), recorded.append))

    def test_missing_sdk_modules_are_skipped(self):
        modules = {"openai": stubs._module("openai", package=False)}
        self.instrument("openai", modules)
        self.assertEqual(self.events(), [])


if __name__ == "__main__":
    unittest.main()
