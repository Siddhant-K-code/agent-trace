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
import json
import os
import sys
import tempfile
import types
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
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


def _interrupt(kwargs):
    raise KeyboardInterrupt


def _exit(kwargs):
    raise SystemExit(3)


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

    def test_interrupted_sync_call_is_recorded_as_cancelled(self):
        sync, _ = self.instrument_chat(_interrupt)
        with self.assertRaises(KeyboardInterrupt):
            sync.create(model="gpt-4o", messages=HI)
        self.assertCancelled(*self.pair(), "gpt-4o")

    def test_async_call_ended_by_system_exit_is_recorded_as_cancelled(self):
        _, client = self.instrument_chat(_exit)
        with self.assertRaises(SystemExit):
            _run(client.create(model="gpt-4o", messages=HI))
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

    def test_interrupted_stream_is_recorded_as_cancelled(self):
        sync, _ = self.instrument_chat(
            lambda kw: stubs.Stream([_chunk()], error=KeyboardInterrupt()))
        stream = sync.create(model="gpt-4o", messages=HI, stream=True)
        with self.assertRaises(KeyboardInterrupt):
            list(stream)
        self.assertCancelled(*self.pair(), "gpt-4o")

    def test_closed_stream_records_only_the_request(self):
        sync, _ = self.instrument_chat(lambda kw: stubs.Stream([_chunk(), _chunk("stop")]))
        stream = sync.create(model="gpt-4o", messages=HI, stream=True)
        self.assertIsNotNone(next(stream))
        stream._iterator.close()
        (request,) = self.events()
        self.assertEqual(request.event_type, EventType.LLM_REQUEST)

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

    def test_async_stream_ended_by_system_exit_is_recorded_as_cancelled(self):
        _, client = self.instrument_chat(lambda kw: stubs.AsyncStream(
            [_chunk()], error=SystemExit(3)))

        async def read():
            stream = await client.create(model="gpt-4o", messages=HI, stream=True)
            await _drain(stream)

        with self.assertRaises(SystemExit):
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
# Agent frameworks: one session per run, tool results linked to their calls
# ---------------------------------------------------------------------------

E = EventType


class FrameworkCase(SDKCase):
    def sessions(self):
        """(meta, events) of each recorded session, oldest first."""
        store = TraceStore(self.trace_dir)
        metas = sorted(store.list_sessions(), key=lambda meta: meta.started_at)
        return [(meta, store.load_events(meta.session_id)) for meta in metas]

    def assertRun(self, meta, events, kinds, counts):
        """A finished run session with these events and counters."""
        self.assertEqual([e.event_type for e in events], kinds)
        self.assertIsNotNone(meta.ended_at)
        self.assertGreater(meta.total_duration_ms, 0)
        self.assertEqual((meta.tool_calls, meta.llm_requests, meta.errors), counts)

    def assertLinkedTo(self, events, response_index, request_index):
        self.assertEqual(events[response_index].parent_id, events[request_index].event_id)

    def assertNoMarkers(self):
        self.assertEqual(list(Path(self.trace_dir).glob(".active-session*")), [])

    def originals(self, owner, *names):
        return [vars(owner)[name] for name in names]


class TestCrewAISessions(FrameworkCase):
    def test_each_kickoff_is_its_own_finished_session(self):
        modules = stubs.crewai_modules()
        self.instrument("crewai", modules)
        crewai = modules["crewai"]
        agent = crewai.Agent()
        for description in ("a", "fail", "b"):
            crew = crewai.Crew([agent], [crewai.Task(description, agent)])
            if description == "fail":
                with self.assertRaises(RuntimeError):
                    crew.kickoff()
            else:
                crew.kickoff()

        ok, failed, last = self.sessions()
        done = [E.SESSION_START, E.TOOL_CALL, E.LLM_REQUEST, E.LLM_RESPONSE, E.TOOL_RESULT,
                E.SESSION_END]
        self.assertRun(*ok, done, (1, 1, 0))
        self.assertRun(*last, done, (1, 1, 0))
        self.assertRun(*failed, [E.SESSION_START, E.TOOL_CALL, E.LLM_REQUEST, E.ERROR, E.ERROR,
                                 E.ERROR, E.SESSION_END], (1, 1, 3))
        events = ok[1]
        self.assertLinkedTo(events, 3, 2)
        self.assertLinkedTo(events, 4, 1)
        events = failed[1]
        self.assertLinkedTo(events, 3, 2)
        self.assertLinkedTo(events, 4, 1)
        self.assertEqual(events[4].data["error_type"], "RuntimeError")
        self.assertEqual(events[5].parent_id, "")
        self.assertNoMarkers()


class TestOpenAIAgentsSessions(FrameworkCase):
    """Tools run through FunctionTool.on_invoke_tool inside a function span."""

    def setUp(self):
        super().setUp()
        self.tools = {}
        self.modules = stubs.agents_modules(self.run_agent)
        self.agents = self.modules["agents"]

    def run_agent(self, agent, prompt):
        if prompt == "fail":
            raise RuntimeError("run failed")
        return self.call_tool("search", prompt)

    def call_tool(self, name, query):
        functions = {"search": lambda query: "found " + query, "broken": self.broken}
        tool = self.agents.FunctionTool(name, functions[name])
        return self.agents.run_tool(tool, json.dumps({"query": query}))

    @staticmethod
    def broken(query):
        raise ValueError("bad input")

    def test_each_run_is_its_own_finished_session(self):
        self.instrument("openai_agents", self.modules)
        self.assertEqual(self.agents.Runner.run_sync("agent", "a"), "found a")
        with self.assertRaises(RuntimeError):
            self.agents.Runner.run_sync("agent", "fail")
        self.agents.Runner.run_sync("agent", "b")

        ok, failed, last = self.sessions()
        done = [E.SESSION_START, E.TOOL_CALL, E.TOOL_RESULT, E.SESSION_END]
        self.assertRun(*ok, done, (1, 0, 0))
        self.assertRun(*last, done, (1, 0, 0))
        self.assertRun(*failed, [E.SESSION_START, E.ERROR, E.SESSION_END], (0, 0, 1))
        self.assertLinkedTo(ok[1], 2, 1)
        self.assertEqual(ok[1][2].data["arguments"], {"query": "a"})
        self.assertNoMarkers()

    def test_tool_result_is_linked_to_its_call(self):
        self.instrument("openai_agents", self.modules)
        self.call_tool("search", "x")
        call, result = self.events()
        self.assertEqual((call.event_type, result.event_type), (E.TOOL_CALL, E.TOOL_RESULT))
        self.assertEqual(result.parent_id, call.event_id)
        self.assertEqual(call.data["tool_name"], "search")
        self.assertEqual(result.data["result"], "found x")

    def test_tool_error_is_linked_to_its_call(self):
        self.instrument("openai_agents", self.modules)
        self.call_tool("broken", "x")
        call, error = self.events()
        self.assertEqual((call.event_type, error.event_type), (E.TOOL_CALL, E.ERROR))
        self.assertEqual(error.parent_id, call.event_id)
        self.assertIn("bad input", error.data["error"])

    def test_tool_call_is_recorded_while_the_tool_runs_and_kept_if_abandoned(self):
        integration = self.instrument("openai_agents", self.modules)
        processor, = self.modules["agents.tracing"].processors
        span = types.SimpleNamespace(
            span_id="span_hung", error=None,
            span_data=types.SimpleNamespace(type="function", name="search",
                                            input=None, output=None))
        processor.on_span_start(span)
        call, = self.events()
        self.assertEqual(call.event_type, E.TOOL_CALL)
        self.assertEqual(call.data["arguments"], {})
        integration.uninstrument_openai_agents()
        span.span_data.input = json.dumps({"query": "x"})
        processor.on_span_end(span)
        self.assertEqual([e.event_id for e in self.events()], [call.event_id])

    def test_uninstrument_restores_and_reinstrumenting_records_once(self):
        originals = self.originals(self.agents.Runner, "run_sync")
        integration = self.instrument("openai_agents", self.modules)
        integration.uninstrument_openai_agents()
        self.assertEqual(self.originals(self.agents.Runner, "run_sync"), originals)
        self.agents.Runner.run_sync("agent", "a")
        self.assertEqual(self.events(), [])
        integration.instrument_openai_agents(agent_name="openai_agents-test")
        integration._PATCHED = False
        integration.instrument_openai_agents(agent_name="openai_agents-test")
        self.assertEqual(len(self.modules["agents.tracing"].processors), 1)
        self.agents.Runner.run_sync("agent", "b")
        (meta, events), = self.sessions()
        self.assertRun(meta, events, [E.SESSION_START, E.TOOL_CALL, E.TOOL_RESULT,
                                      E.SESSION_END], (1, 0, 0))


class TestStrandsSessions(FrameworkCase):
    """Tools run through the executor's tool hooks around tool.stream()."""

    def setUp(self):
        super().setUp()
        self.modules = stubs.strands_modules()
        self.strands = self.modules["strands"]

    def tool(self, name, function):
        def tool_func(tool_use, **invocation_state):
            value = function(tool_use["input"]["prompt"])
            if isinstance(value, dict):
                return value
            return {"toolUseId": tool_use["toolUseId"], "status": "success",
                    "content": [{"text": value}]}

        return self.modules["strands.tools"].PythonAgentTool(name, {"name": name}, tool_func)

    def search_agent(self):
        return self.strands.Agent([self.tool("search", lambda prompt: "found " + prompt)])

    def test_each_call_is_its_own_finished_session(self):
        self.instrument("strands", self.modules)
        agent = self.search_agent()
        agent("a")
        with self.assertRaises(RuntimeError):
            agent("crash")
        _run(agent.invoke_async("b"))

        ok, failed, last = self.sessions()
        done = [E.SESSION_START, E.TOOL_CALL, E.TOOL_RESULT, E.SESSION_END]
        self.assertRun(*ok, done, (1, 0, 0))
        self.assertRun(*last, done, (1, 0, 0))
        self.assertRun(*failed, [E.SESSION_START, E.ERROR, E.SESSION_END], (0, 0, 1))
        self.assertLinkedTo(ok[1], 2, 1)
        self.assertEqual(ok[1][1].data["arguments"], {"prompt": "a"})
        self.assertEqual(ok[1][2].data["result"], "found a")
        self.assertNoMarkers()

    def test_tool_errors_are_linked_to_their_calls(self):
        def broken(prompt):
            raise ValueError("bad input")

        def refused(prompt):
            return {"toolUseId": "tooluse_1", "status": "error",
                    "content": [{"text": "not allowed"}]}

        self.instrument("strands", self.modules)
        self.strands.Agent([self.tool("broken", broken), self.tool("refused", refused)])("x")
        (meta, events), = self.sessions()
        self.assertRun(meta, events, [E.SESSION_START, E.TOOL_CALL, E.ERROR, E.TOOL_CALL,
                                      E.ERROR, E.SESSION_END], (2, 0, 2))
        self.assertLinkedTo(events, 2, 1)
        self.assertLinkedTo(events, 4, 3)
        self.assertEqual((events[2].data["error"], events[2].data["error_type"]),
                         ("bad input", "ValueError"))
        self.assertEqual(events[4].data["error"], "not allowed")

    def test_uninstrument_restores_and_reinstrumenting_records_once(self):
        agent_class = self.strands.Agent
        names = ("__init__", "__call__", "invoke_async")
        originals = self.originals(agent_class, *names)
        made_before = self.search_agent()
        integration = self.instrument("strands", self.modules)
        made_before("a")
        integration.uninstrument_strands()
        self.assertEqual(self.originals(agent_class, *names), originals)
        made_before("b")
        self.assertEqual(len(self.sessions()), 1)
        integration.instrument_strands(agent_name="strands-test")
        integration._PATCHED = False
        integration.instrument_strands(agent_name="strands-test")
        made_after = self.search_agent()
        made_before("c")
        made_after("d")
        self.assertEqual(len(made_before.hooks.callbacks), 2)
        self.assertEqual(len(made_after.hooks.callbacks), 2)
        done = [E.SESSION_START, E.TOOL_CALL, E.TOOL_RESULT, E.SESSION_END]
        for meta, events in self.sessions():
            self.assertRun(meta, events, done, (1, 0, 0))
        self.assertEqual(len(self.sessions()), 3)


class TestLangChainTools(FrameworkCase):
    def instrument_tools(self):
        modules, base, graph = stubs.langchain_tool_modules()
        self.instrument("langchain", modules)
        return base, graph

    def test_structured_tool_call_is_linked_to_its_result(self):
        base, _ = self.instrument_tools()

        class StructuredTool(base):
            name = "add"

            def _run(self, a, b):
                return a + b

        self.assertEqual(StructuredTool().invoke({"a": 1, "b": 2}), 3)
        self.assertEqual(_run(StructuredTool().ainvoke({"a": 2, "b": 2})), 4)
        call, result, async_call, async_result = self.events()
        self.assertEqual((call.event_type, result.event_type), (E.TOOL_CALL, E.TOOL_RESULT))
        self.assertEqual(call.data["arguments"], {"a": 1, "b": 2})
        self.assertEqual(result.parent_id, call.event_id)
        self.assertEqual(async_result.parent_id, async_call.event_id)
        self.assertEqual(async_result.data["result"], "4")

    def test_tool_error_is_linked_to_its_call(self):
        base, _ = self.instrument_tools()

        class FailingTool(base):
            name = "fail"

            def _run(self, query):
                raise ValueError("bad input")

        with self.assertRaises(ValueError):
            FailingTool().invoke("x")
        call, error = self.events()
        self.assertEqual(call.data["arguments"], {"input": "x"})
        self.assertEqual(error.event_type, E.ERROR)
        self.assertEqual(error.parent_id, call.event_id)
        self.assertEqual(error.data["error_type"], "ValueError")

    def test_uninstrument_restores_run_and_arun(self):
        modules, base, _ = stubs.langchain_tool_modules()
        originals = self.originals(base, "run", "arun")
        integration = self.instrument("langchain", modules)
        self.assertNotEqual(self.originals(base, "run", "arun"), originals)
        integration.uninstrument_langchain()
        self.assertEqual(self.originals(base, "run", "arun"), originals)

    def langgraph(self):
        """Stub LangGraph modules, not instrumented yet, and their StateGraph."""
        modules, _, state_graph = stubs.langchain_tool_modules()
        return modules, state_graph

    def assertNodeRuns(self, events, *names):
        """Each node run recorded as decision, tool_call and a linked tool_result."""
        self.assertEqual([e.event_type for e in events],
                         [E.DECISION, E.TOOL_CALL, E.TOOL_RESULT] * len(names))
        for index, name in enumerate(names):
            decision, call, result = events[3 * index:3 * index + 3]
            tool_name = f"langgraph.node.{name}"
            self.assertEqual(decision.data, {"choice": name, "reason": f"LangGraph node: {name}"})
            self.assertEqual(call.data, {"tool_name": tool_name})
            self.assertEqual(result.parent_id, call.event_id)
            self.assertEqual(sorted(result.data), ["duration_ms", "tool_name"])
            self.assertEqual(result.data["tool_name"], tool_name)

    def test_langgraph_graph_runs_as_untraced_and_records_each_node(self):
        modules, state_graph = self.langgraph()
        nodes = {"plan": lambda state: state + 1, "act": lambda state: state * 10}
        untraced = state_graph(nodes).compile()
        self.instrument("langchain", modules)
        traced = state_graph(nodes).compile()
        self.assertEqual((untraced.invoke(1), traced.invoke(1)), (20, 20))
        self.assertNodeRuns(self.events(), "plan", "act")

    def test_langgraph_ainvoke_runs_an_async_only_node(self):
        modules, state_graph = self.langgraph()

        async def work(state):
            return state + 1

        untraced = state_graph({"work": work}).compile()
        self.instrument("langchain", modules)
        traced = state_graph({"work": work}).compile()
        self.assertEqual((_run(untraced.ainvoke(1)), _run(traced.ainvoke(1))), (2, 2))
        self.assertNodeRuns(self.events(), "work")

    def test_langgraph_node_error_is_linked_and_raised_unchanged(self):
        _, state_graph = self.instrument_tools()
        failure = ValueError("node failed")

        def boom(state):
            raise failure

        async def aboom(state):
            raise failure

        runs = {"invoke": lambda: state_graph({"boom": boom}).compile().invoke(1),
                "ainvoke": lambda: _run(state_graph({"boom": aboom}).compile().ainvoke(1))}
        for method, run in runs.items():
            with self.subTest(method=method):
                before = len(self.events())
                with self.assertRaises(ValueError) as caught:
                    run()
                self.assertIs(caught.exception, failure)
                decision, call, error = self.events()[before:]
                self.assertEqual((decision.event_type, error.event_type), (E.DECISION, E.ERROR))
                self.assertEqual(error.parent_id, call.event_id)
                self.assertEqual(error.data, {"tool_name": "langgraph.node.boom",
                                              "error": "node failed", "error_type": "ValueError"})

    def test_langgraph_interrupt_is_a_control_flow_result_not_an_error(self):
        modules, state_graph = self.langgraph()
        interrupt = modules["langgraph.errors"].GraphInterrupt("ask")

        def ask(state):
            raise interrupt

        self.instrument("langchain", modules)
        with self.assertRaises(type(interrupt)) as caught:
            state_graph({"ask": ask}).compile().invoke(1)
        self.assertIs(caught.exception, interrupt)
        decision, call, result = self.events()
        self.assertEqual(result.event_type, E.TOOL_RESULT)
        self.assertEqual(result.parent_id, call.event_id)
        self.assertEqual(sorted(result.data), ["control_flow", "duration_ms", "tool_name"])
        self.assertEqual(result.data["control_flow"], "GraphInterrupt")

    def test_langgraph_node_of_another_shape_is_left_untraced(self):
        modules, state_graph = self.langgraph()
        pregel = modules["langgraph.pregel"]
        not_runnable = pregel.PregelNode(["plain"], types.SimpleNamespace(
            invoke=lambda state, config: state + 1))
        no_copy = types.SimpleNamespace(triggers=["no_copy"],
                                        bound=pregel.RunnableCallable(lambda state: state * 10))
        self.instrument("langchain", modules)
        graph = state_graph({"plain": not_runnable, "no_copy": no_copy,
                             "work": lambda state: state - 1}).compile()
        self.assertIs(graph.nodes["plain"], not_runnable)
        self.assertIs(graph.nodes["no_copy"], no_copy)
        self.assertEqual(graph.invoke(1), 19)
        self.assertNodeRuns(self.events(), "work")

    def test_langgraph_no_node_is_wrapped_when_one_fails_to_copy(self):
        modules, state_graph = self.langgraph()
        pregel = modules["langgraph.pregel"]

        class Uncopyable(pregel.PregelNode):
            def copy(self, update):
                raise RuntimeError("cannot copy")

        work = pregel.PregelNode(["work"], pregel.RunnableCallable(lambda state: state + 1))
        stuck = Uncopyable(["stuck"], pregel.RunnableCallable(lambda state: state * 10))
        self.instrument("langchain", modules)
        graph = state_graph({"work": work, "stuck": stuck}).compile()
        self.assertIs(graph.nodes["work"], work)
        self.assertIs(graph.nodes["stuck"], stuck)
        self.assertEqual(graph.invoke(1), 20)
        self.assertEqual(self.events(), [])

    def test_langgraph_subgraph_node_keeps_its_subgraphs(self):
        _, state_graph = self.instrument_tools()
        inner = state_graph({"s1": lambda state: state + 1}).compile()
        outer = state_graph({"sub": inner}).compile()
        self.assertEqual(outer.nodes["sub"].subgraphs, [inner])
        self.assertEqual(outer.invoke(1), 2)
        events = self.events()
        self.assertNodeRuns(events[2:5], "s1")
        self.assertNodeRuns(events[:2] + events[5:], "sub")

    def test_langgraph_only_start_and_end_nodes_are_skipped(self):
        modules, state_graph = self.langgraph()
        pregel = modules["langgraph.pregel"]
        end = pregel.PregelNode([], pregel.RunnableCallable(lambda state: state))
        self.instrument("langchain", modules)
        graph = state_graph({"work": lambda state: state + 1, "__end__": end,
                             "__error_handler__work": lambda state: state * 10}).compile()
        self.assertIs(type(graph.nodes["__start__"].bound), pregel.RunnableCallable)
        self.assertIs(graph.nodes["__end__"], end)
        self.assertEqual(graph.invoke(1), 20)
        self.assertNodeRuns(self.events(), "work", "__error_handler__work")

    def test_langgraph_failure_to_record_never_reaches_the_graph(self):
        from agent_trace.integrations import _base

        modules, state_graph = self.langgraph()
        real_emit, failing = _base.emit, set()

        def emit(event_type, *args, **kwargs):
            if event_type in failing:
                raise OSError("disk full")
            return real_emit(event_type, *args, **kwargs)

        failure = ValueError("node failed")

        def boom(state):
            raise failure

        self.instrument("langchain", modules)
        with patch.object(_base, "emit", emit):
            graph = state_graph({"work": lambda state: state + 1}).compile()
            failing_graph = state_graph({"boom": boom}).compile()
        for kinds in ({E.DECISION}, {E.TOOL_RESULT, E.ERROR}):
            with self.subTest(failing=kinds):
                failing.clear()
                failing.update(kinds)
                self.assertEqual(graph.invoke(1), 2)
                with self.assertRaises(ValueError) as caught:
                    failing_graph.invoke(1)
                self.assertIs(caught.exception, failure)

    def test_langgraph_sequence_node_is_left_untraced(self):
        modules, state_graph = self.langgraph()
        runnables, pregel = modules["langchain_core.runnables"], modules["langgraph.pregel"]
        sequence = runnables.RunnableSequence(pregel.RunnableCallable(lambda state: state + 1))
        self.instrument("langchain", modules)
        graph = state_graph({"chain": sequence, "work": lambda state: state * 10}).compile()
        self.assertIs(graph.nodes["chain"].bound, sequence)
        self.assertEqual(graph.invoke(1), 20)
        self.assertNodeRuns(self.events(), "work")

    def test_langgraph_astream_passes_the_node_stream_through(self):
        modules, state_graph = self.langgraph()
        failure = ValueError("stream failed")

        class Streamer(modules["langchain_core.runnables"].Runnable):
            def __init__(self, fail=False):
                self.fail = fail

            def invoke(self, input, config=None, **kwargs):
                return "ab"

            async def astream(self, input, config=None, **kwargs):
                yield "a"
                if self.fail:
                    raise failure
                yield "b"

        async def chunks(graph):
            return [chunk async for chunk in graph.astream(1)]

        untraced = state_graph({"stream": Streamer()}).compile()
        self.instrument("langchain", modules)
        traced = state_graph({"stream": Streamer()}).compile()
        self.assertEqual((_run(chunks(untraced)), _run(chunks(traced))),
                         ([1, "a", "b"], [1, "a", "b"]))
        with self.assertRaises(ValueError) as caught:
            _run(chunks(state_graph({"stream": Streamer(fail=True)}).compile()))
        self.assertIs(caught.exception, failure)
        events = self.events()
        self.assertNodeRuns(events[:3], "stream")
        self.assertEqual([e.event_type for e in events[3:]], [E.DECISION, E.TOOL_CALL, E.ERROR])
        self.assertLinkedTo(events, 5, 4)

    def test_langgraph_recorder_reads_attributes_of_the_node_runnable(self):
        modules, state_graph = self.langgraph()

        class ToolNode(modules["langchain_core.runnables"].Runnable):
            tools_by_name = {"search": object()}

            def invoke(self, input, config=None, **kwargs):
                return input

        self.instrument("langchain", modules)
        graph = state_graph({"tools": ToolNode()}).compile()
        self.assertIs(graph.nodes["tools"].bound.tools_by_name, ToolNode.tools_by_name)

    def test_langgraph_node_keeps_its_cache_identifier(self):
        modules, state_graph = self.langgraph()
        identifier = modules["langgraph.pregel._call"].identifier

        def work(state):
            return state + 1

        untraced = state_graph({"work": work}).compile()
        self.instrument("langchain", modules)
        traced = state_graph({"work": work}).compile()
        self.assertEqual(identifier(traced.nodes["work"].bound),
                         identifier(untraced.nodes["work"].bound))
        self.assertIn("<locals>.work", identifier(traced.nodes["work"].bound))


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
