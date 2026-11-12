"""Stand-ins for the model SDKs, laid out like the real packages.

The integrations patch the SDK resource classes, so these stubs provide the
same modules and classes, with a ``create`` method that hands its keyword
arguments to a test-supplied function, and a ``with_raw_response.create``
that returns the result in a raw response. The streams store their items in a
generator in ``_iterator``, as the OpenAI and Anthropic SDK streams do.
"""

from __future__ import annotations

import functools
import inspect
import types
from copy import copy
from functools import cached_property
from typing import Any, Callable


def _module(name: str, package: bool = True, **attrs: Any) -> types.ModuleType:
    module = types.ModuleType(name)
    if package:
        module.__path__ = []
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _required_args(function: Callable[..., Any]) -> Callable[..., Any]:
    """Like the OpenAI and Anthropic SDK decorator, a plain function wrapper.

    It hides that the async ``create`` methods are coroutine functions.
    """
    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return function(*args, **kwargs)

    return wrapper


class NotGiven:
    """The SDKs' placeholder for an argument the caller left out."""

    def __bool__(self) -> bool:
        return False


class Omit(NotGiven):
    """The newer SDK placeholder, which the ``.stream()`` helpers pass on."""


class Stream:
    """An OpenAI SDK ``Stream``: ``_iterator`` is set in ``__init__``."""

    def __init__(self, items: list[Any], error: Exception | None = None) -> None:
        self._items = items
        self._error = error
        self.closed = False
        self._iterator = self.__stream__()

    def __stream__(self) -> Any:
        yield from self._items
        if self._error is not None:
            raise self._error

    def __next__(self) -> Any:
        return self._iterator.__next__()

    def __iter__(self) -> Any:
        for item in self._iterator:
            yield item

    def close(self) -> None:
        self.closed = True


class AsyncStream:
    """An OpenAI SDK ``AsyncStream``."""

    def __init__(self, items: list[Any], error: BaseException | None = None,
                 wait: Any = None) -> None:
        self._items = items
        self._error = error
        self._wait = wait
        self._iterator = self.__stream__()

    async def __stream__(self) -> Any:
        for item in self._items:
            yield item
        if self._wait is not None:
            await self._wait
        if self._error is not None:
            raise self._error

    async def __anext__(self) -> Any:
        return await self._iterator.__anext__()

    async def __aiter__(self) -> Any:
        async for item in self._iterator:
            yield item


class CachedStream(Stream):
    """An Anthropic SDK ``Stream``: ``_iterator`` is made on first use."""

    def __init__(self, items: list[Any], error: Exception | None = None) -> None:
        self._items = items
        self._error = error
        self.closed = False

    @cached_property
    def _iterator(self) -> Any:
        return copy(self).__stream__()


class AsyncCachedStream(AsyncStream):
    """An Anthropic SDK ``AsyncStream``."""

    def __init__(self, items: list[Any], error: BaseException | None = None,
                 wait: Any = None) -> None:
        self._items = items
        self._error = error
        self._wait = wait

    @cached_property
    def _iterator(self) -> Any:
        return copy(self).__stream__()


class Opaque:
    """A stream the integrations cannot see inside: it has no ``_iterator``."""

    def __init__(self, items: list[Any]) -> None:
        self._items = items

    def __iter__(self) -> Any:
        return iter(self._items)


RAW_RESPONSE_HEADER = "X-Stainless-Raw-Response"


class RawResponse:
    """A ``with_raw_response`` result, like the SDKs' ``APIResponse``.

    ``parse()`` passes the body through the request's ``post_parser``
    option and caches the result by the type asked for.
    """

    def __init__(self, body: Any, post_parser: Any = None) -> None:
        self.http_response = types.SimpleNamespace(status_code=200)
        self._body = body
        self._options = types.SimpleNamespace(
            post_parser=post_parser if post_parser is not None else NotGiven())
        self._parsed: dict[Any, Any] = {}

    def parse(self, *, to: Any = None) -> Any:
        if to in self._parsed:
            return self._parsed[to]
        parsed = self._body if to is None else to(self._body)
        if not isinstance(self._options.post_parser, NotGiven):
            parsed = self._options.post_parser(parsed)
        self._parsed[to] = parsed
        return parsed


class AsyncRawResponse(RawResponse):
    """An Anthropic SDK ``AsyncAPIResponse``, whose ``parse()`` is async."""

    async def parse(self, *, to: Any = None) -> Any:  # type: ignore[override]
        return RawResponse.parse(self, to=to)


class _WithRawResponse:
    """A resource's ``with_raw_response``: ``create`` sets the raw header."""

    def __init__(self, resource: Any) -> None:
        self._resource = resource

    def create(self, **kwargs: Any) -> Any:
        headers = {**(kwargs.get("extra_headers") or {}), RAW_RESPONSE_HEADER: "raw"}
        return self._resource.create(**{**kwargs, "extra_headers": headers})


def _wants_raw(kwargs: dict) -> bool:
    headers = kwargs.pop("extra_headers", None) or {}
    return headers.get(RAW_RESPONSE_HEADER) == "raw"


def _resource_classes(name: str, handler: Callable[[dict], Any]) -> tuple[type, type]:
    """A sync and an async resource class whose ``create`` calls *handler*."""

    def create(self: Any, **kwargs: Any) -> Any:
        raw = _wants_raw(kwargs)
        result = handler(kwargs)
        return RawResponse(result) if raw else result

    async def acreate(self: Any, **kwargs: Any) -> Any:
        raw = _wants_raw(kwargs)
        result = handler(kwargs)
        if inspect.isawaitable(result):
            result = await result
        return AsyncRawResponse(result) if raw else result

    with_raw_response = property(_WithRawResponse)
    sync = type(name, (), {"create": _required_args(create),
                           "with_raw_response": with_raw_response})
    async_ = type("Async" + name, (), {"create": _required_args(acreate),
                                       "with_raw_response": with_raw_response})
    return sync, async_


def openai_modules(chat: Callable[[dict], Any] | None = None,
                   responses: Callable[[dict], Any] | None = None) -> dict[str, Any]:
    """``sys.modules`` entries for an OpenAI SDK whose API calls go to the handlers."""
    completions, async_completions = _resource_classes(
        "Completions", chat or _unexpected)
    responses_, async_responses = _resource_classes(
        "Responses", responses or _unexpected)
    completions_module = _module(
        "openai.resources.chat.completions",
        Completions=completions, AsyncCompletions=async_completions)
    chat_module = _module("openai.resources.chat", completions=completions_module)
    responses_module = _module(
        "openai.resources.responses",
        Responses=responses_, AsyncResponses=async_responses)
    resources = _module("openai.resources", chat=chat_module, responses=responses_module)
    root = _module("openai", resources=resources)
    return {
        "openai": root,
        "openai.resources": resources,
        "openai.resources.chat": chat_module,
        "openai.resources.chat.completions": completions_module,
        "openai.resources.responses": responses_module,
    }


def anthropic_modules(messages: Callable[[dict], Any]) -> dict[str, Any]:
    """``sys.modules`` entries for an Anthropic SDK whose calls go to *messages*."""
    sync, async_ = _resource_classes("Messages", messages)
    messages_module = _module("anthropic.resources.messages", Messages=sync, AsyncMessages=async_)
    resources = _module("anthropic.resources", messages=messages_module)
    root = _module("anthropic", resources=resources)
    return {
        "anthropic": root,
        "anthropic.resources": resources,
        "anthropic.resources.messages": messages_module,
    }


def litellm_module(handler: Callable[[dict], Any]) -> dict[str, Any]:
    """``sys.modules`` entry for a LiteLLM whose completions go to *handler*.

    Its functions take ``model`` and ``messages`` positionally too.
    """

    def completion(model: str = "", messages: Any = None, **kwargs: Any) -> Any:
        return handler({"model": model, "messages": messages, **kwargs})

    async def acompletion(model: str = "", messages: Any = None, **kwargs: Any) -> Any:
        return handler({"model": model, "messages": messages, **kwargs})

    return {"litellm": _module("litellm", completion=completion, acompletion=acompletion)}


def langchain_modules() -> tuple[dict[str, Any], type]:
    """``sys.modules`` entries for a LangChain Core with a ``BaseChatModel``.

    Like LangChain's, its ``_generate_with_cache`` passes ``run_manager``
    to ``_generate`` only when ``_generate``'s signature takes it, and its
    default ``_agenerate`` runs ``_generate`` in a thread with the caller's
    context.
    """
    import asyncio
    import contextvars

    class BaseChatModel:
        model_name = "unknown"

        def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None,
                      **kwargs: Any) -> Any:
            raise NotImplementedError

        async def _agenerate(self, messages: Any, stop: Any = None,
                             run_manager: Any = None, **kwargs: Any) -> Any:
            context = contextvars.copy_context()
            call = functools.partial(self._generate, messages, stop=stop,
                                     run_manager=run_manager, **kwargs)
            return await asyncio.get_running_loop().run_in_executor(
                None, context.run, call)

        def _generate_with_cache(self, messages: Any, stop: Any = None,
                                 run_manager: Any = None, **kwargs: Any) -> Any:
            if "run_manager" in inspect.signature(self._generate).parameters:
                return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            return self._generate(messages, stop=stop, **kwargs)

        async def _agenerate_with_cache(self, messages: Any, stop: Any = None,
                                        run_manager: Any = None, **kwargs: Any) -> Any:
            if "run_manager" in inspect.signature(self._agenerate).parameters:
                return await self._agenerate(messages, stop=stop, run_manager=run_manager,
                                             **kwargs)
            return await self._agenerate(messages, stop=stop, **kwargs)

        def invoke(self, messages: Any, **kwargs: Any) -> Any:
            return self._generate_with_cache(messages, run_manager="manager", **kwargs)

        async def ainvoke(self, messages: Any, **kwargs: Any) -> Any:
            return await self._agenerate_with_cache(messages, run_manager="manager", **kwargs)

    models = _module("langchain_core.language_models", BaseChatModel=BaseChatModel)
    root = _module("langchain_core", language_models=models)
    return {
        "langchain_core": root,
        "langchain_core.language_models": models,
        "langchain_core.tools": None,
        "langgraph": None,
        "langgraph.graph": None,
    }, BaseChatModel


def chat_result(usage_metadata: Any = None) -> Any:
    """A LangChain ``ChatResult`` whose message reports *usage_metadata*."""
    message = types.SimpleNamespace(content="hi", usage_metadata=usage_metadata)
    return types.SimpleNamespace(generations=[types.SimpleNamespace(message=message)])


def _unexpected(kwargs: dict) -> Any:
    raise AssertionError(f"unexpected call: {kwargs}")


def ns(**kwargs: Any) -> types.SimpleNamespace:
    """A plain object with the given attributes, like an SDK response model."""
    return types.SimpleNamespace(**kwargs)
