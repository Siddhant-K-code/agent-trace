"""Stand-ins for the model SDKs, laid out like the real packages.

The integrations patch the SDK resource classes, so these stubs provide the
same modules and classes, with a ``create`` method that hands its keyword
arguments to a test-supplied function, and a ``with_raw_response.create``
that returns the result in a raw response. The streams store their items in a
generator in ``_iterator``, as the OpenAI and Anthropic SDK streams do.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import functools
import inspect
import itertools
import json
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

    def __init__(self, items: list[Any], error: BaseException | None = None) -> None:
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

    def __init__(self, items: list[Any], error: BaseException | None = None) -> None:
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


# ---------------------------------------------------------------------------
# Agent frameworks
# ---------------------------------------------------------------------------

def crewai_modules() -> dict[str, Any]:
    """``sys.modules`` entries for a CrewAI whose crews run their tasks.

    ``Crew.kickoff`` runs each task's ``execute_sync``, which has its agent
    ``execute_task``. A task whose description is ``"fail"`` raises.
    """
    class Agent:
        role = "researcher"

        def execute_task(self, task: Any, *args: Any, **kwargs: Any) -> Any:
            if task.description == "fail":
                raise RuntimeError("task failed")
            return f"done: {task.description}"

    class Task:
        def __init__(self, description: str, agent: Any) -> None:
            self.description = description
            self.agent = agent

        def execute_sync(self, *args: Any, **kwargs: Any) -> Any:
            return self.agent.execute_task(self)

    class Crew:
        name = "crew"

        def __init__(self, agents: list, tasks: list) -> None:
            self.agents = agents
            self.tasks = tasks

        def kickoff(self, *args: Any, **kwargs: Any) -> Any:
            return [task.execute_sync() for task in self.tasks]

    return {"crewai": _module("crewai", Agent=Agent, Task=Task, Crew=Crew)}


def agents_modules(run: Callable[..., Any]) -> dict[str, Any]:
    """``sys.modules`` entries for an OpenAI Agents SDK.

    ``Runner.run_sync`` is a classmethod that returns ``run(agent, input)``.
    ``run_tool(tool, arguments)`` runs a ``FunctionTool`` as the SDK does:
    inside a function span that each processor added with
    ``agents.tracing.add_trace_processor`` sees, through the tool's async
    ``on_invoke_tool`` instance attribute. The span's input is set after it
    starts. A tool exception becomes the output and the span's error, as with
    the SDK's default failure handler.
    """
    processors: list = []
    span_ids = itertools.count(1)

    class TracingProcessor:
        pass

    def add_trace_processor(processor: Any) -> None:
        processors.append(processor)

    class FunctionSpanData:
        type = "function"

        def __init__(self, name: str) -> None:
            self.name = name
            self.input: Any = None
            self.output: Any = None

    class Span:
        def __init__(self, span_data: Any) -> None:
            self.span_id = f"span_{next(span_ids)}"
            self.span_data = span_data
            self.error: Any = None

    class FunctionTool:
        def __init__(self, name: str, function: Callable[..., Any]) -> None:
            self.name = name

            async def on_invoke_tool(ctx: Any, input: str) -> Any:
                return function(**json.loads(input))

            self.on_invoke_tool = on_invoke_tool

    def run_tool(tool: Any, arguments: str) -> Any:
        span = Span(FunctionSpanData(tool.name))
        for processor in processors:
            processor.on_span_start(span)
        span.span_data.input = arguments
        try:
            output = asyncio.run(tool.on_invoke_tool(None, arguments))
        except Exception as exc:
            output = f"An error occurred while running the tool. Error: {exc}"
            span.error = {"message": "Error running tool (non-fatal)",
                          "data": {"tool_name": tool.name, "error": str(exc)}}
        span.span_data.output = output
        for processor in processors:
            processor.on_span_end(span)
        return output

    class Runner:
        @classmethod
        def run_sync(cls, starting_agent: Any, input: Any, **kwargs: Any) -> Any:
            return run(starting_agent, input)

    tracing = _module("agents.tracing", TracingProcessor=TracingProcessor,
                      add_trace_processor=add_trace_processor, processors=processors)
    return {
        "agents": _module("agents", Runner=Runner, FunctionTool=FunctionTool,
                          run_tool=run_tool, tracing=tracing),
        "agents.tracing": tracing,
    }


def strands_modules() -> dict[str, Any]:
    """``sys.modules`` entries for AWS Strands Agents.

    Tools run as in the SDK: ``Agent.invoke_async(prompt)`` hands each tool
    to ``SequentialToolExecutor._stream``, which invokes the agent's
    ``hooks`` with a ``BeforeToolCallEvent``, iterates the async generator
    ``tool.stream(tool_use, invocation_state)`` and then invokes them with an
    ``AfterToolCallEvent``. A tool exception becomes an error result and the
    after event's ``exception``. ``PythonAgentTool`` calls
    ``tool_func(tool_use, **invocation_state)``, which returns a ToolResult
    dict. The prompt ``"crash"`` makes the agent raise. Calling the agent runs
    ``invoke_async`` on another thread in a copy of the context, as the SDK
    does.
    """
    class HookRegistry:
        def __init__(self) -> None:
            self.callbacks: list = []

        def add_callback(self, event_type: type, callback: Callable[..., Any]) -> None:
            self.callbacks.append((event_type, callback))

        def invoke_callbacks(self, event: Any) -> None:
            for event_type, callback in self.callbacks:
                if isinstance(event, event_type):
                    callback(event)

    class BeforeToolCallEvent:
        def __init__(self, agent: Any, selected_tool: Any, tool_use: dict,
                     invocation_state: dict) -> None:
            self.agent = agent
            self.selected_tool = selected_tool
            self.tool_use = tool_use
            self.invocation_state = invocation_state

    class AfterToolCallEvent:
        def __init__(self, agent: Any, selected_tool: Any, tool_use: dict,
                     invocation_state: dict, result: dict, exception: Any = None) -> None:
            self.agent = agent
            self.selected_tool = selected_tool
            self.tool_use = tool_use
            self.invocation_state = invocation_state
            self.result = result
            self.exception = exception

    class AgentTool:
        async def stream(self, tool_use: dict, invocation_state: dict, **kwargs: Any) -> Any:
            raise NotImplementedError
            yield

    class PythonAgentTool(AgentTool):
        def __init__(self, tool_name: str, tool_spec: dict,
                     tool_func: Callable[..., Any]) -> None:
            self.tool_name = tool_name
            self.tool_spec = tool_spec
            self._tool_func = tool_func

        async def stream(self, tool_use: dict, invocation_state: dict, **kwargs: Any) -> Any:
            yield self._tool_func(tool_use, **invocation_state)

    class SequentialToolExecutor:
        async def _stream(self, agent: Any, tool_use: dict, tool_results: list,
                          invocation_state: dict) -> Any:
            tool = agent.tool_registry[tool_use["name"]]
            agent.hooks.invoke_callbacks(
                BeforeToolCallEvent(agent, tool, tool_use, invocation_state))
            exception = None
            result: Any = None
            try:
                async for event in tool.stream(tool_use, invocation_state):
                    result = event
                    yield event
            except Exception as exc:
                exception = exc
                result = {"toolUseId": tool_use["toolUseId"], "status": "error",
                          "content": [{"text": f"Error: {exc}"}]}
            agent.hooks.invoke_callbacks(AfterToolCallEvent(
                agent, tool, tool_use, invocation_state, result, exception))
            tool_results.append(result)

    class Agent:
        name = "strands-agent"

        def __init__(self, tools: Any = None) -> None:
            self.tool_registry = {tool.tool_name: tool for tool in tools or []}
            self.tool_executor = SequentialToolExecutor()
            self.hooks = HookRegistry()

        async def invoke_async(self, prompt: str) -> Any:
            if prompt == "crash":
                raise RuntimeError("model failed")
            tool_results: list = []
            for index, name in enumerate(self.tool_registry):
                tool_use = {"toolUseId": f"tooluse_{index}", "name": name,
                            "input": {"prompt": prompt}}
                async for _ in self.tool_executor._stream(self, tool_use, tool_results, {}):
                    pass
            return tool_results

        def __call__(self, prompt: str) -> Any:
            context = contextvars.copy_context()
            with concurrent.futures.ThreadPoolExecutor(1) as executor:
                return executor.submit(context.run, asyncio.run,
                                       self.invoke_async(prompt)).result()

    hooks = _module("strands.hooks", BeforeToolCallEvent=BeforeToolCallEvent,
                    AfterToolCallEvent=AfterToolCallEvent, HookRegistry=HookRegistry)
    executors = _module("strands.tools.executors", SequentialToolExecutor=SequentialToolExecutor)
    tools = _module("strands.tools", PythonAgentTool=PythonAgentTool, AgentTool=AgentTool,
                    executors=executors)
    return {
        "strands": _module("strands", Agent=Agent, hooks=hooks, tools=tools),
        "strands.hooks": hooks,
        "strands.tools": tools,
        "strands.tools.executors": executors,
    }


def langchain_tool_modules() -> tuple[dict[str, Any], type, type]:
    """``sys.modules`` entries for LangChain Core tools and LangGraph.

    Like LangChain's, ``BaseTool.invoke`` and ``ainvoke`` call ``run`` and
    ``arun``, which call the ``_run`` and ``_arun`` that tool classes such
    as ``StructuredTool`` override.

    ``StateGraph(nodes).compile()`` returns a ``Pregel`` whose ``nodes`` maps
    each name, ``__start__`` included, to a ``PregelNode``. A node given as a
    function runs in a ``RunnableCallable``, a runnable is the node's
    ``bound``, and any other object is the node itself. Like LangGraph:

    - a run reads every node's ``triggers``, then runs each node's ``bound``
      in turn, so a node that is a plain function fails;
    - ``RunnableCallable.invoke`` fails for a node with only an async function;
    - ``PregelNode.copy`` finds the node's subgraphs again in its new
      ``bound``, as LangGraph 0.2 does.
    """
    modules, _ = langchain_modules()

    class BaseTool:
        name = "tool"

        def _run(self, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError

        async def _arun(self, *args: Any, **kwargs: Any) -> Any:
            return self._run(*args, **kwargs)

        def run(self, tool_input: Any, callbacks: Any = None, **kwargs: Any) -> Any:
            return self._run(**tool_input) if isinstance(tool_input, dict) else self._run(tool_input)

        async def arun(self, tool_input: Any, callbacks: Any = None, **kwargs: Any) -> Any:
            if isinstance(tool_input, dict):
                return await self._arun(**tool_input)
            return await self._arun(tool_input)

        def invoke(self, tool_input: Any, **kwargs: Any) -> Any:
            return self.run(tool_input, callbacks=object(), **kwargs)

        async def ainvoke(self, tool_input: Any, **kwargs: Any) -> Any:
            return await self.arun(tool_input, callbacks=object(), **kwargs)

    class Runnable:
        """Like LangChain's, ``ainvoke`` and ``astream`` call ``invoke`` by default."""

        def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
            raise NotImplementedError

        async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
            return self.invoke(input, config, **kwargs)

        async def astream(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
            yield await self.ainvoke(input, config, **kwargs)

    class RunnableSequence(Runnable):
        def __init__(self, *steps: Any) -> None:
            self.steps = steps

        def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
            for step in self.steps:
                input = step.invoke(input, config)
            return input

    class RunnableCallable(Runnable):
        def __init__(self, func: Any, afunc: Any = None) -> None:
            self.func = func
            self.afunc = afunc

        def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
            if self.func is None:
                raise TypeError("No synchronous function provided")
            return self.func(input)

        async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
            if self.afunc is None:
                return self.invoke(input, config, **kwargs)
            return await self.afunc(input)

    class Pregel(Runnable):
        def __init__(self, nodes: dict[str, Any]) -> None:
            self.nodes = nodes

        def _ready(self) -> list[Any]:
            return [node for node in self.nodes.values() if node.triggers]

        def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
            for node in self._ready():
                input = node.bound.invoke(input, config)
            return input

        async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
            for node in self._ready():
                input = await node.bound.ainvoke(input, config)
            return input

        async def astream(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
            # A node's last chunk is the next node's input.
            for node in self._ready():
                async for chunk in node.bound.astream(input, config):
                    input = chunk
                    yield chunk

    class PregelNode:
        def __init__(self, triggers: list[str], bound: Any) -> None:
            self.triggers = triggers
            self.bound = bound
            self.subgraphs = [bound] if isinstance(bound, Pregel) else []

        def copy(self, update: dict[str, Any]) -> Any:
            attrs = {**self.__dict__, **update}
            attrs.pop("subgraphs")
            return PregelNode(**attrs)

    def identifier(obj: Any) -> str | None:
        """Like LangGraph's, the name of a node's cache entries."""
        if isinstance(obj, RunnableCallable):
            obj = obj.func
        name = getattr(obj, "__qualname__", None)
        module = getattr(obj, "__module__", None)
        return None if name is None or module is None else f"{module}.{name}"

    def as_node(name: str, node: Any) -> Any:
        if isinstance(node, Runnable):
            return PregelNode([name], node)
        if inspect.iscoroutinefunction(node):
            return PregelNode([name], RunnableCallable(None, node))
        if callable(node):
            return PregelNode([name], RunnableCallable(node))
        return node

    class StateGraph:
        def __init__(self, nodes: dict[str, Any]) -> None:
            self._nodes = nodes

        def compile(self) -> Any:
            nodes = {"__start__": PregelNode(["__start__"], RunnableCallable(lambda state: state))}
            nodes.update((name, as_node(name, node)) for name, node in self._nodes.items())
            return Pregel(nodes)

    class GraphBubbleUp(Exception):
        pass

    class GraphInterrupt(GraphBubbleUp):
        pass

    tools = _module("langchain_core.tools", BaseTool=BaseTool)
    runnables = _module("langchain_core.runnables", Runnable=Runnable,
                        RunnableSequence=RunnableSequence)
    graph = _module("langgraph.graph", StateGraph=StateGraph)
    errors = _module("langgraph.errors", GraphBubbleUp=GraphBubbleUp,
                     GraphInterrupt=GraphInterrupt)
    call = _module("langgraph.pregel._call", identifier=identifier)
    pregel = _module("langgraph.pregel", Pregel=Pregel, PregelNode=PregelNode,
                     RunnableCallable=RunnableCallable, _call=call)
    modules.update({
        "langchain_core.tools": tools,
        "langchain_core.runnables": runnables,
        "langgraph": _module("langgraph", graph=graph, errors=errors, pregel=pregel),
        "langgraph.graph": graph,
        "langgraph.errors": errors,
        "langgraph.pregel": pregel,
        "langgraph.pregel._call": call,
    })
    modules["langchain_core"].tools = tools
    modules["langchain_core"].runnables = runnables
    return modules, BaseTool, StateGraph
