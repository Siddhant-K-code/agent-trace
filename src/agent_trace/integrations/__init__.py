"""Auto-instrumentation integrations for agent frameworks.

Each integration monkey-patches a framework to emit agent-trace events
without requiring any code changes to the application.

Usage:
    from agent_trace.integrations import instrument_openai_agents
    instrument_openai_agents()

Or via environment variable (no code changes):
    AGENT_STRACE_AUTO_INSTRUMENT=openai-agents,langchain python my_agent.py

Or via CLI:
    agent-strace auto --framework openai-agents -- python my_agent.py
    agent-strace auto --detect -- python my_agent.py

Each integration ships as an optional extra:
    pip install agent-strace[openai-agents]
    pip install agent-strace[langchain]
    pip install agent-strace[litellm]
    pip install agent-strace[strands]
    pip install agent-strace[all-integrations]
"""

from __future__ import annotations

import os
import sys

# Registry: name → (module_path, function_name)
_INTEGRATIONS: dict[str, tuple[str, str]] = {
    "openai-agents": ("agent_trace.integrations.openai_agents", "instrument_openai_agents"),
    "openai_agents": ("agent_trace.integrations.openai_agents", "instrument_openai_agents"),
    "langchain": ("agent_trace.integrations.langchain", "instrument_langchain"),
    "litellm": ("agent_trace.integrations.litellm", "instrument_litellm"),
    "anthropic": ("agent_trace.integrations.anthropic", "instrument_anthropic"),
    "openai": ("agent_trace.integrations.openai", "instrument_openai"),
    "strands": ("agent_trace.integrations.strands", "instrument_strands"),
}

# Frameworks that can be auto-detected by checking importability
_DETECTABLE: list[str] = [
    "openai-agents",
    "langchain",
    "litellm",
    "anthropic",
    "openai",
    "strands",
]

_FRAMEWORK_PROBE: dict[str, str] = {
    "openai-agents": "agents",
    "langchain": "langchain_core",
    "litellm": "litellm",
    "anthropic": "anthropic",
    "openai": "openai",
    "strands": "strands",
}


def _import_integration(name: str):
    """Import and return the instrument_* function for a named integration."""
    entry = _INTEGRATIONS.get(name)
    if entry is None:
        raise ValueError(
            f"Unknown integration: {name!r}. "
            f"Available: {', '.join(sorted(_INTEGRATIONS))}"
        )
    module_path, func_name = entry
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, func_name)


def instrument_openai_agents(**kwargs):
    """Instrument the OpenAI Agents SDK."""
    return _import_integration("openai-agents")(**kwargs)


def instrument_langchain(**kwargs):
    """Instrument LangChain / LangGraph."""
    return _import_integration("langchain")(**kwargs)


def instrument_litellm(**kwargs):
    """Instrument LiteLLM."""
    return _import_integration("litellm")(**kwargs)


def instrument_anthropic(**kwargs):
    """Instrument the Anthropic Python SDK."""
    return _import_integration("anthropic")(**kwargs)


def instrument_openai(**kwargs):
    """Instrument the OpenAI Python SDK."""
    return _import_integration("openai")(**kwargs)


def instrument_strands(**kwargs):
    """Instrument AWS Strands Agents."""
    return _import_integration("strands")(**kwargs)


def detect_and_instrument() -> list[str]:
    """Auto-detect installed frameworks and instrument all of them.

    Returns a list of framework names that were successfully instrumented.
    """
    instrumented: list[str] = []
    for name, probe_module in _FRAMEWORK_PROBE.items():
        try:
            import importlib
            importlib.import_module(probe_module)
        except ImportError:
            continue
        try:
            fn = _import_integration(name)
            fn()
            instrumented.append(name)
        except Exception as exc:
            sys.stderr.write(f"[agent-strace] auto-instrument {name} failed: {exc}\n")
    return instrumented


def auto_instrument_from_env() -> list[str]:
    """Read AGENT_STRACE_AUTO_INSTRUMENT and instrument listed frameworks.

    Value is a comma-separated list of framework names, or 'detect' to
    auto-detect all installed frameworks.
    """
    env = os.environ.get("AGENT_STRACE_AUTO_INSTRUMENT", "").strip()
    if not env:
        return []
    if env.lower() == "detect":
        return detect_and_instrument()
    names = [n.strip() for n in env.split(",") if n.strip()]
    instrumented: list[str] = []
    for name in names:
        try:
            fn = _import_integration(name)
            fn()
            instrumented.append(name)
        except Exception as exc:
            sys.stderr.write(f"[agent-strace] auto-instrument {name} failed: {exc}\n")
    return instrumented
