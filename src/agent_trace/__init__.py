"""agent-trace: strace for AI agents."""

from __future__ import annotations

__version__ = "0.103.0"

# The decorator API is exported lazily (PEP 562) so ``import agent_trace``
# stays cheap and does not load the store until a name is first used.
_DECORATOR_EXPORTS = (
    "end_session",
    "log_decision",
    "start_session",
    "trace_llm_call",
    "trace_tool",
)

__all__ = list(_DECORATOR_EXPORTS)


def __getattr__(name: str):
    if name in _DECORATOR_EXPORTS:
        from . import decorator

        value = getattr(decorator, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
