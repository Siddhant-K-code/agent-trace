"""Auto-instrumentation entry point.

Importing this module triggers auto-instrumentation based on the
AGENT_STRACE_AUTO_INSTRUMENT environment variable.

Usage in sitecustomize.py (instrument at interpreter startup):
    import agent_trace.auto

Or via environment variable:
    AGENT_STRACE_AUTO_INSTRUMENT=openai-agents,langchain python my_agent.py
"""

from .integrations import auto_instrument_from_env

_instrumented = auto_instrument_from_env()
