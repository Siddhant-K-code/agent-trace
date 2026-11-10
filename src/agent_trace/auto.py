"""Auto-instrumentation entry point.

Importing this module triggers auto-instrumentation based on the
AGENT_STRACE_AUTO_INSTRUMENT environment variable. Setting the variable
does nothing until this module is imported.

`agent-strace auto -- <command>` imports it at interpreter startup through
a generated sitecustomize (see auto_bootstrap.py). Without the wrapper,
import it first, before any framework import:

    import agent_trace.auto   # first line of my_agent.py

    AGENT_STRACE_AUTO_INSTRUMENT=openai-agents,langchain python my_agent.py
"""

from .integrations import auto_instrument_from_env

_instrumented = auto_instrument_from_env()
