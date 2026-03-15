"""Example: tracing a simple agent with the decorator API.

Run:
    python examples/basic_agent.py
    agent-trace replay
"""

import sys
import os
import time
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace.decorator import (
    end_session,
    log_decision,
    start_session,
    trace_llm_call,
    trace_tool,
)


# --- Simulated tools ---


@trace_tool
def read_file(path: str) -> str:
    """Read a file from disk."""
    time.sleep(random.uniform(0.01, 0.05))
    return f"contents of {path}: def hello(): print('world')"


@trace_tool
def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    time.sleep(random.uniform(0.01, 0.05))
    return f"wrote {len(content)} bytes to {path}"


@trace_tool
def search_codebase(query: str) -> str:
    """Search the codebase for a pattern."""
    time.sleep(random.uniform(0.02, 0.1))
    return f"found 3 matches for '{query}' in src/"


@trace_tool
def run_tests(test_path: str) -> str:
    """Run tests."""
    time.sleep(random.uniform(0.1, 0.3))
    if random.random() < 0.3:
        raise RuntimeError(f"Test failed: {test_path}")
    return "All 12 tests passed"


# --- Simulated LLM ---


@trace_llm_call
def call_llm(messages: list, model: str = "claude-4") -> str:
    """Simulate an LLM call."""
    time.sleep(random.uniform(0.05, 0.15))
    return "I'll read the file first, then make the changes."


# --- Agent loop ---


def main():
    session_id = start_session(name="coding-agent")
    print(f"Started trace session: {session_id}")

    # Step 1: agent gets a task
    response = call_llm(
        messages=[{"role": "user", "content": "Fix the bug in auth.py"}],
        model="claude-4",
    )

    # Step 2: agent decides to read the file first
    log_decision(
        choice="read_file_first",
        reason="Need to understand current implementation before making changes",
        alternatives=["read_file_first", "search_codebase", "write_fix_directly"],
    )

    # Step 3: agent reads the file
    content = read_file("src/auth.py")

    # Step 4: agent searches for related code
    results = search_codebase("authenticate")

    # Step 5: agent asks LLM for the fix
    response = call_llm(
        messages=[
            {"role": "user", "content": "Fix the bug in auth.py"},
            {"role": "assistant", "content": response},
            {"role": "user", "content": f"File contents: {content}\nSearch: {results}"},
        ],
        model="claude-4",
    )

    # Step 6: agent decides to write the fix
    log_decision(
        choice="apply_fix",
        reason="LLM provided a clear fix, confidence is high",
        alternatives=["apply_fix", "ask_for_clarification", "run_tests_first"],
    )

    # Step 7: agent writes the fix
    write_file("src/auth.py", "def authenticate(user):\n    return validate(user.token)")

    # Step 8: agent runs tests
    try:
        run_tests("tests/test_auth.py")
    except RuntimeError:
        # agent handles test failure
        log_decision(
            choice="retry_fix",
            reason="Tests failed, need to adjust the implementation",
        )
        write_file("src/auth.py", "def authenticate(user):\n    if not user.token:\n        raise AuthError()\n    return validate(user.token)")
        run_tests("tests/test_auth.py")

    meta = end_session()
    print(f"\nSession complete: {meta.session_id}")
    print(f"  Tool calls: {meta.tool_calls}")
    print(f"  LLM requests: {meta.llm_requests}")
    print(f"  Errors: {meta.errors}")
    print(f"  Duration: {meta.total_duration_ms:.0f}ms")
    print(f"\nReplay with: agent-strace replay {meta.session_id}")


if __name__ == "__main__":
    main()
