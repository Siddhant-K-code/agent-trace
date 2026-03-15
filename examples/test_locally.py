"""Test agent-strace locally without any external dependencies.

Run:
    uvx agent-strace list                    # should show "No traces found"
    python examples/test_locally.py          # generates a realistic trace
    uvx agent-strace list                    # should show the session
    uvx agent-strace replay                  # replay the timeline
    uvx agent-strace stats                   # tool call frequency
    uvx agent-strace inspect <session-id>    # raw JSON
    uvx agent-strace export <session-id> --format ndjson
"""

import sys
import os
import time
import random

# Use the local source if available, otherwise the installed package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace.decorator import (
    end_session,
    log_decision,
    start_session,
    trace_llm_call,
    trace_tool,
)


# ── Simulated tools (these would be real in your agent) ──


@trace_tool
def read_file(path: str) -> str:
    time.sleep(random.uniform(0.02, 0.08))
    files = {
        "src/auth.py": "def authenticate(user):\n    return check_password(user.password)",
        "src/db.py": "def get_user(id):\n    return db.query('SELECT * FROM users WHERE id = ?', id)",
        "tests/test_auth.py": "def test_login():\n    assert authenticate(mock_user) == True",
    }
    return files.get(path, f"# empty file: {path}")


@trace_tool
def write_file(path: str, content: str) -> str:
    time.sleep(random.uniform(0.01, 0.04))
    return f"wrote {len(content)} bytes to {path}"


@trace_tool
def search_codebase(query: str) -> str:
    time.sleep(random.uniform(0.03, 0.1))
    return f"found 3 matches for '{query}' in src/auth.py, src/db.py, src/middleware.py"


@trace_tool
def run_tests(path: str) -> str:
    time.sleep(random.uniform(0.1, 0.3))
    return "12 tests passed, 0 failed"


@trace_tool
def run_linter(path: str) -> str:
    time.sleep(random.uniform(0.05, 0.1))
    return "no issues found"


@trace_llm_call
def ask_llm(messages: list, model: str = "claude-4") -> str:
    time.sleep(random.uniform(0.05, 0.15))
    responses = [
        "I'll read the auth module first to understand the current implementation.",
        "The bug is in the password check. It doesn't handle None values. Here's the fix.",
        "The fix looks correct. Let me verify by running the tests.",
    ]
    return random.choice(responses)


# ── Agent loop ──


def main():
    session_id = start_session(name="bugfix-agent")
    print(f"Recording session: {session_id}\n")

    # Step 1: agent receives task
    print("  [1/7] Asking LLM for plan...")
    ask_llm(
        messages=[{"role": "user", "content": "Fix the authentication bug in src/auth.py"}],
        model="claude-4",
    )

    # Step 2: agent decides approach
    log_decision(
        choice="read_then_fix",
        reason="Need to understand the current auth flow before changing it",
        alternatives=["read_then_fix", "search_first", "write_fix_directly"],
    )

    # Step 3: read relevant files
    print("  [2/7] Reading source files...")
    read_file("src/auth.py")
    read_file("src/db.py")

    # Step 4: search for related code
    print("  [3/7] Searching codebase...")
    search_codebase("authenticate")
    search_codebase("check_password")

    # Step 5: ask LLM for the fix
    print("  [4/7] Asking LLM for fix...")
    ask_llm(
        messages=[
            {"role": "user", "content": "Fix the authentication bug"},
            {"role": "assistant", "content": "I'll read the auth module first."},
            {"role": "user", "content": "Here's auth.py: def authenticate(user): ..."},
        ],
        model="claude-4",
    )

    # Step 6: apply fix
    log_decision(
        choice="apply_fix",
        reason="LLM provided a clear fix for the None check",
        alternatives=["apply_fix", "ask_for_clarification", "run_tests_first"],
    )

    print("  [5/7] Writing fix...")
    write_file(
        "src/auth.py",
        "def authenticate(user):\n    if user.password is None:\n        return False\n    return check_password(user.password)",
    )

    # Step 7: verify
    print("  [6/7] Running tests...")
    run_tests("tests/")

    print("  [7/7] Running linter...")
    run_linter("src/auth.py")

    # Step 8: final LLM confirmation
    ask_llm(
        messages=[{"role": "user", "content": "Tests pass. Summarize what you changed."}],
        model="claude-4",
    )

    meta = end_session()

    print(f"\n{'─' * 50}")
    print(f"Session complete: {meta.session_id}")
    print(f"  Tool calls:  {meta.tool_calls}")
    print(f"  LLM requests: {meta.llm_requests}")
    print(f"  Errors:      {meta.errors}")
    print(f"  Duration:    {meta.total_duration_ms:.0f}ms")
    print(f"{'─' * 50}")
    print(f"\nNow try:")
    print(f"  uvx agent-strace replay {meta.session_id[:8]}")
    print(f"  uvx agent-strace stats {meta.session_id[:8]}")
    print(f"  uvx agent-strace inspect {meta.session_id[:8]}")
    print(f"  uvx agent-strace export {meta.session_id[:8]} --format ndjson")


if __name__ == "__main__":
    main()
