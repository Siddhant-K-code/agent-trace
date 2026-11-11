"""Tests for session-relative paths and shell file targets."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace.models import EventType, TraceEvent
from agent_trace.pr_comment import session_file_activity
from agent_trace.session_paths import (
    command_text,
    cwd_forms,
    is_shell_call,
    normalise_path,
    session_cwd,
    shell_reads,
    shell_writes,
)


def _ev(kind: EventType, **data) -> TraceEvent:
    return TraceEvent(event_type=kind, timestamp=1.0, session_id="s", data=data)


def _bash(command: str) -> TraceEvent:
    return _ev(EventType.TOOL_CALL, tool_name="bash", arguments={"command": command})


class TestSessionPaths(unittest.TestCase):
    def test_session_cwd_from_session_start(self):
        events = [
            _ev(EventType.USER_PROMPT, prompt="x"),
            _ev(EventType.SESSION_START, cwd="/private/tmp/atdemo/"),
        ]
        self.assertEqual(session_cwd(events), "/private/tmp/atdemo")
        self.assertEqual(session_cwd([]), "")

    def test_cwd_forms_include_macos_alias(self):
        self.assertEqual(cwd_forms("/private/tmp/a"), ("/private/tmp/a", "/tmp/a"))
        self.assertEqual(cwd_forms("/tmp/a"), ("/tmp/a", "/private/tmp/a"))
        self.assertEqual(cwd_forms("/work/repo"), ("/work/repo",))
        self.assertEqual(cwd_forms(""), ())

    def test_normalise_path(self):
        cwd = "/private/tmp/atdemo"
        self.assertEqual(normalise_path("/tmp/atdemo/README.md", cwd), "README.md")
        self.assertEqual(normalise_path("/private/tmp/atdemo/src/a.py", cwd), "src/a.py")
        self.assertEqual(normalise_path("./src/a.py", cwd), "src/a.py")
        self.assertEqual(normalise_path("/elsewhere/a.py", cwd), "/elsewhere/a.py")
        self.assertEqual(normalise_path("C:\\repo\\a.py", "C:/repo"), "a.py")

    def test_command_text_and_shell_call(self):
        call = _bash("ls")
        self.assertEqual(command_text(call), "ls")
        self.assertTrue(is_shell_call(call))
        self.assertFalse(is_shell_call(_ev(EventType.TOOL_RESULT, command="ls")))

    def test_shell_reads_and_writes_are_cwd_relative(self):
        cwd = "/private/tmp/atdemo"
        call = _bash("cd /tmp/atdemo && cat src/a.py > out.txt")
        self.assertEqual(shell_reads(call, cwd), ["src/a.py"])
        self.assertEqual(shell_writes(call, cwd), ["out.txt"])


class TestSessionFileActivity(unittest.TestCase):
    def test_combines_explicit_and_shell_activity(self):
        events = [
            _ev(EventType.SESSION_START, cwd="/private/tmp/atdemo"),
            _bash("cd /private/tmp/atdemo && cat src/a.py README.md"),
            _ev(EventType.TOOL_CALL, tool_name="view", arguments={"path": "/tmp/atdemo/src/b.py"}),
            _ev(EventType.TOOL_CALL, tool_name="view", arguments={"path": "/private/tmp/atdemo/src/b.py"}),
            _ev(EventType.TOOL_CALL, tool_name="edit", arguments={"path": "/tmp/atdemo/README.md"}),
            _bash("echo x > /private/tmp/atdemo/notes.txt"),
        ]
        activity = session_file_activity(events)
        self.assertEqual(activity.read, ["README.md", "src/a.py", "src/b.py"])
        self.assertEqual(activity.written, ["README.md", "notes.txt"])
        self.assertEqual(activity.explicit_written, frozenset({"README.md"}))
        self.assertEqual(activity.redundant, {"src/b.py": 2})


if __name__ == "__main__":
    unittest.main()
