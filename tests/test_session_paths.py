"""Tests for session-relative paths and shell file targets."""

import os
import sys
import tempfile
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
        self.assertEqual(session_cwd([_ev(EventType.SESSION_START, cwd="/")]), "/")
        self.assertEqual(session_cwd([_ev(EventType.SESSION_START, cwd="C:\\")]), "C:/")
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
        self.assertEqual(normalise_path("/workspace/file.py", "/"), "workspace/file.py")
        self.assertEqual(normalise_path("c:/REPO/src/file.py", "C:/repo"), "src/file.py")
        self.assertEqual(
            normalise_path("C:/repo-sibling/file.py", "C:/repo"),
            "C:/repo-sibling/file.py",
        )
        self.assertEqual(
            normalise_path("C:/repo/src/../file.py", "C:/repo"), "file.py"
        )
        self.assertEqual(normalise_path("./src/file.py", "C:/repo"), "src/file.py")

    def test_normalise_path_under_unc_share_root(self):
        self.assertEqual(
            normalise_path("//server/share/a.py", "//server/share"), "a.py"
        )
        self.assertEqual(
            normalise_path("//SERVER/SHARE/src/a.py", "//server/share"),
            "src/a.py",
        )
        self.assertEqual(
            normalise_path("//server/share/src/a.py", "//SERVER/SHARE/"),
            "src/a.py",
        )
        self.assertEqual(
            normalise_path("//server/share/src/../a.py", "//server/share"),
            "a.py",
        )
        self.assertEqual(
            normalise_path("//server/share-sibling/a.py", "//server/share"),
            "//server/share-sibling/a.py",
        )
        self.assertEqual(
            normalise_path("//server/other/a.py", "//server/share"),
            "//server/other/a.py",
        )

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


class TestResolvePath(unittest.TestCase):
    """watch resolves relative paths as text against an absolute session cwd."""

    def test_join_and_normalize(self):
        from agent_trace.session_paths import resolve_path

        cases = (
            ("a.py", "/app", "/app/a.py"),
            ("./src/./a.py", "/app", "/app/src/a.py"),
            ("../a.py", "/app/src", "/app/a.py"),
            ("/app/src/../a.py", "", "/app/a.py"),
            ("/app//src/a.py", "/other", "/app/src/a.py"),
            ("a.py", "", "a.py"),
            ("", "/app", ""),
            ("file:///app/a.py", "/app", "file:///app/a.py"),
            ("~/a.py", "/app", "~/a.py"),
            ("config\\production.env", "C:\\repo", "C:/repo/config/production.env"),
            ("..\\a.py", "C:\\repo\\src", "C:/repo/a.py"),
            ("C:\\repo\\src\\..\\a.py", "/app", "C:/repo/a.py"),
            ("\\\\server\\share\\a.py", "", "//server/share/a.py"),
            ("a.py", "\\\\server\\share", "//server/share/a.py"),
        )
        for path, cwd, expected in cases:
            with self.subTest(path=path, cwd=cwd):
                self.assertEqual(resolve_path(path, cwd), expected)

    def test_symlinks_are_not_followed(self):
        from agent_trace.session_paths import resolve_path

        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.realpath(tmp)
            os.makedirs(os.path.join(root, "real", "sub"))
            try:
                os.symlink(os.path.join(root, "real", "sub"), os.path.join(root, "link"))
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"cannot create a symlink: {exc}")
            self.assertEqual(resolve_path("link/../a.py", root), f"{root}/a.py")

    def test_absolute_cwd(self):
        from agent_trace.session_paths import absolute_cwd

        for value in ("/app", "C:\\repo", "C:/repo", "\\\\server\\share"):
            with self.subTest(value=value):
                self.assertEqual(absolute_cwd(value), value)
        for value in ("app", "", "C:repo", "~/app", None, 42, ["/app"]):
            with self.subTest(value=value):
                self.assertEqual(absolute_cwd(value), "")


class TestSessionFileActivity(unittest.TestCase):
    def test_root_cwd_and_windows_case_variants_are_repo_relative(self):
        root_activity = session_file_activity([
            _ev(EventType.SESSION_START, cwd="/"),
            _ev(EventType.TOOL_CALL, tool_name="view", arguments={"path": "/workspace/a.py"}),
        ])
        self.assertEqual(root_activity.read, ["workspace/a.py"])

        windows_activity = session_file_activity([
            _ev(EventType.SESSION_START, cwd="C:\\Repo\\"),
            _ev(EventType.TOOL_CALL, tool_name="view", arguments={"path": "c:/repo/src/a.py"}),
            _ev(EventType.TOOL_CALL, tool_name="edit", arguments={"path": "C:/REPO/src/a.py"}),
        ])
        self.assertEqual(windows_activity.read, ["src/a.py"])
        self.assertEqual(windows_activity.written, ["src/a.py"])

    def test_unc_share_root_and_case_variants_are_repo_relative(self):
        activity = session_file_activity([
            _ev(EventType.SESSION_START, cwd="\\\\server\\share\\"),
            _ev(
                EventType.TOOL_CALL,
                tool_name="view",
                arguments={"path": "//SERVER/SHARE/src/a.py"},
            ),
            _ev(
                EventType.TOOL_CALL,
                tool_name="edit",
                arguments={"path": "\\\\server\\share\\src\\a.py"},
            ),
        ])
        self.assertEqual(activity.read, ["src/a.py"])
        self.assertEqual(activity.written, ["src/a.py"])

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
