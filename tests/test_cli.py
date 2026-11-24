"""How the CLI refuses a session argument it cannot use.

An argument that cannot name a session directory, such as ``..`` or a path,
is refused in one line that shows it with repr(). A session whose directory
or meta.json is a symlink is not read and counts as not found.
"""

import concurrent.futures
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import agent_trace
from agent_trace.freeze import freeze_session
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore

try:
    from _no_telemetry import without_telemetry
    from test_store import INVALID_SESSION_ARGUMENTS, make_run1_not_a_session
except ImportError:
    from tests._no_telemetry import without_telemetry
    from tests.test_store import INVALID_SESSION_ARGUMENTS, make_run1_not_a_session

SRC_ROOT = Path(agent_trace.__file__).resolve().parent.parent
# Only a command that reads the session outside the store can print this.
OUTSIDE_MARKER = "outside-the-store-5e2b"


def link_snapshot(root: Path) -> dict:
    """Map each path under root to its bytes, ("link", target), or "dir".

    A symlink is recorded with its target and not followed.
    """
    snapshot = {}
    for directory, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            path = Path(directory, name)
            key = path.relative_to(root).as_posix()
            if path.is_symlink():
                snapshot[key] = ("link", os.readlink(path))
            elif path.is_dir():
                snapshot[key] = "dir"
            else:
                snapshot[key] = path.read_bytes()
    return snapshot


class SessionArgumentCase(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="session-argument-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        (self.root / "home").mkdir()
        self.env = without_telemetry({
            key: value
            for key, value in os.environ.items()
            if not key.startswith("AGENT_STRACE_") and key != "AGENT_TRACE_DIR"
        })
        self.env["PYTHONPATH"] = str(SRC_ROOT)
        self.env["HOME"] = self.env["USERPROFILE"] = str(self.root / "home")
        self.env["PYTHONIOENCODING"] = "utf-8"

    def make_store(self, name: str) -> TraceStore:
        """Return a store under root/name with the sessions okay and run1."""
        store = TraceStore(str(self.root / name / "traces"), use_workspace_env=False)
        for number, (session_id, text) in enumerate(
            (("okay", "hi"), ("run1", OUTSIDE_MARKER)), start=1
        ):
            store.create_session(
                SessionMeta(session_id=session_id, agent_name=text, started_at=float(number))
            )
            store.append_event(session_id, TraceEvent(
                EventType.USER_PROMPT, timestamp=float(number), data={"prompt": text}
            ))
        return store

    def run_commands(self, trace_dir: Path, commands: list) -> list:
        def run(command):
            return subprocess.run(
                [sys.executable, "-m", "agent_trace.entry", "--trace-dir", str(trace_dir),
                 *command],
                capture_output=True, text=True, encoding="utf-8", env=self.env,
                cwd=self.root, stdin=subprocess.DEVNULL, timeout=120,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            return list(pool.map(run, commands))


class TestInvalidSessionArgument(SessionArgumentCase):
    """Every command that takes a session ID or prefix refuses one that cannot
    name a session directory in one line. Nothing is written."""

    def test_each_command_refuses_it_in_one_line(self):
        store = self.make_store("store")
        fixture = self.root / "fixture.json"
        fixture.write_text(freeze_session(store, "okay").to_json() + "\n", encoding="utf-8")
        # eval gate reads its config before it looks up the session.
        gate_config = self.root / "gate.yaml"
        gate_config.write_text(
            "evals:\n"
            "  - name: no-errors\n"
            "    scorer: error_count\n"
            "    threshold: 0\n"
            "    fail_on: above\n",
            encoding="utf-8",
        )
        # ID stands for the argument. why and eval gate get the arguments
        # they need to reach the session lookup.
        forms = [
            ("replay", "ID"),
            ("tree", "ID"),
            ("inspect", "ID"),
            ("export", "ID"),
            ("export", "ID", "--format", "eu-ai-act"),
            ("export", "ID", "--anonymize"),
            ("stats", "ID"),
            ("explain", "ID"),
            ("timeline", "ID"),
            ("why", "ID", "1"),
            ("cost", "ID"),
            ("audit", "ID"),
            ("verify", "ID"),
            ("health", "ID"),
            ("postmortem", "ID"),
            ("eval", "gate", "ID", "--config", str(gate_config)),
            ("eval", "run", "ID"),
            ("eval", "ci", "ID"),
            ("cognitive-debt", "--session", "ID"),
            ("mcp-scan", "--session", "ID"),
            ("annotate", "ID"),
            ("token-budget", "ID"),
            ("a2a-tree", "ID"),
            ("baseline", "check", "ID"),
            ("compliance", "export", "ID"),
            ("identity", "sign", "ID"),
            ("identity", "verify", "ID"),
            ("freeze", "ID"),
            ("optimize", "ID"),
            ("standup", "--no-llm", "ID"),
            ("lint", "ID"),
            ("diff", "okay", "ID"),
            ("eval", "compare", "okay", "ID"),
            ("replay", "okay", "--diff", "ID"),
            ("record", "--parent", "ID", "--", sys.executable, "-c", "pass"),
            ("record-http", "--parent", "ID", "--url", "http://127.0.0.1:9", "--port", "0"),
            ("share", "ID"),
            ("share", "ID", "--stdout"),
            ("share", "ID", "--assignment"),
            ("watch", "ID"),
            ("skills", "ID"),
            ("compaction", "ID"),
            ("policy", "generate", "--dry-run", "ID"),
            ("regression", str(fixture), "ID"),
            ("compare", "ID"),
            ("compare", "okay", "ID"),
            ("stats", "ID", "--include-subagents"),
            ("replay", "ID", "--tree"),
            ("export", "ID", "--backend", "langfuse"),
            ("eval", "dataset", "add", "--session", "ID"),
            ("pr-comment", "ID"),
            ("review", "ID"),
        ]
        cases, commands = [], []
        for value in INVALID_SESSION_ARGUMENTS:
            for form in forms:
                cases.append((value, form))
                commands.append([value if part == "ID" else part for part in form])

        before = link_snapshot(self.root)
        results = self.run_commands(store.base_dir, commands)
        for (value, form), result in zip(cases, results):
            line = f"invalid session ID or prefix: {value!r}\n"
            if form[0] == "review":
                expected = (2, "", f"review: {line}")
            elif form[0] == "pr-comment":
                expected = (1, "", f"pr-comment: {line}")
            else:
                expected = (1, "", f"Error: {line}")
            with self.subTest(command=form, value=value):
                self.assertNotIn("Traceback", result.stderr)
                self.assertNotIn("\x01", result.stdout + result.stderr)
                self.assertEqual((result.returncode, result.stdout, result.stderr), expected)
        self.assertEqual(link_snapshot(self.root), before)


class TestSessionTheStoreRefuses(SessionArgumentCase):
    """A session whose directory or meta.json is a symlink counts as not
    found. Nothing outside the store is read, and nothing is written."""

    def test_each_command_reports_it_not_found(self):
        commands = [
            ("replay", "run1"),
            ("inspect", "run1"),
            ("stats", "run1"),
            ("export", "run1"),
            ("export", "run1", "--anonymize"),
            ("mcp-scan", "--session", "run1"),
            ("verify", "run1"),
            ("export", "run1", "--format", "eu-ai-act"),
            ("replay", "okay", "--diff", "run1"),
            ("cognitive-debt", "--session", "run1"),
            ("health", "run1"),
        ]
        for state in ("run1 a symlinked directory", "meta.json a symlink"):
            with self.subTest(state=state):
                store = self.make_store(state.replace(" ", "-"))
                make_run1_not_a_session(store.base_dir, store.base_dir.parent, state)
                before = link_snapshot(self.root)
                results = self.run_commands(store.base_dir, commands)
                for command, result in zip(commands, results):
                    with self.subTest(command=command):
                        self.assertNotIn(OUTSIDE_MARKER, result.stdout + result.stderr)
                        self.assertEqual(
                            (result.returncode, result.stdout, result.stderr),
                            (1, "", "Session not found: run1\n"),
                        )
                self.assertEqual(link_snapshot(self.root), before)


if __name__ == "__main__":
    unittest.main()
