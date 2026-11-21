"""Tests for export --output and for lone surrogates in export --format csv.

With --output, json, csv and ndjson write the file and print nothing on
stdout. Each export path that takes --output turns a file it cannot write into
one Error line and exit status 1. The CSV writes a lone UTF-16 surrogate in a
cell as its escape, so a strict UTF-8 stdout or file can take it.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import _escapes
except ImportError:
    from tests import _escapes

try:
    from _no_telemetry import without_telemetry
except ImportError:
    from tests._no_telemetry import without_telemetry

import agent_trace
from agent_trace.cli import build_parser, cmd_export
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore

SRC = os.path.dirname(os.path.dirname(os.path.abspath(agent_trace.__file__)))
LONE = _escapes.LONE_SURROGATE
SHOWN = _escapes.LONE_SURROGATE_SHOWN
PLAIN = _escapes.PLAIN
# Every export path that writes a file named by --output.
FILE_EXPORTS = (
    ("--format", "json"),
    ("--format", "csv"),
    ("--format", "ndjson"),
    ("--format", "otlp"),
    ("--format", "otlp-genai"),
    ("--format", "temporal"),
    ("--format", "eu-ai-act"),
    ("--anonymize",),
)


class _ExportCase(unittest.TestCase):
    """Run export on a session with a start event and a tool call."""

    event_id = "call-1"
    parent_id = ""
    data = {"tool_name": "Bash", "arguments": {"command": "ls"}}

    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = temp.name
        self.trace_dir = os.path.join(self.root, "traces")
        self.cwd = os.path.join(self.root, "cwd")
        os.makedirs(self.cwd)
        store = TraceStore(self.trace_dir)
        meta = SessionMeta(agent_name="probe", trace_id="a" * 32, parent_span_id="b" * 16, trace_flags="01")
        store.create_session(meta)
        for event in (
            TraceEvent(event_type=EventType.SESSION_START, session_id=meta.session_id, timestamp=1.0, data={}),
            TraceEvent(
                event_type=EventType.TOOL_CALL, session_id=meta.session_id, timestamp=2.0,
                event_id=self.event_id, parent_id=self.parent_id, data=self.data,
            ),
        ):
            store.append_event(meta.session_id, event)
        self.session_id = meta.session_id

    def export(self, *flags: str) -> tuple[int, str, str]:
        """Run export in-process in self.cwd and return (code, stdout, stderr).

        stdout refuses a lone surrogate, as a UTF-8 terminal does.
        """
        args = build_parser().parse_args(["--trace-dir", self.trace_dir, "export", self.session_id, *flags])
        out, err = _escapes.strict_utf8_stream(), io.StringIO()
        previous = os.getcwd()
        os.chdir(self.cwd)
        try:
            with redirect_stdout(out), redirect_stderr(err):
                code = cmd_export(args)
        finally:
            os.chdir(previous)
        return code, _escapes.written_text(out), err.getvalue()

    def run_cli(self, *flags: str) -> subprocess.CompletedProcess:
        """Run the agent-strace command in self.cwd with a strict UTF-8 stdout."""
        home = os.path.join(self.root, "home")
        os.makedirs(home, exist_ok=True)
        env = without_telemetry({
            key: value
            for key, value in os.environ.items()
            if not key.startswith("AGENT_STRACE_") and key != "AGENT_TRACE_DIR"
        })
        env["PYTHONPATH"] = os.pathsep.join(item for item in (SRC, env.get("PYTHONPATH", "")) if item)
        env["PYTHONIOENCODING"] = "utf-8:strict"
        env["HOME"] = home
        return subprocess.run(
            [sys.executable, "-m", "agent_trace.entry", "--trace-dir", self.trace_dir,
             "export", self.session_id, *flags],
            cwd=self.cwd, env=env, capture_output=True, text=True, encoding="utf-8", timeout=60,
        )


class TestOutputFile(_ExportCase):
    """With --output, json, csv and ndjson write the file and print nothing on stdout."""

    def test_file_holds_what_stdout_gets_without_it(self):
        for fmt in ("json", "csv", "ndjson"):
            with self.subTest(fmt=fmt):
                code, printed, err = self.export("--format", fmt)
                self.assertEqual((code, err), (0, ""))
                path = os.path.join(self.root, f"out.{fmt}")
                self.assertEqual(
                    self.export("--format", fmt, "--output", path),
                    (0, "", f"{fmt.upper()} export written to {path}\n"),
                )
                # The csv module writes CRLF, and the file keeps it.
                with open(path, encoding="utf-8", newline="" if fmt == "csv" else None) as f:
                    self.assertEqual(f.read(), printed)
        self.assertEqual(os.listdir(self.cwd), [])

    def test_symlinked_parent_directory(self):
        # /tmp is a symlink on macOS, and the VS Code extension passes the
        # path the user picked to export --format otlp --output.
        real = os.path.join(self.root, "real")
        os.makedirs(real)
        link = os.path.join(self.root, "link")
        try:
            os.symlink(real, link, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("cannot create a symlink here")
        for fmt in ("json", "otlp"):
            with self.subTest(fmt=fmt):
                code, out, _ = self.export("--format", fmt, "--output", os.path.join(link, f"out.{fmt}"))
                self.assertEqual((code, out), (0, ""))
                self.assertTrue(os.path.isfile(os.path.join(real, f"out.{fmt}")))


class TestUnwritableOutput(_ExportCase):
    """A file that export cannot write gives one Error line and exit status 1."""

    def test_missing_directory(self):
        path = os.path.join(self.root, "missing", "out.json")
        for flags in FILE_EXPORTS:
            with self.subTest(flags=flags):
                self.assertEqual(
                    self.export(*flags, "--output", path),
                    (1, "", f"Error: cannot write {path}: No such file or directory\n"),
                )
        self.assertEqual(os.listdir(self.cwd), [])

    def test_path_is_a_directory(self):
        # The reason is "Is a directory", or "Permission denied" on Windows.
        line = re.compile(rf"\AError: cannot write {re.escape(self.root)}: [^\n]+\n\Z")
        for flags in FILE_EXPORTS:
            with self.subTest(flags=flags):
                code, out, err = self.export(*flags, "--output", self.root)
                self.assertEqual((code, out), (1, ""))
                self.assertRegex(err, line)
        self.assertEqual(os.listdir(self.cwd), [])

    def test_cli_prints_no_traceback(self):
        path = os.path.join(self.root, "missing", "out.json")
        for fmt in ("json", "otlp"):
            with self.subTest(fmt=fmt):
                run = self.run_cli("--format", fmt, "--output", path)
                self.assertEqual(
                    (run.returncode, run.stdout, run.stderr),
                    (1, "", f"Error: cannot write {path}: No such file or directory\n"),
                )


class TestCsvLoneSurrogate(_ExportCase):
    """A stored event_id or parent_id can hold a lone UTF-16 surrogate.

    The CSV writes it as its escape, the text the stored line holds. Every
    other character stays as it is, and the data cell is json.dumps(data).
    """

    event_id = "call-" + LONE
    parent_id = PLAIN + " " + LONE
    data = {"command": "cat " + LONE + " " + PLAIN}
    row = ["2.0", "tool_call", "call-" + SHOWN, PLAIN + " " + SHOWN, "", json.dumps(data)]

    def test_stdout(self):
        code, out, err = self.export("--format", "csv")
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(list(csv.reader(io.StringIO(out, newline="")))[2], self.row)

    def test_output_file(self):
        path = os.path.join(self.root, "out.csv")
        code, out, _ = self.export("--format", "csv", "--output", path)
        self.assertEqual((code, out), (0, ""))
        with open(path, encoding="utf-8", newline="") as f:
            self.assertEqual(list(csv.reader(f))[2], self.row)

    def test_cli_exits_0(self):
        run = self.run_cli("--format", "csv")
        self.assertEqual((run.returncode, run.stderr), (0, ""))
        self.assertIn(f"call-{SHOWN},{PLAIN} {SHOWN},", run.stdout)


if __name__ == "__main__":
    unittest.main()
