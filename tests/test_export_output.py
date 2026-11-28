"""Tests for export --output and for lone surrogates in export.

With --output, json, csv and ndjson write the file and print nothing on
stdout. Each export path that takes --output turns a file it cannot write into
one Error line and exit status 1. The CSV writes a lone UTF-16 surrogate in a
cell as its escape, so a strict UTF-8 stdout or file can take it. OTLP export
takes an event ID or parent ID that holds one, to a file or to a collector,
and reports a conversion error in one line either way.
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
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

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
from agent_trace import otlp, temporal
from agent_trace.cli import build_parser, cmd_export
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.otlp import _to_span_id
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

    def run_cli(self, *flags: str, program: str = "") -> subprocess.CompletedProcess:
        """Run the agent-strace command in self.cwd with a strict UTF-8 stdout.

        A *program* runs with ``python -c`` in place of the entry module.
        """
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
        command = ["-c", program] if program else ["-m", "agent_trace.entry"]
        return subprocess.run(
            [sys.executable, *command, "--trace-dir", self.trace_dir,
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


class _Collector(ThreadingHTTPServer):
    """A local collector that answers each POST with 200 and keeps its path and body."""

    daemon_threads = True
    block_on_close = False

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _CollectorHandler)
        self.posts: list[tuple[str, bytes]] = []
        self.url = f"http://127.0.0.1:{self.server_address[1]}"


class _CollectorHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.server.posts.append((self.path, body))
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args: object) -> None:
        pass


class _OtlpCase(_ExportCase):
    """Add a result for the tool call, and run a local collector."""

    result_id = "result-1"

    def setUp(self) -> None:
        super().setUp()
        TraceStore(self.trace_dir).append_event(self.session_id, TraceEvent(
            event_type=EventType.TOOL_RESULT, session_id=self.session_id, timestamp=3.0,
            event_id=self.result_id, parent_id=self.event_id, duration_ms=250.0,
            data={"tool_name": "Bash", "result": "ok"},
        ))
        # Keep the POST to 127.0.0.1 off any proxy the machine sets.
        environ = patch.dict(os.environ, {"no_proxy": "*"})
        environ.start()
        self.addCleanup(environ.stop)
        self.collector = _Collector()
        threading.Thread(
            target=self.collector.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True,
        ).start()
        self.addCleanup(self.collector.server_close)
        self.addCleanup(self.collector.shutdown)


class TestOtlpLoneSurrogate(_OtlpCase):
    """OTLP export takes a session whose event IDs and parent IDs hold a lone surrogate.

    The span ID is the hash of the ID's surrogatepass bytes. json.dumps writes
    the surrogate in an attribute as its escape, so the file and the POST body
    are ASCII.
    """

    event_id = "call-" + LONE
    parent_id = PLAIN + " " + LONE
    result_id = "result-" + LONE

    def check_payload(self, text: str, fmt: str) -> None:
        """Check that *text* holds the call's span, paired with its result."""
        self.assertTrue(text.isascii())
        self.assertIn('"call-' + SHOWN + '"', text)
        spans = json.loads(text)["resourceSpans"][0]["scopeSpans"][0]["spans"]
        (call,) = [span for span in spans if span["spanId"] == _to_span_id(self.event_id)]
        # The result, found by its parent ID, ends the span 250 ms after the call.
        self.assertEqual((call["startTimeUnixNano"], call["endTimeUnixNano"]), ("2000000000", "2250000000"))
        attrs = {a["key"]: a["value"].get("stringValue") for a in call["attributes"]}
        self.assertEqual(attrs["gen_ai.tool.call.id"], self.event_id)
        if fmt != "otlp":
            self.assertEqual(attrs["agent.strace.request.parent_event_id"], self.parent_id)
            self.assertEqual(attrs["agent.strace.response.event_id"], self.result_id)

    def test_output_file(self):
        for fmt in ("otlp", "otlp-genai", "temporal"):
            with self.subTest(fmt=fmt):
                path = os.path.join(self.root, f"{fmt}.json")
                self.assertEqual(self.export("--format", fmt, "--output", path), (0, "", (
                    f"OTLP payload written to {path}\n"
                    f"Send to a collector: agent-strace export --format {fmt} --endpoint <url>\n"
                )))
                with open(path, encoding="utf-8") as f:
                    self.check_payload(f.read(), fmt)

    def test_endpoint(self):
        for fmt in ("otlp", "otlp-genai", "temporal"):
            with self.subTest(fmt=fmt):
                self.collector.posts.clear()
                self.assertEqual(
                    self.export("--format", fmt, "--endpoint", self.collector.url),
                    (0, "", f"Exported 3 events to {self.collector.url} (HTTP 200)\n"),
                )
                ((path, body),) = self.collector.posts
                self.assertEqual(path, "/v1/traces")
                self.check_payload(body.decode("utf-8"), fmt)

    def test_cli_exits_0(self):
        run = self.run_cli("--format", "otlp", "--endpoint", self.collector.url)
        self.assertEqual(
            (run.returncode, run.stdout, run.stderr),
            (0, "", f"Exported 3 events to {self.collector.url} (HTTP 200)\n"),
        )
        self.assertEqual(len(self.collector.posts), 1)


class TestOtlpConversionError(_OtlpCase):
    """A ValueError from the GenAI conversion gives one line and exit status 1.

    otlp-genai and temporal both convert with session_to_otlp_genai. These
    tests patch it to raise one. Export writes no file and sends nothing,
    with or without --endpoint.
    """

    labels = (("otlp-genai", "OTLP"), ("temporal", "Temporal"))
    # A child process does not see the patch in setUp, so it makes its own.
    PATCHED_CLI = (
        "from unittest.mock import patch\n"
        "from agent_trace.entry import main\n"
        "with patch('agent_trace.otlp.session_to_otlp_genai',"
        " side_effect=ValueError('no spans')):\n"
        "    main()\n"
    )

    def setUp(self) -> None:
        super().setUp()
        # temporal.py imports the function by name, so it is patched there too.
        for module in (otlp, temporal):
            converter = patch.object(module, "session_to_otlp_genai", side_effect=ValueError("no spans"))
            converter.start()
            self.addCleanup(converter.stop)

    def test_output_file(self):
        for fmt, label in self.labels:
            with self.subTest(fmt=fmt):
                path = os.path.join(self.root, f"{fmt}.json")
                self.assertEqual(
                    self.export("--format", fmt, "--output", path),
                    (1, "", f"{label} export failed: no spans\n"),
                )
                self.assertFalse(os.path.exists(path))

    def test_endpoint(self):
        for fmt, label in self.labels:
            with self.subTest(fmt=fmt):
                self.assertEqual(
                    self.export("--format", fmt, "--endpoint", self.collector.url),
                    (1, "", f"{label} export failed: no spans\n"),
                )
        self.assertEqual(self.collector.posts, [])
        self.assertEqual(os.listdir(self.cwd), [])

    def test_cli_prints_no_traceback(self):
        run = self.run_cli(
            "--format", "otlp-genai", "--endpoint", self.collector.url, program=self.PATCHED_CLI,
        )
        self.assertEqual(
            (run.returncode, run.stdout, run.stderr), (1, "", "OTLP export failed: no spans\n"),
        )
        self.assertEqual(self.collector.posts, [])


class TestOtlpConverterError(_OtlpCase):
    """otlp turns a ValueError from session_to_otlp into one line and exit status 1.

    No stored value makes session_to_otlp raise one today, so these tests
    patch it. export_otlp catches it on the --endpoint path, as cmd_export
    does on the file path.
    """

    def setUp(self) -> None:
        super().setUp()
        converter = patch("agent_trace.otlp.session_to_otlp", side_effect=ValueError("no spans"))
        converter.start()
        self.addCleanup(converter.stop)

    def test_output_file(self):
        path = os.path.join(self.root, "otlp.json")
        self.assertEqual(
            self.export("--format", "otlp", "--output", path),
            (1, "", "OTLP export failed: no spans\n"),
        )
        self.assertFalse(os.path.exists(path))

    def test_endpoint(self):
        self.assertEqual(
            self.export("--format", "otlp", "--endpoint", self.collector.url),
            (1, "", "OTLP export failed: no spans\n"),
        )
        self.assertEqual(self.collector.posts, [])
        self.assertEqual(os.listdir(self.cwd), [])


if __name__ == "__main__":
    unittest.main()
