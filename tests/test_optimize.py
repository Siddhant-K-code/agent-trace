"""Tests for agent-strace optimize."""

import argparse
import contextlib
import io
import json
import os
import re
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import agent_trace
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.optimize import (
    FailurePattern,
    OptimizeProposal,
    _call_llm,
    _extract_patterns_heuristic,
    _parse_llm_patterns,
    cmd_optimize,
    print_proposal,
    run_optimize,
)
from agent_trace.store import TraceStore

try:
    from _no_telemetry import without_telemetry
except ImportError:
    from tests._no_telemetry import without_telemetry

try:
    from _redirect_servers import ALL_REDIRECTS, RedirectServers
except ImportError:
    from tests._redirect_servers import ALL_REDIRECTS, RedirectServers

try:
    import _escapes
except ImportError:
    from tests import _escapes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(tmp: str) -> TraceStore:
    return TraceStore(tmp)


def _add_session(
    store: TraceStore,
    session_id: str,
    events: list[TraceEvent],
    started_at: float | None = None,
) -> None:
    ts = started_at or time.time()
    meta = SessionMeta(session_id=session_id, started_at=ts, ended_at=ts + 60)
    store.create_session(meta)
    for ev in events:
        store.append_event(session_id, ev)


def _tool(name: str, ts: float = 0.0) -> TraceEvent:
    return TraceEvent(event_type=EventType.TOOL_CALL, timestamp=ts, data={"tool_name": name})


def _error(ts: float = 0.0) -> TraceEvent:
    return TraceEvent(event_type=EventType.ERROR, timestamp=ts, data={"message": "exit 1"})


def _write(path: str, ts: float = 0.0) -> TraceEvent:
    return TraceEvent(event_type=EventType.FILE_WRITE, timestamp=ts, data={"path": path})


# ---------------------------------------------------------------------------
# Heuristic pattern extraction
# ---------------------------------------------------------------------------

class TestExtractPatternsHeuristic(unittest.TestCase):
    def test_no_events_no_patterns(self):
        patterns = _extract_patterns_heuristic({"s1": []})
        self.assertEqual(patterns, [])

    def test_blind_retry_detected(self):
        events = [_tool("Bash"), _tool("Bash"), _tool("Bash")]
        patterns = _extract_patterns_heuristic({"s1": events})
        names = [p.name for p in patterns]
        self.assertIn("blind-retry", names)

    def test_blind_retry_not_triggered_for_two_calls(self):
        # Two consecutive calls of the same tool should NOT trigger (need 3+)
        events = [_tool("Bash"), _tool("Bash")]
        patterns = _extract_patterns_heuristic({"s1": events})
        names = [p.name for p in patterns]
        self.assertNotIn("blind-retry", names)

    def test_error_no_change_detected(self):
        events = [_tool("Bash"), _error(), _tool("Bash")]
        patterns = _extract_patterns_heuristic({"s1": events})
        names = [p.name for p in patterns]
        self.assertIn("error-no-change", names)

    def test_error_no_change_cleared_by_write(self):
        events = [_tool("Bash"), _error(), _write("src/a.py"), _tool("Bash")]
        patterns = _extract_patterns_heuristic({"s1": events})
        names = [p.name for p in patterns]
        self.assertNotIn("error-no-change", names)

    def test_wide_blast_radius_detected(self):
        events = [_write(f"src/file{i}.py") for i in range(9)]
        patterns = _extract_patterns_heuristic({"s1": events})
        names = [p.name for p in patterns]
        self.assertIn("wide-blast-radius", names)

    def test_wide_blast_radius_not_triggered_below_threshold(self):
        events = [_write(f"src/file{i}.py") for i in range(5)]
        patterns = _extract_patterns_heuristic({"s1": events})
        names = [p.name for p in patterns]
        self.assertNotIn("wide-blast-radius", names)

    def test_multiple_sessions_aggregated(self):
        s1 = [_tool("Bash"), _tool("Bash"), _tool("Bash")]
        s2 = [_tool("Read"), _tool("Read"), _tool("Read")]
        patterns = _extract_patterns_heuristic({"s1": s1, "s2": s2})
        blind = next((p for p in patterns if p.name == "blind-retry"), None)
        self.assertIsNotNone(blind)
        self.assertIn("s1", blind.session_ids)
        self.assertIn("s2", blind.session_ids)

    def test_proposed_instruction_is_nonempty(self):
        events = [_tool("Bash"), _tool("Bash"), _tool("Bash")]
        patterns = _extract_patterns_heuristic({"s1": events})
        for p in patterns:
            self.assertTrue(p.proposed_instruction.strip())


# ---------------------------------------------------------------------------
# LLM response parsing
# ---------------------------------------------------------------------------

class TestParseLlmPatterns(unittest.TestCase):
    def _sessions(self) -> dict[str, list[TraceEvent]]:
        return {"abcdef12": [_tool("Bash")]}

    def test_valid_json_parsed(self):
        response = json.dumps({
            "patterns": [
                {
                    "name": "blind-retry",
                    "description": "Agent retried without reading stderr",
                    "affected_sessions": ["abcdef12"],
                    "proposed_addition": "## Retry policy\nRead stderr before retrying.",
                }
            ]
        })
        patterns = _parse_llm_patterns(response, self._sessions())
        self.assertEqual(len(patterns), 1)
        self.assertEqual(patterns[0].name, "blind-retry")
        self.assertIn("Retry policy", patterns[0].proposed_instruction)

    def test_markdown_fences_stripped(self):
        response = "```json\n" + json.dumps({
            "patterns": [{"name": "x", "description": "d",
                          "affected_sessions": [], "proposed_addition": "## X\nY"}]
        }) + "\n```"
        patterns = _parse_llm_patterns(response, self._sessions())
        self.assertEqual(len(patterns), 1)

    def test_invalid_json_returns_empty(self):
        patterns = _parse_llm_patterns("not json at all", self._sessions())
        self.assertEqual(patterns, [])

    def test_empty_proposed_addition_skipped(self):
        response = json.dumps({
            "patterns": [
                {"name": "x", "description": "d",
                 "affected_sessions": [], "proposed_addition": ""}
            ]
        })
        patterns = _parse_llm_patterns(response, self._sessions())
        self.assertEqual(patterns, [])

    def test_missing_patterns_key_returns_empty(self):
        patterns = _parse_llm_patterns(json.dumps({}), self._sessions())
        self.assertEqual(patterns, [])


# ---------------------------------------------------------------------------
# OptimizeProposal
# ---------------------------------------------------------------------------

class TestOptimizeProposal(unittest.TestCase):
    def _proposal(self, existing: str = "") -> OptimizeProposal:
        patterns = [
            FailurePattern(
                name="blind-retry",
                description="Retried without reading stderr",
                session_ids=["s1"],
                example_events=["s1: Bash called 3x"],
                proposed_instruction="## Retry policy\nRead stderr before retrying.",
            )
        ]
        return OptimizeProposal(
            target_file="AGENTS.md",
            patterns=patterns,
            existing_content=existing,
        )

    def test_diff_text_contains_plus_lines(self):
        diff = self._proposal().diff_text()
        self.assertIn("+ ## Retry policy", diff)
        self.assertIn("+ Read stderr", diff)

    def test_diff_text_contains_cluster_header(self):
        diff = self._proposal().diff_text()
        self.assertIn("blind-retry", diff)

    def test_apply_appends_to_existing(self):
        result = self._proposal("# Existing\n\nSome content.").apply()
        self.assertIn("# Existing", result)
        self.assertIn("## Retry policy", result)

    def test_apply_empty_existing(self):
        result = self._proposal("").apply()
        self.assertIn("## Retry policy", result)

    def test_apply_multiple_patterns(self):
        p2 = FailurePattern(
            name="scope",
            description="Wide blast radius",
            session_ids=["s2"],
            example_events=[],
            proposed_instruction="## Scope\nLimit writes.",
        )
        proposal = OptimizeProposal(
            target_file="AGENTS.md",
            patterns=[self._proposal().patterns[0], p2],
            existing_content="# Existing\n",
        )
        result = proposal.apply()
        self.assertIn("## Retry policy", result)
        self.assertIn("## Scope", result)


# ---------------------------------------------------------------------------
# print_proposal
# ---------------------------------------------------------------------------

class TestPrintProposal(unittest.TestCase):
    def test_no_patterns_message(self):
        proposal = OptimizeProposal(
            target_file="AGENTS.md", patterns=[], existing_content=""
        )
        buf = io.StringIO()
        print_proposal(proposal, out=buf)
        self.assertIn("No failure patterns", buf.getvalue())

    def test_patterns_shown(self):
        patterns = [
            FailurePattern(
                name="blind-retry",
                description="Retried without reading stderr",
                session_ids=["s1", "s2"],
                example_events=["s1: Bash x3"],
                proposed_instruction="## Retry policy\nRead stderr.",
            )
        ]
        proposal = OptimizeProposal(
            target_file="AGENTS.md", patterns=patterns, existing_content=""
        )
        buf = io.StringIO()
        print_proposal(proposal, out=buf)
        output = buf.getvalue()
        self.assertIn("blind-retry", output)
        self.assertIn("Retry policy", output)
        self.assertIn("2 session", output)


# ---------------------------------------------------------------------------
# run_optimize (integration)
# ---------------------------------------------------------------------------

class TestRunOptimize(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = _make_store(self.tmp)

    def test_no_sessions_returns_empty_patterns(self):
        proposal = run_optimize(self.store, [], target_file="AGENTS.md")
        self.assertEqual(proposal.patterns, [])

    def test_blind_retry_session_produces_proposal(self):
        events = [_tool("Bash"), _tool("Bash"), _tool("Bash")]
        _add_session(self.store, "s1", events)
        proposal = run_optimize(self.store, ["s1"], target_file="AGENTS.md")
        names = [p.name for p in proposal.patterns]
        self.assertIn("blind-retry", names)

    def test_target_file_content_read(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, dir=self.tmp
        ) as f:
            f.write("# Existing\n\nSome content.\n")
            target = f.name
        events = [_tool("Bash"), _tool("Bash"), _tool("Bash")]
        _add_session(self.store, "s2", events)
        proposal = run_optimize(self.store, ["s2"], target_file=target)
        self.assertIn("# Existing", proposal.existing_content)

    def test_apply_writes_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, dir=self.tmp
        ) as f:
            f.write("# Existing\n")
            target = f.name
        events = [_tool("Bash"), _tool("Bash"), _tool("Bash")]
        _add_session(self.store, "s3", events)
        proposal = run_optimize(self.store, ["s3"], target_file=target)
        new_content = proposal.apply()
        Path(target).write_text(new_content)
        written = Path(target).read_text()
        self.assertIn("# Existing", written)
        self.assertIn("Retry policy", written)

    def test_missing_session_skipped_gracefully(self):
        proposal = run_optimize(
            self.store, ["nonexistent_session_id"], target_file="AGENTS.md"
        )
        # Should not raise; patterns may be empty
        self.assertIsInstance(proposal.patterns, list)


class TestCallLlmRefusesRedirects(unittest.TestCase):
    """urlopen re-sent the POST as a GET with no body and the key, and the
    Location host's reply became the LLM's answer."""

    def test_redirect_returns_none(self):
        servers = RedirectServers(self)
        self.assertEqual(_call_llm("prompt", servers.b.url, "model", "llm-key"), "from B")
        for status in ALL_REDIRECTS:
            with self.subTest(status=status):
                servers.redirect(status)
                self.assertIsNone(_call_llm("prompt", servers.url, "model", "llm-key"))
                self.assertEqual(servers.a.methods(), ["POST"])
                self.assertEqual(servers.b.received(), [])


def _proposal(name="plain", description="plain", addition="plain"):
    pattern = FailurePattern(name=name, description=description, session_ids=["s1"],
                             example_events=[], proposed_instruction=addition)
    return OptimizeProposal(target_file="AGENTS.md", patterns=[pattern], existing_content="")


def _render(proposal):
    buf = io.StringIO()
    print_proposal(proposal, out=buf)
    return buf.getvalue()


def _added_lines(output):
    return [line for line in output.split("\n") if line.startswith("  + ")]


# (label, kind, build): build(value) returns a proposal with value in the
# place the label names. The kinds are in tests/_escapes.py. The proposed
# addition keeps its lines, so it has its own tests.
PRINT_PROPOSAL_CASES = [
    ("cluster name", "text", lambda v: _proposal(name=v)),
    ("description", "free", lambda v: _proposal(description=v)),
]


class TestPrintProposalShowsEscapesAsText(unittest.TestCase):
    """optimize sends tool names and error text from traces to the LLM, so a
    trace can steer its reply. The preview shows escape codes in the cluster
    name, the description and the proposed addition as text."""

    def test_each_value_shows_escapes_as_text(self):
        _escapes.check_each_value(self, PRINT_PROPOSAL_CASES, _render)

    def test_plain_text_prints_unchanged(self):
        _escapes.check_plain_text_unchanged(self, PRINT_PROPOSAL_CASES, _render)
        self.assertIn("  + " + _escapes.PLAIN + "\n", _render(_proposal(addition=_escapes.PLAIN)))

    def test_whitespace_in_free_text_shows_as_one_space(self):
        _escapes.check_whitespace_in_free_text(self, PRINT_PROPOSAL_CASES, _render)

    def test_sgr_code_shows_as_text(self):
        sgr = "a\x1b[8mhidden"
        output = _render(_proposal(name=sgr, description=sgr, addition=sgr))
        self.assertEqual(output.count(r"a\x1b[8mhidden"), 3)
        self.assertNotIn("\x1b", output)

    def test_proposed_addition_line_keeps_its_indentation(self):
        output = _render(_proposal(addition="  A\x1b]52;c;aWQK\x07B\x1b[2K\x9bC\u202eD\tE"))
        self.assertEqual(_added_lines(output), [r"  +   A\x1b]52;c;aWQK\x07B\x1b[2K\x9bC\u202eD\tE"])

    def test_proposed_addition_keeps_its_lines(self):
        # PAYLOAD has a lone CR before C. --apply writes that CR inside the
        # line, so the preview shows it there as an escape.
        output = _render(_proposal(addition=_escapes.PAYLOAD))
        self.assertEqual(_added_lines(output), [
            r"  + A\x1b]52;c;aWQK\x07B\x1b[2K\rC",
            r"  + D\x9bE\u202eF",
        ])
        for line in output.split("\n"):
            self.assertTrue(line.isprintable(), ascii(line))

    def test_proposed_addition_splits_at_lf_only(self):
        for ch in "\r\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029":
            with self.subTest(ch=ascii(ch)):
                output = _render(_proposal(addition=f"A{ch}B\nC"))
                self.assertEqual(_added_lines(output), ["  + A" + repr(ch)[1:-1] + "B", "  + C"])

    def test_final_lf_adds_no_empty_line(self):
        output = _render(_proposal(addition="A\n\nB\n"))
        self.assertEqual(_added_lines(output), ["  + A", "  + ", "  + B"])

    def test_closing_rule_is_one_line(self):
        output = _render(_proposal())
        rules = [line for line in output.split("\n") if line and set(line) == {"─"}]
        self.assertEqual(rules, ["─" * 60] * 2)
        self.assertTrue(output.endswith(
            "\n" + "─" * 60 + "\nApply? Run with --apply to write changes to AGENTS.md\n\n"
        ), output)

    def test_crlf_reply_gives_lf_lines(self):
        reply = json.dumps({"patterns": [{"proposed_addition": "## Rule\r\nline one\r\nline two\r\n"}]})
        proposal = OptimizeProposal(target_file="AGENTS.md", existing_content="",
                                    patterns=_parse_llm_patterns(reply, {"s1": []}))
        self.assertEqual(_added_lines(_render(proposal)), ["  + ## Rule", "  + line one", "  + line two"])

    def test_apply_keeps_the_raw_text(self):
        self.assertEqual(_proposal(addition=_escapes.PAYLOAD).apply(), _escapes.PAYLOAD + "\n")


# ---------------------------------------------------------------------------
# A malformed LLM reply falls back to the heuristics
# ---------------------------------------------------------------------------

SRC = os.path.dirname(os.path.dirname(os.path.abspath(agent_trace.__file__)))
LLM_ENV = ("OPENAI_BASE_URL", "AGENT_STRACE_LLM_URL", "OPENAI_API_KEY",
           "AGENT_STRACE_LLM_KEY", "AGENT_STRACE_LLM_MODEL")
STORE_ENV = ("AGENT_TRACE_DIR", "AGENT_STRACE_WORKSPACE", "AGENT_STRACE_TENANT_ID")
EXISTING = b"# Existing\n"
# Python 3.14 limits JSON nesting by stack size, not by a count, so an 8 MB
# stack parses 100000 levels. A million levels raise RecursionError on 3.10+.
TOO_DEEP = 1_000_000
REFUSAL = (
    "Error: the addition in Cluster {} has a control or format character,"
    " so nothing was written. Run without --apply to see the escapes.\n"
)


def _http(body: bytes) -> bytes:
    head = f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n"
    return head.encode() + body


def _chat(content) -> bytes:
    return _http(json.dumps({"choices": [{"message": {"content": content}}]}).encode())


def _patterns(*additions: str) -> bytes:
    patterns = [{"name": "p", "description": "d", "affected_sessions": [], "proposed_addition": a}
                for a in additions]
    return _chat(json.dumps({"patterns": patterns}))


class _LlmHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_POST(self):
        self.server.bodies.append(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        if self.server.reply is not None:
            self.wfile.write(self.server.reply)


class _LlmServer(ThreadingHTTPServer):
    """Answers each request with the raw bytes in ``reply``. With a reply of
    None it resets the connection instead. ``bodies`` holds the body of each
    POST."""

    daemon_threads = True
    block_on_close = False

    def __init__(self, reply: bytes | None):
        super().__init__(("127.0.0.1", 0), _LlmHandler)
        self.reply = reply
        self.bodies: list[bytes] = []
        self.url = f"http://127.0.0.1:{self.server_address[1]}"

    def handle_error(self, request, client_address):
        pass

    def shutdown_request(self, request):
        if self.reply is not None:
            super().shutdown_request(request)
            return
        # A linger time of 0 makes close send RST, not FIN. The client then
        # gets ConnectionResetError, an OSError that is no HTTPException.
        linger = struct.pack("HH" if os.name == "nt" else "ii", 1, 0)
        request.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, linger)
        self.close_request(request)


def _serve(test: unittest.TestCase, reply: bytes | None) -> str:
    """Start an LLM server for the test and return its base URL."""
    return _start(test, reply).url


def _start(test: unittest.TestCase, reply: bytes | None) -> _LlmServer:
    """Start an LLM server for the test and return it."""
    env = mock.patch.dict(os.environ, {"no_proxy": "*"})
    env.start()
    test.addCleanup(env.stop)
    server = _LlmServer(reply)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    test.addCleanup(thread.join)
    test.addCleanup(server.server_close)
    test.addCleanup(server.shutdown)
    return server


class TestCallLlmReturnsNoneOnABadReply(unittest.TestCase):
    """_call_llm caught only URLError, KeyError, JSONDecodeError and
    IndexError. A reset, a bad status line, a body that is not UTF-8 or JSON
    of another shape crashed optimize with a traceback."""

    def call(self, reply):
        return _call_llm("prompt", _serve(self, reply), "model", "llm-key")

    def test_failed_request_returns_none(self):
        for label, reply in [("reset", None), ("bad status line", b"garbage\r\n\r\n")]:
            with self.subTest(label):
                self.assertIsNone(self.call(reply))

    def test_body_that_cannot_be_decoded_returns_none(self):
        cases = [
            ("not JSON", b"not json"),
            ("not UTF-8", b'{"choices": "\xff"}'),
            ("nested too deep", b"[" * TOO_DEEP + b"]" * TOO_DEEP),
        ]
        for label, body in cases:
            with self.subTest(label):
                self.assertIsNone(self.call(_http(body)))

    def test_json_of_another_shape_returns_none(self):
        for body in [b"[1]", b'{"choices": "x"}', b"{}", b'{"choices": []}']:
            with self.subTest(body=body):
                self.assertIsNone(self.call(_http(body)))

    def test_content_that_is_not_text_returns_none(self):
        for content in [5, None, ["x"]]:
            with self.subTest(content=content):
                self.assertIsNone(self.call(_chat(content)))
        self.assertEqual(self.call(_chat("ok")), "ok")


class TestParseLlmPatternsChecksEachValue(unittest.TestCase):
    """_parse_llm_patterns called .get on the reply and on each pattern, and
    str() on each value. A reply of another shape crashed optimize with a
    traceback, and a name of 5 became "5"."""

    SESSIONS = {"5aaaaaaa": [], "bbbbbbbb": []}
    GOOD = {"name": "p", "description": "d", "affected_sessions": [], "proposed_addition": "## Rule"}

    def parse(self, reply):
        return _parse_llm_patterns(json.dumps(reply), self.SESSIONS)

    def parse_pattern(self, **fields):
        return self.parse({"patterns": [{**self.GOOD, **fields}]})

    def test_reply_that_is_not_an_object_gives_nothing(self):
        for reply in [[1, 2], "x", 5, None]:
            with self.subTest(reply=reply):
                self.assertEqual(self.parse(reply), [])

    def test_patterns_that_is_not_a_list_gives_nothing(self):
        for value in [5, None, "x", {"a": 1}]:
            with self.subTest(patterns=value):
                self.assertEqual(self.parse({"patterns": value}), [])

    def test_pattern_that_is_not_an_object_is_skipped(self):
        patterns = self.parse({"patterns": ["x", 5, None, [1], self.GOOD]})
        self.assertEqual([p.proposed_instruction for p in patterns], ["## Rule"])

    def test_addition_that_is_not_text_is_skipped(self):
        for value in [5, None, [], ["## Rule"], {"a": "b"}, ""]:
            with self.subTest(addition=value):
                self.assertEqual(self.parse_pattern(proposed_addition=value), [])
        no_addition = {k: v for k, v in self.GOOD.items() if k != "proposed_addition"}
        self.assertEqual(self.parse({"patterns": [no_addition]}), [])

    def test_affected_sessions_that_is_not_a_list_counts_as_empty(self):
        for value in [5, "bbbb", None, {"bbbb": 1}]:
            with self.subTest(affected_sessions=value):
                [pattern] = self.parse_pattern(affected_sessions=value)
                self.assertEqual(pattern.session_ids, ["5aaaaaaa", "bbbbbbbb"])

    def test_affected_session_that_is_not_text_is_skipped(self):
        [pattern] = self.parse_pattern(affected_sessions=[5, None, ["5a"], "bbbb"])
        self.assertEqual(pattern.session_ids, ["bbbbbbbb"])

    def test_name_and_description_that_are_not_text_count_as_missing(self):
        [pattern] = self.parse_pattern(name=5, description=["x"])
        self.assertEqual((pattern.name, pattern.description), ("unknown", ""))
        [pattern] = self.parse({"patterns": [{"proposed_addition": "## Rule"}]})
        self.assertEqual((pattern.name, pattern.description), ("unknown", ""))

    def test_reply_that_json_cannot_decode_gives_nothing(self):
        cases = [
            ("nested too deep", "[" * TOO_DEEP + "]" * TOO_DEEP),
            ("number too long", '{"patterns": [' + "1" * 5000 + "]}"),
        ]
        for label, text in cases:
            with self.subTest(label):
                self.assertEqual(_parse_llm_patterns(text, self.SESSIONS), [])

    def test_crlf_in_addition_becomes_lf(self):
        [pattern] = self.parse_pattern(proposed_addition="## Rule\r\nA\rB")
        self.assertEqual(pattern.proposed_instruction, "## Rule\nA\rB")


# ---------------------------------------------------------------------------
# cmd_optimize
# ---------------------------------------------------------------------------

class _OptimizeCommandCase(unittest.TestCase):
    """A trace whose 3 Bash calls make the heuristics propose a retry
    policy, and a target that holds EXISTING."""

    def setUp(self):
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for name in LLM_ENV + STORE_ENV:
            os.environ.pop(name, None)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.traces = str(self.root / "traces")
        _add_session(TraceStore(self.traces), "aaaaaaaaaaaaaaaa",
                     [_tool("Bash"), _tool("Bash"), _tool("Bash")])
        self.target = self.root / "AGENTS.md"
        self.target.write_bytes(EXISTING)

    def run_apply(self, url=None, target=None):
        """Run optimize --apply in this process. Return (exit code, stdout, stderr).

        Only --apply runs here: print_proposal binds sys.stdout when it is
        imported, so a preview would not be captured.
        """
        args = argparse.Namespace(
            trace_dir=self.traces, session_id=None, dataset=None,
            target=str(target or self.target), dry_run=False, apply=True,
            base_url=url, model="m", api_key="k" if url else None,
        )
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cmd_optimize(args)
        return code, out.getvalue(), err.getvalue()

    def child_env(self, **extra):
        env = {
            name: value for name, value in os.environ.items()
            if not name.startswith(("AGENT_STRACE_", "OPENAI_")) and name != "AGENT_TRACE_DIR"
        }
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [SRC, os.environ.get("PYTHONPATH")]))
        env.update(HOME=str(self.root), no_proxy="*", PYTHONIOENCODING="utf-8", **extra)
        return without_telemetry(env)

    def run_cli(self, *args, env=None):
        """Run agent-strace optimize in a child process. Return (exit code, stdout, stderr)."""
        result = subprocess.run(
            [sys.executable, "-m", "agent_trace.entry", "--trace-dir", self.traces,
             "optimize", "--target", str(self.target), *args],
            capture_output=True, encoding="utf-8", env=env or self.child_env(),
            cwd=str(self.root), timeout=120,
        )
        return result.returncode, result.stdout, result.stderr


class TestModelComesFromFlagThenEnvironment(_OptimizeCommandCase):
    """--model defaulted to gpt-4o-mini, so AGENT_STRACE_LLM_MODEL was never read."""

    def models_sent(self, *args, **env):
        """Run optimize --dry-run against an LLM server. Return the model of each POST."""
        server = _start(self, _patterns())
        env = self.child_env(AGENT_STRACE_LLM_URL=server.url, AGENT_STRACE_LLM_KEY="k", **env)
        code, _, err = self.run_cli("--dry-run", *args, env=env)
        self.assertEqual((code, err), (0, ""))
        return [json.loads(body)["model"] for body in server.bodies]

    def test_environment_model_is_sent_without_the_flag(self):
        self.assertEqual(self.models_sent(AGENT_STRACE_LLM_MODEL="llama3"), ["llama3"])

    def test_flag_overrides_the_environment(self):
        self.assertEqual(
            self.models_sent("--model", "qwen2", AGENT_STRACE_LLM_MODEL="llama3"), ["qwen2"])

    def test_unset_or_empty_gives_the_default(self):
        for label, env in [("unset", {}), ("empty", {"AGENT_STRACE_LLM_MODEL": ""})]:
            with self.subTest(label):
                self.assertEqual(self.models_sent(**env), ["gpt-4o-mini"])


class TestDryRunNeverWrites(_OptimizeCommandCase):
    """--dry-run --apply printed the preview and then wrote the target."""

    def test_dry_run_with_apply_prints_the_preview_and_writes_nothing(self):
        dry_run = self.run_cli("--dry-run")
        self.assertEqual((dry_run[0], dry_run[2]), (0, ""))
        self.assertIn("Cluster 1: blind-retry", dry_run[1])
        self.assertEqual(self.run_cli("--dry-run", "--apply"), dry_run)
        self.assertEqual(self.target.read_bytes(), EXISTING)

    def test_apply_alone_prints_only_the_written_line(self):
        self.assertEqual(self.run_cli("--apply"), (0, f"Written 1 addition(s) to {self.target}\n", ""))
        self.assertIn(b"## Retry policy", self.target.read_bytes())


class TestMalformedLlmReplyFallsBack(_OptimizeCommandCase):
    """Each of these replies crashed optimize with a traceback. Now it uses
    the heuristics, as when no LLM is configured."""

    REPLIES = [
        ("pattern is not an object", _chat(json.dumps({"patterns": ["x"]}))),
        ("reply is a list", _chat(json.dumps([1, 2]))),
        ("content is a number", _chat(5)),
        ("choices is a string", _http(b'{"choices": "x"}')),
        ("body is a list", _http(b"[1]")),
    ]

    def test_bad_reply_falls_back_to_the_heuristics(self):
        for label, reply in self.REPLIES:
            with self.subTest(label):
                self.target.write_bytes(EXISTING)
                written = f"Written 1 addition(s) to {self.target}\n"
                self.assertEqual(self.run_apply(_serve(self, reply)), (0, written, ""))
                self.assertIn(b"## Retry policy", self.target.read_bytes())


class TestApplyRefusesControlCharacters(_OptimizeCommandCase):
    """--apply shows no preview, and it wrote control, format and bidi
    characters from the LLM reply into the target raw."""

    def test_unprintable_character_is_refused(self):
        for ch in ["\x1b", "\x07", "\u202e", "\u2028", "\xa0", "\r"]:
            with self.subTest(ch=ascii(ch)):
                url = _serve(self, _patterns(f"## Rule\nok {ch} here"))
                self.assertEqual(self.run_apply(url), (1, "", REFUSAL.format(1)))
                self.assertEqual(self.target.read_bytes(), EXISTING)

    def test_refusal_names_the_cluster(self):
        url = _serve(self, _patterns("## Fine", "## Rule\nok \x1b[2J here"))
        self.assertEqual(self.run_apply(url), (1, "", REFUSAL.format(2)))
        self.assertEqual(self.target.read_bytes(), EXISTING)


class TestApplyWritesTheTarget(_OptimizeCommandCase):
    def test_tab_and_arrow_are_written(self):
        url = _serve(self, _patterns("## Rule\n\tuse → here"))
        self.assertEqual(self.run_apply(url)[0], 0)
        self.assertEqual(self.target.read_bytes(),
                         "# Existing\n\n---\n\n## Rule\n\tuse → here\n".encode("utf-8"))

    def test_crlf_reply_is_written_with_lf(self):
        url = _serve(self, _patterns("## Rule\r\nline one\r\nline two"))
        self.assertEqual(self.run_apply(url)[0], 0)
        self.assertEqual(self.target.read_bytes(), b"# Existing\n\n---\n\n## Rule\nline one\nline two\n")

    def test_symlinked_target_is_written_through_the_link_and_keeps_its_mode(self):
        real = self.root / "real.md"
        real.write_bytes(EXISTING)
        real.chmod(0o640)
        link = self.root / "link.md"
        try:
            link.symlink_to(real)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"cannot create a symlink: {exc}")
        self.assertEqual(self.run_apply(target=link)[0], 0)
        self.assertTrue(link.is_symlink())
        self.assertTrue(real.read_bytes().startswith(b"# Existing\n\n---\n\n## Retry policy\n"))
        self.assertEqual(stat.S_IMODE(real.stat().st_mode), 0o640)

    def test_crlf_target_keeps_crlf(self):
        self.target.write_bytes(b"# Existing\r\n")
        self.assertEqual(self.run_apply()[0], 0)
        data = self.target.read_bytes()
        self.assertIn(b"# Existing\r\n\r\n---\r\n\r\n## Retry policy\r\n", data)
        self.assertEqual(data.count(b"\n"), data.count(b"\r\n"))


class TestTargetReadAndWriteErrors(_OptimizeCommandCase):
    """optimize read the target in the locale encoding, and a target it
    could not decode, read or write crashed it with a traceback."""

    def test_target_that_is_not_utf8_is_refused(self):
        self.target.write_bytes(b"# Caf\xe9\n")
        message = f"Error: {self.target} is not valid UTF-8. Save it as UTF-8 and run again.\n"
        for args in [(), ("--apply",)]:
            with self.subTest(args=args):
                self.assertEqual(self.run_cli(*args), (1, "", message))
                self.assertEqual(self.target.read_bytes(), b"# Caf\xe9\n")

    def test_directory_target_is_refused(self):
        folder = self.root / "folder"
        folder.mkdir()
        code, out, err = self.run_apply(target=folder)
        self.assertEqual((code, out), (1, ""))
        self.assertRegex(err, rf"\AError: cannot read {re.escape(str(folder))}: [^\n]+\n\Z")

    def test_read_only_target_is_left_unchanged(self):
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            self.skipTest("root can write a read-only file")
        self.target.chmod(0o444)
        self.addCleanup(self.target.chmod, 0o644)
        code, out, err = self.run_apply()
        self.assertEqual((code, out), (1, ""))
        self.assertRegex(err, rf"\AError: cannot write {re.escape(str(self.target))}: [^\n]+\n\Z")
        self.assertEqual(self.target.read_bytes(), EXISTING)

    def test_target_under_a_file_is_refused(self):
        parent = self.root / "file.txt"
        parent.write_bytes(b"x")
        target = parent / "AGENTS.md"
        code, out, err = self.run_apply(target=target)
        self.assertEqual((code, out), (1, ""))
        self.assertRegex(err, rf"\AError: cannot write {re.escape(str(target))}: [^\n]+\n\Z")
        self.assertEqual(parent.read_bytes(), b"x")


class TestApplyUnderNonUtf8Locale(_OptimizeCommandCase):
    """Under Latin-1, write_text raised UnicodeEncodeError on an addition
    with → after it had emptied the target. Linux with PYTHONUTF8=0 and a
    missing locale uses ASCII, and Windows uses cp1252. Neither has →."""

    def test_target_is_read_and_written_as_utf8(self):
        env = self.child_env(PYTHONUTF8="0", LC_ALL="en_US.ISO8859-1")
        encoding = subprocess.run(
            [sys.executable, "-c",
             "import codecs, locale; print(codecs.lookup(locale.getpreferredencoding(False)).name)"],
            capture_output=True, encoding="utf-8", env=env, timeout=60, check=True,
        ).stdout.strip()
        if encoding == "utf-8":
            self.skipTest("the locale encoding is UTF-8 here")
        self.target.write_bytes("# Existing café\n".encode("utf-8"))
        url = _serve(self, _patterns("## Rule\nuse → here"))
        code, out, err = self.run_cli("--apply", "--base-url", url, "--api-key", "k", env=env)
        self.assertEqual((code, err), (0, ""), f"locale encoding {encoding}")
        self.assertEqual(self.target.read_bytes(),
                         "# Existing café\n\n---\n\n## Rule\nuse → here\n".encode("utf-8"))


if __name__ == "__main__":
    unittest.main()
