"""Tests for agent-strace optimize."""

import io
import json
import tempfile
import time
import unittest
from pathlib import Path

from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.optimize import (
    FailurePattern,
    OptimizeProposal,
    _call_llm,
    _extract_patterns_heuristic,
    _parse_llm_patterns,
    print_proposal,
    run_optimize,
)
from agent_trace.store import TraceStore

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
        output = _render(_proposal(addition=_escapes.PAYLOAD))
        self.assertEqual(_added_lines(output), [
            r"  + A\x1b]52;c;aWQK\x07B\x1b[2K",
            "  + C",
            r"  + D\x9bE\u202eF",
        ])
        for line in output.split("\n"):
            self.assertTrue(line.isprintable(), ascii(line))

    def test_apply_keeps_the_raw_text(self):
        self.assertEqual(_proposal(addition=_escapes.PAYLOAD).apply(), _escapes.PAYLOAD + "\n")


if __name__ == "__main__":
    unittest.main()
