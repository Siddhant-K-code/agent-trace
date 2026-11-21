"""Top-level event fields with the wrong type.

Four top-level fields of an event have a fixed type: duration_ms is a number,
event_id and parent_id are strings, and redacted is a boolean. The collector
refuses any other type, but a hand edit, an older store or another tool can
still write one to events.ndjson.

Commands read such a value as missing. A duration_ms that is not a usable
number gives the output of an event without one. An event_id or parent_id that
is not a non-empty string gives the output of a stored "", so nothing links to
it. Only a stored true counts as redacted. No command crashes, and no NaN or
Infinity from these fields reaches the JSON that a command builds.

test_wrong_typed_event_fields covers the fields inside ``data``.
"""

from __future__ import annotations

import io
import json
import re
import tempfile
import unittest
from pathlib import Path

try:  # keeps product telemetry off in this process and its children
    import _no_telemetry
except ImportError:  # pragma: no cover - run as a package module
    from tests import _no_telemetry

try:
    from test_meta_text_fields import strict_json
    from test_non_object_event_data import CliTestCase, _tree_snapshot, _write_session
except ImportError:  # pragma: no cover - run as a package module
    from tests.test_meta_text_fields import strict_json
    from tests.test_non_object_event_data import CliTestCase, _tree_snapshot, _write_session

from agent_trace import (
    a2a,
    compliance,
    correlation,
    disclosure,
    evidence_health,
    evidence_packet,
    failures,
    langfuse_export,
    mcp_server,
    otlp,
    pairing,
    policy_backtest,
    replay,
    review,
    review_surface,
    runner_summary,
    subagent,
    timeline,
    why,
)
from agent_trace.models import SessionMeta, TraceEvent
from agent_trace.store import TraceStore

SID = "f1e2d3c4b5a60001"
SID2 = "f1e2d3c4b5a60002"
START = 1767225600.0  # 2026-01-01T00:00:00Z
META = SessionMeta(session_id=SID, started_at=START, ended_at=START + 8, agent_name="claude")
NAN = float("nan")
# _with leaves a field set to MISSING out of the stored event.
MISSING = object()

# Positions in _records().
REQUEST, RESPONSE, BASH_CALL, BASH_RESULT, EDIT_CALL, EDIT_RESULT, ERROR = 2, 3, 4, 5, 6, 7, 8
TIMED = (RESPONSE, BASH_RESULT, EDIT_RESULT)

# Stored durations that are not a usable number of milliseconds.
BAD_DURATIONS = {
    "numeric text": "1000",
    "object": {"ms": 1000},
    "list": [1000],
    "boolean": True,
    "negative": -5,
    "nan": NAN,
    "infinity": float("inf"),
    "too large for a float": 10**400,
}
# Stored IDs that are not a non-empty string.
BAD_IDS = {
    "null": None,
    "number": 5,
    "boolean": True,
    "object": {"id": "c1"},
    "list": ["c1"],
    "nan": NAN,
}
# Stored redacted values other than true.
NOT_TRUE = {
    "text": "yes",
    "text true": "true",
    "number": 1,
    "null": None,
    "object": {},
}
SPAN_ID = re.compile(r"[0-9a-f]{16}")


def _records(session_id: str = SID) -> list[dict]:
    """Stored events of one session: an LLM call, two tool calls and an error."""

    def event(offset, event_type, event_id, data, **fields):
        return {
            "event_type": event_type,
            "timestamp": START + offset,
            "event_id": event_id,
            "session_id": session_id,
            "data": data,
            **fields,
        }

    model = "claude-sonnet-4-5"
    usage = {"input_tokens": 4000, "output_tokens": 20}
    return [
        event(0, "session_start", "e01", {"provider": "claude"}),
        event(1, "user_prompt", "e02", {"prompt": "Fix the failing test"}),
        event(2, "llm_request", "r1", {"model": model}),
        event(2.25, "llm_response", "e04", {"model": model, "usage": usage},
              parent_id="r1", duration_ms=250),
        event(3, "tool_call", "c1", {"tool_name": "Bash", "arguments": {"command": "cat src/a.py"}}),
        event(4, "tool_result", "e06", {"tool_name": "Bash", "result": "print('hi')", "exit_code": 0},
              parent_id="c1", duration_ms=1000),
        event(5, "tool_call", "c2", {"tool_name": "Edit", "arguments": {"file_path": "src/a.py"}}),
        event(5.5, "tool_result", "e08", {"tool_name": "Edit", "result": "ok"},
              parent_id="c2", duration_ms=500),
        event(6, "error", "e09", {"message": "pytest failed"}),
        event(7, "assistant_response", "e10", {"text": "Fixed it."}),
        event(8, "session_end", "e11", {"reason": "completed"}),
    ]


def _with(field: str, value, *positions: int) -> list[dict]:
    """Return _records() with ``field`` stored as ``value`` at ``positions``.

    No positions means every event.
    """
    records = _records()
    for position in positions or range(len(records)):
        records[position].pop(field, None)
        if value is not MISSING:
            records[position][field] = value
    return records


def _linked(field_value, call: int = BASH_CALL, response: int = BASH_RESULT) -> list[dict]:
    """Return _records() with one value as the call's event_id and the response's parent_id."""
    records = _records()
    records[call]["event_id"] = field_value
    records[response]["parent_id"] = field_value
    return records


def _nan_fields(records: list[dict]) -> list[dict]:
    """Store NaN as each event's event_id, parent_id and duration_ms."""
    for record in records:
        record.update(event_id=NAN, parent_id=NAN, duration_ms=NAN)
    return records


def _empty_fields(records: list[dict]) -> list[dict]:
    """Store "" as each event's event_id and parent_id, and drop duration_ms."""
    for record in records:
        record.update(event_id="", parent_id="")
        record.pop("duration_ms", None)
    return records


def _events(records: list[dict]) -> list[TraceEvent]:
    """Read records as the store does: from JSON text."""
    return [TraceEvent.from_json(json.dumps(record)) for record in records]


def _spans(payload: dict) -> list[dict]:
    return [
        span
        for resource in payload["resourceSpans"]
        for scope in resource["scopeSpans"]
        for span in scope["spans"]
    ]


def _exports(records: list[dict]) -> tuple[str, str]:
    """Return the otlp and otlp-genai exports of ``records`` as sorted JSON text."""
    return tuple(
        json.dumps(export(META, _events(records)), sort_keys=True)
        for export in (otlp.session_to_otlp, otlp.session_to_otlp_genai)
    )


def _script_json(page: str, pattern: str):
    """Strictly parse the JSON that ``pattern`` captures from one line of an HTML page."""
    match = re.search(pattern, page, re.MULTILINE)
    assert match is not None, pattern
    return strict_json(match.group(1))


def _values(data, key: str) -> list:
    """Return every value stored under ``key`` anywhere in ``data``."""
    if isinstance(data, dict):
        found = [data[key]] if key in data else []
        return found + [value for item in data.values() for value in _values(item, key)]
    if isinstance(data, list):
        return [value for item in data for value in _values(item, key)]
    return []


class StoreTestCase(unittest.TestCase):
    def tmpdir(self) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return Path(tmp.name)

    def store(self, records: list[dict], other: list[dict] | None = None, **meta) -> TraceStore:
        """Write ``records`` as session SID, and ``other`` as SID2, to a new store."""
        traces = self.tmpdir() / ".agent-traces"
        _write_session(traces, SID, START, records, **meta)
        if other is not None:
            _write_session(traces, SID2, START, other, **meta)
        return TraceStore(str(traces), read_only=True)


class TestDuration(StoreTestCase):
    """A duration_ms that is not a usable number reads as missing."""

    def test_otlp_exports_use_the_fallback_for_a_missing_duration(self):
        missing = _exports(_with("duration_ms", MISSING, *TIMED))
        for name, value in BAD_DURATIONS.items():
            with self.subTest(value=name):
                self.assertEqual(_exports(_with("duration_ms", value, *TIMED)), missing)
        spans = _spans(otlp.session_to_otlp(META, _events(_records())))
        bash = next(span for span in spans if span["name"] == "tool/Bash")
        self.assertEqual(int(bash["endTimeUnixNano"]) - int(bash["startTimeUnixNano"]), 10**9)

    def test_replay_prints_no_duration(self):
        def lines(value):
            events = _events(_with("duration_ms", value, *TIMED))
            return [replay.format_event(event, base_ts=START) for event in events]

        missing = lines(MISSING)
        for name, value in BAD_DURATIONS.items():
            with self.subTest(value=name):
                self.assertEqual(lines(value), missing)
        self.assertIn(" (1.00s)", replay.format_event(_events(_records())[BASH_RESULT]))

    def test_timeline_prints_no_duration(self):
        def render(value):
            result = timeline.build_timeline(self.store(_with("duration_ms", value, *TIMED)), SID)
            text, data = io.StringIO(), io.StringIO()
            timeline.format_timeline(result, out=text)
            timeline.format_timeline_json(result, out=data)
            return text.getvalue(), strict_json(data.getvalue())

        missing = render(MISSING)
        for name, value in BAD_DURATIONS.items():
            with self.subTest(value=name):
                self.assertEqual(render(value), missing)


class TestIds(StoreTestCase):
    """An event_id or parent_id that is not a non-empty string reads as missing."""

    def test_tool_pairing_links_nothing(self):
        def pair(value):
            call, result = _linked(value)[BASH_CALL:BASH_RESULT + 1]
            found = pairing.pair_tool_events(_events([call, result]))
            return found.pairs, found.unpaired_responses, found.unmatched_requests

        self.assertEqual(pair("c1"), ({1: 0}, [], []))
        for name, value in BAD_IDS.items():
            with self.subTest(value=name):
                self.assertEqual(pair(value), ({}, [1], [0]))

    def test_otlp_span_ids_stay_valid_and_unique(self):
        # Each list of IDs is stored in turn across the events of the session.
        variants = {name: [value] for name, value in BAD_IDS.items()}
        variants["empty"] = [""]
        variants["mixed"] = [*BAD_IDS.values(), ""]
        for name, ids in variants.items():
            records = _records()
            for position, record in enumerate(records):
                record["event_id"] = ids[position % len(ids)]
            for export in (otlp.session_to_otlp, otlp.session_to_otlp_genai):
                with self.subTest(ids=name, export=export.__name__):
                    spans = _spans(export(META, _events(records)))
                    span_ids = [span["spanId"] for span in spans]
                    self.assertTrue(all(SPAN_ID.fullmatch(span_id) for span_id in span_ids), span_ids)
                    self.assertEqual(len(set(span_ids)), len(span_ids), span_ids)
                    parents = {span["parentSpanId"] for span in spans if span.get("parentSpanId")}
                    self.assertLessEqual(parents, set(span_ids))

    def test_a2a_call_span_ids_stay_valid_and_unique(self):
        def a2a_call(offset, event_id):
            return {
                "event_type": "tool_call",
                "timestamp": START + offset,
                "event_id": event_id,
                "session_id": SID,
                "data": {
                    "event_subtype": a2a.A2A_CALL, "agent_id": "helper",
                    "agent_url": "http://agents.example/a2a/helper", "task": "summarize the diff",
                    "response": {}, "sub_session_id": "", "duration_ms": 1500.0,
                    "cost_usd": 0.0, "success": True, "error": "",
                },
            }

        records = _records()
        records = [records[0], a2a_call(1, None), a2a_call(2, ""), a2a_call(3, NAN), records[-1]]
        tree = a2a.build_a2a_tree(self.store(records, total_tokens=0), SID)
        spans = a2a.a2a_calls_to_otlp_spans(tree)
        span_ids = [span["spanId"] for span in spans]
        self.assertEqual(sum(span["name"].startswith("a2a_call:") for span in spans), 3)
        self.assertTrue(all(SPAN_ID.fullmatch(span_id) for span_id in span_ids), span_ids)
        self.assertEqual(len(set(span_ids)), len(span_ids), span_ids)

    def test_health_and_review_report_a_bad_parent_as_no_parent(self):
        def health(records):
            return evidence_health.assess_evidence_health(_events(records)).to_dict()

        def projection(records):
            return review.build_review_projection(_events(records)).to_dict()

        for field, position in (("parent_id", BASH_RESULT), ("event_id", BASH_CALL)):
            expected = health(_with(field, "", position)), projection(_with(field, "", position))
            codes = [reason["code"] for reason in expected[0]["reasons"]]
            self.assertIn("orphan_tool_result", codes)
            for name, value in BAD_IDS.items():
                with self.subTest(field=field, value=name):
                    records = _with(field, value, position)
                    self.assertEqual((health(records), projection(records)), expected)

    def test_correlation_errors_and_why_link_nothing(self):
        def correlate(records):
            return correlation.correlate_tool_events(_events(records))

        def chain(records):
            found = why.build_causal_chain(_events(records), BASH_RESULT)
            return found.target_index, [(link.event_index, link.reason) for link in found.links]

        def errors(records):
            return failures.session_errors(_events(records))

        def error_link(value):
            return _linked(value, call=REQUEST, response=ERROR)

        expected = correlate(_linked("")), chain(_linked("")), errors(error_link(""))
        for name, value in BAD_IDS.items():
            with self.subTest(value=name):
                found = correlate(_linked(value)), chain(_linked(value)), errors(error_link(value))
                self.assertEqual(found, expected)

    def test_evidence_packet_links_nothing(self):
        def packet(value):
            return evidence_packet.build_evidence_packet(self.store(_linked(value)), SID).events

        expected = packet("")
        # The packet digest refuses NaN anywhere in a session, as share does.
        for name, value in BAD_IDS.items():
            if name == "nan":
                continue
            with self.subTest(value=name):
                self.assertEqual(packet(value), expected)

    def test_session_tree_reads_a_bad_id_as_missing(self):
        def tree(value):
            node = subagent.build_tree(self.store(_with("event_id", value)), SID)
            out = io.StringIO()
            subagent.format_tree(node, out=out)
            return out.getvalue()

        expected = tree("")
        for name, value in BAD_IDS.items():
            with self.subTest(value=name):
                self.assertEqual(tree(value), expected)


class TestBuiltJson(StoreTestCase):
    """JSON that a command builds gets no NaN or Infinity from these fields."""

    def test_mcp_event(self):
        event = _events(_nan_fields(_records()))[BASH_RESULT]
        data = strict_json(json.dumps(mcp_server._event_to_dict(event)))
        self.assertEqual((data["event_id"], data["parent_id"], data["duration_ms"]), ("", "", None))

    def test_session_review(self):
        def session_review(records):
            built = review_surface.build_session_review(self.store(records), SID)
            return strict_json(json.dumps(built))

        self.assertEqual(
            session_review(_nan_fields(_records())), session_review(_empty_fields(_records())),
        )

    def test_eu_ai_act_export(self):
        store = self.store(_nan_fields(_records()))
        # Edit one stored event so the hash chain breaks at the next one.
        events_path = Path(store.base_dir) / SID / "events.ndjson"
        events_path.write_text(
            events_path.read_text(encoding="utf-8").replace("cat src/a.py", "cat src/b.py"),
            encoding="utf-8",
        )
        data = strict_json(json.dumps(compliance.export_eu_ai_act(store, [SID])))
        broken = _values(data, "broken_event_id")
        self.assertTrue(broken)
        self.assertEqual(set(broken), {""})

    def test_langfuse_observation_ids(self):
        for name, value in {**BAD_IDS, "empty": ""}.items():
            with self.subTest(value=name):
                events = _events(_with("event_id", value))
                observations = langfuse_export._events_to_langfuse_observations(SID, events)
                ids = [observation["id"] for observation in strict_json(json.dumps(observations))]
                self.assertEqual(ids, [None, None, None])

    def test_policy_coverage(self):
        policy = self.tmpdir() / "policy.json"
        policy.write_text(json.dumps({"commands": {"allow": ["pytest *"]}}), encoding="utf-8")
        store = self.store(_nan_fields(_records()))
        coverage = policy_backtest.policy_coverage(store, str(policy), days=None)
        data = strict_json(json.dumps(coverage.to_dict(show_uncovered=True)))
        self.assertEqual([call["event_id"] for call in data["uncovered_calls"]], ["", ""])

    def test_replay_page(self):
        page = replay.replay_to_html(self.store(_nan_fields(_records())), SID)
        events = _script_json(page, r"^const EVENTS = (.*);$")
        self.assertEqual({(event["event_id"], event["duration_ms"]) for event in events}, {("", None)})

    def test_replay_diff_page(self):
        other = _nan_fields(_records(SID2))
        other[BASH_CALL]["data"]["arguments"]["command"] = "cat src/b.py"
        page = replay.replay_to_html_diff(self.store(_nan_fields(_records()), other), SID, SID2)
        for name in ("eventsA", "eventsB"):
            events = _script_json(page, rf"^const {name} = (.*);$")
            self.assertEqual(
                {(event["event_id"], event["duration_ms"]) for event in events}, {("", None)},
            )
        # An event without a usable ID is never highlighted.
        for name in ("onlyA", "onlyB"):
            self.assertEqual(_script_json(page, rf"^const {name} = new Set\((.*)\);$"), [])


class TestRedacted(StoreTestCase):
    """Only a stored true counts as redacted."""

    def test_otlp_exports_count_only_true(self):
        missing = _exports(_with("redacted", MISSING))
        for name, value in NOT_TRUE.items():
            with self.subTest(value=name):
                self.assertEqual(_exports(_with("redacted", value)), missing)
        payload = otlp.session_to_otlp_genai(META, _events(_with("redacted", True, BASH_RESULT)))
        root = _spans(payload)[0]
        attributes = {attribute["key"]: attribute["value"] for attribute in root["attributes"]}
        self.assertEqual(attributes["agent.strace.redacted_event_count"], {"intValue": "1"})

    def test_disclosure_review_and_runner_summary_read_other_values_as_false(self):
        for name, value in NOT_TRUE.items():
            with self.subTest(value=name):
                events = _events(_with("redacted", value))
                self.assertEqual(disclosure.count_redacted_events(events), 0)
                self.assertEqual(runner_summary.output_text(events[BASH_RESULT]), "print('hi')")
                items = review.build_review_projection(events).to_dict()["items"]
                self.assertEqual(
                    {(item["redacted"], "privacy" in item["categories"]) for item in items},
                    {(False, False)},
                )


class TestCommands(CliTestCase):
    """Commands run in-process give the output of a session without the bad fields."""

    def workspace(self, records: list[dict]) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name).resolve()
        (root / "home").mkdir()
        traces = root / "work" / ".agent-traces"
        _write_session(traces, SID, START, records)
        (traces / ".gitignore").write_text("*\n", encoding="utf-8")
        return root

    def output(self, records: list[dict], *argv: str) -> tuple[int, str, str]:
        root = self.workspace(records)
        before = _tree_snapshot(root)
        result = self.run_cli(root, *argv)
        self.assertEqual(_tree_snapshot(root), before)
        return result.code, result.out, result.err

    def test_stats_reads_a_bad_field_as_missing(self):
        cases = (
            ("duration_ms", TIMED, MISSING, BAD_DURATIONS),
            ("event_id", (BASH_CALL,), "", BAD_IDS),
            ("parent_id", (BASH_RESULT,), "", BAD_IDS),
        )
        for field, positions, empty, values in cases:
            expected = self.output(_with(field, empty, *positions), "stats", SID)
            for name, value in values.items():
                with self.subTest(field=field, value=name):
                    found = self.output(_with(field, value, *positions), "stats", SID)
                    self.assertEqual(found, expected)
        self.assertIn("avg: 1000ms", self.output(_records(), "stats", SID)[1])

    def test_read_only_commands(self):
        commands = (
            ("replay", SID), ("tree", SID), ("explain", SID), ("timeline", SID), ("cost", SID),
            ("stats",), ("health", SID), ("review", SID), ("why", SID, "2"),
            ("pr-comment", "--dry-run"),
        )
        wrong = _records()
        for record in wrong:
            record.update(event_id={"id": "x"}, parent_id=5, duration_ms="1000", redacted="yes")
        empty = _empty_fields(_records())
        for argv in commands:
            with self.subTest(command=argv[0]):
                found = self.output(wrong, *argv)
                self.assertNotIn("Traceback", found[2])
                self.assertEqual(found, self.output(empty, *argv))


if __name__ == "__main__":
    unittest.main()
