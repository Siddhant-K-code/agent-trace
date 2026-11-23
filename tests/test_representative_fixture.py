import csv
import io
import json
import re
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

try:  # keeps product telemetry off in this process and its children
    import _no_telemetry  # noqa: F401
except ImportError:  # pragma: no cover - run as a package module
    from tests import _no_telemetry  # noqa: F401

import agent_trace
from agent_trace.annotate import Annotation
from agent_trace.anonymize import cmd_anonymize_export
from agent_trace.cli import build_parser, cmd_export
from agent_trace.health import assess_session, load_capture_matrix
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.otlp import _to_span_id
from agent_trace.store import TraceStore


FIXTURE_DIR = (
    Path(agent_trace.__file__).parent
    / "corpus"
    / "v1"
    / "fixture-review-001"
)
REDACTION_MARKER = "[REDACTED synthetic request]"


class TestRepresentativeEvidenceFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.events = [
            TraceEvent.from_json(line)
            for line in (FIXTURE_DIR / "events.ndjson").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        cls.annotations = [
            Annotation.from_json(line)
            for line in (FIXTURE_DIR / "annotations.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        cls.expected = json.loads((FIXTURE_DIR / "expected.json").read_text(encoding="utf-8"))

    def test_fixture_uses_one_session_and_expected_count(self):
        self.assertEqual(len(self.events), self.expected["expected_event_count"])
        self.assertEqual({event.session_id for event in self.events}, {self.expected["session_id"]})

    def test_fixture_metadata_matches_session_start(self):
        starts = [event for event in self.events if event.event_type == EventType.SESSION_START]
        self.assertEqual(len(starts), 1)
        self.assertEqual(starts[0].data["provider"], self.expected["provider"])
        self.assertEqual(starts[0].data["fixture_version"], self.expected["fixture_version"])
        self.assertEqual(starts[0].session_id, self.expected["session_id"])

    def test_failed_call_has_one_terminal_outcome(self):
        by_id = {event.event_id: event for event in self.events}
        failure = by_id[self.expected["expected_failure_event_id"]]
        children = [
            child for child in self.events
            if child.parent_id == failure.parent_id
            and child.event_type in (EventType.TOOL_RESULT, EventType.ERROR)
        ]
        self.assertEqual([child.event_id for child in children], [failure.event_id])

    def test_fixture_covers_review_workflow(self):
        kinds = {event.event_type for event in self.events}
        for required in (
            EventType.SESSION_START,
            EventType.SESSION_END,
            EventType.USER_PROMPT,
            EventType.ASSISTANT_RESPONSE,
            EventType.FILE_READ,
            EventType.FILE_WRITE,
            EventType.TOOL_CALL,
            EventType.TOOL_RESULT,
            EventType.ERROR,
            EventType.DECISION,
        ):
            self.assertIn(required, kinds)

    def test_fixture_contains_failure_retry_and_recovery(self):
        by_id = {event.event_id: event for event in self.events}
        failure = by_id[self.expected["expected_failure_event_id"]]
        retry = by_id[self.expected["expected_retry_call_id"]]
        recovery = by_id[self.expected["expected_recovery_result_id"]]
        self.assertEqual(failure.event_type, EventType.ERROR)
        self.assertEqual(retry.data["retry_of"], failure.parent_id)
        self.assertEqual(recovery.parent_id, retry.event_id)
        self.assertEqual(recovery.data["exit_code"], 0)

    def test_fixture_has_explicit_redaction_and_human_correction(self):
        self.assertTrue(any(event.redacted for event in self.events))
        self.assertEqual(len(self.annotations), 1)
        annotation = self.annotations[0]
        self.assertEqual(annotation.annotation_id, self.expected["expected_annotation_id"])
        self.assertIn(annotation.event_id, {event.event_id for event in self.events})
        self.assertTrue(annotation.note.startswith("Human correction:"))

    def test_fixture_has_one_deliberate_orphan_gap(self):
        by_id = {event.event_id: event for event in self.events}
        gap = self.expected["intentional_gap"]
        orphan = by_id[gap["event_id"]]
        self.assertEqual(orphan.parent_id, gap["parent_id"])
        self.assertNotIn(orphan.parent_id, by_id)
        self.assertTrue(orphan.data["fixture_gap"])

    def test_walkthrough_states_the_fixture_version(self):
        doc = (
            Path(__file__).resolve().parents[1] / "docs" / "representative-fixture.md"
        ).read_text(encoding="utf-8")
        stated = re.findall(r"This walkthrough uses `fixture_version` (\d+)\.", doc)
        self.assertEqual(stated, [str(self.expected["fixture_version"])])


class TestRepresentativeFixtureHealth(unittest.TestCase):
    def test_health_is_partial_with_only_the_deliberate_gap(self):
        expected = json.loads((FIXTURE_DIR / "expected.json").read_text(encoding="utf-8"))
        meta = SessionMeta.from_json((FIXTURE_DIR / "meta.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            store = TraceStore(tmp, redact=False)
            store.create_session(meta)
            for line in (FIXTURE_DIR / "events.ndjson").read_text(encoding="utf-8").splitlines():
                if line.strip():
                    store.append_event(meta.session_id, TraceEvent.from_json(line))
            report = assess_session(
                store,
                meta.session_id,
                store.load_meta(meta.session_id),
                store.load_events(meta.session_id),
                load_capture_matrix(),
            )

        health = expected["expected_health"]
        self.assertEqual(report.result.status.value, health["status"])
        self.assertEqual(report.exit_code, health["exit_code"])
        # A synthetic session names no hook provider, so health claims none.
        self.assertEqual(report.provider.provider, health["provider"])
        self.assertEqual(
            [
                {"code": reason.code, "kind": reason.kind.value, "event_id": reason.event_id}
                for reason in report.result.reasons
            ],
            health["reasons"],
        )
        self.assertEqual(
            [reason["event_id"] for reason in health["reasons"]],
            [expected["intentional_gap"]["event_id"]],
        )


def _otlp_attributes(item: dict) -> dict:
    return {
        attribute["key"]: next(iter(attribute["value"].values()))
        for attribute in item.get("attributes", [])
    }


class TestRepresentativeFixtureExports(unittest.TestCase):
    """Each export that applies to this fixture keeps the facts it exists for.

    temporal needs a Temporal workflow context and --scores needs a backend,
    so neither applies to this session.
    """

    EVENT_FORMATS = ("json", "ndjson", "csv", "anonymize")
    ALL_FORMATS = EVENT_FORMATS + ("eu-ai-act", "otlp", "otlp-genai")

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.out_dir = Path(tmp.name)
        self.trace_dir = self.out_dir / "traces"
        self.expected = json.loads((FIXTURE_DIR / "expected.json").read_text(encoding="utf-8"))
        self.events = [
            TraceEvent.from_json(line)
            for line in (FIXTURE_DIR / "events.ndjson").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        meta = SessionMeta.from_json((FIXTURE_DIR / "meta.json").read_text(encoding="utf-8"))
        self.session_id = meta.session_id
        # Without write-time redaction the store holds the fixture verbatim,
        # so only the export can keep or lose its redaction.
        store = TraceStore(str(self.trace_dir), redact=False)
        store.create_session(meta)
        for event in self.events:
            store.append_event(self.session_id, event)
        self.fixture_ids = [event.event_id for event in self.events]
        self.failure_id = self.expected["expected_failure_event_id"]
        self.retry_id = self.expected["expected_retry_call_id"]
        self.recovery_id = self.expected["expected_recovery_result_id"]
        self.gap = self.expected["intentional_gap"]
        prompts = [event for event in self.events if event.event_type == EventType.USER_PROMPT]
        self.assertEqual(len(prompts), 1)
        self.prompt = prompts[0]

    def _export(self, fmt: str) -> str:
        output = self.out_dir / f"export-{fmt}.out"
        argv = [
            "--trace-dir", str(self.trace_dir),
            "export", self.session_id,
            "--output", str(output),
        ]
        argv += ["--anonymize"] if fmt == "anonymize" else ["--format", fmt]
        args = build_parser().parse_args(argv)
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            if fmt == "anonymize":
                # cmd_export hands --anonymize to this function, whose
                # out argument is bound to sys.stdout at import.
                status = cmd_anonymize_export(args, out=io.StringIO())
            else:
                status = cmd_export(args)
        self.assertEqual(status, 0, stderr.getvalue())
        # json, ndjson and csv print to stdout; the others write --output.
        if output.exists():
            return output.read_text(encoding="utf-8")
        return stdout.getvalue()

    def _exported_events(self, fmt: str, text: str) -> list[dict]:
        if fmt == "json":
            return json.loads(text)
        if fmt == "ndjson":
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        if fmt == "anonymize":
            return json.loads(text)["events"]
        return [
            {
                "event_id": row["event_id"],
                "event_type": row["event_type"],
                "parent_id": row["parent_id"] or None,
                "data": json.loads(row["data"]),
            }
            for row in csv.DictReader(io.StringIO(text))
        ]

    def _assert_failure_retry_recovery_order(
        self, ids: list[str], failed_call: str,
    ) -> None:
        self.assertLess(ids.index(failed_call), ids.index(self.failure_id))
        self.assertLess(ids.index(self.failure_id), ids.index(self.retry_id))
        self.assertLess(ids.index(self.retry_id), ids.index(self.recovery_id))

    def test_event_exports_keep_order_failure_retry_recovery_and_orphan(self):
        fixture_types = [event.event_type.value for event in self.events]
        for fmt in self.EVENT_FORMATS:
            with self.subTest(format=fmt):
                exported = self._exported_events(fmt, self._export(fmt))
                ids = [event["event_id"] for event in exported]
                types = {event["event_id"]: event["event_type"] for event in exported}
                parents = {event["event_id"]: event.get("parent_id") for event in exported}
                data = {event["event_id"]: event["data"] for event in exported}

                self.assertEqual(len(ids), self.expected["expected_event_count"])
                self.assertEqual(ids, self.fixture_ids)
                self.assertEqual([event["event_type"] for event in exported], fixture_types)

                failed_call = parents[self.failure_id]
                self.assertEqual(types[self.failure_id], "error")
                self.assertEqual(types[failed_call], "tool_call")
                self.assertEqual(data[self.retry_id]["retry_of"], failed_call)
                self.assertEqual(parents[self.recovery_id], self.retry_id)
                self.assertEqual(data[self.recovery_id]["exit_code"], 0)
                self._assert_failure_retry_recovery_order(ids, failed_call)

                self.assertEqual(parents[self.gap["event_id"]], self.gap["parent_id"])
                self.assertNotIn(self.gap["parent_id"], ids)

                self.assertTrue(
                    data[self.prompt.event_id]["prompt"].startswith(REDACTION_MARKER)
                )
                if fmt != "csv":  # csv has no column for the redacted flag
                    self.assertEqual(
                        [event["event_id"] for event in exported if event.get("redacted")],
                        [self.prompt.event_id],
                    )

    def test_eu_ai_act_export_keeps_order_failure_retry_and_recovery(self):
        report = json.loads(self._export("eu-ai-act"))
        log = report["article_12"]["logging_obligations"][0]
        self.assertEqual(log["trace_integrity"]["event_count"], self.expected["expected_event_count"])
        self.assertTrue(log["trace_integrity"]["hash_chain_valid"])

        rows = log["events"]
        ids = [row["event_id"] for row in rows]
        self.assertEqual(ids, self.fixture_ids)
        self.assertEqual([row["seq"] for row in rows], list(range(1, len(rows) + 1)))
        by_id = {row["event_id"]: row for row in rows}

        self.assertEqual(by_id[self.failure_id]["event_type"], "error")
        failed_call = json.loads(by_id[self.retry_id]["input_summary"])["retry_of"]
        self.assertEqual(by_id[failed_call]["event_type"], "tool_call")
        self.assertEqual(json.loads(by_id[self.recovery_id]["output_summary"])["exit_code"], 0)
        self._assert_failure_retry_recovery_order(ids, failed_call)

        documentation = report["article_13"]["transparency_documentation"][0]
        self.assertEqual(documentation["limitations"]["errors_encountered"], 1)
        self.assertEqual(
            [(point["type"], point["timestamp"]) for point in documentation["human_oversight_points"]],
            [("ERROR_RECORDED", by_id[self.failure_id]["timestamp"])],
        )

        # This export has no parent field, so it cannot invent the missing call.
        self.assertTrue(json.loads(by_id[self.gap["event_id"]]["output_summary"])["fixture_gap"])
        self.assertNotIn(self.gap["parent_id"], ids)
        self.assertTrue(
            json.loads(by_id[self.prompt.event_id]["input_summary"])["prompt"]
            .startswith(REDACTION_MARKER)
        )

    def _assert_recovered_root(self, attributes):
        self.assertNotIn("error.type", attributes)
        self.assertEqual(
            (
                attributes["agent.strace.behavior.failed_operation_count"],
                attributes["agent.strace.behavior.recovered_operation_count"],
                attributes["agent.strace.behavior.unresolved_failed_operation_count"],
            ),
            ("1", "1", "0"),
        )
        self.assertEqual(attributes["agent.errors"], "1")

    def test_otlp_export_keeps_failure_retry_recovery_and_unparented_orphan(self):
        payload = json.loads(self._export("otlp"))
        root, *children = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]
        self.assertFalse(root.get("parentSpanId"))
        # The retry recovered the failed operation, so the root is not an error.
        self.assertEqual(root["status"], {"code": 0})
        self._assert_recovered_root(_otlp_attributes(root))
        failed_call = self.events[self.fixture_ids.index(self.failure_id)].parent_id
        failure_message = self.events[self.fixture_ids.index(self.failure_id)].data["error"]

        # One span per call, in recorded order: the failed call, its retry,
        # and the orphaned result, which keeps the session root as parent.
        self.assertEqual(
            [span["spanId"] for span in children],
            [_to_span_id(failed_call), _to_span_id(self.retry_id), _to_span_id(self.gap["event_id"])],
        )
        self.assertEqual(
            sorted(children, key=lambda span: int(span["startTimeUnixNano"])), children,
        )
        self.assertEqual({span["parentSpanId"] for span in children}, {root["spanId"]})
        failed, retry, orphan = children

        self.assertEqual(failed["status"], {"code": 2, "message": failure_message})
        self.assertEqual(
            [event["name"] for event in failed.get("events", [])], ["exception"],
        )
        self.assertEqual(_otlp_attributes(retry)["gen_ai.tool.call.id"], self.retry_id)
        self.assertEqual(retry["status"]["code"], 0)
        self.assertEqual(_otlp_attributes(orphan)["gen_ai.tool.call.id"], self.gap["event_id"])
        self.assertNotIn(
            _to_span_id(self.gap["parent_id"]),
            {span["spanId"] for span in children} | {root["spanId"]},
        )

    def test_otlp_genai_export_keeps_failure_recovery_redaction_and_gap(self):
        payload = json.loads(self._export("otlp-genai"))
        root, *children = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]
        root_attributes = _otlp_attributes(root)
        self.assertEqual(root["status"], {"code": 0})
        self._assert_recovered_root(root_attributes)
        self.assertIs(root_attributes["agent.strace.redacted"], True)
        self.assertEqual(root_attributes["agent.strace.redacted_event_count"], "1")
        self.assertEqual(root_attributes["agent.strace.evidence.status"], "partial")
        self.assertEqual(
            [
                (gap["code"], gap["event_id"])
                for gap in json.loads(root_attributes["agent.strace.evidence.gaps"])
            ],
            [("orphan_tool_result", self.gap["event_id"])],
        )
        messages = {
            _otlp_attributes(event)["agent.strace.event.event_id"]: _otlp_attributes(event)
            for event in root.get("events", [])
            if event["name"].startswith("gen_ai.")
        }
        self.assertIs(messages[self.prompt.event_id]["agent.strace.event.redacted"], True)
        self.assertEqual(
            [event_id for event_id, attributes in messages.items()
             if attributes["agent.strace.event.redacted"]],
            [self.prompt.event_id],
        )

        failed_call = self.events[self.fixture_ids.index(self.failure_id)].parent_id
        spans = {_otlp_attributes(span)["gen_ai.tool.call.id"]: span for span in children}
        self.assertEqual(list(spans), [failed_call, self.retry_id, self.gap["event_id"]])
        self.assertEqual({span["parentSpanId"] for span in children}, {root["spanId"]})

        failed = _otlp_attributes(spans[failed_call])
        self.assertEqual(spans[failed_call]["status"]["code"], 2)
        self.assertEqual(failed["agent.strace.request.event_id"], failed_call)
        self.assertEqual(failed["agent.strace.response.event_id"], self.failure_id)

        retry = _otlp_attributes(spans[self.retry_id])
        self.assertEqual(spans[self.retry_id]["status"]["code"], 0)
        self.assertEqual(retry["agent.strace.request.event_id"], self.retry_id)
        self.assertEqual(retry["agent.strace.response.event_id"], self.recovery_id)

        orphan = _otlp_attributes(spans[self.gap["event_id"]])
        self.assertEqual(orphan["agent.strace.response.event_id"], self.gap["event_id"])
        self.assertEqual(orphan["agent.strace.response.parent_event_id"], self.gap["parent_id"])
        self.assertEqual(
            [key for key in orphan if key.startswith("agent.strace.request.")], [],
        )

    def test_no_export_shows_the_prompt_without_its_redaction_marker(self):
        content = self.prompt.data["prompt"]
        self.assertTrue(content.startswith(REDACTION_MARKER))
        request = content[len(REDACTION_MARKER):].strip()
        self.assertTrue(request)
        for fmt in self.ALL_FORMATS:
            with self.subTest(format=fmt):
                text = self._export(fmt)
                self.assertEqual(text.count(request), text.count(content))
                if fmt in self.EVENT_FORMATS + ("eu-ai-act",):
                    self.assertGreaterEqual(text.count(content), 1)


if __name__ == "__main__":
    unittest.main()
