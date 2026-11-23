"""meta.json IDs, tenant, attribution, redacted flag and times with the wrong type.

meta.json can hold any JSON value. parent_event_id, trace_id, parent_span_id,
trace_flags and tenant_id should be strings, attribution an object, redacted
a boolean, and started_at and ended_at Unix times in seconds. A hand edit, an
older store or another tool can still write another type there.

Commands read such a value as missing. An ID that is not a string gives the
output of a stored "". An attribution that is not an object gives the output
of {}, so a list of pairs does not link a session by branch. Only a stored
true counts as redacted. In tree and replay --expand-subagents, a time that
is not usable gives the output of a stored null, and the replay measures from
the root's first event. A child tenant_id that is not a string blocks
backfill-meta from rebuilding its parent, as an int already does. No command
crashes.

test_meta_text_fields covers agent_name, command and team.
"""

from __future__ import annotations

import functools
import http.client
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:  # keeps product telemetry off in this process and its children
    import _no_telemetry  # noqa: F401
except ImportError:  # pragma: no cover - run as a package module
    from tests import _no_telemetry  # noqa: F401

try:
    from test_backfill import write_metaless
    from test_meta_text_fields import (
        CLOCK_TEXT,
        FIRST,
        MISSING,
        SECOND,
        SHA256,
        START,
        CommandTest,
        Run,
        strict_json,
        without_clock,
    )
except ImportError:  # pragma: no cover - run as a package module
    from tests.test_backfill import write_metaless
    from tests.test_meta_text_fields import (
        CLOCK_TEXT,
        FIRST,
        MISSING,
        SECOND,
        SHA256,
        START,
        CommandTest,
        Run,
        strict_json,
        without_clock,
    )

from agent_trace import otlp, web_dashboard
from agent_trace.cli import build_parser
from agent_trace.models import EventType, TraceEvent
from agent_trace.server import cmd_server
from agent_trace.store import TraceStore

BRANCH = "feat/meta-types"
# write_session gives the parent's one tool_call this event ID.
SPAWN_EVENT = "e1" + FIRST[:6]
TRACE_CONTEXT = {"trace_id": "a" * 32, "parent_span_id": "b" * 16, "trace_flags": "01"}


def not_text(valid: str) -> dict:
    """JSON values that are not strings, by label.

    The list and the object hold ``valid``, so a reader that uses the raw
    value still finds it.
    """
    return {"int": 5, "list": [valid], "object": {"value": valid}, "true": True, "null": None}


# Values of attribution that are not objects. dict() turns "pairs" into an
# object that names the branch.
NOT_OBJECT = {
    "int": 5,
    "list": [BRANCH],
    "true": True,
    "null": None,
    "string": BRANCH,
    "pairs": [["git_branch", BRANCH]],
}
# Values of redacted other than true. All but null are truthy.
NOT_TRUE = {"int": 1, "list": [True], "object": {"value": True}, "string": "yes", "null": None}


def spans_of(payload: dict) -> list:
    return [
        span
        for resource in payload["resourceSpans"]
        for scope in resource["scopeSpans"]
        for span in scope["spans"]
    ]


def root_attributes(payload: dict) -> dict:
    """Return the attributes of the invoke_agent root span by key."""
    root = next(span for span in spans_of(payload) if span["name"].startswith("invoke_agent"))
    return {item["key"]: item["value"] for item in root["attributes"]}


def review_view(text: str):
    # The review names a digest of meta.json, which differs with the value.
    return without_clock(strict_json(SHA256.sub("<sha256>", text)))


def share_view(text: str) -> str:
    return CLOCK_TEXT.sub("<clock>", SHA256.sub("<sha256>", text))


class FieldTest(CommandTest):
    """Runs agent-strace on one store per value of a meta.json field."""

    def outputs_by_label(self, write, runs: list, values: dict) -> dict:
        """Return {label: [view of each run]}, from one store per value.

        write(store, value) fills a store. "{label}" in an argument becomes
        the label. The store's directory shows as <root>.
        """
        self.roots = {}
        for label, value in values.items():
            store = self.new_store()
            write(store, value)
            self.roots[label] = str(store.base_dir.parent)
        results = iter(self.run_cli([
            [self.roots[label], [arg.replace("{label}", label) for arg in run.argv]]
            for label in values
            for run in runs
        ]))
        outputs = {}
        for label, root in self.roots.items():
            paths = sorted({root, os.path.realpath(root)}, key=len, reverse=True)
            outputs[label] = []
            for run in runs:
                code, stdout, stderr = next(results)
                data = None
                if run.file and os.path.exists(os.path.join(root, run.file)):
                    with open(os.path.join(root, run.file), encoding="utf-8") as f:
                        data = f.read()
                texts = [stdout, stderr, data]
                for path in paths:
                    texts = [None if text is None else text.replace(path, "<root>") for text in texts]
                outputs[label].append(self.view(run, (code, *texts)))
        return outputs

    def assert_same_as(self, outputs: dict, baseline: str, labels, codes: list) -> None:
        """The outputs of each label equal those of ``baseline``.

        ``codes`` holds the exit code of each run for ``baseline``, so equal
        crashes do not pass.
        """
        expected = outputs[baseline]
        self.assertEqual([output[0] for output in expected], codes, expected)
        for label in labels:
            with self.subTest(label):
                self.assertEqual(outputs[label], expected)


class TestSubagentParentEventId(FieldTest):
    """A child's parent_event_id names the tool_call that spawned it."""

    RUNS = [
        Run(["replay", FIRST, "--expand-subagents"]),
        Run(["tree", FIRST]),
        Run(["tree", FIRST, "--format", "json"], kind="json"),
    ]
    VALUES = {"empty": "", "text": SPAWN_EVENT, **not_text(SPAWN_EVENT)}

    def write_tree(self, store: TraceStore, value) -> None:
        self.write_session(store, FIRST)
        self.write_session(
            store, SECOND, started_at=START + 0.5, parent_session_id=FIRST, parent_event_id=value,
        )

    def test_value_that_is_not_text_ties_the_child_to_no_event(self):
        outputs = self.outputs_by_label(self.write_tree, self.RUNS, self.VALUES)
        self.assert_same_as(outputs, "empty", not_text(SPAWN_EVENT), [0, 0, 0])
        # The stored ID inlines the child's events under its tool_call.
        self.assertEqual(outputs["text"][0][0], 0)
        self.assertNotEqual(outputs["text"][0], outputs["empty"][0])
        for label, shown in (("text", SPAWN_EVENT), ("empty", "")):
            with self.subTest(label, run="tree --format json"):
                child = outputs[label][2][3]["tree"]["children"][0]
                self.assertEqual(child["parent_event_id"], shown)

    def test_tree_to_otlp_parents_the_child_to_the_parent_root_span(self):
        payloads = {}
        for label, value in self.VALUES.items():
            store = self.new_store()
            self.write_tree(store, value)
            try:
                payloads[label] = without_clock(otlp.tree_to_otlp(store, FIRST))
            except Exception as exc:
                payloads[label] = f"{type(exc).__name__}: {exc}"
        child_root = otlp._to_span_id(f"root-{SECOND}")
        for label, parent in (("text", SPAWN_EVENT), ("empty", f"root-{FIRST}")):
            with self.subTest(label, parent=parent):
                span = next(span for span in spans_of(payloads[label]) if span["spanId"] == child_root)
                self.assertEqual(span["parentSpanId"], otlp._to_span_id(parent))
        for label in not_text(SPAWN_EVENT):
            with self.subTest(label):
                self.assertEqual(payloads[label], payloads["empty"])


class TestSubagentTreeTimes(FieldTest):
    """tree and replay --expand-subagents read started_at and ended_at."""

    RUNS = TestSubagentParentEventId.RUNS
    # Values that are not a usable time. Each must give the output of a stored null.
    NOT_TIME = {"text": "soon", "list": [START], "object": {"value": START}, "true": True}

    def write_tree(self, store: TraceStore, value, sid: str, field: str) -> None:
        """Write FIRST with SECOND under its tool_call, then set one time of ``sid``.

        No total_duration_ms is stored, so tree measures each session from
        its times. FIRST's one event is at START + 0.5.
        """
        self.write_session(store, FIRST, total_duration_ms=MISSING)
        self.write_session(
            store, SECOND, started_at=START + 0.5, total_duration_ms=MISSING,
            parent_session_id=FIRST, parent_event_id=SPAWN_EVENT,
        )
        path = store.base_dir / sid / "meta.json"
        meta = json.loads(path.read_text(encoding="utf-8"))
        meta[field] = value
        path.write_text(json.dumps(meta), encoding="utf-8")

    def write_eventless_root(self, store: TraceStore, value) -> None:
        self.write_tree(store, value, sid=FIRST, field="started_at")
        (store.base_dir / FIRST / "events.ndjson").write_text("", encoding="utf-8")

    def test_time_that_is_not_usable_reads_as_missing(self):
        for sid in (FIRST, SECOND):
            for field in ("started_at", "ended_at"):
                with self.subTest(session=sid, field=field):
                    outputs = self.outputs_by_label(
                        functools.partial(self.write_tree, sid=sid, field=field),
                        self.RUNS,
                        {"null": None, **self.NOT_TIME},
                    )
                    self.assert_same_as(outputs, "null", self.NOT_TIME, [0, 0, 0])

    def test_replay_without_a_usable_start_measures_from_the_first_event(self):
        replay = self.RUNS[:1]
        outputs = self.outputs_by_label(
            functools.partial(self.write_tree, sid=FIRST, field="started_at"),
            replay,
            {"start": START, "first event": START + 0.5, "null": None},
        )
        self.assert_same_as(outputs, "first event", ["null"], [0])
        self.assertEqual(outputs["start"][0][0], 0)
        self.assertNotEqual(outputs["start"], outputs["first event"])
        # With no events the replay shows no time, so any start gives the same output.
        with self.subTest("no events"):
            outputs = self.outputs_by_label(
                self.write_eventless_root, replay, {"start": START, "null": None, **self.NOT_TIME},
            )
            self.assert_same_as(outputs, "start", ["null", *self.NOT_TIME], [0])


class Collector(ThreadingHTTPServer):
    """Records the body of each POST by its path, and answers 200."""

    daemon_threads = True
    block_on_close = False

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), CollectorHandler)
        self.lock = threading.Lock()
        self.bodies: dict[str, str] = {}

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def body(self, path: str):
        with self.lock:
            text = self.bodies.get(path)
        return None if text is None else without_clock(strict_json(text))


class CollectorHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        with self.server.lock:
            self.server.bodies[self.path] = body.decode("utf-8")
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args) -> None:
        pass


class TestTemporalTraceContext(FieldTest):
    """export --format temporal reads trace_id, parent_span_id and trace_flags."""

    def setUp(self) -> None:
        super().setUp()
        os.environ["no_proxy"] = "*"
        self.collector = Collector()
        thread = threading.Thread(target=self.collector.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self.collector.server_close)
        self.addCleanup(self.collector.shutdown)

    def exports(self, field: str, valid: str, codes: list) -> tuple[dict, dict]:
        """Export with one field set per label and the other two valid.

        Returns the outputs and the bodies the collector received, by label.
        """

        def write(store: TraceStore, value) -> None:
            self.write_session(store, FIRST, **{**TRACE_CONTEXT, field: value})

        runs = [
            Run(
                ["export", FIRST, "--format", "temporal", "-o", "temporal.json"],
                kind="json",
                file="temporal.json",
            ),
            Run([
                "export", FIRST, "--format", "temporal",
                "--endpoint", f"{self.collector.url}/{field}/{{label}}",
            ]),
        ]
        wrong = not_text(TRACE_CONTEXT[field])
        values = {"empty": "", "text": valid, **wrong}
        outputs = self.outputs_by_label(write, runs, values)
        self.assert_same_as(outputs, "empty", wrong, codes)
        sent = {label: self.collector.body(f"/{field}/{label}/v1/traces") for label in values}
        for label in wrong:
            with self.subTest(label, sent=True):
                self.assertEqual(sent[label], sent["empty"])
        self.assertEqual([output[0] for output in outputs["text"]], [0, 0], outputs["text"])
        self.assertEqual(sent["text"], outputs["text"][0][3])
        return outputs, sent

    def test_trace_id_that_is_not_text_reads_as_missing(self):
        outputs, sent = self.exports("trace_id", "c" * 32, [1, 1])
        self.assertIsNone(sent["empty"])
        self.assertEqual({span["traceId"] for span in spans_of(sent["text"])}, {"c" * 32})

    def test_parent_span_id_that_is_not_text_reads_as_missing(self):
        outputs, sent = self.exports("parent_span_id", "c" * 16, [1, 1])
        self.assertIsNone(sent["empty"])
        self.assertIn("c" * 16, {span.get("parentSpanId") for span in spans_of(sent["text"])})

    def test_trace_flags_that_are_not_text_default_to_01(self):
        outputs, sent = self.exports("trace_flags", "00", [0, 0])
        self.assertEqual({span["flags"] for span in spans_of(sent["empty"])}, {1})
        self.assertEqual({span["flags"] for span in spans_of(sent["text"])}, {0})


class TestBackfillChildTenant(FieldTest):
    """backfill-meta rebuilds a parent only when each child shares its tenant."""

    RUNS = [
        Run(["backfill-meta"]),
        Run(["backfill-meta", "--format", "json"], kind="json"),
        Run(["backfill-meta", "--apply"]),
    ]

    def write_family(self, store: TraceStore, value) -> None:
        # The parent has no meta.json and untenanted events. A child tenant
        # read as "" would match it.
        write_metaless(store.base_dir, FIRST, [
            TraceEvent(
                EventType.USER_PROMPT, timestamp=START, data={"prompt": "go"},
                event_id="p1" + FIRST[:6], session_id=FIRST, tenant_id="",
            ),
            TraceEvent(
                EventType.TOOL_CALL, timestamp=START + 0.5,
                data={"tool_name": "Bash", "arguments": {"command": "ls"}},
                event_id=SPAWN_EVENT, session_id=FIRST, tenant_id="",
            ),
        ])
        self.write_session(
            store, SECOND, started_at=START + 1, parent_session_id=FIRST, tenant_id=value,
        )

    def test_child_tenant_that_is_not_text_blocks_the_rebuild(self):
        wrong = {"list": ["acme"], "object": {"value": "acme"}, "true": True, "null": None}
        values = {"empty": "", "int": 5, **wrong}
        outputs = self.outputs_by_label(self.write_family, self.RUNS, values)
        self.assert_same_as(outputs, "int", wrong, [0, 0, 0])
        for label in values:
            with self.subTest(label, run="backfill-meta --format json"):
                report = outputs[label][1][3]
                item = next(item for item in report["sessions"] if item["session_id"] == FIRST)
                written = os.path.exists(
                    os.path.join(self.roots[label], ".agent-traces", FIRST, "meta.json")
                )
                if label == "empty":
                    self.assertEqual(item["status"], "would_write")
                    self.assertTrue(written)
                else:
                    self.assertEqual((item["status"], item["reason"]), ("skipped", "cross_tenant_child"))
                    self.assertFalse(written)


@unittest.skipUnless(shutil.which("git"), "needs git")
class TestAttributionBranchLink(FieldTest):
    """pr-comment and review link a session whose attribution names the branch."""

    RUNS = [
        Run(["pr-comment", "--dry-run"]),
        Run(["pr-comment", "--dry-run", "--review"]),
        Run(["review"]),
        Run(["review", "--format", "json"], kind="json"),
    ]

    def setUp(self) -> None:
        super().setUp()
        for name in list(os.environ):
            if name == "CI" or name.startswith(("GITHUB_", "GITLAB_", "CI_", "GIT_")):
                del os.environ[name]
        # The head commit is closer in time to SECOND than to FIRST.
        commit_date = f"{int(START) + 60} +0000"
        os.environ.update({
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
            "GIT_AUTHOR_DATE": commit_date,
            "GIT_COMMITTER_DATE": commit_date,
        })

    @staticmethod
    def view(run: Run, output: tuple) -> tuple:
        code = output[0]
        if run.kind == "json" and code == 1:
            # The review gate fails and the report still prints, so parse it.
            return (code, *CommandTest.view(run, (0, *output[1:]))[1:])
        return CommandTest.view(run, output)

    def write_repo(self, store: TraceStore, value) -> None:
        self.write_session(store, FIRST, attribution=value)
        self.write_session(store, SECOND, started_at=START + 60)
        for args in (["init", "-q", "-b", BRANCH], ["commit", "-q", "--allow-empty", "-m", "change"]):
            subprocess.run(
                ["git", "-c", "commit.gpgsign=false", *args],
                cwd=store.base_dir.parent, check=True, capture_output=True,
            )

    def test_attribution_that_is_not_an_object_names_no_branch(self):
        values = {"empty": {}, "object": {"git_branch": BRANCH}, **NOT_OBJECT}
        outputs = self.outputs_by_label(self.write_repo, self.RUNS, values)
        # The session has no validation evidence, so each review exits 1.
        self.assert_same_as(outputs, "empty", NOT_OBJECT, [0, 1, 1, 1])
        for label, linked in (("object", [FIRST, "branch"]), ("empty", [SECOND, "commit_time"])):
            with self.subTest(label, run="review --format json"):
                sessions = outputs[label][3][3]["sessions"]
                self.assertEqual([[item["session_id"], item["link"]] for item in sessions], [linked])


class TestDashboardReview(FieldTest):
    """The dashboard's review and share routes read attribution and redacted."""

    ROUTES = (f"/api/sessions/{FIRST}/review", f"/api/sessions/{FIRST}/share?mode=minimized")

    def start_server(self) -> tuple[int, TraceStore]:
        storage = tempfile.mkdtemp(dir=self.root)
        args = build_parser().parse_args(
            ["server", "--port", "0", "--storage", storage, "--dashboard"]
        )
        servers = []
        started = threading.Event()

        class RecordingServer(HTTPServer):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                servers.append(self)
                started.set()

        for patcher in (
            mock.patch("agent_trace.server.HTTPServer", RecordingServer),
            mock.patch("sys.stdout", io.StringIO()),
            mock.patch("sys.stderr", io.StringIO()),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        os.environ.pop("AGENT_STRACE_AUTH_KEY", None)
        thread = threading.Thread(target=cmd_server, args=(args,), daemon=True)
        thread.start()
        self.assertTrue(started.wait(5))
        server = servers[0]
        self.addCleanup(thread.join, 5)
        self.addCleanup(server.shutdown)
        return server.server_address[1], TraceStore(storage)

    @staticmethod
    def fetch(port: int, path: str, view) -> tuple:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        try:
            conn.request("GET", path)
            response = conn.getresponse()
            status, body = response.status, response.read().decode("utf-8")
        except (http.client.HTTPException, OSError) as exc:
            return type(exc).__name__, ""
        finally:
            conn.close()
        return status, view(body) if status == 200 else body

    @staticmethod
    def call(function, store: TraceStore, view):
        try:
            return view(function(store, FIRST))
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"

    def responses(self, field: str, values: dict) -> dict:
        """Return, by label, each route through the server and each function."""
        port, store = self.start_server()
        self.write_session(store, FIRST)
        path = store.base_dir / FIRST / "meta.json"
        meta = json.loads(path.read_text(encoding="utf-8"))
        responses = {}
        for label, value in values.items():
            path.write_text(json.dumps({**meta, field: value}), encoding="utf-8")
            responses[label] = [
                self.fetch(port, self.ROUTES[0], review_view),
                self.fetch(port, self.ROUTES[1], share_view),
                self.call(web_dashboard.api_session_review, store, review_view),
                self.call(web_dashboard.api_minimized_review_share, store, share_view),
            ]
        return responses

    def assert_same(self, responses: dict, baseline: str, labels) -> None:
        expected = responses[baseline]
        self.assertEqual([status for status, _ in expected[:2]], [200, 200], expected)
        self.assertEqual(expected[0][1], expected[2])
        self.assertEqual(expected[1][1], expected[3])
        for label in labels:
            for route, got, want in zip(("review", "share", "api_session_review", "share function"),
                                        responses[label], expected):
                with self.subTest(label, route=route):
                    self.assertEqual(got, want)

    def test_attribution_that_is_not_an_object_reads_as_empty(self):
        values = {"empty": {}, "object": {"git_branch": BRANCH}, **NOT_OBJECT}
        responses = self.responses("attribution", values)
        self.assert_same(responses, "empty", NOT_OBJECT)
        for label, shown in (("object", {"git_branch": BRANCH}), ("empty", {})):
            with self.subTest(label):
                self.assertEqual(responses[label][0][1]["session"]["attribution"], shown)

    def test_redacted_other_than_true_reads_as_false(self):
        values = {"false": False, "true": True, **NOT_TRUE}
        responses = self.responses("redacted", values)
        self.assert_same(responses, "false", NOT_TRUE)
        for label, shown in (("true", True), ("false", False)):
            with self.subTest(label):
                self.assertIs(responses[label][0][1]["session"]["redacted_metadata"], shown)


class TestGenAIRootSpan(FieldTest):
    """export --format otlp-genai copies meta.json fields to the root span."""

    def exports(self, sid: str, write, values: dict, baseline: str, wrong) -> dict:
        run = Run(
            ["export", sid, "--format", "otlp-genai", "-o", "genai.json"],
            kind="json",
            file="genai.json",
        )
        outputs = self.outputs_by_label(write, [run], values)
        self.assert_same_as(outputs, baseline, wrong, [0])
        return {label: root_attributes(output[0][3]) for label, output in outputs.items()
                if label not in wrong}

    def test_redacted_other_than_true_exports_false(self):
        values = {"false": False, "true": True, **NOT_TRUE}
        roots = self.exports(
            FIRST, lambda store, value: self.write_session(store, FIRST, redacted=value),
            values, "false", NOT_TRUE,
        )
        self.assertEqual(roots["true"]["agent.strace.redacted"], {"boolValue": True})
        self.assertEqual(roots["false"]["agent.strace.redacted"], {"boolValue": False})

    def test_attribution_that_is_not_an_object_exports_no_provenance(self):
        values = {"empty": {}, "object": {"git_branch": BRANCH}, **NOT_OBJECT}
        roots = self.exports(
            FIRST, lambda store, value: self.write_session(store, FIRST, attribution=value),
            values, "empty", NOT_OBJECT,
        )
        self.assertIn("agent.strace.provenance", roots["object"])
        self.assertNotIn("agent.strace.provenance", roots["empty"])

    def test_parent_event_id_that_is_not_text_is_left_out(self):
        def write(store: TraceStore, value) -> None:
            self.write_session(store, FIRST)
            self.write_session(
                store, SECOND, started_at=START + 0.5, parent_session_id=FIRST, parent_event_id=value,
            )

        wrong = not_text(SPAWN_EVENT)
        roots = self.exports(SECOND, write, {"empty": "", "text": SPAWN_EVENT, **wrong}, "empty", wrong)
        self.assertEqual(roots["text"]["agent.strace.parent_event_id"], {"stringValue": SPAWN_EVENT})
        self.assertNotIn("agent.strace.parent_event_id", roots["empty"])


class TestHealthRedaction(FieldTest):
    """health says the session metadata is redacted only for a stored true."""

    RUNS = [Run(["health", FIRST]), Run(["health", FIRST, "--format", "json"], kind="json")]

    def test_redacted_other_than_true_is_not_redacted_metadata(self):
        values = {"false": False, "true": True, **NOT_TRUE}
        outputs = self.outputs_by_label(
            lambda store, value: self.write_session(store, FIRST, redacted=value), self.RUNS, values,
        )
        self.assert_same_as(outputs, "false", NOT_TRUE, [0, 0])
        for label, shown in (("true", True), ("false", False)):
            with self.subTest(label):
                self.assertIs(outputs[label][1][3]["redaction"]["metadata_redacted"], shown)
                self.assertEqual("plus session metadata" in outputs[label][0][3], shown)


if __name__ == "__main__":
    unittest.main()
