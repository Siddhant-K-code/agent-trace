"""Tests for issue #629: `eval run` and `eval ci` save eval.json, and its readers.

The commands run as a user runs them, so each test reads back a file the
command wrote rather than one written by hand.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

try:  # keeps product telemetry off in this process and its children
    import _no_telemetry
    from _redirect_servers import post_opener
except ImportError:  # pragma: no cover - package-style test discovery
    from tests import _no_telemetry
    from tests._redirect_servers import post_opener

from agent_trace import __version__
from agent_trace.cli import main
from agent_trace.eval.results import load_stored_evaluations
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
EVALUATION = "gen_ai.evaluation.result"

TWO_SCORERS = (
    "scorers:\n"
    "  - type: no_errors\n"
    "    threshold: 1.0\n"
    "  - type: duration_under\n"
    "    max_seconds: 60\n"
    "    threshold: 0.8\n"
)


def _add_session(
    store: TraceStore,
    session_id: str,
    *,
    errors: int = 0,
    start: float | None = None,
    parent: str = "",
    parent_event_id: str = "",
    agent_call_id: str = "",
) -> None:
    start = time.time() - 600 if start is None else start
    events = [
        TraceEvent(event_type=EventType.USER_PROMPT, timestamp=start,
                   session_id=session_id, data={"prompt": "run tests"}),
        TraceEvent(event_type=EventType.TOOL_CALL, timestamp=start + 1,
                   session_id=session_id,
                   data={"tool_name": "Bash", "arguments": {"command": "pytest"}}),
        TraceEvent(event_type=EventType.TOOL_RESULT, timestamp=start + 2,
                   session_id=session_id, data={"result": "5 passed"}),
    ]
    if agent_call_id:
        events.append(TraceEvent(
            event_type=EventType.TOOL_CALL, timestamp=start + 2.5, session_id=session_id,
            event_id=agent_call_id,
            data={"tool_name": "Agent", "arguments": {"prompt": "subtask"}},
        ))
    for n in range(errors):
        events.append(TraceEvent(event_type=EventType.ERROR, timestamp=start + 3 + n / 10,
                                 session_id=session_id, data={"message": "exit 1"}))
    events.append(TraceEvent(event_type=EventType.SESSION_END, timestamp=start + 5,
                             session_id=session_id, data={"exit_code": 0}))
    meta = SessionMeta(
        session_id=session_id,
        started_at=start,
        ended_at=start + 5,
        total_duration_ms=5000,
        parent_session_id=parent or None,
        parent_event_id=parent_event_id or None,
        depth=1 if parent else 0,
    )
    store.create_session(meta)
    for event in events:
        store.append_event(session_id, event)
    store.update_meta(meta)


def _run(trace_dir: str, *argv: str) -> subprocess.CompletedProcess:
    """Run agent-strace in a child process, as a user runs it."""
    env = _no_telemetry.without_telemetry({
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("AGENT_STRACE_", "OPENAI_", "OTEL_", "LANGFUSE_"))
        and key != "AGENT_TRACE_DIR"
    })
    env["PYTHONPATH"] = os.pathsep.join(item for item in (SRC, env.get("PYTHONPATH", "")) if item)
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, "-m", "agent_trace.entry", "--trace-dir", trace_dir, *argv],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        cwd=trace_dir,
        timeout=120,
    )


def _attrs(attributes: list[dict]) -> dict:
    return {a["key"]: next(iter(a["value"].values())) for a in attributes}


def _spans(payload: dict) -> list[dict]:
    return [
        span
        for rs in payload["resourceSpans"]
        for ss in rs["scopeSpans"]
        for span in ss["spans"]
    ]


def _root(payload: dict) -> dict:
    roots = [s for s in _spans(payload) if not s.get("parentSpanId")]
    assert len(roots) == 1, roots
    return roots[0]


def _evaluations(span: dict) -> list[dict]:
    return [e for e in span.get("events", []) if e["name"] == EVALUATION]


def _without_evaluations(payload: dict) -> dict:
    for span in _spans(payload):
        if "events" in span:
            span["events"] = [e for e in span["events"] if e["name"] != EVALUATION]
    return payload


class _TraceDirCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.trace_dir = str(self.tmp / "traces")
        self.store = TraceStore(self.trace_dir)
        self.config = self._config("two.yaml", TWO_SCORERS)

    def _config(self, name: str, text: str) -> str:
        path = self.tmp / name
        path.write_text(text, encoding="utf-8")
        return str(path)

    def _eval_json(self, session_id: str) -> Path:
        return Path(self.trace_dir) / session_id / "eval.json"

    def _run(self, *argv: str) -> subprocess.CompletedProcess:
        return _run(self.trace_dir, *argv)

    def _eval(self, command: str, session_id: str, *extra: str,
              config: str | None = None) -> subprocess.CompletedProcess:
        return self._run("eval", command, session_id, "--config", config or self.config, *extra)

    def _export(self, session_id: str, fmt: str, *extra: str) -> bytes:
        out = self.tmp / f"export-{time.monotonic_ns()}.json"
        result = self._run("export", session_id, "--format", fmt, "-o", str(out), *extra)
        self.assertEqual(result.returncode, 0, result.stderr)
        return out.read_bytes()


class TestEvalCommandsSaveResults(_TraceDirCase):
    def test_eval_run_writes_every_result(self):
        _add_session(self.store, "aaaa1111")
        before = time.time()
        result = self._eval("run", "aaaa1111")
        after = time.time()

        self.assertEqual(result.returncode, 0, result.stderr)
        path = self._eval_json("aaaa1111")
        self.assertIn("Eval results written to ", result.stderr)
        self.assertTrue(result.stderr.rstrip().endswith("eval.json"), result.stderr)
        data = json.loads(path.read_text(encoding="utf-8"))
        evaluated_at = data.pop("evaluated_at")
        self.assertLessEqual(before, evaluated_at)
        self.assertLessEqual(evaluated_at, after)
        self.assertEqual(data, {
            "version": 1,
            "session_id": "aaaa1111",
            "command": "eval run",
            "agent_strace_version": __version__,
            "passed": True,
            "pass_count": 2,
            "fail_count": 0,
            "weighted_score": 1.0,
            "results": [
                {"scorer": "no_errors", "score": 1.0, "threshold": 1.0, "passed": True,
                 "reason": "no errors"},
                {"scorer": "duration_under", "score": 1.0, "threshold": 0.8, "passed": True,
                 "reason": data["results"][1]["reason"]},
            ],
        })
        self.assertTrue(data["results"][1]["reason"])

    def test_failed_scorer_is_saved_as_a_verdict(self):
        _add_session(self.store, "bbbb2222", errors=2)
        result = self._eval("run", "bbbb2222")

        self.assertEqual(result.returncode, 1, result.stderr)
        data = json.loads(self._eval_json("bbbb2222").read_text(encoding="utf-8"))
        self.assertFalse(data["passed"])
        self.assertEqual((data["pass_count"], data["fail_count"]), (1, 1))
        self.assertEqual(data["results"][0], {
            "scorer": "no_errors", "score": 0.0, "threshold": 1.0, "passed": False,
            "reason": "2 error(s) found",
        })

    def test_rerun_replaces_the_file(self):
        _add_session(self.store, "cccc3333")
        self.assertEqual(self._eval("run", "cccc3333").returncode, 0)
        absent = str(self.tmp / "absent.yaml")
        result = self._eval("ci", "cccc3333", config=absent)

        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(self._eval_json("cccc3333").read_text(encoding="utf-8"))
        self.assertEqual(data["command"], "eval ci")
        self.assertEqual([r["scorer"] for r in data["results"]], ["no_errors"])

    def test_eval_ci_writes_with_save_baseline(self):
        _add_session(self.store, "dddd4444")
        baseline = self.tmp / "baseline.json"
        result = self._eval("ci", "dddd4444", "--save-baseline", str(baseline))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(baseline.is_file())
        data = json.loads(self._eval_json("dddd4444").read_text(encoding="utf-8"))
        self.assertEqual(data["command"], "eval ci")
        self.assertEqual(len(data["results"]), 2)

    def test_json_format_keeps_stdout_json(self):
        _add_session(self.store, "eeee5555")
        result = self._eval("run", "eeee5555", "--format", "json")

        self.assertEqual(result.returncode, 0, result.stderr)
        printed = json.loads(result.stdout)
        saved = json.loads(self._eval_json("eeee5555").read_text(encoding="utf-8"))
        self.assertEqual(printed["session_id"], saved["session_id"])
        self.assertIn("Eval results written to ", result.stderr)

    def test_unknown_session_writes_nothing(self):
        _add_session(self.store, "ffff6666")
        for command in ("run", "ci"):
            with self.subTest(command=command):
                result = self._eval(command, "zzzz9999")
                self.assertEqual(result.returncode, 1)
                self.assertEqual(result.stderr, "Session not found: zzzz9999\n")
                self.assertEqual(list(Path(self.trace_dir).rglob("eval.json")), [])

    def test_scorer_that_cannot_judge_keeps_the_earlier_file(self):
        _add_session(self.store, "abab1212")
        self.assertEqual(self._eval("run", "abab1212").returncode, 0)
        path = self._eval_json("abab1212")
        earlier = path.read_bytes()
        cases = {
            "unknown scorer": (
                "scorers:\n  - type: no_errors\n  - type: nope\n",
                "Eval results not saved: scorer nope could not judge the session: "
                "unknown scorer: nope\n",
            ),
            "llm judge without a key": (
                "scorers:\n  - type: llm_judge\n",
                "Eval results not saved: scorer llm_judge could not judge the session: ",
            ),
            "regex with a bad event type": (
                "scorers:\n  - type: regex\n    pattern: ok\n    event_type: nope\n",
                "Eval results not saved: scorer regex could not judge the session: "
                "unknown event_type: nope\n",
            ),
            "threshold that is not a number": (
                "scorers:\n  - type: no_errors\n    threshold: nan\n",
                "Eval results not saved: scorer no_errors has a score or threshold "
                "that is not a finite number\n",
            ),
        }
        for name, (config, note) in cases.items():
            for command in ("run", "ci"):
                with self.subTest(case=name, command=command):
                    result = self._eval(command, "abab1212",
                                        config=self._config(f"{name}.yaml", config))
                    self.assertIn(note, result.stderr)
                    self.assertNotIn("Eval results written", result.stderr)
                    self.assertEqual(path.read_bytes(), earlier)

    def test_gate_writes_nothing(self):
        _add_session(self.store, "acac1313")
        gate = self._config("gate.yaml", (
            "evals:\n"
            "  - name: no-errors\n"
            "    scorer: error_count\n"
            "    threshold: 0\n"
            "    fail_on: above\n"
        ))
        result = self._eval("gate", "acac1313", config=gate)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self._eval_json("acac1313").exists())


class TestOtlpGenaiEvaluationEvents(_TraceDirCase):
    def test_export_carries_one_event_per_saved_result(self):
        _add_session(self.store, "1111aaaa", errors=1)
        before = self._export("1111aaaa", "otlp-genai")
        self.assertEqual(self._eval("run", "1111aaaa").returncode, 1)
        saved = json.loads(self._eval_json("1111aaaa").read_text(encoding="utf-8"))
        after = json.loads(self._export("1111aaaa", "otlp-genai"))

        root = _root(after)
        self.assertTrue(root["name"].startswith("invoke_agent"), root["name"])
        events = _evaluations(root)
        self.assertEqual([_attrs(e["attributes"]) for e in events], [
            {
                "gen_ai.evaluation.name": "no_errors",
                "gen_ai.evaluation.score.value": 0.0,
                "gen_ai.evaluation.score.label": "fail",
                "gen_ai.evaluation.explanation": "1 error(s) found",
            },
            {
                "gen_ai.evaluation.name": "duration_under",
                "gen_ai.evaluation.score.value": 1.0,
                "gen_ai.evaluation.score.label": "pass",
                "gen_ai.evaluation.explanation": saved["results"][1]["reason"],
            },
        ])
        self.assertEqual({e["timeUnixNano"] for e in events}, {root["endTimeUnixNano"]})
        for span in _spans(after):
            if span is not root:
                self.assertEqual(_evaluations(span), [])
        self.assertEqual(_without_evaluations(after), json.loads(before))

    def test_session_without_eval_json_exports_unchanged(self):
        _add_session(self.store, "2222bbbb")
        before = self._export("2222bbbb", "otlp-genai")
        self.assertNotIn(EVALUATION.encode(), before)
        self.assertEqual(self._eval("run", "2222bbbb").returncode, 0)
        self._eval_json("2222bbbb").unlink()

        self.assertEqual(self._export("2222bbbb", "otlp-genai"), before)

    def test_unreadable_eval_json_adds_nothing(self):
        _add_session(self.store, "3333cccc")
        before = self._export("3333cccc", "otlp-genai")
        for text in ("{not json", "[]", '{"results": {}}', '{"results": [1, null]}'):
            with self.subTest(text=text):
                self._eval_json("3333cccc").write_text(text, encoding="utf-8")
                self.assertEqual(self._export("3333cccc", "otlp-genai"), before)

    def test_result_without_a_stored_verdict_is_left_out(self):
        _add_session(self.store, "4444dddd")
        self._eval_json("4444dddd").write_text(json.dumps({"results": [
            {"scorer": "no_verdict", "score": 0.9, "threshold": 0.5},
            {"scorer": "string_verdict", "score": 0.9, "passed": "true"},
            {"scorer": "string_score", "score": "0.9", "passed": True},
            {"scorer": "bool_score", "score": True, "passed": True},
            {"scorer": "", "score": 1.0, "passed": True},
            {"score": 1.0, "passed": True},
            {"scorer": "kept", "score": 1, "passed": True, "reason": 5},
            {"scorer": "long", "score": 0.25, "passed": False, "reason": "x" * 1500},
        ]}), encoding="utf-8")
        with open(self._eval_json("4444dddd"), "a", encoding="utf-8") as f:
            f.write("\n")
        payload = json.loads(self._export("4444dddd", "otlp-genai"))

        self.assertEqual([_attrs(e["attributes"]) for e in _evaluations(_root(payload))], [
            {
                "gen_ai.evaluation.name": "kept",
                "gen_ai.evaluation.score.value": 1.0,
                "gen_ai.evaluation.score.label": "pass",
            },
            {
                "gen_ai.evaluation.name": "long",
                "gen_ai.evaluation.score.value": 0.25,
                "gen_ai.evaluation.score.label": "fail",
                "gen_ai.evaluation.explanation": "x" * 1000,
            },
        ])

    def test_nan_score_is_left_out(self):
        _add_session(self.store, "5555eeee")
        self._eval_json("5555eeee").write_text(
            '{"results": [{"scorer": "nan", "score": NaN, "passed": true},'
            ' {"scorer": "inf", "score": Infinity, "passed": true}]}',
            encoding="utf-8",
        )
        self.assertEqual(load_stored_evaluations(self.store, "5555eeee"), [])

    def test_classic_otlp_is_unchanged(self):
        _add_session(self.store, "6666ffff")
        before = self._export("6666ffff", "otlp")
        self.assertEqual(self._eval("run", "6666ffff").returncode, 0)

        self.assertEqual(self._export("6666ffff", "otlp"), before)

    def test_include_subagents_puts_each_sessions_results_on_its_root(self):
        _add_session(self.store, "root0001", agent_call_id="evt_agent")
        _add_session(self.store, "child001", errors=1, parent="root0001",
                     parent_event_id="evt_agent")
        absent = str(self.tmp / "absent.yaml")
        self.assertEqual(self._eval("run", "root0001", config=absent).returncode, 0)
        self.assertEqual(self._eval("run", "child001", config=absent).returncode, 1)
        payload = json.loads(self._export("root0001", "otlp-genai", "--include-subagents"))

        labels = {}
        for span in _spans(payload):
            events = _evaluations(span)
            if events:
                self.assertTrue(span["name"].startswith("invoke_agent"), span["name"])
                self.assertEqual({e["timeUnixNano"] for e in events},
                                 {span["endTimeUnixNano"]})
                session = _attrs(span["attributes"])["agent.session_id"]
                labels[session] = [
                    _attrs(e["attributes"])["gen_ai.evaluation.score.label"] for e in events
                ]
        self.assertEqual(labels, {"root0001": ["pass"], "child001": ["fail"]})


class TestReadersUseSavedResults(_TraceDirCase):
    def _main(self, *argv: str) -> tuple[object, str]:
        stderr = io.StringIO()
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("OTEL_", "LANGFUSE_"))}
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(sys, "argv", ["agent-strace", "--trace-dir", self.trace_dir,
                                                *argv]), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(stderr):
            try:
                code = main()
            except SystemExit as exc:
                code = exc.code
        return code, stderr.getvalue()

    def _evaluated(self, session_id: str, *, errors: int = 0, start: float | None = None) -> None:
        _add_session(self.store, session_id, errors=errors, start=start)
        result = self._eval("run", session_id)
        self.assertIn("Eval results written to ", result.stderr)

    @staticmethod
    def _response() -> mock.MagicMock:
        response = mock.MagicMock()
        response.status = 200
        response.__enter__ = lambda s: s
        response.__exit__ = mock.MagicMock(return_value=False)
        return response

    def test_dashboard_trend_shows_each_scorer(self):
        now = time.time()
        self._evaluated("7777aaaa", start=now - 7200)
        self._evaluated("8888bbbb", errors=1, start=now - 3600)
        result = self._run("dashboard", "--trend")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Quality trend (eval pass rate):", result.stdout)
        self.assertRegex(result.stdout, r"no_errors\s+100% → 0%  ↓ -100pp")
        self.assertRegex(result.stdout, r"duration_under\s+100% → 100%  → \+0pp")

    def test_dataset_auto_low_eval_score_selects_the_low_session(self):
        self._evaluated("9999cccc")
        self._evaluated("aaaa0000", errors=1)
        result = self._run("eval", "dataset", "auto", "--filter", "low-eval-score:0.9")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Added 1 session(s)", result.stdout)
        dataset = Path(self.trace_dir) / "datasets" / "default.jsonl"
        rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["session_id"] for row in rows], ["aaaa0000"])

    def test_export_scores_otlp_sends_a_gauge_per_result(self):
        self._evaluated("bbbb0000", errors=1)
        with post_opener("agent_trace.langfuse_export.no_redirect_opener",
                         return_value=self._response()) as open_mock:
            code, stderr = self._main("export", "bbbb0000", "--scores", "--backend", "otlp",
                                      "--otlp-endpoint", "http://127.0.0.1:4318")

        self.assertEqual(code, 0, stderr)
        body = json.loads(open_mock.call_args[0][0].data)
        gauges = {
            _attrs(point["attributes"])["judge"]: point["asDouble"]
            for rm in body["resourceMetrics"]
            for sm in rm["scopeMetrics"]
            for metric in sm["metrics"]
            if metric["name"] == "agent_strace.eval.score"
            for point in metric["gauge"]["dataPoints"]
        }
        self.assertEqual(gauges, {"no_errors": 0.0, "duration_under": 1.0})

    def test_export_scores_langfuse_sends_a_score_per_result(self):
        self._evaluated("cccc0000", errors=1)
        with post_opener("agent_trace.langfuse_export.no_redirect_opener",
                         return_value=self._response()) as open_mock:
            code, stderr = self._main("export", "cccc0000", "--scores", "--backend", "langfuse",
                                      "--langfuse-public-key", "pk",
                                      "--langfuse-secret-key", "sk",
                                      "--langfuse-host", "http://127.0.0.1:3000")

        self.assertEqual(code, 0, stderr)
        batch = json.loads(open_mock.call_args[0][0].data)["batch"]
        scores = {
            item["body"]["name"]: item["body"]["value"]
            for item in batch
            if item["type"] == "score-create"
        }
        self.assertEqual(scores, {"no_errors": 0.0, "duration_under": 1.0})


if __name__ == "__main__":
    unittest.main()
