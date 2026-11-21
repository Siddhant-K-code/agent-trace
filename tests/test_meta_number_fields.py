"""meta.json durations, counts and depths that are not usable numbers.

A hand-edited or foreign meta.json can hold text, an object, null, NaN,
Infinity, a negative number, a boolean or an int too large for a float where
total_duration_ms, tool_calls, llm_requests, errors, total_tokens or depth
should be a number. A count or a depth must also be an int, so 7.5 and 7.0
are not counts. Each reader below treats such a value as missing. It does not
crash, the raw value does not reach the output, and a command gives the same
output as it gives when the field is missing. A command that only reads
writes nothing to meta.json.

baseline update leaves out a session whose duration, tool_calls or
llm_requests holds such a value, and share still refuses NaN and Infinity. A
hook that adds to such a count starts from 0.
"""

from __future__ import annotations

import io
import json
import math
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

try:  # keeps product telemetry off in this process and its children
    import _no_telemetry  # noqa: F401
    from test_copilot_subagent_link import CHILD, PARENT, _SubagentHookTest, local
    from test_meta_text_fields import (
        FIRST,
        MISSING,
        SECOND,
        START,
        CommandTest,
        Run,
        strict_json,
        without_clock,
    )
except ImportError:  # pragma: no cover - run as a package module
    from tests import _no_telemetry  # noqa: F401
    from tests.test_copilot_subagent_link import CHILD, PARENT, _SubagentHookTest, local
    from tests.test_meta_text_fields import (
        FIRST,
        MISSING,
        SECOND,
        START,
        CommandTest,
        Run,
        strict_json,
        without_clock,
    )

from agent_trace import context_score, langfuse_export, web_dashboard
from agent_trace.cli import _parent_depth, build_parser, cmd_export

COUNTS = ("tool_calls", "llm_requests", "errors", "total_tokens", "depth")

# Each command reads FIRST and its child SECOND, or the store as a whole.
RUNS = [
    Run(["list"]),
    Run(["stats", FIRST]),
    Run(["replay", FIRST]),
    Run(["tree", FIRST]),
    Run(["tree", FIRST, "--format", "json"], kind="json"),
    Run(["a2a-tree", FIRST]),
    Run(["a2a-tree", FIRST, "--format", "json"], kind="json"),
    Run(["explain", FIRST]),
    Run(["timeline", FIRST]),
    Run(["cost", FIRST]),
    Run(["postmortem", FIRST]),
    Run(["standup", FIRST]),
    Run(["review", FIRST, "--fail-on", "none"]),
    Run(["pr-comment", FIRST, "--dry-run"]),
    Run(["dashboard"]),
    Run(["dashboard", "--trend"]),
    Run(["diff", FIRST, SECOND]),
    Run(["compare", FIRST, SECOND]),
    Run(["export", FIRST, "--format", "otlp", "-o", "otlp.json"], kind="json", file="otlp.json"),
    Run(["export", FIRST, "--format", "otlp-genai", "-o", "genai.json"], kind="json", file="genai.json"),
    Run(
        ["compliance", "export", FIRST, "--framework", "all", "-o", "compliance.json"],
        kind="json", file="compliance.json",
    ),
]


def not_a_number(n) -> dict:
    """JSON values that are not a usable number, by label.

    The text and the object hold ``n``, so a reader that turns the value
    into text with str() shows them. json.dumps writes NaN, Infinity and
    10**400 as bare tokens, which json.loads reads back.
    """
    return {
        "text": str(n),
        "object": {"value": n},
        "null": None,
        "nan": float("nan"),
        "infinity": float("inf"),
        "huge": 10**400,
        "negative": -n,
        "boolean": True,
    }


def not_a_count(n: int) -> dict:
    """not_a_number(n), plus finite numbers that are not an int."""
    return {**not_a_number(n), "fraction": n + 0.5, "whole float": float(n)}


def meta_files(root: str) -> dict:
    """Return the bytes of each meta.json under ``root``, by path."""
    return {str(path): path.read_bytes() for path in Path(root).rglob("meta.json")}


class NumberTest(CommandTest):
    """Runs agent-strace on one store per value of a field."""

    def write_session(
        self, store, sid: str, started_at: float = START, tool_events: bool = True, **fields
    ) -> None:
        """Write a 5 second session that ran one tool, unless not ``tool_events``."""
        meta = {
            "session_id": sid,
            "started_at": started_at,
            "ended_at": started_at + 5,
            "agent_name": "claude-code",
            "command": "claude-code (hook)",
            "tool_calls": 1,
            "total_tokens": 100,
            "total_duration_ms": 5000.0,
            **fields,
        }
        meta = {key: value for key, value in meta.items() if value is not MISSING}
        events = [("session_start", 0, {})]
        if tool_events:
            events += [
                ("tool_call", 1, {"tool_name": "Bash", "arguments": {"command": "ls"}}),
                ("tool_result", 2, {"tool_name": "Bash", "result": "ok"}),
            ]
        events.append(("session_end", 5, {}))
        lines = [
            json.dumps({
                "event_type": event_type,
                "timestamp": started_at + offset,
                "event_id": f"e{n}{sid[:6]}",
                "session_id": sid,
                "data": data,
            })
            for n, (event_type, offset, data) in enumerate(events)
        ]
        session_dir = store.base_dir / sid
        session_dir.mkdir(parents=True)
        (session_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        (session_dir / "events.ndjson").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def write_pair(self, store, **fields) -> None:
        """Write FIRST and its child SECOND, both with ``fields``."""
        self.write_session(store, FIRST, **fields)
        self.write_session(store, SECOND, started_at=START + 0.5, parent_session_id=FIRST, **fields)

    def write_duration(self, store, value) -> None:
        self.write_pair(store, total_duration_ms=value)

    def write_counts(self, store, value) -> None:
        self.write_pair(store, **dict.fromkeys(COUNTS, value))

    def outputs(self, write, runs: list[Run], values: dict) -> dict:
        """Return {label: [output per run]}, from one store per value.

        write(store, value) fills a store. An output is (exit code, stdout,
        stderr, text of the run's file or None), with the store's directory
        shown as <root>. Fails if a run changes a meta.json.
        """
        roots = {}
        for label, value in values.items():
            store = self.new_store()
            write(store, value)
            roots[label] = str(store.base_dir.parent)
        before = {label: meta_files(root) for label, root in roots.items()}
        results = iter(self.run_cli([[root, run.argv] for root in roots.values() for run in runs]))
        outputs = {}
        for label, root in roots.items():
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
                outputs[label].append((code, *texts))
            self.assertEqual(meta_files(root), before[label], f"{label}: a run changed meta.json")
        return outputs

    def assert_reads_as_missing(self, write, runs, valid, bad: dict, shown=None, refused=()) -> dict:
        """Each value in ``bad`` gives the output a missing field gives.

        ``shown`` is text that the output holds for ``valid`` and not for a
        missing field: one for all runs, or a list with one per run, None to
        skip. A label in ``refused`` is left to the caller. Returns the
        outputs of the values in ``bad``.
        """
        outputs = self.outputs(write, runs, {"missing": MISSING, "valid": valid, **bad})
        missing = outputs.pop("missing")
        good = outputs.pop("valid")
        expected = []
        for run, output in zip(runs, missing):
            self.assertEqual(output[0], 0, (run.argv, output))
            expected.append(self.view(run, output))
        for label, got in outputs.items():
            if label in refused:
                continue
            for run, output, want in zip(runs, got, expected):
                with self.subTest(label, argv=run.argv):
                    self.assertEqual(self.view(run, output), want)
        if not isinstance(shown, list):
            shown = [shown] * len(runs)
        for run, output, empty, text in zip(runs, good, missing, shown):
            with self.subTest("valid", argv=run.argv):
                self.assertEqual(output[0], 0, output)
                self.view(run, output)
                if text is not None:
                    self.assertIn(text, "".join(part for part in output[1:] if part))
                    self.assertNotIn(text, "".join(part for part in empty[1:] if part))
        return outputs


class TestDuration(NumberTest):
    def test_commands_read_the_value_as_missing(self):
        # Where a reader fell back to ended_at - started_at on main, a
        # missing duration still shows as 5 s. Other readers show 0 s or "?".
        runs = [
            *RUNS,
            Run(["eval", "dataset", "auto", "--filter", "long-duration:1s", "--since", "100000d",
                 "--dataset", "ds.jsonl"]),
        ]
        shown = {
            ("stats", FIRST): "Duration:   4.20s",
            ("tree", FIRST): "1 tools  4.2s",
            tuple(runs[-1].argv): "Added 2 session(s) to ds.jsonl",
        }
        self.assert_reads_as_missing(
            self.write_duration, runs, 4200.0, not_a_number(4200.0),
            shown=[shown.get(tuple(run.argv)) for run in runs],
        )


class TestCounts(NumberTest):
    def test_commands_read_the_value_as_missing(self):
        shown = {
            ("stats", FIRST): "Tool calls: 7",
            ("tree", FIRST, "--format", "json"): '"depth": 7',
        }
        self.assert_reads_as_missing(
            self.write_counts, RUNS, 7, not_a_count(7),
            shown=[shown.get(tuple(run.argv)) for run in RUNS],
        )


class TestShare(NumberTest):
    # The evidence packet digests meta.json as strict JSON, so share refuses
    # a session that holds NaN or Infinity in any field. main does the same.
    REFUSED = ("nan", "infinity")
    RUNS = [Run(["share", FIRST, "-o", "share.html"], kind="html", file="share.html")]

    def assert_refused(self, outputs) -> None:
        for label in self.REFUSED:
            with self.subTest(label):
                [(code, stdout, stderr, data)] = outputs[label]
                self.assertEqual((code, stdout, data), (1, "", None))
                self.assertIn("could not build evidence packet: Out of range float values", stderr)

    def test_duration(self):
        outputs = self.assert_reads_as_missing(
            self.write_duration, self.RUNS, 4200.0, not_a_number(4200.0), refused=self.REFUSED,
        )
        self.assert_refused(outputs)

    def test_counts(self):
        outputs = self.assert_reads_as_missing(
            self.write_counts, self.RUNS, 7, not_a_count(7), refused=self.REFUSED,
        )
        self.assert_refused(outputs)


class TestTreeDepth(NumberTest):
    RUNS = [Run(["tree", FIRST]), Run(["tree", FIRST, "--format", "json"], kind="json")]

    def test_output_does_not_grow_with_the_stored_depth(self):
        # The indent comes from the tree, so a large depth prints as a small
        # one does. The JSON still shows a usable depth as stored.
        def write(store, value):
            self.write_pair(store, depth=value)

        outputs = self.outputs(write, self.RUNS, {"missing": MISSING, "large": 10**6})
        [text, data] = [self.view(run, output) for run, output in zip(self.RUNS, outputs["large"])]
        [expected_text, expected_data] = [
            self.view(run, output) for run, output in zip(self.RUNS, outputs["missing"])
        ]
        self.assertEqual(text, expected_text)
        tree = data[3]["tree"]
        self.assertEqual((tree["depth"], tree["children"][0]["depth"]), (10**6, 10**6))
        tree["depth"] = tree["children"][0]["depth"] = 0
        self.assertEqual(data, expected_data)


class TestBaseline(NumberTest):
    UPDATE = ["baseline", "update", "--since", "100000", "--output", "baseline.json"]

    def update(self, first: dict, second: dict | None) -> tuple:
        """Run baseline update on FIRST and SECOND, then check SECOND.

        Leaves SECOND out if ``second`` is None. Returns the results and the
        text of baseline.json.
        """
        store = self.new_store()
        self.write_session(store, FIRST, **first)
        argvs = [self.UPDATE]
        if second is not None:
            self.write_session(store, SECOND, started_at=START + 60, **second)
            argvs.append(["baseline", "check", SECOND, "--baseline", "baseline.json"])
        root = str(store.base_dir.parent)
        results = self.run_cli([[root, argv] for argv in argvs])
        with open(os.path.join(root, "baseline.json"), encoding="utf-8") as f:
            return results, f.read()

    def test_update_leaves_out_a_session_whose_value_is_not_usable(self):
        [update], text = self.update({}, None)
        self.assertEqual(update, [0, "Baseline updated: 1 sessions, saved to baseline.json\n", ""])
        expected = strict_json(text)
        expected.pop("created_at")
        # A null value reads as it does on main: a null duration counts as
        # 0 ms, and a null count leaves the session out.
        cases = [
            *(("total_duration_ms", label, value) for label, value in not_a_number(4200.0).items()),
            *((field, label, value) for field in ("tool_calls", "llm_requests")
              for label, value in not_a_count(7).items()),
        ]
        for field, label, value in cases:
            if label == "null":
                continue
            with self.subTest(field=field, label=label):
                [update, check], text = self.update({}, {field: value})
                self.assertEqual(update, [0, "Baseline updated: 1 sessions, saved to baseline.json\n", ""])
                profile = strict_json(text)
                profile.pop("created_at")
                self.assertEqual(profile, expected)
                self.assertEqual(check[0], 1)
                self.assertIn("[UNSCORED]", check[1])
                self.assertIn("Could not score this session: its metadata has non-numeric counters", check[1])

    def test_update_with_durations_near_the_float_limit_writes_strict_json(self):
        # The sum and the squares overflow a float, so a plain mean and
        # stddev would be Infinity.
        [update, _check], text = self.update({"total_duration_ms": 1e308}, {"total_duration_ms": 1.7e308})
        self.assertEqual(update[0], 0)
        stats = strict_json(text)["duration_ms"]
        self.assertEqual((stats["min"], stats["max"], stats["count"]), (1e308, 1.7e308, 2))
        self.assertTrue(math.isclose(stats["mean"], 1.35e308), stats)
        self.assertTrue(math.isclose(stats["stddev"], 0.35e308 * math.sqrt(2)), stats)


class TestToolCallsWithoutToolEvents(NumberTest):
    """team-report and context-score count the tool call events, and read
    tool_calls from meta.json only for a session with none."""

    def write(self, store, value) -> None:
        self.write_session(store, FIRST, tool_events=False, tool_calls=value)

    def test_team_report(self):
        runs = [
            Run(
                ["team-report", "--since", "2025-12-01", "--until", "2026-02-01", "--export", "json"],
                kind="json",
            ),
        ]
        self.assert_reads_as_missing(self.write, runs, 7, not_a_count(7), shown='"total_tool_calls": 7.0')

    def test_context_score(self):
        def tool_calls(value) -> int:
            store = self.new_store()
            self.write(store, value)
            meta = store.load_meta(FIRST)
            return context_score._session_metrics(store, meta, "", "", [], []).tool_calls

        self.assertEqual(tool_calls(7), 7)
        for label, value in {"missing": MISSING, **not_a_count(7)}.items():
            with self.subTest(label):
                self.assertEqual(tool_calls(value), 0)


class TestWebDashboard(NumberTest):
    def payloads(self, fields: dict) -> list:
        """Return each route's parsed payload."""
        store = self.new_store()
        self.write_pair(store, **fields)
        root = str(store.base_dir.parent)
        texts = [
            web_dashboard.api_sessions(store),
            web_dashboard.api_session_review(store, FIRST),
            web_dashboard.api_session_compare(store, FIRST, SECOND),
            web_dashboard.api_session_otlp(store, FIRST),
        ]
        for path in sorted({root, os.path.realpath(root)}, key=len, reverse=True):
            texts = [text.replace(path, "<root>") for text in texts]
        return [without_clock(strict_json(text)) for text in texts]

    def test_each_route_reads_the_value_as_missing(self):
        for fields, valid, bad in (
            (["total_duration_ms"], 4200.0, not_a_number(4200.0)),
            (COUNTS, 7, not_a_count(7)),
        ):
            expected = self.payloads(dict.fromkeys(fields, MISSING))
            sessions = self.payloads(dict.fromkeys(fields, valid))[0]
            self.assertEqual([session[fields[0]] for session in sessions], [valid, valid])
            for label, value in bad.items():
                with self.subTest(fields[0], label=label):
                    self.assertEqual(self.payloads(dict.fromkeys(fields, value)), expected)


class TestLangfuseExport(NumberTest):
    def posts(self, fields: dict) -> tuple:
        """Run the Langfuse and OTLP metrics exports, and return what they post."""
        store = self.new_store()
        self.write_session(store, FIRST, **fields)
        bodies = []

        class Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                return False

        class Opener:
            def open(self, request, timeout=None):
                bodies.append((request.full_url, request.data.decode("utf-8")))
                return Response()

        common = ["--trace-dir", str(store.base_dir), "export", FIRST]
        argvs = [
            [*common, "--scores", "--backend", "langfuse", "--langfuse-public-key", "pk",
             "--langfuse-secret-key", "sk", "--langfuse-host", "https://langfuse.invalid"],
            [*common, "--metrics", "--backend", "otlp", "--otlp-endpoint", "https://otlp.invalid"],
        ]
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(langfuse_export, "no_redirect_opener", Opener), \
                redirect_stdout(out), redirect_stderr(err):
            codes = [cmd_export(build_parser().parse_args(argv)) for argv in argvs]
        posted = [(url, without_clock(strict_json(body))) for url, body in bodies]
        return codes, posted, out.getvalue(), err.getvalue()

    def test_posts_read_the_value_as_missing(self):
        for fields, valid, bad in (
            (["total_duration_ms"], 4200.0, not_a_number(4200.0)),
            (COUNTS, 7, not_a_count(7)),
        ):
            expected = self.posts(dict.fromkeys(fields, MISSING))
            self.assertEqual(expected[0], [0, 0])
            self.assertEqual(len(expected[1]), 2)
            self.assertNotEqual(self.posts(dict.fromkeys(fields, valid))[1], expected[1])
            for label, value in bad.items():
                with self.subTest(fields[0], label=label):
                    self.assertEqual(self.posts(dict.fromkeys(fields, value)), expected)


class TestHookCounts(_SubagentHookTest):
    TOKENS = 40

    def counts_after_hooks(self, n: int, value) -> tuple:
        """Store ``value`` as three counts of a Codex session, then run hooks.

        The hooks record a tool call that fails and a model request that
        fails. Returns the stored tool_calls, errors and total_tokens.
        """
        sid = f"code{n:04d}-0000-4000-8000-{n:012d}"
        self.dispatch("SessionStart", {"session_id": sid, "source": "startup"}, provider="codex")
        path = self.root / local(sid) / "meta.json"
        meta = json.loads(path.read_text(encoding="utf-8"))
        meta.update(dict.fromkeys(("tool_calls", "errors", "total_tokens"), value))
        path.write_text(json.dumps(meta), encoding="utf-8")
        payload = {
            "session_id": sid,
            "tool_name": "Bash",
            "tool_input": {"command": "true"},
            "error": "boom",
            "token_usage": {"total_tokens": self.TOKENS},
        }
        for event in ("PreToolUse", "PostToolUseFailure", "PostModelRequest"):
            self.dispatch(event, payload, provider="codex")
        stored = json.loads(path.read_text(encoding="utf-8"))
        return stored["tool_calls"], stored["errors"], stored["total_tokens"]

    def test_count_that_is_not_a_count_restarts_at_zero(self):
        # Each failure adds an error, so errors grows by 2.
        self.assertEqual(self.counts_after_hooks(1, 7), (8, 9, 7 + self.TOKENS))
        for n, (label, value) in enumerate(not_a_count(7).items(), start=2):
            with self.subTest(label):
                self.assertEqual(self.counts_after_hooks(n, value), (1, 2, self.TOKENS))


class TestSubagentDepth(_SubagentHookTest):
    def test_parent_depth_that_is_not_a_count_reads_as_zero(self):
        for label, value in not_a_count(2).items():
            with self.subTest(label):
                self.root = Path(tempfile.mkdtemp(dir=self._tmp.name)) / ".agent-traces"
                os.environ["AGENT_TRACE_DIR"] = str(self.root)
                self.start_parent()
                path = self.root / local(PARENT) / "meta.json"
                meta = json.loads(path.read_text(encoding="utf-8"))
                meta["depth"] = value
                path.write_text(json.dumps(meta), encoding="utf-8")
                self.subagent_start()
                self.child_runs()
                self.subagent_stop()
                self.assertEqual(self.store().load_meta(local(CHILD)).depth, 1)


class TestRecordParentDepth(NumberTest):
    """record and record-http give a subagent its parent's depth plus 1."""

    def test_parent_depth_that_is_not_a_count_reads_as_zero(self):
        def child_depth(value) -> int:
            store = self.new_store()
            self.write_session(store, FIRST, depth=value)
            return _parent_depth(store, FIRST)

        self.assertEqual(child_depth(2), 3)
        for label, value in {"missing": MISSING, **not_a_count(2)}.items():
            with self.subTest(label):
                self.assertEqual(child_depth(value), 1)


if __name__ == "__main__":
    unittest.main()
