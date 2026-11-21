"""meta.json text fields that are not strings.

A hand-edited or foreign meta.json can hold a list, an object, a number, a
boolean, null, NaN or Infinity where agent_name, command or team should be a
string. Each reader below treats such a value as an empty string: it does not
crash, and the raw value does not reach the output. A command gives the same
output as it gives when the field is missing.
"""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from typing import NamedTuple
from unittest import mock

try:  # keeps product telemetry off in this process and its children
    import _no_telemetry
except ImportError:  # pragma: no cover - run as a package module
    from tests import _no_telemetry

from agent_trace import cognitive_debt, langfuse_export, team_report, web_dashboard
from agent_trace.budget_report import (
    build_report,
    format_report_json,
    format_report_markdown,
    format_report_text,
    team_summary,
)
from agent_trace.cli import build_parser, cmd_export
from agent_trace.cognitive_debt import build_cognitive_debt_report
from agent_trace.curve import DEFAULT_TASK_TYPE, analyse_curve, classify_session
from agent_trace.dashboard import build_dashboard
from agent_trace.freeze import freeze_session
from agent_trace.health import ProviderEvidence, detect_provider, redaction_state_for
from agent_trace.hooks import IMPLICIT_SESSION_SOURCE
from agent_trace.integrations import _base
from agent_trace.live_verify import inspect_live_capture
from agent_trace.models import EventType
from agent_trace.store import TraceStore
from agent_trace.team_report import build_team_report

START = 1767225600.0  # 2026-01-01T00:00:00Z
WINDOW = (START - 3600, START + 86400)
IMPLICIT = f"claude-code ({IMPLICIT_SESSION_SOURCE})"
FIRST = "a1b2c3d4e5f60001"
SECOND = "b1b2c3d4e5f60002"
THIRD = "c1b2c3d4e5f60003"
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
# write_session leaves a field set to MISSING out of meta.json.
MISSING = object()
SHA256 = re.compile(r"\b[0-9a-f]{64}\b")
# Keys whose values come from the clock, not from the session.
CLOCK_KEYS = frozenset({"generated_at", "exported_at", "generation_timestamp", "timeUnixNano"})
# The share page prints the time it was made in this form.
CLOCK_TEXT = re.compile(r"\b\d{4}-\d\d-\d\d \d\d:\d\d:\d\d UTC\b")


def not_text(words: str) -> dict:
    """JSON values that are not strings, by label.

    The list and the object hold ``words``, so a reader that turns the value
    into text with str() shows them. json.dumps writes NaN and Infinity as
    bare tokens, which json.loads reads back.
    """
    return {
        "list": [words],
        "object": {"value": words},
        "number": 5,
        "boolean": True,
        "null": None,
        "nan": float("nan"),
        "infinity": float("inf"),
    }


def strict_json(text: str):
    """Parse ``text`` as JSON, refusing NaN, Infinity and -Infinity."""

    def refuse(token: str):
        raise ValueError(f"not strict JSON: {token}")

    return json.loads(text, parse_constant=refuse)


def without_clock(value):
    """Return ``value`` with the value of each key in CLOCK_KEYS replaced."""
    if isinstance(value, dict):
        return {
            key: "<clock>" if key in CLOCK_KEYS else without_clock(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [without_clock(item) for item in value]
    return value


# Runs agent-strace once per [cwd, argv] read from stdin, in this one
# process, and writes [exit code, stdout, stderr] per run to the file named
# by argv[1]. An exception that escapes the command shows as its type and
# message in place of the exit code.
DRIVER = r"""
import io
import json
import os
import sys

if sys.path and sys.path[0] == "":
    del sys.path[0]


class Switch(io.TextIOBase):
    # Modules bind sys.stdout as a default argument at import. So each run
    # points one stream object at a new buffer, not a new stream object.
    encoding = "utf-8"

    def __init__(self):
        self.target = io.StringIO()

    def writable(self):
        return True

    def write(self, text):
        return self.target.write(text)


results_path = sys.argv[1]
out, err = Switch(), Switch()
sys.stdout, sys.stderr = out, err
from agent_trace.entry import main

results = []
for cwd, argv in json.load(sys.stdin):
    out.target, err.target = io.StringIO(), io.StringIO()
    os.chdir(cwd)
    sys.argv = ["agent-strace", "--trace-dir", ".agent-traces", *argv]
    try:
        main()
        code = 0
    except SystemExit as exc:
        if exc.code is None or isinstance(exc.code, int):
            code = exc.code or 0
        else:
            err.write(f"{exc.code}\n")
            code = 1
    except Exception as exc:
        code = f"{type(exc).__name__}: {exc}"
    results.append([code, out.target.getvalue(), err.target.getvalue()])
with open(results_path, "w", encoding="utf-8") as f:
    json.dump(results, f)
"""


class Run(NamedTuple):
    """One agent-strace command line and how to compare what it writes.

    ``kind`` is text, json, jsonl or html. With ``file``, the output is that
    file, relative to the store's directory; else it is stdout.
    """

    argv: list
    kind: str = "text"
    file: str = ""


class MetaTextTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for name in (
            "AGENT_STRACE_WORKSPACE",
            "AGENT_STRACE_TENANT_ID",
            "AGENT_STRACE_ENDPOINT",
            "AGENT_TRACE_DIR",
        ):
            os.environ.pop(name, None)
        os.environ["HOME"] = os.environ["USERPROFILE"] = self.root

    def new_store(self) -> TraceStore:
        return TraceStore(os.path.join(tempfile.mkdtemp(dir=self.root), ".agent-traces"))

    def write_session(self, store: TraceStore, sid: str, started_at: float = START, **fields) -> None:
        meta = {
            "session_id": sid,
            "started_at": started_at,
            "ended_at": started_at + 1.5,
            "agent_name": "claude-code",
            "command": "claude-code (hook)",
            "tool_calls": 1,
            "total_tokens": 100,
            "total_duration_ms": 1500.0,
            **fields,
        }
        meta = {key: value for key, value in meta.items() if value is not MISSING}
        event = {
            "event_type": "tool_call",
            "timestamp": started_at + 0.5,
            "event_id": "e1" + sid[:6],
            "session_id": sid,
            "data": {"tool_name": "Bash", "arguments": {"command": "ls"}},
        }
        session_dir = store.base_dir / sid
        session_dir.mkdir(parents=True)
        (session_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        (session_dir / "events.ndjson").write_text(json.dumps(event) + "\n", encoding="utf-8")


class TestCurve(MetaTextTest):
    def test_value_that_is_not_text_classifies_as_general(self):
        for label, value in not_text("fix the bug").items():
            with self.subTest(label):
                self.assertEqual(classify_session(value, ""), DEFAULT_TASK_TYPE)
                self.assertEqual(classify_session("", value), DEFAULT_TASK_TYPE)

    def test_the_other_field_still_classifies(self):
        for label, value in not_text("fix the bug").items():
            with self.subTest(label):
                self.assertEqual(classify_session(value, "pytest tests/"), "Unit test writing")
                self.assertEqual(classify_session("pytest", value), "Unit test writing")

    def test_curve_counts_such_sessions(self):
        for label, value in not_text("fix the bug").items():
            with self.subTest(label):
                store = self.new_store()
                self.write_session(store, FIRST)
                self.write_session(store, SECOND, started_at=START + 60, agent_name=value)
                self.write_session(store, THIRD, started_at=START + 120, command=value)
                report = analyse_curve(store, min_sessions=1)
                self.assertEqual(
                    [(stat.task_type, stat.session_count) for stat in report.stats],
                    [(DEFAULT_TASK_TYPE, 3)],
                )


class TestDashboard(MetaTextTest):
    def test_agent_filter_passes_over_a_name_that_is_not_text(self):
        for label, value in not_text("claude-code").items():
            with self.subTest(label):
                store = self.new_store()
                self.write_session(store, FIRST)
                self.write_session(store, SECOND, started_at=START + 60, agent_name=value)
                filtered = build_dashboard(store, agent_filter="claude")
                self.assertEqual([s.session_id for s in filtered.summaries], [FIRST])
                everything = build_dashboard(store)
                self.assertEqual(
                    sorted(s.session_id for s in everything.summaries), [FIRST, SECOND]
                )


class TestBudgetReport(MetaTextTest):
    def render(self, report) -> tuple[str, str, dict]:
        text = io.StringIO()
        format_report_text(report, out=text)
        markdown = io.StringIO()
        format_report_markdown(report, out=markdown)
        top = strict_json(format_report_json(report))["top_sessions"]
        return text.getvalue(), markdown.getvalue(), {s["session_id"]: s["agent_name"] for s in top}

    def test_agent_name_that_is_not_text_falls_back_to_the_command(self):
        for label, value in not_text("fix the bug").items():
            with self.subTest(label):
                store = self.new_store()
                self.write_session(store, FIRST, agent_name=value)
                self.write_session(
                    store, SECOND, started_at=START + 60, agent_name=value, command=value
                )
                report = build_report(store, *WINDOW, include_prior=False)
                names = {FIRST: "claude-code (hook)", SECOND: ""}
                self.assertEqual({s.session_id: s.agent_name for s in report.sessions}, names)
                text, markdown, top = self.render(report)
                self.assertEqual(top, names)
                # A session without a name shows its short ID in the name column.
                for sid, shown in ((FIRST, "claude-code (hook)"), (SECOND, SECOND[:12])):
                    name = re.escape(shown)
                    self.assertRegex(
                        text, re.compile(rf"^  \d+\. {sid[:12]}  \$\d+\.\d\d  {name}  ", re.M)
                    )
                    self.assertRegex(
                        markdown,
                        re.compile(rf"^\| \d+ \| `{sid[:12]}` \| \$\d+\.\d\d \| {name} \| ", re.M),
                    )

    def test_team_that_is_not_text_reads_as_unassigned(self):
        for label, value in not_text("platform").items():
            with self.subTest(label):
                store = self.new_store()
                self.write_session(store, FIRST, team="platform")
                self.write_session(store, SECOND, started_at=START + 60, team=value)
                report = build_report(store, *WINDOW, include_prior=False)
                self.assertEqual(
                    {s.session_id: s.team for s in report.sessions}, {FIRST: "platform", SECOND: ""}
                )
                self.assertEqual(sorted(team_summary(report.sessions)), ["(unassigned)", "platform"])
                text, _, _ = self.render(report)
                self.assertRegex(text, re.compile(r"^  \(unassigned\) +\$", re.M))


class TestFreeze(MetaTextTest):
    def test_command_that_is_not_text_falls_back_to_the_agent_name(self):
        for label, value in not_text("fix the bug").items():
            with self.subTest(label):
                store = self.new_store()
                self.write_session(store, FIRST, command=value)
                self.write_session(
                    store, SECOND, started_at=START + 60, agent_name=value, command=value
                )
                self.assertEqual(freeze_session(store, FIRST).task, "claude-code")
                self.assertEqual(freeze_session(store, SECOND).task, "")


class TestBranchFromCommand(MetaTextTest):
    COMMAND = "claude (branch: feat/x)"

    def store_with(self, value) -> TraceStore:
        store = self.new_store()
        self.write_session(store, FIRST, command=self.COMMAND)
        self.write_session(store, SECOND, started_at=START + 60, command=value)
        return store

    def test_team_report(self):
        for label, value in not_text(self.COMMAND).items():
            with self.subTest(label):
                store = self.store_with(value)
                with mock.patch.object(team_report, "_git", return_value=""):
                    report = build_team_report(store, *WINDOW, group_by="branch")
                self.assertEqual(
                    {s.session_id: s.branch for s in report.sessions},
                    {FIRST: "feat/x", SECOND: "(unknown)"},
                )
                self.assertEqual(sorted(report.rows), ["(unknown)", "feat/x"])

    def test_cognitive_debt(self):
        for label, value in not_text(self.COMMAND).items():
            with self.subTest(label):
                store = self.store_with(value)
                with mock.patch.object(cognitive_debt, "_git", return_value=""):
                    report = build_cognitive_debt_report(store, *WINDOW, group_by="branch")
                self.assertEqual(
                    {s.session_id: s.branch for s in report.sessions},
                    {FIRST: "feat/x", SECOND: "(unknown)"},
                )
                self.assertEqual(sorted(report.rows), ["(unknown)", "feat/x"])


class TestHealth(MetaTextTest):
    def test_hook_metadata_names_the_provider(self):
        store = self.new_store()
        self.write_session(store, FIRST, command=IMPLICIT)
        meta = store.load_meta(FIRST)
        self.assertEqual(detect_provider(meta, []), ProviderEvidence("claude", "hook_metadata"))
        self.assertEqual(len(redaction_state_for(meta, []).session_start_modes), 1)

    def test_agent_name_that_is_not_text_is_not_hook_metadata(self):
        for label, value in not_text("claude-code").items():
            with self.subTest(label):
                store = self.new_store()
                self.write_session(store, FIRST, agent_name=value, command=IMPLICIT)
                self.write_session(store, SECOND, started_at=START + 60, agent_name="", command=IMPLICIT)
                meta = store.load_meta(FIRST)
                self.assertEqual(detect_provider(meta, []), ProviderEvidence())
                self.assertEqual(
                    redaction_state_for(meta, []),
                    redaction_state_for(store.load_meta(SECOND), []),
                )
                self.assertEqual(redaction_state_for(meta, []).session_start_modes, ())


class TestLiveVerify(MetaTextTest):
    def assert_selects_older(self, **newer_fields) -> None:
        store = self.new_store()
        self.write_session(store, FIRST, started_at=START - 600, command=IMPLICIT)
        self.write_session(store, SECOND, **newer_fields)
        result = inspect_live_capture(store.base_dir, "claude")
        self.assertEqual(result.report.session_id, FIRST)
        self.assertEqual(result.selection, "newest matching recorded session")

    def test_newer_session_with_an_agent_name_that_is_not_text_is_passed_over(self):
        for label, value in not_text("claude-code").items():
            with self.subTest(label):
                self.assert_selects_older(agent_name=value, command=IMPLICIT)

    def test_newer_session_with_a_command_that_is_not_text_is_passed_over(self):
        for label, value in not_text("claude-code (hook)").items():
            with self.subTest(label):
                self.assert_selects_older(command=value)


class CommandTest(MetaTextTest):
    """Runs agent-strace on one store per value of a field."""

    def run_cli(self, runs: list) -> list:
        """Run agent-strace once per [cwd, argv], all in one child process."""
        env = _no_telemetry.without_telemetry({
            key: value
            for key, value in os.environ.items()
            if not key.startswith("AGENT_STRACE_") and key != "AGENT_TRACE_DIR"
        })
        env["PYTHONPATH"] = os.pathsep.join(item for item in (SRC, env.get("PYTHONPATH", "")) if item)
        env["PYTHONIOENCODING"] = "utf-8"
        results = os.path.join(self.root, "results.json")
        proc = subprocess.run(
            [sys.executable, "-c", DRIVER, results],
            input=json.dumps(runs),
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=self.root,
            env=env,
            timeout=300,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        with open(results, encoding="utf-8") as f:
            return json.load(f)

    def outputs(self, write, runs: list[Run], words: str) -> dict:
        """Return {label: [output per run]}, from one store per value.

        write(store, value) fills a store. The labels are "missing", "text"
        (``words`` as a string) and those of not_text(words). An output is
        (exit code, stdout, stderr, text of the run's file or None), with the
        store's directory shown as <root>.
        """
        labels = {"missing": MISSING, "text": words, **not_text(words)}
        roots = {}
        for label, value in labels.items():
            store = self.new_store()
            write(store, value)
            roots[label] = str(store.base_dir.parent)
        results = iter(self.run_cli([[roots[label], run.argv] for label in labels for run in runs]))
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
        return outputs

    @staticmethod
    def view(run: Run, output: tuple) -> tuple:
        """Return the part of ``output`` that must not depend on the value.

        JSON is parsed strictly, so NaN or Infinity in the output fails.
        """
        code, stdout, stderr, data = output
        text = data if run.file else stdout
        if code == 0 and run.kind == "json":
            text = without_clock(strict_json(text))
        elif code == 0 and run.kind == "jsonl":
            text = [without_clock(strict_json(line)) for line in text.splitlines()]
        elif run.kind == "html" and text is not None:
            # A digest of the source meta.json differs with the value.
            text = CLOCK_TEXT.sub("<clock>", SHA256.sub("<sha256>", text))
        return code, stdout if run.file else None, stderr, text

    def assert_reads_as_missing(self, write, runs, words, shown, refused=()) -> dict:
        """Each value that is not a string gives the output a missing field gives.

        ``shown`` is text that the output holds for ``words`` as a string and
        not for a missing field: one for all runs, or a list with one per run,
        None to skip. A label in ``refused`` is left to the caller. Returns
        the outputs of the values that are not strings.
        """
        outputs = self.outputs(write, runs, words)
        missing = outputs.pop("missing")
        valid = outputs.pop("text")
        expected = []
        for run, output in zip(runs, missing):
            self.assertEqual(output[0], 0, (run.argv, output))
            expected.append(self.view(run, output))
        for label, got in outputs.items():
            if label not in refused:
                with self.subTest(label):
                    self.assertEqual([self.view(run, output) for run, output in zip(runs, got)], expected)
        if not isinstance(shown, list):
            shown = [shown] * len(runs)
        for run, output, empty, text in zip(runs, valid, missing, shown):
            with self.subTest("text", argv=run.argv):
                self.assertEqual(output[0], 0, output)
                self.view(run, output)
                if text is not None:
                    self.assertIn(text, "".join(part for part in output[1:] if part))
                    self.assertNotIn(text, "".join(part for part in empty[1:] if part))
        return outputs


class TestCompareTag(CommandTest):
    RUNS = [
        Run(["compare", "--tag", "claude"]),
        Run(["compare", "--tag", "claude", "--format", "json"], kind="json"),
    ]

    def test_agent_name_label(self):
        # THIRD matches the tag by its command. Without a name, its label is
        # its short ID.
        def write(store, value):
            self.write_session(store, FIRST)
            self.write_session(store, SECOND, started_at=START + 60)
            self.write_session(store, THIRD, started_at=START + 120, agent_name=value)

        self.assert_reads_as_missing(write, self.RUNS, "named-agent", shown="named-agent")

    def test_agent_name_match(self):
        # THIRD's command does not hold the tag, so only its name can match.
        def write(store, value):
            self.write_session(store, FIRST)
            self.write_session(store, SECOND, started_at=START + 60)
            self.write_session(
                store, THIRD, started_at=START + 120, agent_name=value, command="other (hook)"
            )

        self.assert_reads_as_missing(write, self.RUNS, "claude-agent", shown="claude-agent")

    def test_command(self):
        # THIRD's name does not hold the tag, so only its command can match.
        def write(store, value):
            self.write_session(store, FIRST)
            self.write_session(store, SECOND, started_at=START + 60)
            self.write_session(
                store, THIRD, started_at=START + 120, agent_name="other-agent", command=value
            )

        self.assert_reads_as_missing(write, self.RUNS, "claude (named-command)", shown="other-agent")


class TestOtlpExport(CommandTest):
    RUNS = [
        Run(["export", FIRST, "--format", "otlp", "-o", "otlp.json"], kind="json", file="otlp.json"),
        Run(["export", FIRST, "--format", "otlp-genai", "-o", "genai.json"], kind="json", file="genai.json"),
    ]

    def test_agent_name(self):
        # The classic export picks gen_ai.system from words such as "claude"
        # in the name.
        self.assert_reads_as_missing(
            lambda store, value: self.write_session(store, FIRST, agent_name=value),
            self.RUNS, "claude-named-agent", shown="claude-named-agent",
        )

    def test_command(self):
        # The GenAI payload does not carry the command.
        self.assert_reads_as_missing(
            lambda store, value: self.write_session(store, FIRST, command=value),
            self.RUNS, "named-command", shown=["named-command", None],
        )


class TestTree(CommandTest):
    RUNS = [Run(["tree", FIRST]), Run(["tree", FIRST, "--format", "json"], kind="json")]

    def write_tree(self, store, **fields) -> None:
        self.write_session(store, FIRST, **fields)
        self.write_session(store, SECOND, started_at=START + 0.5, parent_session_id=FIRST, **fields)

    def test_agent_name_falls_back_to_the_command(self):
        self.assert_reads_as_missing(
            lambda store, value: self.write_tree(store, agent_name=value),
            self.RUNS, "named-agent", shown="named-agent",
        )

    def test_command_without_an_agent_name(self):
        self.assert_reads_as_missing(
            lambda store, value: self.write_tree(store, agent_name=MISSING, command=value),
            self.RUNS, "named-command", shown="named-command",
        )


class TestReplay(CommandTest):
    RUNS = [
        Run(["replay", FIRST]),
        Run(["replay", FIRST, "--expand-subagents"]),
        Run(["stats", FIRST]),
    ]

    def test_agent_name(self):
        self.assert_reads_as_missing(
            lambda store, value: self.write_session(store, FIRST, agent_name=value),
            self.RUNS, "named-agent", shown="named-agent",
        )

    def test_command(self):
        # The expanded replay names the agent only.
        self.assert_reads_as_missing(
            lambda store, value: self.write_session(store, FIRST, command=value),
            self.RUNS, "named-command", shown=["named-command", None, "named-command"],
        )


class TestA2ATree(CommandTest):
    def test_agent_name(self):
        runs = [Run(["a2a-tree", FIRST]), Run(["a2a-tree", FIRST, "--format", "json"], kind="json")]
        self.assert_reads_as_missing(
            lambda store, value: self.write_session(store, FIRST, agent_name=value),
            runs, "named-agent", shown="named-agent",
        )


class TestCompliance(CommandTest):
    RUNS = [
        Run(
            ["compliance", "export", FIRST, "--framework", "all", "-o", "compliance.json"],
            kind="json", file="compliance.json",
        ),
        Run(["export", FIRST, "--format", "eu-ai-act", "-o", "eu.json"], kind="json", file="eu.json"),
    ]

    def test_agent_name(self):
        self.assert_reads_as_missing(
            lambda store, value: self.write_session(store, FIRST, agent_name=value),
            self.RUNS, "named-agent", shown="named-agent",
        )

    def test_command(self):
        # Only the EU AI Act record reads the command.
        self.assert_reads_as_missing(
            lambda store, value: self.write_session(store, FIRST, command=value),
            self.RUNS, "named-command", shown=[None, "named-command"],
        )

    def test_agent_name_without_a_command(self):
        # The EU AI Act record falls back from the command to the name.
        self.assert_reads_as_missing(
            lambda store, value: self.write_session(store, FIRST, agent_name=value, command=MISSING),
            self.RUNS[1:], "named-agent", shown='"intended_purpose": "named-agent"',
        )

    def test_team(self):
        # Only the SOC 2 report reads the team.
        self.assert_reads_as_missing(
            lambda store, value: self.write_session(store, FIRST, team=value),
            self.RUNS, "named-team", shown=["named-team", None],
        )


class TestSample(CommandTest):
    RUNS = [Run(["sample", "--n", "5", "-o", "sample.jsonl"], kind="jsonl", file="sample.jsonl")]

    def test_agent_name(self):
        self.assert_reads_as_missing(
            lambda store, value: self.write_session(store, FIRST, agent_name=value),
            self.RUNS, "named-agent", shown="named-agent",
        )

    def test_command(self):
        self.assert_reads_as_missing(
            lambda store, value: self.write_session(store, FIRST, command=value),
            self.RUNS, "named-command", shown="named-command",
        )


class TestShare(CommandTest):
    # The evidence packet digests meta.json as strict JSON, so share refuses
    # a session that holds NaN or Infinity in any field. main does the same.
    REFUSED = ("nan", "infinity")

    def assert_refused(self, outputs) -> None:
        for label in self.REFUSED:
            with self.subTest(label):
                [(code, stdout, stderr, data)] = outputs[label]
                self.assertEqual((code, stdout, data), (1, "", None))
                self.assertIn("could not build evidence packet: Out of range float values", stderr)

    def test_agent_name_with_content(self):
        runs = [Run(["share", FIRST, "--include-content", "-o", "share.html"], kind="html", file="share.html")]
        outputs = self.assert_reads_as_missing(
            lambda store, value: self.write_session(store, FIRST, agent_name=value),
            runs, "named-agent", shown="named-agent", refused=self.REFUSED,
        )
        self.assert_refused(outputs)

    def test_minimized_share_shows_no_name(self):
        runs = [Run(["share", FIRST, "-o", "share.html"], kind="html", file="share.html")]
        outputs = self.assert_reads_as_missing(
            lambda store, value: self.write_session(store, FIRST, agent_name=value),
            runs, "named-agent", shown=None, refused=self.REFUSED,
        )
        self.assert_refused(outputs)


class TestWebDashboard(MetaTextTest):
    def payloads(self, field: str, value) -> tuple[list, list]:
        """Return each route's parsed payload, and the raw text."""
        store = self.new_store()
        self.write_session(store, FIRST, **{field: value})
        self.write_session(store, SECOND, started_at=START + 60)
        root = str(store.base_dir.parent)
        texts = [
            web_dashboard.api_sessions(store),
            web_dashboard.api_session_review(store, FIRST),
            web_dashboard.api_session_compare(store, FIRST, SECOND),
            web_dashboard.api_session_otlp(store, FIRST),
            web_dashboard.api_minimized_review_share(store, FIRST),
        ]
        for path in sorted({root, os.path.realpath(root)}, key=len, reverse=True):
            texts = [text.replace(path, "<root>") for text in texts]
        parsed = [without_clock(strict_json(text)) for text in texts[:-1]]
        return [*parsed, SHA256.sub("<sha256>", texts[-1])], texts

    def test_each_route_reads_the_value_as_missing(self):
        for field, words in (
            ("agent_name", "named-agent"),
            ("command", "named-command"),
            ("team", "named-team"),
        ):
            expected, _ = self.payloads(field, MISSING)
            _, valid = self.payloads(field, words)
            self.assertIn(words, valid[0])
            for label, value in not_text(words).items():
                with self.subTest(field=field, label=label):
                    self.assertEqual(self.payloads(field, value)[0], expected)

    def test_session_list_leaves_out_a_value_that_is_not_text(self):
        store = self.new_store()
        self.write_session(store, FIRST, agent_name=["named-agent"], command=5, team=float("nan"))
        [session] = strict_json(web_dashboard.api_sessions(store))
        self.assertEqual(session["session_id"], FIRST)
        self.assertFalse({"agent_name", "command", "team"} & set(session))


class TestLangfuseExport(MetaTextTest):
    def posts(self, field: str, value) -> tuple:
        """Run the Langfuse and OTLP metrics exports, and return what they post."""
        store = self.new_store()
        self.write_session(store, FIRST, **{field: value})
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
        for field, words in (("agent_name", "named-agent"), ("command", "named-command")):
            expected = self.posts(field, MISSING)
            self.assertEqual(expected[0], [0, 0])
            self.assertEqual(len(expected[1]), 2)
            self.assertIn(words, json.dumps(self.posts(field, words)[1]))
            for label, value in not_text(words).items():
                with self.subTest(field=field, label=label):
                    self.assertEqual(self.posts(field, value), expected)


class TestIntegrationEmit(MetaTextTest):
    def emit_after_end(self, value) -> tuple:
        """Emit one event for an ended session whose agent_name is ``value``."""
        store = self.new_store()
        self.write_session(store, FIRST, agent_name=value)
        event = _base.emit(EventType.TOOL_CALL, FIRST, store, tool_name="Bash")
        sessions = sorted(path.name for path in store.base_dir.iterdir() if path.is_dir())
        return event.session_id, sessions, len(store.load_events(FIRST))

    def test_event_for_an_agent_name_that_is_not_text_is_dropped(self):
        # Without a name there is no current session to move the event to.
        expected = self.emit_after_end(MISSING)
        self.assertEqual(expected, (FIRST, [FIRST], 1))
        for label, value in not_text("named-agent").items():
            with self.subTest(label):
                self.assertEqual(self.emit_after_end(value), expected)
        sid, sessions, count = self.emit_after_end("named-agent")
        self.assertNotEqual(sid, FIRST)
        self.assertEqual(sessions, sorted([FIRST, sid]))
        self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
