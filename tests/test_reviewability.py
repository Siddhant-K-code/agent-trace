"""Tests for the pull-request reviewability check (agent-strace review)."""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from agent_trace.approval import ApprovalRequest, save_request
from agent_trace.cli import build_parser
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.pr_comment import (
    COMMENT_MARKER,
    cmd_pr_comment,
    head_commit_timestamp,
    pull_request_head_sha,
    select_sessions_with_link,
)
from agent_trace.reviewability import (
    FINDINGS,
    Artifacts,
    LinkedSession,
    ReviewConfigError,
    Threshold,
    Finding,
    artifacts_from_args,
    build_report,
    classify_validation,
    cmd_review,
    format_text,
    link_sessions,
    load_threshold,
    read_cognitive_debt,
    render_markdown,
)
from agent_trace.store import TraceStore

SECRET_PROMPT = "Refactor src/auth.py for the zebra-quartz-77 rollout"
BRANCH = "feature/review"
CWD = "/work/repo"


# Sanitized from a recorded GitHub Copilot CLI shell call: file contents are
# written with heredocs, then the test suite runs on the last line.
MULTI_LINE_SCRIPT = (
    "cd /work/demo && cat > src/pkg/sentences.py <<'EOF'\n"
    "import re\n"
    "\n"
    "\n"
    "def sentence_count(text):\n"
    "    \"\"\"Count sentences.\"\"\"\n"
    "    return len(re.findall(r\"[.!?]+(?=\\s|$)\", text))\n"
    "EOF\n"
    "cat > src/pkg/__init__.py <<'EOF'\n"
    "from .sentences import sentence_count\n"
    "EOF\n"
    "cat > tests/test_sentences.py <<'EOF'\n"
    "import unittest\n"
    "\n"
    "class T(unittest.TestCase):\n"
    "    def test_it(self):\n"
    "        self.assertEqual(1, 1)\n"
    "EOF\n"
    "PYTHONPATH=src python3 -m unittest discover -s tests 2>&1 | tail -4"
)


class _Builder:
    def __init__(self, store: TraceStore, session_id: str, *, branch: str = BRANCH,
                 agent: str = "claude-code", start: float = 1000.0):
        self.store = store
        self.session_id = session_id
        self.ts = start
        self.store.create_session(SessionMeta(
            session_id=session_id,
            started_at=start,
            ended_at=start + 500,
            agent_name="claude-code",
            attribution={"git_branch": branch},
        ))
        self.events: list[TraceEvent] = []
        self.add(EventType.SESSION_START, mode=f"{agent}-hooks", cwd=CWD,
                 redaction="on")

    def add(self, kind: EventType, parent: str = "", redacted: bool = False, **data) -> TraceEvent:
        self.ts += 1
        event = TraceEvent(event_type=kind, timestamp=self.ts, session_id=self.session_id,
                           parent_id=parent, data=data, redacted=redacted)
        self.events.append(event)
        return event

    def prompt(self, text: str = SECRET_PROMPT, **data) -> TraceEvent:
        return self.add(EventType.USER_PROMPT, prompt=text, **data)

    def tool(self, name: str, result: str | None = "ok", failed: bool = False, **args) -> TraceEvent:
        call = self.add(EventType.TOOL_CALL, tool_name=name, arguments=args)
        if failed:
            self.add(EventType.ERROR, parent=call.event_id, tool_name=name, error="boom")
        elif result is not None:
            self.add(EventType.TOOL_RESULT, parent=call.event_id, tool_name=name, result=result)
        return call

    def save(self, end: bool = True) -> str:
        if end:
            self.add(EventType.ASSISTANT_RESPONSE, text="done", hook_event="stop", provider="claude")
            self.add(EventType.SESSION_END)
        for event in self.events:
            self.store.append_event(self.session_id, event)
        return self.session_id


def _good_session(store: TraceStore, session_id: str = "good0001aaaa") -> str:
    b = _Builder(store, session_id)
    b.prompt()
    b.tool("Read", file_path=f"{CWD}/src/auth.py")
    b.tool("Edit", file_path=f"{CWD}/src/auth.py")
    b.tool("Bash", result="{'stdout': 'ok', 'exit_code': 0}",
           command="cd repo && python -m pytest tests -q")
    b.prompt("also run the linter", permission_mode="default")
    b.tool("Bash", result="Exited with exit code 0", command="ruff check src")
    return b.save()


class _StoreCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = TraceStore(self.tmp.name)
        cwd = os.getcwd()
        os.chdir(self.tmp.name)
        self.addCleanup(os.chdir, cwd)

    def report(self, *session_ids: str, link: str = "branch", threshold: Threshold | None = None):
        linked = [LinkedSession(sid, link) for sid in session_ids]
        return build_report(self.store, linked, threshold or Threshold(), branch=BRANCH)

    def finding(self, report, name: str) -> Finding:
        return next(f for f in report.findings if f.name == name)


class TestClassifyValidation(unittest.TestCase):
    def test_known_runners_and_wrappers(self):
        cases = {
            "python -m pytest -q": [("pytest", "test")],
            "cd x && uv run pytest tests": [("pytest", "test")],
            "CI=1 npm test": [("npm test", "test")],
            "npx eslint . && npx tsc --noEmit": [("eslint", "check"), ("tsc", "check")],
            "go vet ./... ; go test ./...": [("go vet", "check"), ("go test", "test")],
            "/usr/bin/pytest": [("pytest", "test")],
            "bundle exec rspec": [("rspec", "test")],
        }
        for command, expected in cases.items():
            with self.subTest(command=command):
                self.assertEqual(classify_validation(command), expected)

    def test_non_validation_commands(self):
        for command in ("echo pytest", "ls -la", "git status", "cat pytest.ini", ""):
            with self.subTest(command=command):
                self.assertEqual(classify_validation(command), [])

    def test_unbalanced_quotes_do_not_raise(self):
        self.assertEqual(classify_validation("pytest -k 'oops"), [("pytest", "test")])

    def test_multi_line_script_finds_run_after_heredocs(self):
        # Shape of a real Copilot CLI call: heredocs write files, then the
        # last line runs the tests.
        self.assertEqual(classify_validation(MULTI_LINE_SCRIPT), [("unittest", "test")])

    def test_heredoc_bodies_are_not_commands(self):
        for command in (
            "cat > run.sh <<EOF\npytest -q\nEOF",
            "cat > run.sh <<'EOF'\npytest -q\nEOF\n",
            'cat > run.sh <<"EOF"\nnpm test\nEOF',
            "cat > run.sh <<-EOF\n\tgo test ./...\n\tEOF",
            "python3 - <<'PY'\nimport subprocess\nsubprocess.run(['pytest'])\nPY",
            "# pytest later\nls",
        ):
            with self.subTest(command=command):
                self.assertEqual(classify_validation(command), [])

    def test_newlines_separate_commands(self):
        self.assertEqual(
            classify_validation("cd repo\nruff check src\npytest -q"),
            [("ruff", "check"), ("pytest", "test")],
        )
        self.assertEqual(classify_validation("python -m \\\n  pytest -q"), [("pytest", "test")])


class TestThreshold(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        cwd = os.getcwd()
        os.chdir(self.tmp.name)
        self.addCleanup(os.chdir, cwd)

    def _finding(self, status: str, confidence: str, name: str = "validation") -> Finding:
        return Finding(name, status, confidence, "src", "detail")

    def test_default_fails_only_missing_at_medium_or_higher(self):
        threshold = load_threshold()
        self.assertEqual(threshold.source, "default")
        self.assertTrue(threshold.fails(self._finding("missing", "medium")))
        self.assertTrue(threshold.fails(self._finding("missing", "high")))
        self.assertFalse(threshold.fails(self._finding("missing", "low")))
        for status in ("present", "partial", "inconclusive"):
            for confidence in ("low", "medium", "high"):
                self.assertFalse(threshold.fails(self._finding(status, confidence)))

    def test_config_file_review_block(self):
        Path(".agent-strace.yaml").write_text(
            "budget:\n  weekly: 10\nreview:\n  fail_on: missing, partial  # strict\n"
            "  min_confidence: high\n  findings: [validation, rationale]\n"
        )
        threshold = load_threshold()
        self.assertEqual(threshold.fail_on, ("partial", "missing"))
        self.assertEqual(threshold.min_confidence, "high")
        self.assertEqual(threshold.findings, ("rationale", "validation"))
        self.assertEqual(threshold.source, ".agent-strace.yaml")
        self.assertFalse(threshold.fails(self._finding("missing", "high", "scope")))

    def test_flags_override_config(self):
        Path(".agent-strace.yaml").write_text("review:\n  fail_on: partial\n")
        threshold = load_threshold(fail_on="inconclusive", min_confidence="low")
        self.assertEqual(threshold.fail_on, ("inconclusive",))
        self.assertEqual(threshold.min_confidence, "low")
        self.assertEqual(threshold.source, ".agent-strace.yaml + flags")

    def test_json_config_and_none(self):
        Path("cfg.json").write_text(json.dumps({"review": {"fail_on": "none"}}))
        threshold = load_threshold("cfg.json")
        self.assertEqual(threshold.fail_on, ())
        self.assertEqual(threshold.describe(), "never fails")
        self.assertFalse(threshold.fails(self._finding("missing", "high")))

    def test_invalid_values_raise(self):
        with self.assertRaises(ReviewConfigError):
            load_threshold(fail_on="present")
        with self.assertRaises(ReviewConfigError):
            load_threshold(fail_on="bad")
        with self.assertRaises(ReviewConfigError):
            load_threshold(min_confidence="certain")
        with self.assertRaises(ReviewConfigError):
            load_threshold(findings="vibes")
        with self.assertRaises(ReviewConfigError):
            load_threshold("missing.yaml")
        Path(".agent-strace.yaml").write_text("review:\n  threshold: 3\n")
        with self.assertRaises(ReviewConfigError):
            load_threshold()


class TestFindings(_StoreCase):
    def test_good_session(self):
        report = self.report(_good_session(self.store))
        statuses = {f.name: (f.status, f.confidence) for f in report.findings}
        self.assertEqual(statuses["rationale"], ("present", "high"))
        self.assertEqual(statuses["scope"], ("present", "medium"))
        self.assertEqual(statuses["validation"], ("present", "high"))
        self.assertEqual(statuses["stewardship"], ("present", "medium"))
        self.assertEqual(statuses["evidence"][0], "partial")
        self.assertTrue(report.passed)
        self.assertEqual(report.exit_code, 0)
        validation = self.finding(report, "validation")
        self.assertEqual(
            validation.facts["latest"],
            [
                {"label": "pytest", "kind": "test", "outcome": "passed", "exit_code": 0,
                 "changed_after": 0},
                {"label": "ruff", "kind": "check", "outcome": "passed", "exit_code": 0,
                 "changed_after": 0},
            ],
        )
        stewardship = self.finding(report, "stewardship")
        self.assertEqual(stewardship.facts["human_turns"], 2)
        self.assertEqual(stewardship.facts["interventions"], 1)
        self.assertEqual(stewardship.facts["permission_modes"], ["default"])
        self.assertIn("Redaction: on.", self.finding(report, "evidence").detail)

    def test_no_validation_fails_default_threshold(self):
        b = _Builder(self.store, "notest000001")
        b.prompt()
        b.tool("Edit", file_path=f"{CWD}/src/auth.py")
        b.tool("Bash", command="git status")
        report = self.report(b.save())
        validation = self.finding(report, "validation")
        self.assertEqual((validation.status, validation.confidence), ("missing", "medium"))
        self.assertFalse(report.passed)
        self.assertEqual([f.name for f in report.failing], ["validation"])

    def test_failed_and_stale_validation_is_partial(self):
        b = _Builder(self.store, "failed000001")
        b.prompt()
        b.tool("Bash", failed=True, command="pytest")
        report = self.report(b.save())
        validation = self.finding(report, "validation")
        self.assertEqual(validation.status, "partial")
        self.assertIn("pytest failed", validation.detail)

        b = _Builder(self.store, "stale0000001", start=5000)
        b.prompt()
        b.tool("Bash", result="exit_code: 0", command="pytest")
        b.tool("Edit", file_path=f"{CWD}/src/auth.py")
        report = self.report(b.save())
        validation = self.finding(report, "validation")
        self.assertEqual(validation.status, "partial")
        self.assertEqual(validation.facts["writes_after_last_run"], 1)

    def test_status_less_result_is_unknown_and_success_flag_is_medium(self):
        b = _Builder(self.store, "nocode000001")
        b.prompt()
        b.tool("Bash", result="all good", command="pytest")
        validation = self.finding(self.report(b.save()), "validation")
        self.assertEqual((validation.status, validation.confidence), ("partial", "medium"))
        self.assertEqual(validation.facts["latest"][0]["outcome"], "unknown")
        self.assertIn("pytest outcome unknown (no exit status)", validation.detail)

        b = _Builder(self.store, "flag00000001", start=5000)
        b.prompt()
        call = b.add(EventType.TOOL_CALL, tool_name="Bash", arguments={"command": "pytest"})
        b.add(EventType.TOOL_RESULT, parent=call.event_id, result="ok", success=True)
        validation = self.finding(self.report(b.save()), "validation")
        self.assertEqual((validation.status, validation.confidence), ("present", "medium"))

    def test_scope_lists_ungrounded_paths(self):
        b = _Builder(self.store, "scope0000001")
        b.prompt()
        b.tool("Edit", file_path=f"{CWD}/src/auth.py")
        b.tool("Write", file_path=f"{CWD}/src/unrelated.py")
        scope = self.finding(self.report(b.save()), "scope")
        self.assertEqual(scope.status, "partial")
        self.assertEqual(scope.facts["ungrounded"], ["src/unrelated.py"])
        self.assertEqual(scope.facts["grounded"], 1)

    def test_no_writes_is_inconclusive(self):
        b = _Builder(self.store, "nowrite00001")
        b.prompt()
        scope = self.finding(self.report(b.save()), "scope")
        self.assertEqual((scope.status, scope.confidence), ("inconclusive", "low"))

    def test_scope_reads_shell_redirection_in_bash_only_session(self):
        # Demo shape from a Copilot CLI run: every file is written through
        # bash heredocs, so the trace has no edit or create events.
        b = _Builder(self.store, "shellwrite01", agent="copilot")
        b.prompt("Add a sentence counter in src/pkg/sentences.py with tests")
        b.tool("bash", command=f"cd {CWD} && ls -R src tests && cat README.md",
               description="Inspect")
        b.tool("bash", result="Ran 1 test\n\nOK",
               command=MULTI_LINE_SCRIPT.replace("/work/demo", CWD),
               description="Write files and run tests")
        report = self.report(b.save())
        scope = self.finding(report, "scope")
        self.assertEqual((scope.status, scope.confidence), ("partial", "low"))
        self.assertEqual(scope.facts["files_written"], 3)
        self.assertEqual(scope.facts["shell_writes"], 3)
        self.assertEqual(scope.facts["ungrounded"], ["src/pkg/__init__.py", "tests/test_sentences.py"])
        self.assertIn("shell redirection", scope.detail)
        self.assertIn("shell redirection", scope.source)
        self.assertNotIn("scope", [f.name for f in report.failing])
        text = io.StringIO()
        format_text(report, text)
        for fragment in ("cat >", "EOF", "sentence_count", "cd /work"):
            self.assertNotIn(fragment, text.getvalue())
            self.assertNotIn(fragment, render_markdown(report))
            self.assertNotIn(fragment, json.dumps(report.to_dict()))

    def test_shell_writes_do_not_raise_explicit_write_confidence(self):
        b = _Builder(self.store, "shellmixed01")
        b.prompt()
        b.tool("Edit", file_path=f"{CWD}/src/auth.py")
        b.tool("Bash", command="echo x > notes.txt 2>&1; echo y > /dev/null")
        scope = self.finding(self.report(b.save()), "scope")
        self.assertEqual((scope.status, scope.confidence), ("partial", "medium"))
        self.assertEqual(scope.facts["shell_writes"], 1)
        self.assertEqual(scope.facts["ungrounded"], ["notes.txt"])

        b = _Builder(self.store, "shellsame001", start=5000)
        b.prompt()
        b.tool("Edit", file_path=f"{CWD}/src/auth.py")
        b.tool("Bash", command="cat src/auth.py | tee -a src/auth.py")
        scope = self.finding(self.report(b.save()), "scope")
        self.assertEqual((scope.status, scope.confidence), ("present", "medium"))
        self.assertEqual((scope.facts["files_written"], scope.facts["shell_writes"]), (1, 0))

    def test_shell_write_target_under_macos_tmp_alias(self):
        store_cwd = "/private/tmp/demo"
        b = _Builder(self.store, "shelltmp0001")
        b.events[0].data["cwd"] = store_cwd
        b.prompt("fix the app")
        b.tool("bash", command="cd /tmp/demo && cat > app.py <<'EOF'\nx = 1\nEOF")
        scope = self.finding(self.report(b.save()), "scope")
        self.assertEqual((scope.status, scope.confidence), ("missing", "low"))
        self.assertEqual(scope.facts["ungrounded"], ["app.py"])

    def test_shell_write_after_last_run_marks_validation_stale(self):
        b = _Builder(self.store, "shellstale01")
        b.prompt()
        b.tool("Bash", result="exit_code: 0", command="pytest")
        b.tool("Bash", command="cat > src/auth.py <<'EOF'\npass\nEOF")
        validation = self.finding(self.report(b.save()), "validation")
        self.assertEqual(validation.status, "partial")
        self.assertEqual(validation.facts["writes_after_last_run"], 1)
        self.assertEqual(validation.facts["latest"][0]["changed_after"], 1)

    def test_rationale_missing_when_provider_captures_prompts(self):
        b = _Builder(self.store, "noprompt0001")
        b.tool("Bash", command="ls")
        report = self.report(b.save())
        rationale = self.finding(report, "rationale")
        self.assertEqual((rationale.status, rationale.confidence), ("missing", "high"))
        stewardship = self.finding(report, "stewardship")
        self.assertEqual((stewardship.status, stewardship.confidence), ("missing", "high"))

    def test_rationale_inconclusive_without_known_provider(self):
        b = _Builder(self.store, "unknown00001", agent="nobody")
        rationale = self.finding(self.report(b.save(end=False)), "rationale")
        self.assertEqual((rationale.status, rationale.confidence), ("inconclusive", "low"))

    def test_redacted_prompt_counts_as_present(self):
        b = _Builder(self.store, "redact000001")
        b.add(EventType.USER_PROMPT, redacted=True, prompt="[REDACTED]")
        rationale = self.finding(self.report(b.save()), "rationale")
        self.assertEqual(rationale.status, "present")
        self.assertEqual(rationale.facts["redacted_prompts"], 1)

    def test_single_prompt_is_partial_stewardship_and_approvals_count(self):
        b = _Builder(self.store, "single000001")
        b.prompt()
        sid = b.save()
        self.assertEqual(self.finding(self.report(sid), "stewardship").status, "partial")
        save_request(self.store, ApprovalRequest(session_id=sid, state="approved"))
        save_request(self.store, ApprovalRequest(session_id="other", state="denied"))
        stewardship = self.finding(self.report(sid), "stewardship")
        self.assertEqual(stewardship.status, "present")
        self.assertEqual(stewardship.facts["approvals_approved"], 1)
        self.assertEqual(stewardship.facts["approvals_denied"], 0)

    def test_weak_link_lowers_evidence_confidence(self):
        evidence = self.finding(self.report(_good_session(self.store), link="commit_time"), "evidence")
        self.assertEqual(evidence.confidence, "low")
        self.assertEqual(evidence.facts["link"], ["commit_time"])

    def test_no_sessions_is_inconclusive_and_passes_by_default(self):
        report = self.report()
        self.assertEqual({f.status for f in report.findings}, {"inconclusive"})
        self.assertTrue(report.passed)
        strict = Threshold(fail_on=("inconclusive",), min_confidence="low")
        self.assertFalse(self.report(threshold=strict).passed)

    def test_link_sessions(self):
        self.assertEqual(link_sessions(self.store, branch=BRANCH), [])
        sid = _good_session(self.store)
        _Builder(self.store, "otherbranch1", branch="main", start=9000).save()
        self.assertEqual(link_sessions(self.store, branch=BRANCH), [LinkedSession(sid, "branch")])
        self.assertEqual(link_sessions(self.store, ["good0001"]), [LinkedSession(sid, "explicit")])

    def test_output_is_deterministic_and_never_shows_prompt(self):
        sid = _good_session(self.store)
        first = self.report(sid)
        second = self.report(sid)
        self.assertEqual(first.to_dict(), second.to_dict())
        rendered = json.dumps(first.to_dict()) + render_markdown(first)
        self.assertNotIn("zebra-quartz-77", rendered)
        self.assertNotIn("python -m pytest", rendered)
        self.assertNotIn("\u2014", rendered)


class TestValidationOutcomes(_StoreCase):
    def _validate(self, sid: str, command: str, *results: dict, failed: bool = False,
                 before: bool = False):
        b = _Builder(self.store, sid)
        b.prompt()
        call = TraceEvent(event_type=EventType.TOOL_CALL, timestamp=b.ts + 5, session_id=sid,
                          data={"tool_name": "Bash", "arguments": {"command": command}})
        if not before:
            b.events.append(call)
        for data in results:
            b.add(EventType.ERROR if failed else EventType.TOOL_RESULT,
                  parent=call.event_id, tool_name="Bash", **data)
            b.events[-1].timestamp = call.timestamp + 1
        if before:
            b.events.append(call)
        validation = self.finding(self.report(b.save()), "validation")
        return validation, validation.facts["latest"][0]

    def test_single_command_owns_exit_code(self):
        validation, run = self._validate("single000001", "pytest -k 'a && b'", {"result": "exit_code: 0"})
        self.assertEqual((validation.status, run["outcome"], run["exit_code"]), ("present", "passed", 0))
        _, run = self._validate("single000002", "pytest", {"result": "exit code 1"})
        self.assertEqual((run["outcome"], run["exit_code"]), ("failed", 1))
        _, run = self._validate("single000003", "pytest", {"error": "boom"}, failed=True)
        self.assertEqual(run["outcome"], "failed")

    def test_compound_shell_shapes_are_unknown(self):
        for index, command in enumerate((
            "pytest || true", "pytest | tee log.txt", "pytest > log.txt", "pytest &",
            "pytest; echo done", "pytest $(cat args)", "pytest\necho done",
        )):
            with self.subTest(command=command):
                validation, run = self._validate(f"compound{index:04d}", command,
                                                {"result": "exit_code: 0"})
                self.assertEqual(run["outcome"], "unknown")
                self.assertIsNone(run["exit_code"])
                self.assertEqual(validation.status, "partial")

    def test_multi_line_script_is_partial_not_missing(self):
        validation, run = self._validate(
            "multiline001", MULTI_LINE_SCRIPT,
            {"result": "Ran 7 tests in 0.000s\n\nOK\n<shellId: 1 completed with exit code 0>"},
        )
        self.assertEqual((validation.status, validation.confidence), ("partial", "medium"))
        self.assertEqual((run["label"], run["outcome"]), ("unittest", "unknown"))
        self.assertIsNone(run["exit_code"])
        report = self.report("multiline001")
        self.assertNotIn("validation", [f.name for f in report.failing])
        text = io.StringIO()
        format_text(report, text)
        for fragment in ("sentence_count", "PYTHONPATH", "EOF", "tail -4"):
            self.assertNotIn(fragment, text.getvalue())
            self.assertNotIn(fragment, render_markdown(report))

    def test_and_chain_carries_only_a_pass(self):
        validation, run = self._validate("chain0000001", "cd repo && pytest", {"result": "exit_code: 0"})
        self.assertEqual((validation.status, run["outcome"]), ("present", "passed"))
        _, run = self._validate("chain0000002", "cd repo && pytest", {"result": "exit_code: 2"})
        self.assertEqual(run["outcome"], "unknown")
        self.assertIn("failing part not known", self.finding(self.report("chain0000002"), "validation").detail)

    def test_result_must_be_single_valid_and_later(self):
        cases = {
            "none00000001": ((), False, "no valid result recorded"),
            "dupe00000001": (({"result": "exit_code: 0"}, {"result": "exit_code: 0"}), False,
                             "more than one result"),
            "before000001": (({"result": "exit_code: 0"},), True, "no valid result recorded"),
            "conflict0001": (({"result": "exit_code: 0", "success": False},), False,
                             "conflicting status"),
            "codes0000001": (({"result": "exit code 0 then exit code 3"},), False,
                             "conflicting status"),
            "errzero00001": (({"error": "exit_code: 0"},), False, "conflicting status"),
        }
        for sid, (results, before, reason) in cases.items():
            with self.subTest(sid=sid):
                failed = sid.startswith("errzero")
                validation, run = self._validate(sid, "pytest", *results, failed=failed, before=before)
                self.assertEqual(run["outcome"], "unknown")
                self.assertEqual(validation.status, "partial")
                self.assertIn(reason, validation.detail)

    def test_freshness_is_global_across_sessions(self):
        a = _Builder(self.store, "aaaa00000001", start=1000)
        a.prompt()
        a.tool("Bash", result="exit_code: 0", command="pytest")
        a.save()
        b = _Builder(self.store, "bbbb00000001", start=2000)
        b.prompt()
        b.tool("Edit", file_path=f"{CWD}/src/auth.py")
        b.save()
        for order in (("aaaa00000001", "bbbb00000001"), ("bbbb00000001", "aaaa00000001")):
            with self.subTest(order=order):
                report = self.report(*order)
                validation = self.finding(report, "validation")
                self.assertEqual(validation.status, "partial")
                self.assertEqual(validation.facts["latest"][0]["changed_after"], 1)
                self.assertEqual(validation.facts["writes_after_last_run"], 1)
                self.assertIn("Files changed after the last run of: pytest (1 file)", validation.detail)
        first = self.finding(self.report("aaaa00000001", "bbbb00000001"), "validation")
        second = self.finding(self.report("bbbb00000001", "aaaa00000001"), "validation")
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_validation_in_later_session_clears_staleness(self):
        a = _Builder(self.store, "cccc00000001", start=1000)
        a.prompt()
        a.tool("Edit", file_path=f"{CWD}/src/auth.py")
        a.save()
        b = _Builder(self.store, "dddd00000001", start=2000)
        b.prompt()
        b.tool("Bash", result="exit_code: 0", command="pytest")
        b.save()
        validation = self.finding(self.report("cccc00000001", "dddd00000001"), "validation")
        self.assertEqual((validation.status, validation.confidence), ("present", "high"))


class TestConfigFailsClosed(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        cwd = os.getcwd()
        os.chdir(self.tmp.name)
        self.addCleanup(os.chdir, cwd)

    def _yaml(self, text: str) -> Threshold:
        Path(".agent-strace.yaml").write_text(text)
        return load_threshold()

    def test_block_and_flow_lists(self):
        threshold = self._yaml(
            "budget:\n  weekly: 3\nreview:\n  fail_on:\n    - missing\n    - partial  # strict\n"
            "  findings: [validation, 'scope']\nother: 1\n"
        )
        self.assertEqual(threshold.fail_on, ("partial", "missing"))
        self.assertEqual(threshold.findings, ("scope", "validation"))

    def test_only_literal_none_disables(self):
        self.assertEqual(self._yaml("review:\n  fail_on: none\n").fail_on, ())
        self.assertEqual(self._yaml("review:\n  fail_on: \"none\"\n").fail_on, ())
        bad = (
            "review:\n  fail_on:\n",
            "review:\n  fail_on: null\n",
            "review:\n  fail_on: ~\n",
            "review:\n  fail_on: []\n",
            "review:\n  fail_on: [none]\n",
            "review:\n  fail_on:\n    - none\n",
            "review:\n  fail_on: missing, none\n",
            "review:\n  fail_on: missing,,partial\n",
            "review:\n  findings:\n",
            "review:\n  findings: []\n",
            "review:\n  min_confidence:\n",
        )
        for text in bad:
            with self.subTest(text=text), self.assertRaises(ReviewConfigError):
                self._yaml(text)

    def test_unsupported_structures_fail(self):
        bad = (
            "review: {fail_on: missing}\n",
            "review:\n  fail_on: {missing: true}\n",
            "review:\n  fail_on:\n    missing: 1\n",
            "review:\n  fail_on: [missing\n",
            "review:\n  fail_on: &anchor missing\n",
            "review:\n  fail_on: missing\n  fail_on: partial\n",
            "review:\n  - missing\n",
            "review:\n  fail_on: missing\n      deeper: 1\n",
        )
        for text in bad:
            with self.subTest(text=text), self.assertRaises(ReviewConfigError):
                self._yaml(text)

    def test_json_types(self):
        Path("cfg.json").write_text(json.dumps({"review": {"fail_on": ["missing", "partial"]}}))
        self.assertEqual(load_threshold("cfg.json").fail_on, ("partial", "missing"))
        for block in ({"fail_on": None}, {"fail_on": []}, {"fail_on": [1]}, {"fail_on": True},
                      {"min_confidence": None}, {"findings": None}, None, "missing"):
            with self.subTest(block=block), self.assertRaises(ReviewConfigError):
                Path("cfg.json").write_text(json.dumps({"review": block}))
                load_threshold("cfg.json")
        Path("cfg.json").write_text("{not json")
        with self.assertRaises(ReviewConfigError):
            load_threshold("cfg.json")

    def test_empty_flag_does_not_disable(self):
        for value in ("", " ", ","):
            with self.subTest(value=value), self.assertRaises(ReviewConfigError):
                load_threshold(fail_on=value)


class TestPullRequestHead(_StoreCase):
    SHA = "a" * 40

    def _env(self, **values):
        keys = ("GITHUB_EVENT_PATH", "GITHUB_HEAD_REF", "CI_MERGE_REQUEST_IID",
                "CI_MERGE_REQUEST_SOURCE_BRANCH_SHA", "CI_COMMIT_SHA")
        env = {key: "" for key in keys}
        env.update(values)
        return patch.dict(os.environ, env)

    def _event(self, payload) -> str:
        path = Path(self.tmp.name) / "event.json"
        path.write_text(json.dumps(payload))
        return str(path)

    def test_github_pull_request_uses_event_head_sha(self):
        calls = []

        def fake_git(_cwd, *args):
            calls.append(args)
            return "195"

        event = self._event({"pull_request": {"head": {"sha": self.SHA}}})
        with self._env(GITHUB_EVENT_PATH=event, GITHUB_HEAD_REF="feature"), \
                patch("agent_trace.pr_comment._git", side_effect=fake_git):
            self.assertEqual(pull_request_head_sha(), (True, self.SHA))
            self.assertEqual(head_commit_timestamp(), 195.0)
        self.assertEqual(calls, [("show", "-s", "--format=%ct", f"{self.SHA}^{{commit}}")])

    def test_pull_request_without_resolvable_head_skips_commit_time(self):
        _Builder(self.store, "far000000001", branch="", start=100).save()
        _Builder(self.store, "near00000001", branch="", start=195).save()
        cases = (
            {"GITHUB_EVENT_PATH": self._event({"pull_request": {"head": {}}})},
            {"GITHUB_EVENT_PATH": self._event({"pull_request": {"head": {"sha": "--bad"}}})},
            {"GITHUB_HEAD_REF": "feature"},
        )
        for env in cases:
            with self.subTest(env=env), self._env(**env), \
                    patch("agent_trace.pr_comment._git", return_value="200") as git:
                self.assertIsNone(head_commit_timestamp())
                git.assert_not_called()
                ids, link = select_sessions_with_link(self.store, branch="missing")
                self.assertEqual(link, "latest")
        event = self._event({"pull_request": {"head": {"sha": self.SHA}}})
        with self._env(GITHUB_EVENT_PATH=event), \
                patch("agent_trace.pr_comment._git", return_value=""):
            self.assertIsNone(head_commit_timestamp())
            self.assertEqual(select_sessions_with_link(self.store, branch="missing")[1], "latest")
        with self._env(GITHUB_EVENT_PATH=event), \
                patch("agent_trace.pr_comment._git", return_value="200"):
            self.assertEqual(
                select_sessions_with_link(self.store, branch="missing"),
                (["near00000001"], "commit_time"),
            )

    def test_local_and_push_use_head(self):
        push = self._event({"ref": "refs/heads/main"})
        for env in ({}, {"GITHUB_EVENT_PATH": push}):
            with self.subTest(env=env), self._env(**env), \
                    patch("agent_trace.pr_comment._git", return_value="200") as git:
                self.assertEqual(head_commit_timestamp(), 200.0)
                git.assert_called_once_with(".", "log", "-1", "--format=%ct")

    def test_gitlab_merge_request_prefers_source_branch_sha(self):
        other = "b" * 40
        with self._env(CI_MERGE_REQUEST_IID="5", CI_MERGE_REQUEST_SOURCE_BRANCH_SHA=self.SHA,
                       CI_COMMIT_SHA=other):
            self.assertEqual(pull_request_head_sha(), (True, self.SHA))
        with self._env(CI_MERGE_REQUEST_IID="5", CI_COMMIT_SHA=other):
            self.assertEqual(pull_request_head_sha(), (True, other))
        with self._env(CI_MERGE_REQUEST_IID="5"):
            self.assertEqual(pull_request_head_sha(), (True, ""))


class TestCommands(_StoreCase):
    def _run(self, argv: list[str]):
        args = build_parser().parse_args(["--trace-dir", self.tmp.name, *argv])
        out, err = io.StringIO(), io.StringIO()
        handler = cmd_pr_comment if args.command == "pr-comment" else cmd_review
        with redirect_stdout(out), redirect_stderr(err), \
                patch("agent_trace.pr_comment.current_branch", return_value=BRANCH), \
                patch("agent_trace.reviewability.current_branch", return_value=BRANCH):
            code = handler(args)
        return code, out.getvalue(), err.getvalue()

    def test_review_formats(self):
        _good_session(self.store)
        code, out, _ = self._run(["review"])
        self.assertEqual(code, 0)
        self.assertIn("Reviewability: pass", out)
        self.assertIn("session git branch matches the PR branch", out)
        self.assertNotIn("zebra-quartz-77", out)

        code, out, _ = self._run(["review", "--format", "json"])
        payload = json.loads(out)
        self.assertEqual(payload["result"], "pass")
        self.assertEqual([f["name"] for f in payload["findings"]], list(FINDINGS))
        for finding in payload["findings"]:
            self.assertIn(finding["status"], ("present", "partial", "missing", "inconclusive"))
            self.assertIn(finding["confidence"], ("low", "medium", "high"))
            self.assertTrue(finding["source"])

        code, out, _ = self._run(["review", "--format", "markdown"])
        self.assertTrue(out.startswith("## agent-trace reviewability"))
        self.assertIn("| Validation | present | high |", out)

    def test_review_threshold_exit_codes(self):
        _good_session(self.store)
        code, out, _ = self._run(["review", "--fail-on", "partial", "--format", "json"])
        self.assertEqual(code, 1)
        self.assertIn("evidence", json.loads(out)["failing"])
        code, _, err = self._run(["review", "--fail-on", "wrong"])
        self.assertEqual(code, 2)
        self.assertIn("invalid fail_on", err)
        code, _, err = self._run(["review", "nope"])
        self.assertEqual(code, 2)
        self.assertIn("Session not found", err)

    def test_review_with_no_sessions(self):
        code, out, _ = self._run(["review"])
        self.assertEqual(code, 0)
        self.assertIn("none (no session linked)", out)

    def test_pr_comment_review_dry_run(self):
        b = _Builder(self.store, "notest000002")
        b.prompt()
        b.tool("Bash", command="git status")
        b.save()
        code, out, _ = self._run(["pr-comment", "--dry-run", "--review"])
        self.assertEqual(code, 1)
        self.assertTrue(out.startswith(COMMENT_MARKER))
        self.assertEqual(out.count(COMMENT_MARKER), 1)
        self.assertIn("### Reviewability", out)
        self.assertIn("**missing** (fails)", out)
        self.assertLess(out.index("### Reviewability"), out.index("### What the agent did"))

        code, out, _ = self._run(["pr-comment", "--dry-run"])
        self.assertEqual(code, 0)
        self.assertNotIn("### Reviewability", out)

        code, _, _ = self._run(["pr-comment", "--dry-run", "--review", "--fail-on", "none"])
        self.assertEqual(code, 0)

    def test_pr_comment_review_posts_then_fails(self):
        _Builder(self.store, "notest000003").save()
        posted = {}

        def fake_post(target, body, token, **_kwargs):
            posted["body"] = body
            return "created"

        with patch.dict(os.environ, {"GITHUB_TOKEN": "t"}), \
                patch("agent_trace.pr_comment.resolve_target") as resolve, \
                patch("agent_trace.pr_comment.post_or_update_comment", side_effect=fake_post):
            resolve.return_value.number = 7
            code, out, err = self._run(["pr-comment", "--review", "--fail-on", "missing",
                                        "--min-confidence", "low"])
        self.assertEqual(code, 1)
        self.assertIn("Created agent-trace comment on PR #7", out)
        self.assertIn("reviewability check failed", err)
        self.assertIn("### Reviewability", posted["body"])


class TestArtifacts(_StoreCase):
    def _write(self, name: str, payload) -> str:
        path = Path(self.tmp.name) / name
        path.write_text(payload if isinstance(payload, str) else json.dumps(payload))
        return str(path)

    def test_read_cognitive_debt_uses_schema_fields(self):
        good = {"schema_version": 1, "metric": "cognitive_debt"}
        cases = {
            "clean": ({**good, "status": "clean"}, "pass"),
            "partial": ({**good, "status": "partial"}, "pass"),
            "above": ({**good, "status": "above_threshold"}, "fail"),
            "unknown": ({**good, "status": "unknown", "color": "brightgreen"}, "unavailable"),
            "v2": ({**good, "schema_version": 2, "status": "clean"}, "unavailable"),
            "metric": ({"schema_version": 1, "metric": "other", "status": "clean"}, "unavailable"),
            "badge": ({"schemaVersion": 1, "color": "green", "message": "0%"}, "unavailable"),
            "odd": ({**good, "status": "great"}, "unavailable"),
            "list": ([], "unavailable"),
            "broken": ("{not json", "unavailable"),
        }
        for name, (payload, result) in cases.items():
            with self.subTest(name=name):
                evidence = read_cognitive_debt(self._write(f"{name}.json", payload))
                self.assertEqual(evidence.result, result)
        missing = read_cognitive_debt(Path(self.tmp.name) / "absent.json")
        self.assertEqual((missing.result, missing.reason), ("unavailable", "report not generated"))

    def test_artifact_flags(self):
        parser = build_parser()
        args = parser.parse_args(["review", "--artifact-url", "http://x"])
        with self.assertRaises(ReviewConfigError):
            artifacts_from_args(args)
        args = parser.parse_args(["review", "--artifact-url", "https://x/a)b"])
        with self.assertRaises(ReviewConfigError):
            artifacts_from_args(args)
        self.assertIsNone(artifacts_from_args(parser.parse_args(["review"])))
        args = parser.parse_args(["review", "--cognitive-debt-mode", "check"])
        artifacts = artifacts_from_args(args)
        self.assertTrue(artifacts.cognitive_debt_fails)

    def test_markdown_never_claims_an_upload(self):
        _good_session(self.store)
        packet = self._write("evidence-packet.html", "<html></html>")
        url = "https://github.com/o/r/actions/runs/1/artifacts/2"
        states = {
            Artifacts(packet): "`evidence-packet.html` (local file, not uploaded)",
            Artifacts(packet, url=url): f"`evidence-packet.html` in the [workflow artifact]({url})",
            Artifacts(str(Path(self.tmp.name) / "none.html"), url=url): "Evidence packet: not generated.",
            Artifacts(self._write("empty.html", "")): "Evidence packet: not generated.",
        }
        for artifacts, expected in states.items():
            with self.subTest(expected=expected):
                report = build_report(self.store, [LinkedSession("good0001aaaa", "branch")],
                                      Threshold(), artifacts=artifacts)
                markdown = render_markdown(report)
                self.assertIn(expected, markdown)
                self.assertNotIn(self.tmp.name, markdown)
                if "not uploaded" in expected or "not generated" in expected:
                    self.assertNotIn("in the [workflow artifact]", markdown)

    def test_cognitive_check_changes_result_not_findings(self):
        _good_session(self.store)
        unknown = read_cognitive_debt(self._write(
            "debt.json", {"schema_version": 1, "metric": "cognitive_debt", "status": "unknown"}))
        linked = [LinkedSession("good0001aaaa", "branch")]
        report = build_report(self.store, linked, Threshold(fail_on=()),
                              artifacts=Artifacts(cognitive_debt=unknown))
        self.assertTrue(report.passed)
        report = build_report(self.store, linked, Threshold(fail_on=()),
                              artifacts=Artifacts(cognitive_debt=unknown, cognitive_debt_mode="check"))
        self.assertFalse(report.passed)
        self.assertTrue(report.review_passed)
        self.assertEqual(report.to_dict()["result"], "fail")
        self.assertEqual(report.to_dict()["review_result"], "pass")
        self.assertIn("**Fails the check.**", render_markdown(report))


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "pr-evidence.sh"

_SHIM = """#!{python}
import json, os, sys
sys.path.insert(0, {src!r})
args = sys.argv[1:]
with open(os.environ["SHIM_LOG"], "a", encoding="utf-8") as log:
    log.write(json.dumps(args) + "\\n")
command = next((a for a in args if a in ("share", "cognitive-debt", "pr-comment", "review")), "")
if command == "share" and os.environ.get("SHIM_FAIL_SHARE"):
    sys.exit(1)
if command == "cognitive-debt" and os.environ.get("SHIM_COGNITIVE_STATUS"):
    out = args[args.index("--output") + 1]
    if args[args.index("--format") + 1] == "json":
        payload = {{"schema_version": 1, "metric": "cognitive_debt",
                   "status": os.environ["SHIM_COGNITIVE_STATUS"]}}
    else:
        payload = {{"schemaVersion": 1, "label": "cognitive debt", "message": "x", "color": "green"}}
    open(out, "w", encoding="utf-8").write(json.dumps(payload))
    sys.exit(0)
if command == "pr-comment":
    sys.exit(int(os.environ.get("SHIM_COMMENT_EXIT", "0")))
from agent_trace.cli import main
sys.argv = ["agent-strace"] + args
main()
"""


class TestAction(unittest.TestCase):
    def test_composite_action_wires_evidence_steps(self):
        action = (ROOT / "action.yml").read_text()
        for name in ("run-eval", "review", "review-fail-on", "review-min-confidence",
                     "cognitive-debt"):
            self.assertIn(f"  {name}:\n", action)
        for name in ("passed", "summary-path", "review-passed", "review-result",
                     "review-markdown-path", "review-json-path", "evidence-packet-path",
                     "cognitive-debt-json-path", "cognitive-debt-badge-path",
                     "cognitive-debt-status", "cognitive-debt-result", "artifacts-dir",
                     "evidence-artifact-url"):
            self.assertIn(f"  {name}:\n", action)
        self.assertIn("if: ${{ inputs.run-eval == 'true' }}", action)
        self.assertIn("inputs.post-pr-comment == 'true' && inputs.review != 'true'", action)
        self.assertEqual(action.count("if: ${{ always() && inputs.review == 'true' }}"), 2)
        self.assertEqual(action.count("always() && inputs.review == 'true' && "), 2)
        self.assertIn('bash "$GITHUB_ACTION_PATH/scripts/pr-evidence.sh" prepare', action)
        self.assertIn('bash "$GITHUB_ACTION_PATH/scripts/pr-evidence.sh" check', action)
        self.assertIn("name: agent-trace-pr-evidence", action)
        self.assertIn("name: agent-trace-pr-review", action)
        self.assertIn("ARTIFACT_URL: ${{ steps.upload-evidence.outputs.artifact-url }}", action)
        self.assertEqual(action.count("continue-on-error: true"), 2)
        self.assertNotIn("--include-content", action)
        self.assertNotIn("\u2014", action)
        # Inputs reach the shell only through env, never inline in run blocks.
        for block in action.split("run: ")[1:]:
            body = block.split("\n    - name:")[0]
            self.assertNotIn("${{ inputs.review-fail-on }}", body.split("env:")[0])
        script = SCRIPT.read_text()
        self.assertNotIn("eval ", script)
        self.assertNotIn("\u2014", script)


@unittest.skipUnless(shutil.which("bash"), "bash is required")
class TestActionScript(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.work = root / "work dir"
        self.work.mkdir()
        self.trace_dir = self.work / "agent traces"
        self.store = TraceStore(str(self.trace_dir))
        self.out_dir = root / "runner temp" / "agent-trace-pr-evidence"
        bin_dir = root / "bin"
        bin_dir.mkdir()
        shim = bin_dir / "agent-strace"
        shim.write_text(_SHIM.format(python=sys.executable, src=str(ROOT / "src")))
        shim.chmod(0o755)
        self.log = root / "calls.log"
        self.output = root / "github_output"
        self.summary = root / "summary.md"
        self.env = {
            "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
            "HOME": self.tmp.name,
            "TMPDIR": self.tmp.name,
            "SHIM_LOG": str(self.log),
            "GITHUB_OUTPUT": str(self.output),
            "GITHUB_STEP_SUMMARY": str(self.summary),
            "GITHUB_HEAD_REF": BRANCH,
            "AGENT_TRACE_DIR": str(self.trace_dir),
            "PR_EVIDENCE_DIR": str(self.out_dir),
        }

    def run_script(self, mode: str, **env) -> subprocess.CompletedProcess:
        self.output.write_text("")
        merged = dict(self.env)
        merged.update(env)
        return subprocess.run(
            ["bash", str(SCRIPT), mode], cwd=self.work, env=merged,
            capture_output=True, text=True, timeout=120,
        )

    def outputs(self) -> dict[str, str]:
        pairs = [line.split("=", 1) for line in self.output.read_text().splitlines() if line]
        return dict(pairs)

    def calls(self, command: str) -> list[list[str]]:
        if not self.log.exists():
            return []
        rows = [json.loads(line) for line in self.log.read_text().splitlines()]
        return [row for row in rows if command in row]

    def test_full_flow_with_spaces_and_unknown_cognitive_status(self):
        _good_session(self.store)
        prepared = self.run_script("prepare")
        self.assertEqual(prepared.returncode, 0, prepared.stderr + prepared.stdout)
        out = self.outputs()
        self.assertEqual(out["session-id"], "good0001aaaa")
        packet = Path(out["evidence-packet-path"])
        self.assertEqual(packet, self.out_dir / "evidence-packet.html")
        self.assertTrue(packet.is_file())
        self.assertNotIn("zebra-quartz-77", packet.read_text())
        self.assertNotIn("--include-content", json.dumps(self.calls("share")))
        self.assertEqual(Path(out["cognitive-debt-json-path"]).name, "cognitive-debt.json")
        self.assertEqual(Path(out["cognitive-debt-badge-path"]).name, "cognitive-debt-badge.json")
        self.assertEqual(out["artifacts-dir"], str(self.out_dir))

        checked = self.run_script("check")
        self.assertEqual(checked.returncode, 0, checked.stderr + checked.stdout)
        out = self.outputs()
        self.assertEqual(out["review-result"], "pass")
        self.assertEqual(out["passed"], "true")
        self.assertEqual(out["cognitive-debt-status"], "unknown")
        self.assertEqual(out["cognitive-debt-result"], "unavailable")
        report = json.loads(Path(out["review-json-path"]).read_text())
        self.assertEqual(report["artifacts"]["evidence_packet"]["state"], "local")
        self.assertIsNone(report["artifacts"]["url"])
        markdown = Path(out["review-markdown-path"]).read_text()
        self.assertIn("`evidence-packet.html` (local file, not uploaded)", markdown)
        self.assertIn("Unavailable, not a pass. (report only)", markdown)
        self.assertNotIn("zebra-quartz-77", markdown)
        self.assertEqual(self.summary.read_text(), markdown)
        self.assertEqual(self.calls("pr-comment"), [])

        url = "https://github.com/o/r/actions/runs/1/artifacts/2"
        self.run_script("check", ARTIFACT_URL=url)
        markdown = Path(self.outputs()["review-markdown-path"]).read_text()
        self.assertIn(f"`evidence-packet.html` in the [workflow artifact]({url})", markdown)
        self.run_script("check", ARTIFACT_URL="javascript:alert(1)")
        self.assertNotIn("javascript", Path(self.outputs()["review-markdown-path"]).read_text())

    def test_unknown_cognitive_status_fails_check_mode(self):
        _good_session(self.store)
        for status, code, result in (("unknown", 1, "unavailable"), ("above_threshold", 1, "fail"),
                                     ("partial", 0, "pass"), ("clean", 0, "pass")):
            with self.subTest(status=status):
                env = {"SHIM_COGNITIVE_STATUS": status, "COGNITIVE_DEBT_MODE": "check"}
                self.assertEqual(self.run_script("prepare", **env).returncode, 0)
                checked = self.run_script("check", **env)
                self.assertEqual(checked.returncode, code, checked.stderr)
                out = self.outputs()
                self.assertEqual(out["review-result"], "pass" if code == 0 else "fail")
                self.assertEqual(out["cognitive-debt-status"], status)
                self.assertEqual(out["cognitive-debt-result"], result)
                report = json.loads(Path(out["review-json-path"]).read_text())
                self.assertEqual(report["review_result"], "pass")
                self.assertEqual(report["artifacts"]["cognitive_debt"]["fails_check"], code == 1)

    def test_badge_color_is_not_read(self):
        _good_session(self.store)
        env = {"SHIM_COGNITIVE_STATUS": "unknown", "COGNITIVE_DEBT_MODE": "check"}
        self.run_script("prepare", **env)
        badge = self.out_dir / "cognitive-debt-badge.json"
        self.assertEqual(json.loads(badge.read_text())["color"], "green")
        self.assertEqual(self.run_script("check", **env).returncode, 1)

    def test_packet_generation_failure(self):
        _good_session(self.store)
        prepared = self.run_script("prepare", SHIM_FAIL_SHARE="1")
        self.assertEqual(prepared.returncode, 0)
        self.assertIn("Evidence packet was not generated", prepared.stdout)
        self.assertNotIn("evidence-packet-path", self.outputs())
        self.assertFalse((self.out_dir / "evidence-packet.html").exists())
        checked = self.run_script("check", ARTIFACT_URL="https://example.test/a")
        self.assertEqual(checked.returncode, 0)
        markdown = Path(self.outputs()["review-markdown-path"]).read_text()
        self.assertIn("Evidence packet: not generated.", markdown)
        self.assertNotIn("in the [workflow artifact]", markdown)

    def test_cognitive_debt_false_skips_reports(self):
        _good_session(self.store)
        self.run_script("prepare", COGNITIVE_DEBT_MODE="false")
        self.assertEqual(self.calls("cognitive-debt"), [])
        self.assertNotIn("cognitive-debt-json-path", self.outputs())
        self.run_script("check", COGNITIVE_DEBT_MODE="false")
        out = self.outputs()
        self.assertNotIn("cognitive-debt-result", out)
        self.assertNotIn("cognitive_debt", Path(out["review-json-path"]).read_text())
        bad = self.run_script("prepare", COGNITIVE_DEBT_MODE="maybe")
        self.assertEqual(bad.returncode, 2)

    def test_shell_values_are_not_executed(self):
        _good_session(self.store)
        marker = Path(self.tmp.name) / "pwned"
        payload = f"$(touch {marker}); `touch {marker}`"
        checked = self.run_script("check", REVIEW_FAIL_ON=payload, REVIEW_MIN_CONFIDENCE=payload)
        self.assertEqual(checked.returncode, 2)
        self.assertFalse(marker.exists())
        self.assertEqual(self.outputs()["review-result"], "error")
        self.assertEqual(self.outputs()["passed"], "false")
        self.assertTrue(all(row.count(payload) == 2 for row in self.calls("review")))

    def test_comment_posts_once_and_skips_forks(self):
        _good_session(self.store)
        base = {"REVIEW_POST_COMMENT": "true", "PR_BASE_REPOSITORY": "o/r"}
        forked = self.run_script("check", PR_HEAD_REPOSITORY="fork/r", **base)
        self.assertIn("Skipping PR comment for a fork", forked.stdout)
        self.assertEqual(self.calls("pr-comment"), [])
        self.assertEqual(self.run_script("check", PR_HEAD_REPOSITORY="o/r", **base).returncode, 0)
        posted = self.calls("pr-comment")
        self.assertEqual(len(posted), 1)
        self.assertIn("--review", posted[0])
        self.assertNotIn("--dry-run", posted[0])
        failed = self.run_script("check", PR_HEAD_REPOSITORY="o/r", SHIM_COMMENT_EXIT="1", **base)
        self.assertEqual(failed.returncode, 1)
        self.assertIn("PR comment was not posted", failed.stdout)

    def test_no_sessions(self):
        prepared = self.run_script("prepare")
        self.assertEqual(prepared.returncode, 0)
        self.assertEqual(self.outputs()["session-id"], "")
        checked = self.run_script("check", REVIEW_FAIL_ON="none")
        self.assertEqual(checked.returncode, 0)
        out = self.outputs()
        self.assertEqual(out["cognitive-debt-result"], "unavailable")
        self.assertIn("Evidence packet: not generated.",
                      Path(out["review-markdown-path"]).read_text())


if __name__ == "__main__":
    unittest.main()
