"""Tests for the pull-request reviewability check (agent-strace review)."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from agent_trace.approval import ApprovalRequest, save_request
from agent_trace.cli import build_parser
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.pr_comment import COMMENT_MARKER, cmd_pr_comment
from agent_trace.reviewability import (
    FINDINGS,
    LinkedSession,
    ReviewConfigError,
    Threshold,
    Finding,
    build_report,
    classify_validation,
    cmd_review,
    link_sessions,
    load_threshold,
    render_markdown,
)
from agent_trace.store import TraceStore

SECRET_PROMPT = "Refactor src/auth.py for the zebra-quartz-77 rollout"
BRANCH = "feature/review"
CWD = "/work/repo"


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
                {"label": "pytest", "kind": "test", "outcome": "passed", "exit_code": 0},
                {"label": "ruff", "kind": "check", "outcome": "passed", "exit_code": 0},
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

    def test_success_without_exit_code_is_medium(self):
        b = _Builder(self.store, "nocode000001")
        b.prompt()
        b.tool("Bash", result="all good", command="pytest")
        validation = self.finding(self.report(b.save()), "validation")
        self.assertEqual((validation.status, validation.confidence), ("present", "medium"))
        self.assertEqual(validation.facts["latest"][0]["outcome"], "succeeded")

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


class TestAction(unittest.TestCase):
    def test_composite_action_registers_review_step(self):
        action = (Path(__file__).parents[1] / "action.yml").read_text()
        for name in ("run-eval", "review", "review-fail-on", "review-min-confidence"):
            self.assertIn(f"  {name}:\n", action)
        self.assertIn("  review-passed:\n", action)
        self.assertIn("if: ${{ inputs.run-eval == 'true' }}", action)
        self.assertIn("inputs.post-pr-comment == 'true' && inputs.review != 'true'", action)
        self.assertIn("if: ${{ always() && inputs.review == 'true' }}", action)
        self.assertIn('pr-comment --review "${ARGS[@]}" || status=$?', action)
        self.assertIn('review --format markdown "${ARGS[@]}" >> "$GITHUB_STEP_SUMMARY"', action)
        self.assertIn('exit "$status"', action)
        self.assertNotIn("\u2014", action)


if __name__ == "__main__":
    unittest.main()
