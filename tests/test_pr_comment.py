"""Tests for PR/MR session annotation (GitHub issue #212)."""

from __future__ import annotations

import argparse
import io
import json
import os
import tempfile
import unittest
import urllib.error
from http import HTTPStatus
from pathlib import Path
from unittest.mock import patch

from agent_trace.cli import build_parser
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.pr_comment import (
    COMMENT_MARKER,
    PRCommentError,
    ReviewTarget,
    SessionSummary,
    _platform_token,
    _repo_from_remote,
    _request_json,
    _safe_code_text,
    _safe_text,
    cmd_pr_comment,
    post_or_update_comment,
    render_comment,
    resolve_target,
    select_sessions,
    summarize_session,
)
from agent_trace.store import TraceStore

try:
    from _redirect_servers import ALL_REDIRECTS, ORIGIN_NOTE, REDIRECT_NOTE, RedirectServers
except ImportError:
    from tests._redirect_servers import (
        ALL_REDIRECTS, ORIGIN_NOTE, REDIRECT_NOTE, RedirectServers,
    )

try:
    import _escapes
except ImportError:
    from tests import _escapes


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        if self.payload is None:
            return b""
        return json.dumps(self.payload).encode("utf-8")


class _Opener:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request, timeout):
        self.requests.append((request, timeout))
        if not self.responses:
            raise AssertionError("unexpected HTTP request")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return _Response(response)


def _event(session_id: str, kind: EventType, timestamp: float, **data) -> TraceEvent:
    return TraceEvent(
        event_type=kind,
        timestamp=timestamp,
        session_id=session_id,
        data=data,
    )


def _add_session(
    store: TraceStore,
    session_id: str,
    *,
    started_at: float = 100.0,
    branch: str = "feature/trace",
    ended: bool = True,
) -> None:
    meta = SessionMeta(
        session_id=session_id,
        started_at=started_at,
        ended_at=started_at + 134 if ended else None,
        total_duration_ms=134_000 if ended else 0,
        attribution={"git_branch": branch},
    )
    store.create_session(meta)
    events = [
        _event(session_id, EventType.SESSION_START, started_at),
        _event(session_id, EventType.USER_PROMPT, started_at + 1, prompt="fix auth"),
        _event(
            session_id,
            EventType.LLM_REQUEST,
            started_at + 2,
            model="claude-opus-4-6",
            prompt="inspect",
        ),
        _event(
            session_id,
            EventType.TOOL_CALL,
            started_at + 3,
            tool_name="Read",
            arguments={"file_path": "src/auth.py"},
        ),
        _event(
            session_id,
            EventType.TOOL_CALL,
            started_at + 4,
            tool_name="Read",
            arguments={"file_path": "src/auth.py"},
        ),
        _event(
            session_id,
            EventType.TOOL_CALL,
            started_at + 5,
            tool_name="Read",
            arguments={"file_path": "src/auth.py"},
        ),
        _event(
            session_id,
            EventType.TOOL_CALL,
            started_at + 6,
            tool_name="Edit",
            arguments={"file_path": "src/auth.py"},
        ),
        _event(
            session_id,
            EventType.TOOL_CALL,
            started_at + 7,
            tool_name="Bash",
            arguments={"command": "python -m pytest tests/test_auth.py -q"},
        ),
        _event(session_id, EventType.ASSISTANT_RESPONSE, started_at + 8, text="done"),
    ]
    if ended:
        events.append(_event(session_id, EventType.SESSION_END, started_at + 134, exit_code=0))
    for event in events:
        store.append_event(session_id, event)


class TestSessionSummary(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = TraceStore(self.temp.name, redact=False)
        _add_session(self.store, "abc123")

    def tearDown(self):
        self.temp.cleanup()

    def test_collects_required_review_metrics(self):
        summary = summarize_session(self.store, "abc123")

        self.assertEqual(summary.duration_seconds, 134)
        self.assertEqual(summary.tool_calls, 5)
        self.assertEqual(summary.models, ["claude-opus-4-6"])
        self.assertEqual(summary.status, "completed")
        self.assertEqual(summary.files_read, ["src/auth.py"])
        self.assertEqual(summary.files_modified, ["src/auth.py"])
        self.assertEqual(summary.redundant_reads, {"src/auth.py": 3})
        self.assertEqual(summary.redundant_read_count, 2)
        self.assertEqual(summary.test_runs, 1)
        self.assertGreater(summary.cost_usd, 0)
        self.assertEqual(summary.phases, ["explore", "implement", "verify"])
        self.assertNotIn("fix auth", summary.phases)

    def test_in_progress_without_session_end(self):
        _add_session(self.store, "still-running", started_at=300, ended=False)
        summary = summarize_session(self.store, "still-running")
        self.assertEqual(summary.status, "in progress")

    def test_failed_exit_code_is_reported(self):
        meta = self.store.load_meta("abc123")
        self.store.append_event(
            "abc123",
            _event("abc123", EventType.SESSION_END, 240, exit_code=2),
        )
        self.store.update_meta(meta)
        self.assertEqual(summarize_session(self.store, "abc123").status, "failed")

    def test_records_lint_rules_that_could_not_run(self):
        def broken(events, cfg):
            raise RuntimeError("boom")

        with patch.dict("agent_trace.lint._RULES", {"no-output": broken}):
            summary = summarize_session(self.store, "abc123")
        self.assertEqual(summary.lint_rules_not_run, ["no-output"])


# Shape of the public demo session recorded by GitHub Copilot CLI 1.0.103
# (Siddhant-K-code/agent-trace-demo#1): hooks carry no model or token usage,
# the session cwd is /private/tmp/atdemo, the agent uses both /private/tmp
# and /tmp paths, and it reads files only through a shell ``cat``.
DEMO_CWD = "/private/tmp/atdemo"
DEMO_PROMPT = (
    "Add a sentence_count(text) function to tinytext. Put it in a new module "
    "src/tinytext/sentences.py, export it from the package, add unittest tests "
    "in tests/test_sentences.py, and document it in README.md."
)


def _add_demo_session(store: TraceStore, session_id: str = "b13539e7-7f48-4d") -> None:
    meta = SessionMeta(
        session_id=session_id,
        started_at=100.0,
        ended_at=120.0,
        total_duration_ms=20_000,
        tool_calls=7,
        attribution={"git_branch": "add-sentence-count", "working_dir": DEMO_CWD},
    )
    store.create_session(meta)
    events = [
        _event(session_id, EventType.USER_PROMPT, 100.0, prompt=DEMO_PROMPT),
        _event(
            session_id, EventType.SESSION_START, 100.1,
            mode="github-copilot-hooks", provider="copilot", model="", cwd=DEMO_CWD,
        ),
        _event(
            session_id, EventType.TOOL_CALL, 101.0, tool_name="bash",
            arguments={"command": (
                "cd /private/tmp/atdemo && git ls-files && "
                "cat src/tinytext/__init__.py README.md && ls tests && "
                "head -20 tests/*.py | head -40"
            )},
        ),
        _event(session_id, EventType.TOOL_RESULT, 102.0, tool_name="bash", result="ok"),
        _event(
            session_id, EventType.TOOL_CALL, 103.0, tool_name="create",
            arguments={"path": "/private/tmp/atdemo/src/tinytext/sentences.py"},
        ),
        _event(
            session_id, EventType.TOOL_CALL, 104.0, tool_name="create",
            arguments={"path": "/private/tmp/atdemo/tests/test_sentences.py"},
        ),
        _event(
            session_id, EventType.TOOL_CALL, 105.0, tool_name="edit",
            arguments={"path": "/tmp/atdemo/src/tinytext/__init__.py"},
        ),
        _event(
            session_id, EventType.TOOL_CALL, 106.0, tool_name="edit",
            arguments={"path": "/tmp/atdemo/README.md"},
        ),
        _event(
            session_id, EventType.TOOL_CALL, 106.5, tool_name="bash",
            arguments={"command": (
                "cd /private/tmp/atdemo && PYTHONPATH=src python3 -m unittest 2>&1 "
                "| tail -5; python3 -m unittest 2>&1 | tail -5"
            )},
        ),
        _event(session_id, EventType.TOOL_RESULT, 106.6, tool_name="bash", result="NO TESTS RAN"),
        _event(
            session_id, EventType.TOOL_CALL, 107.0, tool_name="bash",
            arguments={"command": (
                "cd /private/tmp/atdemo && PYTHONPATH=src python3 -m unittest "
                "discover -s tests 2>&1 | tail -5"
            )},
        ),
        _event(session_id, EventType.TOOL_RESULT, 108.0, tool_name="bash", result="OK"),
        _event(
            session_id, EventType.ASSISTANT_RESPONSE, 119.0,
            text="", provider="copilot", cwd=DEMO_CWD,
        ),
        _event(session_id, EventType.SESSION_END, 120.0, reason="complete", cwd=DEMO_CWD),
    ]
    for event in events:
        store.append_event(session_id, event)


DEMO_MODIFIED = [
    "README.md",
    "src/tinytext/__init__.py",
    "src/tinytext/sentences.py",
    "tests/test_sentences.py",
]


class TestDemoSessionSummary(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = TraceStore(self.temp.name, redact=False)
        _add_demo_session(self.store)

    def tearDown(self):
        self.temp.cleanup()

    def test_cost_is_not_captured_without_model_or_token_usage(self):
        summary = summarize_session(self.store, "b13539e7-7f48-4d")
        self.assertIsNone(summary.cost_usd)
        self.assertEqual(summary.models, [])

    def test_shell_read_targets_count_as_reads(self):
        summary = summarize_session(self.store, "b13539e7-7f48-4d")
        self.assertEqual(summary.files_read, ["README.md", "src/tinytext/__init__.py"])

    def test_paths_are_relative_to_the_session_cwd(self):
        summary = summarize_session(self.store, "b13539e7-7f48-4d")
        self.assertEqual(summary.files_modified, DEMO_MODIFIED)

    def test_rendered_summary_matches_reviewability_scope(self):
        from agent_trace.reviewability import LinkedSession, Threshold, build_report

        summary = summarize_session(self.store, "b13539e7-7f48-4d")
        report = build_report(
            self.store,
            [LinkedSession("b13539e7-7f48-4d", "branch")],
            Threshold(),
            branch="add-sentence-count",
        )
        scope = next(f for f in report.findings if f.name == "scope")
        self.assertEqual(scope.facts["files_written"], len(summary.files_modified))
        self.assertEqual(scope.status, "present")

        body = render_comment([summary])
        self.assertIn(
            "| **Cost** | not captured: no model or token usage recorded |", body
        )
        self.assertIn("| **Tool calls** | 7 |", body)
        self.assertIn("| **Model** | not captured |", body)
        self.assertIn("- Read 2 files, modified 4\n", body)
        self.assertIn(
            "- Modified: `README.md`, `src/tinytext/__init__.py`, "
            "`src/tinytext/sentences.py`, `tests/test_sentences.py`\n",
            body,
        )
        self.assertNotIn("$", body)
        self.assertNotIn("…", body)
        self.assertNotIn("/tmp", body)

    def test_recorded_model_shows_a_labelled_estimate(self):
        self.store.append_event(
            "b13539e7-7f48-4d",
            _event("b13539e7-7f48-4d", EventType.LLM_REQUEST, 109.0, model="gpt-5"),
        )
        summary = summarize_session(self.store, "b13539e7-7f48-4d")
        self.assertIsNotNone(summary.cost_usd)
        self.assertGreater(summary.cost_usd, 0)
        body = render_comment([summary])
        self.assertIn(f"| **Cost** | ${summary.cost_usd:.4f} (estimate) |", body)

    def test_recorded_token_usage_shows_a_labelled_estimate(self):
        self.store.append_event(
            "b13539e7-7f48-4d",
            _event(
                "b13539e7-7f48-4d", EventType.LLM_RESPONSE, 109.0,
                usage={"input_tokens": 1000, "output_tokens": 200},
            ),
        )
        summary = summarize_session(self.store, "b13539e7-7f48-4d")
        self.assertIsNotNone(summary.cost_usd)
        self.assertIn("(estimate)", render_comment([summary]))

    def test_dry_run_comment_for_demo_session(self):
        args = argparse.Namespace(
            trace_dir=self.temp.name,
            session_id="b13539e7",
            platform="github",
            dry_run=True,
            share_url="",
        )
        stdout = io.StringIO()
        with patch.dict(os.environ, {}, clear=True), patch(
            "urllib.request.urlopen", side_effect=AssertionError("network attempted")
        ), patch(
            "agent_trace.pr_comment.no_redirect_opener",
            side_effect=AssertionError("network attempted"),
        ), patch(
            "agent_trace.pr_comment.same_origin_opener",
            side_effect=AssertionError("network attempted"),
        ), patch("sys.stdout", stdout):
            result = cmd_pr_comment(args)
        self.assertEqual(result, 0)
        body = stdout.getvalue()
        self.assertIn("not captured", body)
        self.assertNotIn("$0.", body)
        self.assertIn("- Read 2 files, modified 4", body)
        self.assertIn("`src/tinytext/sentences.py`", body)


class TestRenderCostCapture(unittest.TestCase):
    def test_missing_cost_is_not_shown_as_dollars(self):
        body = render_comment([_summary(cost_usd=None, models=[])])
        self.assertIn(
            "| **Cost** | not captured: no model or token usage recorded |", body
        )
        self.assertIn("| **Model** | not captured |", body)
        self.assertNotIn("$", body)

    def test_multi_session_missing_cost_cell(self):
        body = render_comment([
            _summary("one", cost_usd=None, models=[]),
            _summary("two", cost_usd=0.2),
        ])
        self.assertIn("| `one` | 2m 14s | not captured | 142 | not captured |", body)
        self.assertIn("| `two` | 2m 14s | $0.2000 | 142 |", body)


class TestSessionSelection(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = TraceStore(self.temp.name, redact=False)

    def tearDown(self):
        self.temp.cleanup()

    def test_explicit_prefix_selects_one_session(self):
        _add_session(self.store, "abc123456", branch="one")
        _add_session(self.store, "def987654", branch="one", started_at=200)
        self.assertEqual(select_sessions(self.store, "abc123"), ["abc123456"])

    def test_all_branch_sessions_are_returned_oldest_first(self):
        _add_session(self.store, "older", branch="feature", started_at=100)
        _add_session(self.store, "newer", branch="feature", started_at=200)
        _add_session(self.store, "other", branch="main", started_at=300)
        self.assertEqual(
            select_sessions(self.store, branch="feature"),
            ["older", "newer"],
        )

    def test_unattributed_session_closest_to_commit_is_selected(self):
        _add_session(self.store, "far", branch="", started_at=100)
        _add_session(self.store, "near", branch="", started_at=195)
        with patch("agent_trace.pr_comment._git", return_value="200"):
            selected = select_sessions(self.store, branch="missing")
        self.assertEqual(selected, ["near"])

    def test_missing_session_has_clear_error(self):
        _add_session(self.store, "exists")
        with self.assertRaisesRegex(PRCommentError, "Session not found"):
            select_sessions(self.store, "missing")


def _summary(session_id: str = "abc123", **changes) -> SessionSummary:
    values = {
        "session_id": session_id,
        "duration_seconds": 134,
        "cost_usd": 0.31,
        "tool_calls": 142,
        "models": ["claude-opus-4-6"],
        "status": "completed",
        "files_read": ["a.py", "b.py"],
        "files_modified": ["a.py"],
        "phases": ["explore", "implement", "verify"],
        "test_runs": 3,
        "context_resets": 1,
        "lint_violations": 0,
        "redundant_reads": {"a.py": 3},
        "errors": 0,
    }
    values.update(changes)
    return SessionSummary(**values)


class TestRenderComment(unittest.TestCase):
    def test_single_session_contains_acceptance_fields(self):
        body = render_comment([_summary()])

        self.assertTrue(body.startswith(COMMENT_MARKER))
        self.assertIn("`abc123` · 2m 14s", body)
        self.assertIn("| **Cost** | $0.3100 (estimate) |", body)
        self.assertIn("| **Tool calls** | 142 |", body)
        self.assertIn("claude-opus-4-6", body)
        self.assertIn("Read 2 files, modified 1", body)
        self.assertIn("2 redundant reads", body)
        self.assertIn("No lint violations", body)
        self.assertIn("No errors", body)
        self.assertIn("agent-strace replay abc123", body)

    def test_lint_rules_that_could_not_run_are_not_reported_as_clean(self):
        body = render_comment([
            _summary("one", lint_rules_not_run=["no-output"]),
            _summary("two", lint_rules_not_run=["tool-loop", "no-output"]),
        ])
        self.assertNotIn("No lint violations", body)
        self.assertIn("- ⚠️ Lint could not run 2 rules: `no-output`, `tool-loop`\n", body)

    def test_multiple_sessions_have_individual_cost_status_rows(self):
        body = render_comment([
            _summary("one", cost_usd=0.10),
            _summary("two", cost_usd=0.20, status="failed"),
        ])
        self.assertIn("| Session | Duration | Est. cost |", body)
        self.assertIn("| `one` | 2m 14s | $0.1000", body)
        self.assertIn("| `two` | 2m 14s | $0.2000", body)
        self.assertIn("failed", body)

    def test_optional_share_url_template(self):
        body = render_comment(
            [_summary("abc123")],
            share_url="https://trace.example/s/{session_id}",
        )
        self.assertIn("https://trace.example/s/abc123", body)

    def test_unsafe_share_url_is_omitted(self):
        body = render_comment([_summary()], share_url="javascript:alert(1)")
        self.assertNotIn("javascript", body)
        self.assertNotIn("view the HTML replay", body)

    def test_trace_text_cannot_inject_markdown_table_or_html(self):
        body = render_comment([
            _summary(models=["bad|model\n<script>"], phases=["x\n- injected"])
        ])
        self.assertIn("bad\\|model &lt;script&gt;", body)
        self.assertNotIn("\n- injected", body)
        self.assertNotIn("<script>", body)

    def test_paths_and_inline_markdown_are_privacy_safe(self):
        body = render_comment([
            _summary(
                phases=["@reviewers [click me](https://evil.example)"],
                files_modified=[
                    "/home/alice/private/src/auth.py",
                    "reports/alice@example.com.txt",
                ],
                redundant_reads={"/Users/alice/company/src/secret.py": 2},
            )
        ])

        self.assertIn("- Modified:", body)
        self.assertIn("`…/private/src/auth.py`", body)
        self.assertIn("reports/&lt;email&gt;", body)
        self.assertNotIn("/home/alice", body)
        self.assertNotIn("/Users/alice", body)
        self.assertNotIn("@reviewers", body)
        self.assertNotIn("[click me](https://evil.example)", body)
        self.assertIn("&#64;reviewers", body)
        self.assertIn(r"\[click me\]\(https://evil.example\)", body)


class TestTargetResolution(unittest.TestCase):
    def setUp(self):
        # Target resolution intentionally consults CI metadata. Keep these
        # unit tests deterministic when the suite itself runs inside a PR job.
        self.ci_env = patch.dict(os.environ, {
            "GITHUB_EVENT_PATH": "",
            "GITHUB_REPOSITORY": "",
            "GITHUB_API_URL": "",
            "GITHUB_HEAD_REPOSITORY": "",
            "CI_PROJECT_PATH": "",
            "CI_API_V4_URL": "",
            "CI_MERGE_REQUEST_IID": "",
        })
        self.ci_env.start()
        self.addCleanup(self.ci_env.stop)

    def test_remote_url_formats(self):
        self.assertEqual(
            _repo_from_remote("git@github.com:owner/repo.git"),
            "owner/repo",
        )
        self.assertEqual(
            _repo_from_remote("https://gitlab.com/group/sub/repo.git"),
            "group/sub/repo",
        )

    def test_explicit_github_target_needs_no_request(self):
        opener = _Opener()
        target = resolve_target(
            "github",
            "secret",
            project="owner/repo",
            number=12,
            opener=opener,
        )
        self.assertEqual(target, ReviewTarget("github", "owner/repo", 12, "https://api.github.com"))
        self.assertEqual(opener.requests, [])

    def test_github_open_pr_is_discovered_from_branch(self):
        opener = _Opener([{"number": 42}])
        target = resolve_target(
            "github",
            "secret",
            project="owner/repo",
            branch="feature",
            opener=opener,
        )
        self.assertEqual(target.number, 42)
        request = opener.requests[0][0]
        self.assertIn("head=owner%3Afeature", request.full_url)
        self.assertEqual(request.get_header("Authorization"), "Bearer secret")

    def test_github_discovery_uses_tracked_fork_owner(self):
        opener = _Opener([{"number": 42}])

        def fake_git(_cwd, *args):
            if args == ("config", "--get", "branch.feature.remote"):
                return "origin"
            if args == ("remote", "get-url", "origin"):
                return "git@github.com:fork-owner/repo.git"
            if args == ("remote", "get-url", "upstream"):
                return "git@github.com:base-owner/repo.git"
            return ""

        with patch("agent_trace.pr_comment._git", side_effect=fake_git):
            target = resolve_target(
                "github",
                "secret",
                project="base-owner/repo",
                branch="feature",
                opener=opener,
            )

        self.assertEqual(target.project, "base-owner/repo")
        self.assertIn("head=fork-owner%3Afeature", opener.requests[0][0].full_url)

    def test_gitlab_open_mr_uses_encoded_project(self):
        opener = _Opener([{"iid": 7}])
        target = resolve_target(
            "gitlab",
            "secret",
            project="group/sub/repo",
            branch="feature",
            opener=opener,
        )
        self.assertEqual(target.number, 7)
        self.assertIn("group%2Fsub%2Frepo", opener.requests[0][0].full_url)


class TestCommentPosting(unittest.TestCase):
    def test_github_creates_comment_when_marker_is_absent(self):
        opener = _Opener([], {"id": 99})
        target = ReviewTarget("github", "owner/repo", 12, "https://api.github.com")
        action = post_or_update_comment(target, COMMENT_MARKER + "\nbody", "secret", opener=opener)

        self.assertEqual(action, "created")
        self.assertEqual([request.method for request, _ in opener.requests], ["GET", "POST"])
        request = opener.requests[1][0]
        self.assertEqual(json.loads(request.data), {"body": COMMENT_MARKER + "\nbody"})
        self.assertNotIn("secret", request.full_url)

    def test_github_updates_existing_marker_comment(self):
        opener = _Opener([{"id": 88, "body": COMMENT_MARKER + " old"}], {"id": 88})
        target = ReviewTarget("github", "owner/repo", 12, "https://api.github.com")
        action = post_or_update_comment(target, COMMENT_MARKER + " new", "secret", opener=opener)

        self.assertEqual(action, "updated")
        request = opener.requests[1][0]
        self.assertEqual(request.method, "PATCH")
        self.assertTrue(request.full_url.endswith("/issues/comments/88"))

    def test_github_finds_marker_after_first_comment_page(self):
        first_page = [
            {"id": index, "body": "ordinary comment"}
            for index in range(1, 101)
        ]
        opener = _Opener(
            first_page,
            [{"id": 188, "body": COMMENT_MARKER + " old"}],
            {"id": 188},
        )
        target = ReviewTarget("github", "owner/repo", 12, "https://api.github.com")

        action = post_or_update_comment(
            target, COMMENT_MARKER + " new", "secret", opener=opener
        )

        self.assertEqual(action, "updated")
        self.assertEqual(
            [request.method for request, _timeout in opener.requests],
            ["GET", "GET", "PATCH"],
        )
        self.assertIn("page=1", opener.requests[0][0].full_url)
        self.assertIn("page=2", opener.requests[1][0].full_url)
        self.assertTrue(opener.requests[2][0].full_url.endswith("/issues/comments/188"))

    def test_gitlab_updates_existing_note(self):
        opener = _Opener([{"id": 55, "body": COMMENT_MARKER}], {"id": 55})
        target = ReviewTarget("gitlab", "group/repo", 4, "https://gitlab.example/api/v4")
        action = post_or_update_comment(
            target,
            COMMENT_MARKER + " new",
            "secret",
            opener=opener,
            token_header="PRIVATE-TOKEN",
        )

        self.assertEqual(action, "updated")
        request = opener.requests[1][0]
        self.assertEqual(request.method, "PUT")
        self.assertTrue(request.full_url.endswith("/notes/55"))
        self.assertEqual(request.get_header("Private-token"), "secret")

    def test_gitlab_finds_marker_after_first_note_page(self):
        first_page = [
            {"id": index, "body": "ordinary note"}
            for index in range(1, 101)
        ]
        opener = _Opener(
            first_page,
            [{"id": 155, "body": COMMENT_MARKER}],
            {"id": 155},
        )
        target = ReviewTarget("gitlab", "group/repo", 4, "https://gitlab.example/api/v4")

        action = post_or_update_comment(
            target,
            COMMENT_MARKER + " new",
            "secret",
            opener=opener,
            token_header="PRIVATE-TOKEN",
        )

        self.assertEqual(action, "updated")
        self.assertEqual(
            [request.method for request, _timeout in opener.requests],
            ["GET", "GET", "PUT"],
        )
        self.assertIn("page=2", opener.requests[1][0].full_url)

    def test_gitlab_job_token_is_not_accepted_for_writes(self):
        with patch.dict(os.environ, {"CI_JOB_TOKEN": "read-only-job-token"}, clear=True):
            self.assertEqual(_platform_token("gitlab"), ("", "PRIVATE-TOKEN"))

        with patch.dict(os.environ, {"GITLAB_ACCESS_TOKEN": "project-token"}, clear=True):
            self.assertEqual(
                _platform_token("gitlab"),
                ("project-token", "PRIVATE-TOKEN"),
            )

    def test_missing_token_cannot_write(self):
        target = ReviewTarget("github", "owner/repo", 1, "https://api.github.com")
        with self.assertRaisesRegex(PRCommentError, "token is required"):
            post_or_update_comment(target, COMMENT_MARKER, "", opener=_Opener())

    def test_http_error_does_not_expose_token(self):
        error = urllib.error.HTTPError(
            "https://api.github.com", 403, "Forbidden", {}, None
        )
        with self.assertRaises(PRCommentError) as raised:
            _request_json(
                "GET",
                "https://api.github.com/repos/owner/repo",
                "top-secret-token",
                platform="github",
                opener=_Opener(error),
            )
        self.assertNotIn("top-secret-token", str(raised.exception))

    def _request_error(self, error):
        with self.assertRaises(PRCommentError) as raised:
            _request_json(
                "GET",
                "https://api.github.com/repos/owner/repo",
                "token",
                platform="github",
                opener=_Opener(error),
            )
        return str(raised.exception)

    def test_network_error_reason_is_inert(self):
        # urllib raises this when an HTTPS proxy refuses CONNECT.
        error = urllib.error.URLError(
            OSError("Tunnel connection failed: 403 Denied\x1b[2K\nforged: ok\x9b8m")
        )
        self.assertEqual(
            self._request_error(error),
            "Github API request failed: <urlopen error Tunnel connection failed: "
            "403 Denied\\x1b[2K\\nforged: ok\\x9b8m>",
        )

    def test_printable_error_reason_is_unchanged(self):
        url = "https://api.github.com"
        cases = [
            (urllib.error.HTTPError(url, 404, "Not Found", {}, None),
             "Github API request failed with HTTP 404: Not Found"),
            (urllib.error.HTTPError(url, 503, "Busy: café → 日本", {}, None),
             "Github API request failed with HTTP 503: Busy: café → 日本"),
            (urllib.error.URLError(ConnectionRefusedError(61, "Connection refused")),
             "Github API request failed: <urlopen error [Errno 61] Connection refused>"),
            (TimeoutError("timed out"), "Github API request failed: timed out"),
        ]
        for error, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(self._request_error(error), expected)


class TestRequestRedirects(unittest.TestCase):
    """urllib re-sent a redirected POST as a GET with no body and the token."""

    def test_gitlab_post_redirect_is_an_error(self):
        servers = RedirectServers(self, a_reply=[])
        target = ReviewTarget("gitlab", "group/repo", 4, servers.url + "/api/v4")
        for status in ALL_REDIRECTS:
            with self.subTest(status=status):
                servers.redirect(status)
                with self.assertRaises(PRCommentError) as caught:
                    post_or_update_comment(
                        target, COMMENT_MARKER + "\nbody", "gitlab-token",
                        token_header="PRIVATE-TOKEN",
                    )
                self.assertEqual(
                    str(caught.exception),
                    f"Gitlab API request failed with HTTP {status}: "
                    f"{HTTPStatus(status).phrase}{REDIRECT_NOTE}",
                )
                self.assertEqual(servers.a.methods(), ["GET", "POST"])
                self.assertEqual(servers.b.received(), [])

    def test_gitlab_put_redirect_names_the_cause(self):
        servers = RedirectServers(
            self, methods=("PUT",), a_reply=[{"id": 7, "body": COMMENT_MARKER + " old"}]
        )
        servers.redirect(301)
        target = ReviewTarget("gitlab", "group/repo", 4, servers.url + "/api/v4")
        with self.assertRaises(PRCommentError) as caught:
            post_or_update_comment(target, COMMENT_MARKER + " new", "gitlab-token")
        self.assertEqual(
            str(caught.exception),
            f"Gitlab API request failed with HTTP 301: Moved Permanently{REDIRECT_NOTE}",
        )
        self.assertEqual(servers.a.methods(), ["GET", "PUT"])
        self.assertEqual(servers.b.received(), [])

    def test_get_still_follows_a_redirect(self):
        # GitHub answers a GET for a renamed repository with 301 to
        # https://api.github.com/repositories/<id>, the same origin.
        notes = [{"id": 1, "body": "note"}]
        servers = RedirectServers(self, methods=("GET",), reply=notes)
        servers.redirect(301, servers.moved)
        path = "/repos/old/name/issues/1/comments"
        result = _request_json("GET", servers.url + path, "token", platform="github")
        self.assertEqual(result, notes)
        self.assertEqual(
            [(r.path, r.headers["Authorization"]) for r in servers.a.received()],
            [(path, "Bearer token"), ("/moved" + path, "Bearer token")],
        )
        self.assertEqual(servers.b.received(), [])

    def test_get_redirect_to_another_origin_is_an_error(self):
        # urlopen sent the token to the host in Location.
        servers = RedirectServers(self, methods=("GET",), reply=[])
        cases = [
            (ReviewTarget("github", "owner/repo", 1, servers.url), "Authorization"),
            (ReviewTarget("gitlab", "group/repo", 4, servers.url + "/api/v4"), "PRIVATE-TOKEN"),
        ]
        for target, header in cases:
            for status in ALL_REDIRECTS:
                with self.subTest(platform=target.platform, status=status):
                    servers.redirect(status)
                    with self.assertRaises(PRCommentError) as caught:
                        post_or_update_comment(target, COMMENT_MARKER + "\nbody", "probe-token")
                    self.assertEqual(
                        str(caught.exception),
                        f"{target.platform.title()} API request failed with HTTP {status}: "
                        f"{HTTPStatus(status).phrase}{ORIGIN_NOTE}",
                    )
                    self.assertTrue(caught.exception.__cause__.fp.closed)
                    self.assertEqual(servers.a.methods(), ["GET"])
                    self.assertIn("probe-token", servers.a.received()[0].headers[header])
                    self.assertEqual(servers.b.received(), [])

    def test_merge_request_lookup_refuses_a_redirect_to_another_origin(self):
        servers = RedirectServers(self, methods=("GET",), reply=[{"iid": 9}])
        servers.redirect(302)
        with patch.dict(os.environ, {"CI_MERGE_REQUEST_IID": ""}):
            with self.assertRaises(PRCommentError) as caught:
                resolve_target(
                    "gitlab", "probe-token", project="group/repo",
                    api_url=servers.url + "/api/v4", branch="feature",
                )
        self.assertEqual(
            str(caught.exception),
            f"Gitlab API request failed with HTTP 302: Found{ORIGIN_NOTE}",
        )
        self.assertEqual(servers.a.methods(), ["GET"])
        self.assertEqual(servers.b.received(), [])

    def test_get_error_has_no_redirect_note(self):
        # urllib does not follow a 300; it raises. That is not a refused
        # redirect, so neither note applies.
        servers = RedirectServers(self, methods=("GET",))
        servers.redirect(300)
        url = servers.url + "/repos/owner/repo/issues/1/comments"
        with self.assertRaises(PRCommentError) as caught:
            _request_json("GET", url, "token", platform="github")
        self.assertEqual(
            str(caught.exception), "Github API request failed with HTTP 300: Multiple Choices"
        )
        self.assertEqual(servers.a.methods(), ["GET"])
        self.assertEqual(servers.b.received(), [])


class TestCommand(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = TraceStore(self.temp.name, redact=False)
        _add_session(self.store, "dry-run")

    def tearDown(self):
        self.temp.cleanup()

    def test_dry_run_prints_without_token_or_network(self):
        args = argparse.Namespace(
            trace_dir=self.temp.name,
            session_id="dry-run",
            platform="github",
            dry_run=True,
            share_url="",
        )
        stdout = io.StringIO()
        with patch.dict(os.environ, {}, clear=True), patch(
            "urllib.request.urlopen", side_effect=AssertionError("network attempted")
        ), patch(
            "agent_trace.pr_comment.no_redirect_opener",
            side_effect=AssertionError("network attempted"),
        ), patch(
            "agent_trace.pr_comment.same_origin_opener",
            side_effect=AssertionError("network attempted"),
        ), patch("sys.stdout", stdout):
            result = cmd_pr_comment(args)

        self.assertEqual(result, 0)
        self.assertIn(COMMENT_MARKER, stdout.getvalue())
        self.assertIn("dry-run", stdout.getvalue())

    def test_api_error_reason_is_one_inert_line(self):
        # The API server writes the reason phrase that pr-comment prints.
        error = urllib.error.HTTPError(
            "https://api.github.com", 500, "Oops\x1b[2J\r\nforged: ok\x9b8m\u202e", {}, None
        )
        args = argparse.Namespace(
            trace_dir=self.temp.name,
            session_id="dry-run",
            platform="github",
            dry_run=False,
            share_url="",
            repo="owner/repo",
            pr=7,
            api_url="https://api.github.com",
        )
        stderr = io.StringIO()
        # Patch OpenerDirector.open, not urlopen, so a request through any
        # opener fails here and none reaches api.github.com.
        with patch.dict(os.environ, {"GITHUB_TOKEN": "token"}, clear=True), patch(
            "urllib.request.OpenerDirector.open", side_effect=error
        ), patch("sys.stderr", stderr):
            result = cmd_pr_comment(args)

        self.assertEqual(result, 1)
        self.assertEqual(
            stderr.getvalue(),
            "pr-comment: Github API request failed with HTTP 500: "
            "Oops\\x1b[2J\\r\\nforged: ok\\x9b8m\\u202e\n",
        )

    def test_get_redirect_to_another_origin_fails_the_command(self):
        servers = RedirectServers(self, methods=("GET",), reply=[])
        servers.redirect(302)
        args = argparse.Namespace(
            trace_dir=self.temp.name,
            session_id="dry-run",
            platform="gitlab",
            dry_run=False,
            share_url="",
            repo="group/repo",
            pr=4,
            api_url=servers.url + "/api/v4",
        )
        stderr = io.StringIO()
        with patch.dict(os.environ, {"GITLAB_TOKEN": "probe-token"}), patch(
            "sys.stdout", io.StringIO()
        ), patch("sys.stderr", stderr):
            result = cmd_pr_comment(args)
        self.assertEqual(result, 1)
        self.assertEqual(
            stderr.getvalue(),
            f"pr-comment: Gitlab API request failed with HTTP 302: Found{ORIGIN_NOTE}\n",
        )
        self.assertEqual(servers.a.methods(), ["GET"])
        self.assertEqual(servers.b.received(), [])

    def test_no_sessions_returns_nonzero(self):
        empty = tempfile.TemporaryDirectory()
        args = argparse.Namespace(
            trace_dir=empty.name,
            session_id=None,
            platform="github",
            dry_run=True,
            share_url="",
        )
        stderr = io.StringIO()
        try:
            with patch("sys.stderr", stderr):
                result = cmd_pr_comment(args)
        finally:
            empty.cleanup()
        self.assertEqual(result, 1)
        self.assertIn("No agent-trace sessions found", stderr.getvalue())

    def test_cli_registers_pr_comment_flags(self):
        args = build_parser().parse_args([
            "pr-comment", "abc123", "--dry-run", "--platform", "gitlab",
            "--repo", "group/project", "--pr", "17",
            "--api-url", "https://gitlab.example/api/v4",
            "--share-url", "https://traces.example/{session_id}",
        ])
        self.assertEqual(args.command, "pr-comment")
        self.assertEqual(args.session_id, "abc123")
        self.assertTrue(args.dry_run)
        self.assertEqual(args.platform, "gitlab")
        self.assertEqual(args.repo, "group/project")
        self.assertEqual(args.pr, 17)

    def test_composite_action_registers_safe_opt_in_comment_step(self):
        action = (Path(__file__).parents[1] / "action.yml").read_text()
        self.assertIn('bash "$GITHUB_ACTION_PATH/scripts/install-agent-strace.sh"', action)
        self.assertNotIn("pip install agent-trace", action)
        self.assertIn("  post-pr-comment:\n", action)
        self.assertIn("  github-token:\n", action)
        self.assertIn("inputs.post-pr-comment == 'true'", action)
        self.assertIn("GITHUB_TOKEN: ${{ inputs.github-token }}", action)
        self.assertIn("PR_HEAD_REPOSITORY", action)
        self.assertIn("No agent-trace sessions found; skipping PR comment", action)
        self.assertIn("pr-comment", action)


# How _safe_text and _table_text show PAYLOAD. The Markdown escape puts a
# backslash before each backslash and bracket, so GitHub shows PAYLOAD_FREE.
PAYLOAD_MARKDOWN = r"A\\x1b\]52;c;aWQK\\x07B\\x1b\[2K C D\\x9bE\\u202eF"

# (label, helper, expected): how each Markdown helper shows PAYLOAD.
HELPER_CASES = [
    ("_safe_text", _safe_text, PAYLOAD_MARKDOWN),
    ("_safe_code_text", _safe_code_text, _escapes.PAYLOAD_FREE),
]

# (label, expected, build): build(value) returns the summaries for
# render_comment(), with value in the place the label names and plain text
# everywhere else.
RENDER_COMMENT_CASES = [
    ("model, one session", PAYLOAD_MARKDOWN, lambda v: [_summary(models=[v])]),
    (
        "model, two sessions",
        PAYLOAD_MARKDOWN,
        lambda v: [_summary("one", models=[v]), _summary("two")],
    ),
    ("modified path", _escapes.PAYLOAD_FREE, lambda v: [_summary(files_modified=[v])]),
    ("redundant read", _escapes.PAYLOAD_FREE, lambda v: [_summary(redundant_reads={v: 3})]),
]


class TestTraceTextShowsAsText(unittest.TestCase):
    """Model names and paths in a trace can hold escape codes and line breaks.
    The comment shows the escapes as text and each line break as a space."""

    def test_each_helper_shows_escapes_as_text(self):
        for label, helper, expected in HELPER_CASES:
            with self.subTest(label):
                self.assertEqual(helper(_escapes.PAYLOAD), expected)

    def test_each_value_shows_escapes_as_text(self):
        for label, expected, build in RENDER_COMMENT_CASES:
            with self.subTest(label):
                _escapes.assert_shows_as_text(
                    self,
                    render_comment(build(_escapes.PAYLOAD)),
                    expected,
                    render_comment(build("plain")),
                )

    def test_plain_text_prints_unchanged(self):
        for label, helper, _ in HELPER_CASES:
            with self.subTest(label):
                self.assertEqual(helper(_escapes.PLAIN), _escapes.PLAIN)
        for label, _, build in RENDER_COMMENT_CASES:
            with self.subTest(label):
                self.assertIn(_escapes.PLAIN, render_comment(build(_escapes.PLAIN)))

    def test_sgr_code_shows_as_text(self):
        self.assertEqual(_safe_text("a\x1b[8mhidden"), r"a\\x1b\[8mhidden")
        self.assertEqual(_safe_code_text("a\x1b[8mhidden"), r"a\x1b[8mhidden")


class TestPayloadTrace(unittest.TestCase):
    """pr-comment posts the body that --dry-run prints, with each escape from
    the trace shown as text."""

    SESSION = "payload"

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        store = TraceStore(self.temp.name, redact=False)
        store.create_session(SessionMeta(
            session_id=self.SESSION, started_at=100.0, ended_at=234.0
        ))
        sid = self.SESSION
        events = [
            _event(sid, EventType.SESSION_START, 100.0),
            _event(sid, EventType.USER_PROMPT, 101.0, prompt="fix it"),
            _event(sid, EventType.LLM_REQUEST, 102.0, model=_escapes.PAYLOAD, prompt="inspect"),
        ]
        # Three reads of one path make two redundant reads.
        for timestamp in (103.0, 104.0, 105.0):
            events.append(_event(
                sid, EventType.TOOL_CALL, timestamp,
                tool_name="Read", arguments={"file_path": "r" + _escapes.PAYLOAD},
            ))
        events.append(_event(
            sid, EventType.TOOL_CALL, 106.0,
            tool_name="Edit", arguments={"file_path": "e" + _escapes.PAYLOAD},
        ))
        events.append(_event(sid, EventType.SESSION_END, 234.0, exit_code=0))
        for event in events:
            store.append_event(sid, event)
        # --review reads its config file from the working directory.
        cwd = os.getcwd()
        os.chdir(self.temp.name)
        self.addCleanup(os.chdir, cwd)

    def _run(self, *flags):
        """Run pr-comment and return its stdout and the bodies it posted."""
        args = build_parser().parse_args(
            ["--trace-dir", self.temp.name, "pr-comment", self.SESSION, *flags]
        )
        stdout, posted = io.StringIO(), []

        def post(_target, body, _token, **_kwargs):
            posted.append(body)
            return "created"

        with patch.dict(os.environ, {"GITHUB_TOKEN": "token"}), patch(
            "agent_trace.pr_comment.current_branch", return_value="feature/trace"
        ), patch("agent_trace.pr_comment.resolve_target") as resolve, patch(
            "agent_trace.pr_comment.post_or_update_comment", side_effect=post
        ), patch("sys.stdout", stdout), patch("sys.stderr", io.StringIO()):
            resolve.return_value.number = 7
            cmd_pr_comment(args)
        return stdout.getvalue(), posted

    def test_posted_body_is_the_dry_run_output(self):
        for flags in ((), ("--review",)):
            with self.subTest(flags=flags):
                body, posted = self._run("--dry-run", *flags)
                self.assertEqual(posted, [])
                stdout, posted = self._run(*flags)
                self.assertEqual(stdout, "Created agent-trace comment on PR #7.\n")
                self.assertEqual(posted, [body])
                self.assertEqual("### Reviewability" in body, bool(flags))
                for line in body.split("\n"):
                    self.assertTrue(line.isprintable(), ascii(line))
                self.assertIn(f"| **Model** | {PAYLOAD_MARKDOWN} |", body)
                self.assertIn(f"- Modified: `e{_escapes.PAYLOAD_FREE}`", body)
                self.assertIn(f"(`r{_escapes.PAYLOAD_FREE}` read 3×)", body)


if __name__ == "__main__":
    unittest.main()
