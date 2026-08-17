"""Tests for PR/MR session annotation (GitHub issue #212)."""

from __future__ import annotations

import argparse
import io
import json
import os
import tempfile
import unittest
import urllib.error
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
    cmd_pr_comment,
    post_or_update_comment,
    render_comment,
    resolve_target,
    select_sessions,
    summarize_session,
)
from agent_trace.store import TraceStore


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
        self.assertIn("$0.3100 (142 tool calls)", body)
        self.assertIn("claude-opus-4-6", body)
        self.assertIn("Read 2 files, modified 1", body)
        self.assertIn("2 redundant reads", body)
        self.assertIn("No lint violations", body)
        self.assertIn("No errors", body)
        self.assertIn("agent-strace replay abc123", body)

    def test_multiple_sessions_have_individual_cost_status_rows(self):
        body = render_comment([
            _summary("one", cost_usd=0.10),
            _summary("two", cost_usd=0.20, status="failed"),
        ])
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
        ), patch("sys.stdout", stdout):
            result = cmd_pr_comment(args)

        self.assertEqual(result, 0)
        self.assertIn(COMMENT_MARKER, stdout.getvalue())
        self.assertIn("dry-run", stdout.getvalue())

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
        self.assertIn('pip install "agent-strace[$EXTRAS]"', action)
        self.assertIn("pip install agent-strace", action)
        self.assertNotIn("pip install agent-trace", action)
        self.assertIn("  post-pr-comment:\n", action)
        self.assertIn("  github-token:\n", action)
        self.assertIn("inputs.post-pr-comment == 'true'", action)
        self.assertIn("GITHUB_TOKEN: ${{ inputs.github-token }}", action)
        self.assertIn("PR_HEAD_REPOSITORY", action)
        self.assertIn("No agent-trace sessions found; skipping PR comment", action)
        self.assertIn("pr-comment", action)


if __name__ == "__main__":
    unittest.main()
