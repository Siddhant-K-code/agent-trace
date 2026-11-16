"""Tests for external evidence signals read by ``health`` and ``review``."""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from agent_trace.cli import build_parser
from agent_trace.external_evidence import (
    ExternalEvidenceError,
    assess_commits,
    assess_otel_logs,
    collect_external_evidence,
    copilot_agent_coauthored,
    copilot_log_urls,
    entire_checkpoints,
    final_trailer_block,
    is_copilot_agent,
)
from agent_trace.health import cmd_health
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.pr_comment import COMMENT_MARKER, NO_SESSION_NOTE, cmd_pr_comment
from agent_trace.reviewability import (
    LinkedSession,
    Threshold,
    build_report,
    cmd_review,
    render_markdown,
)
from agent_trace.store import TraceStore

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "external_evidence"
SECRET = "zebra-quartz-77"
ULID = "01JABCDEFGHJKMNPQRSTVWXYZ0"
LEGACY = "abcdef123456"
AGENT_COAUTHOR = (
    "Co-authored-by: copilot-swe-agent[bot] <198982749+Copilot@users.noreply.github.com>"
)
APP_COAUTHOR = "Co-authored-by: Copilot App <223556219+Copilot@users.noreply.github.com>"
SESSION_URL = "https://github.com/octo-org/sheet/sessions/7f6a2b1c-1d2e-4f3a-9b8c-0d1e2f3a4b5c"
HAS_GIT = shutil.which("git") is not None


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _by_source(signals, source):
    return [signal for signal in signals if signal.source == source]


class TestTrailerParsing(unittest.TestCase):
    def test_final_block_links_checkpoint(self):
        message = f"feat: a\n\nBody.\n\nEntire-Checkpoint: {ULID}\nCo-authored-by: A <a@b>\n"
        self.assertEqual(entire_checkpoints(message), ([ULID], 0))

    def test_legacy_id_and_crlf(self):
        message = f"fix\r\n\r\nEntire-Checkpoint: {LEGACY}\r\n"
        self.assertEqual(entire_checkpoints(message), ([LEGACY], 0))

    def test_body_line_does_not_link(self):
        linked, unlinked = entire_checkpoints(_fixture("entire_body_trailer_message.txt"))
        self.assertEqual(linked, [])
        self.assertEqual(unlinked, 1)

    def test_subject_is_never_a_trailer_block(self):
        self.assertEqual(final_trailer_block(f"Entire-Checkpoint: {ULID}"), [])
        self.assertEqual(entire_checkpoints(f"Entire-Checkpoint: {ULID}"), ([], 1))

    def test_invalid_ids_do_not_link(self):
        # A ULID cannot start above 7; uppercase hex is not a legacy ID.
        for bad in ("81JABCDEFGHJKMNPQRSTVWXYZ0", "ABCDEF123456", "abc"):
            with self.subTest(bad=bad):
                self.assertEqual(entire_checkpoints(f"x\n\nEntire-Checkpoint: {bad}\n"), ([], 1))

    def test_prose_in_last_paragraph_breaks_the_block(self):
        message = f"x\n\nSee the notes.\nEntire-Checkpoint: {ULID}\n"
        self.assertEqual(final_trailer_block(message), [])

    def test_git_generated_trailer_allows_other_lines(self):
        message = f"x\n\nSigned-off-by: A <a@b>\nnote\nEntire-Checkpoint: {ULID}\n"
        self.assertEqual(entire_checkpoints(message), ([ULID], 0))

    def test_copilot_urls_in_squash_body(self):
        self.assertEqual(copilot_log_urls(_fixture("copilot_squash_message.txt")), [SESSION_URL])

    def test_copilot_agent_identity_is_strict(self):
        self.assertTrue(is_copilot_agent("copilot-swe-agent[bot]", "x@example.com"))
        self.assertTrue(is_copilot_agent("Copilot", "198982749+Copilot@users.noreply.github.com"))
        self.assertTrue(is_copilot_agent(
            "x", "198982749+copilot-swe-agent[bot]@users.noreply.github.com"
        ))
        for name, email in (
            ("Copilot App", "223556219+Copilot@users.noreply.github.com"),
            ("Copilot", "copilot@example.com"),
            ("copilot-swe-agent", "1198982749+Copilot@users.noreply.github.com"),
            ("octocat", "198982749+Copilot@users.noreply.github.com.evil"),
        ):
            with self.subTest(name=name, email=email):
                self.assertFalse(is_copilot_agent(name, email))

    def test_copilot_agent_coauthor_lines(self):
        self.assertTrue(copilot_agent_coauthored(f"Add helper\n\n{AGENT_COAUTHOR}\n"))
        self.assertTrue(copilot_agent_coauthored(_fixture("copilot_coauthor_squash_message.txt")))
        self.assertFalse(copilot_agent_coauthored(f"Local change\n\n{APP_COAUTHOR}\n"))
        self.assertFalse(copilot_agent_coauthored("x\n\nThanks copilot-swe-agent[bot] for this.\n"))


@unittest.skipUnless(HAS_GIT, "git is required")
class _RepoCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.repo = str(Path(tmp.name).resolve() / "repo")
        os.makedirs(self.repo)
        env = patch.dict(os.environ, {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com",
        })
        env.start()
        self.addCleanup(env.stop)
        self.git("init", "-q", "-b", "main")
        self.commit("base")
        self.base = self.git("rev-parse", "HEAD")

    def git(self, *args: str, input: str | None = None, env: dict | None = None) -> str:
        proc = subprocess.run(
            ["git", "-c", "commit.gpgsign=false", *args], cwd=self.repo, input=input,
            capture_output=True, text=True, check=True, env={**os.environ, **(env or {})},
        )
        return proc.stdout.strip()

    def commit(self, message: str) -> str:
        self.git("commit", "-q", "--allow-empty", "-F", "-", input=message)
        return self.git("rev-parse", "HEAD")

    def empty_commit(self) -> str:
        tree = self.git("mktree", input="")
        return self.git("commit-tree", tree, "-m", "checkpoint")

    def entire_ref(self, checkpoint_id: str, *, queued: bool) -> str:
        ref = f"refs/entire/checkpoints/{checkpoint_id[-2:]}/{checkpoint_id}"
        sha = self.empty_commit()
        self.git("update-ref", ref, sha)
        if queued:
            with open(Path(self.repo) / ".git" / "entire-checkpoint-push-queue.jsonl", "a") as fh:
                fh.write(json.dumps({"ref": ref, "hash": sha}) + "\n")
        return ref

    def v1_commit(self, checkpoint_id: str | None) -> str:
        index = str(Path(self.repo) / ".git" / f"v1-index-{checkpoint_id}")
        env = {"GIT_INDEX_FILE": index}
        if checkpoint_id:
            blob = self.git("hash-object", "-w", "--stdin", input='{"checkpoint_id": "x"}')
            path = f"{checkpoint_id[:2]}/{checkpoint_id[2:]}/metadata.json"
            self.git("update-index", "--add", "--cacheinfo", f"100644,{blob},{path}", env=env)
            tree = self.git("write-tree", env=env)
        else:
            tree = self.git("mktree", input="")
        return self.git("commit-tree", tree, "-m", f"Checkpoint: {checkpoint_id}")

    def policy(self, value: str) -> None:
        path = Path(self.repo) / ".git" / "entire"
        path.mkdir(exist_ok=True)
        (path / "checkpoint-push-policy").write_text(value + "\n")

    def signals(self, rev_range: str = ""):
        return assess_commits(rev_range or f"{self.base}..HEAD", self.repo)


class TestEntireSignals(_RepoCase):
    def test_trailer_present_not_pushed(self):
        self.commit(f"feat\n\nEntire-Checkpoint: {ULID}\n")
        self.entire_ref(ULID, queued=True)
        self.policy("manual")
        [signal] = _by_source(self.signals(), "entire_checkpoint")
        self.assertEqual((signal.status, signal.confidence), ("unpushed", "high"))
        self.assertEqual(signal.subject, ULID)
        self.assertIn("Trailer present in 1 commit, not pushed", signal.detail)
        self.assertIn("push policy manual", signal.detail)
        self.assertEqual(signal.facts["backend"], "git-refs")
        self.assertTrue(signal.facts["queued"])
        self.assertEqual(signal.facts["push_policy"], "manual")

    def test_local_and_not_queued_is_present_with_remote_unchecked(self):
        self.commit(f"feat\n\nEntire-Checkpoint: {ULID}\n")
        self.entire_ref(ULID, queued=False)
        [signal] = _by_source(self.signals(), "entire_checkpoint")
        self.assertEqual((signal.status, signal.confidence), ("present", "medium"))
        self.assertIn("The remote was not checked", signal.detail)
        self.assertEqual(signal.facts["push_policy"], "automatic")

    def test_unreadable_policy_and_push_sessions_off(self):
        self.commit(f"feat\n\nEntire-Checkpoint: {ULID}\n")
        self.entire_ref(ULID, queued=False)
        self.policy("someday")
        settings = Path(self.repo) / ".entire"
        settings.mkdir()
        (settings / "settings.json").write_text('{"strategy_options": {"push_sessions": true}}')
        (settings / "settings.local.json").write_text('{"strategy_options": {"push_sessions": false}}')
        [signal] = _by_source(self.signals(), "entire_checkpoint")
        self.assertEqual((signal.status, signal.confidence), ("unpushed", "medium"))
        self.assertEqual(signal.facts["push_policy"], "unreadable")
        self.assertIs(signal.facts["push_sessions"], False)

    def test_checkpoint_not_in_clone_is_inconclusive(self):
        self.commit(f"feat\n\nEntire-Checkpoint: {ULID}\n")
        [signal] = _by_source(self.signals(), "entire_checkpoint")
        self.assertEqual((signal.status, signal.confidence), ("inconclusive", "low"))
        self.assertIn("may be on the remote", signal.detail)

    def test_git_branch_backend(self):
        self.commit(f"feat\n\nEntire-Checkpoint: {LEGACY}\n")
        with_checkpoint = self.v1_commit(LEGACY)
        self.git("update-ref", "refs/heads/entire/checkpoints/v1", with_checkpoint)
        [signal] = _by_source(self.signals(), "entire_checkpoint")
        self.assertEqual((signal.status, signal.confidence), ("unpushed", "low"))
        self.assertEqual(signal.facts["backend"], "git-branch")

        self.git("update-ref", "refs/remotes/origin/entire/checkpoints/v1", self.v1_commit(None))
        [signal] = _by_source(self.signals(), "entire_checkpoint")
        self.assertEqual((signal.status, signal.confidence), ("unpushed", "medium"))

        self.git("update-ref", "refs/remotes/origin/entire/checkpoints/v1", with_checkpoint)
        [signal] = _by_source(self.signals(), "entire_checkpoint")
        self.assertEqual((signal.status, signal.confidence), ("present", "medium"))
        self.assertEqual(
            signal.facts["on_remote_tracking"], ["refs/remotes/origin/entire/checkpoints/v1"]
        )

    def test_body_trailer_is_inconclusive_not_linked(self):
        self.commit(_fixture("entire_body_trailer_message.txt"))
        [signal] = _by_source(self.signals(), "entire_checkpoint")
        self.assertEqual((signal.status, signal.confidence), ("inconclusive", "medium"))
        self.assertIn("do not link", signal.detail)
        self.assertEqual(signal.subject, "")


class TestCommitSignals(_RepoCase):
    def test_no_trailers_is_missing_for_both(self):
        self.commit("plain change\n\nNo trailers here.\n")
        signals = self.signals()
        self.assertEqual(
            [(s.source, s.status, s.confidence) for s in signals],
            [("entire_checkpoint", "missing", "high"), ("copilot_agent_logs", "missing", "high")],
        )
        self.assertEqual(signals[0].facts["commits_read"], 1)

    def test_copilot_link_is_present_at_low_confidence(self):
        sha = self.commit(_fixture("copilot_squash_message.txt"))
        [signal] = _by_source(self.signals(), "copilot_agent_logs")
        self.assertEqual((signal.status, signal.confidence), ("present", "low"))
        self.assertEqual(signal.subject, SESSION_URL)
        self.assertEqual(signal.facts["commits"], [sha])
        self.assertIn("not read", signal.detail)
        self.assertIn("does not show complete evidence", signal.detail)

    def test_agent_coauthor_without_link_is_inconclusive(self):
        # The commit shape seen on Siddhant-K-code/agent-trace-demo#3 on 2026-11-16.
        self.commit("Initial plan")
        sha = self.commit(f"Add char_count text helper\n\n{AGENT_COAUTHOR}\n")
        [signal] = _by_source(self.signals(), "copilot_agent_logs")
        self.assertEqual((signal.status, signal.confidence), ("inconclusive", "medium"))
        self.assertEqual(signal.subject, "")
        self.assertEqual(
            signal.detail,
            "1 of 2 commits read names the Copilot cloud agent as author or co-author, but "
            "no commit has an Agent-Logs-Url trailer, so no session log link was found.",
        )
        self.assertEqual(signal.facts["commits"], [sha])
        self.assertEqual((signal.facts["agent_authored"], signal.facts["agent_coauthored"]), (0, 1))
        self.assertEqual(signal.facts["commits_read"], 2)

    def test_agent_author_without_link_is_inconclusive(self):
        agent = {
            "GIT_AUTHOR_NAME": "Copilot",
            "GIT_AUTHOR_EMAIL": "198982749+Copilot@users.noreply.github.com",
        }
        self.git("commit", "-q", "--allow-empty", "-m", "a", env=agent)
        self.git("commit", "-q", "--allow-empty", "-m", "b", env=agent)
        [signal] = _by_source(self.signals(), "copilot_agent_logs")
        self.assertEqual((signal.status, signal.confidence), ("inconclusive", "medium"))
        self.assertIn("2 of 2 commits read name the Copilot cloud agent", signal.detail)
        self.assertEqual((signal.facts["agent_authored"], signal.facts["agent_coauthored"]), (2, 0))

    def test_agent_coauthor_in_squash_body_is_inconclusive(self):
        self.commit(_fixture("copilot_coauthor_squash_message.txt"))
        [signal] = _by_source(self.signals(), "copilot_agent_logs")
        self.assertEqual((signal.status, signal.confidence), ("inconclusive", "medium"))

    def test_copilot_app_coauthor_is_still_missing(self):
        self.commit(f"Local change\n\n{APP_COAUTHOR}\n")
        [signal] = _by_source(self.signals(), "copilot_agent_logs")
        self.assertEqual((signal.status, signal.confidence), ("missing", "high"))
        self.assertEqual(
            signal.detail,
            "No Agent-Logs-Url trailer and no Copilot cloud agent author or co-author; "
            "1 commit read.",
        )

    def test_link_wins_over_agent_commits(self):
        self.commit(f"a\n\n{AGENT_COAUTHOR}\n")
        self.commit(f"b\n\nAgent-Logs-Url: {SESSION_URL}\n{AGENT_COAUTHOR}\n")
        signals = _by_source(self.signals(), "copilot_agent_logs")
        self.assertEqual([(s.status, s.confidence) for s in signals], [("present", "low")])

    def test_unrecognized_copilot_link_is_inconclusive(self):
        self.commit("x\n\nAgent-Logs-Url: https://example.com/logs?id=1\n")
        self.commit("y\n\nAgent-Logs-Url: javascript:alert(1)\n")
        signals = _by_source(self.signals(), "copilot_agent_logs")
        self.assertEqual([s.status for s in signals], ["inconclusive", "inconclusive"])
        self.assertEqual([s.subject for s in signals], ["https://example.com/logs?id=1", ""])

    def test_unreadable_and_empty_ranges(self):
        [signal] = self.signals("no-such-ref..HEAD")
        self.assertEqual((signal.source, signal.status), ("commit_trailers", "inconclusive"))
        [signal] = self.signals("HEAD..HEAD")
        self.assertEqual((signal.source, signal.status), ("commit_trailers", "inconclusive"))

    def test_range_is_validated(self):
        for bad in ("--output=/tmp/x", "a b", "", "HEAD;rm"):
            with self.subTest(bad=bad), self.assertRaises(ExternalEvidenceError):
                collect_external_evidence(bad or " ", cwd=self.repo)
        self.assertIsNone(collect_external_evidence(cwd=self.repo))

    def test_not_a_git_clone(self):
        with tempfile.TemporaryDirectory() as plain:
            [signal] = assess_commits("HEAD", plain)
        self.assertEqual((signal.source, signal.status), ("commit_trailers", "inconclusive"))


class TestClaudeOtel(unittest.TestCase):
    def test_safety_events_present(self):
        signal = assess_otel_logs(str(FIXTURES / "claude_otel_safety.json"))
        self.assertEqual((signal.status, signal.confidence), ("present", "medium"))
        self.assertEqual(signal.subject, "claude_otel_safety.json")
        facts = signal.facts
        self.assertEqual(facts["safety_events"], 3)
        self.assertEqual(
            facts["safety_categories"],
            {"hook_review": 1, "permission_rule": 1, "sandbox_network": 1},
        )
        self.assertEqual(facts["safety_decisions"], {"reject": 2, "sent_back": 1})
        self.assertEqual(facts["tool_decision_rejects"], 2)
        self.assertEqual(facts["rejects_without_safety_event"], 1)
        self.assertEqual(facts["versions"], ["2.1.310"])
        self.assertEqual(facts["sessions"], 1)
        self.assertIn("1 rejected tool decision has no matching safety event", signal.detail)
        self.assertIn("count is a floor", signal.detail)
        self.assertNotIn(SECRET, json.dumps(signal.to_dict()))

    def test_no_safety_events_is_inconclusive_not_none(self):
        signal = assess_otel_logs(str(FIXTURES / "claude_otel_no_safety.jsonl"))
        self.assertEqual((signal.status, signal.confidence), ("inconclusive", "medium"))
        self.assertEqual(signal.facts["unparsed_lines"], 1)
        self.assertEqual(signal.facts["claude_code_events"], 3)
        self.assertEqual(signal.facts["sessions"], 2)
        self.assertEqual(signal.facts["tool_decision_rejects"], 1)
        self.assertIn("Absence does not show that nothing was refused", signal.detail)
        self.assertNotIn(SECRET, json.dumps(signal.to_dict()))

    def test_old_version_predates_safety_event(self):
        signal = assess_otel_logs(str(FIXTURES / "claude_otel_old_version.jsonl"))
        self.assertEqual(signal.status, "inconclusive")
        self.assertIn("predates 2.1.307", signal.detail)

    def test_other_tool_export_is_missing(self):
        signal = assess_otel_logs(str(FIXTURES / "other_service_otel.json"))
        self.assertEqual((signal.status, signal.confidence), ("missing", "high"))

    def test_unreadable_file(self):
        signal = assess_otel_logs(str(FIXTURES / "does-not-exist.json"))
        self.assertEqual((signal.status, signal.confidence), ("inconclusive", "low"))


class TestShowsAgent(_RepoCase):
    def test_commit_signals(self):
        self.commit("plain change\n")
        human = collect_external_evidence(f"{self.base}..HEAD", cwd=self.repo)
        self.assertFalse(human.shows_agent)
        self.assertFalse(human.to_dict()["shows_agent"])

        start = self.git("rev-parse", "HEAD")
        self.commit(f"Add helper\n\n{AGENT_COAUTHOR}\n")
        self.commit("ci: run checks\n")
        agent = collect_external_evidence(f"{start}..HEAD", cwd=self.repo)
        self.assertEqual([s.shows_agent for s in agent.signals], [False, True])
        self.assertTrue(agent.to_dict()["shows_agent"])

    def test_entire_and_copilot_link_signals(self):
        self.commit(f"feat\n\nEntire-Checkpoint: {ULID}\n")
        [signal] = _by_source(self.signals(), "entire_checkpoint")
        self.assertEqual(signal.status, "inconclusive")
        self.assertTrue(signal.shows_agent)
        self.entire_ref(ULID, queued=True)
        [signal] = _by_source(self.signals(), "entire_checkpoint")
        self.assertEqual(signal.status, "unpushed")
        self.assertTrue(signal.shows_agent)
        self.commit(_fixture("copilot_squash_message.txt"))
        [signal] = _by_source(self.signals(), "copilot_agent_logs")
        self.assertEqual(signal.status, "present")
        self.assertTrue(signal.shows_agent)

    def test_unreadable_or_empty_range_does_not_count(self):
        for rev_range in ("no-such-ref..HEAD", "HEAD..HEAD"):
            with self.subTest(rev_range=rev_range):
                evidence = collect_external_evidence(rev_range, cwd=self.repo)
                [signal] = evidence.signals
                self.assertEqual(signal.status, "inconclusive")
                self.assertFalse(evidence.shows_agent)
        [signal] = self.signals("no-such-ref..HEAD")
        self.assertTrue(signal.facts["read_error"])
        [signal] = self.signals("HEAD..HEAD")
        self.assertNotIn("read_error", signal.facts)

    def test_otel_signals(self):
        cases = {
            "claude_otel_safety.json": True,
            "claude_otel_no_safety.jsonl": True,
            "claude_otel_old_version.jsonl": True,
            "other_service_otel.json": False,
            "does-not-exist.json": False,
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                signal = assess_otel_logs(str(FIXTURES / name))
                self.assertEqual(signal.shows_agent, expected)
        signal = assess_otel_logs(str(FIXTURES / "does-not-exist.json"))
        self.assertEqual(signal.status, "inconclusive")
        self.assertTrue(signal.facts["read_error"])


def _session(store: TraceStore, session_id: str = "ext000000001") -> str:
    store.create_session(SessionMeta(
        session_id=session_id, started_at=1000.0, ended_at=1100.0, agent_name="claude-code",
        attribution={"git_branch": "main"},
    ))
    for offset, (kind, data) in enumerate((
        (EventType.SESSION_START, {"mode": "claude-code-hooks", "cwd": "/w", "redaction": "on"}),
        (EventType.USER_PROMPT, {"prompt": f"fix it {SECRET}"}),
        (EventType.ASSISTANT_RESPONSE, {"text": "done", "hook_event": "stop", "provider": "claude"}),
        (EventType.SESSION_END, {}),
    )):
        store.append_event(session_id, TraceEvent(
            event_type=kind, timestamp=1001.0 + offset, session_id=session_id, data=data,
        ))
    return session_id


class TestHealthAndReview(_RepoCase):
    def setUp(self):
        super().setUp()
        self.commit(f"feat\n\nEntire-Checkpoint: {ULID}\n")
        self.entire_ref(ULID, queued=True)
        self.commit(_fixture("copilot_squash_message.txt"))
        self.trace_dir = str(Path(self.repo) / ".agent-traces")
        self.store = TraceStore(self.trace_dir)
        self.session_id = _session(self.store)
        cwd = os.getcwd()
        os.chdir(self.repo)
        self.addCleanup(os.chdir, cwd)
        self.range = f"{self.base}..HEAD"
        self.otel = str(FIXTURES / "claude_otel_safety.json")

    def run_cmd(self, handler, *argv):
        args = build_parser().parse_args(["--trace-dir", self.trace_dir, *argv])
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = handler(args)
        return code, out.getvalue(), err.getvalue()

    def test_health_json_adds_facts_and_keeps_status(self):
        code, plain, _ = self.run_cmd(cmd_health, "health", "--format", "json")
        code_ext, out, err = self.run_cmd(
            cmd_health, "health", "--format", "json", "--commits", self.range,
            "--otel-logs", self.otel,
        )
        self.assertEqual(err, "")
        before, after = json.loads(plain), json.loads(out)
        self.assertNotIn("external", before)
        self.assertEqual(code, code_ext)
        self.assertEqual(before["status"], after["status"])
        external = after.pop("external")
        self.assertEqual(before, after)
        self.assertFalse(external["graded"])
        self.assertEqual(external["commits"], self.range)
        self.assertEqual(external["otel_logs"], "claude_otel_safety.json")
        self.assertEqual(
            [(s["source"], s["status"]) for s in external["signals"]],
            [
                ("entire_checkpoint", "unpushed"),
                ("copilot_agent_logs", "present"),
                ("claude_code_otel", "present"),
            ],
        )
        self.assertNotIn(SECRET, out)

    def test_health_text_section(self):
        code, out, err = self.run_cmd(cmd_health, "health", "--commits", self.range)
        self.assertEqual(err, "")
        self.assertIn(f"External evidence (commits {self.range}):", out)
        self.assertIn(f"Entire checkpoint {ULID}: unpushed (high). Trailer present", out)
        self.assertIn("Reported, not graded", out)

    def test_health_rejects_bad_range(self):
        code, out, err = self.run_cmd(cmd_health, "health", "--commits=--all")
        self.assertEqual(code, 2)
        self.assertIn("--commits must be a plain git revision range", err)

    def test_review_facts_never_change_findings_or_exit(self):
        linked = [LinkedSession(self.session_id, "branch")]
        external = collect_external_evidence(self.range, self.otel, cwd=self.repo)
        plain = build_report(self.store, linked, Threshold(), branch="main")
        report = build_report(self.store, linked, Threshold(), branch="main", external=external)
        self.assertEqual(
            [(f.name, f.status, f.confidence, f.detail) for f in plain.findings],
            [(f.name, f.status, f.confidence, f.detail) for f in report.findings],
        )
        self.assertEqual(plain.exit_code, report.exit_code)
        evidence = next(f for f in report.findings if f.name == "evidence")
        self.assertEqual(evidence.facts["external"], external.to_dict())
        markdown = render_markdown(report)
        self.assertIn("**External evidence** (commits `", markdown)
        self.assertIn(f"- Entire checkpoint `{ULID}`: unpushed (high).", markdown)
        self.assertIn("- Claude Code OTel `claude_otel_safety.json`: present (medium).", markdown)
        self.assertNotIn(SECRET, markdown)

    def test_agent_commits_without_link_never_change_findings_or_exit(self):
        start = self.git("rev-parse", "HEAD")
        self.commit(f"Add helper\n\n{AGENT_COAUTHOR}\n")
        external = collect_external_evidence(f"{start}..HEAD", cwd=self.repo)
        [signal] = _by_source(external.signals, "copilot_agent_logs")
        self.assertEqual(signal.status, "inconclusive")
        linked = [LinkedSession(self.session_id, "branch")]
        plain = build_report(self.store, linked, Threshold(), branch="main")
        report = build_report(self.store, linked, Threshold(), branch="main", external=external)
        self.assertEqual(
            [(f.name, f.status, f.confidence, f.detail) for f in plain.findings],
            [(f.name, f.status, f.confidence, f.detail) for f in report.findings],
        )
        self.assertEqual(plain.exit_code, report.exit_code)

    def test_review_cli_json_and_no_session(self):
        code, out, err = self.run_cmd(
            cmd_review, "review", "--format", "json", "--commits", self.range,
        )
        self.assertEqual(err, "")
        payload = json.loads(out)
        evidence = next(f for f in payload["findings"] if f["name"] == "evidence")
        self.assertEqual(
            [s["source"] for s in evidence["facts"]["external"]["signals"]],
            ["entire_checkpoint", "copilot_agent_logs"],
        )
        code, out, err = self.run_cmd(cmd_review, "review", "--commits", "a b")
        self.assertEqual(code, 2)
        self.assertIn("review: --commits", err)

    def test_review_json_has_top_level_external(self):
        code, out, err = self.run_cmd(
            cmd_review, "review", "--format", "json", "--commits", self.range,
        )
        payload = json.loads(out)
        evidence = next(f for f in payload["findings"] if f["name"] == "evidence")
        self.assertEqual(payload["external"], evidence["facts"]["external"])
        self.assertTrue(payload["external"]["shows_agent"])
        code, out, err = self.run_cmd(cmd_review, "review", "--format", "json")
        self.assertNotIn("external", json.loads(out))

    def test_pr_comment_review_dry_run(self):
        code, out, err = self.run_cmd(
            cmd_pr_comment, "pr-comment", "--dry-run", "--review", "--commits", self.range,
        )
        self.assertIn("**External evidence**", out)
        self.assertIn("Copilot agent logs", out)


class TestPrCommentWithoutSession(_RepoCase):
    def setUp(self):
        super().setUp()
        self.commit("Initial plan")
        self.commit(f"Add char_count text helper\n\n{AGENT_COAUTHOR}\n")
        self.commit("ci: run checks on the cloud agent PR")
        self.trace_dir = str(Path(self.repo) / ".agent-traces")
        cwd = os.getcwd()
        os.chdir(self.repo)
        self.addCleanup(os.chdir, cwd)
        self.range = f"{self.base}..HEAD"

    def run_cmd(self, *argv):
        args = build_parser().parse_args(["--trace-dir", self.trace_dir, *argv])
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cmd_pr_comment(args)
        return code, out.getvalue(), err.getvalue()

    def test_review_renders_findings_and_external_evidence(self):
        code, out, err = self.run_cmd(
            "pr-comment", "--dry-run", "--review", "--commits", self.range,
        )
        self.assertEqual((code, err), (0, ""))
        self.assertTrue(out.startswith(COMMENT_MARKER))
        self.assertEqual(out.count(COMMENT_MARKER), 1)
        self.assertIn("## agent-trace reviewability", out)
        self.assertIn(NO_SESSION_NOTE, out)
        self.assertIn("### Findings", out)
        for label in ("Rationale", "Scope", "Validation", "Stewardship", "Evidence health"):
            self.assertIn(f"| {label} | inconclusive | low |", out)
        self.assertIn("**External evidence** (commits `", out)
        self.assertIn(
            "- Copilot agent logs: inconclusive (medium). 1 of 3 commits read names the "
            "Copilot cloud agent", out,
        )
        self.assertIn("- Entire checkpoint: missing (high).", out)
        for absent in ("session summary\n", "### What the agent did", "### Flags",
                       "agent-strace replay", "| **Session** |", "No redundant reads"):
            self.assertNotIn(absent, out)

    def test_threshold_still_sets_the_exit_code(self):
        code, out, _ = self.run_cmd(
            "pr-comment", "--dry-run", "--review", "--commits", self.range,
            "--fail-on", "inconclusive", "--min-confidence", "low",
        )
        self.assertEqual(code, 1)
        self.assertIn("**inconclusive** (fails)", out)

    def test_without_review_or_with_a_named_session_it_still_errors(self):
        code, _, err = self.run_cmd("pr-comment", "--dry-run")
        self.assertEqual(code, 1)
        self.assertIn("No agent-trace sessions found", err)
        code, _, err = self.run_cmd("pr-comment", "nope", "--dry-run", "--review")
        self.assertEqual(code, 1)
        self.assertIn("Session not found", err)


class TestActionPassthrough(unittest.TestCase):
    def test_action_inputs_reach_the_script(self):
        action = (ROOT / "action.yml").read_text(encoding="utf-8")
        script = (ROOT / "scripts" / "pr-evidence.sh").read_text(encoding="utf-8")
        for name, env in (("review-commits", "REVIEW_COMMITS"), ("review-otel-logs", "REVIEW_OTEL_LOGS")):
            self.assertIn(f"  {name}:\n", action)
            self.assertIn(f"{env}: ${{{{ inputs.{name} }}}}", action)
            self.assertIn(env, script)
        self.assertIn('args+=(--commits "$REVIEW_COMMITS")', script)
        self.assertIn('args+=(--otel-logs "$REVIEW_OTEL_LOGS")', script)


if __name__ == "__main__":
    unittest.main()
