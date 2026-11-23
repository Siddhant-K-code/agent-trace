"""Tests for agent-strace freshness: what changed since the last session."""

from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from agent_trace.models import SessionMeta
from agent_trace.store import TraceStore

try:  # keeps product telemetry off in this process and its children
    import _no_telemetry
except ImportError:  # pragma: no cover - run as a package module
    from tests import _no_telemetry

SRC = str(Path(__file__).resolve().parent.parent / "src")
DAY = 86400
NOV15 = 1763164800  # 2025-11-15T00:00:00Z
T0 = NOV15 - 10 * DAY
SEP = "─" * 55
NO_TIME = (
    "session s1 has no usable start time. "
    "Pass --since YYYY-MM-DD to compare with a date."
)
NO_SESSIONS = "No sessions found. Pass --since YYYY-MM-DD to compare with a date.\n"


def _lines(count: int, word: str = "line") -> str:
    return "".join(f"{word} {n}\n" for n in range(count))


class TestScopeList(unittest.TestCase):
    """_parse_scope_from_agents_md reads the current directory, for context-score."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        old = os.getcwd()
        os.chdir(self.tmp)
        self.addCleanup(os.chdir, old)

    def test_parse_scope_no_file(self):
        from agent_trace.freshness import _parse_scope_from_agents_md
        self.assertEqual(_parse_scope_from_agents_md(), [])

    def test_parse_scope_from_claude_md(self):
        from agent_trace.freshness import _parse_scope_from_agents_md
        Path("CLAUDE.md").write_text("# Instructions\n\nScope:\n- src/**\n- tests/**\n")
        self.assertEqual(_parse_scope_from_agents_md(), ["src/**", "tests/**"])

    def test_star_bullets_and_backticks_are_stripped(self):
        from agent_trace.freshness import _parse_scope_from_agents_md
        Path("CLAUDE.md").write_text(
            "Files to watch:\n* `src/**`\n*\ttests/**\n- `docs/*.md`\n-\tlib/**\n**/*.py\n"
        )
        self.assertEqual(
            _parse_scope_from_agents_md(),
            ["src/**", "tests/**", "docs/*.md", "lib/**", "**/*.py"],
        )


class TestParser(unittest.TestCase):
    def test_cli_has_freshness_command(self):
        from agent_trace.cli import build_parser
        args = build_parser().parse_args(
            ["freshness", "--since", "2026-01-01", "--scope", "src/**", "--repo", "x"]
        )
        self.assertEqual((args.since, args.scope, args.repo), ("2026-01-01", "src/**", "x"))


@unittest.skipIf(shutil.which("git") is None, "git is not installed")
class TestFreshness(unittest.TestCase):
    def setUp(self):
        self.root = os.path.realpath(tempfile.mkdtemp(prefix="freshness-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for key in list(os.environ):
            if key.startswith("GIT_") or key in ("AGENT_TRACE_DIR", "AGENT_STRACE_WORKSPACE"):
                del os.environ[key]
        # A developer's git config (signing, diff.renames, core.quotePath)
        # must not change the result, and git must not find a parent repo.
        os.environ.update(
            GIT_CONFIG_GLOBAL=os.devnull,
            GIT_CONFIG_NOSYSTEM="1",
            GIT_CEILING_DIRECTORIES=self.root,
            GIT_AUTHOR_NAME="t",
            GIT_AUTHOR_EMAIL="t@example.com",
            GIT_COMMITTER_NAME="t",
            GIT_COMMITTER_EMAIL="t@example.com",
        )
        self.chdir(self.root)
        self.repo = os.path.join(self.root, "repo")
        os.makedirs(self.repo)
        self.git("init", "-q", "-b", "main")
        self.traces = os.path.join(self.root, "traces")
        self.store = TraceStore(self.traces)

    # -- helpers -----------------------------------------------------------

    def chdir(self, path: str) -> None:
        old = os.getcwd()
        os.chdir(path)
        self.addCleanup(os.chdir, old)

    def git(self, *args: str, when: int | None = None, cwd: str | None = None) -> str:
        env = None
        if when is not None:
            stamp = f"@{when} +0000"
            env = dict(os.environ, GIT_AUTHOR_DATE=stamp, GIT_COMMITTER_DATE=stamp)
        result = subprocess.run(
            ["git", "-C", cwd or self.repo, *args], env=env, capture_output=True
        )
        if result.returncode:
            self.fail(f"git {' '.join(args)}: {result.stderr.decode(errors='replace')}")
        return result.stdout.decode().strip()

    def write(self, name: str, content: str | bytes = "") -> None:
        path = Path(self.repo, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content if isinstance(content, bytes) else content.encode())

    def remove(self, name: str) -> None:
        os.remove(os.path.join(self.repo, name))

    def rename(self, old: str, new: str) -> None:
        os.makedirs(os.path.dirname(os.path.join(self.repo, new)), exist_ok=True)
        os.rename(os.path.join(self.repo, old), os.path.join(self.repo, new))

    def commit(self, when: int) -> None:
        self.git("add", "-A")
        self.git("commit", "-q", "--allow-empty", "-m", f"at {when}", when=when)

    def session(self, started_at: float, session_id: str = "s1") -> None:
        self.store.create_session(SessionMeta(session_id=session_id, started_at=started_at))

    def raw_session(self, started_at: str, session_id: str = "s1") -> None:
        """Store a session whose meta.json holds ``started_at`` as raw JSON."""
        path = Path(self.store.base_dir, session_id)
        path.mkdir(parents=True)
        (path / "meta.json").write_text(
            '{"session_id": "%s", "started_at": %s}' % (session_id, started_at)
        )
        (path / "events.ndjson").write_text("")

    def analyse(self, **kwargs):
        from agent_trace.freshness import analyse_freshness
        kwargs.setdefault("repo", self.repo)
        return analyse_freshness(self.store, **kwargs)

    def text(self, report) -> str:
        from agent_trace.freshness import format_freshness
        out = io.StringIO()
        format_freshness(report, out=out)
        return out.getvalue()

    @staticmethod
    def changes(report) -> list[tuple[str, str, int]]:
        return [(f.path, f.change_type, f.lines_changed) for f in report.stale_files]

    def run_cli(self, *argv: str) -> tuple[int, str, str]:
        from agent_trace.cli import main
        out, err = io.StringIO(), io.StringIO()
        with patch.object(sys, "argv", ["agent-strace", "--trace-dir", self.traces, *argv]), \
                patch("agent_trace.cli._product_telemetry.capture"), \
                patch("agent_trace.cli._product_telemetry.maybe_show_telemetry_notice"), \
                redirect_stdout(out), redirect_stderr(err), \
                self.assertRaises(SystemExit) as raised:
            main()
        return raised.exception.code, out.getvalue(), err.getvalue()

    def run_child(self, *argv: str) -> subprocess.CompletedProcess:
        path = os.pathsep.join(p for p in (SRC, os.environ.get("PYTHONPATH", "")) if p)
        env = _no_telemetry.without_telemetry(
            dict(os.environ, PYTHONPATH=path, PYTHONIOENCODING="utf-8", HOME=self.root)
        )
        return subprocess.run(
            [sys.executable, "-m", "agent_trace.entry", "--trace-dir", self.traces,
             "freshness", *argv],
            cwd=self.root, env=env, capture_output=True, text=True, encoding="utf-8",
        )

    # -- the report ----------------------------------------------------------

    def test_linear_history_report_keeps_its_layout(self):
        self.write("src/app.py", "alpha\nbeta\ngamma\n")
        self.write("docs/old.md", "one\ntwo\nthree\nfour\nfive\nsix\n")
        self.commit(T0)
        self.session(T0 + 60, "s-linear")
        self.write("src/app.py", "alpha\nBETA\nGAMMA\n")
        self.write("src/new.py", "x = 1\ny = 2\nz = 3\nw = 4\n")
        self.remove("docs/old.md")
        self.commit(T0 + DAY)

        lines = self.text(self.analyse(scope_glob="src/**")).split("\n")
        self.assertEqual(lines[:3], ["", "Context Freshness Report", SEP])
        self.assertRegex(lines[3], r"^Last session: s-linear \(\d+d ago\)$")
        self.assertEqual("\n".join(lines[4:]), (
            "Scope: src/**  [--scope flag]\n"
            "Files changed: 3 total, 2 in scope\n"
            "Freshness score: ⚠️  60/100  [" + "█" * 12 + "░" * 8 + "]\n"
            f"{SEP}\n"
            "\n"
            "Stale files in scope:\n"
            "\n"
            "  ⚠️  src/app.py\n"
            "       modified · 4 lines changed\n"
            "  ⚠️  src/new.py\n"
            "       added · 4 lines changed\n"
            "\n"
            "Out-of-scope changes: 1 files\n"
            "\n"
            "Estimated catch-up time: 2min\n"
            f"{SEP}\n"
            "\n"
        ))

    def test_no_change_since_the_session_is_fully_fresh(self):
        self.write("a.txt", "a\n")
        self.commit(T0)
        self.session(T0 + 60)
        report = self.analyse()
        self.assertEqual((report.files_changed_total, report.freshness_score), (0, 100))
        self.assertIn("Context is fully fresh", self.text(report))

    def test_newest_session_by_start_time_is_the_base(self):
        self.write("a.txt", "a\n")
        self.commit(T0)
        self.write("b.txt", "b\n")
        self.commit(T0 + 2 * DAY)
        self.session(T0 + 3 * DAY, "aa-new")
        self.session(T0 + DAY, "zz-old")
        report = self.analyse()
        self.assertEqual(report.last_session_id, "aa-new")
        self.assertEqual(report.last_session_ts, T0 + 3 * DAY)
        self.assertEqual(report.stale_files, [])

    def test_since_replaces_the_session_time(self):
        self.write("a.txt", "a\n")
        self.commit(NOV15 - 2 * DAY)
        self.write("b.txt", "b\n")
        self.commit(NOV15 + 2 * DAY)
        self.session(NOV15 + 3 * DAY)
        report = self.analyse(since_date="2025-11-15")
        self.assertEqual(self.changes(report), [("b.txt", "added", 1)])
        self.assertEqual(report.last_session_id, "s1")

    # -- the base commit -----------------------------------------------------

    def test_a_branch_merged_after_the_session_is_not_seen(self):
        # main2.txt was on main when the session started. side.txt was
        # committed on a branch before that and merged after it.
        self.write("a.txt", "a\n")
        self.commit(T0)
        self.git("checkout", "-q", "-b", "side")
        self.write("side.txt", "side\n")
        self.commit(T0 + 2 * DAY + DAY // 2)
        self.git("checkout", "-q", "main")
        self.write("main2.txt", "main2\n")
        self.commit(T0 + 2 * DAY)
        self.session(T0 + 3 * DAY)
        self.git("merge", "-q", "--no-ff", "-m", "merge", "side", when=T0 + 4 * DAY)
        self.assertEqual(self.changes(self.analyse()), [("side.txt", "added", 1)])

    def test_a_session_before_the_first_commit_sees_every_file_as_added(self):
        self.write("a.py", _lines(10))
        self.write("src/b.py", _lines(3, "b"))
        self.commit(T0)
        self.session(T0 - DAY)
        report = self.analyse()
        self.assertEqual(
            sorted(self.changes(report)),
            [("a.py", "added", 10), ("src/b.py", "added", 3)],
        )
        self.assertLess(report.freshness_score, 100)

    def test_since_before_1970_sees_every_file_as_added(self):
        self.write("a.py", "a\n")
        self.commit(T0)
        self.assertEqual(
            self.changes(self.analyse(since_date="1900-01-01")), [("a.py", "added", 1)]
        )

    def test_the_empty_tree_comes_from_the_repository(self):
        self.repo = os.path.join(self.root, "sha256")
        os.makedirs(self.repo)
        made = subprocess.run(
            ["git", "-C", self.repo, "init", "-q", "--object-format=sha256"],
            capture_output=True,
        )
        if made.returncode:
            self.skipTest("this git cannot make a SHA-256 repository")
        self.write("a.py", "a\n")
        self.commit(T0)
        self.session(T0 - DAY)
        self.assertEqual(self.changes(self.analyse()), [("a.py", "added", 1)])

    def test_a_time_after_2099_still_finds_the_base(self):
        # git cannot read an ISO date after 2099.
        self.write("a.py", "a\n")
        self.commit(4000000000)  # 2096-10-02
        self.session(7258118400)  # 2200-01-01
        report = self.analyse()
        self.assertEqual((report.stale_files, report.freshness_score), ([], 100))

    def test_a_file_named_like_a_revision_is_not_read_as_a_path(self):
        # git stops when an argument names both a revision and a file.
        self.write("a.txt", "a\n")
        self.commit(T0)
        self.session(T0 + 60)
        self.write("HEAD", "h\n")
        self.commit(T0 + DAY)
        self.write(self.git("rev-parse", "--verify", "HEAD"), "untracked\n")
        self.assertEqual(self.changes(self.analyse()), [("HEAD", "added", 1)])

    def test_since_means_midnight_utc(self):
        # Local time is UTC-12. git would read a bare date as that day at the
        # current local time, which is after the 06:00 UTC commit.
        old = os.environ.get("TZ")

        def restore_tz():
            if old is None:
                os.environ.pop("TZ", None)
            else:
                os.environ["TZ"] = old
            if hasattr(time, "tzset"):
                time.tzset()

        self.addCleanup(restore_tz)
        os.environ["TZ"] = "TST+12"
        if hasattr(time, "tzset"):
            time.tzset()
        self.write("a.txt", "a\n")
        self.commit(NOV15 - DAY)
        self.write("b.txt", "b\n")
        self.commit(NOV15 + 6 * 3600)
        report = self.analyse(since_date="2025-11-15")
        self.assertEqual(self.changes(report), [("b.txt", "added", 1)])

    def test_a_shallow_clone_without_the_base_is_an_error(self):
        from agent_trace.freshness import FreshnessError
        self.write("a.txt", "a\n")
        self.commit(T0)
        self.write("b.txt", "b\n")
        self.commit(T0 + DAY)
        clone = os.path.join(self.root, "clone")
        subprocess.run(
            ["git", "clone", "-q", "--depth", "1", Path(self.repo).as_uri(), clone],
            check=True, capture_output=True,
        )
        self.session(T0 + 60)
        with self.assertRaises(FreshnessError) as caught:
            self.analyse(repo=clone)
        self.assertIn(f"the git repository at {clone} is a shallow clone", str(caught.exception))

    # -- names and change types ----------------------------------------------

    def test_a_rename_inside_a_directory_uses_the_new_name(self):
        self.write("src/old/x.py", _lines(10))
        self.write("README.md", "readme\n")
        self.commit(T0)
        self.session(T0 + 60)
        self.rename("src/old/x.py", "src/new/x.py")
        self.write("src/new/x.py", _lines(9) + "changed\n")
        self.commit(T0 + DAY)
        report = self.analyse(scope_glob="src/**")
        self.assertEqual(self.changes(report), [("src/new/x.py", "renamed", 2)])
        self.assertEqual(report.files_in_scope, 1)

    def test_change_types_come_from_git(self):
        self.write("src/a.py", _lines(10, "a"))
        self.write("docs/b.md", _lines(10, "b"))
        self.commit(T0)
        self.session(T0 + 60)
        self.write("src/a.py", _lines(15, "a"))  # only gained lines
        self.write("docs/b.md", _lines(4, "b"))  # only lost lines
        self.write("empty.txt", "")
        self.write("img.bin", b"\x00\xff" * 8)
        self.commit(T0 + DAY)
        self.assertEqual(sorted(self.changes(self.analyse())), [
            ("docs/b.md", "modified", 6),
            ("empty.txt", "added", 0),
            ("img.bin", "added", 0),
            ("src/a.py", "modified", 5),
        ])

    def test_a_file_that_becomes_a_symlink_is_modified(self):
        self.write("target.txt", "t\n")
        self.write("link.txt", "")
        self.commit(T0)
        self.session(T0 + 60)
        self.remove("link.txt")
        try:
            os.symlink("target.txt", os.path.join(self.repo, "link.txt"))
        except (OSError, NotImplementedError):
            self.skipTest("cannot make a symlink here")
        self.commit(T0 + DAY)
        # git reports a type change. The file only gained a line.
        self.assertEqual(self.changes(self.analyse()), [("link.txt", "modified", 1)])

    def test_since_reports_types_and_new_names(self):
        # The commits are two days from the date, so the base is the same
        # in every time zone.
        self.write("src/old/x.py", _lines(10))
        self.write("gone.txt", "gone\n" * 3)
        self.commit(NOV15 - 2 * DAY)
        self.rename("src/old/x.py", "src/new/x.py")
        self.write("src/new/x.py", _lines(9) + "changed\n")
        self.write("new.txt", "new\n")
        self.remove("gone.txt")
        self.commit(NOV15 + 2 * DAY)
        report = self.analyse(since_date="2025-11-15", scope_glob="src/**")
        self.assertEqual(sorted(self.changes(report)), [
            ("gone.txt", "deleted", 3),
            ("new.txt", "added", 1),
            ("src/new/x.py", "renamed", 2),
        ])
        self.assertEqual(report.files_in_scope, 1)
        self.assertIn("\nLast session: none found\n", self.text(report))

    def test_a_non_ascii_name_is_not_quoted(self):
        name = "caf\u00e9.py"
        self.write("README.md", "readme\n")
        self.commit(T0)
        self.session(T0 + 60)
        self.write(name, "c\n")
        self.commit(T0 + DAY)
        report = self.analyse(scope_glob="*.py")
        self.assertEqual(self.changes(report), [(name, "added", 1)])
        self.assertEqual(report.files_in_scope, 1)
        self.assertIn(f"  ⚠️  {name}\n", self.text(report))

    def test_a_control_character_in_a_name_is_printed_as_an_escape(self):
        name = "evil\x1b[2J.txt"
        self.write("README.md", "readme\n")
        self.commit(T0)
        self.session(T0 + 60)
        try:
            self.write(name, "e\n")
        except OSError:
            self.skipTest("this file system refuses the name")
        self.commit(T0 + DAY)
        report = self.analyse()
        self.assertEqual(self.changes(report), [(name, "added", 1)])
        text = self.text(report)
        self.assertIn("  ⚠️  evil\\x1b[2J.txt\n", text)
        self.assertNotIn("\x1b", text)

    def test_names_stay_relative_to_the_top_level(self):
        self.write("src/old.py", _lines(10))
        self.write("docs/b.md", "b\n")
        self.commit(T0)
        self.session(T0 + 60)
        self.rename("src/old.py", "src/new.py")
        self.write("docs/b.md", "b\nc\n")
        self.commit(T0 + DAY)
        for key, value in (
            ("diff.relative", "true"),
            ("diff.renames", "false"),
            ("color.ui", "always"),
        ):
            self.git("config", key, value)
        report = self.analyse(repo=os.path.join(self.repo, "docs"))
        self.assertEqual(
            sorted(self.changes(report)),
            [("docs/b.md", "modified", 1), ("src/new.py", "renamed", 0)],
        )

    # -- the scope list ------------------------------------------------------

    def _scope_repo(self) -> None:
        self.write("src/a.py", "a\n")
        self.write("docs/b.md", "b\n")
        self.commit(T0)
        self.session(T0 + 60)
        self.write("src/a.py", "a\nb\n")
        self.write("docs/b.md", "b\nc\n")
        self.commit(T0 + DAY)

    def assert_src_scope(self, report) -> None:
        self.assertEqual(report.scope_source, "CLAUDE.md / AGENTS.md")
        self.assertEqual(report.scope_glob, "src/**")
        self.assertEqual(
            [(f.path, f.in_scope) for f in report.stale_files],
            [("src/a.py", True), ("docs/b.md", False)],
        )

    def test_the_scope_list_comes_from_the_top_level(self):
        self.write("CLAUDE.md", "# Rules\n\nScope:\n- src/**\n")
        self._scope_repo()
        elsewhere = os.path.join(self.root, "elsewhere")
        os.makedirs(elsewhere)
        Path(elsewhere, "CLAUDE.md").write_text("Scope:\n- docs/**\n")
        self.chdir(elsewhere)
        with self.subTest("--repo is the top level"):
            self.assert_src_scope(self.analyse(repo=self.repo))
        with self.subTest("--repo is a subdirectory"):
            self.assert_src_scope(self.analyse(repo=os.path.join(self.repo, "docs")))
        self.chdir(os.path.join(self.repo, "docs"))
        with self.subTest("run from a subdirectory"):
            self.assert_src_scope(self.analyse(repo="."))

    def test_agents_md_with_star_bullets(self):
        self.write("AGENTS.md", "Files to watch:\n* `src/**`\n")
        self._scope_repo()
        self.assert_src_scope(self.analyse())

    def test_claude_md_comes_first_and_scope_flag_wins(self):
        self.write("AGENTS.md", "Scope:\n- docs/**\n")
        self.write("CLAUDE.md", "Scope:\n- src/**\n")
        self._scope_repo()
        self.assert_src_scope(self.analyse())
        Path(self.repo, "CLAUDE.md").write_text("No scope list here.\n")
        self.assertEqual(self.analyse().scope_glob, "docs/**")
        report = self.analyse(scope_glob="src/a.py")
        self.assertEqual((report.scope_source, report.files_in_scope), ("--scope flag", 1))

    # -- errors --------------------------------------------------------------

    def test_not_a_repository(self):
        from agent_trace.freshness import FreshnessError
        plain = os.path.join(self.root, "plain")
        os.makedirs(plain)
        self.session(T0)
        with self.assertRaises(FreshnessError) as caught:
            self.analyse(repo=plain)
        self.assertTrue(
            str(caught.exception).startswith(f"git rev-parse failed in {plain}: "),
            caught.exception,
        )

    def test_missing_directory(self):
        from agent_trace.freshness import FreshnessError
        missing = os.path.join(self.root, "nope")
        self.session(T0)
        with self.assertRaises(FreshnessError) as caught:
            self.analyse(repo=missing)
        self.assertEqual(str(caught.exception), f"{missing} does not exist")

    def test_repository_without_commits(self):
        from agent_trace.freshness import FreshnessError
        self.session(T0)
        with self.assertRaises(FreshnessError) as caught:
            self.analyse()
        self.assertEqual(
            str(caught.exception), f"the git repository at {self.repo} has no commits"
        )

    def test_git_not_installed(self):
        from agent_trace.freshness import FreshnessError
        self.session(T0)
        empty = os.path.join(self.root, "bin")
        os.makedirs(empty)
        os.environ["PATH"] = empty
        with self.assertRaises(FreshnessError) as caught:
            self.analyse()
        self.assertEqual(
            str(caught.exception),
            f"git was not found on PATH, so freshness cannot read {self.repo}",
        )

    def test_git_timeout(self):
        from agent_trace.freshness import FreshnessError
        self.session(T0)
        timeout = subprocess.TimeoutExpired("git", 30)
        with patch("agent_trace.freshness.subprocess.run", side_effect=timeout), \
                self.assertRaises(FreshnessError) as caught:
            self.analyse()
        self.assertEqual(
            str(caught.exception), f"git rev-parse timed out after 30s in {self.repo}"
        )

    def test_a_failed_diff_is_an_error(self):
        from agent_trace.freshness import FreshnessError
        self.write("a.txt", "one\n")
        self.commit(T0)
        self.session(T0 + 60)
        self.write("a.txt", "two\n")
        self.commit(T0 + DAY)
        blob = self.git("rev-parse", "HEAD:a.txt")
        loose = Path(self.repo, ".git", "objects", blob[:2], blob[2:])
        loose.chmod(0o644)
        loose.unlink()
        with self.assertRaises(FreshnessError) as caught:
            self.analyse()
        self.assertTrue(
            str(caught.exception).startswith(f"git diff failed in {self.repo}: "),
            caught.exception,
        )

    def test_since_must_be_a_date(self):
        from agent_trace.freshness import FreshnessError
        for since in (
            "not-a-date", "2026-13-45", "2026-02-30", "2026-1-5",
            "2026-01-01T00:00", " 2026-01-01", "yesterday",
        ):
            with self.subTest(since=since):
                with patch("agent_trace.freshness.subprocess.run") as run, \
                        self.assertRaises(FreshnessError) as caught:
                    self.analyse(since_date=since)
                run.assert_not_called()
                self.assertEqual(
                    str(caught.exception),
                    f"--since must be a date in the form YYYY-MM-DD, not {since!r}",
                )

    def test_no_session_without_since(self):
        from agent_trace.freshness import NoSessionsError
        self.write("a.txt", "a\n")
        self.commit(T0)
        with self.assertRaises(NoSessionsError):
            self.analyse()

    def test_a_session_without_a_usable_start_time(self):
        from agent_trace.freshness import FreshnessError
        self.write("a.txt", "a\n")
        self.commit(T0)
        for n, started_at in enumerate(("NaN", '"2026-01-01"', "1e300")):
            with self.subTest(started_at=started_at):
                self.store = TraceStore(os.path.join(self.root, f"traces-{n}"))
                self.raw_session(started_at)
                with self.assertRaises(FreshnessError) as caught:
                    self.analyse()
                self.assertEqual(str(caught.exception), NO_TIME)

    def test_since_with_a_session_without_a_usable_start_time(self):
        self.write("a.txt", "a\n")
        self.commit(NOV15 - DAY)
        self.raw_session("NaN")
        report = self.analyse(since_date="2025-11-15")
        self.assertIsNone(report.last_session_ts)
        self.assertIn("\nLast session: s1 (start time unknown)\n", self.text(report))

    # -- the command line ----------------------------------------------------

    def test_cli_errors_are_one_line_and_exit_1(self):
        plain = os.path.join(self.root, "plain")
        os.makedirs(plain)
        self.write("a.txt", "a\n")
        self.commit(T0)
        with self.subTest("no session"):
            self.assertEqual(self.run_cli("freshness", "--repo", self.repo), (1, "", NO_SESSIONS))
        self.session(T0 + 60)
        with self.subTest("not a repository"):
            rc, out, err = self.run_cli("freshness", "--repo", plain)
            self.assertEqual((rc, out), (1, ""))
            self.assertRegex(
                err, rf"\AError: git rev-parse failed in {re.escape(plain)}: [^\n]+\n\Z"
            )
        with self.subTest("bad --since"):
            self.assertEqual(
                self.run_cli("freshness", "--repo", self.repo, "--since", "not-a-date"),
                (1, "", "Error: --since must be a date in the form YYYY-MM-DD, "
                        "not 'not-a-date'\n"),
            )
        with self.subTest("a report"):
            rc, out, err = self.run_cli("freshness", "--repo", self.repo)
            self.assertEqual((rc, err), (0, ""))
            self.assertIn("Context is fully fresh", out)

    def test_child_process_errors_have_no_traceback(self):
        plain = os.path.join(self.root, "plain")
        os.makedirs(plain)
        self.write("a.txt", "a\n")
        self.commit(T0)
        with self.subTest("no session"):
            result = self.run_child("--repo", self.repo)
            self.assertEqual((result.returncode, result.stdout, result.stderr),
                             (1, "", NO_SESSIONS))
        self.raw_session("NaN")
        with self.subTest("started_at is NaN"):
            result = self.run_child("--repo", self.repo)
            self.assertEqual((result.returncode, result.stdout, result.stderr),
                             (1, "", f"Error: {NO_TIME}\n"))
        with self.subTest("not a repository"):
            result = self.run_child("--repo", plain, "--since", "2025-11-15")
            self.assertEqual((result.returncode, result.stdout), (1, ""))
            self.assertRegex(
                result.stderr,
                rf"\AError: git rev-parse failed in {re.escape(plain)}: [^\n]+\n\Z",
            )


if __name__ == "__main__":
    unittest.main()
