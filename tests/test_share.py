"""Tests for HTML share generation and postmortem analysis."""

import contextlib
import io
import json
import os
import re
import socket
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from agent_trace.annotate import Annotation, add_annotation
from agent_trace.cli import build_parser
from agent_trace.disclosure import build_disclosure_preview
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore
from agent_trace.share import cmd_share, render_html, _event_summary, _esc
from agent_trace.postmortem import (
    analyze_session,
    format_postmortem,
    render_postmortem_html,
    _load_agents_md,
    _detect_agents_md_violations,
    _find_root_cause,
    _build_timeline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(event_type: EventType, ts: float, session_id: str = "s1", **data) -> TraceEvent:
    return TraceEvent(event_type=event_type, timestamp=ts, session_id=session_id, data=data)


def _make_store(events: list[TraceEvent], session_id: str = "s1") -> tuple[TraceStore, tempfile.TemporaryDirectory]:
    tmp = tempfile.TemporaryDirectory()
    store = TraceStore(tmp.name)
    meta = SessionMeta(
        session_id=session_id,
        started_at=events[0].timestamp if events else 0.0,
        total_duration_ms=(events[-1].timestamp - events[0].timestamp) * 1000 if len(events) > 1 else 0,
    )
    store.create_session(meta)
    for e in events:
        store.append_event(session_id, e)
    store.update_meta(meta)
    return store, tmp


def _basic_events(session_id: str = "s1") -> list[TraceEvent]:
    return [
        _make_event(EventType.SESSION_START, 0.0, session_id),
        _make_event(EventType.USER_PROMPT, 1.0, session_id, prompt="run the tests"),
        _make_event(EventType.TOOL_CALL, 2.0, session_id,
                    tool_name="Bash", arguments={"command": "pytest"}),
        _make_event(EventType.TOOL_RESULT, 3.0, session_id, result="5 passed"),
        _make_event(EventType.ASSISTANT_RESPONSE, 4.0, session_id, text="All tests passed."),
        _make_event(EventType.SESSION_END, 5.0, session_id, exit_code=0),
    ]


def _failed_events(session_id: str = "s1") -> list[TraceEvent]:
    return [
        _make_event(EventType.SESSION_START, 0.0, session_id),
        _make_event(EventType.USER_PROMPT, 1.0, session_id, prompt="install deps"),
        _make_event(EventType.TOOL_CALL, 2.0, session_id,
                    tool_name="Bash", arguments={"command": "pip install psycopg2"}),
        _make_event(EventType.ERROR, 3.0, session_id, message="missing libpq-dev"),
        _make_event(EventType.TOOL_CALL, 4.0, session_id,
                    tool_name="Bash", arguments={"command": "pip install psycopg2"}),
        _make_event(EventType.ERROR, 5.0, session_id, message="missing libpq-dev"),
        _make_event(EventType.SESSION_END, 6.0, session_id, exit_code=1),
    ]


# ---------------------------------------------------------------------------
# share.py tests
# ---------------------------------------------------------------------------

class TestEscaping(unittest.TestCase):
    def test_esc_html_entities(self):
        self.assertEqual(_esc("<script>"), "&lt;script&gt;")
        self.assertEqual(_esc('"hello"'), "&quot;hello&quot;")
        self.assertEqual(_esc("a & b"), "a &amp; b")


class TestEventSummary(unittest.TestCase):
    def test_tool_call_bash(self):
        e = _make_event(EventType.TOOL_CALL, 0.0,
                        tool_name="Bash", arguments={"command": "ls -la"})
        summary = _event_summary(e)
        self.assertIn("ls -la", summary)

    def test_tool_call_read(self):
        e = _make_event(EventType.TOOL_CALL, 0.0,
                        tool_name="Read", arguments={"file_path": "src/main.py"})
        summary = _event_summary(e)
        self.assertIn("src/main.py", summary)

    def test_user_prompt(self):
        e = _make_event(EventType.USER_PROMPT, 0.0, prompt="fix the bug")
        self.assertIn("fix the bug", _event_summary(e))

    def test_error_event(self):
        e = _make_event(EventType.ERROR, 0.0, message="something went wrong")
        self.assertIn("something went wrong", _event_summary(e))

    def test_session_end(self):
        e = _make_event(EventType.SESSION_END, 0.0, exit_code=0)
        self.assertIn("exit=0", _event_summary(e))


class TestRenderHtml(unittest.TestCase):
    def setUp(self):
        events = _basic_events("html1")
        self.store, self.tmp = _make_store(events, "html1")

    def tearDown(self):
        self.tmp.cleanup()

    def test_returns_string(self):
        html = render_html(self.store, "html1")
        self.assertIsInstance(html, str)

    def test_valid_html_structure(self):
        html = render_html(self.store, "html1")
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("<html", html)
        self.assertIn("</html>", html)
        self.assertIn("<head>", html)
        self.assertIn("</head>", html)
        self.assertIn("<body>", html)
        self.assertIn("</body>", html)

    def test_no_external_urls(self):
        html = render_html(self.store, "html1")
        # Should not reference any CDN or external resource
        self.assertNotIn("cdn.jsdelivr.net", html)
        self.assertNotIn("unpkg.com", html)
        self.assertNotIn("googleapis.com", html)
        self.assertNotIn('src="http', html)
        self.assertNotIn('href="http', html)

    def test_session_id_in_output(self):
        html = render_html(self.store, "html1")
        self.assertIn("html1", html)

    def test_css_inlined(self):
        html = render_html(self.store, "html1")
        self.assertIn("<style>", html)

    def test_js_inlined(self):
        html = render_html(self.store, "html1")
        self.assertIn("<script>", html)

    def test_phase_section_present(self):
        html = render_html(self.store, "html1")
        self.assertIn("Phase", html)

    def test_failed_session_shows_failed(self):
        events = _failed_events("html2")
        store2, tmp2 = _make_store(events, "html2")
        try:
            html = render_html(store2, "html2")
            self.assertIn("FAILED", html)
        finally:
            tmp2.cleanup()

    def test_postmortem_html_included_when_provided(self):
        html = render_html(self.store, "html1", postmortem_html="<div>POSTMORTEM</div>")
        self.assertIn("POSTMORTEM", html)


_SENTINELS = (
    "sentinel-prompt-41f7",
    "sentinel-command-8c2e",
    "/sentinel/dir-5d90/file.py",
    "sentinel-result-b3a1",
    "sentinel-response-06cd",
    "sentinel-label-7e44",
    "sentinel-note-2f58",
)


def _sentinel_events(session_id: str) -> list[TraceEvent]:
    return [
        _make_event(EventType.SESSION_START, 0.0, session_id),
        _make_event(EventType.USER_PROMPT, 1.0, session_id, prompt="sentinel-prompt-41f7"),
        _make_event(EventType.TOOL_CALL, 2.0, session_id,
                    tool_name="Bash", arguments={"command": "sentinel-command-8c2e"}),
        _make_event(EventType.TOOL_RESULT, 3.0, session_id, result="sentinel-result-b3a1"),
        _make_event(EventType.FILE_READ, 4.0, session_id, path="/sentinel/dir-5d90/file.py"),
        _make_event(EventType.ASSISTANT_RESPONSE, 5.0, session_id, text="sentinel-response-06cd"),
        _make_event(EventType.SESSION_END, 6.0, session_id, exit_code=0),
    ]


def _tree_snapshot(root: Path) -> dict[str, tuple[int, bytes] | None]:
    """Map every path under *root* to its file mtime and bytes (None for directories)."""
    return {
        str(path.relative_to(root)): (
            (path.stat().st_mtime_ns, path.read_bytes()) if path.is_file() else None
        )
        for path in root.rglob("*")
    }


@contextlib.contextmanager
def _deny_network():
    def fail(*args, **kwargs):
        raise AssertionError("share --dry-run must not use the network")

    with mock.patch.object(socket.socket, "connect", fail), \
            mock.patch.object(socket.socket, "connect_ex", fail), \
            mock.patch("socket.create_connection", fail), \
            mock.patch("urllib.request.urlopen", fail):
        yield


class TestShareDryRun(unittest.TestCase):
    session_id = "dryrun0123456789"

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.trace_dir = str(self.root / "traces")
        self.workdir = self.root / "work"
        self.workdir.mkdir()

        self.store = TraceStore(self.trace_dir)
        meta = SessionMeta(session_id=self.session_id, started_at=0.0, total_duration_ms=6000)
        self.store.create_session(meta)
        for event in _sentinel_events(self.session_id):
            self.store.append_event(self.session_id, event)
        self.store.update_meta(meta)
        add_annotation(self.store, self.session_id,
                       Annotation(label="sentinel-label-7e44", note="sentinel-note-2f58"))
        add_annotation(self.store, self.session_id, Annotation(label="decision"))

        # Default share output lands in the working directory, which is inside the snapshot root.
        old_cwd = os.getcwd()
        os.chdir(self.workdir)
        self.addCleanup(os.chdir, old_cwd)

    def _run(self, *argv: str) -> tuple[int, str, str]:
        args = build_parser().parse_args(["--trace-dir", self.trace_dir, "share", *argv])
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cmd_share(args)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_parser_flags(self):
        args = build_parser().parse_args(["share"])
        self.assertFalse(args.dry_run)
        self.assertIsNone(args.format)
        args = build_parser().parse_args(["share", "abc", "--dry-run", "--format", "json"])
        self.assertTrue(args.dry_run)
        self.assertEqual(args.format, "json")

    def test_text_preview_reports_include_content_counts(self):
        code, out, err = self._run(self.session_id[:8], "--dry-run")
        self.assertEqual(code, 0)
        self.assertEqual(err, "")
        self.assertIn("no HTML written or opened", out)
        self.assertIn(f"Session: {self.session_id}\n", out)
        self.assertIn("Mode: include_content", out)
        expected = {
            "events": 7,
            "prompts": 1,
            "responses": 1,
            "tool inputs": 1,
            "tool results": 1,
            "command-bearing events": 1,
            "path-bearing events": 1,
            "annotations": 2,
            "content-rich events": 5,
            "redacted events": 0,
            "unredacted content-rich events": 5,
        }
        for label, count in expected.items():
            self.assertRegex(out, rf"(?m)^  {re.escape(label)}:\s+{count}$")

    def test_json_preview_matches_disclosure_core(self):
        code, out, err = self._run(self.session_id, "--dry-run", "--format", "json")
        self.assertEqual(code, 0)
        self.assertEqual(err, "")
        expected = build_disclosure_preview(
            self.store.load_events(self.session_id), annotation_count=2, include_content=True
        ).to_dict()
        payload = json.loads(out)
        self.assertEqual(payload, {"session_id": self.session_id, **expected})
        self.assertEqual(payload["mode"], "include_content")

    def test_latest_session_is_previewed_by_default(self):
        code, out, _ = self._run("--dry-run", "--format", "json")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["session_id"], self.session_id)

    def test_preview_never_prints_values(self):
        html = render_html(self.store, self.session_id)
        for value in _SENTINELS:
            self.assertIn(value, html)
        for argv in (("--dry-run",), ("--dry-run", "--format", "json")):
            code, out, err = self._run(self.session_id, *argv)
            self.assertEqual(code, 0)
            for value in _SENTINELS:
                self.assertNotIn(value, out)
                self.assertNotIn(value, err)

    def test_dry_run_writes_opens_and_sends_nothing(self):
        target = self.root / "out.html"
        before = _tree_snapshot(self.root)
        with mock.patch("agent_trace.share.subprocess.run") as run, \
                mock.patch("agent_trace.share.os.startfile", create=True) as startfile, \
                mock.patch("agent_trace.share.render_html") as render, \
                _deny_network():
            code, out, err = self._run(
                self.session_id, "--dry-run", "-o", str(target), "--stdout", "--open"
            )
        self.assertEqual(code, 0)
        run.assert_not_called()
        startfile.assert_not_called()
        render.assert_not_called()
        self.assertFalse(target.exists())
        self.assertEqual(_tree_snapshot(self.root), before)
        self.assertNotIn("<html", out)
        self.assertIn("Mode: include_content", out)
        self.assertIn("ignores --output, --stdout, --open; no HTML was written or opened", err)

    def test_dry_run_without_output_flags_writes_nothing(self):
        before = _tree_snapshot(self.root)
        code, _, err = self._run(self.session_id, "--dry-run")
        self.assertEqual(code, 0)
        self.assertEqual(err, "")
        self.assertEqual(_tree_snapshot(self.root), before)

    def test_assignment_with_dry_run_is_rejected_without_writing(self):
        before = _tree_snapshot(self.root)
        code, out, err = self._run(self.session_id, "--assignment", "--dry-run", "-o", "bundle.zip")
        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        self.assertIn("cannot be combined with --assignment", err)
        self.assertFalse((self.workdir / "bundle.zip").exists())
        self.assertEqual(_tree_snapshot(self.root), before)

    def test_format_requires_dry_run(self):
        before = _tree_snapshot(self.root)
        for extra in ((), ("--assignment",)):
            code, out, err = self._run(self.session_id, "--format", "json", *extra)
            self.assertEqual(code, 1)
            self.assertEqual(out, "")
            self.assertIn("--format requires --dry-run", err)
        self.assertEqual(_tree_snapshot(self.root), before)

    def test_postmortem_does_not_change_counts(self):
        _, plain, _ = self._run(self.session_id, "--dry-run", "--format", "json")
        code, with_postmortem, err = self._run(
            self.session_id, "--dry-run", "--postmortem", "--format", "json"
        )
        self.assertEqual(code, 0)
        self.assertEqual(err, "")
        self.assertEqual(json.loads(with_postmortem), json.loads(plain))

    def test_unknown_session_is_an_error(self):
        code, out, err = self._run("nosuchsession", "--dry-run")
        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        self.assertIn("Session not found: nosuchsession", err)

    def test_default_share_still_writes_html(self):
        code, out, err = self._run(self.session_id)
        self.assertEqual(code, 0)
        self.assertEqual(out, "")
        self.assertIn("Created:", err)
        html = (self.workdir / f"session-{self.session_id[:12]}.html").read_text(encoding="utf-8")
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("sentinel-prompt-41f7", html)


# ---------------------------------------------------------------------------
# postmortem.py tests
# ---------------------------------------------------------------------------

class TestFindRootCause(unittest.TestCase):
    def test_finds_first_error(self):
        events = _failed_events()
        base_ts = events[0].timestamp
        idx, desc, offset = _find_root_cause(events, base_ts)
        self.assertGreater(idx, 0)
        self.assertIn("missing libpq-dev", desc)
        self.assertAlmostEqual(offset, 3.0, places=1)

    def test_no_error_returns_minus_one(self):
        events = _basic_events()
        base_ts = events[0].timestamp
        idx, desc, offset = _find_root_cause(events, base_ts)
        self.assertEqual(idx, -1)


class TestBuildTimeline(unittest.TestCase):
    def test_timeline_has_entries(self):
        events = _basic_events()
        base_ts = events[0].timestamp
        timeline = _build_timeline(events, base_ts, root_cause_idx=-1)
        self.assertGreater(len(timeline), 0)

    def test_root_cause_flagged(self):
        events = _failed_events()
        base_ts = events[0].timestamp
        idx, _, _ = _find_root_cause(events, base_ts)
        timeline = _build_timeline(events, base_ts, root_cause_idx=idx)
        root_entries = [e for e in timeline if e.is_root_cause]
        self.assertEqual(len(root_entries), 1)

    def test_retry_flagged(self):
        events = _failed_events()
        base_ts = events[0].timestamp
        timeline = _build_timeline(events, base_ts, root_cause_idx=-1)
        retries = [e for e in timeline if e.is_retry]
        self.assertGreater(len(retries), 0)


class TestAnalyzeSession(unittest.TestCase):
    def setUp(self):
        events = _failed_events("pm1")
        self.store, self.tmp = _make_store(events, "pm1")

    def tearDown(self):
        self.tmp.cleanup()

    def test_returns_report(self):
        from agent_trace.postmortem import PostmortemReport
        report = analyze_session(self.store, "pm1")
        self.assertIsInstance(report, PostmortemReport)

    def test_failed_session_detected(self):
        report = analyze_session(self.store, "pm1")
        self.assertTrue(report.failed)

    def test_root_cause_populated(self):
        report = analyze_session(self.store, "pm1")
        self.assertIn("libpq-dev", report.root_cause)

    def test_recommendations_non_empty(self):
        report = analyze_session(self.store, "pm1")
        self.assertGreater(len(report.recommendations), 0)

    def test_wasted_seconds_positive(self):
        report = analyze_session(self.store, "pm1")
        self.assertGreater(report.wasted_seconds, 0)


class TestFormatPostmortem(unittest.TestCase):
    def setUp(self):
        events = _failed_events("pm2")
        self.store, self.tmp = _make_store(events, "pm2")

    def tearDown(self):
        self.tmp.cleanup()

    def test_output_contains_session_id(self):
        report = analyze_session(self.store, "pm2")
        buf = io.StringIO()
        format_postmortem(report, out=buf)
        self.assertIn("pm2", buf.getvalue())

    def test_output_contains_root_cause(self):
        report = analyze_session(self.store, "pm2")
        buf = io.StringIO()
        format_postmortem(report, out=buf)
        self.assertIn("Root cause", buf.getvalue())

    def test_output_contains_recommendations(self):
        report = analyze_session(self.store, "pm2")
        buf = io.StringIO()
        format_postmortem(report, out=buf)
        self.assertIn("Recommendations", buf.getvalue())


class TestRenderPostmortemHtml(unittest.TestCase):
    def setUp(self):
        events = _failed_events("pm3")
        self.store, self.tmp = _make_store(events, "pm3")

    def tearDown(self):
        self.tmp.cleanup()

    def test_returns_html_for_failed_session(self):
        report = analyze_session(self.store, "pm3")
        html = render_postmortem_html(report)
        self.assertIn("<div", html)
        self.assertIn("Postmortem", html)

    def test_returns_empty_for_ok_session(self):
        events = _basic_events("pm4")
        store2, tmp2 = _make_store(events, "pm4")
        try:
            report = analyze_session(store2, "pm4")
            html = render_postmortem_html(report)
            self.assertEqual(html, "")
        finally:
            tmp2.cleanup()


class TestAgentsMdViolations(unittest.TestCase):
    def test_detects_pip_when_uv_required(self):
        events = [
            _make_event(EventType.TOOL_CALL, 0.0,
                        tool_name="Bash", arguments={"command": "pip install requests"}),
        ]
        agents_md_lines = ["use uv, not pip for package installation"]
        violations = _detect_agents_md_violations(events, agents_md_lines)
        self.assertGreater(len(violations), 0)
        self.assertTrue(any("pip" in v for v in violations))

    def test_no_violation_when_correct_tool_used(self):
        events = [
            _make_event(EventType.TOOL_CALL, 0.0,
                        tool_name="Bash", arguments={"command": "uv install requests"}),
        ]
        agents_md_lines = ["use uv, not pip"]
        violations = _detect_agents_md_violations(events, agents_md_lines)
        self.assertEqual(len(violations), 0)

    def test_empty_agents_md_no_violations(self):
        events = [
            _make_event(EventType.TOOL_CALL, 0.0,
                        tool_name="Bash", arguments={"command": "pip install x"}),
        ]
        violations = _detect_agents_md_violations(events, [])
        self.assertEqual(len(violations), 0)


if __name__ == "__main__":
    unittest.main()
