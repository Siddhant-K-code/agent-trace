"""Trace-store files are written with LF line endings on every OS.

Text-mode writes translate "\\n" to os.linesep unless ``newline`` is set, so
Windows used to write CRLF into events.ndjson and other store files. Each
test runs the real writer twice: once as is, and once under
``windows_newlines()``, which makes every text-mode open that leaves
``newline`` unset translate "\\n" to "\\r\\n" the way Windows does. A writer
that forgets ``newline`` therefore fails on Linux and macOS too.
"""

import builtins
import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace import annotate, baseline, compaction, dashboard, hooks, identity
from agent_trace import postmortem, retention
from agent_trace.audit import verify_chain
from agent_trace.backfill import _write_meta_exclusive
from agent_trace.cli import cmd_verify
from agent_trace.compliance_report import _parse_event_snapshot
from agent_trace.eval import dataset
from agent_trace.fileutil import write_atomic
from agent_trace.integrations import _base as integrations_base
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.store import TraceStore, ensure_trace_dir_gitignore
from agent_trace.tenancy import delete_tenant_data
from agent_trace.validation import validate_trace_directory

_REAL_OPEN = io.open
MODES = ("native", "windows")


def _windows_open(file, mode="r", buffering=-1, encoding=None, errors=None,
                  newline=None, closefd=True, opener=None):
    if "b" not in mode and newline is None and any(c in mode for c in "wax+"):
        newline = "\r\n"
    return _REAL_OPEN(file, mode, buffering, encoding, errors, newline, closefd, opener)


@contextlib.contextmanager
def windows_newlines():
    """Translate "\\n" to "\\r\\n" in text-mode writes that leave newline unset."""
    with patch.object(builtins, "open", _windows_open), patch.object(io, "open", _windows_open):
        yield


class _LineEndingCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._env = patch.dict(os.environ, {"AGENT_TRACE_DIR": str(self.root / "traces")})
        self._env.start()
        for name in ("AGENT_STRACE_WORKSPACE", "AGENT_STRACE_TENANT_ID", "AGENT_STRACE_ENDPOINT"):
            os.environ.pop(name, None)

    def tearDown(self):
        self._env.stop()
        self._tmp.cleanup()

    @contextlib.contextmanager
    def in_mode(self, mode: str):
        """Run one subtest in a fresh trace directory, natively or as Windows."""
        trace_dir = self.root / mode
        newlines = windows_newlines if mode == "windows" else contextlib.nullcontext
        with self.subTest(mode=mode), patch.dict(os.environ, {"AGENT_TRACE_DIR": str(trace_dir)}):
            with newlines():
                yield trace_dir

    def assertLF(self, path: Path, *, expect_newline: bool = True) -> bytes:
        raw = Path(path).read_bytes()
        self.assertNotIn(b"\r", raw, f"{path.name} has CR bytes: {raw[:200]!r}")
        if expect_newline:
            self.assertIn(b"\n", raw, f"{path.name} has no line break to check")
        return raw


def _event(session_id: str, n: int) -> TraceEvent:
    return TraceEvent(
        event_type=EventType.TOOL_CALL,
        session_id=session_id,
        data={"tool_name": "Bash", "n": n},
    )


class TestSimulation(unittest.TestCase):
    def test_simulation_translates_unset_newline(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.txt"
            with windows_newlines():
                path.write_text("a\nb\n", encoding="utf-8")
            self.assertEqual(path.read_bytes(), b"a\r\nb\r\n")
            with windows_newlines():
                path.write_text("a\nb\n", encoding="utf-8", newline="\n")
            self.assertEqual(path.read_bytes(), b"a\nb\n")


class TestTraceStoreLineEndings(_LineEndingCase):
    def test_events_meta_and_gitignore_are_lf(self):
        for mode in MODES:
            with self.in_mode(mode) as trace_dir:
                store = TraceStore(trace_dir)
                meta = SessionMeta(agent_name="agent")
                store.create_session(meta)
                for n in range(3):
                    store.append_event(meta.session_id, _event(meta.session_id, n))

                session_dir = trace_dir / meta.session_id
                raw = self.assertLF(session_dir / "events.ndjson")
                self.assertTrue(raw.endswith(b"\n"))
                self.assertLF(session_dir / "meta.json")
                self.assertEqual(
                    (trace_dir / ".gitignore").read_bytes(),
                    b"# Created by agent-strace: traces may contain prompts and tool output.\n*\n",
                )

                lines = raw.split(b"\n")[:-1]
                self.assertEqual(len(lines), 3)
                for previous, current in zip(lines, lines[1:]):
                    self.assertEqual(
                        json.loads(current)["prev_hash"],
                        hashlib.sha256(previous).hexdigest(),
                    )
                self.assertTrue(verify_chain(store, meta.session_id).ok)

    def test_ensure_trace_dir_gitignore_is_lf(self):
        for mode in MODES:
            with self.in_mode(mode) as trace_dir:
                trace_dir.mkdir()
                ensure_trace_dir_gitignore(trace_dir)
                self.assertLF(trace_dir / ".gitignore")

    def test_tag_session_and_tenant_audits_are_lf(self):
        for mode in MODES:
            with self.in_mode(mode) as trace_dir:
                store = TraceStore(trace_dir)
                meta = SessionMeta()
                store.create_session(meta)
                for n in range(2):
                    store.append_event(meta.session_id, _event(meta.session_id, n))
                store.tag_session(meta.session_id, "acme")

                self.assertLF(trace_dir / meta.session_id / "events.ndjson")
                self.assertLF(trace_dir / meta.session_id / "meta.json")
                self.assertLF(trace_dir / "tenant-audit.ndjson")
                self.assertTrue(verify_chain(store, meta.session_id).ok)

                delete_tenant_data(store, "acme")
                raw = self.assertLF(trace_dir / "tenant-deletions.ndjson")
                self.assertEqual(len(raw.splitlines()), 2)

    def test_write_atomic_is_lf(self):
        for mode in MODES:
            with self.in_mode(mode) as trace_dir:
                target = trace_dir / "file.json"
                write_atomic(target, json.dumps({"a": 1}, indent=2) + "\n")
                self.assertLF(target)


class TestLegacyCrlfEvents(_LineEndingCase):
    """Event files written by older Windows versions end lines with CRLF.

    Their prev_hash values cover each line without its line ending, so they
    still verify, and new appends link to them the same way.
    """

    def _legacy_session(self, trace_dir: Path) -> tuple[TraceStore, str, bytes]:
        store = TraceStore(trace_dir)
        meta = SessionMeta()
        store.create_session(meta)
        lines = []
        previous = ""
        for n in range(2):
            event = _event(meta.session_id, n)
            event.prev_hash = hashlib.sha256(previous.encode()).hexdigest() if previous else ""
            previous = event.to_json()
            lines.append(previous)
        raw = "".join(line + "\r\n" for line in lines).encode()
        (trace_dir / meta.session_id / "events.ndjson").write_bytes(raw)
        return store, meta.session_id, raw

    def test_legacy_crlf_file_verifies(self):
        store, session_id, _ = self._legacy_session(self.root / "traces")
        self.assertTrue(verify_chain(store, session_id).ok)
        self.assertEqual(len(store.load_events(session_id)), 2)
        args = SimpleNamespace(
            trace_dir=str(store.base_dir), session_id=session_id,
            format="json", from_export="",
        )
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(cmd_verify(args), 0)
        self.assertTrue(json.loads(out.getvalue())["ok"])

    def test_append_to_legacy_crlf_file_links_to_line_without_cr(self):
        for mode in MODES:
            with self.in_mode(mode) as trace_dir:
                store, session_id, legacy = self._legacy_session(trace_dir)
                store.append_event(session_id, _event(session_id, 9))

                raw = (trace_dir / session_id / "events.ndjson").read_bytes()
                self.assertTrue(raw.startswith(legacy))
                appended = raw[len(legacy):]
                self.assertTrue(appended.endswith(b"\n"))
                self.assertNotIn(b"\r", appended)
                last_legacy_line = legacy.split(b"\r\n")[-2]
                self.assertEqual(
                    json.loads(appended)["prev_hash"],
                    hashlib.sha256(last_legacy_line).hexdigest(),
                )
                self.assertTrue(verify_chain(store, session_id).ok)
                meta = store.load_meta(session_id)
                _, state = _parse_event_snapshot(raw, meta)
                self.assertEqual(state, "unanchored")
                report = validate_trace_directory(trace_dir / session_id, allow_active=True)
                self.assertEqual(report.hash_chain["state"], "unanchored")

    def test_tag_session_rewrites_legacy_crlf_file_as_lf(self):
        for mode in MODES:
            with self.in_mode(mode) as trace_dir:
                store, session_id, _ = self._legacy_session(trace_dir)
                self.assertEqual(store.tag_session(session_id, "acme"), 2)
                self.assertLF(trace_dir / session_id / "events.ndjson")
                self.assertTrue(verify_chain(store, session_id).ok)

    def test_failed_tag_restores_legacy_bytes_exactly(self):
        store, session_id, legacy = self._legacy_session(self.root / "traces")
        events_path = store.base_dir / session_id / "events.ndjson"

        def fail_after_rewrite(_self, _journal, _record):
            events_path.write_bytes(b"partial\n")
            raise RuntimeError("boom")

        with patch.object(TraceStore, "_complete_tag_journal_locked", fail_after_rewrite):
            with self.assertRaises(RuntimeError):
                store.tag_session(session_id, "acme")
        self.assertEqual(events_path.read_bytes(), legacy)


class TestSidecarLineEndings(_LineEndingCase):
    def _session(self, trace_dir: Path) -> tuple[TraceStore, str]:
        store = TraceStore(trace_dir)
        meta = SessionMeta()
        store.create_session(meta)
        store.append_event(meta.session_id, _event(meta.session_id, 0))
        return store, meta.session_id

    def test_hook_state_files_are_lf(self):
        for mode in MODES:
            with self.in_mode(mode) as trace_dir:
                trace_dir.mkdir()
                hooks._write_active_session("abc123", provider="claude")
                hooks._write_pending_calls({"call": {"tool": "Bash"}}, provider="claude")
                for marker in trace_dir.glob(".active-session*"):
                    self.assertLF(marker, expect_newline=False)
                    self.assertEqual(marker.read_bytes(), b"abc123")
                pending = list(trace_dir.glob(".pending-calls*.json"))
                self.assertEqual(len(pending), 1)
                self.assertLF(pending[0], expect_newline=False)

    def test_integration_active_session_is_lf(self):
        for mode in MODES:
            with self.in_mode(mode) as trace_dir:
                store = TraceStore(trace_dir)
                session_id = integrations_base._get_or_create_session(store, "agent")
                marker = trace_dir / ".active-session.agent"
                self.assertEqual(marker.read_bytes(), session_id.encode())
                self.assertLF(trace_dir / session_id / "meta.json")

    def test_annotations_are_lf(self):
        for mode in MODES:
            with self.in_mode(mode) as trace_dir:
                store, session_id = self._session(trace_dir)
                first = annotate.Annotation(note="one")
                annotate.add_annotation(store, session_id, first)
                annotate.add_annotation(store, session_id, annotate.Annotation(note="two"))
                path = store.annotations_path(session_id)
                self.assertEqual(len(self.assertLF(path).splitlines()), 2)
                self.assertTrue(annotate.delete_annotation(store, session_id, first.annotation_id))
                self.assertEqual(len(self.assertLF(path).splitlines()), 1)

    def test_trend_annotations_are_lf(self):
        for mode in MODES:
            with self.in_mode(mode) as trace_dir:
                store = TraceStore(trace_dir)
                dashboard.save_annotation(store, dashboard.TrendAnnotation(date="2026-01-01", note="é"))
                dashboard.save_annotation(store, dashboard.TrendAnnotation(date="2026-01-02", note="b"))
                raw = self.assertLF(store.base_dir / "annotations.jsonl")
                self.assertEqual(len(raw.splitlines()), 2)
                self.assertEqual([a.note for a in dashboard.load_annotations(store)], ["é", "b"])

    def test_retention_log_is_lf(self):
        for mode in MODES:
            with self.in_mode(mode) as trace_dir:
                log = trace_dir / "retention.log"
                retention._log_deletion("s1", str(log))
                retention._log_deletion("s2", str(log))
                self.assertEqual(len(self.assertLF(log).splitlines()), 2)

    def test_dataset_is_lf(self):
        for mode in MODES:
            with self.in_mode(mode) as trace_dir:
                path = trace_dir / "datasets" / "default.jsonl"
                dataset.add_entry(path, dataset.DatasetEntry(session_id="s1"))
                dataset.add_entry(path, dataset.DatasetEntry(session_id="s2"))
                self.assertEqual(len(self.assertLF(path).splitlines()), 2)

    def test_identity_signature_is_lf(self):
        for mode in MODES:
            with self.in_mode(mode) as trace_dir:
                store, session_id = self._session(trace_dir)
                identity._save_sig(store, session_id, {"signature": "x", "agent_id": "y"})
                self.assertLF(trace_dir / session_id / "identity.json")

    def test_checkpoint_is_lf(self):
        for mode in MODES:
            with self.in_mode(mode) as trace_dir:
                store, session_id = self._session(trace_dir)
                path = compaction.write_checkpoint(store, session_id, timestamp=0.0)
                self.assertLF(path)

    def test_heartbeat_and_crash_postmortem_are_lf(self):
        for mode in MODES:
            with self.in_mode(mode) as trace_dir:
                store, session_id = self._session(trace_dir)
                postmortem.write_heartbeat(store, session_id, agent_pid=1)
                self.assertLF(postmortem.heartbeat_path(store, session_id), expect_newline=False)
                report = postmortem.PostmortemReport(
                    session_id=session_id, failed=True, status_summary="Failed",
                    root_cause="crash", root_cause_offset=0.0, timeline=[],
                    wasted_seconds=0.0, total_seconds=1.0, wasted_cost=0.0,
                    total_cost=0.0, recommendations=["retry"], agents_md_violations=[],
                )
                self.assertLF(postmortem.write_crash_postmortem(store, report))

    def test_baseline_is_lf(self):
        for mode in MODES:
            with self.in_mode(mode) as trace_dir:
                path = trace_dir / "baseline.json"
                baseline.save_baseline(baseline.BaselineProfile(session_count=1), str(path))
                self.assertLF(path)

    def test_backfill_meta_is_lf(self):
        for mode in MODES:
            with self.in_mode(mode) as trace_dir:
                store = TraceStore(trace_dir)
                meta = SessionMeta(agent_name="agent")
                (trace_dir / meta.session_id).mkdir(parents=True)
                _write_meta_exclusive(store, meta.session_id, meta)
                self.assertLF(trace_dir / meta.session_id / "meta.json")


if __name__ == "__main__":
    unittest.main()
