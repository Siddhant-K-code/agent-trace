"""Trace-store files and the files agent-strace reads back are LF on every OS.

Text-mode writes translate "\\n" to os.linesep unless ``newline`` is set, so
Windows used to write CRLF into events.ndjson and other store files. Each
test runs the real writer twice: once as is, and once under
``windows_newlines()``, which makes every text-mode open that leaves
``newline`` unset translate "\\n" to "\\r\\n" the way Windows does. A writer
that forgets ``newline`` therefore fails on Linux and macOS too.
"""

import argparse
import builtins
import contextlib
import hashlib
import io
import json
import os
import pathlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace import annotate, auto_bootstrap, baseline, budget_report, cli, compaction
from agent_trace import copilot_cloud, dashboard, drift, freeze, hooks, identity, optimize
from agent_trace import policy, policy_backtest, postmortem, retention, skills_eval, watch
from agent_trace.audit import verify_chain
from agent_trace.backfill import _write_meta_exclusive
from agent_trace.cli import cmd_verify
from agent_trace.compliance_report import _parse_event_snapshot
from agent_trace.eval import dataset
from agent_trace.eval import runner as eval_runner
from agent_trace.eval.config import EvalConfig
from agent_trace.eval.scorers import ScoreResult
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
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(builtins, "open", _windows_open))
        stack.enter_context(patch.object(io, "open", _windows_open))
        # Python 3.10 pathlib binds io.open on a module-level accessor.
        accessor = getattr(pathlib, "_normal_accessor", None)
        if accessor is not None:
            stack.enter_context(patch.object(accessor, "open", _windows_open))
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


def _crlf(path: Path) -> None:
    """Rewrite an LF file the way an older Windows version wrote it."""
    raw = path.read_bytes()
    assert b"\r" not in raw
    path.write_bytes(raw.replace(b"\n", b"\r\n"))


def _gate_report() -> "eval_runner.GateReport":
    return eval_runner.GateReport(session_id="gate", results=[
        eval_runner.CriterionResult(
            name="no-errors", scorer="error_count", score=0, fail_on="above",
            threshold=0, criterion_passed=True,
        ),
    ])


def _eval_report() -> "eval_runner.EvalReport":
    return eval_runner.EvalReport(
        session_id="eval",
        results=[ScoreResult(scorer="no_errors", score=1.0, threshold=1.0, passed=True, reason="")],
        config=EvalConfig.default(),
    )


class TestReadBackFileLineEndings(_LineEndingCase):
    """Configs, fixtures, baselines, policies and logs that agent-strace reads back.

    Each test also rewrites the file with CRLF, as older Windows versions
    did, and checks that it still loads.
    """

    def _store_with_session(self, trace_dir: Path) -> tuple[TraceStore, str]:
        store = TraceStore(trace_dir)
        meta = SessionMeta(agent_name="agent")
        store.create_session(meta)
        store.append_event(meta.session_id, _event(meta.session_id, 0))
        return store, meta.session_id

    def test_alerts_log_is_lf(self):
        for mode in MODES:
            with self.in_mode(mode) as trace_dir:
                log = trace_dir / "alerts.log"
                watch._alert_file("first", log_path=str(log))
                watch._alert_file("second", log_path=str(log))
                self.assertEqual(len(self.assertLF(log).splitlines()), 2)

    def test_watchdog_postmortem_is_lf(self):
        for mode in MODES:
            with self.in_mode(mode) as trace_dir:
                store, session_id = self._store_with_session(trace_dir)
                path = watch._write_watchdog_postmortem(
                    store, session_id, watch.WatchState(), "budget",
                )
                self.assertIsNotNone(path)
                self.assertLF(path)
                _crlf(path)
                loaded = budget_report._read_postmortem(store, session_id)
                self.assertEqual(loaded["session_id"], session_id)

    def test_setup_hook_configs_are_lf(self):
        args = argparse.Namespace(redact=False, no_redact=False)
        for mode in MODES:
            with self.in_mode(mode) as trace_dir:
                env = {
                    "CODEX_HOME": str(trace_dir / "codex"),
                    "GEMINI_CLI_HOME": str(trace_dir / "gemini"),
                    "CURSOR_CONFIG_DIR": str(trace_dir / "cursor"),
                    "COPILOT_HOME": str(trace_dir / "copilot"),
                }
                (trace_dir / "codex").mkdir(parents=True)
                with patch.dict(os.environ, env):
                    paths = [
                        cli._write_codex_hooks_config(args)[0],
                        *cli._write_gemini_extension(args),
                        cli._write_cursor_hooks_config(args),
                        cli._write_copilot_hooks_config(args),
                    ]
                for path in paths:
                    self.assertLF(path)
                    expected = json.loads(path.read_text(encoding="utf-8"))
                    _crlf(path)
                    self.assertEqual(cli._read_json_or_none(path), expected)

    def test_codex_setup_backs_up_legacy_crlf_config_exactly(self):
        args = argparse.Namespace(redact=False, no_redact=False)
        for mode in MODES:
            with self.in_mode(mode) as trace_dir:
                codex = trace_dir / "codex"
                codex.mkdir(parents=True)
                legacy = b'{\r\n  "hooks": {}\r\n}\r\n'
                (codex / "hooks.json").write_bytes(legacy)
                with patch.dict(os.environ, {"CODEX_HOME": str(codex)}):
                    hooks_path, backup, created = cli._write_codex_hooks_config(args)
                self.assertTrue(created)
                self.assertEqual(backup.read_bytes(), legacy)
                self.assertLF(hooks_path)

    def test_copilot_cloud_repository_files_are_lf(self):
        for mode in MODES:
            with self.in_mode(mode) as trace_dir:
                repo = trace_dir / "repo"
                hooks_path, script_path = copilot_cloud.write_repository_files(repo)
                self.assertLF(hooks_path)
                script = self.assertLF(script_path)
                self.assertTrue(script.startswith(b"#!"))

    def test_policy_is_lf(self):
        for mode in MODES:
            with self.in_mode(mode) as trace_dir:
                self._store_with_session(trace_dir)
                out = trace_dir / "policy.json"
                args = argparse.Namespace(
                    trace_dir=str(trace_dir), session_ids=[], output=str(out), dry_run=False,
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(policy.cmd_policy(args), 0)
                self.assertLF(out)
                _crlf(out)
                policy_backtest._load_policy(out)

    def test_drift_fingerprints_are_lf(self):
        for mode in MODES:
            with self.in_mode(mode) as trace_dir:
                self._store_with_session(trace_dir)
                saved = trace_dir / "fp" / "fingerprint.json"
                args = argparse.Namespace(
                    trace_dir=str(trace_dir), sessions=20, output=str(saved), id="fp",
                    compare=None, threshold=0.2, format="text",
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(drift.cmd_fingerprint(args), 0)
                self.assertLF(saved)

                baseline_path = trace_dir / "baseline" / "drift.json"
                args = argparse.Namespace(
                    trace_dir=str(trace_dir), threshold=0.2, baseline=None, since="30d",
                    current=None, baseline_range=None, save_baseline=str(baseline_path),
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(drift.cmd_drift(args), 0)
                self.assertLF(baseline_path)

                _crlf(saved)
                loaded = drift.BehavioralFingerprint.from_json(saved.read_text())
                self.assertEqual(loaded.fingerprint_id, "fp")

    def test_freeze_fixture_is_lf(self):
        for mode in MODES:
            with self.in_mode(mode) as trace_dir:
                _, session_id = self._store_with_session(trace_dir)
                out = trace_dir / "fixtures" / "fixture.json"
                args = argparse.Namespace(
                    trace_dir=str(trace_dir), session_id=session_id, output=str(out),
                    task="", format="text",
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(freeze.cmd_freeze(args), 0)
                self.assertLF(out)
                _crlf(out)
                args = argparse.Namespace(
                    trace_dir=str(trace_dir), fixture_file=str(out), session_id=session_id,
                    threshold=0.0, format="json",
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(freeze.cmd_regression(args), 0)

    def test_eval_baselines_and_summaries_are_lf(self):
        for mode in MODES:
            with self.in_mode(mode) as trace_dir:
                gate = trace_dir / "gate-baseline.json"
                eval_runner.save_gate_baseline(gate, _gate_report())
                self.assertLF(gate)
                _crlf(gate)
                self.assertEqual(eval_runner.load_gate_baseline(gate), {"no-errors": 0})

                legacy = trace_dir / "eval-baseline.json"
                eval_runner._save_baseline(str(legacy), _eval_report())
                self.assertLF(legacy)
                _crlf(legacy)
                self.assertEqual(eval_runner._load_baseline(str(legacy)), {"no_errors": 1.0})

                summary = trace_dir / "step-summary.md"
                eval_runner.write_gate_github_summary(_gate_report(), summary)
                with patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": str(summary)}):
                    with contextlib.redirect_stderr(io.StringIO()):
                        eval_runner._write_github_summary(_eval_report(), {}, tolerance=0.0)
                self.assertIn(b"agent-strace eval", self.assertLF(summary))

    def test_skills_eval_hook_configs_are_lf(self):
        for mode in MODES:
            with self.in_mode(mode) as trace_dir:
                paths = skills_eval.write_hook_configs(trace_dir)
                plugin_dir = Path(paths["hooks_plugin_dir"])
                self.assertLF(plugin_dir / "plugin.json")
                self.assertLF(plugin_dir / "hooks.json")
                self.assertLF(Path(paths["claude_settings"]))

    def test_skills_eval_run_info_is_lf(self):
        case = skills_eval.EvalCase(id="case", skill="demo", prompt="hi", should_trigger=True)
        spec = skills_eval.AgentSpec(name="agent", template="", tokens=["agent"], timeout=1.0)
        outcome = {"exit_code": 3, "timed_out": False, "duration_s": 0.5, "error": "boom"}
        for mode in MODES:
            with self.in_mode(mode) as trace_dir:
                with patch.object(skills_eval, "run_agent", return_value=dict(outcome)), \
                        patch.object(skills_eval, "score_trace_dir", return_value={}), \
                        patch.object(skills_eval, "classify_run", side_effect=lambda result: result):
                    skills_eval.run_case_live(case, 1, spec, save_root=trace_dir / "saved")
                run_dir = trace_dir / "saved" / "case" / "run-1"
                self.assertLF(run_dir / "run.json")
                _crlf(run_dir / "run.json")
                warnings: list[str] = []
                info = skills_eval._read_run_info(run_dir, warnings)
                self.assertEqual((info["exit_code"], warnings), (3, []))

    def test_sitecustomize_is_lf(self):
        for mode in MODES:
            with self.in_mode(mode):
                bootstrap_dir = auto_bootstrap.create_bootstrap_dir()
                self.addCleanup(auto_bootstrap.remove_bootstrap_dir, bootstrap_dir)
                self.assertLF(Path(bootstrap_dir) / "sitecustomize.py")

    def test_machine_identity_is_lf(self):
        for mode in MODES:
            with self.in_mode(mode) as trace_dir:
                path = trace_dir / "identity.json"
                with patch.dict(os.environ, {"AGENT_TRACE_IDENTITY_FILE": str(path)}):
                    created = identity.load_or_create_identity("agent")
                    self.assertLF(path)
                    _crlf(path)
                    self.assertEqual(identity.load_identity().identity_id, created.identity_id)

    def test_optimize_keeps_the_target_file_line_endings(self):
        cases = (
            (None, b"# Added\n"),
            (b"# Rules\n", b"# Rules\n\n---\n\n# Added\n"),
            (b"# Rules\r\n", b"# Rules\r\n\r\n---\r\n\r\n# Added\r\n"),
        )
        pattern = optimize.FailurePattern(
            name="p", description="d", session_ids=["s"], example_events=[],
            proposed_instruction="# Added",
        )

        def propose(_store, _ids, *, target_file, **_kwargs):
            path = Path(target_file)
            existing = path.read_text() if path.exists() else ""
            return optimize.OptimizeProposal(target_file, [pattern], existing)

        for existing, expected in cases:
            for mode in MODES:
                with self.in_mode(mode) as trace_dir, self.subTest(existing=existing):
                    self._store_with_session(trace_dir)
                    target = trace_dir / "AGENTS.md"
                    if existing is not None:
                        target.write_bytes(existing)
                    args = argparse.Namespace(
                        trace_dir=str(trace_dir), target=str(target), apply=True, dry_run=False,
                        session_id=None, dataset=None,
                        base_url=None, model=None, api_key=None,
                    )
                    with patch.object(optimize, "run_optimize", propose), \
                            contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(optimize.cmd_optimize(args), 0)
                    self.assertEqual(target.read_bytes(), expected)

if __name__ == "__main__":
    unittest.main()
