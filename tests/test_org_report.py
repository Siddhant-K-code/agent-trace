"""Tests for organization-wide local and remote-collector reporting."""

from __future__ import annotations

import io
import json
import os
import socket
import tempfile
import threading
import unittest
import urllib.error
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import urlsplit

from agent_trace.cli import build_parser
from agent_trace.collector_client import (
    CollectorAuthenticationError,
    CollectorClient,
    CollectorClientError,
    CollectorTraceStore,
    validate_collector_endpoint,
)
from agent_trace.lint import LintReport, LintResult
from agent_trace.models import EventType, SessionMeta, TraceEvent
from agent_trace.org_report import (
    MIN_ANOMALY_COVERAGE,
    MIN_ANOMALY_PEERS,
    MIN_ANOMALY_SESSIONS,
    TeamBreakdown,
    _SessionMetric,
    _build_anomalies,
    _write_atomic,
    build_org_report,
    format_org_report_html,
    format_org_report_json,
    format_org_report_text,
    month_bounds_utc,
)
from agent_trace.store import TraceStore
from agent_trace.team_report import _fallback_author, session_engineer_attribution
from agent_trace.tenancy import enumerate_trace_stores


def _timestamp(year: int, month: int, day: int = 1) -> float:
    return datetime(year, month, day, tzinfo=timezone.utc).timestamp()


def _event(session_id: str, timestamp: float, **kwargs) -> TraceEvent:
    return TraceEvent(
        event_type=kwargs.pop("event_type", EventType.USER_PROMPT),
        event_id=kwargs.pop("event_id", f"event-{session_id}"),
        session_id=session_id,
        timestamp=timestamp,
        data=kwargs.pop("data", {"prompt": "x" * 200}),
        **kwargs,
    )


def _add_session(
    store: TraceStore,
    session_id: str,
    timestamp: float,
    *,
    team: str = "",
    tenant: str = "",
    attribution: dict | None = None,
    agent_name: str = "code review",
) -> SessionMeta:
    meta = SessionMeta(
        session_id=session_id,
        started_at=timestamp,
        team=team,
        tenant_id=tenant,
        attribution=attribution or {},
        agent_name=agent_name,
    )
    store.create_session(meta)
    store.append_event(
        session_id,
        _event(session_id, timestamp + 1, tenant_id=tenant),
    )
    store.append_event(
        session_id,
        _event(
            session_id,
            timestamp + 2,
            event_type=EventType.ASSISTANT_RESPONSE,
            event_id=f"response-{session_id}",
            tenant_id=tenant,
            data={"text": "y" * 200},
        ),
    )
    return meta


class _MemoryStore:
    def __init__(self, metas, events=None, workspace_id=""):
        self.metas = list(metas)
        self.events = events or {meta.session_id: [] for meta in metas}
        self.workspace_id = workspace_id
        self.loads: list[str] = []

    def list_sessions_strict(self, tenant_id=None, *, validate_events=True):
        del validate_events
        if tenant_id is None:
            return list(self.metas)
        return [meta for meta in self.metas if meta.tenant_id == tenant_id]

    def load_meta(self, session_id):
        return next(meta for meta in self.metas if meta.session_id == session_id)

    def load_events(self, session_id):
        self.loads.append(session_id)
        return list(self.events[session_id])


class _Response(io.BytesIO):
    def __init__(self, body: bytes, *, status: int = 200, length=True):
        super().__init__(body)
        self.status = status
        self.headers = {}
        if length:
            self.headers["Content-Length"] = str(len(body))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class _RouteOpener:
    def __init__(self, sessions, events, *, enforce_auth=True):
        self.sessions = sessions
        self.events = events
        self.enforce_auth = enforce_auth
        self.requests: list[tuple[str, bool]] = []

    def open(self, request, timeout=0):
        del timeout
        path = urlsplit(request.full_url).path
        authenticated = bool(request.get_header("Authorization"))
        self.requests.append((path, authenticated))
        if path == "/sessions" and not authenticated and self.enforce_auth:
            raise urllib.error.HTTPError(
                request.full_url, 401, "unauthorized", {}, io.BytesIO(b"{}")
            )
        if path == "/sessions":
            return _Response(json.dumps(self.sessions).encode())
        encoded = path.removeprefix("/sessions/").removesuffix("/events")
        body = self.events[encoded]
        return _Response(body)


class TestMonthAndLocalReport(unittest.TestCase):
    def test_utc_month_bounds_are_half_open_and_handle_december(self):
        start, end = month_bounds_utc("2026-12")
        self.assertEqual(start, _timestamp(2026, 12))
        self.assertEqual(end, _timestamp(2027, 1))
        with self.assertRaisesRegex(ValueError, "YYYY-MM"):
            month_bounds_utc("2026-2")
        with self.assertRaisesRegex(ValueError, "YYYY-MM"):
            month_bounds_utc("2026-13")

    def test_flat_and_workspace_sessions_with_same_id_remain_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            flat = TraceStore(directory, use_workspace_env=False, redact=False)
            workspace = TraceStore(
                directory, workspace_id="payments", use_workspace_env=False,
                redact=False,
            )
            when = _timestamp(2026, 8, 3)
            _add_session(flat, "same-id", when, team="platform")
            _add_session(workspace, "same-id", when, team="payments")

            report = build_org_report(
                enumerate_trace_stores(directory), "2026-08",
                generated_at="2026-09-01T00:00:00Z",
            )

            self.assertEqual(report.session_count, 2)
            self.assertEqual({row.team for row in report.teams}, {"platform", "payments"})

    def test_month_and_team_filters_run_before_event_loads(self):
        inside = SessionMeta(
            session_id="inside", started_at=_timestamp(2026, 8, 2), team="blue"
        )
        other_team = SessionMeta(
            session_id="other", started_at=_timestamp(2026, 8, 3), team="red"
        )
        other_month = SessionMeta(
            session_id="old", started_at=_timestamp(2026, 7, 31), team="blue"
        )
        store = _MemoryStore([inside, other_team, other_month])
        with patch("agent_trace.org_report.estimate_cost", return_value=SimpleNamespace(total_cost=1.0)), patch(
            "agent_trace.org_report.lint_session", return_value=LintReport("inside")
        ):
            report = build_org_report([store], "2026-08", team="tag:blue")
        self.assertEqual(report.session_count, 1)
        self.assertEqual(store.loads, ["inside"])

    def test_ambiguous_unprefixed_selector_fails_before_event_loads(self):
        when = _timestamp(2026, 8, 2)
        store = _MemoryStore(
            [SessionMeta(session_id="one", started_at=when, team="same")],
            workspace_id="same",
        )
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            build_org_report([store], "2026-08", team="same")
        self.assertEqual(store.loads, [])

    def test_local_metadata_and_selected_events_use_strict_report_schema(self):
        when = _timestamp(2026, 8, 2)
        bad_meta = SessionMeta(session_id="bad-meta", started_at=when)
        bad_meta.started_at = "2026-08-02"  # type: ignore[assignment]
        store = _MemoryStore([bad_meta])
        with self.assertRaisesRegex(ValueError, "started_at must be a number"):
            build_org_report([store], "2026-08")
        self.assertEqual(store.loads, [])

        bad_meta.started_at = 10**1000
        with self.assertRaisesRegex(ValueError, "finite"):
            build_org_report([store], "2026-08")

        meta = SessionMeta(session_id="bad-event", started_at=when)
        event = _event("bad-event", when + 1)
        event.data = {"nested": float("nan")}
        store = _MemoryStore([meta], {meta.session_id: [event]})
        with self.assertRaisesRegex(ValueError, "numbers must be finite"):
            build_org_report([store], "2026-08")

        unsafe = SessionMeta(
            session_id="unsafe-label", started_at=when, team="team\u202ename"
        )
        with self.assertRaisesRegex(ValueError, "control or format"):
            build_org_report([_MemoryStore([unsafe])], "2026-08")
        overlong = SessionMeta(
            session_id="long-label", started_at=when, team="x" * 201
        )
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            build_org_report([_MemoryStore([overlong])], "2026-08")
        padded = SessionMeta(
            session_id="padded-label", started_at=when, team="team "
        )
        padded_store = _MemoryStore([padded])
        with self.assertRaisesRegex(ValueError, "leading or trailing"):
            build_org_report([padded_store], "2026-08")
        self.assertEqual(padded_store.loads, [])

    def test_deep_local_event_file_fails_without_recursion_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TraceStore(directory, use_workspace_env=False, redact=False)
            meta = SessionMeta(
                session_id="deep-local", started_at=_timestamp(2026, 8, 2)
            )
            store.create_session(meta)
            payload = (
                '{"event_type":"user_prompt","timestamp":1,"event_id":"e",'
                '"session_id":"deep-local","data":{"nested":'
                + "[" * 10_000 + "0" + "]" * 10_000 + "}}\n"
            )
            (store._session_dir(meta.session_id) / "events.ndjson").write_text(
                payload, encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "nesting exceeds"):
                build_org_report([store], "2026-08")

    def test_tenants_are_tags_not_org_boundaries_and_are_not_emitted(self):
        when = _timestamp(2026, 8, 2)
        metas = [
            SessionMeta(session_id="a", started_at=when, tenant_id="customer-secret"),
            SessionMeta(session_id="b", started_at=when, tenant_id="customer-other"),
        ]
        store = _MemoryStore(metas)
        with patch("agent_trace.org_report.estimate_cost", return_value=SimpleNamespace(total_cost=0.5)), patch(
            "agent_trace.org_report.lint_session", side_effect=lambda _s, sid: LintReport(sid)
        ):
            report = build_org_report([store], "2026-08")
        payload = format_org_report_json(report)
        self.assertEqual(report.session_count, 2)
        self.assertNotIn("customer-secret", payload)
        self.assertNotIn("customer-other", payload)

    def test_unknown_costs_are_excluded_and_lint_counts_are_distinct(self):
        when = _timestamp(2026, 8, 2)
        metas = [
            SessionMeta(session_id="a", started_at=when, team="blue"),
            SessionMeta(session_id="b", started_at=when, team="blue"),
        ]
        store = _MemoryStore(metas)

        def cost(_store, session_id, model="sonnet"):
            del model
            if session_id == "b":
                raise ValueError("unknown")
            return SimpleNamespace(total_cost=2.0)

        def lint(_store, session_id):
            findings = []
            if session_id == "a":
                findings = [
                    LintResult("tool-loop", "WARN", "one"),
                    LintResult("tool-loop", "WARN", "two"),
                ]
            return LintReport(session_id, findings)

        with patch("agent_trace.org_report.estimate_cost", side_effect=cost), patch(
            "agent_trace.org_report.lint_session", side_effect=lint
        ):
            report = build_org_report([store], "2026-08")
        team = report.teams[0]
        self.assertEqual(report.cost_estimated_sessions, 1)
        self.assertEqual(report.avg_estimated_cost_usd, 2.0)
        self.assertEqual(report.cost_estimate_coverage_percent, 50.0)
        self.assertEqual(team.lint_findings, 2)
        self.assertEqual(team.sessions_with_lint_signals, 1)

    def test_attribution_uses_persisted_fields_without_changing_legacy_fallback(self):
        explicit = SessionMeta(attribution={"actor_id": "stable-id"})
        composite = SessionMeta(attribution={"os_user": "alice", "hostname": "build-1"})
        self.assertEqual(session_engineer_attribution(explicit), ("stable-id", "explicit"))
        self.assertEqual(session_engineer_attribution(composite), ("alice@build-1", "host-user"))
        with patch("agent_trace.team_report._git", return_value="legacy@example.com"):
            self.assertEqual(_fallback_author(SessionMeta()), "legacy@example.com")


class TestAnomaliesAndFormats(unittest.TestCase):
    @staticmethod
    def _team(name, average, sessions=MIN_ANOMALY_SESSIONS, coverage=100.0):
        return TeamBreakdown(
            team=name,
            sessions=sessions,
            cost_estimated_sessions=sessions,
            cost_estimate_coverage_percent=coverage,
            estimated_spend_usd=average * sessions,
            avg_estimated_cost_usd=average,
            active_attributed_identities=1,
            lint_findings=0,
            sessions_with_lint_signals=0,
            lint_signals={},
        )

    def test_anomaly_detection_uses_peer_averages_and_floors(self):
        teams = [
            self._team("a", 1), self._team("b", 1), self._team("c", 1),
            self._team("high", 4),
        ]
        anomalies = _build_anomalies([], teams)
        self.assertEqual(MIN_ANOMALY_PEERS, 3)
        self.assertEqual([item.subject for item in anomalies], ["high"])
        self.assertEqual(anomalies[0].ratio_to_peer_median, 4)
        self.assertEqual(anomalies[0].eligible_peers, 3)

        too_small = [self._team("a", 1), self._team("b", 1), self._team("high", 4)]
        self.assertEqual(_build_anomalies([], too_small), [])
        low_coverage = [
            self._team("a", 1), self._team("b", 1), self._team("c", 1),
            self._team("high", 4, coverage=MIN_ANOMALY_COVERAGE * 100 - 1),
        ]
        self.assertEqual(_build_anomalies([], low_coverage), [])

    def test_identity_anomaly_anonymizes_full_composite_stably(self):
        metrics = []
        for identity, cost in (
            ("alice@host-a", 1.0), ("bob@host-b", 1.0),
            ("carol@host-c", 1.0), ("secret@host-d", 4.0),
        ):
            metrics.extend(
                _SessionMetric("team", identity, "host-user", "other", cost, ())
                for _ in range(MIN_ANOMALY_SESSIONS)
            )
        raw = _build_anomalies(metrics, [])
        self.assertEqual(raw[0].subject, "secret@host-d")

        metas = [
            SessionMeta(
                session_id=f"s{i}", started_at=_timestamp(2026, 8, 2), team="<team>",
                attribution={"os_user": identity.split("@")[0], "hostname": identity.split("@")[1]},
            )
            for i, identity in enumerate(
                ["alice@host-a"] * 3 + ["bob@host-b"] * 3
                + ["carol@host-c"] * 3 + ["secret@host-d"] * 3
            )
        ]
        store = _MemoryStore(metas)

        def estimate(_store, sid, model="sonnet"):
            del model
            return SimpleNamespace(total_cost=4.0 if int(sid[1:]) >= 9 else 1.0)

        with patch("agent_trace.org_report.estimate_cost", side_effect=estimate), patch(
            "agent_trace.org_report.lint_session", side_effect=lambda _s, sid: LintReport(sid)
        ):
            first = build_org_report(
                [store], "2026-08", team="tag:<team>", anonymize=True,
                generated_at="same",
            )
            second = build_org_report(
                [store], "2026-08", team="tag:<team>", anonymize=True,
                generated_at="same",
            )
        self.assertEqual(format_org_report_json(first), format_org_report_json(second))
        for output in (
            format_org_report_json(first), format_org_report_text(first),
            format_org_report_html(first),
        ):
            self.assertNotIn("secret@host-d", output)
            self.assertNotIn("<team>", output)
            self.assertNotIn("tag:<team>", output)
            self.assertIn("Identity-004", output)

    def test_html_is_escaped_and_self_contained(self):
        meta = SessionMeta(
            session_id="escape", started_at=_timestamp(2026, 8, 2),
            team='<script src="https://bad.example/x.js">x</script>',
        )
        store = _MemoryStore([meta])
        with patch("agent_trace.org_report.estimate_cost", return_value=SimpleNamespace(total_cost=1)), patch(
            "agent_trace.org_report.lint_session", return_value=LintReport("escape")
        ):
            report = build_org_report([store], "2026-08")
        rendered = format_org_report_html(report)
        self.assertNotIn("<script", rendered)
        self.assertNotIn('src="https://bad.example', rendered)
        self.assertNotIn("<link", rendered)
        self.assertIn("&lt;script", rendered)

    def test_json_is_strict_and_declares_estimation_semantics(self):
        report = build_org_report([], "2026-08", generated_at="2026-09-01T00:00:00Z")
        payload = json.loads(format_org_report_json(report))
        self.assertEqual(payload["schema"], "agent-strace-org-report/v1")
        self.assertTrue(payload["spend_is_estimate"])
        self.assertIn("snapshot", payload["snapshot_limitation"].lower())
        self.assertNotIn("NaN", format_org_report_json(report))


class TestCollectorClient(unittest.TestCase):
    def _meta(self, session_id="remote", **values):
        item = {
            "session_id": session_id,
            "started_at": _timestamp(2026, 8, 2),
            "team": "blue",
            "attribution": {"actor_id": "alice"},
        }
        item.update(values)
        return item

    def _event_bytes(self, session_id="remote", **values):
        item = {
            "event_type": "user_prompt",
            "timestamp": _timestamp(2026, 8, 2) + 1,
            "event_id": "event-1",
            "session_id": session_id,
            "data": {"prompt": "hello"},
        }
        item.update(values)
        return (json.dumps(item) + "\n").encode()

    def test_endpoint_security_contract(self):
        self.assertEqual(validate_collector_endpoint("http://localhost:8000/"), "http://localhost:8000")
        self.assertEqual(validate_collector_endpoint("http://[::1]:8000"), "http://[::1]:8000")
        for endpoint in (
            "http://example.com", "ftp://example.com", "https://user:pass@example.com",
            "https://example.com?q=1", "https://example.com#frag", "https://example.com?",
            "https://example.com#",
        ):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                validate_collector_endpoint(endpoint)
        self.assertEqual(
            validate_collector_endpoint("http://example.com/base", allow_insecure_http=True),
            "http://example.com/base",
        )

    def test_auth_enforcement_and_percent_encoded_ids(self):
        meta = self._meta("space id")
        opener = _RouteOpener([meta], {"space%20id": self._event_bytes("space id")})
        client = CollectorClient("http://localhost:8000", "secret", opener=opener)
        sessions = client.list_sessions()
        events = client.load_events(sessions[0])
        self.assertEqual(events[0].session_id, "space id")
        self.assertIn(("/sessions/space%20id/events", True), opener.requests)

        unauthenticated = _RouteOpener([meta], {}, enforce_auth=False)
        with self.assertRaisesRegex(CollectorAuthenticationError, "does not enforce"):
            CollectorClient(
                "http://localhost:8000", "secret", opener=unauthenticated
            ).list_sessions()

    def test_redirect_size_and_truncation_fail_closed(self):
        class RedirectOpener:
            def open(self, request, timeout=0):
                del timeout
                raise urllib.error.HTTPError(request.full_url, 302, "redirect", {}, None)

        with self.assertRaisesRegex(CollectorClientError, "redirect"):
            CollectorClient("http://localhost:8000", "key", opener=RedirectOpener()).list_sessions()

        client = CollectorClient("http://localhost:8000", "key", opener=object(), max_metadata_bytes=1)
        with self.assertRaisesRegex(CollectorClientError, "size limit"):
            client._read_bounded(_Response(b"[]"), client.max_metadata_bytes)

        truncated = _Response(b"[]")
        truncated.headers["Content-Length"] = "20"
        with self.assertRaisesRegex(CollectorClientError, "truncated"):
            CollectorClient("http://localhost:8000", "key", opener=object())._read_bounded(
                truncated, 100
            )

        no_length = _Response(b"123", length=False)
        total_limited = CollectorClient(
            "http://localhost:8000", "key", opener=object(), max_total_bytes=2
        )
        with self.assertRaisesRegex(CollectorClientError, "size limit"):
            total_limited._read_bounded(no_length, 100)

    def test_strict_metadata_and_event_validation(self):
        bad_payloads = [
            b'[{"session_id":"x","started_at":NaN}]',
            b'[{"session_id":"x","session_id":"y","started_at":1}]',
            b'[{"session_id":"x","started_at":"now"}]',
            ('[{"session_id":"x","started_at":' + str(10**1000) + '}]').encode(),
            b'[{"session_id":"x","started_at":1,"attribution":[]}]',
            ("[" * 10_000 + "0" + "]" * 10_000).encode(),
        ]
        for body in bad_payloads:
            class BadOpener:
                def open(self, request, timeout=0):
                    del timeout
                    if not request.get_header("Authorization"):
                        raise urllib.error.HTTPError(request.full_url, 401, "no", {}, None)
                    return _Response(body)
            with self.subTest(body=body), self.assertRaises(CollectorClientError):
                CollectorClient("http://localhost:8000", "key", opener=BadOpener()).list_sessions()

        meta = self._meta(tenant_id="")
        mixed = self._event_bytes(tenant_id="one") + self._event_bytes(
            event_id="event-2", tenant_id="two"
        )
        client = CollectorClient("http://localhost:8000", "key", opener=object())
        client._get = lambda *args, **kwargs: mixed
        with self.assertRaisesRegex(CollectorClientError, "mixed tenants"):
            client.load_events(SessionMeta.from_json(json.dumps(meta)))

        wrong_data = self._event_bytes(data=[])
        client._get = lambda *args, **kwargs: wrong_data
        with self.assertRaisesRegex(CollectorClientError, "malformed"):
            client.load_events(SessionMeta.from_json(json.dumps(meta)))

        deeply_nested = (
            '{"event_type":"user_prompt","timestamp":1,"event_id":"e",'
            '"session_id":"remote","data":'
            + "[" * 10_000 + "0" + "]" * 10_000 + "}\n"
        ).encode()
        client._get = lambda *args, **kwargs: deeply_nested
        with self.assertRaisesRegex(CollectorClientError, "malformed"):
            client.load_events(SessionMeta.from_json(json.dumps(meta)))

    def test_remote_metadata_filters_before_n_plus_one_event_reads(self):
        selected = self._meta("selected")
        outside = self._meta("outside", started_at=_timestamp(2026, 7, 2))
        other_team = self._meta("red", team="red")
        opener = _RouteOpener(
            [selected, outside, other_team],
            {
                "selected": self._event_bytes("selected"),
                "outside": self._event_bytes("outside"),
                "red": self._event_bytes("red"),
            },
        )
        store = CollectorTraceStore.load(
            CollectorClient("http://localhost:8000", "key", opener=opener)
        )
        report = build_org_report(
            [store], "2026-08", team="tag:blue", source_mode="collector-instance"
        )
        self.assertEqual(report.session_count, 1)
        event_paths = [path for path, _ in opener.requests if path.endswith("/events")]
        self.assertEqual(event_paths, ["/sessions/selected/events"])
        self.assertEqual(store._events, {})
        self.assertIn("N+1", report.snapshot_limitation)

    def test_ambient_proxy_is_never_used_for_loopback_bearer(self):
        captured = []

        class ProxyHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                captured.append((self.path, self.headers.get("Authorization")))
                body = b"[]"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        proxy = ThreadingHTTPServer(("127.0.0.1", 0), ProxyHandler)
        thread = threading.Thread(target=proxy.serve_forever, daemon=True)
        thread.start()
        unavailable = socket.socket()
        unavailable.bind(("127.0.0.1", 0))
        port = unavailable.getsockname()[1]
        proxy_url = f"http://127.0.0.1:{proxy.server_port}"
        try:
            with patch.dict(
                os.environ,
                {"http_proxy": proxy_url, "HTTP_PROXY": proxy_url, "NO_PROXY": "", "no_proxy": ""},
                clear=False,
            ):
                with self.assertRaises(CollectorClientError):
                    CollectorClient(
                        f"http://127.0.0.1:{port}", "TOPSECRET", timeout=0.2
                    ).list_sessions()
            self.assertEqual(captured, [])
        finally:
            unavailable.close()
            proxy.shutdown()
            proxy.server_close()
            thread.join(timeout=2)


class TestCliAndOutput(unittest.TestCase):
    def test_cli_registers_org_report_and_does_not_read_endpoint_from_env(self):
        args = build_parser().parse_args(["org-report", "--month", "2026-08"])
        self.assertEqual(args.command, "org-report")
        self.assertIsNone(args.endpoint)

    def test_atomic_output_rejects_symlink_and_preserves_existing_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.txt"
            target.write_text("old", encoding="utf-8")
            link = root / "report.txt"
            link.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "symlink"):
                _write_atomic(str(link), "new")
            self.assertEqual(target.read_text(encoding="utf-8"), "old")

            parent_target = root / "real"
            parent_target.mkdir()
            parent_link = root / "linked"
            parent_link.symlink_to(parent_target, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                _write_atomic(str(parent_link / "report.txt"), "new")

            private = root / "private.txt"
            _write_atomic(str(private), "secret")
            self.assertEqual(private.stat().st_mode & 0o077, 0)

            stable = root / "stable.txt"
            stable.write_text("original", encoding="utf-8")
            with patch("agent_trace.org_report.os.replace", side_effect=OSError("stop")):
                with self.assertRaises(OSError):
                    _write_atomic(str(stable), "replacement")
            self.assertEqual(stable.read_text(encoding="utf-8"), "original")
            self.assertEqual(list(root.glob(".stable.txt.*")), [])


if __name__ == "__main__":
    unittest.main()
