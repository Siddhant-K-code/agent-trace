"""Tests for the web dashboard module and server --dashboard integration."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import HTTPServer
from urllib.parse import quote

sys.path.insert(0, "src")

from agent_trace.web_dashboard import (
    render_sessions_page,
    render_detail_page,
    render_cost_page,
    render_violations_page,
    render_health_page,
    api_sessions,
    api_session_events,
    api_session_review,
    api_session_compare,
    api_minimized_review_share,
    api_session_otlp,
)
from agent_trace.server import _make_handler
from agent_trace.store import TraceStore
from agent_trace.models import SessionMeta, TraceEvent, EventType
from agent_trace.event_data import NonObjectEventDataError

try:
    from test_html_replay import LEGACY_ID, SKIP_LEGACY_ON_WINDOWS, write_legacy_session
    from test_store import OTHER_SESSION, event_line, write_session_lines
except ImportError:  # pragma: no cover - package-style test discovery
    from tests.test_html_replay import LEGACY_ID, SKIP_LEGACY_ON_WINDOWS, write_legacy_session
    from tests.test_store import OTHER_SESSION, event_line, write_session_lines

# The detail page title cuts the id at 12 characters, then escapes it.
LEGACY_TITLE = "<title>Session L&lt;&quot;&amp;a #?%20&amp; | agent-trace</title>"


def _start_server(tmp_dir, dashboard=True):
    store = TraceStore(tmp_dir)
    lock = threading.Lock()
    handler = _make_handler(store, lock, auth_key="", dashboard=dashboard)
    server = HTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, port, store


def _append_raw_event(store, session_id, data):
    line = json.dumps({
        "event_type": "tool_call",
        "timestamp": 2,
        "event_id": "bad-data",
        "session_id": session_id,
        "data": data,
    })
    path = store.base_dir / session_id / "events.ndjson"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _tool_call_session(store, session_id, timestamp=2.0):
    store.create_session(SessionMeta(session_id=session_id, started_at=1.0))
    store.append_event(session_id, TraceEvent(
        event_type=EventType.TOOL_CALL,
        timestamp=timestamp,
        session_id=session_id,
        event_id="call",
        data={"tool_name": "Bash"},
    ))


def _set_raw_meta(store, session_id, **fields):
    """Write *fields* into meta.json as given, past the store's checks."""
    path = store.base_dir / session_id / "meta.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta.update(fields)
    path.write_text(json.dumps(meta), encoding="utf-8")


def _otlp_refusal(port, session_id):
    """The status and JSON body of the OTLP download, or 200 and None when it is served."""
    url = f"http://127.0.0.1:{port}/api/sessions/{session_id}/otlp"
    try:
        with urllib.request.urlopen(url):
            pass
    except urllib.error.HTTPError as error:
        with error:
            return error.code, json.loads(error.read())
    return 200, None


class TestRenderPages(unittest.TestCase):
    def test_sessions_page_contains_nav(self):
        html = render_sessions_page()
        self.assertIn("<nav>", html)
        self.assertIn('href="/"', html)
        self.assertIn('href="/cost"', html)
        self.assertIn('href="/violations"', html)
        self.assertIn('href="/health"', html)

    def test_sessions_page_active_nav(self):
        self.assertIn('class="active"', render_sessions_page())

    def test_detail_page_contains_session_id(self):
        html = render_detail_page("abc123def456")
        self.assertIn("abc123def4", html)
        self.assertIn("Local evidence review", html)
        self.assertIn("Save minimized share", html)
        self.assertIn("Download OTLP JSON", html)
        self.assertIn("Compare sessions", html)
        self.assertIn("J/K or arrows move", html)

    def test_cost_page_title(self):
        html = render_cost_page()
        self.assertIn("Cost", html)
        self.assertIn("By Team", html)

    def test_violations_page_title(self):
        self.assertIn("Violations", render_violations_page())

    def test_health_page_title(self):
        self.assertIn("Health", render_health_page())

    def test_all_pages_are_valid_html(self):
        for html in [render_sessions_page(), render_detail_page("t"),
                     render_cost_page(), render_violations_page(), render_health_page()]:
            self.assertTrue(html.startswith("<!DOCTYPE html>"))
            self.assertIn("</html>", html)
            self.assertIn("<script>", html)

    def test_detail_page_title_shows_legacy_id_as_text(self):
        html = render_detail_page(LEGACY_ID)
        self.assertIn(LEGACY_TITLE, html)
        self.assertNotIn('L<"', html)


class TestPageScriptsShowDataAsText(unittest.TestCase):
    """No page script parses session or event data as HTML."""

    @staticmethod
    def _scripts() -> dict:
        from agent_trace.replay import _DIFF_HTML_TEMPLATE, _HTML_TEMPLATE
        pages = {
            "sessions": render_sessions_page(),
            "detail": render_detail_page("t"),
            "cost": render_cost_page(),
            "violations": render_violations_page(),
            "health": render_health_page(),
            "replay": _HTML_TEMPLATE,
            "diff": _DIFF_HTML_TEMPLATE,
        }
        return {
            name: "".join(re.findall(r"<script>(.*?)</script>", page, re.S))
            for name, page in pages.items()
        }

    def test_no_html_parsing_except_clearing(self):
        sinks = re.compile(r"innerHTML|outerHTML|insertAdjacentHTML|document\.write")
        clearing = re.compile(r"\.innerHTML\s*=\s*(?:''|\"\")")
        for name, script in self._scripts().items():
            with self.subTest(page=name):
                self.assertTrue(script.strip())
                self.assertIsNone(sinks.search(clearing.sub("", script)))

    def test_session_ids_in_urls_are_encoded(self):
        scripts = self._scripts()
        for name, script in scripts.items():
            with self.subTest(page=name):
                for match in re.finditer(r"/(?:session|api/sessions)/", script):
                    self.assertEqual(
                        script[match.end():match.end() + 21], "${encodeURIComponent(",
                        script[match.start():match.end() + 40],
                    )
        self.assertIn("/session/${encodeURIComponent(", scripts["sessions"])
        self.assertIn("/api/sessions/${encodeURIComponent(", scripts["violations"])


# Runs the detail page script under Node with a minimal fake DOM and fetch.
# Each compare response waits its own delay, so a test can make an earlier
# request answer after a later one.
_COMPARE_HARNESS = r"""
const script = __SCRIPT__;
const responses = __RESPONSES__;
const order = __ORDER__;
const elements = {};
function makeEl() {
  return {
    value: '', textContent: '', className: '', disabled: false, style: {},
    children: [], options: [], listeners: {},
    classList: {add() {}, remove() {}},
    replaceChildren(...c) { this.children = c; },
    append(...c) { this.children.push(...c); },
    appendChild(c) { this.children.push(c); return c; },
    addEventListener(type, fn) { this.listeners[type] = fn; },
    querySelector() { return null; },
  };
}
const document = {
  getElementById(id) { return elements[id] || (elements[id] = makeEl()); },
  createElement() { return makeEl(); },
  createTextNode(t) { return {textContent: String(t), children: []}; },
  addEventListener() {},
  querySelectorAll() { return []; },
};
const window = {location: {pathname: '/session/self', href: 'http://dash/session/self'}};
async function fetch(url) {
  if (!url.includes('/compare?')) return {ok: false, json: async () => ({error: 'no review'})};
  const reply = responses[new URL(url, 'http://dash').searchParams.get('other')];
  await new Promise(resolve => setTimeout(resolve, reply.delay));
  return {ok: reply.ok, json: async () => reply.body};
}
new Function('document', 'window', 'fetch', script)(document, window, fetch);
const text = e => [e.textContent || '', ...(e.children || []).map(text)].join(' ');
const select = document.getElementById('compare-session');
const click = document.getElementById('compare-action').listeners.click;
for (const other of order) { select.value = other; click(); }
setTimeout(() => {
  process.stdout.write(JSON.stringify(text(document.getElementById('compare-panel'))));
}, 300);
"""


def _compare_report(label, tool_calls):
    return {
        "label_a": "self", "label_b": label, "verdict": f"verdict {label}",
        "duration_a": 1, "duration_b": 2, "cost_a": 0, "cost_b": 0,
        "tool_calls_a": 1, "tool_calls_b": tool_calls,
        "files_modified_a": 0, "files_modified_b": 0,
        "errors_a": 0, "errors_b": 0, "decision_divergence": 0,
    }


class TestDetailPageCompareRace(unittest.TestCase):
    """Only the newest compare request may paint the compare panel."""

    @staticmethod
    def _detail_script():
        return "".join(re.findall(r"<script>(.*?)</script>", render_detail_page("t"), re.S))

    def _run(self, responses, order):
        node = shutil.which("node")
        if node is None:
            self.skipTest("node is not installed")
        harness = (_COMPARE_HARNESS
                   .replace("__SCRIPT__", json.dumps(self._detail_script()))
                   .replace("__RESPONSES__", json.dumps(responses))
                   .replace("__ORDER__", json.dumps(order)))
        result = subprocess.run([node, "-"], input=harness, capture_output=True,
                                text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_script_ignores_responses_of_older_requests(self):
        script = self._detail_script()
        compare = script[script.index("async function compare()"):script.index("async function load()")]
        self.assertEqual(compare.count("if(request!==compareRequest)return;"), 2)

    def test_slow_earlier_response_does_not_replace_newer_comparison(self):
        panel = self._run({
            "slow": {"delay": 150, "ok": True, "body": _compare_report("slow", 6)},
            "fast": {"delay": 0, "ok": True, "body": _compare_report("fast", 1)},
        }, ["slow", "fast"])
        self.assertIn("verdict fast", panel)
        self.assertNotIn("slow", panel)

    def test_slow_earlier_error_does_not_replace_newer_comparison(self):
        panel = self._run({
            "slow": {"delay": 150, "ok": False, "body": {"error": "slow failed"}},
            "fast": {"delay": 0, "ok": True, "body": _compare_report("fast", 1)},
        }, ["slow", "fast"])
        self.assertIn("verdict fast", panel)
        self.assertNotIn("slow failed", panel)

    def test_newest_slow_response_and_error_still_show(self):
        panel = self._run({
            "fast": {"delay": 0, "ok": True, "body": _compare_report("fast", 1)},
            "slow": {"delay": 150, "ok": True, "body": _compare_report("slow", 6)},
        }, ["fast", "slow"])
        self.assertIn("verdict slow", panel)
        self.assertNotIn("fast", panel)
        panel = self._run({
            "slow": {"delay": 150, "ok": False, "body": {"error": "slow failed"}},
        }, ["slow"])
        self.assertIn("Error: slow failed", panel)


class TestApiHelpers(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def test_api_sessions_empty(self):
        self.assertEqual(json.loads(api_sessions(TraceStore(self._tmp))), [])

    def test_api_sessions_returns_list(self):
        store = TraceStore(self._tmp)
        store.create_session(SessionMeta(session_id="sess001", agent_name="agent"))
        data = json.loads(api_sessions(store))
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["session_id"], "sess001")

    def test_api_session_events_not_found(self):
        self.assertIsNone(api_session_events(TraceStore(self._tmp), "nope"))

    def test_api_session_events_returns_events(self):
        store = TraceStore(self._tmp)
        store.create_session(SessionMeta(session_id="s2"))
        store.append_event("s2", TraceEvent(event_type=EventType.TOOL_CALL,
                                            session_id="s2", data={"tool_name": "bash"}))
        data = json.loads(api_session_events(store, "s2"))
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["event_type"], "tool_call")

    def test_api_session_events_prefix_lookup(self):
        store = TraceStore(self._tmp)
        store.create_session(SessionMeta(session_id="abcdef123456"))
        store.append_event("abcdef123456",
                           TraceEvent(event_type=EventType.SESSION_START,
                                      session_id="abcdef123456"))
        data = json.loads(api_session_events(store, "abcdef"))
        self.assertEqual(len(data), 1)

    def test_api_session_review_returns_projected_evidence(self):
        store = TraceStore(self._tmp)
        store.create_session(SessionMeta(
            session_id="review001",
            started_at=1,
            ended_at=3,
            agent_name="review-agent",
        ))
        store.append_event("review001", TraceEvent(
            event_type=EventType.SESSION_START,
            timestamp=1,
            session_id="review001",
            event_id="start",
            data={"provider": "synthetic", "capture_method": "fixture"},
        ))
        store.append_event("review001", TraceEvent(
            event_type=EventType.TOOL_RESULT,
            timestamp=2,
            session_id="review001",
            event_id="orphan",
            parent_id="missing",
            data={"output": "private output"},
        ))
        store.append_event("review001", TraceEvent(
            event_type=EventType.SESSION_END,
            timestamp=3,
            session_id="review001",
            event_id="end",
        ))

        payload = json.loads(api_session_review(store, "review001"))

        self.assertEqual(payload["session"]["provider"], "synthetic")
        self.assertEqual(payload["items"][1]["relationship"], "orphaned")
        self.assertIn("gaps", payload["items"][1]["categories"])

    def test_minimized_share_and_otlp_are_local_artifacts(self):
        store = TraceStore(self._tmp)
        store.create_session(SessionMeta(
            session_id="artifact001",
            started_at=1,
            ended_at=2,
        ))
        store.append_event("artifact001", TraceEvent(
            event_type=EventType.SESSION_START,
            timestamp=1,
            session_id="artifact001",
            event_id="start",
            data={"provider": "synthetic", "secret": "do-not-share"},
        ))
        store.append_event("artifact001", TraceEvent(
            event_type=EventType.SESSION_END,
            timestamp=2,
            session_id="artifact001",
            event_id="end",
        ))

        share = api_minimized_review_share(store, "artifact001")
        self.assertIn("Privacy-minimized evidence review", share)
        self.assertNotIn("do-not-share", share)
        self.assertNotIn("synthetic", share)
        otlp = json.loads(api_session_otlp(store, "artifact001"))
        self.assertIn("resourceSpans", otlp)

    def test_api_session_compare_reuses_existing_report(self):
        store = TraceStore(self._tmp)
        for session_id, ended_at in (("compare-a", 2), ("compare-b", 3)):
            store.create_session(SessionMeta(
                session_id=session_id,
                started_at=1,
                ended_at=ended_at,
            ))
            store.append_event(session_id, TraceEvent(
                event_type=EventType.SESSION_START,
                timestamp=1,
                session_id=session_id,
                event_id=f"{session_id}-start",
            ))
            store.append_event(session_id, TraceEvent(
                event_type=EventType.SESSION_END,
                timestamp=ended_at,
                session_id=session_id,
                event_id=f"{session_id}-end",
            ))

        report = json.loads(api_session_compare(store, "compare-a", "compare-b"))

        self.assertEqual(report["session_a"], "compare-a")
        self.assertEqual(report["session_b"], "compare-b")
        self.assertIn("decision_divergence", report)

    def test_review_helpers_reject_non_object_event_data(self):
        store = TraceStore(self._tmp)
        store.create_session(SessionMeta(session_id="invalid-review", started_at=1))
        _append_raw_event(store, "invalid-review", ["not", "an", "object"])

        for operation in (api_session_review, api_minimized_review_share):
            with self.subTest(operation=operation.__name__):
                with self.assertRaisesRegex(
                    NonObjectEventDataError,
                    "session invalid-review event data must be an object",
                ):
                    operation(store, "invalid-review")


class TestServerDashboardRoutes(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def test_root_returns_html(self):
        server, port, _ = _start_server(self._tmp)
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{port}/")
            self.assertEqual(r.status, 200)
            self.assertIn("text/html", r.headers.get("Content-Type", ""))
            self.assertIn("<!DOCTYPE html>", r.read().decode())
        finally:
            server.shutdown()
            server.server_close()

    def test_cost_route(self):
        server, port, _ = _start_server(self._tmp)
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{port}/cost")
            self.assertEqual(r.status, 200)
            self.assertIn("Cost", r.read().decode())
        finally:
            server.shutdown()
            server.server_close()

    def test_violations_route(self):
        server, port, _ = _start_server(self._tmp)
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{port}/violations")
            self.assertEqual(r.status, 200)
        finally:
            server.shutdown()
            server.server_close()

    def test_health_route_returns_html_when_dashboard(self):
        server, port, _ = _start_server(self._tmp)
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{port}/health")
            self.assertEqual(r.status, 200)
            self.assertIn("text/html", r.headers.get("Content-Type", ""))
        finally:
            server.shutdown()
            server.server_close()

    def test_session_detail_route(self):
        server, port, _ = _start_server(self._tmp)
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{port}/session/abc123")
            self.assertEqual(r.status, 200)
            self.assertIn("<!DOCTYPE html>", r.read().decode())
        finally:
            server.shutdown()
            server.server_close()

    def test_api_sessions_route(self):
        server, port, store = _start_server(self._tmp)
        try:
            store.create_session(SessionMeta(session_id="live001", agent_name="a"))
            r = urllib.request.urlopen(f"http://127.0.0.1:{port}/api/sessions")
            self.assertEqual(r.status, 200)
            data = json.loads(r.read())
            self.assertTrue(any(s["session_id"] == "live001" for s in data))
        finally:
            server.shutdown()
            server.server_close()

    def test_api_session_events_route(self):
        server, port, store = _start_server(self._tmp)
        try:
            store.create_session(SessionMeta(session_id="live002"))
            store.append_event("live002",
                               TraceEvent(event_type=EventType.TOOL_CALL,
                                          session_id="live002",
                                          data={"tool_name": "read_file"}))
            r = urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/sessions/live002/events")
            self.assertEqual(r.status, 200)
            data = json.loads(r.read())
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["event_type"], "tool_call")
        finally:
            server.shutdown()
            server.server_close()

    @SKIP_LEGACY_ON_WINDOWS
    def test_legacy_id_routes_decode_the_encoded_id(self):
        self.addCleanup(shutil.rmtree, self._tmp, True)
        server, port, store = _start_server(self._tmp)
        try:
            write_legacy_session(store, LEGACY_ID, [TraceEvent(
                event_type=EventType.TOOL_CALL, session_id=LEGACY_ID,
                event_id="legacy-call", data={"tool_name": "Bash"},
            )])
            # The second form is what encodeURIComponent sends: it keeps ' raw.
            for encoded in (quote(LEGACY_ID, safe=""), quote(LEGACY_ID, safe="!'()*")):
                with self.subTest(encoded=encoded):
                    with urllib.request.urlopen(
                            f"http://127.0.0.1:{port}/api/sessions/{encoded}/events") as r:
                        self.assertEqual(r.status, 200)
                        events = json.loads(r.read())
                    self.assertEqual(
                        [(e["event_id"], e["session_id"]) for e in events],
                        [("legacy-call", LEGACY_ID)],
                    )
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/session/{encoded}") as r:
                        self.assertEqual(r.status, 200)
                        self.assertIn(LEGACY_TITLE, r.read().decode())
        finally:
            server.shutdown()
            server.server_close()

    def test_api_session_events_404(self):
        server, port, _ = _start_server(self._tmp)
        try:
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/api/sessions/nosuch/events")
            self.assertEqual(ctx.exception.code, 404)
        finally:
            server.shutdown()
            server.server_close()

    def test_review_download_routes(self):
        server, port, store = _start_server(self._tmp)
        try:
            store.create_session(SessionMeta(
                session_id="downloads001",
                started_at=1,
                ended_at=2,
            ))
            store.append_event("downloads001", TraceEvent(
                event_type=EventType.SESSION_START,
                timestamp=1,
                session_id="downloads001",
                event_id="start",
            ))
            store.append_event("downloads001", TraceEvent(
                event_type=EventType.SESSION_END,
                timestamp=2,
                session_id="downloads001",
                event_id="end",
            ))

            review = urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/sessions/downloads001/review"
            )
            self.assertEqual(review.status, 200)
            self.assertEqual(json.loads(review.read())["session"]["session_id"], "downloads001")

            share = urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/sessions/downloads001/share?mode=minimized"
            )
            self.assertIn("attachment", share.headers.get("Content-Disposition", ""))
            self.assertIn("Privacy-minimized evidence review", share.read().decode())

            otlp = urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/sessions/downloads001/otlp"
            )
            self.assertIn("attachment", otlp.headers.get("Content-Disposition", ""))
            self.assertIn("resourceSpans", json.loads(otlp.read()))
        finally:
            server.shutdown()
            server.server_close()

    def test_review_routes_return_400_for_non_object_event_data(self):
        server, port, store = _start_server(self._tmp)
        try:
            store.create_session(SessionMeta(
                session_id="invalid-review-route",
                started_at=1,
            ))
            _append_raw_event(
                store,
                "invalid-review-route",
                "not-an-object",
            )
            for suffix in ("review", "share?mode=minimized"):
                with self.subTest(suffix=suffix):
                    with self.assertRaises(urllib.error.HTTPError) as context:
                        urllib.request.urlopen(
                            f"http://127.0.0.1:{port}/api/sessions/"
                            f"invalid-review-route/{suffix}"
                        )
                    self.assertEqual(context.exception.code, 400)
                    payload = json.loads(context.exception.read())
                    self.assertEqual(
                        payload["error"],
                        "session invalid-review-route event data must be an object: "
                        "event 1 of 1 (tool_call) has string data",
                    )
        finally:
            server.shutdown()
            server.server_close()

    def test_session_routes_return_400_for_a_refused_session(self):
        server, port, store = _start_server(self._tmp)
        try:
            other = write_session_lines(store, [event_line()])
            mismatch = write_session_lines(
                store, [event_line(), event_line(session_id=OTHER_SESSION)]
            )
            unreadable = write_session_lines(store, [event_line(), "[1]"])
            for session_id, reason in (
                (mismatch, "event session ID does not match its directory"),
                (unreadable, "line 2 of events.ndjson is an array, not an object"),
            ):
                for suffix in (
                    "events", "review", "share?mode=minimized", "otlp", f"compare?other={other}",
                ):
                    with self.subTest(reason=reason, suffix=suffix):
                        with self.assertRaises(urllib.error.HTTPError) as context:
                            urllib.request.urlopen(
                                f"http://127.0.0.1:{port}/api/sessions/{session_id}/{suffix}"
                            )
                        with context.exception as error:
                            self.assertEqual(error.code, 400)
                            self.assertEqual(
                                json.loads(error.read()),
                                {"error": f"cannot read session {session_id}: {reason}"},
                            )
        finally:
            server.shutdown()
            server.server_close()

    def test_dashboard_disabled_root_falls_through(self):
        server, port, _ = _start_server(self._tmp, dashboard=False)
        try:
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/")
            self.assertEqual(ctx.exception.code, 404)
        finally:
            server.shutdown()
            server.server_close()

    def test_otlp_route_returns_400_for_a_time_past_the_otlp_limit(self):
        year_3000 = 32503680000.0
        limit = "is past the latest time OTLP can carry (2554-07-21T23:34:33.709551615Z)"
        server, port, store = _start_server(self._tmp)
        try:
            cases = (
                ("otlp-limit-start", {"started_at": year_3000}, 2.0,
                 f"session started_at {limit}"),
                ("otlp-limit-end", {"started_at": 1.0, "ended_at": year_3000}, 2.0,
                 f"session ended_at {limit}"),
                ("otlp-limit-event", {"started_at": 1.0}, year_3000,
                 f"session otlp-limit-event event timestamp {limit}: event 1 of 1 (tool_call)"),
            )
            for session_id, times, timestamp, message in cases:
                store.create_session(SessionMeta(session_id=session_id, **times))
                store.append_event(session_id, TraceEvent(
                    event_type=EventType.TOOL_CALL,
                    timestamp=timestamp,
                    session_id=session_id,
                    event_id="call",
                    data={"tool_name": "Bash"},
                ))
                with self.subTest(session_id=session_id):
                    with self.assertRaises(urllib.error.HTTPError) as context:
                        urllib.request.urlopen(
                            f"http://127.0.0.1:{port}/api/sessions/{session_id}/otlp"
                        )
                    with context.exception as error:
                        self.assertEqual(error.code, 400)
                        self.assertEqual(json.loads(error.read()), {"error": message})
        finally:
            server.shutdown()
            server.server_close()

    def test_otlp_route_returns_400_for_a_tenant_the_store_would_refuse(self):
        refused = [(value, "must be a string") for value in (
            None, 0, False, [], {}, 5, True, 1.5, float("nan"), float("inf"), ["acme"], {"a": 1},
        )] + [
            ("   ", "cannot be empty"),
            ("ac\x01me", "cannot contain control characters"),
            ("x" * 300, "cannot exceed 256 characters"),
        ]
        server, port, store = _start_server(self._tmp)
        try:
            for index, (value, reason) in enumerate(refused):
                session_id = f"refused-tenant-{index}"
                _tool_call_session(store, session_id)
                _set_raw_meta(store, session_id, tenant_id=value)
                with self.subTest(tenant=value):
                    self.assertEqual(
                        _otlp_refusal(port, session_id),
                        (400, {"error": f"session tenant_id {reason}"}),
                    )
        finally:
            server.shutdown()
            server.server_close()

    def test_otlp_route_exports_an_accepted_tenant_stripped(self):
        server, port, store = _start_server(self._tmp)
        try:
            for index, (value, expected) in enumerate((
                (" acme ", [{"stringValue": "acme"}]),
                ("", []),
            )):
                session_id = f"accepted-tenant-{index}"
                _tool_call_session(store, session_id)
                _set_raw_meta(store, session_id, tenant_id=value)
                with self.subTest(tenant=value):
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/api/sessions/{session_id}/otlp"
                    ) as response:
                        self.assertEqual(response.status, 200)
                        self.assertIn("attachment", response.headers.get("Content-Disposition", ""))
                        payload = json.loads(response.read())
                    resource_spans = payload["resourceSpans"][0]
                    holders = [resource_spans["resource"], *resource_spans["scopeSpans"][0]["spans"]]
                    values = [
                        [a["value"] for a in holder.get("attributes", []) if a["key"] == "tenant_id"]
                        for holder in holders
                    ]
                    self.assertGreaterEqual(len(values), 3)
                    self.assertEqual(values, [expected] * len(values))
        finally:
            server.shutdown()
            server.server_close()

    def test_otlp_route_checks_the_tenant_after_the_other_session_checks(self):
        limit = "is past the latest time OTLP can carry (2554-07-21T23:34:33.709551615Z)"
        year_3000 = 32503680000.0

        def null_timestamp(store, session_id):
            path = store.base_dir / session_id / "events.ndjson"
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "event_type": "tool_call",
                    "timestamp": None,
                    "event_id": "null-time",
                    "session_id": session_id,
                    "data": {"tool_name": "Bash"},
                }) + "\n")

        # Each case: the event timestamp, a change to the stored session, the
        # meta.json fields besides the tenant, and the message.
        cases = (
            ("tenant-alone", 2.0, None, {}, "session tenant_id must be a string"),
            ("tenant-data", 2.0, lambda store, sid: _append_raw_event(store, sid, "text"), {},
             "session tenant-data event data must be an object: "
             "event 2 of 2 (tool_call) has string data"),
            ("tenant-event-time", 2.0, null_timestamp, {},
             "session tenant-event-time event timestamp must be a Unix time in seconds: "
             "event 2 of 2 (tool_call) has a null timestamp"),
            ("tenant-start", 2.0, None, {"started_at": float("nan")},
             "session started_at must be a Unix time in seconds"),
            ("tenant-start-limit", 2.0, None, {"started_at": year_3000},
             f"session started_at {limit}"),
            ("tenant-event-limit", year_3000, None, {},
             f"session tenant-event-limit event timestamp {limit}: event 1 of 1 (tool_call)"),
        )
        server, port, store = _start_server(self._tmp)
        try:
            for session_id, timestamp, change, fields, message in cases:
                _tool_call_session(store, session_id, timestamp)
                if change:
                    change(store, session_id)
                _set_raw_meta(store, session_id, tenant_id=5, **fields)
                with self.subTest(session_id=session_id):
                    self.assertEqual(_otlp_refusal(port, session_id), (400, {"error": message}))
        finally:
            server.shutdown()
            server.server_close()

    def test_health_json_when_dashboard_disabled(self):
        server, port, _ = _start_server(self._tmp, dashboard=False)
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{port}/health")
            self.assertEqual(r.status, 200)
            self.assertIn("application/json", r.headers.get("Content-Type", ""))
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
