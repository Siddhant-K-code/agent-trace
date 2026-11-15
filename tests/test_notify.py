"""Tests for Slack/Teams/webhook notifications (issue #138)."""

import io
import json
import os
import socket
import sys
import threading
import unittest
from contextlib import redirect_stderr
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import call, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace.notify import (
    _slack_payload,
    _teams_payload,
    _generic_payload,
    _post,
    notify,
    notify_violation,
    notify_budget_alert,
    notify_session_end,
    _SLACK_ENV,
    _TEAMS_ENV,
    _GENERIC_ENV,
)


class TestSlackPayload(unittest.TestCase):
    def test_contains_title_and_message(self):
        p = _slack_payload("Test title", "Test message")
        att = p["attachments"][0]
        self.assertEqual(att["title"], "Test title")
        self.assertEqual(att["text"], "Test message")

    def test_error_level_uses_red_color(self):
        p = _slack_payload("t", "m", level="error")
        self.assertEqual(p["attachments"][0]["color"], "#cc0000")

    def test_warning_level_uses_orange(self):
        p = _slack_payload("t", "m", level="warning")
        self.assertEqual(p["attachments"][0]["color"], "#ff9900")

    def test_fields_included(self):
        p = _slack_payload("t", "m", fields={"tool": "Bash", "action": "block"})
        fields = p["attachments"][0]["fields"]
        titles = [f["title"] for f in fields]
        self.assertIn("tool", titles)
        self.assertIn("action", titles)

    def test_session_id_appended_to_fields(self):
        p = _slack_payload("t", "m", session_id="abc123def456")
        fields = p["attachments"][0].get("fields", [])
        values = [f["value"] for f in fields]
        self.assertTrue(any("abc123def456"[:12] in v for v in values))


class TestTeamsPayload(unittest.TestCase):
    def test_message_card_type(self):
        p = _teams_payload("Title", "Body")
        self.assertEqual(p["@type"], "MessageCard")

    def test_error_theme_color(self):
        p = _teams_payload("t", "m", level="error")
        self.assertEqual(p["themeColor"], "CC0000")

    def test_facts_from_fields(self):
        p = _teams_payload("t", "m", fields={"rule": "no-network"})
        facts = p["sections"][0]["facts"]
        names = [f["name"] for f in facts]
        self.assertIn("rule", names)

    def test_session_id_in_facts(self):
        p = _teams_payload("t", "m", session_id="sess001")
        facts = p["sections"][0]["facts"]
        self.assertTrue(any(f["name"] == "Session" for f in facts))


class TestGenericPayload(unittest.TestCase):
    def test_has_required_keys(self):
        p = _generic_payload("t", "m", level="warning", session_id="s1")
        self.assertEqual(p["title"], "t")
        self.assertEqual(p["level"], "warning")
        self.assertEqual(p["session_id"], "s1")
        self.assertIn("timestamp", p)


# No failure line may show these parts of a URL: the token lives there.
_MARKERS = ("SECRET", "token")
_NOT_FOLLOWED = " (redirects are not followed; give the final URL)"


def _closed_port() -> int:
    """Return a local port that nothing listens on."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _PostServer(ThreadingHTTPServer):
    """Answer each POST with the next scripted (status, hold) reply.

    A reply waits *hold* seconds, or until ``release`` is set, before it is
    sent. A 3xx reply points Location at a URL with markers in it, and every
    GET gets 200.
    """

    daemon_threads = True
    block_on_close = False

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _PostHandler)
        self.lock = threading.Lock()
        self.requests: list[tuple[str, str, bytes]] = []
        self.replies: list[tuple[int, float]] = []
        self.release = threading.Event()

    def handle_error(self, request, client_address) -> None:
        pass  # the client timed out and closed the connection

    def script(self, *replies: tuple[int, float]) -> None:
        with self.lock:
            self.requests.clear()
            self.replies[:] = replies
        self.release.clear()

    def methods(self) -> list[str]:
        with self.lock:
            return [method for method, _, _ in self.requests]


class _PostHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        server = self.server
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        with server.lock:
            server.requests.append(("POST", self.path, body))
            status, hold = server.replies.pop(0) if server.replies else (200, 0.0)
        if hold:
            server.release.wait(hold)
        self.send_response(status)
        if 300 <= status < 400:
            self.send_header("Location", "/SECRET-LOCATION?token=SECRET-QUERY")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        with self.server.lock:
            self.server.requests.append(("GET", self.path, b""))
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args: object) -> None:
        pass


class _PostServerTestCase(unittest.TestCase):
    """Run a scripted local receiver, with no channel env vars and no sleeps."""

    def setUp(self) -> None:
        self.server = _PostServer()
        self.port = self.server.server_address[1]
        threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True,
        ).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.addCleanup(self.server.release.set)
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for var in (_SLACK_ENV, _TEAMS_ENV, _GENERIC_ENV):
            os.environ.pop(var, None)
        os.environ["no_proxy"] = "*"
        # notify's time is the time module, so this records every retry delay.
        sleep = patch("agent_trace.notify.time.sleep")
        self.sleep = sleep.start()
        self.addCleanup(sleep.stop)
        self.url = f"http://127.0.0.1:{self.port}/SECRET-PATH?token=SECRET-QUERY"
        self.endpoint = f"http://127.0.0.1:{self.port}"


class TestPost(_PostServerTestCase):
    # Guards: unchanged from main.

    def test_returns_true_on_200(self):
        self.server.script((200, 0.0))
        result = _post(self.url, {"text": "hi"}, retries=1)
        self.assertTrue(result)
        self.assertEqual(
            self.server.requests,
            [("POST", "/SECRET-PATH?token=SECRET-QUERY", json.dumps({"text": "hi"}).encode())],
        )

    def test_returns_false_on_4xx(self):
        self.server.script((400, 0.0))
        with redirect_stderr(io.StringIO()):
            result = _post(self.url, {}, retries=1)
        self.assertFalse(result)

    def test_returns_false_on_repeated_failure(self):
        with redirect_stderr(io.StringIO()):
            result = _post(f"http://127.0.0.1:{_closed_port()}/hook", {}, retries=2)
        self.assertFalse(result)


class TestPostFailureReports(_PostServerTestCase):
    """A final delivery failure prints one line naming the channel, only
    scheme://host[:port], and the reason."""

    def _post(self, url: str | None = None, retries: int = 3) -> tuple[bool, str]:
        err = io.StringIO()
        with redirect_stderr(err):
            result = _post(self.url if url is None else url, {"text": "hi"}, retries=retries)
        for marker in _MARKERS:
            self.assertNotIn(marker, err.getvalue())
        return result, err.getvalue()

    def test_any_2xx_answer_succeeds(self):
        for status in (200, 201, 202, 203, 204):
            with self.subTest(status=status):
                self.server.script((status, 0.0))
                self.assertEqual(self._post(), (True, ""))
                self.assertEqual(self.server.methods(), ["POST"])
                self.sleep.assert_not_called()

    def test_3xx_and_4xx_answers_are_reported_at_once(self):
        for status in (301, 302, 303, 307, 308, 400, 403, 404, 410):
            with self.subTest(status=status):
                self.sleep.reset_mock()
                self.server.script((status, 0.0))
                hint = _NOT_FOLLOWED if status < 400 else ""
                self.assertEqual(self._post(), (
                    False,
                    f"[notify] Webhook POST to {self.endpoint} failed: "
                    f"HTTP Error {status}: {HTTPStatus(status).phrase}{hint}\n",
                ))
                # Not retried, and a redirect is not followed.
                self.assertEqual(self.server.methods(), ["POST"])
                self.sleep.assert_not_called()

    def test_5xx_answers_are_reported_after_the_retries(self):
        for status in (500, 502, 503):
            with self.subTest(status=status):
                self.sleep.reset_mock()
                self.server.script((status, 0.0), (status, 0.0), (status, 0.0))
                self.assertEqual(self._post(), (
                    False,
                    f"[notify] Webhook POST to {self.endpoint} failed after 3 attempts: "
                    f"HTTP Error {status}: {HTTPStatus(status).phrase}\n",
                ))
                self.assertEqual(self.server.methods(), ["POST"] * 3)
                self.assertEqual(self.sleep.call_args_list, [call(1.0), call(2.0)])

    def test_single_attempt_has_no_attempt_count(self):
        self.server.script((503, 0.0))
        self.assertEqual(self._post(retries=1), (
            False,
            f"[notify] Webhook POST to {self.endpoint} failed: "
            "HTTP Error 503: Service Unavailable\n",
        ))
        self.sleep.assert_not_called()

    def test_refused_connection_is_reported_after_the_retries(self):
        port = _closed_port()
        result, err = self._post(
            f"http://127.0.0.1:{port}/SECRET-PATH?token=SECRET-QUERY", retries=2,
        )
        self.assertFalse(result)
        self.assertRegex(
            err,
            rf"\A\[notify\] Webhook POST to http://127\.0\.0\.1:{port} failed after "
            r"2 attempts: \[Errno \d+\] Connection refused\n\Z",
        )
        self.assertEqual(self.sleep.call_args_list, [call(1.0)])

    def test_timeout_is_reported(self):
        self.server.script((200, 30.0))
        with patch("agent_trace.notify._POST_TIMEOUT", 0.2):
            self.assertEqual(self._post(retries=1), (
                False, f"[notify] Webhook POST to {self.endpoint} failed: TimeoutError\n",
            ))

    def test_unusable_urls_are_not_posted(self):
        cases = {
            "it must start with http:// or https://": (
                "hooks.example.com/services/T0/B0/SECRET-PATH?token=SECRET-QUERY",
                f"127.0.0.1:{self.port}/SECRET-PATH?token=SECRET-QUERY",
            ),
            "it must not contain a user name or password": (
                f"http://SECRET-USER:SECRET-PASS@127.0.0.1:{self.port}/SECRET-PATH",
            ),
            "it must not contain spaces or control characters": (
                f"http://127.0.0.1:{self.port}/SECRET PATH?token=SECRET-QUERY",
            ),
        }
        for reason, urls in cases.items():
            for url in urls:
                with self.subTest(url=url):
                    self.sleep.reset_mock()
                    self.server.script()
                    self.assertEqual(
                        self._post(url), (False, f"[notify] Webhook URL not used: {reason}\n"),
                    )
                    self.assertEqual(self.server.methods(), [])
                    self.sleep.assert_not_called()

    def test_retry_after_5xx_can_succeed(self):
        # Guard: unchanged from main.
        self.server.script((500, 0.0), (200, 0.0))
        self.assertEqual(self._post(), (True, ""))
        self.assertEqual(self.server.methods(), ["POST", "POST"])
        self.assertEqual(self.sleep.call_args_list, [call(1.0)])


class TestNotifyFailureLines(_PostServerTestCase):
    """notify() names the channel that failed."""

    def _notify(self) -> tuple[dict, str]:
        err = io.StringIO()
        with redirect_stderr(err):
            results = notify("t", "m")
        for marker in _MARKERS:
            self.assertNotIn(marker, err.getvalue())
        return results, err.getvalue()

    def test_slack_and_teams(self):
        os.environ[_SLACK_ENV] = self.url
        os.environ[_TEAMS_ENV] = self.url.replace("SECRET-PATH", "SECRET-TEAMS")
        self.server.script((404, 0.0), (404, 0.0))
        self.assertEqual(self._notify(), (
            {"slack": False, "teams": False},
            f"[notify] Slack POST to {self.endpoint} failed: HTTP Error 404: Not Found\n"
            f"[notify] Teams POST to {self.endpoint} failed: HTTP Error 404: Not Found\n",
        ))

    def test_generic_webhook(self):
        os.environ[_GENERIC_ENV] = self.url
        self.server.script((404, 0.0))
        self.assertEqual(self._notify(), (
            {"webhook": False},
            f"[notify] Webhook POST to {self.endpoint} failed: HTTP Error 404: Not Found\n",
        ))


class TestNotify(unittest.TestCase):
    def setUp(self):
        # Clear env vars before each test, and restore them after it so a
        # channel URL set here never reaches another test module.
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for var in (_SLACK_ENV, _TEAMS_ENV, _GENERIC_ENV):
            os.environ.pop(var, None)

    def test_no_channels_returns_empty(self):
        results = notify("title", "message")
        self.assertEqual(results, {})

    def test_slack_channel_used_when_env_set(self):
        os.environ[_SLACK_ENV] = "http://slack.example.com/hook"
        with patch("agent_trace.notify._post", return_value=True) as mock_post:
            results = notify("t", "m")
        self.assertIn("slack", results)
        mock_post.assert_called_once()
        payload = mock_post.call_args[0][1]
        self.assertIn("attachments", payload)

    def test_teams_channel_used_when_env_set(self):
        os.environ[_TEAMS_ENV] = "http://teams.example.com/hook"
        with patch("agent_trace.notify._post", return_value=True) as mock_post:
            results = notify("t", "m")
        self.assertIn("teams", results)
        payload = mock_post.call_args[0][1]
        self.assertEqual(payload["@type"], "MessageCard")

    def test_generic_fallback_when_no_provider(self):
        os.environ[_GENERIC_ENV] = "http://generic.example.com/hook"
        with patch("agent_trace.notify._post", return_value=True) as mock_post:
            results = notify("t", "m")
        self.assertIn("webhook", results)

    def test_generic_not_used_when_slack_set(self):
        os.environ[_SLACK_ENV] = "http://slack.example.com/hook"
        os.environ[_GENERIC_ENV] = "http://generic.example.com/hook"
        with patch("agent_trace.notify._post", return_value=True):
            results = notify("t", "m")
        self.assertIn("slack", results)
        self.assertNotIn("webhook", results)

    def test_explicit_url_overrides_env(self):
        os.environ[_SLACK_ENV] = "http://env-slack.example.com"
        with patch("agent_trace.notify._post", return_value=True) as mock_post:
            notify("t", "m", slack_url="http://explicit-slack.example.com")
        called_url = mock_post.call_args[0][0]
        self.assertEqual(called_url, "http://explicit-slack.example.com")


class TestConvenienceWrappers(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for var in (_SLACK_ENV, _TEAMS_ENV, _GENERIC_ENV):
            os.environ.pop(var, None)

    def test_notify_violation_returns_empty_when_no_channels(self):
        result = notify_violation("no-network", "Bash")
        self.assertEqual(result, {})

    def test_notify_budget_alert_message_contains_pct(self):
        os.environ[_GENERIC_ENV] = "http://example.com"
        with patch("agent_trace.notify._post", return_value=True) as mock_post:
            notify_budget_alert("my-agent", spent=4.0, budget=5.0)
        payload = mock_post.call_args[0][1]
        self.assertIn("80%", payload["message"])

    def test_notify_session_end_includes_tool_calls(self):
        os.environ[_GENERIC_ENV] = "http://example.com"
        with patch("agent_trace.notify._post", return_value=True) as mock_post:
            notify_session_end("agent", "sess123", cost=0.05, tool_calls=7, duration_ms=3000)
        payload = mock_post.call_args[0][1]
        self.assertIn("7", payload["message"])


if __name__ == "__main__":
    unittest.main()
