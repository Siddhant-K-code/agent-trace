"""Tests for Slack/Teams/webhook notifications (issue #138)."""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

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


class TestPost(unittest.TestCase):
    def test_returns_true_on_200(self):
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.status = 200
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = _post("http://example.com/hook", {"text": "hi"}, retries=1)
        self.assertTrue(result)

    def test_returns_false_on_4xx(self):
        import urllib.error
        with patch("urllib.request.urlopen",
                   side_effect=urllib.error.HTTPError("url", 400, "Bad Request", {}, None)):
            result = _post("http://example.com/hook", {}, retries=1)
        self.assertFalse(result)

    def test_returns_false_on_repeated_failure(self):
        with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            with patch("time.sleep"):  # don't actually sleep in tests
                result = _post("http://example.com/hook", {}, retries=2)
        self.assertFalse(result)


class TestNotify(unittest.TestCase):
    def setUp(self):
        # Clear env vars before each test
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
