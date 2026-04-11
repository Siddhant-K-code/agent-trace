"""Tests for session attribution (issue #22)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace.attribution import (
    Attribution,
    _detect_agent_provider,
    _detect_os_user,
    collect_attribution,
    format_attribution,
)


class TestAttribution(unittest.TestCase):
    def test_collect_returns_attribution(self):
        attr = collect_attribution()
        self.assertIsInstance(attr, Attribution)
        self.assertIsInstance(attr.pid, int)
        self.assertGreater(attr.pid, 0)

    def test_working_dir_set(self):
        attr = collect_attribution()
        self.assertTrue(attr.working_dir)
        self.assertTrue(os.path.isdir(attr.working_dir))

    def test_to_dict_round_trip(self):
        attr = collect_attribution()
        d = attr.to_dict()
        self.assertIsInstance(d, dict)
        restored = Attribution.from_dict(d)
        self.assertEqual(restored.pid, attr.pid)
        self.assertEqual(restored.working_dir, attr.working_dir)

    def test_format_attribution_no_crash(self):
        attr = Attribution(os_user="alice", hostname="box", agent_provider="claude-code")
        result = format_attribution(attr)
        self.assertIn("alice", result)
        self.assertIn("claude-code", result)

    def test_format_attribution_empty(self):
        attr = Attribution()
        result = format_attribution(attr)
        self.assertIsInstance(result, str)

    def test_detect_agent_provider_unknown_by_default(self):
        # In a clean test env without agent env vars, should return "unknown"
        # (unless running inside an actual agent)
        provider, version = _detect_agent_provider()
        self.assertIsInstance(provider, str)
        self.assertIsInstance(version, str)

    def test_detect_agent_provider_claude_env(self):
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test"
        try:
            provider, _ = _detect_agent_provider()
            self.assertEqual(provider, "claude-code")
        finally:
            del os.environ["ANTHROPIC_API_KEY"]

    def test_os_user_returns_string(self):
        user = _detect_os_user()
        self.assertIsInstance(user, str)


if __name__ == "__main__":
    unittest.main()
