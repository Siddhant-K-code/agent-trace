"""Tests for live PII masking (issue #20)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace.masking import MaskingConfig, mask_data, mask_event_data, _mask_string


class TestMaskString(unittest.TestCase):
    def setUp(self):
        self.cfg = MaskingConfig.default()

    def test_masks_email(self):
        result = _mask_string("Contact alice@example.com for help", self.cfg)
        self.assertNotIn("alice@example.com", result)
        self.assertIn("[MASKED]", result)

    def test_masks_phone_us(self):
        result = _mask_string("Call 555-867-5309 now", self.cfg)
        self.assertNotIn("555-867-5309", result)

    def test_masks_credit_card(self):
        result = _mask_string("Card: 4111 1111 1111 1111", self.cfg)
        self.assertNotIn("4111 1111 1111 1111", result)

    def test_masks_ssn(self):
        result = _mask_string("SSN: 123-45-6789", self.cfg)
        self.assertNotIn("123-45-6789", result)

    def test_safe_string_unchanged(self):
        result = _mask_string("hello world", self.cfg)
        self.assertEqual(result, "hello world")

    def test_custom_pattern(self):
        cfg = MaskingConfig(custom_patterns=[r"\bACCT-\d+\b"])
        result = _mask_string("Account ACCT-12345 is active", cfg)
        self.assertNotIn("ACCT-12345", result)

    def test_public_ip_off_by_default(self):
        cfg = MaskingConfig(mask_public_ips=False)
        result = _mask_string("Server at 8.8.8.8", cfg)
        self.assertIn("8.8.8.8", result)

    def test_public_ip_masked_when_enabled(self):
        cfg = MaskingConfig(mask_public_ips=True)
        result = _mask_string("Server at 8.8.8.8", cfg)
        self.assertNotIn("8.8.8.8", result)


class TestMaskData(unittest.TestCase):
    def setUp(self):
        self.cfg = MaskingConfig.default()

    def test_masks_email_in_dict_value(self):
        data = {"message": "Send to bob@test.org please"}
        result = mask_data(data, self.cfg)
        self.assertNotIn("bob@test.org", result["message"])

    def test_masks_sensitive_key(self):
        data = {"email": "alice@example.com", "name": "Alice"}
        result = mask_data(data, self.cfg)
        self.assertEqual(result["email"], "[MASKED]")
        self.assertEqual(result["name"], "Alice")

    def test_masks_nested(self):
        data = {"user": {"contact": "call 555-123-4567"}}
        result = mask_data(data, self.cfg)
        self.assertNotIn("555-123-4567", result["user"]["contact"])

    def test_masks_list_items(self):
        data = {"notes": ["email: test@example.com", "safe note"]}
        result = mask_data(data, self.cfg)
        self.assertNotIn("test@example.com", result["notes"][0])
        self.assertEqual(result["notes"][1], "safe note")

    def test_non_string_unchanged(self):
        data = {"count": 42, "flag": True}
        result = mask_data(data, self.cfg)
        self.assertEqual(result["count"], 42)
        self.assertEqual(result["flag"], True)


class TestMaskEventData(unittest.TestCase):
    def test_combined_secrets_and_pii(self):
        data = {
            "api_key": "sk-abc123def456ghi789jkl012mno345pqr678",
            "message": "Contact alice@example.com",
        }
        result = mask_event_data(data)
        # Secret redacted
        self.assertNotIn("sk-abc", str(result.get("api_key", "")))
        # PII masked
        self.assertNotIn("alice@example.com", result.get("message", ""))

    def test_no_redact_secrets_flag(self):
        data = {"message": "email: test@example.com"}
        result = mask_event_data(data, redact_secrets=False)
        self.assertNotIn("test@example.com", result["message"])

    def test_none_config_uses_defaults(self):
        data = {"note": "phone: 555-123-4567"}
        result = mask_event_data(data, config=None)
        self.assertNotIn("555-123-4567", result["note"])


if __name__ == "__main__":
    unittest.main()
