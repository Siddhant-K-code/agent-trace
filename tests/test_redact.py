"""Tests for secret redaction."""

import base64
import json
import os
import random
import re
import sys
import time
import tracemalloc
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_trace.redact import (
    SECRET_PATTERNS,
    _contains_secret,
    redact_data,
    redact_data_with_status,
    redact_value,
    redaction_marker,
)

KEY_MARKER = redaction_marker("private-key")
SUFFIX = "." * 3 + " (truncated)"


class TestRedactValue(unittest.TestCase):
    def test_openai_key(self):
        result = redact_value("sk-abc123def456ghi789jkl012mno345pqr678")
        self.assertEqual(result, redaction_marker("openai-key"))

    def test_github_token(self):
        result = redact_value("ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijkl")
        self.assertEqual(result, redaction_marker("github-token"))

    def test_github_pat(self):
        result = redact_value("github_pat_ABCDEFGHIJKLMNOPQRSTUV_1234567890")
        self.assertEqual(result, redaction_marker("github-token"))

    def test_aws_key(self):
        result = redact_value("AKIAIOSFODNN7EXAMPLE")
        self.assertEqual(result, redaction_marker("aws-access-key"))

    def test_bearer_token(self):
        result = redact_value("Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test")
        self.assertIn(redaction_marker("bearer-token"), result)

    def test_connection_string(self):
        result = redact_value("postgres://user:pass@host:5432/db")
        self.assertEqual(result, redaction_marker("connection-string"))

    def test_jwt(self):
        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        result = redact_value(jwt)
        self.assertEqual(result, redaction_marker("jwt"))

    def test_anthropic_key(self):
        result = redact_value("sk-ant-api03-abcdefghijklmnopqrstuvwxyz")
        self.assertEqual(result, redaction_marker("anthropic-key"))

    def test_slack_token(self):
        # Use a clearly fake token that still matches the xox[bpras]-... pattern
        result = redact_value("xoxb-fake-test-token-value")
        self.assertEqual(result, redaction_marker("slack-token"))

    def test_private_key(self):
        key = "-----BEGIN PRIVATE KEY-----\nabc123\n-----END PRIVATE KEY-----"
        result = redact_value(key)
        self.assertEqual(result, redaction_marker("private-key"))

    def test_basic_auth_url(self):
        result = redact_value("https://alice:secret@example.com/path")
        self.assertEqual(result, f"https://{redaction_marker('basic-auth')}@example.com/path")

    def test_safe_string_unchanged(self):
        result = redact_value("hello world")
        self.assertEqual(result, "hello world")

    def test_file_path_unchanged(self):
        result = redact_value("/usr/local/bin/python3")
        self.assertEqual(result, "/usr/local/bin/python3")

    def test_inline_redaction(self):
        result = redact_value("Authorization: Bearer sk-abc123def456ghi789jkl012mno345pqr678")
        self.assertIn("[REDACTED:", result)
        self.assertNotIn("sk-abc", result)


class TestRedactData(unittest.TestCase):
    def test_redact_by_key_name(self):
        data = {"username": "alice", "password": "secret123"}
        result = redact_data(data)
        self.assertEqual(result["username"], "alice")
        self.assertEqual(result["password"], redaction_marker("sensitive"))

    def test_redact_api_key(self):
        data = {"api_key": "sk-abc123def456ghi789jkl012mno345pqr678"}
        result = redact_data(data)
        self.assertEqual(result["api_key"], redaction_marker("sensitive"))

    def test_redact_token_key(self):
        data = {"token": "some-secret-value", "name": "test"}
        result = redact_data(data)
        self.assertEqual(result["token"], redaction_marker("sensitive"))
        self.assertEqual(result["name"], "test")

    def test_redact_nested_dict(self):
        data = {
            "config": {
                "database_url": "postgres://user:pass@host/db",
                "name": "myapp",
            }
        }
        result = redact_data(data)
        self.assertEqual(result["config"]["database_url"], redaction_marker("sensitive"))
        self.assertEqual(result["config"]["name"], "myapp")

    def test_redact_list_values(self):
        data = {
            "headers": [
                {"name": "Authorization", "value": "Bearer sk-abc123def456ghi789jkl012mno345pqr678"},
                {"name": "Content-Type", "value": "application/json"},
            ]
        }
        result = redact_data(data)
        self.assertEqual(result["headers"][1]["value"], "application/json")

    def test_redact_value_by_pattern(self):
        data = {
            "arguments": {
                "url": "postgres://admin:hunter2@db.example.com:5432/prod",
                "query": "SELECT * FROM users",
            }
        }
        result = redact_data(data)
        self.assertEqual(result["arguments"]["url"], redaction_marker("connection-string"))
        self.assertEqual(result["arguments"]["query"], "SELECT * FROM users")

    def test_non_string_values_unchanged(self):
        data = {"count": 42, "enabled": True, "ratio": 3.14}
        result = redact_data(data)
        self.assertEqual(result, data)

    def test_empty_data(self):
        self.assertEqual(redact_data({}), {})
        self.assertEqual(redact_data([]), [])
        self.assertEqual(redact_data("hello"), "hello")

    def test_sensitive_key_case_insensitive(self):
        data = {"Password": "secret", "API_KEY": "key123"}
        # keys are checked lowercase
        result = redact_data(data)
        # "Password" lowered is "password" which is in SENSITIVE_KEYS
        self.assertEqual(result["Password"], redaction_marker("sensitive"))

    def test_aws_env_var_key_names(self):
        data = {"AWS_SECRET_ACCESS_KEY": "secret", "AWS_SESSION_TOKEN": "token"}
        result = redact_data(data)
        self.assertEqual(result["AWS_SECRET_ACCESS_KEY"], redaction_marker("sensitive"))
        self.assertEqual(result["AWS_SESSION_TOKEN"], redaction_marker("sensitive"))

    def test_tool_call_with_secrets(self):
        """Simulate a real tool call event data with secrets."""
        data = {
            "tool_name": "http_request",
            "arguments": {
                "url": "https://api.example.com/data",
                "headers": {
                    "Authorization": "Bearer sk-abc123def456ghi789jkl012mno345pqr678",
                    "Content-Type": "application/json",
                },
                "api_key": "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijkl",
            },
        }
        result = redact_data(data)
        self.assertEqual(result["tool_name"], "http_request")
        self.assertEqual(result["arguments"]["url"], "https://api.example.com/data")
        self.assertIn("[REDACTED:", result["arguments"]["headers"]["Authorization"])
        self.assertEqual(result["arguments"]["api_key"], redaction_marker("sensitive"))

    def test_redact_data_with_status(self):
        data = {"text": "hello sk-abc123def456ghi789jkl012mno345pqr678"}
        result, changed = redact_data_with_status(data)
        self.assertTrue(changed)
        self.assertNotIn("sk-abc123", str(result))


def fake_key(label="RSA PRIVATE KEY", seed=1, size=1190):
    """Return a PEM block of random base64. The RSA default is 1675 characters."""
    text = base64.b64encode(random.Random(seed).randbytes(size)).decode()
    lines = [text[i : i + 64] for i in range(0, len(text), 64)]
    return "\n".join([f"-----BEGIN {label}-----", *lines, f"-----END {label}-----", ""])


def fake_pgp_key(seed=2):
    """Return an armored PGP private key block with a header and a checksum."""
    text = base64.b64encode(random.Random(seed).randbytes(900)).decode()
    lines = [text[i : i + 64] for i in range(0, len(text), 64)]
    return "\n".join([
        "-----BEGIN PGP PRIVATE KEY BLOCK-----",
        "Comment: Alice <alice@example.com>",
        "",
        *lines,
        "=Nm2X",
        "-----END PGP PRIVATE KEY BLOCK-----",
        "",
    ])


KEY_LABELS = (
    "PRIVATE KEY",
    "RSA PRIVATE KEY",
    "EC PRIVATE KEY",
    "DSA PRIVATE KEY",
    "OPENSSH PRIVATE KEY",
    "ENCRYPTED PRIVATE KEY",
    "ANY PRIVATE KEY",
    "A B C D E F G H PRIVATE KEY",
)


class TestPrivateKeyBlocks(unittest.TestCase):
    def keys(self):
        for label in KEY_LABELS:
            yield label, fake_key(label)
        yield "PGP PRIVATE KEY BLOCK", fake_pgp_key()

    def test_complete_block_for_each_label(self):
        for label, key in self.keys():
            with self.subTest(label):
                text = "$ cat id\n" + key + "$ ls"
                self.assertEqual(redact_value(text), "$ cat id\n" + KEY_MARKER + "\n$ ls")

    def test_block_cut_before_its_end_line(self):
        for label, key in self.keys():
            with self.subTest(label):
                cut = key[: len(key) // 2].rstrip("\n")
                self.assertEqual(redact_value("$ cat id\n" + cut), "$ cat id\n" + KEY_MARKER)
                self.assertEqual(redact_value(cut + SUFFIX), KEY_MARKER + SUFFIX)

    def test_end_label_may_differ_from_begin_label(self):
        for begin, end in (
            ("RSA PRIVATE KEY", "EC PRIVATE KEY"),
            ("PGP PRIVATE KEY BLOCK", "PRIVATE KEY"),
            ("ENCRYPTED PRIVATE KEY", "PGP PRIVATE KEY BLOCK"),
        ):
            with self.subTest(begin=begin, end=end):
                text = f"-----BEGIN {begin}-----\nMIIE\n-----END {end}-----\nafter"
                self.assertEqual(redact_value(text), KEY_MARKER + "\nafter")

    def test_end_certificate_does_not_close_a_key(self):
        text = (
            "-----BEGIN PRIVATE KEY-----\nMIIE\n-----END CERTIFICATE-----\n"
            "QQ==\n-----END PRIVATE KEY-----\nafter"
        )
        self.assertEqual(redact_value(text), KEY_MARKER + "\nafter")
        text = "-----BEGIN PRIVATE KEY-----\nMIIE\n-----END CERTIFICATE-----\nafter"
        self.assertEqual(
            redact_value(text), KEY_MARKER + "\n-----END CERTIFICATE-----\nafter"
        )

    def test_nested_begin_gives_two_markers(self):
        text = (
            "-----BEGIN RSA PRIVATE KEY-----\nAAAA\n"
            "-----BEGIN RSA PRIVATE KEY-----\nBBBB\n-----END RSA PRIVATE KEY-----\nafter"
        )
        self.assertEqual(redact_value(text), KEY_MARKER + "\n" + KEY_MARKER + "\nafter")

    def test_text_before_a_nested_begin_stays_visible(self):
        text = (
            "-----BEGIN RSA PRIVATE KEY-----\nAAAA\n$ echo done\n"
            "-----BEGIN RSA PRIVATE KEY-----\nBBBB\n-----END RSA PRIVATE KEY-----"
        )
        self.assertEqual(redact_value(text), KEY_MARKER + "\n$ echo done\n" + KEY_MARKER)


class TestCutPrivateKeyBody(unittest.TestCase):
    """A key with no END line is redacted from BEGIN to the end of its body."""

    BEGIN = "-----BEGIN PRIVATE KEY-----"

    def test_lone_begin_line(self):
        self.assertEqual(redact_value(self.BEGIN), KEY_MARKER)
        self.assertEqual(redact_value("x " + self.BEGIN + "\n"), "x " + KEY_MARKER + "\n")

    def test_each_line_break_form(self):
        # A real break, JSON-escaped breaks, and breaks escaped twice.
        for brk in ("\n", "\r\n", "\\n", "\\r\\n", "\\\\n", "\\\\r\\\\n"):
            with self.subTest(brk=brk):
                text = self.BEGIN + brk + "MIIE" + brk + "QQ==" + brk + "$ next"
                self.assertEqual(redact_value(text), KEY_MARKER + brk + "$ next")

    def test_single_space_separator(self):
        # echo $KEY prints the key on one line with single spaces.
        key = fake_key().replace("\n", " ")
        self.assertEqual(redact_value("$ echo " + key), "$ echo " + KEY_MARKER + " ")
        self.assertEqual(redact_value("$ echo " + key[:1000]), "$ echo " + KEY_MARKER)
        self.assertEqual(redact_value(self.BEGIN + " MIIE QQ==  next"), KEY_MARKER + "  next")

    def test_header_lines(self):
        for header in (
            "Proc-Type: 4,ENCRYPTED",
            "DEK-Info: AES-256-CBC,3F17F5316E2BAC89",
            "Version: GnuPG v2",
            "Comment: alice@example.com",
        ):
            with self.subTest(header):
                text = self.BEGIN + "\n" + header + "\n\nMIIE\nQQ==\n$ next"
                self.assertEqual(redact_value(text), KEY_MARKER + "\n$ next")

    def test_header_keeps_a_backslash_that_is_not_a_line_break(self):
        text = self.BEGIN + "\\nComment: C:\\keys\\id_rsa\\nMIIE\\n$ next"
        self.assertEqual(redact_value(text), KEY_MARKER + "\\n$ next")

    def test_blank_lines_and_indentation(self):
        text = "  " + self.BEGIN + "\n\n  MIIE\n\t QQ==\n\n  $ next"
        self.assertEqual(redact_value(text), "  " + KEY_MARKER + "\n\n  $ next")

    def test_long_runs_of_blank_lines(self):
        for brk in ("\n", "\\n"):
            with self.subTest(brk=brk):
                text = self.BEGIN + brk + "MIIE" + brk * 100 + "QQ==" + brk + "$ next"
                self.assertEqual(redact_value(text), KEY_MARKER + brk + "$ next")

    def test_long_body(self):
        text = self.BEGIN + "\n" + "\n".join(["A" * 64] * 300) + "\n$ next"
        self.assertEqual(redact_value(text), KEY_MARKER + "\n$ next")

    def test_body_ends_at_the_first_other_character(self):
        text = self.BEGIN + "\nMIIE\nthe rest of the key was cut, see notes"
        self.assertEqual(redact_value(text), KEY_MARKER + ", see notes")

    def test_key_in_json_text(self):
        once = json.dumps({"private_key": fake_key("PRIVATE KEY")[:900]})
        self.assertEqual(redact_value(once), '{"private_key": "' + KEY_MARKER + '"}')
        twice = json.dumps(once)
        self.assertEqual(
            redact_value(twice), '"{\\"private_key\\": \\"' + KEY_MARKER + '\\"}"'
        )


class TestTokenPrefixes(unittest.TestCase):
    def test_openai_project_service_admin_and_none_keys(self):
        marker = redaction_marker("openai-key")
        for prefix in ("sk-proj-", "sk-svcacct-", "sk-admin-", "sk-None-"):
            with self.subTest(prefix):
                text = "OPENAI_API_KEY=" + prefix + "Ab_9-" * 8 + "\n"
                self.assertEqual(redact_value(text), "OPENAI_API_KEY=" + marker + "\n")

    def test_openai_prefix_inside_a_word_is_kept(self):
        for text in ("risk-admin-panel-configuration-file", "disk-proj-backup-volume-snapshots"):
            with self.subTest(text):
                self.assertEqual(redact_value(text), text)

    def test_github_token_prefixes(self):
        marker = redaction_marker("github-token")
        for prefix in ("ghp_", "gho_", "ghu_", "ghs_", "ghr_"):
            with self.subTest(prefix):
                text = "GH_TOKEN=" + prefix + "A1b2" * 9
                self.assertEqual(redact_value(text), "GH_TOKEN=" + marker)

    def test_anthropic_key_with_an_underscore(self):
        marker = redaction_marker("anthropic-key")
        early = "sk-ant-api03_" + "Ab9" * 10
        late = "sk-ant-api03-" + "Ab9" * 10 + "_" + "Cd8" * 6
        for name, token in (("early", early), ("late", late)):
            with self.subTest(name):
                self.assertEqual(redact_value("key: " + token + "."), "key: " + marker + ".")

    def test_anthropic_key_inside_a_word(self):
        token = "sk-ant-api03-" + "Ab9" * 10
        self.assertEqual(redact_value("x" + token), "x" + redaction_marker("anthropic-key"))


# Fake keys, built in parts so no full key appears in the source.
STRIPE_BODY = "A1b2" * 6
GOOGLE_KEY = "AI" + "za" + "Sy" + "B-_c" * 8 + "Q"
GITLAB_BODY = "Ab_-" * 7
NPM_KEY = "npm" + "_" + "A1b2" * 9
HF_KEY = "hf" + "_" + "A1b2" * 8


class TestProviderKeys(unittest.TestCase):
    def assert_redacted(self, text, expected):
        self.assertEqual(redact_value(text), expected)
        self.assertTrue(_contains_secret(text))

    def assert_kept(self, text):
        self.assertEqual(redact_value(text), text)
        self.assertFalse(_contains_secret(text))

    def test_stripe_secret_and_restricted_keys(self):
        marker = redaction_marker("stripe-key")
        for prefix in ("sk_", "rk_"):
            for mode in ("live_", "test_"):
                for body in (STRIPE_BODY, "A1b2" * 25):
                    with self.subTest(prefix + mode, length=len(body)):
                        text = "STRIPE_SECRET_KEY=" + prefix + mode + body + "\n"
                        self.assert_redacted(text, "STRIPE_SECRET_KEY=" + marker + "\n")

    def test_stripe_publishable_keys_are_kept(self):
        for mode in ("live_", "test_"):
            with self.subTest(mode):
                self.assert_kept("STRIPE_PUBLISHABLE_KEY=" + "pk_" + mode + STRIPE_BODY)

    def test_stripe_organization_keys(self):
        marker = redaction_marker("stripe-key")
        for mode in ("", "live_", "test_"):
            with self.subTest(mode=mode):
                text = "STRIPE_ORG_KEY=" + "sk_" + "org_" + mode + STRIPE_BODY + "\n"
                self.assert_redacted(text, "STRIPE_ORG_KEY=" + marker + "\n")

    def test_stripe_organization_near_misses_are_kept(self):
        cases = (
            "sk_" + "org_settings_page_controller_name",
            "sk_" + "org_" + "A1b2" * 5 + "Q23",
            "sk_" + "org_" + "live_" + "A1b2" * 5 + "Q23",
            "ask_" + "org_" + STRIPE_BODY,
        )
        for text in cases:
            with self.subTest(text):
                self.assert_kept(text)

    def test_stripe_webhook_secret(self):
        text = "endpoint secret: " + "wh" + "sec_" + "A1b2" * 8
        self.assert_redacted(text, "endpoint secret: " + redaction_marker("stripe-webhook-secret"))

    def test_stripe_webhook_secret_in_base64_leaves_no_tail(self):
        marker = redaction_marker("stripe-webhook-secret")
        bodies = {
            "padded tail": "A1b2" * 8 + "+/aB=",
            "two pad characters": "A1b2" * 8 + "+/aB==",
            "early plus and slash": "A1b2+/" * 6,
        }
        for name, body in bodies.items():
            with self.subTest(name):
                text = "secret: " + "wh" + "sec_" + body + " end"
                self.assert_redacted(text, "secret: " + marker + " end")

    def test_google_api_key(self):
        self.assertEqual(len(GOOGLE_KEY), 39)
        marker = redaction_marker("google-api-key")
        self.assert_redacted("https://maps.example/api?key=" + GOOGLE_KEY + "&q=1",
                             "https://maps.example/api?key=" + marker + "&q=1")

    def test_gitlab_token_formats(self):
        marker = redaction_marker("gitlab-token")
        prefix = "gl" + "pat-"
        cases = {
            "legacy": prefix + "Ab_-" * 5,
            "routable": prefix + GITLAB_BODY + ".x1" + "abcdefg",
            "versioned": prefix + GITLAB_BODY + ".01.x1" + "abcdefg",
        }
        for name, token in cases.items():
            with self.subTest(name):
                self.assert_redacted("token " + token + ".", "token " + marker + ".")
        with self.subTest("instance prefix"):
            self.assert_redacted("acme-" + cases["legacy"], "acme-" + marker)

    def test_gitlab_token_types(self):
        marker = redaction_marker("gitlab-token")
        routable = GITLAB_BODY + ".01.x1" + "abcdefg"
        jwt = "eyJ" + "a1B2c3D4e5" + ".eyJ" + "f6G7h8I9j0" + "." + "K1l2M3n4O5p6Q7r8S9t0"
        cases = {
            "pipeline trigger": ("ptt", "A1b2" * 10),
            "deploy": ("dt", "Ab_-" * 5),
            "routable deploy": ("dt", routable),
            "runner": ("rt", "t1_" + "Ab_-" * 5),
            "routable runner": ("rt", routable),
            "runner, rtr": ("rtr", "Ab_-" * 5),
            "runner controller": ("rct", "Ab_-" * 5),
            "CI/CD job": ("cbt", "64_" + "Ab_-" * 5),
            "CI/CD job JWT": ("cbt", jwt),
            "feed": ("ft", "Ab_-" * 5),
            "path-dependent feed": ("ft", "0123456789abcdef" * 4 + "-42"),
            "incoming email": ("imt", "a1b2c3" * 4),
            "Kubernetes agent": ("agent", "Ab_-" * 12 + "Q7"),
            "routable Kubernetes agent": ("agent", routable),
            "OAuth application secret": ("oas", "Ab_-" * 16),
            "SCIM": ("soat", "Ab_-" * 5),
            "feature flags client": ("ffct", "Ab_-" * 5),
            "workspace": ("wt", routable),
        }
        for name, (kind, body) in cases.items():
            with self.subTest(name):
                token = "gl" + kind + "-" + body
                self.assert_redacted("token " + token + ".", "token " + marker + ".")
        with self.subTest("instance prefix"):
            self.assert_redacted("acme-" + "gl" + "rt-" + GITLAB_BODY, "acme-" + marker)

    def test_gitlab_routable_token_before_a_hyphen(self):
        token = "gl" + "pat-" + GITLAB_BODY + ".01.x1" + "abcdefg"
        self.assert_redacted(token + "-backup", redaction_marker("gitlab-token") + "-backup")

    def test_gitlab_near_misses_are_kept(self):
        cases = [
            "x" + "gl" + "rt-" + GITLAB_BODY,
            "_" + "gl" + "ft-" + GITLAB_BODY,
            "9" + "gl" + "dt-" + GITLAB_BODY,
            "gl" + "ob-pattern-matching-for-the-file-tree",
            "Runner authentication tokens have the prefix " + "gl" + "rt-.",
        ]
        for kind in ("ptt", "dt", "rt", "rct", "cbt", "ft", "imt", "agent", "oas",
                     "soat", "ffct", "wt"):
            cases.append("gl" + kind + "-" + "Ab_-" * 4 + "Q23")
        for text in cases:
            with self.subTest(text):
                self.assert_kept(text)

    def test_long_prefixes_match_after_a_hyphen(self):
        cases = {
            "header": ("x-", "sk_" + "live_" + STRIPE_BODY, "stripe-key"),
            "flag": ("--", "sk_" + "test_" + STRIPE_BODY, "stripe-key"),
            "removed diff line": ("-", "rk_" + "live_" + STRIPE_BODY, "stripe-key"),
            "joined to a word": ("key-", "sk_" + "live_" + STRIPE_BODY, "stripe-key"),
            "organization key": ("x-", "sk_" + "org_" + STRIPE_BODY, "stripe-key"),
            "webhook secret": ("-", "wh" + "sec_" + "A1b2" * 8, "stripe-webhook-secret"),
        }
        for name, (lead, token, kind) in cases.items():
            with self.subTest(name):
                self.assert_redacted(lead + token, lead + redaction_marker(kind))

    def test_short_prefixes_after_a_hyphen_are_kept(self):
        # A "-" can sit inside a base64url body such as a JWT, so the short
        # prefixes still need a separator before them.
        cases = (
            "-" + HF_KEY,
            "Llama-2-7b-" + HF_KEY,
            "x-" + NPM_KEY,
            "-" + GOOGLE_KEY,
            "x-risk_" + "live_" + STRIPE_BODY,
            "x-disk_" + "test_" + STRIPE_BODY,
            "risk-admin-panel-configuration-file",
        )
        for text in cases:
            with self.subTest(text):
                self.assert_kept(text)

    def test_npm_token(self):
        self.assert_redacted("//registry.npmjs.org/:_authToken=" + NPM_KEY,
                             "//registry.npmjs.org/:_authToken=" + redaction_marker("npm-token"))

    def test_huggingface_token(self):
        marker = redaction_marker("huggingface-token")
        self.assert_redacted("HF_TOKEN=" + HF_KEY, "HF_TOKEN=" + marker)
        self.assert_redacted("Bearer " + HF_KEY, "Bearer " + marker)

    def test_prefix_inside_a_longer_identifier_is_kept(self):
        cases = (
            "risk_" + "live_" + STRIPE_BODY,
            "task_" + "test_" + STRIPE_BODY,
            "disk_" + "live_" + STRIPE_BODY,
            "work_" + "test_" + STRIPE_BODY,
            "x" + "wh" + "sec_" + "A1b2" * 8,
            "x" + GOOGLE_KEY,
            "x" + "gl" + "pat-" + "Ab_-" * 5,
            "x" + NPM_KEY,
            "a" + HF_KEY,
        )
        for text in cases:
            with self.subTest(text):
                self.assert_kept(text)

    def test_wrong_lengths_are_kept(self):
        cases = {
            "stripe 23": "sk_" + "live_" + "A1b2" * 5 + "Q23",
            "webhook 31": "wh" + "sec_" + "A1b2" * 7 + "Q2Z",
            "base64 webhook 31": "wh" + "sec_" + "A1b2+/" * 5 + "Q",
            "google 34": GOOGLE_KEY[:-1],
            "google 36": GOOGLE_KEY + "Z",
            "gitlab 19": "gl" + "pat-" + "Ab_-" * 4 + "Q23",
            "npm 35": NPM_KEY[:-1],
            "npm 37": NPM_KEY + "Z",
            "huggingface 29": HF_KEY[:-3],
        }
        for name, text in cases.items():
            with self.subTest(name):
                self.assert_kept(text)

    def test_generic_patterns_do_not_cut_a_provider_key(self):
        # sk-, key- and token- have no left boundary and would match inside
        # these bodies if they ran first.
        cases = {
            "google-api-key": "AI" + "za" + "Sy" + "key-" + "A1b2" * 7 + "Q",
            "gitlab-token": "gl" + "pat-" + "sk-" + "A1b2" * 6,
        }
        for kind, token in cases.items():
            with self.subTest(kind):
                self.assertEqual(redact_value(token), redaction_marker(kind))

    def test_redact_data_by_pattern_under_any_key(self):
        stripe = "sk_" + "live_" + STRIPE_BODY
        data = {"output": [stripe, {"log": "using " + GOOGLE_KEY}], "count": 2}
        self.assertEqual(redact_data(data), {
            "output": [
                redaction_marker("stripe-key"),
                {"log": "using " + redaction_marker("google-api-key")},
            ],
            "count": 2,
        })


# Main's patterns before this change. On the inputs main handled, the new
# patterns must give the same result.
MAIN_PRIVATE_KEY = re.compile(
    r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"
    r"[\s\S]*?"
    r"-----END (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"
)
MAIN_KEY_BEGIN = re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----")
MAIN_KEY_END = re.compile(r"-----END (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----")
MAIN_KEY_LABELS = {
    "PRIVATE KEY",
    "RSA PRIVATE KEY",
    "EC PRIVATE KEY",
    "DSA PRIVATE KEY",
    "OPENSSH PRIVATE KEY",
}
ANY_KEY_LABEL = re.compile(
    r"(?=-----(?:BEGIN|END) (PGP PRIVATE KEY BLOCK|(?:[A-Z0-9]+ ){0,8}PRIVATE KEY)-----)"
)
MAIN_JWT = re.compile(r"eyJ[a-zA-Z0-9_-]{10,}\.eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}")
MAIN_BASIC_AUTH = re.compile(r"\b([a-zA-Z][a-zA-Z0-9+.-]*://)([^/\s:@]+:[^/\s@]+)@")
MAIN_TOKENS = {
    "anthropic-key": [r"sk-ant-[a-zA-Z0-9\-]{20,}"],
    "openai-key": [r"sk-[a-zA-Z0-9]{20,}"],
    "github-token": [r"gh[ps]_[a-zA-Z0-9]{36,}", r"github_pat_[a-zA-Z0-9_]{22,}"],
}
HEX_TOKEN = re.compile(r"[0-9a-f]{40,}")
GLUED_GITHUB_TOKENS = re.compile(r"gh[pousr]_[A-Za-z0-9]*gh[pousr]_")

# The body of a key with no END line, as one regex. It needs memory for each
# line, so it only runs on short strings here.
_BREAK = r"(?:\r?\n|\\r\\n|\\n|\\\\r\\\\n|\\\\n)"
_HEADER = (
    r"(?:Proc-Type|DEK-Info|Version|Comment):"
    r"[^\r\n\\]*(?:\\(?!n|r\\n|\\n|\\r\\\\n)[^\r\n\\]*)*"
)
_BASE64 = r"[A-Za-z0-9+/=]"
KEY_BODY = re.compile(
    _BASE64 + r"*(?:(?:" + _BREAK + r"[ \t]*)+(?:" + _HEADER + "|" + _BASE64 + r"+)"
    r"| " + _BASE64 + r"+)*"
)


def in_main_domain(text):
    """Main's labels only, and each BEGIN has its END before another BEGIN."""
    if any(m.group(1) not in MAIN_KEY_LABELS for m in ANY_KEY_LABEL.finditer(text)):
        return False
    for begin in MAIN_KEY_BEGIN.finditer(text):
        end = MAIN_KEY_END.search(text, begin.end())
        other = text.find("-----BEGIN ", begin.end())
        if end is None or -1 < other < end.start():
            return False
    return True


def covered(patterns, text):
    """Return the positions that any of the patterns match."""
    positions = set()
    for pattern in patterns:
        for match in pattern.finditer(text):
            positions.update(range(match.start(), match.end()))
    return positions


class TestPatternsMatchMain(unittest.TestCase):
    """Seeded fuzz of the rewritten patterns against main's regexes."""

    def test_jwt(self):
        rng = random.Random(1)
        marker = redaction_marker("jwt")
        ends = (".",) * 7 + ("", " ", "eyJ", "/", "é", "=", "+")
        found = 0
        for _ in range(20000):
            parts = []
            for _ in range(rng.randrange(7)):
                size = rng.choice((0, 1, 9, 10, 10, 11, 13, 13))
                run = "".join(rng.choice("aZ0_-") for _ in range(size))
                if run and rng.random() < 0.3:
                    cut = rng.randrange(len(run))
                    run = run[:cut] + "eyJ" + run[cut:]
                if rng.random() < 0.8:
                    run = "eyJ" + run
                parts.append(run + rng.choice(ends))
            text = "".join(parts)
            if HEX_TOKEN.search(text):
                continue
            expected = MAIN_JWT.sub(marker, text)
            found += expected != text
            self.assertEqual(redact_value(text), expected, repr(text))
            self.assertEqual(_contains_secret(text), expected != text, repr(text))
        self.assertGreater(found, 1000)

    def test_basic_auth(self):
        rng = random.Random(2)
        marker = redaction_marker("basic-auth")
        pieces = (
            "https", "a", "Z", "9", "1", "u", "p", "host", "+", ".", "-", "_", "é", "ß",
            "://", ":", "@", "/", " ", "\n", "x:y@", "://u:p@",
        )
        found = 0
        for _ in range(20000):
            text = "".join(rng.choice(pieces) for _ in range(rng.randrange(16)))
            expected = MAIN_BASIC_AUTH.sub(lambda m: m.group(1) + marker + "@", text)
            found += expected != text
            self.assertEqual(redact_value(text), expected, repr(text))
            self.assertEqual(_contains_secret(text), expected != text, repr(text))
        self.assertGreater(found, 1000)

    def test_private_key(self):
        rng = random.Random(3)
        labels = ("", "RSA ", "EC ", "DSA ", "OPENSSH ")

        def begin():
            return "-----BEGIN " + rng.choice(labels) + "PRIVATE KEY-----"

        def end():
            return "-----END " + rng.choice(labels) + "PRIVATE KEY-----"

        def block():
            return begin() + rng.choice(("\nMII\n", "", "abc", "\\nQQ\\n")) + end()

        pieces = [begin, end, block] + [
            lambda text=text: text
            for text in (
                "MIIE", "\n", "\\n", " ", "x.", "-----", "----", "-", "BEGIN ", "END ",
                "PRIVATE KEY", "-----END CERTIFICATE-----", "-----BEGIN CERTIFICATE-----",
                "Proc-Type: 4,ENCRYPTED\n",
            )
        ]
        found = 0
        for _ in range(20000):
            text = "".join(rng.choice(pieces)() for _ in range(rng.randrange(9)))
            if not in_main_domain(text):
                continue
            expected = MAIN_PRIVATE_KEY.sub(KEY_MARKER, text)
            found += expected != text
            self.assertEqual(redact_value(text), expected, repr(text))
            self.assertEqual(_contains_secret(text), expected != text, repr(text))
        self.assertGreater(found, 1000)

    def test_token_patterns_cover_main(self):
        rng = random.Random(4)
        main = {kind: [re.compile(s) for s in sources] for kind, sources in MAIN_TOKENS.items()}
        new = {kind: [p.pattern for p in SECRET_PATTERNS if p.kind == kind] for kind in main}
        pieces = (
            "sk-", "sk-ant-", "sk-proj-", "sk-svcacct-", "sk-admin-", "sk-None-", "ant-",
            "proj-", "api03", "_", "-", "A", "z", "0", "x" * 10, "x" * 18, "Q" * 36, "gh",
            "ghp_", "gho_", "ghs_", "ghu_", "ghr_", "p_", "github_pat_", "ri", "é", " ", ".",
        )
        found = dict.fromkeys(main, 0)
        for _ in range(30000):
            text = "".join(rng.choice(pieces) for _ in range(rng.randrange(12)))
            # A GitHub token glued to the next one is out of scope. The first
            # match takes the start of the second, as on main.
            if GLUED_GITHUB_TOKENS.search(text):
                continue
            for kind in main:
                before = covered(main[kind], text)
                found[kind] += bool(before)
                self.assertLessEqual(before, covered(new[kind], text), (kind, text))
        for kind, count in found.items():
            self.assertGreater(count, 200, kind)

    def test_cut_key_body_follows_the_rule(self):
        rng = random.Random(5)
        pieces = (
            "A", "b", "9", "+", "/", "=", "MIIE", "n", "r", "\n", "\r\n", "\r", "\\n",
            "\\r\\n", "\\\\n", "\\\\r\\\\n", "\\", "\\\\", " ", "  ", "\t", "Proc-Type:",
            "DEK-Info:", "Version:", "Comment:", "Proc-Type", " 4,ENCRYPTED", ",", ":", "-",
            ".", "x y", "é", "\n" * 70, "\\n" * 70, "\n " * 66, "a " * 70,
        )
        for _ in range(15000):
            body = "".join(rng.choice(pieces) for _ in range(rng.randrange(14)))
            expected = KEY_MARKER + body[KEY_BODY.match(body).end() :]
            self.assertEqual(
                redact_value("-----BEGIN PRIVATE KEY-----" + body), expected, repr(body)
            )


class TestPatternCost(unittest.TestCase):
    """Inputs that took quadratic time on main stay fast and small."""

    BEGIN = "-----BEGIN PRIVATE KEY-----"

    def test_long_inputs_finish_fast(self):
        shapes = {
            "BEGIN lines": "-----BEGIN RSA PRIVATE KEY-----\n" * 10_000,
            "one-line BEGIN run": "-----BEGIN " * 29_000,
            "eyJ run": "eyJ" * 53_000,
            "a. run": "a." * 80_000,
            "a. run, then ://": "a." * 80_000 + "://",
            "label words": "-----BEGIN " + "A " * 160_000,
            "END label words": self.BEGIN + "\n-----END " + "A " * 160_000,
            "blank lines": self.BEGIN + "\n" * 320_000,
            "escaped blank lines": self.BEGIN + "\\n" * 160_000,
            "base64 lines": self.BEGIN + ("\n" + "A" * 64) * 5_000,
            "spaced words": self.BEGIN + "\n" + "a " * 160_000,
            "header backslashes": self.BEGIN + "\nComment: " + "\\x" * 160_000,
        }
        for name, text in shapes.items():
            with self.subTest(name):
                for check in (_contains_secret, redact_value):
                    start = time.perf_counter()
                    check(text)
                    self.assertLess(time.perf_counter() - start, 1.0, check.__name__)

    def test_provider_key_near_misses_finish_fast(self):
        # About 1 MB each: prefixes with no body, prefixes inside a word, and
        # bodies one character short.
        n = 1_000_000
        shapes = {
            "sk_live_ run": "sk_" + "live_" * (n // 5),
            "risk_live_ words": ("risk_" + "live_" + "a" * 30 + " ") * (n // 41),
            "short Stripe keys": ("sk_" + "live_" + "a" * 23 + " ") * (n // 32),
            "whsec_ run": ("wh" + "sec_") * (n // 6),
            "AIza run": ("AI" + "za") * (n // 4),
            "short Google keys": (" AI" + "za" + "a" * 34) * (n // 39),
            "glpat- run": ("-gl" + "pat-") * (n // 7),
            "short GitLab tokens": ("gl" + "pat-" + "a" * 19 + ".") * (n // 26),
            "GitLab dots": ("gl" + "pat-" + "a" * 20 + ".ab.") * (n // 30),
            "short npm tokens": ("npm" + "_" + "a" * 35 + "_") * (n // 40),
            "xhf_ words": ("xhf" + "_" + "a" * 29) * (n // 33),
            "hf_ run": ("hf" + "_") * (n // 3),
            "-sk_live_ run": ("-sk_" + "live_") * (n // 9),
            "-sk_org_ run": ("-sk_" + "org_") * (n // 8),
            "short Stripe org keys": (
                "-sk_" + "org_" + "a" * 10 + "_" + "a" * 23 + " "
            ) * (n // 43),
            "short base64 webhook secrets": ("-wh" + "sec_" + "+/" * 15 + " ") * (n // 38),
            "-gl run": "-gl" * (n // 3),
            "-glrtr- run": ("-gl" + "rtr-") * (n // 7),
            "short GitLab JWTs": (
                "-gl" + "cbt-eyJ" + "a" * 10 + ".eyJ" + "a" * 10 + "." + "a" * 19 + " "
            ) * (n // 55),
            "GitLab JWT headers": ("-gl" + "cbt-eyJ" + "a" * 10 + ".") * (n // 21),
        }
        for name, text in shapes.items():
            with self.subTest(name):
                for check in (_contains_secret, redact_value):
                    start = time.perf_counter()
                    check(text)
                    self.assertLess(time.perf_counter() - start, 1.0, check.__name__)

    def test_long_runs_need_little_memory(self):
        # 320K characters each. A regex that keeps state for each word or line
        # needs 25 MB or more here.
        shapes = {
            "label words": "-----BEGIN " + "A " * 160_000,
            "END label words": self.BEGIN + "\n-----END " + "A " * 160_000,
            "blank lines": self.BEGIN + "\n" * 320_000,
            "escaped blank lines": self.BEGIN + "\\n" * 160_000,
            "spaced words": self.BEGIN + "\n" + "a " * 160_000,
        }
        for name, text in shapes.items():
            with self.subTest(name):
                tracemalloc.start()
                try:
                    redact_value(text)
                    peak = tracemalloc.get_traced_memory()[1]
                finally:
                    tracemalloc.stop()
                self.assertLess(peak, 8_000_000)


if __name__ == "__main__":
    unittest.main()
