"""Secret redaction for trace events.

Scans event data for values that look like secrets (API keys, tokens,
passwords, connection strings) and replaces them with typed redaction
markers such as [REDACTED:api-key].

Patterns detected:
  - API keys: sk-*, ghp_*, gho_*, github_pat_*, xoxb-*, xoxp-*,
    AKIA*, key-*, Bearer *, token-*
  - Passwords and credentials in sensitive key names
  - Connection strings: postgres://, mysql://, mongodb://, redis://
  - Private key blocks, basic-auth URLs, AWS access keys, JWT tokens

Works on nested dicts and lists. Redacts values, not keys.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

REDACTED = "[REDACTED]"

_NO_REDACT_ENV_VARS = ("AGENT_TRACE_NO_REDACT", "AGENT_STRACE_NO_REDACT")

# Keys whose values should always be redacted (case-insensitive)
SENSITIVE_KEYS = {
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "api-key",
    "access_token",
    "refresh_token",
    "auth_token",
    "authorization",
    "credential",
    "credentials",
    "private_key",
    "privatekey",
    "client_secret",
    "connection_string",
    "database_url",
    "db_url",
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
}


@dataclass(frozen=True)
class SecretPattern:
    kind: str
    pattern: re.Pattern[str]


def redaction_marker(kind: str) -> str:
    """Return a typed redaction marker."""
    return f"[REDACTED:{kind}]"


# Regex patterns that match secret-looking values.
SECRET_PATTERNS = [
    SecretPattern(
        "private-key",
        re.compile(
            r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"
            r"[\s\S]*?"
            r"-----END (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"
        ),
    ),
    SecretPattern("anthropic-key", re.compile(r"sk-ant-[a-zA-Z0-9\-]{20,}")),
    SecretPattern("openai-key", re.compile(r"sk-[a-zA-Z0-9]{20,}")),
    SecretPattern("github-token", re.compile(r"gh[ps]_[a-zA-Z0-9]{36,}")),
    SecretPattern("github-token", re.compile(r"github_pat_[a-zA-Z0-9_]{22,}")),
    SecretPattern("slack-token", re.compile(r"xox[bpras]-[a-zA-Z0-9\-]{10,}")),
    SecretPattern("aws-access-key", re.compile(r"AKIA[0-9A-Z]{16}")),
    SecretPattern("bearer-token", re.compile(r"Bearer\s+[a-zA-Z0-9\-._~+/]+=*")),
    SecretPattern(
        "connection-string",
        re.compile(r"(postgres|mysql|mongodb|redis|amqp)://[^\s]+"),
    ),
    SecretPattern(
        "basic-auth",
        re.compile(r"\b([a-zA-Z][a-zA-Z0-9+.-]*://)([^/\s:@]+:[^/\s@]+)@"),
    ),
    SecretPattern("api-key", re.compile(r"key-[a-zA-Z0-9]{16,}")),
    SecretPattern("token", re.compile(r"token-[a-zA-Z0-9]{16,}")),
    SecretPattern(
        "jwt",
        re.compile(
            r"eyJ[a-zA-Z0-9_-]{10,}\."
            r"eyJ[a-zA-Z0-9_-]{10,}\."
            r"[a-zA-Z0-9_-]{10,}"
        ),
    ),
    SecretPattern("hex-token", re.compile(r"[0-9a-f]{40,}")),
]


def _is_sensitive_key(key: str) -> bool:
    """Check if a key name suggests its value is a secret."""
    normalized = key.lower().strip().replace("-", "_")
    return normalized in SENSITIVE_KEYS


def _contains_secret(value: str) -> bool:
    """Check if a string value matches any secret pattern."""
    for secret in SECRET_PATTERNS:
        if secret.pattern.search(value):
            return True
    return False


def redact_value(value: str) -> str:
    """Redact secrets from a string value.

    Replaces matched patterns with typed redaction markers inline.
    """
    redacted = value
    for secret in SECRET_PATTERNS:
        marker = redaction_marker(secret.kind)
        if secret.kind == "basic-auth":
            redacted = secret.pattern.sub(
                lambda m: f"{m.group(1)}{marker}@",
                redacted,
            )
        else:
            redacted = secret.pattern.sub(marker, redacted)
    return redacted


def redact_data(data: Any, parent_key: str = "") -> Any:
    """Recursively redact secrets from event data.

    Handles dicts, lists, and string values.
    """
    if isinstance(data, dict):
        result = {}
        for k, v in data.items():
            if isinstance(v, str) and _is_sensitive_key(k):
                result[k] = redaction_marker("sensitive")
            else:
                result[k] = redact_data(v, parent_key=k)
        return result

    if isinstance(data, list):
        return [redact_data(item, parent_key=parent_key) for item in data]

    if isinstance(data, str):
        if _is_sensitive_key(parent_key):
            return redaction_marker("sensitive")
        if _contains_secret(data):
            return redact_value(data)
        return data

    return data


def contains_redaction_marker(data: Any) -> bool:
    """Return True when data contains an existing redaction marker."""
    if isinstance(data, dict):
        return any(contains_redaction_marker(v) for v in data.values())
    if isinstance(data, list):
        return any(contains_redaction_marker(item) for item in data)
    if isinstance(data, str):
        return "[REDACTED" in data
    return False


def redact_data_with_status(data: Any) -> tuple[Any, bool]:
    """Return redacted data and whether redaction is present."""
    redacted = redact_data(data)
    return redacted, redacted != data or contains_redaction_marker(redacted)


def redaction_enabled(default: bool = True) -> bool:
    """Return whether automatic secret redaction should run."""
    for name in _NO_REDACT_ENV_VARS:
        if os.environ.get(name, "").lower() in ("1", "true", "yes"):
            return False
    return default
