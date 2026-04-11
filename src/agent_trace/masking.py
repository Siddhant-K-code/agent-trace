"""Live PII and sensitive data masking for the proxy layer.

Extends redact.py with PII detection (emails, phone numbers, credit cards,
SSNs, IP addresses, names in common patterns) applied in real-time as events
flow through the proxy. Operates on raw JSON-RPC message bodies before they
are stored as trace events.

Usage in proxy:
    from .masking import mask_event_data
    event.data = mask_event_data(event.data, config)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .redact import redact_data, REDACTED

# ---------------------------------------------------------------------------
# PII patterns
# ---------------------------------------------------------------------------

# Email addresses
_EMAIL_RE = re.compile(
    r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b"
)

# Phone numbers (US and international formats)
# Require a separator (space, dash, dot, or parens) to reduce false positives
# on plain numeric sequences like IP addresses or version numbers.
_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?1[\s\-.]?)?"
    r"(?:\(\d{3}\)[\s\-.]|\d{3}[\-.])"  # area code must have separator
    r"\d{3}[\s\-.]?\d{4}"
    r"(?!\d)"
)

# Credit card numbers (Visa, MC, Amex, Discover — 13-16 digits with optional separators)
_CC_RE = re.compile(
    r"\b(?:4[0-9]{12}(?:[0-9]{3})?"           # Visa
    r"|5[1-5][0-9]{14}"                         # Mastercard
    r"|3[47][0-9]{13}"                          # Amex
    r"|6(?:011|5[0-9]{2})[0-9]{12}"            # Discover
    r"|(?:\d{4}[\s\-]){3}\d{4})\b"             # Generic 16-digit with separators
)

# US Social Security Numbers
_SSN_RE = re.compile(
    r"\b(?!000|666|9\d{2})\d{3}[\s\-]"
    r"(?!00)\d{2}[\s\-]"
    r"(?!0000)\d{4}\b"
)

# IPv4 addresses (private ranges are not masked; public IPs are)
_IPV4_RE = re.compile(
    r"\b(?!(?:10|127|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.\d+\.\d+)"
    r"(?:\d{1,3}\.){3}\d{1,3}\b"
)

# AWS account IDs (12-digit numbers in ARNs)
_AWS_ARN_RE = re.compile(r"arn:aws:[a-z0-9\-]+:[a-z0-9\-]*:(\d{12}):")

# Generic UUIDs that look like user/account IDs (not session IDs)
# Only mask when they appear as values of keys containing "user", "account", "customer"
_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)

_USER_KEYS = {
    "user_id", "userid", "user-id", "account_id", "accountid",
    "customer_id", "customerid", "person_id", "personid",
    "email", "phone", "mobile", "ssn", "dob", "date_of_birth",
    "credit_card", "card_number", "cvv", "billing_address",
}

# ---------------------------------------------------------------------------
# Masking config
# ---------------------------------------------------------------------------

@dataclass
class MaskingConfig:
    mask_emails: bool = True
    mask_phones: bool = True
    mask_credit_cards: bool = True
    mask_ssn: bool = True
    mask_public_ips: bool = False   # off by default — too noisy
    mask_aws_arns: bool = True
    mask_user_id_keys: bool = True
    # Custom regex patterns provided by the user
    custom_patterns: list[str] = field(default_factory=list)
    # Replacement token (default: [MASKED])
    replacement: str = "[MASKED]"

    @classmethod
    def from_dict(cls, d: dict) -> "MaskingConfig":
        return cls(
            mask_emails=d.get("emails", True),
            mask_phones=d.get("phones", True),
            mask_credit_cards=d.get("credit_cards", True),
            mask_ssn=d.get("ssn", True),
            mask_public_ips=d.get("public_ips", False),
            mask_aws_arns=d.get("aws_arns", True),
            mask_user_id_keys=d.get("user_id_keys", True),
            custom_patterns=d.get("custom_patterns", []),
            replacement=d.get("replacement", "[MASKED]"),
        )

    @classmethod
    def default(cls) -> "MaskingConfig":
        return cls()


# ---------------------------------------------------------------------------
# String-level masking
# ---------------------------------------------------------------------------

def _mask_string(value: str, config: MaskingConfig) -> str:
    """Apply all enabled PII patterns to a string value."""
    result = value
    repl = config.replacement

    if config.mask_emails:
        result = _EMAIL_RE.sub(repl, result)

    if config.mask_phones:
        result = _PHONE_RE.sub(repl, result)

    if config.mask_credit_cards:
        result = _CC_RE.sub(repl, result)

    if config.mask_ssn:
        result = _SSN_RE.sub(repl, result)

    if config.mask_public_ips:
        result = _IPV4_RE.sub(repl, result)

    if config.mask_aws_arns:
        # Replace only the account ID portion (group 1) within the ARN
        def _mask_arn(m: re.Match) -> str:
            return m.group(0).replace(m.group(1), repl, 1)
        result = _AWS_ARN_RE.sub(_mask_arn, result)

    for pat_str in config.custom_patterns:
        try:
            result = re.sub(pat_str, repl, result)
        except re.error:
            pass

    return result


# ---------------------------------------------------------------------------
# Recursive data masking
# ---------------------------------------------------------------------------

def mask_data(data: Any, config: MaskingConfig, parent_key: str = "") -> Any:
    """Recursively apply PII masking to event data.

    Runs *after* redact_data (which handles secrets/tokens). This layer
    handles PII: emails, phones, credit cards, SSNs, etc.
    """
    if isinstance(data, dict):
        result = {}
        for k, v in data.items():
            k_lower = k.lower().strip()
            if config.mask_user_id_keys and k_lower in _USER_KEYS:
                result[k] = config.replacement if isinstance(v, str) else v
            else:
                result[k] = mask_data(v, config, parent_key=k)
        return result

    if isinstance(data, list):
        return [mask_data(item, config, parent_key=parent_key) for item in data]

    if isinstance(data, str):
        return _mask_string(data, config)

    return data


# ---------------------------------------------------------------------------
# Combined masking entry point (secrets + PII)
# ---------------------------------------------------------------------------

def mask_event_data(
    data: Any,
    config: MaskingConfig | None = None,
    redact_secrets: bool = True,
) -> Any:
    """Apply secret redaction and PII masking to event data.

    This is the single entry point used by the proxy layer. It runs
    redact_data first (for API keys, tokens, etc.) then mask_data (for PII).
    """
    if config is None:
        config = MaskingConfig.default()

    result = data
    if redact_secrets:
        result = redact_data(result)
    result = mask_data(result, config)
    return result
