"""Secret redaction for trace events.

Scans event data for values that look like secrets (API keys, tokens,
passwords, connection strings) and replaces them with typed redaction
markers such as [REDACTED:api-key].

Patterns detected:
  - API keys: sk-*, sk-proj-*, sk-svcacct-*, sk-admin-*, sk-None-*,
    sk-ant-*, ghp_*, gho_*, ghu_*, ghs_*, ghr_*, github_pat_*,
    xox[bpras]-*, AKIA*, key-*, token-*, Bearer *
  - Provider keys: Stripe sk_live_*, sk_test_*, rk_live_*, rk_test_*,
    sk_org_*, whsec_* (not the public pk_*), Google AIza*, GitLab glpat-*,
    glptt-*, gldt-*, glrt-*, glcbt-* and the other documented gl*- prefixes,
    npm_*, Hugging Face hf_*
  - Values under sensitive key names such as password or api_key, at any
    depth below the key
  - Connection strings: postgres://, mysql://, mongodb://, redis://, amqp://
  - Private key blocks: any label that ends in PRIVATE KEY, and PGP.
    A block cut before its END line is redacted up to the end of its body.
    Body lines that end in an END line with no BEGIN line are redacted too.
    Line numbers from cat -n or Claude Code output may start body lines.
  - Basic-auth URLs, JWTs, and hex strings of 40 or more characters

Works on nested dicts and lists. Redacts values, not keys.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Callable

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


def redaction_marker(kind: str) -> str:
    """Return a typed redaction marker."""
    return f"[REDACTED:{kind}]"


@dataclass(frozen=True)
class SecretPattern:
    """A secret regex and how to replace its matches.

    ``literal`` skips the regex when the value lacks that text. ``replace``
    builds the replacement for one match. It returns the match unchanged
    when the match holds no secret.
    """

    kind: str
    pattern: re.Pattern[str]
    literal: str = ""
    replace: Callable[[re.Match[str], str], str] | None = None

    def found_in(self, value: str) -> bool:
        if self.literal and self.literal not in value:
            return False
        if self.replace is None:
            return self.pattern.search(value) is not None
        marker = redaction_marker(self.kind)
        return any(
            self.replace(match, marker) != match.group(0)
            for match in self.pattern.finditer(value)
        )

    def redact(self, value: str) -> str:
        if self.literal and self.literal not in value:
            return value
        marker = redaction_marker(self.kind)
        replace = self.replace
        if replace is None:
            return self.pattern.sub(marker, value)
        return self.pattern.sub(lambda match: replace(match, marker), value)


# A PEM label is words of uppercase letters and digits that end in PRIVATE
# KEY, or the PGP label. A label never holds "-", so a run of BEGIN lines
# stays linear. At most 8 words come before PRIVATE KEY, so a long run of
# words needs no extra memory.
_PEM_LABEL = r"(?:PGP PRIVATE KEY BLOCK|(?:[A-Z0-9]+ ){0,8}PRIVATE KEY)"
# Real line breaks, JSON-escaped ones, and ones escaped twice, as repr()
# prints JSON text.
_PEM_BREAK = r"(?:\r?\n|\\r\\n|\\n|\\\\r\\\\n|\\\\n)"
# A header line runs to the next line break. A backslash that does not start
# an escaped line break is part of the header.
_PEM_HEADER = (
    r"(?:Proc-Type|DEK-Info|Version|Comment):"
    r"[^\r\n]*?(?=[\r\n]|\\(?:n|r\\n|\\n|\\r\\\\n)|\Z)"
)
_PEM_BASE64 = r"[A-Za-z0-9+/=]"
# A line number as cat -n ("2<tab>") or Claude Code ("2→") prints it, with
# the tab or arrow as itself, JSON-escaped, or escaped twice.
_PEM_NUMBER = r"[0-9]+(?:\t|\\t|\\\\t|\u2192|\\u2192|\\\\u2192)"
_PEM_LINE_NUMBER = r"(?:" + _PEM_NUMBER + r"[ \t]*)?"
_PEM_GAP = r"(?:" + _PEM_BREAK + r"[ \t]*" + _PEM_LINE_NUMBER + r")"
# A line number is never the base64 run of a line.
_PEM_LINE = (
    r"(?:" + _PEM_HEADER + r"|(?!" + _PEM_NUMBER + r")" + _PEM_BASE64 + r"+)"
)
_PEM_FIRST_RUN = re.compile(_PEM_BASE64 + r"*")
# Up to 64 steps. A step is up to 64 line breaks and then a header or a
# base64 run, or a space and then a base64 run.
_PEM_STEPS = re.compile(
    r"(?:" + _PEM_GAP + r"{1,64}" + _PEM_LINE + r"| " + _PEM_BASE64 + r"+){0,64}"
)
_PEM_GAPS = re.compile(_PEM_GAP + r"{1,64}")
_PEM_LINE_START = re.compile(_PEM_LINE)
# A BEGIN line, then the text up to the first END line, the next BEGIN line,
# or the end of the string. The scan moves one character at a time, so it
# needs no memory per character.
_PRIVATE_KEY = re.compile(
    r"(?P<head>-----BEGIN " + _PEM_LABEL + r"-----)"
    r"(?s:.)*?(?=-----(?:END " + _PEM_LABEL + r"-----|BEGIN )|\Z)"
    r"(?P<end>-----END " + _PEM_LABEL + r"-----)?"
)


def _pem_body_end(text: str, pos: int, end: int) -> int:
    """Return where the body of a key cut before its END line ends.

    The body is base64 runs split by line breaks or single spaces, header
    lines, blank lines and indentation. Each regex call reads a bounded
    number of lines, so memory stays flat on long input.
    """
    pos = _PEM_FIRST_RUN.match(text, pos, end).end()
    while True:
        step_end = _PEM_STEPS.match(text, pos, end).end()
        if step_end > pos:
            pos = step_end
            continue
        # More than 64 line breaks in a row: skip them, then look for a line.
        gap_end = pos
        gaps = _PEM_GAPS.match(text, gap_end, end)
        while gaps is not None:
            gap_end = gaps.end()
            gaps = _PEM_GAPS.match(text, gap_end, end)
        line = _PEM_LINE_START.match(text, gap_end, end) if gap_end > pos else None
        if line is None:
            return pos
        pos = line.end()


def _replace_private_key(match: re.Match[str], marker: str) -> str:
    if match.group("end") is not None:
        return marker
    # No END line before the next BEGIN line: redact the key body and keep
    # the rest.
    body_end = _pem_body_end(match.string, match.end("head"), match.end())
    return marker + match.string[body_end : match.end()]


_PEM_END_LINE = re.compile(r"-----END " + _PEM_LABEL + r"-----")
_PEM_LINE_PREFIX = re.compile(r"[ \t]*" + _PEM_LINE_NUMBER)
# One whole line of a key body seen from its END line: a base64 run, a
# header line above a base64 run, or a blank line. A run holds no spaces, so
# a line of prose is not a body line.
_PEM_TAIL_LINE = re.compile(
    r"[ \t]*" + _PEM_LINE_NUMBER
    + r"(?:(?P<header>(?:Proc-Type|DEK-Info|Version|Comment):[^\r\n]*)|"
    + r"(?P<body>" + _PEM_BASE64 + r"+))?[ \t]*"
)
# A long base64 run at the end of a line that holds other text first.
_PEM_TAIL_RUN = re.compile(
    r"(?<![A-Za-z0-9+/=])" + _PEM_BASE64 + r"{32,}[ \t]*\Z"
)
# Line breaks as _PEM_BREAK reads them, longest first.
_PEM_BREAK_FORMS = ("\\\\r\\\\n", "\\r\\n", "\\\\n", "\r\n", "\\n", "\n")


class _LineStarts:
    """Find line starts walking up the text from one point, not before low.

    Each search starts below the last one found, so a walk over many lines
    reads each character once.
    """

    def __init__(self, text: str, low: int) -> None:
        self.text = text
        self.low = low
        self.real: int | None = None
        self.escaped: int | None = None

    def start(self, pos: int) -> int:
        """Return where the line that ends at pos starts."""
        text, low = self.text, self.low
        if self.real is None or self.real >= pos:
            self.real = text.rfind("\n", low, pos)
        if self.escaped is None or self.escaped + 2 > pos:
            self.escaped = text.rfind("\\n", low, pos)
        start = low
        if self.real >= 0:
            start = max(start, self.real + 1)
        if self.escaped >= 0:
            start = max(start, self.escaped + 2)
        return start

    def break_before(self, pos: int) -> int:
        """Return the length of the line break that ends at pos, or 0."""
        for form in _PEM_BREAK_FORMS:
            if pos - len(form) >= self.low and self.text.startswith(
                form, pos - len(form), pos
            ):
                return len(form)
        return 0


def _pem_tail_start(lines: _LineStarts, line_start: int) -> int | None:
    """Return where the body lines above an END line start, or None.

    Walks up one whole line at a time and stops at a line that is not a body
    line or at the start of the text that is left.
    """
    top = None
    pos = line_start
    while True:
        brk = lines.break_before(pos)
        if not brk:
            return top
        line_end = pos - brk
        start = lines.start(line_end)
        line = _PEM_TAIL_LINE.fullmatch(lines.text, start, line_end)
        if line is None:
            # The first body line may follow other text, as in "out": "MIIE.
            run = _PEM_TAIL_RUN.search(lines.text, start, line_end)
            return run.start() if run else top
        if line.group("body") is not None:
            top = line.start("body")
        elif line.group("header") is not None and top is not None:
            top = line.start("header")
        pos = start


def _pem_tails(text: str):
    """Yield (start, end) for each key body cut before its BEGIN line.

    The span runs from the first body line through an END line that no
    BEGIN line claimed, as in the output of tail on a key file.
    """
    if "-----END " not in text:
        return
    low = 0
    for end in _PEM_END_LINE.finditer(text):
        lines = _LineStarts(text, low)
        line_start = lines.start(end.start())
        if _PEM_LINE_PREFIX.fullmatch(text, line_start, end.start()):
            top = _pem_tail_start(lines, line_start)
            if top is not None:
                yield top, end.end()
        low = end.end()


@dataclass(frozen=True)
class _KeyTailPattern(SecretPattern):
    """Body lines of a key that end in an END line with no BEGIN line."""

    def found_in(self, value: str) -> bool:
        return next(_pem_tails(value), None) is not None

    def redact(self, value: str) -> str:
        marker = redaction_marker(self.kind)
        parts = []
        last = 0
        for start, end in _pem_tails(value):
            parts.append(value[last:start])
            parts.append(marker)
            last = end
        if not parts:
            return value
        parts.append(value[last:])
        return "".join(parts)


# Each match takes the rest of its run, so the scan never starts again
# inside a run. Only the first eyJ of a run can start a JWT.
_JWT = re.compile(
    r"eyJ(?:[a-zA-Z0-9_-]{10,}"
    r"(\.eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,})?"
    r"|[a-zA-Z0-9_-]*)"
)


def _replace_jwt(match: re.Match[str], marker: str) -> str:
    return marker if match.group(1) is not None else match.group(0)


# The match starts at the start of the scheme run. The replace function
# redacts only when a URL scheme can start inside that run: an ASCII letter
# at a word boundary, or an ASCII letter after "+", "." or "-".
_BASIC_AUTH = re.compile(
    r"(?<![a-zA-Z0-9+.-])([a-zA-Z0-9+.-]+)://[^/\s:@]+:[^/\s@]+@"
)
_WORD_CHAR = re.compile(r"\w")
_INNER_SCHEME_START = re.compile(r"[+.-][a-zA-Z]")


def _replace_basic_auth(match: re.Match[str], marker: str) -> str:
    run = match.group(1)
    start = match.start()
    at_boundary = start == 0 or not _WORD_CHAR.match(match.string, start - 1)
    if (run[0].isascii() and run[0].isalpha() and at_boundary) or (
        _INNER_SCHEME_START.search(run)
    ):
        return f"{run}://{marker}@"
    return match.group(0)


# Regex patterns that match secret-looking values. Order matters: each one
# runs on the output of the one before.
SECRET_PATTERNS = [
    SecretPattern("private-key", _PRIVATE_KEY, replace=_replace_private_key),
    # Runs after the BEGIN pattern, so every END line left has no BEGIN line.
    _KeyTailPattern("private-key", _PEM_END_LINE),
    # Provider keys with a left boundary. They run before the patterns
    # without one (sk-, key-, token-), which could otherwise match inside
    # their bodies. Each lookbehind sits after the prefix so the regex keeps
    # its literal prefix. Stripe publishable keys (pk_) are public.
    # A letter, digit or "_" before the prefix stops a match. A "-" stops it
    # only before the short prefixes AIza, npm_ and hf_: the longer ones
    # match after "-", as in x-sk_live_ or a removed diff line.
    SecretPattern(
        "stripe-key",
        re.compile(r"[rs]k_(?<![A-Za-z0-9_][rs]k_)(?:live|test)_[0-9A-Za-z]{24,}"),
    ),
    # Stripe documents only the sk_org prefix of organization keys, so one
    # short segment such as live_ may come before the body.
    SecretPattern(
        "stripe-key",
        re.compile(
            r"sk_org_(?<![A-Za-z0-9_]sk_org_)(?:[0-9A-Za-z]{1,10}_)?[0-9A-Za-z]{24,}"
        ),
    ),
    # Some webhook secrets are base64, so "+", "/" and "=" are part of them.
    SecretPattern(
        "stripe-webhook-secret",
        re.compile(r"whsec_(?<![A-Za-z0-9_]whsec_)[0-9A-Za-z+/]{32,}={0,2}"),
    ),
    SecretPattern(
        "google-api-key",
        re.compile(r"AIza(?<![A-Za-z0-9_-]AIza)[0-9A-Za-z_-]{35}(?![0-9A-Za-z_-])"),
    ),
    # The prefixes in GitLab's token prefix table. "-" may come before them
    # so instance prefixes such as acme-glpat- match. CI/CD job tokens can be
    # JWTs. Routable tokens end in .<9>, versioned ones in .<2>.<9>.
    SecretPattern(
        "gitlab-token",
        re.compile(
            r"gl(?<![A-Za-z0-9_]gl)"
            r"(?:pat|ptt|dt|rtr?|rct|cbt|ft|imt|agent|oas|soat|ffct|wt)-"
            r"(?:eyJ[0-9A-Za-z_-]{10,}\.eyJ[0-9A-Za-z_-]{10,}\.[0-9A-Za-z_-]{20,}"
            r"|[0-9A-Za-z_-]{20,}(?:(?:\.[0-9a-z]{2})?\.[0-9a-z]{9}(?![0-9A-Za-z_]))?)"
        ),
    ),
    SecretPattern(
        "npm-token",
        re.compile(r"npm_(?<![A-Za-z0-9_-]npm_)[0-9A-Za-z]{36}(?![0-9A-Za-z])"),
    ),
    SecretPattern(
        "huggingface-token",
        re.compile(r"hf_(?<![A-Za-z0-9_-]hf_)[0-9A-Za-z]{30,}"),
    ),
    SecretPattern("anthropic-key", re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}")),
    # The lookbehind sits after "sk-" so the regex keeps its literal prefix.
    SecretPattern(
        "openai-key",
        re.compile(
            r"sk-(?<![A-Za-z0-9_-]sk-)(?:proj|svcacct|admin|None)-[A-Za-z0-9_-]{20,}"
        ),
    ),
    SecretPattern("openai-key", re.compile(r"sk-[a-zA-Z0-9]{20,}")),
    SecretPattern("github-token", re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}")),
    SecretPattern("github-token", re.compile(r"github_pat_[a-zA-Z0-9_]{22,}")),
    SecretPattern("slack-token", re.compile(r"xox[bpras]-[a-zA-Z0-9\-]{10,}")),
    SecretPattern("aws-access-key", re.compile(r"AKIA[0-9A-Z]{16}")),
    SecretPattern("bearer-token", re.compile(r"Bearer\s+[a-zA-Z0-9\-._~+/]+=*")),
    SecretPattern(
        "connection-string",
        re.compile(r"(postgres|mysql|mongodb|redis|amqp)://[^\s]+"),
    ),
    SecretPattern(
        "basic-auth", _BASIC_AUTH, literal="://", replace=_replace_basic_auth
    ),
    SecretPattern("api-key", re.compile(r"key-[a-zA-Z0-9]{16,}")),
    SecretPattern("token", re.compile(r"token-[a-zA-Z0-9]{16,}")),
    SecretPattern("jwt", _JWT, replace=_replace_jwt),
    SecretPattern("hex-token", re.compile(r"[0-9a-f]{40,}")),
]


def _is_sensitive_key(key: str) -> bool:
    """Check if a key name suggests its value is a secret."""
    normalized = key.lower().strip().replace("-", "_")
    return normalized in SENSITIVE_KEYS


def _contains_secret(value: str) -> bool:
    """Check if a string value matches any secret pattern."""
    return any(secret.found_in(value) for secret in SECRET_PATTERNS)


def redact_value(value: str) -> str:
    """Redact secrets from a string value.

    Replaces matched patterns with typed redaction markers inline.
    """
    redacted = value
    for secret in SECRET_PATTERNS:
        redacted = secret.redact(redacted)
    return redacted


def redact_data(data: Any, parent_key: str = "") -> Any:
    """Recursively redact secrets from event data.

    Handles dicts, lists, and string values. Every string under a sensitive
    key is redacted, however deep it sits.
    """
    return _redact_data(data, parent_key, _is_sensitive_key(parent_key))


def _redact_data(data: Any, parent_key: str, sensitive: bool) -> Any:
    if isinstance(data, dict):
        result = {}
        for k, v in data.items():
            # The compliance-report authorization evidence contract includes
            # a policy content digest. Preserve only that exact namespaced,
            # schema-validated digest shape; arbitrary 64-character values
            # remain covered by the ordinary hex-token redactor.
            is_authorization_policy_digest = (
                parent_key == "agent_trace_authorization"
                and data.get("schema") == "agent-strace-authorization/v1"
                and k == "policy_sha256"
                and isinstance(v, str)
                and re.fullmatch(r"sha256:[0-9a-f]{64}", v) is not None
            )
            if is_authorization_policy_digest:
                result[k] = v
            else:
                result[k] = _redact_data(v, k, sensitive or _is_sensitive_key(k))
        return result

    if isinstance(data, list):
        return [_redact_data(item, parent_key, sensitive) for item in data]

    if isinstance(data, str):
        if sensitive:
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
