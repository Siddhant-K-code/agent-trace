"""Trace anonymization: strip identifying information from exported traces.

Applied at export time — original session data is never modified.
Complements secret redaction (redact.py), which strips secrets at capture
time. Anonymization strips identity: paths, hostnames, usernames, emails.

Rules applied:
  - Home directory paths → ~/relative/path
  - Hostnames (from socket.gethostname()) → <hostname>
  - Email addresses → <email>
  - OS username (from os.getlogin / env) → <user>
  - Custom regex patterns from .agent-strace/anonymize.yaml

Usage:
    agent-strace export <session-id> --anonymize --output trace.json
    agent-strace export <session-id> --anonymize --dry-run
"""

from __future__ import annotations

import json
import os
import re
import socket
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from .models import TraceEvent, SessionMeta
from .store import TraceStore


# ---------------------------------------------------------------------------
# Built-in anonymization rules
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(
    r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b"
)


def _get_home_dir() -> str:
    return str(Path.home())


def _get_hostname() -> str:
    try:
        return socket.gethostname()
    except Exception:
        return ""


def _get_username() -> str:
    for key in ("USER", "USERNAME", "LOGNAME"):
        val = os.environ.get(key, "")
        if val:
            return val
    try:
        return os.getlogin()
    except Exception:
        return ""


@dataclass
class AnonymizationRule:
    """A single find-and-replace rule."""
    pattern: re.Pattern
    replacement: str
    description: str = ""


@dataclass
class AnonymizationResult:
    """Summary of what was anonymized."""
    rules_applied: dict[str, int] = field(default_factory=dict)  # description → count

    @property
    def total_replacements(self) -> int:
        return sum(self.rules_applied.values())

    def record(self, description: str, count: int = 1) -> None:
        self.rules_applied[description] = self.rules_applied.get(description, 0) + count


def _build_builtin_rules() -> list[AnonymizationRule]:
    """Build the default set of anonymization rules from the current environment."""
    rules: list[AnonymizationRule] = []

    # Home directory paths
    home = _get_home_dir()
    if home and home != "/":
        # Match the home dir prefix in paths, replace with ~
        escaped = re.escape(home)
        rules.append(AnonymizationRule(
            pattern=re.compile(escaped + r"(/[^\s\"']*)?"),
            replacement=lambda m: "~" + (m.group(1) or ""),
            description="home directory paths",
        ))

    # Hostname
    hostname = _get_hostname()
    if hostname:
        rules.append(AnonymizationRule(
            pattern=re.compile(re.escape(hostname)),
            replacement="<hostname>",
            description="hostname",
        ))

    # Username
    username = _get_username()
    if username and len(username) >= 3:
        # Only replace when it looks like a standalone word to avoid false positives
        rules.append(AnonymizationRule(
            pattern=re.compile(r"\b" + re.escape(username) + r"\b"),
            replacement="<user>",
            description="username",
        ))

    # Email addresses
    rules.append(AnonymizationRule(
        pattern=_EMAIL_RE,
        replacement="<email>",
        description="email addresses",
    ))

    return rules


def _load_custom_rules(config_path: str | Path | None = None) -> list[AnonymizationRule]:
    """Load custom rules from .agent-strace/anonymize.yaml or the given path."""
    paths = []
    if config_path:
        paths.append(Path(config_path))
    paths += [
        Path(".agent-strace/anonymize.yaml"),
        Path(".agent-strace/anonymize.yml"),
    ]

    for p in paths:
        if p.exists():
            try:
                return _parse_custom_rules(p.read_text())
            except Exception:
                pass
    return []


def _parse_custom_rules(text: str) -> list[AnonymizationRule]:
    """Parse a minimal YAML rules file into AnonymizationRule objects.

    Expected format:
        rules:
          - pattern: "regex here"
            replacement: "<REDACTED>"
    """
    rules: list[AnonymizationRule] = []
    current: dict = {}
    in_rules = False

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "rules:":
            in_rules = True
            continue
        if not in_rules:
            continue
        if stripped.startswith("- "):
            if current:
                _try_add_rule(current, rules)
            current = {}
            rest = stripped[2:].strip()
            if ":" in rest:
                k, _, v = rest.partition(":")
                current[k.strip()] = v.strip().strip('"').strip("'")
        elif current is not None and ":" in stripped:
            k, _, v = stripped.partition(":")
            current[k.strip()] = v.strip().strip('"').strip("'")

    if current:
        _try_add_rule(current, rules)

    return rules


def _try_add_rule(d: dict, rules: list[AnonymizationRule]) -> None:
    pattern_str = d.get("pattern", "")
    replacement = d.get("replacement", "<REDACTED>")
    description = d.get("description", f"custom: {pattern_str[:40]}")
    if pattern_str:
        try:
            rules.append(AnonymizationRule(
                pattern=re.compile(pattern_str),
                replacement=replacement,
                description=description,
            ))
        except re.error:
            pass


# ---------------------------------------------------------------------------
# Core anonymization engine
# ---------------------------------------------------------------------------

def _apply_rules_to_string(
    text: str,
    rules: list[AnonymizationRule],
    result: AnonymizationResult,
) -> str:
    """Apply all rules to a string, recording replacement counts."""
    for rule in rules:
        new_text, count = rule.pattern.subn(rule.replacement, text)
        if count:
            result.record(rule.description, count)
            text = new_text
    return text


def _anonymize_value(
    value: Any,
    rules: list[AnonymizationRule],
    result: AnonymizationResult,
) -> Any:
    """Recursively anonymize a value (str, dict, list, or scalar)."""
    if isinstance(value, str):
        return _apply_rules_to_string(value, rules, result)
    if isinstance(value, dict):
        return {k: _anonymize_value(v, rules, result) for k, v in value.items()}
    if isinstance(value, list):
        return [_anonymize_value(item, rules, result) for item in value]
    return value


def anonymize_event(
    event: TraceEvent,
    rules: list[AnonymizationRule],
    result: AnonymizationResult,
) -> TraceEvent:
    """Return a new TraceEvent with anonymized data. Original is unchanged."""
    new_data = _anonymize_value(event.data, rules, result)
    # Build a new event with the same fields but anonymized data
    import copy
    new_event = copy.copy(event)
    new_event.data = new_data
    return new_event


def anonymize_session(
    store: TraceStore,
    session_id: str,
    custom_config: str | Path | None = None,
) -> tuple[list[TraceEvent], AnonymizationResult]:
    """Load and anonymize all events for a session.

    Returns (anonymized_events, result_summary). Original store is unchanged.
    """
    events = store.load_events(session_id)
    rules = _build_builtin_rules() + _load_custom_rules(custom_config)
    result = AnonymizationResult()
    anonymized = [anonymize_event(ev, rules, result) for ev in events]
    return anonymized, result


# ---------------------------------------------------------------------------
# CLI handler
# ---------------------------------------------------------------------------

def cmd_anonymize_export(args, out: TextIO = sys.stdout) -> int:
    """Export a session with anonymization applied."""
    store = TraceStore(args.trace_dir)

    session_id = getattr(args, "session_id", None)
    if not session_id:
        session_id = store.get_latest_session_id()
    if not session_id:
        sys.stderr.write("No sessions found.\n")
        return 1

    if not store.session_exists(session_id):
        found = store.find_session(session_id)
        if found:
            session_id = found
        else:
            sys.stderr.write(f"Session not found: {session_id}\n")
            return 1

    custom_config = getattr(args, "anonymize_config", None)
    dry_run = getattr(args, "dry_run", False)

    anonymized_events, result = anonymize_session(store, session_id, custom_config)

    if dry_run:
        out.write(f"Anonymizing session {session_id[:16]}...\n\n")
        if result.total_replacements == 0:
            out.write("Nothing to anonymize.\n")
        else:
            out.write("Would redact:\n")
            for desc, count in sorted(result.rules_applied.items()):
                out.write(f"  {count:3d} {desc}\n")
        return 0

    output_path = getattr(args, "output", "") or f"session-{session_id[:12]}-anon.json"
    meta = store.load_meta(session_id)

    export_data = {
        "session_id": session_id,
        "meta": json.loads(meta.to_json()),
        "anonymized": True,
        "events": [json.loads(ev.to_json()) for ev in anonymized_events],
    }

    Path(output_path).write_text(json.dumps(export_data, indent=2))

    out.write(f"Anonymizing session {session_id[:16]}...\n\n")
    if result.total_replacements == 0:
        out.write("Nothing to anonymize.\n")
    else:
        out.write("Redacted:\n")
        for desc, count in sorted(result.rules_applied.items()):
            out.write(f"  {count:3d} {desc}\n")
    out.write(f"\nExported anonymized trace to {output_path}\n")
    return 0
