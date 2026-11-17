"""Permission audit trail for agent sessions.

Checks every tool_call event against a policy file (.agent-scope.json)
and auto-flags sensitive file access even without a policy.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TextIO

from .event_data import object_field, require_object_event_data, str_field
from .inert import inert
from .models import EventType, TraceEvent
from .store import TraceStore


# ---------------------------------------------------------------------------
# Sensitive file patterns (auto-flagged regardless of policy)
# ---------------------------------------------------------------------------

SENSITIVE_PATTERNS: list[str] = [
    ".env", ".env.*", "*.env",
    "secrets.json", "secrets.yaml", "secrets.yml", "secrets.toml",
    "config/secrets*",
    "*.pem", "*.key", "*.p12", "*.pfx",
    ".ssh/*", "id_rsa", "id_ed25519",
    ".aws/credentials", ".aws/config",
    ".netrc", ".npmrc", ".pypirc",
    ".github/workflows/*",
    "*.token", "*.password",
]

Verdict = Literal["allowed", "denied", "no_policy"]


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------

@dataclass
class Policy:
    file_read_allow: list[str] = field(default_factory=list)
    file_read_deny: list[str] = field(default_factory=list)
    file_write_allow: list[str] = field(default_factory=list)
    file_write_deny: list[str] = field(default_factory=list)
    cmd_allow: list[str] = field(default_factory=list)
    cmd_deny: list[str] = field(default_factory=list)
    network_deny_all: bool = False
    network_allow: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> Policy:
        files = d.get("files", {})
        read = files.get("read", {})
        write = files.get("write", {})
        cmds = d.get("commands", {})
        net = d.get("network", {})
        return cls(
            file_read_allow=read.get("allow", []),
            file_read_deny=read.get("deny", []),
            file_write_allow=write.get("allow", []),
            file_write_deny=write.get("deny", []),
            cmd_allow=cmds.get("allow", []),
            cmd_deny=cmds.get("deny", []),
            network_deny_all=net.get("deny_all", False),
            network_allow=net.get("allow", []),
        )

    @classmethod
    def load(cls, path: str | Path) -> Policy | None:
        p = Path(path)
        if not p.exists():
            return None
        try:
            return cls.from_dict(json.loads(p.read_text()))
        except json.JSONDecodeError as exc:
            sys.stderr.write(f"Warning: malformed policy file {p}: {exc}\n")
            return None
        except OSError as exc:
            sys.stderr.write(f"Warning: could not read policy file {p}: {exc}\n")
            return None


# ---------------------------------------------------------------------------
# Audit result types
# ---------------------------------------------------------------------------

@dataclass
class AuditEntry:
    event: TraceEvent
    event_index: int          # 1-based
    action: str               # human-readable description
    verdict: Verdict
    reason: str               # why this verdict was reached
    sensitive: bool = False   # auto-flagged as sensitive


@dataclass
class AuditReport:
    session_id: str
    total_events: int
    total_tool_calls: int
    entries: list[AuditEntry]
    policy_loaded: bool

    @property
    def allowed(self) -> list[AuditEntry]:
        return [e for e in self.entries if e.verdict == "allowed"]

    @property
    def denied(self) -> list[AuditEntry]:
        return [e for e in self.entries if e.verdict == "denied"]

    @property
    def no_policy(self) -> list[AuditEntry]:
        return [e for e in self.entries if e.verdict == "no_policy"]

    @property
    def sensitive_accesses(self) -> list[AuditEntry]:
        return [e for e in self.entries if e.sensitive]


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------

def _glob_match(path: str, patterns: list[str]) -> bool:
    """Match *path* against any of *patterns*.

    Supports ``**`` as a recursive wildcard (matches zero or more path
    components) in addition to the standard fnmatch ``*`` and ``?``.
    Also matches against the bare filename so ``.env`` catches
    ``config/.env``.
    """
    name = Path(path).name
    norm_path = path.replace("\\", "/")
    for pat in patterns:
        norm_pat = pat.replace("\\", "/")
        if fnmatch.fnmatch(norm_path, norm_pat):
            return True
        if fnmatch.fnmatch(name, norm_pat):
            return True
        if "**" in norm_pat and _glob_match_recursive(norm_path, norm_pat):
            return True
    return False


def _glob_match_recursive(path: str, pattern: str) -> bool:
    """Match path against a pattern that may contain ``**`` wildcards.

    ``**`` matches zero or more path components (including none).
    """
    path_parts = path.split("/")
    pat_parts = pattern.split("/")
    return _match_parts(path_parts, pat_parts)


def _match_parts(path_parts: list[str], pat_parts: list[str]) -> bool:
    """Recursively match path_parts against pat_parts."""
    if not pat_parts:
        return not path_parts
    if not path_parts:
        # Only match if remaining pattern is all **
        return all(p == "**" for p in pat_parts)

    if pat_parts[0] == "**":
        # ** can match zero components (skip it) or one+ components (consume one path part)
        return (
            _match_parts(path_parts, pat_parts[1:])          # match zero
            or _match_parts(path_parts[1:], pat_parts)        # match one, keep **
        )

    if fnmatch.fnmatch(path_parts[0], pat_parts[0]):
        return _match_parts(path_parts[1:], pat_parts[1:])

    return False


def _is_sensitive(path: str) -> bool:
    return _glob_match(path, SENSITIVE_PATTERNS)


def _cmd_matches(cmd: str, patterns: list[str]) -> bool:
    cmd_lower = cmd.lower().strip()
    for pat in patterns:
        pat_lower = pat.lower().strip()
        if cmd_lower == pat_lower:
            return True
        # Prefix match only at a word boundary (followed by space or end of string)
        if cmd_lower.startswith(pat_lower) and (
            len(cmd_lower) == len(pat_lower)
            or cmd_lower[len(pat_lower)] == " "
        ):
            return True
        if fnmatch.fnmatch(cmd_lower, pat_lower):
            return True
    return False


_URL_RE = re.compile(r"https?://([^/\s]+)")


def _extract_urls(text: str) -> list[str]:
    # Strip port from host (e.g. "localhost:8080" → "localhost")
    return [host.split(":")[0] for host in _URL_RE.findall(text)]


def _url_allowed(host: str, policy: Policy) -> bool:
    if not policy.network_deny_all:
        return True
    return any(
        fnmatch.fnmatch(host, allowed) or host == allowed
        for allowed in policy.network_allow
    )


# ---------------------------------------------------------------------------
# Per-event audit logic
# ---------------------------------------------------------------------------

def _audit_event(
    event: TraceEvent,
    index: int,
    policy: Policy | None,
) -> list[AuditEntry]:
    """Return zero or more AuditEntry objects for a single tool_call event."""
    entries: list[AuditEntry] = []
    data = event.data
    tool_name = str_field(data, "tool_name", "").lower()
    args = object_field(data, "arguments", {}) or {}

    # --- File read (Read, View, Grep, Glob) ---
    if tool_name in ("read", "view", "grep", "glob"):
        # Grep uses "path" or "pattern"; Glob uses "pattern" or "path"
        path = str(
            args.get("file_path")
            or args.get("path")
            or args.get("pattern")
            or ""
        )
        if path:
            sensitive = _is_sensitive(path)
            if policy and (policy.file_read_allow or policy.file_read_deny):
                if _glob_match(path, policy.file_read_deny):
                    verdict, reason = "denied", "denied by files.read.deny"
                elif policy.file_read_allow and not _glob_match(path, policy.file_read_allow):
                    verdict, reason = "denied", "not in files.read.allow"
                else:
                    verdict, reason = "allowed", "matches files.read.allow"
            else:
                verdict, reason = "no_policy", "no file read policy"
            action_verb = "Glob" if tool_name == "glob" else ("Grep" if tool_name == "grep" else "Read")
            entries.append(AuditEntry(
                event=event, event_index=index,
                action=f"{action_verb} {path}",
                verdict=verdict, reason=reason, sensitive=sensitive,
            ))

    # --- File write / edit ---
    elif tool_name in ("write", "edit", "create"):
        path = str(args.get("file_path") or args.get("path") or "")
        if path:
            sensitive = _is_sensitive(path)
            if policy and (policy.file_write_allow or policy.file_write_deny):
                if _glob_match(path, policy.file_write_deny):
                    verdict, reason = "denied", "denied by files.write.deny"
                elif policy.file_write_allow and not _glob_match(path, policy.file_write_allow):
                    verdict, reason = "denied", "not in files.write.allow"
                else:
                    verdict, reason = "allowed", "matches files.write.allow"
            else:
                verdict, reason = "no_policy", "no file write policy"
            entries.append(AuditEntry(
                event=event, event_index=index,
                action=f"Write {path}",
                verdict=verdict, reason=reason, sensitive=sensitive,
            ))

    # --- Bash / command execution ---
    elif tool_name == "bash":
        cmd = str(args.get("command", "")).strip()
        if cmd:
            # Network access check: scan command for URLs
            urls = _extract_urls(cmd)
            network_denied = False
            for url_host in urls:
                if policy:
                    net_ok = _url_allowed(url_host, policy)
                    net_verdict: Verdict = "allowed" if net_ok else "denied"
                    net_reason = (
                        "allowed by network.allow"
                        if net_ok
                        else "denied by network.deny_all"
                    )
                    if not net_ok:
                        network_denied = True
                else:
                    net_verdict = "no_policy"
                    net_reason = "no network policy"
                entries.append(AuditEntry(
                    event=event, event_index=index,
                    action=f"Network access {url_host}",
                    verdict=net_verdict, reason=net_reason,
                ))

            # Command policy check — skip if a network violation already covers
            # this event to avoid double-counting the same command.
            if network_denied:
                pass
            elif policy and (policy.cmd_allow or policy.cmd_deny):
                if _cmd_matches(cmd, policy.cmd_deny):
                    verdict, reason = "denied", "denied by commands.deny"
                elif policy.cmd_allow and not _cmd_matches(cmd, policy.cmd_allow):
                    verdict, reason = "denied", "not in commands.allow"
                else:
                    verdict, reason = "allowed", "matches commands.allow"
            else:
                verdict, reason = "no_policy", "no command policy"

            if not network_denied:
                entries.append(AuditEntry(
                    event=event, event_index=index,
                    action=f"Ran: {cmd[:80]}{'...' if len(cmd) > 80 else ''}",
                    verdict=verdict, reason=reason,
                ))

    # --- Generic tool call (MCP tools, Agent, TodoWrite, etc.) ---
    # No policy rules cover arbitrary tool types; always no_policy regardless
    # of whether a policy file is loaded. Add explicit tool rules to the policy
    # file's "commands" section to cover these if needed.
    else:
        entries.append(AuditEntry(
            event=event, event_index=index,
            action=f"Tool: {str_field(data, 'tool_name', '?')}",
            verdict="no_policy",
            reason="no policy rule for this tool type",
        ))

    return entries


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# audit_session reads the data of every event of these types.
AUDIT_DATA_EVENT_TYPES = frozenset({EventType.TOOL_CALL})


def audit_session(
    store: TraceStore,
    session_id: str,
    policy_path: str | Path = ".agent-scope.json",
) -> AuditReport:
    """Audit all tool_call events in *session_id* against *policy_path*."""
    events = store.load_events(session_id)
    policy = Policy.load(policy_path)

    entries: list[AuditEntry] = []
    tool_call_count = 0

    for i, event in enumerate(events):
        if event.event_type != EventType.TOOL_CALL:
            continue
        tool_call_count += 1
        entries.extend(_audit_event(event, i + 1, policy))

    return AuditReport(
        session_id=session_id,
        total_events=len(events),
        total_tool_calls=tool_call_count,
        entries=entries,
        policy_loaded=policy is not None,
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_audit(report: AuditReport, out: TextIO = sys.stdout) -> None:
    """Write a human-readable audit report to *out*.

    Each action and reason goes through inert(), so an escape code or a line
    break from the trace or the policy file shows as text.
    """
    w = out.write
    policy_note = "" if report.policy_loaded else " (no policy file)"
    w(f"\nAUDIT: Session {report.session_id[:12]}"
      f" ({report.total_events} events, {report.total_tool_calls} tool calls)"
      f"{policy_note}\n\n")

    if report.allowed:
        w(f"✅ Allowed ({len(report.allowed)}):\n")
        for e in report.allowed[:20]:
            w(f"  {inert(e.action)}\n")
        if len(report.allowed) > 20:
            w(f"  ... and {len(report.allowed) - 20} more\n")
        w("\n")

    if report.no_policy:
        w(f"⚠️  No policy ({len(report.no_policy)}):\n")
        for e in report.no_policy[:20]:
            w(f"  {inert(e.action)}  ({inert(e.reason)})\n")
        if len(report.no_policy) > 20:
            w(f"  ... and {len(report.no_policy) - 20} more\n")
        w("\n")

    if report.denied:
        w(f"❌ Violations ({len(report.denied)}):\n")
        for e in report.denied:
            w(f"  {inert(e.action)}  ← {inert(e.reason)}\n")
        w("\n")

    if report.sensitive_accesses:
        w(f"🔐 Sensitive files accessed ({len(report.sensitive_accesses)}):\n")
        for e in report.sensitive_accesses:
            w(f"  {inert(e.action)}  (event #{e.event_index})\n")
        w("\n")

    if not report.denied and not report.sensitive_accesses:
        w("No violations found.\n\n")


# ---------------------------------------------------------------------------
# Hash-chain verification
# ---------------------------------------------------------------------------

@dataclass
class ChainVerifyResult:
    """The hash-chain check of one session.

    ``state`` is one of the hash-chain states in docs/trace-schema.md, or
    ``missing`` when the session has no events.ndjson. ``ok`` is true only
    for ``unanchored``: every event after the first links to the exact
    previous line. ``broken`` is true when a link does not match, a line
    can't be read, or events.ndjson is missing. A stream with missing links
    is neither ``ok`` nor ``broken``.
    """

    session_id: str
    ok: bool
    total_events: int
    broken_at: int | None = None   # 0-based index of first broken link
    broken_event_id: str = ""
    state: str = ""
    broken: bool = False
    # Links that are present and match, counted up to a broken link.
    links_verified: int = 0
    # 0-based index of the first event after the first with no prev_hash.
    first_unlinked_at: int | None = None
    # SHA-256 of the last line, as in the EU AI Act export. "" for no lines.
    root_hash: str = ""

    def format(self, out: TextIO = sys.stdout) -> None:
        """Write the result to *out*.

        The event ID comes from the event line and can be any JSON value, so
        it goes through inert() and an escape code in it shows as text.
        """
        if self.ok:
            out.write(f"✅ Chain intact — {self.total_events} events verified\n")
        elif self.state == "partial":
            out.write(
                f"❌ Chain incomplete: event #{self.first_unlinked_at} has no hash link "
                f"({self.links_verified} of {self.total_events - 1} links present)\n"
                "   An edit to the event before a missing link can't be detected.\n"
            )
        elif self.state == "legacy":
            out.write(
                f"❌ No hash links in {self.total_events} events\n"
                "   The session was recorded before v0.62.0, or its links were removed. "
                "An edit can't be detected.\n"
            )
        elif self.state == "empty":
            out.write("❌ No events to verify\n")
        elif self.state == "missing":
            out.write("❌ No event stream to verify: events.ndjson is missing\n")
        else:
            out.write(
                f"❌ Chain broken at event #{self.broken_at} "
                f"(id={inert(self.broken_event_id)})\n"
                f"   {self.total_events} events checked — session may have been tampered with\n"
            )


def verify_chain(store: TraceStore, session_id: str) -> ChainVerifyResult:
    """Verify the SHA-256 hash chain for a session's event log.

    Each event after the first stores prev_hash = SHA-256 of the previous
    event's exact JSON line. See verify_chain_lines for the states.
    """
    events_path = store._session_file(session_id, "events.ndjson", "event stream")
    if not events_path.exists():
        return ChainVerifyResult(
            session_id=session_id, ok=False, total_events=0,
            state="missing", broken=True,
        )

    # Split the bytes the way validate does, so both see the same lines.
    # surrogateescape keeps a line that is not UTF-8, so the check reports
    # it as broken instead of raising.
    lines = [
        line.decode("utf-8", "surrogateescape")
        for line in events_path.read_bytes().splitlines()
        if line.strip()
    ]
    return verify_chain_lines(lines, session_id=session_id)


def verify_chain_lines(
    lines: list[str],
    *,
    session_id: str = "",
) -> ChainVerifyResult:
    """Check the hash links in already-read event lines, one event per line.

    The first event must have no ``prev_hash``. Each later event must hold
    the SHA-256 of the exact previous line. The state is:

    - ``empty``: no lines
    - ``legacy``: two or more events and no links
    - ``partial``: some links missing, and every present link matches
    - ``unanchored``: every link present and matching. One event is
      ``unanchored``.
    - ``broken``: a link does not match, the first event has a
      ``prev_hash``, a ``prev_hash`` is not a string, or a line is not a
      UTF-8 JSON object. The check stops at that line.

    Only ``unanchored`` sets ``ok``. Only ``broken`` sets ``broken``.
    """
    import hashlib

    total = len(lines)
    root_hash = ""
    if lines:
        try:
            root_hash = hashlib.sha256(
                lines[-1].encode("utf-8", "surrogateescape")
            ).hexdigest()
        except UnicodeEncodeError:
            root_hash = ""

    links = 0
    unlinked = 0
    first_unlinked: int | None = None
    broken_at: int | None = None
    broken_event_id = ""
    prev_digest = ""
    for i, line in enumerate(lines):
        try:
            obj = json.loads(line)
        except (ValueError, RecursionError):
            broken_at = i
            break
        if not isinstance(obj, dict):
            broken_at = i
            break
        event_id = obj.get("event_id", "")
        stored = obj.get("prev_hash", "")
        try:
            raw = line.encode("utf-8")
        except UnicodeEncodeError:
            broken_at, broken_event_id = i, event_id
            break
        if i == 0:
            if stored != "":
                broken_at, broken_event_id = i, event_id
                break
        elif not isinstance(stored, str) or (stored and stored != prev_digest):
            broken_at, broken_event_id = i, event_id
            break
        elif stored:
            links += 1
        else:
            unlinked += 1
            if first_unlinked is None:
                first_unlinked = i
        prev_digest = hashlib.sha256(raw).hexdigest()

    if broken_at is not None:
        state = "broken"
    elif total == 0:
        state = "empty"
    elif unlinked == 0:
        state = "unanchored"
    elif unlinked == total - 1:
        state = "legacy"
    else:
        state = "partial"
    return ChainVerifyResult(
        session_id=session_id,
        ok=state == "unanchored",
        total_events=total,
        broken_at=broken_at,
        broken_event_id=broken_event_id,
        state=state,
        broken=state == "broken",
        links_verified=links,
        first_unlinked_at=first_unlinked,
        root_hash=root_hash,
    )


# ---------------------------------------------------------------------------
# CLI handler
# ---------------------------------------------------------------------------

def cmd_audit(args: argparse.Namespace) -> int:
    store = TraceStore(args.trace_dir)

    session_id = args.session_id
    if not session_id:
        session_id = store.get_latest_session_id()
    if not session_id:
        sys.stderr.write("No sessions found.\n")
        return 1
    full_id = store.find_session(session_id)
    if not full_id:
        sys.stderr.write(f"Session not found: {session_id}\n")
        return 1

    # --verify-chain: check hash chain integrity before policy audit
    chain_result = None
    if getattr(args, "verify_chain", False):
        chain_result = verify_chain(store, full_id)
        if not chain_result.ok:
            chain_result.format()
            return 1

    require_object_event_data(
        store.load_events(full_id), event_types=AUDIT_DATA_EVENT_TYPES
    )
    if chain_result is not None:
        chain_result.format()

    policy_path = getattr(args, "policy", ".agent-scope.json")
    report = audit_session(store, full_id, policy_path=policy_path)
    format_audit(report)

    # Exit 1 if there are violations so CI can catch them
    return 1 if report.denied else 0
