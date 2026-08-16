"""Backtest scope policies against stored agent sessions.

The evaluator intentionally reuses the permission-audit policy model and
matching helpers.  A backtest is read-only: it selects sessions from the
existing :class:`~agent_trace.store.TraceStore`, evaluates every tool call,
and returns structured reports suitable for terminal or JSON output.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TextIO

from .audit import (
    Policy,
    _audit_event,
    _cmd_matches,
    _extract_urls,
    _glob_match,
)
from .models import EventType, SessionMeta, TraceEvent
from .store import TraceStore


RuleEffect = Literal["allow", "deny", "default"]


class PolicyBacktestError(ValueError):
    """Raised when a policy cannot be loaded or a window is invalid."""


@dataclass(frozen=True)
class PolicyRule:
    """One explicit policy pattern, or the implicit unmatched default."""

    key: str
    label: str
    scope: str
    effect: RuleEffect
    pattern: str = ""


@dataclass
class PolicyRuleStats:
    """Observed outcomes attributed to a policy rule."""

    rule: PolicyRule
    matched: int = 0
    would_block: int = 0
    would_allow: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.rule.key,
            "rule": self.rule.label,
            "scope": self.rule.scope,
            "effect": self.rule.effect,
            "pattern": self.rule.pattern,
            "matched": self.matched,
            "would_block": self.would_block,
            "would_allow": self.would_allow,
        }


@dataclass(frozen=True)
class CallEvaluation:
    """Policy outcome for one stored tool call."""

    session_id: str
    event_id: str
    event_index: int
    action: str
    verdict: Literal["allowed", "denied"]
    reason: str
    rule_key: str | None = None
    rule_label: str = "default (no rule matched)"

    @property
    def covered(self) -> bool:
        return self.rule_key is not None

    @property
    def identity(self) -> tuple[str, int]:
        # Event IDs were optional in early sessions, while the session-local
        # event index is stable for the append-only event stream.
        return (self.session_id, self.event_index)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "event_id": self.event_id,
            "event_index": self.event_index,
            "action": self.action,
            "verdict": self.verdict,
            "reason": self.reason,
            "covered": self.covered,
            "rule": self.rule_label,
        }


@dataclass
class BacktestReport:
    """Aggregate result of simulating one policy over one history window."""

    policy_path: str
    days: int | None
    window_start: float | None
    window_end: float
    session_count: int
    calls: list[CallEvaluation] = field(default_factory=list)
    rules: list[PolicyRuleStats] = field(default_factory=list)

    @property
    def total_tool_calls(self) -> int:
        return len(self.calls)

    @property
    def covered_tool_calls(self) -> int:
        return sum(1 for call in self.calls if call.covered)

    @property
    def uncovered_tool_calls(self) -> int:
        return self.total_tool_calls - self.covered_tool_calls

    @property
    def coverage_percent(self) -> float:
        if not self.total_tool_calls:
            return 0.0
        return self.covered_tool_calls / self.total_tool_calls * 100.0

    @property
    def blocked_calls(self) -> list[CallEvaluation]:
        return [call for call in self.calls if call.verdict == "denied"]

    @property
    def affected_session_ids(self) -> list[str]:
        return sorted({call.session_id for call in self.blocked_calls})

    def to_dict(self) -> dict:
        return {
            "type": "policy_backtest",
            "policy": self.policy_path,
            "days": self.days,
            "window": {
                "start": self.window_start,
                "end": self.window_end,
            },
            "sessions": self.session_count,
            "tool_calls": self.total_tool_calls,
            "coverage": {
                "matched": self.covered_tool_calls,
                "uncovered": self.uncovered_tool_calls,
                "percent": round(self.coverage_percent, 2),
            },
            "would_block": {
                "tool_calls": len(self.blocked_calls),
                "sessions": len(self.affected_session_ids),
                "session_ids": self.affected_session_ids,
            },
            "rules": [rule.to_dict() for rule in self.rules],
        }


@dataclass
class PolicyDiffReport:
    """Behavioral difference between two policies on identical calls."""

    old: BacktestReport
    new: BacktestReport
    introduced: list[CallEvaluation]
    removed: list[CallEvaluation]

    @property
    def blocked_delta(self) -> int:
        return len(self.new.blocked_calls) - len(self.old.blocked_calls)

    @property
    def affected_sessions_delta(self) -> int:
        return (
            len(self.new.affected_session_ids)
            - len(self.old.affected_session_ids)
        )

    @staticmethod
    def _by_rule(calls: list[CallEvaluation]) -> list[dict]:
        counts = Counter(call.rule_label for call in calls)
        return [
            {"rule": rule, "tool_calls": count}
            for rule, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        ]

    def to_dict(self) -> dict:
        return {
            "type": "policy_diff",
            "old_policy": self.old.policy_path,
            "new_policy": self.new.policy_path,
            "days": self.old.days,
            "window": {
                "start": self.old.window_start,
                "end": self.old.window_end,
            },
            "sessions": self.old.session_count,
            "tool_calls": self.old.total_tool_calls,
            "old": {
                "blocked": len(self.old.blocked_calls),
                "affected_sessions": len(self.old.affected_session_ids),
            },
            "new": {
                "blocked": len(self.new.blocked_calls),
                "affected_sessions": len(self.new.affected_session_ids),
            },
            "delta": {
                "blocked": self.blocked_delta,
                "affected_sessions": self.affected_sessions_delta,
            },
            "new_blocks": {
                "tool_calls": len(self.introduced),
                "by_rule": self._by_rule(self.introduced),
                "calls": [call.to_dict() for call in self.introduced],
            },
            "blocks_removed": {
                "tool_calls": len(self.removed),
                "by_rule": self._by_rule(self.removed),
                "calls": [call.to_dict() for call in self.removed],
            },
        }


@dataclass
class CoverageReport:
    """Explicit-rule coverage derived from a backtest."""

    backtest: BacktestReport

    @property
    def total_tool_calls(self) -> int:
        return self.backtest.total_tool_calls

    @property
    def covered_tool_calls(self) -> int:
        return self.backtest.covered_tool_calls

    @property
    def uncovered_tool_calls(self) -> int:
        return self.backtest.uncovered_tool_calls

    @property
    def coverage_percent(self) -> float:
        return self.backtest.coverage_percent

    @property
    def uncovered(self) -> list[CallEvaluation]:
        return [call for call in self.backtest.calls if not call.covered]

    def to_dict(self, show_uncovered: bool = False) -> dict:
        data = {
            "type": "policy_coverage",
            "policy": self.backtest.policy_path,
            "days": self.backtest.days,
            "window": {
                "start": self.backtest.window_start,
                "end": self.backtest.window_end,
            },
            "sessions": self.backtest.session_count,
            "tool_calls": self.total_tool_calls,
            "covered": self.covered_tool_calls,
            "uncovered": self.uncovered_tool_calls,
            "coverage_percent": round(self.coverage_percent, 2),
        }
        if show_uncovered:
            data["uncovered_calls"] = [call.to_dict() for call in self.uncovered]
        return data


_DEFAULT_RULE = PolicyRule(
    key="default",
    label="default (no rule matched)",
    scope="default",
    effect="default",
)

_MALFORMED_ARGUMENTS_REASON = (
    "malformed tool call arguments; no policy rule evaluated"
)
_MALFORMED_DATA_REASON = "malformed tool call data; no policy rule evaluated"


def _rule_key(scope: str, effect: str, index: int) -> str:
    return f"{scope}.{effect}.{index}"


def _policy_rules(policy: Policy) -> list[PolicyRule]:
    """Return policy patterns in a stable, human-readable order."""
    rules: list[PolicyRule] = []

    def add(scope: str, verb: str, effect: Literal["allow", "deny"], patterns: list[str]) -> None:
        for index, pattern in enumerate(patterns):
            rules.append(PolicyRule(
                key=_rule_key(scope, effect, index),
                label=f"{effect}: {verb} {pattern}",
                scope=scope,
                effect=effect,
                pattern=pattern,
            ))

    add("files.read", "read", "deny", policy.file_read_deny)
    add("files.read", "read", "allow", policy.file_read_allow)
    add("files.write", "write", "deny", policy.file_write_deny)
    add("files.write", "write", "allow", policy.file_write_allow)
    add("commands", "bash", "deny", policy.cmd_deny)
    add("commands", "bash", "allow", policy.cmd_allow)
    add("network", "network", "allow", policy.network_allow)
    if policy.network_deny_all:
        rules.append(PolicyRule(
            key="network.deny_all",
            label="deny: network *",
            scope="network",
            effect="deny",
            pattern="*",
        ))
    return rules


def _first_path_rule(
    path: str,
    scope: str,
    effect: Literal["allow", "deny"],
    patterns: list[str],
) -> str | None:
    for index, pattern in enumerate(patterns):
        if _glob_match(path, [pattern]):
            return _rule_key(scope, effect, index)
    return None


def _first_command_rule(
    command: str,
    effect: Literal["allow", "deny"],
    patterns: list[str],
) -> str | None:
    for index, pattern in enumerate(patterns):
        if _cmd_matches(command, [pattern]):
            return _rule_key("commands", effect, index)
    return None


def _first_network_allow_rule(host: str, policy: Policy) -> str | None:
    for index, pattern in enumerate(policy.network_allow):
        if host == pattern or fnmatch.fnmatch(host, pattern):
            return _rule_key("network", "allow", index)
    return None


def _effective_rule(event: TraceEvent, policy: Policy) -> str | None:
    """Return the explicit rule that determines a tool call's outcome.

    A call is attributed to at most one rule so per-rule match totals remain
    directly comparable with the total number of tool calls.  Deny rules take
    precedence exactly as they do in ``audit --policy``.  Calls rejected only
    because they fall outside an allow-list are intentionally uncovered: no
    explicit pattern matched them.
    """
    tool_name = str(event.data.get("tool_name", "")).lower()
    args = event.data.get("arguments", {}) or {}

    if tool_name in ("read", "view", "grep", "glob"):
        path = str(
            args.get("file_path")
            or args.get("path")
            or args.get("pattern")
            or ""
        )
        if not path:
            return None
        return (
            _first_path_rule(path, "files.read", "deny", policy.file_read_deny)
            or _first_path_rule(path, "files.read", "allow", policy.file_read_allow)
        )

    if tool_name in ("write", "edit", "create"):
        path = str(args.get("file_path") or args.get("path") or "")
        if not path:
            return None
        return (
            _first_path_rule(path, "files.write", "deny", policy.file_write_deny)
            or _first_path_rule(path, "files.write", "allow", policy.file_write_allow)
        )

    if tool_name != "bash":
        return None

    command = str(args.get("command", "")).strip()
    if not command:
        return None

    network_fallback: str | None = None
    if policy.network_deny_all:
        for host in _extract_urls(command):
            allow_rule = _first_network_allow_rule(host, policy)
            if allow_rule is None:
                return "network.deny_all"
            if network_fallback is None:
                network_fallback = allow_rule

    deny_rule = _first_command_rule(command, "deny", policy.cmd_deny)
    if deny_rule:
        return deny_rule

    allow_rule = _first_command_rule(command, "allow", policy.cmd_allow)
    if allow_rule:
        return allow_rule

    # An unmatched command allow-list produces an implicit denial.  It must be
    # reported as uncovered even when the command's URL matched a network rule.
    if policy.cmd_allow:
        return None
    return network_fallback


def _evaluate_call(
    event: TraceEvent,
    event_index: int,
    session_id: str,
    policy: Policy,
    rules_by_key: dict[str, PolicyRule],
) -> CallEvaluation:
    if not isinstance(event.data, Mapping):
        return CallEvaluation(
            session_id=session_id,
            event_id=event.event_id,
            event_index=event_index,
            action="Tool: ? (malformed data)",
            verdict="allowed",
            reason=_MALFORMED_DATA_REASON,
        )

    data = event.data
    arguments = data.get("arguments", {})
    if arguments is not None and not isinstance(arguments, Mapping):
        tool_name = str(data.get("tool_name", "?"))
        return CallEvaluation(
            session_id=session_id,
            event_id=event.event_id,
            event_index=event_index,
            action=f"Tool: {tool_name} (malformed arguments)",
            verdict="allowed",
            reason=_MALFORMED_ARGUMENTS_REASON,
        )

    entries = _audit_event(event, event_index, policy)
    denied = next((entry for entry in entries if entry.verdict == "denied"), None)
    verdict: Literal["allowed", "denied"] = "denied" if denied else "allowed"
    selected = denied or (entries[-1] if entries else None)
    action = selected.action if selected else f"Tool: {event.data.get('tool_name', '?')}"
    reason = selected.reason if selected else "no policy rule matched"

    rule_key = _effective_rule(event, policy)
    rule = rules_by_key.get(rule_key or "")
    return CallEvaluation(
        session_id=session_id,
        event_id=event.event_id,
        event_index=event_index,
        action=action,
        verdict=verdict,
        reason=reason,
        rule_key=rule.key if rule else None,
        rule_label=rule.label if rule else _DEFAULT_RULE.label,
    )


def _policy_mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise PolicyBacktestError(f"policy field {field_name!r} must be an object")
    return value


def _policy_patterns(section: Mapping[str, object], key: str, field_name: str) -> list[str]:
    value = section.get(key, [])
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise PolicyBacktestError(
            f"policy field {field_name!r} must be a list of strings"
        )
    return value


def _load_policy(path: str | Path) -> Policy:
    """Load and validate a policy before evaluating historical actions."""
    policy_path = Path(path)
    try:
        raw = json.loads(policy_path.read_text())
    except FileNotFoundError as exc:
        raise PolicyBacktestError(f"Could not load policy file: {path}") from exc
    except OSError as exc:
        raise PolicyBacktestError(f"Could not load policy file: {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise PolicyBacktestError(f"Malformed policy file {path}: {exc}") from exc

    root = _policy_mapping(raw, "<root>")
    files = _policy_mapping(root.get("files", {}), "files")
    reads = _policy_mapping(files.get("read", {}), "files.read")
    writes = _policy_mapping(files.get("write", {}), "files.write")
    commands = _policy_mapping(root.get("commands", {}), "commands")
    network = _policy_mapping(root.get("network", {}), "network")

    deny_all = network.get("deny_all", False)
    if not isinstance(deny_all, bool):
        raise PolicyBacktestError("policy field 'network.deny_all' must be a boolean")

    return Policy(
        file_read_allow=_policy_patterns(reads, "allow", "files.read.allow"),
        file_read_deny=_policy_patterns(reads, "deny", "files.read.deny"),
        file_write_allow=_policy_patterns(writes, "allow", "files.write.allow"),
        file_write_deny=_policy_patterns(writes, "deny", "files.write.deny"),
        cmd_allow=_policy_patterns(commands, "allow", "commands.allow"),
        cmd_deny=_policy_patterns(commands, "deny", "commands.deny"),
        network_deny_all=deny_all,
        network_allow=_policy_patterns(network, "allow", "network.allow"),
    )


def _select_sessions(
    store: TraceStore,
    days: int | None,
    now: float | None,
) -> tuple[list[SessionMeta], float | None, float]:
    if days is not None and days <= 0:
        raise PolicyBacktestError("days must be greater than zero")
    window_end = time.time() if now is None else now
    window_start = None if days is None else window_end - days * 86400
    sessions = [
        meta for meta in store.list_sessions()
        if (window_start is None or meta.started_at >= window_start)
        and meta.started_at <= window_end
    ]
    return sessions, window_start, window_end


def _snapshot_sessions(
    store: TraceStore,
    sessions: list[SessionMeta],
) -> list[tuple[SessionMeta, list[TraceEvent]]]:
    """Load each selected event stream exactly once.

    Active sessions are append-only.  Keeping one in-memory snapshot ensures
    policy diffs compare the same calls even when new events arrive while the
    command is running.
    """
    return [
        (meta, store.load_events(meta.session_id))
        for meta in sessions
    ]


def _backtest_selected(
    policy: Policy,
    policy_path: str,
    snapshots: list[tuple[SessionMeta, list[TraceEvent]]],
    days: int | None,
    window_start: float | None,
    window_end: float,
) -> BacktestReport:
    declared_rules = _policy_rules(policy)
    rules_by_key = {rule.key: rule for rule in declared_rules}
    calls: list[CallEvaluation] = []

    for meta, events in snapshots:
        for event_index, event in enumerate(events, start=1):
            if event.event_type != EventType.TOOL_CALL:
                continue
            calls.append(_evaluate_call(
                event,
                event_index,
                meta.session_id,
                policy,
                rules_by_key,
            ))

    stats_by_key = {
        rule.key: PolicyRuleStats(rule=rule)
        for rule in [*declared_rules, _DEFAULT_RULE]
    }
    for call in calls:
        key = call.rule_key or _DEFAULT_RULE.key
        stats = stats_by_key[key]
        stats.matched += 1
        if call.verdict == "denied":
            stats.would_block += 1
        else:
            stats.would_allow += 1

    return BacktestReport(
        policy_path=policy_path,
        days=days,
        window_start=window_start,
        window_end=window_end,
        session_count=len(snapshots),
        calls=calls,
        rules=[stats_by_key[rule.key] for rule in [*declared_rules, _DEFAULT_RULE]],
    )


def backtest_policy(
    store: TraceStore,
    policy_path: str | Path,
    days: int | None = 30,
    now: float | None = None,
) -> BacktestReport:
    """Simulate *policy_path* over sessions in the requested window."""
    policy = _load_policy(policy_path)
    sessions, window_start, window_end = _select_sessions(store, days, now)
    snapshots = _snapshot_sessions(store, sessions)
    return _backtest_selected(
        policy,
        str(policy_path),
        snapshots,
        days,
        window_start,
        window_end,
    )


def diff_policies(
    store: TraceStore,
    old_policy_path: str | Path,
    new_policy_path: str | Path,
    days: int | None = 30,
    now: float | None = None,
) -> PolicyDiffReport:
    """Compare two policies against the exact same historical calls."""
    old_policy = _load_policy(old_policy_path)
    new_policy = _load_policy(new_policy_path)
    sessions, window_start, window_end = _select_sessions(store, days, now)
    snapshots = _snapshot_sessions(store, sessions)
    old_report = _backtest_selected(
        old_policy, str(old_policy_path), snapshots,
        days, window_start, window_end,
    )
    new_report = _backtest_selected(
        new_policy, str(new_policy_path), snapshots,
        days, window_start, window_end,
    )

    old_by_id = {call.identity: call for call in old_report.calls}
    new_by_id = {call.identity: call for call in new_report.calls}
    introduced = [
        call for identity, call in new_by_id.items()
        if call.verdict == "denied"
        and old_by_id[identity].verdict != "denied"
    ]
    removed = [
        call for identity, call in old_by_id.items()
        if call.verdict == "denied"
        and new_by_id[identity].verdict != "denied"
    ]
    return PolicyDiffReport(
        old=old_report,
        new=new_report,
        introduced=introduced,
        removed=removed,
    )


def policy_coverage(
    store: TraceStore,
    policy_path: str | Path,
    days: int | None = 30,
    now: float | None = None,
) -> CoverageReport:
    """Report the fraction of tool calls matching an explicit rule."""
    return CoverageReport(backtest_policy(store, policy_path, days=days, now=now))


def format_backtest(
    report: BacktestReport,
    out: TextIO = sys.stdout,
    show_sessions: bool = False,
) -> None:
    """Render a human-readable policy backtest."""
    period = "all history" if report.days is None else f"last {report.days} days"
    out.write(
        f"Policy backtest — {period} "
        f"({report.session_count:,} sessions, {report.total_tool_calls:,} tool calls)\n"
    )
    width = max(28, *(len(stats.rule.label) for stats in report.rules))
    divider = "─" * (width + 43)
    out.write(f"{divider}\n")
    out.write(
        f"{'Rule':<{width}}  {'Matched':>9}  {'Would block':>13}  {'Would allow':>13}\n"
    )
    out.write(f"{divider}\n")
    for stats in report.rules:
        out.write(
            f"{stats.rule.label:<{width}}  {stats.matched:>9,}  "
            f"{stats.would_block:>13,}  {stats.would_allow:>13,}\n"
        )
    out.write(f"{divider}\n")
    out.write(
        f"Coverage: {report.coverage_percent:.1f}% of tool calls matched an explicit rule\n"
    )
    out.write(
        f"Would block: {len(report.blocked_calls):,} tool calls across "
        f"{len(report.affected_session_ids):,} sessions\n"
    )
    if show_sessions and report.affected_session_ids:
        out.write("Affected sessions:\n")
        for session_id in report.affected_session_ids:
            out.write(f"  {session_id}\n")


def _signed(value: int) -> str:
    return f"{value:+d}"


def format_policy_diff(report: PolicyDiffReport, out: TextIO = sys.stdout) -> None:
    """Render a human-readable comparison of two policy simulations."""
    period = "all history" if report.old.days is None else f"last {report.old.days} days"
    out.write(f"Policy diff — old vs new ({period})\n")
    out.write("─" * 72 + "\n")
    out.write(f"{'Change':<34} {'Old policy':>11} {'New policy':>12} {'Delta':>9}\n")
    out.write("─" * 72 + "\n")
    out.write(
        f"{'Total blocked':<34} {len(report.old.blocked_calls):>11,} "
        f"{len(report.new.blocked_calls):>12,} { _signed(report.blocked_delta):>9}\n"
    )
    out.write(
        f"{'Sessions affected':<34} {len(report.old.affected_session_ids):>11,} "
        f"{len(report.new.affected_session_ids):>12,} "
        f"{_signed(report.affected_sessions_delta):>9}\n"
    )
    out.write(f"{'New blocks introduced':<34} {'—':>11} {len(report.introduced):>12,} {_signed(len(report.introduced)):>9}\n")
    for item in PolicyDiffReport._by_rule(report.introduced):
        out.write(f"  → {item['rule']:<52} {item['tool_calls']:>8,} new\n")
    out.write(f"{'Blocks removed':<34} {len(report.removed):>11,} {'—':>12} {_signed(-len(report.removed)):>9}\n")
    for item in PolicyDiffReport._by_rule(report.removed):
        out.write(f"  → {item['rule']:<52} {item['tool_calls']:>8,} removed\n")
    out.write("─" * 72 + "\n")
    if report.introduced:
        out.write(
            f"Recommendation: review {len(report.introduced):,} new blocks before enforcing\n"
        )
    else:
        out.write("No new blocks introduced.\n")


def format_coverage(
    report: CoverageReport,
    out: TextIO = sys.stdout,
    show_uncovered: bool = False,
) -> None:
    """Render a human-readable explicit-rule coverage report."""
    backtest = report.backtest
    period = "all history" if backtest.days is None else f"last {backtest.days} days"
    out.write(
        f"Policy coverage — {period} "
        f"({backtest.session_count:,} sessions)\n"
    )
    out.write(f"Tool calls: {report.total_tool_calls:,}\n")
    out.write(f"Explicitly covered: {report.covered_tool_calls:,}\n")
    out.write(f"Uncovered: {report.uncovered_tool_calls:,}\n")
    out.write(f"Coverage: {report.coverage_percent:.1f}%\n")
    if show_uncovered and report.uncovered:
        out.write("Uncovered tool calls:\n")
        for call in report.uncovered:
            out.write(
                f"  {call.session_id} event #{call.event_index}: {call.action}\n"
            )


def _days_arg(args: argparse.Namespace) -> int | None:
    return getattr(args, "days", 30)


def cmd_policy_backtest(args: argparse.Namespace) -> int:
    """CLI handler for ``policy backtest``."""
    try:
        report = backtest_policy(
            TraceStore(args.trace_dir),
            getattr(args, "policy", ".agent-scope.json"),
            days=_days_arg(args),
        )
    except (OSError, PolicyBacktestError) as exc:
        sys.stderr.write(f"Policy backtest failed: {exc}\n")
        return 1
    if getattr(args, "format", "text") == "json":
        sys.stdout.write(json.dumps(report.to_dict(), indent=2) + "\n")
    else:
        format_backtest(report, show_sessions=getattr(args, "show_sessions", False))
    return 0


def cmd_policy_diff(args: argparse.Namespace) -> int:
    """CLI handler for ``policy diff``."""
    try:
        report = diff_policies(
            TraceStore(args.trace_dir),
            args.old_policy,
            args.new_policy,
            days=_days_arg(args),
        )
    except (OSError, PolicyBacktestError) as exc:
        sys.stderr.write(f"Policy diff failed: {exc}\n")
        return 1
    if getattr(args, "format", "text") == "json":
        sys.stdout.write(json.dumps(report.to_dict(), indent=2) + "\n")
    else:
        format_policy_diff(report)
    return 0


def cmd_policy_coverage(args: argparse.Namespace) -> int:
    """CLI handler for ``policy coverage``."""
    try:
        report = policy_coverage(
            TraceStore(args.trace_dir),
            getattr(args, "policy", ".agent-scope.json"),
            days=_days_arg(args),
        )
    except (OSError, PolicyBacktestError) as exc:
        sys.stderr.write(f"Policy coverage failed: {exc}\n")
        return 1
    show_uncovered = getattr(args, "show_uncovered", False)
    if getattr(args, "format", "text") == "json":
        sys.stdout.write(
            json.dumps(report.to_dict(show_uncovered=show_uncovered), indent=2) + "\n"
        )
    else:
        format_coverage(report, show_uncovered=show_uncovered)
    return 0
