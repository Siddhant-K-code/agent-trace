"""Deterministic context-compaction analysis and checkpoint generation.

Compaction is inferred from consecutive LLM request contexts: when the input
token count drops by more than ``DEFAULT_COMPACTION_THRESHOLD``, the later
request is treated as the first request after compaction.  The implementation
uses stored events only; it never sends trace contents to an LLM or a remote
service.

Checkpoint files are additive Markdown sidecars under
``.agent-traces/checkpoints``.  Event NDJSON and session metadata are never
rewritten.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, TextIO

from .cost import get_model_pricing
from .models import EventType, TraceEvent
from .store import TraceStore
from .token_budget import DEFAULT_CONTEXT_LIMIT, _resolve_limit


DEFAULT_COMPACTION_THRESHOLD = 0.50
DEFAULT_CHECKPOINT_AT = 0.80
DEFAULT_BEHAVIOR_WINDOW = 50

_INPUT_TOKEN_KEYS = (
    "input_tokens",
    "prompt_tokens",
    "inputTokens",
    "promptTokenCount",
    "prompt_token_count",
)
_CACHE_TOKEN_KEYS = (
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "cached_input_tokens",
)
_USAGE_KEYS = ("usage", "usage_metadata", "usageMetadata", "token_usage")
_CONTEXT_LIMIT_KEYS = (
    "context_window",
    "context_window_tokens",
    "max_context_tokens",
)
_READ_TOOLS = {"read", "read_file", "file_read"}
_WRITE_TOOLS = {
    "write",
    "edit",
    "create",
    "str_replace",
    "str_replace_based_edit_tool",
    "multiedit",
    "notebook_edit",
}
_CONSTRAINT_RE = re.compile(
    r"\b(?:must|shall|required?|requirement|should|need(?:s)? to|"
    r"do not|don't|never|only|ensure|cannot|can't|without)\b",
    re.IGNORECASE,
)
_DECISION_RE = re.compile(
    r"\b(?:decid(?:e|ed|ing)|chose|chosen|select(?:ed)?|"
    r"reject(?:ed)?|abandon(?:ed)?|avoid(?:ed)?|instead of)\b",
    re.IGNORECASE,
)
_EXPLORATION_RE = re.compile(
    r"^\s*(?:ls(?:\s|$)|find(?:\s|$)|tree(?:\s|$)|"
    r"rg\s+--files(?:\s|$)|git\s+ls-files(?:\s|$))",
    re.IGNORECASE,
)
_SPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[a-z0-9][a-z0-9_.:/-]*")
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "was", "were", "with",
}


@dataclass
class ContextItem:
    """One reconstructable fact that may survive or be lost at compaction."""

    kind: str
    text: str
    event_index: int
    high_risk: bool = False


@dataclass
class CompactionContextDiff:
    survived: list[ContextItem] = field(default_factory=list)
    likely_dropped: list[ContextItem] = field(default_factory=list)
    context_available: bool = True

    @property
    def high_risk_drops(self) -> list[ContextItem]:
        return [item for item in self.likely_dropped if item.high_risk]


@dataclass(frozen=True)
class BehaviorMetrics:
    unique_files_read: int
    redundant_read_violations: int
    tool_loop_violations: int
    reexploration_calls: int
    files_read: tuple[str, ...] = ()
    context_saturation_violations: int = 0
    lint_violations: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class CompactionBehaviorDiff:
    before: BehaviorMetrics
    after: BehaviorMetrics
    reread_files: tuple[str, ...] = ()
    regressions: tuple[str, ...] = ()
    lint_deltas: tuple[tuple[str, int, int], ...] = ()

    @property
    def regressed(self) -> bool:
        return bool(self.regressions)


@dataclass
class CompactionEvent:
    sequence: int
    event_index: int
    before_event_index: int
    before_request_index: int
    after_request_index: int
    timestamp: float
    offset_seconds: float
    tokens_before: int
    tokens_after: int
    tokens_dropped: int
    drop_ratio: float
    model: str
    estimated_cost_usd: float
    stream_id: str = "main"
    context_boundary_index: int = -1
    post_context: Mapping[str, Any] = field(default_factory=dict, repr=False)
    context_diff: CompactionContextDiff | None = None
    behavior_diff: CompactionBehaviorDiff | None = None


@dataclass
class CompactionReport:
    session_id: str
    threshold: float
    events: list[CompactionEvent] = field(default_factory=list)

    @property
    def total_tokens_dropped(self) -> int:
        return sum(event.tokens_dropped for event in self.events)

    @property
    def estimated_cost_usd(self) -> float:
        return sum(event.estimated_cost_usd for event in self.events)


@dataclass
class _RequestSample:
    request_index: int
    event_index: int
    timestamp: float
    input_tokens: int
    model: str
    context_data: Mapping[str, Any]
    estimated: bool = False
    stream_id: str = "main"
    context_boundary_index: int = -1


def _event_stream(event: TraceEvent) -> str:
    value = event.data.get("context_stream", "main")
    return str(value) if value else "main"


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number > 0 else None


def _first_token_value(data: Mapping[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        if key in data:
            value = _positive_int(data[key])
            if value is not None:
                return value
    return None


def _input_token_count(data: Mapping[str, Any]) -> int | None:
    """Read a provider token count, including Anthropic cache context."""
    candidates: list[Mapping[str, Any]] = [data]
    for key in _USAGE_KEYS:
        nested = data.get(key)
        if isinstance(nested, Mapping):
            candidates.append(nested)
    response_metadata = data.get("response_metadata")
    if isinstance(response_metadata, Mapping):
        nested = response_metadata.get("token_usage")
        if isinstance(nested, Mapping):
            candidates.append(nested)

    for candidate in candidates:
        primary = _first_token_value(candidate, _INPUT_TOKEN_KEYS)
        cache = sum(
            value or 0
            for value in (_positive_int(candidate.get(key)) for key in _CACHE_TOKEN_KEYS)
        )
        if primary is not None or cache:
            return (primary or 0) + cache
    return None


def _model_name(data: Mapping[str, Any]) -> str:
    for key in ("model", "model_id", "modelId"):
        value = data.get(key)
        if value:
            return str(value)
    return ""


def _estimate_request_tokens(data: Mapping[str, Any]) -> int:
    ignored = set(_INPUT_TOKEN_KEYS) | set(_CACHE_TOKEN_KEYS) | set(_USAGE_KEYS)
    payload = {key: value for key, value in data.items() if key not in ignored}
    return max(1, len(json.dumps(payload, sort_keys=True, default=str)) // 4)


def _request_samples(events: list[TraceEvent]) -> list[_RequestSample]:
    """Build logical request samples from request/paired response events.

    Several integrations learn provider usage only after the response.  Those
    counts replace a request-side estimate.  Response-only samples also allow
    imported logs to participate when their source exposes one usage-bearing
    assistant event per turn.
    """
    samples: list[_RequestSample] = []
    pending_by_stream: dict[str, int] = {}
    summary_by_stream: dict[str, int] = {}

    for event_index, event in enumerate(events):
        stream_id = _event_stream(event)
        if event.data.get("is_compaction_summary"):
            summary_by_stream[stream_id] = event_index

        if event.event_type == EventType.LLM_REQUEST:
            explicit = _input_token_count(event.data)
            boundary_index = summary_by_stream.pop(stream_id, event_index)
            samples.append(_RequestSample(
                request_index=len(samples) + 1,
                event_index=event_index,
                timestamp=event.timestamp,
                input_tokens=explicit or _estimate_request_tokens(event.data),
                model=_model_name(event.data),
                context_data=event.data,
                estimated=explicit is None,
                stream_id=stream_id,
                context_boundary_index=boundary_index,
            ))
            pending_by_stream[stream_id] = len(samples) - 1
            continue

        if event.event_type not in {EventType.LLM_RESPONSE, EventType.ASSISTANT_RESPONSE}:
            continue

        explicit = _input_token_count(event.data)
        if explicit is None:
            continue
        pending_index = pending_by_stream.get(stream_id)
        if pending_index is not None:
            sample = samples[pending_index]
            sample.input_tokens = explicit
            sample.estimated = False
            if not sample.model:
                sample.model = _model_name(event.data)
            summary = event.data.get("context_summary")
            if isinstance(summary, str) and summary:
                sample.context_data = {"context_summary": summary}
                sample.context_boundary_index = summary_by_stream.pop(
                    stream_id, sample.context_boundary_index,
                )
            pending_by_stream.pop(stream_id, None)
        else:
            summary = event.data.get("context_summary")
            context_data = (
                {"context_summary": summary}
                if isinstance(summary, str) and summary else {}
            )
            samples.append(_RequestSample(
                request_index=len(samples) + 1,
                event_index=event_index,
                timestamp=event.timestamp,
                input_tokens=explicit,
                model=_model_name(event.data),
                context_data=context_data,
                estimated=False,
                stream_id=stream_id,
                context_boundary_index=summary_by_stream.pop(stream_id, event_index),
            ))
    return samples


def detect_compactions(
    events: list[TraceEvent],
    threshold: float = DEFAULT_COMPACTION_THRESHOLD,
) -> list[CompactionEvent]:
    """Return compactions inferred from consecutive logical LLM requests."""
    if not 0 < threshold < 1:
        raise ValueError("compaction threshold must be between 0 and 1")

    samples = _request_samples(events)
    base_timestamp = events[0].timestamp if events else 0.0
    compactions: list[CompactionEvent] = []
    previous_by_stream: dict[str, _RequestSample] = {}
    for after in samples:
        before = previous_by_stream.get(after.stream_id)
        previous_by_stream[after.stream_id] = after
        if before is None:
            continue
        # JSON-size estimates are useful for checkpointing but are not precise
        # enough to claim or price a compaction event.
        if before.estimated or after.estimated:
            continue
        if before.input_tokens <= 0 or after.input_tokens >= before.input_tokens:
            continue
        dropped = before.input_tokens - after.input_tokens
        drop_ratio = dropped / before.input_tokens
        if drop_ratio <= threshold:
            continue
        model = after.model or before.model
        price = get_model_pricing(model)
        input_rate = price.input_per_million if price is not None else 3.0
        compactions.append(CompactionEvent(
            sequence=len(compactions) + 1,
            event_index=after.event_index,
            before_event_index=before.event_index,
            before_request_index=before.request_index,
            after_request_index=after.request_index,
            timestamp=after.timestamp,
            offset_seconds=max(0.0, after.timestamp - base_timestamp),
            tokens_before=before.input_tokens,
            tokens_after=after.input_tokens,
            tokens_dropped=dropped,
            drop_ratio=drop_ratio,
            model=model or "unknown",
            estimated_cost_usd=dropped * input_rate / 1_000_000,
            stream_id=after.stream_id,
            context_boundary_index=(
                after.context_boundary_index
                if after.context_boundary_index >= 0 else after.event_index
            ),
            post_context=after.context_data,
        ))
    return compactions


def _flatten_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        strings: list[str] = []
        for key, nested in value.items():
            if key in _USAGE_KEYS or key in _INPUT_TOKEN_KEYS or key in _CACHE_TOKEN_KEYS:
                continue
            strings.extend(_flatten_strings(nested))
        return strings
    if isinstance(value, (list, tuple)):
        strings = []
        for nested in value:
            strings.extend(_flatten_strings(nested))
        return strings
    return []


def _event_text(event: TraceEvent) -> str:
    preferred = (
        "prompt", "text", "content", "summary", "decision", "reason",
        "message", "task", "goal",
    )
    for key in preferred:
        value = event.data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return " ".join(_flatten_strings(event.data)).strip()


def _statements(text: str) -> list[str]:
    parts = re.split(r"(?:\r?\n)+|(?<=[.!?])\s+", text)
    result: list[str] = []
    for part in parts:
        cleaned = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", part).strip()
        cleaned = _SPACE_RE.sub(" ", cleaned)
        if len(cleaned) >= 4:
            result.append(cleaned[:400])
    return result


def _read_path(event: TraceEvent) -> str:
    if event.event_type == EventType.FILE_READ:
        return str(
            event.data.get("path") or event.data.get("file_path")
            or event.data.get("uri") or ""
        ).strip()
    if event.event_type != EventType.TOOL_CALL:
        return ""
    tool = str(event.data.get("tool_name", "")).lower()
    if tool not in _READ_TOOLS:
        return ""
    arguments = event.data.get("arguments", {})
    if not isinstance(arguments, Mapping):
        return ""
    return str(arguments.get("file_path") or arguments.get("path") or "").strip()


def _write_path(event: TraceEvent) -> str:
    if event.event_type == EventType.FILE_WRITE:
        return str(
            event.data.get("path") or event.data.get("file_path")
            or event.data.get("uri") or ""
        ).strip()
    if event.event_type != EventType.TOOL_CALL:
        return ""
    tool = str(event.data.get("tool_name", "")).lower()
    if tool not in _WRITE_TOOLS:
        return ""
    arguments = event.data.get("arguments", {})
    if not isinstance(arguments, Mapping):
        return ""
    return str(arguments.get("file_path") or arguments.get("path") or "").strip()


def _normalise(value: str) -> str:
    return " ".join(_WORD_RE.findall(value.lower()))


def _context_items(
    events: list[TraceEvent],
    end_index: int,
    stream_id: str | None = None,
) -> list[ContextItem]:
    items: list[ContextItem] = []
    task_added = False
    seen: set[tuple[str, str]] = set()

    def add(kind: str, text: str, event_index: int, high_risk: bool = False) -> None:
        text = _SPACE_RE.sub(" ", text).strip()[:400]
        key = (kind, _normalise(text))
        if not text or not key[1] or key in seen:
            return
        seen.add(key)
        items.append(ContextItem(kind, text, event_index, high_risk))

    for event_index, event in enumerate(events[:end_index]):
        if stream_id is not None and _event_stream(event) != stream_id:
            continue
        path = _read_path(event)
        if path:
            add("File read", path, event_index)
        path = _write_path(event)
        if path:
            add("File modified", path, event_index)

        text = _event_text(event)
        if event.event_type == EventType.USER_PROMPT and text:
            statements = _statements(text)
            if statements and not task_added:
                add("Task goal", statements[0], event_index)
                task_added = True
            for statement in statements:
                if _CONSTRAINT_RE.search(statement):
                    add("Constraint", statement, event_index, high_risk=True)

        if event.event_type == EventType.LLM_REQUEST and text:
            for statement in _statements(text):
                if _CONSTRAINT_RE.search(statement):
                    add("Constraint", statement, event_index, high_risk=True)

        if event.event_type == EventType.DECISION and text:
            add("Decision", text, event_index, high_risk=True)
        elif event.event_type == EventType.ASSISTANT_RESPONSE and text:
            for statement in _statements(text):
                if _DECISION_RE.search(statement):
                    add("Decision", statement, event_index, high_risk=True)
    return items


def _item_survived(
    item: ContextItem,
    context_text: str,
    context_words: set[str],
) -> bool:
    item_text = _normalise(item.text)
    if not item_text or not context_text:
        return False
    if item_text in context_text:
        return True
    if item.kind in {"Task goal", "Decision"}:
        ordered = [
            word for word in item_text.split()
            if word not in _STOP_WORDS and len(word) > 2
        ]
        if any(
            f"{left} {right}" in context_text
            for left, right in zip(ordered, ordered[1:])
        ):
            return True
    significant = {
        word for word in item_text.split()
        if word not in _STOP_WORDS and len(word) > 2
    }
    if len(significant) < 3:
        return False
    return len(significant & context_words) / len(significant) >= 0.80


def context_diff_for_compaction(
    events: list[TraceEvent],
    compaction: CompactionEvent,
) -> CompactionContextDiff:
    """Infer which reconstructable pre-compaction facts survived."""
    boundary_index = (
        compaction.context_boundary_index
        if compaction.context_boundary_index >= 0 else compaction.event_index
    )
    items = _context_items(events, boundary_index, compaction.stream_id)
    post_context = " ".join(_flatten_strings(compaction.post_context))
    context_text = _normalise(post_context)
    if not context_text:
        return CompactionContextDiff(context_available=False)
    context_words = set(context_text.split())
    diff = CompactionContextDiff(context_available=True)
    for item in items:
        if _item_survived(item, context_text, context_words):
            diff.survived.append(item)
        else:
            diff.likely_dropped.append(item)
    return diff


def _tool_loop_count(events: list[TraceEvent], threshold: int) -> int:
    count = 0
    run_tool = ""
    run_length = 0
    for event in events + [TraceEvent(event_type=EventType.SESSION_END)]:
        if event.event_type == EventType.TOOL_CALL:
            tool = str(event.data.get("tool_name", ""))
            if tool and tool == run_tool:
                run_length += 1
            else:
                if run_length >= threshold:
                    count += 1
                run_tool = tool
                run_length = 1 if tool else 0
        else:
            if run_length >= threshold:
                count += 1
            run_tool = ""
            run_length = 0
    return count


def _behavior_metrics(
    events: list[TraceEvent],
    redundant_read_threshold: int,
    tool_loop_threshold: int,
) -> BehaviorMetrics:
    paths = [path for event in events if (path := _read_path(event))]
    counts: dict[str, int] = {}
    for path in paths:
        counts[path] = counts.get(path, 0) + 1

    reexploration_calls = 0
    for event in events:
        if event.event_type != EventType.TOOL_CALL:
            continue
        tool = str(event.data.get("tool_name", "")).lower()
        arguments = event.data.get("arguments", {})
        arguments = arguments if isinstance(arguments, Mapping) else {}
        command = str(arguments.get("command", ""))
        if tool in {"glob", "list", "list_dir", "directory_tree"}:
            reexploration_calls += 1
        elif tool in {"bash", "shell", "exec"} and _EXPLORATION_RE.search(command):
            reexploration_calls += 1

    # Reuse the registered lint rules so behavior diffs evolve with the lint
    # command.  The compaction rule itself is excluded to avoid recursion.
    from . import lint as lint_module

    lint_counts: dict[str, int] = {}
    for rule_name, rule_fn in lint_module._RULES.items():
        if rule_name == "post-compaction-regression":
            continue
        rule_config = dict(lint_module.DEFAULT_CONFIG.get(rule_name, {}))
        if rule_name == "redundant-read":
            rule_config["threshold"] = redundant_read_threshold
        elif rule_name == "tool-loop":
            rule_config["threshold"] = tool_loop_threshold
        try:
            lint_counts[rule_name] = len(rule_fn(events, rule_config))
        except Exception:
            # Match lint_session's rule isolation without printing repeatedly
            # while generating a behavior diff.
            lint_counts[rule_name] = 0

    explicit_context = [
        context_fill_ratio(event)
        for event in events
        if _input_token_count(event.data) is not None
    ]
    if explicit_context:
        saturation_threshold = float(
            lint_module.DEFAULT_CONFIG["context-saturation"].get("threshold", 0.8)
        )
        lint_counts["context-saturation"] = int(any(
            ratio is not None and ratio >= saturation_threshold
            for ratio in explicit_context
        ))

    return BehaviorMetrics(
        unique_files_read=len(set(paths)),
        redundant_read_violations=sum(
            1 for count in counts.values() if count >= redundant_read_threshold
        ),
        tool_loop_violations=_tool_loop_count(events, tool_loop_threshold),
        reexploration_calls=reexploration_calls,
        files_read=tuple(dict.fromkeys(paths)),
        context_saturation_violations=lint_counts.get("context-saturation", 0),
        lint_violations=tuple(sorted(lint_counts.items())),
    )


def behavior_diff_for_compaction(
    events: list[TraceEvent],
    compaction: CompactionEvent,
    window: int = DEFAULT_BEHAVIOR_WINDOW,
    redundant_read_threshold: int = 3,
    tool_loop_threshold: int = 5,
) -> CompactionBehaviorDiff:
    """Compare bounded tool behavior before and after one compaction."""
    if window <= 0:
        raise ValueError("behavior window must be positive")
    index = (
        compaction.context_boundary_index
        if compaction.context_boundary_index >= 0 else compaction.event_index
    )
    before_events = [
        event for event in events[:index]
        if _event_stream(event) == compaction.stream_id
    ][-window:]
    after_events = [
        event for event in events[index:]
        if _event_stream(event) == compaction.stream_id
    ][:window]
    before = _behavior_metrics(
        before_events, redundant_read_threshold, tool_loop_threshold,
    )
    after = _behavior_metrics(
        after_events, redundant_read_threshold, tool_loop_threshold,
    )
    all_prior_paths = {
        _read_path(event) for event in events[:index]
        if _event_stream(event) == compaction.stream_id
    }
    all_prior_paths.discard("")
    reread_files = tuple(sorted(all_prior_paths & set(after.files_read)))

    regressions: list[str] = []
    if reread_files:
        regressions.append(f"re-read {len(reread_files)} pre-compaction file(s)")
    if after.reexploration_calls > before.reexploration_calls:
        regressions.append(
            "repository exploration calls increased "
            f"{before.reexploration_calls} -> {after.reexploration_calls}"
        )

    before_lint = dict(before.lint_violations)
    after_lint = dict(after.lint_violations)
    lint_deltas = tuple(
        (rule, before_lint.get(rule, 0), after_lint.get(rule, 0))
        for rule in sorted(before_lint.keys() | after_lint.keys())
    )
    for rule, before_count, after_count in lint_deltas:
        if after_count > before_count:
            regressions.append(
                f"{rule} lint violations increased {before_count} -> {after_count}"
            )

    return CompactionBehaviorDiff(
        before=before,
        after=after,
        reread_files=reread_files,
        regressions=tuple(regressions),
        lint_deltas=lint_deltas,
    )


def analyze_compactions(
    events: list[TraceEvent],
    session_id: str = "",
    threshold: float = DEFAULT_COMPACTION_THRESHOLD,
    behavior_window: int = DEFAULT_BEHAVIOR_WINDOW,
) -> CompactionReport:
    """Build a complete deterministic compaction report from events."""
    report = CompactionReport(session_id=session_id, threshold=threshold)
    report.events = detect_compactions(events, threshold)
    for compaction in report.events:
        compaction.context_diff = context_diff_for_compaction(events, compaction)
        compaction.behavior_diff = behavior_diff_for_compaction(
            events, compaction, window=behavior_window,
        )
    return report


def analyse_compactions(
    events: list[TraceEvent],
    session_id: str = "",
    threshold: float = DEFAULT_COMPACTION_THRESHOLD,
    behavior_window: int = DEFAULT_BEHAVIOR_WINDOW,
) -> CompactionReport:
    """British-English alias used by other analysis modules."""
    return analyze_compactions(events, session_id, threshold, behavior_window)


def analyse_compaction_session(
    store: TraceStore,
    session_id: str,
    threshold: float = DEFAULT_COMPACTION_THRESHOLD,
    behavior_window: int = DEFAULT_BEHAVIOR_WINDOW,
) -> CompactionReport:
    return analyze_compactions(
        store.load_events(session_id), session_id, threshold, behavior_window,
    )


def _format_offset(seconds: float) -> str:
    minutes, seconds = divmod(max(0, int(seconds)), 60)
    return f"{minutes}m {seconds:02d}s"


def format_compaction_report(
    report: CompactionReport,
    out: TextIO = sys.stdout,
    show_diff: bool = False,
    show_behavior_diff: bool = False,
) -> None:
    """Write a stable human-readable report."""
    if not report.events:
        out.write(f"No compaction events detected in session {report.session_id[:12]}.\n")
        return

    out.write(f"Compaction events - session {report.session_id[:12]}\n")
    out.write("─" * 78 + "\n")
    out.write("Event  Timestamp  Tokens before  Tokens after  Dropped\n")
    out.write("─" * 78 + "\n")
    for event in report.events:
        out.write(
            f"#{event.sequence:<5} {_format_offset(event.offset_seconds):<10} "
            f"{event.tokens_before:>13,}  {event.tokens_after:>12,}  "
            f"{event.tokens_dropped:>10,} ({event.drop_ratio * 100:.0f}%)\n"
        )
    out.write("─" * 78 + "\n")
    out.write(
        f"{len(report.events)} compaction event(s). "
        f"{report.total_tokens_dropped:,} tokens dropped total.\n"
        f"Estimated cost of dropped context: ${report.estimated_cost_usd:.4f}\n"
    )

    if show_diff:
        for event in report.events:
            diff = event.context_diff or CompactionContextDiff()
            out.write(f"\nCompaction #{event.sequence} - context diff\n")
            out.write("─" * 78 + "\n")
            if not diff.context_available:
                out.write(
                    "Post-compaction summary/context was not captured; "
                    "survival cannot be assessed.\n"
                )
                continue
            out.write("Survived in summary:\n")
            if diff.survived:
                for item in diff.survived:
                    out.write(f"  ✅ {item.kind}: {item.text}\n")
            else:
                out.write("  (none detected)\n")
            out.write("Likely dropped:\n")
            if diff.likely_dropped:
                for item in diff.likely_dropped:
                    out.write(f"  ❌ {item.kind} at event #{item.event_index + 1}: {item.text}\n")
            else:
                out.write("  (none detected)\n")
            for item in diff.high_risk_drops:
                out.write(f"⚠️  High-risk drop: {item.kind.lower()} {item.text!r}\n")

    if show_behavior_diff:
        for event in report.events:
            behavior = event.behavior_diff
            if behavior is None:
                continue
            out.write(f"\nBehavior change after compaction #{event.sequence}\n")
            out.write("─" * 78 + "\n")
            out.write(
                f"Before: {behavior.before.unique_files_read} unique files read, "
                f"{behavior.before.redundant_read_violations} redundant-read violations\n"
                f"After:  {behavior.after.unique_files_read} unique files read, "
                f"{behavior.after.redundant_read_violations} redundant-read violations\n"
                "Lint delta:\n"
            )
            for rule, before_count, after_count in behavior.lint_deltas:
                out.write(f"  {rule}: {before_count} -> {after_count}\n")
            for regression in behavior.regressions:
                out.write(f"  - {regression}\n")
            verdict = "regressed" if behavior.regressed else "no measurable regression"
            out.write(f"Verdict: behavior {verdict} after compaction #{event.sequence}\n")


def _context_limit(data: Mapping[str, Any]) -> int:
    for key in _CONTEXT_LIMIT_KEYS:
        limit = _positive_int(data.get(key))
        if limit is not None:
            return limit
    return _resolve_limit(_model_name(data)) or DEFAULT_CONTEXT_LIMIT


def context_fill_ratio(event: TraceEvent) -> float | None:
    """Return current-request context fill, or ``None`` without token usage."""
    if event.event_type not in {
        EventType.LLM_REQUEST, EventType.LLM_RESPONSE, EventType.ASSISTANT_RESPONSE,
    }:
        return None
    input_tokens = _input_token_count(event.data)
    if input_tokens is None:
        return None
    return input_tokens / _context_limit(event.data)


def build_checkpoint_markdown(
    events: list[TraceEvent],
    session_id: str,
    timestamp: float | None = None,
) -> str:
    """Summarize recoverable session state as deterministic Markdown."""
    if timestamp is None:
        timestamp = events[-1].timestamp if events else 0.0
    base_timestamp = events[0].timestamp if events else timestamp
    offset = max(0.0, timestamp - base_timestamp)
    items = _context_items(events, len(events))
    task = next((item.text for item in items if item.kind == "Task goal"), "Not detected")
    constraints = [item.text for item in items if item.kind == "Constraint"]
    decisions = [item.text for item in items if item.kind == "Decision"]

    read_entries: list[tuple[str, float]] = []
    seen_reads: set[str] = set()
    modified: list[str] = []
    seen_modified: set[str] = set()
    for event in events:
        path = _read_path(event)
        if path and path not in seen_reads:
            seen_reads.add(path)
            read_entries.append((path, max(0.0, event.timestamp - base_timestamp)))
        path = _write_path(event)
        if path and path not in seen_modified:
            seen_modified.add(path)
            modified.append(path)

    last_response = next(
        (
            _event_text(event) for event in reversed(events)
            if event.event_type in {EventType.ASSISTANT_RESPONSE, EventType.LLM_RESPONSE}
            and _event_text(event)
        ),
        "",
    )
    current_parts: list[str] = []
    if modified:
        current_parts.append("Modified files: " + ", ".join(modified))
    if last_response:
        current_parts.append("Latest agent state: " + _SPACE_RE.sub(" ", last_response)[:400])
    if not current_parts:
        current_parts.append("No file modifications or agent response detected yet")

    lines = [
        f"# Session checkpoint - {session_id} - {_format_offset(offset)}",
        "",
        "## Task",
        task,
        "",
        "## Constraints",
    ]
    lines.extend(f"- {value}" for value in constraints)
    if not constraints:
        lines.append("- None detected")
    lines.extend(["", "## Files read"])
    lines.extend(
        f"- {path} (read at {_format_offset(read_offset)})"
        for path, read_offset in read_entries
    )
    if not read_entries:
        lines.append("- None detected")
    lines.extend(["", "## Decisions made"])
    lines.extend(f"- {value}" for value in decisions)
    if not decisions:
        lines.append("- None detected")
    lines.extend(["", "## Current state"])
    lines.extend(f"- {value}" for value in current_parts)
    return "\n".join(lines) + "\n"


def write_checkpoint(
    store: TraceStore,
    session_id: str,
    events: list[TraceEvent] | None = None,
    timestamp: float | None = None,
) -> Path:
    """Write ``checkpoints/<session>.md`` without touching trace storage."""
    if events is None:
        events = store.load_events(session_id)
    path = store.checkpoint_path(session_id)
    checkpoint_dir = path.parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(
        build_checkpoint_markdown(events, session_id, timestamp), encoding="utf-8",
    )
    return path


@dataclass
class CompactionCheckpointWatcher:
    """Stateful watch-loop adapter that writes once per context-growth cycle."""

    store: TraceStore
    session_id: str
    checkpoint_at: float = DEFAULT_CHECKPOINT_AT
    compaction_threshold: float = DEFAULT_COMPACTION_THRESHOLD
    events: list[TraceEvent] = field(default_factory=list)
    _armed: bool = field(default=True, init=False, repr=False)
    _last_tokens: int | None = field(default=None, init=False, repr=False)
    _last_tokens_by_stream: dict[str, int] = field(
        default_factory=dict, init=False, repr=False,
    )
    _pending_previous_by_stream: dict[str, int | None] = field(
        default_factory=dict, init=False, repr=False,
    )

    def __post_init__(self) -> None:
        if not 0 < self.checkpoint_at <= 1:
            raise ValueError("checkpoint threshold must be between 0 and 1")
        if not 0 < self.compaction_threshold < 1:
            raise ValueError("compaction threshold must be between 0 and 1")
        if not self.events:
            self.events = self.store.load_events(self.session_id)
        samples = _request_samples(self.events)
        if samples:
            self._last_tokens = samples[-1].input_tokens
            for sample in samples:
                if not sample.estimated:
                    self._last_tokens_by_stream[sample.stream_id] = sample.input_tokens

        # Preserve a request/response pairing that was still open when watch
        # attached.  Subsequent updates stay O(1) instead of reconstructing all
        # request samples for every tailed LLM event.
        previous_exact: dict[str, int] = {}
        for existing in self.events:
            stream_id = _event_stream(existing)
            if existing.event_type == EventType.LLM_REQUEST:
                self._pending_previous_by_stream[stream_id] = previous_exact.get(
                    stream_id
                )
                explicit = _input_token_count(existing.data)
                if explicit is not None:
                    previous_exact[stream_id] = explicit
            elif existing.event_type in {
                EventType.LLM_RESPONSE, EventType.ASSISTANT_RESPONSE,
            }:
                explicit = _input_token_count(existing.data)
                if explicit is not None:
                    self._pending_previous_by_stream.pop(stream_id, None)
                    previous_exact[stream_id] = explicit

    def checkpoint_current(self) -> Path | None:
        """Checkpoint an already-saturated session when watch starts."""
        samples = _request_samples(self.events)
        if not samples or not self._armed:
            return None
        latest_exact_by_stream: dict[str, _RequestSample] = {}
        for sample in samples:
            if not sample.estimated:
                latest_exact_by_stream[sample.stream_id] = sample
        saturated: list[tuple[float, _RequestSample, TraceEvent]] = []
        for sample in latest_exact_by_stream.values():
            source_event = self.events[sample.event_index]
            ratio = sample.input_tokens / _context_limit(source_event.data)
            if ratio >= self.checkpoint_at:
                saturated.append((ratio, sample, source_event))
        if not saturated:
            return None
        _ratio, _sample, source_event = max(
            saturated, key=lambda item: (item[0], item[1].timestamp),
        )
        self._armed = False
        return write_checkpoint(
            self.store, self.session_id, self.events, source_event.timestamp,
        )

    def update(self, event: TraceEvent) -> Path | None:
        """Process a newly tailed event and return a written path, if any."""
        self.events.append(event)
        if event.event_type not in {
            EventType.LLM_REQUEST,
            EventType.LLM_RESPONSE,
            EventType.ASSISTANT_RESPONSE,
        }:
            return None
        stream_id = _event_stream(event)
        explicit = _input_token_count(event.data)

        if event.event_type == EventType.LLM_REQUEST:
            previous_tokens = self._last_tokens_by_stream.get(stream_id)
            self._pending_previous_by_stream[stream_id] = previous_tokens
            # Checkpoints, like compaction reports, require provider-reported
            # usage.  A paired response can supply it later.
            if explicit is None:
                return None
        else:
            if explicit is None:
                return None
            previous_tokens = self._pending_previous_by_stream.pop(
                stream_id, self._last_tokens_by_stream.get(stream_id),
            )

        if explicit is None:
            return None
        if (
            previous_tokens is not None
            and explicit < previous_tokens
            and (previous_tokens - explicit) / previous_tokens
            > self.compaction_threshold
        ):
            self._armed = True
        self._last_tokens = explicit
        self._last_tokens_by_stream[stream_id] = explicit

        ratio = explicit / _context_limit(event.data)
        if ratio < self.checkpoint_at or not self._armed:
            return None
        self._armed = False
        return write_checkpoint(
            self.store, self.session_id, self.events, event.timestamp,
        )


def _rule_post_compaction_regression(events: list[TraceEvent], cfg: dict) -> list[Any]:
    """Lint rule adapter; imported by ``lint.py`` to avoid a module cycle."""
    from .lint import LintLevel, LintResult

    level = cfg.get("level", LintLevel.WARN)
    threshold = float(cfg.get("compaction_threshold", DEFAULT_COMPACTION_THRESHOLD))
    behavior_window = int(cfg.get("behavior_window", DEFAULT_BEHAVIOR_WINDOW))
    redundant_threshold = int(cfg.get("redundant_read_threshold", 3))
    tool_loop_threshold = int(cfg.get("tool_loop_threshold", 5))
    findings: list[LintResult] = []
    for compaction in detect_compactions(events, threshold):
        behavior = behavior_diff_for_compaction(
            events,
            compaction,
            window=behavior_window,
            redundant_read_threshold=redundant_threshold,
            tool_loop_threshold=tool_loop_threshold,
        )
        if not behavior.regressed:
            continue
        findings.append(LintResult(
            rule="post-compaction-regression",
            level=level,
            message=(
                f"Behavior regressed after compaction #{compaction.sequence} "
                f"({compaction.drop_ratio * 100:.0f}% context drop): "
                + "; ".join(behavior.regressions)
                + "."
            ),
            line_start=compaction.event_index + 1,
        ))
    return findings


def cmd_compaction(args: argparse.Namespace) -> int:
    """CLI handler registered by :mod:`agent_trace.cli`."""
    store = TraceStore(args.trace_dir)
    raw_session_id = getattr(args, "session_id", None) or store.get_latest_session_id()
    if not raw_session_id:
        sys.stderr.write("No sessions found.\n")
        return 1
    session_id = store.find_session(raw_session_id)
    if not session_id:
        sys.stderr.write(f"Session not found: {raw_session_id}\n")
        return 1
    threshold = float(
        getattr(args, "compaction_threshold", DEFAULT_COMPACTION_THRESHOLD)
    )
    try:
        report = analyse_compaction_session(store, session_id, threshold)
    except ValueError as exc:
        sys.stderr.write(f"compaction: {exc}\n")
        return 2
    format_compaction_report(
        report,
        out=sys.stdout,
        show_diff=bool(getattr(args, "diff", False)),
        show_behavior_diff=bool(getattr(args, "behavior_diff", False)),
    )
    return 0
