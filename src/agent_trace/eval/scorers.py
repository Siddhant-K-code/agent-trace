"""Built-in scorer implementations.

A scorer takes a list of TraceEvent objects and returns a score between
0.0 and 1.0, plus an optional reason string. Zero new dependencies for
built-in scorers.
"""

from __future__ import annotations

import importlib
import inspect
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Callable

from ..cost import estimate_cost
from ..models import EventType, TraceEvent
from ..store import TraceStore


@dataclass
class ScoreResult:
    scorer: str
    score: float          # 0.0 – 1.0
    threshold: float      # minimum passing score
    passed: bool
    reason: str = ""

    @property
    def status(self) -> str:
        return "pass" if self.passed else "fail"


class MetricError(ValueError):
    """Raised when a raw eval metric cannot be computed reliably."""


MetricValue = int | float | str | bool


_READ_TOOLS = {
    "read",
    "read_file",
    "file_read",
    "view",
    "open_file",
}


def _as_number(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def measure_cost_usd(store: TraceStore, session_id: str) -> float:
    """Return estimated total session cost in US dollars."""

    return estimate_cost(store, session_id).total_cost


def measure_error_count(events: list[TraceEvent]) -> int:
    """Return the number of explicit error events."""

    return sum(1 for event in events if event.event_type == EventType.ERROR)


def measure_session_status(events: list[TraceEvent], meta: object | None = None) -> str:
    """Classify a session as completed, killed, or timeout."""

    end_event = next(
        (event for event in reversed(events) if event.event_type == EventType.SESSION_END),
        None,
    )
    if end_event is None:
        return "completed" if getattr(meta, "ended_at", None) else "killed"

    data = end_event.data
    explicit = data.get("status") or data.get("session_status")
    if explicit:
        normalized = str(explicit).strip().lower().replace("-", "_")
        if normalized in {"completed", "complete", "success", "succeeded"}:
            return "completed"
        if normalized in {"timeout", "timed_out"}:
            return "timeout"
        if normalized in {
            "killed", "cancelled", "canceled", "interrupted", "terminated",
            "failed", "failure", "aborted", "error",
        }:
            return "killed"

    reason = " ".join(
        str(data.get(key, ""))
        for key in ("reason", "stop_reason", "termination_reason")
    ).lower()
    if "timeout" in reason or "timed out" in reason:
        return "timeout"
    if any(word in reason for word in ("kill", "cancel", "interrupt", "terminate")):
        return "killed"

    exit_code = _as_number(data.get("exit_code"))
    if exit_code == 124:
        return "timeout"
    if exit_code is not None and exit_code != 0:
        return "killed"
    return "completed"


def _read_path(event: TraceEvent) -> str:
    if event.event_type == EventType.FILE_READ:
        return str(
            event.data.get("path")
            or event.data.get("file_path")
            or event.data.get("uri")
            or ""
        )
    if event.event_type != EventType.TOOL_CALL:
        return ""
    tool_name = str(event.data.get("tool_name", "")).strip().lower().replace("-", "_")
    if tool_name not in _READ_TOOLS and not tool_name.endswith(("read_file", "file_read")):
        return ""
    arguments = event.data.get("arguments", {}) or {}
    if not isinstance(arguments, dict):
        return ""
    return str(arguments.get("file_path") or arguments.get("path") or arguments.get("uri") or "")


def measure_redundant_read_ratio(events: list[TraceEvent]) -> float:
    """Return repeated reads after the first read divided by all file reads."""

    seen: set[str] = set()
    reads = 0
    redundant = 0
    for event in events:
        path = _read_path(event)
        if not path:
            continue
        normalized = os.path.normcase(os.path.normpath(path))
        reads += 1
        if normalized in seen:
            redundant += 1
        else:
            seen.add(normalized)
    return redundant / reads if reads else 0.0


def measure_tool_call_count(events: list[TraceEvent]) -> int:
    """Return the number of tool-call events."""

    return sum(1 for event in events if event.event_type == EventType.TOOL_CALL)


def _context_payloads(data: dict[str, Any]) -> list[dict[str, Any]]:
    payloads = [data]
    usage = data.get("usage")
    if isinstance(usage, dict):
        payloads.insert(0, usage)
    return payloads


def measure_context_fill_ratio(
    events: list[TraceEvent],
    context_window: float = 200_000,
) -> float:
    """Return the maximum observed prompt/context window utilization."""

    configured_window = _as_number(context_window)
    if configured_window is None or configured_window <= 0:
        raise MetricError("context_window must be greater than zero")

    maximum = 0.0
    for event in events:
        if event.event_type not in {EventType.LLM_REQUEST, EventType.LLM_RESPONSE}:
            continue
        data = event.data
        payloads = _context_payloads(data)

        for payload in payloads:
            for key in ("context_fill_ratio", "context_usage_ratio"):
                explicit_ratio = _as_number(payload.get(key))
                if explicit_ratio is not None:
                    maximum = max(maximum, explicit_ratio)

        input_tokens: float | None = None
        for payload in payloads:
            primary = next(
                (
                    number
                    for key in ("input_tokens", "prompt_tokens")
                    if (number := _as_number(payload.get(key))) is not None
                ),
                None,
            )
            if primary is not None:
                cache_tokens = sum(
                    _as_number(payload.get(key)) or 0.0
                    for key in ("cache_read_input_tokens", "cache_creation_input_tokens")
                )
                input_tokens = primary + cache_tokens
                break

        if input_tokens is None and event.event_type == EventType.LLM_REQUEST:
            context_content = {
                key: data[key]
                for key in ("messages", "prompt", "content")
                if key in data
            }
            if context_content:
                input_tokens = max(1, len(json.dumps(context_content, default=str)) // 4)

        event_window = next(
            (
                number
                for payload in payloads
                for key in ("context_window", "context_window_tokens", "max_context_tokens")
                if (number := _as_number(payload.get(key))) is not None and number > 0
            ),
            configured_window,
        )
        if input_tokens is not None:
            maximum = max(maximum, input_tokens / event_window)
    return maximum


def measure_duration_seconds(events: list[TraceEvent], meta: object | None = None) -> float:
    """Return wall-clock session duration in seconds."""

    duration_ms = _as_number(getattr(meta, "total_duration_ms", None))
    if duration_ms is not None and duration_ms > 0:
        return duration_ms / 1000.0
    started_at = _as_number(getattr(meta, "started_at", None))
    ended_at = _as_number(getattr(meta, "ended_at", None))
    if started_at is not None and ended_at is not None and ended_at >= started_at:
        return ended_at - started_at
    if len(events) < 2:
        return 0.0
    timestamps = [event.timestamp for event in events]
    return max(0.0, max(timestamps) - min(timestamps))


def measure_lint_violations(store: TraceStore, session_id: str) -> int:
    """Return the total number of findings from all enabled lint rules."""

    from ..lint import lint_session

    return len(lint_session(store, session_id).findings)


BUILTIN_METRICS = {
    "cost_usd",
    "error_count",
    "session_status",
    "redundant_read_ratio",
    "tool_call_count",
    "context_fill_ratio",
    "max_context_fill_ratio",
    "duration_seconds",
    "lint_violations",
}


def _run_custom_metric(
    reference: str,
    events: list[TraceEvent],
    store: TraceStore,
    session_id: str,
    params: dict[str, Any],
) -> MetricValue:
    module_name, separator, attribute = reference.partition(":")
    if not separator:
        module_name, dot, attribute = reference.rpartition(".")
        if not dot:
            raise MetricError(f"unknown scorer: {reference}")
    if not module_name or not attribute:
        raise MetricError(f"invalid custom scorer reference: {reference}")

    try:
        function = getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as exc:
        raise MetricError(f"could not load custom scorer {reference!r}: {exc}") from exc
    if not callable(function):
        raise MetricError(f"custom scorer {reference!r} is not callable")

    try:
        signature = inspect.signature(function)
        parameters = signature.parameters
        has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())
        if "events" not in parameters and not has_kwargs:
            return function(events)
        available = {"events": events, "store": store, "session_id": session_id, **params}
        kwargs = {
            key: value
            for key, value in available.items()
            if has_kwargs or key in parameters
        }
        return function(**kwargs)
    except MetricError:
        raise
    except Exception as exc:
        raise MetricError(f"custom scorer {reference!r} failed: {exc}") from exc


def measure_metric(
    name: str,
    events: list[TraceEvent],
    store: TraceStore,
    session_id: str,
    params: dict[str, Any] | None = None,
) -> MetricValue:
    """Measure a raw built-in or importable custom scorer value."""

    params = params or {}
    meta = store.load_meta(session_id)
    if name == "cost_usd":
        value: MetricValue = measure_cost_usd(store, session_id)
    elif name == "error_count":
        value = measure_error_count(events)
    elif name == "session_status":
        value = measure_session_status(events, meta)
    elif name == "redundant_read_ratio":
        value = measure_redundant_read_ratio(events)
    elif name == "tool_call_count":
        value = measure_tool_call_count(events)
    elif name in {"context_fill_ratio", "max_context_fill_ratio"}:
        value = measure_context_fill_ratio(
            events,
            context_window=params.get("context_window", params.get("context_window_tokens", 200_000)),
        )
    elif name == "duration_seconds":
        value = measure_duration_seconds(events, meta)
    elif name == "lint_violations":
        value = measure_lint_violations(store, session_id)
    else:
        value = _run_custom_metric(name, events, store, session_id, params)

    if isinstance(value, float) and not math.isfinite(value):
        raise MetricError(f"scorer {name!r} returned a non-finite number")
    if not isinstance(value, (int, float, str, bool)):
        raise MetricError(f"scorer {name!r} returned unsupported value {type(value).__name__}")
    return value


# ---------------------------------------------------------------------------
# Built-in scorers
# ---------------------------------------------------------------------------

def score_no_errors(events: list[TraceEvent], threshold: float = 1.0) -> ScoreResult:
    """1.0 if no ERROR events, 0.0 otherwise."""
    errors = [e for e in events if e.event_type == EventType.ERROR]
    score = 0.0 if errors else 1.0
    reason = f"{len(errors)} error(s) found" if errors else "no errors"
    return ScoreResult("no_errors", score, threshold, score >= threshold, reason)


def score_regex(
    events: list[TraceEvent],
    pattern: str,
    event_type: str = "assistant_response",
    threshold: float = 1.0,
) -> ScoreResult:
    """1.0 if any event of *event_type* matches *pattern*, 0.0 otherwise."""
    try:
        et = EventType(event_type)
    except ValueError:
        return ScoreResult("regex", 0.0, threshold, False, f"unknown event_type: {event_type}")

    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        return ScoreResult("regex", 0.0, threshold, False, f"invalid pattern: {exc}")

    for event in events:
        if event.event_type != et:
            continue
        text = " ".join(str(v) for v in event.data.values())
        if compiled.search(text):
            return ScoreResult("regex", 1.0, threshold, True, f"pattern matched in {event_type}")

    return ScoreResult("regex", 0.0, threshold, False, f"pattern not found in any {event_type} event")


def score_cost_under(
    store: TraceStore,
    session_id: str,
    max_dollars: float,
    threshold: float = 1.0,
) -> ScoreResult:
    """1.0 if estimated cost ≤ max_dollars, else proportional score."""
    try:
        result = estimate_cost(store, session_id)
        actual = result.total_cost
    except Exception as exc:
        return ScoreResult("cost_under", 0.0, threshold, False, f"cost estimation failed: {exc}")

    if actual <= max_dollars:
        score = 1.0
        reason = f"${actual:.4f} ≤ ${max_dollars}"
    else:
        # Proportional: score = max_dollars / actual (capped at 1.0)
        score = min(1.0, max_dollars / actual) if actual > 0 else 1.0
        reason = f"${actual:.4f} actual > ${max_dollars} limit"

    return ScoreResult("cost_under", score, threshold, score >= threshold, reason)


def score_files_scoped(
    events: list[TraceEvent],
    allowed_paths: list[str],
    threshold: float = 1.0,
) -> ScoreResult:
    """1.0 if all file operations are within allowed_paths."""
    if not allowed_paths:
        return ScoreResult("files_scoped", 1.0, threshold, True, "no path restrictions")

    violations: list[str] = []
    for event in events:
        if event.event_type != EventType.TOOL_CALL:
            continue
        name = event.data.get("tool_name", "").lower()
        if name not in ("read", "write", "edit", "view", "create"):
            continue
        args = event.data.get("arguments", {}) or {}
        path = str(args.get("file_path") or args.get("path") or "")
        if not path:
            continue
        if not any(path.startswith(allowed) for allowed in allowed_paths):
            violations.append(path)

    if not violations:
        return ScoreResult("files_scoped", 1.0, threshold, True, "all files within allowed paths")

    # Count total scoped file ops to compute a meaningful ratio
    total_ops = sum(
        1 for e in events
        if e.event_type == EventType.TOOL_CALL
        and e.data.get("tool_name", "").lower() in ("read", "write", "edit", "view", "create")
        and (e.data.get("arguments") or {}).get("file_path") or (e.data.get("arguments") or {}).get("path")
    )
    score = max(0.0, 1.0 - len(violations) / max(1, total_ops))
    reason = f"{len(violations)} file(s) outside allowed paths: {', '.join(violations[:3])}"
    return ScoreResult("files_scoped", score, threshold, score >= threshold, reason)


def score_duration_under(
    events: list[TraceEvent],
    max_seconds: float,
    threshold: float = 1.0,
) -> ScoreResult:
    """1.0 if session duration ≤ max_seconds."""
    if len(events) < 2:
        return ScoreResult("duration_under", 1.0, threshold, True, "insufficient events to measure duration")

    duration = events[-1].timestamp - events[0].timestamp
    if duration <= max_seconds:
        return ScoreResult("duration_under", 1.0, threshold, True, f"{duration:.1f}s ≤ {max_seconds}s")

    score = min(1.0, max_seconds / duration) if duration > 0 else 1.0
    reason = f"{duration:.1f}s actual > {max_seconds}s limit"
    return ScoreResult("duration_under", score, threshold, score >= threshold, reason)


def score_custom(
    events: list[TraceEvent],
    fn: Callable[[list[TraceEvent]], float],
    name: str = "custom",
    threshold: float = 1.0,
) -> ScoreResult:
    """Run a user-supplied callable that returns a float in [0, 1]."""
    try:
        score = float(fn(events))
        score = max(0.0, min(1.0, score))
    except Exception as exc:
        return ScoreResult(name, 0.0, threshold, False, f"scorer raised: {exc}")
    return ScoreResult(name, score, threshold, score >= threshold, "custom scorer")


def score_llm_judge(
    events: list[TraceEvent],
    prompt: str,
    base_url: str,
    api_key: str,
    model: str = "gpt-4o-mini",
    threshold: float = 1.0,
) -> ScoreResult:
    """Call an LLM to judge the session. Returns a score in [0, 1].

    The prompt receives a compact session summary. The LLM must respond
    with JSON: {"score": <float 0-1>, "reason": "<string>"}.
    Uses urllib.request — zero new dependencies.
    """
    import json as _json
    import urllib.error
    import urllib.request

    if not base_url or not api_key:
        return ScoreResult("llm_judge", 0.0, threshold, False,
                           "base_url and api_key required for llm_judge scorer")

    # Build a compact session summary for the LLM
    tool_calls = [e for e in events if e.event_type == EventType.TOOL_CALL]
    errors = [e for e in events if e.event_type == EventType.ERROR]
    summary_lines = [
        f"Session: {len(events)} events, {len(tool_calls)} tool calls, {len(errors)} errors.",
    ]
    for ev in tool_calls[:15]:
        name = ev.data.get("tool_name", "unknown")
        summary_lines.append(f"  TOOL_CALL: {name}")
    for ev in errors[:5]:
        msg = str(ev.data.get("message", ""))[:80]
        summary_lines.append(f"  ERROR: {msg}")
    session_summary = "\n".join(summary_lines)

    full_prompt = (
        f"{prompt}\n\n"
        f"Session summary:\n{session_summary}\n\n"
        "Respond with JSON only: {\"score\": <float 0.0-1.0>, \"reason\": \"<one sentence>\"}"
    )

    payload = _json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": full_prompt}],
        "temperature": 0.1,
        "max_tokens": 256,
    }).encode()

    url = base_url.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = _json.loads(resp.read())
            content = data["choices"][0]["message"]["content"].strip()
            # Strip markdown fences if present
            if content.startswith("```"):
                content = "\n".join(content.split("\n")[1:])
            if content.endswith("```"):
                content = "\n".join(content.split("\n")[:-1])
            result = _json.loads(content)
            score = float(max(0.0, min(1.0, result.get("score", 0.0))))
            reason = str(result.get("reason", ""))[:200]
            return ScoreResult("llm_judge", score, threshold, score >= threshold, reason)
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        return ScoreResult("llm_judge", 0.0, threshold, False, f"LLM request failed: {exc}")
    except (_json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return ScoreResult("llm_judge", 0.0, threshold, False, f"LLM response parse error: {exc}")


# ---------------------------------------------------------------------------
# Scorer registry (name → factory)
# ---------------------------------------------------------------------------

def run_scorer(
    name: str,
    config: dict,
    events: list[TraceEvent],
    store: TraceStore | None = None,
    session_id: str = "",
) -> ScoreResult:
    """Dispatch to the appropriate built-in scorer by name."""
    threshold = float(config.get("threshold", config.get("weight", 1.0)))

    if name == "no_errors":
        return score_no_errors(events, threshold=threshold)

    if name == "regex":
        return score_regex(
            events,
            pattern=config.get("pattern", ""),
            event_type=config.get("event_type", "assistant_response"),
            threshold=threshold,
        )

    if name == "cost_under":
        if store is None or not session_id:
            return ScoreResult(name, 0.0, threshold, False, "store/session_id required for cost_under")
        return score_cost_under(store, session_id, max_dollars=float(config.get("max_dollars", 10.0)), threshold=threshold)

    if name == "files_scoped":
        return score_files_scoped(
            events,
            allowed_paths=config.get("allowed_paths", []),
            threshold=threshold,
        )

    if name == "duration_under":
        return score_duration_under(
            events,
            max_seconds=float(config.get("max_seconds", 120.0)),
            threshold=threshold,
        )

    if name == "llm_judge":
        import os
        return score_llm_judge(
            events,
            prompt=config.get("prompt", "Did the agent complete the task correctly?"),
            base_url=config.get("base_url", "") or os.environ.get("OPENAI_BASE_URL", "") or os.environ.get("AGENT_STRACE_LLM_URL", ""),
            api_key=config.get("api_key", "") or os.environ.get("OPENAI_API_KEY", "") or os.environ.get("AGENT_STRACE_LLM_KEY", ""),
            model=config.get("model", "gpt-4o-mini"),
            threshold=threshold,
        )

    return ScoreResult(name, 0.0, threshold, False, f"unknown scorer: {name}")
