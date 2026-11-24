"""Token inflation calculator: measure tokenizer cost impact across model versions.

Compares token counts across model versions for stored session content.
Uses character-based estimation (no API calls required) with per-model
tokenizer inflation factors derived from community measurements.

Usage:
    agent-strace inflation
    agent-strace inflation --compare claude-opus-4-6,claude-opus-4-7
    agent-strace inflation --sessions 30
    agent-strace inflation --daily-sessions 8
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from typing import TextIO

from .cost import ModelPrice, _estimate_tokens, _match_model_key, get_model_pricing
from .event_data import list_field, require_object_event_data
from .models import EventType, TraceEvent
from .store import TraceStore


# ---------------------------------------------------------------------------
# Tokenizer inflation factors
#
# Measured community inflation ratios relative to a baseline tokenizer.
# claude-opus-4-6 is the baseline (factor = 1.0).
# claude-opus-4-7 introduced a new tokenizer with ~1.3–1.47x inflation
# on real Claude Code inputs (community measurements, April 2026).
# ---------------------------------------------------------------------------

TOKENIZER_FACTORS: dict[str, float] = {
    # Anthropic
    "claude-opus-4-6":       1.00,
    "claude-sonnet-4-5":     1.00,
    "claude-haiku-3-5":      1.00,
    "claude-3-opus":         1.00,
    "claude-3-sonnet":       1.00,
    "claude-3-haiku":        1.00,
    "claude-opus-4-7":       1.38,   # community median: 1.3–1.47x
    "claude-sonnet-4-7":     1.35,
    # OpenAI (tiktoken cl100k_base → o200k_base)
    "gpt-4":                 1.00,
    "gpt-4o":                1.05,
    "gpt-4o-mini":           1.05,
    "o1":                    1.05,
    "o3":                    1.05,
    # Google
    "gemini-1.5-pro":        0.95,
    "gemini-2.0-flash":      0.95,
}

# Content type labels for breakdown
CONTENT_TYPES = ["system_prompt", "tool_definitions", "user_messages", "assistant_messages"]

# Content types priced at the output rate. The others use the input rate.
OUTPUT_CONTENT_TYPES = frozenset({"assistant_messages"})

DEFAULT_COMPARE = ("claude-opus-4-6", "claude-opus-4-7")
DEFAULT_SESSIONS = 30
DEFAULT_DAILY_SESSIONS = 8.0


def _resolve_factor(model: str) -> float | None:
    """Return the tokenizer factor for *model*, or ``None`` if it has none.

    Names resolve as in :func:`agent_trace.cost.get_model_pricing`, so a
    dated or provider-prefixed name gets the factor of its base model.
    """
    key = _match_model_key(model, TOKENIZER_FACTORS)
    return None if key is None else TOKENIZER_FACTORS[key]


def _no_factor_message(model: str) -> str:
    return (
        f"model {model!r} has no tokenizer factor. "
        f"Models with a factor: {', '.join(sorted(TOKENIZER_FACTORS))}"
    )


def _session_word(count: float) -> str:
    return "session" if count == 1 else "sessions"


def _format_count(value: float) -> str:
    """Return *value* with no rounding: 8.0 is ``8`` and 2.5 is ``2.5``."""
    text = repr(float(value))
    return text[:-2] if text.endswith(".0") else text


# _extract_tokens_by_type reads the data of every event of these types.
INFLATION_DATA_EVENT_TYPES = frozenset({
    EventType.LLM_REQUEST,
    EventType.USER_PROMPT,
    EventType.ASSISTANT_RESPONSE,
    EventType.LLM_RESPONSE,
})


def _extract_tokens_by_type(event: TraceEvent) -> dict[str, int]:
    """Return a mapping of content_type → estimated token count for one event.

    LLM_REQUEST events may contain multiple content types simultaneously
    (system prompt + tool definitions + user messages). Each is measured
    independently so the per-type breakdown is accurate.
    """
    result: dict[str, int] = {}
    et = event.event_type
    data = event.data

    if et == EventType.LLM_REQUEST:
        # System prompt — Anthropic top-level key or OpenAI role=system message
        system_text = ""
        if "system" in data:
            system_text = json.dumps(data["system"])
        elif "system_prompt" in data:
            system_text = json.dumps(data["system_prompt"])
        else:
            # OpenAI format: look for role=system inside messages array
            for msg in list_field(data, "messages", []):
                if isinstance(msg, dict) and msg.get("role") == "system":
                    system_text += json.dumps(msg.get("content", ""))
        if system_text:
            result["system_prompt"] = _estimate_tokens(system_text)

        # Tool definitions
        tools_text = ""
        if "tools" in data:
            tools_text = json.dumps(data["tools"])
        elif "tool_definitions" in data:
            tools_text = json.dumps(data["tool_definitions"])
        if tools_text:
            result["tool_definitions"] = _estimate_tokens(tools_text)

        # User messages (non-system messages in the messages array)
        user_text = ""
        for msg in list_field(data, "messages", []):
            if isinstance(msg, dict) and msg.get("role") != "system":
                user_text += json.dumps(msg.get("content", ""))
        # Also count any top-level prompt field
        if "prompt" in data:
            user_text += json.dumps(data["prompt"])
        if user_text:
            result["user_messages"] = _estimate_tokens(user_text)

        # Fallback: if none of the above matched, count the whole payload
        if not result:
            result["user_messages"] = _estimate_tokens(json.dumps(data))

    elif et == EventType.USER_PROMPT:
        content = json.dumps(data.get("content", data))
        result["user_messages"] = _estimate_tokens(content)

    elif et in (EventType.ASSISTANT_RESPONSE, EventType.LLM_RESPONSE):
        content = json.dumps(data.get("content", data))
        result["assistant_messages"] = _estimate_tokens(content)

    return result


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ContentTypeInflation:
    content_type: str
    tokens_baseline: int
    tokens_inflated: int
    inflation_pct: float


@dataclass
class InflationReport:
    session_count: int
    model_baseline: str
    model_inflated: str
    factor_baseline: float
    factor_inflated: float
    # Per content type
    by_content_type: list[ContentTypeInflation]
    # Per session averages
    avg_tokens_baseline: float
    avg_tokens_inflated: float
    # None when the model has no bundled price
    avg_cost_baseline: float | None
    avg_cost_inflated: float | None
    # Projections
    daily_sessions: float
    monthly_cost_baseline: float | None
    monthly_cost_inflated: float | None
    largest_inflation_source: str


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _avg_cost(
    raw_tokens: dict[str, int],
    sessions: int,
    factor: float,
    price: ModelPrice | None,
) -> float | None:
    """Return the cost of an average session, or ``None`` with no price."""
    if price is None:
        return None
    if sessions == 0:
        return 0.0
    cost = 0.0
    for ct in CONTENT_TYPES:
        if ct in OUTPUT_CONTENT_TYPES:
            rate = price.output_per_million
        else:
            rate = price.input_per_million
        cost += raw_tokens.get(ct, 0) / sessions * factor / 1_000_000 * rate
    return cost


def _monthly_cost(avg_cost: float | None, daily_sessions: float) -> float | None:
    return None if avg_cost is None else avg_cost * daily_sessions * 30


def analyse_inflation(
    store: TraceStore,
    model_baseline: str = "claude-opus-4-6",
    model_inflated: str = "claude-opus-4-7",
    session_limit: int = 30,
    daily_sessions: float = 8.0,
) -> InflationReport:
    """Compare token counts between two model versions across stored sessions.

    Raises ValueError when a model has no tokenizer factor. Costs use the
    bundled prices from :func:`agent_trace.cost.get_model_pricing`. A model
    with no bundled price gets ``None`` for each cost.
    """
    factor_b = _resolve_factor(model_baseline)
    factor_i = _resolve_factor(model_inflated)
    for model, factor in ((model_baseline, factor_b), (model_inflated, factor_i)):
        if factor is None:
            raise ValueError(_no_factor_message(model))
    price_b = get_model_pricing(model_baseline)
    price_i = get_model_pricing(model_inflated)

    all_metas = store.list_sessions()
    if not all_metas:
        cost_b = _avg_cost({}, 0, factor_b, price_b)
        cost_i = _avg_cost({}, 0, factor_i, price_i)
        return InflationReport(
            session_count=0,
            model_baseline=model_baseline,
            model_inflated=model_inflated,
            factor_baseline=factor_b,
            factor_inflated=factor_i,
            by_content_type=[],
            avg_tokens_baseline=0,
            avg_tokens_inflated=0,
            avg_cost_baseline=cost_b,
            avg_cost_inflated=cost_i,
            daily_sessions=daily_sessions,
            monthly_cost_baseline=_monthly_cost(cost_b, daily_sessions),
            monthly_cost_inflated=_monthly_cost(cost_i, daily_sessions),
            largest_inflation_source="",
        )

    # Accumulate raw token counts per content type (at baseline factor=1.0)
    raw_tokens: dict[str, int] = {ct: 0 for ct in CONTENT_TYPES}
    total_raw = 0
    sessions_analysed = 0

    for meta in all_metas[:session_limit]:
        sid = meta.session_id
        try:
            events = store.load_events(sid)
        except Exception:
            continue
        require_object_event_data(
            events, event_types=INFLATION_DATA_EVENT_TYPES, session_id=sid
        )

        session_raw = 0
        for event in events:
            by_type = _extract_tokens_by_type(event)
            for ct, count in by_type.items():
                raw_tokens[ct] = raw_tokens.get(ct, 0) + count
                session_raw += count
                total_raw += count

        # Only count sessions that contributed LLM/prompt content — sessions
        # with only tool calls would dilute the per-session average.
        if session_raw > 0:
            sessions_analysed += 1

    # Build per-content-type inflation breakdown. The largest source is the
    # type that adds the most tokens. A tie keeps the first type.
    by_ct: list[ContentTypeInflation] = []
    largest_source = ""
    largest_added = 0

    for ct in CONTENT_TYPES:
        raw = raw_tokens.get(ct, 0)
        tok_b = round(raw * factor_b)
        tok_i = round(raw * factor_i)
        pct = ((tok_i - tok_b) / tok_b * 100) if tok_b > 0 else 0.0
        by_ct.append(ContentTypeInflation(
            content_type=ct,
            tokens_baseline=tok_b,
            tokens_inflated=tok_i,
            inflation_pct=pct,
        ))
        if tok_i - tok_b > largest_added:
            largest_added = tok_i - tok_b
            largest_source = ct

    # Per-session averages. With no session of content they are 0.
    avg_raw = total_raw / sessions_analysed if sessions_analysed else 0.0
    avg_cost_b = _avg_cost(raw_tokens, sessions_analysed, factor_b, price_b)
    avg_cost_i = _avg_cost(raw_tokens, sessions_analysed, factor_i, price_i)

    return InflationReport(
        session_count=sessions_analysed,
        model_baseline=model_baseline,
        model_inflated=model_inflated,
        factor_baseline=factor_b,
        factor_inflated=factor_i,
        by_content_type=by_ct,
        avg_tokens_baseline=avg_raw * factor_b,
        avg_tokens_inflated=avg_raw * factor_i,
        avg_cost_baseline=avg_cost_b,
        avg_cost_inflated=avg_cost_i,
        daily_sessions=daily_sessions,
        monthly_cost_baseline=_monthly_cost(avg_cost_b, daily_sessions),
        monthly_cost_inflated=_monthly_cost(avg_cost_i, daily_sessions),
        largest_inflation_source=largest_source,
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _cost_cell(value: float | None, precision: int) -> str:
    if value is None:
        return f"{'n/a':>14}"
    return f"${value:>13.{precision}f}"


def _cost_delta(baseline: float | None, inflated: float | None) -> str:
    if baseline is None or inflated is None:
        return f"{'n/a':>8}"
    return f"${inflated - baseline:>+7.2f}"


def format_inflation(report: InflationReport, out: TextIO = sys.stdout) -> None:
    w = out.write
    sep = "─" * 55

    mb = report.model_baseline
    mi = report.model_inflated
    count = report.session_count

    w(f"\nToken Inflation Report ({count} {_session_word(count)})\n{sep}\n")
    w(f"  {'':22}  {mb:>14}  {mi:>14}  {'delta':>8}\n")
    w(f"{sep}\n")

    def _pct(a: float | None, b: float | None) -> str:
        if a is None or b is None:
            return "n/a"
        if a == 0:
            return "  n/a"
        pct = (b - a) / a * 100
        sign = "+" if pct >= 0 else ""
        return f"{sign}{pct:.1f}%"

    for ct in report.by_content_type:
        label = ct.content_type.replace("_", " ").title()
        w(
            f"  {label:<22}  {ct.tokens_baseline:>14,}  "
            f"{ct.tokens_inflated:>14,}  {_pct(ct.tokens_baseline, ct.tokens_inflated):>8}\n"
        )

    w(f"{sep}\n")
    w(
        f"  {'Per session (tokens)':<22}  "
        f"{report.avg_tokens_baseline:>14,.0f}  "
        f"{report.avg_tokens_inflated:>14,.0f}  "
        f"{_pct(report.avg_tokens_baseline, report.avg_tokens_inflated):>8}\n"
    )
    w(
        f"  {'Per session (cost)':<22}  "
        f"{_cost_cell(report.avg_cost_baseline, 4)}  "
        f"{_cost_cell(report.avg_cost_inflated, 4)}  "
        f"{_pct(report.avg_cost_baseline, report.avg_cost_inflated):>8}\n"
    )
    daily = report.daily_sessions
    daily_b = None if report.avg_cost_baseline is None else report.avg_cost_baseline * daily
    daily_i = None if report.avg_cost_inflated is None else report.avg_cost_inflated * daily
    daily_label = f"Daily ({_format_count(daily)} {_session_word(daily)})"
    w(
        f"  {daily_label:<22}  "
        f"{_cost_cell(daily_b, 2)}  "
        f"{_cost_cell(daily_i, 2)}  "
        f"{_cost_delta(daily_b, daily_i)}\n"
    )
    w(
        f"  {'Monthly':<22}  "
        f"{_cost_cell(report.monthly_cost_baseline, 2)}  "
        f"{_cost_cell(report.monthly_cost_inflated, 2)}  "
        f"{_cost_delta(report.monthly_cost_baseline, report.monthly_cost_inflated)}\n"
    )
    w(f"{sep}\n")

    if report.largest_inflation_source:
        label = report.largest_inflation_source.replace("_", " ").title()
        w(f"Largest inflation source: {label}\n")

    w("\n")


# ---------------------------------------------------------------------------
# CLI handler
# ---------------------------------------------------------------------------

def _compare_models(value: object) -> tuple[str, str]:
    if value is None:
        return DEFAULT_COMPARE
    names = [name.strip() for name in str(value).split(",")]
    if len(names) != 2 or not all(names):
        raise ValueError(
            f"--compare must be two model names separated by one comma, got {value!r}"
        )
    for name in names:
        if _resolve_factor(name) is None:
            raise ValueError(f"--compare {_no_factor_message(name)}")
    return names[0], names[1]


def _session_limit(value: object) -> int:
    limit = DEFAULT_SESSIONS if value is None else value
    if limit < 1:
        raise ValueError(f"--sessions must be 1 or more, got {limit}")
    return limit


def _daily_sessions(value: object) -> float:
    daily = DEFAULT_DAILY_SESSIONS if value is None else float(value)
    if not math.isfinite(daily) or daily < 0:
        raise ValueError(
            f"--daily-sessions must be a finite number, 0 or more, got {_format_count(daily)}"
        )
    # -0.0 would print as "-0".
    return 0.0 if daily == 0 else daily


def cmd_inflation(args: argparse.Namespace) -> int:
    try:
        model_b, model_i = _compare_models(getattr(args, "compare", None))
        session_limit = _session_limit(getattr(args, "sessions", None))
        daily = _daily_sessions(getattr(args, "daily_sessions", None))
    except ValueError as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 2

    store = TraceStore(args.trace_dir)
    report = analyse_inflation(
        store,
        model_baseline=model_b,
        model_inflated=model_i,
        session_limit=session_limit,
        daily_sessions=daily,
    )
    format_inflation(report, out=sys.stdout)
    return 0
