"""Token inflation calculator: measure tokenizer cost impact across model versions.

Compares token counts across model versions for stored session content.
Uses character-based estimation (no API calls required) with per-model
tokenizer inflation factors derived from community measurements.

Usage:
    agent-strace inflation
    agent-strace inflation --compare claude-opus-4-6,claude-opus-4-7
    agent-strace inflation --sessions 30
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import TextIO

from .cost import _estimate_tokens, PRICING
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

# Pricing per 1M tokens (input) — used for cost delta calculation
MODEL_INPUT_PRICE: dict[str, float] = {
    "claude-opus-4-6":   15.00,
    "claude-opus-4-7":   15.00,
    "claude-sonnet-4-5":  3.00,
    "claude-sonnet-4-7":  3.00,
    "gpt-4o":             5.00,
    "gpt-4":             30.00,
}
DEFAULT_INPUT_PRICE = 3.00  # sonnet-class fallback


def _resolve_factor(model: str) -> float:
    m = model.lower().strip()
    if m in TOKENIZER_FACTORS:
        return TOKENIZER_FACTORS[m]
    for key, factor in TOKENIZER_FACTORS.items():
        if m.startswith(key) or key.startswith(m):
            return factor
    return 1.0


def _resolve_price(model: str) -> float:
    m = model.lower().strip()
    if m in MODEL_INPUT_PRICE:
        return MODEL_INPUT_PRICE[m]
    for key, price in MODEL_INPUT_PRICE.items():
        if m.startswith(key) or key.startswith(m):
            return price
    return DEFAULT_INPUT_PRICE


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
            for msg in data.get("messages", []):
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
        for msg in data.get("messages", []):
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
    avg_cost_baseline: float
    avg_cost_inflated: float
    # Projections
    daily_sessions: float
    monthly_cost_baseline: float
    monthly_cost_inflated: float
    largest_inflation_source: str


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse_inflation(
    store: TraceStore,
    model_baseline: str = "claude-opus-4-6",
    model_inflated: str = "claude-opus-4-7",
    session_limit: int = 30,
    daily_sessions: float = 8.0,
) -> InflationReport:
    """Compare token counts between two model versions across stored sessions."""
    all_metas = store.list_sessions()
    if not all_metas:
        return InflationReport(
            session_count=0,
            model_baseline=model_baseline,
            model_inflated=model_inflated,
            factor_baseline=_resolve_factor(model_baseline),
            factor_inflated=_resolve_factor(model_inflated),
            by_content_type=[],
            avg_tokens_baseline=0,
            avg_tokens_inflated=0,
            avg_cost_baseline=0,
            avg_cost_inflated=0,
            daily_sessions=daily_sessions,
            monthly_cost_baseline=0,
            monthly_cost_inflated=0,
            largest_inflation_source="",
        )

    factor_b = _resolve_factor(model_baseline)
    factor_i = _resolve_factor(model_inflated)
    price_b = _resolve_price(model_baseline)
    price_i = _resolve_price(model_inflated)

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

    if sessions_analysed == 0:
        sessions_analysed = 1

    # Build per-content-type inflation breakdown
    by_ct: list[ContentTypeInflation] = []
    largest_source = ""
    largest_pct = 0.0

    for ct in CONTENT_TYPES:
        raw = raw_tokens.get(ct, 0)
        tok_b = int(raw * factor_b)
        tok_i = int(raw * factor_i)
        pct = ((tok_i - tok_b) / tok_b * 100) if tok_b > 0 else 0.0
        by_ct.append(ContentTypeInflation(
            content_type=ct,
            tokens_baseline=tok_b,
            tokens_inflated=tok_i,
            inflation_pct=pct,
        ))
        if pct > largest_pct:
            largest_pct = pct
            largest_source = ct

    # Per-session averages
    avg_raw = total_raw / sessions_analysed
    avg_b = avg_raw * factor_b
    avg_i = avg_raw * factor_i
    avg_cost_b = avg_b / 1_000_000 * price_b
    avg_cost_i = avg_i / 1_000_000 * price_i

    monthly = daily_sessions * 30
    monthly_cost_b = avg_cost_b * monthly
    monthly_cost_i = avg_cost_i * monthly

    return InflationReport(
        session_count=sessions_analysed,
        model_baseline=model_baseline,
        model_inflated=model_inflated,
        factor_baseline=factor_b,
        factor_inflated=factor_i,
        by_content_type=by_ct,
        avg_tokens_baseline=avg_b,
        avg_tokens_inflated=avg_i,
        avg_cost_baseline=avg_cost_b,
        avg_cost_inflated=avg_cost_i,
        daily_sessions=daily_sessions,
        monthly_cost_baseline=monthly_cost_b,
        monthly_cost_inflated=monthly_cost_i,
        largest_inflation_source=largest_source,
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_inflation(report: InflationReport, out: TextIO = sys.stdout) -> None:
    w = out.write
    sep = "─" * 55

    mb = report.model_baseline
    mi = report.model_inflated

    w(f"\nToken Inflation Report ({report.session_count} sessions)\n{sep}\n")
    w(f"  {'':22}  {mb:>14}  {mi:>14}  {'delta':>8}\n")
    w(f"{sep}\n")

    def _pct(a: float, b: float) -> str:
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
        f"${report.avg_cost_baseline:>13.4f}  "
        f"${report.avg_cost_inflated:>13.4f}  "
        f"{_pct(report.avg_cost_baseline, report.avg_cost_inflated):>8}\n"
    )
    w(
        f"  {'Daily ({:.0f} sessions)'.format(report.daily_sessions):<22}  "
        f"${report.avg_cost_baseline * report.daily_sessions:>13.2f}  "
        f"${report.avg_cost_inflated * report.daily_sessions:>13.2f}  "
        f"${(report.avg_cost_inflated - report.avg_cost_baseline) * report.daily_sessions:>+7.2f}\n"
    )
    w(
        f"  {'Monthly':<22}  "
        f"${report.monthly_cost_baseline:>13.2f}  "
        f"${report.monthly_cost_inflated:>13.2f}  "
        f"${report.monthly_cost_inflated - report.monthly_cost_baseline:>+7.2f}\n"
    )
    w(f"{sep}\n")

    if report.largest_inflation_source:
        label = report.largest_inflation_source.replace("_", " ").title()
        w(f"Largest inflation source: {label}\n")

    w("\n")


# ---------------------------------------------------------------------------
# CLI handler
# ---------------------------------------------------------------------------

def cmd_inflation(args: argparse.Namespace) -> int:
    store = TraceStore(args.trace_dir)

    compare_raw = getattr(args, "compare", "") or ""
    if compare_raw and "," in compare_raw:
        parts = [p.strip() for p in compare_raw.split(",", 1)]
        model_b, model_i = parts[0], parts[1]
    else:
        model_b = "claude-opus-4-6"
        model_i = "claude-opus-4-7"

    session_limit = getattr(args, "sessions", 30) or 30
    daily = getattr(args, "daily_sessions", 8.0) or 8.0

    report = analyse_inflation(
        store,
        model_baseline=model_b,
        model_inflated=model_i,
        session_limit=session_limit,
        daily_sessions=daily,
    )
    format_inflation(report)
    return 0
