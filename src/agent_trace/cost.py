"""Cost estimation for agent sessions.

Estimates token usage from event payload sizes (len(content) / 4) and maps
tokens to dollar cost using configurable per-model pricing. No API calls.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _datetime
import json
import re
import sys
import time as _time
from dataclasses import dataclass, field, replace as _replace
from typing import Any, Mapping, TextIO

from .event_data import require_object_event_data
from .explain import PHASE_DATA_EVENT_TYPES, ExplainResult, explain_session
from .models import EventType, TraceEvent
from .pairing import Pairer, event_model, llm_pairer
from .store import TraceStore


# ---------------------------------------------------------------------------
# Pricing table  (dollars per 1M tokens)
# ---------------------------------------------------------------------------

PRICING_SNAPSHOT_DATE = "2026-11-11"
"""Effective date of the bundled, offline pricing snapshot.

Rates are estimates in USD per one million text tokens.  Each model has an
``input`` rate for uncached input and an ``output`` rate, which also covers
reasoning and thinking tokens.  Where the provider publishes them, a model
also has ``cache_read`` (cached input, a cache hit), ``cache_write`` (a cache
write with the default lifetime: 5 minutes for Anthropic and Bedrock, 30
minutes for OpenAI) and ``cache_write_long`` (a cache write with the long
lifetime: 1 hour for Anthropic and Bedrock, 12 hours for OpenAI).  A model
without a cache read or write rate bills those tokens at its input rate, and
a long write without a long rate uses the default write rate.

Each model uses its standard (non-batch) rate in its shortest context or
smallest prompt tier.  When a pricing page lists a temporary promotional
price together with the price that follows it, the following price is used.
Batch, flex, fast and priority tiers, regional or data-residency uplifts,
long-context tiers, cache storage fees and other non-token charges are
excluded.

Updating this table requires checking each URL in ``PRICING_SOURCES``,
updating ``PRICING`` (``_MODEL_PROVIDERS`` is derived from its keys),
advancing the snapshot date, and extending the provider-cost tests for every
added model family.
"""

PRICING_SOURCES: dict[str, str] = {
    "Anthropic": "https://platform.claude.com/docs/en/about-claude/pricing",
    "OpenAI": "https://developers.openai.com/api/docs/pricing",
    "AWS Bedrock": "https://aws.amazon.com/bedrock/pricing/",
    "Gemini": "https://ai.google.dev/gemini-api/docs/pricing",
}


def _rates(
    input_rate: float,
    output_rate: float,
    cache_read: float | None = None,
    cache_write: float | None = None,
    cache_write_long: float | None = None,
) -> dict[str, float]:
    rates = {"input": input_rate, "output": output_rate}
    for key, value in (
        ("cache_read", cache_read),
        ("cache_write", cache_write),
        ("cache_write_long", cache_write_long),
    ):
        if value is not None:
            rates[key] = value
    return rates


PRICING: dict[str, dict[str, float]] = {
    # Backward-compatible short names used by the existing per-session report.
    # sonnet and opus carry the Claude Sonnet 4.x and Opus 4.1 cache rates.
    "sonnet": _rates(3.00, 15.00, 0.30, 3.75, 6.00),
    "opus":   _rates(15.00, 75.00, 1.50, 18.75, 30.00),
    "haiku":  _rates(0.25, 1.25),
    "gpt4":   _rates(30.00, 60.00),
    "gpt4o":  _rates(5.00, 15.00),

    # Anthropic direct API: input, output, cache hit, 5-minute write, 1-hour
    # write.  Haiku 5.5 uses its tier for prompts up to 100k tokens.
    "claude-fable-5-2": _rates(8.00, 40.00, 0.20, 10.00, 16.00),
    "claude-mythos-5-2": _rates(8.00, 40.00, 0.20, 10.00, 16.00),
    "claude-fable-5-1": _rates(10.00, 50.00, 0.25, 12.50, 20.00),
    "claude-mythos-5-1": _rates(10.00, 50.00, 0.25, 12.50, 20.00),
    "claude-fable-5": _rates(10.00, 50.00, 1.00, 12.50, 20.00),
    "claude-mythos-5": _rates(10.00, 50.00, 1.00, 12.50, 20.00),
    "claude-opus-5-5": _rates(4.00, 20.00, 0.20, 5.00, 8.00),
    "claude-opus-5": _rates(5.00, 25.00, 0.50, 6.25, 10.00),
    "claude-opus-4-8": _rates(5.00, 25.00, 0.50, 6.25, 10.00),
    "claude-opus-4-7": _rates(5.00, 25.00, 0.50, 6.25, 10.00),
    "claude-opus-4-6": _rates(5.00, 25.00, 0.50, 6.25, 10.00),
    "claude-opus-4-5": _rates(5.00, 25.00, 0.50, 6.25, 10.00),
    "claude-opus-4-1": _rates(15.00, 75.00, 1.50, 18.75, 30.00),
    "claude-opus-4": _rates(15.00, 75.00, 1.50, 18.75, 30.00),
    "claude-sonnet-5-5": _rates(2.00, 10.00, 0.10, 2.50, 4.00),
    "claude-sonnet-5": _rates(2.00, 10.00, 0.20, 2.50, 4.00),
    "claude-sonnet-4-6": _rates(3.00, 15.00, 0.30, 3.75, 6.00),
    "claude-sonnet-4-5": _rates(3.00, 15.00, 0.30, 3.75, 6.00),
    "claude-sonnet-4": _rates(3.00, 15.00, 0.30, 3.75, 6.00),
    "claude-haiku-5-5": _rates(0.10, 0.50, 0.01, 0.125, 0.20),
    "claude-haiku-4-5": _rates(1.00, 5.00, 0.10, 1.25, 2.00),
    # Retired Claude 3 models are no longer on the pricing page; their last
    # input and output rates are kept, without cache rates.
    "claude-3-7-sonnet": _rates(3.00, 15.00),
    "claude-3-5-sonnet": _rates(3.00, 15.00),
    "claude-3-haiku": _rates(0.25, 1.25),
    "claude-3-opus": _rates(15.00, 75.00),

    # OpenAI direct API, standard tier, short context (up to 272K input
    # tokens): input, output, cached input, 30-minute write, 12-hour write.
    # Models without write rates bill cache writes as ordinary input.
    "gpt-6-astra": _rates(10.00, 50.00, 1.00, 12.50, 25.00),
    "gpt-6.1-sol": _rates(2.00, 10.00, 0.10, 2.50, 5.00),
    "gpt-6.1-terra": _rates(1.00, 5.00, 0.05, 1.25, 2.50),
    "gpt-6-sol": _rates(2.00, 10.00, 0.20, 2.50, 5.00),
    "gpt-6-luna": _rates(0.10, 0.50, 0.01, 0.125, 0.25),
    "gpt-5.6-sol": _rates(4.00, 20.00, 0.40, 5.00, 10.00),
    "gpt-5.6-terra": _rates(2.00, 12.00, 0.20, 2.50, 5.00),
    "gpt-5.6-luna": _rates(0.20, 1.20, 0.02, 0.25, 0.50),
    "gpt-5.5": _rates(5.00, 30.00, 0.50),
    "gpt-5.5-pro": _rates(30.00, 180.00),
    "gpt-5.4": _rates(2.50, 15.00, 0.25),
    "gpt-5.4-pro": _rates(30.00, 180.00),
    "gpt-5.4-mini": _rates(0.75, 4.50, 0.075),
    "gpt-5.4-nano": _rates(0.20, 1.25, 0.02),
    "gpt-5.3-codex": _rates(1.75, 14.00, 0.175),
    "gpt-5.2": _rates(1.75, 14.00, 0.175),
    "gpt-5.2-pro": _rates(21.00, 168.00),
    "gpt-5.1": _rates(1.25, 10.00, 0.125),
    "gpt-5": _rates(1.25, 10.00, 0.125),
    "gpt-5-mini": _rates(0.25, 2.00, 0.025),
    "gpt-5-nano": _rates(0.05, 0.40, 0.005),
    "gpt-5-pro": _rates(15.00, 120.00),
    "gpt-5-search-api": _rates(1.25, 10.00, 0.125),
    "gpt-4.1": _rates(2.00, 8.00, 0.50),
    "gpt-4.1-mini": _rates(0.40, 1.60, 0.10),
    "gpt-4o": _rates(2.50, 10.00, 1.25),
    "gpt-4o-mini": _rates(0.15, 0.60, 0.075),
    "o3-pro": _rates(20.00, 80.00),
    "o3": _rates(2.00, 8.00, 0.50),
    # No longer on the pricing page; last input and output rates are kept.
    "gpt-4.1-nano": _rates(0.10, 0.40),
    "o3-mini": _rates(1.10, 4.40),
    "o4-mini": _rates(1.10, 4.40),

    # Amazon models through AWS Bedrock, us-east-1 on-demand: input, output,
    # cache read, cache write.  Actual charges vary by region and tier.
    "amazon.nova-pro-v1": _rates(0.80, 3.20, 0.20, 0.00),
    "amazon.nova-lite-v1": _rates(0.06, 0.24, 0.015, 0.00),
    "amazon.nova-micro-v1": _rates(0.035, 0.14, 0.00875, 0.00),

    # Google Gemini Developer API, paid tier: input, output, context caching.
    # Pro models use the tier for prompts up to 200k tokens, and Gemini 4 and
    # 3.8/3.6 Flash the rates that follow their promotional prices
    # (December 1, 2026 and January 1, 2027).
    "gemini-4-carbon": _rates(4.00, 20.00, 0.20),
    "gemini-4-argon": _rates(4.00, 20.00, 0.20),
    "gemini-3.8-flash": _rates(1.50, 7.50, 0.15),
    "gemini-3.6-flash": _rates(1.50, 7.50, 0.15),
    "gemini-3.5-flash-lite": _rates(0.30, 2.50, 0.03),
    "gemini-3.1-flash-lite": _rates(0.25, 1.50, 0.025),
    "gemini-3.1-pro-preview": _rates(2.00, 12.00, 0.20),
    "gemini-3-flash-preview": _rates(0.50, 3.00, 0.05),
    "gemini-2.5-pro": _rates(1.25, 10.00, 0.125),
    "gemini-2.5-flash": _rates(0.30, 2.50, 0.03),
    "gemini-2.5-flash-lite": _rates(0.10, 0.40, 0.01),
    # No longer on the pricing page; last input and output rates are kept.
    "gemini-2.0-flash": _rates(0.10, 0.40),
}
DEFAULT_MODEL = "sonnet"

_SHORT_MODEL_NAMES = frozenset({"sonnet", "opus", "haiku", "gpt4", "gpt4o"})


def _provider_for_key(key: str) -> str:
    if key.startswith("claude-"):
        return "Anthropic"
    if key.startswith("amazon."):
        return "AWS Bedrock"
    if key.startswith("gemini-"):
        return "Gemini"
    return "OpenAI"


_MODEL_PROVIDERS: dict[str, str] = {
    model: _provider_for_key(model)
    for model in PRICING
    if model not in _SHORT_MODEL_NAMES
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TokenUsage:
    """Token counts split by the rate each part is billed at.

    ``input_tokens`` is the whole input: uncached input, cache reads and
    cache writes.  ``cache_write_long_tokens`` is the part of
    ``cache_write_tokens`` written with the long cache lifetime.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cache_write_long_tokens: int = 0

    @property
    def uncached_input_tokens(self) -> int:
        return max(0, self.input_tokens - self.cache_read_tokens - self.cache_write_tokens)

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        if not isinstance(other, TokenUsage):
            return NotImplemented
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            cache_write_long_tokens=(
                self.cache_write_long_tokens + other.cache_write_long_tokens
            ),
        )


@dataclass
class PhaseCost:
    phase_index: int
    phase_name: str
    input_tokens: int
    output_tokens: int
    cost_dollars: float
    failed: bool = False
    # Parts of input_tokens read from or written to a prompt cache.
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class CostResult:
    session_id: str
    model: str
    total_cost: float
    input_tokens: int
    output_tokens: int
    phase_costs: list[PhaseCost]
    wasted_cost: float      # cost from failed phases
    # Parts of input_tokens read from or written to a prompt cache.
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass(frozen=True)
class TenantCostResult:
    tenant_id: str
    since: str
    sessions: int
    total_cost: float
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass(frozen=True)
class ModelPrice:
    """One model's bundled, offline token rates.

    A cache rate is ``None`` when the provider publishes none for the model;
    :meth:`cost` then bills those tokens at the input rate (or, for a long
    cache write, at the default write rate).
    """

    model_key: str
    provider: str
    input_per_million: float
    output_per_million: float
    cache_read_per_million: float | None = None
    cache_write_per_million: float | None = None
    cache_write_long_per_million: float | None = None

    def cost(self, usage: TokenUsage) -> float:
        """Return the USD cost of *usage* at these rates."""
        rates = {"input": self.input_per_million, "output": self.output_per_million}
        for key, value in (
            ("cache_read", self.cache_read_per_million),
            ("cache_write", self.cache_write_per_million),
            ("cache_write_long", self.cache_write_long_per_million),
        ):
            if value is not None:
                rates[key] = value
        return _usage_cost(usage, rates)


@dataclass(frozen=True)
class ProviderCostRecord:
    """Raw provider cost for one model used in one session.

    ``input_tokens`` is the whole input; ``cache_read_tokens`` and
    ``cache_write_tokens`` are the parts of it read from or written to a
    prompt cache.
    """

    provider: str
    model: str
    session_id: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass(frozen=True)
class ProviderCostSummary:
    """Aggregated provider/model totals used by the text dashboard."""

    provider: str
    model: str
    sessions: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass
class ProviderCostReport:
    """Provider cost records for a selected time window."""

    records: list[ProviderCostRecord]
    since: str
    since_timestamp: float
    generated_at: float
    pricing_snapshot_date: str = PRICING_SNAPSHOT_DATE

    @property
    def summaries(self) -> list[ProviderCostSummary]:
        grouped: dict[tuple[str, str], dict[str, float | int]] = {}
        for record in self.records:
            key = (record.provider, record.model)
            values = grouped.setdefault(
                key,
                {
                    "sessions": 0, "input_tokens": 0, "output_tokens": 0,
                    "cache_read_tokens": 0, "cache_write_tokens": 0, "cost_usd": 0.0,
                },
            )
            values["sessions"] += 1
            values["input_tokens"] += record.input_tokens
            values["output_tokens"] += record.output_tokens
            values["cache_read_tokens"] += record.cache_read_tokens
            values["cache_write_tokens"] += record.cache_write_tokens
            values["cost_usd"] += record.cost_usd
        summaries = [
            ProviderCostSummary(
                provider=provider,
                model=model,
                sessions=int(values["sessions"]),
                input_tokens=int(values["input_tokens"]),
                output_tokens=int(values["output_tokens"]),
                cost_usd=float(values["cost_usd"]),
                cache_read_tokens=int(values["cache_read_tokens"]),
                cache_write_tokens=int(values["cache_write_tokens"]),
            )
            for (provider, model), values in grouped.items()
        ]
        return sorted(summaries, key=lambda row: (-row.cost_usd, row.provider, row.model))

    @property
    def total_cost_usd(self) -> float:
        return sum(record.cost_usd for record in self.records)

    @property
    def session_count(self) -> int:
        return len({record.session_id for record in self.records})


class LivePricingUnavailable(RuntimeError):
    """Raised when live pricing is requested without reliable provider APIs."""


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Estimate token count from character length (4 chars ≈ 1 token)."""
    return max(1, len(text) // 4)


INPUT_TYPES = {EventType.USER_PROMPT, EventType.LLM_REQUEST, EventType.TOOL_CALL}
OUTPUT_TYPES = {EventType.ASSISTANT_RESPONSE, EventType.LLM_RESPONSE, EventType.TOOL_RESULT}


def _event_tokens(event: TraceEvent) -> tuple[int, int]:
    """Return (input_tokens, output_tokens) estimated for one event."""
    if event.event_type in INPUT_TYPES:
        return _estimate_tokens(json.dumps(event.data)), 0

    if event.event_type in OUTPUT_TYPES:
        return 0, _estimate_tokens(json.dumps(event.data))

    return 0, 0


def _recorded_token_usage(data: Any) -> TokenUsage | None:
    """Return the token usage an event recorded, or None when it has none."""
    if not isinstance(data, Mapping):
        return None
    usage, has_inp, has_out = _usage_detail(data)
    if not (has_inp or has_out):
        return None
    return usage


def _recorded_usage(data: Any) -> tuple[int, int] | None:
    """Return the (input, output) tokens an event recorded, or None."""
    usage = _recorded_token_usage(data)
    if usage is None:
        return None
    return usage.input_tokens, usage.output_tokens


class TokenCounter:
    """Count (input, output) tokens per event of one session, in trace order.

    Each event is estimated from its JSON size by default. Some providers
    record each model request only as an ``llm_response`` with no
    ``llm_request`` (Codex ``PostModelRequest``), and that response carries
    the request's exact usage. Once such a response with token counts is
    seen, the counter switches to recorded usage: every ``llm_response``
    counts its own recorded tokens, or none when it recorded none (a failed
    or cancelled request), and every other event counts nothing, because its
    content is already part of some request's input.
    """

    def __init__(self, recorded_usage: bool = False) -> None:
        self.recorded_usage = recorded_usage
        self._requests: Pairer[bool] = llm_pairer()

    def observe(self, event: TraceEvent) -> None:
        """Update the request pairing and switch to recorded usage if due."""
        if event.event_type == EventType.LLM_REQUEST:
            self._requests.add(event, True)
        elif event.event_type == EventType.LLM_RESPONSE:
            paired = self._requests.match(event) is not None
            if not paired and _recorded_usage(event.data) is not None:
                self.recorded_usage = True

    def usage(self, event: TraceEvent) -> TokenUsage:
        """Observe *event* and return its token usage.

        Size estimates have no cache split; recorded usage keeps the cache
        reads and writes the provider reported.
        """
        if not self.recorded_usage:
            self.observe(event)
        if not self.recorded_usage:
            return TokenUsage(*_event_tokens(event))
        if event.event_type != EventType.LLM_RESPONSE:
            return TokenUsage()
        return _recorded_token_usage(event.data) or TokenUsage()

    def count(self, event: TraceEvent) -> tuple[int, int]:
        """Observe *event* and return its (input, output) tokens."""
        usage = self.usage(event)
        return usage.input_tokens, usage.output_tokens


def session_records_usage(events: list[TraceEvent]) -> bool:
    """True when *events* hold an unpaired ``llm_response`` with token counts."""
    counter = TokenCounter()
    for event in events:
        counter.observe(event)
        if counter.recorded_usage:
            return True
    return False


def session_event_usage(events: list[TraceEvent]) -> list[TokenUsage]:
    """Return the token usage of each event of one session.

    A session that records usage (see :class:`TokenCounter`) counts it for
    every event, including those before the first usage response. Any other
    session gets the size estimate of :func:`_event_tokens` for each event.
    """
    counter = TokenCounter(recorded_usage=session_records_usage(events))
    return [counter.usage(event) for event in events]


def session_event_tokens(events: list[TraceEvent]) -> list[tuple[int, int]]:
    """Return (input, output) tokens for each event of one session.

    See :func:`session_event_usage`; input includes cache reads and writes.
    """
    return [
        (usage.input_tokens, usage.output_tokens)
        for usage in session_event_usage(events)
    ]


def _usage_cost(usage: TokenUsage, rates: Mapping[str, float]) -> float:
    """Return the USD cost of *usage* at *rates* (USD per 1M tokens).

    Cache reads and writes without a rate of their own are billed at the
    input rate, and long cache writes without one at the default write rate.
    """
    input_rate = float(rates["input"])
    read_rate = float(rates.get("cache_read", input_rate))
    write_rate = float(rates.get("cache_write", input_rate))
    long_write_rate = float(rates.get("cache_write_long", write_rate))
    long_writes = min(usage.cache_write_long_tokens, usage.cache_write_tokens)
    short_writes = usage.cache_write_tokens - long_writes
    return (
        usage.uncached_input_tokens / 1_000_000 * input_rate
        + usage.cache_read_tokens / 1_000_000 * read_rate
        + short_writes / 1_000_000 * write_rate
        + long_writes / 1_000_000 * long_write_rate
        + usage.output_tokens / 1_000_000 * float(rates["output"])
    )


def _usage_dollars(usage: TokenUsage, model: str) -> float:
    """Price *usage* at the ``PRICING`` entry *model*, or the default model."""
    return _usage_cost(usage, PRICING.get(model, PRICING[DEFAULT_MODEL]))


def _dollars(input_tokens: int, output_tokens: int, model: str) -> float:
    """Price input and output tokens with no cache reads or writes."""
    return _usage_dollars(TokenUsage(input_tokens, output_tokens), model)


# ---------------------------------------------------------------------------
# Provider cost dashboard helpers
# ---------------------------------------------------------------------------

_BEDROCK_MODEL_RE = re.compile(
    r"^(?:(?:us|eu|apac|global)\.)?"
    r"(?:amazon|anthropic|ai21|cohere|meta|mistral|openai|qwen|writer)\."
)
_RELATIVE_SINCE_RE = re.compile(r"^(\d+(?:\.\d+)?)([dhw])$", re.IGNORECASE)


def infer_provider(model: str) -> str:
    """Infer a billing provider from a stored model identifier.

    Bedrock inference-profile IDs are detected before their underlying model
    vendor, so ``us.anthropic.claude-*`` is correctly attributed to AWS rather
    than Anthropic's direct API.
    """
    value = str(model or "").strip().lower()
    if not value:
        return "Unknown"
    if value.startswith(("bedrock/", "aws/bedrock/")) or _BEDROCK_MODEL_RE.match(value):
        return "AWS Bedrock"
    if "claude" in value or value.startswith("anthropic/"):
        return "Anthropic"
    if "gemini" in value or value.startswith(("google/", "models/gemini")):
        return "Gemini"
    if (
        value.startswith(("gpt-", "chatgpt-", "openai/", "azure/openai/"))
        or re.match(r"^o[134](?:-|$)", value)
    ):
        return "OpenAI"
    return "Unknown"


def _pricing_candidates(model: str) -> list[str]:
    value = str(model or "").strip().lower()
    candidates = [value]

    for prefix in ("bedrock/", "aws/bedrock/", "openai/", "anthropic/", "google/"):
        if value.startswith(prefix):
            candidates.append(value[len(prefix):])
    if value.startswith("models/"):
        candidates.append(value[len("models/"):])

    for candidate in list(candidates):
        without_region = re.sub(r"^(?:us|eu|apac|global)\.", "", candidate)
        candidates.append(without_region)

    # Keep order deterministic while removing duplicates.
    return list(dict.fromkeys(candidates))


def get_model_pricing(
    model: str,
    pricing_table: Mapping[str, Mapping[str, float]] = PRICING,
) -> ModelPrice | None:
    """Return bundled pricing for *model*, including dated model variants.

    Unknown models deliberately return ``None`` rather than silently using the
    Sonnet fallback retained by :func:`_dollars` for backward compatibility.
    This keeps the cross-provider dashboard from fabricating spend.
    """
    model_keys = sorted(_MODEL_PROVIDERS, key=len, reverse=True)
    for candidate in _pricing_candidates(model):
        for key in model_keys:
            if (
                candidate == key
                or candidate.startswith(f"{key}:")
                or candidate.startswith(f"{key}@")
                or candidate.startswith(f"{key}-")
            ):
                rates = pricing_table.get(key)
                if rates is None:
                    return None
                return ModelPrice(
                    model_key=key,
                    provider=infer_provider(model),
                    input_per_million=float(rates["input"]),
                    output_per_million=float(rates["output"]),
                    cache_read_per_million=_optional_rate(rates, "cache_read"),
                    cache_write_per_million=_optional_rate(rates, "cache_write"),
                    cache_write_long_per_million=_optional_rate(rates, "cache_write_long"),
                )
    return None


def _optional_rate(rates: Mapping[str, float], key: str) -> float | None:
    value = rates.get(key)
    return None if value is None else float(value)


def parse_since(value: str | None, now: float | None = None) -> float:
    """Parse ``Nd``/``Nh``/``Nw`` or an ISO date into an epoch cutoff.

    Date-only values are interpreted as midnight UTC, making reports stable
    across machines with different local time zones.
    """
    current = _time.time() if now is None else float(now)
    raw = (value or "30d").strip()
    match = _RELATIVE_SINCE_RE.fullmatch(raw)
    if match:
        amount = float(match.group(1))
        unit_seconds = {"d": 86400, "h": 3600, "w": 7 * 86400}
        return current - amount * unit_seconds[match.group(2).lower()]

    try:
        parsed = _datetime.datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            f"Cannot parse --since value: {raw!r}. "
            "Use an ISO date (2026-06-01) or relative duration (30d)."
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_datetime.timezone.utc)
    return parsed.timestamp()


_INPUT_TOKEN_KEYS = (
    "input_tokens", "prompt_tokens", "inputTokens", "promptTokenCount",
    "prompt_token_count",
)
_OUTPUT_TOKEN_KEYS = (
    "output_tokens", "completion_tokens", "outputTokens", "candidatesTokenCount",
    "candidates_token_count", "completion_token_count",
)
_GEMINI_INPUT_KEYS = frozenset({"promptTokenCount", "prompt_token_count"})
_GEMINI_OUTPUT_KEYS = frozenset({"candidatesTokenCount", "candidates_token_count"})

# (cache read, cache write, long-lifetime cache write) token counts.
_CacheSplit = tuple[int, int, int]


def _count(value: Any) -> int | None:
    """Return *value* as a non-negative token count, or None."""
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number >= 0 else None


def _token_entry(data: Mapping[str, Any], keys: tuple[str, ...]) -> tuple[int, str | None]:
    for key in keys:
        if key not in data:
            continue
        number = _count(data[key])
        if number is not None:
            return number, key
    return 0, None


def _token_value(data: Mapping[str, Any], keys: tuple[str, ...]) -> tuple[int, bool]:
    number, key = _token_entry(data, keys)
    return number, key is not None


def _first_count(data: Any, *keys: str) -> int | None:
    if not isinstance(data, Mapping):
        return None
    number, key = _token_entry(data, keys)
    return number if key is not None else None


def _anthropic_cache(data: Mapping[str, Any]) -> _CacheSplit | None:
    """Anthropic Messages: cache tokens are reported beside ``input_tokens``."""
    read = _first_count(data, "cache_read_input_tokens")
    write = _first_count(data, "cache_creation_input_tokens")
    split = data.get("cache_creation")
    short_write = _first_count(split, "ephemeral_5m_input_tokens")
    long_write = _first_count(split, "ephemeral_1h_input_tokens")
    if read is None and write is None and short_write is None and long_write is None:
        return None
    if write is None:
        write = (short_write or 0) + (long_write or 0)
    return read or 0, write, min(long_write or 0, write)


def _bedrock_cache(data: Mapping[str, Any]) -> _CacheSplit | None:
    """Bedrock Converse: cache tokens are reported beside ``inputTokens``."""
    read = _first_count(data, "cacheReadInputTokens")
    write = _first_count(data, "cacheWriteInputTokens")
    details = data.get("cacheDetails")
    detail_writes = long_write = 0
    has_details = False
    if isinstance(details, list):
        for item in details:
            tokens = _first_count(item, "inputTokens")
            if tokens is None:
                continue
            has_details = True
            detail_writes += tokens
            if item.get("ttl") == "1h":
                long_write += tokens
    if read is None and write is None and not has_details:
        return None
    if write is None:
        write = detail_writes
    return read or 0, write, min(long_write, write)


def _codex_cache(data: Mapping[str, Any]) -> _CacheSplit | None:
    """Codex ``token_usage``: cache tokens are part of ``input_tokens``."""
    read = _first_count(data, "cached_input_tokens")
    write = _first_count(data, "cache_write_input_tokens")
    if read is None and write is None:
        return None
    return read or 0, write or 0, 0


def _openai_cache(details: Any) -> _CacheSplit | None:
    """OpenAI ``input_tokens_details`` or ``prompt_tokens_details``.

    Their cache tokens are part of ``input_tokens`` or ``prompt_tokens``.
    """
    if not isinstance(details, Mapping):
        return None
    read = _first_count(details, "cached_tokens")
    write = _first_count(details, "cache_write_tokens")
    ttl = details.get("cache_write_tokens_details")
    short_write = _first_count(ttl, "ttl_30m_tokens")
    long_write = _first_count(ttl, "ttl_12h_tokens")
    if read is None and write is None and short_write is None and long_write is None:
        return None
    if write is None:
        write = (short_write or 0) + (long_write or 0)
    return read or 0, write, min(long_write or 0, write)


def _usage_shape(data: Mapping[str, Any]) -> tuple[TokenUsage, bool, bool, bool]:
    """Normalize one usage mapping.

    Returns ``(usage, has_input, has_output, has_cache_detail)``.  The usage
    input is the whole input whichever way the provider counts it:

    - Anthropic ``input_tokens`` and Bedrock ``inputTokens`` exclude cache
      reads and writes, so they are added.
    - Codex ``input_tokens``, OpenAI ``input_tokens``/``prompt_tokens`` and
      Gemini ``promptTokenCount`` include them, so they are not.
    - Gemini tool-use prompt tokens are added to input and thinking tokens to
      output, because Gemini bills them as such.  OpenAI and Codex output
      already include reasoning tokens.
    """
    input_tokens, input_key = _token_entry(data, _INPUT_TOKEN_KEYS)
    output_tokens, output_key = _token_entry(data, _OUTPUT_TOKEN_KEYS)
    has_input = input_key is not None
    has_output = output_key is not None
    cache: _CacheSplit | None = None
    beside_input = False

    if input_key == "input_tokens":
        cache = _anthropic_cache(data)
        beside_input = cache is not None
        if cache is None:
            cache = _codex_cache(data) or _openai_cache(data.get("input_tokens_details"))
    elif input_key == "prompt_tokens":
        cache = _openai_cache(data.get("prompt_tokens_details"))
    elif input_key == "inputTokens":
        cache = _bedrock_cache(data)
        beside_input = cache is not None
    elif input_key in _GEMINI_INPUT_KEYS:
        read = _first_count(data, "cachedContentTokenCount", "cached_content_token_count")
        if read is not None:
            cache = (read, 0, 0)
        tool_prompt = _first_count(
            data, "toolUsePromptTokenCount", "tool_use_prompt_token_count",
        )
        input_tokens += tool_prompt or 0

    if input_key in _GEMINI_INPUT_KEYS or output_key in _GEMINI_OUTPUT_KEYS:
        thoughts = _first_count(data, "thoughtsTokenCount", "thoughts_token_count")
        if thoughts is not None:
            output_tokens += thoughts
            has_output = True

    read = write = long_write = 0
    if cache is not None:
        read, write, long_write = cache
        if beside_input:
            input_tokens += read + write
        else:
            read = min(read, input_tokens)
            write = min(write, input_tokens - read)
            long_write = min(long_write, write)
    usage = TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=read,
        cache_write_tokens=write,
        cache_write_long_tokens=long_write,
    )
    return usage, has_input, has_output, cache is not None


def _nested_usage(data: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    candidates = [
        data.get("usage"),
        data.get("usage_metadata"),
        data.get("usageMetadata"),
        data.get("token_usage"),
    ]
    response_metadata = data.get("response_metadata")
    if isinstance(response_metadata, Mapping):
        candidates.append(response_metadata.get("token_usage"))
    return [candidate for candidate in candidates if isinstance(candidate, Mapping)]


def _usage_detail(data: Mapping[str, Any]) -> tuple[TokenUsage, bool, bool]:
    """Read common provider token-counter shapes without an SDK dependency.

    Returns ``(usage, has_input, has_output)``.  Counters at the top level of
    *data* win over nested usage objects.  When the top-level counters carry
    no cache detail but repeat a nested object's input and output counts, as
    the Codex hooks copy them from ``token_usage``, the nested object's cache
    detail is used.
    """
    usage, has_input, has_output, detailed = _usage_shape(data)
    if has_input or has_output:
        if not detailed:
            counters = (
                _token_value(data, _INPUT_TOKEN_KEYS),
                _token_value(data, _OUTPUT_TOKEN_KEYS),
            )
            for nested in _nested_usage(data):
                if (
                    _token_value(nested, _INPUT_TOKEN_KEYS),
                    _token_value(nested, _OUTPUT_TOKEN_KEYS),
                ) != counters:
                    continue
                inner, inner_input, inner_output, inner_detailed = _usage_shape(nested)
                if inner_detailed:
                    return inner, inner_input, inner_output
        return usage, has_input, has_output

    for nested in _nested_usage(data):
        usage, has_input, has_output, _ = _usage_shape(nested)
        if has_input or has_output:
            return usage, has_input, has_output
    return TokenUsage(), False, False


def _usage_tokens(data: Mapping[str, Any]) -> tuple[int, int, bool, bool]:
    """Return ``(input, output, has_input, has_output)``; see :func:`_usage_detail`."""
    usage, has_input, has_output = _usage_detail(data)
    return usage.input_tokens, usage.output_tokens, has_input, has_output


@dataclass
class _UsageSample:
    model: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    has_input: bool = False
    has_output: bool = False

    def record(self, usage: TokenUsage, has_input: bool, has_output: bool) -> None:
        """Replace the input side, the output side, or both with *usage*."""
        if has_input:
            self.usage = _replace(usage, output_tokens=self.usage.output_tokens)
            self.has_input = True
        if has_output:
            self.usage = _replace(self.usage, output_tokens=usage.output_tokens)
            self.has_output = True


_NO_SAMPLE = -1

# Shown in the text dashboard for usage recorded without a model name.
UNRECORDED_MODEL_LABEL = "(not recorded)"

# _session_model_usage reads the data of every event of these types.
MODEL_USAGE_EVENT_TYPES = frozenset({
    EventType.LLM_REQUEST,
    EventType.LLM_RESPONSE,
    EventType.ASSISTANT_RESPONSE,
})


def _session_model_token_usage(events: list[TraceEvent]) -> dict[str, TokenUsage]:
    """Aggregate usage by model while avoiding request/response double counts.

    Responses pair with requests under the pairing.py LLM rules. An
    assistant_response counts as a response too, because it can carry usage.
    A request that names no model is not a usage sample. It still joins the
    pairer, so a response that names it by parent_id does not pair with
    another request; that response counts on its own.

    An llm_response that pairs with no request and names no model, but
    records token counts (a Codex request whose model was not reported),
    is a sample under the empty model name, so its tokens are reported
    without inventing a model or a price.
    """
    samples: list[_UsageSample] = []
    requests: Pairer[int] = llm_pairer()

    for event in events:
        if event.event_type not in MODEL_USAGE_EVENT_TYPES:
            continue

        model = event_model(event.data)
        recorded, has_inp, has_out = _usage_detail(event.data)

        if event.event_type == EventType.LLM_REQUEST:
            if not model:
                requests.add(event, _NO_SAMPLE)
                continue
            sample = _UsageSample(model)
            sample.record(recorded, has_inp, has_out)
            samples.append(sample)
            requests.add(event, len(samples) - 1)
            continue

        index = requests.match(event)
        if index is None or index == _NO_SAMPLE:
            unrequested_usage = (
                index is None
                and event.event_type == EventType.LLM_RESPONSE
                and (has_inp or has_out)
            )
            if not model and not unrequested_usage:
                continue
            samples.append(_UsageSample(model))
            index = len(samples) - 1
        sample = samples[index]
        if not model:
            model = sample.model
        sample.record(recorded, has_inp, has_out)

    usage: dict[str, TokenUsage] = {}
    for sample in samples:
        usage[sample.model] = usage.get(sample.model, TokenUsage()) + sample.usage
    return usage


def _session_model_usage(events: list[TraceEvent]) -> dict[str, tuple[int, int]]:
    """Return ``{model: (input, output)}``; see :func:`_session_model_token_usage`."""
    return {
        model: (usage.input_tokens, usage.output_tokens)
        for model, usage in _session_model_token_usage(events).items()
    }


def build_provider_cost_report(
    store: TraceStore,
    since: str | None = "30d",
    *,
    now: float | None = None,
    live_pricing: bool = False,
    pricing_table: Mapping[str, Mapping[str, float]] = PRICING,
    tenant_id: str | None = None,
) -> ProviderCostReport:
    """Aggregate stored session usage by inferred provider and model.

    The default mode is fully offline.  Live pricing fails explicitly because
    the four providers do not expose a compatible, authoritative pricing API;
    scraping their billing pages would be less reliable than a dated snapshot.
    """
    if live_pricing:
        raise LivePricingUnavailable(
            "Live pricing is unavailable: Anthropic, OpenAI, AWS Bedrock, and "
            "Gemini do not expose one compatible pricing API. Use the bundled "
            f"offline estimate (snapshot {PRICING_SNAPSHOT_DATE})."
        )

    generated_at = _time.time() if now is None else float(now)
    since_value = since or "30d"
    cutoff = parse_since(since_value, now=generated_at)
    records: list[ProviderCostRecord] = []

    for meta in store.list_sessions(tenant_id=tenant_id):
        if meta.started_at < cutoff or meta.started_at > generated_at:
            continue
        try:
            events = store.load_events(meta.session_id)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        require_object_event_data(
            events, event_types=MODEL_USAGE_EVENT_TYPES, session_id=meta.session_id
        )
        for model, usage in _session_model_token_usage(events).items():
            price = get_model_pricing(model, pricing_table=pricing_table)
            cost = 0.0 if price is None else price.cost(usage)
            records.append(ProviderCostRecord(
                provider=infer_provider(model),
                model=model,
                session_id=meta.session_id,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cost_usd=cost,
                cache_read_tokens=usage.cache_read_tokens,
                cache_write_tokens=usage.cache_write_tokens,
            ))

    records.sort(key=lambda row: (row.provider, row.model, row.session_id))
    return ProviderCostReport(
        records=records,
        since=since_value,
        since_timestamp=cutoff,
        generated_at=generated_at,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_cost(
    store: TraceStore,
    session_id: str,
    model: str = DEFAULT_MODEL,
    input_price: float | None = None,
    output_price: float | None = None,
) -> CostResult:
    """Estimate cost for *session_id*, broken down by phase.

    Recorded cache reads and writes are billed at the model's cache rates.
    A custom ``input_price`` applies to them too, because it has none.
    """
    if (input_price is None) != (output_price is None):
        raise ValueError(
            "Both --input-price and --output-price must be provided together"
        )
    if input_price is not None and output_price is not None:
        PRICING["custom"] = {"input": input_price, "output": output_price}
        model = "custom"

    result = explain_session(store, session_id)
    # Phases hold every event of the session, in order.
    usages = iter(session_event_usage(
        [event for phase in result.phases for event in phase.events]
    ))

    phase_costs: list[PhaseCost] = []
    total = TokenUsage()

    for phase in result.phases:
        usage = TokenUsage()
        for _ in phase.events:
            usage = usage + next(usages)
        phase_costs.append(PhaseCost(
            phase_index=phase.index,
            phase_name=phase.name,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_dollars=_usage_dollars(usage, model),
            failed=phase.failed,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
        ))
        total = total + usage

    total_cost = _usage_dollars(total, model)
    wasted_cost = sum(pc.cost_dollars for pc in phase_costs if pc.failed)

    return CostResult(
        session_id=session_id,
        model=model,
        total_cost=total_cost,
        input_tokens=total.input_tokens,
        output_tokens=total.output_tokens,
        phase_costs=phase_costs,
        wasted_cost=wasted_cost,
        cache_read_tokens=total.cache_read_tokens,
        cache_write_tokens=total.cache_write_tokens,
    )


def build_tenant_cost_report(
    store: TraceStore,
    tenant_id: str,
    since: str = "30d",
    model: str = DEFAULT_MODEL,
    input_price: float | None = None,
    output_price: float | None = None,
    now: float | None = None,
) -> TenantCostResult:
    """Aggregate legacy per-session estimates for one exact tenant tag."""
    generated_at = _time.time() if now is None else float(now)
    cutoff = parse_since(since, now=generated_at)
    sessions = total_input = total_output = total_read = total_write = 0
    total_cost = 0.0
    for meta in store.list_sessions(tenant_id=tenant_id):
        if meta.started_at < cutoff or meta.started_at > generated_at:
            continue
        try:
            events = store.load_events(meta.session_id)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        require_object_event_data(
            events, event_types=PHASE_DATA_EVENT_TYPES, session_id=meta.session_id
        )
        try:
            result = estimate_cost(
                store,
                meta.session_id,
                model=model,
                input_price=input_price,
                output_price=output_price,
            )
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        sessions += 1
        total_input += result.input_tokens
        total_output += result.output_tokens
        total_read += result.cache_read_tokens
        total_write += result.cache_write_tokens
        total_cost += result.total_cost
    return TenantCostResult(
        tenant_id=tenant_id,
        since=since,
        sessions=sessions,
        total_cost=total_cost,
        input_tokens=total_input,
        output_tokens=total_output,
        cache_read_tokens=total_read,
        cache_write_tokens=total_write,
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _cache_note(read_tokens: int, write_tokens: int) -> str:
    """Describe the cached part of an input count, or return ''."""
    parts = []
    if read_tokens:
        parts.append(f"{read_tokens:,} cache read")
    if write_tokens:
        parts.append(f"{write_tokens:,} cache write")
    return f" ({', '.join(parts)})" if parts else ""


def format_cost(result: CostResult, out: TextIO = sys.stdout) -> None:
    w = out.write
    w(f"\nSession: {result.session_id} — Estimated cost: ${result.total_cost:.4f}\n")
    cache = _cache_note(result.cache_read_tokens, result.cache_write_tokens)
    w(f"Model: {result.model}  |  "
      f"{result.input_tokens:,} input tokens{cache}, "
      f"{result.output_tokens:,} output tokens\n\n")

    if result.phase_costs:
        total = result.total_cost or 1e-9  # avoid div/0
        for pc in result.phase_costs:
            pct = pc.cost_dollars / total * 100
            wasted_tag = "  ← wasted" if pc.failed else ""
            w(f"  Phase {pc.phase_index}: {pc.phase_name[:40]:<40}  "
              f"${pc.cost_dollars:.4f}  ({pct:.0f}%)  "
              f"{pc.input_tokens:,}in {pc.output_tokens:,}out"
              f"{wasted_tag}\n")

    w("\n")

    if result.wasted_cost > 0:
        wasted_pct = result.wasted_cost / (result.total_cost or 1e-9) * 100
        w(f"Wasted on failed phases: ${result.wasted_cost:.4f} ({wasted_pct:.0f}%)\n\n")


def format_tenant_cost(result: TenantCostResult, out: TextIO = sys.stdout) -> None:
    out.write(
        f"Tenant: {result.tenant_id}\n"
        f"Window: {_since_description(result.since)}\n"
        f"Sessions: {result.sessions}\n"
        f"Estimated cost: ${result.total_cost:.4f}\n"
        f"Tokens: {result.input_tokens:,} input"
        f"{_cache_note(result.cache_read_tokens, result.cache_write_tokens)}, "
        f"{result.output_tokens:,} output\n"
    )


def _since_description(value: str) -> str:
    match = _RELATIVE_SINCE_RE.fullmatch(value.strip())
    if not match:
        return f"since {value}"
    amount = float(match.group(1))
    unit = match.group(2).lower()
    labels = {"d": "day", "h": "hour", "w": "week"}
    display_amount = str(int(amount)) if amount.is_integer() else str(amount)
    plural = "" if amount == 1 else "s"
    return f"last {display_amount} {labels[unit]}{plural}"


def format_provider_cost_report(
    report: ProviderCostReport,
    out: TextIO = sys.stdout,
) -> None:
    """Write the offline provider/model cost dashboard."""
    summaries = report.summaries
    models = [row.model or UNRECORDED_MODEL_LABEL for row in summaries]
    provider_width = max([len("Provider"), *(len(row.provider) for row in summaries)])
    model_width = max([len("Model"), *(len(model) for model in models)])
    model_width = min(max(model_width, 24), 44)
    divider = "─" * (provider_width + model_width + 26)

    out.write(
        f"Cost breakdown — {_since_description(report.since)}\n"
        f"Offline estimates · pricing snapshot {report.pricing_snapshot_date}\n"
        f"{divider}\n"
        f"{'Provider':<{provider_width}}  {'Model':<{model_width}}  {'Sessions':>8}  {'Cost':>10}\n"
        f"{divider}\n"
    )
    for row, model in zip(summaries, models):
        if len(model) > model_width:
            model = model[:model_width - 1] + "…"
        out.write(
            f"{row.provider:<{provider_width}}  {model:<{model_width}}  "
            f"{row.sessions:>8}  ${row.cost_usd:>9.2f}\n"
        )
    out.write(f"{divider}\n")
    out.write(
        f"{'Total':<{provider_width}}  {'':<{model_width}}  "
        f"{report.session_count:>8}  ${report.total_cost_usd:>9.2f}\n"
    )
    out.write(f"{divider}\n")


def format_provider_cost_csv(
    report: ProviderCostReport,
    out: TextIO = sys.stdout,
) -> None:
    """Write raw per-session/model rows.

    The first six columns are stable; the cache columns were appended later.
    """
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow([
        "provider", "model", "session_id", "input_tokens", "output_tokens", "cost_usd",
        "cache_read_tokens", "cache_write_tokens",
    ])
    for row in report.records:
        writer.writerow([
            row.provider,
            row.model,
            row.session_id,
            row.input_tokens,
            row.output_tokens,
            f"{row.cost_usd:.8f}",
            row.cache_read_tokens,
            row.cache_write_tokens,
        ])


# ---------------------------------------------------------------------------
# CLI handler
# ---------------------------------------------------------------------------

def cmd_cost(args: argparse.Namespace) -> int:
    store = TraceStore(args.trace_dir)
    tenant_id = getattr(args, "tenant", None)
    if tenant_id is not None:
        from .store import validate_tenant_id
        try:
            tenant_id = validate_tenant_id(tenant_id)
        except ValueError as exc:
            sys.stderr.write(f"Error: {exc}\n")
            return 1

    breakdown = getattr(args, "breakdown", None)
    if breakdown:
        if breakdown != "provider":
            sys.stderr.write(f"Unsupported cost breakdown: {breakdown}\n")
            return 1
        try:
            report = build_provider_cost_report(
                store,
                since=getattr(args, "since", None) or "30d",
                live_pricing=bool(getattr(args, "live_pricing", False)),
                tenant_id=tenant_id,
            )
        except (ValueError, LivePricingUnavailable) as exc:
            sys.stderr.write(f"Error: {exc}\n")
            return 1
        if getattr(args, "csv", False):
            format_provider_cost_csv(report, out=sys.stdout)
        else:
            format_provider_cost_report(report, out=sys.stdout)
        return 0

    session_id = getattr(args, "session_id", None)
    model = getattr(args, "model", DEFAULT_MODEL) or DEFAULT_MODEL
    input_price = getattr(args, "input_price", None)
    output_price = getattr(args, "output_price", None)

    if tenant_id is not None and not session_id:
        if (input_price is None) != (output_price is None):
            sys.stderr.write(
                "Error: --input-price and --output-price must be provided together.\n"
            )
            return 1
        try:
            report = build_tenant_cost_report(
                store,
                tenant_id,
                since=getattr(args, "since", None) or "30d",
                model=model,
                input_price=input_price,
                output_price=output_price,
            )
        except ValueError as exc:
            sys.stderr.write(f"Error: {exc}\n")
            return 1
        format_tenant_cost(report)
        return 0

    if not session_id:
        session_id = store.get_latest_session_id()
    if not session_id:
        sys.stderr.write("No sessions found.\n")
        return 1
    full_id = store.find_session(session_id, tenant_id=tenant_id)
    if not full_id:
        if tenant_id:
            sys.stderr.write(f"Session not found for tenant {tenant_id}: {session_id}\n")
        else:
            sys.stderr.write(f"Session not found: {session_id}\n")
        return 1

    if (input_price is None) != (output_price is None):
        sys.stderr.write(
            "Error: --input-price and --output-price must be provided together.\n"
        )
        return 1

    require_object_event_data(
        store.load_events(full_id), event_types=PHASE_DATA_EVENT_TYPES
    )
    result = estimate_cost(
        store, full_id,
        model=model,
        input_price=input_price,
        output_price=output_price,
    )
    format_cost(result)
    return 0
