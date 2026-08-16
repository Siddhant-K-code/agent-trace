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
from dataclasses import dataclass, field
from typing import Any, Mapping, TextIO

from .explain import ExplainResult, Phase, explain_session
from .models import EventType, TraceEvent
from .store import TraceStore


# ---------------------------------------------------------------------------
# Pricing table  (dollars per 1M tokens)
# ---------------------------------------------------------------------------

PRICING_SNAPSHOT_DATE = "2026-08-16"
"""Effective date of the bundled, offline pricing snapshot.

Rates are estimates in USD per one million text tokens.  They intentionally
exclude provider-specific discounts, caching, batch rates, regional pricing,
long-context tiers, and non-token charges.  Updating this table requires
checking each URL in ``PRICING_SOURCES``, updating ``PRICING`` and
``_MODEL_PROVIDERS``, advancing the snapshot date, and extending the
provider-cost tests for every added model family.
"""

PRICING_SOURCES: dict[str, str] = {
    "Anthropic": "https://platform.claude.com/docs/en/about-claude/pricing",
    "OpenAI": "https://openai.com/api/pricing/",
    "AWS Bedrock": "https://aws.amazon.com/bedrock/pricing/",
    "Gemini": "https://ai.google.dev/gemini-api/docs/pricing",
}


PRICING: dict[str, dict[str, float]] = {
    # Backward-compatible short names used by the existing per-session report.
    "sonnet": {"input": 3.00,  "output": 15.00},
    "opus":   {"input": 15.00, "output": 75.00},
    "haiku":  {"input": 0.25,  "output": 1.25},
    "gpt4":   {"input": 30.00, "output": 60.00},
    "gpt4o":  {"input": 5.00,  "output": 15.00},

    # Anthropic direct API.
    "claude-opus-4-6": {"input": 5.00, "output": 25.00},
    "claude-opus-4-5": {"input": 5.00, "output": 25.00},
    "claude-opus-4-1": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4": {"input": 3.00, "output": 15.00},
    "claude-3-7-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "claude-3-opus": {"input": 15.00, "output": 75.00},

    # OpenAI direct API (standard, non-batch text-token rates).
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "o3": {"input": 2.00, "output": 8.00},
    "o3-mini": {"input": 1.10, "output": 4.40},
    "o4-mini": {"input": 1.10, "output": 4.40},

    # Amazon models through AWS Bedrock.  Actual charges can vary by region
    # and inference tier; these are standard on-demand snapshot rates.
    "amazon.nova-pro-v1": {"input": 0.80, "output": 3.20},
    "amazon.nova-lite-v1": {"input": 0.06, "output": 0.24},
    "amazon.nova-micro-v1": {"input": 0.035, "output": 0.14},

    # Google Gemini Developer API paid tier.  The 2.5 Pro entry uses the
    # <=200k-token tier; long-context requests can cost more.
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
}
DEFAULT_MODEL = "sonnet"


_MODEL_PROVIDERS: dict[str, str] = {
    model: provider
    for provider, models in {
        "Anthropic": (
            "claude-opus-4-6", "claude-opus-4-5", "claude-opus-4-1",
            "claude-sonnet-4", "claude-3-7-sonnet", "claude-3-5-sonnet",
            "claude-haiku-4-5", "claude-3-haiku", "claude-3-opus",
        ),
        "OpenAI": (
            "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o",
            "gpt-4o-mini", "o3", "o3-mini", "o4-mini",
        ),
        "AWS Bedrock": (
            "amazon.nova-pro-v1", "amazon.nova-lite-v1",
            "amazon.nova-micro-v1",
        ),
        "Gemini": (
            "gemini-2.5-pro", "gemini-2.5-flash",
            "gemini-2.5-flash-lite", "gemini-2.0-flash",
        ),
    }.items()
    for model in models
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PhaseCost:
    phase_index: int
    phase_name: str
    input_tokens: int
    output_tokens: int
    cost_dollars: float
    failed: bool = False

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


@dataclass(frozen=True)
class TenantCostResult:
    tenant_id: str
    since: str
    sessions: int
    total_cost: float
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class ModelPrice:
    """One model's bundled, offline token rates."""

    model_key: str
    provider: str
    input_per_million: float
    output_per_million: float


@dataclass(frozen=True)
class ProviderCostRecord:
    """Raw provider cost for one model used in one session."""

    provider: str
    model: str
    session_id: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass(frozen=True)
class ProviderCostSummary:
    """Aggregated provider/model totals used by the text dashboard."""

    provider: str
    model: str
    sessions: int
    input_tokens: int
    output_tokens: int
    cost_usd: float


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
                {"sessions": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
            )
            values["sessions"] += 1
            values["input_tokens"] += record.input_tokens
            values["output_tokens"] += record.output_tokens
            values["cost_usd"] += record.cost_usd
        summaries = [
            ProviderCostSummary(
                provider=provider,
                model=model,
                sessions=int(values["sessions"]),
                input_tokens=int(values["input_tokens"]),
                output_tokens=int(values["output_tokens"]),
                cost_usd=float(values["cost_usd"]),
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


def _phase_tokens(phase: Phase) -> tuple[int, int]:
    inp = out = 0
    for event in phase.events:
        i, o = _event_tokens(event)
        inp += i
        out += o
    return inp, out


def _dollars(input_tokens: int, output_tokens: int, model: str) -> float:
    pricing = PRICING.get(model, PRICING[DEFAULT_MODEL])
    return (
        input_tokens  / 1_000_000 * pricing["input"] +
        output_tokens / 1_000_000 * pricing["output"]
    )


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
                )
    return None


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
    "completion_token_count",
)


def _token_value(data: Mapping[str, Any], keys: tuple[str, ...]) -> tuple[int, bool]:
    for key in keys:
        if key not in data:
            continue
        value = data[key]
        if isinstance(value, bool):
            continue
        try:
            number = int(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if number >= 0:
            return number, True
    return 0, False


def _usage_tokens(data: Mapping[str, Any]) -> tuple[int, int, bool, bool]:
    """Read common provider token-counter shapes without an SDK dependency."""
    input_tokens, has_input = _token_value(data, _INPUT_TOKEN_KEYS)
    output_tokens, has_output = _token_value(data, _OUTPUT_TOKEN_KEYS)
    if has_input or has_output:
        return input_tokens, output_tokens, has_input, has_output

    nested_candidates = (
        data.get("usage"),
        data.get("usage_metadata"),
        data.get("usageMetadata"),
        data.get("token_usage"),
    )
    response_metadata = data.get("response_metadata")
    if isinstance(response_metadata, Mapping):
        nested_candidates += (response_metadata.get("token_usage"),)
    for nested in nested_candidates:
        if not isinstance(nested, Mapping):
            continue
        input_tokens, has_input = _token_value(nested, _INPUT_TOKEN_KEYS)
        output_tokens, has_output = _token_value(nested, _OUTPUT_TOKEN_KEYS)
        if has_input or has_output:
            return input_tokens, output_tokens, has_input, has_output
    return 0, 0, False, False


def _event_model(data: Mapping[str, Any]) -> str:
    for key in ("model", "model_id", "modelId"):
        value = data.get(key)
        if value:
            return str(value).strip()
    return ""


@dataclass
class _UsageSample:
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    has_input: bool = False
    has_output: bool = False


def _session_model_usage(events: list[TraceEvent]) -> dict[str, tuple[int, int]]:
    """Aggregate usage by model while avoiding request/response double counts."""
    samples: list[_UsageSample] = []
    pending: set[int] = set()
    request_indexes: dict[str, int] = {}

    for event in events:
        if event.event_type not in {
            EventType.LLM_REQUEST,
            EventType.LLM_RESPONSE,
            EventType.ASSISTANT_RESPONSE,
        }:
            continue

        model = _event_model(event.data)
        inp, out, has_inp, has_out = _usage_tokens(event.data)

        if event.event_type == EventType.LLM_REQUEST:
            if not model:
                continue
            sample = _UsageSample(model, inp, out, has_inp, has_out)
            samples.append(sample)
            index = len(samples) - 1
            pending.add(index)
            if event.event_id:
                request_indexes[event.event_id] = index
            continue

        index = request_indexes.get(event.parent_id) if event.parent_id else None
        if index is None:
            for candidate in reversed(range(len(samples))):
                if candidate in pending and (not model or samples[candidate].model == model):
                    index = candidate
                    break

        if index is None:
            if not model:
                continue
            samples.append(_UsageSample(model))
            index = len(samples) - 1
        sample = samples[index]
        if not model:
            model = sample.model
        if has_inp:
            sample.input_tokens = inp
            sample.has_input = True
        if has_out:
            sample.output_tokens = out
            sample.has_output = True
        pending.discard(index)

    usage: dict[str, list[int]] = {}
    for sample in samples:
        totals = usage.setdefault(sample.model, [0, 0])
        totals[0] += sample.input_tokens
        totals[1] += sample.output_tokens
    return {model: (values[0], values[1]) for model, values in usage.items()}


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
        for model, (input_tokens, output_tokens) in _session_model_usage(events).items():
            price = get_model_pricing(model, pricing_table=pricing_table)
            cost = 0.0
            if price is not None:
                cost = (
                    input_tokens / 1_000_000 * price.input_per_million
                    + output_tokens / 1_000_000 * price.output_per_million
                )
            records.append(ProviderCostRecord(
                provider=infer_provider(model),
                model=model,
                session_id=meta.session_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
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
    """Estimate cost for *session_id*, broken down by phase."""
    if (input_price is None) != (output_price is None):
        raise ValueError(
            "Both --input-price and --output-price must be provided together"
        )
    if input_price is not None and output_price is not None:
        PRICING["custom"] = {"input": input_price, "output": output_price}
        model = "custom"

    result = explain_session(store, session_id)

    phase_costs: list[PhaseCost] = []
    total_input = total_output = 0

    for phase in result.phases:
        inp, out = _phase_tokens(phase)
        cost = _dollars(inp, out, model)
        phase_costs.append(PhaseCost(
            phase_index=phase.index,
            phase_name=phase.name,
            input_tokens=inp,
            output_tokens=out,
            cost_dollars=cost,
            failed=phase.failed,
        ))
        total_input += inp
        total_output += out

    total_cost = _dollars(total_input, total_output, model)
    wasted_cost = sum(pc.cost_dollars for pc in phase_costs if pc.failed)

    return CostResult(
        session_id=session_id,
        model=model,
        total_cost=total_cost,
        input_tokens=total_input,
        output_tokens=total_output,
        phase_costs=phase_costs,
        wasted_cost=wasted_cost,
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
    sessions = total_input = total_output = 0
    total_cost = 0.0
    for meta in store.list_sessions(tenant_id=tenant_id):
        if meta.started_at < cutoff or meta.started_at > generated_at:
            continue
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
        total_cost += result.total_cost
    return TenantCostResult(
        tenant_id=tenant_id,
        since=since,
        sessions=sessions,
        total_cost=total_cost,
        input_tokens=total_input,
        output_tokens=total_output,
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_cost(result: CostResult, out: TextIO = sys.stdout) -> None:
    w = out.write
    w(f"\nSession: {result.session_id} — Estimated cost: ${result.total_cost:.4f}\n")
    w(f"Model: {result.model}  |  "
      f"{result.input_tokens:,} input tokens, {result.output_tokens:,} output tokens\n\n")

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
        f"Tokens: {result.input_tokens:,} input, {result.output_tokens:,} output\n"
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
    provider_width = max([len("Provider"), *(len(row.provider) for row in summaries)])
    model_width = max([len("Model"), *(len(row.model) for row in summaries)])
    model_width = min(max(model_width, 24), 44)
    divider = "─" * (provider_width + model_width + 26)

    out.write(
        f"Cost breakdown — {_since_description(report.since)}\n"
        f"Offline estimates · pricing snapshot {report.pricing_snapshot_date}\n"
        f"{divider}\n"
        f"{'Provider':<{provider_width}}  {'Model':<{model_width}}  {'Sessions':>8}  {'Cost':>10}\n"
        f"{divider}\n"
    )
    for row in summaries:
        model = row.model
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
    """Write raw per-session/model rows using the issue's stable columns."""
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow([
        "provider", "model", "session_id", "input_tokens", "output_tokens", "cost_usd",
    ])
    for row in report.records:
        writer.writerow([
            row.provider,
            row.model,
            row.session_id,
            row.input_tokens,
            row.output_tokens,
            f"{row.cost_usd:.8f}",
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

    result = estimate_cost(
        store, full_id,
        model=model,
        input_price=input_price,
        output_price=output_price,
    )
    format_cost(result)
    return 0
