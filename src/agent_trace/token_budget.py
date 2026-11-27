"""Token budget tracking: context window fill and early warning.

Each LLM request's input token count is the whole prompt the model received,
so it is the context fill at that request. This module compares that count
with the model's context limit; it never adds requests together.
Provides:

  - `token-budget` command: per-request context fill table, with the peak
  - `TokenBudgetWatcher`: fires when one request's context fill crosses a
    threshold, and re-arms when the fill drops below it again (for example
    after compaction)

The input token count comes from the request data, read the same way as the
`compaction` command reads it (provider usage keys, nested usage objects and
Anthropic cache tokens). When a request has no count, the fill is estimated
from the size of that request's own data. The recorded data may not hold the
whole prompt, so an estimated fill can be lower than the real one.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import TextIO

from .cost import _estimate_tokens, _match_model_key
from .event_data import (
    number_field,
    require_object_event_data,
    require_usable_timestamps,
    str_field,
)
from .inert import inert
from .models import EventType, TraceEvent
from .store import TraceStore


# ---------------------------------------------------------------------------
# Context limits (tokens)
# ---------------------------------------------------------------------------

CONTEXT_LIMITS: dict[str, int] = {
    # Each value is the most input tokens one request can carry: the model
    # page's maximum input when it states one no larger than the context
    # window, and the context window otherwise.
    # Anthropic: each Claude model's context window, from
    # https://platform.claude.com/docs/en/about-claude/models/overview
    "claude-fable-5-2":     1_000_000,
    "claude-mythos-5-2":    1_000_000,
    "claude-fable-5-1":     1_000_000,
    "claude-mythos-5-1":    1_000_000,
    "claude-fable-5":       1_000_000,
    "claude-mythos-5":      1_000_000,
    "claude-opus-5-5":      1_000_000,
    "claude-opus-5":        1_000_000,
    "claude-opus-4-8":      1_000_000,
    "claude-opus-4-7":      1_000_000,
    "claude-opus-4-6":      1_000_000,
    "claude-opus-4-5":        200_000,
    "claude-opus-4-1":        200_000,
    "claude-opus-4":          200_000,
    "claude-sonnet-5-5":    1_000_000,
    "claude-sonnet-5":      1_000_000,
    "claude-sonnet-4-6":    1_000_000,
    "claude-sonnet-4-5":      200_000,
    "claude-sonnet-4":        200_000,
    "claude-haiku-5-5":     1_000_000,
    "claude-haiku-4-5":       200_000,
    # Retired Claude 3 models, no longer on that page
    "claude-3-7-sonnet":      200_000,
    "claude-3-5-sonnet":      200_000,
    "claude-3-5-haiku":       200_000,
    "claude-3-opus":          200_000,
    "claude-3-sonnet":        200_000,
    "claude-3-haiku":         200_000,
    # OpenAI: each model's page under
    # https://developers.openai.com/api/docs/models, and for gpt-5-search-api
    # the web search guide,
    # https://developers.openai.com/api/docs/guides/tools-web-search
    "gpt-6-astra":                     922_000,
    "gpt-6.1-sol":                     922_000,
    "gpt-6.1-terra":                   922_000,
    "gpt-6-sol":                       922_000,
    "gpt-6-luna":                      922_000,
    "gpt-5.6-sol":                     922_000,
    "gpt-5.6-terra":                   922_000,
    "gpt-5.6-luna":                    922_000,
    "gpt-5.5":                       1_050_000,
    "gpt-5.5-pro":                   1_050_000,
    "gpt-5.4":                       1_050_000,
    "gpt-5.4-pro":                   1_050_000,
    "gpt-5.4-mini":                    272_000,
    "gpt-5.4-nano":                    272_000,
    "gpt-5.3-codex":                   272_000,
    "gpt-5.2":                         400_000,
    "gpt-5.2-pro":                     400_000,
    "gpt-5.2-codex":                   272_000,
    "gpt-5.2-chat-latest":             128_000,
    "gpt-5.1":                         400_000,
    "gpt-5.1-chat-latest":             128_000,
    "gpt-5":                           272_000,
    "gpt-5-mini":                      272_000,
    "gpt-5-nano":                      272_000,
    "gpt-5-pro":                       400_000,
    "gpt-5-search-api":                200_000,
    "gpt-5-chat-latest":               128_000,
    "gpt-4.1":                       1_047_576,
    "gpt-4.1-mini":                  1_047_576,
    "gpt-4.1-nano":                  1_047_576,
    "gpt-4o":                          128_000,
    "gpt-4o-realtime-preview":          32_000,
    "gpt-4o-transcribe":                16_000,
    "gpt-4o-mini":                     128_000,
    "gpt-4o-mini-realtime-preview":     16_000,
    "gpt-4o-mini-transcribe":           16_000,
    "gpt-4o-mini-tts":                   2_000,
    "gpt-4-turbo":                     128_000,
    "gpt-4-0125-preview":              128_000,
    "gpt-4-1106-vision-preview":       128_000,
    "gpt-4":                             8_192,
    "gpt-3.5-turbo":                    16_385,
    "o1":                              200_000,
    "o1-mini":                         128_000,
    "o1-preview":                      128_000,
    "o3":                              200_000,
    "o3-mini":                         200_000,
    "o3-pro":                          200_000,
    "o4-mini":                         200_000,
    # Amazon Nova through AWS Bedrock: each model card, such as
    # https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-amazon-nova-pro.html
    "amazon.nova-pro-v1":              300_000,
    "amazon.nova-lite-v1":             300_000,
    "amazon.nova-micro-v1":            128_000,
    # Google: the input token limit on each model's page under
    # https://ai.google.dev/gemini-api/docs/models. Gemini 4 pages state only
    # a context window. Gemini 1.5 is from that page as archived on
    # 2025-02-28 by the Wayback Machine.
    "gemini-4-carbon":               1_048_576,
    "gemini-4-argon":                1_048_576,
    "gemini-3.8-flash":              1_048_576,
    "gemini-3.8-flash-tts":              8_192,
    "gemini-3.8-flash-lite-tts":         8_192,
    "gemini-3.6-flash":              1_048_576,
    "gemini-3.5-flash-lite":         1_048_576,
    "gemini-3.1-flash-lite":         1_048_576,
    "gemini-3.1-flash-lite-image":      65_536,
    "gemini-3.1-pro-preview":        1_048_576,
    "gemini-3-flash-preview":        1_048_576,
    "gemini-2.5-pro":                1_048_576,
    "gemini-2.5-pro-preview-tts":        8_192,
    "gemini-2.5-flash":              1_048_576,
    "gemini-2.5-flash-image":           65_536,
    "gemini-2.5-flash-native-audio":   131_072,
    "gemini-2.5-flash-preview-tts":      8_192,
    "gemini-2.5-flash-lite":         1_048_576,
    "gemini-2.0-flash":              1_048_576,
    "gemini-1.5-pro":                2_097_152,
    "gemini-1.5-flash":              1_048_576,
    # Meta
    "llama-3.1-405b":        131_072,
    "llama-3.1-70b":         131_072,
}

DEFAULT_CONTEXT_LIMIT = 200_000  # conservative fallback


def _resolve_limit(model: str) -> int | None:
    """Return the context limit for *model*, or None if it is unknown.

    The name finds its ``CONTEXT_LIMITS`` key by the rule that ``cost`` uses
    to price a recorded model name. Case is ignored, and a provider prefix, a
    Bedrock prefix or a Bedrock ARN is read past. In a ``claude-`` name, a dot
    between two digits reads as a dash. The name uses the longest key that it
    equals, or that it starts with followed by ``-``, ``:`` or ``@``.
    """
    if not model:
        return None
    key = _match_model_key(model, CONTEXT_LIMITS)
    return None if key is None else CONTEXT_LIMITS[key]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RequestBudget:
    request_index: int
    offset_seconds: float
    input_tokens: int        # context tokens of this request
    output_tokens: int
    cumulative_input: int    # input tokens sent so far, summed over requests
    context_limit: int | None
    pct_used: float | None   # this request's context fill; None when limit unknown
    estimated: bool = False  # input_tokens estimated from the request data


@dataclass
class TokenBudgetReport:
    session_id: str
    model: str
    context_limit: int | None
    requests: list[RequestBudget]
    final_cumulative: int     # input tokens sent, summed over all requests
    final_pct: float | None   # context fill of the last request
    warning_threshold: float  # 0.0 to 1.0
    final_context_tokens: int = 0
    peak_context_tokens: int = 0
    peak_pct: float | None = None
    peak_request_index: int | None = None


def _request_context_tokens(data) -> tuple[int, bool]:
    """Return (context tokens, estimated) for one LLM request's data."""
    # compaction imports this module, so its helper is imported here.
    from .compaction import _input_token_count

    explicit = _input_token_count(data)
    if explicit is not None:
        return explicit, False
    import json
    return _estimate_tokens(json.dumps(data)), True


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def token_budget_data_indexes(events: list[TraceEvent]) -> list[int]:
    """Return the positions of the events whose data analyse_token_budget reads.

    It reads every llm_request and every llm_response after the first request.
    """
    indexes = []
    seen_request = False
    for index, event in enumerate(events):
        if event.event_type == EventType.LLM_REQUEST:
            seen_request = True
            indexes.append(index)
        elif event.event_type == EventType.LLM_RESPONSE and seen_request:
            indexes.append(index)
    return indexes


def analyse_token_budget(
    store: TraceStore,
    session_id: str,
    warning_threshold: float = 0.9,
) -> TokenBudgetReport:
    """Build a TokenBudgetReport for *session_id*.

    Raises NonObjectEventDataError when it would read event data that is not
    an object, and UnusableTimestampError when the first event's or an
    llm_request's timestamp is not a Unix time in seconds.
    """
    meta = store.load_meta(session_id)
    events = store.load_events(session_id)
    require_object_event_data(events, indexes=token_budget_data_indexes(events))
    # Each request's offset is its timestamp minus the first event's.
    require_usable_timestamps(
        events,
        indexes={0}
        | {
            index
            for index, event in enumerate(events)
            if event.event_type == EventType.LLM_REQUEST
        },
    )

    base_ts = events[0].timestamp if events else meta.started_at
    model = ""
    context_limit: int | None = None
    cumulative_input = 0
    requests: list[RequestBudget] = []
    req_idx = 0

    for event in events:
        if event.event_type == EventType.LLM_REQUEST:
            req_idx += 1
            offset = event.timestamp - base_ts

            inp_tok, estimated = _request_context_tokens(event.data)

            out_tok = number_field(event.data, "output_tokens", 0)

            # Detect model
            if not model:
                model = str_field(event.data, "model", "")
                if model:
                    context_limit = _resolve_limit(model)

            cumulative_input += inp_tok
            pct = (inp_tok / context_limit) if context_limit else None

            requests.append(RequestBudget(
                request_index=req_idx,
                offset_seconds=offset,
                input_tokens=inp_tok,
                output_tokens=out_tok,
                cumulative_input=cumulative_input,
                context_limit=context_limit,
                pct_used=pct,
                estimated=estimated,
            ))

        elif event.event_type == EventType.LLM_RESPONSE:
            # Update output tokens on the last request if available
            if requests:
                out_tok = number_field(event.data, "output_tokens", 0)
                if out_tok:
                    requests[-1].output_tokens = out_tok

    final_context = requests[-1].input_tokens if requests else 0
    peak = max(requests, key=lambda r: r.input_tokens) if requests else None

    def _fill(tokens: int) -> float | None:
        return (tokens / context_limit) if context_limit and requests else None

    return TokenBudgetReport(
        session_id=session_id,
        model=model or "unknown",
        context_limit=context_limit,
        requests=requests,
        final_cumulative=cumulative_input,
        final_pct=_fill(final_context),
        warning_threshold=warning_threshold,
        final_context_tokens=final_context,
        peak_context_tokens=peak.input_tokens if peak else 0,
        peak_pct=_fill(peak.input_tokens) if peak else None,
        peak_request_index=peak.request_index if peak else None,
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt_pct(pct: float | None) -> str:
    if pct is None:
        return "  n/a"
    return f"{pct*100:5.1f}%"


def _fmt_offset(s: float) -> str:
    m = int(s) // 60
    sec = int(s) % 60
    return f"+{m}:{sec:02d}"


def format_token_budget(report: TokenBudgetReport, out: TextIO = sys.stdout) -> None:
    """Write the per-request context fill table for one session.

    The model goes through inert(), so an escape code or a line break in it
    shows as text.
    """
    w = out.write
    limit_str = f"{report.context_limit:,}" if report.context_limit else "unknown"
    w(f"\nSession: {report.session_id[:12]}  Model: {inert(report.model)}  "
      f"Context limit: {limit_str} tokens\n")
    w("─" * 72 + "\n")
    w(f"  {'Req':>4}  {'Offset':>7}  {'Context tok':>12}  {'Output tok':>10}  "
      f"{'Fill':>7}\n")
    w("─" * 72 + "\n")

    for r in report.requests:
        warn = ""
        if r.pct_used is not None and r.pct_used >= report.warning_threshold:
            warn = "  ← warning"
        context = f"{'~' if r.estimated else ''}{r.input_tokens:,}"
        w(f"  {r.request_index:>4}  {_fmt_offset(r.offset_seconds):>7}  "
          f"{context:>12}  {r.output_tokens:>10,}  "
          f"{_fmt_pct(r.pct_used):>7}{warn}\n")

    w("─" * 72 + "\n")
    w(f"Current: {report.final_context_tokens:,} tokens  "
      f"({_fmt_pct(report.final_pct).strip()} of limit)\n")
    if report.peak_request_index is not None:
        w(f"Peak:    {report.peak_context_tokens:,} tokens  "
          f"({_fmt_pct(report.peak_pct).strip()} of limit) at request "
          f"{report.peak_request_index}\n")
    if report.context_limit and report.requests:
        headroom = max(report.context_limit - report.final_context_tokens, 0)
        w(f"Headroom: {headroom:,} tokens\n")
    if report.requests:
        w(f"Input sent: {report.final_cumulative:,} tokens over "
          f"{len(report.requests)} request(s), summed\n")
    if any(r.estimated for r in report.requests):
        w("~ estimated from the size of the request data, which may not hold "
          "the whole prompt\n")
    w("\n")


# ---------------------------------------------------------------------------
# TokenBudgetWatcher (used by watch.py)
# ---------------------------------------------------------------------------

@dataclass
class TokenBudgetWatcher:
    """Stateful watcher that fires when one request's context fill crosses a threshold.

    It fires once per crossing: a request whose fill is below the threshold
    re-arms it, so it can fire again after compaction.
    """
    threshold: float = 0.9      # 0.0 to 1.0
    model: str = ""
    context_limit: int | None = None
    cumulative_input: int = 0   # input tokens sent, summed over requests
    fired: bool = False
    context_tokens: int = 0     # context tokens of the latest request

    def update(self, event: TraceEvent) -> str | None:
        """Process one event. Returns a violation message or None."""
        if event.event_type != EventType.LLM_REQUEST:
            return None

        # Detect model on first LLM request
        if not self.model:
            self.model = str_field(event.data, "model", "")
            if self.model:
                self.context_limit = _resolve_limit(self.model)

        if self.context_limit is None:
            return None  # unknown model — watcher disabled

        inp_tok, _estimated = _request_context_tokens(event.data)
        self.context_tokens = inp_tok
        self.cumulative_input += inp_tok
        pct = inp_tok / self.context_limit

        if pct < self.threshold:
            self.fired = False
            return None
        if self.fired:
            return None
        self.fired = True
        return (
            f"TokenBudgetWatcher: {inp_tok:,} / "
            f"{self.context_limit:,} tokens "
            f"({pct*100:.1f}%): context window "
            f"{self.threshold*100:.0f}% full"
        )


# ---------------------------------------------------------------------------
# CLI handler
# ---------------------------------------------------------------------------

def cmd_token_budget(args: argparse.Namespace) -> int:
    store = TraceStore(args.trace_dir)

    session_id = getattr(args, "session_id", None)
    if not session_id:
        session_id = store.get_latest_session_id()
    if not session_id:
        sys.stderr.write("No sessions found.\n")
        return 1
    full_id = store.find_session(session_id)
    if not full_id:
        sys.stderr.write(f"Session not found: {session_id}\n")
        return 1

    threshold = getattr(args, "warning_threshold", 0.9) or 0.9
    report = analyse_token_budget(store, full_id, warning_threshold=threshold)
    format_token_budget(report)
    return 0
