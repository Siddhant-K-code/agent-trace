"""Token budget tracking: context window usage and early warning.

Analyses LLM request events to track cumulative input token usage against
the model's context window limit. Provides:

  - `token-budget` command: per-request accumulation table
  - `TokenBudgetWatcher`: fires when context usage crosses a threshold
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import TextIO

from .cost import _estimate_tokens
from .models import EventType, TraceEvent
from .store import TraceStore


# ---------------------------------------------------------------------------
# Context window limits (tokens)
# ---------------------------------------------------------------------------

CONTEXT_LIMITS: dict[str, int] = {
    # Anthropic
    "claude-opus-4":          200_000,
    "claude-sonnet-4-5":      200_000,
    "claude-sonnet-4":        200_000,
    "claude-haiku-3-5":       200_000,
    "claude-haiku-3":         200_000,
    "claude-3-opus":          200_000,
    "claude-3-sonnet":        200_000,
    "claude-3-haiku":         200_000,
    # OpenAI
    "gpt-4o":                 128_000,
    "gpt-4o-mini":            128_000,
    "gpt-4-turbo":            128_000,
    "gpt-4":                   8_192,
    "gpt-3.5-turbo":          16_385,
    "o1":                     200_000,
    "o1-mini":                128_000,
    "o3":                     200_000,
    # Google
    "gemini-1.5-pro":       1_048_576,
    "gemini-1.5-flash":     1_048_576,
    "gemini-2.0-flash":     1_048_576,
    # Meta
    "llama-3.1-405b":        131_072,
    "llama-3.1-70b":         131_072,
}

DEFAULT_CONTEXT_LIMIT = 200_000  # conservative fallback


def _resolve_limit(model: str) -> int | None:
    """Return context limit for *model*, or None if unknown."""
    if not model:
        return None
    m = model.lower().strip()
    # Exact match
    if m in CONTEXT_LIMITS:
        return CONTEXT_LIMITS[m]
    # Prefix match (e.g. "claude-sonnet-4-5-20251022" → "claude-sonnet-4-5")
    for key, limit in CONTEXT_LIMITS.items():
        if m.startswith(key) or key.startswith(m):
            return limit
    return None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RequestBudget:
    request_index: int
    offset_seconds: float
    input_tokens: int
    output_tokens: int
    cumulative_input: int
    context_limit: int | None
    pct_used: float | None   # None when limit unknown


@dataclass
class TokenBudgetReport:
    session_id: str
    model: str
    context_limit: int | None
    requests: list[RequestBudget]
    final_cumulative: int
    final_pct: float | None
    warning_threshold: float  # 0.0–1.0


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse_token_budget(
    store: TraceStore,
    session_id: str,
    warning_threshold: float = 0.9,
) -> TokenBudgetReport:
    """Build a TokenBudgetReport for *session_id*."""
    meta = store.load_meta(session_id)
    events = store.load_events(session_id)

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

            # Try to get explicit token count from event data
            inp_tok = event.data.get("input_tokens", 0)
            if not inp_tok:
                # Fall back to estimation from payload size
                import json
                inp_tok = _estimate_tokens(json.dumps(event.data))

            out_tok = event.data.get("output_tokens", 0)

            # Detect model
            if not model:
                model = event.data.get("model", "")
                if model:
                    context_limit = _resolve_limit(model)

            cumulative_input += inp_tok
            pct = (cumulative_input / context_limit) if context_limit else None

            requests.append(RequestBudget(
                request_index=req_idx,
                offset_seconds=offset,
                input_tokens=inp_tok,
                output_tokens=out_tok,
                cumulative_input=cumulative_input,
                context_limit=context_limit,
                pct_used=pct,
            ))

        elif event.event_type == EventType.LLM_RESPONSE:
            # Update output tokens on the last request if available
            if requests:
                out_tok = event.data.get("output_tokens", 0)
                if out_tok:
                    requests[-1].output_tokens = out_tok

    final_pct = (cumulative_input / context_limit) if context_limit else None

    return TokenBudgetReport(
        session_id=session_id,
        model=model or "unknown",
        context_limit=context_limit,
        requests=requests,
        final_cumulative=cumulative_input,
        final_pct=final_pct,
        warning_threshold=warning_threshold,
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
    w = out.write
    limit_str = f"{report.context_limit:,}" if report.context_limit else "unknown"
    w(f"\nSession: {report.session_id[:12]}  Model: {report.model}  "
      f"Context limit: {limit_str} tokens\n")
    w("─" * 72 + "\n")
    w(f"  {'Req':>4}  {'Offset':>7}  {'Input tok':>10}  {'Output tok':>10}  "
      f"{'Cumulative':>11}  {'% limit':>7}\n")
    w("─" * 72 + "\n")

    for r in report.requests:
        warn = ""
        if r.pct_used is not None and r.pct_used >= report.warning_threshold:
            warn = "  ← warning"
        w(f"  {r.request_index:>4}  {_fmt_offset(r.offset_seconds):>7}  "
          f"{r.input_tokens:>10,}  {r.output_tokens:>10,}  "
          f"{r.cumulative_input:>11,}  {_fmt_pct(r.pct_used):>7}{warn}\n")

    w("─" * 72 + "\n")
    final_pct_str = _fmt_pct(report.final_pct)
    w(f"Current: {report.final_cumulative:,} tokens  ({final_pct_str} of limit)\n")

    if report.context_limit and report.final_cumulative > 0:
        remaining = report.context_limit - report.final_cumulative
        avg_per_req = report.final_cumulative / max(len(report.requests), 1)
        est_remaining = int(remaining / avg_per_req) if avg_per_req > 0 else 0
        w(f"Est. remaining: ~{est_remaining} request(s)\n")
    w("\n")


# ---------------------------------------------------------------------------
# TokenBudgetWatcher (used by watch.py)
# ---------------------------------------------------------------------------

@dataclass
class TokenBudgetWatcher:
    """Stateful watcher that fires when context usage crosses a threshold."""
    threshold: float = 0.9      # 0.0–1.0
    model: str = ""
    context_limit: int | None = None
    cumulative_input: int = 0
    fired: bool = False

    def update(self, event: TraceEvent) -> str | None:
        """Process one event. Returns a violation message or None."""
        if event.event_type != EventType.LLM_REQUEST:
            return None

        # Detect model on first LLM request
        if not self.model:
            self.model = event.data.get("model", "")
            if self.model:
                self.context_limit = _resolve_limit(self.model)

        if self.context_limit is None:
            return None  # unknown model — watcher disabled

        inp_tok = event.data.get("input_tokens", 0)
        if not inp_tok:
            import json
            inp_tok = _estimate_tokens(json.dumps(event.data))

        self.cumulative_input += inp_tok
        pct = self.cumulative_input / self.context_limit

        if pct >= self.threshold and not self.fired:
            self.fired = True
            return (
                f"TokenBudgetWatcher: {self.cumulative_input:,} / "
                f"{self.context_limit:,} tokens "
                f"({pct*100:.1f}%) — context window "
                f"{self.threshold*100:.0f}% full"
            )
        return None


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
