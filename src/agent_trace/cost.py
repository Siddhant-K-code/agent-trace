"""Cost estimation for agent sessions.

Estimates token usage from event payload sizes (len(content) / 4) and maps
tokens to dollar cost using configurable per-model pricing. No API calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import TextIO

from .explain import ExplainResult, Phase, explain_session
from .models import EventType, TraceEvent
from .store import TraceStore


# ---------------------------------------------------------------------------
# Pricing table  (dollars per 1M tokens)
# ---------------------------------------------------------------------------

PRICING: dict[str, dict[str, float]] = {
    "sonnet": {"input": 3.00,  "output": 15.00},
    "opus":   {"input": 15.00, "output": 75.00},
    "haiku":  {"input": 0.25,  "output": 1.25},
    "gpt4":   {"input": 30.00, "output": 60.00},
    "gpt4o":  {"input": 5.00,  "output": 15.00},
}
DEFAULT_MODEL = "sonnet"


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


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Estimate token count from character length (4 chars ≈ 1 token)."""
    return max(1, len(text) // 4)


def _event_tokens(event: TraceEvent) -> tuple[int, int]:
    """Return (input_tokens, output_tokens) estimated for one event."""
    if event.event_type in (EventType.USER_PROMPT, EventType.LLM_REQUEST):
        text = json.dumps(event.data)
        return _estimate_tokens(text), 0

    if event.event_type in (EventType.ASSISTANT_RESPONSE, EventType.LLM_RESPONSE):
        text = json.dumps(event.data)
        return 0, _estimate_tokens(text)

    if event.event_type == EventType.TOOL_CALL:
        text = json.dumps(event.data)
        return _estimate_tokens(text), 0

    if event.event_type == EventType.TOOL_RESULT:
        text = json.dumps(event.data)
        return 0, _estimate_tokens(text)

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


# ---------------------------------------------------------------------------
# CLI handler
# ---------------------------------------------------------------------------

def cmd_cost(args: argparse.Namespace) -> int:
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

    model = getattr(args, "model", DEFAULT_MODEL) or DEFAULT_MODEL
    input_price = getattr(args, "input_price", None)
    output_price = getattr(args, "output_price", None)

    result = estimate_cost(
        store, full_id,
        model=model,
        input_price=input_price,
        output_price=output_price,
    )
    format_cost(result)
    return 0
