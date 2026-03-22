# ADR-0008: Token and Cost Estimation via Character-Count Heuristic

**Status:** Accepted  
**Date:** 2025-03  
**Deciders:** Siddhant Khare

## Context

Developers want to know how much an agent session cost and where tokens were spent. Accurate token counting requires a tokenizer library (tiktoken, tokenizers) which conflicts with ADR-0003 (zero runtime dependencies). An approximation is sufficient for the use case: identifying expensive phases and wasted spend on retries.

## Decision

Token estimation uses `max(1, len(text) // 4)` — the "4 characters per token" heuristic. This is a well-known approximation for English text with GPT/Claude tokenizers. The `max(1, ...)` ensures at least 1 token is counted for any non-empty payload.

Token counts are estimated from `json.dumps(event.data)` — the serialized event payload — which includes JSON key overhead. This slightly overestimates tokens but is consistent across all event types.

**Input vs output classification by event type:**

| Event type | Token class |
|---|---|
| `user_prompt`, `llm_request`, `tool_call` | Input |
| `assistant_response`, `llm_response`, `tool_result` | Output |
| All others | Not counted |

**Pricing** is a hardcoded table (dollars per 1M tokens) for 5 models: `sonnet`, `opus`, `haiku`, `gpt4`, `gpt4o`. Custom pricing is supported via `--input-price` and `--output-price` CLI flags.

Cost is broken down by phase (from `explain`) and wasted cost is calculated as the sum of cost from failed phases.

## Consequences

- **Zero new dependencies** — consistent with ADR-0003.
- **Accuracy is ±30–50%** compared to actual token counts. Sufficient for identifying expensive phases; not suitable for billing reconciliation.
- **JSON overhead is included** in the estimate. A `tool_call` event with a short command will still count the JSON key names (`tool_name`, `arguments`, etc.) as tokens.
- **Pricing table requires manual updates** when model prices change. There is no automatic price fetching.
- **Custom pricing mutates the global `PRICING` dict** — repeated calls with different custom prices within the same process overwrite each other. This is a known limitation acceptable for a CLI tool where each invocation is a separate process.
- **Actual token counts from JSONL import** (present in Claude Code's `usage` field) are not used by the cost estimator — it always applies the heuristic. Using real counts where available is a future improvement.
