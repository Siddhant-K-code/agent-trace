# #241 owner checklist: what to check in Telemetry.dev

Use this after `send.py` reports `HTTP 200` for each payload. Write what you **see** in the Observed column. The Predicted column comes from [MAPPING.md](MAPPING.md) and is doc-based, not observed. #241 says a 2xx alone does not show compatibility, so every row matters.

## Run record

| | Fill in |
| --- | --- |
| Date and time (UTC) | `sent_at` in `sent/<run id>/manifest.json` |
| agent-strace | `agent_strace_version` in the manifest (0.120.0 when this was prepared) |
| Telemetry.dev | No version is exposed. Record the date, the plan, and the `GET https://api.telemetry.dev/health` reply. |
| Run ID | |
| Environment the key sends to | |
| Prompt capture for that environment | on / off (on is the default) |
| HTTP status per payload | otlp: ___ genai: ___ genai-tree: ___ |
| Trace IDs | From the send output. Paste each one in the trace list search, or open it through the Date range. |

Do not paste the key anywhere. `sent/<run id>/` holds the exact bytes sent and their sha256, and no key, so you can compare sent and received without secrets.

## Before you send

- Use a key with **Send telemetry** access. A key limited to one environment sends there. Use `--environment NAME` only for a key that is not limited, or the send fails with `403 environment_mismatch`.
- If **Issue checks** is on, Telemetry.dev sends stored content to its AI sub-processor. The payloads are synthetic, so this is harmless here.
- Run it once. A second run gets new IDs and makes a second copy of every trace.

## Checks by acceptance criterion

### 1. A real export reaches a Telemetry.dev project

| Check | Predicted | Observed |
| --- | --- | --- |
| **Traces** list, Date range Last 7 days: new traces | 5 (2 otlp, 2 genai, 1 tree) | |
| Turn on the **Environment** column (**Columns**; it is hidden by default) | The key's environment, else `default` | |
| The trace names | `agent-session (claude-code)` for otlp, `invoke_agent claude-code` for genai and tree | |

### 2. Versions and date recorded

Fill in the run record above. **Source** in a trace's **Metadata** tab shows **Service** `agent-trace`. The `service.version` value 0.120.0 is in the resource attributes.

### 3. Sent and received can be compared without secrets

| Check | Predicted | Observed |
| --- | --- | --- |
| Open a GenAI `chat` span, **Attributes** | Every key in MAPPING.md's `chat` table is listed with the value from `sent/<run id>/payload-otlp-genai.json` | |
| Any key missing from **Attributes** | none (`validate.py` predicts no secret-key drops) | |
| Any value shown as redacted by Telemetry.dev | none. AgentTrace's own `[REDACTED…]` markers show as sent. | |

### 4. Dropped, renamed or flattened attributes

Check these against MAPPING.md and note every difference.

| Check | Predicted | Observed |
| --- | --- | --- |
| `chat` span **Model call** section | Requested model `claude-sonnet-4-5`, Provider `anthropic`, Finish reason | |
| `execute_tool` span **Tool call** section | Tool name `bash`, `Task` or `read_file` | |
| Span list **Tool / Agent** column | `claude-code` on both `invoke_agent` roots | |
| `gen_ai.tool.input.*`, `gen_ai.tool.output`, `input.value`, `output.value` | Kept in **Attributes**, not in Arguments or Result | |
| `agent.strace.provenance`, `agent.strace.evidence.gaps` | One opaque JSON string each, not expanded into separate keys | |
| Integer attributes such as `gen_ai.usage.input_tokens` | Shown as numbers | |
| The trace's **Metadata** under **Source** | Empty (the `metadata` attribute is not read as metadata) | |

### 5. Hierarchy, errors, tokens, cost, redaction, provenance and gaps

**Hierarchy**

| Check | Predicted | Observed |
| --- | --- | --- |
| genai-tree trace, **Trace** tab, **Expand all** | One tree: `invoke_agent claude-code` > `execute_tool Task` > child `invoke_agent claude-code` > `chat` and `execute_tool read_file` | |
| genai parent and child | Two separate traces, not linked | |
| genai child root **Attributes** | `agent.strace.parent_session_id`, `agent.strace.parent_event_id`, `agent.strace.depth` = 1 | |
| otlp child | A separate trace, linked only by `agent.depth` = 1 | |
| genai-tree **Session** | Unknown: `synthetic-conversation-td241-tree` or `td241-tree-child` | |

**Errors and status**

| Check | Predicted | Observed |
| --- | --- | --- |
| **Spans** column, genai parent and tree | 3 errors: the root, `chat` with `overloaded_error`, `execute_tool bash` | |
| **Spans** column, otlp parent | 2 errors: the root and `bash` | |
| Children | 0 errors | |
| Failed `execute_tool bash`, **Error** section | Error type `_OTHER` (from `error.type`). The details add exception type `Error` and the message `example test failed`. | |
| Failed `chat`, **Error** section | Error type `overloaded_error`, status message `synthetic overload, retrying` | |
| **Logs** (level filter starts at `error`) | 1 error record for each parent and tree trace, from the `exception` event, linked to its trace | |
| Status filter `error` | The parent roots are listed even though both retries passed (an AgentTrace design choice) | |
| Retry link | None. The retry is only a second span with the same name. | |

**Tokens and cost**

| Check | Predicted | Observed |
| --- | --- | --- |
| genai parent **Tokens** | 12,000 in / 1,500 out; hover: cache read 8,000 | |
| genai child **Tokens** | 2,400 in / 300 out | |
| genai-tree **Tokens** | 14,400 in / 1,800 out | |
| genai parent **Cost** | $0.03765 | |
| **genai child Cost (the key test)** | **$0.0117**, computed by Telemetry.dev. If it shows $0.01234, Telemetry.dev reads AgentTrace's cost after all. | |
| genai-tree **Cost** | $0.04935 | |
| Failed `chat` span **Cost** | none | |
| otlp traces **Tokens** and **Cost** | none. Model calls exist only as events on the root. | |
| Model **Models** column | `claude-sonnet-4-5` for genai, empty for otlp | |

**Content and redaction**

| Check | Predicted | Observed |
| --- | --- | --- |
| Trace list **Input** column | Empty for every trace | |
| Trace list **Output** column | Empty, or a synthetic reply from the finish reason | |
| `execute_tool` **Arguments** and **Result** | Empty | |
| Root **Events**: `gen_ai.user.message` | `gen_ai.prompt` starts with `[REDACTED synthetic request]` | |
| Child `execute_tool read_file` **Attributes** | `gen_ai.tool.input.auth_header` = `[REDACTED:bearer]`, kept | |
| A field or badge showing AgentTrace redaction | None. `agent.strace.redacted` and the `*.redacted` keys are plain attributes. | |

**Provenance and evidence health**

| Check | Predicted | Observed |
| --- | --- | --- |
| Parent root **Attributes** | `agent.strace.evidence.status` = `partial`, `agent.strace.evidence.gap_count` = 1 | |
| Child root | `agent.strace.evidence.status` = `healthy` | |
| `agent.strace.provenance` | Present, an opaque JSON string, with `synthetic-user`, `synthetic-host` and `[REDACTED:hex-token]` | |
| Searchable or filterable evidence | No | |

**The capture gap**

| Check | Predicted | Observed |
| --- | --- | --- |
| genai parent: the orphan `execute_tool bash` span | Has `gen_ai.tool.output` but no `gen_ai.tool.input.*`, and `agent.strace.response.parent_event_id` = `missing-tool-call` | |
| Anything in Telemetry.dev marking it as a gap | No. Only the root's `agent.strace.evidence.gap_codes` names `orphan_tool_result`. | |

### 6. Decision

| | |
| --- | --- |
| Predicted | **PARTIAL**: accepted, with model, provider, tools, tokens, errors and status mapped; cost, content, cross-session hierarchy, retry and evidence lost or opaque |
| Observed | PASS / PARTIAL / FAIL |
| Rows where Observed differs from Predicted | |
| Bounded limits to document if PARTIAL | |

### 7. Documentation and outreach

Do this only after a PASS or a clearly bounded PARTIAL in section 6. Only then add a backend guide to `docs/`, list the limits prominently, and propose the Telemetry.dev directory listing. This branch adds neither.
