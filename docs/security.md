# Security

agent-strace provides complementary mechanisms for keeping sensitive data out
of traces and derived reports: secret redaction at capture time, PII
anonymization at export time, and privacy-minimized report formats. A policy
file lets you audit and restrict what the agent is allowed to do.

Anonymous product telemetry is a separate, default-on data path. CLI telemetry
can be disabled with `agent-strace telemetry disable`; extension telemetry can
be disabled with the `agentTrace.telemetry.enabled` setting or the Command
Palette. `DO_NOT_TRACK=1`, `AGENT_STRACE_TELEMETRY=0`, and the editor-wide
telemetry switch also disable extension telemetry. Both clients use fixed
schemas that exclude prompts, responses, command arguments, paths, repository
data, session IDs, endpoints, and exception messages. The extension sends only
session success, duration, and aggregate tool/error counts—not trace contents.
See [Product telemetry](telemetry.md) for the complete schemas and controls.

---

## Secret redaction

Strips API keys, tokens, and credentials before they hit disk. The original data is never stored. Redaction is enabled by default for all trace writes.

```bash
# Default: redact before writing traces
agent-strace record -- npx -y @modelcontextprotocol/server-filesystem /tmp
agent-strace record-http https://mcp.example.com

# Trusted local traces only
agent-strace record --no-redact -- npx -y @modelcontextprotocol/server-filesystem /tmp
agent-strace setup --no-redact
```

Detected patterns:

| Secret type | Pattern |
|---|---|
| OpenAI API keys | `sk-*` |
| GitHub tokens | `ghp_*`, `github_pat_*` |
| AWS credentials | `AKIA*`, `aws_secret_access_key` |
| Anthropic API keys | `sk-ant-*` |
| Slack tokens | `xox*` |
| JWTs | Three base64 segments separated by `.` |
| Bearer tokens | `Bearer [A-Za-z0-9+/=]{20,}` |
| Connection strings | `postgres://`, `mysql://`, `mongodb://` |
| Basic-auth URLs | `https://user:pass@example.com` |
| Key-named values | Any value under keys: `password`, `secret`, `token`, `api_key`, `authorization` |
| EKM shared secrets | 64-char hex strings |
| Private keys | PEM blocks |

Redacted values become typed markers such as `[REDACTED:openai-key]`, `[REDACTED:bearer-token]`, or `[REDACTED:sensitive]`. Events with redaction are marked with `"redacted": true`. See [ADR-0007](../ADRs/0007-heuristic-redaction.md) for design rationale.

---

## PII masking

Masks personally identifiable information before it hits disk. Separate from secret redaction — use both for maximum coverage.

```bash
agent-strace record --mask -- npx -y @modelcontextprotocol/server-filesystem /tmp
agent-strace record-http https://mcp.example.com --mask
```

Masked by default: email addresses, phone numbers, credit card numbers, US Social Security Numbers, AWS ARNs.

Call `mask_event_data()` directly to sanitise events from an existing session before sharing or exporting:

```python
from agent_trace.masking import mask_event_data
sanitised = mask_event_data(event)
```

---

## Trace anonymization

Strip identifying information from traces at export time. Original session data is never modified.

```bash
# Preview what would be anonymized
agent-strace export SESSION_ID --anonymize --dry-run

# Export with anonymization applied
agent-strace export SESSION_ID --anonymize --output trace-anon.json
```

Anonymized by default:
- Home directory paths → `~/relative/path`
- Hostnames → `<hostname>`
- OS usernames → `<user>`
- Email addresses → `<email>`

Add custom patterns via `.agent-strace/anonymize.yaml`:

```yaml
rules:
  - pattern: "ACME Corp"
    replacement: "<company>"
  - pattern: "192\\.168\\.\\d+\\.\\d+"
    replacement: "<internal-ip>"
```

### Organization-report anonymization

`agent-strace org-report --anonymize` replaces complete persisted identity
labels (including `user@hostname` composites) and team/workspace labels with
deterministic aliases such as `Identity-001` and `Team-001`. Aliases are sorted
and local to that report; they are not unsalted hashes and never use the
collector authentication key. Aggregate output has no standalone session ID,
path, branch, hostname, tenant, or raw-event fields. The default,
non-anonymized report may show team labels and an attributed identity such as
`user@hostname` when it appears in an anomaly callout; use `--anonymize` before
sharing a report outside the organization.

For remote collector input, one authenticated collector instance and key is the
organization boundary. Team, workspace, and tenant metadata are reporting
labels, not authorization controls. Use separate authenticated collectors for
organizations that must be isolated. Static HTML escapes all labels and has no
external assets.

### Privacy-minimized reports and bundles

Use the purpose-built derived formats when raw trace content is not required:

- assignment submissions include allowlisted, anonymous process telemetry; see
  [Assignment submissions and review](hiring.md#responsible-interpretation)
- compliance crosswalks omit prompts, commands, URLs, and raw identifiers; see
  [Compliance evidence crosswalk](compliance-report.md#privacy-and-integrity)
- organization reports support `--anonymize` for external sharing; see
  [Organization reporting](org-report.md#output-and-privacy)

These formats have narrower disclosure boundaries than ordinary
`agent-strace share` output, which can include replay data such as prompts,
tool results, commands, and paths.

---

## Policy files

Audit and restrict what the agent is allowed to do. Exits 1 on violations — usable in CI.

```bash
agent-strace audit                          # latest session, no policy required
agent-strace audit abc123 --policy .agent-scope.json

# CI gate
agent-strace audit --policy .agent-scope.json || exit 1
```

**Policy file** (`.agent-scope.json`):

```json
{
  "files": {
    "read":  { "allow": ["src/**", "tests/**"], "deny": [".env"] },
    "write": { "allow": ["src/**"], "deny": [".github/**"] }
  },
  "commands": {
    "allow": ["pytest", "uv run", "cat"],
    "deny":  ["curl", "wget", "rm -rf"]
  },
  "network": { "deny_all": true, "allow": ["localhost"] }
}
```

Glob patterns support `**` as a recursive wildcard. File read policy applies to `Read`, `View`, `Grep`, and `Glob` tool calls. Network policy checks URLs embedded in `Bash` commands.

### Auto-generate a policy from traces

```bash
# Dry-run: print the suggested policy
agent-strace policy

# Write it to disk
agent-strace policy --output .agent-scope.json

# Observe a specific set of sessions
agent-strace policy --last 20 --output .agent-scope.json
```

---

## Role-based access control

Manage org-level and workspace-scoped permissions for users and groups.

```bash
# Assign an org-level role
agent-strace rbac assign --user alice@example.com --role admin --by security@example.com

# Scope to a specific workspace
agent-strace rbac assign --user bob@example.com --role workspace:viewer --workspace prod

# Assign a role to a group
agent-strace rbac assign --group engineering --role member

# Check access
agent-strace rbac check --user alice@example.com --action export_compliance
agent-strace rbac check --user bob@example.com --action read_sessions --workspace prod

# List all assignments
agent-strace rbac list

# Revoke the workspace assignment
agent-strace rbac revoke --user bob@example.com --workspace prod
```

| Role | Scope and permissions |
|---|---|
| `owner` | Full org access, including RBAC, billing, and SSO management |
| `admin` | Read and run agents; annotate; manage policies, identities, and workspaces; export compliance data |
| `member` | Read sessions, run agents, and annotate |
| `viewer` | Read sessions only |
| `machine` | Programmatic identity with read-session access |
| `workspace:admin` | `admin` permissions in the named workspace |
| `workspace:member` | `member` permissions in the named workspace |
| `workspace:viewer` | `viewer` permissions in the named workspace |

Workspace roles require `--workspace ID`; org roles must omit it. Assignments are
stored locally in `.agent-traces/rbac.json` by default.

---

## Workspace isolation

Workspaces provide separate session stores. Use them to isolate production traces from development, or to separate sessions by team.

```bash
# Create a workspace
agent-strace workspace new prod

# Switch to it (sets AGENT_STRACE_STORAGE)
eval $(agent-strace workspace use prod)

# List workspaces
agent-strace workspace list

# Delete a workspace and all its sessions
agent-strace workspace rm staging
```

Each workspace is a separate directory under the base storage path. RBAC assignments can be scoped per workspace.

---

## Shadow AI detection

Detect undeclared agent activity and Shadow MCP servers in any repo. No network calls, no API keys.

```bash
agent-strace audit-tools
agent-strace audit-tools --repo . --since "90 days ago" --approved cursor,copilot
```

Detected tools: Claude Code, Cursor, GitHub Copilot, Codex/ChatGPT, Windsurf, Aider, Gemini CLI. Identified via file signals (`.cursorrules`, `CLAUDE.md`, `.github/copilot-instructions.md`, etc.) and commit message patterns.

---

## Runtime MCP poisoning scan

`agent-strace mcp-scan` scans the session store for runtime MCP poisoning indicators. It works on the tool descriptions and tool calls the agent actually saw during the session.

```bash
agent-strace mcp-scan
agent-strace mcp-scan --session abc123
agent-strace mcp-scan --watch
agent-strace watch --rules mcp-poisoning,budget:$5,timeout:30m
```

The scanner checks:

| Check | What it catches |
|---|---|
| Description pattern match | Tool descriptions containing instruction override text such as `SYSTEM:`, `ignore previous instructions`, or `<HIDDEN>` |
| Description drift | A tool name whose runtime description changed since an earlier session |
| Behavioural sequence | Credential reads followed by external HTTP, environment dumps followed by external HTTP, mass reads followed by compression, and writes outside the project root |

Add custom regexes to `~/.agent-strace/mcp-patterns.txt`, one pattern per line. The scan is local and deterministic; it does not call a remote reputation service.

---

## Recommended setup for sensitive repos

Commit `.claude/settings.json` to the repo root so every developer gets the same instrumentation:

```json
{
  "hooks": {
    "PreToolUse": [{
      "matcher": ".*",
      "hooks": [{ "type": "command", "command": "agent-strace hook pre-tool" }]
    }],
    "PostToolUse": [{
      "matcher": ".*",
      "hooks": [{ "type": "command", "command": "agent-strace hook post-tool" }]
    }]
  }
}
```

Or use the setup command:

```bash
cd your-sensitive-repo
agent-strace setup
```

Combine with [agentic-authz](https://github.com/Siddhant-K-code/agentic-authz) to block agents from security-critical components entirely, and use agent-strace to audit everything they do access.
