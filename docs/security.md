# Security

agent-strace provides two complementary mechanisms for keeping sensitive data out of traces: secret redaction (at capture time) and PII anonymization (at export time). A policy file lets you audit and restrict what the agent is allowed to do.

---

## Secret redaction

Strips API keys, tokens, and credentials before they hit disk. The original data is never stored.

```bash
# Enable redaction when capturing
agent-strace record --redact -- npx -y @modelcontextprotocol/server-filesystem /tmp
agent-strace record-http https://mcp.example.com --redact

# Or via setup (Claude Code hooks)
agent-strace setup --redact
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
| Key-named values | Any value under keys: `password`, `secret`, `token`, `api_key`, `authorization` |
| EKM shared secrets | 64-char hex strings |
| Private keys | PEM blocks |

Redacted values become `[REDACTED]`. See [ADR-0007](../ADRs/0007-heuristic-redaction.md) for design rationale.

### Custom patterns

```bash
agent-strace setup --redact --redact-pattern "ATTESTATION_KEY=[A-Fa-f0-9]{64}"
```

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

## Shadow AI detection

Detect undeclared agent activity and Shadow MCP servers in any repo. No network calls, no API keys.

```bash
agent-strace audit-tools
agent-strace audit-tools --repo . --since "90 days ago" --approved cursor,copilot
```

Detected tools: Claude Code, Cursor, GitHub Copilot, Codex/ChatGPT, Windsurf, Aider, Gemini CLI. Identified via file signals (`.cursorrules`, `CLAUDE.md`, `.github/copilot-instructions.md`, etc.) and commit message patterns.

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
agent-strace setup --redact
```

Combine with [agentic-authz](https://github.com/Siddhant-K-code/agentic-authz) to block agents from security-critical components entirely, and use agent-strace to audit everything they do access.
