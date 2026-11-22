# AgentTrace documentation

Use this index to find the shortest path from a question to the relevant
command and its limits. The [command reference](commands.md) remains the
canonical flag reference.

## Share your use case

Want to help improve the docs? Start a
[GitHub Discussion](https://github.com/Siddhant-K-code/agent-trace/discussions)
with your main use and a question you want the docs to answer. Participation
is optional.

| Main use | Start here |
|---|---|
| Debug an agent run | [Session replay](commands.md#replay) and [evidence health](evidence-health.md) |
| Review an agent-produced change | [PR evidence workflow](ai-agent-pr-evidence.md) |
| Evaluate agents | [Evaluation command and limits](commands.md#eval) |
| Integrate or export evidence | [Integrations](integrations.md) and [OTLP export](production.md) |

Still exploring? Say what you hope to try. Use public or synthetic examples.
Do not post private code, transcripts, customer information, or contact
details. Discussions are public; posting requires a GitHub account. Sharing
your use case does not opt you into outreach.

## Setup and releases

- [Setup](setup.md): hook, proxy, and library capture paths.
- [Platform-team install](platform-install.md): machine-wide, metadata-only
  capture through Copilot CLI policy hooks and Claude Code managed settings.
- [Maintainer release runbook](releases.md): independent core and VS Code
  release lines, version parity, publishing, and recovery.
- [Security](security.md): local data, redaction, authentication, and sharing
  boundaries.

## Local review

- [Claude Code session replay](claude-code-session-replay.md)
- [OpenAI Codex trace viewer](codex-trace-viewer.md)
- [GitHub Copilot CLI observability](github-copilot-cli-observability.md)
- [Evidence health](evidence-health.md): what a trace could not prove, and
  [external evidence](evidence-health.md#external-evidence) from Entire,
  Copilot cloud agent commits, and Claude Code OpenTelemetry.
- [Session evidence review](session-review-projection.md): shared dashboard and
  VS Code review schema, filters, relationships, and local actions.
- [VS Code evidence review](vscode.md#evidence-review)
- [PR reviewability check](review.md)
- [Local review workflow](v1-workflow.md)

## Provider guides

- [Capture matrix](capture-matrix.md): provider event coverage, verification
  status, and known gaps.
- [Claude Code session replay](claude-code-session-replay.md)
- [Claude Code plugin](claude-code-plugin.md)
- [OpenAI Codex trace viewer and verified capture](codex-trace-viewer.md)
- [OpenAI Codex plugin](codex-plugin.md)
- [GitHub Copilot CLI observability](github-copilot-cli-observability.md)
- [Copilot CLI plugin](copilot-plugin.md)
- [Copilot cloud agent capture](copilot-cloud-agent.md): partial, unverified
  repository hooks that stream events to a collector.
- [Setup](setup.md): Gemini CLI and Cursor are also covered here.
- [Live capture verification](setup.md#live-capture-verification): local
  recorded-evidence verification for Claude Code and Codex.

## Evidence handoff

- [AI agent pull request evidence](ai-agent-pr-evidence.md)
- [PR reviewability check and Action artifacts](review.md)
- [Share disclosure preview](share-disclosure-preview.md)
- [Cognitive debt schema and badge](cognitive-debt.md)
- [Compliance evidence crosswalk](compliance-report.md)
- [Assignment review](hiring.md)

## Interoperability

- [MCP trace recording and trace-reader server](mcp-trace-recording.md)
- [MCP client setup and published server metadata](mcp.md)
- [Framework integrations](integrations.md)
- [OTLP GenAI and OpenInference export](production.md#openinference-compatibility)
- [Collector server](server.md)
- [Public trace schema and validator](trace-schema.md)
- [Agent Evidence Corpus](agent-evidence-corpus.md)

## Experimental analysis

- [AGENTS.md and CLAUDE.md instruction compliance](agent-instruction-compliance.md)
- [Instruction compliance benchmark](instruction-compliance.md)
- [Cognitive debt and review coverage](cognitive-debt-review-coverage.md)
- [Skill adoption](skills.md)
- [Skill trigger evaluation](skills-eval.md)
- [Organization reporting](org-report.md)

Experimental reports are review aids. They do not prove causation, task
success, policy compliance, or complete provider capture.
