# Copilot cloud agent capture

`agent-strace setup --cli copilot-cloud` writes repository hooks that record the hook-visible events of a [Copilot cloud agent](https://docs.github.com/en/copilot/concepts/security-governance-and-network-settings/about-cloud-sandboxes) job and stream them to an [`agent-strace server`](server.md) collector before the sandbox is discarded.

Coverage is **partial** and **not verified in a live cloud agent job**. The template follows the GitHub documentation cited below, and the tests run the generated script locally against Copilot hook payloads and an authenticated collector. See the [capture matrix](capture-matrix.md#github-copilot-cloud-agent) for the exact captured and missing signals.

## Why a separate setup

Local Copilot CLI hooks and plugins do not run in cloud agent. In a cloud agent job, the only hook configuration that exists is `.github/hooks/*.json` in the cloned repository; the sandbox has no user-level hook files, `settings.json`, or installed plugins. The sandbox filesystem is ephemeral, so files written by hooks are discarded when the job ends. ([Hooks reference: cloud agent execution environment](https://docs.github.com/en/copilot/reference/hooks-reference#cloud-agent-execution-environment))

This is the transcript-access gap reported in [github/gh-aw discussion #74442](https://github.com/github/gh-aw/discussions/74442): coding agent transcripts could not be retrieved after the job.

## Setup

Run from anywhere inside the repository:

```bash
agent-strace setup --cli copilot-cloud
```

It writes two files at the repository root and prints the hooks JSON:

| File | Purpose |
|---|---|
| `.github/hooks/agent-strace.json` | Registers 9 hook events as `bash` command hooks |
| `.github/hooks/agent-strace-cloud.sh` | Runs `agent-strace hook --provider copilot EVENT` and always exits 0 |

`--cli all` does not include `copilot-cloud`, and `--verify` does not support it. `--no-redact` adds `AGENT_TRACE_NO_REDACT=1` to each hook's `env`.

Then:

1. Commit both files and merge them to the default branch. Cloud agent uses a hooks file only when it is on the default branch. ([Customize agent workflows with hooks](https://docs.github.com/en/copilot/how-tos/copilot-on-github/customize-copilot/customize-cloud-agent/use-hooks))
2. Run a collector the agent can reach over HTTPS, with a key:
   ```bash
   agent-strace server keygen
   AGENT_STRACE_AUTH_KEY=ast_<key> agent-strace server --host 0.0.0.0 --port 4317
   ```
   Put it behind TLS. See [server.md](server.md).
3. In the repository settings, add two **Agents** secrets: `AGENT_STRACE_ENDPOINT` (the collector URL) and `AGENT_STRACE_AUTH_KEY` (the key). In a cloud sandbox, Agents secrets are exposed as environment variables to the commands and tools Copilot runs; Agents variables are not available there, so use secrets. ([Configure secrets and variables](https://docs.github.com/en/copilot/how-tos/copilot-on-github/customize-copilot/customize-cloud-agent/configure-secrets-and-variables))
4. Add the collector host to the cloud agent firewall allowlist. By default only GitHub and Copilot hosts, plus the recommended allowlist, are reachable. ([Customizing or disabling the firewall](https://docs.github.com/en/copilot/how-tos/copilot-on-github/customize-copilot/customize-cloud-agent/customize-the-agent-firewall))
5. Assign a task to Copilot, then list the job's session on the collector host with `agent-strace list` and `agent-strace replay`.

## What the hook script does

Cloud agent runs hooks on Linux, honors only the `bash` field (or the `command` fallback), and starts in `/workspace` when a repository is cloned. ([Hooks reference](https://docs.github.com/en/copilot/reference/hooks-reference#cloud-agent-execution-environment)) Each entry runs `bash .github/hooks/agent-strace-cloud.sh EVENT` with `cwd` set to the repository root. The script:

- Sets `AGENT_TRACE_DIR` to `/tmp/agent-strace-traces` unless it is already set, so local state is never written inside the repository.
- Uses `agent-strace` from `PATH`, else the copy in `/tmp/agent-strace-pkg`. If neither exists, it runs `python3 -m pip install --target /tmp/agent-strace-pkg agent-strace==VERSION` once per job, pinned to the version that generated the script. The `pypi.org` and `files.pythonhosted.org` hosts are on the recommended allowlist. ([Copilot allowlist reference](https://docs.github.com/en/copilot/reference/copilot-allowlist-reference)) Set the `AGENT_STRACE_INSTALL` secret to `0` to disable the install.
- Warns on stderr at session start when `AGENT_STRACE_ENDPOINT` is not set.
- Always exits 0. A `preToolUse` command hook that exits non-zero denies the tool call, so a capture problem must never fail the hook. ([Hooks reference: exit codes](https://docs.github.com/en/copilot/reference/hooks-reference#exit-codes-for-command-hooks))

With `AGENT_STRACE_ENDPOINT` set, each event is redacted (unless `--no-redact` was used) and sent to the collector when it is recorded, with a 5 second timeout. A failed send is dropped; nothing is retried or buffered.

`sessionStart` and `userPromptSubmitted` get a 120 second timeout because either can be the first hook of a job and may run the install. The others get 15 seconds, and `sessionEnd` 30. A timed-out hook is skipped and the agent continues.

## Sessions on GitHub Actions

Sessions run in a cloud sandbox by default or, when chosen at the start of a session, in a GitHub Actions job. Setup steps in `copilot-setup-steps.yml` apply only to sessions on Actions. ([About cloud sandboxes](https://docs.github.com/en/copilot/concepts/security-governance-and-network-settings/about-cloud-sandboxes)) For those sessions, `setup --cli copilot-cloud` prints a workflow that preinstalls the same pinned version into `/tmp/agent-strace-pkg` before the agent starts. It does not write the file. The job must be named `copilot-setup-steps` and the file must be on the default branch. ([Configure the development environment](https://docs.github.com/en/copilot/how-tos/copilot-on-github/customize-copilot/customize-cloud-agent/customize-the-agent-environment)) If the repository already has this file, add the install step to its existing job.

## Event coverage

The cloud agent column of the [hook events table](https://docs.github.com/en/copilot/reference/hooks-reference#hook-events) decides what can be captured:

| Copilot hook | In cloud agent | Registered | Recorded as |
|---|---|---|---|
| `sessionStart` | once per job, as a new session | yes | `session_start` |
| `userPromptSubmitted` | at most once, for the job prompt | yes | `user_prompt` |
| `preToolUse` | fires; `ask` is treated as deny | yes | `tool_call` |
| `postToolUse` | fires | yes | `tool_result` |
| `postToolUseFailure` | fires | yes | `error` |
| `agentStop` | fires | yes | stop marker, no response text |
| `subagentStart`, `subagentStop` | fire | yes | `subagent_start`, `subagent_stop` |
| `sessionEnd` | once per job: complete, error, or timeout | yes | `session_end` |
| `errorOccurred`, `userPromptTransformed` | fire | no | not recorded |
| `preCompact` | only with trigger `auto` | no | not recorded |
| `permissionRequest`, `notification` | do not fire | no | not recorded |

Each job is recorded as its own session. The template does not link sessions from separate jobs on the same pull request.

## Upload paths that are not supported

The landscape plan asked for an upload as a workflow artifact or pull request attachment before the sandbox ends. Neither is shipped, because neither could be verified:

- **Workflow artifact.** Setup steps run before the agent starts, not after it. The hook environment has `GITHUB_COPILOT_API_TOKEN` and `GITHUB_COPILOT_GIT_TOKEN`, but `GITHUB_TOKEN` is not set, and GitHub documents no token or step that can upload an artifact after the agent finishes. ([Hooks reference](https://docs.github.com/en/copilot/reference/hooks-reference#cloud-agent-execution-environment)) The scope of `GITHUB_COPILOT_GIT_TOKEN` is not documented, so the template does not use it.
- **Pull request attachment.** GitHub documents no token-authenticated API for pull request attachments.
- **Native `http` hooks.** The hooks reference recommends an `http` hook entry to retain output. Such a hook posts the raw Copilot payload to a URL fixed in the committed JSON file. `agent-strace server` accepts only agent-strace events, so the template uses command hooks that convert each payload first.

## Security

- The agent can read Agents secrets, so a prompt-injected agent can read `AGENT_STRACE_AUTH_KEY`. The same key also reads sessions from the collector. Run a dedicated collector for cloud agent traces, with a key that guards nothing else.
- Redaction runs in the sandbox before events are sent. Do not use `--no-redact` for cloud agent jobs.
- The collector host must be on the firewall allowlist, which opens it to every command the agent runs.
- See [security.md](security.md) for redaction details.

## Not verified

- No live cloud agent job has been recorded with this template.
- That Agents secrets reach hook processes. GitHub documents them for "the commands and tools that Copilot runs"; hooks are not named.
- That `python3` and `pip` are present in the cloud sandbox image.
- That the setup-steps install in `/tmp/agent-strace-pkg` is visible to hooks in an Actions session.
- Hook payload field names in cloud agent. The script uses the Copilot CLI payload handling verified in the [capture matrix](capture-matrix.md#github-copilot-cli).

## Remove

Delete `.github/hooks/agent-strace.json` and `.github/hooks/agent-strace-cloud.sh` from the default branch, then remove the `AGENT_STRACE_ENDPOINT` and `AGENT_STRACE_AUTH_KEY` secrets and the collector allow rule.
