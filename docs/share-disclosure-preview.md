# Share disclosure preview core

`agent_trace.disclosure.build_disclosure_preview()` provides a value-free summary of the content-rich categories present in a trace. `agent-strace share SESSION --dry-run` uses it to preview ordinary share HTML before the HTML is written; see [`share`](commands.md#share).

The preview reports counts for prompts, responses, tool inputs/results, command-bearing events, path-bearing events, annotations, redacted events, and unredacted content-rich events. It never copies prompt text, tool payload values, command strings, filesystem paths, result text, or annotation text into the preview object.

Command-bearing and path-bearing events are matched by key name, on every event type and inside a tool call's `arguments`. A `command`, `cmd`, or `argv` key marks a command, including the server launch command that the MCP proxy records on `session_start`. A `path`, `file`, `file_path`, `cwd`, or `workdir` key marks a path, and file reads and writes always count as path-bearing. Values are never inspected, so a path or token inside a command line is covered by the command-bearing count, not the path count.

Two modes are represented: `minimized` (the safe default) and `include_content` (an explicit content-sharing choice). The mode flag changes the disclosure contract only; the preview itself remains value-free in both modes.

## Scope

`share --dry-run` reports mode `include_content`, because ordinary share HTML embeds full event data, and counts every annotation, because the HTML lists them all. `--format json` prints `session_id` plus the fields above. Default `agent-strace share` output is unchanged, and the preview does not claim that heuristic redaction finds every sensitive value. Making share minimized by default is tracked in #252; this module lets that work reuse one disclosure calculation instead of duplicating category logic.
