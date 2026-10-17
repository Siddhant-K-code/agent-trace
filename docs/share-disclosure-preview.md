# Share disclosure preview core

`agent_trace.disclosure.build_disclosure_preview()` provides a value-free summary of the content-rich categories present in a trace. It is intended as the calculation layer for a future `agent-strace share SESSION --dry-run` flow.

The preview reports counts for prompts, responses, tool inputs/results, command-bearing events, path-bearing events, annotations, redacted events, and unredacted content-rich events. It never copies prompt text, tool payload values, command strings, filesystem paths, result text, or annotation text into the preview object.

Two modes are represented: `minimized` (the safe default) and `include_content` (an explicit content-sharing choice). The mode flag changes the disclosure contract only; the preview itself remains value-free in both modes.

## Scope

This module does not yet change `agent-strace share` output and does not claim that heuristic redaction finds every sensitive value. It is a small, testable foundation for #252 so the CLI/HTML wiring can reuse one disclosure calculation instead of duplicating category logic.
