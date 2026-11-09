# Share disclosure preview and manifest

`agent_trace.disclosure.build_disclosure_preview()` provides a value-free
summary of the categories present in a trace. `agent-strace share SESSION
--dry-run` uses it before any HTML is written. See [`share`](commands.md#share).

The preview schema is version 2. It reports:

- the exact selected mode, `minimized` or `include_content`
- counts for prompts, responses, tool inputs, tool results, command-bearing
  events, path-bearing events, source annotations, included annotations,
  omitted annotations, content-rich events, redacted events, and unredacted
  content-rich events
- content minimization, anonymization action, and sensitive-field warning counts
- evidence-health status and known-gap count
- provider, capture method, and capture-matrix version
- redaction and anonymization state
- included and omitted field categories
- fixed limitations and warnings without trace values

The preview never copies prompt text, tool payload values, command strings,
filesystem paths, result text, or annotation text into the preview object.

Command-bearing and path-bearing events are matched by key name, on every event
type and inside a tool call's `arguments`. A `command`, `cmd`, or `argv` key
marks a command, including a server launch command recorded on `session_start`.
A `path`, `file`, `file_path`, `cwd`, or `workdir` key marks a path. File reads
and writes always count as path-bearing. Values are never inspected, so a path
or token inside a command line is covered by the command-bearing count, not the
path count.

Every event's `data` must be an object. If any event's `data` is an array,
string, number, boolean, or null, `build_disclosure_preview()` raises
`NonObjectEventDataError`, a `ValueError`. Key names cannot be read from such a
payload, so counting the event without commands or paths would under-count. The
message names the first event by position and event type and never includes data
values. `share` runs the same check before every mode, so the preview and HTML
refuse the same sessions.

## Evidence packet manifest

Every share HTML file embeds an `agent-strace.evidence-packet` manifest. The
manifest records:

- packet schema and agent-strace version
- selected disclosure mode
- provider and capture provenance
- evidence health, known gaps, and redaction state
- anonymization state and replacement count
- value-free failure, retry, and recovery counts
- source-trace, selected-event, included-annotation, and manifest SHA-256
  integrity references
- explicit limitations

The manifest is deterministic for the same source snapshot, selected mode, and
capture state. The HTML generation timestamp is not part of the manifest.
Local SHA-256 values are integrity references, not signatures or independent
timestamps.

## Modes and annotations

Default `share --dry-run` reports `minimized`. The matching HTML omits
content-rich values and includes only annotations created with
`agent-strace annotate --shareable`. Event, parent, and annotation identifiers
are replaced with packet-local references. Shareable annotation labels and
notes are included verbatim, so mark them shareable only after reviewing their
content.

`share --include-content --dry-run` reports `include_content`. The matching HTML
contains full event payloads and all annotations. It can contain sensitive
values even when capture-time redaction was enabled. Heuristic redaction does
not detect every secret or personal identifier.

`--format json` prints `session_id` plus the complete preview object. Assignment
ZIP files use a separate privacy format and are not described by this preview
or manifest.
