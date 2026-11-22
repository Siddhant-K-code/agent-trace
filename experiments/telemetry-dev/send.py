#!/usr/bin/env python3
"""Send the #241 payloads to Telemetry.dev through agent-strace's OTLP exporter.

The owner runs this once, with a Telemetry.dev key that has Send telemetry
access:

    read -rs TELEMETRY_DEV_API_KEY && export TELEMETRY_DEV_API_KEY
    python3 experiments/telemetry-dev/send.py

The key is read only from the TELEMETRY_DEV_API_KEY environment variable. It
is never a command-line argument, so it never shows in ``ps``. The script
refuses to run without it and never prints it: the exporter's output is
captured and scrubbed of the key before it is shown.

Each committed payload is sent as its own request. The POST goes through
``agent_trace.otlp.export_otlp_payload``, the code that
``agent-strace export <sid> --format ... --endpoint URL`` uses. Before
sending, the script changes two things in each payload:

* It moves every timestamp so the newest span ends about now. The committed
  times are in 2026-09, and data older than the project's retention is
  hidden (the Free plan keeps 14 days).
* It gives every trace ID and span ID a new value, the first 32 or 16 hex
  characters of sha256("<run id>:<old id>"), and rewrites parentSpanId the
  same way. Telemetry.dev treats a body it has stored in the last hour as a
  duplicate, and this keeps each run's body unique.

Everything else is sent unchanged. The exact bytes sent, their sha256, and
the map from old to new IDs go to sent/<run id>/, which git ignores. Nothing
in that directory holds the key.

Exit status: 0 when every payload got a 2xx answer, 1 when any did not or a
rewritten payload failed validation, 2 for a missing or malformed key or
another bad argument.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import ipaddress
import json
import os
import re
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(HERE))

from agent_trace import __version__  # noqa: E402
from agent_trace.otlp import export_otlp_payload  # noqa: E402

import validate  # noqa: E402

KEY_ENV = "TELEMETRY_DEV_API_KEY"
KEY_PREFIX = "td_live_"
DEFAULT_ENDPOINT = "https://api.telemetry.dev"
PAYLOADS = list(validate.PAYLOADS)
SENT = HERE / "sent"
STATUS = re.compile(r"\bHTTP(?: Error)? (\d{3})\b")
LOOPBACK_NAMES = {"localhost"}


def _fail(message: str) -> int:
    sys.stderr.write(f"send.py: {message}\n")
    return 2


def _read_key() -> str | None:
    key = os.environ.get(KEY_ENV, "")
    if not key:
        _fail(f"{KEY_ENV} is not set. Refusing to send. "
              f"Run: read -rs {KEY_ENV} && export {KEY_ENV}")
        return None
    if any(ch.isspace() or not ch.isprintable() or ord(ch) > 126 for ch in key):
        _fail(f"{KEY_ENV} contains whitespace or a non-ASCII character. Refusing to send.")
        return None
    if not key.startswith(KEY_PREFIX):
        _fail(f"{KEY_ENV} does not start with {KEY_PREFIX}. Telemetry.dev keys do. Refusing to send.")
        return None
    return key


def _check_endpoint(endpoint: str) -> str | None:
    parts = urllib.parse.urlsplit(endpoint)
    host = parts.hostname or ""
    if parts.username or parts.password or parts.query or parts.fragment:
        return "the endpoint must not carry user info, a query or a fragment"
    if parts.scheme == "https" and host:
        return None
    if parts.scheme == "http" and host:
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = host in LOOPBACK_NAMES
        if loopback:
            return None
        return "plain http is allowed only for a loopback test server; the key would travel unencrypted"
    return "the endpoint must be an https URL"


def _endpoint_label(endpoint: str) -> str:
    parts = urllib.parse.urlsplit(endpoint)
    return f"{parts.scheme}://{parts.hostname}" + (f":{parts.port}" if parts.port else "")


def _new_id(run_id: str, old: str) -> str:
    new = hashlib.sha256(f"{run_id}:{old}".encode()).hexdigest()[: len(old)]
    if set(new) == {"0"}:  # never all zeros; astronomically unlikely
        new = "1" + new[1:]
    return new


def _spans(payload: dict):
    for rs in payload["resourceSpans"]:
        for ss in rs["scopeSpans"]:
            yield from ss["spans"]


def latest_end_ns(payloads: dict[str, dict]) -> int:
    return max(int(s["endTimeUnixNano"]) for p in payloads.values() for s in _spans(p))


def rewrite(payload: dict, run_id: str, shift_ns: int, environment: str | None) -> tuple[dict, dict]:
    """Return a copy with shifted times and new IDs, and the old-to-new ID map."""
    payload = json.loads(json.dumps(payload))
    ids: dict[str, str] = {}

    def remap(old: str) -> str:
        if old not in ids:
            ids[old] = _new_id(run_id, old)
        return ids[old]

    for rs in payload["resourceSpans"]:
        if environment:
            rs.setdefault("resource", {}).setdefault("attributes", []).append(
                {"key": "deployment.environment.name", "value": {"stringValue": environment}}
            )
        for ss in rs["scopeSpans"]:
            for span in ss["spans"]:
                span["traceId"] = remap(span["traceId"])
                span["spanId"] = remap(span["spanId"])
                if span.get("parentSpanId"):
                    span["parentSpanId"] = remap(span["parentSpanId"])
                for field in ("startTimeUnixNano", "endTimeUnixNano"):
                    span[field] = str(int(span[field]) + shift_ns)
                for event in span.get("events", []):
                    event["timeUnixNano"] = str(int(event["timeUnixNano"]) + shift_ns)
    return payload, ids


def _post(payload: dict, endpoint: str, key: str, span_count: int) -> tuple[bool, str | None, str]:
    """POST through agent-strace's exporter; return (ok, HTTP status, scrubbed output)."""
    captured = io.StringIO()
    with contextlib.redirect_stderr(captured):
        ok = export_otlp_payload(
            payload,
            endpoint,
            headers={"Authorization": f"Bearer {key}"},
            event_count=span_count,
        )
    text = captured.getvalue().replace(key, "[TELEMETRY_DEV_API_KEY]")
    match = STATUS.search(text)
    return ok, match.group(1) if match else None, text.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT,
                        help=f"OTLP/HTTP base URL; /v1/traces is appended (default {DEFAULT_ENDPOINT})")
    parser.add_argument("--only", action="append", choices=PAYLOADS, metavar="PAYLOAD",
                        help=f"send only this payload; repeat for more (default: all of {', '.join(PAYLOADS)})")
    parser.add_argument("--run-id", help="seed for the new IDs and the sent/ directory name (default: UTC time)")
    parser.add_argument("--environment",
                        help="add deployment.environment.name to every resource. Leave unset for a key limited to "
                             "one environment, or the send fails with 403 environment_mismatch")
    args = parser.parse_args(argv)

    key = _read_key()
    if key is None:
        return 2
    problem = _check_endpoint(args.endpoint)
    if problem:
        return _fail(problem)
    if args.environment is not None and not args.environment.strip():
        return _fail("--environment is empty")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", run_id):
        return _fail("--run-id may use only letters, digits, '.', '_' and '-' (at most 64)")

    names = args.only or PAYLOADS
    originals = {name: json.loads((HERE / name).read_text(encoding="utf-8")) for name in names}
    now_ns = time.time_ns()
    shift_ns = (now_ns - 1_000_000_000) - latest_end_ns(originals)

    out_dir = SENT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "sent_at": datetime.fromtimestamp(now_ns / 1e9, timezone.utc).isoformat(timespec="seconds"),
        "agent_strace_version": __version__,
        "endpoint": _endpoint_label(args.endpoint),
        "environment": args.environment,
        "time_shift_seconds": round(shift_ns / 1e9, 3),
        "payloads": [],
    }

    failed = False
    for name in names:
        payload, ids = rewrite(originals[name], run_id, shift_ns, args.environment)
        body = json.dumps(payload).encode("utf-8")  # the bytes export_otlp_payload sends
        errors: list[str] = []
        result = validate.check_payload(name, body, errors, [])
        span_count = len(result["spans"])
        entry = {
            "file": name,
            "spans": span_count,
            "trace_ids": sorted(result["traces"]),
            "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
            "id_map": ids,
        }
        (out_dir / name).write_bytes(body)
        if errors:
            failed = True
            entry["status"] = "not sent: " + "; ".join(errors)
            print(f"{name}: NOT SENT, the rewritten payload failed validation:")
            for error in errors:
                print(f"  {error}")
        else:
            ok, status, text = _post(payload, args.endpoint, key, span_count)
            failed = failed or not ok
            entry["http_status"] = int(status) if status else None
            entry["ok"] = ok
            entry["exporter_output"] = text
            print(f"{name}: HTTP {status or 'unknown'} ({'accepted' if ok else 'FAILED'}), "
                  f"{span_count} spans, {len(body)} bytes")
            for line in text.splitlines():
                print(f"  exporter: {line}")
        for trace_id in entry["trace_ids"]:
            print(f"  trace {trace_id}")
        manifest["payloads"].append(entry)

    manifest_text = json.dumps(manifest, indent=2) + "\n"
    (out_dir / "manifest.json").write_text(manifest_text.replace(key, "[TELEMETRY_DEV_API_KEY]"), encoding="utf-8")
    print(f"agent-strace {__version__}, run {run_id}, sent at {manifest['sent_at']}")
    print(f"Bodies, sha256 and ID map: {out_dir.relative_to(HERE.parents[1])}/")
    print("Next: open each trace ID in Telemetry.dev and fill in CHECKLIST.md.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
