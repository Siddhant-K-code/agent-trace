#!/usr/bin/env python3
"""Check the committed payloads against Telemetry.dev's documented OTLP JSON rules.

This is an offline check that sends nothing. The rules come from
https://telemetry.dev/docs/reference/otlp ("OTLP JSON rules", "IDs",
"Body limits", "Error responses"), fetched 2026-11-22:

* no duplicate JSON keys
* lowerCamelCase field names
* ``kind`` and ``status.code`` are integers
* hex trace IDs (32 characters) and span IDs (16), never all zeros
* nanosecond timestamps
* at least one span
* at most 16 MiB

It also checks that:

* every ``parentSpanId`` resolves inside its own trace
* no trace ID or span ID is shared between payloads
* no local path, user name or host name has leaked into a payload
* MAPPING.md names every attribute key and event name the payloads send

It also prints which keys Telemetry.dev's documented secret-key rule would
drop, and which string values its documented value patterns would redact.
These are predictions from the docs, not observed behavior.

    python3 experiments/telemetry-dev/validate.py
"""

from __future__ import annotations

import getpass
import json
import re
import socket
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAYLOADS = ["payload-otlp.json", "payload-otlp-genai.json", "payload-otlp-genai-tree.json"]
MAPPING = HERE / "MAPPING.md"
MAX_BYTES = 16 * 1024 * 1024
CAMEL = re.compile(r"^[a-z][a-zA-Z0-9]*$")
HEX = re.compile(r"^[0-9a-f]+$")
DIGITS = re.compile(r"^[0-9]+$")

# Telemetry.dev "Attributes with secret-like keys" (docs/platform/privacy).
SECRET_WORDS = {"authorization", "cookie", "credential", "credentials", "password", "secret", "token"}
# "Token usage and timing keys stay, such as gen_ai.usage.input_tokens,
# llm.token_count.prompt, and time_to_first_token."
USAGE_PREFIXES = ("gen_ai.usage.", "llm.token_count.", "ai.usage.")
# Telemetry.dev "Secret and personal values in text". These are rough local
# stand-ins for the documented patterns, used only to flag likely matches.
VALUE_PATTERNS = {
    "private key block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "telemetry.dev key": re.compile(r"td_live_[A-Za-z0-9]"),
    "bearer credential": re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"),
    "provider key": re.compile(r"\b(sk-[A-Za-z0-9]|AKIA[0-9A-Z]{8}|AIza[0-9A-Za-z_-]{8}|gh[pousr]_[A-Za-z0-9]|github_pat_|xox[abposr]-|xapp-)"),
    "name=value secret": re.compile(r"(api_key|access_token|refresh_token|password|secret)\"?\s*[:=]"),
    "url credentials": re.compile(r"://[^/\s:@]+:[^/\s@]+@"),
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z]{2,}"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone": re.compile(r"(\(\d{3}\)\s?\d{3}-\d{4}|\b\d{3}-\d{3}-\d{4}\b|\+\d[\d\s-]{7,})"),
}
CARD_RUN = re.compile(r"(?<!\d)\d{13,19}(?!\d)")


def _luhn(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def _attr_strings(payload):
    """String attribute values only; IDs and timestamps are structural, not text."""
    for rs in payload["resourceSpans"]:
        blocks = [rs.get("resource", {}).get("attributes", [])]
        for ss in rs["scopeSpans"]:
            for span in ss["spans"]:
                blocks.append(span.get("attributes", []))
                blocks.extend(e.get("attributes", []) for e in span.get("events", []))
                if span.get("status", {}).get("message"):
                    yield span["status"]["message"]
                yield span["name"]
        for attrs in blocks:
            for attr in attrs:
                yield from _strings(attr["value"])


def _words(key: str) -> list[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", key)
    return [w.lower() for w in re.split(r"[.\-_\s]+", spaced) if w]


def secret_key_drop(key: str) -> str:
    """Return why Telemetry.dev's key rule would drop ``key``, or ''."""
    words = _words(key)
    reason = ""
    if key.lower() == "apikey":
        reason = "whole key is apikey"
    elif any(w in SECRET_WORDS for w in words):
        reason = "word " + next(w for w in words if w in SECRET_WORDS)
    elif any(a == "api" and b == "key" for a, b in zip(words, words[1:])):
        reason = "api followed by key"
    if reason and key.startswith(USAGE_PREFIXES):
        return ""
    return reason


def _no_duplicates(pairs):
    seen = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"duplicate JSON key {key!r}")
        seen[key] = value
    return seen


def _field_names(node, path, errors):
    """Check structural field names. Attribute keys are values, so they are skipped."""
    if isinstance(node, dict):
        for key, value in node.items():
            if not CAMEL.match(key):
                errors.append(f"{path}.{key}: field name is not lowerCamelCase")
            _field_names(value, f"{path}.{key}", errors)
    elif isinstance(node, list):
        for i, item in enumerate(node):
            _field_names(item, f"{path}[{i}]", errors)


def _strings(node):
    if isinstance(node, dict):
        for value in node.values():
            yield from _strings(value)
    elif isinstance(node, list):
        for item in node:
            yield from _strings(item)
    elif isinstance(node, str):
        yield node


def _attr_keys(payload):
    keys = set()
    for rs in payload["resourceSpans"]:
        keys.update(a["key"] for a in rs.get("resource", {}).get("attributes", []))
        for ss in rs["scopeSpans"]:
            for span in ss["spans"]:
                keys.update(a["key"] for a in span.get("attributes", []))
                for event in span.get("events", []):
                    keys.update(a["key"] for a in event.get("attributes", []))
    return keys


def _event_names(payload):
    return {
        event["name"]
        for rs in payload["resourceSpans"] for ss in rs["scopeSpans"]
        for span in ss["spans"] for event in span.get("events", [])
    }


def _check_id(value, length, label, errors):
    if not isinstance(value, str) or len(value) != length or not HEX.match(value):
        errors.append(f"{label}: {value!r} is not {length} lowercase hex characters")
    elif set(value) == {"0"}:
        errors.append(f"{label}: all zeros")


def check_payload(name: str, raw: bytes, errors: list[str], warnings: list[str]) -> dict:
    if len(raw) > MAX_BYTES:
        errors.append(f"{name}: {len(raw)} bytes is over the 16 MiB limit (413)")
    payload = json.loads(raw, object_pairs_hook=_no_duplicates)
    _field_names(payload, name, errors)
    spans = []
    for rs in payload.get("resourceSpans", []):
        resource_keys = {a["key"] for a in rs.get("resource", {}).get("attributes", [])}
        missing_env = not resource_keys & {"deployment.environment.name", "deployment.environment"}
        if missing_env and not any(w.startswith(f"{name}: resource") for w in warnings):
            warnings.append(
                f"{name}: resource {sorted(resource_keys)} has no deployment.environment(.name); "
                "Telemetry.dev uses the key's environment, or 'default'"
            )
        for ss in rs.get("scopeSpans", []):
            spans.extend(ss.get("spans", []))
    if not spans:
        errors.append(f"{name}: no spans (400 empty_payload)")
    by_trace: dict[str, set[str]] = {}
    for span in spans:
        label = f"{name}:{span.get('name')}"
        _check_id(span.get("traceId"), 32, f"{label} traceId", errors)
        _check_id(span.get("spanId"), 16, f"{label} spanId", errors)
        by_trace.setdefault(span["traceId"], set()).add(span["spanId"])
        if not isinstance(span.get("kind"), int):
            errors.append(f"{label}: kind {span.get('kind')!r} is not an integer")
        status = span.get("status", {})
        if "code" in status and not isinstance(status["code"], int):
            errors.append(f"{label}: status.code {status['code']!r} is not an integer")
        start, end = span.get("startTimeUnixNano"), span.get("endTimeUnixNano")
        for field, value in (("startTimeUnixNano", start), ("endTimeUnixNano", end)):
            if not isinstance(value, str) or not DIGITS.match(value):
                errors.append(f"{label}: {field} {value!r} is not a decimal nanosecond string")
        if DIGITS.match(str(start)) and DIGITS.match(str(end)):
            if int(end) < int(start):
                errors.append(f"{label}: ends before it starts")
            for event in span.get("events", []):
                t = int(event["timeUnixNano"])
                if not int(start) <= t <= int(end):
                    warnings.append(f"{label}: event {event['name']} is outside the span's time range")
    for span in spans:
        parent = span.get("parentSpanId", "")
        if parent:
            _check_id(parent, 16, f"{name}:{span['name']} parentSpanId", errors)
            if parent not in by_trace.get(span["traceId"], set()):
                errors.append(f"{name}:{span['name']}: parentSpanId {parent} is not a span in its trace")
    return {"payload": payload, "spans": spans, "traces": by_trace}


def check_classic_tree(errors: list[str]) -> str:
    """Build the sessions in a temp store and check tree_to_otlp's parent links.

    The CLI's --format otlp has no tree option, so this library path is not
    committed as a payload. It is checked here instead.
    """
    import tempfile

    sys.path.insert(0, str(HERE))
    import generate  # noqa: E402  (puts src/ on sys.path)
    from agent_trace.otlp import tree_to_otlp
    from agent_trace.store import TraceStore

    prefix = "td241-ctree"
    with tempfile.TemporaryDirectory() as tmp:
        store = TraceStore(tmp)
        generate.write_sessions(store, generate.build_sessions(prefix))
        payload = tree_to_otlp(store, f"{prefix}-parent")
    spans = [s for rs in payload["resourceSpans"] for ss in rs["scopeSpans"] for s in ss["spans"]]
    traces = {s["traceId"] for s in spans}
    ids = {s["spanId"] for s in spans}
    if len(traces) != 1:
        errors.append(f"classic tree: {len(traces)} traces, expected 1")
    for span in spans:
        if span.get("parentSpanId") and span["parentSpanId"] not in ids:
            errors.append(f"classic tree: {span['name']} parent {span['parentSpanId']} does not resolve")
    task = [s for s in spans if s["name"] == "tool/Task"]
    roots = [s for s in spans if s["name"].startswith("agent-session") and s.get("parentSpanId")]
    if len(task) != 1 or len(roots) != 1 or roots[0]["parentSpanId"] != task[0]["spanId"]:
        errors.append("classic tree: the child session's root is not under the tool/Task span")
    return f"classic tree_to_otlp (library only): {len(spans)} spans in {len(traces)} trace(s), child root under tool/Task"


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []
    results = {}
    for name in PAYLOADS:
        raw = (HERE / name).read_bytes()
        try:
            results[name] = check_payload(name, raw, errors, warnings)
        except ValueError as exc:
            errors.append(f"{name}: {exc}")

    seen_traces: dict[str, str] = {}
    seen_spans: dict[tuple[str, str], str] = {}
    for name, result in results.items():
        for trace_id in result["traces"]:
            if trace_id in seen_traces and seen_traces[trace_id] != name:
                errors.append(f"trace {trace_id} is in {seen_traces[trace_id]} and {name}")
            seen_traces[trace_id] = name
        for span in result["spans"]:
            key = (span["traceId"], span["spanId"])
            if key in seen_spans:
                errors.append(f"span {key} appears twice ({seen_spans[key]}, {name})")
            seen_spans[key] = name

    local = ["/Users/", "/home/", "/var/folders", "/private/", "/tmp/", "td241-tmp"]
    user, host = getpass.getuser(), socket.gethostname().split(".")[0]
    local += [v for v in (user, host) if v and len(v) >= 4]
    for name, result in results.items():
        text = "\n".join(_strings(result["payload"]))
        for needle in local:
            if needle.lower() in text.lower():
                errors.append(f"{name}: contains a local value ({'user or host name' if needle in (user, host) else needle})")

    all_keys = set().union(*(_attr_keys(r["payload"]) for r in results.values())) if results else set()
    all_events = set().union(*(_event_names(r["payload"]) for r in results.values())) if results else set()
    if MAPPING.exists():
        mapping = MAPPING.read_text(encoding="utf-8")
        for key in sorted(all_keys | all_events):
            if f"`{key}`" not in mapping:
                errors.append(f"MAPPING.md does not name `{key}`")
    else:
        errors.append("MAPPING.md is missing")

    print("Predicted key drops (doc-based, not observed):")
    drops = {key: secret_key_drop(key) for key in sorted(all_keys)}
    dropped = {k: v for k, v in drops.items() if v}
    for key, why in dropped.items():
        print(f"  drop {key}: {why}")
    if not dropped:
        print("  none")
    print("Predicted value redactions (doc-based, not observed):")
    hits = 0
    for name, result in results.items():
        for text in _attr_strings(result["payload"]):
            for label, pattern in VALUE_PATTERNS.items():
                if pattern.search(text):
                    hits += 1
                    print(f"  {name}: {label} in {text[:60]!r}")
            for run in CARD_RUN.findall(text):
                if _luhn(run):
                    hits += 1
                    print(f"  {name}: Luhn-valid {len(run)}-digit run in {text[:60]!r}")
    if not hits:
        print("  none")

    for name, result in results.items():
        print(f"{name}: {len(result['spans'])} spans in {len(result['traces'])} trace(s)")
    print(check_classic_tree(errors))
    print(f"{len(all_keys)} attribute keys, {len(all_events)} event names")
    for warning in warnings:
        print(f"warning: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    print("FAIL" if errors else "OK")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
