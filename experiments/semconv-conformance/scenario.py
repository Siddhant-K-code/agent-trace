#!/usr/bin/env python3
"""Export a synthetic agent-strace session and send it to Weaver live-check.

The conformance runner starts Weaver, sets ``OTEL_EXPORTER_OTLP_ENDPOINT``
and runs this script once per scenario:

* ``session``: ``agent-strace export <parent> --format otlp-genai`` and the
  same for the child subagent session, sent in one request (two traces).
* ``tree``: ``agent-strace export <parent> --format otlp-genai
  --include-subagents`` (one trace, the child under the parent's ``Task``
  span).
* ``sdk``: ``sdk_calls.py`` makes OpenAI and Anthropic SDK calls against the
  runner's mock server (``$MOCK_SERVER_URL``) with agent-strace's SDK
  integrations on, and the session they record is exported the same way.
  Needs ``openai`` and ``anthropic`` installed next to the runner.
* ``invalid-content``: the ``session`` payload with every
  ``gen_ai.input.messages`` (one on each session's root span) replaced by a
  message that has no ``parts``. Weaver must report ``genai_content_schema``
  for each; it is the negative check.

The ``session`` and ``tree`` sessions are the deterministic #241 sessions
that ``experiments/telemetry-dev/generate.py`` builds: a failed model call
and its retry (a ``retry_of`` span link), a tool error and its retry, a
redacted argument, and a ``Task`` call that starts a child subagent session.

agent-strace sends OTLP over HTTP only, as JSON or protobuf (ADR-0006);
Weaver only accepts OTLP/gRPC. So the payload goes through a shim: OTLP JSON
hex IDs become base64, ``json_format`` parses the result into an
``ExportTraceServiceRequest``, and it goes to the endpoint over insecure gRPC.
Before it sends, the script checks that every span link survived the parse
with the same trace and span IDs, and exits 1 if one did not.

``--output FILE`` writes the OTLP JSON that would be sent and sends nothing,
so a payload can be read without the runner.

Needs ``grpcio`` and ``opentelemetry-proto``, which the conformance runner
installs. agent-strace itself runs from this checkout's ``src`` in a clean
environment, so no installed agent-strace and no ``AGENT_STRACE_*`` or
``OTEL_*`` setting changes what it exports.
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SRC = REPO / "src"
GENERATOR = REPO / "experiments" / "telemetry-dev"

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(GENERATOR))

from generate import build_sessions, write_sessions  # noqa: E402

from agent_trace.store import TraceStore  # noqa: E402

PREFIX = "semconv-conformance"
# A message must have "parts"; this one has none.
INVALID_MESSAGES = json.dumps([{"role": "user"}])
_ID_KEYS = ("traceId", "spanId", "parentSpanId")


def _clean_env(home: str) -> dict[str, str]:
    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("AGENT_STRACE_", "AGENT_TRACE_", "OTEL_"))
    }
    env.update({
        "PYTHONPATH": str(SRC),
        "HOME": home,
        "DO_NOT_TRACK": "1",
        "AGENT_STRACE_TELEMETRY": "0",
    })
    return env


def _export(trace_dir: str, home: str, session_id: str, out: Path,
            include_subagents: bool = False) -> dict:
    cmd = [
        sys.executable, "-m", "agent_trace.entry", "--trace-dir", trace_dir,
        "export", session_id, "--format", "otlp-genai", "--output", str(out),
    ]
    if include_subagents:
        cmd.append("--include-subagents")
    subprocess.run(cmd, check=True, cwd=home, env=_clean_env(home),
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return json.loads(out.read_text(encoding="utf-8"))


def _spans(payload: dict):
    for rs in payload["resourceSpans"]:
        for ss in rs["scopeSpans"]:
            yield from ss["spans"]


def _break_input_messages(payload: dict) -> int:
    broken = 0
    for span in _spans(payload):
        for attr in span.get("attributes", []):
            if attr["key"] == "gen_ai.input.messages":
                attr["value"] = {"stringValue": INVALID_MESSAGES}
                broken += 1
    return broken


def _record_sdk_calls(trace_dir: str, home: str, base_url: str) -> str:
    env = _clean_env(home)
    env["AGENT_TRACE_DIR"] = trace_dir
    done = subprocess.run(
        [sys.executable, str(HERE / "sdk_calls.py"), base_url],
        check=True, cwd=home, env=env, capture_output=True, text=True,
    )
    return done.stdout.strip().splitlines()[-1]


def build_payload(scenario: str, base_url: str | None = None) -> dict:
    with tempfile.TemporaryDirectory(prefix="semconv-conformance-") as tmp:
        home = os.path.join(tmp, "home")
        trace_dir = os.path.join(tmp, "traces")
        os.makedirs(home)
        if scenario == "sdk":
            if not base_url:
                raise SystemExit("sdk: set MOCK_SERVER_URL or pass --base-url")
            session_id = _record_sdk_calls(trace_dir, home, base_url)
            return _export(trace_dir, home, session_id, Path(tmp) / "sdk.json")
        write_sessions(TraceStore(trace_dir), build_sessions(PREFIX))
        parent, child = f"{PREFIX}-parent", f"{PREFIX}-child"
        if scenario == "tree":
            return _export(trace_dir, home, parent, Path(tmp) / "tree.json",
                           include_subagents=True)
        parts = [_export(trace_dir, home, sid, Path(tmp) / f"{sid}.json")
                 for sid in (parent, child)]
    payload = {"resourceSpans": parts[0]["resourceSpans"] + parts[1]["resourceSpans"]}
    if scenario == "invalid-content" and not _break_input_messages(payload):
        raise SystemExit("invalid-content: no gen_ai.input.messages to break")
    return payload


def _hex_ids_to_base64(node) -> None:
    if isinstance(node, dict):
        for key, value in list(node.items()):
            if key in _ID_KEYS and isinstance(value, str):
                node[key] = base64.b64encode(bytes.fromhex(value)).decode() if value else ""
            else:
                _hex_ids_to_base64(value)
    elif isinstance(node, list):
        for item in node:
            _hex_ids_to_base64(item)


def _links(payload: dict) -> list[tuple[str, str, str]]:
    return sorted(
        (span["spanId"], link["traceId"], link["spanId"])
        for span in _spans(payload) for link in span.get("links", [])
    )


def to_request(payload: dict):
    """The protobuf request for ``payload`` and its number of span links.

    Exits if a span link did not survive the conversion unchanged.
    """
    from google.protobuf import json_format
    from opentelemetry.proto.collector.trace.v1 import trace_service_pb2

    converted = copy.deepcopy(payload)
    _hex_ids_to_base64(converted)
    request = json_format.ParseDict(converted, trace_service_pb2.ExportTraceServiceRequest())
    sent = _links(payload)
    parsed = sorted(
        (span.span_id.hex(), link.trace_id.hex(), link.span_id.hex())
        for rs in request.resource_spans for ss in rs.scope_spans
        for span in ss.spans for link in span.links
    )
    if parsed != sent:
        raise SystemExit(f"span links changed in the shim: {sent} became {parsed}")
    return request, len(sent)


def send(payload: dict) -> None:
    import grpc
    from opentelemetry.proto.collector.trace.v1 import trace_service_pb2_grpc

    request, links = to_request(payload)
    endpoint = os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"]
    target = endpoint.removeprefix("http://").removeprefix("https://")
    with grpc.insecure_channel(target) as channel:
        trace_service_pb2_grpc.TraceServiceStub(channel).Export(request, timeout=10)
    spans = sum(1 for _ in _spans(payload))
    print(f"sent {spans} spans and {links} span links to {target}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scenario", choices=("session", "tree", "sdk", "invalid-content"))
    parser.add_argument("--output", type=Path,
                        help="write the OTLP JSON here and send nothing")
    parser.add_argument("--base-url", default=os.environ.get("MOCK_SERVER_URL"),
                        help="the mock server for the sdk scenario "
                        "(default $MOCK_SERVER_URL)")
    args = parser.parse_args(argv)
    payload = build_payload(args.scenario, args.base_url)
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return 0
    send(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
