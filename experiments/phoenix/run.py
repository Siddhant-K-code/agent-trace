#!/usr/bin/env python3
"""Send the synthetic #241 sessions to a local Phoenix over OTLP/HTTP protobuf.

This is the #560 check. Start Phoenix first (see README.md), then run:

    python3 experiments/phoenix/run.py [--phoenix http://127.0.0.1:6016]

The script:

1. writes the synthetic #241 sessions (``experiments/telemetry-dev/generate.py``)
   into a temporary trace store;
2. sends them with the real CLI: ``export --endpoint`` with the default JSON
   (Phoenix answers 415), then with protobuf for ``otlp-genai`` (``--protocol``)
   and ``otlp`` (``OTEL_EXPORTER_OTLP_TRACES_PROTOCOL``), and ``--protocol grpc``,
   which must be refused;
3. sends the parent and its subagent as one trace with
   ``export --include-subagents --endpoint --protocol http/protobuf``;
4. streams one session with ``watch --stream-otlp``, once with protobuf and
   once with the default JSON, through a local relay that records each POST
   and forwards it to Phoenix;
5. reads back what Phoenix stored over its REST and GraphQL APIs and compares
   it with what was sent.

It prints a JSON report on stdout. It needs no key and refuses a Phoenix URL
that is not a loopback address. Session IDs start with ``px560-``, so run it
against a fresh Phoenix working directory.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments" / "telemetry-dev"))

import generate  # noqa: E402
from agent_trace import __version__  # noqa: E402
from agent_trace.store import TraceStore  # noqa: E402

PREFIX = "px560"
GROUPS = {
    "otlp-genai": f"{PREFIX}-genai",
    "otlp": f"{PREFIX}-otlp",
    "tree": f"{PREFIX}-tree",
    "watch-protobuf": f"{PREFIX}-watch",
    "watch-json": f"{PREFIX}-watchjson",
}
STATUS = {0: "UNSET", 1: "OK", 2: "ERROR"}
ROOT_COUNTS = ("agent.tool_calls", "agent.llm_requests", "agent.errors")


# ---------------------------------------------------------------------------
# Phoenix HTTP
# ---------------------------------------------------------------------------

def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read()


def _graphql(base: str, query: str, variables: dict | None = None) -> dict:
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    request = urllib.request.Request(
        f"{base}/graphql", data=body, headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        reply = json.load(response)
    if reply.get("errors"):
        raise RuntimeError(f"GraphQL errors: {reply['errors']}")
    return reply["data"]


def rest_spans(base: str) -> list[dict]:
    spans, cursor = [], None
    while True:
        url = f"{base}/v1/projects/default/spans?limit=1000"
        if cursor:
            url += f"&cursor={cursor}"
        page = json.loads(_get(url))
        spans += page["data"]
        cursor = page.get("next_cursor")
        if not cursor:
            return spans


_PROJECT_QUERY = """
query ($id: ID!) {
  node(id: $id) {
    ... on Project {
      spans(first: 1000) { edges { node {
        name spanKind statusCode tokenCountTotal tokenCountPrompt tokenCountCompletion
        costSummary { total { cost } }
        input { value mimeType } output { value mimeType }
        context { spanId traceId }
      } } }
      sessions(first: 100) { edges { node {
        sessionId numTraces
        tokenUsage { total prompt completion }
        costSummary { total { cost } }
      } } }
    }
  }
}
"""


def graphql_project(base: str) -> dict:
    projects = _graphql(base, "{ projects { edges { node { id name } } } }")
    [project_id] = [
        edge["node"]["id"] for edge in projects["projects"]["edges"]
        if edge["node"]["name"] == "default"
    ]
    return _graphql(base, _PROJECT_QUERY, {"id": project_id})["node"]


# ---------------------------------------------------------------------------
# Relay for watch --stream-to: records each POST and forwards it to Phoenix
# ---------------------------------------------------------------------------

class _RelayHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        content_type = self.headers.get("Content-Type")
        request = urllib.request.Request(
            self.server.target + self.path, data=body,
            headers={"Content-Type": content_type}, method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                status, reply = response.status, response.read()
        except urllib.error.HTTPError as exc:
            status, reply = exc.code, exc.read()
        with self.server.lock:
            self.server.posts.append(
                {"path": self.path, "content_type": content_type, "status": status, "body": body}
            )
        self.send_response(status)
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, *args) -> None:
        pass


class Relay(ThreadingHTTPServer):
    def __init__(self, target: str) -> None:
        super().__init__(("127.0.0.1", 0), _RelayHandler)
        self.target = target
        self.posts: list[dict] = []
        self.lock = threading.Lock()

    def take(self) -> list[dict]:
        with self.lock:
            posts, self.posts = self.posts, []
        return posts


# ---------------------------------------------------------------------------
# Minimal protobuf reader, for the spans in each relayed POST
# ---------------------------------------------------------------------------

def _varint(buf: bytes, i: int) -> tuple[int, int]:
    shift = result = 0
    while True:
        byte = buf[i]
        i += 1
        result |= (byte & 0x7F) << shift
        if byte < 0x80:
            return result, i
        shift += 7


def _fields(buf: bytes):
    i = 0
    while i < len(buf):
        key, i = _varint(buf, i)
        number, wire_type = key >> 3, key & 7
        if wire_type == 0:
            value, i = _varint(buf, i)
        elif wire_type == 1:
            value, i = int.from_bytes(buf[i:i + 8], "little"), i + 8
        elif wire_type == 2:
            size, i = _varint(buf, i)
            value, i = buf[i:i + size], i + size
        elif wire_type == 5:
            value, i = buf[i:i + 4], i + 4
        else:
            raise ValueError(f"wire type {wire_type}")
        yield number, value


def protobuf_spans(body: bytes) -> list[dict]:
    """Read span IDs, names, times and string or int attributes from a body."""
    spans = []
    for number, resource_spans in _fields(body):
        if number != 1:
            continue
        for number, scope_spans in _fields(resource_spans):
            if number != 2:
                continue
            for number, raw in _fields(scope_spans):
                if number != 2:
                    continue
                span = {"attributes": {}}
                for field, value in _fields(raw):
                    if field == 2:
                        span["span_id"] = value.hex()
                    elif field == 4:
                        span["parent_span_id"] = value.hex()
                    elif field == 5:
                        span["name"] = value.decode()
                    elif field == 7:
                        span["start_us"] = _ns_to_us(value)
                    elif field == 8:
                        span["end_us"] = _ns_to_us(value)
                    elif field == 9:
                        kv = dict(_fields(value))
                        any_value = dict(_fields(kv.get(2, b"")))
                        key = kv[1].decode()
                        if 3 in any_value:
                            span["attributes"][key] = any_value[3]
                        elif 1 in any_value:
                            span["attributes"][key] = any_value[1].decode()
                spans.append(span)
    return spans


# ---------------------------------------------------------------------------
# CLI runs
# ---------------------------------------------------------------------------

class Runner:
    def __init__(self, tmp: str) -> None:
        self.tmp = tmp
        self.home = os.path.join(tmp, "home")
        self.trace_dir = os.path.join(tmp, "traces")
        os.makedirs(self.home)
        self.store = TraceStore(self.trace_dir)
        self.runs: list[dict] = []

    def _scrub(self, text: str) -> str:
        return text.replace(self.tmp, "<tmpdir>")

    def _argv(self, args) -> list[str]:
        return [sys.executable, "-m", "agent_trace.entry", "--trace-dir", self.trace_dir, *args]

    def _env(self, extra: dict | None) -> dict:
        env = generate._clean_env(self.home)
        env.update(extra or {})
        return env

    def cli(self, *args: str, env: dict | None = None, record: bool = True) -> dict:
        proc = subprocess.run(
            self._argv(args), cwd=self.home, env=self._env(env),
            capture_output=True, text=True, timeout=120,
        )
        run = {
            "command": " ".join(["agent-strace", *args]),
            "env": env or {},
            "exit": proc.returncode,
            "stdout": self._scrub(proc.stdout),
            "stderr": self._scrub(proc.stderr),
        }
        if record:
            self.runs.append(run)
        return run

    def sent_payload(self, session_id: str, fmt: str, *flags: str) -> dict:
        """Return the JSON payload the CLI builds for *session_id* (the --output file)."""
        out = os.path.join(self.tmp, f"{session_id}-{fmt}{''.join(flags)}.json")
        run = self.cli("export", session_id, "--format", fmt, *flags, "--output", out, record=False)
        if run["exit"] != 0:
            raise RuntimeError(run["stderr"])
        with open(out, encoding="utf-8") as handle:
            return json.load(handle)

    def watch(self, prefix: str, relay_url: str, protocol_args: list[str]) -> dict:
        """Watch a session that grows while watch runs, streaming to *relay_url*."""
        meta, events, _, _ = generate.build_sessions(prefix)
        session_id = meta.session_id
        self.store.create_session(meta)
        self.store.append_events(session_id, events[:1])
        args = [
            "watch", session_id, "--stream-to", relay_url, "--stream-otlp", *protocol_args,
            "--stream-flush-interval", "1",
        ]
        proc = subprocess.Popen(
            self._argv(args), cwd=self.home, env=self._env(None),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        time.sleep(4)
        self.store.append_events(session_id, events[1:])
        out, err = proc.communicate(timeout=180)
        parts = urlsplit(relay_url)
        origin = f"{parts.scheme}://{parts.netloc}"
        run = {
            "command": " ".join(["agent-strace", *args]).replace(relay_url, "RELAY_URL"),
            "env": {},
            "exit": proc.returncode,
            "stdout_lines": len(out.splitlines()),
            "stderr": self._scrub(err).replace(origin, "RELAY_URL"),
            "events": len(events),
        }
        self.runs.append(run)
        return run


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _value(value: dict):
    if "stringValue" in value:
        return value["stringValue"]
    if "intValue" in value:
        return int(value["intValue"])
    if "doubleValue" in value:
        return float(value["doubleValue"])
    if "boolValue" in value:
        return value["boolValue"]
    if "arrayValue" in value:
        return [_value(item) for item in value["arrayValue"].get("values", [])]
    if "kvlistValue" in value:
        return {kv["key"]: _value(kv["value"]) for kv in value["kvlistValue"].get("values", [])}
    return value.get("bytesValue")


def _attrs(items) -> dict:
    return {kv["key"]: _value(kv["value"]) for kv in items or []}


_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _iso_us(text: str) -> int:
    return (datetime.fromisoformat(text) - _EPOCH) // timedelta(microseconds=1)


def _ns_to_us(nanos) -> int:
    # Phoenix keeps microseconds, rounded. agent-strace's nanoseconds come from
    # float seconds, so they can sit just below a whole microsecond.
    return (int(nanos) + 500) // 1000


def _payload_spans(payload: dict) -> list[dict]:
    return [
        span
        for resource_spans in payload["resourceSpans"]
        for scope_spans in resource_spans["scopeSpans"]
        for span in scope_spans["spans"]
    ]


def compare(sent_spans: list[dict], stored: dict[str, dict], gql: dict[str, dict]) -> dict:
    """Compare the spans sent with what Phoenix's REST and GraphQL APIs return."""
    missing_spans, mismatched = [], Counter()
    lost, changed, added, kinds = Counter(), Counter(), Counter(), Counter()
    sent_keys, event_lost = Counter(), Counter()
    links = {"sent": 0, "stored": 0, "rest_api_has_links_field": None}
    tokens, io = [], Counter()
    for span in sent_spans:
        span_id = span["spanId"]
        got = stored.get(span_id)
        if got is None:
            missing_spans.append(span["name"])
            continue
        status = span.get("status", {})
        checks = {
            "name": got["name"] == span["name"],
            "trace_id": got["context"]["trace_id"] == span["traceId"],
            "parent": (got["parent_id"] or "") == span.get("parentSpanId", ""),
            "start": _iso_us(got["start_time"]) == _ns_to_us(span["startTimeUnixNano"]),
            "end": _iso_us(got["end_time"]) == _ns_to_us(span["endTimeUnixNano"]),
            "status_code": got["status_code"] == STATUS[status.get("code", 0)],
            "status_message": got["status_message"] == status.get("message", ""),
        }
        for check, ok in checks.items():
            if not ok:
                mismatched[check] += 1
        sent_attrs = _attrs(span.get("attributes"))
        sent_keys.update(sent_attrs.keys())
        got_attrs = got["attributes"]
        for key, value in sent_attrs.items():
            if key not in got_attrs:
                lost[key] += 1
            elif got_attrs[key] != value:
                changed[key] += 1
        for key in got_attrs:
            if key not in sent_attrs:
                added["metadata.*" if key.startswith("metadata.") else key] += 1
        kinds[(sent_attrs.get("openinference.span.kind"), got["span_kind"])] += 1
        got_events = {(e["name"], _iso_us(e["timestamp"])): e for e in got["events"]}
        for event in span.get("events", []):
            stored_event = got_events.get((event["name"], _ns_to_us(event["timeUnixNano"])))
            if stored_event is None:
                event_lost[f"{event['name']} (whole event)"] += 1
                continue
            for name in _attrs(event.get("attributes")):
                if name not in stored_event["attributes"]:
                    event_lost[f"{event['name']}: {name}"] += 1
        if "links" in got:
            links["rest_api_has_links_field"] = True
        elif links["rest_api_has_links_field"] is None:
            links["rest_api_has_links_field"] = False
        links["sent"] += len(span.get("links", []))
        links["stored"] += len(got.get("links") or [])
        node = gql.get(span_id, {})
        if node.get("tokenCountTotal") or "llm.cost.total" in sent_attrs:
            tokens.append({
                "span": span["name"],
                "status": got["status_code"],
                "prompt": node.get("tokenCountPrompt"),
                "completion": node.get("tokenCountCompletion"),
                "total": node.get("tokenCountTotal"),
                "sent_llm_cost_total": sent_attrs.get("llm.cost.total"),
                "phoenix_cost": ((node.get("costSummary") or {}).get("total") or {}).get("cost"),
            })
        io[(got["span_kind"], bool(node.get("input")), bool(node.get("output")))] += 1
    return {
        "sent": len(sent_spans),
        "stored": len(sent_spans) - len(missing_spans),
        "missing_spans": missing_spans,
        "field_mismatches": dict(mismatched),
        "span_kinds": [
            {"sent_openinference_kind": sent, "phoenix_span_kind": got, "spans": n}
            for (sent, got), n in sorted(kinds.items(), key=str)
        ],
        "attributes_sent": dict(sorted(sent_keys.items())),
        "attributes_not_stored": dict(sorted(lost.items())),
        "attributes_changed": dict(sorted(changed.items())),
        "attributes_added_by_phoenix": dict(sorted(added.items())),
        "event_attributes_not_stored": dict(sorted(event_lost.items())),
        "links": links,
        "tokens_and_cost": tokens,
        "input_output": [
            {"phoenix_span_kind": kind, "input": has_in, "output": has_out, "spans": n}
            for (kind, has_in, has_out), n in sorted(io.items(), key=str)
        ],
    }


def watch_report(posts: list[dict], stored_spans: list[dict], session_id: str) -> dict:
    """Describe each relayed protobuf POST and which root span copy Phoenix kept."""
    report: dict = {"posts": []}
    roots = []
    for number, post in enumerate(posts, 1):
        entry = {"status": post["status"], "content_type": post["content_type"], "path": post["path"]}
        spans = protobuf_spans(post["body"])
        entry["spans"] = len(spans)
        for span in spans:
            if "parent_span_id" not in span:
                roots.append((number, span))
                entry["root"] = {key: span["attributes"].get(key) for key in ROOT_COUNTS}
        report["posts"].append(entry)
    root_ids = {span["span_id"] for _, span in roots}
    kept = [span for span in stored_spans if span["context"]["span_id"] in root_ids]
    report["stored_root_copies"] = len(kept)
    report["stored_root_matches_post"] = [
        number for span in kept for number, sent in roots
        if sent["start_us"] == _iso_us(span["start_time"])
        and sent["end_us"] == _iso_us(span["end_time"])
        and all(sent["attributes"].get(key) == span["attributes"].get(key) for key in ROOT_COUNTS)
    ]
    session = [span for span in stored_spans if span["attributes"].get("session.id") == session_id]
    report["stored"] = {
        "spans": len(session),
        "traces": len({span["context"]["trace_id"] for span in session}),
        "names": dict(sorted(Counter(span["name"] for span in session).items())),
        "status": dict(sorted(Counter(span["status_code"] for span in session).items())),
        "kinds": dict(sorted(Counter(span["span_kind"] for span in session).items())),
    }
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _wait_for(base: str, span_ids: set[str], timeout: float = 90.0) -> list[dict]:
    # Phoenix stores spans a few seconds after it answers 200.
    deadline = time.monotonic() + timeout
    while True:
        stored = {span["context"]["span_id"] for span in rest_spans(base)}
        if span_ids <= stored or time.monotonic() > deadline:
            time.sleep(3)
            return rest_spans(base)
        time.sleep(2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--phoenix", default="http://127.0.0.1:6016",
                        help="Phoenix base URL on a loopback address")
    args = parser.parse_args()
    base = args.phoenix.rstrip("/")
    if urlsplit(base).hostname not in ("127.0.0.1", "localhost", "::1"):
        sys.stderr.write("run.py sends only to a Phoenix on a loopback address\n")
        return 2
    for key in ("AGENT_STRACE_WORKSPACE", "AGENT_STRACE_TENANT_ID"):
        os.environ.pop(key, None)

    report: dict = {
        "agent_strace_version": __version__,
        "phoenix_version": _get(f"{base}/arize_phoenix_version").decode().strip(),
        "python_version": sys.version.split()[0],
        "graphql_span_has_links_field": "links" in {
            field["name"] for field in
            _graphql(base, '{ __type(name: "Span") { fields { name } } }')["__type"]["fields"]
        },
    }
    sent: dict[str, list[dict]] = {}
    with tempfile.TemporaryDirectory(prefix="px560-") as tmp:
        runner = Runner(tmp)
        for group in ("otlp-genai", "otlp", "tree"):
            generate.write_sessions(runner.store, generate.build_sessions(GROUPS[group]))

        # 1. The documented command, unchanged: JSON, which Phoenix refuses.
        genai = GROUPS["otlp-genai"]
        runner.cli("export", f"{genai}-parent", "--format", "otlp-genai", "--endpoint", base)

        # 2. Protobuf through the flag (otlp-genai) and the variable (otlp).
        sent["otlp-genai"], sent["otlp"] = [], []
        for role in ("parent", "child"):
            session_id = f"{genai}-{role}"
            sent["otlp-genai"] += _payload_spans(runner.sent_payload(session_id, "otlp-genai"))
            runner.cli("export", session_id, "--format", "otlp-genai", "--endpoint", base,
                       "--protocol", "http/protobuf")
        for role in ("parent", "child"):
            session_id = f"{GROUPS['otlp']}-{role}"
            sent["otlp"] += _payload_spans(runner.sent_payload(session_id, "otlp"))
            runner.cli("export", session_id, "--format", "otlp", "--endpoint", base,
                       env={"OTEL_EXPORTER_OTLP_TRACES_PROTOCOL": "http/protobuf"})

        # 3. gRPC is refused before anything is read or sent.
        runner.cli("export", f"{genai}-parent", "--format", "otlp-genai", "--endpoint", base,
                   "--protocol", "grpc")

        # 4. The parent and its subagent in one trace.
        tree = f"{GROUPS['tree']}-parent"
        sent["tree"] = _payload_spans(
            runner.sent_payload(tree, "otlp-genai", "--include-subagents"),
        )
        runner.cli("export", tree, "--format", "otlp-genai", "--include-subagents",
                   "--endpoint", base, "--protocol", "http/protobuf")

        # 5. watch --stream-otlp through the relay: protobuf, then the default JSON.
        relay = Relay(base)
        threading.Thread(target=relay.serve_forever, daemon=True).start()
        relay_url = f"http://127.0.0.1:{relay.server_address[1]}"
        try:
            runner.watch(GROUPS["watch-protobuf"], relay_url, ["--protocol", "http/protobuf"])
            protobuf_posts = relay.take()
            runner.watch(GROUPS["watch-json"], relay_url, [])
            json_posts = relay.take()
        finally:
            relay.shutdown()
            relay.server_close()

        report["cli_runs"] = runner.runs
        sent_ids = {span["spanId"] for spans in sent.values() for span in spans}
        stored_spans = _wait_for(base, sent_ids)
        project = graphql_project(base)

    stored = {span["context"]["span_id"]: span for span in stored_spans}
    gql = {edge["node"]["context"]["spanId"]: edge["node"] for edge in project["spans"]["edges"]}
    report["groups"] = {group: compare(spans, stored, gql) for group, spans in sent.items()}
    traces = defaultdict(set)
    for group, spans in sent.items():
        for span in spans:
            if span["spanId"] in stored:
                traces[group].add(stored[span["spanId"]]["context"]["trace_id"])
    report["traces_per_group"] = {group: len(ids) for group, ids in sorted(traces.items())}
    report["watch_protobuf"] = watch_report(
        protobuf_posts, stored_spans, f"{GROUPS['watch-protobuf']}-parent",
    )
    report["watch_json"] = {
        "posts": [{"status": post["status"], "content_type": post["content_type"]}
                  for post in json_posts],
        "stored_spans": sum(
            1 for span in stored_spans
            if span["attributes"].get("session.id") == f"{GROUPS['watch-json']}-parent"
        ),
    }
    report["sessions"] = [
        {
            "session_id": edge["node"]["sessionId"],
            "traces": edge["node"]["numTraces"],
            "tokens": edge["node"]["tokenUsage"],
            "cost": ((edge["node"]["costSummary"] or {}).get("total") or {}).get("cost"),
        }
        for edge in project["sessions"]["edges"]
        if PREFIX in edge["node"]["sessionId"]
    ]
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
