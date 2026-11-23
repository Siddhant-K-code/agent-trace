#!/usr/bin/env python3
"""Send agent-strace traces through otelcol-contrib's azuremonitorexporter.

This is the #586 check. It needs no Azure account. The exporter's connection
string points its ``IngestionEndpoint`` at a server on 127.0.0.1 that records
what Application Insights would receive. Download otelcol-contrib first (see
README.md), then run:

    python3 experiments/azure-monitor/run.py --otelcol /path/to/otelcol-contrib

The script:

1. starts a capture server that unzips each POST to ``.../v2.1/track`` and
   answers the way the ingestion endpoint does (200, all items accepted);
2. starts the collector with ``collector.yaml``: an OTLP/HTTP receiver and
   two azure_monitor exporters, one with the defaults and one with
   ``spaneventsenabled: true``, each with its own capture path;
3. writes synthetic sessions into a temporary trace store and sends them with
   the real CLI over OTLP/HTTP protobuf:

   - the #241 parent and its subagent as one ``otlp-genai`` trace, with
     ``--agent-id``;
   - the same pair in the classic ``otlp`` format, with
     ``AGENT_STRACE_AGENT_ID`` instead of the flag;
   - a ten-turn session whose ``gen_ai.input.messages`` is longer than
     8192 bytes;
   - a session sent without an agent ID;
   - a growing session streamed with ``watch --stream-otlp --agent-id``;

4. compares each span sent (the same export written with ``--output``) with
   the envelope the exporter built for it.

It prints a JSON report on stdout, with temporary paths, ports, the user name
and the host name replaced. The instrumentation key is the all-zero fake one.
"""

from __future__ import annotations

import argparse
import getpass
import gzip
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter, defaultdict
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments" / "telemetry-dev"))

import generate  # noqa: E402
from agent_trace import __version__  # noqa: E402
from agent_trace.models import EventType, SessionMeta, TraceEvent  # noqa: E402
from agent_trace.store import TraceStore  # noqa: E402

PREFIX = "am586"
AGENT_ID = "demo-agent-v1"
FAKE_IKEY = "00000000-0000-0000-0000-000000000000"
CAPTURES = ("default", "spanevents")
SPAN_KINDS = {0: "UNSPECIFIED", 1: "INTERNAL", 2: "SERVER", 3: "CLIENT", 4: "PRODUCER", 5: "CONSUMER"}
SPAN_DATA = {"RequestData", "RemoteDependencyData"}


# ---------------------------------------------------------------------------
# Capture server: stands in for the Application Insights ingestion endpoint
# ---------------------------------------------------------------------------

class _CaptureHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        encoding = self.headers.get("Content-Encoding", "")
        body = gzip.decompress(raw) if encoding == "gzip" else raw
        envelopes = [json.loads(line) for line in body.decode("utf-8").splitlines() if line.strip()]
        with self.server.lock:
            self.server.posts.append({
                "path": self.path,
                "content_type": self.headers.get("Content-Type"),
                "content_encoding": encoding,
                "bytes_sent": len(raw),
                "bytes_unzipped": len(body),
                "envelopes": envelopes,
            })
        reply = json.dumps({
            "itemsReceived": len(envelopes), "itemsAccepted": len(envelopes), "errors": [],
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, *args) -> None:
        pass


class Capture(ThreadingHTTPServer):
    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _CaptureHandler)
        self.posts: list[dict] = []
        self.lock = threading.Lock()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def envelopes(self, capture: str) -> list[dict]:
        with self.lock:
            return [
                envelope for post in self.posts if post["path"].startswith(f"/{capture}/")
                for envelope in post["envelopes"]
            ]


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Collector:
    def __init__(self, binary: str, tmp: str, capture_url: str) -> None:
        self.port = _free_port()
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": tmp,
            # The exporter writes envelope times in the local time zone.
            "TZ": "UTC",
            "OTLP_HTTP_ENDPOINT": f"127.0.0.1:{self.port}",
        }
        for capture in CAPTURES:
            env[f"CONNECTION_STRING_{capture.upper()}"] = (
                f"InstrumentationKey={FAKE_IKEY};IngestionEndpoint={capture_url}/{capture}/"
            )
        self.log_path = os.path.join(tmp, "otelcol.log")
        self._log = open(self.log_path, "w", encoding="utf-8")
        self.proc = subprocess.Popen(
            [binary, "--config", str(HERE / "collector.yaml")],
            env=env, stdout=self._log, stderr=subprocess.STDOUT, cwd=tmp,
        )
        deadline = time.monotonic() + 30
        while True:
            try:
                socket.create_connection(("127.0.0.1", self.port), timeout=1).close()
                return
            except OSError:
                if self.proc.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(f"otelcol-contrib did not start; see {self.log_path}")
                time.sleep(0.2)

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def stop(self) -> int:
        self.proc.terminate()
        code = self.proc.wait(timeout=60)
        self._log.close()
        return code

    def warnings(self) -> list[str]:
        """Return each warning or error as level, message and component.

        The time and the collector's random service.instance.id are left out.
        """
        lines = []
        with open(self.log_path, encoding="utf-8") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4 or parts[1] not in ("warn", "error"):
                    continue
                component = ""
                if len(parts) > 4:
                    try:
                        component = json.loads(parts[4]).get("otelcol.component.id", "")
                    except ValueError:
                        pass
                lines.append(f"{parts[1]}: {parts[3]} [{component}]")
        return lines


def otelcol_version(binary: str) -> dict:
    with open(binary, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    out = subprocess.run([binary, "--version"], capture_output=True, text=True, check=True).stdout
    return {"version": out.strip().rsplit(" ", 1)[-1], "sha256": digest}


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

def long_session(prefix: str) -> tuple[SessionMeta, list[TraceEvent]]:
    """Ten prompts and replies, each cut to 1000 characters by agent-strace.

    The prompts are non-ASCII text, which the JSON in ``gen_ai.input.messages``
    escapes as six bytes a character. The tool argument and result are longer
    than agent-strace's own limits.
    """
    session_id = f"{prefix}-parent"
    start = generate.BASE
    meta = SessionMeta(
        session_id=session_id, started_at=start, ended_at=start + 12,
        agent_name="claude-code", command="claude", tool_calls=1,
        conversation_id=f"synthetic-conversation-{prefix}", tenant_id="",
    )
    events = [TraceEvent(
        event_type=EventType.SESSION_START, timestamp=start, event_id=f"{session_id}-start",
        session_id=session_id, data={"provider": "claude-code"},
    )]
    t = start
    for turn in range(10):
        t += 1
        prompt = f"turn {turn}: " + "é" * 1100
        events.append(TraceEvent(
            event_type=EventType.USER_PROMPT, timestamp=t, event_id=f"{session_id}-p{turn}",
            session_id=session_id, data={"prompt": prompt},
        ))
        events.append(TraceEvent(
            event_type=EventType.ASSISTANT_RESPONSE, timestamp=t + 0.5,
            event_id=f"{session_id}-r{turn}", session_id=session_id,
            data={"text": f"reply {turn}: " + "ok " * 400},
        ))
    events.append(TraceEvent(
        event_type=EventType.TOOL_CALL, timestamp=t + 0.6, event_id=f"{session_id}-tool",
        session_id=session_id,
        data={"tool_name": "Bash", "arguments": {"command": "echo " + "a" * 5000}},
    ))
    events.append(TraceEvent(
        event_type=EventType.TOOL_RESULT, timestamp=t + 0.8, event_id=f"{session_id}-tool-result",
        session_id=session_id, parent_id=f"{session_id}-tool", duration_ms=200.0,
        data={"tool_name": "Bash", "result": "b" * 5000},
    ))
    events.append(TraceEvent(
        event_type=EventType.SESSION_END, timestamp=t + 1, event_id=f"{session_id}-end",
        session_id=session_id, data={"reason": "complete"},
    ))
    return meta, events


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
        self.replace: dict[str, str] = {tmp: "<tmpdir>"}

    def scrub(self, text: str) -> str:
        for old, new in self.replace.items():
            text = text.replace(old, new)
        return text

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
            "command": self.scrub(" ".join(["agent-strace", *args])),
            "env": env or {},
            "exit": proc.returncode,
            "stdout": self.scrub(proc.stdout),
            "stderr": self.scrub(proc.stderr),
        }
        if record:
            self.runs.append(run)
        return run

    def sent_payload(self, session_id: str, *flags: str, env: dict | None = None) -> dict:
        """Return the JSON payload the CLI builds for an export (the --output file)."""
        out = os.path.join(self.tmp, f"sent-{len(os.listdir(self.tmp))}.json")
        run = self.cli("export", session_id, *flags, "--output", out, env=env, record=False)
        if run["exit"] != 0:
            raise RuntimeError(run["stderr"])
        with open(out, encoding="utf-8") as handle:
            return json.load(handle)

    def send(self, session_id: str, endpoint: str, *flags: str, env: dict | None = None) -> list[dict]:
        """Send an export over OTLP/HTTP protobuf and return the spans it holds."""
        spans = _payload_spans(self.sent_payload(session_id, *flags, env=env))
        run = self.cli("export", session_id, *flags, "--endpoint", endpoint,
                       "--protocol", "http/protobuf", env=env)
        if run["exit"] != 0:
            raise RuntimeError(run["stderr"])
        return spans

    def watch(self, prefix: str, endpoint: str) -> str:
        """Stream a session that grows while watch runs."""
        meta, events, _, _ = generate.build_sessions(prefix)
        session_id = meta.session_id
        self.store.create_session(meta)
        self.store.append_events(session_id, events[:1])
        args = [
            "watch", session_id, "--stream-to", endpoint, "--stream-otlp",
            "--protocol", "http/protobuf", "--agent-id", AGENT_ID,
            "--stream-flush-interval", "1",
        ]
        proc = subprocess.Popen(
            self._argv(args), cwd=self.home, env=self._env(None),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        time.sleep(4)
        self.store.append_events(session_id, events[1:])
        _, err = proc.communicate(timeout=180)
        self.runs.append({
            "command": self.scrub(" ".join(["agent-strace", *args])),
            "env": {},
            "exit": proc.returncode,
            "stderr": self.scrub(err),
            "events": len(events),
        })
        return session_id


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


def _payload_spans(payload: dict) -> list[dict]:
    spans = []
    for resource_spans in payload["resourceSpans"]:
        resource = _attrs(resource_spans.get("resource", {}).get("attributes"))
        for scope_spans in resource_spans["scopeSpans"]:
            for span in scope_spans["spans"]:
                spans.append({**span, "_resource": resource, "_scope": scope_spans.get("scope", {})})
    return spans


def _as_property(value) -> str | None:
    """Return the string the exporter writes for an attribute, or None if it drops it."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format(Decimal(repr(value)), "f")
    if isinstance(value, str):
        return value
    return None


def _type_name(value) -> str:
    return {list: "array", dict: "map", bool: "bool", int: "int", float: "double", str: "string"}.get(
        type(value), type(value).__name__,
    )


def _changed_value(key: str, sent: str, got: str, resource: dict) -> dict:
    """Describe a property whose value is not the attribute value sent."""
    entry: dict = {"attribute": key}
    if len(sent) <= 100 and len(got) <= 100:
        entry.update({"sent": sent, "stored": got})
    entry.update({
        "sent_bytes": len(sent.encode("utf-8")),
        "stored_bytes": len(got.encode("utf-8")),
        "stored_is_prefix_of_sent": sent.startswith(got),
        "stored_is_the_resource_attribute": key in resource and _as_property(resource[key]) == got,
    })
    for name, text in (("sent", sent), ("stored", got)):
        try:
            json.loads(text)
            entry[f"{name}_is_json"] = True
        except ValueError:
            entry[f"{name}_is_json"] = False
    if key in ("gen_ai.input.messages", "gen_ai.output.messages"):
        entry["messages_sent"] = len(json.loads(sent))
        entry["messages_begun_in_stored"] = got.count('"role"')
    return entry


def _index(envelopes: list[dict]) -> tuple[dict, dict]:
    spans: dict[tuple[str, str], list[dict]] = defaultdict(list)
    events: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for envelope in envelopes:
        data = envelope["data"]
        operation = envelope["tags"].get("ai.operation.id", "")
        if data["baseType"] in SPAN_DATA:
            spans[(operation, data["baseData"]["id"])].append(envelope)
        else:
            events[(operation, envelope["tags"].get("ai.operation.parentId", ""))].append(envelope)
    return spans, events


def compare(sent_spans: list[dict], capture: Capture) -> dict:
    """Compare the spans sent with the envelopes each exporter produced."""
    spans_default, events_default = _index(capture.envelopes("default"))
    _, events_spanevents = _index(capture.envelopes("spanevents"))
    mapping = Counter()
    kept, changed, dropped, added = Counter(), Counter(), Counter(), Counter()
    values_changed, links, agent_ids, missing = [], [], Counter(), []
    event_fates: dict[str, Counter] = {capture_name: Counter() for capture_name in CAPTURES}
    event_agent_id = Counter()
    for span in sent_spans:
        key = (span["traceId"], span["spanId"])
        found = spans_default.get(key, [])
        if not found:
            missing.append(span["name"])
            continue
        envelope = found[-1]
        data = envelope["data"]["baseData"]
        properties = data.get("properties", {})
        mapping[(
            span["name"].split(" ")[0], SPAN_KINDS[span.get("kind", 0)],
            envelope["data"]["baseType"], data.get("type", ""),
        )] += 1
        sent = _attrs(span.get("attributes"))
        agent_ids[properties.get("gen_ai.agent.id", "<absent>")] += 1
        for name, value in sent.items():
            expected = _as_property(value)
            if expected is None:
                dropped[f"{name} ({_type_name(value)})"] += 1
            elif name not in properties:
                dropped[f"{name} ({_type_name(value)})"] += 1
            elif properties[name] == expected:
                kept[name] += 1
            else:
                changed[name] += 1
                values_changed.append(
                    _changed_value(name, expected, properties[name], span["_resource"]),
                )
        for name in properties:
            if name not in sent:
                added[name] += 1
        sent_links = span.get("links", [])
        if sent_links or "_MS.links" in properties:
            stored = json.loads(properties.get("_MS.links", "[]"))
            links.append({
                "span": span["name"],
                "sent": [{"trace_id": link["traceId"], "span_id": link["spanId"],
                          "attributes": sorted(_attrs(link.get("attributes")))}
                         for link in sent_links],
                "_MS.links": stored,
                "same_targets": sorted((link["traceId"], link["spanId"]) for link in sent_links)
                == sorted((link.get("operation_Id"), link.get("id")) for link in stored),
            })
        for capture_name, index in (("default", events_default), ("spanevents", events_spanevents)):
            got = Counter(envelope["data"]["baseType"] for envelope in index.get(key, []))
            for event in span.get("events", []):
                base = "ExceptionData" if event["name"] == "exception" else "MessageData"
                if got[base] > 0:
                    got[base] -= 1
                    event_fates[capture_name][f"{event['name']} -> {base}"] += 1
                else:
                    event_fates[capture_name][f"{event['name']} -> dropped"] += 1
            for envelope in index.get(key, []):
                props = envelope["data"]["baseData"].get("properties", {})
                event_agent_id[(
                    capture_name, envelope["data"]["baseType"],
                    "gen_ai.agent.id" in props,
                )] += 1
    return {
        "spans_sent": len(sent_spans),
        "spans_with_an_envelope": len(sent_spans) - len(missing),
        "spans_without_an_envelope": missing,
        "span_mapping": [
            {"span": name, "kind": kind, "baseType": base, "type": dep_type, "spans": count}
            for (name, kind, base, dep_type), count in sorted(mapping.items())
        ],
        "gen_ai.agent.id_on_span_envelopes": dict(agent_ids),
        "gen_ai.agent.id_on_event_envelopes": [
            {"exporter": capture_name, "baseType": base, "has_agent_id": has, "envelopes": count}
            for (capture_name, base, has), count in sorted(event_agent_id.items())
        ],
        "attributes_kept": dict(sorted(kept.items())),
        "attributes_dropped": dict(sorted(dropped.items())),
        "attributes_changed": dict(sorted(changed.items())),
        "values_changed": values_changed,
        "properties_added_by_exporter": dict(sorted(added.items())),
        "links": links,
        "span_events": {name: dict(sorted(fates.items())) for name, fates in event_fates.items()},
    }


def watch_report(capture: Capture, session_id: str) -> dict:
    span_envelopes = [
        envelope for envelope in capture.envelopes("default")
        if envelope["data"]["baseType"] in SPAN_DATA
        and envelope["data"]["baseData"].get("properties", {}).get("session.id") == session_id
    ]
    ids = Counter(envelope["data"]["baseData"]["id"] for envelope in span_envelopes)
    roots = [envelope for envelope in span_envelopes if not envelope["tags"].get("ai.operation.parentId")]
    return {
        "span_envelopes": len(span_envelopes),
        "distinct_span_ids": len(ids),
        "span_ids_sent_more_than_once": sum(1 for count in ids.values() if count > 1),
        "root_envelopes": len(roots),
        "names": dict(sorted(Counter(
            envelope["data"]["baseData"]["name"].split(" ")[0] for envelope in span_envelopes
        ).items())),
        "gen_ai.agent.id": dict(Counter(
            envelope["data"]["baseData"]["properties"].get("gen_ai.agent.id", "<absent>")
            for envelope in span_envelopes
        )),
    }


def _short(value, limit: int = 160):
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + f"... ({len(value)} characters)"
    if isinstance(value, dict):
        return {key: _short(item, limit) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [_short(item, limit) for item in value]
    return value


def samples(capture: Capture, trace_id: str) -> dict:
    """One envelope of each kind from the otlp-genai tree, as the exporter sent it."""
    wanted = {
        "invoke_agent root": lambda e: e["data"]["baseData"].get("name", "").startswith("invoke_agent")
        and not e["tags"].get("ai.operation.parentId"),
        "chat": lambda e: e["data"]["baseData"].get("name", "").startswith("chat"),
        "execute_tool": lambda e: e["data"]["baseData"].get("name", "").startswith("execute_tool"),
        "exception": lambda e: e["data"]["baseType"] == "ExceptionData",
    }
    found = {}
    for envelope in capture.envelopes("default"):
        if envelope["tags"].get("ai.operation.id") != trace_id:
            continue
        for label, match in wanted.items():
            if label not in found and match(envelope):
                found[label] = _short(envelope)
    for envelope in capture.envelopes("spanevents"):
        if (envelope["tags"].get("ai.operation.id") == trace_id
                and envelope["data"]["baseType"] == "MessageData"):
            found["span event (spaneventsenabled)"] = _short(envelope)
            break
    return found


def transport(capture: Capture) -> dict:
    with capture.lock:
        posts = list(capture.posts)
    return {
        "posts": len(posts),
        "paths": sorted({post["path"] for post in posts}),
        "content_types": sorted({post["content_type"] for post in posts}),
        "content_encodings": sorted({post["content_encoding"] for post in posts}),
        "envelopes": {name: len(capture.envelopes(name)) for name in CAPTURES},
        "envelopes_by_baseType": {
            name: dict(sorted(Counter(e["data"]["baseType"] for e in capture.envelopes(name)).items()))
            for name in CAPTURES
        },
        "envelope_names": dict(sorted(Counter(e["name"] for e in capture.envelopes("default")).items())),
        "envelope_tags": sorted({tag for e in capture.envelopes("default") for tag in e["tags"]}),
        "iKeys": sorted({e.get("iKey", "") for e in capture.envelopes("default")}),
        "largest_post_bytes_sent": max((post["bytes_sent"] for post in posts), default=0),
        "largest_post_bytes_unzipped": max((post["bytes_unzipped"] for post in posts), default=0),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _wait_for(capture: Capture, expected: set[tuple[str, str]], timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        spans, _ = _index(capture.envelopes("default"))
        if expected <= set(spans):
            break
        time.sleep(1)
    time.sleep(3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--otelcol", required=True, help="path to the otelcol-contrib binary")
    args = parser.parse_args()
    for key in ("AGENT_STRACE_WORKSPACE", "AGENT_STRACE_TENANT_ID", "AGENT_STRACE_AGENT_ID"):
        os.environ.pop(key, None)

    report: dict = {
        "agent_strace_version": __version__,
        "otelcol_contrib": otelcol_version(args.otelcol),
        "python_version": sys.version.split()[0],
        "agent_id": AGENT_ID,
    }
    groups = {name: f"{PREFIX}-{name}" for name in ("genai", "otlp", "long", "noid", "watch")}
    sent: dict[str, list[dict]] = {}
    capture = Capture()
    threading.Thread(target=capture.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory(prefix=f"{PREFIX}-") as tmp:
        runner = Runner(tmp)
        collector = Collector(args.otelcol, tmp, capture.url)
        runner.replace.update({
            collector.endpoint: "<otlp-receiver>",
            capture.url: "<capture>",
            getpass.getuser(): "<user>",
            socket.gethostname(): "<host>",
        })
        try:
            for name in ("genai", "otlp", "noid"):
                generate.write_sessions(runner.store, generate.build_sessions(groups[name]))
            meta, events = long_session(groups["long"])
            runner.store.create_session(meta)
            runner.store.append_events(meta.session_id, events)

            endpoint = collector.endpoint
            sent["otlp-genai tree, --agent-id"] = runner.send(
                f"{groups['genai']}-parent", endpoint, "--format", "otlp-genai",
                "--include-subagents", "--agent-id", AGENT_ID,
            )
            sent["otlp tree, AGENT_STRACE_AGENT_ID"] = runner.send(
                f"{groups['otlp']}-parent", endpoint, "--format", "otlp", "--include-subagents",
                env={"AGENT_STRACE_AGENT_ID": AGENT_ID},
            )
            sent["otlp-genai long session, --agent-id"] = runner.send(
                f"{groups['long']}-parent", endpoint, "--format", "otlp-genai",
                "--agent-id", AGENT_ID,
            )
            sent["otlp-genai tree, no agent ID"] = runner.send(
                f"{groups['noid']}-parent", endpoint, "--format", "otlp-genai",
                "--include-subagents",
            )
            watch_session = runner.watch(groups["watch"], endpoint)

            _wait_for(capture, {
                (span["traceId"], span["spanId"]) for spans in sent.values() for span in spans
            })
        finally:
            report["collector_exit"] = collector.stop()
            report["collector_warnings"] = [runner.scrub(line) for line in collector.warnings()]
            capture.shutdown()
            capture.server_close()
        report["cli_runs"] = runner.runs

    report["transport"] = transport(capture)
    report["exports"] = {name: compare(spans, capture) for name, spans in sent.items()}
    report["watch --stream-otlp --agent-id"] = watch_report(capture, watch_session)
    genai_trace = sent["otlp-genai tree, --agent-id"][0]["traceId"]
    report["sample_envelopes"] = samples(capture, genai_trace)

    text = runner.scrub(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
