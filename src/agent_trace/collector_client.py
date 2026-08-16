"""Bounded, authenticated read client for an agent-strace collector.

The client deliberately exposes only the existing collector read API.  It
does not infer an organization from tags: one authenticated collector
instance and key are the organization boundary for a remote report.
"""

from __future__ import annotations

import ipaddress
import http.client
import json
import math
import ssl
import urllib.error
import urllib.parse
import urllib.request

from .models import SessionMeta, TraceEvent
from .store import validate_stored_id


MAX_METADATA_BYTES = 16 * 1024 * 1024
MAX_SESSION_EVENT_BYTES = 32 * 1024 * 1024
MAX_EVENT_LINE_BYTES = 2 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024
MAX_SESSIONS = 10_000
MAX_EVENTS_PER_SESSION = 500_000
MAX_JSON_DEPTH = 64

_META_STRING_FIELDS = {
    "agent_name", "command", "parent_session_id", "parent_event_id", "team",
    "workspace_id", "trace_id", "parent_span_id", "trace_flags", "tenant_id",
}
_META_COUNT_FIELDS = {
    "tool_calls", "llm_requests", "errors", "total_tokens", "depth",
}
_EVENT_STRING_FIELDS = {
    "event_id", "session_id", "parent_id", "prev_hash", "tenant_id",
}


class CollectorClientError(RuntimeError):
    """Raised when a collector cannot provide one complete trusted snapshot."""


class CollectorAuthenticationError(CollectorClientError):
    """Raised when collector authentication is absent or rejected."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r} is not allowed")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _strict_json_loads(text: str):  # noqa: ANN201
    try:
        value = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except RecursionError as exc:
        raise ValueError("collector JSON nesting exceeds the limit") from exc
    stack = [(value, 0)]
    while stack:
        item, depth = stack.pop()
        if depth > MAX_JSON_DEPTH:
            raise ValueError("collector JSON nesting exceeds the limit")
        if isinstance(item, dict):
            stack.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            stack.extend((child, depth + 1) for child in item)
    return value


def _finite_number(value, field: str, *, optional: bool = False) -> float | None:  # noqa: ANN001
    if optional and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{field} must be a finite non-negative number") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{field} must be a finite non-negative number")
    return number


def _validate_meta_item(item: object) -> SessionMeta:
    if not isinstance(item, dict):
        raise ValueError("session metadata must be an object")
    if not isinstance(item.get("session_id"), str) or not item["session_id"]:
        raise ValueError("session_id must be a non-empty string")
    if "started_at" not in item:
        raise ValueError("started_at is required")
    _finite_number(item["started_at"], "started_at")
    if "ended_at" in item:
        _finite_number(item["ended_at"], "ended_at", optional=True)
    if "total_duration_ms" in item:
        _finite_number(item["total_duration_ms"], "total_duration_ms")
    for field in _META_STRING_FIELDS:
        if field in item and not isinstance(item[field], str):
            raise ValueError(f"{field} must be a string")
    for field in _META_COUNT_FIELDS:
        value = item.get(field, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field} must be a non-negative integer")
    if "redacted" in item and not isinstance(item["redacted"], bool):
        raise ValueError("redacted must be a boolean")
    if "attribution" in item and not isinstance(item["attribution"], dict):
        raise ValueError("attribution must be an object")
    if item.get("parent_session_id"):
        validate_stored_id(item["parent_session_id"], "collector parent session ID")
    # Serializing the already strictly-decoded object avoids the permissive
    # json.loads path in the compatibility model constructor.
    meta = SessionMeta.from_json(json.dumps(item, allow_nan=False))
    meta.session_id = validate_stored_id(meta.session_id, "collector session ID")
    return meta


def _validate_event_item(item: object, meta: SessionMeta) -> TraceEvent:
    if not isinstance(item, dict):
        raise ValueError("event must be an object")
    if not isinstance(item.get("event_type"), str):
        raise ValueError("event_type must be a string")
    if "timestamp" not in item:
        raise ValueError("timestamp is required")
    _finite_number(item["timestamp"], "timestamp")
    for field in _EVENT_STRING_FIELDS:
        if field not in item and field in {"event_id", "session_id"}:
            raise ValueError(f"{field} is required")
        if field in item and not isinstance(item[field], str):
            raise ValueError(f"{field} must be a string")
    if not item.get("event_id") or not item.get("session_id"):
        raise ValueError("event_id and session_id must be non-empty")
    validate_stored_id(item["session_id"], "collector event session ID")
    if "duration_ms" in item:
        _finite_number(item["duration_ms"], "duration_ms", optional=True)
    if "data" in item and not isinstance(item["data"], dict):
        raise ValueError("event data must be an object")
    if "redacted" in item and not isinstance(item["redacted"], bool):
        raise ValueError("event redacted must be a boolean")
    event = TraceEvent.from_json(json.dumps(item, allow_nan=False))
    if event.session_id != meta.session_id:
        raise ValueError("event session ID does not match its metadata")
    return event


def _is_loopback(hostname: str) -> bool:
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def validate_collector_endpoint(
    endpoint: str, *, allow_insecure_http: bool = False,
) -> str:
    """Validate and normalize an explicit collector base URL."""
    value = str(endpoint)
    if not value or value != value.strip() or any(ord(char) < 32 for char in value):
        raise ValueError("collector endpoint must be a non-empty URL without controls")
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("collector endpoint must use https or http")
    if not parsed.hostname:
        raise ValueError("collector endpoint must include a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("collector endpoint cannot include user information")
    # urlsplit cannot distinguish an absent delimiter from a present empty
    # query/fragment. Reject the delimiters themselves before credentials are
    # ever attached to a derived URL.
    if "?" in value or "#" in value or parsed.query or parsed.fragment:
        raise ValueError("collector endpoint cannot include a query or fragment")
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError("collector endpoint has an invalid port") from exc
    if (
        parsed.scheme == "http"
        and not allow_insecure_http
        and not _is_loopback(parsed.hostname)
    ):
        raise ValueError(
            "plain HTTP is allowed only for loopback collectors; "
            "use --allow-insecure-http to opt in"
        )
    return value.rstrip("/")


class CollectorClient:
    """Read one complete, size-bounded snapshot from a collector instance."""

    def __init__(
        self,
        endpoint: str,
        auth_key: str,
        *,
        allow_insecure_http: bool = False,
        ca_file: str = "",
        timeout: float = 15.0,
        max_metadata_bytes: int = MAX_METADATA_BYTES,
        max_session_event_bytes: int = MAX_SESSION_EVENT_BYTES,
        max_event_line_bytes: int = MAX_EVENT_LINE_BYTES,
        max_total_bytes: int = MAX_TOTAL_BYTES,
        max_sessions: int = MAX_SESSIONS,
        max_events_per_session: int = MAX_EVENTS_PER_SESSION,
        opener=None,
    ):
        self.endpoint = validate_collector_endpoint(
            endpoint, allow_insecure_http=allow_insecure_http
        )
        key = str(auth_key).strip()
        if not key or any(char in key for char in "\r\n"):
            raise CollectorAuthenticationError(
                "remote collector org-report requires a single-line auth key"
            )
        self._auth_key = key
        self.timeout = float(timeout)
        self.max_metadata_bytes = int(max_metadata_bytes)
        self.max_session_event_bytes = int(max_session_event_bytes)
        self.max_event_line_bytes = int(max_event_line_bytes)
        self.max_total_bytes = int(max_total_bytes)
        self.max_sessions = int(max_sessions)
        self.max_events_per_session = int(max_events_per_session)
        self._bytes_read = 0
        if (
            not math.isfinite(self.timeout)
            or self.timeout <= 0
            or min(
                self.max_metadata_bytes,
                self.max_session_event_bytes,
                self.max_event_line_bytes,
                self.max_total_bytes,
                self.max_sessions,
                self.max_events_per_session,
            ) <= 0
        ):
            raise ValueError("collector timeouts and limits must be positive")

        if opener is not None:
            self._opener = opener
        else:
            # Disable ambient proxy configuration. Otherwise even the explicit
            # loopback HTTP exception can send the bearer token to http_proxy.
            handlers: list[urllib.request.BaseHandler] = [
                urllib.request.ProxyHandler({}),
                _NoRedirect(),
            ]
            if urllib.parse.urlsplit(self.endpoint).scheme == "https":
                try:
                    context = ssl.create_default_context(cafile=ca_file or None)
                except (OSError, ssl.SSLError) as exc:
                    raise CollectorClientError(
                        f"could not load collector CA file: {exc}"
                    ) from exc
                handlers.append(urllib.request.HTTPSHandler(context=context))
            self._opener = urllib.request.build_opener(*handlers)

    @property
    def bytes_read(self) -> int:
        return self._bytes_read

    def _url(self, path: str) -> str:
        return self.endpoint + path

    def _read_bounded(self, response, limit: int) -> bytes:  # noqa: ANN001
        header = response.headers.get("Content-Length", "")
        declared: int | None = None
        if header:
            try:
                declared = int(header)
            except ValueError as exc:
                raise CollectorClientError(
                    "collector returned an invalid Content-Length"
                ) from exc
            if declared < 0 or declared > limit:
                raise CollectorClientError("collector response exceeds the size limit")
            if self._bytes_read + declared > self.max_total_bytes:
                raise CollectorClientError("collector snapshot exceeds the total size limit")
        remaining = self.max_total_bytes - self._bytes_read
        allowed = min(limit, remaining)
        if allowed < 0:
            raise CollectorClientError("collector snapshot exceeds the total size limit")
        try:
            body = response.read(allowed + 1)
        except (http.client.HTTPException, OSError) as exc:
            raise CollectorClientError("collector response body was truncated") from exc
        if len(body) > allowed:
            raise CollectorClientError("collector response exceeds the size limit")
        if declared is not None and len(body) != declared:
            raise CollectorClientError("collector response body was truncated")
        self._bytes_read += len(body)
        return body

    def _get(self, path: str, *, authenticated: bool, limit: int) -> bytes:
        headers = {"Accept": "application/json, application/x-ndjson"}
        if authenticated:
            headers["Authorization"] = f"Bearer {self._auth_key}"
        request = urllib.request.Request(
            self._url(path), headers=headers, method="GET"
        )
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                status = getattr(response, "status", None)
                if status is None:
                    status = response.getcode()
                if status != 200:
                    raise CollectorClientError(
                        f"collector returned unexpected HTTP status {status}"
                    )
                return self._read_bounded(response, limit)
        except urllib.error.HTTPError as exc:
            if exc.code in {401, 403}:
                raise CollectorAuthenticationError(
                    "collector authentication was required or rejected"
                ) from exc
            if 300 <= exc.code < 400:
                raise CollectorClientError("collector redirects are not allowed") from exc
            raise CollectorClientError(
                f"collector request failed with HTTP {exc.code}"
            ) from exc
        except (
            urllib.error.URLError,
            TimeoutError,
            http.client.HTTPException,
            OSError,
        ) as exc:
            raise CollectorClientError(f"collector request failed: {exc}") from exc

    def list_sessions(self) -> list[SessionMeta]:
        """Authenticate and return all metadata visible to this collector key."""
        # Refuse an instance that exposes session metadata without the key.  A
        # successful report must be an authenticated snapshot, not merely a
        # request that happened to include an ignored Authorization header.
        try:
            self._get("/sessions", authenticated=False, limit=self.max_metadata_bytes)
        except CollectorAuthenticationError:
            pass
        else:
            raise CollectorAuthenticationError(
                "collector does not enforce authentication for session reads"
            )

        raw = self._get(
            "/sessions", authenticated=True, limit=self.max_metadata_bytes
        )
        try:
            payload = _strict_json_loads(raw.decode("utf-8", errors="strict"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise CollectorClientError("collector returned malformed session metadata") from exc
        if not isinstance(payload, list):
            raise CollectorClientError("collector session response must be a JSON array")
        if len(payload) > self.max_sessions:
            raise CollectorClientError("collector session count exceeds the limit")

        sessions: list[SessionMeta] = []
        seen: set[str] = set()
        for item in payload:
            try:
                meta = _validate_meta_item(item)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise CollectorClientError("collector returned invalid session metadata") from exc
            if meta.session_id in seen:
                raise CollectorClientError("collector returned a duplicate session ID")
            seen.add(meta.session_id)
            sessions.append(meta)
        return sorted(
            sessions,
            key=lambda meta: (meta.started_at, meta.session_id),
            reverse=True,
        )

    def load_events(self, meta: SessionMeta) -> list[TraceEvent]:
        """Load and strictly validate one session's NDJSON event stream."""
        encoded_id = urllib.parse.quote(meta.session_id, safe="")
        raw = self._get(
            f"/sessions/{encoded_id}/events",
            authenticated=True,
            limit=self.max_session_event_bytes,
        )
        try:
            text = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise CollectorClientError("collector returned non-UTF-8 event data") from exc
        events: list[TraceEvent] = []
        stream_tenant = ""
        for line in text.splitlines():
            if not line.strip():
                continue
            if len(line.encode("utf-8")) > self.max_event_line_bytes:
                raise CollectorClientError("collector event line exceeds the size limit")
            if len(events) >= self.max_events_per_session:
                raise CollectorClientError("collector event count exceeds the limit")
            try:
                event = _validate_event_item(_strict_json_loads(line), meta)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise CollectorClientError("collector returned malformed event data") from exc
            if meta.tenant_id and event.tenant_id and event.tenant_id != meta.tenant_id:
                raise CollectorClientError(
                    "collector event tenant does not match its metadata"
                )
            if meta.tenant_id:
                event.tenant_id = meta.tenant_id
            elif event.tenant_id:
                if stream_tenant and event.tenant_id != stream_tenant:
                    raise CollectorClientError(
                        "collector event stream contains mixed tenants"
                    )
                stream_tenant = event.tenant_id
            events.append(event)
        return events


class CollectorTraceStore:
    """Read-only TraceStore-compatible view backed by a CollectorClient."""

    def __init__(self, client: CollectorClient, sessions: list[SessionMeta]):
        self.client = client
        self.workspace_id = ""
        self._metas = {meta.session_id: meta for meta in sessions}
        self._events: dict[str, list[TraceEvent]] = {}

    @classmethod
    def load(cls, client: CollectorClient) -> "CollectorTraceStore":
        return cls(client, client.list_sessions())

    def list_sessions_strict(
        self,
        tenant_id: str | None = None,
        *,
        validate_events: bool = True,
    ) -> list[SessionMeta]:
        # Metadata is already strictly validated. Event loading remains lazy so
        # org-report can apply its UTC month and team filters first.
        del validate_events
        metas = list(self._metas.values())
        if tenant_id is not None:
            metas = [meta for meta in metas if meta.tenant_id == tenant_id]
        return sorted(
            metas,
            key=lambda meta: (meta.started_at, meta.session_id),
            reverse=True,
        )

    def load_meta(self, session_id: str) -> SessionMeta:
        try:
            return self._metas[session_id]
        except KeyError as exc:
            raise FileNotFoundError(f"collector session not found: {session_id}") from exc

    def load_events(self, session_id: str) -> list[TraceEvent]:
        if session_id not in self._events:
            self._events[session_id] = self.client.load_events(
                self.load_meta(session_id)
            )
        return list(self._events[session_id])

    def release_events(self, session_id: str) -> None:
        """Release parsed event objects after all per-session analyses finish."""
        self._events.pop(session_id, None)
