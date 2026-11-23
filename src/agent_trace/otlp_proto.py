"""OTLP/HTTP protocol choice and protobuf encoding. Standard library only.

The exporters build one OTLP JSON payload, a dict. :func:`encode_body` sends
it as that JSON or, for ``http/protobuf``, as the protobuf bytes that
:func:`encode_trace_request` makes from the same dict, so the two encodings
carry the same data. Some collectors take only one: Phoenix's ``/v1/traces``
takes only ``application/x-protobuf``.

:func:`encode_trace_request` reads the dict by the OTLP JSON rules
(https://opentelemetry.io/docs/specs/otlp/#json-protobuf-encoding):

- ``traceId``, ``spanId`` and ``parentSpanId`` are hex; ``bytesValue`` is
  base64.
- 64-bit integers may be JSON strings or numbers.
- Enums may be integers or their names.
- Keys may be lowerCamelCase or the proto field names.
- A null value is the field's default.

A field with its default value is left out, except in the ``AnyValue``
oneof, where the value is always written so the receiver sees which kind it
is. A key the schema does not have raises :class:`EncodeError`, so a field an
exporter adds cannot be dropped from the protobuf body without a test noticing.

Field numbers come from opentelemetry-proto v1.11.1:
``collector/trace/v1/trace_service.proto``, ``trace/v1/trace.proto``,
``common/v1/common.proto`` and ``resource/v1/resource.proto``. The
profiling-only ``string_value_strindex`` and ``key_strindex`` are left out.
"""

from __future__ import annotations

import base64
import json
import math
import re
import struct
from collections.abc import Mapping
from typing import Any, NamedTuple

PROTOCOL_JSON = "http/json"
PROTOCOL_PROTOBUF = "http/protobuf"
PROTOCOLS = (PROTOCOL_JSON, PROTOCOL_PROTOBUF)
# Read in this order when --protocol is not given; the first one set wins.
PROTOCOL_ENV_VARS = ("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "OTEL_EXPORTER_OTLP_PROTOCOL")
CONTENT_TYPES = {
    PROTOCOL_JSON: "application/json",
    PROTOCOL_PROTOBUF: "application/x-protobuf",
}
_GRPC_NOTE = "agent-strace sends OTLP over HTTP only; use http/protobuf or http/json"


class EncodeError(ValueError):
    """The payload cannot be encoded as an OTLP protobuf request.

    The message names the field by its path, never its value.
    """


def resolve_protocol(flag: str | None, environ: Mapping[str, str]) -> str:
    """Return the OTLP/HTTP protocol to send with.

    ``--protocol`` (*flag*) wins, then OTEL_EXPORTER_OTLP_TRACES_PROTOCOL,
    then OTEL_EXPORTER_OTLP_PROTOCOL, then ``http/json``. An empty variable
    counts as unset, as the OpenTelemetry spec says. Only the value that
    wins is checked. Raises ValueError with a refusal line for ``grpc`` or
    any other value; the line never repeats the value.
    """
    if flag is not None:
        source, value = "--protocol", flag
    else:
        source, value = "", ""
        for name in PROTOCOL_ENV_VARS:
            value = (environ.get(name) or "").strip()
            if value:
                source = name
                break
        if not source:
            return PROTOCOL_JSON
    normalized = value.strip().lower()
    if normalized in PROTOCOLS:
        return normalized
    env_note = "" if source == "--protocol" else "; or pass --protocol to override it"
    if normalized == "grpc":
        refusal = f"{source} grpc is not supported: {_GRPC_NOTE}{env_note}"
    else:
        refusal = f"invalid {source} value: use http/json or http/protobuf{env_note}"
    raise ValueError(refusal)


def unsupported_media_hint(protocol: str) -> str:
    """Return the note for an HTTP 415 answer to a request sent with *protocol*."""
    if protocol == PROTOCOL_PROTOBUF:
        return "This collector does not accept OTLP protobuf; send JSON with --protocol http/json"
    return "This collector does not accept OTLP JSON; send protobuf with --protocol http/protobuf"


def encode_body(payload: Mapping[str, Any], protocol: str) -> bytes:
    """Return the request body for an OTLP JSON *payload* sent with *protocol*."""
    if protocol == PROTOCOL_PROTOBUF:
        return encode_trace_request(payload)
    return json.dumps(payload).encode("utf-8")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class _Field(NamedTuple):
    name: str           # proto field name
    number: int
    kind: str           # see _encode_value
    message: str = ""   # message type, for kind "message"
    repeated: bool = False
    oneof: bool = False
    enum: Mapping[str, int] | None = None


_SPAN_KIND = {
    "SPAN_KIND_UNSPECIFIED": 0, "SPAN_KIND_INTERNAL": 1, "SPAN_KIND_SERVER": 2,
    "SPAN_KIND_CLIENT": 3, "SPAN_KIND_PRODUCER": 4, "SPAN_KIND_CONSUMER": 5,
}
_STATUS_CODE = {"STATUS_CODE_UNSET": 0, "STATUS_CODE_OK": 1, "STATUS_CODE_ERROR": 2}

_SCHEMA: dict[str, tuple[_Field, ...]] = {
    "ExportTraceServiceRequest": (
        _Field("resource_spans", 1, "message", "ResourceSpans", repeated=True),
    ),
    "ResourceSpans": (
        _Field("resource", 1, "message", "Resource"),
        _Field("scope_spans", 2, "message", "ScopeSpans", repeated=True),
        _Field("schema_url", 3, "string"),
    ),
    "ScopeSpans": (
        _Field("scope", 1, "message", "InstrumentationScope"),
        _Field("spans", 2, "message", "Span", repeated=True),
        _Field("schema_url", 3, "string"),
    ),
    "Span": (
        _Field("trace_id", 1, "hex"),
        _Field("span_id", 2, "hex"),
        _Field("trace_state", 3, "string"),
        _Field("parent_span_id", 4, "hex"),
        _Field("name", 5, "string"),
        _Field("kind", 6, "enum", enum=_SPAN_KIND),
        _Field("start_time_unix_nano", 7, "fixed64"),
        _Field("end_time_unix_nano", 8, "fixed64"),
        _Field("attributes", 9, "message", "KeyValue", repeated=True),
        _Field("dropped_attributes_count", 10, "uint32"),
        _Field("events", 11, "message", "Event", repeated=True),
        _Field("dropped_events_count", 12, "uint32"),
        _Field("links", 13, "message", "Link", repeated=True),
        _Field("dropped_links_count", 14, "uint32"),
        _Field("status", 15, "message", "Status"),
        _Field("flags", 16, "fixed32"),
    ),
    "Event": (
        _Field("time_unix_nano", 1, "fixed64"),
        _Field("name", 2, "string"),
        _Field("attributes", 3, "message", "KeyValue", repeated=True),
        _Field("dropped_attributes_count", 4, "uint32"),
    ),
    "Link": (
        _Field("trace_id", 1, "hex"),
        _Field("span_id", 2, "hex"),
        _Field("trace_state", 3, "string"),
        _Field("attributes", 4, "message", "KeyValue", repeated=True),
        _Field("dropped_attributes_count", 5, "uint32"),
        _Field("flags", 6, "fixed32"),
    ),
    "Status": (
        _Field("message", 2, "string"),
        _Field("code", 3, "enum", enum=_STATUS_CODE),
    ),
    "Resource": (
        _Field("attributes", 1, "message", "KeyValue", repeated=True),
        _Field("dropped_attributes_count", 2, "uint32"),
        _Field("entity_refs", 3, "message", "EntityRef", repeated=True),
    ),
    "EntityRef": (
        _Field("schema_url", 1, "string"),
        _Field("type", 2, "string"),
        _Field("id_keys", 3, "string", repeated=True),
        _Field("description_keys", 4, "string", repeated=True),
    ),
    "InstrumentationScope": (
        _Field("name", 1, "string"),
        _Field("version", 2, "string"),
        _Field("attributes", 3, "message", "KeyValue", repeated=True),
        _Field("dropped_attributes_count", 4, "uint32"),
    ),
    "KeyValue": (
        _Field("key", 1, "string"),
        _Field("value", 2, "message", "AnyValue"),
    ),
    "AnyValue": (
        _Field("string_value", 1, "string", oneof=True),
        _Field("bool_value", 2, "bool", oneof=True),
        _Field("int_value", 3, "int64", oneof=True),
        _Field("double_value", 4, "double", oneof=True),
        _Field("array_value", 5, "message", "ArrayValue", oneof=True),
        _Field("kvlist_value", 6, "message", "KeyValueList", oneof=True),
        _Field("bytes_value", 7, "bytes", oneof=True),
    ),
    "ArrayValue": (
        _Field("values", 1, "message", "AnyValue", repeated=True),
    ),
    "KeyValueList": (
        _Field("values", 1, "message", "KeyValue", repeated=True),
    ),
}


def _camel(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(part.capitalize() for part in rest)


# Each message's fields by JSON name and by proto name.
_FIELDS: dict[str, dict[str, _Field]] = {
    message: {
        key: field for field in fields for key in (_camel(field.name), field.name)
    }
    for message, fields in _SCHEMA.items()
}

# ---------------------------------------------------------------------------
# Wire format
# ---------------------------------------------------------------------------

_VARINT, _I64, _LEN, _I32 = 0, 1, 2, 5
_WIRE_TYPES = {
    "string": _LEN, "bytes": _LEN, "hex": _LEN, "message": _LEN,
    "bool": _VARINT, "int64": _VARINT, "uint32": _VARINT, "enum": _VARINT,
    "fixed64": _I64, "double": _I64, "fixed32": _I32,
}
# Inclusive ranges of the integer kinds. An enum is an int32.
_RANGES = {
    "int64": (-(1 << 63), (1 << 63) - 1),
    "uint32": (0, (1 << 32) - 1),
    "fixed64": (0, (1 << 64) - 1),
    "fixed32": (0, (1 << 32) - 1),
    "enum": (-(1 << 31), (1 << 31) - 1),
}
_INTEGER_TEXT = re.compile(r"-?[0-9]+")
_NUMBER_TEXT = re.compile(r"-?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
_SPECIAL_DOUBLES = {"NaN": math.nan, "Infinity": math.inf, "-Infinity": -math.inf}
_HEX_TEXT = re.compile(r"(?:[0-9a-fA-F]{2})*")
_BASE64_TEXT = re.compile(r"[A-Za-z0-9+/_-]*={0,2}")


def _varint(value: int) -> bytes:
    """Return *value* as a base-128 varint; a negative value takes 10 bytes."""
    value &= (1 << 64) - 1
    out = bytearray()
    while value > 0x7F:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _integer(value: object, kind: str, path: str) -> int:
    if isinstance(value, bool):
        raise EncodeError(f"{path}: expected an integer, got a boolean")
    if isinstance(value, int):
        number = value
    elif isinstance(value, float) and value.is_integer():
        number = int(value)
    elif isinstance(value, str) and _INTEGER_TEXT.fullmatch(value):
        number = int(value)
    else:
        raise EncodeError(f"{path}: expected an integer")
    low, high = _RANGES[kind]
    if not low <= number <= high:
        raise EncodeError(f"{path}: integer out of range for {kind}")
    return number


def _double(value: object, path: str) -> float:
    if isinstance(value, bool):
        raise EncodeError(f"{path}: expected a number, got a boolean")
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except OverflowError:
            raise EncodeError(f"{path}: number out of range for double") from None
    if isinstance(value, str):
        if value in _SPECIAL_DOUBLES:
            return _SPECIAL_DOUBLES[value]
        if _NUMBER_TEXT.fullmatch(value):
            return float(value)
    raise EncodeError(f"{path}: expected a number")


def _utf8(value: object, path: str) -> bytes:
    if not isinstance(value, str):
        raise EncodeError(f"{path}: expected a string")
    try:
        return value.encode("utf-8")
    except UnicodeEncodeError:
        # A lone surrogate, which UTF-8 cannot hold. JSON receivers read the
        # \\udXXX escape json.dumps writes as U+FFFD, so this does the same;
        # a surrogate pair split across two characters is joined.
        joined = value.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "replace")
        return joined.encode("utf-8")


def _hex_bytes(value: object, path: str) -> bytes:
    if not isinstance(value, str) or not _HEX_TEXT.fullmatch(value):
        raise EncodeError(f"{path}: expected an even number of hex digits")
    return bytes.fromhex(value)


def _base64_bytes(value: object, path: str) -> bytes:
    if not isinstance(value, str) or not _BASE64_TEXT.fullmatch(value):
        raise EncodeError(f"{path}: expected base64")
    text = value.rstrip("=").replace("-", "+").replace("_", "/")
    if len(text) % 4 == 1:
        raise EncodeError(f"{path}: expected base64")
    return base64.b64decode(text + "=" * (-len(text) % 4))


def _encode_value(field: _Field, value: object, path: str, explicit: bool) -> bytes:
    """Return one field's tag and value, or b"" for an implicit default."""
    kind = field.kind
    if kind == "message":
        data = _encode_message(field.message, value, path)
    elif kind in ("string", "hex", "bytes"):
        if kind == "string":
            data = _utf8(value, path)
        elif kind == "hex":
            data = _hex_bytes(value, path)
        else:
            data = _base64_bytes(value, path)
        if not data and not explicit:
            return b""
    elif kind == "bool":
        if not isinstance(value, bool):
            raise EncodeError(f"{path}: expected a boolean")
        if not value and not explicit:
            return b""
        data = b"\x01" if value else b"\x00"
    elif kind == "enum" and isinstance(value, str) and value in (field.enum or {}):
        number = field.enum[value]
        if not number and not explicit:
            return b""
        data = _varint(number)
    elif kind in ("int64", "uint32", "enum"):
        number = _integer(value, kind, path)
        if not number and not explicit:
            return b""
        data = _varint(number)
    elif kind in ("fixed64", "fixed32"):
        number = _integer(value, kind, path)
        if not number and not explicit:
            return b""
        data = struct.pack("<Q" if kind == "fixed64" else "<I", number)
    else:  # double
        packed = struct.pack("<d", _double(value, path))
        if packed == bytes(8) and not explicit:
            return b""
        data = packed
    wire = _WIRE_TYPES[kind]
    tag = _varint(field.number << 3 | wire)
    if wire == _LEN:
        return tag + _varint(len(data)) + data
    return tag + data


def _encode_message(message: str, value: object, path: str) -> bytes:
    where = f"{path}: " if path else ""
    if not isinstance(value, Mapping):
        raise EncodeError(f"{where}expected an object")
    fields = _FIELDS[message]
    present: dict[int, tuple[_Field, object, str]] = {}
    for key, item in value.items():
        field = fields.get(key) if isinstance(key, str) else None
        if field is None:
            raise EncodeError(f"{where}{message} has no field {key!r}")
        if item is None:
            continue
        if field.number in present:
            raise EncodeError(f"{where}field {field.name} is given twice")
        key_path = f"{path}.{key}" if path else key
        present[field.number] = (field, item, key_path)
    if sum(1 for field, _, _ in present.values() if field.oneof) > 1:
        raise EncodeError(f"{where}more than one value is set")
    out = bytearray()
    for number in sorted(present):
        field, item, key_path = present[number]
        if field.repeated:
            if not isinstance(item, (list, tuple)):
                raise EncodeError(f"{key_path}: expected an array")
            for index, element in enumerate(item):
                out += _encode_value(field, element, f"{key_path}[{index}]", True)
        else:
            out += _encode_value(field, item, key_path, field.oneof or field.kind == "message")
    return bytes(out)


def encode_trace_request(payload: Mapping[str, Any]) -> bytes:
    """Return an OTLP JSON ExportTraceServiceRequest as protobuf bytes.

    Raises EncodeError for a value the OTLP JSON rules do not allow, naming
    the field by its path, such as
    ``resourceSpans[0].scopeSpans[0].spans[2].traceId``.
    """
    return _encode_message("ExportTraceServiceRequest", payload, "")
