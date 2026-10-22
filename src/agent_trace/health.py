"""Evidence health for one recorded session (``agent-strace health``).

Adapts session metadata, hook state and the packaged capture matrix into
:func:`agent_trace.evidence_health.assess_evidence_health`. The core stays free
of file and metadata access; this module is the CLI adapter around it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from importlib import resources
from typing import Any, Sequence, TextIO

from .backfill import hook_agent_names
from .disclosure import count_redacted_events
from .evidence_health import (
    EvidenceHealthReasonKind,
    EvidenceHealthResult,
    EvidenceHealthStatus,
    assess_evidence_health,
)
from .hooks import (
    IMPLICIT_SESSION_SOURCE,
    REDACTION_MODE_KEY,
    REDACTION_OFF,
    REDACTION_ON,
    _PROVIDER_AGENT,
)
from .models import EventType, SessionMeta, TraceEvent
from .store import TraceStore

CAPTURE_MATRIX_RESOURCE = "capture_matrix.json"
CAPTURE_MATRIX_SCHEMA_VERSION = 1

REDACTION_MIXED = "mixed"
REDACTION_NOT_RECORDED = "not_recorded"
_RECORDED_REDACTION_MODES = (REDACTION_ON, REDACTION_OFF)
_REDACTION_LABELS = {
    REDACTION_ON: "on (recorded by the session start hook)",
    REDACTION_OFF: "off (recorded by the session start hook)",
    REDACTION_NOT_RECORDED: "not recorded (this trace does not show whether redaction was on)",
}

_AGENT_PROVIDER = {agent: provider for provider, agent in _PROVIDER_AGENT.items()}
_PASSING_STATUSES = frozenset({EvidenceHealthStatus.HEALTHY, EvidenceHealthStatus.PARTIAL})
_ENTRY_FIELDS = (
    "id",
    "name",
    "capture_method",
    "verification_status",
    "tested_version",
    "last_verified",
    "coverage",
)
_SOURCE_LABELS = {"hook_events": "hook events", "hook_metadata": "hook metadata"}
_FINALIZED_LABELS = {
    True: "yes",
    False: "no (active-session marker present)",
    None: "unknown",
}


@dataclass(frozen=True)
class CaptureMatrix:
    """The provider capture matrix shipped in the package."""

    schema_version: int
    version: str
    providers: tuple[dict[str, Any], ...]

    def entry(self, provider: str) -> dict[str, Any] | None:
        for entry in self.providers:
            if entry["id"] == provider:
                return entry
        return None


def load_capture_matrix() -> CaptureMatrix:
    """Load the packaged capture matrix.

    ``version`` is the SHA-256 of the packaged bytes: a stable reference to the
    exact matrix revision that declared a provider limitation.
    """
    raw = resources.files("agent_trace").joinpath(CAPTURE_MATRIX_RESOURCE).read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict) or data.get("schema_version") != CAPTURE_MATRIX_SCHEMA_VERSION:
        raise ValueError("unsupported capture matrix schema_version")
    providers = data.get("providers")
    if not isinstance(providers, list):
        raise ValueError("capture matrix providers must be a list")
    for entry in providers:
        if not (
            isinstance(entry, dict)
            and isinstance(entry.get("id"), str)
            and isinstance(entry.get("capture_method"), str)
            and isinstance(entry.get("coverage"), dict)
            and isinstance(entry.get("known_gaps"), list)
            and all(isinstance(gap, str) for gap in entry["known_gaps"])
            and _valid_event_types(entry.get("events_before_start", []))
        ):
            raise ValueError("capture matrix provider entry is malformed")
    return CaptureMatrix(
        schema_version=CAPTURE_MATRIX_SCHEMA_VERSION,
        version="sha256:" + hashlib.sha256(raw).hexdigest(),
        providers=tuple(providers),
    )


def _valid_event_types(value: object) -> bool:
    if not isinstance(value, list):
        return False
    known = {event_type.value for event_type in EventType}
    return all(isinstance(item, str) and item in known for item in value)


@dataclass(frozen=True)
class BlindSpot:
    """A provider limitation declared by a capture-matrix entry."""

    source: str
    text: str
    capability: str = ""

    def to_dict(self) -> dict[str, str]:
        payload = {"source": self.source, "text": self.text}
        if self.capability:
            payload["capability"] = self.capability
        return payload


def blind_spots_for(entry: dict[str, Any]) -> tuple[BlindSpot, ...]:
    """Return the limitations an entry declares, and nothing else.

    Every ``known_gaps`` string is used verbatim, followed by one line per
    ``coverage`` capability marked ``unavailable``. ``provider_defined``
    coverage adds no line; the entry's own gaps describe it.
    """
    spots = [
        BlindSpot("known_gaps", gap.strip())
        for gap in entry["known_gaps"]
        if gap.strip()
    ]
    name = entry.get("name") or entry["id"]
    for capability, value in entry["coverage"].items():
        if value == "unavailable":
            spots.append(
                BlindSpot(
                    "coverage",
                    f"Capture matrix: {capability} is unavailable for {name}.",
                    capability,
                )
            )
    return tuple(spots)


@dataclass(frozen=True)
class ProviderEvidence:
    """The hook provider of a session and where it was read from."""

    provider: str = ""
    source: str = ""
    conflicting: tuple[str, ...] = ()


def detect_provider(meta: SessionMeta, events: Sequence[TraceEvent]) -> ProviderEvidence:
    """Identify the hook provider from hook-written evidence only.

    The ``session_start`` and stop events written by hooks name the provider.
    Sessions whose first hook was not SessionStart fall back to the metadata
    the hooks write for them. A name chosen with ``record --name``, the SDK or
    an importer is never used, because it does not identify a hook surface.
    """
    providers = sorted({_AGENT_PROVIDER[name] for name in hook_agent_names(list(events))})
    if len(providers) > 1:
        return ProviderEvidence(conflicting=tuple(providers))
    if providers:
        return ProviderEvidence(providers[0], "hook_events")
    if _implicit_hook_session(meta):
        return ProviderEvidence(_AGENT_PROVIDER[meta.agent_name], "hook_metadata")
    return ProviderEvidence()


def _implicit_hook_session(meta: SessionMeta) -> bool:
    """Return True for metadata the hooks wrote because the first hook was not SessionStart."""
    implicit_command = f"{meta.agent_name} ({IMPLICIT_SESSION_SOURCE})"
    return meta.agent_name in _AGENT_PROVIDER and meta.command == implicit_command


def _has_active_marker(store: TraceStore, session_id: str) -> bool:
    """Return True when a hook active-session marker names ``session_id``."""
    try:
        markers = list(store.base_dir.glob(".active-session*"))
    except OSError:
        return False
    for marker in markers:
        try:
            if marker.is_symlink() or not marker.is_file():
                continue
            if marker.read_text(encoding="utf-8").strip() == session_id:
                return True
        except (OSError, ValueError):
            continue
    return False


def session_finalized_for(
    store: TraceStore,
    session_id: str,
    meta: SessionMeta,
    entry: dict[str, Any] | None,
) -> bool | None:
    """Return the ``session_finalized`` value passed to the core.

    ``True`` when the metadata records an end time. ``False`` only when the
    matrix says the provider captures ``session_end`` and a hook active-session
    marker still names the session. Otherwise ``None`` (not known).
    """
    if meta.ended_at is not None:
        return True
    if (
        entry is not None
        and entry["coverage"].get("session_end") == "captured"
        and _has_active_marker(store, session_id)
    ):
        return False
    return None


def _matrix_note(provider: ProviderEvidence, entry: dict[str, Any] | None) -> str:
    if entry is not None:
        return ""
    if provider.conflicting:
        return (
            f"Hook events name more than one provider ({', '.join(provider.conflicting)}); "
            "no capture-matrix entry was applied, so provider limitations were not assessed."
        )
    if not provider.provider:
        return (
            "Provider not identified from hook-written evidence; no capture-matrix entry "
            "was applied, so provider limitations were not assessed."
        )
    return (
        f"The capture matrix has no entry for provider {provider.provider!r}; "
        "provider limitations were not assessed."
    )


def event_type_counts(events: Sequence[TraceEvent]) -> dict[str, int]:
    """Count the observed event types, in ``EventType`` declaration order."""
    counts = Counter(event.event_type for event in events)
    return {kind.value: counts[kind] for kind in EventType if counts[kind]}


@dataclass(frozen=True)
class RedactionState:
    """Secret-redaction facts read from a trace. They never affect status.

    ``mode`` is what the session starts recorded: ``on`` or ``off`` when all
    of them recorded the same value, ``mixed`` when they differ or only some
    recorded one, and ``not_recorded`` when none did. ``session_start_modes``
    has one value per start, in order: an implicit start first when the first
    hook was not SessionStart, then each ``session_start`` event.
    """

    mode: str
    session_start_modes: tuple[str, ...]
    redacted_event_count: int
    metadata_redacted: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "session_start_modes": list(self.session_start_modes),
            "redacted_event_count": self.redacted_event_count,
            "metadata_redacted": self.metadata_redacted,
        }


def _session_start_redaction(event: TraceEvent) -> str:
    data = event.data if isinstance(event.data, dict) else {}
    mode = data.get(REDACTION_MODE_KEY)
    if mode in _RECORDED_REDACTION_MODES and hook_agent_names([event]):
        return mode
    return REDACTION_NOT_RECORDED


def redaction_state_for(meta: SessionMeta, events: Sequence[TraceEvent]) -> RedactionState:
    """Report redaction from what the trace records, without guessing the mode.

    Only hooks record the mode, on each ``session_start`` they write. Older
    traces and other capture paths read as ``not_recorded``. When the first
    hook was not SessionStart and resume or compact kept that metadata, the
    implicit start counts as ``not_recorded``. Zero redacted events does not
    prove that redaction was on.
    """
    starts = tuple(
        _session_start_redaction(event)
        for event in events
        if event.event_type == EventType.SESSION_START
    )
    if _implicit_hook_session(meta):
        starts = (REDACTION_NOT_RECORDED, *starts)
    recorded = set(starts) - {REDACTION_NOT_RECORDED}
    if not recorded:
        mode = REDACTION_NOT_RECORDED
    elif len(recorded) == 1 and REDACTION_NOT_RECORDED not in starts:
        mode = starts[0]
    else:
        mode = REDACTION_MIXED
    return RedactionState(
        mode=mode,
        session_start_modes=starts,
        redacted_event_count=count_redacted_events(events),
        metadata_redacted=bool(meta.redacted),
    )


@dataclass(frozen=True)
class HealthReport:
    """An evidence-health result plus the inputs the CLI supplied to the core."""

    session_id: str
    result: EvidenceHealthResult
    provider: ProviderEvidence
    session_finalized: bool | None
    matrix: CaptureMatrix
    entry: dict[str, Any] | None
    blind_spots: tuple[BlindSpot, ...]
    note: str
    event_types: dict[str, int]
    redaction: RedactionState

    @property
    def exit_code(self) -> int:
        return 0 if self.result.status in _PASSING_STATUSES else 1

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"session_id": self.session_id}
        payload.update(self.result.to_dict())
        payload["session_finalized"] = self.session_finalized
        payload["provider_source"] = self.provider.source or None
        payload["event_types"] = dict(self.event_types)
        payload["redaction"] = self.redaction.to_dict()
        payload["capture_matrix"] = {
            "schema_version": self.matrix.schema_version,
            "version": self.matrix.version,
            "entry": (
                None
                if self.entry is None
                else {key: self.entry.get(key) for key in _ENTRY_FIELDS}
            ),
            "blind_spots": [spot.to_dict() for spot in self.blind_spots],
            "note": self.note or None,
        }
        return payload


def assess_session(
    store: TraceStore,
    session_id: str,
    meta: SessionMeta,
    events: Sequence[TraceEvent],
    matrix: CaptureMatrix,
) -> HealthReport:
    """Assess one loaded session against its capture-matrix entry."""
    provider = detect_provider(meta, events)
    entry = matrix.entry(provider.provider) if provider.provider else None
    blind_spots = blind_spots_for(entry) if entry is not None else ()
    finalized = session_finalized_for(store, session_id, meta, entry)
    result = assess_evidence_health(
        events,
        provider=provider.provider,
        capture_method=entry["capture_method"] if entry is not None else "",
        capture_matrix_version=matrix.version if entry is not None else "",
        provider_blind_spots=[spot.text for spot in blind_spots],
        session_finalized=finalized,
        allowed_pre_start_event_types=(
            entry.get("events_before_start", []) if entry is not None else ()
        ),
    )
    return HealthReport(
        session_id=session_id,
        result=result,
        provider=provider,
        session_finalized=finalized,
        matrix=matrix,
        entry=entry,
        blind_spots=blind_spots,
        note=_matrix_note(provider, entry),
        event_types=event_type_counts(events),
        redaction=redaction_state_for(meta, events),
    )


def _redaction_text(redaction: RedactionState) -> str:
    if redaction.mode == REDACTION_MIXED:
        starts = ", ".join(mode.replace("_", " ") for mode in redaction.session_start_modes)
        return f"mixed (session starts: {starts})"
    return _REDACTION_LABELS[redaction.mode]


def format_health_text(report: HealthReport, out: TextIO) -> None:
    result = report.result
    w = out.write
    w(f"Evidence health: {result.status.value}\n")
    w(f"Session:   {report.session_id}\n")
    provider = result.provider or "unknown"
    if report.provider.source:
        provider += f" (from {_SOURCE_LABELS[report.provider.source]})"
    w(f"Provider:  {provider}\n")
    w(f"Finalized: {_FINALIZED_LABELS[report.session_finalized]}\n")
    w(f"Events:    {result.event_count}\n")
    types = ", ".join(f"{kind} {count}" for kind, count in report.event_types.items())
    w(f"Types:     {types or 'none'}\n")
    redaction = report.redaction
    w(f"Redaction: {_redaction_text(redaction)}\n")
    redacted = f"{redaction.redacted_event_count} of {result.event_count} events"
    if redaction.metadata_redacted:
        redacted += ", plus session metadata"
    w(f"Redacted:  {redacted}\n")

    observed = [r for r in result.reasons if r.kind == EvidenceHealthReasonKind.OBSERVED]
    limitations = [
        r for r in result.reasons if r.kind == EvidenceHealthReasonKind.PROVIDER_LIMITATION
    ]

    w(f"\nObserved ({len(observed)}):\n")
    for reason in observed:
        w(f"  - {reason.code}: {reason.message}\n")
    if not observed:
        w("  none\n")

    w(f"\nProvider limitations ({len(limitations)}):\n")
    for reason in limitations:
        w(f"  - {reason.message}\n")
    if not limitations:
        w("  none declared\n" if report.entry is not None else "  not assessed\n")

    w("\n")
    if report.entry is not None:
        entry = report.entry
        w(
            f"Capture matrix: {entry.get('name') or entry['id']} ({entry['id']}), "
            f"capture_method {entry['capture_method']}, "
            f"verification_status {entry.get('verification_status') or 'unknown'}\n"
        )
    else:
        w("Capture matrix: no entry applied\n")
        w(f"Note: {report.note}\n")
    w(f"Matrix version: {report.matrix.version}\n")


def _resolve_session(store: TraceStore, requested: str | None) -> str | None:
    if not requested:
        return store.get_latest_session_id()
    try:
        if store.session_exists(requested):
            return requested
        return store.find_session(requested)
    except ValueError:
        return None


def cmd_health(args: argparse.Namespace) -> int:
    """Report evidence health for one session. Exit 0 for healthy or partial."""
    store = TraceStore(args.trace_dir)
    requested = getattr(args, "session_id", None)
    session_id = _resolve_session(store, requested)
    if not session_id:
        if requested:
            sys.stderr.write(f"Session not found: {requested}\n")
        else:
            sys.stderr.write("No sessions found.\n")
        return 1

    try:
        matrix = load_capture_matrix()
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"Could not load the packaged capture matrix: {exc}\n")
        return 1

    try:
        meta = store.load_meta(session_id)
        events = store.load_events(session_id)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        sys.stderr.write(f"Could not read session {session_id}: {exc}\n")
        return 1

    report = assess_session(store, session_id, meta, events, matrix)
    if getattr(args, "format", "text") == "json":
        sys.stdout.write(json.dumps(report.to_dict(), indent=2) + "\n")
    else:
        format_health_text(report, sys.stdout)
    return report.exit_code
