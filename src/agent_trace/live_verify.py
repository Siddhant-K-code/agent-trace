"""Verify that installed provider hooks produced reviewable local evidence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from .evidence_health import EvidenceHealthReasonKind, EvidenceHealthStatus
from .health import HealthReport, assess_session, load_capture_matrix
from .hooks import _PROVIDER_AGENT
from .models import SessionMeta
from .store import TraceStore

_SUPPORTED_PROVIDERS = frozenset({"claude", "codex"})
_PROVIDER_NAMES = {
    "claude": "Claude Code",
    "codex": "OpenAI Codex",
}
_CAPTURE_GUIDANCE = {
    "claude": (
        "Start Claude Code in this project, accept workspace trust, submit a harmless "
        "prompt, exit the session cleanly, then rerun this command."
    ),
    "codex": (
        "Open Codex, review and trust the AgentTrace commands in /hooks, run a harmless "
        "local prompt, exit the session cleanly, then rerun this command."
    ),
}


class LiveCaptureError(ValueError):
    """A user-actionable live verification failure."""


@dataclass(frozen=True)
class LiveCaptureVerification:
    """One matrix-backed assessment of recorded provider evidence."""

    report: HealthReport
    selection: str

    @property
    def exit_code(self) -> int:
        return self.report.exit_code


def _metadata_matches_provider(meta: SessionMeta, provider: str) -> bool:
    agent_name = _PROVIDER_AGENT.get(provider)
    if not agent_name or meta.agent_name != agent_name:
        return False
    return meta.command.startswith(f"{agent_name} (")


def _resolve_requested_session(store: TraceStore, requested: str) -> str | None:
    try:
        if store.session_exists(requested):
            return requested
        return store.find_session(requested)
    except ValueError as exc:
        raise LiveCaptureError(f"Invalid session ID or prefix {requested!r}: {exc}") from exc


def _load_report(
    store: TraceStore,
    session_id: str,
    provider: str,
    matrix,
    *,
    selection: str,
) -> LiveCaptureVerification:
    try:
        meta = store.load_meta(session_id)
        events = store.load_events(session_id)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise LiveCaptureError(
            f"Session {session_id} could not be read as recorded evidence: {exc}. "
            "Recapture the session, or select another with --session SESSION_ID."
        ) from exc

    if not events:
        raise LiveCaptureError(
            f"Session {session_id} contains no captured events. The configuration is installed, "
            f"but no {_PROVIDER_NAMES[provider]} events were observed. {_CAPTURE_GUIDANCE[provider]}"
        )

    report = assess_session(store, session_id, meta, events, matrix)
    detected = report.provider.provider
    if detected != provider:
        if report.provider.conflicting:
            found = ", ".join(report.provider.conflicting)
            detail = f"its hook events name conflicting providers: {found}"
        elif detected:
            detail = f"it contains {detected!r} hook evidence"
        else:
            detail = "it contains no provider-identifying hook evidence"
        raise LiveCaptureError(
            f"Session {session_id} does not match requested provider {provider!r}: {detail}. "
            "Select a session recorded by the requested provider with --session SESSION_ID."
        )

    return LiveCaptureVerification(report=report, selection=selection)


def inspect_live_capture(
    trace_dir: str | Path,
    provider: str,
    requested_session: str | None = None,
) -> LiveCaptureVerification:
    """Select and assess one recorded session for ``provider``.

    Without an explicit session, the newest session whose hook evidence names
    the provider is selected. Hook-shaped metadata is used only to surface an
    unreadable or empty newest candidate instead of silently falling back to an
    older healthy trace.
    """
    if provider not in _SUPPORTED_PROVIDERS:
        raise LiveCaptureError(
            "Live capture verification supports only --cli codex or --cli claude, "
            "one provider per run."
        )
    try:
        matrix = load_capture_matrix()
    except (OSError, ValueError) as exc:
        raise LiveCaptureError(f"Could not load the packaged capture matrix: {exc}") from exc
    if matrix.entry(provider) is None:
        raise LiveCaptureError(
            f"The packaged capture matrix has no entry for provider {provider!r}."
        )

    store = TraceStore(trace_dir)
    if requested_session:
        session_id = _resolve_requested_session(store, requested_session)
        if session_id is None:
            raise LiveCaptureError(
                f"Session not found: {requested_session}. Run agent-strace list, then pass "
                "--session SESSION_ID."
            )
        return _load_report(
            store,
            session_id,
            provider,
            matrix,
            selection=f"requested session {requested_session}",
        )

    for meta in store.list_sessions():
        hinted = _metadata_matches_provider(meta, provider)
        try:
            events = store.load_events(meta.session_id)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            if hinted:
                raise LiveCaptureError(
                    f"Newest {_PROVIDER_NAMES[provider]} capture candidate {meta.session_id} "
                    f"could not be read: {exc}. Recapture the session, or select another with "
                    "--session SESSION_ID."
                ) from exc
            continue
        if not events and hinted:
            raise LiveCaptureError(
                f"Newest {_PROVIDER_NAMES[provider]} capture candidate {meta.session_id} "
                f"contains no captured events. The configuration is installed, but no events "
                f"were observed. {_CAPTURE_GUIDANCE[provider]}"
            )
        report = assess_session(store, meta.session_id, meta, events, matrix)
        if report.provider.provider == provider:
            return LiveCaptureVerification(
                report=report,
                selection="newest matching recorded session",
            )
        if hinted:
            return _load_report(
                store,
                meta.session_id,
                provider,
                matrix,
                selection="newest matching recorded session",
            )

    raise LiveCaptureError(
        f"No recorded {_PROVIDER_NAMES[provider]} session was found in {store.base_dir}. "
        f"The configuration is installed, but matching events were not observed. "
        f"{_CAPTURE_GUIDANCE[provider]} The verifier selects the newest matching session by "
        "default; pass --session SESSION_ID to select another recorded session."
    )


def format_live_capture_text(result: LiveCaptureVerification, out: TextIO) -> None:
    """Print the acceptance evidence for a live capture verification."""
    report = result.report
    entry = report.entry
    if entry is None:
        raise ValueError("live capture verification requires a capture-matrix entry")

    out.write("Capture evidence: observed\n")
    out.write(f"Selection: {result.selection}\n")
    out.write(f"Session: {report.session_id}\n")
    out.write(f"Provider: {report.provider.provider}\n")
    out.write(f"Capture method: {entry['capture_method']}\n")
    out.write(f"Evidence health: {report.result.status.value}\n")
    types = ", ".join(f"{kind} {count}" for kind, count in report.event_types.items())
    out.write(f"Observed event types: {types or 'none'}\n")
    out.write(
        f"Start marker: {'observed' if report.result.observed_start else 'missing'}\n"
    )
    out.write(f"End marker: {'observed' if report.result.observed_end else 'missing'}\n")
    out.write(f"Redaction: {report.redaction.mode.replace('_', ' ')}\n")
    out.write(f"Redacted events: {report.redaction.redacted_event_count}\n")
    out.write(f"Capture matrix: {entry['name']} ({report.matrix.version})\n")

    observed = [
        reason
        for reason in report.result.reasons
        if reason.kind == EvidenceHealthReasonKind.OBSERVED
    ]
    out.write(f"Observed diagnostics ({len(observed)}):\n")
    if observed:
        for reason in observed:
            out.write(f"  - {reason.code}: {reason.message}\n")
    else:
        out.write("  none\n")

    out.write(f"Capture matrix gaps ({len(report.blind_spots)}):\n")
    if report.blind_spots:
        for spot in report.blind_spots:
            out.write(f"  - {spot.text}\n")
    else:
        out.write("  none declared\n")


def live_capture_failure_diagnostic(result: LiveCaptureVerification) -> str:
    """Return an actionable message for a non-passing health result."""
    status = result.report.result.status
    session_id = result.report.session_id
    if status == EvidenceHealthStatus.INVALID:
        return (
            f"Session {session_id} has invalid evidence. Recapture it after fixing the "
            "reported diagnostics, or select another with --session SESSION_ID."
        )
    if status == EvidenceHealthStatus.UNKNOWN:
        return (
            f"Session {session_id} does not yet contain reviewable completed evidence. "
            "Finish the provider session cleanly, then rerun this command."
        )
    return ""
