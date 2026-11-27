"""Privacy-preserving product telemetry for the agent-strace CLI.

Telemetry is enabled by default and can be disabled at any time.  Nothing is
sent until the one-time notice has been shown or the user explicitly enabled
telemetry.  The client sends a small allow-listed event directly to PostHog's
public capture API.  It never reads or sends trace events, prompts, command
arguments, file paths, repository metadata, or trace session identifiers.
Events carry a random PostHog ``$session_id`` that only groups one period of
activity on this installation; see ``_analytics_session_id``.

The PostHog project token is intentionally a public, write-only ingestion
token.  Set ``DEFAULT_POSTHOG_PROJECT_TOKEN`` before publishing a release, or
use ``AGENT_STRACE_TELEMETRY_TOKEN`` while developing and testing.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, TextIO

from . import __version__
from .fileutil import file_lock, write_atomic

if TYPE_CHECKING:
    import argparse


def _urllib_request():
    """Return ``urllib.request``, importing it on first use.

    It is slow to import and most hook processes never send an event.
    """
    module = globals().get("request")
    if module is None:
        from urllib import request as module

        globals()["request"] = module
    return module


def __getattr__(name: str):
    # Keep ``telemetry.request`` available to callers that patch its urlopen.
    if name == "request":
        return _urllib_request()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# PostHog project tokens are public ingestion tokens, not personal API keys.
# Replace this value with the project token from Project settings before a
# release.  The environment override is useful for local verification.
DEFAULT_POSTHOG_PROJECT_TOKEN = (
    "phc_yWjxK7ibJuFfskDuTr4HbXGdyGvFC6V3Y3fRwNDiRK6x"
)
DEFAULT_POSTHOG_HOST = "https://us.i.posthog.com"

TELEMETRY_SCHEMA_VERSION = 1

CLI_COMMAND_COMPLETED = "agent_strace_cli_command_completed"
SESSION_COMPLETED = "agent_strace_session_completed"
TELEMETRY_ENABLED = "agent_strace_telemetry_enabled"

_CONFIG_ENV = "AGENT_STRACE_TELEMETRY_CONFIG"
_ENABLED_ENV = "AGENT_STRACE_TELEMETRY"
_TOKEN_ENV = "AGENT_STRACE_TELEMETRY_TOKEN"
_HOST_ENV = "AGENT_STRACE_TELEMETRY_HOST"
_TIMEOUT_ENV = "AGENT_STRACE_TELEMETRY_TIMEOUT"

# Hooks and CLI commands must never wait long on another process.
_LOCK_TIMEOUT = 2.0

# PostHog's own session rules: a new session after 30 minutes without an
# event, and no session longer than 24 hours.
_SESSION_IDLE_MS = 30 * 60 * 1000
_SESSION_MAX_MS = 24 * 60 * 60 * 1000
_SESSION_ID_KEY = "analytics_session_id"
_SESSION_LAST_EVENT_KEY = "analytics_session_last_event_ms"
_UUID7_MAX_MS = (1 << 48) - 1
_UUID7 = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)

_TRUE = {"1", "true", "yes", "on", "enabled"}
_FALSE = {"0", "false", "no", "off", "disabled"}
_SAFE_TEXT = re.compile(r"^[A-Za-z0-9_.:/-]{1,80}$")

# Only these properties can leave the process.  Free-form command arguments,
# exception messages, paths, trace data, and trace session IDs are not accepted.
_EVENT_FIELDS: dict[str, dict[str, type]] = {
    CLI_COMMAND_COMPLETED: {
        "command": str,
        "subcommand": str,
        "success": bool,
        "exit_code": int,
        "duration_ms": int,
        "error_type": str,
        "integration": str,
        "export_format": str,
        "backend": str,
    },
    SESSION_COMPLETED: {
        "provider": str,
        "capture_method": str,
        "success": bool,
        "duration_ms": int,
    },
    TELEMETRY_ENABLED: {
        "source": str,
    },
}


def _config_path() -> Path:
    override = os.environ.get(_CONFIG_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".agent-strace" / "telemetry.json"


def _read_config() -> dict[str, Any]:
    path = _config_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _write_config(data: dict[str, Any]) -> bool:
    """Atomically write the config. Use ``_update_config`` for changes."""
    path = _config_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        write_atomic(
            path,
            json.dumps(data, indent=2, sort_keys=True) + "\n",
            mode=0o600,
            allow_symlinks=True,
        )
        return True
    except (OSError, ValueError):
        return False


def _update_config(
    update: Callable[[dict[str, Any]], dict[str, Any] | None],
) -> dict[str, Any] | None:
    """Run one read-modify-write of the config under a cross-process lock.

    *update* gets the current config and returns the dict to write, or None
    to leave the file unchanged.  Returns the config now on disk, or None if
    the lock or the write failed.  Never raises: telemetry is best-effort.
    """
    try:
        path = _config_path()
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with file_lock(path, timeout=_LOCK_TIMEOUT):
            current = _read_config()
            updated = update(current)
            if updated is None:
                return current
            return updated if _write_config(updated) else None
    except Exception:
        return None


def _parse_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    normalised = value.strip().lower()
    if normalised in _TRUE:
        return True
    if normalised in _FALSE:
        return False
    return None


def _in_ci() -> bool:
    return any(
        os.environ.get(name)
        for name in (
            "CI",
            "GITHUB_ACTIONS",
            "GITLAB_CI",
            "BUILDKITE",
            "JENKINS_URL",
            "TF_BUILD",
        )
    )


def consent_state() -> str:
    """Return the persisted preference state: enabled, disabled, or unset."""
    enabled = _read_config().get("enabled")
    if enabled is True:
        return "enabled"
    if enabled is False:
        return "disabled"
    return "unset"


def telemetry_enabled() -> bool:
    """Return whether product telemetry is enabled for this process."""
    if _parse_bool(os.environ.get("DO_NOT_TRACK")) is True:
        return False

    override = _parse_bool(os.environ.get(_ENABLED_ENV))
    if override is not None:
        return override

    # Avoid ephemeral CI/test identifiers and accidental events from the test
    # suite.  CI can opt in explicitly with AGENT_STRACE_TELEMETRY=1.
    if _in_ci() or os.environ.get("PYTEST_CURRENT_TEST"):
        return False

    # An unset preference uses the product default.  Only an explicit opt-out
    # disables telemetry outside CI and test processes.
    return consent_state() != "disabled"


def disclosure_satisfied() -> bool:
    """Return whether the user saw the notice or explicitly enabled telemetry.

    Non-interactive commands never show the notice, and hooks show it only
    where the agent displays hook output to the user (see hooks.py), so an
    install that only runs those may send nothing until one of these is true.
    """
    if _parse_bool(os.environ.get(_ENABLED_ENV)) is True:
        return True
    data = _read_config()
    return data.get("enabled") is True or data.get("notice_shown") is True


def set_telemetry_enabled(enabled: bool) -> bool:
    """Persist the user's telemetry preference.

    Disabling telemetry deletes the anonymous installation identifier and
    the analytics session.  A later re-enable therefore starts with a new
    identity.
    """
    def update(current: dict[str, Any]) -> dict[str, Any]:
        data: dict[str, Any] = {
            "enabled": bool(enabled),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if enabled:
            existing_id = current.get("anonymous_id", "")
            if _valid_anonymous_id(existing_id):
                data["anonymous_id"] = existing_id
                # Same identity, so the current analytics session goes on.
                for key in (_SESSION_ID_KEY, _SESSION_LAST_EVENT_KEY):
                    if key in current:
                        data[key] = current[key]
            else:
                data["anonymous_id"] = uuid.uuid4().hex
        return data

    return _update_config(update) is not None


def _valid_anonymous_id(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{32}", value))


def _anonymous_id() -> str:
    value = _read_config().get("anonymous_id", "")
    if _valid_anonymous_id(value):
        return value
    if not telemetry_enabled():
        return ""

    def assign(current: dict[str, Any]) -> dict[str, Any] | None:
        # Another process may have created an id or opted out since the
        # unlocked read above.  Keep its result instead of overwriting it.
        if _valid_anonymous_id(current.get("anonymous_id")):
            return None
        opted_out = current.get("enabled") is False
        if opted_out and _parse_bool(os.environ.get(_ENABLED_ENV)) is not True:
            return None
        current["anonymous_id"] = uuid.uuid4().hex
        current["updated_at"] = datetime.now(timezone.utc).isoformat()
        return current

    saved = _update_config(assign) or {}
    value = saved.get("anonymous_id", "")
    return value if _valid_anonymous_id(value) else ""


def _uuid7_time(unix_ms: int) -> int:
    """Clamp *unix_ms* to the 48-bit range a UUIDv7 timestamp can hold."""
    return min(max(int(unix_ms), 0), _UUID7_MAX_MS)


def _uuid7(unix_ms: int) -> str:
    """Return a random UUIDv7 whose timestamp is *unix_ms* (RFC 9562).

    A time outside the 48-bit range is clamped, not wrapped, so a pre-epoch
    clock never yields a far-future start.
    """
    value = _uuid7_time(unix_ms) << 80
    value |= 0x7 << 76
    value |= int.from_bytes(os.urandom(10), "big") & ((1 << 76) - 1)
    value &= ~(0x3 << 62)
    value |= 0x2 << 62
    return str(uuid.UUID(int=value))


def _session_start_ms(session_id: Any) -> int | None:
    """Return the start time a UUIDv7 session ID encodes, or None."""
    if not isinstance(session_id, str) or not _UUID7.fullmatch(session_id):
        return None
    return int(session_id[:8] + session_id[9:13], 16)


def _analytics_session_id() -> tuple[str, int | None]:
    """Return the PostHog ``$session_id`` for an event sent now, and now in ms.

    PostHog's server SDKs send no session ID, so its session analytics
    (unique sessions, session duration) see nothing.  This keeps one random
    UUIDv7 per period of activity on this installation, shared by every CLI
    and hook process through the config.  It meets PostHog's rules for a
    custom session ID: it starts at or before the session's first event, and
    rotates after 30 idle minutes or 24 hours.  It is not derived from, and
    cannot be linked to, any trace session.

    The returned time is the one the rotation used, so the event's timestamp
    stays inside the session.  Returns ("", None) when no session can be
    stored; the event is then sent without one.
    """
    result: tuple[str, int | None] = ("", None)

    def touch(current: dict[str, Any]) -> dict[str, Any] | None:
        nonlocal result
        # Disabling deletes the installation ID: a concurrent opt-out must
        # not bring a session back.
        if not _valid_anonymous_id(current.get("anonymous_id")):
            return None
        # Clamped like the UUIDv7 start, and used as the event's timestamp,
        # so a clock outside that range never puts the start after the event.
        now_ms = _uuid7_time(time.time() * 1000)
        session_id = current.get(_SESSION_ID_KEY)
        last_ms = current.get(_SESSION_LAST_EVENT_KEY)
        start_ms = _session_start_ms(session_id)
        if (
            start_ms is None
            or not isinstance(last_ms, int)
            or isinstance(last_ms, bool)
            or now_ms < start_ms
            or now_ms - start_ms >= _SESSION_MAX_MS
            or now_ms - max(last_ms, start_ms) >= _SESSION_IDLE_MS
        ):
            session_id = _uuid7(now_ms)
            last_ms = now_ms
        current[_SESSION_ID_KEY] = session_id
        current[_SESSION_LAST_EVENT_KEY] = max(last_ms, now_ms)
        result = (session_id, now_ms)
        return current

    if _update_config(touch) is None:
        return "", None
    return result


def _project_token() -> str:
    return os.environ.get(_TOKEN_ENV, DEFAULT_POSTHOG_PROJECT_TOKEN).strip()


def _posthog_host() -> str:
    return os.environ.get(_HOST_ENV, DEFAULT_POSTHOG_HOST).strip().rstrip("/")


def telemetry_configured() -> bool:
    """Return whether a PostHog project token is available."""
    token = _project_token()
    return bool(token and "REPLACE" not in token.upper())


def _timeout() -> float:
    try:
        value = float(os.environ.get(_TIMEOUT_ENV, "0.5"))
        return max(0.05, min(value, 2.0))
    except ValueError:
        return 0.5


def _safe_value(value: Any, expected: type) -> Any | None:
    if expected is bool:
        return value if isinstance(value, bool) else None
    if expected is int:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return max(-1, min(int(value), 2_678_400_000))  # up to 31 days in ms
    if expected is str:
        if not isinstance(value, str):
            return None
        value = value.strip()
        return value if _SAFE_TEXT.fullmatch(value) else None
    return None


def _sanitise_properties(event: str, properties: dict[str, Any]) -> dict[str, Any]:
    allowed = _EVENT_FIELDS.get(event, {})
    clean: dict[str, Any] = {}
    for name, expected in allowed.items():
        if name not in properties:
            continue
        value = _safe_value(properties[name], expected)
        if value is not None:
            clean[name] = value
    return clean


def build_payload(
    event: str,
    properties: dict[str, Any],
    anonymous_id: str,
    session_id: str = "",
    timestamp_ms: int | None = None,
) -> dict[str, Any] | None:
    """Build a PostHog capture payload from an allow-listed event.

    *session_id* is added as ``$session_id`` only when it is a UUIDv7.
    *timestamp_ms* defaults to now.
    """
    if event not in _EVENT_FIELDS or not _valid_anonymous_id(anonymous_id):
        return None

    import platform

    clean = _sanitise_properties(event, properties)
    if _session_start_ms(session_id) is not None:
        clean["$session_id"] = session_id
    clean.update({
        "$process_person_profile": False,
        "$geoip_disable": True,
        "$lib": "agent-strace",
        "$lib_version": __version__,
        "telemetry_schema_version": TELEMETRY_SCHEMA_VERSION,
        "agent_strace_version": __version__,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "os": platform.system().lower() or "unknown",
        "ci": _in_ci(),
    })
    timestamp = (
        datetime.now(timezone.utc)
        if timestamp_ms is None
        else datetime.fromtimestamp(timestamp_ms / 1000, timezone.utc)
    )
    return {
        "api_key": _project_token(),
        "event": event,
        "distinct_id": anonymous_id,
        "properties": clean,
        "timestamp": timestamp.isoformat(),
        "uuid": str(uuid.uuid4()),
    }


def capture(event: str, properties: dict[str, Any] | None = None) -> bool:
    """Send one product event, returning False silently on any failure.

    Telemetry must never change command output, exit status, or control flow.
    Every caller goes through here, so the disclosure check is enforced here,
    before the anonymous ID is created.
    """
    try:
        if not telemetry_enabled() or not telemetry_configured():
            return False
        if not disclosure_satisfied():
            return False
        anonymous_id = _anonymous_id()
        if not _valid_anonymous_id(anonymous_id):
            return False
        session_id, timestamp_ms = _analytics_session_id()
        payload = build_payload(
            event, properties or {}, anonymous_id, session_id, timestamp_ms
        )
        if payload is None:
            return False
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request = _urllib_request()
        req = request.Request(
            _posthog_host() + "/i/v0/e/",
            data=body,
            headers={
                "Content-Type": "application/json",
                "User-Agent": f"agent-strace/{__version__}",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=_timeout()) as response:
            return 200 <= getattr(response, "status", 200) < 300
    except Exception:
        return False


_NOTICE_SENTENCES = (
    "Anonymous product telemetry is enabled by default.",
    "No prompts, arguments, paths, repository data, or trace contents are sent.",
    "Disable at any time with: agent-strace telemetry disable",
)
CLI_NOTICE = f"{_NOTICE_SENTENCES[0]} {_NOTICE_SENTENCES[1]}\n{_NOTICE_SENTENCES[2]}\n"
# One line: agent UIs show it among other session messages.
HOOK_NOTICE = "agent-strace: " + " ".join(_NOTICE_SENTENCES)


def notice_due() -> bool:
    """Return whether telemetry is on by default and the notice is unseen."""
    data = _read_config()
    return not (
        isinstance(data.get("enabled"), bool)
        or data.get("notice_shown") is True
        or _parse_bool(os.environ.get(_ENABLED_ENV)) is not None
        or _parse_bool(os.environ.get("DO_NOT_TRACK")) is True
        or _in_ci()
        or os.environ.get("PYTEST_CURRENT_TEST")
        or not telemetry_configured()
    )


def _mark_notice_shown(current: dict[str, Any]) -> dict[str, Any]:
    current["notice_shown"] = True
    current["updated_at"] = datetime.now(timezone.utc).isoformat()
    return current


def maybe_show_telemetry_notice(
    output_stream: TextIO | None = None,
) -> bool:
    """Show the default-on disclosure once on an interactive CLI."""
    output_stream = output_stream or sys.stderr
    if not notice_due():
        return False
    try:
        if not output_stream.isatty():
            return False
        output_stream.write(CLI_NOTICE)
        output_stream.flush()

        def mark_shown(current: dict[str, Any]) -> dict[str, Any] | None:
            if current.get("notice_shown") is True:
                return None
            return _mark_notice_shown(current)

        _update_config(mark_shown)
        return True
    except OSError:
        return False


def show_hook_notice(write: Callable[[str], None]) -> bool:
    """Show the notice once through an agent hook's user-visible output.

    *write* must deliver ``HOOK_NOTICE`` to the user or raise.  The check,
    the write and the mark run under the config lock, so concurrent hooks
    show it once.  ``notice_shown`` is set only after *write* returns; if
    the lock or the write fails, nothing is marked.  Never raises.
    """
    if not notice_due():
        return False
    shown = False

    def show_and_mark(current: dict[str, Any]) -> dict[str, Any] | None:
        nonlocal shown
        if current.get("notice_shown") is True or isinstance(current.get("enabled"), bool):
            return None
        write(HOOK_NOTICE)
        shown = True
        return _mark_notice_shown(current)

    _update_config(show_and_mark)
    return shown


def cmd_telemetry(args: argparse.Namespace) -> int:
    """Manage the anonymous product telemetry preference."""
    action = getattr(args, "telemetry_command", None) or "status"
    if action == "enable":
        if not set_telemetry_enabled(True):
            sys.stderr.write(f"Could not write telemetry preference to {_config_path()}\n")
            return 1
        sys.stdout.write("Anonymous product telemetry enabled.\n")
        if telemetry_configured():
            capture(TELEMETRY_ENABLED, {"source": "cli"})
        else:
            sys.stdout.write(
                "PostHog is not configured in this build; no events will be sent.\n"
            )
        return 0

    if action == "disable":
        if not set_telemetry_enabled(False):
            sys.stderr.write(f"Could not write telemetry preference to {_config_path()}\n")
            return 1
        sys.stdout.write("Anonymous product telemetry disabled; the local anonymous ID was deleted.\n")
        return 0

    state = consent_state()
    effective = "enabled" if telemetry_enabled() else "disabled"
    source = "stored preference"
    if _parse_bool(os.environ.get("DO_NOT_TRACK")) is True:
        source = "DO_NOT_TRACK"
    elif _parse_bool(os.environ.get(_ENABLED_ENV)) is not None:
        source = _ENABLED_ENV
    elif _in_ci():
        source = "CI default"
    elif state == "unset":
        source = "enabled by default"

    pending = ""
    if effective == "enabled" and telemetry_configured() and not disclosure_satisfied():
        pending = (
            "Waiting for the one-time notice: nothing is sent until an "
            "interactive command or an interactive Claude Code, Codex, or "
            "Gemini CLI session shows it, or telemetry is explicitly enabled.\n"
        )

    sys.stdout.write(
        f"Anonymous product telemetry: {effective} ({source})\n"
        f"{pending}"
        f"Stored preference: {state}\n"
        f"PostHog destination: "
        f"{_posthog_host() if telemetry_configured() else 'not configured'}\n"
        "Collected: command/subcommand, success, duration, integration/export format, "
        "agent-strace/Python version, OS, CI flag, a random installation ID, and a "
        "random analytics session ID that rotates after 30 idle minutes.\n"
        "Never collected: prompts, responses, arguments, paths, repository data, "
        "trace contents, trace session IDs, or exception messages.\n"
    )
    return 0


def monotonic_ms_since(started_at: float) -> int:
    """Return a non-negative elapsed duration in milliseconds."""
    return max(0, int((time.monotonic() - started_at) * 1000))
