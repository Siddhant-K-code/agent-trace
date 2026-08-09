"""Privacy-preserving product telemetry for the agent-strace CLI.

Telemetry is disabled until the user explicitly opts in.  When enabled, the
client sends a small allow-listed event directly to PostHog's public capture
API.  It never reads or sends trace events, prompts, command arguments, file
paths, repository metadata, or session identifiers.

The PostHog project token is intentionally a public, write-only ingestion
token.  Set ``DEFAULT_POSTHOG_PROJECT_TOKEN`` before publishing a release, or
use ``AGENT_STRACE_TELEMETRY_TOKEN`` while developing and testing.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO
from urllib import request

from . import __version__


# PostHog project tokens are public ingestion tokens, not personal API keys.
# Replace this value with the project token from Project settings before a
# release.  The environment override is useful for local verification.
DEFAULT_POSTHOG_PROJECT_TOKEN = ""
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

_TRUE = {"1", "true", "yes", "on", "enabled"}
_FALSE = {"0", "false", "no", "off", "disabled"}
_SAFE_TEXT = re.compile(r"^[A-Za-z0-9_.:/-]{1,80}$")

# Only these properties can leave the process.  Free-form command arguments,
# exception messages, paths, trace data, and session IDs are not accepted.
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
    path = _config_path()
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        try:
            tmp.chmod(0o600)
        except OSError:
            pass
        tmp.replace(path)
        return True
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return False


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
    """Return the persisted consent state: enabled, disabled, or unset."""
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

    return consent_state() == "enabled"


def set_telemetry_enabled(enabled: bool) -> bool:
    """Persist the user's telemetry preference.

    Disabling telemetry deletes the anonymous installation identifier.  A
    later opt-in therefore starts with a new identity.
    """
    current = _read_config()
    data: dict[str, Any] = {
        "enabled": bool(enabled),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if enabled:
        existing_id = current.get("anonymous_id", "")
        data["anonymous_id"] = (
            existing_id if _valid_anonymous_id(existing_id) else uuid.uuid4().hex
        )
    return _write_config(data)


def _valid_anonymous_id(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{32}", value))


def _anonymous_id() -> str:
    data = _read_config()
    value = data.get("anonymous_id", "")
    if _valid_anonymous_id(value):
        return value
    if not telemetry_enabled():
        return ""
    value = uuid.uuid4().hex
    data["anonymous_id"] = value
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    return value if _write_config(data) else ""


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
) -> dict[str, Any] | None:
    """Build a PostHog capture payload from an allow-listed event."""
    if event not in _EVENT_FIELDS or not _valid_anonymous_id(anonymous_id):
        return None

    clean = _sanitise_properties(event, properties)
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
    return {
        "api_key": _project_token(),
        "event": event,
        "distinct_id": anonymous_id,
        "properties": clean,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uuid": str(uuid.uuid4()),
    }


def capture(event: str, properties: dict[str, Any] | None = None) -> bool:
    """Send one product event, returning False silently on any failure.

    Telemetry must never change command output, exit status, or control flow.
    """
    try:
        if not telemetry_enabled() or not telemetry_configured():
            return False
        anonymous_id = _anonymous_id()
        payload = build_payload(event, properties or {}, anonymous_id)
        if payload is None:
            return False
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
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


def maybe_prompt_for_consent(
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
) -> bool:
    """Prompt once on an interactive CLI, returning True if a choice was made."""
    input_stream = input_stream or sys.stdin
    output_stream = output_stream or sys.stderr
    if (
        consent_state() != "unset"
        or os.environ.get(_ENABLED_ENV) is not None
        or _parse_bool(os.environ.get("DO_NOT_TRACK")) is True
        or _in_ci()
        or os.environ.get("PYTEST_CURRENT_TEST")
        or not telemetry_configured()
    ):
        return False
    try:
        if not input_stream.isatty() or not output_stream.isatty():
            return False
        output_stream.write(
            "Help improve agent-strace by sending anonymous CLI usage?\n"
            "No prompts, arguments, paths, repository data, or trace contents are sent. "
            "[y/N] "
        )
        output_stream.flush()
        answer = input_stream.readline().strip().lower()
        enabled = answer in {"y", "yes"}
        set_telemetry_enabled(enabled)
        if enabled:
            capture(TELEMETRY_ENABLED, {"source": "first_run_prompt"})
            output_stream.write("Anonymous telemetry enabled. Disable with: agent-strace telemetry disable\n")
        else:
            output_stream.write("Anonymous telemetry disabled. Enable with: agent-strace telemetry enable\n")
        return True
    except (OSError, EOFError):
        return False


def cmd_telemetry(args: argparse.Namespace) -> int:
    """Manage anonymous product telemetry consent."""
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
    elif state == "unset":
        source = "no consent recorded"
    elif _in_ci():
        source = "CI default"

    sys.stdout.write(
        f"Anonymous product telemetry: {effective} ({source})\n"
        f"Stored consent: {state}\n"
        f"PostHog destination: "
        f"{_posthog_host() if telemetry_configured() else 'not configured'}\n"
        "Collected: command/subcommand, success, duration, integration/export format, "
        "agent-strace/Python version, OS, and CI flag.\n"
        "Never collected: prompts, responses, arguments, paths, repository data, "
        "trace contents, session IDs, or exception messages.\n"
    )
    return 0


def monotonic_ms_since(started_at: float) -> int:
    """Return a non-negative elapsed duration in milliseconds."""
    return max(0, int((time.monotonic() - started_at) * 1000))
