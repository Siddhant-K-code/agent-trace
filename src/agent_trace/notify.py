"""Slack, Microsoft Teams, and generic webhook notifications.

Sends structured notifications on watch violations, session end, and
budget alerts. All HTTP — no third-party SDKs required.

Configuration (env vars):
    AGENT_STRACE_SLACK_WEBHOOK      Slack incoming webhook URL
    AGENT_STRACE_TEAMS_WEBHOOK      Microsoft Teams incoming webhook URL
    AGENT_STRACE_NOTIFY_WEBHOOK     Generic JSON webhook URL (fallback)

Usage:
    from agent_trace.notify import notify

    notify(
        title="Policy violation",
        message="Bash tool blocked by scope policy",
        level="error",          # info | warning | error
        session_id="abc123",
        fields={"tool": "Bash", "policy": "no-network"},
    )
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Literal

Level = Literal["info", "warning", "error"]

# Env var names
_SLACK_ENV = "AGENT_STRACE_SLACK_WEBHOOK"
_TEAMS_ENV = "AGENT_STRACE_TEAMS_WEBHOOK"
_GENERIC_ENV = "AGENT_STRACE_NOTIFY_WEBHOOK"

# Slack colour map
_SLACK_COLORS: dict[str, str] = {
    "info": "#36a64f",
    "warning": "#ff9900",
    "error": "#cc0000",
}

# Retry config
_MAX_RETRIES = 3
_RETRY_DELAY = 1.0  # seconds, doubled each attempt


# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------

def _slack_payload(
    title: str,
    message: str,
    level: Level = "info",
    session_id: str = "",
    fields: dict[str, str] | None = None,
) -> dict:
    """Build a Slack Block Kit message payload."""
    color = _SLACK_COLORS.get(level, "#36a64f")
    attachment: dict = {
        "color": color,
        "title": title,
        "text": message,
        "footer": "agent-strace",
        "ts": int(time.time()),
    }
    if fields:
        attachment["fields"] = [
            {"title": k, "value": str(v), "short": True}
            for k, v in fields.items()
        ]
    if session_id:
        attachment["footer_icon"] = ""
        attachment["fields"] = attachment.get("fields", []) + [
            {"title": "session", "value": session_id[:12], "short": True}
        ]
    return {"attachments": [attachment]}


def _teams_payload(
    title: str,
    message: str,
    level: Level = "info",
    session_id: str = "",
    fields: dict[str, str] | None = None,
) -> dict:
    """Build a Microsoft Teams Adaptive Card payload (legacy connector format)."""
    theme_color = {"info": "0076D7", "warning": "FF9900", "error": "CC0000"}.get(level, "0076D7")
    facts = []
    if session_id:
        facts.append({"name": "Session", "value": session_id[:12]})
    if fields:
        facts.extend({"name": k, "value": str(v)} for k, v in fields.items())

    payload: dict = {
        "@type": "MessageCard",
        "@context": "http://schema.org/extensions",
        "themeColor": theme_color,
        "summary": title,
        "sections": [
            {
                "activityTitle": f"**{title}**",
                "activityText": message,
                "facts": facts,
                "markdown": True,
            }
        ],
    }
    return payload


def _generic_payload(
    title: str,
    message: str,
    level: Level = "info",
    session_id: str = "",
    fields: dict[str, str] | None = None,
) -> dict:
    return {
        "title": title,
        "message": message,
        "level": level,
        "session_id": session_id,
        "fields": fields or {},
        "timestamp": time.time(),
        "source": "agent-strace",
    }


# ---------------------------------------------------------------------------
# HTTP delivery with retry
# ---------------------------------------------------------------------------

def _post(url: str, payload: dict, retries: int = _MAX_RETRIES) -> bool:
    """POST JSON payload to url. Returns True on success."""
    body = json.dumps(payload).encode("utf-8")
    delay = _RETRY_DELAY
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status in (200, 202, 204):
                    return True
        except urllib.error.HTTPError as exc:
            if exc.code < 500:
                # 4xx — don't retry
                import sys
                sys.stderr.write(f"[notify] HTTP {exc.code} from {url}: {exc.reason}\n")
                return False
        except Exception:
            pass
        if attempt < retries - 1:
            time.sleep(delay)
            delay *= 2
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def notify(
    title: str,
    message: str,
    level: Level = "info",
    session_id: str = "",
    fields: dict[str, str] | None = None,
    slack_url: str = "",
    teams_url: str = "",
    webhook_url: str = "",
) -> dict[str, bool]:
    """Send a notification to all configured channels.

    URL priority: explicit argument > env var.
    Returns a dict of {channel: success} for each attempted delivery.
    """
    results: dict[str, bool] = {}

    slack = slack_url or os.environ.get(_SLACK_ENV, "")
    teams = teams_url or os.environ.get(_TEAMS_ENV, "")
    generic = webhook_url or os.environ.get(_GENERIC_ENV, "")

    if slack:
        payload = _slack_payload(title, message, level, session_id, fields)
        results["slack"] = _post(slack, payload)

    if teams:
        payload = _teams_payload(title, message, level, session_id, fields)
        results["teams"] = _post(teams, payload)

    if generic and not (slack or teams):
        # Only fall back to generic if no provider-specific URL is set
        payload = _generic_payload(title, message, level, session_id, fields)
        results["webhook"] = _post(generic, payload)

    return results


def notify_violation(
    rule_name: str,
    tool_name: str,
    session_id: str = "",
    action: str = "alert",
    **kwargs,
) -> dict[str, bool]:
    """Convenience wrapper for watch rule violations."""
    return notify(
        title=f"Agent policy violation: {rule_name}",
        message=f"Rule `{rule_name}` fired on tool `{tool_name}` — action: {action}",
        level="error",
        session_id=session_id,
        fields={"rule": rule_name, "tool": tool_name, "action": action},
        **kwargs,
    )


def notify_budget_alert(
    agent_name: str,
    spent: float,
    budget: float,
    session_id: str = "",
    **kwargs,
) -> dict[str, bool]:
    """Convenience wrapper for budget threshold alerts."""
    pct = int(spent / budget * 100) if budget else 0
    return notify(
        title=f"Budget alert: {agent_name}",
        message=f"Agent `{agent_name}` has spent ${spent:.4f} ({pct}% of ${budget:.2f} budget)",
        level="warning",
        session_id=session_id,
        fields={"agent": agent_name, "spent": f"${spent:.4f}", "budget": f"${budget:.2f}"},
        **kwargs,
    )


def notify_session_end(
    agent_name: str,
    session_id: str,
    cost: float,
    tool_calls: int,
    duration_ms: float,
    **kwargs,
) -> dict[str, bool]:
    """Convenience wrapper for session-end summaries."""
    return notify(
        title=f"Session complete: {agent_name}",
        message=f"Session `{session_id[:12]}` finished — {tool_calls} tool calls, ${cost:.4f}, {duration_ms/1000:.1f}s",
        level="info",
        session_id=session_id,
        fields={
            "agent": agent_name,
            "tool_calls": str(tool_calls),
            "cost": f"${cost:.4f}",
            "duration": f"{duration_ms/1000:.1f}s",
        },
        **kwargs,
    )
