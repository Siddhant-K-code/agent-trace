"""Per-project rolling budget guardrails.

Reads the ``budget`` block from ``.agent-strace.yaml`` and evaluates local
session spend over a rolling seven-day window. The module is intentionally
stdlib-only so budget enforcement stays available in the core package.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, TextIO

from .cost import estimate_cost
from .store import TraceStore


DEFAULT_CONFIG_PATHS = (
    ".agent-strace.yaml",
    ".agent-strace.yml",
    ".agent-strace.json",
)
ROLLING_WINDOW_SECONDS = 7 * 24 * 60 * 60


@dataclass(frozen=True)
class ProjectBudgetConfig:
    """Config for a per-project rolling cost budget."""

    weekly: float = 0.0
    warn_at: float = 0.80
    stop_at: float = 1.00
    per_session_max: float | None = None

    @property
    def enabled(self) -> bool:
        return self.weekly > 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectBudgetConfig":
        budget = data.get("budget", data)
        if not isinstance(budget, dict):
            return cls()

        weekly = _to_float(budget.get("weekly"), 0.0)
        warn_at = _to_float(budget.get("warn_at"), 0.80)
        stop_at = _to_float(budget.get("stop_at"), 1.00)
        per_session_raw = budget.get("per_session_max")
        per_session_max = (
            None if per_session_raw in (None, "", "null", "~")
            else _to_float(per_session_raw, 0.0)
        )
        if per_session_max is not None and per_session_max <= 0:
            per_session_max = None

        return cls(
            weekly=max(0.0, weekly),
            warn_at=max(0.0, warn_at),
            stop_at=max(0.0, stop_at),
            per_session_max=per_session_max,
        )


@dataclass(frozen=True)
class ProjectBudgetStatus:
    """Current spend against the configured rolling budget."""

    config: ProjectBudgetConfig
    spent: float
    window_start: float
    window_end: float

    @property
    def weekly(self) -> float:
        return self.config.weekly

    @property
    def warn_amount(self) -> float:
        return self.weekly * self.config.warn_at

    @property
    def stop_amount(self) -> float:
        return self.weekly * self.config.stop_at

    @property
    def percent(self) -> float:
        if self.weekly <= 0:
            return 0.0
        return self.spent / self.weekly

    @property
    def should_warn(self) -> bool:
        return self.config.enabled and self.spent >= self.warn_amount

    @property
    def should_stop(self) -> bool:
        return self.config.enabled and self.spent >= self.stop_amount


def load_project_budget_config(
    config_path: str | Path | None = None,
) -> ProjectBudgetConfig:
    """Load project budget config from JSON or minimal YAML.

    If ``config_path`` is omitted, the standard project config filenames are
    checked in order. Missing or invalid files fall back to a disabled config.
    """

    for path in _candidate_paths(config_path):
        if not path.exists():
            continue
        try:
            text = path.read_text()
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = _parse_simple_yaml(text)
            return ProjectBudgetConfig.from_dict(data)
        except Exception:
            return ProjectBudgetConfig()
    return ProjectBudgetConfig()


def rolling_project_spend(
    store: TraceStore,
    now: float | None = None,
    window_seconds: float = ROLLING_WINDOW_SECONDS,
    exclude_session_id: str = "",
) -> float:
    """Estimate spend for sessions started in the rolling project window."""

    window_end = time.time() if now is None else now
    window_start = window_end - window_seconds
    total = 0.0

    for meta in store.list_sessions():
        if meta.session_id == exclude_session_id:
            continue
        if not (window_start <= meta.started_at < window_end):
            continue
        try:
            total += estimate_cost(store, meta.session_id).total_cost
        except Exception:
            continue
    return total


def project_budget_status(
    store: TraceStore,
    config: ProjectBudgetConfig,
    now: float | None = None,
    current_session_spend: float = 0.0,
    exclude_session_id: str = "",
) -> ProjectBudgetStatus:
    """Return current rolling budget status."""

    window_end = time.time() if now is None else now
    window_start = window_end - ROLLING_WINDOW_SECONDS
    spent = rolling_project_spend(
        store,
        now=window_end,
        exclude_session_id=exclude_session_id,
    ) + max(0.0, current_session_spend)
    return ProjectBudgetStatus(
        config=config,
        spent=spent,
        window_start=window_start,
        window_end=window_end,
    )


def format_project_budget_status(status: ProjectBudgetStatus) -> str:
    """Format a concise status line for terminal/watch alerts."""

    pct = status.percent * 100
    return (
        f"ProjectBudget: ${status.spent:.2f} spent in rolling 7 days "
        f"({pct:.0f}% of ${status.weekly:.2f})"
    )


def warn_if_budget_near_limit(
    store: TraceStore,
    config: ProjectBudgetConfig,
    out: TextIO,
    now: float | None = None,
) -> ProjectBudgetStatus:
    """Print a warning when the rolling budget is at or above warn_at."""

    status = project_budget_status(store, config, now=now)
    if status.should_warn:
        out.write(f"{format_project_budget_status(status)}\n")
        out.flush()
    return status


def enforce_new_session_budget(
    store: TraceStore,
    config: ProjectBudgetConfig,
    out: TextIO,
    now: float | None = None,
) -> bool:
    """Return False and print an error when a new session should be refused."""

    status = project_budget_status(store, config, now=now)
    if not status.should_stop:
        return True

    out.write(
        f"agent-strace: weekly project budget exceeded; refusing new session "
        f"(${status.spent:.2f} spent, stop threshold ${status.stop_amount:.2f})\n"
    )
    out.flush()
    return False


def _candidate_paths(config_path: str | Path | None) -> Iterable[Path]:
    if config_path:
        yield Path(config_path)
        return
    for path in DEFAULT_CONFIG_PATHS:
        yield Path(path)


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse top-level scalar keys and one level of nested scalar mappings."""

    result: dict[str, Any] = {}
    current_section: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        stripped = line.lstrip()
        if not stripped:
            continue

        indent = len(line) - len(stripped)
        if indent == 0:
            current_section = None
            if stripped.endswith(":"):
                current_section = stripped[:-1].strip()
                result[current_section] = {}
            elif ":" in stripped:
                key, _, raw_value = stripped.partition(":")
                result[key.strip()] = _coerce(raw_value.strip())
        elif current_section and ":" in stripped:
            section = result.get(current_section)
            if isinstance(section, dict):
                key, _, raw_value = stripped.partition(":")
                section[key.strip()] = _coerce(raw_value.strip())

    return result


def _coerce(value: str) -> Any:
    if value in ("", "null", "~"):
        return None
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    return value


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
