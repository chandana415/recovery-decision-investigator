from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from recovery_workspace.events import normalize_events
from recovery_workspace.models import Event, LogEntry, Scenario


def parse_log_entry(raw: dict[str, Any]) -> LogEntry:
    """Validate a single raw log record and convert it to a LogEntry."""
    return LogEntry.model_validate(raw)


def parse_logs(raw_logs: list[dict[str, Any]]) -> list[LogEntry]:
    """Parse a list of raw log records into structured LogEntry objects."""
    return [parse_log_entry(raw_log) for raw_log in raw_logs]


def parse_events(raw_logs: list[dict[str, Any]]) -> list[Event]:
    """Parse raw log records and normalize them into Event objects."""
    return normalize_events(parse_logs(raw_logs))


def parse_scenario(path: str | Path) -> Scenario:
    raw = json.loads(Path(path).read_text())
    return Scenario.model_validate(raw)
