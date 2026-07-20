from __future__ import annotations

from recovery_workspace.models import Event, LogEntry


def normalize_events(logs: list[LogEntry]) -> list[Event]:
    return [Event.from_log_entry(log) for log in logs]
