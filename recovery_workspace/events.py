from __future__ import annotations

from recovery_workspace.models import Event, LogEntry
from recovery_workspace.semantics import apply_event_semantics


def normalize_events(logs: list[LogEntry]) -> list[Event]:
    events = [Event.from_log_entry(log) for log in logs]
    return [apply_event_semantics(event) for event in events]
