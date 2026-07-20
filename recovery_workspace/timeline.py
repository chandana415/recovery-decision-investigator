from __future__ import annotations

from recovery_workspace.models import Event, Timeline, TimelineEntry


def build_timeline(events: list[Event]) -> Timeline:
    if not events:
        raise ValueError("cannot build a timeline from zero events")

    correlation_ids = {event.correlation_id for event in events}
    if len(correlation_ids) > 1:
        raise ValueError(f"events span multiple correlation IDs: {sorted(correlation_ids)}")

    ordered = sorted(events, key=lambda event: event.timestamp)

    entries: list[TimelineEntry] = []
    previous_timestamp = ordered[0].timestamp
    for sequence, event in enumerate(ordered, start=1):
        gap_before_seconds = (event.timestamp - previous_timestamp).total_seconds()
        entries.append(
            TimelineEntry(sequence=sequence, event=event, gap_before_seconds=gap_before_seconds)
        )
        previous_timestamp = event.timestamp

    return Timeline(
        correlation_id=ordered[0].correlation_id,
        started_at=ordered[0].timestamp,
        ended_at=ordered[-1].timestamp,
        duration_seconds=(ordered[-1].timestamp - ordered[0].timestamp).total_seconds(),
        entries=entries,
    )
