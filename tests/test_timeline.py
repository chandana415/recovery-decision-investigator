from pathlib import Path

import pytest

from recovery_workspace.events import normalize_events
from recovery_workspace.models import Event
from recovery_workspace.parser import parse_scenario
from recovery_workspace.timeline import build_timeline

SCENARIO_PATH = (
    Path(__file__).parent.parent / "mock-data" / "scenarios" / "storage-quota-exceeded.json"
)


def _load_events() -> list[Event]:
    scenario = parse_scenario(SCENARIO_PATH)
    return normalize_events(scenario.logs)


def test_timeline_entry_count_matches_events():
    events = _load_events()
    timeline = build_timeline(events)

    assert len(timeline.entries) == len(events)


def test_timeline_entries_are_chronologically_ordered():
    events = _load_events()
    timeline = build_timeline(events)

    timestamps = [entry.event.timestamp for entry in timeline.entries]
    assert timestamps == sorted(timestamps)


def test_timeline_sequence_numbers_are_1_indexed_and_contiguous():
    events = _load_events()
    timeline = build_timeline(events)

    assert [entry.sequence for entry in timeline.entries] == list(range(1, len(events) + 1))


def test_timeline_bounds_and_duration():
    events = _load_events()
    timeline = build_timeline(events)

    assert timeline.correlation_id == "corr-b19c44d0"
    assert timeline.started_at == min(e.timestamp for e in events)
    assert timeline.ended_at == max(e.timestamp for e in events)
    # 2026-06-28T04:00:00Z -> 2026-06-28T10:03:20Z
    assert timeline.duration_seconds == 21800.0


def test_first_entry_has_zero_gap():
    events = _load_events()
    timeline = build_timeline(events)

    assert timeline.entries[0].gap_before_seconds == 0.0


def test_gap_before_seconds_matches_consecutive_timestamp_deltas():
    events = _load_events()
    timeline = build_timeline(events)

    for previous, current in zip(timeline.entries, timeline.entries[1:]):
        expected_gap = (current.event.timestamp - previous.event.timestamp).total_seconds()
        assert current.gap_before_seconds == expected_gap


def test_build_timeline_rejects_mixed_correlation_ids():
    events = _load_events()
    tampered = events[0].model_copy(update={"correlation_id": "corr-different"})

    with pytest.raises(ValueError, match="multiple correlation IDs"):
        build_timeline([tampered, events[1]])


def test_build_timeline_rejects_empty_input():
    with pytest.raises(ValueError, match="zero events"):
        build_timeline([])
