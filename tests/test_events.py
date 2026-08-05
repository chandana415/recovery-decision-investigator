from pathlib import Path

from recovery_workspace.events import normalize_events
from recovery_workspace.parser import parse_scenario


def test_normalize_events_derives_semantic_fields_before_investigation_context():
    scenario = parse_scenario(SCENARIO_PATH)
    events = normalize_events(scenario.logs)

    storage_event = next(event for event in events if event.component == "storage" and event.level == "ERROR")

    assert storage_event.semantic_error_code == "QUOTA_EXCEEDED"
    assert storage_event.cause_type == "STORAGE_CAPACITY"
    assert storage_event.event_type == "TERMINAL_FAILURE"

SCENARIO_PATH = (
    Path(__file__).parent.parent / "mock-data" / "scenarios" / "storage-quota-exceeded.json"
)


def test_normalize_events_is_1_to_1_with_logs():
    scenario = parse_scenario(SCENARIO_PATH)
    events = normalize_events(scenario.logs)

    assert len(events) == len(scenario.logs)


def test_normalize_events_preserves_field_values_and_order():
    scenario = parse_scenario(SCENARIO_PATH)
    events = normalize_events(scenario.logs)

    for log, event in zip(scenario.logs, events):
        assert event.timestamp == log.timestamp
        assert event.correlation_id == log.correlation_id
        assert event.trace_id == log.trace_id
        assert event.component_id == log.component_id
        assert event.component == log.component
        assert event.level == log.level
        assert event.error_code == log.error_code
        assert event.message == log.message


def test_normalize_events_returns_event_instances_not_log_entries():
    scenario = parse_scenario(SCENARIO_PATH)
    events = normalize_events(scenario.logs)

    from recovery_workspace.models import Event

    assert all(isinstance(event, Event) for event in events)
