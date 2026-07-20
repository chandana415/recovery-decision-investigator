import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from recovery_workspace.models import Event, LogEntry
from recovery_workspace.parser import parse_events, parse_logs, parse_scenario

SCENARIO_PATH = (
    Path(__file__).parent.parent / "mock-data" / "scenarios" / "storage-quota-exceeded.json"
)


def test_parses_scenario_metadata():
    scenario = parse_scenario(SCENARIO_PATH)

    assert scenario.scenario_id == "storage-quota-exceeded"
    assert scenario.customer == "Initech LLC"
    assert scenario.job_id == "job-weekly-initech-15029"
    assert scenario.correlation_id == "corr-b19c44d0"
    assert scenario.components_involved == ["backup_service", "storage", "monitoring"]
    assert scenario.expected_outcome == "FAILED"


def test_parses_all_log_entries():
    scenario = parse_scenario(SCENARIO_PATH)

    assert len(scenario.logs) == 8


def test_log_entry_field_types():
    scenario = parse_scenario(SCENARIO_PATH)
    first = scenario.logs[0]

    assert first.timestamp.isoformat() == "2026-06-28T04:00:00+00:00"
    assert first.correlation_id == "corr-b19c44d0"
    assert first.component == "monitoring"
    assert first.component_id == "comp-monitoring"
    assert first.level == "WARN"
    assert first.error_code == "QUOTA_WARNING"
    assert first.trace_id is None


def test_log_entries_preserve_source_order():
    scenario = parse_scenario(SCENARIO_PATH)
    messages = [log.message for log in scenario.logs]

    assert messages[0].startswith("Storage account initech-backups")
    assert messages[-1].startswith("Partial blob cleanup completed")


def test_error_code_is_none_for_info_logs():
    scenario = parse_scenario(SCENARIO_PATH)
    info_logs = [log for log in scenario.logs if log.level == "INFO"]

    assert len(info_logs) == 4
    assert all(log.error_code is None for log in info_logs)


def test_terminal_error_logs_have_error_codes():
    scenario = parse_scenario(SCENARIO_PATH)
    error_logs = [log for log in scenario.logs if log.level == "ERROR"]

    assert [log.error_code for log in error_logs] == ["QUOTA_EXCEEDED", "QUOTA_EXCEEDED", "JOB_FAILED"]


def test_parse_logs_converts_raw_dicts_to_log_entry_instances():
    raw_scenario = json.loads(Path(SCENARIO_PATH).read_text())
    raw_logs = raw_scenario["logs"]

    log_entries = parse_logs(raw_logs)

    assert len(log_entries) == len(raw_logs)
    assert all(isinstance(entry, LogEntry) for entry in log_entries)
    assert [entry.message for entry in log_entries] == [raw_log["message"] for raw_log in raw_logs]


def test_parse_events_converts_raw_dicts_to_event_instances():
    raw_scenario = json.loads(Path(SCENARIO_PATH).read_text())
    raw_logs = raw_scenario["logs"]

    events = parse_events(raw_logs)

    assert len(events) == len(raw_logs)
    assert all(isinstance(event, Event) for event in events)
    assert [event.message for event in events] == [raw_log["message"] for raw_log in raw_logs]


def test_parse_logs_rejects_invalid_log_records():
    invalid_raw_logs = [
        {
            "timestamp": "2026-06-28T04:00:00Z",
            "correlationId": "corr-b19c44d0",
            "componentId": "comp-monitoring",
            "component": "monitoring",
            "level": "UNKNOWN_LEVEL",
            "errorCode": "QUOTA_WARNING",
            "message": "Invalid log level",
        }
    ]

    with pytest.raises(ValidationError):
        parse_logs(invalid_raw_logs)
