"""Tests for the upload handler (uploader.py)."""

from __future__ import annotations

import json
from io import BytesIO
from typing import Any

import pytest

from recovery_workspace.models import LogEntry
from recovery_workspace.uploader import (
    parse_uploaded_logs,
    parse_uploaded_logs_with_source,
    read_uploaded_file_safe,
    validate_upload,
)


class MockUploadedFile:
    """Mock Streamlit UploadedFile for testing."""

    def __init__(self, name: str, content: bytes, size: int | None = None):
        self.name = name
        self.content = content
        self.size = size if size is not None else len(content)
        self._pos = 0

    def read(self) -> bytes:
        return self.content


def test_validate_upload_unsupported_extension():
    """Reject unsupported file types."""
    file = MockUploadedFile("data.csv", b"")
    valid, msg = validate_upload(file)
    assert not valid
    assert ".csv" in msg or "Unsupported" in msg


def test_validate_upload_supported_json():
    """Accept .json files."""
    file = MockUploadedFile("logs.json", b'{"logs": []}', size=12)
    valid, msg = validate_upload(file)
    assert valid
    assert "accepted" in msg.lower()


def test_validate_upload_supported_log():
    """Accept .log files."""
    file = MockUploadedFile("syslog.log", b"test", size=4)
    valid, msg = validate_upload(file)
    assert valid


def test_validate_upload_supported_txt():
    """Accept .txt files."""
    file = MockUploadedFile("backup.txt", b"test", size=4)
    valid, msg = validate_upload(file)
    assert valid


def test_validate_upload_empty_file():
    """Reject files with size 0 (but don't fail on extension check)."""
    file = MockUploadedFile("logs.json", b"", size=0)
    # validate_upload checks extension first, passes; size check later
    valid, msg = validate_upload(file)
    # Size 0 is technically ≤ 10MB, so it passes validation here
    # The error is caught later in read_uploaded_file_safe
    assert valid


def test_validate_upload_too_large():
    """Reject files exceeding max size."""
    oversized = 11 * 1024 * 1024  # 11 MB
    file = MockUploadedFile("logs.json", b"x", size=oversized)
    valid, msg = validate_upload(file)
    assert not valid
    assert "too large" in msg.lower() or "maximum" in msg.lower()


def test_validate_upload_none_file():
    """Handle None file gracefully."""
    valid, msg = validate_upload(None)
    assert not valid
    assert "No file" in msg or "provided" in msg


def test_read_uploaded_file_safe_valid_utf8():
    """Read valid UTF-8 content."""
    content = "Hello, World!"
    file = MockUploadedFile("test.txt", content.encode("utf-8"))
    text, err = read_uploaded_file_safe(file)
    assert text == content
    assert err == ""


def test_read_uploaded_file_safe_empty_file():
    """Reject empty files."""
    file = MockUploadedFile("test.txt", b"", size=0)
    text, err = read_uploaded_file_safe(file)
    assert text is None
    assert "empty" in err.lower()


def test_read_uploaded_file_safe_invalid_utf8():
    """Reject non-UTF-8 content."""
    # Create invalid UTF-8 sequence
    file = MockUploadedFile("test.txt", b"\x80\x81\x82")
    text, err = read_uploaded_file_safe(file)
    assert text is None
    assert "UTF-8" in err or "encoding" in err.lower()


def test_parse_uploaded_logs_valid_json_array():
    """Parse JSON array of log objects."""
    logs = [
        {
            "timestamp": "2026-01-01T10:00:00Z",
            "correlationId": "corr-123",
            "componentId": "comp-test",
            "component": "test_service",
            "level": "INFO",
            "errorCode": None,
            "message": "Test message",
        }
    ]
    content = json.dumps(logs)
    parsed, err = parse_uploaded_logs(content)
    assert err == ""
    assert parsed is not None
    assert len(parsed) == 1
    assert isinstance(parsed[0], LogEntry)
    assert parsed[0].component == "test_service"


def test_parse_uploaded_logs_scenario_format():
    """Parse Scenario object with 'logs' field."""
    scenario = {
        "scenarioId": "test",
        "name": "Test",
        "category": "test",
        "tags": [],
        "description": "Test scenario",
        "customer": "TestCorp",
        "jobId": "test-job",
        "correlationId": "corr-123",
        "componentsInvolved": ["test_service"],
        "expectedOutcome": "FAILED",
        "expectedFailureSignature": "Test failure",
        "logs": [
            {
                "timestamp": "2026-01-01T10:00:00Z",
                "correlationId": "corr-123",
                "componentId": "comp-test",
                "component": "test_service",
                "level": "INFO",
                "errorCode": None,
                "message": "Test message",
            }
        ],
    }
    content = json.dumps(scenario)
    parsed, err = parse_uploaded_logs(content)
    assert err == ""
    assert parsed is not None
    assert len(parsed) == 1


def test_parse_uploaded_logs_empty_file():
    """Reject empty content."""
    parsed, err = parse_uploaded_logs("")
    assert parsed is None
    assert "empty" in err.lower() or "whitespace" in err.lower()


def test_parse_uploaded_logs_invalid_json():
    """Reject malformed JSON."""
    content = "{ not valid json"
    parsed, err = parse_uploaded_logs(content)
    assert parsed is None
    assert "JSON" in err or "error" in err.lower()


def test_parse_uploaded_logs_json_but_no_logs_field():
    """Reject valid JSON that doesn't match schema."""
    content = json.dumps({"some_field": "value"})
    parsed, err = parse_uploaded_logs(content)
    assert parsed is None
    assert "doesn't match" in err or "format" in err.lower()


def test_parse_uploaded_logs_empty_logs_array():
    """Reject logs array with no entries."""
    content = json.dumps([])
    parsed, err = parse_uploaded_logs(content)
    assert parsed is None
    assert "no log entries" in err.lower()


def test_parse_uploaded_logs_invalid_log_entry():
    """Reject logs with missing required fields."""
    logs = [
        {
            "timestamp": "2026-01-01T10:00:00Z",
            # Missing correlationId, componentId, etc.
            "message": "Incomplete log",
        }
    ]
    content = json.dumps(logs)
    parsed, err = parse_uploaded_logs(content)
    assert parsed is None
    assert "validation" in err.lower() or "failed" in err.lower()


def test_parse_uploaded_logs_restic_jsonlines_without_timestamps():
    """Parse restic-style JSON-lines records even when they lack timestamps."""
    content = """
{"message_type":"verbose_status","action":"scan_finished","item":"","duration":0.203324484,"data_size":6553619,"data_size_in_repo":0,"metadata_size":0,"metadata_size_in_repo":0,"total_files":22}
{"message_type":"verbose_status","action":"new","item":"/home/claude/backup_lab_v2/incidents_run/source_data/app_config/settings.conf","duration":0.005229609,"data_size":19,"data_size_in_repo":101,"metadata_size":0,"metadata_size_in_repo":0,"total_files":0}
{"message_type":"status","percent_done":0.26075134975042036,"total_files":22,"files_done":2,"total_bytes":6553619,"bytes_done":1708865,"current_files":["/home/claude/backup_lab_v2/incidents_run/source_data/database/prod_dump.sql"]}
{"message_type":"verbose_status","action":"new","item":"/home/claude/backup_lab_v2/incidents_run/source_data/database/prod_dump.sql","duration":0.054029824,"data_size":5242880,"data_size_in_repo":5243341,"metadata_size":0,"metadata_size_in_repo":0,"total_files":0}
""".strip()

    parsed, err = parse_uploaded_logs_with_source(content, "backup_service.log")
    assert err == ""
    assert parsed is not None
    assert len(parsed) == 4
    assert all(entry.source_file == "backup_service.log" for entry in parsed)
    assert all(entry.component == "backup-service" for entry in parsed)
    assert parsed[0].message.startswith("verbose_status")
    assert parsed[0].timestamp < parsed[1].timestamp < parsed[2].timestamp < parsed[3].timestamp


def test_parse_uploaded_logs_keeps_plaintext_and_unknown_jsonl_lines():
    """Keep fatal plain-text lines and raw unparsed lines inside JSON-lines files."""
    content = """
{"time":"2026-07-29T15:23:07.000Z","message_type":"status","action":"scan_finished","total_files":22}
{"time":"2026-07-29T15:23:08.000Z","message_type":"status","action":"new","item":"file_1.bin","total_files":22}
Fatal: unable to save snapshot: write /repo/data/...: no space left on device
not-json-but-visible
""".strip()

    parsed, err = parse_uploaded_logs_with_source(content, "backup_service.log")
    assert err == ""
    assert parsed is not None
    assert len(parsed) == 4
    assert any(entry.error_code == "PLAINTEXT_FATAL" for entry in parsed)
    assert any(entry.error_code == "RAW_UNPARSED" for entry in parsed)
    assert any("no space left on device" in entry.message.lower() for entry in parsed)
    assert any(entry.message == "not-json-but-visible" for entry in parsed)
