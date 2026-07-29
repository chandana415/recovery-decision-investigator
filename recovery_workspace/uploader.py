"""Upload handler for user-supplied backup/recovery logs"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from recovery_workspace.models import LogEntry


MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
SUPPORTED_EXTENSIONS = {".log", ".txt", ".json"}


def extract_job_id(message: str) -> Optional[str]:
    """
    Extract job ID from log message if present.

    Looks for patterns like:
    - job job-0002
    - job-0002
    - jobId: abc-123

    Args:
        message: Log message text

    Returns:
        Job ID string or None if not found
    """
    patterns = [
        r"job\s+(?P<id>job-[\w]+)",  # "job job-0002"
        r"(?P<id>job-[\w]+)",          # "job-0002"
        r"jobId[:\s]+(?P<id>[\w-]+)",  # "jobId: abc-123"
    ]

    for pattern in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            return match.group("id")

    return None


def extract_component_from_filename(filename: str) -> str:
    """
    Extract component name from log filename.

    Examples:
    - "backup_service.log" → "backup-service"
    - "monitoring.log" → "monitoring"
    - "restic-backup.log" → "restic-backup"

    Args:
        filename: Log filename

    Returns:
        Component name
    """
    # Remove extension
    name = filename.rsplit(".", 1)[0] if "." in filename else filename
    # Replace underscores with hyphens for consistency
    name = name.replace("_", "-")
    return name or "unknown"


def validate_upload(uploaded_file: Any) -> tuple[bool, str]:
    """
    Validate uploaded file extension and size.

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        (is_valid: bool, message: str)
        - If valid: (True, "File accepted")
        - If invalid: (False, error_description)
    """
    if uploaded_file is None:
        return False, "No file provided."

    # Check extension
    name = uploaded_file.name.lower()
    ext = ""
    if "." in name:
        ext = "." + name.split(".")[-1]

    if ext not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        return False, f"Unsupported file type '{ext}'. Supported: {supported}"

    # Check size
    if uploaded_file.size > MAX_UPLOAD_SIZE_BYTES:
        max_mb = MAX_UPLOAD_SIZE_BYTES / (1024 * 1024)
        return (
            False,
            f"File too large ({uploaded_file.size / 1024 / 1024:.1f} MB). Maximum: {max_mb:.0f} MB",
        )

    return True, "File accepted"


def _build_structured_message(json_obj: dict) -> str:
    parts: list[str] = []
    for key in ("message", "msg", "details", "description", "text", "content"):
        value = json_obj.get(key)
        if isinstance(value, str) and value.strip():
            return value

    message_type = json_obj.get("message_type")
    action = json_obj.get("action")
    error_text = json_obj.get("error")
    item = json_obj.get("item")
    if isinstance(message_type, str) and message_type.strip():
        parts.append(message_type.strip())
    if isinstance(action, str) and action.strip():
        parts.append(action.strip())
    if isinstance(error_text, str) and error_text.strip():
        parts.append(error_text.strip())
    if isinstance(item, str) and item.strip():
        parts.append(item.strip())

    if not parts:
        parts.append(str(json_obj)[:200])

    if "percent_done" in json_obj:
        parts.append(f"{json_obj['percent_done']:.2%}" if isinstance(json_obj["percent_done"], float) else str(json_obj["percent_done"]))
    if "files_done" in json_obj and "total_files" in json_obj:
        parts.append(f"{json_obj['files_done']}/{json_obj['total_files']} files")
    if "bytes_done" in json_obj and "total_bytes" in json_obj:
        parts.append(f"{json_obj['bytes_done']}/{json_obj['total_bytes']} bytes")

    return " | ".join(str(part) for part in parts if part)


def _parse_iso_timestamp(ts_str: str) -> Optional[datetime]:
    try:
        if "T" in ts_str:
            return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        timestamp = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        return timestamp.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _extract_correlation_id_from_json_obj(json_obj: dict) -> Optional[str]:
    for corr_field in ["correlationId", "correlation_id", "job", "jobId", "job_id", "trace_id", "traceId"]:
        value = json_obj.get(corr_field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    message = json_obj.get("message")
    if isinstance(message, str):
        job_id = extract_job_id(message)
        if job_id:
            return job_id
    return None


def _is_known_plaintext_failure_line(line: str) -> bool:
    return bool(re.match(r"^\s*(fatal|error|panic)\s*:", line, re.IGNORECASE))


def _make_raw_log_entry(
    *,
    line: str,
    source_filename: str,
    file_correlation_id: str,
    level: str,
    error_code: str,
    parse_status: str,
    timestamp: Optional[datetime] = None,
) -> dict:
    component = extract_component_from_filename(source_filename) if source_filename else "unknown"
    return {
        "timestamp": timestamp,
        "correlationId": file_correlation_id,
        "componentId": component.lower(),
        "component": component,
        "level": level,
        "errorCode": error_code,
        "message": line,
        "traceId": None,
        "structuredFields": {
            "parse_status": parse_status,
            "raw_line": line,
            "source_filename": source_filename,
        },
    }


def _interpolate_missing_timestamps(records: list[dict], default_start: Optional[datetime] = None) -> None:
    known_indices = [i for i, record in enumerate(records) if isinstance(record.get("timestamp"), datetime)]
    if not records:
        return

    if not known_indices:
        base = default_start or datetime.now(timezone.utc).replace(microsecond=0)
        for i, record in enumerate(records):
            record["timestamp"] = (base + timedelta(seconds=i)).isoformat()
        return

    for i, record in enumerate(records):
        if isinstance(record.get("timestamp"), datetime):
            continue

        prev_known = max((idx for idx in known_indices if idx < i), default=None)
        next_known = min((idx for idx in known_indices if idx > i), default=None)

        if prev_known is not None and next_known is not None:
            prev_ts = records[prev_known]["timestamp"]
            next_ts = records[next_known]["timestamp"]
            missing_between = [idx for idx in range(prev_known + 1, next_known) if not isinstance(records[idx].get("timestamp"), datetime)]
            if missing_between:
                position = missing_between.index(i) + 1
                step = (next_ts - prev_ts) / (len(missing_between) + 1)
                record["timestamp"] = prev_ts + (step * position)
            else:
                record["timestamp"] = prev_ts + (next_ts - prev_ts) / 2
        elif prev_known is not None:
            prev_ts = records[prev_known]["timestamp"]
            offset = sum(1 for idx in range(prev_known + 1, i) if not isinstance(records[idx].get("timestamp"), datetime))
            record["timestamp"] = prev_ts + timedelta(seconds=offset + 1)
        elif next_known is not None:
            next_ts = records[next_known]["timestamp"]
            offset = sum(1 for idx in range(i + 1, next_known) if not isinstance(records[idx].get("timestamp"), datetime))
            record["timestamp"] = next_ts - timedelta(seconds=offset + 1)
        else:
            base = default_start or datetime.now(timezone.utc).replace(microsecond=0)
            record["timestamp"] = base + timedelta(seconds=i)

    for record in records:
        if isinstance(record.get("timestamp"), datetime):
            record["timestamp"] = record["timestamp"].isoformat()


def normalize_json_to_log_entry(
    json_obj: dict,
    source_filename: str = "",
    *,
    event_index: int = 0,
    synthetic_timestamp_base: Optional[datetime] = None,
    allow_missing_timestamp: bool = False,
) -> Optional[dict]:
    """
    Convert an arbitrary JSON object into LogEntry-compatible format.
    
    Tries to extract/infer required fields from the JSON object.
    
    Args:
        json_obj: Raw JSON object from JSON-lines parsing
        source_filename: Source filename for component tagging
        
    Returns:
        dict with LogEntry-compatible fields, or None if required fields can't be extracted
    """
    # Try to extract timestamp (required for LogEntry)
    timestamp = None
    for ts_field in ["timestamp", "ts", "time", "datetime", "date"]:
        if ts_field in json_obj:
            timestamp = json_obj[ts_field]
            break
    
    if not timestamp:
        if allow_missing_timestamp:
            timestamp = None
        elif synthetic_timestamp_base is None:
            return None  # Can't convert without timestamp unless caller allows synthetic time
        else:
            timestamp = synthetic_timestamp_base + timedelta(seconds=event_index)
    
    # Extract other fields with fallbacks
    message = _build_structured_message(json_obj)
    
    # Extract component
    component = None
    for comp_field in ["component", "service", "app", "source"]:
        if comp_field in json_obj:
            component = json_obj[comp_field]
            break
    
    if not component:
        component = extract_component_from_filename(source_filename)
    
    # Extract correlation ID
    correlation_id = None
    for corr_field in ["correlationId", "correlation_id", "job", "jobId", "job_id", "trace_id", "traceId"]:
        if corr_field in json_obj:
            correlation_id = json_obj[corr_field]
            break
    
    if not correlation_id:
        correlation_id = f"plaintext-{extract_component_from_filename(source_filename)}"
    
    # Extract level
    level = None
    for level_field in ["level", "severity", "status"]:
        if level_field in json_obj:
            level = str(json_obj[level_field]).upper()
            break
    
    if not level or level not in ["INFO", "WARN", "ERROR", "DEBUG", "WARNING"]:
        message_type = str(json_obj.get("message_type", "")).lower()
        action = str(json_obj.get("action", "")).lower()
        if any(token in message_type for token in ("error", "fatal")) or action in {"error", "failed", "fatal"}:
            level = "ERROR"
        elif any(token in message_type for token in ("warn", "warning")):
            level = "WARN"
        else:
            level = "INFO"
    
    # Return normalized LogEntry dict
    return {
        "timestamp": timestamp,
        "correlationId": correlation_id,
        "componentId": component.lower() if isinstance(component, str) else "unknown",
        "component": component,
        "level": level,
        "errorCode": "JSON" if level == "ERROR" else None,
        "message": message,
        "traceId": None,
        "structuredFields": json_obj,
    }


def read_uploaded_file_safe(uploaded_file: Any) -> tuple[Optional[str], str]:
    """
    Read uploaded file into memory with safe UTF-8 decoding.

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        (content: Optional[str], error_message: str)
        - If successful: (content_string, "")
        - If failed: (None, error_description)
    """
    if uploaded_file is None:
        return None, "No file provided."

    if uploaded_file.size == 0:
        return None, "File is empty."

    try:
        # Read raw bytes
        raw_bytes = uploaded_file.read()
        if not raw_bytes:
            return None, "File is empty."

        # Decode as UTF-8
        content = raw_bytes.decode("utf-8")
        return content, ""
    except UnicodeDecodeError as e:
        return (
            None,
            f"File is not valid UTF-8 text. Encoding error: {str(e)[:100]}",
        )
    except Exception as e:
        return None, f"Error reading file: {str(e)[:100]}"


def try_parse_jsonlines(content: str, source_filename: str = "") -> tuple[Optional[list[dict]], str]:
    """
    Try parsing content as JSON-lines (one JSON object per line).

    Args:
        content: File content as string

    Returns:
        (logs: Optional[list[dict]], error_message: str)
    """
    lines = content.strip().split("\n")
    records: list[dict] = []
    known_timestamps: list[datetime] = []
    file_correlation_id = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        job_id = extract_job_id(line)
        if job_id and file_correlation_id is None:
            file_correlation_id = job_id

        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            if _is_known_plaintext_failure_line(line):
                records.append(
                    _make_raw_log_entry(
                        line=line,
                        source_filename=source_filename,
                        file_correlation_id=file_correlation_id or f"jsonlines-{extract_component_from_filename(source_filename)}",
                        level="ERROR",
                        error_code="PLAINTEXT_FATAL",
                        parse_status="fatal_plaintext",
                    )
                )
            else:
                records.append(
                    _make_raw_log_entry(
                        line=line,
                        source_filename=source_filename,
                        file_correlation_id=file_correlation_id or f"jsonlines-{extract_component_from_filename(source_filename)}",
                        level="INFO",
                        error_code="RAW_UNPARSED",
                        parse_status="raw_unparsed",
                    )
                )
            continue

        if isinstance(obj, dict):
            correlation_id = _extract_correlation_id_from_json_obj(obj)
            if correlation_id and file_correlation_id is None:
                file_correlation_id = correlation_id
            normalized = normalize_json_to_log_entry(
                obj,
                source_filename,
                allow_missing_timestamp=True,
            )
            if normalized is None:
                records.append(
                    _make_raw_log_entry(
                        line=line,
                        source_filename=source_filename,
                        file_correlation_id=file_correlation_id or f"jsonlines-{extract_component_from_filename(source_filename)}",
                        level="INFO",
                        error_code="RAW_UNPARSED",
                        parse_status="json_unparsed",
                    )
                )
                continue

            if file_correlation_id and normalized.get("correlationId", "").startswith("plaintext-"):
                normalized["correlationId"] = file_correlation_id
            elif file_correlation_id is None:
                file_correlation_id = normalized.get("correlationId")
            normalized["structuredFields"] = obj
            records.append(normalized)
            if isinstance(normalized.get("timestamp"), datetime):
                known_timestamps.append(normalized["timestamp"])
            continue

        # Non-dict JSON scalar; keep visible.
        records.append(
            _make_raw_log_entry(
                line=line,
                source_filename=source_filename,
                file_correlation_id=file_correlation_id or f"jsonlines-{extract_component_from_filename(source_filename)}",
                level="INFO",
                error_code="RAW_UNPARSED",
                parse_status="json_scalar_unparsed",
            )
        )

    if not records:
        return None, "No valid JSON lines found."

    if not file_correlation_id:
        file_correlation_id = f"jsonlines-{extract_component_from_filename(source_filename)}"

    for record in records:
        if not record.get("correlationId"):
            record["correlationId"] = file_correlation_id

    _interpolate_missing_timestamps(records)
    return records, ""


def try_parse_plaintext(content: str, source_filename: str = "") -> tuple[Optional[list[dict]], str]:
    """
    Try parsing content as plaintext logs using regex patterns.

    Extracts ALL timestamped lines and associates orphaned lines (without timestamp)
    with the nearest preceding timestamped line.

    All events from the same file share a single correlation ID (either extracted job ID
    or derived from filename) to ensure timeline builder accepts them.

    Supports formats like:
    - 2025-01-01T10:00:00Z component ComponentName LEVEL message text here
    - [2025-01-01 10:00:00] component: message
    - component | LEVEL | message
    - Bare lines (associated with previous timestamp)

    Args:
        content: File content as string
        source_filename: Source filename for component tagging (e.g., "backup_service.log")

    Returns:
        (logs: Optional[list[dict]], error_message: str)
    """
    lines = content.strip().split("\n")
    logs = []
    last_timestamp = None
    last_component = extract_component_from_filename(source_filename) if source_filename else "unknown"
    last_level = "INFO"

    # First pass: scan for job ID to use as correlation ID for all events in this file
    file_correlation_id = None
    for line in lines:
        line = line.strip()
        if line:
            job_id = extract_job_id(line)
            if job_id:
                file_correlation_id = job_id
                break
    
    # If no job ID found, derive one from filename or use generic
    if not file_correlation_id:
        if source_filename:
            file_correlation_id = f"plaintext-{extract_component_from_filename(source_filename)}"
        else:
            file_correlation_id = "plaintext"

    # Multiple regex patterns to try for each line
    patterns = [
        # ISO 8601 timestamp + space-separated fields
        r"^(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?[Z+-].{0,6}?)\s+(?P<component>\w+)\s+(?P<component_id>\w+)\s+(?P<level>INFO|WARN|ERROR|WARNING|DEBUG)\s+(?P<message>.+)$",
        # ISO timestamp with [LEVEL] component: message
        r"^(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?[Z+-].{0,6}?)\s+\[(?P<level>\w+)\]\s+(?P<component>[\w-]+):\s+(?P<message>.+)$",
        # [timestamp] component: message
        r"^\[(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})\]\s+(?P<component>[\w-]+):\s+(?P<message>.+)$",
        # timestamp | component | level | message
        r"^(?P<timestamp>\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2})\s*\|\s*(?P<component>[\w-]+)\s*\|\s*(?P<level>\w+)\s*\|\s*(?P<message>.+)$",
        # syslog-like: timestamp hostname component[pid]: message
        r"^(?P<timestamp>\w+\s+\d+\s+\d{2}:\d{2}:\d{2})\s+[\w.-]+\s+(?P<component>[\w-]+)(?:\[\d+\])?:\s+(?P<message>.+)$",
        # ISO timestamp + any message (catch-all for "2026-07-29T15:23:07.000Z restic backup invoked...")
        r"^(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?[Z+-].{0,6}?)\s+(?P<message>.+)$",
    ]

    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue

        matched = False
        for pattern in patterns:
            match = re.match(pattern, line)
            if match:
                data = match.groupdict()

                # Normalize timestamp
                ts_str = data.get("timestamp", "")
                try:
                    # Try ISO 8601 first
                    if "T" in ts_str:
                        timestamp = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    else:
                        # Try space-separated format
                        timestamp = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                        # Assume UTC if not specified
                        timestamp = timestamp.replace(tzinfo=timezone.utc)
                except (ValueError, AttributeError):
                    # If parsing fails, use last timestamp or current time
                    timestamp = last_timestamp if last_timestamp else datetime.now(timezone.utc)

                # Update tracking variables for orphaned lines
                last_timestamp = timestamp
                last_component = data.get("component", last_component)
                last_level = data.get("level", "INFO").upper()

                # Extract message (for logging/display purposes, not for correlation ID)
                message_text = data.get("message", line)

                # Detect ERROR level from message content if not already set
                if last_level == "INFO" and re.search(r"\b(fatal|error|failed|failed|corruption|cannot|unable)\b", message_text, re.IGNORECASE):
                    last_level = "ERROR"

                log_entry = {
                    "timestamp": timestamp.isoformat(),
                    "correlationId": file_correlation_id,  # Use consistent ID for all events in file
                    "componentId": data.get("component_id", last_component).lower(),
                    "component": last_component,
                    "level": last_level,
                    "errorCode": "PLAINTEXT" if last_level == "ERROR" else None,
                    "message": message_text,
                    "traceId": None,
                }
                logs.append(log_entry)
                matched = True
                break

        if not matched and last_timestamp:
            # Orphaned line: associate with last timestamp
            # Check if the line contains error keywords
            orphaned_level = last_level
            if orphaned_level == "INFO" and re.search(r"\b(fatal|error|failed|corruption|cannot|unable)\b", line, re.IGNORECASE):
                orphaned_level = "ERROR"
            
            log_entry = {
                "timestamp": last_timestamp.isoformat(),
                "correlationId": file_correlation_id,  # Use consistent ID for all events in file
                "componentId": last_component.lower(),
                "component": last_component,
                "level": orphaned_level,
                "errorCode": "PLAINTEXT" if orphaned_level == "ERROR" else None,
                "message": line,
                "traceId": None,
            }
            logs.append(log_entry)
            last_level = orphaned_level

    return logs, "" if logs else "No parseable lines found in plaintext log."


def parse_uploaded_logs(content: str, source_filename: str = "") -> tuple[Optional[list[LogEntry]], str]:
    """
    Parse file content into LogEntry objects using multi-format strategy.

    Tries in order:
    1. JSON-lines: one JSON object per line (fast detection)
    2. JSON array: [{...}, {...}]
    3. Scenario format: {"logs": [{...}]}
    4. Plaintext: regex-based extraction (timestamp, component, level, message)

    Args:
        content: File content as string
        source_filename: Source filename for component tagging (optional)

    Returns:
        (logs: Optional[list[LogEntry]], error_message: str)
        - If parsed: (LogEntry_list, "")
        - If failed: (None, error_description)
    """
    if not content.strip():
        return None, "File is empty or contains only whitespace."

    content_start = content.strip()[:1]

    # Strategy 1a: Quick check for JSON-lines (multiline JSON objects)
    # If content has multiple lines and doesn't start with [ or {, try JSON-lines first
    if content_start not in "{[" or "\n" in content.strip():
        logs_data, _ = try_parse_jsonlines(content, source_filename)
        if logs_data:
            # Try direct LogEntry validation first
            try:
                parsed_logs = [LogEntry.model_validate(log) for log in logs_data]
                if parsed_logs:
                    return parsed_logs, ""
            except Exception:
                pass  # JSON-lines objects don't match LogEntry schema, try normalizing

    # Strategy 1b: Try JSON array or Scenario format
    # BUT: Skip this if content looks like multiline JSON-lines (multiple lines each starting with { or [)
    # because json.loads(entire_content) will fail with "Extra data" error
    json_error = None
    logs_data = None
    
    # Check if this looks like JSON-lines format (multiple independent JSON objects)
    is_likely_jsonlines = False
    if "\n" in content.strip():
        lines = content.strip().split("\n")
        json_line_count = sum(1 for line in lines if line.strip() and line.strip()[0] in "{[")
        if json_line_count >= 2:  # At least 2 lines that look like JSON objects
            is_likely_jsonlines = True
    
    # Only try json.loads on entire content if it doesn't look like JSON-lines
    if not is_likely_jsonlines:
        try:
            obj = json.loads(content)
            if isinstance(obj, list):
                logs_data = obj
            elif isinstance(obj, dict) and "logs" in obj:
                logs_data = obj["logs"]
            elif isinstance(obj, dict):
                # JSON but not in expected format
                json_error = "File is JSON but format doesn't match expected schema. Provide a JSON array or {logs: [...]} object."
            else:
                json_error = "File is JSON but format doesn't match expected schema."
        except json.JSONDecodeError as e:
            json_error = f"Invalid JSON syntax: {str(e)[:80]}"

    # If JSON parsing succeeded, validate entries
    if logs_data is not None:
        try:
            if not logs_data:
                return None, "File contains no log entries."
            parsed_logs = [LogEntry.model_validate(log) for log in logs_data]
            if parsed_logs:
                return parsed_logs, ""
        except Exception as e:
            json_error = f"JSON validation failed: {str(e)[:80]}"

    # If JSON had obvious syntax errors (starts with { or [), reject plaintext fallback
    if content_start in "{[" and not is_likely_jsonlines:
        # Looks like JSON but failed to parse — don't try plaintext
        return (
            None,
            json_error or "Could not parse JSON file. " 
            "Provide a JSON array or {logs: [...]} object.",
        )

    # Strategy 2: Try plaintext parsing (only if content doesn't look like JSON)
    logs_data, plaintext_error = try_parse_plaintext(content, source_filename)
    if logs_data:
        try:
            parsed_logs = [LogEntry.model_validate(log) for log in logs_data]
            if parsed_logs:
                # Success with plaintext parsing
                return parsed_logs, ""
        except Exception as e:
            return (
                None,
                f"Plaintext parsing extracted entries but validation failed: {str(e)[:80]}",
            )

    # All strategies failed
    error_msg = (
        json_error 
        or "Could not parse file as JSON, JSON-lines, or plaintext. "
        "Supported formats: JSON array, JSON-lines (one object per line), or plaintext logs."
    )
    return None, error_msg


def parse_uploaded_logs_with_source(
    content: str, source_filename: str
) -> tuple[Optional[list[LogEntry]], str]:
    """
    Parse file content and tag each LogEntry with source filename.

    Args:
        content: File content as string
        source_filename: Name of the source file (e.g., "monitoring.log")

    Returns:
        (logs: Optional[list[LogEntry]], error_message: str)
    """
    logs, error = parse_uploaded_logs(content, source_filename)
    if error:
        return None, error

    # Tag each log entry with source file
    for log in logs:
        log.source_file = source_filename

    return logs, ""


def process_multiple_uploads(uploaded_files: list[Any]) -> tuple[list[LogEntry], dict[str, str]]:
    """
    Process multiple uploaded files, collecting results and tracking errors.

    Args:
        uploaded_files: List of Streamlit UploadedFile objects

    Returns:
        (merged_logs: list[LogEntry], file_results: dict)
        where file_results = {
            "backup_service.log": "✅ 42 events",
            "storage.log": "⚠️ Invalid JSON: ...",
        }
    """
    merged_logs = []
    file_results = {}

    for uploaded_file in uploaded_files:
        filename = uploaded_file.name

        # Validate
        is_valid, validation_msg = validate_upload(uploaded_file)
        if not is_valid:
            file_results[filename] = f"⚠️ {validation_msg}"
            continue

        # Read
        content, read_error = read_uploaded_file_safe(uploaded_file)
        if read_error:
            file_results[filename] = f"⚠️ {read_error}"
            continue

        # Parse with source tagging
        logs, parse_error = parse_uploaded_logs_with_source(content, filename)
        if parse_error:
            file_results[filename] = f"⚠️ {parse_error}"
            continue

        # Success
        merged_logs.extend(logs)
        file_results[filename] = f"✅ {len(logs)} events"

    return merged_logs, file_results
