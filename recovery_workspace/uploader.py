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
            "timestamp_source": "unavailable",
        },
    }


def _interpolate_missing_timestamps(records: list[dict], default_start: Optional[datetime] = None) -> None:
    if not records:
        return

    base = default_start or datetime.now(timezone.utc).replace(microsecond=0)
    for index, record in enumerate(records):
        structured_fields = record.setdefault("structuredFields", {})
        if not isinstance(structured_fields, dict):
            structured_fields = {}
            record["structuredFields"] = structured_fields

        if isinstance(record.get("timestamp"), datetime):
            structured_fields.setdefault("timestamp_source", "parsed")
            if record["timestamp"].tzinfo is None:
                record["timestamp"] = record["timestamp"].replace(tzinfo=timezone.utc)
        elif isinstance(record.get("timestamp"), str):
            parsed = _parse_iso_timestamp(record["timestamp"])
            if parsed is not None:
                record["timestamp"] = parsed
                structured_fields.setdefault("timestamp_source", "parsed")
            else:
                record["timestamp"] = base + timedelta(seconds=index)
                structured_fields["timestamp_source"] = "missing"
                structured_fields["ingestion_sequence"] = index
        else:
            record["timestamp"] = base + timedelta(seconds=index)
            structured_fields["timestamp_source"] = "missing"
            structured_fields["ingestion_sequence"] = index


def _coerce_structured_value(value: str) -> Any:
    value = value.strip()
    if not value:
        return value
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)
    return value


def _extract_generic_level(fields: dict[str, Any], *, fallback: str = "INFO") -> str:
    level = fields.get("level") or fields.get("severity") or fields.get("status")
    if isinstance(level, str):
        level_text = level.strip().upper()
        if level_text in {"INFO", "WARN", "WARNING", "ERROR", "DEBUG"}:
            return level_text if level_text != "WARNING" else "WARN"

    for key in ("msg", "message", "summary", "reason"):
        value = fields.get(key)
        if isinstance(value, str):
            value_text = value.lower()
            if any(token in value_text for token in ("error", "failed", "failure", "fatal", "denied", "timeout", "cancelled", "critical")):
                return "ERROR"
            if any(token in value_text for token in ("warn", "warning", "degraded")):
                return "WARN"
    return fallback


def _build_structural_event(
    *,
    source_filename: str,
    file_correlation_id: str,
    message: str,
    level: str,
    structured_fields: dict[str, Any],
    timestamp: Optional[datetime] = None,
    raw_line: str = "",
    error_code: Optional[str] = None,
    sequence: Optional[int] = None,
) -> dict:
    component = extract_component_from_filename(source_filename) if source_filename else "unknown"
    timestamp_source = "parsed" if timestamp is not None else "missing"
    if structured_fields.get("timestamp_source") in {"missing", "parsed", "sequence"}:
        timestamp_source = structured_fields.get("timestamp_source") or timestamp_source
    return {
        "timestamp": timestamp,
        "correlationId": file_correlation_id,
        "componentId": component.lower(),
        "component": component,
        "level": level,
        "errorCode": error_code,
        "message": message,
        "traceId": None,
        "structuredFields": {
            **structured_fields,
            "raw_line": raw_line,
            "source_filename": source_filename,
            "timestamp_source": timestamp_source,
        },
    }


def _score_key_value_lines(lines: list[str]) -> tuple[float, dict[str, Any]]:
    meaningful_lines = 0
    meaningful_fields = set()
    assignment_count = 0
    recognized_keys = {"time", "timestamp", "level", "msg", "message", "cause", "source", "stack"}
    for line in lines:
        if line.startswith("$") or line.startswith("#"):
            continue
        matches = list(re.finditer(r"(^|\s)([A-Za-z0-9_.-]+)\s*=", line))
        if matches:
            assignment_count += len(matches)
            for key in recognized_keys:
                if re.search(rf"\b{key}\s*=", line):
                    meaningful_fields.add(key)
            if re.search(r"\b(msg|message|cause|source|stack|level|time|timestamp)\s*=", line):
                meaningful_lines += 1

    if assignment_count < 2:
        return 0.0, {"matched_lines": 0, "assignment_count": assignment_count, "meaningful_lines": 0, "meaningful_fields": []}

    matched_ratio = meaningful_lines / max(1, len(lines))
    field_ratio = len(meaningful_fields) / len(recognized_keys)
    confidence = min(0.95, 0.2 + 0.35 * min(1.0, matched_ratio * 2.0) + 0.2 * field_ratio)
    return confidence, {
        "matched_lines": meaningful_lines,
        "assignment_count": assignment_count,
        "meaningful_lines": meaningful_lines,
        "meaningful_fields": sorted(meaningful_fields),
    }


def _score_command_status_lines(lines: list[str]) -> tuple[float, dict[str, Any]]:
    prompt_lines = sum(1 for line in lines if line.startswith("$"))
    assignment_lines = sum(1 for line in lines if re.match(r"^[A-Za-z0-9_.-]+=", line))
    summary_lines = sum(1 for line in lines if re.search(r"\b(state|summary|status|health|count|available|entity|section)\b", line, re.IGNORECASE))
    section_lines = sum(1 for line in lines if re.search(r"^section\s*=", line, re.IGNORECASE))
    state_lines = sum(1 for line in lines if re.search(r"^state\s*=", line, re.IGNORECASE))
    metric_lines = sum(1 for line in lines if re.search(r"\b(count|available|total|used|down|up)\b", line, re.IGNORECASE))
    confidence = 0.0
    if prompt_lines:
        confidence += 0.2
    if assignment_lines:
        confidence += 0.2
    if summary_lines:
        confidence += 0.2
    if section_lines:
        confidence += 0.15
    if state_lines:
        confidence += 0.15
    if metric_lines:
        confidence += 0.1
    return min(0.95, confidence), {
        "prompt_lines": prompt_lines,
        "assignment_lines": assignment_lines,
        "summary_lines": summary_lines,
        "section_lines": section_lines,
        "state_lines": state_lines,
        "metric_lines": metric_lines,
    }


def _normalize_detection_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _is_timestamped_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return bool(
        re.match(r"^\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}", stripped)
        or re.match(r"^\[\d{4}-\d{2}-\d{2}", stripped)
        or re.match(r"^\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}", stripped)
    )


def _looks_like_exception_field(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    label = stripped.split(":", 1)[0].strip()
    normalized_label = _normalize_detection_token(label)
    if not normalized_label:
        return False

    recognized = {"level", "logger", "message", "thread", "throwable", "cause", "stack", "suppressed", "exception", "class", "trace", "error", "detail"}
    if normalized_label in recognized:
        return True
    for token in recognized:
        if normalized_label.startswith(token) or token.startswith(normalized_label):
            return True
    return False


def _looks_like_frame_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if re.match(r"^(?:at|caused by|suppressed)\b", stripped, re.IGNORECASE):
        return True
    normalized = _normalize_detection_token(stripped)
    if normalized.startswith("at") and len(normalized) > 2:
        return True
    if "causedby" in normalized or "suppressed" in normalized:
        return True
    if re.search(r"\b(?:class|method|file|line|trace)\b", stripped, re.IGNORECASE) and re.search(r"\d+", stripped):
        return True
    if re.search(r"\(", stripped) and re.search(r"\d+", stripped):
        return True
    return False


def detect_document_type(content: str, source_filename: str = "") -> str:
    if not content or not content.strip():
        return "PLAINTEXT"

    stripped = content.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            return "JSON"
        if isinstance(obj, list):
            return "JSON"
        if isinstance(obj, dict) and "logs" in obj:
            return "JSON"
        return "JSON"

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if not lines:
        return "PLAINTEXT"

    if any(line.startswith("$") for line in lines) and any(re.search(r"\b(state|summary|health|count|available|entity|section)\b", line, re.IGNORECASE) for line in lines):
        return "COMMAND_STATUS"

    if len(lines) >= 2:
        header_line = lines[0]
        header_tokens = [token for token in re.split(r"\s{2,}|\s+", header_line) if token]
        if len(header_tokens) >= 2 and any(token.lower() in {"first_seen", "last_seen", "count", "source", "reason", "message", "details"} for token in header_tokens):
            return "TABULAR_REPORT"

    timestamped_lines = sum(1 for line in lines if _is_timestamped_line(line))
    frame_lines = [line for line in lines if _looks_like_frame_line(line)]
    field_lines = [line for line in lines if _looks_like_exception_field(line)]
    header_signal = any(re.search(r"\b(error|failed|unable|exception|fatal|timeout|denied|interrupted|unavailable)\b", line, re.IGNORECASE) for line in lines)
    diagnostic_signal = any(re.search(r"\b(cause|throwable|suppressed|stack|trace|exception)\b", line, re.IGNORECASE) for line in lines)

    score = 0
    if header_signal:
        score += 2
    if field_lines:
        score += min(3, len(field_lines))
    if frame_lines:
        score += min(3, len(frame_lines))
    if diagnostic_signal:
        score += 1
    if len(frame_lines) >= 2 and len(field_lines) >= 2:
        score += 1
    if timestamped_lines >= 2:
        score -= 3

    if score >= 6 and len(frame_lines) >= 2 and (len(field_lines) >= 2 or header_signal):
        return "EXCEPTION_DOCUMENT"

    if timestamped_lines >= 2:
        return "TIMESTAMPED_LOG"

    return "PLAINTEXT"


def detect_input_structure(content: str, source_filename: str = "") -> tuple[str, dict[str, Any]]:
    lines = [line.strip() for line in content.strip().splitlines() if line.strip()]
    if not lines:
        return "plain_text", {"confidence": 0.0, "reason": "empty"}

    key_confidence, key_stats = _score_key_value_lines(lines)
    command_confidence, command_stats = _score_command_status_lines(lines)

    if key_confidence >= 0.6 and key_confidence >= command_confidence + 0.1:
        return "key_value", {
            "confidence": key_confidence,
            "matched_line_ratio": key_stats["meaningful_lines"] / max(1, len(lines)),
            "meaningful_lines": key_stats["meaningful_lines"],
            "assignment_count": key_stats["assignment_count"],
            "meaningful_fields": key_stats["meaningful_fields"],
        }

    if command_confidence >= 0.6 and command_confidence >= key_confidence + 0.1:
        return "command_status", {
            "confidence": command_confidence,
            "matched_line_ratio": (command_stats["assignment_lines"] + command_stats["summary_lines"]) / max(1, len(lines)),
            "meaningful_lines": command_stats["assignment_lines"] + command_stats["summary_lines"],
            "prompt_lines": command_stats["prompt_lines"],
            "state_lines": command_stats["state_lines"],
            "metric_lines": command_stats["metric_lines"],
        }

    return "plain_text", {"confidence": max(key_confidence, command_confidence), "reason": "generic"}


def _parse_exception_document(content: str, source_filename: str = "") -> Optional[list[dict]]:
    lines = [line.rstrip() for line in content.strip().splitlines() if line.strip()]
    if not lines:
        return None

    if detect_document_type(content, source_filename) != "EXCEPTION_DOCUMENT":
        return None

    stack_lines: list[str] = []
    header_line = None
    for line in lines:
        stripped = line.strip()
        if re.match(r"^\s*(at|caused by)\b", stripped, re.IGNORECASE) or re.search(r"[A-Za-z0-9_.$]+\.java:\d+", stripped):
            stack_lines.append(stripped)
            continue
        if header_line is None:
            header_line = stripped

    if not header_line:
        return None

    message = header_line
    for line in lines:
        stripped = line.strip()
        if re.search(r"http\s+4\d\d|http\s+5\d\d|unauthorized|forbidden|access denied", stripped, re.IGNORECASE):
            message = stripped
            break

    diagnostic_blocks = []
    if stack_lines:
        diagnostic_blocks.append({
            "kind": "DIAGNOSTIC_BLOCK",
            "message": f"Stack trace — {len(stack_lines)} frames",
            "raw_lines": stack_lines,
        })

    return [
        _build_structural_event(
            source_filename=source_filename,
            file_correlation_id=f"structured-{extract_component_from_filename(source_filename)}",
            message=message,
            level="ERROR",
            structured_fields={
                "document_type": "EXCEPTION_DOCUMENT",
                "record_type": "EXCEPTION",
                "diagnostic_blocks": diagnostic_blocks,
                "timestamp_source": "missing",
            },
            timestamp=None,
            raw_line="\n".join(lines),
            error_code="HTTP_4XX" if re.search(r"http\s+4\d\d", message, re.IGNORECASE) else "HTTP_5XX" if re.search(r"http\s+5\d\d", message, re.IGNORECASE) else None,
            sequence=0,
        )
    ]


def _build_command_status_event(
    *,
    lines: list[str],
    source_filename: str,
    fields: dict[str, Any],
    default_message: str,
    extra_structured_fields: Optional[dict[str, Any]] = None,
) -> list[dict]:
    state = str(fields.get("state", "")).strip().lower()
    level = "ERROR" if state in {"down", "error", "failed", "degraded", "critical"} else "WARN" if state in {"warning", "warn"} else "INFO"
    message = str(fields.get("summary") or fields.get("message") or "").strip()
    if not message and fields.get("section") and fields.get("state"):
        message = f"{fields['section']} status: {fields['state']}"
    elif not message:
        message = default_message

    structured_fields = dict(fields)
    if extra_structured_fields:
        structured_fields.update(extra_structured_fields)

    return [
        _build_structural_event(
            source_filename=source_filename,
            file_correlation_id=f"structured-{extract_component_from_filename(source_filename)}",
            message=message,
            level=level,
            structured_fields=structured_fields,
            timestamp=None,
            raw_line=" | ".join(lines),
            error_code=None,
            sequence=0,
        )
    ]


def _parse_command_status_document(content: str, source_filename: str = "") -> Optional[list[dict]]:
    lines = [line.strip() for line in content.strip().splitlines() if line.strip()]
    if not lines:
        return None

    if detect_document_type(content, source_filename) != "COMMAND_STATUS":
        return None

    fields: dict[str, Any] = {}
    for line in lines:
        if line.startswith("$") or line.startswith("#"):
            continue
        if re.match(r"^[A-Za-z0-9_.-]+=", line):
            key, _, value = line.partition("=")
            fields[key.strip()] = _coerce_structured_value(value.strip())

    if not fields:
        return None

    return _build_command_status_event(
        lines=lines,
        source_filename=source_filename,
        fields=fields,
        default_message="status snapshot",
        extra_structured_fields={
            "document_type": "COMMAND_STATUS",
            "record_type": "OPERATIONAL_EVENT",
            "timestamp_source": "missing",
        },
    )


def _build_tabular_events(
    *,
    lines: list[str],
    source_filename: str,
    recognized: dict[str, int],
    base_structured_fields: Optional[dict[str, Any]] = None,
) -> Optional[list[dict]]:
    records: list[dict] = []
    file_correlation_id = f"structured-{extract_component_from_filename(source_filename)}"

    for index, line in enumerate(lines[1:], start=1):
        if line.startswith("$") or line.startswith("#"):
            continue
        tokens = line.split()
        if not tokens or len(tokens) < 3:
            continue
        if all(token.lower() in {"first_seen", "last_seen", "count", "source", "reason", "message"} for token in tokens):
            continue

        values: dict[str, str] = {}
        for key, idx in recognized.items():
            if key in {"message", "details"}:
                values[key] = " ".join(tokens[idx:]).strip() if idx < len(tokens) else ""
            elif idx < len(tokens):
                values[key] = tokens[idx]

        if not values:
            continue

        first_seen = values.get("first_seen") or values.get("time") or values.get("timestamp")
        last_seen = values.get("last_seen")
        count_value = values.get("count")
        source_value = values.get("source") or values.get("from")
        reason_value = values.get("reason")
        message_value = values.get("message") or values.get("details") or ""

        timestamp = _parse_iso_timestamp(first_seen) if first_seen else None
        if timestamp is None and last_seen:
            timestamp = _parse_iso_timestamp(last_seen)

        structured_fields: dict[str, Any] = dict(base_structured_fields or {})
        if count_value is not None:
            try:
                structured_fields["count"] = int(count_value)
            except ValueError:
                structured_fields["count"] = count_value
        if source_value:
            structured_fields["source"] = source_value
        if reason_value:
            structured_fields["reason"] = reason_value
        if message_value:
            structured_fields["message"] = message_value
        if first_seen:
            structured_fields["first_seen"] = first_seen
        if last_seen:
            structured_fields["last_seen"] = last_seen
        if "timestamp_source" in structured_fields:
            structured_fields["timestamp_source"] = "parsed" if timestamp is not None else "missing"

        level = "ERROR" if any(token in (message_value + " " + (reason_value or "")).lower() for token in ("error", "failed", "failure", "fatal", "timeout", "cancelled", "denied")) else "INFO"
        records.append(
            _build_structural_event(
                source_filename=source_filename,
                file_correlation_id=file_correlation_id,
                message=message_value or reason_value or "structured table event",
                level=level,
                structured_fields=structured_fields,
                timestamp=timestamp,
                raw_line=line,
                error_code="STRUCTURED_EVENT" if level == "ERROR" else None,
                sequence=index,
            )
        )

    return records or None


def _parse_tabular_report_document(content: str, source_filename: str = "") -> Optional[list[dict]]:
    lines = [line.strip() for line in content.strip().splitlines() if line.strip()]
    if len(lines) < 2:
        return None

    if detect_document_type(content, source_filename) != "TABULAR_REPORT":
        return None

    header_line = lines[0]
    header_tokens = [token for token in re.split(r"\s{2,}|\s+", header_line) if token]
    normalized_header = [token.lower() for token in header_tokens]
    recognized = {name: idx for idx, name in enumerate(normalized_header) if name in {"first_seen", "last_seen", "count", "source", "from", "reason", "message", "details"}}
    if not recognized:
        return None

    return _build_tabular_events(
        lines=lines,
        source_filename=source_filename,
        recognized=recognized,
        base_structured_fields={
            "document_type": "TABULAR_REPORT",
            "record_type": "OPERATIONAL_EVENT",
            "timestamp_source": "missing",
        },
    )


def _parse_key_value_block(content: str, source_filename: str = "") -> Optional[list[dict]]:
    lines = [line.strip() for line in content.strip().splitlines() if line.strip()]
    if not lines:
        return None

    parser_name, metadata = detect_input_structure(content, source_filename)
    if parser_name != "key_value":
        return None
    if metadata.get("confidence", 0.0) < 0.6:
        return None
    if metadata.get("matched_line_ratio", 0.0) < 0.25:
        return None

    assignment_count = 0
    for line in lines:
        if line.startswith("$"):
            continue
        assignment_count += len(re.findall(r"([A-Za-z0-9_.-]+)\s*=", line))

    if assignment_count < 2:
        return None

    file_correlation_id = f"structured-{extract_component_from_filename(source_filename)}"
    records: list[dict] = []
    for line in lines:
        if not line or line.startswith("$"):
            continue
        field_matches = list(re.finditer(r"([A-Za-z0-9_.-]+)\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s]+))", line))
        if not field_matches:
            continue

        fields: dict[str, Any] = {}
        for match in field_matches:
            key = match.group(1)
            value = match.group(2) if match.group(2) is not None else match.group(3) if match.group(3) is not None else match.group(4) or ""
            fields[key] = _coerce_structured_value(value)

        timestamp = None
        for key in ("time", "timestamp", "ts"):
            value = fields.get(key)
            if isinstance(value, str):
                timestamp = _parse_iso_timestamp(value)
                if timestamp is not None:
                    break

        level = _extract_generic_level(fields, fallback="INFO")
        message = ""
        for key in ("msg", "message", "summary", "cause"):
            if isinstance(fields.get(key), str) and fields[key].strip():
                message = fields[key]
                break
        if not message and fields.get("cause"):
            message = str(fields["cause"])

        records.append(
            _build_structural_event(
                source_filename=source_filename,
                file_correlation_id=file_correlation_id,
                message=message or "structured event",
                level=level,
                structured_fields=fields,
                timestamp=timestamp,
                raw_line=line,
                error_code="STRUCTURED_EVENT" if level == "ERROR" else None,
                sequence=len(records),
            )
        )

    return records or None


def _parse_tabular_events(content: str, source_filename: str = "") -> Optional[list[dict]]:
    lines = [line.strip() for line in content.strip().splitlines() if line.strip()]
    if len(lines) < 2:
        return None

    header_line = lines[0]
    header_tokens = [token for token in re.split(r"\s{2,}|\s+", header_line) if token]
    if not header_tokens:
        return None

    normalized_header = [token.lower() for token in header_tokens]
    recognized = {name: idx for idx, name in enumerate(normalized_header) if name in {"first_seen", "last_seen", "count", "source", "from", "reason", "message", "details"}}
    if not recognized:
        return None

    return _build_tabular_events(
        lines=lines,
        source_filename=source_filename,
        recognized=recognized,
    )


def _parse_command_status_block(content: str, source_filename: str = "") -> Optional[list[dict]]:
    lines = [line.strip() for line in content.strip().splitlines() if line.strip()]
    if not lines:
        return None

    parser_name, metadata = detect_input_structure(content, source_filename)
    if parser_name != "command_status":
        return None
    if metadata.get("confidence", 0.0) < 0.6:
        return None

    if not any(line.startswith("$") for line in lines) and not any(re.match(r"^[A-Za-z0-9_.-]+=", line) for line in lines):
        return None

    fields: dict[str, Any] = {}
    for line in lines:
        if line.startswith("$") or line.startswith("#"):
            continue
        if re.match(r"^[A-Za-z0-9_.-]+=", line):
            key, _, value = line.partition("=")
            fields[key.strip()] = _coerce_structured_value(value.strip())

    if not fields:
        return None

    return _build_command_status_event(
        lines=lines,
        source_filename=source_filename,
        fields=fields,
        default_message="status block",
    )


def try_parse_document_structures(content: str, source_filename: str = "") -> tuple[Optional[list[dict]], str]:
    for parser in (
        _parse_exception_document,
        _parse_command_status_document,
        _parse_tabular_report_document,
    ):
        records = parser(content, source_filename)
        if records:
            return records, ""
    return None, ""


def try_parse_structural_formats(content: str, source_filename: str = "") -> tuple[Optional[list[dict]], str]:
    for parser in (
        _parse_key_value_block,
        _parse_tabular_events,
        _parse_command_status_block,
    ):
        records = parser(content, source_filename)
        if records:
            return records, ""
    return None, ""


def _validate_log_entries(logs_data: list[dict]) -> list[LogEntry]:
    _interpolate_missing_timestamps(logs_data)
    return [LogEntry.model_validate(log) for log in logs_data]


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
    saw_json_object = False

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

        saw_json_object = True
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

    if not saw_json_object:
        return None, "Not JSON-lines."

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
        # Generic component: message lines (e.g. "monitoring: snapshot failed")
        r"^(?P<component>[A-Za-z][A-Za-z0-9_.-]*):\s*(?P<message>.+)$",
        # Shell prompt and command output block lines like "[ceph@mon1 ~]$ ceph -s"
        r"^\[(?P<component>[A-Za-z0-9_.@-]+)\s+~\]\$\s*(?P<message>.+)$",
        # Timestamp + component + level + message (e.g. "2026-07-29T15:23:07.000Z monitoring ERROR snapshot failed")
        r"^(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)\s+(?P<component>[A-Za-z0-9_.-]+)\s+(?P<level>INFO|WARN|ERROR|WARNING|DEBUG)\s+(?P<message>.+)$",
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
                        # Try to assume UTC if not specified
                        timestamp = timestamp.replace(tzinfo=timezone.utc)
                except (ValueError, AttributeError):
                    # If parsing fails, use last timestamp or current time
                    timestamp = last_timestamp if last_timestamp else datetime.now(timezone.utc)

                # Update tracking variables for orphaned lines
                last_timestamp = timestamp
                raw_component = data.get("component", last_component)
                normalized_component = str(raw_component).replace("_", "-").strip()
                last_component = normalized_component or last_component
                last_level = data.get("level", "INFO").upper()

                # Extract message (for logging/display purposes, not for correlation ID)
                message_text = data.get("message", line)

                # Detect ERROR level from message content if not already set
                if last_level == "INFO" and re.search(r"\b(fatal|error|failed|failed|corruption|cannot|unable)\b", message_text, re.IGNORECASE):
                    last_level = "ERROR"

                if not last_level or last_level == "INFO":
                    lowered_message = message_text.lower()
                    if any(token in lowered_message for token in ("health_err", "error", "failed", "fatal", "denied", "timeout", "cancelled", "critical", "stuck", "degraded", "inactive", "stale", "unclean", "undersized")):
                        last_level = "ERROR"
                    elif any(token in lowered_message for token in ("warn", "warning", "degraded")):
                        last_level = "WARN"

                log_entry = {
                    "timestamp": timestamp.isoformat(),
                    "correlationId": file_correlation_id,  # Use consistent ID for all events in file
                    "componentId": str(data.get("component_id") or last_component).lower(),
                    "component": last_component,
                    "level": last_level,
                    "errorCode": "PLAINTEXT" if last_level == "ERROR" else None,
                    "message": message_text,
                    "traceId": None,
                    "structuredFields": {"timestamp_source": "parsed"},
                }
                logs.append(log_entry)
                matched = True
                break

        if not matched and last_timestamp:
            # Orphaned line: associate with last timestamp
            # Check if the line contains error keywords
            orphaned_level = last_level
            lowered_line = line.lower()
            if orphaned_level == "INFO" and re.search(r"\b(fatal|error|failed|corruption|cannot|unable|health_err|stuck|degraded|inactive|stale|unclean|undersized)\b", line, re.IGNORECASE):
                orphaned_level = "ERROR"
            elif orphaned_level == "INFO" and re.search(r"\b(warn|warning|degraded)\b", line, re.IGNORECASE):
                orphaned_level = "WARN"
            
            log_entry = {
                "timestamp": last_timestamp.isoformat(),
                "correlationId": file_correlation_id,  # Use consistent ID for all events in file
                "componentId": last_component.lower(),
                "component": last_component,
                "level": orphaned_level,
                "errorCode": "PLAINTEXT" if orphaned_level == "ERROR" else None,
                "message": line,
                "traceId": None,
                "structuredFields": {"timestamp_source": "inherited"},
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
    looks_like_plaintext = bool(
        re.search(r"^\s*(?:\d{4}-\d{2}-\d{2}T|\[)", content, re.MULTILINE)
    )

    # Strategy 1a: Quick check for JSON-lines (multiline JSON objects)
    # If content has multiple lines and doesn't start with [ or {, try JSON-lines first
    if content_start not in "{[" or "\n" in content.strip():
        logs_data, _ = try_parse_jsonlines(content, source_filename)
        if logs_data:
            # Try direct LogEntry validation first
            try:
                parsed_logs = _validate_log_entries(logs_data)
                if parsed_logs:
                    return parsed_logs, ""
            except Exception:
                pass  # JSON-lines objects don't match LogEntry schema, try normalizing

    if not (content.strip().startswith("{") or content.strip().startswith("[")):
        json_error = None
        logs_data = None
    else:
        json_error = None
        logs_data = None

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
            parsed_logs = _validate_log_entries(logs_data)
            if parsed_logs:
                return parsed_logs, ""
        except Exception as e:
            json_error = f"JSON validation failed: {str(e)[:80]}"

    # If the content clearly looks like JSON but failed to parse, reject plaintext fallback.
    if (content_start in "{[" or content.strip().startswith("{") or content.strip().startswith("[")) and not is_likely_jsonlines:
        # Looks like JSON but failed to parse — don't try plaintext
        return (
            None,
            json_error or "Could not parse JSON file. " 
            "Provide a JSON array or {logs: [...]} object.",
        )

    # Strategy 2: Try logical document parsers before generic structural/line fallback.
    logs_data, _ = try_parse_document_structures(content, source_filename)
    if logs_data:
        try:
            parsed_logs = _validate_log_entries(logs_data)
            if parsed_logs:
                return parsed_logs, ""
        except Exception as e:
            return (
                None,
                f"Document parsing extracted entries but validation failed: {str(e)[:80]}",
            )

    # Strategy 3: Try structural parsers before generic plaintext fallback.
    logs_data, _ = try_parse_structural_formats(content, source_filename)
    if logs_data:
        try:
            parsed_logs = _validate_log_entries(logs_data)
            if parsed_logs:
                return parsed_logs, ""
        except Exception as e:
            return (
                None,
                f"Structural parsing extracted entries but validation failed: {str(e)[:80]}",
            )

    # Strategy 4: Try plaintext parsing (only if content doesn't look like JSON, document structures, or structural formats)
    logs_data, plaintext_error = try_parse_plaintext(content, source_filename)
    if logs_data:
        try:
            parsed_logs = _validate_log_entries(logs_data)
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
