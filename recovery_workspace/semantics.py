from __future__ import annotations

import re
from typing import Any, Optional

from recovery_workspace.models import Event, EventSemantics

TERMINAL_OUTCOME_CODES = {"JOB_FAILED", "RETRY_EXHAUSTED", "PERMISSION_DENIED", "PARTIAL_SUCCESS"}
SUCCESS_EXIT_CODES = {"EXIT_SUCCESS", "SUCCESS", "SUCCEEDED", "OK", "0"}
FAILURE_EXIT_CODES = {"EXIT_FAILURE", "FAILURE", "FAILED", "ERROR", "1"}

MODE_TO_CAUSE_TYPE = {
    "quota_exhaustion": "STORAGE_CAPACITY",
    "permission_failure": "PERMISSION_DENIED",
    "corruption": "REPOSITORY_CORRUPTION",
    "network_timeout": "NETWORK_TIMEOUT",
    "service_unavailable": "SERVICE_UNAVAILABLE",
    "clock_skew": "CLOCK_SKEW",
    "partial_failure": "PARTIAL_FAILURE",
    "resource_access_failure": "RESOURCE_ACCESS",
    "generic_fatal": "UNKNOWN",
}

TERMINAL_SEMANTIC_CODES = {
    "EXIT_FAILURE",
    "FAILURE",
    "FAILED",
    "JOB_FAILED",
    "RETRY_EXHAUSTED",
    "PERMISSION_DENIED",
    "PARTIAL_SUCCESS",
    "QUOTA_EXCEEDED",
    "ACCESS_DENIED",
    "CORRUPTION_DETECTED",
    "ETIMEDOUT",
    "ECONNRESET",
    "ENETUNREACH",
    "SERVICE_UNAVAILABLE",
    "REQUEST_TIME_TOO_SKEWED",
    "UNKNOWN_TERMINAL_FAILURE",
}

NONTERMINAL_SEMANTIC_CODES = {
    "ACCESS_OR_RESOURCE_CONFLICT",
    "LOCK_OR_CONCURRENCY_FAILURE",
    "OPERATION_ABORTED_OR_CANCELLED",
    "RESOURCE_ACCESS_FAILURE",
    "UNKNOWN_OPERATIONAL_ERROR",
}

SEMANTIC_SUPPORTING_FIELDS = (
    "cause",
    "reason",
    "error",
    "exception",
    "detail",
    "failure_reason",
)

SEMANTIC_PLACEHOLDER_CODES = {"PLAINTEXT_FATAL", "RAW_UNPARSED", "NO_CODE", "UNKNOWN_CODE", "JSON", "STRUCTURED_EVENT"}


def _component_key(component: str) -> str:
    return component.replace("_", "-").strip().lower()


def _entry_text(entry: Any) -> str:
    component = str(getattr(entry, "component", "") or "")
    raw_error_code = str(getattr(entry, "error_code", "") or "")
    evidence_text = _canonical_semantic_text(entry)
    return f"{component} {raw_error_code} {evidence_text}".strip().lower()


def _structured_text(entry: Any) -> str:
    return " ".join(_canonical_semantic_supporting_fields(entry)).lower()


def _normalize_semantic_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return " ".join(value.split()).strip()
    return ""


def _looks_like_diagnostic_blob(text: str) -> bool:
    lowered = text.lower()
    if not text:
        return True
    if len(text) > 280:
        return True
    if text.startswith(("{", "[")):
        return True
    if lowered.startswith(("traceback", "stack trace", "suppressed:")):
        return True
    if any(marker in lowered for marker in ("\nat ", " stack=", " traceback ", " traceback:", ".go:", ".py:", ".java:")):
        return True
    if text.count("\n") > 1:
        return True
    return False


def _canonical_semantic_supporting_fields(entry: Any) -> list[str]:
    structured_fields = getattr(entry, "structured_fields", None) or {}
    primary = _normalize_semantic_text(getattr(entry, "message", "") or "")
    seen = {primary.lower()} if primary else set()
    values: list[str] = []

    raw_error_code = _normalize_semantic_text(getattr(entry, "error_code", None))
    if raw_error_code and raw_error_code.upper() not in SEMANTIC_PLACEHOLDER_CODES:
        seen.add(raw_error_code.lower())
        values.append(raw_error_code)

    for key in SEMANTIC_SUPPORTING_FIELDS:
        value = structured_fields.get(key)
        text = _normalize_semantic_text(value)
        if not text:
            continue
        if _looks_like_diagnostic_blob(text):
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        values.append(text)
    return values


def _canonical_semantic_text(entry: Any) -> str:
    primary = _normalize_semantic_text(getattr(entry, "message", "") or "")
    supporting = _canonical_semantic_supporting_fields(entry)
    parts = [part for part in [primary, *supporting] if part]
    return " ".join(parts)


def _observed_cause_text(entry: Any) -> Optional[str]:
    supporting = _canonical_semantic_supporting_fields(entry)
    return supporting[0] if supporting else None


def _structured_message_type(entry: Any) -> str:
    structured_fields = getattr(entry, "structured_fields", None) or {}
    return str(structured_fields.get("message_type", "")).lower()


def _raw_line_text(entry: Any) -> str:
    structured_fields = getattr(entry, "structured_fields", None) or {}
    raw_line = structured_fields.get("raw_line")
    if isinstance(raw_line, str) and raw_line.strip():
        return raw_line.strip()
    return entry.message.strip()


def _has_failure_prefix(entry: Any) -> bool:
    text = _raw_line_text(entry)
    return bool(re.match(r"^\s*(fatal|error|panic)\s*:", text, re.IGNORECASE))


def _has_explicit_terminal_evidence(entry: Any) -> bool:
    raw_line = _raw_line_text(entry).lower()
    message = str(getattr(entry, "message", "") or "").lower()
    structured_fields = getattr(entry, "structured_fields", None) or {}

    explicit_terms = (
        "fatal",
        "panic",
        "job failed",
        "process failed",
        "operation failed",
        "failed to complete",
        "terminal failure",
        "terminated",
        "termination",
        "crashed",
    )

    if any(term in raw_line for term in explicit_terms):
        return True
    if any(term in message for term in explicit_terms):
        return True

    structured_event_type = structured_fields.get("event_type")
    if isinstance(structured_event_type, str) and structured_event_type.strip().upper() == "TERMINAL_FAILURE":
        return True

    structured_terminal_state = structured_fields.get("terminal_state")
    if isinstance(structured_terminal_state, str) and structured_terminal_state.strip().lower() == "failure":
        return True

    return False


def _is_failure_related(entry: Any) -> bool:
    text = _entry_text(entry)
    failure_terms = (
        "fail",
        "failed",
        "unable",
        "cannot",
        "could not",
        "denied",
        "blocked",
        "held",
        "locked",
        "error",
        "fatal",
        "canceled",
        "cancelled",
    )
    return any(term in text for term in failure_terms)


def _mode_from_text(text: str) -> Optional[str]:
    lowered = text.lower()
    if any(term in lowered for term in ("no space left on device", "out of space", "disk full", "quota", "capacity", "enospc")):
        return "quota_exhaustion"
    if any(term in lowered for term in ("wrong password", "no key found", "access denied", "permission denied", "unauthorized", "forbidden", "authentication failed")):
        return "permission_failure"
    if any(term in lowered for term in ("pack id does not match", "checksum mismatch", "corrupt", "corruption", "invalid block", "blob corruption")):
        return "corruption"
    if any(term in lowered for term in ("timed out", "timeout", "packet loss", "connection reset", "network unreachable", "unreachable")):
        return "network_timeout"
    if any(term in lowered for term in ("service unavailable", "503", "maintenance mode", "dependency outage")):
        return "service_unavailable"
    if any(term in lowered for term in ("clock skew", "time skew", "skewed", "ahead of server time")):
        return "clock_skew"
    if any(term in lowered for term in ("partial success", "completed with", "3/4", "volumes succeeded")):
        return "partial_failure"
    if any(term in lowered for term in ("bad file descriptor", "stale file handle", "input/output error", "i/o error", "too many open files", "read-only file system")):
        return "resource_access_failure"
    return None


def _generic_semantic_family(entry: Any) -> Optional[str]:
    haystack = _canonical_semantic_text(entry).lower()

    if any(term in haystack for term in ("access conflict", "resource conflict", "resource busy", "contention", "in use", "busy", "conflict")):
        return "ACCESS_OR_RESOURCE_CONFLICT"
    if any(term in haystack for term in ("lock", "locked", "concurrency", "race", "contention")):
        return "LOCK_OR_CONCURRENCY_FAILURE"
    if any(term in haystack for term in ("cancelled", "canceled", "abort", "aborted")):
        return "OPERATION_ABORTED_OR_CANCELLED"
    if any(term in haystack for term in ("bad file descriptor", "stale file handle", "input/output error", "i/o error", "too many open files", "read-only file system")):
        return "RESOURCE_ACCESS_FAILURE"
    return None


def canonical_error_code(entry: Any) -> Optional[str]:
    semantic_error_code = getattr(entry, "semantic_error_code", None)
    if isinstance(semantic_error_code, str) and semantic_error_code.strip():
        return semantic_error_code.strip().upper()

    exit_code = _extract_exit_code(entry)
    if exit_code in SUCCESS_EXIT_CODES:
        return "EXIT_SUCCESS"
    if exit_code in FAILURE_EXIT_CODES:
        return "EXIT_FAILURE"

    child_exit_codes = _child_exit_codes(entry)
    if child_exit_codes:
        has_success = any(code in SUCCESS_EXIT_CODES for code in child_exit_codes)
        has_failure = any(code in FAILURE_EXIT_CODES for code in child_exit_codes)
        if has_success and has_failure:
            return "MIXED_CHILD_EXITS"
        if has_failure:
            return "EXIT_FAILURE"
        if has_success:
            return "EXIT_SUCCESS"

    text = _entry_text(entry)
    structured_text = _structured_text(entry)
    message_type = _structured_message_type(entry)
    if message_type not in {"error", "exit_error"} and getattr(entry, "level", None) != "ERROR":
        return None

    candidates = [
        ("QUOTA_EXCEEDED", ("no space left on device", "out of space", "disk full", "quota", "capacity", "enospc")),
        ("ACCESS_DENIED", ("wrong password", "no key found", "access denied", "permission denied", "unauthorized", "forbidden", "authentication failed")),
        ("CORRUPTION_DETECTED", ("pack id does not match", "checksum mismatch", "corrupt", "corruption", "integrity", "blob corruption")),
        ("ETIMEDOUT", ("timed out", "timeout", "network timeout")),
        ("ECONNRESET", ("connection reset", "reset by peer")),
        ("ENETUNREACH", ("network unreachable", "unreachable")),
        ("SERVICE_UNAVAILABLE", ("service unavailable", "503", "maintenance mode", "dependency outage")),
        ("REQUEST_TIME_TOO_SKEWED", ("clock skew", "time skew", "skewed", "ahead of server time")),
        ("PARTIAL_SUCCESS", ("partial success", "completed with", "3/4", "volumes succeeded")),
        ("JOB_FAILED", ("job failed", "job marked failed", "marked failed")),
        ("OOM", ("out of memory", "oom", "memory exhaustion")),
        ("RESOURCE_ACCESS_FAILURE", ("bad file descriptor", "stale file handle", "input/output error", "i/o error", "too many open files", "read-only file system")),
    ]

    haystack = f"{structured_text} {text}"
    for code, terms in candidates:
        if any(term in haystack for term in terms):
            return code

    generic_family = _generic_semantic_family(entry)
    if generic_family is not None:
        return generic_family

    if _has_failure_prefix(entry) and _has_explicit_terminal_evidence(entry):
        return "UNKNOWN_TERMINAL_FAILURE"
    if message_type in {"error", "exit_error"} or getattr(entry, "level", None) == "ERROR":
        return "UNKNOWN_OPERATIONAL_ERROR"
    return None


def _infer_event_type(entry: Any, semantic_error_code: Optional[str] = None) -> Optional[str]:
    structured_fields = getattr(entry, "structured_fields", None) or {}
    value = structured_fields.get("event_type")
    if isinstance(value, str) and value.strip():
        return value.strip().upper()

    semantic_error_code = (semantic_error_code or getattr(entry, "semantic_error_code", None) or "").strip().upper()
    if semantic_error_code in {"ACCESS_OR_RESOURCE_CONFLICT"}:
        return "ACCESS_OR_RESOURCE_CONFLICT"
    if semantic_error_code in {"LOCK_OR_CONCURRENCY_FAILURE"}:
        return "LOCK_OR_CONCURRENCY_FAILURE"
    if semantic_error_code in {"OPERATION_ABORTED_OR_CANCELLED"}:
        return "OPERATION_ABORTED_OR_CANCELLED"
    if semantic_error_code in {"RESOURCE_ACCESS_FAILURE"}:
        return "RESOURCE_ACCESS_FAILURE"
    if semantic_error_code in {"UNKNOWN_OPERATIONAL_ERROR"}:
        return "OPERATIONAL_ERROR"
    if semantic_error_code in {"UNKNOWN_TERMINAL_FAILURE", "EXIT_FAILURE", "FAILURE", "FAILED", "JOB_FAILED", "RETRY_EXHAUSTED", "PERMISSION_DENIED", "PARTIAL_SUCCESS", "QUOTA_EXCEEDED", "ACCESS_DENIED", "CORRUPTION_DETECTED", "ETIMEDOUT", "ECONNRESET", "ENETUNREACH", "SERVICE_UNAVAILABLE", "REQUEST_TIME_TOO_SKEWED"}:
        return "TERMINAL_FAILURE"
    if getattr(entry, "level", None) == "WARN":
        return "WARNING"
    if getattr(entry, "level", None) == "INFO":
        return "INFO"
    if getattr(entry, "level", None) == "ERROR":
        return "OPERATIONAL_ERROR"
    return None


def _extract_exit_code(entry: Any) -> Optional[str]:
    structured_fields = getattr(entry, "structured_fields", None) or {}
    for key in ("exit_code", "exitCode", "exit_status", "exitStatus", "status_code", "statusCode"):
        value = structured_fields.get(key)
        if isinstance(value, (int, float)):
            return str(int(value))
        if isinstance(value, str) and value.strip():
            return value.strip().upper()
    if isinstance(getattr(entry, "error_code", None), str) and getattr(entry, "error_code", None).strip():
        return getattr(entry, "error_code", None).strip().upper()
    return None


def _child_exit_codes(entry: Any) -> list[str]:
    structured_fields = getattr(entry, "structured_fields", None) or {}
    values: list[str] = []
    for key in ("child_exit_codes", "childExitCodes", "exit_codes", "exitCodes", "child_statuses", "childStatuses"):
        value = structured_fields.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, (int, float)):
                    values.append(str(int(item)))
                elif isinstance(item, str) and item.strip():
                    values.append(item.strip().upper())
            if values:
                return values
        elif isinstance(value, tuple):
            for item in value:
                if isinstance(item, (int, float)):
                    values.append(str(int(item)))
                elif isinstance(item, str) and item.strip():
                    values.append(item.strip().upper())
            if values:
                return values
    return values


def _infer_terminal_state(entry: Any, semantic_error_code: Optional[str] = None) -> Optional[str]:
    if not semantic_error_code:
        semantic_error_code = getattr(entry, "semantic_error_code", None)
    if isinstance(semantic_error_code, str) and semantic_error_code.strip():
        upper = semantic_error_code.strip().upper()
        if upper in {"EXIT_SUCCESS", "SUCCESS", "SUCCEEDED"}:
            return "success"
        if upper == "MIXED_CHILD_EXITS":
            return "partial"
        if upper in TERMINAL_SEMANTIC_CODES:
            return "failure"
        if upper in NONTERMINAL_SEMANTIC_CODES:
            if upper == "OPERATION_ABORTED_OR_CANCELLED" and _has_explicit_terminal_evidence(entry):
                return "failure"
            return None

    exit_code = _extract_exit_code(entry)
    if exit_code in SUCCESS_EXIT_CODES:
        return "success"
    if exit_code in FAILURE_EXIT_CODES:
        return "failure"

    child_exit_codes = _child_exit_codes(entry)
    if child_exit_codes:
        has_success = any(code in SUCCESS_EXIT_CODES for code in child_exit_codes)
        has_failure = any(code in FAILURE_EXIT_CODES for code in child_exit_codes)
        if has_success and has_failure:
            return "partial"
        if has_failure:
            return "failure"
        if has_success:
            return "success"

    if _component_key(getattr(entry, "component", "")) == "backup-service" and any(
        term in _entry_text(entry)
        for term in ("marked failed", "completed with", "partial_success")
    ):
        return "failure"
    if _component_key(getattr(entry, "component", "")) == "backup-service" and any(
        term in _entry_text(entry)
        for term in ("succeeded", "completed successfully")
    ):
        return "success"
    return None


def _infer_cause_type(entry: Any, semantic_error_code: Optional[str] = None) -> Optional[str]:
    if not semantic_error_code:
        semantic_error_code = getattr(entry, "semantic_error_code", None)
    if isinstance(semantic_error_code, str) and semantic_error_code.strip():
        semantic_error_code = semantic_error_code.strip().upper()
        if semantic_error_code in {"QUOTA_EXCEEDED", "QUOTA_WARNING"}:
            return "STORAGE_CAPACITY"
        if semantic_error_code in {"ACCESS_OR_RESOURCE_CONFLICT"}:
            return "RESOURCE_OR_ACCESS_CONTENTION"
        if semantic_error_code in {"LOCK_OR_CONCURRENCY_FAILURE"}:
            return "LOCK_OR_CONCURRENCY"
        if semantic_error_code in {"RESOURCE_ACCESS_FAILURE"}:
            return "RESOURCE_ACCESS"
        if semantic_error_code in {"UNKNOWN_OPERATIONAL_ERROR", "UNKNOWN_TERMINAL_FAILURE", "OPERATION_ABORTED_OR_CANCELLED"}:
            return None
        if semantic_error_code in {"ACCESS_DENIED", "PERMISSION_DENIED", "UNAUTHORIZED"}:
            return "PERMISSION_DENIED"
        if semantic_error_code in {"CORRUPTION_DETECTED", "PACK_ID_MISMATCH", "CHECKSUM_MISMATCH"}:
            return "REPOSITORY_CORRUPTION"
        if semantic_error_code in {"ETIMEDOUT", "ENETUNREACH", "ECONNRESET"}:
            return "NETWORK_TIMEOUT"
        if semantic_error_code in {"SERVICE_UNAVAILABLE"}:
            return "SERVICE_UNAVAILABLE"
        if semantic_error_code in {"REQUEST_TIME_TOO_SKEWED", "CLOCK_SKEW_DETECTED"}:
            return "CLOCK_SKEW"
        if semantic_error_code in {"PARTIAL_SUCCESS"}:
            return "PARTIAL_FAILURE"
        if semantic_error_code in {"OOM", "OUT_OF_MEMORY", "MEMORY_EXHAUSTION"}:
            return "RESOURCE_EXHAUSTION"

    text = _entry_text(entry)
    if _has_failure_prefix(entry) or _structured_message_type(entry) in {"error", "exit_error"} or getattr(entry, "level", None) in {"WARN", "ERROR"}:
        if any(term in text for term in ("out of memory", "oom", "memory exhaustion")):
            return "RESOURCE_EXHAUSTION"
        mode = _mode_from_text(text)
        if mode is not None:
            return MODE_TO_CAUSE_TYPE.get(mode)

    return None


def derive_event_semantics(event: Event) -> EventSemantics:
    semantic_error_code = canonical_error_code(event)
    return EventSemantics(
        event_type=_infer_event_type(event, semantic_error_code),
        terminal_state=_infer_terminal_state(event, semantic_error_code),
        cause_type=_infer_cause_type(event, semantic_error_code),
        semantic_error_code=semantic_error_code,
        observed_cause=_observed_cause_text(event),
    )


def apply_event_semantics(event: Event) -> Event:
    semantics = derive_event_semantics(event)
    event.event_type = semantics.event_type
    event.terminal_state = semantics.terminal_state
    event.cause_type = semantics.cause_type
    event.semantic_error_code = semantics.semantic_error_code
    event.observed_cause = semantics.observed_cause
    return event
