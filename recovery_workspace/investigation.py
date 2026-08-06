from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)
from pydantic import ValidationError

try:
    import streamlit as st
except ImportError:
    st = None  # For non-Streamlit contexts (tests, etc.)

from recovery_workspace.models import (
    ConfidenceDimensionView,
    Event,
    EventContext,
    EvidenceItem,
    IncidentSummaryView,
    InvestigationContext,
    InvestigationEvidence,
    InvestigationReportView,
    InvestigationSummary,
    RecommendedActionView,
    SupportingEvidenceItem,
    TimelineFindingDetail,
    TimelineEntryContext,
    TimelineReportItem,
    Scenario,
)
from recovery_workspace.evidence_grouping import group_evidence_for_timeline
from recovery_workspace.semantics import canonical_error_code

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

client: Optional[OpenAI] = None
client_timeout_seconds: Optional[float] = None
DEFAULT_OPENAI_TIMEOUT_SECONDS = 8.0


class LLMRequestError(Exception):
    pass


def _format_event_context(event: Event) -> EventContext:
    return EventContext(
        timestamp=event.timestamp.isoformat(),
        component=event.component,
        level=event.level,
        error_code=event.error_code,
        message=event.message,
        source_file=event.source_file,
        structured_fields=event.structured_fields,
        event_type=event.event_type,
        terminal_state=event.terminal_state,
        cause_type=event.cause_type,
        semantic_error_code=event.semantic_error_code,
        observed_cause=event.observed_cause,
    )


def _format_timeline_entry_context(entry: Any) -> TimelineEntryContext:
    timestamp_source = _timeline_entry_timestamp_source(entry)
    timestamp_text = entry.event.timestamp.isoformat() if timestamp_source in {"available", "parsed"} else None
    return TimelineEntryContext(
        sequence=entry.sequence,
        timestamp=timestamp_text,
        component=entry.event.component,
        level=entry.event.level,
        error_code=entry.event.error_code,
        message=entry.event.message,
        gap_before_seconds=entry.gap_before_seconds,
        source_file=entry.event.source_file,
        structured_fields=entry.event.structured_fields,
        event_type=entry.event.event_type,
        terminal_state=entry.event.terminal_state,
        cause_type=entry.event.cause_type,
        semantic_error_code=entry.event.semantic_error_code,
        observed_cause=entry.event.observed_cause,
    )


def _get_llm_client(timeout_seconds: float = DEFAULT_OPENAI_TIMEOUT_SECONDS) -> OpenAI:
    global client, client_timeout_seconds
    if client is not None:
        return client
    
    # Try Streamlit secrets first (for Streamlit Cloud deployment)
    api_key = None
    if st is not None and hasattr(st, "secrets"):
        try:
            api_key = st.secrets.get("OPENAI_API_KEY")
        except Exception:
            pass  # st.secrets not available (e.g., in tests)
    
    # Fall back to environment variable
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        raise LLMRequestError(
            "OPENAI_API_KEY not found. Configure it via: "
            "1. (Streamlit Cloud) App Secrets dashboard, or "
            "2. (Local dev) Environment variable OPENAI_API_KEY or .env file"
        )
    
    client = OpenAI(api_key=api_key, max_retries=0, timeout=timeout_seconds)
    client_timeout_seconds = timeout_seconds
    return client


def build_investigation_context(
    customer: str,
    job_id: str,
    outcome: str,
    expected_failure_signature: str,
    timeline: Any,
    events: list[Event],
) -> InvestigationContext:
    warning_events = [event for event in events if event.level == "WARN"]
    error_events = [event for event in events if event.level == "ERROR"]

    return InvestigationContext(
        customer=customer,
        jobId=job_id,
        outcome=outcome,
        expectedFailureSignature=expected_failure_signature,
        recovery_timeline=[_format_timeline_entry_context(entry) for entry in timeline.entries],
        warning_events=[_format_event_context(event) for event in warning_events],
        error_events=[_format_event_context(event) for event in error_events],
    )


def _extract_json(text: str) -> str:
    text = text.strip()

    # Strip markdown code fences if present.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\n", "", text)
        text = re.sub(r"\n```$", "", text)

    # Attempt to find the first JSON object in the text.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)

    return text


def _is_insufficient_quota_error(exc: RateLimitError) -> bool:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        if body.get("code") == "insufficient_quota" or body.get("type") == "insufficient_quota":
            return True
        error = body.get("error")
        if isinstance(error, dict):
            return (
                error.get("code") == "insufficient_quota"
                or error.get("type") == "insufficient_quota"
            )
    return False


def _call_llm(
    context: InvestigationContext,
    model: str = "gpt-4.1-mini",
    timeout_seconds: float = DEFAULT_OPENAI_TIMEOUT_SECONDS,
) -> str:
    prompt = (
        "Analyze the following structured investigation context and return only valid JSON with the exact "
        "fields likely_root_cause, supporting_evidence, next_actions, and confidence. "
        "Do not include any raw log text or any additional fields."
    )

    content = json.dumps(_reasoning_context_payload(context), indent=2)
    client = _get_llm_client(timeout_seconds=timeout_seconds)
    try:
        response = client.responses.create(
            model=model,
            input=content,
            instructions=prompt,
            temperature=0.0,
            max_output_tokens=400,
        )
    except RateLimitError as exc:
        if _is_insufficient_quota_error(exc):
            raise LLMRequestError(
                "OpenAI request failed: insufficient quota for the configured API key."
            ) from exc
        raise LLMRequestError(f"LLM request failed: {exc}") from exc
    except APITimeoutError as exc:
        raise LLMRequestError("OpenAI request timed out.") from exc
    except APIConnectionError as exc:
        raise LLMRequestError("OpenAI request failed due to a network connection error.") from exc
    except OpenAIError as exc:
        raise LLMRequestError(f"LLM request failed: {exc}") from exc
    except Exception as exc:
        raise LLMRequestError(f"LLM request failed: {exc}") from exc

    text = getattr(response, "output_text", None)
    if not isinstance(text, str) or not text.strip():
        raise LLMRequestError("LLM response did not contain any text output")

    return text


def _reasoning_context_payload(context: InvestigationContext) -> dict[str, Any]:
    payload = context.model_dump(by_alias=True)
    payload.pop("expectedFailureSignature", None)
    return payload


def _generate_investigation_summary(
    context: InvestigationContext,
    model: str = "gpt-4.1-mini",
    timeout_seconds: float = DEFAULT_OPENAI_TIMEOUT_SECONDS,
) -> InvestigationSummary:
    raw_response = _call_llm(context, model=model, timeout_seconds=timeout_seconds)
    json_text = _extract_json(raw_response)

    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM response is not valid JSON") from exc

    try:
        return InvestigationSummary.model_validate(parsed)
    except ValidationError as exc:
        raise ValueError(
            "LLM response did not match the expected investigation summary schema"
        ) from exc


TERMINAL_OUTCOME_CODES = {"JOB_FAILED", "RETRY_EXHAUSTED", "PERMISSION_DENIED", "PARTIAL_SUCCESS"}
FAILURE_RULES = [
    (
        "clock_skew",
        {"REQUEST_TIME_TOO_SKEWED", "CLOCK_SKEW_DETECTED"},
        ("clock skew", "ahead of server time", "request_time_too_skewed", "ntp", "time skew"),
        {"storage", "monitoring", "backup_service"},
    ),
    (
        "quota_exhaustion",
        {"QUOTA_EXCEEDED", "QUOTA_WARNING"},
        ("quota", "capacity", "provisioned", "gb/", "used)", "disk full", "no space left on device", "out of space", "enospc"),
        {"storage", "monitoring", "backup_service"},
    ),
    (
        "permission_failure",
        {"ACCESS_DENIED", "PERMISSION_DENIED", "UNAUTHORIZED"},
        ("forbidden", "lacks", "access denied", "permission denied", "principal", "wrong password", "no key found", "authentication failed"),
        {"storage", "auth", "backup_service"},
    ),
    (
        "corruption",
        {"CORRUPTION_DETECTED", "PACK_ID_MISMATCH", "CHECKSUM_MISMATCH"},
        ("pack id does not match", "corrupt", "corruption", "checksum mismatch", "invalid block", "blob corruption"),
        {"storage", "backup_service"},
    ),
    (
        "network_timeout",
        {"ETIMEDOUT", "ENETUNREACH", "ECONNRESET", "PACKET_LOSS"},
        ("timeout", "timed out", "packet loss", "tunnel", "connection reset", "network"),
        {"network", "storage", "backup_service"},
    ),
    (
        "service_unavailable",
        {"SERVICE_UNAVAILABLE"},
        ("503", "maintenance mode", "service unavailable", "dependency outage", "unavailable"),
        None,
    ),
    (
        "partial_failure",
        {"PARTIAL_SUCCESS"},
        ("partial_success", "completed with", "3/4", "task", "volumes succeeded"),
        {"backup_service", "network", "storage"},
    ),
    (
        "system_health_degradation",
        {"SYSTEM_HEALTH_DEGRADATION"},
        (
            "health_err",
            "health_warn",
            "critical health",
            "degraded",
            "inactive",
            "stale",
            "undersized",
            "unclean",
            "down",
            "unavailable",
        ),
        None,
    ),
    (
        "resource_access_failure",
        {"RESOURCE_ACCESS_FAILURE"},
        ("bad file descriptor", "stale file handle", "input/output error", "i/o error", "too many open files", "read-only file system"),
        None,
    ),
]
EXACT_SIGNAL_WEIGHTS = {
    "QUOTA_EXCEEDED": 0.3,
    "ACCESS_DENIED": 0.3,
    "CORRUPTION_DETECTED": 0.3,
    "PACK_ID_MISMATCH": 0.3,
    "CHECKSUM_MISMATCH": 0.3,
    "ETIMEDOUT": 0.28,
    "ENETUNREACH": 0.28,
    "ECONNRESET": 0.28,
    "SERVICE_UNAVAILABLE": 0.28,
    "REQUEST_TIME_TOO_SKEWED": 0.3,
    "CLOCK_SKEW_DETECTED": 0.3,
    "PARTIAL_SUCCESS": 0.18,
    "SYSTEM_HEALTH_DEGRADATION": 0.22,
}


def _evidence_timestamp_source(entry: TimelineEntryContext) -> str:
    structured_fields = getattr(entry, "structured_fields", None) or {}
    value = structured_fields.get("timestamp_source")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "parsed"


def _evidence_source_file(entry: TimelineEntryContext) -> Optional[str]:
    source_file = getattr(entry, "source_file", None)
    if isinstance(source_file, str) and source_file.strip():
        return source_file.strip()
    structured_fields = getattr(entry, "structured_fields", None) or {}
    for key in ("source_file", "source_filename", "file", "path"):
        value = structured_fields.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _legacy_evidence_event_type_fallback(entry: TimelineEntryContext) -> Optional[str]:
    structured_fields = getattr(entry, "structured_fields", None) or {}
    value = structured_fields.get("event_type") or structured_fields.get("message_type")
    if isinstance(value, str) and value.strip():
        return value.strip().upper()
    if entry.level == "ERROR":
        return "TERMINAL_FAILURE"
    if entry.level == "WARN":
        return "WARNING"
    if entry.level == "INFO":
        return "INFO"
    return None


def _evidence_event_type(entry: TimelineEntryContext) -> Optional[str]:
    value = getattr(entry, "event_type", None)
    if isinstance(value, str) and value.strip():
        return value.strip().upper()
    return _legacy_evidence_event_type_fallback(entry)


def _legacy_evidence_terminal_state_fallback(entry: TimelineEntryContext) -> Optional[str]:
    return None


def _evidence_terminal_state(entry: TimelineEntryContext, role: str) -> Optional[str]:
    value = getattr(entry, "terminal_state", None)
    if isinstance(value, str) and value.strip():
        return value.strip().lower()
    return _legacy_evidence_terminal_state_fallback(entry)


def _legacy_evidence_cause_type_fallback(entry: TimelineEntryContext) -> Optional[str]:
    return None


def _evidence_cause_type(entry: TimelineEntryContext) -> Optional[str]:
    value = getattr(entry, "cause_type", None)
    if isinstance(value, str) and value.strip():
        return value.strip().upper()
    return _legacy_evidence_cause_type_fallback(entry)


def _shorten_message(message: str, *, limit: int = 96) -> str:
    cleaned = " ".join(message.split()).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def _humanize_identifier(value: Optional[str]) -> str:
    if not value:
        return ""
    text = re.sub(r"[_\-]+", " ", value).strip()
    if not text:
        return ""
    return " ".join(part.capitalize() for part in text.split())


def _strip_internal_placeholders(text: str) -> str:
    cleaned = text
    for token in ("PLAINTEXT_FATAL", "RAW_UNPARSED", "NO_CODE", "UNKNOWN_CODE"):
        cleaned = re.sub(rf"\b{re.escape(token)}\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" \t\n\r:-—")
    return cleaned


def _role_fallback(role: str) -> str:
    return {
        "CONTEXT": "Operation context recorded.",
        "CONTRIBUTING": "A contributing operational issue was detected.",
        "PRIMARY_CAUSE": "A direct failure signal was identified.",
        "PROPAGATION": "The failure prevented the next processing step.",
        "OUTCOME": "The operation ended unsuccessfully.",
    }.get(role, "Operational failure detected.")


def _generic_sentence(text: str) -> str:
    cleaned = _strip_internal_placeholders(" ".join(text.split()))
    if not cleaned:
        return ""
    cleaned = re.sub(r"^(?:fatal|error|panic|warning)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip(" \t\n\r:-—")
    if not cleaned:
        return ""
    return cleaned[0].upper() + cleaned[1:]


def _concise_observed_message(text: str) -> str:
    cleaned = _strip_internal_placeholders(" ".join(text.split()))
    cleaned = re.sub(
        r"^[A-Za-z0-9_.-]+\s+[A-Za-z0-9_.-]+\s+(INFO|WARN|ERROR|WARNING|DEBUG)\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"^(level|logger|message|thread|throwable|cause|stack|suppressed)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    return _generic_sentence(cleaned)


def _format_report_timestamp(timestamp_text: Optional[str], timestamp_source: Optional[str]) -> str:
    if not timestamp_text or timestamp_source not in {"available", "parsed"}:
        return "Timestamp unavailable"
    parsed = datetime.fromisoformat(timestamp_text.replace("Z", "+00:00"))
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _extract_http_failure_observation(text: str) -> Optional[str]:
    compact = " ".join(text.split())
    status_match = re.search(r"\bHTTP\s*(?P<status>[45]\d\d)\b", compact, re.IGNORECASE)
    if not status_match:
        status_match = re.search(r"\b(?P<status>[45]\d\d)\s+(?P<phrase>Unauthorized|Forbidden|Not Found|Conflict|Bad Request)\b", compact, re.IGNORECASE)
    if not status_match:
        return None

    status = status_match.group("status")
    reason_match = re.search(r"\b(Unauthorized|Forbidden|Conflict|Bad Request|Service Unavailable)\b", compact, re.IGNORECASE)
    reason = reason_match.group(1) if reason_match else "response"
    method_match = re.search(r"\b(GET|PUT|POST|DELETE|PATCH|HEAD|OPTIONS)\b", compact, re.IGNORECASE)
    method = method_match.group(1).upper() if method_match else "request"
    if method == "REQUEST":
        return f"A downstream request received HTTP {status} {reason}."
    return f"A downstream {method} request received HTTP {status} {reason}."


def _timeline_entry_timestamp_source(entry: Any) -> str:
    structured_fields = getattr(entry.event, "structured_fields", None) or {}
    value = structured_fields.get("timestamp_source")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "parsed"


def _derive_incident_operation(scenario: Scenario) -> str:
    observed_text = " ".join(
        value for value in (scenario.name, scenario.description, scenario.job_id) if isinstance(value, str)
    ).lower()
    if "restore" in observed_text:
        return "Recovery Restore"
    if "weekly" in observed_text and "backup" in observed_text:
        return "Scheduled Full Backup"
    if "nightly" in observed_text and "backup" in observed_text:
        return "Scheduled Nightly Backup"
    if "backup" in observed_text and scenario.category != "uploaded":
        return "Scheduled Backup"
    return "Unknown"


def _derive_incident_target(timeline: Any) -> str:
    patterns = (
        r"storage account ([A-Za-z0-9\-_]+)",
        r"on ([A-Za-z0-9\-_]+-bucket)",
        r"for ([A-Za-z0-9\-_]+/[A-Za-z0-9\-_.]+)",
        r"endpoint\s+([A-Za-z0-9:/._\-]+)",
        r"url\s+([A-Za-z0-9:/._\-]+)",
    )
    for entry in timeline.entries:
        message = entry.event.message
        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                return match.group(1)
    return "Not provided"


def _derive_incident_environment(scenario: Scenario, timeline: Any) -> str:
    candidates = [scenario.name, scenario.description, scenario.job_id]
    candidates.extend(entry.event.message for entry in timeline.entries)
    joined = " ".join(text for text in candidates if isinstance(text, str)).lower()
    if re.search(r"\bproduction\b", joined):
        return "Production"
    if re.search(r"\bstaging\b", joined):
        return "Staging"
    if re.search(r"\bdevelopment\b", joined) or re.search(r"\bdev\b", joined):
        return "Development"
    return "Not provided"


def _build_incident_summary_view(scenario: Scenario, timeline: Any) -> IncidentSummaryView:
    first_entry = timeline.entries[0]
    last_entry = timeline.entries[-1]
    started_source = _timeline_entry_timestamp_source(first_entry)
    ended_source = _timeline_entry_timestamp_source(last_entry)
    started = _format_report_timestamp(first_entry.event.timestamp.isoformat(), started_source)
    ended = _format_report_timestamp(last_entry.event.timestamp.isoformat(), ended_source)
    duration = (
        f"{timeline.duration_seconds:.0f}s"
        if started_source in {"available", "parsed"} and ended_source in {"available", "parsed"}
        else "Unknown"
    )

    return IncidentSummaryView(
        customer=scenario.customer,
        job_id=scenario.job_id,
        status=scenario.expected_outcome,
        operation=_derive_incident_operation(scenario),
        target=_derive_incident_target(timeline),
        environment=_derive_incident_environment(scenario, timeline),
        started=started,
        ended_label="Failed At" if scenario.expected_outcome == "FAILED" else "Ended At",
        ended=ended,
        duration=duration,
    )


def _report_confidence_explanation(evidence: InvestigationEvidence) -> list[str]:
    lines: list[str] = []
    for line in evidence.confidence_explanation:
        if line not in lines:
            lines.append(line)
    for line in evidence.limitations:
        formatted = line if line.startswith(("•", "✓")) else f"• {line}"
        if formatted not in lines:
            lines.append(formatted)
    return lines


def build_evidence_display(entry: TimelineEntryContext, role: str) -> tuple[str, str]:
    structured_fields = getattr(entry, "structured_fields", None) or {}
    semantic_code = (getattr(entry, "semantic_error_code", None) or canonical_error_code(entry) or "").upper()
    observed_cause = (getattr(entry, "observed_cause", None) or "").strip()

    if semantic_code == "RESOURCE_ACCESS_FAILURE":
        summary = "Object metadata access failed." if "object info" in entry.message.lower() else "Resource access failed."
        if "object info" in entry.message.lower():
            detail = (
                f"The service could not read object metadata because the underlying system returned {observed_cause}."
                if observed_cause
                else "The service could not read object metadata because the underlying system returned a low-level resource-access error."
            )
        else:
            detail = (
                f"The service could not access the required resource because the underlying system returned {observed_cause}."
                if observed_cause
                else "The service could not access the required resource because the underlying system returned a low-level resource-access error."
            )
        return summary, detail

    if semantic_code == "SYSTEM_HEALTH_DEGRADATION":
        return (
            "Critical system health degradation detected.",
            "The supplied status snapshot shows degraded/inactive resources and unavailable services.",
        )

    semantic_bits: list[str] = []
    for value in (
        getattr(entry, "cause_type", None),
        getattr(entry, "event_type", None),
        getattr(entry, "terminal_state", None),
        structured_fields.get("message_type"),
    ):
        if isinstance(value, str) and value.strip():
            humanized = _humanize_identifier(value)
            if humanized and humanized.upper() not in {
                "PLAINTEXT FATAL",
                "RAW UNPARSED",
                "NO CODE",
                "UNKNOWN CODE",
                "UNKNOWN",
                "ERROR",
                "INFO",
                "WARN",
                "WARNING",
                "TERMINAL FAILURE",
            }:
                semantic_bits.append(humanized)

    clean_message = _generic_sentence(entry.message)
    summary = ""
    if semantic_bits:
        if role == "PRIMARY_CAUSE":
            summary = f"{semantic_bits[0]} failure detected."
        elif role == "OUTCOME":
            summary = f"{semantic_bits[0]} outcome recorded."
        elif role == "CONTRIBUTING":
            summary = f"{semantic_bits[0]} contributing condition."
        elif role == "PROPAGATION":
            summary = f"{semantic_bits[0]} propagation event."
        else:
            summary = f"{semantic_bits[0]} context."
    elif clean_message:
        summary = clean_message
    else:
        summary = _role_fallback(role)

    detail = ""
    if clean_message:
        if role == "CONTEXT":
            detail = f"Context: {clean_message}"
        elif role == "CONTRIBUTING":
            detail = f"Contributing issue: {clean_message}"
        elif role == "PRIMARY_CAUSE":
            detail = f"Primary cause: {clean_message}"
        elif role == "PROPAGATION":
            detail = f"Propagation: {clean_message}"
        elif role == "OUTCOME":
            detail = f"Outcome: {clean_message}"
        else:
            detail = clean_message
    else:
        detail = _role_fallback(role)

    if not summary:
        summary = _role_fallback(role)
    if not detail:
        detail = summary
    return summary, detail


def _evidence_signature(entry: TimelineEntryContext) -> tuple[str, str, str, str]:
    return (
        entry.component.lower().strip(),
        (entry.error_code or "").strip().lower(),
        entry.level.lower().strip(),
        _shorten_message(entry.message, limit=120).lower(),
    )


def _semantic_causal_strength(entry: TimelineEntryContext) -> int:
    score = 0
    event_type = (getattr(entry, "event_type", None) or "").upper()
    semantic_error_code = (getattr(entry, "semantic_error_code", None) or canonical_error_code(entry) or "").upper()
    cause_type = (getattr(entry, "cause_type", None) or "").upper()

    if cause_type in {
        "STORAGE_CAPACITY",
        "RESOURCE_EXHAUSTION",
        "RESOURCE_ACCESS",
        "SYSTEM_DEGRADATION",
        "AUTHENTICATION",
        "AUTHORIZATION",
        "REPOSITORY_CORRUPTION",
        "NETWORK_TIMEOUT",
        "SERVICE_UNAVAILABLE",
        "CONFIGURATION_FAILURE",
        "DEPENDENCY_FAILURE",
        "LOCK_OR_CONCURRENCY",
        "RESOURCE_OR_ACCESS_CONTENTION",
    }:
        score += 4

    if event_type in {
        "LOCK_OR_CONCURRENCY_FAILURE",
        "TERMINAL_FAILURE",
        "PARTIAL_FAILURE",
        "SYSTEM_HEALTH_DEGRADATION",
    }:
        score += 3

    if semantic_error_code in {
        "LOCK_OR_CONCURRENCY_FAILURE",
        "RESOURCE_ACCESS_FAILURE",
        "RESOURCE_EXHAUSTION",
        "SYSTEM_HEALTH_DEGRADATION",
        "QUOTA_EXCEEDED",
        "ACCESS_DENIED",
        "PERMISSION_DENIED",
        "CORRUPTION_DETECTED",
        "ETIMEDOUT",
        "ENETUNREACH",
        "ECONNRESET",
        "SERVICE_UNAVAILABLE",
        "REQUEST_TIME_TOO_SKEWED",
        "UNKNOWN_TERMINAL_FAILURE",
    }:
        score += 2

    if event_type in {"ACCESS_OR_RESOURCE_CONFLICT"}:
        score += 1

    if entry.level == "ERROR":
        score += 1
    elif entry.level == "WARN":
        score += 0

    if entry.error_code and entry.error_code not in {"PLAINTEXT", "RAW_UNPARSED", "ERROR", "PLAINTEXT_FATAL"}:
        score += 1

    return score


def _is_outcome_signal(entry: TimelineEntryContext) -> bool:
    event_type = (getattr(entry, "event_type", None) or "").upper()
    semantic_error_code = (getattr(entry, "semantic_error_code", None) or canonical_error_code(entry) or "").upper()
    terminal_state = (getattr(entry, "terminal_state", None) or "").lower()
    raw_error_code = (getattr(entry, "error_code", None) or "").upper()

    if event_type in {"TERMINAL_FAILURE", "TERMINAL_SUCCESS", "PARTIAL_FAILURE", "OPERATION_ABORTED_OR_CANCELLED"}:
        return True
    if semantic_error_code in {"EXIT_FAILURE", "EXIT_SUCCESS", "MIXED_CHILD_EXITS", "JOB_FAILED", "RETRY_EXHAUSTED", "PARTIAL_SUCCESS", "OPERATION_ABORTED_OR_CANCELLED"}:
        return True
    if raw_error_code in {"JOB_FAILED", "EXIT_FAILURE", "EXIT_SUCCESS", "PARTIAL_SUCCESS", "RETRY_EXHAUSTED"}:
        return True
    return False


def _is_direct_causal_signal(entry: TimelineEntryContext) -> bool:
    return _semantic_causal_strength(entry) > 0 and not _is_outcome_signal(entry)


def _is_explicit_propagation_signal(entry: TimelineEntryContext) -> bool:
    event_type = (getattr(entry, "event_type", None) or "").upper()
    return event_type == "PROPAGATION"


def _is_progress_noise(entry: TimelineEntryContext) -> bool:
    text = _entry_text(entry)
    noisy_terms = (
        "progress",
        "scanning",
        "copying",
        "transferring",
        "remaining",
        "bytes",
        "items processed",
        "files processed",
        "percent",
        "step",
    )
    if any(term in text for term in noisy_terms) and not any(
        term in text for term in ("fail", "error", "fatal", "unable", "cannot", "warn")
    ):
        return True
    return False


def _is_failure_related(entry: TimelineEntryContext) -> bool:
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


def _primary_candidate_score(entry: TimelineEntryContext) -> int:
    score = 0
    if _is_progress_noise(entry):
        return -100
    if _is_failure_related(entry):
        score += 2
    if entry.level == "ERROR":
        score += 2
    elif entry.level == "WARN":
        score += 1
    canonical = _canonical_error_code(entry)
    if canonical in EXACT_SIGNAL_WEIGHTS:
        score += 3
    elif entry.error_code and entry.error_code not in {"PLAINTEXT", "RAW_UNPARSED", "ERROR", "PLAINTEXT_FATAL"}:
        score += 1
    if _evidence_timestamp_source(entry) in {"inherited", "interpolated", "unavailable"}:
        score -= 1
    return score


def _evidence_item(
    entry: TimelineEntryContext,
    *,
    role: str,
    mode: str,
) -> EvidenceItem:
    display_summary, display_detail = build_evidence_display(entry, role)
    return EvidenceItem(
        event_id=f"{entry.sequence}",
        sequence=entry.sequence,
        timestamp=entry.timestamp,
        timestamp_source=_evidence_timestamp_source(entry),
        component=entry.component,
        event_type=_evidence_event_type(entry),
        terminal_state=_evidence_terminal_state(entry, role),
        cause_type=_evidence_cause_type(entry),
        error_code=entry.error_code or canonical_error_code(entry),
        semantic_error_code=getattr(entry, "semantic_error_code", None) or canonical_error_code(entry),
        observed_cause=getattr(entry, "observed_cause", None),
        original_message=entry.message,
        message=entry.message,
        role=role,  # type: ignore[arg-type]
        source_file=_evidence_source_file(entry),
        structured_fields=dict(entry.structured_fields),
        summary=display_summary,
        display_summary=display_summary,
        display_detail=display_detail,
    )


def _entry_text(entry: TimelineEntryContext) -> str:
    structured_fields = getattr(entry, "structured_fields", None) or {}
    structured_bits: list[str] = []
    for key in (
        "message_type",
        "action",
        "error",
        "message",
        "details",
        "description",
        "status",
        "item",
        "raw_line",
        "parse_status",
    ):
        value = structured_fields.get(key)
        if isinstance(value, str) and value.strip():
            structured_bits.append(value)
    return f"{entry.component} {entry.error_code or ''} {entry.message} {' '.join(structured_bits)}".lower()


def _component_key(component: str) -> str:
    return component.replace("_", "-").strip().lower()


def _raw_line_text(entry: TimelineEntryContext) -> str:
    structured_fields = getattr(entry, "structured_fields", None) or {}
    raw_line = structured_fields.get("raw_line")
    if isinstance(raw_line, str) and raw_line.strip():
        return raw_line.strip()
    return entry.message.strip()


def _has_failure_prefix(entry: TimelineEntryContext) -> bool:
    text = _raw_line_text(entry)
    return bool(re.match(r"^\s*(fatal|error|panic)\s*:", text, re.IGNORECASE))


def _failure_blob(entries: list[TimelineEntryContext], index: int, radius: int = 1) -> str:
    start = max(0, index - radius)
    end = min(len(entries), index + radius + 1)
    return " ".join(_entry_text(entries[i]) for i in range(start, end))


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


def _structured_text(entry: TimelineEntryContext) -> str:
    structured_fields = getattr(entry, "structured_fields", None) or {}
    bits: list[str] = []
    for key in (
        "message_type",
        "action",
        "error",
        "message",
        "details",
        "description",
        "status",
        "item",
        "raw_line",
        "parse_status",
    ):
        value = structured_fields.get(key)
        if isinstance(value, str) and value.strip():
            bits.append(value.strip())
    return " ".join(bits).lower()


def _structured_message_type(entry: TimelineEntryContext) -> str:
    structured_fields = getattr(entry, "structured_fields", None) or {}
    return str(structured_fields.get("message_type", "")).lower()


def _canonical_error_code(entry: TimelineEntryContext) -> Optional[str]:
    return canonical_error_code(entry)


def _inferred_failure_mode(
    entries: list[TimelineEntryContext],
    index: int,
) -> Optional[str]:
    entry = entries[index]
    canonical = _canonical_error_code(entry)
    if canonical in {"QUOTA_EXCEEDED"}:
        return "quota_exhaustion"
    if canonical in {"ACCESS_DENIED"}:
        return "permission_failure"
    if canonical in {"CORRUPTION_DETECTED", "PACK_ID_MISMATCH", "CHECKSUM_MISMATCH"}:
        return "corruption"
    if canonical in {"ETIMEDOUT", "ENETUNREACH", "ECONNRESET"}:
        return "network_timeout"
    if canonical in {"SERVICE_UNAVAILABLE"}:
        return "service_unavailable"
    if canonical in {"REQUEST_TIME_TOO_SKEWED"}:
        return "clock_skew"
    if canonical in {"PARTIAL_SUCCESS"}:
        return "partial_failure"
    if canonical in {"SYSTEM_HEALTH_DEGRADATION"}:
        return "system_health_degradation"
    if canonical in {"RESOURCE_ACCESS_FAILURE"}:
        return "resource_access_failure"
    if _has_failure_prefix(entry) or _structured_message_type(entry) in {"error", "exit_error"} or entry.level in {"WARN", "ERROR"}:
        mode = _mode_from_text(_failure_blob(entries, index, radius=1))
        if mode:
            return mode
        if _has_failure_prefix(entry) or _structured_message_type(entry) in {"error", "exit_error"} or entry.level == "ERROR":
            return "generic_fatal"
    return None


def _is_terminal_outcome_event(entry: TimelineEntryContext) -> bool:
    text = _entry_text(entry)
    if entry.error_code in TERMINAL_OUTCOME_CODES:
        return True
    return (
        _component_key(entry.component) == "backup-service"
        and ("marked failed" in text or "completed with" in text or "partial_success" in text)
    )


def _find_terminal_sequence(entries: list[TimelineEntryContext]) -> Optional[int]:
    terminal_entries = [entry for entry in entries if _is_terminal_outcome_event(entry)]
    if not terminal_entries:
        fallback_entries = [
            entry
            for entry in entries
            if entry.level == "ERROR"
            and (_is_failure_related(entry) or _has_failure_prefix(entry) or _structured_message_type(entry) in {"error", "exit_error"})
        ]
        if not fallback_entries:
            return None
        return fallback_entries[-1].sequence
    return terminal_entries[-1].sequence


def _find_matching_entries(
    entries: list[TimelineEntryContext],
    *,
    mode: str,
    error_codes: set[str],
    message_terms: tuple[str, ...],
    components: Optional[set[str]] = None,
) -> list[TimelineEntryContext]:
    matched: list[TimelineEntryContext] = []
    component_keys = {_component_key(component) for component in components} if components is not None else None
    for index, entry in enumerate(entries):
        text = _failure_blob(entries, index, radius=1) if (_has_failure_prefix(entry) or _structured_message_type(entry) in {"error", "exit_error"} or entry.level in {"WARN", "ERROR"}) else _entry_text(entry)
        code_match = entry.error_code in error_codes if entry.error_code else False
        message_match = any(term in text for term in message_terms)
        component_match = component_keys is None or _component_key(entry.component) in component_keys
        inferred_mode = _inferred_failure_mode(entries, index)
        mode_match = inferred_mode == mode
        if component_match and (code_match or message_match or mode_match):
            matched.append(entry)
    return matched


def _latest_causal_event(
    matches: list[TimelineEntryContext],
    terminal_sequence: Optional[int],
) -> Optional[TimelineEntryContext]:
    non_terminal = [entry for entry in matches if not _is_terminal_outcome_event(entry)]
    if terminal_sequence is not None:
        before_terminal = [entry for entry in non_terminal if entry.sequence <= terminal_sequence]
        if before_terminal:
            return before_terminal[-1]
    if non_terminal:
        return non_terminal[-1]
    if terminal_sequence is not None:
        before_terminal_any = [entry for entry in matches if entry.sequence <= terminal_sequence]
        if before_terminal_any:
            return before_terminal_any[-1]
    if matches:
        return matches[-1]
    return None


def _rule_match_collections(context: InvestigationContext) -> dict[str, list[TimelineEntryContext]]:
    entries = context.recovery_timeline
    collections: dict[str, list[TimelineEntryContext]] = {}
    for mode, codes, terms, components in FAILURE_RULES:
        collections[mode] = _find_matching_entries(
            entries,
            mode=mode,
            error_codes=codes,
            message_terms=terms,
            components=components,
        )
    return collections


def _fatal_fallback_entries(entries: list[TimelineEntryContext]) -> list[TimelineEntryContext]:
    return [
        entry
        for entry in entries
        if (
            _has_failure_prefix(entry)
            or (entry.level == "ERROR" and _is_failure_related(entry))
            or entry.error_code in {"PLAINTEXT_FATAL", "RAW_UNPARSED"}
        )
    ]


def _classify_failure_mode(
    context: InvestigationContext,
) -> tuple[str, list[TimelineEntryContext], dict[str, list[TimelineEntryContext]], dict[str, int]]:
    entries = context.recovery_timeline
    terminal_sequence = _find_terminal_sequence(entries)
    collections = _rule_match_collections(context)
    best_mode = "unknown"
    best_matches: list[TimelineEntryContext] = []
    best_score = -1
    rule_scores: dict[str, int] = {}

    for mode, codes, _terms, _components in FAILURE_RULES:
        matches = collections[mode]
        if not matches:
            rule_scores[mode] = 0
            continue

        score = len(matches)
        causal = _latest_causal_event(matches, terminal_sequence)
        if causal is not None and _canonical_error_code(causal) in codes:
            score += 3
        if any(entry.level == "WARN" for entry in matches):
            score += 1
        if terminal_sequence is not None and any(entry.sequence <= terminal_sequence for entry in matches):
            score += 1
        if mode == "partial_failure" and any(
            "completed with" in _entry_text(entry) or (entry.error_code == "PARTIAL_SUCCESS")
            for entry in matches
        ):
            score += 2
        rule_scores[mode] = score

        if score > best_score:
            best_mode = mode
            best_matches = matches
            best_score = score

    if best_mode == "unknown":
        fatal_entries = _fatal_fallback_entries(entries)
        if fatal_entries:
            best_mode = "generic_fatal"
            best_matches = fatal_entries
            best_score = len(fatal_entries)
            rule_scores["generic_fatal"] = best_score

    return best_mode, best_matches, collections, rule_scores


def _evidence_grounded_actions(
    mode: str, matched_entries: list[TimelineEntryContext]
) -> list[str]:
    """Generate actions grounded in actual evidence from matched entries, augmented by failure mode."""
    
    # Always use mode-specific actions as baseline, then enhance with evidence-specific details
    # This ensures partial_failure gets partial_failure actions, not just network actions
    
    mode_specific_actions = {
        "quota_exhaustion": [
            "Increase destination storage capacity or prune old backup artifacts before the next run.",
            "Trigger proactive quota alerts earlier so backup jobs are blocked before upload begins.",
            "Mark quota-exceeded failures as non-transient and skip retry loops until capacity is restored.",
        ],
        "permission_failure": [
            "Restore read permissions for the backup service principal on the affected storage path.",
            "Audit the recent policy change and roll back least-privilege scope reductions that removed required actions.",
            "Add a preflight authorization check before restore jobs start.",
        ],
        "network_timeout": [
            "Stabilize the failing network path (tunnel flap/packet loss) before rerunning the restore.",
            "Tune timeout and retry policies for intermittent transport instability.",
            "Add targeted alerts for ETIMEDOUT/ENETUNREACH/ECONNRESET spikes on backup traffic.",
        ],
        "service_unavailable": [
            "Coordinate with the dependency owner to schedule maintenance windows outside backup execution.",
            "Keep backoff-retry behavior for transient 503 outages and alert when outage duration exceeds retry budget.",
            "Add dependency health prechecks before starting large upload phases.",
        ],
        "clock_skew": [
            "Repair host time synchronization (NTP) on the backup nodes before the next signed request.",
            "Add clock-skew drift alerts with thresholds below the storage signature tolerance window.",
            "Force a time-sync validation precheck before backup upload signing.",
        ],
        "partial_failure": [
            "Isolate failed child tasks and re-run only the failed volumes instead of the entire job.",
            "Track per-task success/failure metrics so partial outcomes trigger explicit remediation workflows.",
            "Review whether child task failures are related to resource contention or environmental issues.",
        ],
        "system_health_degradation": [
            "Inspect unavailable daemon, service, and resource status logs around the captured snapshot window.",
            "Verify host and device health for resources reported degraded, inactive, stale, or down.",
            "Review recovery, rebalance, or remapping state for resources stuck in degraded or inactive states.",
            "Collect deeper service and operating-system evidence to identify the underlying infrastructure cause.",
        ],
        "resource_access_failure": [
            "Inspect service, operating-system, and I/O logs around the failure timestamp.",
            "Verify the health of the affected resource and host before retrying.",
            "Correlate repeated metadata-access failures using the affected resource and surrounding time window.",
            "Retry only after confirming the resource and service are healthy.",
            "Alert on repeated metadata-access or low-level I/O failures and preserve correlation identifiers with structured cause fields.",
        ],
        "corruption": [
            "Verify repository integrity and repair or recreate corrupted packs before rerunning the job.",
            "Check for disk or storage path issues that may be truncating writes or corrupting blob data.",
            "Add integrity checks earlier so corruption is surfaced before the terminal backup step.",
        ],
        "generic_fatal": [
            "Inspect the fatal line together with the immediately surrounding log lines for the missing failure family.",
            "Preserve raw plaintext or structured error output so generic fatal lines are not lost during normalization.",
            "Add richer fatal-line heuristics for unknown-but-failing events.",
        ],
    }
    
    # If we have mode-specific actions, start with those
    if mode in mode_specific_actions:
        return mode_specific_actions[mode]
    
    # Fallback for unknown modes: use evidence-grounded detection
    evidence_text = " ".join(_entry_text(entry).lower() for entry in matched_entries)
    error_codes = {(entry.error_code or "").lower() for entry in matched_entries}
    
    # Check for specific symptoms
    has_quota = any(
        phrase in evidence_text
        for phrase in ["quota", "space", "capacity", "disk full", "no space left", "enospc"]
    ) or "quota_exceeded" in error_codes or "quota_warning" in error_codes
    
    has_oom = any(
        phrase in evidence_text
        for phrase in ["oom", "out of memory", "memory exhaustion", "sigkill", "memory watchdog"]
    )
    
    has_corruption = any(
        phrase in evidence_text
        for phrase in ["corrupt", "checksum", "pack id", "blob", "integrity", "mismatch"]
    ) or any(code in error_codes for code in ["corruption_detected", "pack_id_mismatch", "checksum_mismatch"])
    
    has_auth = (
        "access_denied" in error_codes
        or "permission_denied" in error_codes
        or "unauthorized" in error_codes
        or any(
            phrase in evidence_text
            for phrase in ["permission denied", "access denied", "authentication failed", "wrong password", "no key found", "principal lacks", "lacks ", "scope narrowed"]
        )
    )
    
    has_network = any(
        phrase in evidence_text
        for phrase in ["etimedout", "enetunreach", "econnreset", "packet loss", "connection reset", "tunnel"]
    ) or any(code in error_codes for code in ["etimedout", "enetunreach", "econnreset"])
    
    # For unknown modes, check symptoms in order of priority
    if has_quota:
        return mode_specific_actions["quota_exhaustion"]
    if has_corruption:
        return mode_specific_actions["corruption"]
    if has_oom:
        return [
            "Review memory allocation and limits for backup worker processes — consider increasing heap/RSS caps or reducing concurrent job parallelism.",
            "Enable memory pressure monitoring with earlier warnings before OOM conditions trigger system watchdogs.",
            "Profile peak memory usage during large data scans to establish resource requirements.",
        ]
    if has_auth:
        return mode_specific_actions["permission_failure"]
    if has_network:
        return mode_specific_actions["network_timeout"]
    
    # Generic fallback for truly unknown modes
    return [
        "Review the most recent non-terminal error event and its upstream dependencies.",
        "Add targeted alerts around repeated WARN/ERROR patterns preceding terminal job outcomes.",
        "Capture richer component diagnostics at failure boundaries to improve deterministic classification.",
    ]


def _root_cause_text(mode: str, causal_event: Optional[TimelineEntryContext]) -> str:
    if causal_event is not None:
        observation = _extract_http_failure_observation(causal_event.message)
        if observation is not None and "HTTP 401 Unauthorized" in observation:
            return observation + " The rejecting service and exact identity failure mode could not be determined from the supplied evidence."

    if mode == "quota_exhaustion":
        return "Observed storage capacity or quota exhaustion interrupted the operation."
    if mode == "permission_failure":
        return "Observed authorization or permission denial interrupted the operation."
    if mode == "network_timeout":
        return "Observed network timeout or unreachable transport errors interrupted the operation."
    if mode == "service_unavailable":
        return "Observed service outage or service-unavailable responses interrupted the operation."
    if mode == "clock_skew":
        return "Observed clock skew caused signed requests to be rejected."
    if mode == "partial_failure":
        return "Observed partial failure after one or more child tasks failed."
    if mode == "system_health_degradation":
        return "The supplied status snapshot shows degraded/inactive resources and unavailable services."
    if mode == "resource_access_failure":
        observed_cause = (getattr(causal_event, "observed_cause", None) or "").strip() if causal_event is not None else ""
        if causal_event is not None and "object info" in causal_event.message.lower():
            if observed_cause:
                return f"The service could not read object metadata because the underlying system returned {observed_cause}."
            return "The service could not read object metadata because the underlying system returned a low-level resource-access error."
        if observed_cause:
            return f"The service could not access a required resource because the underlying system returned {observed_cause}."
        return "The service could not access a required resource because the underlying system returned a low-level resource-access error."
    if mode == "corruption":
        return "Observed repository or blob corruption before the operation failed."
    if causal_event is not None:
        semantic_code = (_canonical_error_code(causal_event) or "").upper()
        cause_type = (getattr(causal_event, "cause_type", None) or "").upper()

        if semantic_code == "QUOTA_EXCEEDED" or cause_type == "STORAGE_CAPACITY":
            return "The selected evidence shows that a storage-capacity failure interrupted the operation."
        if semantic_code == "LOCK_OR_CONCURRENCY_FAILURE" or cause_type == "LOCK_OR_CONCURRENCY":
            return "The selected evidence shows that a lock-refresh failure interrupted the operation."
        if semantic_code in {"ACCESS_DENIED", "PERMISSION_DENIED", "UNAUTHORIZED"}:
            return "The selected evidence shows an access-denied response, but the rejecting service and exact identity failure mode could not be determined from the supplied evidence."
        if semantic_code == "RESOURCE_ACCESS_FAILURE" or cause_type == "RESOURCE_ACCESS":
            observed_cause = (getattr(causal_event, "observed_cause", None) or "").strip()
            if "object info" in causal_event.message.lower() and observed_cause:
                return f"The service could not read object metadata because the underlying system returned {observed_cause}."
            if observed_cause:
                return f"The service could not access a required resource because the underlying system returned {observed_cause}."
        if semantic_code == "SYSTEM_HEALTH_DEGRADATION" or cause_type == "SYSTEM_DEGRADATION":
            return "The supplied status snapshot shows degraded/inactive resources and unavailable services."
    if mode == "generic_fatal":
        if causal_event is not None:
            concise_message = _concise_observed_message(causal_event.message)
            if concise_message:
                return f"The selected evidence shows: {concise_message}"
        return "The selected evidence shows a fatal failure, but the available evidence does not support a more specific causal attribution."
    return "The timeline shows repeated error activity preceding the final job outcome, but the dominant failure class is inconclusive."


def _family_consistency_ratio(
   best_matches: list[TimelineEntryContext],
   all_collections: dict[str, list[TimelineEntryContext]],
) -> float:
   total_classified = sum(len(matches) for matches in all_collections.values())
   if total_classified == 0:
       return 0.0
   return len(best_matches) / total_classified


def _supports_corroboration(entry: TimelineEntryContext) -> bool:
   if entry.level in {"WARN", "ERROR"}:
       return True
   if _has_failure_prefix(entry):
       return True
   if entry.error_code in {"PLAINTEXT_FATAL", "RAW_UNPARSED"} and _raw_line_text(entry).lower().startswith(("fatal:", "error:", "panic:")):
       return True
   return False


def _corroboration_identity_key(entry: TimelineEntryContext) -> tuple[str, str]:
    non_identity_tokens = {
        "error",
        "fatal",
        "warn",
        "warning",
        "info",
        "debug",
        "trace",
        "unknown",
        "plaintext",
        "raw",
        "parser",
        "generic",
    }

    component = _component_key(entry.component)
    if component and component not in non_identity_tokens:
        return ("component", component)

    source_file = _evidence_source_file(entry)
    if source_file:
        return ("source", source_file.lower().strip())

    return ("unknown", "unknown")


def _temporal_chain_strength(
   matched_entries: list[TimelineEntryContext],
   terminal_entries: list[TimelineEntryContext],
) -> tuple[float, list[str]]:
   explanation: list[str] = []
   timestamp_sources = {(_evidence_timestamp_source(entry) or "").lower().strip() for entry in matched_entries + terminal_entries}
   observed_timestamps_available = bool(timestamp_sources.intersection({"available", "parsed"}))
   warnings = [entry for entry in matched_entries if entry.level == "WARN"]
   errors = [entry for entry in matched_entries if entry.level == "ERROR" and not _is_terminal_outcome_event(entry)]
   if not observed_timestamps_available:
       explanation.append("Event sequence is preserved from document order, but observed timestamps are unavailable, so temporal causality cannot be verified.")
       if warnings and errors and terminal_entries:
           return 0.2, explanation
       if errors and terminal_entries and errors[-1].sequence < terminal_entries[-1].sequence:
           return 0.14, explanation
       if errors:
           return 0.07, explanation
       return 0.02, explanation
   if warnings and errors and terminal_entries:
       if warnings[0].sequence < errors[-1].sequence < terminal_entries[-1].sequence:
           explanation.append("✓ Clear temporal chain: warning → causal error → terminal outcome")
           return 0.2, explanation
   if errors and terminal_entries and errors[-1].sequence < terminal_entries[-1].sequence:
       explanation.append("✓ Causal error appears before the terminal outcome")
       return 0.14, explanation
   if errors:
       explanation.append("• Causal error is present, but the supporting event sequence is limited")
       return 0.07, explanation
   explanation.append("• Temporal ordering is weak or unclear")
   return 0.02, explanation


def _compute_confidence(
   *,
   mode: str,
   matched_entries: list[TimelineEntryContext],
   causal_event: Optional[TimelineEntryContext],
   all_collections: dict[str, list[TimelineEntryContext]],
   rule_scores: dict[str, int],
   terminal_entries: list[TimelineEntryContext],
) -> tuple[float, dict[str, float], list[str]]:
   breakdown = {
       "causal_signal": 0.0,
       "corroboration": 0.0,
       "temporal_coherence": 0.0,
       "consistency": 0.0,
       "ambiguity_penalty": 0.0,
   }
   explanation: list[str] = []

   if causal_event is not None and not _is_terminal_outcome_event(causal_event):
       canonical_error_code = _canonical_error_code(causal_event) or causal_event.error_code
       if canonical_error_code in EXACT_SIGNAL_WEIGHTS:
           breakdown["causal_signal"] = EXACT_SIGNAL_WEIGHTS[canonical_error_code]
           explanation.append("✓ Exact causal error identified")
       elif _structured_message_type(causal_event) in {"error", "exit_error"}:
           breakdown["causal_signal"] = 0.16
           explanation.append("✓ Structured causal error identified")
       elif causal_event.level == "ERROR":
           breakdown["causal_signal"] = 0.16
           explanation.append("✓ Non-terminal causal error identified")
   elif matched_entries and any((_canonical_error_code(entry) or entry.error_code) in EXACT_SIGNAL_WEIGHTS for entry in matched_entries):
       breakdown["causal_signal"] = 0.12
       explanation.append("• Failure family identified, but the precise causal event is less direct")
   elif terminal_entries:
       breakdown["causal_signal"] = 0.04
       explanation.append("• Only terminal outcome evidence is available")

   corroborating_entries = [entry for entry in matched_entries if _supports_corroboration(entry)]
   unique_components = {_corroboration_identity_key(entry) for entry in corroborating_entries}
   corroborating_components = min(len(unique_components), 3)
   if corroborating_components >= 3:
       breakdown["corroboration"] = 0.22
       explanation.append("✓ Evidence is corroborated across multiple components")
   elif corroborating_components == 2:
       breakdown["corroboration"] = 0.14
       explanation.append("✓ Evidence is corroborated by more than one component")
   elif corroborating_components == 1 and corroborating_entries:
       breakdown["corroboration"] = 0.05
       explanation.append("• Most evidence comes from a single component")

   breakdown["temporal_coherence"], temporal_notes = _temporal_chain_strength(
       matched_entries,
       terminal_entries,
   )
   explanation.extend(temporal_notes)

   consistency_ratio = _family_consistency_ratio(matched_entries, all_collections)
   if consistency_ratio >= 0.8 and matched_entries:
       breakdown["consistency"] = 0.18
       explanation.append("✓ Evidence consistently points to one failure family")
   elif consistency_ratio >= 0.6:
       breakdown["consistency"] = 0.12
       explanation.append("✓ Most evidence points to one failure family")
   elif consistency_ratio >= 0.4:
       breakdown["consistency"] = 0.06
       explanation.append("• Evidence is somewhat mixed across failure families")

   competing_scores = sorted(
       (score for family, score in rule_scores.items() if family != mode and score > 0),
       reverse=True,
   )
   top_competing = competing_scores[0] if competing_scores else 0
   best_score = rule_scores.get(mode, 0)
   if top_competing > 0:
       if best_score - top_competing <= 1:
           breakdown["ambiguity_penalty"] = -0.18
           explanation.append("• Competing hypotheses are nearly as plausible as the chosen root cause")
       elif best_score - top_competing <= 3:
           breakdown["ambiguity_penalty"] = -0.1
           explanation.append("• Some competing evidence reduces certainty")
       else:
           explanation.append("✓ No strong competing hypothesis was detected")
   else:
       explanation.append("✓ No competing hypotheses detected")

   confidence = sum(breakdown.values())
   confidence = max(0.05, min(0.95, confidence))
   rounded_breakdown = {key: round(value, 2) for key, value in breakdown.items()}
   return round(confidence, 2), rounded_breakdown, explanation


def select_investigation_evidence(context: InvestigationContext) -> InvestigationEvidence:
   entries = context.recovery_timeline
   mode, matched_entries, all_collections, rule_scores = _classify_failure_mode(context)
   terminal_sequence = _find_terminal_sequence(entries)
   terminal_entries = [entry for entry in entries if _is_terminal_outcome_event(entry)]
   if not terminal_entries and terminal_sequence is not None:
       terminal_entries = [entry for entry in entries if entry.sequence == terminal_sequence]

   candidate_pool = [entry for entry in entries if not _is_progress_noise(entry)]
   causal_candidates = [entry for entry in candidate_pool if _is_direct_causal_signal(entry)]
   if causal_candidates:
       primary_entry = max(
           causal_candidates,
           key=lambda entry: (_semantic_causal_strength(entry), -entry.sequence),
       )
   elif candidate_pool:
       primary_entry = max(
           candidate_pool,
           key=lambda entry: (_primary_candidate_score(entry), -entry.sequence),
       )
   else:
       primary_entry = _latest_causal_event(matched_entries, terminal_sequence)
   if primary_entry is None and terminal_entries:
       primary_entry = terminal_entries[-1]

   selected_entries: list[tuple[TimelineEntryContext, str]] = []
   seen_signatures: set[tuple[str, str, str, str]] = set()

   def add_entry(entry: TimelineEntryContext, role: str) -> None:
       signature = _evidence_signature(entry)
       if signature in seen_signatures or _is_progress_noise(entry):
           return
       seen_signatures.add(signature)
       selected_entries.append((entry, role))

   for entry in entries:
       if _is_progress_noise(entry):
           continue
       if primary_entry is not None and entry.sequence == primary_entry.sequence:
           add_entry(entry, "PRIMARY_CAUSE")
           continue
       if _is_outcome_signal(entry):
           add_entry(entry, "OUTCOME")
           continue
       if _is_explicit_propagation_signal(entry) and primary_entry is not None and entry.sequence > primary_entry.sequence:
           add_entry(entry, "PROPAGATION")
           continue
       if _is_direct_causal_signal(entry):
           add_entry(entry, "CONTRIBUTING")
           continue
       if entry.level == "INFO":
           add_entry(entry, "CONTEXT")
       elif entry.level in {"WARN", "ERROR"} or entry.error_code or _is_failure_related(entry):
           add_entry(entry, "CONTRIBUTING")
       else:
           add_entry(entry, "CONTEXT")

   causal_items: list[EvidenceItem] = []
   context_items: list[EvidenceItem] = []
   supporting_items: list[EvidenceItem] = []
   outcome_items: list[EvidenceItem] = []
   causal_chain: list[EvidenceItem] = []
   primary_item: Optional[EvidenceItem] = None

   for entry, role in sorted(selected_entries, key=lambda pair: pair[0].sequence):
       item = _evidence_item(entry, role=role, mode=mode)
       causal_chain.append(item)
       if role == "CONTEXT":
           context_items.append(item)
           supporting_items.append(item)
       elif role == "PRIMARY_CAUSE":
           causal_items.append(item)
           primary_item = item
       elif role == "OUTCOME":
           outcome_items.append(item)
           supporting_items.append(item)
       else:
           causal_items.append(item)
           supporting_items.append(item)

   limitations: list[str] = []
   if primary_item is None:
       limitations.append("No defensible primary causal event could be isolated from the available evidence.")
   if primary_item is not None and any(
       item.role == "CONTRIBUTING" and item.sequence is not None and primary_item.sequence is not None and item.sequence < primary_item.sequence
       for item in causal_chain
   ):
       limitations.append("The logs do not prove that the earlier file-access conflict caused the lock-refresh failure.")
   if any(item.timestamp_source not in {"available", "parsed"} for item in causal_chain):
       limitations.append("Observed timestamps are missing for part of the evidence; event order is preserved from ingestion sequence.")
   source_documents = {item.source_file for item in causal_chain if item.source_file}
   if len(source_documents) == 1 and causal_chain:
       limitations.append("The investigation is based on a single source document.")
   elif len({item.component for item in causal_chain}) == 1 and causal_chain:
       limitations.append("Evidence is concentrated in a single component.")
   if primary_item is not None:
       primary_text = primary_item.message.lower()
       primary_code = (primary_item.semantic_error_code or "").upper()
       if "401" in primary_text or "unauthorized" in primary_text or primary_code in {"ACCESS_DENIED", "PERMISSION_DENIED", "UNAUTHORIZED"}:
           limitations.append("The evidence contains an explicit HTTP 401 Unauthorized signal.")
           limitations.append("The rejecting service could not be determined from the supplied evidence.")
           limitations.append("No corroborating gateway, authentication, or target-service logs were supplied.")
           limitations.append("The evidence does not establish whether the failure was authentication, authorization policy, token expiry, gateway enforcement, or another identity issue.")
       if primary_code == "RESOURCE_ACCESS_FAILURE":
           limitations.append("The supplied evidence does not establish whether the underlying problem was filesystem state, device failure, process descriptor state, or an operating-system issue.")
       if primary_code == "SYSTEM_HEALTH_DEGRADATION":
           limitations.append("The snapshot does not identify which host, device, daemon, or network failure caused the degradation.")

   competing_scores = sorted(
       (score for family, score in rule_scores.items() if family != mode and score > 0),
       reverse=True,
   )
   if competing_scores and rule_scores.get(mode, 0) - competing_scores[0] <= 1:
       limitations.append("A competing failure family is nearly as plausible as the selected cause.")

   confidence, confidence_breakdown, confidence_explanation = _compute_confidence(
       mode=mode,
       matched_entries=matched_entries,
       causal_event=primary_entry,
       all_collections=all_collections,
       rule_scores=rule_scores,
       terminal_entries=terminal_entries,
   )

   confidence_inputs = {
       "causal_count": float(len(causal_items)),
       "context_count": float(len(context_items)),
       "supporting_count": float(len(supporting_items)),
       "outcome_count": float(len(outcome_items)),
       "limitations_count": float(len(limitations)),
       "matched_count": float(len(matched_entries)),
   }

   return InvestigationEvidence(
       failure_mode=mode,
       root_cause=_root_cause_text(mode, primary_entry),
       context_events=context_items,
       causal_events=causal_items,
       outcome_events=outcome_items,
       supporting_events=supporting_items,
       limitations=limitations,
       primary_causal_event=primary_item,
       causal_chain=causal_chain,
       confidence_inputs=confidence_inputs,
       confidence=confidence,
       confidence_breakdown=confidence_breakdown,
       confidence_explanation=confidence_explanation,
   )


def _build_report_supporting_evidence(evidence: InvestigationEvidence) -> list[SupportingEvidenceItem]:
    supporting_items: list[SupportingEvidenceItem] = []
    seen_event_ids: set[str] = set()
    for item in evidence.supporting_events:
        if evidence.primary_causal_event is not None and item.event_id == evidence.primary_causal_event.event_id:
            continue
        if item.event_id in seen_event_ids:
            continue
        seen_event_ids.add(item.event_id)
        supporting_items.append(
            SupportingEvidenceItem(
                event_id=item.event_id,
                role=item.role,
                display_summary=item.display_summary or item.summary or item.message,
                why_it_matters=item.display_detail or item.summary or item.message,
                source_document_id=item.source_file,
            )
        )
    if not supporting_items and evidence.causal_chain:
        for item in evidence.causal_chain:
            if evidence.primary_causal_event is not None and item.event_id == evidence.primary_causal_event.event_id:
                continue
            if item.event_id in seen_event_ids:
                continue
            seen_event_ids.add(item.event_id)
            supporting_items.append(
                SupportingEvidenceItem(
                    event_id=item.event_id,
                    role=item.role,
                    display_summary=item.display_summary or item.summary or item.message,
                    why_it_matters=item.display_detail or item.summary or item.message,
                    source_document_id=item.source_file,
                )
            )
    return supporting_items


def _build_report_timeline_items(evidence: InvestigationEvidence) -> list[TimelineReportItem]:
    items: list[TimelineReportItem] = []
    for finding_index, group in enumerate(group_evidence_for_timeline(evidence), start=1):
        representative = group.items[0] if group.items else None
        detail_items = [
            TimelineFindingDetail(
                event_id=item.event_id,
                role=item.role,
                timestamp=item.timestamp,
                timestamp_source=item.timestamp_source,
                time=_format_report_timestamp(item.timestamp, item.timestamp_source),
                display_summary=item.display_summary or item.summary or item.message,
                display_detail=item.display_detail or item.summary or item.message,
                source=item.component,
                component_id=item.component,
                component_name=item.component,
                source_document_id=item.source_file,
            )
            for item in group.items
        ]
        display_detail = group.summary
        if len(group.items) > 1:
            display_detail = f"{group.summary} ({len(group.items)} supporting observations.)"
        items.append(
            TimelineReportItem(
                event_id=representative.event_id if representative is not None else f"finding-{finding_index}",
                role=representative.role if representative is not None else "CONTEXT",
                title=group.title,
                finding_type=group.finding_type,
                stage=group.stage,
                timestamp=representative.timestamp if representative is not None else None,
                timestamp_source=representative.timestamp_source if representative is not None else None,
                time=_format_report_timestamp(representative.timestamp, representative.timestamp_source) if representative is not None else "Timestamp unavailable",
                display_summary=group.title,
                display_detail=display_detail,
                source=representative.component if representative is not None else "investigation",
                component_id=representative.component if representative is not None else "investigation",
                component_name=representative.component if representative is not None else "Investigation",
                source_document_id=representative.source_file if representative is not None else None,
                supporting_evidence_count=len(group.items) if group.items else len(group.underlying_evidence_ids),
                underlying_evidence_ids=list(group.underlying_evidence_ids),
                detail_items=detail_items,
            )
        )
    return items


def _derive_customer_impact(evidence: InvestigationEvidence, scenario: Scenario) -> str:
    if evidence.outcome_events:
        outcome = evidence.outcome_events[-1]
        if outcome.role == "OUTCOME" and outcome.display_detail:
            return outcome.display_detail
    if evidence.primary_causal_event is not None:
        observation = _extract_http_failure_observation(evidence.primary_causal_event.message)
        if observation is not None:
            if "PUT request" in observation:
                return "The downstream PUT request did not complete successfully. Broader job or customer impact could not be determined from the supplied evidence."
            return "The observed downstream request did not complete successfully. Broader job or customer impact could not be determined from the supplied evidence."
    return "Broader job or customer impact could not be determined from the supplied evidence."


def _primary_causal_label(evidence: InvestigationEvidence) -> str:
    primary = evidence.primary_causal_event
    if primary is None:
        return "No defensible primary causal event"

    if (primary.semantic_error_code or "").upper() == "RESOURCE_ACCESS_FAILURE":
        if "object info" in primary.message.lower():
            return "Object metadata access failure"
        return "Resource access failure"

    if (primary.semantic_error_code or "").upper() == "SYSTEM_HEALTH_DEGRADATION":
        return "Critical system health degradation"

    semantic_bits: list[str] = []
    for value in (primary.cause_type, primary.event_type, primary.semantic_error_code, primary.error_code):
        if isinstance(value, str) and value.strip():
            cleaned = _humanize_identifier(value)
            if cleaned and cleaned.upper() not in {"UNKNOWN", "NO CODE", "UNKNOWN CODE", "RAW UNPARSED"}:
                semantic_bits.append(cleaned)

    if semantic_bits:
        return semantic_bits[0]
    if primary.display_summary:
        return _generic_sentence(primary.display_summary)
    return _generic_sentence(primary.message) or "Primary causal signal"


def _observed_failure_text(evidence: InvestigationEvidence) -> str:
    primary = evidence.primary_causal_event
    if primary is None:
        return "No direct causal event was isolated from the supplied evidence."

    if (primary.semantic_error_code or "").upper() == "RESOURCE_ACCESS_FAILURE":
        if primary.display_summary:
            return _generic_sentence(primary.display_summary)
        if "object info" in primary.message.lower():
            return "Object metadata access failed."
        return "Resource access failed."

    if (primary.semantic_error_code or "").upper() == "SYSTEM_HEALTH_DEGRADATION":
        return "Critical system health degradation detected."

    if primary.display_summary:
        return _generic_sentence(primary.display_summary)

    concise = _concise_observed_message(primary.message)
    if concise:
        return concise
    return _generic_sentence(primary.message) or "Failure signal observed in the primary causal event."


def _inferred_explanation_text(evidence: InvestigationEvidence) -> str:
    root = (evidence.root_cause or "").strip()
    if not root:
        return "No evidence-grounded explanation could be inferred from the supplied timeline."
    return root


def _evidence_gap_lines(evidence: InvestigationEvidence) -> list[str]:
    gaps: list[str] = []
    for limitation in evidence.limitations:
        normalized = limitation.removeprefix("•").removeprefix("✓").strip()
        if not normalized:
            continue
        lower = normalized.lower()
        if any(
            token in lower
            for token in (
                "could not",
                "does not",
                "missing",
                "single source",
                "unavailable",
                "not establish",
                "not prove",
                "competing",
                "inconclusive",
                "could not be isolated",
            )
        ):
            if normalized not in gaps:
                gaps.append(normalized)
    return gaps


def _action_category(action: str) -> str:
    text = action.lower()
    if any(token in text for token in ("retry", "restore", "increase", "repair", "stabilize", "isolate", "rollback", "roll back", "prune", "clear")):
        return "Immediate mitigation"
    if any(token in text for token in ("inspect", "search", "verify", "confirm", "identify", "logs", "trace", "collect", "correlate")):
        return "Immediate investigation"
    return "Preventive improvement"


def _action_rationale(action: str, evidence: InvestigationEvidence, evidence_gaps: list[str]) -> str:
    category = _action_category(action)
    if category == "Immediate investigation":
        if evidence_gaps:
            return "Addresses unresolved evidence gaps before attributing a narrower cause."
        return "Validates the observed failure with directly related component evidence."
    if category == "Immediate mitigation":
        if evidence.primary_causal_event is not None and evidence.primary_causal_event.display_summary:
            return f"Targets the observed primary failure signal: {evidence.primary_causal_event.display_summary}"
        return "Attempts to restore service after the observed failure condition."
    return "Reduces recurrence risk by hardening controls around the observed failure mode."


def _recommended_actions_view(actions: list[str], evidence: InvestigationEvidence, evidence_gaps: list[str]) -> list[RecommendedActionView]:
    output: list[RecommendedActionView] = []
    for action in _dedupe_actions(actions):
        output.append(
            RecommendedActionView(
                category=_action_category(action),
                action=action,
                rationale=_action_rationale(action, evidence, evidence_gaps),
            )
        )
    return output


def _confidence_dimension_level(value: float, *, penalty: bool = False) -> str:
    if penalty:
        if value <= -0.15:
            return "High"
        if value < 0:
            return "Medium"
        return "Low"
    if value >= 0.16:
        return "High"
    if value >= 0.08:
        return "Medium"
    if value > 0:
        return "Low"
    return "Unknown"


def _confidence_dimension_rationale(key: str, value: float, evidence: InvestigationEvidence) -> str:
    if key == "causal_signal":
        if evidence.primary_causal_event is not None:
            return "Primary causal event was selected from non-terminal evidence."
        return "No non-terminal primary causal event was isolated."
    if key == "corroboration":
        identities = {
            _corroboration_identity_key(item)
            for item in evidence.causal_chain
            if item.role in {"PRIMARY_CAUSE", "CONTRIBUTING", "OUTCOME"}
        }
        if not identities or identities == {("unknown", "unknown")}:
            return "Corroboration is limited to one source document and no independently identifiable second component was established."
        if len(identities) == 1:
            return "Corroboration comes from one independently identifiable component or source document."
        return f"Corroboration observed across {len(identities)} independently identifiable components or sources."
    if key == "temporal_coherence":
        timestamps_available = any(
            (_evidence_timestamp_source(item) or "").lower().strip() in {"available", "parsed"}
            for item in evidence.causal_chain
        )
        if timestamps_available:
            return "Ordering of warning/error signals relative to terminal outcome supports the inferred chain."
        return "Event sequence is preserved from document order, but observed timestamps are unavailable, so temporal causality cannot be verified."
    if key == "consistency":
        return "Failure-family consistency reflects how strongly evidence aligns to one mode."
    if key == "ambiguity_penalty":
        if value < 0:
            return "Competing hypotheses reduce certainty for the selected explanation."
        return "No strong competing hypothesis reduced certainty."
    return "Confidence contribution derived from selected evidence."


def _confidence_dimensions_view(evidence: InvestigationEvidence) -> list[ConfidenceDimensionView]:
    labels = {
        "causal_signal": "Causal signal",
        "corroboration": "Cross-component corroboration",
        "temporal_coherence": "Temporal coherence",
        "consistency": "Failure-family consistency",
        "ambiguity_penalty": "Ambiguity penalty",
    }
    dimensions: list[ConfidenceDimensionView] = []
    for key in ("causal_signal", "corroboration", "temporal_coherence", "consistency", "ambiguity_penalty"):
        value = float(evidence.confidence_breakdown.get(key, 0.0))
        dimensions.append(
            ConfidenceDimensionView(
                dimension=labels[key],
                level=_confidence_dimension_level(value, penalty=(key == "ambiguity_penalty")),
                rationale=_confidence_dimension_rationale(key, value, evidence),
            )
        )
    return dimensions


def _build_report_actions(actions: list[str]) -> tuple[list[str], list[str]]:
    immediate: list[str] = []
    preventive: list[str] = []
    for action in actions:
        text = action.lower()
        if any(token in text for token in ("retry", "restore", "increase", "stabilize", "repair", "re-run", "isolate", "roll back", "inspect", "search", "verify", "confirm")):
            immediate.append(action)
        else:
            preventive.append(action)
    if not immediate and actions:
        immediate = actions[:1]
        preventive = actions[1:]
    return immediate, preventive


def _dedupe_actions(actions: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for action in actions:
        key = action.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(action)
    return deduped


def _report_actions_from_evidence(
    evidence: InvestigationEvidence,
    context: InvestigationContext,
) -> list[str]:
    primary = evidence.primary_causal_event
    primary_text = (primary.message if primary is not None else "").lower()
    semantic_code = (primary.semantic_error_code if primary is not None and primary.semantic_error_code else "").upper()
    cause_type = (primary.cause_type if primary is not None and primary.cause_type else "").upper()

    if "401" in primary_text or "unauthorized" in primary_text:
        return [
            "Inspect gateway, identity-provider, and target-service logs for the rejected request.",
            "Search by endpoint, caller identity, request ID, or the surrounding time window to identify the rejecting service.",
            "Verify the credentials or token used by the request and confirm the expected scope or policy.",
        ]

    if evidence.failure_mode and evidence.failure_mode != "generic_fatal":
        return _evidence_grounded_actions(evidence.failure_mode, context.recovery_timeline)

    if semantic_code == "RESOURCE_ACCESS_FAILURE" or cause_type == "RESOURCE_ACCESS":
        return _evidence_grounded_actions("resource_access_failure", context.recovery_timeline)

    if semantic_code == "SYSTEM_HEALTH_DEGRADATION" or cause_type == "SYSTEM_DEGRADATION":
        return _evidence_grounded_actions("system_health_degradation", context.recovery_timeline)

    if semantic_code == "QUOTA_EXCEEDED" or cause_type == "STORAGE_CAPACITY" or "no space left" in primary_text:
        return [
            "Increase destination storage capacity or prune old backup artifacts before the next run.",
            "Confirm the failing write or upload path has sufficient free space before retrying.",
            "Add earlier capacity alerts so non-transient quota failures are detected before upload begins.",
        ]

    if semantic_code == "LOCK_OR_CONCURRENCY_FAILURE" or cause_type == "LOCK_OR_CONCURRENCY" or "refresh lock" in primary_text:
        return [
            "Inspect repository-lock or lock-manager logs around the failed refresh window.",
            "Verify whether another writer or a stale lock prevented the operation from completing.",
            "Retry only after confirming the conflicting lock holder or stale lock condition has been cleared.",
        ]

    return _evidence_grounded_actions(evidence.failure_mode or "generic_fatal", context.recovery_timeline)


def build_investigation_report_view(
    evidence: InvestigationEvidence,
    summary: InvestigationSummary,
    scenario: Scenario,
    timeline: Any,
) -> InvestigationReportView:
    evidence_gaps = _evidence_gap_lines(evidence)
    recommended_actions = _recommended_actions_view(summary.next_actions, evidence, evidence_gaps)
    immediate_actions = [item.action for item in recommended_actions if item.category in {"Immediate investigation", "Immediate mitigation"}]
    preventive_actions = [item.action for item in recommended_actions if item.category == "Preventive improvement"]
    if not immediate_actions and summary.next_actions:
        immediate_actions, preventive_actions = _build_report_actions(summary.next_actions)
    confidence_level = "High" if evidence.confidence >= 0.75 else "Medium" if evidence.confidence >= 0.4 else "Low"
    return InvestigationReportView(
        status="Completed",
        incident_summary=_build_incident_summary_view(scenario, timeline),
        root_cause=evidence.root_cause,
        observed_failure=_observed_failure_text(evidence),
        inferred_explanation=_inferred_explanation_text(evidence),
        evidence_gaps=evidence_gaps,
        primary_causal_label=_primary_causal_label(evidence),
        primary_causal_event=evidence.primary_causal_event,
        outcome_summary=evidence.outcome_events[-1].display_summary if evidence.outcome_events else None,
        customer_impact=_derive_customer_impact(evidence, scenario),
        timeline_items=_build_report_timeline_items(evidence),
        supporting_evidence=_build_report_supporting_evidence(evidence),
        evidence_limitations=evidence.limitations,
        immediate_actions=immediate_actions,
        preventive_actions=preventive_actions,
        confidence=evidence.confidence,
        confidence_level=confidence_level,
        confidence_explanation=_report_confidence_explanation(evidence),
        confidence_dimensions=_confidence_dimensions_view(evidence),
        recommended_actions=recommended_actions,
    )


def _generate_fallback_summary(
    context: InvestigationContext,
    evidence: Optional[InvestigationEvidence] = None,
) -> InvestigationSummary:
    """Generate a deterministic, evidence-based investigation summary from timeline context only."""
    evidence = evidence or select_investigation_evidence(context)
    supporting_text = [evidence.root_cause]
    for item in evidence.causal_chain[-3:]:
        detail = item.message
        if item.summary and item.summary != item.message:
            supporting_text.append(f"{item.summary} — {detail}")
        else:
            supporting_text.append(detail)

    if evidence.primary_causal_event is not None:
        primary_detail = evidence.primary_causal_event.message
        if evidence.primary_causal_event.summary and evidence.primary_causal_event.summary != primary_detail:
            supporting_text.insert(
                0,
                f"Causal candidate: {evidence.primary_causal_event.summary} — {primary_detail}",
            )
        else:
            supporting_text.insert(0, f"Causal candidate: {primary_detail}")

    if evidence.outcome_events:
        supporting_text.append(
            f"Outcome evidence: {evidence.outcome_events[-1].message}"
        )

    if not supporting_text and evidence.causal_chain:
        supporting_text = [
            evidence.causal_chain[-1].message,
            "No stronger classified signal was found before the terminal outcome.",
        ]

    return InvestigationSummary(
        likely_root_cause=evidence.root_cause,
        supporting_evidence=supporting_text[:5],
        next_actions=_report_actions_from_evidence(evidence, context),
        confidence=evidence.confidence,
        confidence_breakdown=evidence.confidence_breakdown,
        confidence_explanation=evidence.confidence_explanation,
        evidence=evidence,
    )


def generate_simulated_investigation_summary(
    context: InvestigationContext,
    evidence: Optional[InvestigationEvidence] = None,
) -> InvestigationSummary:
    selected_evidence = evidence or select_investigation_evidence(context)
    return _generate_fallback_summary(context, selected_evidence)


def generate_llm_investigation_summary(
    context: InvestigationContext,
    model: str = "gpt-4.1-mini",
    timeout_seconds: float = DEFAULT_OPENAI_TIMEOUT_SECONDS,
) -> InvestigationSummary:
    return _generate_investigation_summary(
        context,
        model=model,
        timeout_seconds=timeout_seconds,
    )


def generate_live_investigation_summary(
    context: InvestigationContext,
    model: str = "gpt-4.1-mini",
    timeout_seconds: float = DEFAULT_OPENAI_TIMEOUT_SECONDS,
) -> tuple[InvestigationSummary, bool, Optional[str]]:
    try:
        return (
            generate_llm_investigation_summary(
                context,
                model=model,
                timeout_seconds=timeout_seconds,
            ),
            False,
            None,
        )
    except (LLMRequestError, ValueError) as exc:
        return generate_simulated_investigation_summary(context), True, str(exc)


def generate_investigation_summary(
    context: InvestigationContext,
    model: str = "gpt-4.1-mini",
    timeout_seconds: float = DEFAULT_OPENAI_TIMEOUT_SECONDS,
) -> tuple[InvestigationSummary, bool]:
    try:
        summary = generate_llm_investigation_summary(
            context,
            model=model,
            timeout_seconds=timeout_seconds,
        )
        return summary, False
    except LLMRequestError:
        return generate_simulated_investigation_summary(context), True
