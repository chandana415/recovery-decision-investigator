from __future__ import annotations

import json
import os
import re
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

from recovery_workspace.models import (
    Event,
    EventContext,
    InvestigationContext,
    InvestigationSummary,
    TimelineEntryContext,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

client: Optional[OpenAI] = None
client_timeout_seconds: float | None = None
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
    )


def _format_timeline_entry_context(entry: Any) -> TimelineEntryContext:
    return TimelineEntryContext(
        sequence=entry.sequence,
        timestamp=entry.event.timestamp.isoformat(),
        component=entry.event.component,
        level=entry.event.level,
        error_code=entry.event.error_code,
        message=entry.event.message,
        gap_before_seconds=entry.gap_before_seconds,
    )


def _get_llm_client(timeout_seconds: float = DEFAULT_OPENAI_TIMEOUT_SECONDS) -> OpenAI:
    global client, client_timeout_seconds
    if client is not None:
        return client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise LLMRequestError(
            "OPENAI_API_KEY environment variable is required for AI summarization."
        )
    client = OpenAI(max_retries=0, timeout=timeout_seconds)
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
        ("quota", "capacity", "provisioned", "gb/", "used)", "disk full"),
        {"storage", "monitoring", "backup_service"},
    ),
    (
        "permission_failure",
        {"ACCESS_DENIED", "PERMISSION_DENIED", "UNAUTHORIZED"},
        ("forbidden", "lacks", "access denied", "permission denied", "principal"),
        {"storage", "auth", "backup_service"},
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
]
EXACT_SIGNAL_WEIGHTS = {
    "QUOTA_EXCEEDED": 0.3,
    "ACCESS_DENIED": 0.3,
    "ETIMEDOUT": 0.28,
    "ENETUNREACH": 0.28,
    "ECONNRESET": 0.28,
    "SERVICE_UNAVAILABLE": 0.28,
    "REQUEST_TIME_TOO_SKEWED": 0.3,
    "CLOCK_SKEW_DETECTED": 0.3,
    "PARTIAL_SUCCESS": 0.18,
}


def _entry_text(entry: TimelineEntryContext) -> str:
    return f"{entry.component} {entry.error_code or ''} {entry.message}".lower()


def _is_terminal_outcome_event(entry: TimelineEntryContext) -> bool:
    text = _entry_text(entry)
    if entry.error_code in TERMINAL_OUTCOME_CODES:
        return True
    return (
        entry.component == "backup_service"
        and ("marked failed" in text or "completed with" in text or "partial_success" in text)
    )


def _find_terminal_sequence(entries: list[TimelineEntryContext]) -> int | None:
    terminal_entries = [entry for entry in entries if _is_terminal_outcome_event(entry)]
    if not terminal_entries:
        return None
    return terminal_entries[-1].sequence


def _build_evidence_line(entry: TimelineEntryContext) -> str:
    code = entry.error_code or "NO_CODE"
    return (
        f"Event #{entry.sequence}: {entry.component} {entry.level} {code} at {entry.timestamp} — "
        f"{entry.message}"
    )


def _find_matching_entries(
    entries: list[TimelineEntryContext],
    *,
    error_codes: set[str],
    message_terms: tuple[str, ...],
    components: set[str] | None = None,
) -> list[TimelineEntryContext]:
    matched: list[TimelineEntryContext] = []
    for entry in entries:
        text = _entry_text(entry)
        code_match = entry.error_code in error_codes if entry.error_code else False
        message_match = any(term in text for term in message_terms)
        component_match = components is None or entry.component in components
        if component_match and (code_match or message_match):
            matched.append(entry)
    return matched


def _latest_causal_event(
    matches: list[TimelineEntryContext],
    terminal_sequence: int | None,
) -> TimelineEntryContext | None:
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
            error_codes=codes,
            message_terms=terms,
            components=components,
        )
    return collections


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
        if causal is not None and causal.error_code in codes:
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

    return best_mode, best_matches, collections, rule_scores


def _scenario_specific_actions(mode: str) -> list[str]:
    if mode == "quota_exhaustion":
        return [
            "Increase destination storage capacity or prune old backup artifacts before the next run.",
            "Trigger proactive quota alerts earlier so backup jobs are blocked before upload begins.",
            "Mark quota-exceeded failures as non-transient and skip retry loops until capacity is restored.",
        ]
    if mode == "permission_failure":
        return [
            "Restore read permissions for the backup service principal on the affected storage path.",
            "Audit the recent policy change and roll back least-privilege scope reductions that removed required actions.",
            "Add a preflight authorization check before restore jobs start.",
        ]
    if mode == "network_timeout":
        return [
            "Stabilize the failing network path (tunnel flap/packet loss) before rerunning the restore.",
            "Tune timeout and retry policies for intermittent transport instability.",
            "Add targeted alerts for ETIMEDOUT/ENETUNREACH/ECONNRESET spikes on backup traffic.",
        ]
    if mode == "service_unavailable":
        return [
            "Coordinate with the dependency owner to schedule maintenance windows outside backup execution.",
            "Keep backoff-retry behavior for transient 503 outages and alert when outage duration exceeds retry budget.",
            "Add dependency health prechecks before starting large upload phases.",
        ]
    if mode == "clock_skew":
        return [
            "Repair host time synchronization (NTP) on the backup nodes before the next signed request.",
            "Add clock-skew drift alerts with thresholds below the storage signature tolerance window.",
            "Force a time-sync validation precheck before backup upload signing.",
        ]
    if mode == "partial_failure":
        return [
            "Isolate failed child tasks and re-run only the failed volumes instead of the entire job.",
            "Review non-retryable network reset policy for hard connection resets on parallel task workers.",
            "Track per-task success/failure metrics so partial outcomes trigger explicit remediation workflows.",
        ]
    return [
        "Review the most recent non-terminal error event and its upstream dependencies.",
        "Add targeted alerts around repeated WARN/ERROR patterns preceding terminal job outcomes.",
        "Capture richer component diagnostics at failure boundaries to improve deterministic classification.",
    ]


def _root_cause_text(mode: str, causal_event: TimelineEntryContext | None) -> str:
    event_detail = ""
    if causal_event is not None:
        code = causal_event.error_code or "NO_CODE"
        event_detail = (
            f" Most likely causal event: {causal_event.component} {code} at {causal_event.timestamp}."
        )
    if mode == "quota_exhaustion":
        return (
            "Destination storage capacity was exhausted during upload, causing a non-transient quota failure."
            + event_detail
        )
    if mode == "permission_failure":
        return (
            "Restore access failed due to authorization scope/permission denial on storage reads."
            + event_detail
        )
    if mode == "network_timeout":
        return (
            "Restore failed because transport instability caused repeated network timeout/unreachable errors."
            + event_detail
        )
    if mode == "service_unavailable":
        return (
            "A dependency service outage (503/unavailable) interrupted processing; retries indicate transient dependency failure."
            + event_detail
        )
    if mode == "clock_skew":
        return (
            "Signed requests were rejected because producer and storage clocks were outside the allowed skew window."
            + event_detail
        )
    if mode == "partial_failure":
        return (
            "Parallel volume processing ended in partial failure: at least one child task failed while others succeeded."
            + event_detail
        )
    return "The timeline shows repeated error activity preceding the final job outcome, but the dominant failure class is inconclusive."


def _family_consistency_ratio(
   best_matches: list[TimelineEntryContext],
   all_collections: dict[str, list[TimelineEntryContext]],
) -> float:
   total_classified = sum(len(matches) for matches in all_collections.values())
   if total_classified == 0:
       return 0.0
   return len(best_matches) / total_classified


def _temporal_chain_strength(
   matched_entries: list[TimelineEntryContext],
   terminal_entries: list[TimelineEntryContext],
) -> tuple[float, list[str]]:
   explanation: list[str] = []
   warnings = [entry for entry in matched_entries if entry.level == "WARN"]
   errors = [entry for entry in matched_entries if entry.level == "ERROR" and not _is_terminal_outcome_event(entry)]
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
   causal_event: TimelineEntryContext | None,
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
       if causal_event.error_code in EXACT_SIGNAL_WEIGHTS:
           breakdown["causal_signal"] = EXACT_SIGNAL_WEIGHTS[causal_event.error_code]
           explanation.append("✓ Exact causal error identified")
       elif causal_event.level == "ERROR":
           breakdown["causal_signal"] = 0.16
           explanation.append("✓ Non-terminal causal error identified")
   elif matched_entries and any(entry.error_code in EXACT_SIGNAL_WEIGHTS for entry in matched_entries):
       breakdown["causal_signal"] = 0.12
       explanation.append("• Failure family identified, but the precise causal event is less direct")
   elif terminal_entries:
       breakdown["causal_signal"] = 0.04
       explanation.append("• Only terminal outcome evidence is available")

   unique_components = {entry.component for entry in matched_entries}
   corroborating_components = min(len(unique_components), 3)
   if corroborating_components >= 3:
       breakdown["corroboration"] = 0.22
       explanation.append("✓ Evidence is corroborated across multiple components")
   elif corroborating_components == 2:
       breakdown["corroboration"] = 0.14
       explanation.append("✓ Evidence is corroborated by more than one component")
   elif corroborating_components == 1 and matched_entries:
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


def _generate_fallback_summary(context: InvestigationContext) -> InvestigationSummary:
    """Generate a deterministic, evidence-based investigation summary from timeline context only."""
    entries = context.recovery_timeline
    mode, matched_entries, all_collections, rule_scores = _classify_failure_mode(context)
    terminal_sequence = _find_terminal_sequence(entries)
    causal_event = _latest_causal_event(matched_entries, terminal_sequence)
    terminal_entries = [entry for entry in entries if _is_terminal_outcome_event(entry)]

    evidence: list[str] = []
    for entry in matched_entries[-3:]:
        evidence.append(_build_evidence_line(entry))

    if causal_event is not None:
        evidence.insert(0, f"Causal candidate: {_build_evidence_line(causal_event)}")

    if terminal_entries:
        evidence.append(f"Outcome evidence: {_build_evidence_line(terminal_entries[-1])}")

    if not evidence and entries:
        evidence = [
            _build_evidence_line(entries[-1]),
            "No stronger classified signal was found before the terminal outcome.",
        ]

    confidence, confidence_breakdown, confidence_explanation = _compute_confidence(
        mode=mode,
        matched_entries=matched_entries,
        causal_event=causal_event,
        all_collections=all_collections,
        rule_scores=rule_scores,
        terminal_entries=terminal_entries,
    )

    return InvestigationSummary(
        likely_root_cause=_root_cause_text(mode, causal_event),
        supporting_evidence=evidence[:5],
        next_actions=_scenario_specific_actions(mode),
        confidence=confidence,
        confidence_breakdown=confidence_breakdown,
        confidence_explanation=confidence_explanation,
    )


def generate_simulated_investigation_summary(
    context: InvestigationContext,
) -> InvestigationSummary:
    return _generate_fallback_summary(context)


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
) -> tuple[InvestigationSummary, bool, str | None]:
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
