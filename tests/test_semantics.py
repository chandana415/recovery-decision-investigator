from datetime import datetime, timezone
from pathlib import Path

from recovery_workspace.events import normalize_events
from recovery_workspace.investigation import build_investigation_context, select_investigation_evidence
from recovery_workspace.models import Event, InvestigationContext, TimelineEntryContext
from recovery_workspace.parser import parse_scenario
from recovery_workspace.semantics import apply_event_semantics
from recovery_workspace.timeline import build_timeline


def _make_event(*, component="storage", level="ERROR", error_code=None, message="", structured_fields=None):
    return Event(
        timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        correlation_id="corr-1",
        component_id=component.replace("-", "_"),
        component=component,
        level=level,
        error_code=error_code,
        message=message,
        structured_fields=structured_fields or {},
    )


def test_plaintext_access_conflict_stays_nonterminal():
    event = _make_event(
        error_code="PLAINTEXT_FATAL",
        message="file access conflict detected",
        structured_fields={"raw_line": "file access conflict detected", "message_type": "error"},
    )
    apply_event_semantics(event)

    assert event.error_code == "PLAINTEXT_FATAL"
    assert event.semantic_error_code == "ACCESS_OR_RESOURCE_CONFLICT"
    assert event.event_type == "ACCESS_OR_RESOURCE_CONFLICT"
    assert event.terminal_state is None
    assert event.cause_type == "RESOURCE_OR_ACCESS_CONTENTION"


def test_plaintext_lock_failure_stays_nonterminal():
    event = _make_event(
        error_code="PLAINTEXT_FATAL",
        message="lock refresh failed",
        structured_fields={"raw_line": "lock refresh failed", "message_type": "error"},
    )
    apply_event_semantics(event)

    assert event.semantic_error_code == "LOCK_OR_CONCURRENCY_FAILURE"
    assert event.event_type == "LOCK_OR_CONCURRENCY_FAILURE"
    assert event.terminal_state is None
    assert event.cause_type == "LOCK_OR_CONCURRENCY"


def test_plaintext_cancelled_operation_is_recognized_distinctly():
    event = _make_event(
        error_code="PLAINTEXT_FATAL",
        message="snapshot save cancelled",
        structured_fields={"raw_line": "snapshot save cancelled", "message_type": "error"},
    )
    apply_event_semantics(event)

    assert event.semantic_error_code == "OPERATION_ABORTED_OR_CANCELLED"
    assert event.event_type == "OPERATION_ABORTED_OR_CANCELLED"
    assert event.terminal_state is None
    assert event.cause_type is None


def test_json_terminal_failure_derives_semantic_signal_independently():
    event = _make_event(
        error_code="JSON",
        message="quota exceeded",
        structured_fields={"message_type": "error", "details": "quota exceeded"},
    )
    apply_event_semantics(event)

    assert event.semantic_error_code == "QUOTA_EXCEEDED"
    assert event.cause_type == "STORAGE_CAPACITY"
    assert event.terminal_state == "failure"


def test_explicit_exit_success_becomes_successful_terminal_state():
    event = _make_event(
        component="backup-service",
        level="INFO",
        error_code="EXIT_SUCCESS",
        message="completed successfully",
        structured_fields={"exit_code": 0},
    )
    apply_event_semantics(event)

    assert event.semantic_error_code == "EXIT_SUCCESS"
    assert event.terminal_state == "success"
    assert event.event_type == "INFO"


def test_nonterminal_warning_remains_nonterminal():
    event = _make_event(level="WARN", error_code=None, message="checkpoint reached")
    apply_event_semantics(event)

    assert event.event_type == "WARNING"
    assert event.terminal_state is None
    assert event.semantic_error_code is None


def test_unknown_fatal_is_explicit_terminal_failure():
    event = _make_event(level="ERROR", error_code="NO_CODE", message="fatal: something unexpected")
    apply_event_semantics(event)

    assert event.semantic_error_code == "UNKNOWN_TERMINAL_FAILURE"
    assert event.event_type == "TERMINAL_FAILURE"
    assert event.terminal_state == "failure"
    assert event.cause_type is None


def test_unknown_plaintext_error_without_terminal_evidence_stays_nonterminal():
    event = _make_event(level="ERROR", error_code="NO_CODE", message="unexpected operational issue")
    apply_event_semantics(event)

    assert event.semantic_error_code == "UNKNOWN_OPERATIONAL_ERROR"
    assert event.event_type == "OPERATIONAL_ERROR"
    assert event.terminal_state is None
    assert event.cause_type is None


def test_structured_cause_participates_in_semantic_derivation_for_resource_access_failures():
    event = _make_event(
        error_code="JSON",
        message="Unable to fetch object info.",
        structured_fields={
            "message_type": "error",
            "cause": "read /data/fs.json: bad file descriptor",
            "stack": "stack trace omitted",
            "raw_line": "raw diagnostic blob omitted",
        },
    )
    apply_event_semantics(event)

    assert event.semantic_error_code == "RESOURCE_ACCESS_FAILURE"
    assert event.event_type == "RESOURCE_ACCESS_FAILURE"
    assert event.cause_type == "RESOURCE_ACCESS"
    assert event.observed_cause == "read /data/fs.json: bad file descriptor"


def test_unrelated_cause_field_does_not_change_classification_without_matching_content():
    event = _make_event(
        error_code="JSON",
        message="Unexpected operational issue",
        structured_fields={
            "message_type": "error",
            "cause": "operator requested manual follow-up",
        },
    )
    apply_event_semantics(event)

    assert event.semantic_error_code == "UNKNOWN_OPERATIONAL_ERROR"
    assert event.event_type == "OPERATIONAL_ERROR"
    assert event.cause_type is None
    assert event.observed_cause == "operator requested manual follow-up"


def test_exit_code_failure_and_success_are_separate_signals():
    success_event = _make_event(level="INFO", error_code="0", message="child completed", structured_fields={"exit_code": 0})
    failure_event = _make_event(level="ERROR", error_code="1", message="child failed", structured_fields={"exit_code": 1})

    apply_event_semantics(success_event)
    apply_event_semantics(failure_event)

    assert success_event.terminal_state == "success"
    assert success_event.semantic_error_code == "EXIT_SUCCESS"
    assert failure_event.terminal_state == "failure"
    assert failure_event.semantic_error_code == "EXIT_FAILURE"


def test_mixed_child_exits_become_partial():
    event = _make_event(level="ERROR", error_code="CHILD_FAILURE", message="child exits mixed", structured_fields={"child_exit_codes": [0, 1]})
    apply_event_semantics(event)

    assert event.terminal_state == "partial"
    assert event.semantic_error_code == "MIXED_CHILD_EXITS"


def test_kubernetes_assignment_remains_nonterminal():
    event = _make_event(level="INFO", error_code=None, message="Successfully assigned default/web to node-1")
    apply_event_semantics(event)

    assert event.terminal_state is None
    assert event.event_type == "INFO"


def test_cause_type_is_consumed_from_event_semantics_not_failure_mode():
    event = _make_event(error_code="QUOTA_EXCEEDED", message="quota exceeded", structured_fields={"message_type": "error"})
    apply_event_semantics(event)
    timeline_entry = TimelineEntryContext(
        sequence=1,
        timestamp="2026-01-01T12:00:00Z",
        component=event.component,
        level=event.level,
        error_code=event.error_code,
        message=event.message,
        gap_before_seconds=0.0,
        structured_fields=event.structured_fields,
        event_type=event.event_type,
        terminal_state=event.terminal_state,
        cause_type=event.cause_type,
        semantic_error_code=event.semantic_error_code,
    )
    context = InvestigationContext(
        customer="Test Customer",
        jobId="job-test",
        outcome="FAILED",
        expectedFailureSignature="ignored",
        recovery_timeline=[timeline_entry],
        warning_events=[],
        error_events=[],
    )
    evidence = select_investigation_evidence(context)

    assert evidence.primary_causal_event is not None
    assert evidence.primary_causal_event.cause_type == "STORAGE_CAPACITY"


def test_investigation_uses_precomputed_terminal_state_not_role():
    event = _make_event(error_code="JOB_FAILED", message="job failed", structured_fields={"message_type": "error"})
    apply_event_semantics(event)
    timeline_entry = TimelineEntryContext(
        sequence=1,
        timestamp="2026-01-01T12:00:00Z",
        component=event.component,
        level=event.level,
        error_code=event.error_code,
        message=event.message,
        gap_before_seconds=0.0,
        structured_fields=event.structured_fields,
        event_type=event.event_type,
        terminal_state=event.terminal_state,
        cause_type=event.cause_type,
        semantic_error_code=event.semantic_error_code,
    )
    context = InvestigationContext(
        customer="Test Customer",
        jobId="job-test",
        outcome="FAILED",
        expectedFailureSignature="ignored",
        recovery_timeline=[timeline_entry],
        warning_events=[],
        error_events=[],
    )
    evidence = select_investigation_evidence(context)

    assert evidence.causal_chain[0].terminal_state == "failure"
    assert evidence.causal_chain[0].role == "PRIMARY_CAUSE"


def test_normalize_events_applies_semantics_before_timeline_construction():
    scenario = parse_scenario(
        Path(__file__).parent.parent / "mock-data" / "scenarios" / "storage-quota-exceeded.json"
    )
    events = normalize_events(scenario.logs)
    timeline = build_timeline(events)
    context = build_investigation_context(
        customer=scenario.customer,
        job_id=scenario.job_id,
        outcome=scenario.expected_outcome,
        expected_failure_signature=scenario.expected_failure_signature,
        timeline=timeline,
        events=events,
    )

    storage_entry = next(entry for entry in context.recovery_timeline if entry.component == "storage" and entry.level == "ERROR")
    assert storage_entry.semantic_error_code == "QUOTA_EXCEEDED"
    assert storage_entry.cause_type == "STORAGE_CAPACITY"
    assert storage_entry.event_type == "TERMINAL_FAILURE"

    assert len(timeline.entries) == len(events)
