import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import httpx
import pytest
from openai import RateLimitError

from recovery_workspace.events import normalize_events
from recovery_workspace.app import build_uploaded_scenario
from recovery_workspace.investigation import (
    LLMRequestError,
    _reasoning_context_payload,
    build_investigation_context,
    build_investigation_report_view,
    generate_llm_investigation_summary,
    generate_investigation_summary,
    generate_simulated_investigation_summary,
    select_investigation_evidence,
)
from recovery_workspace.models import (
    EventContext,
    EvidenceItem,
    InvestigationContext,
    InvestigationEvidence,
    InvestigationSummary,
    Scenario,
    TimelineEntryContext,
)
from recovery_workspace.parser import parse_scenario
from recovery_workspace.timeline import build_timeline
from recovery_workspace.uploader import parse_uploaded_logs_with_source

SCENARIO_PATH = (
    Path(__file__).parent.parent / "mock-data" / "scenarios" / "storage-quota-exceeded.json"
)
SCENARIOS_DIR = Path(__file__).parent.parent / "mock-data" / "scenarios"


def _mock_response(response_text: str):
    return SimpleNamespace(output_text=response_text)


class DummyResponses:
    def __init__(self, response_text: str):
        self.response_text = response_text

    def create(self, **kwargs):
        return _mock_response(self.response_text)


class DummyClient:
    def __init__(self, response_text: str):
        self.responses = DummyResponses(response_text)


class RateLimitResponses:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        request = httpx.Request("POST", "https://api.openai.com/v1/responses")
        response = httpx.Response(
            429,
            request=request,
            json={"error": {"code": "insufficient_quota", "message": "quota exhausted"}},
        )
        raise RateLimitError("quota exhausted", response=response, body=response.json())


class RateLimitClient:
    def __init__(self):
        self.responses = RateLimitResponses()


def test_build_investigation_context_includes_timeline_and_event_summaries():
    scenario = parse_scenario(SCENARIO_PATH)
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

    assert isinstance(context, InvestigationContext)
    assert context.customer == scenario.customer
    assert context.job_id == scenario.job_id
    assert context.outcome == scenario.expected_outcome
    assert context.expected_failure_signature == scenario.expected_failure_signature
    assert len(context.recovery_timeline) == len(events)
    assert len(context.warning_events) == 1
    assert len(context.error_events) == 3
    assert context.recovery_timeline[0].sequence == 1
    assert context.recovery_timeline[-1].gap_before_seconds == pytest.approx(4.0)

    storage_entry = next(entry for entry in context.recovery_timeline if entry.component == "storage" and entry.level == "ERROR")
    assert storage_entry.semantic_error_code == "QUOTA_EXCEEDED"
    assert storage_entry.cause_type == "STORAGE_CAPACITY"
    assert storage_entry.event_type == "TERMINAL_FAILURE"


def test_build_investigation_report_view_assembles_single_source_ui_payload():
    primary_event = EvidenceItem(
        event_id="2",
        component="backup_service",
        original_message="lock refresh failed",
        message="lock refresh failed",
        role="PRIMARY_CAUSE",
        display_summary="The lock refresh failed",
        display_detail="The lock refresh failed during backup processing.",
    )
    supporting_event = EvidenceItem(
        event_id="1",
        component="storage",
        original_message="access conflict detected",
        message="access conflict detected",
        role="CONTRIBUTING",
        display_summary="An access conflict was observed",
        display_detail="A temporary access conflict appeared before the lock failure.",
    )
    evidence = InvestigationEvidence(
        root_cause="A lock refresh failure interrupted the backup.",
        supporting_events=[supporting_event],
        causal_chain=[primary_event, supporting_event],
        primary_causal_event=primary_event,
        confidence=0.8,
        confidence_explanation=["The causal chain is consistent."],
    )
    summary = InvestigationSummary(
        likely_root_cause="A lock refresh failure interrupted the backup.",
        supporting_evidence=["The causal chain is consistent."],
        next_actions=["Retry the backup after restoring the lock manager.", "Add recovery guards around lock refresh."],
        confidence=0.8,
        confidence_explanation=["The causal chain is consistent."],
    )
    scenario = Scenario(
        scenarioId="scn-1",
        name="Sample scenario",
        category="demo",
        tags=["demo"],
        description="Sample",
        customer="Test Customer",
        jobId="job-test",
        correlationId="corr-1",
        componentsInvolved=["backup_service", "storage"],
        expectedOutcome="FAILED",
        expectedFailureSignature="lock refresh failed",
        logs=[],
    )
    timestamp = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    timeline = SimpleNamespace(
        entries=[
            SimpleNamespace(
                event=SimpleNamespace(
                    timestamp=timestamp,
                    structured_fields={"timestamp_source": "parsed"},
                    message="lock refresh failed",
                )
            )
        ],
        duration_seconds=0.0,
    )

    report = build_investigation_report_view(evidence, summary, scenario, timeline)

    assert report.root_cause == evidence.root_cause
    assert report.incident_summary.operation == "Unknown"
    assert report.primary_causal_event is not None
    assert report.primary_causal_event.event_id == "2"
    assert report.supporting_evidence[0].display_summary == supporting_event.display_summary
    assert report.supporting_evidence[0].source_document_id is None
    assert report.immediate_actions[0].startswith("Retry")
    assert report.preventive_actions[0].startswith("Add")
    assert report.confidence == pytest.approx(0.8)
    assert report.confidence_level == "High"
    assert report.observed_failure
    assert report.inferred_explanation == evidence.root_cause
    assert report.primary_causal_label
    assert len(report.confidence_dimensions) == 5
    assert {item.category for item in report.recommended_actions} == {
        "Immediate mitigation",
        "Preventive improvement",
    }


def _build_uploaded_report(content: str, filename: str):
    logs, error = parse_uploaded_logs_with_source(content, filename)
    assert error == ""
    scenario = build_uploaded_scenario([filename], logs)
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
    evidence = select_investigation_evidence(context)
    summary = generate_simulated_investigation_summary(context, evidence)
    report = build_investigation_report_view(evidence, summary, scenario, timeline)
    return scenario, timeline, evidence, summary, report


def test_http_401_ocr_report_view_is_conservative_and_traceable():
    content = '''Error : downstream PUT request failed
level : ERROR
logger : gateway client
message : HTTP 401 Unauthorized during PUT /upload
thread : main
throwable : request execution failed
cause : unauthorized response
stack : at client upload run (UploadClient java : 42)
at client upload retry (UploadClient java : 88)
suppressed : request context unavailable'''

    _scenario, _timeline, evidence, summary, report = _build_uploaded_report(content, "ocr401.log")

    assert report.incident_summary.operation == "Unknown"
    assert report.incident_summary.target == "Not provided"
    assert report.incident_summary.environment == "Not provided"
    assert report.incident_summary.started == "Timestamp unavailable"
    assert report.root_cause == (
        "A downstream PUT request received HTTP 401 Unauthorized. "
        "The rejecting service and exact identity failure mode could not be determined from the supplied evidence."
    )
    assert "stack trace" not in report.root_cause.lower()
    assert "thread :" not in report.root_cause.lower()
    assert report.customer_impact == (
        "The downstream PUT request did not complete successfully. Broader job or customer impact could not be determined from the supplied evidence."
    )
    assert report.timeline_items[0].time == "Timestamp unavailable"
    assert {item.source_document_id for item in report.timeline_items if item.source_document_id} == {"ocr401.log"}
    assert report.supporting_evidence == []
    assert len(report.timeline_items) == 2
    assert report.timeline_items[0].title == "HTTP 401 Unauthorized observed"
    assert report.timeline_items[0].underlying_evidence_ids == ["1"]
    assert report.timeline_items[0].supporting_evidence_count == 1
    assert report.timeline_items[1].title == "Evidence gap identified"
    assert report.timeline_items[1].underlying_evidence_ids == []
    assert summary.next_actions == [
        "Inspect gateway, identity-provider, and target-service logs for the rejected request.",
        "Search by endpoint, caller identity, request ID, or the surrounding time window to identify the rejecting service.",
        "Verify the credentials or token used by the request and confirm the expected scope or policy.",
    ]
    assert report.immediate_actions == summary.next_actions
    assert all("restic" not in action.lower() and "stderr" not in action.lower() for action in report.immediate_actions)
    assert report.observed_failure
    assert report.inferred_explanation == report.root_cause
    assert any(item.category == "Immediate investigation" for item in report.recommended_actions)
    assert any("rejecting service" in gap.lower() for gap in report.evidence_gaps)
    explanation_text = " ".join(report.confidence_explanation)
    assert "single source document" in explanation_text.lower()
    assert "observed timestamps are missing" in explanation_text.lower()
    assert "http 401 unauthorized" in explanation_text.lower()
    assert "rejecting service could not be determined" in explanation_text.lower()
    assert "no corroborating gateway, authentication, or target-service logs" in explanation_text.lower()
    assert "does not establish whether the failure was authentication" in explanation_text.lower()
    assert evidence.primary_causal_event is not None
    assert evidence.primary_causal_event.source_file == "ocr401.log"


def test_lock_refresh_report_actions_are_not_generic_restic_fallbacks():
    content = '''2026-08-02T14:00:00Z backup-service backup-service INFO repository opened successfully
source file could not be accessed because another process was using it
2026-08-02T14:00:02Z backup-service backup-service ERROR failed to refresh lock in time
2026-08-02T14:00:03Z backup-service backup-service ERROR unable to save snapshot: context canceled
'''

    _scenario, _timeline, _evidence, summary, report = _build_uploaded_report(content, "lock.log")

    assert "lock-refresh failure" in report.root_cause.lower()
    assert any("lock-manager" in action.lower() or "stale lock" in action.lower() for action in summary.next_actions)
    assert all("restic stderr" not in action.lower() for action in summary.next_actions)
    finding_titles = [item.title.lower() for item in report.timeline_items]
    assert finding_titles[:3] == [
        "access conflict observed",
        "lock failure detected",
        "operation cancelled",
    ]


def test_restic_jsonlines_report_actions_follow_quota_signal():
    content = '''{"message_type":"verbose_status","action":"scan_finished","item":"","duration":0.203324484,"data_size":6553619,"data_size_in_repo":0,"metadata_size":0,"metadata_size_in_repo":0,"total_files":22}
{"message_type":"status","percent_done":0.26075134975042036,"total_files":22,"files_done":2,"total_bytes":6553619,"bytes_done":1708865,"current_files":["/repo/prod_dump.sql"]}
Fatal: unable to save snapshot: write /repo/data/...: no space left on device'''

    _scenario, _timeline, _evidence, summary, report = _build_uploaded_report(content, "restic.log")

    assert "storage-capacity failure" in report.root_cause.lower()
    joined_actions = " ".join(summary.next_actions).lower()
    assert "capacity" in joined_actions or "space" in joined_actions or "quota" in joined_actions
    assert "stderr" not in joined_actions


def test_structured_resource_access_failure_uses_semantic_report_outputs():
    content = '''time="2017-03-16T13:36:32+08:00" level=error msg="Unable to fetch object info." cause="read /data/fs.json: bad file descriptor" source="[object-handler]" stack="fs.go:1 handler.go:2"'''

    _scenario, _timeline, evidence, summary, report = _build_uploaded_report(content, "backup_service.log")

    assert evidence.primary_causal_event is not None
    assert evidence.primary_causal_event.semantic_error_code == "RESOURCE_ACCESS_FAILURE"
    assert evidence.primary_causal_event.cause_type == "RESOURCE_ACCESS"
    assert evidence.failure_mode == "resource_access_failure"
    assert report.observed_failure == "Object metadata access failed."
    assert report.inferred_explanation == (
        "The service could not read object metadata because the underlying system returned read /data/fs.json: bad file descriptor."
    )
    assert report.primary_causal_label == "Object metadata access failure"
    assert any("does not establish whether the underlying problem was filesystem state" in gap.lower() for gap in report.evidence_gaps)
    joined_actions = " ".join(summary.next_actions).lower()
    assert "fatal line" not in joined_actions
    assert "restic stderr" not in joined_actions
    categories = [item.category for item in report.recommended_actions]
    assert "Immediate investigation" in categories
    assert "Immediate mitigation" in categories
    assert "Preventive improvement" in categories


def test_grouped_timeline_merges_repeated_quota_observations():
    scenario = parse_scenario(SCENARIOS_DIR / "storage-quota-exceeded.json")
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
    evidence = select_investigation_evidence(context)
    summary = generate_simulated_investigation_summary(context, evidence)
    report = build_investigation_report_view(evidence, summary, scenario, timeline)

    assert len(report.timeline_items) < len(evidence.causal_chain)
    assert any(item.supporting_evidence_count > 1 for item in report.timeline_items)
    assert any("capacity" in item.title.lower() or "quota" in item.title.lower() for item in report.timeline_items)
    assert all("grouped operational observations" not in item.title.lower() for item in report.timeline_items)


def test_timestampless_command_status_report_view_uses_unknown_metadata_and_unavailable_time():
    content = '''$ ./health-check.sh
section=cluster
state=degraded
entity_id=node-7
count=2
available=1
summary=one node down'''

    _scenario, _timeline, _evidence, _summary, report = _build_uploaded_report(content, "status.log")

    assert report.incident_summary.operation == "Unknown"
    assert report.incident_summary.target == "Not provided"
    assert report.incident_summary.environment == "Not provided"
    assert report.incident_summary.started == "Timestamp unavailable"
    assert report.incident_summary.duration == "Unknown"
    assert all(item.time == "Timestamp unavailable" for item in report.timeline_items)


def test_grouped_timeline_preserves_all_underlying_evidence_ids_for_large_snapshot():
    evidence = InvestigationEvidence(
        root_cause="Observed degradation interrupted the operation.",
        primary_causal_event=EvidenceItem(
            event_id="3",
            sequence=3,
            timestamp=None,
            timestamp_source="missing",
            component="monitoring",
            event_type="OPERATIONAL_ERROR",
            terminal_state=None,
            cause_type=None,
            error_code="UNKNOWN_OPERATIONAL_ERROR",
            semantic_error_code="UNKNOWN_OPERATIONAL_ERROR",
            original_message="health HEALTH_ERR",
            message="health HEALTH_ERR",
            role="PRIMARY_CAUSE",
            source_file="monitoring.log",
            structured_fields={},
            summary="Operational Error failure detected.",
            display_summary="Operational Error failure detected.",
            display_detail="Primary cause: health HEALTH_ERR",
        ),
        causal_chain=[
            EvidenceItem(
                event_id=str(index),
                sequence=index,
                timestamp=None,
                timestamp_source="missing",
                component="monitoring",
                event_type="OPERATIONAL_ERROR" if index >= 3 else "INFO",
                terminal_state=None,
                cause_type=None,
                error_code="UNKNOWN_OPERATIONAL_ERROR" if index >= 3 else None,
                semantic_error_code="UNKNOWN_OPERATIONAL_ERROR" if index >= 3 else None,
                original_message=message,
                message=message,
                role=role,
                source_file="monitoring.log",
                structured_fields={},
                summary=summary_text,
                display_summary=summary_text,
                display_detail=summary_text,
            )
            for index, role, message, summary_text in [
                (1, "CONTEXT", "ceph -s", "Ceph -s"),
                (2, "CONTEXT", "cluster 0d7a", "Cluster 0d7a"),
                (3, "PRIMARY_CAUSE", "health HEALTH_ERR", "Operational Error failure detected."),
                (4, "CONTRIBUTING", "128 pgs are stuck inactive", "Operational Error contributing condition."),
                (5, "CONTRIBUTING", "64 pgs degraded", "Operational Error contributing condition."),
                (6, "CONTRIBUTING", "64 pgs stale", "Operational Error contributing condition."),
                (7, "CONTRIBUTING", "64 pgs stuck degraded", "Operational Error contributing condition."),
                (8, "CONTRIBUTING", "64 pgs stuck inactive", "Operational Error contributing condition."),
                (9, "CONTRIBUTING", "64 pgs stuck stale", "Operational Error contributing condition."),
                (10, "CONTEXT", "ceph osd tree", "Ceph osd tree"),
                (11, "CONTEXT", "ID WEIGHT TYPE NAME UP/DOWN", "ID WEIGHT TYPE NAME UP/DOWN"),
                (12, "CONTEXT", "1 0.9 osd.1 down", "1 0.9 osd.1 down"),
                (13, "CONTEXT", "0 0.9 osd.0 down", "0 0.9 osd.0 down"),
            ]
        ],
        supporting_events=[],
        limitations=[
            "Observed timestamps are missing for part of the evidence; event order is preserved from ingestion sequence.",
            "The investigation is based on a single source document.",
        ],
        confidence=0.4,
        confidence_explanation=[],
    )
    scenario = Scenario(
        scenarioId="snapshot",
        name="Snapshot",
        category="uploaded",
        tags=["uploaded"],
        description="Snapshot",
        customer="Test",
        jobId="job-snapshot",
        correlationId="corr-snapshot",
        componentsInvolved=["monitoring"],
        expectedOutcome="FAILED",
        expectedFailureSignature="health err",
        logs=[],
    )
    timeline = SimpleNamespace(
        entries=[
            SimpleNamespace(
                event=SimpleNamespace(
                    timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                    structured_fields={"timestamp_source": "missing"},
                    message="snapshot",
                )
            )
        ],
        duration_seconds=0.0,
    )
    summary = InvestigationSummary(
        likely_root_cause=evidence.root_cause,
        supporting_evidence=[],
        next_actions=[],
        confidence=0.4,
    )
    report = build_investigation_report_view(evidence, summary, scenario, timeline)

    titles = [item.title for item in report.timeline_items]
    assert len(report.timeline_items) <= 6
    assert "Critical system health detected" in titles
    assert "Resource degradation observed" in titles
    assert "Resource availability issues detected" in titles
    assert "Evidence gap identified" in titles
    assert all("Grouped operational observations" != title for title in titles)
    grouped_ids = {event_id for item in report.timeline_items for event_id in item.underlying_evidence_ids}
    assert grouped_ids == {"3", "4", "5", "6", "7", "8", "9", "12", "13"}


def test_select_investigation_evidence_uses_semantic_roles_over_sequence_position():
    entries = [
        TimelineEntryContext(
            sequence=1,
            timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc).isoformat(),
            component="storage",
            level="ERROR",
            error_code="PLAINTEXT_FATAL",
            message="file access conflict detected",
            gap_before_seconds=0.0,
            structured_fields={"raw_line": "file access conflict detected", "message_type": "error"},
            event_type="ACCESS_OR_RESOURCE_CONFLICT",
            terminal_state=None,
            cause_type="RESOURCE_OR_ACCESS_CONTENTION",
            semantic_error_code="ACCESS_OR_RESOURCE_CONFLICT",
        ),
        TimelineEntryContext(
            sequence=2,
            timestamp=datetime(2026, 1, 1, 12, 0, 5, tzinfo=timezone.utc).isoformat(),
            component="storage",
            level="ERROR",
            error_code="PLAINTEXT_FATAL",
            message="lock refresh failed",
            gap_before_seconds=5.0,
            structured_fields={"raw_line": "lock refresh failed", "message_type": "error"},
            event_type="LOCK_OR_CONCURRENCY_FAILURE",
            terminal_state=None,
            cause_type="LOCK_OR_CONCURRENCY",
            semantic_error_code="LOCK_OR_CONCURRENCY_FAILURE",
        ),
        TimelineEntryContext(
            sequence=3,
            timestamp=datetime(2026, 1, 1, 12, 0, 10, tzinfo=timezone.utc).isoformat(),
            component="storage",
            level="ERROR",
            error_code="PLAINTEXT_FATAL",
            message="snapshot save cancelled",
            gap_before_seconds=5.0,
            structured_fields={"raw_line": "snapshot save cancelled", "message_type": "error"},
            event_type="OPERATION_ABORTED_OR_CANCELLED",
            terminal_state=None,
            cause_type=None,
            semantic_error_code="OPERATION_ABORTED_OR_CANCELLED",
        ),
    ]

    context = InvestigationContext(
        customer="Test Customer",
        jobId="job-test",
        outcome="FAILED",
        expectedFailureSignature="ignored",
        recovery_timeline=entries,
        warning_events=[],
        error_events=[],
    )

    evidence = select_investigation_evidence(context)
    roles = {item.original_message: item.role for item in evidence.causal_chain}

    assert roles["file access conflict detected"] == "CONTRIBUTING"
    assert roles["lock refresh failed"] == "PRIMARY_CAUSE"
    assert roles["snapshot save cancelled"] == "OUTCOME"
    assert evidence.primary_causal_event is not None
    assert evidence.primary_causal_event.event_id == "2"
    assert any("earlier file-access conflict caused the lock-refresh failure" in limitation for limitation in evidence.limitations) is True


def test_select_investigation_evidence_treats_later_nonpropagating_errors_as_contributing():
    entries = [
        TimelineEntryContext(
            sequence=1,
            timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc).isoformat(),
            component="storage",
            level="ERROR",
            error_code="PLAINTEXT_FATAL",
            message="lock refresh failed",
            gap_before_seconds=0.0,
            structured_fields={"raw_line": "lock refresh failed", "message_type": "error"},
            event_type="LOCK_OR_CONCURRENCY_FAILURE",
            terminal_state=None,
            cause_type="LOCK_OR_CONCURRENCY",
            semantic_error_code="LOCK_OR_CONCURRENCY_FAILURE",
        ),
        TimelineEntryContext(
            sequence=2,
            timestamp=datetime(2026, 1, 1, 12, 0, 5, tzinfo=timezone.utc).isoformat(),
            component="storage",
            level="ERROR",
            error_code="PLAINTEXT_FATAL",
            message="temporary operational issue",
            gap_before_seconds=5.0,
            structured_fields={"raw_line": "temporary operational issue", "message_type": "error"},
            event_type="OPERATIONAL_ERROR",
            terminal_state=None,
            cause_type=None,
            semantic_error_code="UNKNOWN_OPERATIONAL_ERROR",
        ),
    ]

    context = InvestigationContext(
        customer="Test Customer",
        jobId="job-test",
        outcome="FAILED",
        expectedFailureSignature="ignored",
        recovery_timeline=entries,
        warning_events=[],
        error_events=[],
    )

    evidence = select_investigation_evidence(context)
    roles = {item.original_message: item.role for item in evidence.causal_chain}

    assert roles["lock refresh failed"] == "PRIMARY_CAUSE"
    assert roles["temporary operational issue"] == "CONTRIBUTING"


def test_select_investigation_evidence_filters_context_noise():
    entries = [
        TimelineEntryContext(
            sequence=1,
            timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc).isoformat(),
            component="storage",
            level="INFO",
            error_code=None,
            message="copying 10 files",
            gap_before_seconds=0.0,
            structured_fields={"raw_line": "copying 10 files", "message_type": "info"},
            event_type="INFO",
            terminal_state=None,
            cause_type=None,
            semantic_error_code=None,
        ),
        TimelineEntryContext(
            sequence=2,
            timestamp=datetime(2026, 1, 1, 12, 0, 5, tzinfo=timezone.utc).isoformat(),
            component="storage",
            level="ERROR",
            error_code="PLAINTEXT_FATAL",
            message="lock refresh failed",
            gap_before_seconds=5.0,
            structured_fields={"raw_line": "lock refresh failed", "message_type": "error"},
            event_type="LOCK_OR_CONCURRENCY_FAILURE",
            terminal_state=None,
            cause_type="LOCK_OR_CONCURRENCY",
            semantic_error_code="LOCK_OR_CONCURRENCY_FAILURE",
        ),
    ]

    context = InvestigationContext(
        customer="Test Customer",
        jobId="job-test",
        outcome="FAILED",
        expectedFailureSignature="ignored",
        recovery_timeline=entries,
        warning_events=[],
        error_events=[],
    )

    evidence = select_investigation_evidence(context)
    assert all(item.role != "CONTEXT" for item in evidence.causal_chain if item.original_message == "copying 10 files")
    assert any(item.original_message == "lock refresh failed" and item.role == "PRIMARY_CAUSE" for item in evidence.causal_chain)


def test_generate_investigation_summary_parses_valid_llm_response(monkeypatch):
    scenario = parse_scenario(SCENARIO_PATH)
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

    response_text = json.dumps(
        {
            "likely_root_cause": "Storage quota exhaustion caused the backup job failure.",
            "supporting_evidence": [
                "The storage service reported QUOTA_EXCEEDED during upload.",
                "The backup service marked the job FAILED immediately after the quota error.",
            ],
            "next_actions": [
                "Increase storage quota or clean up old backups.",
                "Add quota monitoring alerts earlier in the pipeline.",
            ],
            "confidence": 0.87,
        }
    )

    import recovery_workspace.investigation as investigation_module

    monkeypatch.setattr(
        investigation_module,
        "client",
        DummyClient(response_text),
    )

    summary, used_fallback = generate_investigation_summary(context)

    assert used_fallback is False
    assert summary.likely_root_cause.startswith("Storage quota exhaustion")
    assert summary.supporting_evidence[0].startswith("The storage service reported")
    assert summary.next_actions[1].startswith("Add quota monitoring")
    assert summary.confidence == pytest.approx(0.87)


def test_generate_investigation_summary_requires_api_key(monkeypatch):
    import recovery_workspace.investigation as investigation_module

    monkeypatch.setattr(investigation_module, "client", None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    scenario = parse_scenario(SCENARIO_PATH)
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

    # With fallback enabled, missing API key should return deterministic fallback
    summary, used_fallback = generate_investigation_summary(context)
    
    assert used_fallback is True
    assert summary.likely_root_cause is not None
    assert len(summary.supporting_evidence) > 0
    assert len(summary.next_actions) > 0
    assert 0.0 <= summary.confidence <= 1.0


def test_generate_investigation_summary_uses_fallback_on_missing_api_key(monkeypatch):
    scenario = parse_scenario(SCENARIO_PATH)
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

    import recovery_workspace.investigation as investigation_module
    monkeypatch.setattr(investigation_module, "client", None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    summary, used_fallback = generate_investigation_summary(context)

    assert used_fallback is True
    # Verify fallback summary is generated from timeline evidence
    assert summary.likely_root_cause is not None
    assert len(summary.supporting_evidence) > 0
    assert all(isinstance(e, str) for e in summary.supporting_evidence)
    assert 0.0 <= summary.confidence <= 1.0


def test_generate_simulated_investigation_summary_never_requires_openai(monkeypatch):
    import recovery_workspace.investigation as investigation_module

    monkeypatch.setattr(
        investigation_module,
        "_get_llm_client",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("OpenAI client should not be created")),
    )

    scenario = parse_scenario(SCENARIO_PATH)
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

    summary = generate_simulated_investigation_summary(context)

    assert summary.likely_root_cause is not None
    assert len(summary.supporting_evidence) > 0


def test_insufficient_quota_falls_back_without_retries(monkeypatch):
    import recovery_workspace.investigation as investigation_module

    failing_client = RateLimitClient()
    monkeypatch.setattr(
        investigation_module,
        "_get_llm_client",
        lambda *args, **kwargs: failing_client,
    )

    scenario = parse_scenario(SCENARIO_PATH)
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

    summary, used_fallback = generate_investigation_summary(context)

    assert used_fallback is True
    assert summary.likely_root_cause is not None
    assert failing_client.responses.calls == 1


def test_generate_investigation_summary_rejects_malformed_response(monkeypatch):
    scenario = parse_scenario(SCENARIO_PATH)
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

    bad_response = "{invalid json"

    import recovery_workspace.investigation as investigation_module

    monkeypatch.setattr(
        investigation_module,
        "client",
        DummyClient(bad_response),
    )

    with pytest.raises(ValueError, match="LLM response is not valid JSON"):
        generate_investigation_summary(context)


def _build_context_for_scenario(
    scenario_filename: str,
    *,
    expected_failure_signature: Optional[str] = None,
):
    scenario = parse_scenario(SCENARIOS_DIR / scenario_filename)
    events = normalize_events(scenario.logs)
    timeline = build_timeline(events)
    context = build_investigation_context(
        customer=scenario.customer,
        job_id=scenario.job_id,
        outcome=scenario.expected_outcome,
        expected_failure_signature=(
            expected_failure_signature
            if expected_failure_signature is not None
            else scenario.expected_failure_signature
        ),
        timeline=timeline,
        events=events,
    )
    return scenario, context


@pytest.mark.parametrize(
    ("scenario_filename", "root_cause_terms", "action_terms", "forbidden_action_terms"),
    [
        (
            "storage-quota-exceeded.json",
            ("capacity", "quota"),
            ("capacity", "quota"),
            ("network",),
        ),
        (
            "permission-denied.json",
            ("authorization", "permission"),
            ("permission", "policy"),
            ("network", "capacity"),
        ),
        (
            "network-timeout.json",
            ("network", "timeout"),
            ("network", "timeout"),
            ("quota", "capacity"),
        ),
        (
            "service-unavailable.json",
            ("service", "outage"),
            ("maintenance", "dependency"),
            ("network connectivity", "capacity"),
        ),
        (
            "clock-skew.json",
            ("clock", "skew"),
            ("ntp", "time"),
            ("network connectivity", "capacity"),
        ),
        (
            "partial-failure.json",
            ("partial", "child"),
            ("failed volumes", "per-task"),
            ("resource utilization",),
        ),
    ],
)
def test_simulated_fallback_is_scenario_specific(
    scenario_filename: str,
    root_cause_terms: tuple[str, ...],
    action_terms: tuple[str, ...],
    forbidden_action_terms: tuple[str, ...],
):
    _, context = _build_context_for_scenario(scenario_filename)
    summary = generate_simulated_investigation_summary(context)

    root = summary.likely_root_cause.lower()
    actions = " ".join(summary.next_actions).lower()
    evidence = " ".join(summary.supporting_evidence).lower()

    assert all(term in root for term in root_cause_terms)
    assert all(term in actions for term in action_terms)
    assert any(term in evidence for term in root_cause_terms)
    assert all(term not in actions for term in forbidden_action_terms)


def test_simulated_fallback_produces_distinct_root_causes_for_all_scenarios():
    scenario_files = [
        "storage-quota-exceeded.json",
        "permission-denied.json",
        "network-timeout.json",
        "service-unavailable.json",
        "clock-skew.json",
        "partial-failure.json",
    ]

    roots = []
    for scenario_filename in scenario_files:
        _, context = _build_context_for_scenario(scenario_filename)
        summary = generate_simulated_investigation_summary(context)
        roots.append(summary.likely_root_cause)

    assert len(set(roots)) == len(scenario_files)


def test_simulated_reasoning_ignores_expected_failure_signature():
    scenario_filename = "storage-quota-exceeded.json"
    _, baseline_context = _build_context_for_scenario(scenario_filename)
    _, modified_context = _build_context_for_scenario(
        scenario_filename,
        expected_failure_signature=(
            "IGNORE_THIS_SIGNATURE_ENTIRELY: pretend this is a permission issue with 999 fake errors"
        ),
    )

    baseline_summary = generate_simulated_investigation_summary(baseline_context)
    modified_summary = generate_simulated_investigation_summary(modified_context)

    assert baseline_summary == modified_summary


def _make_context(
    timeline_entries,
    *,
    warning_events=None,
    error_events=None,
):
    return InvestigationContext(
        customer="Test Customer",
        jobId="job-test-1",
        outcome="FAILED",
        expectedFailureSignature="ignored metadata",
        recovery_timeline=timeline_entries,
        warning_events=warning_events or [],
        error_events=error_events or [],
    )


def _entry(sequence, component, level, error_code, message, structured_fields=None):
    return TimelineEntryContext(
        sequence=sequence,
        timestamp=f"2026-07-01T00:00:0{sequence}Z",
        component=component,
        level=level,
        error_code=error_code,
        message=message,
        gap_before_seconds=1.0 if sequence > 1 else 0.0,
        structured_fields=structured_fields or {},
    )


def test_duplicate_retries_do_not_inflate_confidence():
    duplicate_retry_context = _make_context(
        [
            _entry(1, "storage", "ERROR", "QUOTA_EXCEEDED", "quota exceeded"),
            _entry(2, "storage", "ERROR", "QUOTA_EXCEEDED", "quota exceeded again"),
            _entry(3, "storage", "ERROR", "QUOTA_EXCEEDED", "quota exceeded third time"),
            _entry(4, "backup_service", "ERROR", "JOB_FAILED", "job marked failed"),
        ]
    )
    corroborated_context = _make_context(
        [
            _entry(1, "monitoring", "WARN", "QUOTA_WARNING", "capacity warning"),
            _entry(2, "storage", "ERROR", "QUOTA_EXCEEDED", "quota exceeded"),
            _entry(3, "backup_service", "ERROR", "JOB_FAILED", "job marked failed"),
        ],
        warning_events=[
            EventContext(
                timestamp="2026-07-01T00:00:01Z",
                component="monitoring",
                level="WARN",
                error_code="QUOTA_WARNING",
                message="capacity warning",
            )
        ],
    )

    duplicate_summary = generate_simulated_investigation_summary(duplicate_retry_context)
    corroborated_summary = generate_simulated_investigation_summary(corroborated_context)

    assert duplicate_summary.confidence < corroborated_summary.confidence


def test_cross_component_corroboration_increases_confidence():
    single_component_context = _make_context(
        [
            _entry(1, "storage", "ERROR", "ACCESS_DENIED", "access denied"),
            _entry(2, "backup_service", "ERROR", "JOB_FAILED", "job marked failed"),
        ]
    )
    multi_component_context = _make_context(
        [
            _entry(1, "auth", "WARN", None, "policy update reduced scope"),
            _entry(2, "storage", "ERROR", "ACCESS_DENIED", "access denied"),
            _entry(3, "backup_service", "ERROR", "PERMISSION_DENIED", "job marked failed"),
        ],
        warning_events=[
            EventContext(
                timestamp="2026-07-01T00:00:01Z",
                component="auth",
                level="WARN",
                error_code=None,
                message="policy update reduced scope",
            )
        ],
    )

    single_summary = generate_simulated_investigation_summary(single_component_context)
    multi_summary = generate_simulated_investigation_summary(multi_component_context)

    assert multi_summary.confidence > single_summary.confidence


def test_job_failed_alone_produces_low_confidence():
    context = _make_context(
        [
            _entry(1, "backup_service", "ERROR", "JOB_FAILED", "job marked failed"),
        ]
    )

    summary = generate_simulated_investigation_summary(context)

    assert summary.confidence < 0.4


def test_conflicting_evidence_lowers_confidence():
    consistent_context = _make_context(
        [
            _entry(1, "monitoring", "WARN", "QUOTA_WARNING", "quota nearing limit"),
            _entry(2, "storage", "ERROR", "QUOTA_EXCEEDED", "quota exceeded"),
            _entry(3, "backup_service", "ERROR", "JOB_FAILED", "job marked failed"),
        ],
        warning_events=[
            EventContext(
                timestamp="2026-07-01T00:00:01Z",
                component="monitoring",
                level="WARN",
                error_code="QUOTA_WARNING",
                message="quota nearing limit",
            )
        ],
    )
    conflicting_context = _make_context(
        [
            _entry(1, "monitoring", "WARN", "QUOTA_WARNING", "quota nearing limit"),
            _entry(2, "storage", "ERROR", "QUOTA_EXCEEDED", "quota exceeded"),
            _entry(3, "network", "ERROR", "ETIMEDOUT", "network timeout"),
            _entry(4, "backup_service", "ERROR", "JOB_FAILED", "job marked failed"),
        ],
        warning_events=[
            EventContext(
                timestamp="2026-07-01T00:00:01Z",
                component="monitoring",
                level="WARN",
                error_code="QUOTA_WARNING",
                message="quota nearing limit",
            )
        ],
    )

    consistent_summary = generate_simulated_investigation_summary(consistent_context)
    conflicting_summary = generate_simulated_investigation_summary(conflicting_context)

    assert conflicting_summary.confidence < consistent_summary.confidence


def test_complete_causal_chain_produces_high_confidence():
    context = _make_context(
        [
            _entry(1, "monitoring", "WARN", "QUOTA_WARNING", "quota at 92%"),
            _entry(2, "storage", "ERROR", "QUOTA_EXCEEDED", "quota exceeded"),
            _entry(3, "backup_service", "ERROR", "JOB_FAILED", "job marked failed"),
        ],
        warning_events=[
            EventContext(
                timestamp="2026-07-01T00:00:01Z",
                component="monitoring",
                level="WARN",
                error_code="QUOTA_WARNING",
                message="quota at 92%",
            )
        ],
    )

    summary = generate_simulated_investigation_summary(context)

    assert summary.confidence >= 0.75


def test_structured_error_fields_raise_causal_signal_and_corroboration():
    context = _make_context(
        [
            _entry(1, "monitoring", "WARN", None, "storage volume at 92% capacity"),
            _entry(
                2,
                "storage",
                "ERROR",
                None,
                "verbose_status",
                structured_fields={
                    "message_type": "error",
                    "error": "unable to save blob: no space left on device",
                    "message": "unable to save blob: no space left on device",
                },
            ),
            _entry(3, "backup_service", "ERROR", "JOB_FAILED", "job marked failed"),
        ],
        warning_events=[
            EventContext(
                timestamp="2026-07-01T00:00:01Z",
                component="monitoring",
                level="WARN",
                error_code=None,
                message="storage volume at 92% capacity",
                structured_fields={},
            )
        ],
    )

    summary = generate_simulated_investigation_summary(context)

    assert summary.confidence_breakdown["causal_signal"] >= 0.3
    assert summary.confidence_breakdown["corroboration"] >= 0.14
    assert "quota" in summary.likely_root_cause.lower() or "capacity" in summary.likely_root_cause.lower()


def test_raw_plaintext_failure_lines_contribute_to_causal_signal():
    context = _make_context(
        [
            _entry(1, "monitoring", "WARN", None, "storage volume at 92% capacity"),
            _entry(
                2,
                "backup_service",
                "ERROR",
                "PLAINTEXT_FATAL",
                "Fatal: unable to save snapshot: write /repo/data/...: no space left on device",
                structured_fields={
                    "raw_line": "Fatal: unable to save snapshot: write /repo/data/...: no space left on device",
                    "parse_status": "fatal_plaintext",
                },
            ),
            _entry(3, "backup_service", "ERROR", "JOB_FAILED", "job marked failed"),
        ],
        warning_events=[
            EventContext(
                timestamp="2026-07-01T00:00:01Z",
                component="monitoring",
                level="WARN",
                error_code=None,
                message="storage volume at 92% capacity",
                structured_fields={},
            )
        ],
    )

    summary = generate_simulated_investigation_summary(context)

    assert summary.confidence_breakdown["causal_signal"] >= 0.3
    assert summary.confidence > 0.3
    assert "quota" in summary.likely_root_cause.lower() or "capacity" in summary.likely_root_cause.lower()


def test_generic_fatal_line_uses_neighboring_context_for_family_detection():
    context = _make_context(
        [
            _entry(1, "storage", "WARN", None, "storage volume at 92% capacity"),
            _entry(
                2,
                "backup_service",
                "ERROR",
                "PLAINTEXT_FATAL",
                "Fatal: unable to save snapshot",
                structured_fields={
                    "raw_line": "Fatal: unable to save snapshot",
                    "parse_status": "fatal_plaintext",
                },
            ),
            _entry(3, "backup_service", "ERROR", "JOB_FAILED", "job marked failed"),
        ],
        warning_events=[
            EventContext(
                timestamp="2026-07-01T00:00:01Z",
                component="storage",
                level="WARN",
                error_code=None,
                message="storage volume at 92% capacity",
                structured_fields={},
            )
        ],
    )

    summary = generate_simulated_investigation_summary(context)

    assert summary.confidence_breakdown["causal_signal"] >= 0.16
    assert "quota" in summary.likely_root_cause.lower() or "capacity" in summary.likely_root_cause.lower()
    assert all("restic" not in action.lower() for action in summary.next_actions)


def test_routine_non_alarm_line_does_not_count_as_corroboration():
    context = _make_context(
        [
            _entry(
                1,
                "backup_service",
                "ERROR",
                "PLAINTEXT_FATAL",
                "Fatal: unable to save snapshot",
                structured_fields={
                    "raw_line": "Fatal: unable to save snapshot",
                    "parse_status": "fatal_plaintext",
                },
            ),
            _entry(2, "storage", "INFO", None, "nightly integrity scan started"),
            _entry(3, "backup_service", "ERROR", "JOB_FAILED", "job marked failed"),
        ]
    )

    summary = generate_simulated_investigation_summary(context)

    assert summary.confidence_breakdown["corroboration"] <= 0.05


def test_confidence_scores_are_deterministic_and_bounded():
    _, context = _build_context_for_scenario("network-timeout.json")

    first = generate_simulated_investigation_summary(context)
    second = generate_simulated_investigation_summary(context)

    assert first.confidence == second.confidence
    assert first.confidence_breakdown == second.confidence_breakdown
    assert 0.0 <= first.confidence <= 0.95
    assert all(-0.5 <= value <= 0.5 for value in first.confidence_breakdown.values())


def test_reasoning_context_payload_excludes_expected_failure_signature():
    _, context = _build_context_for_scenario("storage-quota-exceeded.json")

    payload = _reasoning_context_payload(context)

    assert "expectedFailureSignature" not in payload


def test_openai_request_input_excludes_expected_failure_signature(monkeypatch):
    captured = {}

    class CapturingResponses:
        def create(self, **kwargs):
            captured["input"] = kwargs["input"]
            return _mock_response(
                json.dumps(
                    {
                        "likely_root_cause": "Captured result",
                        "supporting_evidence": ["Evidence"],
                        "next_actions": ["Action"],
                        "confidence": 0.5,
                    }
                )
            )

    class CapturingClient:
        def __init__(self):
            self.responses = CapturingResponses()

    import recovery_workspace.investigation as investigation_module

    monkeypatch.setattr(investigation_module, "client", CapturingClient())
    _, context = _build_context_for_scenario("storage-quota-exceeded.json")

    generate_llm_investigation_summary(context)

    assert "expectedFailureSignature" not in captured["input"]


@pytest.mark.parametrize(
    "scenario_filename",
    [
        "storage-quota-exceeded.json",
        "permission-denied.json",
        "network-timeout.json",
        "service-unavailable.json",
        "clock-skew.json",
        "partial-failure.json",
    ],
)
def test_select_investigation_evidence_keeps_primary_cause_structured(scenario_filename: str):
    _, context = _build_context_for_scenario(scenario_filename)

    evidence = select_investigation_evidence(context)

    assert evidence.primary_causal_event is not None
    assert evidence.primary_causal_event.role == "PRIMARY_CAUSE"
    assert evidence.primary_causal_event.event_id in {item.event_id for item in evidence.causal_chain}
    assert len(evidence.causal_chain) >= len(evidence.causal_events)
    assert evidence.root_cause


def test_evidence_items_preserve_original_messages():
    _, context = _build_context_for_scenario("storage-quota-exceeded.json")

    evidence = select_investigation_evidence(context)

    assert any(item.original_message for item in evidence.causal_chain)
    assert all(item.original_message == item.message for item in evidence.causal_chain)
