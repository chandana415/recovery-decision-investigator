from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

LogLevel = Literal["INFO", "WARN", "ERROR"]
EvidenceRole = Literal["CONTEXT", "CONTRIBUTING", "PRIMARY_CAUSE", "PROPAGATION", "OUTCOME"]


class LogEntry(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    timestamp: datetime
    correlation_id: str = Field(alias="correlationId")
    trace_id: Optional[str] = Field(default=None, alias="traceId")
    component_id: str = Field(alias="componentId")
    component: str
    level: LogLevel
    error_code: Optional[str] = Field(default=None, alias="errorCode")
    message: str
    source_file: Optional[str] = Field(default=None)
    structured_fields: dict[str, Any] = Field(default_factory=dict, alias="structuredFields")


class EventSemantics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_type: Optional[str] = None
    terminal_state: Optional[str] = None
    cause_type: Optional[str] = None
    semantic_error_code: Optional[str] = None
    observed_cause: Optional[str] = None


class Event(BaseModel):
    timestamp: datetime
    correlation_id: str
    trace_id: Optional[str] = None
    component_id: str
    component: str
    level: LogLevel
    error_code: Optional[str]
    message: str
    source_file: Optional[str] = None
    structured_fields: dict[str, Any] = Field(default_factory=dict)
    event_type: Optional[str] = None
    terminal_state: Optional[str] = None
    cause_type: Optional[str] = None
    semantic_error_code: Optional[str] = None
    observed_cause: Optional[str] = None

    @classmethod
    def from_log_entry(cls, log: LogEntry) -> "Event":
        return cls(
            timestamp=log.timestamp,
            correlation_id=log.correlation_id,
            trace_id=log.trace_id,
            component_id=log.component_id,
            component=log.component,
            level=log.level,
            error_code=log.error_code,
            message=log.message,
            source_file=log.source_file,
            structured_fields=log.structured_fields,
        )


class TimelineEntry(BaseModel):
    sequence: int
    event: Event
    gap_before_seconds: float


class Timeline(BaseModel):
    correlation_id: str
    started_at: datetime
    ended_at: datetime
    duration_seconds: float
    entries: list[TimelineEntry]


class TimelineEntryContext(BaseModel):
    sequence: int
    timestamp: str
    component: str
    level: LogLevel
    error_code: Optional[str] = None
    message: str
    gap_before_seconds: float
    source_file: Optional[str] = None
    structured_fields: dict[str, Any] = Field(default_factory=dict)
    event_type: Optional[str] = None
    terminal_state: Optional[str] = None
    cause_type: Optional[str] = None
    semantic_error_code: Optional[str] = None
    observed_cause: Optional[str] = None


class EventContext(BaseModel):
    timestamp: str
    component: str
    level: LogLevel
    error_code: Optional[str] = None
    message: str
    source_file: Optional[str] = None
    structured_fields: dict[str, Any] = Field(default_factory=dict)
    event_type: Optional[str] = None
    terminal_state: Optional[str] = None
    cause_type: Optional[str] = None
    semantic_error_code: Optional[str] = None
    observed_cause: Optional[str] = None


class InvestigationContext(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    customer: str
    job_id: str = Field(alias="jobId")
    outcome: str
    expected_failure_signature: str = Field(alias="expectedFailureSignature")
    recovery_timeline: list[TimelineEntryContext]
    warning_events: list[EventContext]
    error_events: list[EventContext]


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    sequence: Optional[int] = None
    timestamp: Optional[str] = None
    timestamp_source: Optional[str] = None
    component: str
    event_type: Optional[str] = None
    terminal_state: Optional[str] = None
    cause_type: Optional[str] = None
    error_code: Optional[str] = None
    semantic_error_code: Optional[str] = None
    observed_cause: Optional[str] = None
    original_message: str
    message: str
    role: EvidenceRole
    source_file: Optional[str] = None
    structured_fields: dict[str, Any] = Field(default_factory=dict)
    summary: Optional[str] = None
    display_summary: Optional[str] = None
    display_detail: Optional[str] = None


class InvestigationEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    failure_mode: Optional[str] = None
    root_cause: str
    context_events: list[EvidenceItem] = Field(default_factory=list)
    causal_events: list[EvidenceItem] = Field(default_factory=list)
    outcome_events: list[EvidenceItem] = Field(default_factory=list)
    supporting_events: list[EvidenceItem] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    primary_causal_event: Optional[EvidenceItem] = None
    causal_chain: list[EvidenceItem] = Field(default_factory=list)
    confidence_inputs: dict[str, float] = Field(default_factory=dict)
    confidence: float = 0.0
    confidence_breakdown: dict[str, float] = Field(default_factory=dict)
    confidence_explanation: list[str] = Field(default_factory=list)


class InvestigationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    likely_root_cause: str
    supporting_evidence: list[str]
    next_actions: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_breakdown: dict[str, float] = Field(default_factory=dict)
    confidence_explanation: list[str] = Field(default_factory=list)
    evidence: Optional[InvestigationEvidence] = None


class SupportingEvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    role: str
    display_summary: str
    why_it_matters: str
    source_document_id: Optional[str] = None


class TimelineFindingDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    role: str
    timestamp: Optional[str] = None
    timestamp_source: Optional[str] = None
    time: str = ""
    display_summary: str
    display_detail: str
    source: str
    component_id: str
    component_name: str
    source_document_id: Optional[str] = None


class TimelineReportItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    role: str
    title: str
    finding_type: str
    stage: str
    timestamp: Optional[str] = None
    timestamp_source: Optional[str] = None
    time: str = ""
    display_summary: str
    display_detail: str
    source: str
    component_id: str
    component_name: str
    source_document_id: Optional[str] = None
    supporting_evidence_count: int = 0
    underlying_evidence_ids: list[str] = Field(default_factory=list)
    detail_items: list[TimelineFindingDetail] = Field(default_factory=list)


class ConfidenceDimensionView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimension: str
    level: str
    rationale: str


class RecommendedActionView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str
    action: str
    rationale: str


class IncidentSummaryView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer: str
    job_id: str
    status: str
    operation: str
    target: str
    environment: str
    started: str
    ended_label: str
    ended: str
    duration: str


class InvestigationReportView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    incident_summary: IncidentSummaryView
    root_cause: str
    observed_failure: str = ""
    inferred_explanation: str = ""
    evidence_gaps: list[str] = Field(default_factory=list)
    primary_causal_label: str = ""
    primary_causal_event: Optional[EvidenceItem] = None
    outcome_summary: Optional[str] = None
    customer_impact: str
    timeline_items: list[TimelineReportItem] = Field(default_factory=list)
    supporting_evidence: list[SupportingEvidenceItem] = Field(default_factory=list)
    evidence_limitations: list[str] = Field(default_factory=list)
    immediate_actions: list[str] = Field(default_factory=list)
    preventive_actions: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_level: str = "Low"
    confidence_explanation: list[str] = Field(default_factory=list)
    confidence_dimensions: list[ConfidenceDimensionView] = Field(default_factory=list)
    recommended_actions: list[RecommendedActionView] = Field(default_factory=list)


class Scenario(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    scenario_id: str = Field(alias="scenarioId")
    name: str
    category: str
    tags: list[str] = []
    description: str
    customer: str
    job_id: str = Field(alias="jobId")
    correlation_id: str = Field(alias="correlationId")
    components_involved: list[str] = Field(alias="componentsInvolved")
    expected_outcome: str = Field(alias="expectedOutcome")
    expected_failure_signature: str = Field(alias="expectedFailureSignature")
    logs: list[LogEntry]
