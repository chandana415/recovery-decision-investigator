from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

LogLevel = Literal["INFO", "WARN", "ERROR"]


class LogEntry(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    timestamp: datetime
    correlation_id: str = Field(alias="correlationId")
    trace_id: str | None = Field(default=None, alias="traceId")
    component_id: str = Field(alias="componentId")
    component: str
    level: LogLevel
    error_code: str | None = Field(alias="errorCode")
    message: str


class Event(BaseModel):
    timestamp: datetime
    correlation_id: str
    trace_id: str | None = None
    component_id: str
    component: str
    level: LogLevel
    error_code: str | None
    message: str

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
    error_code: str | None = None
    message: str
    gap_before_seconds: float


class EventContext(BaseModel):
    timestamp: str
    component: str
    level: LogLevel
    error_code: str | None = None
    message: str


class InvestigationContext(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    customer: str
    job_id: str = Field(alias="jobId")
    outcome: str
    expected_failure_signature: str = Field(alias="expectedFailureSignature")
    recovery_timeline: list[TimelineEntryContext]
    warning_events: list[EventContext]
    error_events: list[EventContext]


class InvestigationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    likely_root_cause: str
    supporting_evidence: list[str]
    next_actions: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_breakdown: dict[str, float] = Field(default_factory=dict)
    confidence_explanation: list[str] = Field(default_factory=list)


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
