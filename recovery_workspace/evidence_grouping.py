from __future__ import annotations

import re
from dataclasses import dataclass, field

from recovery_workspace.models import EvidenceItem, InvestigationEvidence


@dataclass
class EvidenceFindingGroup:
    title: str
    summary: str
    finding_type: str
    stage: str
    items: list[EvidenceItem] = field(default_factory=list)
    underlying_evidence_ids: list[str] = field(default_factory=list)


def _combined_text(item: EvidenceItem) -> str:
    bits = [
        item.message,
        item.display_summary or "",
        item.display_detail or "",
        item.error_code or "",
        item.semantic_error_code or "",
        item.event_type or "",
        item.cause_type or "",
        item.terminal_state or "",
    ]
    return " ".join(bit for bit in bits if bit).lower()


def _contains_any(text: str, tokens: tuple[str, ...]) -> bool:
    return any(token in text for token in tokens)


def _is_pure_metadata(item: EvidenceItem) -> bool:
    text = _combined_text(item)

    if item.role == "PRIMARY_CAUSE":
        return False

    if _contains_any(text, ("ceph -s", "ceph osd tree", "id weight type name up/down", "flags sortbitwise", "cluster ")):
        return True
    if re.search(r"\bcluster\s+[0-9a-f-]{8,}\b", text):
        return True
    if re.search(r"\bmonmap\b", text) or re.search(r"\bquorum\b", text) or re.search(r"\belection epoch\b", text):
        return True
    if re.search(r"^\s*-?\d+\s+\d+(?:\.\d+)?\s+(root|host)\b", item.message.lower()):
        return True
    if re.search(r"\bosd\.\d+\s+up\b", text):
        return True
    if _contains_any(text, ("repository opened successfully", "verbose status context", "scan_finished", "files_done", "bytes_done", "progress")):
        return True
    if _contains_any(text, ("mb used", "gb /", "avail")) and not _contains_any(text, ("quota", "capacity", "full", "no space", "enospc")):
        return True
    return False


def _primary_title(item: EvidenceItem) -> tuple[str, str]:
    semantic_code = (item.semantic_error_code or "").upper()
    cause_type = (item.cause_type or "").upper()
    event_type = (item.event_type or "").upper()

    if semantic_code == "SYSTEM_HEALTH_DEGRADATION" or cause_type == "SYSTEM_DEGRADATION" or event_type == "SYSTEM_HEALTH_DEGRADATION":
        return "Critical system health degradation", "Critical system health degradation was selected as the primary finding."
    if semantic_code == "RESOURCE_ACCESS_FAILURE" or cause_type == "RESOURCE_ACCESS" or event_type == "RESOURCE_ACCESS_FAILURE":
        return "Resource access failure", "A resource-access failure was selected as the primary finding."

    text = _combined_text(item)
    if _contains_any(text, ("http 401", "401 unauthorized", "unauthorized")):
        return "HTTP 401 Unauthorized observed", "A downstream request received HTTP 401 Unauthorized."
    if _contains_any(text, ("http 403", "403 forbidden", "forbidden", "access denied", "permission denied")):
        return "Access denied observed", "An access-denied response was selected as the primary finding."
    if _contains_any(text, ("lock", "refresh lock", "lock_or_concurrency")):
        return "Lock failure detected", "A lock refresh or concurrency failure was selected as the primary finding."
    if _contains_any(text, ("quota", "no space left", "disk full", "capacity", "enospc")):
        return "Quota exhaustion observed", "Storage capacity exhaustion was selected as the primary finding."
    if _contains_any(text, ("cancelled", "canceled", "aborted")):
        return "Operation cancelled", "The selected primary finding indicates the operation was cancelled."
    return item.display_summary or item.summary or item.message, item.display_detail or item.message


def _classify_finding(item: EvidenceItem) -> tuple[str, str, str, str]:
    text = _combined_text(item)

    if item.role == "PRIMARY_CAUSE":
        title, summary = _primary_title(item)
        return title, summary, "PRIMARY_CAUSE", "Primary Causal Event"

    if item.role == "OUTCOME" or _contains_any(text, ("cancelled", "canceled", "aborted", "job failed", "context canceled", "context cancelled")):
        if _contains_any(text, ("cancelled", "canceled", "aborted", "context canceled", "context cancelled")):
            return "Operation cancelled", "The operation was cancelled or terminated after the earlier failure.", "OUTCOME", "Outcome"
        if _contains_any(text, ("completed", "succeeded", "success")):
            return "Operation completed", "The operation reached a terminal completion state.", "OUTCOME", "Outcome"
        return "Operation outcome observed", "A terminal job outcome was recorded.", "OUTCOME", "Outcome"

    if _contains_any(text, ("unauthorized", "forbidden", "permission denied", "access denied", "authentication", "authorization", "credential", "token", "401", "403")):
        return "Authentication/authorization failures observed", "One or more authentication or authorization failures were reported.", "AUTHORIZATION_FAILURE", "Supporting Evidence"

    if _contains_any(text, ("timeout", "timed out", "unreachable", "connection reset", "network", "communication failure", "econnreset", "enetunreach", "etimedout")):
        return "Repeated communication failures observed", "Multiple communication or transport failures were reported.", "COMMUNICATION_FAILURE", "Supporting Evidence"

    if _contains_any(text, ("another process", "access conflict", "contention", "using it", "resource_or_access_contention")):
        return "Access conflict observed", "A resource or file-access conflict was observed before the primary failure.", "ACCESS_CONFLICT", "Supporting Evidence"

    if _contains_any(text, ("lock", "lock_or_concurrency")):
        return "Lock/contention detected", "Lock or concurrency issues were observed.", "LOCK_CONTENTION", "Supporting Evidence"

    if _contains_any(text, ("quota", "capacity", "no space left", "disk full", "enospc")):
        return "Storage capacity pressure detected", "One or more observations indicated rising storage capacity pressure.", "CAPACITY_PRESSURE", "Supporting Evidence"

    if _contains_any(text, ("no daemons active", "service unavailable", "dependency outage", "unavailable")):
        return "Service availability issues detected", "One or more required services were unavailable or inactive.", "SERVICE_AVAILABILITY", "Supporting Evidence"

    if _contains_any(text, ("too few", " down ", "osd.", "creating")) and _contains_any(text, ("too few", " down ", "creating")):
        return "Resource availability issues detected", "One or more required resources were down or below the expected availability level.", "RESOURCE_AVAILABILITY", "Supporting Evidence"

    if _contains_any(text, ("degraded", "stale", "inactive", "unclean", "undersized", "remapped", "health_err", "health_warn")):
        if _contains_any(text, ("health_err", "health_warn")):
            return "Critical system health detected", "Critical system health problems were reported.", "CRITICAL_HEALTH", "Supporting Evidence"
        return "Resource degradation observed", "Multiple resources reported degraded, stale, inactive, or undersized states.", "RESOURCE_DEGRADATION", "Supporting Evidence"

    if item.role == "CONTEXT":
        return "Operational context retained", "A small amount of operational context was retained for the investigation timeline.", "OPERATIONAL_CONTEXT", "Context"

    return "Repeated warning conditions observed", "Multiple warning conditions were observed before or around the failure.", "REPEATED_WARNING", "Supporting Evidence"


def _evidence_gap_group(evidence: InvestigationEvidence) -> EvidenceFindingGroup | None:
    if not evidence.limitations:
        return None

    limitation_text = " ".join(evidence.limitations).lower()
    if "rejecting service could not be determined" in limitation_text:
        summary = "The supplied evidence does not identify which downstream service rejected the request."
    elif "single source document" in limitation_text and "observed timestamps are missing" in limitation_text:
        summary = "The investigation is based on a single source document and missing observed timestamps limit attribution."
    else:
        summary = evidence.limitations[0]

    return EvidenceFindingGroup(
        title="Evidence gap identified",
        summary=summary,
        finding_type="EVIDENCE_GAP",
        stage="Evidence Gap",
        items=[],
        underlying_evidence_ids=[],
    )


def group_evidence_for_timeline(evidence: InvestigationEvidence) -> list[EvidenceFindingGroup]:
    groups: list[EvidenceFindingGroup] = []

    for item in sorted(evidence.causal_chain, key=lambda evidence_item: evidence_item.sequence or 0):
        if _is_pure_metadata(item):
            continue

        title, summary, finding_type, stage = _classify_finding(item)
        if item.role == "PRIMARY_CAUSE":
            groups.append(
                EvidenceFindingGroup(
                    title=title,
                    summary=summary,
                    finding_type=finding_type,
                    stage=stage,
                    items=[item],
                    underlying_evidence_ids=[item.event_id],
                )
            )
            continue

        if groups:
            previous = groups[-1]
            if previous.finding_type == finding_type and previous.stage == stage:
                previous.items.append(item)
                previous.underlying_evidence_ids.append(item.event_id)
                continue

        groups.append(
            EvidenceFindingGroup(
                title=title,
                summary=summary,
                finding_type=finding_type,
                stage=stage,
                items=[item],
                underlying_evidence_ids=[item.event_id],
            )
        )

    evidence_gap = _evidence_gap_group(evidence)
    if evidence_gap is not None:
        groups.append(evidence_gap)

    return groups
