from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from time import perf_counter
import re
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

import streamlit as st

from recovery_workspace.events import normalize_events
from recovery_workspace.investigation import (
    build_investigation_context,
    build_investigation_report_view,
    generate_llm_investigation_summary,
    generate_simulated_investigation_summary,
    select_investigation_evidence,
)
from recovery_workspace.models import Scenario, LogEntry, InvestigationEvidence
from recovery_workspace.parser import parse_scenario
from recovery_workspace.timeline import build_timeline
from recovery_workspace.uploader import (
    parse_uploaded_logs,
    parse_uploaded_logs_with_source,
    process_multiple_uploads,
    read_uploaded_file_safe,
    validate_upload,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENARIOS_DIR = REPO_ROOT / "mock-data" / "scenarios"


def load_scenarios_index() -> list[dict]:
    return json.loads((SCENARIOS_DIR / "index.json").read_text())


@lru_cache(maxsize=1)
def load_scenario_entries() -> list[dict]:
    entries = []
    for entry in load_scenarios_index():
        scenario_path = REPO_ROOT / "mock-data" / entry["file"]
        scenario = parse_scenario(scenario_path)
        entries.append(
            {
                **entry,
                "name": scenario.name,
                "path": scenario_path,
                "job_id": scenario.job_id,
                "customer": scenario.customer,
                "outcome": scenario.expected_outcome,
            }
        )
    return entries


@lru_cache(maxsize=None)
def _load_processed_scenario(path_text: str):
    scenario_path = Path(path_text)
    timings: dict[str, float] = {}

    started = perf_counter()

    checkpoint = perf_counter()
    scenario_path.read_text()
    timings["scenario_file_loading"] = perf_counter() - checkpoint

    checkpoint = perf_counter()
    scenario = parse_scenario(scenario_path)
    timings["parse_scenario"] = perf_counter() - checkpoint

    checkpoint = perf_counter()
    events = normalize_events(scenario.logs)
    timings["normalize_events"] = perf_counter() - checkpoint

    checkpoint = perf_counter()
    timeline = build_timeline(events)
    timings["build_timeline"] = perf_counter() - checkpoint

    checkpoint = perf_counter()
    context = build_investigation_context(
        customer=scenario.customer,
        job_id=scenario.job_id,
        outcome=scenario.expected_outcome,
        expected_failure_signature=scenario.expected_failure_signature,
        timeline=timeline,
        events=events,
    )
    timings["build_investigation_context"] = perf_counter() - checkpoint
    timings["total_page_execution"] = perf_counter() - started

    return scenario, timeline, context, timings


def load_processed_scenario(path: Path):
    scenario, timeline, context, timings = _load_processed_scenario(str(path))
    return scenario, timeline, context, dict(timings)


def find_job_entry(entries: list[dict], job_query: str) -> Optional[dict]:
    query = job_query.strip().rstrip("🔍").strip().lower()
    if not query:
        return None
    for entry in entries:
        if entry["job_id"].lower() == query:
            return entry
    return None


def confidence_label(confidence: float) -> str:
    if confidence >= 0.75:
        return "High"
    if confidence >= 0.4:
        return "Medium"
    return "Low"


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


CONFIDENCE_LABELS = {
    "causal_signal": "Exact causal signal",
    "corroboration": "Cross-component corroboration",
    "temporal_coherence": "Temporal coherence",
    "consistency": "Evidence consistency",
    "ambiguity_penalty": "Ambiguity penalty",
}


def format_contribution(value: float) -> str:
    if value > 0:
        return f"+{value:.2f}"
    if value < 0:
        return f"{value:.2f}"
    return "0.00"


def confidence_breakdown_rows(breakdown: dict[str, float]) -> list[dict[str, str]]:
    rows = []
    for key in (
        "causal_signal",
        "corroboration",
        "temporal_coherence",
        "consistency",
        "ambiguity_penalty",
    ):
        rows.append(
            {
                "Evidence dimension": CONFIDENCE_LABELS[key],
                "Contribution": format_contribution(breakdown.get(key, 0.0)),
            }
        )
    return rows


def confidence_debug_rows(breakdown: dict[str, float]) -> list[dict[str, str]]:
    rows = []
    running_total = 0.0
    for key in ("causal_signal", "corroboration", "temporal_coherence", "consistency", "ambiguity_penalty"):
        value = breakdown.get(key, 0.0)
        running_total += value
        rows.append(
            {
                "Component": key,
                "Value": f"{value:.2f}",
                "Running total": f"{running_total:.2f}",
            }
        )
    return rows


def final_score_text(confidence: float) -> str:
    return f"Final Investigation Confidence: {confidence:.0%}"


def deterministic_mode_message(analysis_mode: str, used_fallback: bool) -> tuple[str, str]:
    if analysis_mode == "Simulated":
        return (
            "info",
            "Deterministic Investigation Mode\n\n"
            "This investigation was generated entirely from the evidence timeline using deterministic reasoning. "
            "Live AI analysis is available when Live OpenAI mode is enabled.",
        )
    if used_fallback:
        return (
            "warning",
            "Evidence-Based Investigation\n\n"
            "This investigation was completed using deterministic evidence analysis after live AI analysis was unavailable.",
        )
    return ("", "")


def infer_operation(job_id: str) -> str:
    text = job_id.lower()
    if "restore" in text:
        return "Recovery Restore"
    if "weekly" in text:
        return "Scheduled Full Backup"
    if "nightly" in text:
        return "Scheduled Nightly Backup"
    return "Scheduled Backup"


def infer_target(timeline) -> str:
    patterns = [
        r"storage account ([a-zA-Z0-9\-_]+)",
        r"on ([a-zA-Z0-9\-_]+-bucket)",
        r"for ([a-zA-Z0-9\-_]+/[a-zA-Z0-9\-_\.]+)",
    ]
    for entry in timeline.entries:
        message = entry.event.message
        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                return match.group(1)
    return "Primary backup target"


def format_component_source(component: str) -> str:
    return component.replace("_", " ").title()


def normalize_component_key(component: str) -> str:
    return component.replace("_", "-").strip().lower()


def format_component_name(component_id: Optional[str]) -> str:
    if not component_id:
        return ""
    normalized = component_id.replace("comp-", "").replace("-", " ").replace("_", " ").strip()
    return normalized.title()


def _evidence_timestamp_display(item) -> str:
    if not item.timestamp or item.timestamp_source not in {"available", "parsed"}:
        return "Timestamp unavailable"
    return format_event_timestamp(item.timestamp)


def split_root_cause_text(text: str) -> tuple[str, Optional[str]]:
    marker = " Most likely causal event: "
    if marker not in text:
        return text, None
    summary_text, event_text = text.split(marker, 1)
    return summary_text.strip(), event_text.strip().rstrip(".")


def summarize_causal_event(event_text: Optional[str]) -> Optional[str]:
    if not event_text:
        return None
    match = re.match(r"(?P<component>\w+)\s+(?P<code>[A-Z0-9_]+)\s+at\s+(?P<timestamp>[^.]+)", event_text)
    if not match:
        return event_text
    component = match.group("component").replace("_", " ")
    code = match.group("code")
    return f"{component} returned {code}."


def parse_causal_event_identity(event_text: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if not event_text:
        return None, None
    match = re.match(r"(?P<component>\w+)\s+(?P<code>[A-Z0-9_]+)\s+at\s+(?P<timestamp>[^.]+)", event_text)
    if not match:
        return None, None
    return match.group("component"), match.group("code")


def terminal_outcome_sequence(timeline) -> Optional[int]:
    terminal_codes = {"JOB_FAILED", "RETRY_EXHAUSTED", "PERMISSION_DENIED", "PARTIAL_SUCCESS"}
    sequence = None
    for entry in timeline.entries:
        message = entry.event.message.lower()
        if entry.event.error_code in terminal_codes or (
            normalize_component_key(entry.event.component) == "backup-service"
            and ("marked failed" in message or "completed with" in message or "marked succeeded" in message)
        ):
            sequence = entry.sequence
    return sequence


def causal_event_sequence(timeline, causal_event_text: Optional[str]) -> Optional[int]:
    component, code = parse_causal_event_identity(causal_event_text)
    if component is None and code is None:
        return None
    for entry in timeline.entries:
        if component is not None and normalize_component_key(entry.event.component) != normalize_component_key(component):
            continue
        if code is not None and entry.event.error_code != code:
            continue
        return entry.sequence
    return None


def to_iso_utc(timestamp_text: str) -> str:
    parsed = datetime.fromisoformat(timestamp_text.replace("Z", "+00:00"))
    utc_time = parsed.astimezone(timezone.utc)
    return utc_time.isoformat().replace("+00:00", "Z")


def format_event_timestamp(
    timestamp_text: str,
    *,
    display_mode: str = "utc",
    reference_timestamp_text: Optional[str] = None,
) -> str:
    """Format an ISO-8601 timestamp for UI display.

    display_mode:
      - "utc" (default): 2026-06-28 10:03:41 UTC
      - "local": local timezone representation
      - "relative": relative duration from reference timestamp (future-ready)
    """
    parsed = datetime.fromisoformat(timestamp_text.replace("Z", "+00:00"))
    utc_time = parsed.astimezone(timezone.utc)

    if display_mode == "local":
        local_time = parsed.astimezone()
        return local_time.strftime("%Y-%m-%d %H:%M:%S %Z")

    if display_mode == "relative":
        if reference_timestamp_text is None:
            return "0s"
        reference = datetime.fromisoformat(reference_timestamp_text.replace("Z", "+00:00"))
        delta = parsed - reference
        seconds = int(abs(delta.total_seconds()))
        prefix = "+" if delta.total_seconds() >= 0 else "-"
        return f"{prefix}{seconds}s"

    return utc_time.strftime("%Y-%m-%d %H:%M:%S UTC")


def evidence_timeline_items(evidence: InvestigationEvidence) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for item in evidence.causal_chain:
        if item.role == "PRIMARY_CAUSE":
            stage = "❌ Primary Failure"
        elif item.role == "OUTCOME":
            stage = "Outcome"
        elif item.role == "CONTRIBUTING":
            stage = "⚠️ Contributing Evidence"
        elif item.role == "PROPAGATION":
            stage = "Propagation"
        else:
            stage = "Evidence"

        items.append(
            {
                "stage": stage,
                "timestamp": item.timestamp or "",
                "time": _evidence_timestamp_display(item),
                "display_summary": item.display_summary or item.summary or item.message,
                "display_detail": item.display_detail or item.summary or item.message,
                "summary": item.display_summary or item.summary or item.message,
                "detail": item.display_detail or item.summary or item.message,
                "source": format_component_source(item.component),
                "component_id": item.component,
                "component_name": format_component_name(item.source_file or item.component),
                "role": item.role,
                "event_id": item.event_id,
            }
        )
    return items


def split_recommendations(
    actions: list[str],
    *,
    root_cause_text: str,
) -> tuple[list[str], list[str]]:
    root_text = root_cause_text.lower()
    if "quota" in root_text or "capacity" in root_text:
        immediate = [
            "Increase destination storage capacity or remove old backup artifacts.",
            "Retry the failed backup after capacity has been restored.",
        ]
        preventative = [
            "Configure earlier quota alerts.",
            "Prevent backup jobs from starting when available capacity is below the required threshold.",
            "Suppress retry loops for non-transient quota failures.",
        ]
        return immediate, preventative

    deduped: list[str] = []
    seen = set()
    for action in actions:
        key = action.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(action)

    immediate: list[str] = []
    preventative: list[str] = []
    for action in deduped:
        text = action.lower()
        if any(
            token in text
            for token in (
                "retry",
                "restore",
                "increase",
                "stabilize",
                "repair",
                "re-run",
                "isolate",
                "roll back",
            )
        ):
            immediate.append(action)
        else:
            preventative.append(action)

    if not immediate and deduped:
        immediate = deduped[:1]
        preventative = deduped[1:]
    if not preventative and len(immediate) > 1:
        preventative = immediate[1:]
        immediate = immediate[:1]
    return immediate, preventative


def root_cause_reason(summary_explanations: list[str]) -> str:
    supportive_lines = [line.lstrip("✓ ").strip() for line in summary_explanations if line.strip().startswith("✓")]
    if len(supportive_lines) >= 3:
        return (
            f"{supportive_lines[0]}. {supportive_lines[1]}. {supportive_lines[2]}. "
            "No conflicting evidence was identified."
        )
    if supportive_lines:
        return ". ".join(supportive_lines) + ". No conflicting evidence was identified."
    return "Evidence is chronologically consistent and no conflicting signals were detected."


def confidence_reason_lines(evidence: InvestigationEvidence) -> tuple[list[str], str]:
    preferred = ["monitoring", "backup_service", "storage", "network", "auth", "encryption"]
    available = {normalize_component_key(item.component) for item in evidence.causal_chain}
    ordered = [component for component in preferred if normalize_component_key(component) in available]
    if not ordered:
        ordered = sorted(available)
    lines = [f"{format_component_source(component)} events" for component in ordered[:3]]
    conclusion = "All available evidence supported the same causal sequence."
    if evidence.limitations:
        conclusion += " " + evidence.limitations[0]
    if evidence.confidence_explanation and any("No competing" in line for line in evidence.confidence_explanation):
        conclusion += " No conflicting evidence was detected."
    return lines, conclusion


def recovered_scenario_sections(summary_text: str, timeline) -> dict[str, str]:
    recovery_entries = [
        entry for entry in timeline.entries
        if entry.event.level == "INFO"
        and any(
            term in entry.event.message.lower()
            for term in ("healthy", "recovered", "retry succeeded", "succeeded", "completed")
        )
    ]
    outcome_entry = timeline.entries[-1].event.message if timeline.entries else ""
    recovery_text = (
        "Retry logic detected service recovery and automatically resumed processing."
        if recovery_entries
        else "Service recovered and processing resumed automatically."
    )
    return {
        "Primary disruption": summary_text,
        "Recovery": recovery_text,
        "Successful outcome": outcome_entry,
    }


def job_option_label(entry: dict) -> str:
    return f"{entry['job_id']} — {entry['outcome']} — {entry['customer']}"


def customer_impact_text(outcome: str) -> str:
    if outcome == "FAILED":
        return "The scheduled backup did not complete successfully."
    if outcome == "RECOVERED":
        return "A transient service disruption occurred, but recovery logic completed the backup successfully."
    if outcome == "PARTIAL_SUCCESS":
        return "The backup completed with partial coverage; at least one workload did not complete successfully."
    return f"The job completed with outcome status: {outcome}."


def status_display(outcome: str) -> str:
    if outcome == "FAILED":
        return "🔴 FAILED"
    if outcome in {"RECOVERED", "SUCCEEDED", "SUCCESS"}:
        return "🟢 SUCCESS"
    if outcome == "PARTIAL_SUCCESS":
        return "🟠 PARTIAL_SUCCESS"
    return outcome


def incident_summary_styles() -> str:
    return """
    <style>
      .incident-group-header {
        font-size: 0.95rem;
        font-weight: 600;
        margin: 0.35rem 0 0.35rem 0;
      }
      .incident-field {
        margin: 0.05rem 0 0.55rem 0;
      }
      .incident-label {
        font-size: 0.76rem;
        color: #6b7280;
        font-weight: 600;
        margin-bottom: 0.08rem;
        letter-spacing: 0.01em;
      }
      .incident-value {
        font-size: 0.95rem;
        font-weight: 500;
        color: #111827;
        line-height: 1.25;
      }
      .incident-status {
        font-size: 1.02rem;
        font-weight: 700;
        color: #111827;
        line-height: 1.2;
      }
      .investigation-status-line {
        margin: 0.12rem 0;
      }
      .timeline-event {
        margin: 0.20rem 0 0.40rem 0;
        padding-bottom: 0.30rem;
        border-bottom: 1px solid rgba(107, 114, 128, 0.18);
      }
      .timeline-time {
        font-size: 0.82rem;
        color: #4b5563;
        margin-bottom: 0.02rem;
      }
      .timeline-title {
        font-size: 0.94rem;
        font-weight: 600;
        margin-bottom: 0.04rem;
      }
      .timeline-meta {
        font-size: 0.82rem;
        color: #374151;
        margin-bottom: 0.06rem;
      }
      .timeline-detail {
        font-size: 0.90rem;
        color: #111827;
      }
    </style>
    """


def render_incident_field(label: str, value: str, *, is_status: bool = False) -> None:
    value_class = "incident-status" if is_status else "incident-value"
    st.markdown(
        (
            "<div class='incident-field'>"
            f"<div class='incident-label'>{label}</div>"
            f"<div class='{value_class}'>{value}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def developer_diagnostics_payload(
    *,
    enabled: bool,
    scenario,
    selected_entry: dict,
    scenario_path: Optional[str],
    timings: dict[str, float],
    timeline,
):
    if not enabled:
        return None
    return {
        "Expected failure signature": scenario.expected_failure_signature,
        "Scenario ID": scenario.scenario_id,
        "Scenario path": scenario_path or "Uploaded logs",
        "Execution timings": timings,
        "Parsed events": [entry.event.model_dump() for entry in timeline.entries],
        "Normalized events": [entry.event.model_dump() for entry in timeline.entries],
        "Raw event payloads": [log.model_dump(by_alias=True) for log in scenario.logs],
    }


def investigation_execution_metadata(total_page_execution: float) -> dict[str, str]:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    if total_page_execution < 1:
        execution_text = f"{total_page_execution * 1000:.0f} ms"
    else:
        execution_text = f"{total_page_execution:.2f} s"
    return {
        "generated": generated,
        "execution_time": execution_text,
    }


def run_analysis(
    context,
    analysis_mode: str,
    should_generate_live: bool,
    evidence: InvestigationEvidence | None = None,
):
    timings = {
        "openai_request": 0.0,
        "fallback_generation": 0.0,
    }

    if analysis_mode == "Simulated":
        checkpoint = perf_counter()
        summary = generate_simulated_investigation_summary(context, evidence=evidence)
        timings["fallback_generation"] = perf_counter() - checkpoint
        return summary, True, None, timings

    if not should_generate_live:
        return None, False, None, timings

    checkpoint = perf_counter()
    try:
        summary = generate_llm_investigation_summary(context)
        timings["openai_request"] = perf_counter() - checkpoint
        return summary, False, None, timings
    except Exception as exc:
        timings["openai_request"] = perf_counter() - checkpoint
        fallback_started = perf_counter()
        summary = generate_simulated_investigation_summary(context, evidence=evidence)
        timings["fallback_generation"] = perf_counter() - fallback_started
        return summary, True, str(exc), timings


def active_scenario_path() -> Path:
    index = load_scenarios_index()
    active = [entry for entry in index if entry.get("status") == "active"]
    if len(active) != 1:
        raise ValueError(f"expected exactly one active scenario, found {len(active)}")
    return REPO_ROOT / "mock-data" / active[0]["file"]


def build_uploaded_scenario(filenames: list[str], logs: list[LogEntry]) -> Scenario:
    """Build a synthetic Scenario from uploaded logs."""
    # Normalize all logs to use the same correlation ID (required by timeline builder)
    # Try to use the most specific job ID if one exists, otherwise generate a unique one
    job_ids = [log.correlation_id for log in logs if log.correlation_id and log.correlation_id.startswith('job-')]
    if job_ids:
        # Use the most common job ID, or the first one found
        normalized_correlation_id = job_ids[0]
    else:
        # Generate a unique correlation ID for this investigation
        normalized_correlation_id = f"uploaded-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    
    # Update all logs to use the normalized correlation ID
    for log in logs:
        log.correlation_id = normalized_correlation_id
    
    # Compute real start/end times from log timestamps
    timestamps = []
    for log in logs:
        try:
            if isinstance(log.timestamp, str):
                # Parse ISO format timestamp
                ts = datetime.fromisoformat(log.timestamp.replace("Z", "+00:00"))
            else:
                ts = log.timestamp
            timestamps.append(ts)
        except Exception:
            pass  # Skip unparseable timestamps
    
    # Determine expected outcome based on error presence
    error_entries = [log for log in logs if log.level and log.level.upper() == "ERROR"]
    if error_entries:
        expected_outcome = "FAILED"
        failure_sig = f"User-uploaded logs: {len(error_entries)} error(s) detected"
    else:
        expected_outcome = "UNKNOWN"
        failure_sig = "User-uploaded logs; investigation driven by evidence."
    
    file_label = ", ".join(filenames) if len(filenames) <= 3 else f"{len(filenames)} files"
    
    return Scenario(
        scenario_id="uploaded",
        name=f"Uploaded Logs: {file_label}",
        category="uploaded",
        tags=["uploaded"],
        description="User-uploaded recovery logs for investigation.",
        customer="[Uploaded Investigation]",
        job_id=f"uploaded-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
        correlation_id=normalized_correlation_id,
        components_involved=list(set(log.component for log in logs if log.component)),
        expected_outcome=expected_outcome,
        expected_failure_signature=failure_sig,
        logs=logs,
    )


def main() -> None:
    st.set_page_config(page_title="Recovery Investigation Workspace", layout="wide")
    st.markdown(incident_summary_styles(), unsafe_allow_html=True)

    scenario_entries = load_scenario_entries()
    default_job_id = next(
        (entry["job_id"] for entry in scenario_entries if entry.get("status") == "active"),
        scenario_entries[0]["job_id"],
    )
    if "searched_job_id" not in st.session_state:
        st.session_state["searched_job_id"] = default_job_id

    # Initialize tab tracking in session state
    if "active_tab" not in st.session_state:
        st.session_state["active_tab"] = "demo"  # "demo" or "upload"
    if "uploaded_scenario" not in st.session_state:
        st.session_state["uploaded_scenario"] = None
    if "uploaded_scenario_name" not in st.session_state:
        st.session_state["uploaded_scenario_name"] = ""

    # Tabs update state when interacted with
    tab_demo, tab_upload = st.tabs(["Demo Scenario", "Upload Your Logs"])

    scenario = None
    scenario_path = None
    upload_error = None

    with tab_demo:
        # User is on demo tab — mark it as active
        st.session_state["active_tab"] = "demo"
        # Clear any previously uploaded scenario when visiting demo tab
        st.session_state["uploaded_scenario"] = None

        st.subheader("Find Recovery Job")
        st.session_state["searched_job_id"] = st.text_input(
            "Job ID",
            value=st.session_state["searched_job_id"],
            placeholder="search for job id or name",
        ).strip()

        selected_entry = find_job_entry(scenario_entries, st.session_state["searched_job_id"])
        if selected_entry is None:
            st.error("No recovery job found for that Job ID.")
        else:
            scenario_path = selected_entry["path"]

    with tab_upload:
        # User is on upload tab — mark it as active and clear demo
        st.session_state["active_tab"] = "upload"
        scenario_path = None  # Force clear demo scenario on this tab

        st.warning(
            "⚠️ **Do not upload confidential production logs, credentials, access tokens, personal data, or customer-identifying information.** "
            "This public prototype is intended for sanitized test data only."
        )

        uploaded_files = st.file_uploader(
            "Upload your logs (.log, .txt, .json)",
            type=["log", "txt", "json"],
            accept_multiple_files=True,
            key="log_uploader",
        )

        if uploaded_files:
            # Process all files
            merged_logs, file_results = process_multiple_uploads(uploaded_files)

            # Show per-file summary
            with st.expander("📋 File Processing Summary", expanded=True):
                for filename, result in file_results.items():
                    st.markdown(f"- {filename}: {result}")

            # Check if any files succeeded
            if not merged_logs:
                st.error("❌ No events could be extracted from any uploaded file.")
                upload_error = "All files failed to parse"
                st.session_state["uploaded_scenario"] = None
            else:
                # Build synthetic scenario from merged logs
                filenames = list(file_results.keys())
                scenario = build_uploaded_scenario(filenames, merged_logs)
                st.session_state["uploaded_scenario"] = scenario
                st.session_state["uploaded_scenario_name"] = f"{len(filenames)} file(s)"

                # Show success with file count
                total_events = len(merged_logs)
                successful_files = sum(
                    1 for r in file_results.values() if r.startswith("✅")
                )
                st.success(
                    f"✅ Loaded {total_events} total events from {successful_files} file(s)"
                )
        else:
            # No files uploaded, check if we have a saved upload from session state
            if st.session_state["uploaded_scenario"] is not None:
                scenario = st.session_state["uploaded_scenario"]
                st.info(f"ℹ️ Using previously uploaded: {st.session_state['uploaded_scenario_name']}")
            else:
                st.info("📤 Select files to upload to begin investigation.")

    # Route to the correct scenario based on which tab is active
    if st.session_state["active_tab"] == "upload":
        # Use uploaded scenario
        scenario = st.session_state.get("uploaded_scenario")
        scenario_path = None
    else:
        # Use demo scenario
        scenario = None
        # scenario_path already set by demo tab logic above

    # If no scenario selected in either tab, show error and return
    if scenario_path is None and scenario is None:
        if upload_error:
            # Already showed error in upload tab
            pass
        else:
            st.info("Select a demo job or upload logs to begin investigation.")
        return

    # Load or use the uploaded scenario
    if scenario_path is not None:
        scenario, timeline, context, processing_timings = load_processed_scenario(scenario_path)
    else:
        # For uploaded scenario, build timeline and context
        try:
            timings = {}
            checkpoint = perf_counter()
            events = normalize_events(scenario.logs)
            timings["normalize_events"] = perf_counter() - checkpoint

            checkpoint = perf_counter()
            timeline = build_timeline(events)
            timings["build_timeline"] = perf_counter() - checkpoint

            checkpoint = perf_counter()
            context = build_investigation_context(
                customer=scenario.customer,
                job_id=scenario.job_id,
                outcome=scenario.expected_outcome,
                expected_failure_signature=scenario.expected_failure_signature,
                timeline=timeline,
                events=events,
            )
            timings["build_investigation_context"] = perf_counter() - checkpoint
            processing_timings = dict(timings)
            processing_timings["total_page_execution"] = sum(timings.values())
        except Exception as e:
            st.error(f"Failed to process uploaded logs: {str(e)}")
            return

    evidence = select_investigation_evidence(context)

    show_developer_diagnostics = st.sidebar.checkbox(
        "Show developer diagnostics",
        value=False,
    )
    analysis_mode = "Simulated"
    should_generate_live = False
    if show_developer_diagnostics:
        st.sidebar.subheader("Developer controls")
        analysis_mode = st.sidebar.radio(
            "Analysis mode",
            ("Simulated", "Live OpenAI"),
            index=0,
        )
        if analysis_mode == "Live OpenAI":
            should_generate_live = st.sidebar.button("Generate AI Investigation", type="primary")

    if analysis_mode == "Live OpenAI" and should_generate_live:
        with st.spinner("Generating AI investigation..."):
            summary, used_fallback, analysis_error, analysis_timings = run_analysis(
                context,
                analysis_mode=analysis_mode,
                should_generate_live=should_generate_live,
                evidence=evidence,
            )
    else:
        summary, used_fallback, analysis_error, analysis_timings = run_analysis(
            context,
            analysis_mode=analysis_mode,
            should_generate_live=should_generate_live,
            evidence=evidence,
        )

    total_page_execution = (
        processing_timings["total_page_execution"]
        + analysis_timings["openai_request"]
        + analysis_timings["fallback_generation"]
    )

    st.title("Recovery Decision Investigator")

    report_view = None
    if summary is not None:
        report_view = build_investigation_report_view(evidence, summary, scenario, timeline)

    st.subheader("Incident Summary")
    incident_summary = report_view.incident_summary if report_view is not None else None

    st.markdown("<div class='incident-group-header'>Recovery Job</div>", unsafe_allow_html=True)
    identity_col1, identity_col2, identity_col3 = st.columns(3)
    with identity_col1:
        render_incident_field("Customer", incident_summary.customer if incident_summary is not None else scenario.customer)
    with identity_col2:
        render_incident_field("Job ID", incident_summary.job_id if incident_summary is not None else scenario.job_id)
    with identity_col3:
        render_incident_field("Status", status_display(incident_summary.status if incident_summary is not None else scenario.expected_outcome), is_status=True)

    st.markdown("<div class='incident-group-header'>Operation</div>", unsafe_allow_html=True)
    operation_col1, operation_col2, operation_col3 = st.columns(3)
    with operation_col1:
        render_incident_field("Operation", incident_summary.operation if incident_summary is not None else "Unknown")
    with operation_col2:
        render_incident_field("Target", incident_summary.target if incident_summary is not None else "Not provided")
    with operation_col3:
        render_incident_field("Environment", incident_summary.environment if incident_summary is not None else "Not provided")

    st.markdown("<div class='incident-group-header'>Timeline</div>", unsafe_allow_html=True)
    timeline_col1, timeline_col2, timeline_col3 = st.columns(3)
    with timeline_col1:
        render_incident_field("Started", incident_summary.started if incident_summary is not None else "Timestamp unavailable")
    with timeline_col2:
        render_incident_field(
            incident_summary.ended_label if incident_summary is not None else "Ended At",
            incident_summary.ended if incident_summary is not None else "Timestamp unavailable",
        )
    with timeline_col3:
        render_incident_field("Duration", incident_summary.duration if incident_summary is not None else "Unknown")

    st.subheader("Investigation Status")
    st.markdown("🟢 Completed")
    st.markdown("<div class='investigation-status-line'>✓ Metadata</div>", unsafe_allow_html=True)
    st.markdown("<div class='investigation-status-line'>✓ Backup Events</div>", unsafe_allow_html=True)
    st.markdown("<div class='investigation-status-line'>✓ Storage Events</div>", unsafe_allow_html=True)
    st.markdown("<div class='investigation-status-line'>✓ Monitoring Alerts</div>", unsafe_allow_html=True)
    st.markdown("<div class='investigation-status-line'>✓ Timeline</div>", unsafe_allow_html=True)
    st.markdown("<div class='investigation-status-line'>✓ Report Generated</div>", unsafe_allow_html=True)
    execution_meta = investigation_execution_metadata(total_page_execution)
    meta_col1, meta_col2 = st.columns(2)
    with meta_col1:
        st.caption("Generated")
        st.caption(execution_meta["generated"])
    with meta_col2:
        st.caption("Execution Time")
        st.caption(execution_meta["execution_time"])

    if report_view is not None:
        st.subheader("Investigation Timeline")
        for item in report_view.timeline_items:
            metadata_line = item.source
            if item.component_name and item.component_name != item.source:
                metadata_line = f"{metadata_line} • {item.component_name}"
            st.markdown(
                (
                    "<div class='timeline-event'>"
                    f"<div class='timeline-time'>{item.time}</div>"
                    f"<div class='timeline-title'>{item.title}</div>"
                    f"<div class='timeline-meta'>{metadata_line}</div>"
                    f"<div class='timeline-detail'>{item.display_detail}</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
            if item.detail_items:
                with st.expander(f"Supporting evidence ({item.supporting_evidence_count})", expanded=False):
                    for detail in item.detail_items:
                        source_bits = [detail.source]
                        if detail.source_document_id:
                            source_bits.append(detail.source_document_id)
                        st.write(f"{detail.event_id}. {detail.display_summary}")
                        st.caption(f"{detail.time} • {' • '.join(source_bits)}")
                        if detail.display_detail and detail.display_detail != detail.display_summary:
                            st.caption(detail.display_detail)

        st.subheader("Investigation Findings")
        st.markdown("**Observed Failure**")
        st.write(report_view.observed_failure or report_view.root_cause)
        st.markdown("**Inferred Explanation**")
        st.write(report_view.inferred_explanation or report_view.root_cause)
        if report_view.primary_causal_event is not None:
            st.markdown("**Primary Causal Event**")
            primary_text = report_view.primary_causal_label or report_view.primary_causal_event.display_summary or report_view.primary_causal_event.message
            st.write(primary_text)
        st.markdown("**Customer Impact**")
        st.write(report_view.customer_impact)
        if report_view.outcome_summary:
            st.markdown("**Outcome Summary**")
            st.write(report_view.outcome_summary)
        if report_view.evidence_gaps:
            st.markdown("**Evidence Gaps**")
            for gap in report_view.evidence_gaps:
                st.write(f"• {gap}")

        st.subheader("Recommended Actions")
        if report_view.recommended_actions:
            action_categories = [
                "Immediate investigation",
                "Immediate mitigation",
                "Preventive improvement",
            ]
            for category in action_categories:
                grouped_actions = [item for item in report_view.recommended_actions if item.category == category]
                if not grouped_actions:
                    continue
                st.markdown(f"**{category}**")
                for i, action in enumerate(grouped_actions, 1):
                    st.write(f"{i}. {action.action}")
                    st.caption(action.rationale)
        else:
            st.markdown("**Immediate Actions**")
            for i, action in enumerate(report_view.immediate_actions, 1):
                st.write(f"{i}. {action}")
            if report_view.preventive_actions:
                st.markdown("**Prevent Future Failures**")
                for i, action in enumerate(report_view.preventive_actions, 1):
                    st.write(f"{i}. {action}")

        st.subheader("Investigation Confidence")
        conf_col1, conf_col2 = st.columns(2)
        conf_col1.metric("Investigation Confidence", f"{report_view.confidence:.0%}")
        conf_col2.metric("Level", report_view.confidence_level)
        st.markdown("**Supporting Evidence**")
        for item in report_view.supporting_evidence:
            if item.source_document_id:
                st.write(f"• {item.display_summary} [{item.event_id}; {item.source_document_id}]")
            else:
                st.write(f"• {item.display_summary} [{item.event_id}]")
        for line in report_view.confidence_explanation:
            st.write(line)
        if report_view.confidence_dimensions:
            st.markdown("**Confidence Dimensions**")
            for dimension in report_view.confidence_dimensions:
                st.write(f"• {dimension.dimension}: {dimension.level}")
                st.caption(dimension.rationale)

        if show_developer_diagnostics:
            with st.expander("Confidence component debug", expanded=False):
                st.write("Component values before clamping and final formatting:")
                st.dataframe(confidence_debug_rows(evidence.confidence_breakdown), use_container_width=True, hide_index=True)
                st.write(f"Raw summed confidence: {sum(evidence.confidence_breakdown.values()):.2f}")

    diagnostics_payload = developer_diagnostics_payload(
        enabled=show_developer_diagnostics,
        scenario=scenario,
        selected_entry=selected_entry,
        scenario_path=str(scenario_path) if scenario_path is not None else None,
        timings={
            **processing_timings,
            "openai_request": analysis_timings["openai_request"],
            "fallback_generation": analysis_timings["fallback_generation"],
            "total_page_execution": total_page_execution,
        },
        timeline=timeline,
    )
    if diagnostics_payload is not None:
        with st.expander("Developer diagnostics", expanded=False):
            if analysis_mode == "Live OpenAI" and not should_generate_live:
                st.info("Click Generate AI Investigation in the sidebar to run live analysis.")
            if analysis_error is not None:
                st.error(f"Live OpenAI analysis failed: {analysis_error}")
            st.write(f"Scenario ID: {diagnostics_payload['Scenario ID']}")
            st.write(f"Scenario path: {diagnostics_payload['Scenario path']}")
            st.subheader("Expected failure signature")
            st.write(diagnostics_payload["Expected failure signature"])
            st.subheader("Execution timings")
            st.write(diagnostics_payload["Execution timings"])
            st.subheader("Parsed events")
            st.write(diagnostics_payload["Parsed events"])
            st.subheader("Normalized events")
            st.write(diagnostics_payload["Normalized events"])
            st.subheader("Raw event payloads")
            st.write(diagnostics_payload["Raw event payloads"])
            if summary is not None:
                st.subheader("Confidence diagnostics")
                for item in summary.confidence_explanation:
                    st.write(item)
                st.dataframe(
                    confidence_breakdown_rows(summary.confidence_breakdown),
                    use_container_width=True,
                    hide_index=True,
                )
                st.write(final_score_text(summary.confidence))


if __name__ == "__main__":
    main()
