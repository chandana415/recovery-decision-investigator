from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from time import perf_counter
import re
from datetime import datetime, timezone
from typing import Optional

import streamlit as st

from recovery_workspace.events import normalize_events
from recovery_workspace.investigation import (
    build_investigation_context,
    generate_llm_investigation_summary,
    generate_simulated_investigation_summary,
)
from recovery_workspace.parser import parse_scenario
from recovery_workspace.timeline import build_timeline

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


def format_component_name(component_id: str | None) -> str:
    if not component_id:
        return ""
    normalized = component_id.replace("comp-", "").replace("-", " ").replace("_", " ").strip()
    return normalized.title()


def split_root_cause_text(text: str) -> tuple[str, str | None]:
    marker = " Most likely causal event: "
    if marker not in text:
        return text, None
    summary_text, event_text = text.split(marker, 1)
    return summary_text.strip(), event_text.strip().rstrip(".")


def summarize_causal_event(event_text: str | None) -> str | None:
    if not event_text:
        return None
    match = re.match(r"(?P<component>\w+)\s+(?P<code>[A-Z0-9_]+)\s+at\s+(?P<timestamp>[^.]+)", event_text)
    if not match:
        return event_text
    component = match.group("component").replace("_", " ")
    code = match.group("code")
    return f"{component} returned {code}."


def parse_causal_event_identity(event_text: str | None) -> tuple[str | None, str | None]:
    if not event_text:
        return None, None
    match = re.match(r"(?P<component>\w+)\s+(?P<code>[A-Z0-9_]+)\s+at\s+(?P<timestamp>[^.]+)", event_text)
    if not match:
        return None, None
    return match.group("component"), match.group("code")


def terminal_outcome_sequence(timeline) -> int | None:
    terminal_codes = {"JOB_FAILED", "RETRY_EXHAUSTED", "PERMISSION_DENIED", "PARTIAL_SUCCESS"}
    sequence = None
    for entry in timeline.entries:
        message = entry.event.message.lower()
        if entry.event.error_code in terminal_codes or (
            entry.event.component == "backup_service"
            and ("marked failed" in message or "completed with" in message or "marked succeeded" in message)
        ):
            sequence = entry.sequence
    return sequence


def causal_event_sequence(timeline, causal_event_text: str | None) -> int | None:
    component, code = parse_causal_event_identity(causal_event_text)
    if component is None and code is None:
        return None
    for entry in timeline.entries:
        if component is not None and entry.event.component != component:
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
    reference_timestamp_text: str | None = None,
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


def evidence_timeline_items(timeline, causal_event_text: str | None) -> list[dict[str, str]]:
    causal_sequence = causal_event_sequence(timeline, causal_event_text)
    outcome_sequence = terminal_outcome_sequence(timeline)
    selected_sequences: set[int] = set()

    for entry in timeline.entries:
        if entry.event.level == "WARN":
            selected_sequences.add(entry.sequence)
    if causal_sequence is not None:
        selected_sequences.add(causal_sequence)
    if outcome_sequence is not None:
        selected_sequences.add(outcome_sequence)

    if not selected_sequences and timeline.entries:
        selected_sequences.add(timeline.entries[-1].sequence)

    items: list[dict[str, str]] = []
    for entry in timeline.entries:
        if entry.sequence not in selected_sequences:
            continue
        if entry.sequence == causal_sequence:
            stage = "❌ Primary Failure"
        elif entry.sequence == outcome_sequence:
            stage = "Outcome"
        elif entry.event.level == "WARN":
            stage = "⚠️ Monitoring Warning"
        else:
            stage = "Evidence"

        code = entry.event.error_code or "NO_CODE"
        timestamp_iso = to_iso_utc(entry.event.timestamp.isoformat())
        items.append(
            {
                "stage": stage,
                "timestamp_iso": timestamp_iso,
                "time": format_event_timestamp(timestamp_iso),
                "summary": f"{entry.event.component} {code}",
                "detail": entry.event.message,
                "source": format_component_source(entry.event.component),
                "component_id": entry.event.component_id,
                "component_name": format_component_name(entry.event.component_id),
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


def confidence_reason_lines(timeline, summary_explanations: list[str]) -> tuple[list[str], str]:
    preferred = ["monitoring", "backup_service", "storage", "network", "auth", "encryption"]
    available = {entry.event.component for entry in timeline.entries}
    ordered = [component for component in preferred if component in available]
    if not ordered:
        ordered = sorted(available)
    lines = [f"{format_component_source(component)} events" for component in ordered[:3]]
    conclusion = "All available evidence supported the same causal sequence."
    if any("No competing" in line for line in summary_explanations):
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
    timings: dict[str, float],
    timeline,
):
    if not enabled:
        return None
    return {
        "Expected failure signature": scenario.expected_failure_signature,
        "Scenario ID": scenario.scenario_id,
        "Scenario path": str(selected_entry["path"]),
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
):
    timings = {
        "openai_request": 0.0,
        "fallback_generation": 0.0,
    }

    if analysis_mode == "Simulated":
        checkpoint = perf_counter()
        summary = generate_simulated_investigation_summary(context)
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
        summary = generate_simulated_investigation_summary(context)
        timings["fallback_generation"] = perf_counter() - fallback_started
        return summary, True, str(exc), timings


def active_scenario_path() -> Path:
    index = load_scenarios_index()
    active = [entry for entry in index if entry.get("status") == "active"]
    if len(active) != 1:
        raise ValueError(f"expected exactly one active scenario, found {len(active)}")
    return REPO_ROOT / "mock-data" / active[0]["file"]


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

    st.subheader("Find Recovery Job")
    st.session_state["searched_job_id"] = st.text_input(
        "Job ID",
        value=st.session_state["searched_job_id"],
        placeholder="search for job id or name",
    ).strip()

    selected_entry = find_job_entry(scenario_entries, st.session_state["searched_job_id"])
    if selected_entry is None:
        st.error("No recovery job found for that Job ID.")
        return

    scenario_path = selected_entry["path"]
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

    scenario, timeline, context, processing_timings = load_processed_scenario(scenario_path)

    if analysis_mode == "Live OpenAI" and should_generate_live:
        with st.spinner("Generating AI investigation..."):
            summary, used_fallback, analysis_error, analysis_timings = run_analysis(
                context,
                analysis_mode=analysis_mode,
                should_generate_live=should_generate_live,
            )
    else:
        summary, used_fallback, analysis_error, analysis_timings = run_analysis(
            context,
            analysis_mode=analysis_mode,
            should_generate_live=should_generate_live,
        )

    total_page_execution = (
        processing_timings["total_page_execution"]
        + analysis_timings["openai_request"]
        + analysis_timings["fallback_generation"]
    )

    st.title("Recovery Decision Investigator")

    st.subheader("Incident Summary")
    started = format_event_timestamp(to_iso_utc(timeline.started_at.isoformat()))
    completed = format_event_timestamp(to_iso_utc(timeline.ended_at.isoformat()))
    end_label = "Failed At" if scenario.expected_outcome == "FAILED" else "Ended At"

    st.markdown("<div class='incident-group-header'>Recovery Job</div>", unsafe_allow_html=True)
    identity_col1, identity_col2, identity_col3 = st.columns(3)
    with identity_col1:
        render_incident_field("Customer", scenario.customer)
    with identity_col2:
        render_incident_field("Job ID", scenario.job_id)
    with identity_col3:
        render_incident_field("Status", status_display(scenario.expected_outcome), is_status=True)

    st.markdown("<div class='incident-group-header'>Operation</div>", unsafe_allow_html=True)
    operation_col1, operation_col2, operation_col3 = st.columns(3)
    with operation_col1:
        render_incident_field("Operation", infer_operation(scenario.job_id))
    with operation_col2:
        render_incident_field("Target", infer_target(timeline))
    with operation_col3:
        render_incident_field("Environment", "Production")

    st.markdown("<div class='incident-group-header'>Timeline</div>", unsafe_allow_html=True)
    timeline_col1, timeline_col2, timeline_col3 = st.columns(3)
    with timeline_col1:
        render_incident_field("Started", started)
    with timeline_col2:
        render_incident_field(end_label, completed)
    with timeline_col3:
        render_incident_field("Duration", format_duration(timeline.duration_seconds))

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

    if summary is not None:
        summary_text, causal_event_text = split_root_cause_text(summary.likely_root_cause)
        primary_causal_event = summarize_causal_event(causal_event_text)

        st.subheader("Evidence Timeline")
        for item in evidence_timeline_items(timeline, causal_event_text):
            metadata_line = item["source"]
            if item["component_name"]:
                metadata_line = f"{metadata_line} • {item['component_name']}"
            st.markdown(
                (
                    "<div class='timeline-event'>"
                    f"<div class='timeline-time'>{item['time']}</div>"
                    f"<div class='timeline-title'>{item['stage']}</div>"
                    f"<div class='timeline-meta'>{metadata_line}</div>"
                    f"<div class='timeline-detail'>{item['detail']}</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )

        st.subheader("Investigation Findings")
        st.markdown("**Root Cause**")
        if scenario.expected_outcome == "RECOVERED":
            sections = recovered_scenario_sections(summary_text, timeline)
            st.write(sections["Primary disruption"])
            if primary_causal_event is not None:
                st.markdown("**Primary Causal Event**")
                st.write(primary_causal_event)
            st.markdown("**Customer Impact**")
            st.write(customer_impact_text(scenario.expected_outcome))
            st.markdown("**Recovery**")
            st.write(sections["Recovery"])
            st.markdown("**Successful Outcome**")
            st.write(sections["Successful outcome"])
        else:
            st.write(summary_text)
            if primary_causal_event is not None:
                st.markdown("**Primary Causal Event**")
                st.write(primary_causal_event)
            st.markdown("**Customer Impact**")
            st.write(customer_impact_text(scenario.expected_outcome))

        st.subheader("Recommended Actions")
        immediate_actions, preventative_actions = split_recommendations(
            summary.next_actions,
            root_cause_text=summary_text,
        )
        st.markdown("**Immediate Actions**")
        for i, action in enumerate(immediate_actions, 1):
            st.write(f"{i}. {action}")
        if preventative_actions:
            st.markdown("**Prevent Future Failures**")
            for i, action in enumerate(preventative_actions, 1):
                st.write(f"{i}. {action}")

        st.subheader("Investigation Confidence")
        conf_col1, conf_col2 = st.columns(2)
        conf_col1.metric("Investigation Confidence", f"{summary.confidence:.0%}")
        conf_col2.metric("Level", confidence_label(summary.confidence))
        st.markdown("**Supporting Evidence**")
        reason_lines, confidence_conclusion = confidence_reason_lines(timeline, summary.confidence_explanation)
        st.write("The investigation considered:")
        for line in reason_lines:
            st.write(f"• {line}")
        st.write(confidence_conclusion)

    diagnostics_payload = developer_diagnostics_payload(
        enabled=show_developer_diagnostics,
        scenario=scenario,
        selected_entry=selected_entry,
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
