from time import perf_counter

from recovery_workspace.app import (
    active_scenario_path,
    confidence_breakdown_rows,
    deterministic_mode_message,
    developer_diagnostics_payload,
    evidence_timeline_items,
    final_score_text,
    format_event_timestamp,
    format_contribution,
    format_duration,
    load_processed_scenario,
    load_scenario_entries,
    load_scenarios_index,
    run_analysis,
    select_investigation_evidence,
    split_root_cause_text,
    summarize_causal_event,
)
from recovery_workspace.app import build_uploaded_scenario
from recovery_workspace.events import normalize_events
from recovery_workspace.investigation import build_investigation_context
from recovery_workspace.parser import parse_scenario
from recovery_workspace.timeline import build_timeline
from recovery_workspace.uploader import parse_uploaded_logs_with_source
from recovery_workspace.models import InvestigationSummary


def test_active_scenario_path_resolves_to_storage_quota_exceeded():
    path = active_scenario_path()

    assert path.name == "storage-quota-exceeded.json"
    assert path.exists()


def test_load_scenarios_index_contains_expected_scenarios():
    scenarios = load_scenarios_index()
    scenario_ids = {entry["scenarioId"] for entry in scenarios}

    expected_ids = {
        "storage-quota-exceeded",
        "permission-denied",
        "network-timeout",
        "service-unavailable",
        "clock-skew",
    }

    assert expected_ids == scenario_ids
    assert len(scenarios) == 5


def test_load_scenario_entries_adds_display_names():
    entries = load_scenario_entries()

    assert len(entries) == 5
    assert all(entry["name"] for entry in entries)
    assert all(entry["path"].exists() for entry in entries)


def test_simulated_mode_never_calls_openai(monkeypatch):
    entry = load_scenario_entries()[0]
    _, _, context, _ = load_processed_scenario(entry["path"])

    def fail_if_called(*args, **kwargs):
        raise AssertionError("OpenAI should not be called in simulated mode")

    monkeypatch.setattr("recovery_workspace.app.generate_llm_investigation_summary", fail_if_called)

    summary, used_fallback, error, timings = run_analysis(
        context,
        analysis_mode="Simulated",
        should_generate_live=False,
    )

    assert used_fallback is True
    assert error is None
    assert summary is not None
    assert timings["openai_request"] == 0.0


def test_switching_scenarios_in_simulated_mode_returns_immediately():
    entries = load_scenario_entries()[:2]

    started = perf_counter()
    for entry in entries:
        _, _, context, _ = load_processed_scenario(entry["path"])
        summary, used_fallback, error, _ = run_analysis(
            context,
            analysis_mode="Simulated",
            should_generate_live=False,
        )
        assert summary is not None
        assert used_fallback is True
        assert error is None
    elapsed = perf_counter() - started

    assert elapsed < 1.0


def test_live_mode_calls_openai_only_after_button_action(monkeypatch):
    entry = load_scenario_entries()[0]
    _, _, context, _ = load_processed_scenario(entry["path"])
    calls = {"count": 0}

    def fake_live_summary(*args, **kwargs):
        calls["count"] += 1
        return InvestigationSummary(
            likely_root_cause="Live result",
            supporting_evidence=["Evidence"],
            next_actions=["Action"],
            confidence=0.6,
        )

    monkeypatch.setattr("recovery_workspace.app.generate_llm_investigation_summary", fake_live_summary)

    summary, used_fallback, error, _ = run_analysis(
        context,
        analysis_mode="Live OpenAI",
        should_generate_live=False,
    )
    assert summary is None
    assert used_fallback is False
    assert error is None
    assert calls["count"] == 0

    summary, used_fallback, error, _ = run_analysis(
        context,
        analysis_mode="Live OpenAI",
        should_generate_live=True,
    )
    assert summary is not None
    assert used_fallback is False
    assert error is None
    assert calls["count"] == 1


def test_confidence_breakdown_rows_are_formatted_for_display():
    rows = confidence_breakdown_rows(
        {
            "causal_signal": 0.30,
            "corroboration": 0.22,
            "temporal_coherence": 0.20,
            "consistency": 0.12,
            "ambiguity_penalty": -0.05,
        }
    )

    assert rows == [
        {"Evidence dimension": "Exact causal signal", "Contribution": "+0.30"},
        {"Evidence dimension": "Cross-component corroboration", "Contribution": "+0.22"},
        {"Evidence dimension": "Temporal coherence", "Contribution": "+0.20"},
        {"Evidence dimension": "Evidence consistency", "Contribution": "+0.12"},
        {"Evidence dimension": "Ambiguity penalty", "Contribution": "-0.05"},
    ]


def test_format_contribution_includes_signs_and_zero_format():
    assert format_contribution(0.3) == "+0.30"
    assert format_contribution(-0.05) == "-0.05"
    assert format_contribution(0.0) == "0.00"


def test_final_score_text_uses_actual_confidence_value():
    assert final_score_text(0.84) == "Final Investigation Confidence: 84%"


def test_format_duration_returns_human_readable_values():
    assert format_duration(21800) == "6h 03m"
    assert format_duration(476) == "7m 56s"
    assert format_duration(583) == "9m 43s"
    assert format_duration(228) == "3m 48s"


def test_deterministic_mode_message_presents_resilience():
    level, text = deterministic_mode_message("Simulated", True)

    assert level == "info"
    assert "Deterministic Investigation Mode" in text
    assert "evidence timeline" in text


def test_root_cause_text_is_split_for_primary_causal_event_display():
    summary_text, causal_event = split_root_cause_text(
        "Destination storage capacity was exhausted. Most likely causal event: backup_service QUOTA_EXCEEDED at 2026-01-01T00:00:00Z."
    )

    assert summary_text == "Destination storage capacity was exhausted."
    assert causal_event == "backup_service QUOTA_EXCEEDED at 2026-01-01T00:00:00Z"


def test_primary_causal_event_display_is_humanized():
    assert (
        summarize_causal_event("backup_service QUOTA_EXCEEDED at 2026-01-01T00:00:00Z")
        == "backup service returned QUOTA_EXCEEDED."
    )


def test_evidence_timeline_items_are_chronological_and_unique():
    entry = load_scenario_entries()[0]
    _, timeline, context, _ = load_processed_scenario(entry["path"])
    evidence = select_investigation_evidence(context)
    items = evidence_timeline_items(evidence)

    labels = [item["stage"] for item in items]
    assert len(items) == len({item["event_id"] for item in items})
    assert any("Primary Failure" in label for label in labels)
    assert any("Outcome" in label for label in labels)
    assert {item["event_id"] for item in items} == {item.event_id for item in evidence.causal_chain}


def test_shared_evidence_model_preserves_lock_refresh_chain():
    content = """2026-08-02T14:00:00Z backup-service backup-service INFO repository opened successfully
source file could not be accessed because another process was using it
2026-08-02T14:00:02Z backup-service backup-service ERROR failed to refresh lock in time
2026-08-02T14:00:02Z backup-service backup-service ERROR failed to refresh lock in time
progress 50% complete
2026-08-02T14:00:03Z backup-service backup-service ERROR unable to save snapshot: context canceled
"""
    logs, error = parse_uploaded_logs_with_source(content, "backup-service.log")
    assert error == ""
    scenario = build_uploaded_scenario(["backup-service.log"], logs)
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
    summary, _, _, _ = run_analysis(
        context,
        analysis_mode="Simulated",
        should_generate_live=False,
        evidence=evidence,
    )

    assert summary.evidence is not None
    timeline_ids = [item["event_id"] for item in evidence_timeline_items(evidence)]
    finding_ids = [item.event_id for item in summary.evidence.causal_chain]
    assert summary.evidence.primary_causal_event is not None
    assert summary.evidence.primary_causal_event.role == "PRIMARY_CAUSE"
    assert summary.evidence.primary_causal_event.event_id in timeline_ids
    assert timeline_ids == finding_ids
    assert len(finding_ids) == len(set(finding_ids))
    assert any(item["time"] == "Timestamp unavailable" for item in evidence_timeline_items(evidence))
    assert all("progress" not in item["summary"].lower() for item in evidence_timeline_items(evidence))
    assert summary.evidence.primary_causal_event.summary is not None


def test_evidence_timeline_uses_human_readable_display_fields():
    content = """2026-08-02T14:00:00Z backupservice backupservice INFO repository opened successfully
source file could not be accessed because another process was using it
2026-08-02T14:00:02Z backupservice backupservice ERROR failed to refresh lock in time
2026-08-02T14:00:03Z backupservice backupservice ERROR unable to save snapshot: context canceled
"""
    logs, error = parse_uploaded_logs_with_source(content, "backupservice.log")
    assert error == ""
    scenario = build_uploaded_scenario(["backupservice.log"], logs)
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
    items = evidence_timeline_items(evidence)

    joined_text = " ".join(
        f"{item['display_summary']} {item['display_detail']} {item['detail']}" for item in items
    ).upper()
    assert "PLAINTEXT_FATAL" not in joined_text
    assert "RAW_UNPARSED" not in joined_text
    assert "NO_CODE" not in joined_text
    assert len({item["display_summary"] for item in items}) == len(items)
    assert items[-2]["display_summary"] != items[-1]["display_summary"]
    assert all(item["display_detail"] for item in items)


def test_format_event_timestamp_returns_full_utc_timestamp():
    assert format_event_timestamp("2026-06-28T10:03:12+00:00") == "2026-06-28 10:03:12 UTC"


def test_format_event_timestamp_supports_relative_mode():
    formatted = format_event_timestamp(
        "2026-06-28T10:03:22+00:00",
        display_mode="relative",
        reference_timestamp_text="2026-06-28T10:03:12+00:00",
    )
    assert formatted == "+10s"


def test_developer_diagnostics_hidden_by_default():
    entry = load_scenario_entries()[0]
    scenario, timeline, _, _ = load_processed_scenario(entry["path"])

    payload = developer_diagnostics_payload(
        enabled=False,
        scenario=scenario,
        selected_entry=entry,
        scenario_path=str(entry["path"]),
        timings={"total_page_execution": 0.1},
        timeline=timeline,
    )

    assert payload is None


def test_developer_diagnostics_include_expected_failure_signature_when_enabled():
    entry = load_scenario_entries()[0]
    scenario, timeline, _, _ = load_processed_scenario(entry["path"])

    payload = developer_diagnostics_payload(
        enabled=True,
        scenario=scenario,
        selected_entry=entry,
        scenario_path=str(entry["path"]),
        timings={"total_page_execution": 0.1},
        timeline=timeline,
    )

    assert payload is not None
    assert payload["Expected failure signature"] == scenario.expected_failure_signature
    assert payload["Scenario ID"] == scenario.scenario_id
    assert "Execution timings" in payload
    assert any("message" in event and event["message"] for event in payload["Raw event payloads"])
