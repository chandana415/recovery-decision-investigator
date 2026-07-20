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
    split_root_cause_text,
    summarize_causal_event,
)
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
    summary, _, _, _ = run_analysis(
        context,
        analysis_mode="Simulated",
        should_generate_live=False,
    )
    _, causal_event_text = split_root_cause_text(summary.likely_root_cause)

    items = evidence_timeline_items(timeline, causal_event_text)

    labels = [item["stage"] for item in items]
    assert len(items) == len({(item["stage"], item["time"], item["summary"]) for item in items})
    assert any("Primary Failure" in label for label in labels)
    assert any("Outcome" in label for label in labels)


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
        timings={"total_page_execution": 0.1},
        timeline=timeline,
    )

    assert payload is not None
    assert payload["Expected failure signature"] == scenario.expected_failure_signature
    assert payload["Scenario ID"] == scenario.scenario_id
    assert "Execution timings" in payload
