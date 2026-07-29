# Recovery Decision Investigator

AI-assisted investigation for enterprise backup and recovery incidents.

Recovery Decision Investigator is a deterministic-first investigation tool that reconstructs the evidence chain for a recovery job, identifies the most likely root cause, and recommends next actions with an explainable confidence score.

It is designed for incident-response workflows where reliability, transparency, and reproducibility matter more than a black-box model.

## What the project does

Given a recovery job or incident scenario, the app:

- parses raw event and log data
- normalizes it into a common schema
- reconstructs a chronological evidence timeline
- classifies likely root causes
- scores confidence based on evidence quality
- recommends next actions
- optionally enriches the result with an LLM-generated narrative

This is not an automated remediation system. It investigates and recommends, but it does not execute fixes.

## Implemented phases

### Phase 1 — Evidence ingestion and normalization
Implemented in the core parsing layer.

- ingest backup, storage, and monitoring-style events
- parse JSON-based synthetic scenarios from the mock-data directory
- normalize timestamps, identifiers, and error fields into a shared model
- validate incoming log payloads and scenario structure

### Phase 2 — Deterministic investigation engine
Implemented in the investigation workflow.

- reconstruct an evidence timeline from parsed events
- detect warning and error patterns across components
- identify likely causal events and terminal outcomes
- score confidence using evidence strength, corroboration, temporal coherence, and consistency
- generate supporting evidence and next-action recommendations

### Phase 3 — Interactive investigation experience
Implemented as a Streamlit application.

- search and review incident scenarios by job ID
- inspect investigation summaries, evidence timelines, and findings
- review recommended actions and confidence explanations
- load demo scenarios without any external dependencies

### Phase 4 — Upload and live AI support
Implemented in the current repository state.

- upload custom log files for ad-hoc analysis
- validate uploaded JSON or text-based content safely in memory
- optionally enable Live OpenAI mode for richer narrative summaries
- support deterministic fallback when live AI is unavailable

### Phase 5 — Deployment readiness
Implemented for local use and Streamlit Cloud-style deployment.

- local development workflow with Python and Streamlit
- configuration for Streamlit app settings
- environment and secrets handling for optional OpenAI usage
- deployment documentation for cloud-based hosting

## Architecture overview

```text
Recovery Job / Scenario
    │
    ▼
Evidence Ingestion
    │
    ▼
Event Parsing and Normalization
    │
    ▼
Timeline Reconstruction
    │
    ▼
Deterministic Investigation
    │
    ▼
(Optional Live AI Explanation)
    │
    ▼
Investigation Report
```

## Key features

- deterministic-first analysis
- evidence-based confidence scoring
- explainable root-cause reasoning
- support for synthetic and uploaded scenarios
- optional LLM-generated summaries
- Streamlit-based user interface

## Running locally

```bash
cd /path/to/workrepo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run recovery_workspace/app.py
```

## Testing

```bash
pytest -q
```

## Current limitations

This project is under active development and has known gaps, surfaced through testing against real tool-generated logs (not synthetic mocks) rather than assumed. Documenting them here rather than only in commit history:

- **Root-cause categorization and evidence scoring currently run through separate code paths.** In several cases the human-readable root-cause explanation correctly identified a failure category while the confidence score did not reflect it (or vice versa). Work is in progress to unify these into a single classification step.
- **Failure-pattern detection is template-based and not fully generalized.** Detection currently relies on a hardcoded set of symptom phrases.
