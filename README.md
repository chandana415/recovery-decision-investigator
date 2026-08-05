# Recovery Decision Investigator

An AI-assisted investigation platform for enterprise backup and recovery incidents.

Recovery Decision Investigator reconstructs operational evidence, identifies the most likely causal event, explains what the available evidence supports (and what it does not), and recommends evidence-based next actions with an explainable confidence score.

Unlike traditional log summarizers, the system separates:

- Observed failures
- Inferred explanations
- Evidence gaps
- Investigation confidence

This enables engineers to understand not only what likely happened, but also the limits of the available evidence before taking action.

The investigation engine is deterministic-first. AI is optional and used only to improve narrative explanations—not to replace the investigation logic.

---

# What the project does

Given uploaded operational logs or a recovery scenario, the application:

- ingests structured and unstructured operational logs
- detects document structure across multiple log formats
- normalizes heterogeneous events into a common evidence model
- reconstructs an investigation timeline
- derives semantic failure classifications
- identifies the most likely causal event
- distinguishes observed failures from inferred explanations
- highlights evidence gaps and investigation limitations
- computes an explainable confidence score
- recommends evidence-based next actions
- optionally generates an LLM-assisted investigation summary

This is **not** an automated remediation system. It investigates and recommends, but never executes corrective actions.

---

# Investigation philosophy

The platform follows a deterministic-first investigation pipeline.

Rather than asking an LLM to infer root cause directly, the application:

1. Extracts operational evidence.
2. Derives semantic meaning from heterogeneous logs.
3. Reconstructs the investigation timeline.
4. Evaluates evidence quality and confidence.
5. Identifies evidence gaps and uncertainty.
6. Produces an explainable investigation report.
7. Optionally generates an AI-assisted narrative.

This approach keeps investigations reproducible, explainable, and auditable.

---

# Architecture overview

```text
Uploaded Logs / Demo Scenario
            │
            ▼
Document Detection
            │
            ▼
Event Parsing & Normalization
            │
            ▼
Semantic Classification
            │
            ▼
Evidence Selection
            │
            ▼
Investigation Engine
            │
            ▼
Evidence Grouping
            │
            ▼
Report Builder
            │
            ▼
Streamlit UI
                 │
                 ▼
(Optional AI Narrative)
```

---

# Example investigation workflow

```text
Upload Logs
      │
      ▼
Normalize Events
      │
      ▼
Derive Semantic Meaning
      │
      ▼
Select Supporting Evidence
      │
      ▼
Reconstruct Investigation
      │
      ▼
Identify Evidence Gaps
      │
      ▼
Generate Investigation Report
```

---

# Key features

- Deterministic-first investigation engine
- Semantic normalization across heterogeneous logs
- Evidence-based causal reasoning
- Investigation timeline reconstruction
- Explicit evidence-gap reporting
- Explainable confidence scoring
- Grouped investigation timeline
- Analysis of uploaded operational logs
- Optional AI-generated investigation narrative
- Interactive Streamlit interface

---

# Running locally

```bash
cd /path/to/workrepo

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env

streamlit run recovery_workspace/app.py
```

---

# Running tests

```bash
python3 -m pytest -q
```

---

# Current limitations

The project is under active development.

Current known limitations include:

- Investigation quality depends on the evidence provided. Missing downstream service logs, trace IDs, timestamps, or cross-system telemetry reduce attribution confidence.

- Cross-service correlation currently relies on uploaded evidence rather than live integrations with observability platforms.

- Extremely noisy OCR output or heavily truncated logs may reduce semantic extraction quality.

- Live OpenAI mode enhances investigation narratives only. Core investigation logic remains deterministic and fully functional without AI.

---

# Future roadmap

Planned enhancements include:

- Live integrations with enterprise observability platforms
- Cross-service evidence correlation using distributed tracing
- Investigation graph visualization
- Multi-incident comparison
- Interactive evidence exploration
- Support for additional operational log formats