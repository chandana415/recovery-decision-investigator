# Mock Backup/Recovery Failure Data

Mock datasets representing realistic backup and recovery failure scenarios,
used for parallel UI/analysis development and testing without depending on
production systems.

## Week 1 scope

This repo is scoped to a **thin vertical slice** for Week 1: only
**`storage-quota-exceeded`** is the active scenario (`status: "active"` in
`scenarios/index.json`). It's the simplest complete failure signature —
3 components, a single `correlationId`, no parallel tasks or multi-stage
recovery — and is the one to build parsing/timeline/evidence logic against
first.

The other 5 scenarios remain in the repo but are marked `status: "deferred"`
in the index. They fulfill the full mock-data ticket's acceptance criteria
(5+ scenarios covering network timeout, permission denied, service
unavailable, clock skew, and partial failure) and are intended for a later
week once the Week 1 slice works end-to-end. Treat them as not-yet-in-scope
rather than deleted.

## Structure

```
mock-data/
  scenarios/
    index.json            Manifest of all scenarios (id, file, category, outcome, components)
    network-timeout.json
    storage-quota-exceeded.json
    permission-denied.json
    service-unavailable.json
    clock-skew.json
    partial-failure.json
```

Each file under `scenarios/` is **self-contained**: it includes its own
metadata and full log stream, so it can be loaded and used in a test in
isolation without joining against any other file — every log line carries
its own `component` and `componentId` inline.

## Loading / switching scenarios in tests

```js
const index = require('./mock-data/scenarios/index.json');
const scenario = require(`./mock-data/${index.find(s => s.scenarioId === 'network-timeout').file}`);
```

Or load a single scenario directly by path when a test only needs one:

```js
const scenario = require('./mock-data/scenarios/storage-quota-exceeded.json');
```

To add a new scenario: drop a new self-contained JSON file into `scenarios/`
following the schema below, then add one entry to `scenarios/index.json`.

## Scenario file schema

| Field | Description |
|---|---|
| `scenarioId` | Stable identifier, matches the filename (minus `.json`) |
| `name` | Human-readable scenario title |
| `category` | Failure-mode category (`network_timeout`, `resource_exhaustion`, `permission_error`, `service_unavailable`, `clock_skew`, `partial_failure`) |
| `description` | What the scenario represents and why it's realistic |
| `customer` / `jobId` | Fictional tenant and job identifiers for realism |
| `correlationId` | Single ID shared by every log line in the scenario, simulating distributed request tracing. `partial-failure` additionally uses per-task `traceId`s under the shared parent `correlationId` |
| `componentsInvolved` | Which of `backup_service`, `storage`, `network`, `encryption`, `auth`, `monitoring` appear in the logs (always 3+) |
| `expectedOutcome` | `FAILED`, `RECOVERED`, or `PARTIAL_SUCCESS` — what a correct parser/timeline implementation should conclude |
| `expectedFailureSignature` | One-sentence description of the log pattern a correct implementation should detect |
| `logs` | Ordered array of log entries: `timestamp`, `correlationId` (and `traceId` where applicable), `componentId`, `component`, `level` (`INFO`/`WARN`/`ERROR`), `errorCode` (structured code, `null` when not an error), `message` |

## Scenarios

| Scenario | Status | Category | Components | Outcome | Summary |
|---|---|---|---|---|---|
| `storage-quota-exceeded` | **active** | Resource Exhaustion | backup_service, storage, monitoring | FAILED | Upload hits a hard storage quota ceiling (`QUOTA_EXCEEDED`, 403) partway through; a monitoring warning earlier in the day went unheeded. |
| `network-timeout` | deferred | Network Timeout | backup_service, network, storage | FAILED | VPN tunnel flaps mid-restore; repeated `ETIMEDOUT` reads exhaust all 3 retries. |
| `permission-denied` | deferred | Permission Error | backup_service, auth, storage | FAILED | An unrelated access-policy change narrows the backup role's scope; every retry gets the same `ACCESS_DENIED` since the problem isn't transient. |
| `service-unavailable` | deferred | Service Unavailable | backup_service, encryption, storage | RECOVERED | The encryption key service goes down for maintenance mid-upload (`SERVICE_UNAVAILABLE`); backoff-retry succeeds once it comes back — a working recovery pattern, not just a failure. |
| `clock-skew` | deferred | Clock Skew | backup_service, storage, monitoring | FAILED | The backup host's clock drifts 6+ minutes ahead of the storage service; signed requests are rejected (`REQUEST_TIME_TOO_SKEWED`) even though nothing else is wrong. Monitoring only confirms the drift after the fact. |
| `partial-failure` | deferred | Partial Failure | backup_service, storage, network | PARTIAL_SUCCESS | A job fans out into 4 parallel per-volume tasks sharing one `correlationId` with per-task `traceId`s: 2 succeed cleanly, 1 recovers after a retry, 1 fails permanently (`ECONNRESET`, non-retryable) — overall job status is `PARTIAL_SUCCESS`. |

## Error code taxonomy used across scenarios

| Component | Codes |
|---|---|
| network | `ETIMEDOUT`, `ENETUNREACH`, `ECONNRESET`, `PACKET_LOSS` |
| storage | `QUOTA_EXCEEDED`, `ACCESS_DENIED`, `REQUEST_TIME_TOO_SKEWED`, `SERVICE_UNAVAILABLE` (propagated from a dependency) |
| encryption | `SERVICE_UNAVAILABLE` |
| auth | (policy changes logged as `WARN`, no dedicated error code — the resulting storage-side `ACCESS_DENIED` is the observable failure) |
| monitoring | `QUOTA_WARNING`, `CLOCK_SKEW_DETECTED` |
| backup_service (job-level) | `RETRY_EXHAUSTED`, `JOB_FAILED`, `PERMISSION_DENIED`, `PARTIAL_SUCCESS` |
