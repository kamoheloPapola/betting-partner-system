
# Deployment Guide

## Prerequisites
- Python 3.11
- Pip
- Valid API Keys (odds-api, football-data, etc.) set in environment variables.

## Installation

1. **Clone Repository**
   ```bash
   git clone <repo-url>
   cd betting_partner_system
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Installation**
   Run the test suite:
   ```bash
   pytest tests/
   ```

## Production Workflow

### 1. Initialize Data
Fetch the latest historical data for all supported leagues:
```bash
python -m src.cli fetch-latest-season --league PL
python -m src.cli fetch-latest-season --league SA
# ... repeat for others
python -m src.cli ingest-results
```

### 2. Train Base Models
Train the foundational models. This usually takes 5-10 minutes.
```bash
python -m src.cli train --mode production
```

### 3. Verify System Health
Check for drift and coverage issues before going live:
```bash
python -m src.cli refresh-drift
python -m src.cli audit-coverage
```

### 4. Schedule the canonical nightly pipeline

The supported deployment topology is Docker Compose on a self-hosted machine.
`Dockerfile.scheduler` runs the canonical seven-step pipeline once; the only
recurring trigger is the version-controlled host-cron definition at
`deploy/cron/betting-nightly`. The scheduler service is profile-gated so a normal
`docker compose up` does not also trigger an unscheduled run.

The committed schedule runs at 06:00 UTC, which is 08:00 in
Africa/Johannesburg. Before installing it, replace
`/opt/betting_partner_system` if the repository is deployed elsewhere, then copy
the file into `/etc/cron.d/` and preserve its root ownership and `0644` mode.
The host requires Docker Compose with `docker compose up --wait` support and
`flock`.

The trigger first waits for the API's strong model-readiness healthcheck, then
runs the scheduler container without starting a second dependency graph. A
non-blocking host lock prevents overlapping scheduler runs. Trigger logs use the
stable events `scheduler_trigger_started`, `scheduler_trigger_completed`,
`scheduler_trigger_failed`, and `scheduler_trigger_overlap_skipped`.

The old Render shell claim, direct prediction cron, Windows batch file, and agent
workflow are retained only for audit history under `legacy/scheduler/`; none is a
supported entrypoint.

## Environment Variables
`.env.example` is generated from the canonical schema in
`src/config/env_contract.py`; do not maintain a second list by hand. Verify and
copy it before filling secrets:

```bash
python scripts/generate_env_example.py --check
cp .env.example .env
```

API startup requires non-empty `ARTIFACT_SIGNING_KEY`, `CLI_AUTH_TOKENS`,
`FOOTBALL_DATA_API_KEY`, `ODDS_API_KEY`, and `SENTRY_DSN`. Scheduler startup
requires `ARTIFACT_SIGNING_KEY` and `SENTRY_DSN`. Missing and whitespace-only
values both fail before model acquisition or any data-provider request.
Environment failures emit `startup_environment_invalid` and exit `2`; model-state
preflight failures emit `startup_model_state_blocked` and exit `3`. These stable
event names are inputs to the alerting work described for the later observability
phase.

The connect timeout bounds TCP establishment. The transfer timeout is curl's
overall download deadline and both the S3 client's socket-read deadline and an
application-level total wall-clock deadline around each S3 operation. Timed-out
downloads are staged in partial files and never replace the last complete local
artifact.

## Signed Artifact Upgrade

Serialized model artifacts are verified before deserialization. Existing manifests
without an `artifact_integrity` record are intentionally rejected.

Before the first deployment using artifact verification:

1. Configure the same `ARTIFACT_SIGNING_KEY` for the trusted training/publishing
   process and every inference process. Keep it in the deployment secret store;
   never commit it.
2. Retrain or republish each artifact from a trusted local source so
   `ModelRegistry.register_model` writes its signed integrity record.
3. Distribute the artifact and its newly-written manifest together.
4. Do not sign artifacts after downloading them from an untrusted or unverifiable
   remote source; doing so would bless the bytes the verification boundary is
   intended to reject.

There is no automatic compatibility fallback for unsigned artifacts. A missing,
malformed, or mismatched signature fails closed with `ArtifactVerificationError`.

## Container Storage Contract

Both the API and scheduler resolve `MODELS_DIR` to `/app/data/models`. They mount
the same host `./data` directory at `/app/data`. The full data mount is
intentional: model locking, features, drift state, results, predictions, and
model-history SQLite state all live under `/app/data`. Maintaining separate
per-file mounts would recreate the visibility gap whenever a new runtime state
file is added.

The API installs the manifest first and treats it as the readiness authority. A
deployment is model-ready only when every active manifest pointer resolves to a
non-empty artifact and both `feature_columns.json` and
`feature_baselines.json` exist. The old `goals_model.pkl` sentinel is not valid;
that filename is not a manifest artifact. An existing persistent manifest is not
overwritten during a warm restart.

The scheduler waits for the API healthcheck. Because model acquisition completes
before Uvicorn starts, this prevents the scheduler from racing the API's initial
download. If `/app/data/.model_state` is absent, the API remains fail-closed as
unlocked and the scheduler does not start.

The recurring trigger itself has a separate silent-failure mode: if host cron
never fires, there is no process failure to log. A successful nightly run now
atomically records `data/monitoring/nightly_heartbeat.json`. The Compose-polled
API health endpoint checks that heartbeat and dispatches a critical alert after
`NIGHTLY_HEARTBEAT_MAX_AGE_HOURS` (30 hours by default). The first check in a new
deployment atomically records the start of a grace window and reports
`not_initialized` without alerting. If no first successful nightly run arrives
before that window expires, it alerts as `missing`; later stale timestamps alert
as `stale`. The heartbeat state is informational and does not make the API
unhealthy, because doing so would prevent the scheduler from starting to repair
its own stale state.

## Operational alerts

CLI authorization denials, artifact-verification failures, model-state blocks,
and stale scheduler heartbeats feed `src.monitoring.alert_pipeline.dispatch_alert`.
The dispatcher reuses the existing Slack, ntfy, and SMTP senders and their
24-hour stable-event de-duplication. Network attempts are bounded to 10 seconds
per configured channel; a channel or dispatcher failure is logged and never
changes the caller's fail-closed behavior.

Sentry is a secondary sink for dispatched events, not the primary delivery
path. The API and scheduler environment contract fails startup when
`SENTRY_DSN` is absent, replacing the previous silent no-op; failed Sentry
delivery still cannot swallow an operational alert. If no Slack, ntfy, or email
destination is configured, the existing alert logger remains the last-resort
sink.

## State authority

PostgreSQL is disabled. Compose contains no PostgreSQL or placeholder database
service, external database URLs are not recognized, and the application has no
manifest, resolved-prediction, or drift dual-write branch.

Authoritative state is deliberately narrow:

- `manifest.json` and its backup hold model registry state.
- `data/eval/prediction_outcomes.csv` holds resolved predictions.
- drift JSON and CSV files hold drift state and events.
- `data/models/model_history.db` holds model lifecycle history in SQLite.

`model_history.db` uses WAL mode and a 30-second busy timeout. The timeout is long
enough for the API and scheduler's short writes to serialize, while still bounded
so a stuck writer fails visibly. RL-bandit and shadow-comparison readers consume
the authoritative outcomes CSV rather than an unpopulated SQL mirror.

The documented Compose deployment never passed `DATABASE_URL` to either
application service, and no repository-owned operational/PostgreSQL data was
found. An external deployment may have independently configured a database;
before upgrading such a deployment, inspect and export it. The repository cannot
verify external Render/dashboard state.

### Deployment verification

Run these checks from a host with Docker Compose. Perform the clean-volume check
from a disposable copy of the repository, because `./data` is a bind mount and is
not isolated by the Compose project name. In that disposable copy, provision a
valid test lock before startup:

```bash
printf 'LOCKED_v14.0' > data/.model_state
```

```bash
docker compose config
docker compose -p betting-path-check up --build
docker compose -p betting-path-check exec api python -c "from src.config import MODELS_DIR; from src.config.model_state import get_state_file; print(MODELS_DIR); print(get_state_file())"
docker compose -p betting-path-check run --rm scheduler python -c "from src.config import MODELS_DIR; from src.config.model_state import get_state_file, get_model_state; print(MODELS_DIR); print(get_state_file()); print(get_model_state())"
```

Expected in both one-off container checks:

```text
/app/data/models
/app/data/.model_state
```

During a genuinely fresh-volume run, the API logs must show manifest installation
followed by downloads for every missing active artifact. It must not report ready
with zero active models. The scheduler must remain pending until the API becomes
healthy, then read the same lock value. Remove only the disposable verification
project from the disposable repository copy when finished. This stops its
containers; it does not delete the bind-mounted `./data` tree:

```bash
docker compose -p betting-path-check down
```
