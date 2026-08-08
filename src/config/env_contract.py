"""Canonical environment-variable schema and startup validation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence


API_PROCESS = "api"
SCHEDULER_PROCESS = "scheduler"
ENRICHMENT_PROCESS = "enrichment"
DATA_FETCH_PROCESS = "data-fetch"
KNOWN_PROCESSES = frozenset(
    {API_PROCESS, SCHEDULER_PROCESS, ENRICHMENT_PROCESS, DATA_FETCH_PROCESS}
)


@dataclass(frozen=True)
class EnvVarSpec:
    name: str
    group: str
    description: str
    required_for: frozenset[str] = frozenset()
    default: Optional[str] = None
    example: Optional[str] = None
    secret: bool = False

    @property
    def is_optional(self) -> bool:
        return not self.required_for


def _required(*processes: str) -> frozenset[str]:
    return frozenset(processes)


ENV_SCHEMA: tuple[EnvVarSpec, ...] = (
    EnvVarSpec("MODELS_DIR", "Storage", "Canonical model artifact directory.", default="/app/data/models"),
    EnvVarSpec(
        "ARTIFACT_SIGNING_KEY",
        "Storage",
        "HMAC key for serialized artifacts; use at least 32 random bytes.",
        required_for=_required(API_PROCESS, SCHEDULER_PROCESS),
        secret=True,
    ),
    EnvVarSpec("ARTIFACT_CONNECT_TIMEOUT_SECONDS", "Storage", "Artifact TCP connect timeout.", default="10"),
    EnvVarSpec("ARTIFACT_TRANSFER_TIMEOUT_SECONDS", "Storage", "Artifact total transfer deadline.", default="120"),
    EnvVarSpec("MANIFEST_SRC", "Storage", "Bundled manifest source path.", default="/app/src/ml/models/manifest.json"),
    EnvVarSpec(
        "MODEL_RELEASE_BASE_URL",
        "Storage",
        "Base URL for release artifacts.",
        default="https://github.com/kamoheloPapola/betting-partner-system/releases/download/v1.0-models",
    ),
    EnvVarSpec(
        "CLI_AUTH_TOKENS",
        "Authentication",
        "JSON object mapping CLI admin tokens to roles.",
        required_for=_required(API_PROCESS),
        secret=True,
    ),
    EnvVarSpec(
        "FOOTBALL_DATA_API_KEY",
        "Data providers",
        "football-data.org v4 API token.",
        required_for=_required(API_PROCESS, DATA_FETCH_PROCESS),
        secret=True,
    ),
    EnvVarSpec(
        "API_FOOTBALL_KEY",
        "Data providers",
        "Direct API-Sports/API-Football token.",
        required_for=_required(ENRICHMENT_PROCESS),
        secret=True,
    ),
    EnvVarSpec(
        "ODDS_API_KEY",
        "Data providers",
        "The Odds API token used for upcoming fixtures and odds.",
        required_for=_required(API_PROCESS),
        secret=True,
    ),
    EnvVarSpec("OPENWEATHER_API_KEY", "Data providers", "Optional OpenWeather token.", secret=True),
    EnvVarSpec("FOOTBALL_DATA_SEASON", "Data providers", "Optional season-year override."),
    EnvVarSpec(
        "SENTRY_DSN",
        "Observability",
        "Sentry project DSN.",
        required_for=_required(API_PROCESS, SCHEDULER_PROCESS),
        secret=True,
    ),
    EnvVarSpec("SENTRY_TRACES_SAMPLE_RATE", "Observability", "Sentry trace sample rate.", default="0.0"),
    EnvVarSpec("APP_VERSION", "Observability", "Release/version tag sent to telemetry."),
    EnvVarSpec("ENV", "Runtime", "Runtime environment name.", default="development"),
    EnvVarSpec("LOG_LEVEL", "Runtime", "Application log level.", default="INFO"),
    EnvVarSpec("PREDICTION_SYSTEM_LOG_FILE", "Runtime", "Optional CLI log-file override."),
    EnvVarSpec("WARMUP_CACHE", "Runtime flags", "Enable API prediction-cache warmup.", default="false"),
    EnvVarSpec("SKIP_MODEL_LOCK_CHECK", "Runtime flags", "Development-only API lock bypass.", default="false"),
    EnvVarSpec("LOKY_MAX_CPU_COUNT", "Runtime flags", "Optional joblib worker CPU limit."),
    EnvVarSpec("SPECIALIST_ENSEMBLE_LGBM_WEIGHT", "Model tuning", "LightGBM ensemble weight.", default="0.6"),
    EnvVarSpec("SPECIALIST_ENSEMBLE_XGB_WEIGHT", "Model tuning", "XGBoost ensemble weight.", default="0.4"),
    EnvVarSpec(
        "SPECIALIST_ENSEMBLE_DIVERGENCE_THRESHOLD",
        "Model tuning",
        "Maximum ensemble disagreement before gating.",
        default="0.20",
    ),
    EnvVarSpec("AWS_ACCESS_KEY_ID", "Artifact sync", "Optional S3 access key.", secret=True),
    EnvVarSpec("AWS_SECRET_ACCESS_KEY", "Artifact sync", "Optional S3 secret key.", secret=True),
    EnvVarSpec("AWS_REGION", "Artifact sync", "Optional S3 region."),
    EnvVarSpec("S3_BUCKET", "Artifact sync", "Optional model-sync bucket."),
    EnvVarSpec("S3_PREFIX", "Artifact sync", "Optional model-sync key prefix."),
    EnvVarSpec("SLACK_WEBHOOK_URL", "Alert channels", "Optional Slack webhook.", secret=True),
    EnvVarSpec("NTFY_TOPIC", "Alert channels", "Optional ntfy topic.", secret=True),
    EnvVarSpec("ALERT_EMAIL_TO", "Alert channels", "Optional alert recipient."),
    EnvVarSpec("ALERT_EMAIL_FROM", "Alert channels", "Alert sender address.", default="betting-system@localhost"),
    EnvVarSpec("SMTP_HOST", "Alert channels", "SMTP server hostname.", default="localhost"),
    EnvVarSpec("SMTP_PORT", "Alert channels", "SMTP server port.", default="25"),
    EnvVarSpec(
        "NIGHTLY_HEARTBEAT_MAX_AGE_HOURS",
        "Alert channels",
        "Maximum age of the nightly success heartbeat before alerting.",
        default="30",
    ),
)


SCHEMA_BY_NAME = {spec.name: spec for spec in ENV_SCHEMA}


class EnvironmentContractError(RuntimeError):
    """Raised before startup when required environment is missing."""

    def __init__(self, process: str, missing: Sequence[str]):
        self.process = process
        self.missing = tuple(sorted(missing))
        joined = ", ".join(self.missing)
        super().__init__(
            f"Environment validation failed for {process}; "
            f"missing required variables: {joined}"
        )


def required_names(process: str) -> tuple[str, ...]:
    if process not in KNOWN_PROCESSES:
        raise ValueError(f"Unknown process environment contract: {process}")
    return tuple(sorted(spec.name for spec in ENV_SCHEMA if process in spec.required_for))


def validate_environment(
    process: str,
    environ: Optional[Mapping[str, str]] = None,
) -> None:
    """Fail when a required value is absent or contains only whitespace."""
    values = os.environ if environ is None else environ
    missing = [
        name
        for name in required_names(process)
        if not str(values.get(name, "")).strip()
    ]
    if missing:
        raise EnvironmentContractError(process, missing)


def render_env_example() -> str:
    """Render .env.example directly from ENV_SCHEMA."""
    lines = [
        "# Generated by scripts/generate_env_example.py from src/config/env_contract.py.",
        "# Do not hand-edit; update ENV_SCHEMA and regenerate.",
    ]
    current_group = None
    for spec in ENV_SCHEMA:
        if spec.group != current_group:
            lines.extend(("", f"# === {spec.group} ==="))
            current_group = spec.group
        requirement = (
            f"Required for: {', '.join(sorted(spec.required_for))}."
            if spec.required_for
            else "Optional."
        )
        lines.append(f"# {requirement} {spec.description}")
        value = spec.default if spec.default is not None else (spec.example or "")
        lines.append(f"{spec.name}={value}")
    return "\n".join(lines) + "\n"


def env_example_path() -> Path:
    return Path(__file__).resolve().parents[2] / ".env.example"
