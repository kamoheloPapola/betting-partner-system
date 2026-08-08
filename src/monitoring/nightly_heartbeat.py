"""Local dead-man heartbeat for the nightly scheduler."""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from src.config import DATA_DIR
from src.monitoring.alert_pipeline import dispatch_alert

logger = logging.getLogger(__name__)

HEARTBEAT_FILE = DATA_DIR / "monitoring" / "nightly_heartbeat.json"
HEARTBEAT_MAX_AGE_ENV = "NIGHTLY_HEARTBEAT_MAX_AGE_HOURS"
DEFAULT_MAX_AGE_HOURS = 30.0

__all__ = [
    "DEFAULT_MAX_AGE_HOURS",
    "HEARTBEAT_FILE",
    "HeartbeatStatus",
    "check_nightly_heartbeat",
    "record_nightly_success",
]


@dataclass(frozen=True)
class HeartbeatStatus:
    state: str
    completed_at: Optional[str] = None
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _write_atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(dict(payload), sort_keys=True), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _watch_file(heartbeat_file: Path) -> Path:
    return heartbeat_file.with_name("nightly_heartbeat_watch.json")


def _max_age_hours() -> float:
    raw_value = os.getenv(HEARTBEAT_MAX_AGE_ENV, str(DEFAULT_MAX_AGE_HOURS))
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        logger.error(
            "nightly_heartbeat_config_invalid name=%s value=%r",
            HEARTBEAT_MAX_AGE_ENV,
            raw_value,
        )
        return DEFAULT_MAX_AGE_HOURS
    if value <= 0:
        logger.error(
            "nightly_heartbeat_config_invalid name=%s value=%r",
            HEARTBEAT_MAX_AGE_ENV,
            raw_value,
        )
        return DEFAULT_MAX_AGE_HOURS
    return value


def record_nightly_success(
    *,
    summary: Optional[Mapping[str, Any]] = None,
    heartbeat_file: Optional[Path] = None,
    now: Optional[datetime] = None,
) -> str:
    """Atomically record the completion of a successful nightly run."""
    heartbeat_file = heartbeat_file or HEARTBEAT_FILE
    completed_at = (now or _utc_now()).astimezone(timezone.utc).isoformat()
    payload = {
        "completed_at": completed_at,
        "status": "ok",
        "summary": dict(summary or {}),
    }
    _write_atomic_json(heartbeat_file, payload)
    logger.info("nightly_heartbeat_recorded completed_at=%s", completed_at)
    return completed_at


def check_nightly_heartbeat(
    *,
    heartbeat_file: Optional[Path] = None,
    now: Optional[datetime] = None,
    max_age_hours: Optional[float] = None,
) -> HeartbeatStatus:
    """Check scheduler freshness and alert once de-duplication permits.

    A missing file means the deployment has never completed a nightly run and
    is intentionally not treated as a missed heartbeat.
    """
    heartbeat_file = heartbeat_file or HEARTBEAT_FILE
    configured_max_age = max_age_hours if max_age_hours is not None else _max_age_hours()
    current_time = (now or _utc_now()).astimezone(timezone.utc)
    stale_after = timedelta(hours=float(configured_max_age))
    if not heartbeat_file.exists():
        watch_file = _watch_file(heartbeat_file)
        if not watch_file.exists():
            try:
                _write_atomic_json(
                    watch_file,
                    {"monitoring_started_at": current_time.isoformat()},
                )
            except OSError as exc:
                logger.error(
                    "nightly_heartbeat_watch_write_failed path=%s detail=%s",
                    watch_file,
                    exc,
                )
                dispatch_alert(
                    source="scheduler_heartbeat",
                    severity="CRITICAL",
                    message="Nightly scheduler heartbeat monitor could not initialize",
                    context={"path": str(watch_file), "reason": "watch_write_failure"},
                )
                return HeartbeatStatus("invalid", max_age_hours=configured_max_age)
            return HeartbeatStatus("not_initialized", max_age_hours=configured_max_age)

        try:
            watch_payload = json.loads(watch_file.read_text(encoding="utf-8"))
            monitoring_started_at = datetime.fromisoformat(
                str(watch_payload["monitoring_started_at"])
            )
            if monitoring_started_at.tzinfo is None:
                raise ValueError("monitoring_started_at must include a timezone")
            monitoring_started_at = monitoring_started_at.astimezone(timezone.utc)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            logger.error(
                "nightly_heartbeat_watch_invalid path=%s error_type=%s detail=%s",
                watch_file,
                type(exc).__name__,
                exc,
            )
            dispatch_alert(
                source="scheduler_heartbeat",
                severity="CRITICAL",
                message="Nightly scheduler heartbeat monitor state is unreadable",
                context={"path": str(watch_file), "reason": "invalid_watch_state"},
            )
            return HeartbeatStatus("invalid", max_age_hours=configured_max_age)

        if current_time - monitoring_started_at > stale_after:
            logger.error(
                "nightly_heartbeat_missing monitoring_started_at=%s max_age_hours=%s",
                monitoring_started_at.isoformat(),
                configured_max_age,
            )
            dispatch_alert(
                source="scheduler_heartbeat",
                severity="CRITICAL",
                message="Nightly scheduler has never recorded a successful run",
                context={
                    "monitoring_started_at": monitoring_started_at.isoformat(),
                    "max_age_hours": float(configured_max_age),
                },
            )
            return HeartbeatStatus("missing", max_age_hours=configured_max_age)
        return HeartbeatStatus("not_initialized", max_age_hours=configured_max_age)

    try:
        payload = json.loads(heartbeat_file.read_text(encoding="utf-8"))
        completed_at_raw = str(payload["completed_at"])
        completed_at = datetime.fromisoformat(completed_at_raw)
        if completed_at.tzinfo is None:
            raise ValueError("completed_at must include a timezone")
        completed_at = completed_at.astimezone(timezone.utc)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.error(
            "nightly_heartbeat_invalid path=%s error_type=%s detail=%s",
            heartbeat_file,
            type(exc).__name__,
            exc,
        )
        dispatch_alert(
            source="scheduler_heartbeat",
            severity="CRITICAL",
            message="Nightly scheduler heartbeat is unreadable",
            context={"path": str(heartbeat_file), "reason": "invalid_heartbeat"},
        )
        return HeartbeatStatus("invalid", max_age_hours=configured_max_age)

    if current_time - completed_at > stale_after:
        logger.error(
            "nightly_heartbeat_stale completed_at=%s max_age_hours=%s",
            completed_at.isoformat(),
            configured_max_age,
        )
        dispatch_alert(
            source="scheduler_heartbeat",
            severity="CRITICAL",
            message="Nightly scheduler heartbeat is stale",
            context={
                "completed_at": completed_at.isoformat(),
                "max_age_hours": float(configured_max_age),
            },
        )
        return HeartbeatStatus(
            "stale",
            completed_at=completed_at.isoformat(),
            max_age_hours=float(configured_max_age),
        )

    return HeartbeatStatus(
        "fresh",
        completed_at=completed_at.isoformat(),
        max_age_hours=float(configured_max_age),
    )
