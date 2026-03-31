"""
Optional Sentry telemetry helpers.

This module is intentionally defensive:
- If sentry-sdk is unavailable, all helpers no-op.
- If SENTRY_DSN is missing, all helpers no-op.
- Telemetry failures never crash application flow.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Sequence

logger = logging.getLogger(__name__)

try:
    import sentry_sdk
except Exception:  # pragma: no cover - import may fail in constrained envs.
    sentry_sdk = None

_sentry_enabled = False
_disabled_notice_logged = False


def init_sentry(
    *,
    component: str,
    dsn: Optional[str] = None,
    integrations: Optional[Sequence[Any]] = None,
) -> bool:
    """Initialize sentry-sdk if configured, otherwise no-op."""
    global _sentry_enabled, _disabled_notice_logged

    if _sentry_enabled:
        return True

    if sentry_sdk is None:
        if not _disabled_notice_logged:
            logger.info("Sentry disabled: sentry_sdk is not available.")
            _disabled_notice_logged = True
        return False

    sentry_dsn = dsn or os.getenv("SENTRY_DSN")
    if not sentry_dsn:
        if not _disabled_notice_logged:
            logger.info("Sentry disabled: SENTRY_DSN is not set.")
            _disabled_notice_logged = True
        return False

    try:
        kwargs: Dict[str, Any] = {
            "dsn": sentry_dsn,
            "environment": os.getenv("ENV", "development"),
            "release": os.getenv("APP_VERSION"),
            "traces_sample_rate": float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.0")),
        }
        if integrations is not None:
            kwargs["integrations"] = list(integrations)

        sentry_sdk.init(**kwargs)
        sentry_sdk.set_tag("component", component)
        _sentry_enabled = True
        logger.info("Sentry enabled for component=%s", component)
        return True
    except Exception as exc:
        logger.warning("Sentry init failed for component=%s: %s", component, exc)
        return False


def capture_exception(exc: BaseException, *, context: Optional[Dict[str, Any]] = None) -> None:
    """Capture exception to Sentry when telemetry is active."""
    if not _sentry_enabled or sentry_sdk is None:
        return
    try:
        with sentry_sdk.push_scope() as scope:
            for key, value in (context or {}).items():
                scope.set_extra(str(key), value)
            sentry_sdk.capture_exception(exc)
    except Exception:
        logger.debug("Failed to send exception to Sentry.", exc_info=True)


def capture_alert(
    *,
    event_name: str,
    message: str,
    level: str = "warning",
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """Capture manual operational alert to Sentry when telemetry is active."""
    if not _sentry_enabled or sentry_sdk is None:
        return
    try:
        with sentry_sdk.push_scope() as scope:
            scope.set_tag("event_name", event_name)
            for key, value in (context or {}).items():
                scope.set_extra(str(key), value)
            sentry_sdk.capture_message(message, level=level)
    except Exception:
        logger.debug("Failed to send alert to Sentry.", exc_info=True)
