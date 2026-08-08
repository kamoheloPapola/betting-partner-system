"""Shared operational alert dispatch.

Operational callers emit a source, severity, message, and stable context here.
The existing :class:`Alerter` owns channel selection and de-duplication while
Sentry remains an optional secondary sink. Dispatch failures never alter the
fail-closed behavior of the caller; they are emitted to the application log.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

from src.monitoring.alerter import Alerter
from src.monitoring.telemetry import capture_alert

logger = logging.getLogger(__name__)

__all__ = ["dispatch_alert"]


def dispatch_alert(
    *,
    source: str,
    severity: str,
    message: str,
    context: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Dispatch one operational event without raising into its caller.

    Slack, ntfy, and SMTP are selected by the existing ``Alerter`` from the
    configured environment. If no channel is configured, or a configured
    channel fails, ``Alerter`` writes the alert to the normal log. Sentry is
    deliberately secondary so a missing ``SENTRY_DSN`` cannot swallow alerts.
    """
    normalized_severity = str(severity or "WARNING").strip().upper()
    event_context = {"source": str(source), **dict(context or {})}

    try:
        sent = Alerter().send_alert(
            message,
            context=event_context,
            severity=normalized_severity,
        )
    except Exception:
        logger.exception(
            "operational_alert_dispatch_failed source=%s severity=%s message=%s",
            source,
            normalized_severity,
            message,
        )
        return False

    if sent:
        capture_alert(
            event_name=str(source),
            message=message,
            level=normalized_severity.lower(),
            context=event_context,
        )
    return bool(sent)
