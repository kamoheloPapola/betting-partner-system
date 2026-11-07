"""Explicit inference-time contracts for non-goal football count means.

Goal lambdas intentionally retain the separate ``clamp_lambda`` contract in
``src.simulation.match_simulator``. Corner and card models predict different
targets and must never inherit the goal-specific upper bound of 4.0.
"""

from __future__ import annotations

import logging
import math


logger = logging.getLogger(__name__)

CORNER_COUNT_MEAN_MIN = 0.05
CARD_COUNT_MEAN_MIN = 0.05


def _sanitize_positive_count_mean(
    value: float,
    *,
    minimum: float,
    domain: str,
    context: str,
) -> float:
    """Return a finite positive mean without applying an unrelated upper cap."""
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{context or domain}: non-finite {domain} count mean: {numeric}")
    if numeric < minimum:
        logger.warning(
            "%s: %s count mean %.6f below numerical floor %.6f; applying floor",
            context or domain,
            domain,
            numeric,
            minimum,
        )
        return minimum
    return numeric


def sanitize_corner_count_mean(value: float, *, context: str = "") -> float:
    """Validate a team or total-corner expected count; no upper cap is applied."""
    return _sanitize_positive_count_mean(
        value,
        minimum=CORNER_COUNT_MEAN_MIN,
        domain="corner",
        context=context,
    )


def sanitize_card_count_mean(value: float, *, context: str = "") -> float:
    """Validate a total-card expected count; no upper cap is applied."""
    return _sanitize_positive_count_mean(
        value,
        minimum=CARD_COUNT_MEAN_MIN,
        domain="card",
        context=context,
    )


__all__ = [
    "CARD_COUNT_MEAN_MIN",
    "CORNER_COUNT_MEAN_MIN",
    "sanitize_card_count_mean",
    "sanitize_corner_count_mean",
]
