"""
Edge evaluation utilities for value-aware recommendation workflows.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

__all__ = ["EdgeEngine"]


class EdgeEngine:
    """Compute price-aware edge metrics without making bankroll decisions."""

    DEFAULT_MIN_EV = 0.02
    DEFAULT_MIN_EDGE = 0.0

    def compute_implied_probability(self, odds: Optional[float]) -> Optional[float]:
        odds_value = self._coerce_float(odds)
        if odds_value is None or odds_value <= 1.0:
            return None
        return 1.0 / odds_value

    def compute_edge(
        self,
        probability: Optional[float],
        odds: Optional[float],
    ) -> Optional[float]:
        prob_value = self._coerce_probability(probability)
        implied_probability = self.compute_implied_probability(odds)
        if prob_value is None or implied_probability is None:
            return None
        return prob_value - implied_probability

    def compute_ev(
        self,
        probability: Optional[float],
        odds: Optional[float],
    ) -> Optional[float]:
        prob_value = self._coerce_probability(probability)
        odds_value = self._coerce_float(odds)
        if prob_value is None or odds_value is None or odds_value <= 1.0:
            return None
        return (prob_value * odds_value) - 1.0

    def passes_value_threshold(
        self,
        probability: Optional[float],
        odds: Optional[float],
        *,
        min_ev: Optional[float] = None,
        min_edge: Optional[float] = None,
    ) -> bool:
        evaluation = self.evaluate_opportunity(
            probability=probability,
            odds=odds,
            min_ev=min_ev,
            min_edge=min_edge,
        )
        return bool(evaluation["passes_value_threshold"])

    def evaluate_opportunity(
        self,
        *,
        probability: Optional[float],
        odds: Optional[float],
        min_ev: Optional[float] = None,
        min_edge: Optional[float] = None,
    ) -> Dict[str, Any]:
        min_ev_value = self.DEFAULT_MIN_EV if min_ev is None else float(min_ev)
        min_edge_value = self.DEFAULT_MIN_EDGE if min_edge is None else float(min_edge)
        prob_value = self._coerce_probability(probability)
        odds_value = self._coerce_float(odds)

        base = {
            "odds": odds_value,
            "implied_probability": None,
            "edge": None,
            "ev": None,
            "priced": False,
            "passes_value_threshold": False,
            "value_reason": "UNPRICED",
            "min_edge": min_edge_value,
            "min_ev": min_ev_value,
        }

        if prob_value is None:
            base["value_reason"] = "INVALID_PROBABILITY"
            return base

        if odds_value is None:
            base["value_reason"] = "MISSING_ODDS"
            return base

        if odds_value <= 1.0:
            base["value_reason"] = "INVALID_ODDS"
            return base

        implied_probability = self.compute_implied_probability(odds_value)
        edge = self.compute_edge(prob_value, odds_value)
        ev = self.compute_ev(prob_value, odds_value)
        passes = bool(
            implied_probability is not None
            and edge is not None
            and ev is not None
            and edge >= min_edge_value
            and ev >= min_ev_value
        )

        reason = "VALUE_OK"
        if not passes:
            if edge is None or edge < min_edge_value:
                reason = "EDGE_TOO_LOW"
            elif ev is None or ev < min_ev_value:
                reason = "EV_TOO_LOW"

        return {
            "odds": odds_value,
            "implied_probability": implied_probability,
            "edge": edge,
            "ev": ev,
            "priced": True,
            "passes_value_threshold": passes,
            "value_reason": reason,
            "min_edge": min_edge_value,
            "min_ev": min_ev_value,
        }

    def annotate_opportunity(
        self,
        opportunity: Dict[str, Any],
        *,
        odds: Optional[float] = None,
        min_ev: Optional[float] = None,
        min_edge: Optional[float] = None,
    ) -> Dict[str, Any]:
        annotated = dict(opportunity)
        probability = annotated.get("probability", annotated.get("confidence"))
        annotated.update(
            self.evaluate_opportunity(
                probability=probability,
                odds=odds if odds is not None else annotated.get("odds"),
                min_ev=min_ev,
                min_edge=min_edge,
            )
        )
        return annotated

    def rank_opportunities(
        self,
        opportunities: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        ranked = sorted(
            (dict(opportunity) for opportunity in opportunities),
            key=self._rank_key,
            reverse=True,
        )

        for index, opportunity in enumerate(ranked, start=1):
            opportunity["value_rank"] = index

        return ranked

    def _rank_key(self, opportunity: Dict[str, Any]) -> tuple[float, float, float, float]:
        pass_flag = 1.0 if opportunity.get("passes_value_threshold") else 0.0
        edge = self._coerce_float(opportunity.get("edge"))
        ev = self._coerce_float(opportunity.get("ev"))
        confidence = self._coerce_float(
            opportunity.get("confidence", opportunity.get("probability"))
        )
        return (
            pass_flag,
            edge if edge is not None else float("-inf"),
            ev if ev is not None else float("-inf"),
            confidence if confidence is not None else float("-inf"),
        )

    @staticmethod
    def _coerce_probability(value: Optional[float]) -> Optional[float]:
        prob_value = EdgeEngine._coerce_float(value)
        if prob_value is None or prob_value < 0.0 or prob_value > 1.0:
            return None
        return prob_value

    @staticmethod
    def _coerce_float(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
