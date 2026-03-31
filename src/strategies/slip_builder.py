"""
Reusable slip-building adapter shared by CLI and API surfaces.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from src.strategies.forbidden_fruit import ForbiddenFruitEngine

__all__ = ["ForbiddenFruitSlipBuilder"]


def _safe_prob(prediction: Dict[str, Any], key: str) -> float:
    """Return a numeric probability or 0.0 for missing values."""
    value = prediction.get(key)
    if value is None:
        return 0.0

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _extract_market_odds(prediction: Dict[str, Any]) -> Dict[str, float]:
    raw_odds = prediction.get("market_odds") or prediction.get("odds")
    if isinstance(raw_odds, dict):
        return raw_odds
    return {}


class ForbiddenFruitSlipBuilder:
    """Translate raw prediction rows into a Forbidden Fruit slip."""

    def __init__(self, engine: Optional[ForbiddenFruitEngine] = None) -> None:
        self.engine = engine or ForbiddenFruitEngine()

    def build_candidates(self, predictions: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Reuse the same market mapping that powers the CLI slip output."""
        candidates: List[Dict[str, Any]] = []

        for prediction in predictions:
            match_info = {
                "home_team": prediction["home_team"],
                "away_team": prediction["away_team"],
                "league": prediction["league"],
                "date": prediction.get("time") or prediction.get("date"),
                "match_id": prediction["match_id"],
                "h2h_match_count": prediction.get("h2h_match_count", 0),
                "days_since_last_h2h": prediction.get("days_since_last_h2h", 0),
            }
            precomputed_confidences = {
                "CORNERS_O7.5": prediction.get("conf_o75"),
                "CARDS_U5.5": prediction.get("conf_u55"),
            }
            strategy_input = {
                "home_win": _safe_prob(prediction, "home"),
                "draw": _safe_prob(prediction, "draw"),
                "away_win": _safe_prob(prediction, "away"),
                "over_2_5": _safe_prob(prediction, "o25"),
                "under_2_5": _safe_prob(prediction, "u25"),
                "btts_yes": _safe_prob(prediction, "btts"),
                "btts_no": _safe_prob(prediction, "btts_no"),
                "over_1_5": _safe_prob(prediction, "over_1_5"),
                "home_under_1_5": _safe_prob(prediction, "home_under_1_5"),
                "away_under_1_5": _safe_prob(prediction, "away_under_1_5"),
                "corn_u11": _safe_prob(prediction, "corn_u11"),
                "corners_over_7_5": _safe_prob(prediction, "corn_o75"),
                "corners_home_win": _safe_prob(prediction, "corn_1x2_h"),
                "corners_draw": _safe_prob(prediction, "corn_1x2_d"),
                "corners_away_win": _safe_prob(prediction, "corn_1x2_a"),
                "dc_1x": _safe_prob(prediction, "dc_1x"),
                "dc_x2": _safe_prob(prediction, "dc_x2"),
                "dc_12": _safe_prob(prediction, "dc_12"),
                "cards_over_2_5": _safe_prob(prediction, "card_o25"),
                "card_u55": _safe_prob(prediction, "card_u55"),
                "card_u45": _safe_prob(prediction, "card_u45"),
                "goals_under_3_5": _safe_prob(prediction, "u35"),
            }
            match_candidates = self.engine.analyze_match(
                match_info,
                strategy_input,
                precomputed_confidences=precomputed_confidences,
            )
            market_odds = _extract_market_odds(prediction)
            match_candidates = self.engine.annotate_candidates_with_edge(
                match_candidates,
                market_odds,
            )
            candidates.extend(
                match_candidates
            )

        return self.engine.edge_engine.rank_opportunities(candidates)

    def generate(
        self,
        predictions: Sequence[Dict[str, Any]],
        min_probability: float = 0.65,
        max_selections: int = 4,
    ) -> List[Dict[str, Any]]:
        candidates = self.build_candidates(predictions)
        return self.engine.construct_slip(
            candidates,
            min_probability=min_probability,
            max_selections=max_selections,
        )
