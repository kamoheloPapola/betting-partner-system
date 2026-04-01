"""
Prediction adapter built on top of the sealed CLI prediction workflow.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from src.cli.commands.prediction import _run_predict_loop
from src.cli.utils import DateFilter, filter_matches_by_date, resolve_league_code
from src.core.container import ServiceContainer
from src.core.exceptions import DataValidationError
from src.core.validators import validate_match_dataframe

logger = logging.getLogger(__name__)

__all__ = ["Predictor"]


class Predictor:
    """Thin adapter that exposes prediction workflow to the API layer."""

    def __init__(self, container: Optional[ServiceContainer] = None) -> None:
        self.container = container or ServiceContainer.get_instance()

    def _resolve_league(self, league: Optional[str]) -> Optional[str]:
        if not league:
            return None

        code = resolve_league_code(league)
        if code is None:
            raise DataValidationError(
                f"Unsupported league: {league}",
                context={"league": league},
            )
        return code.value

    def get_upcoming_matches(self, league: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load future matches using the same filter policy as the CLI."""
        league_code = self._resolve_league(league)
        df = self.container.pipeline.run(league=league_code)
        validate_match_dataframe(df, context="predictor.get_upcoming_matches")

        upcoming = filter_matches_by_date(
            df,
            DateFilter.ALL,
            show_all=True,
            user_timezone="UTC",
        )
        if upcoming.empty:
            return []

        upcoming = upcoming.sort_values("date").reset_index(drop=True)
        return upcoming.to_dict(orient="records")

    def predict_all(self, matches: Sequence[Dict[str, Any]] | pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate probabilities for match records returned by get_upcoming_matches."""
        if isinstance(matches, pd.DataFrame):
            df = matches.copy()
        else:
            df = pd.DataFrame(list(matches))

        if df.empty:
            return []

        if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")

        validate_match_dataframe(df, context="predictor.predict_all")
        predictions = _run_predict_loop(df.sort_values("date"))
        logger.info("Generated %s predictions for API adapter", len(predictions))
        return predictions

    def predict_upcoming(
        self,
        league: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Convenience method for the API layer."""
        matches = self.get_upcoming_matches(league)
        if limit is not None:
            matches = matches[:limit]
        return self.predict_all(matches)

    def predict_for_show_predictions(
        self,
        *,
        league: Optional[str],
        date: str | DateFilter = DateFilter.TODAY,
        show_all: bool = False,
        timezone: str = "LOCAL",
        simulate: bool = True,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Reuse the exact core data/prediction path used by CLI `show-predictions`.
        """
        league_code = self._resolve_league(league)
        df = self.container.pipeline.run(league=league_code)
        validate_match_dataframe(df, context="predictor.predict_for_show_predictions")

        df_target = filter_matches_by_date(
            df,
            date,
            show_all=show_all,
            user_timezone=timezone,
        )
        if df_target.empty:
            return []

        predictions = _run_predict_loop(df_target.sort_values("date"), use_simulator=simulate)
        if limit is not None:
            return predictions[:limit]
        return predictions
