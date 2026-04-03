"""Contextual bandit sidecar for Monte Carlo shadow weighting."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.config import DATA_DIR
from src.core.container import ServiceContainer
from src.db.models import ResolvedPrediction
from src.ml.calibration import calculate_ece

logger = logging.getLogger(__name__)

DEFAULT_ALPHA = 0.1
DEFAULT_STATE_PATH = DATA_DIR / "rl_bandit_state.json"
DEFAULT_WEIGHTS: Dict[str, float] = {
    "tempo_sigma_scale": 1.0,
    "lambda_scale_home": 1.0,
    "lambda_scale_away": 1.0,
}
ECE_BINS = 10
MARKET_BUCKET_ALIASES: Dict[str, set[str]] = {
    "1x2": {"home_win", "draw", "away_win"},
    "over_2_5": {"over_2_5", "goals_over_2_5", "over_2_5_goals"},
    "btts_yes": {"btts_yes"},
    "home_under_1_5": {"home_under_1_5"},
    "away_under_1_5": {"away_under_1_5"},
}
MARKET_TO_BUCKET: Dict[str, str] = {
    alias: bucket
    for bucket, aliases in MARKET_BUCKET_ALIASES.items()
    for alias in aliases
}

__all__ = [
    "ContextualBandit",
    "DEFAULT_STATE_PATH",
    "compute_bucket_ece_by_context",
    "load_resolved_predictions_from_db",
]


def _build_context_key(league: str, market: str) -> str:
    league_token = str(league or "GLOBAL").strip().upper() or "GLOBAL"
    market_token = str(market or "unknown").strip().lower() or "unknown"
    return f"{league_token}:{market_token}"


def _default_bucket_state() -> Dict[str, Any]:
    return {
        "count": 0,
        "last_ece": None,
        "last_reward": None,
        "ece_ema": 0.0,
        "reward_ema": 0.0,
        "weights": dict(DEFAULT_WEIGHTS),
    }


def _clip_scale(value: float) -> float:
    return float(np.clip(value, 0.5, 1.5))


def normalize_market_bucket(market: str) -> Optional[str]:
    market_token = str(market).strip().lower()
    return MARKET_TO_BUCKET.get(market_token)


def load_resolved_predictions_from_db(
    container: Optional[ServiceContainer] = None,
) -> pd.DataFrame:
    """Load resolved predictions from the operational database."""
    active_container = container or ServiceContainer.get_instance()
    try:
        with Session(active_container.engine) as session:
            rows = session.execute(
                select(ResolvedPrediction).where(
                    ResolvedPrediction.outcome.in_(("WON", "LOST"))
                )
            ).scalars().all()
    except Exception as exc:
        logger.warning("RL bandit could not load resolved predictions: %s", exc)
        return pd.DataFrame(
            columns=["league", "market", "probability", "actual_outcome", "resolved_at"]
        )

    if not rows:
        return pd.DataFrame(
            columns=["league", "market", "probability", "actual_outcome", "resolved_at"]
        )

    frame = pd.DataFrame(
        [
            {
                "league": row.league,
                "market": row.market,
                "probability": row.probability,
                "actual_outcome": 1 if row.outcome == "WON" else 0,
                "resolved_at": row.resolved_at.isoformat() if row.resolved_at else None,
            }
            for row in rows
        ]
    )
    return frame


def compute_bucket_ece_by_context(resolved_predictions: pd.DataFrame) -> Dict[str, float]:
    """Aggregate ECE into league/market bandit buckets."""
    required = {"league", "market", "probability", "actual_outcome"}
    if resolved_predictions.empty or not required.issubset(set(resolved_predictions.columns)):
        return {}

    valid = resolved_predictions.copy()
    valid["league"] = valid["league"].astype(str).str.upper()
    valid["market_bucket"] = valid["market"].map(normalize_market_bucket)
    valid["probability"] = pd.to_numeric(valid["probability"], errors="coerce")
    valid["actual_outcome"] = pd.to_numeric(valid["actual_outcome"], errors="coerce")
    valid = valid.dropna(subset=["league", "market_bucket", "probability", "actual_outcome"])
    valid = valid[valid["probability"].between(0.0, 1.0, inclusive="both")]
    if valid.empty:
        return {}

    ece_by_context: Dict[str, float] = {}
    for (league, market_bucket), group in valid.groupby(["league", "market_bucket"]):
        probs = group["probability"].to_numpy(dtype=float)
        actuals = group["actual_outcome"].to_numpy(dtype=float)
        context = _build_context_key(str(league), str(market_bucket))
        ece_by_context[context] = float(calculate_ece(actuals, probs, n_bins=ECE_BINS))
    return ece_by_context


class ContextualBandit:
    """Maintains per-context shadow weights for simulator experiments."""

    def __init__(
        self,
        state_path: Path = DEFAULT_STATE_PATH,
        *,
        alpha: float = DEFAULT_ALPHA,
        auto_load: bool = True,
        auto_bootstrap: bool = True,
    ) -> None:
        self.state_path = state_path
        self.alpha = float(alpha)
        self.state: Dict[str, Dict[str, Any]] = {}

        if auto_load and self.state_path.exists():
            self.load(self.state_path)
        elif auto_bootstrap:
            self.refresh_from_resolved_predictions()
            self.save(self.state_path)

    def context_key(self, league: str, market: str) -> str:
        return _build_context_key(league, market)

    def get_weights(self, context_key: str) -> Dict[str, float]:
        bucket = self.state.get(context_key)
        if bucket is None:
            return dict(DEFAULT_WEIGHTS)

        weights = bucket.get("weights", {})
        return {
            "tempo_sigma_scale": float(weights.get("tempo_sigma_scale", 1.0)),
            "lambda_scale_home": float(weights.get("lambda_scale_home", 1.0)),
            "lambda_scale_away": float(weights.get("lambda_scale_away", 1.0)),
        }

    def update(self, context_key: str, ece: float) -> None:
        bucket = self.state.setdefault(context_key, _default_bucket_state())
        reward = -float(ece)
        prev_reward_ema = float(bucket.get("reward_ema", 0.0))
        prev_ece_ema = float(bucket.get("ece_ema", 0.0))

        reward_ema = ((1.0 - self.alpha) * prev_reward_ema) + (self.alpha * reward)
        ece_ema = ((1.0 - self.alpha) * prev_ece_ema) + (self.alpha * float(ece))

        bucket["count"] = int(bucket.get("count", 0)) + 1
        bucket["last_ece"] = float(ece)
        bucket["last_reward"] = reward
        bucket["reward_ema"] = reward_ema
        bucket["ece_ema"] = ece_ema
        bucket["weights"] = {
            "tempo_sigma_scale": _clip_scale(1.0 + reward_ema),
            "lambda_scale_home": _clip_scale(1.0 + (reward_ema / 2.0)),
            "lambda_scale_away": _clip_scale(1.0 + (reward_ema / 2.0)),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "alpha": self.alpha,
            "contexts": self.state,
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    def load(self, path: Path) -> None:
        if not path.exists():
            return

        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)

        if isinstance(raw, dict) and "contexts" in raw:
            contexts = raw.get("contexts", {})
            alpha = raw.get("alpha")
            if isinstance(alpha, (int, float)):
                self.alpha = float(alpha)
        elif isinstance(raw, dict):
            contexts = raw
        else:
            contexts = {}

        self.state = {}
        for context_key, bucket in contexts.items():
            if not isinstance(bucket, dict):
                continue
            normalized = _default_bucket_state()
            normalized["count"] = int(bucket.get("count", 0))
            normalized["last_ece"] = bucket.get("last_ece")
            normalized["last_reward"] = bucket.get("last_reward")
            normalized["ece_ema"] = float(bucket.get("ece_ema", 0.0))
            normalized["reward_ema"] = float(bucket.get("reward_ema", 0.0))
            weights = bucket.get("weights", {})
            if isinstance(weights, dict):
                normalized["weights"] = {
                    "tempo_sigma_scale": float(weights.get("tempo_sigma_scale", 1.0)),
                    "lambda_scale_home": float(weights.get("lambda_scale_home", 1.0)),
                    "lambda_scale_away": float(weights.get("lambda_scale_away", 1.0)),
                }
            self.state[str(context_key)] = normalized

    def refresh_from_resolved_predictions(self) -> Dict[str, float]:
        """Load history, compute ECE per bucket, and update state."""
        resolved = load_resolved_predictions_from_db(ServiceContainer.get_instance())
        ece_by_context = compute_bucket_ece_by_context(resolved)
        if not ece_by_context:
            logger.info("RL bandit cold start found no resolved prediction buckets to train.")
            return {}

        for context_key, ece in ece_by_context.items():
            self.update(context_key, ece)

        logger.info(
            "RL bandit refreshed %s context bucket(s) from resolved predictions.",
            len(ece_by_context),
        )
        return ece_by_context
