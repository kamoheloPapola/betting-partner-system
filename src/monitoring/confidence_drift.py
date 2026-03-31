"""
Confidence drift compatibility module.

The canonical drift state machine now lives in DriftOrchestrator.
This module keeps the previous ConfidenceDriftMonitor/QuantileStratifier APIs.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.config import DATA_DIR
from src.monitoring.drift_orchestrator import DriftOrchestrator, get_drift_orchestrator

logger = logging.getLogger(__name__)

# Backward-compatible module constants
DRIFT_THRESHOLD = DriftOrchestrator.CONFIDENCE_DRIFT_THRESHOLD
TIER_INVERSION_WINDOW = DriftOrchestrator.TIER_INVERSION_WINDOW
VOLUME_P95_MULTIPLIER = DriftOrchestrator.VOLUME_P95_MULTIPLIER
COOLDOWN_HOURS = DriftOrchestrator.COOLDOWN_HOURS
REFIT_EVERY = DriftOrchestrator.REFIT_EVERY

DRIFT_DIR = DATA_DIR / "drift"
DRIFT_DIR.mkdir(parents=True, exist_ok=True)


class ConfidenceDriftMonitor:
    """
    Compatibility adapter over DriftOrchestrator market drift methods.

    Legacy behavior preserved:
    - check_drift(...) returns bool (True=allow, False=blocked)
    """

    def __init__(self) -> None:
        self._orchestrator = get_drift_orchestrator()

    @property
    def market_status(self) -> Dict[str, str]:
        return self._orchestrator.market_status

    @property
    def cooldowns(self) -> Dict[str, Any]:
        return self._orchestrator.cooldowns

    @property
    def bet_counts(self) -> Dict[str, int]:
        return self._orchestrator.bet_counts

    @property
    def rolling_data(self) -> Dict[str, List[Dict[str, Any]]]:
        return self._orchestrator.rolling_data

    def record_bet(self, market: str, confidence: float, hit: bool, tier: str) -> None:
        self._orchestrator.record_bet(market, confidence, hit, tier)

    def check_drift(self, market: str) -> bool:
        status = self._orchestrator.evaluate_market_drift(market)
        return status != DriftOrchestrator.STOP

    def check_tier_inversion(self, market: str) -> bool:
        status = self._orchestrator.evaluate_tier_inversion(market)
        return status != DriftOrchestrator.STOP

    def check_volume_sanity(self, picks_today: int, historical_p95: int = 10) -> bool:
        status = self._orchestrator.evaluate_volume_sanity(
            picks_today,
            historical_p95=historical_p95,
        )
        return status == DriftOrchestrator.GO

    def should_refit(self, market: str, ece: float) -> bool:
        return self._orchestrator.should_refit(market, ece)

    def apply_cooldown(self, market: str) -> None:
        self._orchestrator.apply_cooldown(market)

    def get_market_health(self, market: str) -> Dict[str, Any]:
        health = self._orchestrator.get_market_health(market)
        # Preserve legacy presentation labels.
        status = health.get("status")
        if status == DriftOrchestrator.GO:
            health["status"] = "ACTIVE"
        elif status == DriftOrchestrator.STOP:
            health["status"] = "FROZEN"
        return health

    def _persist_state(self) -> None:
        self._orchestrator.persist_confidence_state()

    def _load_state(self) -> None:
        self._orchestrator.load_confidence_state()


class QuantileStratifier:
    """
    Quantile-based confidence stratification.

    Replaces fixed thresholds with rolling percentiles.
    """

    MIN_ROLLING_SAMPLES = 500

    def __init__(self):
        self.rolling_predictions: Dict[str, List[float]] = {}
        self.global_predictions: Dict[str, List[float]] = {}
        self.percentiles: Dict[str, Dict[str, float]] = {}
        self.percentile_timestamps: Dict[str, str] = {}

    @staticmethod
    def _local_market_key(market: str, league: Optional[str] = None) -> str:
        return f"{league}::{market}" if league else market

    @staticmethod
    def _compute_percentile_dict(data: List[float]) -> Dict[str, float]:
        return {
            "q80": float(np.percentile(data, 80)),
            "q90": float(np.percentile(data, 90)),
            "q95": float(np.percentile(data, 95)),
        }

    def _compute_blended_percentiles(
        self,
        market: str,
        local_data: List[float],
    ) -> Dict[str, float]:
        n_local = len(local_data)
        local_pct = self._compute_percentile_dict(local_data)

        if n_local >= self.MIN_ROLLING_SAMPLES:
            return local_pct

        global_data = self.global_predictions.get(market, [])
        if not global_data:
            return local_pct

        global_pct = self._compute_percentile_dict(global_data)
        if n_local < 100:
            local_weight, global_weight = 0.3, 0.7
        else:
            local_weight, global_weight = 0.7, 0.3

        return {
            key: float((local_weight * local_pct[key]) + (global_weight * global_pct[key]))
            for key in ("q80", "q90", "q95")
        }

    def add_prediction(self, market: str, p_cal: float, league: Optional[str] = None):
        local_key = self._local_market_key(market, league)
        if local_key not in self.rolling_predictions:
            self.rolling_predictions[local_key] = []
        if market not in self.global_predictions:
            self.global_predictions[market] = []

        self.rolling_predictions[local_key].append(p_cal)
        self.global_predictions[market].append(p_cal)

        if len(self.rolling_predictions[local_key]) > 2000:
            self.rolling_predictions[local_key] = self.rolling_predictions[local_key][-2000:]
        if len(self.global_predictions[market]) > 5000:
            self.global_predictions[market] = self.global_predictions[market][-5000:]

    def compute_percentiles(self, market: str, league: Optional[str] = None):
        local_key = self._local_market_key(market, league)
        data = self.rolling_predictions.get(local_key, [])
        if not data:
            return

        self.percentiles[local_key] = self._compute_blended_percentiles(market, data)
        self.percentile_timestamps[local_key] = datetime.now().isoformat()

    def get_tier(self, market: str, p_cal: float, league: Optional[str] = None) -> str:
        local_key = self._local_market_key(market, league)
        data = self.rolling_predictions.get(local_key, [])

        if len(data) < 50:
            return "NO_BET"

        if local_key not in self.percentiles:
            self.compute_percentiles(market, league=league)

        pct = self.percentiles.get(local_key)
        if not pct:
            return "NO_BET"

        if p_cal >= pct["q95"]:
            return "TIER_A"
        if p_cal >= pct["q90"]:
            return "TIER_B"
        if p_cal >= pct["q80"]:
            return "TIER_C"

        return "NO_BET"

    def get_percentiles(self, market: str, league: Optional[str] = None) -> Optional[Dict[str, float]]:
        local_key = self._local_market_key(market, league)
        return self.percentiles.get(local_key)

    def save(self):
        path = DATA_DIR / "calibration" / "percentiles.json"
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "percentiles": self.percentiles,
            "timestamps": self.percentile_timestamps,
            "sample_sizes": {k: len(v) for k, v in self.rolling_predictions.items()},
            "global_sample_sizes": {k: len(v) for k, v in self.global_predictions.items()},
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self):
        path = DATA_DIR / "calibration" / "percentiles.json"

        if path.exists():
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            self.percentiles = data.get("percentiles", {})
            self.percentile_timestamps = data.get("timestamps", {})


_drift_monitor: Optional[ConfidenceDriftMonitor] = None
_stratifier: Optional[QuantileStratifier] = None


def get_drift_monitor() -> ConfidenceDriftMonitor:
    global _drift_monitor
    if _drift_monitor is None:
        _drift_monitor = ConfidenceDriftMonitor()
    return _drift_monitor


def get_stratifier() -> QuantileStratifier:
    global _stratifier
    if _stratifier is None:
        _stratifier = QuantileStratifier()
        try:
            _stratifier.load()
        except Exception:
            pass
    return _stratifier
