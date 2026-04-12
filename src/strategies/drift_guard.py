"""
Drift Guardrail compatibility adapter.

All drift state-machine logic now lives in `DriftOrchestrator`.
This class keeps the legacy `DriftGuardrail` API stable for callers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Set

import pandas as pd

from src.config import DATA_DIR
from src.monitoring.drift_orchestrator import DriftOrchestrator

__all__ = ["DriftGuardrail"]


class DriftGuardrail:
    """
    Legacy facade over the unified DriftOrchestrator.

    Public contract preserved:
    - check_drift() -> "OK" | "WATCH" | "STOP"
    - inspect_state(), describe_baselines(), detect_drift()
    """

    STATUS_FILE: Path = DATA_DIR / "drift" / "rolling_90d_status.json"
    BASELINE_FILE: Path = DATA_DIR / "models" / "drift_baselines.json"

    BASELINES: Dict[str, float] = dict(DriftOrchestrator.LEGACY_BASELINES)
    THRESHOLDS: Dict[str, float] = dict(DriftOrchestrator.GLOBAL_THRESHOLDS)

    def __init__(
        self,
        status_file: Optional[Path] = None,
        baseline_file: Optional[Path] = None,
    ) -> None:
        self.status_file: Path = Path(status_file) if status_file else self.STATUS_FILE
        self.baseline_file: Path = Path(baseline_file) if baseline_file else self.BASELINE_FILE

        self.status: str = "OK"
        self.alerts: list[str] = []
        self.metrics: Dict[str, Any] = {}
        self.requires_enforcement: bool = True
        self.metrics_included: Set[str] = set()
        self.evaluated_at: Optional[str] = None
        self.legacy_date: Optional[str] = None
        self.baseline_metadata: Dict[str, Any] = {}

        self._orchestrator = DriftOrchestrator(
            status_file=self.status_file,
            baseline_file=self.baseline_file,
        )
        self._sync_from_orchestrator()

    def check_drift(
        self,
        current_session_data: Optional[Dict[str, Any]] = None,
        league: Optional[str] = None,
    ) -> str:
        """
        Return drift status as "OK" | "WATCH" | "STOP".

        When `league` is provided, evaluates and returns the per-league drift
        status from the league-scoped state file. The global state is not
        touched. When `league` is None, falls back to global evaluation
        (original behaviour preserved).
        """
        if league is not None:
            raw = self._orchestrator.evaluate_league_drift(league, current_session_data)
            return self._to_legacy_status(raw)
        self._orchestrator.evaluate_global_drift(current_session_data)
        self._sync_from_orchestrator()
        return self.status

    def inspect_state(self) -> Dict[str, Any]:
        state = self._orchestrator.inspect_global_state()
        state["status"] = self._to_legacy_status(state.get("status", DriftOrchestrator.GO))
        return state

    def describe_baselines(self) -> Dict[str, Any]:
        return self._orchestrator.describe_baselines()

    def detect_drift(self, df: pd.DataFrame, window_days: int = 30) -> Dict[str, float]:
        return self._orchestrator.detect_structural_drift(df, window_days=window_days)

    def _sync_from_orchestrator(self) -> None:
        self.BASELINES = dict(self._orchestrator.baselines)
        self.baseline_metadata = dict(self._orchestrator.baseline_metadata)
        self.metrics = dict(self._orchestrator.global_metrics)
        self.metrics_included = set(self.metrics.keys())
        self.alerts = list(self._orchestrator.global_alerts)
        self.evaluated_at = self._orchestrator.global_evaluated_at
        self.legacy_date = self._orchestrator.global_legacy_date
        self.status = self._to_legacy_status(self._orchestrator.global_status)

    @staticmethod
    def _to_legacy_status(status: str) -> str:
        if status == DriftOrchestrator.GO:
            return "OK"
        if status == DriftOrchestrator.WATCH:
            return "WATCH"
        return "STOP"
