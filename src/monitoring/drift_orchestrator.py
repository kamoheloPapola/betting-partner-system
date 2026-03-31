"""
Unified Drift Orchestrator.

Centralizes all drift decisions behind a single STOP/WATCH/GO state machine.
This module merges:
- Global model drift guardrails (hit_rate / ece / mean_conf checks)
- Market confidence drift guardrails (rolling confidence vs hit rate)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from src.config import DATA_DIR
from src.monitoring.telemetry import capture_alert

logger = logging.getLogger(__name__)

__all__ = ["DriftOrchestrator", "get_drift_orchestrator"]


class DriftOrchestrator:
    """
    Single source of truth for drift state and drift gating.

    The only canonical drift states are:
    - GO
    - WATCH
    - STOP
    """

    # --- Paths ---
    DEFAULT_STATUS_FILE: Path = DATA_DIR / "drift" / "rolling_90d_status.json"
    DEFAULT_BASELINE_FILE: Path = DATA_DIR / "models" / "drift_baselines.json"
    DEFAULT_CONFIDENCE_STATE_FILE: Path = DATA_DIR / "drift" / "confidence_drift_state.json"

    # --- Global drift baselines / thresholds (legacy DriftGuardrail semantics) ---
    LEGACY_BASELINES: Dict[str, float] = {
        "hit_rate": 0.75,
        "ece": 0.044,
        "mean_conf": 0.559,
        "selection_rate": 0.03,
    }
    GLOBAL_THRESHOLDS: Dict[str, float] = {
        "hit_rate_stop": -0.05,
        "ece_stop": 0.03,
        "conf_inflation_stop": 0.05,
    }

    # --- Confidence drift thresholds (legacy ConfidenceDriftMonitor semantics) ---
    CONFIDENCE_DRIFT_THRESHOLD = 0.04
    TIER_INVERSION_WINDOW = 300
    VOLUME_P95_MULTIPLIER = 1.5
    COOLDOWN_HOURS = 24
    REFIT_EVERY = 500

    # --- Canonical states ---
    GO = "GO"
    WATCH = "WATCH"
    STOP = "STOP"

    def __init__(
        self,
        *,
        status_file: Optional[Path] = None,
        baseline_file: Optional[Path] = None,
        confidence_state_file: Optional[Path] = None,
    ) -> None:
        self.status_file = Path(status_file) if status_file else self.DEFAULT_STATUS_FILE
        self.baseline_file = Path(baseline_file) if baseline_file else self.DEFAULT_BASELINE_FILE
        self.confidence_state_file = (
            Path(confidence_state_file)
            if confidence_state_file
            else self.DEFAULT_CONFIDENCE_STATE_FILE
        )

        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        self.confidence_state_file.parent.mkdir(parents=True, exist_ok=True)

        # Global drift state
        self.global_status: str = self.GO
        self.global_alerts: List[str] = []
        self.global_metrics: Dict[str, Any] = {}
        self.global_metrics_included: Set[str] = set()
        self.global_evaluated_at: Optional[str] = None
        self.global_legacy_date: Optional[str] = None

        # Baseline metadata
        self.baseline_metadata: Dict[str, Any] = {}
        self.baselines: Dict[str, float] = self._load_baselines()

        # Confidence drift state
        self.market_status: Dict[str, str] = {}
        self.cooldowns: Dict[str, datetime] = {}
        self.bet_counts: Dict[str, int] = {}
        self.rolling_data: Dict[str, List[Dict[str, Any]]] = {}

        # Load persisted state
        self.load_global_state()
        self.load_confidence_state()

    # ------------------------------------------------------------------
    # Global drift guardrail (legacy DriftGuardrail responsibilities)
    # ------------------------------------------------------------------
    def evaluate_global_drift(self, current_session_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Evaluate global drift and return STOP/WATCH/GO.

        If `current_session_data` is None, reads persisted state only.
        """
        if current_session_data is not None:
            self._evaluate_global_metrics(current_session_data)
            self.persist_global_state()
        else:
            self.load_global_state()
        return self.global_status

    def _evaluate_global_metrics(self, data: Dict[str, Any]) -> None:
        previous_status = self.global_status
        self.global_metrics = dict(data)
        self.global_metrics_included = set(data.keys())
        self.global_alerts = []
        self.global_evaluated_at = None
        self.global_legacy_date = None

        hr = float(data.get("hit_rate", 0.0))
        if hr < (self.baselines["hit_rate"] + self.GLOBAL_THRESHOLDS["hit_rate_stop"]):
            self.global_alerts.append(
                f"HIT_RATE_DRIFT: {hr:.2f} (Baseline {self.baselines['hit_rate']})"
            )

        ece = float(data.get("ece", 0.0))
        if ece > (self.baselines["ece"] + self.GLOBAL_THRESHOLDS["ece_stop"]):
            self.global_alerts.append(
                f"CALIBRATION_DRIFT: {ece:.3f} (Baseline {self.baselines['ece']})"
            )

        mconf = float(data.get("mean_conf", 0.0))
        if mconf > (
            self.baselines["mean_conf"] + self.GLOBAL_THRESHOLDS["conf_inflation_stop"]
        ):
            self.global_alerts.append(
                f"CONFIDENCE_INFLATION: {mconf:.2f} (Baseline {self.baselines['mean_conf']})"
            )

        if self.global_alerts:
            # Current alert set is stop-level by policy.
            self.global_status = self.STOP
        else:
            self.global_status = self.GO

        self._emit_stop_transition(
            scope="global",
            subject="global",
            previous_status=previous_status,
            new_status=self.global_status,
            context={"alerts": list(self.global_alerts)},
        )

    def inspect_global_state(self) -> Dict[str, Any]:
        self.load_global_state()
        note = None
        if self.global_legacy_date and not self.global_evaluated_at:
            note = (
                "Legacy drift files created before the read-side fix may store a "
                "last-access date instead of the true evaluation time."
            )

        return {
            "path": str(self.status_file),
            "exists": self.status_file.exists(),
            "status": self.global_status,
            "alerts": list(self.global_alerts),
            "metrics": dict(self.global_metrics),
            "evaluated_at": self.global_evaluated_at,
            "legacy_date": self.global_legacy_date,
            "note": note,
        }

    def describe_baselines(self) -> Dict[str, Any]:
        return {
            "path": str(self.baseline_file),
            "values": dict(self.baselines),
            "thresholds": dict(self.GLOBAL_THRESHOLDS),
            **self.baseline_metadata,
        }

    def persist_global_state(self) -> None:
        evaluated_at = datetime.now(timezone.utc).isoformat()
        self.global_evaluated_at = evaluated_at
        self.global_legacy_date = None
        payload = {
            "date": evaluated_at[:10],
            "evaluated_at": evaluated_at,
            "status": self.global_status,
            "alerts": self.global_alerts,
            "metrics": self.global_metrics,
        }
        with open(self.status_file, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def load_global_state(self) -> None:
        if not self.status_file.exists():
            self.global_status = self.GO
            self.global_alerts = []
            self.global_metrics = {}
            self.global_metrics_included = set()
            self.global_evaluated_at = None
            self.global_legacy_date = None
            return

        try:
            with open(self.status_file, "r", encoding="utf-8") as handle:
                data = json.load(handle)

            self.global_status = self._normalize_state(data.get("status", self.GO))
            self.global_alerts = data.get("alerts", [])
            self.global_metrics = data.get("metrics", {})
            self.global_metrics_included = set(self.global_metrics.keys())
            self.global_evaluated_at = data.get("evaluated_at")
            legacy_date = data.get("date")
            self.global_legacy_date = (
                legacy_date if legacy_date and not self.global_evaluated_at else None
            )
        except Exception as exc:
            logger.error("Failed to load global drift state: %s", exc)
            # Fail closed for global guardrail
            self.global_status = self.STOP
            self.global_alerts = ["STATUS_LOAD_FAILURE"]
            self.global_metrics = {}
            self.global_metrics_included = set()
            self.global_evaluated_at = None
            self.global_legacy_date = None

    # ------------------------------------------------------------------
    # Confidence drift monitor (legacy ConfidenceDriftMonitor responsibilities)
    # ------------------------------------------------------------------
    def record_bet(self, market: str, confidence: float, hit: bool, tier: str) -> None:
        if market not in self.rolling_data:
            self.rolling_data[market] = []

        self.rolling_data[market].append(
            {
                "conf": confidence,
                "hit": 1 if hit else 0,
                "tier": tier,
                "timestamp": datetime.now().isoformat(),
            }
        )
        if len(self.rolling_data[market]) > 1000:
            self.rolling_data[market] = self.rolling_data[market][-1000:]

        self.bet_counts[market] = self.bet_counts.get(market, 0) + 1

        if self.bet_counts[market] % 50 == 0:
            self.persist_confidence_state()

    def evaluate_market_drift(self, market: str) -> str:
        """
        Evaluate market confidence drift and return STOP/WATCH/GO.
        """
        previous_status = self.market_status.get(market)
        if not self._is_cooldown_expired(market):
            self.market_status[market] = self.STOP
            self._emit_stop_transition(
                scope="market",
                subject=market,
                previous_status=previous_status,
                new_status=self.STOP,
                context={"reason": "cooldown_active"},
            )
            logger.warning("%s: cooldown active - STOP", market)
            return self.STOP

        data = self.rolling_data.get(market, [])
        if len(data) < 100:
            self.market_status[market] = self.WATCH
            return self.WATCH

        recent = data[-300:]
        mean_conf = float(np.mean([d["conf"] for d in recent]))
        hit_rate = float(np.mean([d["hit"] for d in recent]))
        drift = mean_conf - hit_rate

        if drift > self.CONFIDENCE_DRIFT_THRESHOLD:
            logger.error(
                "%s: OVERCONFIDENCE DRIFT | conf=%.3f hit=%.3f drift=%.3f",
                market,
                mean_conf,
                hit_rate,
                drift,
            )
            self.market_status[market] = self.STOP
            self._emit_stop_transition(
                scope="market",
                subject=market,
                previous_status=previous_status,
                new_status=self.STOP,
                context={
                    "reason": "overconfidence_drift",
                    "mean_conf": mean_conf,
                    "hit_rate": hit_rate,
                    "drift": drift,
                },
            )
            return self.STOP

        inversion_status = self.evaluate_tier_inversion(market)
        if inversion_status == self.STOP:
            self.market_status[market] = self.STOP
            self._emit_stop_transition(
                scope="market",
                subject=market,
                previous_status=previous_status,
                new_status=self.STOP,
                context={"reason": "tier_inversion"},
            )
            return self.STOP

        self.market_status[market] = self.GO
        return self.GO

    def evaluate_tier_inversion(self, market: str) -> str:
        data = self.rolling_data.get(market, [])
        if len(data) < self.TIER_INVERSION_WINDOW:
            return self.WATCH

        recent = data[-self.TIER_INVERSION_WINDOW :]
        tier_a = [d for d in recent if d.get("tier") == "TIER_A"]
        tier_b = [d for d in recent if d.get("tier") == "TIER_B"]

        if len(tier_a) < 20 or len(tier_b) < 20:
            return self.WATCH

        tier_a_hit = float(np.mean([d["hit"] for d in tier_a]))
        tier_b_hit = float(np.mean([d["hit"] for d in tier_b]))

        if tier_a_hit < tier_b_hit:
            logger.error("%s: TIER INVERSION | A=%.3f B=%.3f", market, tier_a_hit, tier_b_hit)
            self.market_status[market] = self.STOP
            return self.STOP

        return self.GO

    def evaluate_volume_sanity(self, picks_today: int, historical_p95: int = 10) -> str:
        threshold = int(historical_p95 * self.VOLUME_P95_MULTIPLIER)
        if picks_today > threshold:
            logger.warning("VOLUME THROTTLE: %s picks > %s threshold", picks_today, threshold)
            return self.WATCH
        return self.GO

    def should_refit(self, market: str, ece: float) -> bool:
        bet_count = self.bet_counts.get(market, 0)
        if bet_count > 0 and bet_count % self.REFIT_EVERY == 0:
            if ece > 0.03:
                logger.info("%s: Refit triggered (ECE=%.4f)", market, ece)
                return True
        return False

    def apply_cooldown(self, market: str) -> None:
        previous_status = self.market_status.get(market)
        self.cooldowns[market] = datetime.now() + timedelta(hours=self.COOLDOWN_HOURS)
        self.market_status[market] = self.STOP
        self._emit_stop_transition(
            scope="market",
            subject=market,
            previous_status=previous_status,
            new_status=self.STOP,
            context={
                "reason": "cooldown_applied",
                "cooldown_until": self.cooldowns[market].isoformat(),
            },
        )
        logger.info("%s: cooldown applied until %s", market, self.cooldowns[market].isoformat())
        self.persist_confidence_state()

    def get_market_health(self, market: str) -> Dict[str, Any]:
        data = self.rolling_data.get(market, [])
        if len(data) < 50:
            return {
                "status": self.WATCH,
                "n": len(data),
                "bet_count": self.bet_counts.get(market, 0),
                "cooldown_until": (
                    self.cooldowns[market].isoformat() if market in self.cooldowns else None
                ),
            }

        recent = data[-300:]
        mean_conf = float(np.mean([d["conf"] for d in recent]))
        hit_rate = float(np.mean([d["hit"] for d in recent]))
        return {
            "status": self.market_status.get(market, self.WATCH),
            "n": len(data),
            "mean_conf": mean_conf,
            "hit_rate": hit_rate,
            "drift": mean_conf - hit_rate,
            "bet_count": self.bet_counts.get(market, 0),
            "cooldown_until": (
                self.cooldowns[market].isoformat() if market in self.cooldowns else None
            ),
        }

    def persist_confidence_state(self) -> None:
        state = {
            "market_status": self.market_status,
            "cooldowns": {k: v.isoformat() for k, v in self.cooldowns.items()},
            "bet_counts": self.bet_counts,
            "rolling_data": self.rolling_data,
            "updated": datetime.now().isoformat(),
        }
        with open(self.confidence_state_file, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2)

    def load_confidence_state(self) -> None:
        if not self.confidence_state_file.exists():
            return
        try:
            with open(self.confidence_state_file, "r", encoding="utf-8") as handle:
                state = json.load(handle)

            raw_market_status = state.get("market_status", {})
            if isinstance(raw_market_status, dict):
                self.market_status = {
                    str(market): self._normalize_state(value)
                    for market, value in raw_market_status.items()
                }

            self.bet_counts = state.get("bet_counts", {})

            raw_rolling_data = state.get("rolling_data", {})
            if isinstance(raw_rolling_data, dict):
                self.rolling_data = {
                    str(market): entries
                    for market, entries in raw_rolling_data.items()
                    if isinstance(entries, list)
                }
            else:
                self.rolling_data = {}

            self.cooldowns = {}
            for market, value in state.get("cooldowns", {}).items():
                try:
                    self.cooldowns[str(market)] = datetime.fromisoformat(value)
                except Exception:
                    continue
        except Exception as exc:
            logger.warning("Failed to load confidence drift state: %s", exc)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def detect_structural_drift(self, df: pd.DataFrame, window_days: int = 30) -> Dict[str, float]:
        if df.empty:
            return {"goals_shift": 0.0, "corners_shift": 0.0}

        hist_goals = df["home_score"].mean() + df["away_score"].mean()
        hist_corners = df["home_corners"].mean() + df["away_corners"].mean()

        latest_date = df["date"].max()
        cutoff = latest_date - pd.Timedelta(days=window_days)
        recent_df = df[df["date"] > cutoff]

        if recent_df.empty:
            return {"goals_shift": 0.0, "corners_shift": 0.0}

        recent_goals = recent_df["home_score"].mean() + recent_df["away_score"].mean()
        recent_corners = recent_df["home_corners"].mean() + recent_df["away_corners"].mean()

        goals_shift = abs(recent_goals - hist_goals) / hist_goals if hist_goals > 0 else 0.0
        corners_shift = (
            abs(recent_corners - hist_corners) / hist_corners if hist_corners > 0 else 0.0
        )

        return {
            "goals_shift": goals_shift,
            "corners_shift": corners_shift,
            "hist_goals_avg": hist_goals,
            "recent_goals_avg": recent_goals,
            "sample_size": float(len(recent_df)),
        }

    def _load_baselines(self) -> Dict[str, float]:
        fallback = dict(self.LEGACY_BASELINES)
        self.baseline_metadata = {
            "source": "legacy_fallback",
            "training_date": None,
            "loaded_from_file": False,
        }

        if not self.baseline_file.exists():
            return fallback

        try:
            with open(self.baseline_file, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except Exception as exc:
            logger.warning("Failed to load drift baseline metadata: %s", exc)
            self.baseline_metadata["load_error"] = str(exc)
            return fallback

        nested_metrics = metadata.get("baseline_metrics", {})
        baselines = {
            "hit_rate": float(
                metadata.get(
                    "baseline_hit_rate",
                    nested_metrics.get("hit_rate", fallback["hit_rate"]),
                )
            ),
            "ece": float(
                metadata.get(
                    "baseline_ece",
                    nested_metrics.get("ece", fallback["ece"]),
                )
            ),
            "mean_conf": float(
                metadata.get(
                    "baseline_mean_conf",
                    nested_metrics.get(
                        "mean_confidence",
                        nested_metrics.get("mean_conf", fallback["mean_conf"]),
                    ),
                )
            ),
            "selection_rate": float(
                metadata.get(
                    "baseline_selection_rate",
                    nested_metrics.get("selection_rate", fallback["selection_rate"]),
                )
            ),
        }
        self.baseline_metadata = {
            "source": metadata.get("source", "artifact"),
            "training_date": metadata.get("training_date"),
            "training_window_days": metadata.get("training_window_days"),
            "migrated_at": metadata.get("migrated_at"),
            "loaded_from_file": True,
        }
        return baselines

    def _is_cooldown_expired(self, market: str) -> bool:
        if market not in self.cooldowns:
            return True
        return datetime.now() >= self.cooldowns[market]

    @classmethod
    def _normalize_state(cls, value: Any) -> str:
        raw = str(value).upper()
        if raw in {"GO", "OK", "ACTIVE"}:
            return cls.GO
        if raw in {"WATCH", "WARN", "CAUTION", "INSUFFICIENT_DATA"}:
            return cls.WATCH
        if raw in {"STOP", "FAIL", "CRITICAL", "FROZEN"}:
            return cls.STOP
        return cls.STOP

    def _emit_stop_transition(
        self,
        *,
        scope: str,
        subject: str,
        previous_status: Optional[str],
        new_status: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        if new_status != self.STOP:
            return
        if previous_status == self.STOP:
            return

        payload: Dict[str, Any] = {
            "scope": scope,
            "subject": subject,
            "previous_status": previous_status,
            "new_status": new_status,
        }
        if context:
            payload.update(context)

        capture_alert(
            event_name="drift_status_stop_transition",
            message=f"Drift status transitioned to STOP ({scope}:{subject})",
            level="error",
            context=payload,
        )


_drift_orchestrator: Optional[DriftOrchestrator] = None


def get_drift_orchestrator() -> DriftOrchestrator:
    """Return singleton orchestrator for runtime flow."""
    global _drift_orchestrator
    if _drift_orchestrator is None:
        _drift_orchestrator = DriftOrchestrator()
    return _drift_orchestrator
