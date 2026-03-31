"""
Feature Drift Monitor.

Compares live prediction input snapshots against training baselines
to detect distribution shifts (Feature Drift).
"""
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import MODELS_DIR

logger = logging.getLogger(__name__)

class FeatureDriftMonitor:
    """
    Monitors live feature inputs against training baselines using Z-Scores.
    """
    BASELINE_FILE = MODELS_DIR / "feature_baselines.json"
    
    def __init__(self, z_score_threshold: float = 3.0, structural_drift_threshold: float = 0.05):
        """
        Args:
            z_score_threshold: The Z-score absolute value above which a feature is considered flagged.
            structural_drift_threshold: The fraction of features that must be flagged to trigger a CRITICAL structural drift alert.
        """
        self.z_score_threshold = z_score_threshold
        self.structural_drift_threshold = structural_drift_threshold
        self.baselines: Dict[str, Dict[str, float]] = {}
        self._load_baselines()

    def _load_baselines(self) -> None:
        if not self.BASELINE_FILE.exists():
            logger.warning(f"Drift Monitor: Baseline file not found at {self.BASELINE_FILE}. Drift detection is disabled.")
            return
            
        try:
            with open(self.BASELINE_FILE, "r") as f:
                self.baselines = json.load(f)
            logger.info(f"Drift Monitor: Loaded baselines for {len(self.baselines)} features.")
        except Exception as e:
            logger.error(f"Drift Monitor: Failed to load baselines: {e}")

    def check_live_prediction(self, input_snapshot: Dict[str, Any], context_label: str = "Unknown") -> Dict[str, Any]:
        """
        Evaluates an individual live prediction snapshot for feature drift.
        
        Args:
            input_snapshot: Dictionary mapping feature names to their live values.
            context_label: A label (like Match ID or League) for logging purposes.
            
        Returns:
            A dictionary containing drift analysis results.
        """
        if not self.baselines:
            return {"status": "DISABLED", "flagged_features": []}
            
        flagged_features: List[Dict[str, Any]] = []
        valid_features_checked = 0
        
        for feature, baseline in self.baselines.items():
            if feature not in input_snapshot:
                continue
                
            val = input_snapshot[feature]
            if val is None or not isinstance(val, (int, float)):
                continue
                
            mean = baseline.get("mean", 0.0)
            std = baseline.get("std", 0.0)
            
            if std == 0.0:
                # If a feature has 0 standard deviation in training, it's a constant.
                # If it changes at runtime, that's an immediate structural drift.
                if not math.isclose(val, mean, abs_tol=1e-5):
                    flagged_features.append({
                        "feature": feature, 
                        "value": val, 
                        "mean": mean, 
                        "z_score": float('inf')
                    })
                valid_features_checked += 1
                continue
                
            z_score = abs(val - mean) / std
            valid_features_checked += 1
            
            if z_score > self.z_score_threshold:
                flagged_features.append({
                    "feature": feature,
                    "value": round(float(val), 4),
                    "mean": round(mean, 4),
                    "std": round(std, 4),
                    "z_score": round(z_score, 2)
                })
                
        if valid_features_checked == 0:
            return {"status": "NO_FEATURES", "flagged_features": []}
            
        flagged_ratio = len(flagged_features) / valid_features_checked
        status = "OK"
        
        if flagged_ratio > self.structural_drift_threshold:
            status = "CRITICAL"
            logger.error(
                f"[DRIFT CRITICAL] {context_label}: {len(flagged_features)} features ({flagged_ratio:.1%}) "
                f"exceed Z > {self.z_score_threshold}. Structural drift detected!"
            )
        elif len(flagged_features) > 0:
            status = "WARNING"
            logger.debug(
                f"[DRIFT WARNING] {context_label}: {len(flagged_features)} features "
                f"exceed Z > {self.z_score_threshold}."
            )
            
        return {
            "status": status,
            "flagged_ratio": flagged_ratio,
            "flagged_features": flagged_features
        }
