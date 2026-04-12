"""
Model & Prediction Integrity Guard (MPIG).

A hard pre-execution verification layer designed to prevent silent failures, 
data leakage, and stale model usage in production betting systems.

Advanced Usage Patterns:
    1. Pre-Deployment Validation (Strict Mode):
        guard = MPIG(dry_run=False)
        try:
            results = guard.verify_system(matches, markets, 'production')
            # Success: All gates passed
        except IntegrityError as e:
            # Blocked: Manual intervention required

    2. Continuous Monitoring (Non-blocking):
        guard = MPIG(dry_run=True)
        for pred in predictions:
            try:
                guard.verify_prediction(pred)
            except IntegrityError as e:
                # Log issues without stopping the pipeline
                logger.warning(f"Quality Alert: {e}")

    3. Strategy Validation:
        guard = MPIG()
        result = guard.verify_strategy(candidates, 'accumulator')
        if result['verified']:
            # Candidates meet structural requirements
"""

import json
import logging
import hashlib
import time
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.ml.registry import ModelRegistry
from src.strategies.drift_guard import DriftGuardrail
from src.config import MODELS_DIR

logger = logging.getLogger(__name__)

# Configuration for Integrity Gates
GUARD_VERSION = "1.0.0"
MIN_TRAIN_DATE = date(2025, 12, 1)  # Using date object for robust comparison (Issue #6 Fix)
# Default ECE Threshold
MAX_ECE = 0.08
MIN_SAMPLES = 1000  # Stricter for production integrity
MAX_CONFIDENCE = 0.99  # Threshold for suspicious predictions (Issue #3 Fix)

# Per-Model ECE Overrides (for challenging new markets)
ECE_THRESHOLDS = {
    "xgb_double_chance_1x": 0.30,
    "xgb_double_chance_x2": 0.30,
    "xgb_double_chance_12": 0.30
}

class IntegrityError(Exception):
    """Custom exception for integrity failures."""
    pass

class MPIG:
    """
    Model & Prediction Integrity Guard (MPIG)
    Hard pre-execution verification layer.
    """
    
    def __init__(self, dry_run: bool = False):
        self.registry = ModelRegistry()
        self.drift_guard = DriftGuardrail()
        self.dry_run = dry_run
        self.audit_trail = []  # Track all checks performed
        self.results = {}

    def _audit(self, phase: str, status: str, details: Dict[str, Any] = None):
        """Record audit entry."""
        entry = {
            "phase": phase,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        self.audit_trail.append(entry)

    def verify_system(
        self, 
        matches: List[Dict[str, Any]], 
        markets: List[str], 
        strategy_name: str
    ) -> Dict[str, Any]:
        """
        Main entry point for system-level pre-execution verification.
        
        System Verification Phases (executed here):
        - Phase 1: Model Availability - Verify production models exist
        - Phase 2: Model Health - Check freshness, calibration, sample size
        - Phase 6: Drift Detection - Ensure no active drift alerts
        
        Per-Prediction/Strategy Phases (called separately):
        - Phase 3: Provenance - Use verify_prediction() for traceability
        - Phase 4: Sanity Checks - Use verify_prediction() for bounds checking
        - Phase 5: Strategy Compatibility - Use verify_strategy() for requirements
        
        Args:
            matches: List of match dictionaries to verify models for
            markets: List of market types to check
            strategy_name: Name of betting strategy being used
            
        Returns:
            Verification results dictionary with status and audit trail
            
        Raises:
            TypeError: If inputs are not correct types
            ValueError: If inputs are empty or invalid
            IntegrityError: If any system-level check fails
        """
        # 1. Input Validation (Issue #10.1 Fix)
        if not isinstance(matches, list):
            raise TypeError(f"matches must be list, got {type(matches).__name__}")
        if not isinstance(markets, list):
            raise TypeError(f"markets must be list, got {type(markets).__name__}")
        if not matches:
            raise ValueError("matches list cannot be empty")
        if not markets:
            raise ValueError("markets list cannot be empty")
        if not strategy_name or not isinstance(strategy_name, str):
            raise ValueError("strategy_name must be non-empty string")

        logger.info(
            f"MPIG: Starting verification for '{strategy_name}' "
            f"({len(matches)} matches, {len(markets)} markets)"
        )
        
        start_time = time.time()  # Performance metrics (Design B)
        
        # Reset state for each verification run (Issue #10.4 Fix)
        self.audit_trail = []  
        self.results = {
            "verified": False,
            "guard_version": GUARD_VERSION,
            "checks_passed": [],
            "timestamp": datetime.now().isoformat(),
            "strategy": strategy_name,
            "dry_run": self.dry_run,
            "phases": {
                "phase_1_models": "pending",
                "phase_2_health": "pending",
                "phase_6_drift": "pending"
            }
        }
        
        if self.dry_run:
            logger.info("MPIG: Running in DRY-RUN mode (errors won't block)")

        logger.info(f"MPIG: Starting system verification for strategy: {strategy_name}")
        
        try:
            # Phase 1: Model Availability & Matching
            self._verify_models_exist(matches, markets)
            self.results["phases"]["phase_1_models"] = "passed"
            self.results["checks_passed"].append("model_exists")
            self._audit("phase_1_models", "passed", {"verified_pairs": len(self.results.get("verified_pairs", []))})

            # Phase 2: Freshness & Calibration
            self._verify_model_health(matches, markets)
            self.results["phases"]["phase_2_health"] = "passed"
            self.results["checks_passed"].append("calibrated")
            self._audit("phase_2_health", "passed")

            # Phase 6: Pre-use Drift Check
            self._verify_drift_safety()
            self.results["phases"]["phase_6_drift"] = "passed"
            self.results["checks_passed"].append("no_drift")
            self._audit("phase_6_drift", "passed", {"status": self.results.get("drift_status")})

            self.results["verified"] = True
            logger.info("MPIG: All system integrity checks PASSED.")
            
        except IntegrityError as e:
            if self.dry_run:
                logger.warning(f"MPIG DRY-RUN: Would have blocked: {e}")
                self.results["would_block"] = str(e)
                self._audit("verification", "failed_dry_run", {"error": str(e)})
            else:
                self._audit("verification", "failed", {"error": str(e)})
                self.results["duration_seconds"] = time.time() - start_time
                logger.error(f"MPIG: BLOCKED - {e}")
                raise e
        except Exception as e:
            self._audit("verification", "unexpected_error", {"error": str(e)})
            logger.error(f"MPIG: UNEXPECTED FAILURE - {e}")
            raise IntegrityError(f"System Integrity Compromised: {e}")
        
        self.results["duration_seconds"] = time.time() - start_time
        self.results["audit_trail"] = self.audit_trail
        return self.results

    def _verify_models_exist(self, matches: List[Dict[str, Any]], markets: List[str]):
        """Phase 1: Ensure production models exist for every (league, market) pair."""
        # 1. Validate all matches have leagues (Issue #1 Fix)
        matches_without_league = [
            m for m in matches 
            if not (m.get('league') or m.get('competition'))
        ]
        if matches_without_league:
            raise IntegrityError(
                f"{len(matches_without_league)} matches missing league identifier. "
                f"Examples: {matches_without_league[:3]}"
            )

        leagues = set(m.get('league') or m.get('competition') for m in matches)
        verified_pairs = []
        
        for league in leagues:
            for market_type in markets:
                # We check the registry for a production model
                meta = self.registry.get_production_model_for_league(league, market_type)
                
                if not meta:
                    raise IntegrityError(
                        f"No production model for [{league} | {market_type}]. "
                        f"Global fallback not permitted by MPIG."
                    )
                
                verified_pairs.append((league, market_type))
        
        logger.info(f"MPIG: Verified {len(verified_pairs)} league/market pairs")
        self.results["verified_pairs"] = verified_pairs

    def _verify_model_health(self, matches: List[Dict[str, Any]], markets: List[str]):
        """Phase 2: Check model registration date, ECE, and sample size."""
        leagues = set(m.get('league') or m.get('competition') for m in matches)
        
        for league in leagues:
            # We already validated leagues exist in Phase 1
            for market_type in markets:
                meta = self.registry.get_production_model_for_league(league, market_type)
                if not meta:
                     raise IntegrityError(f"Inconsistent state: Model for [{league} | {market_type}] missing in Phase 2")
                
                # Freshness (Robust Date Comparison Issue #6 Fix)
                reg_date_str = meta.get("registered_at", "1970-01-01")
                try:
                    reg_date = date.fromisoformat(reg_date_str)
                except ValueError:
                    logger.warning(f"Invalid date format for model {meta.get('name')}: {reg_date_str}")
                    reg_date = date(1970, 1, 1)

                if reg_date < MIN_TRAIN_DATE:
                    raise IntegrityError(f"Model {meta['name']} is stale (Trained: {reg_date} < {MIN_TRAIN_DATE})")

                # Calibration
                ece = meta.get("metrics", {}).get("calibration_score", 999.0)
                threshold = ECE_THRESHOLDS.get(market_type, MAX_ECE)
                
                if ece > threshold:
                    raise IntegrityError(f"Model {meta['name']} failed calibration gate (ECE: {ece:.4f} > {threshold})")

                # Sample Size
                n = meta.get("train_size", 0) + meta.get("test_size", 0)
                if n < MIN_SAMPLES:
                    raise IntegrityError(f"Model {meta['name']} failed sample size gate (N: {n} < {MIN_SAMPLES})")

    def _verify_drift_safety(self):
        """Phase 6: Ensure no active drift alerts (Issue #9 Fix)."""
        status = self.drift_guard.check_drift()
        
        logger.info(f"MPIG Phase 6: Drift status = {status}")
        self.results["drift_status"] = status
        
        if status == "STOP":
            raise IntegrityError(
                f"Drift Guard triggered STOP. Alerts: {self.drift_guard.alerts}"
            )
        elif status in ("WATCH", "CAUTION"):
            logger.warning(
                f"MPIG Phase 6: Drift Guard shows {status}. "
                f"Alerts: {self.drift_guard.alerts}"
            )
            self.results["drift_warnings"] = self.drift_guard.alerts
        elif status == "OK":
            logger.debug("MPIG Phase 6: No drift detected")
        else:
            logger.warning(f"MPIG Phase 6: Unknown drift status: {status}")

    def verify_prediction(self, prediction: Dict[str, Any]) -> bool:
        """
        Phase 3 & 4: Traceability and Sanity checks for a SINGLE prediction.
        
        - Phase 3: Provenance - Ensure traceability metadata exists
        - Phase 4: Sanity - Bounds check and suspiciously high confidence
        
        Returns:
            True if prediction passes all verification gates.
            
        Raises:
            IntegrityError: If prediction fails any check.
        """
        # Phase 4: Sanity First
        p = prediction.get("confidence") or prediction.get("probability")
        if p is None:
            raise IntegrityError(f"Missing probability for {prediction.get('match', 'unknown')}")
        
        if not (0.0 <= p <= 1.0):
            raise IntegrityError(f"Probability out of bounds: {p} for {prediction.get('match')}")

        # Phase 4b: Leakage/Sanity Check (Statistically Unjustified Confidence)
        if p > MAX_CONFIDENCE:
            raise IntegrityError(
                f"Suspicious Confidence (>{MAX_CONFIDENCE}): {p:.4f} for "
                f"{prediction.get('match', 'unknown')}. Possible data leakage."
            )

        # Phase 3: Provenance (Traceability)
        prov = prediction.get("provenance", {})
        required = ["model_id", "model_hash", "timestamp"]
        for field in required:
            if field not in prov:
                raise IntegrityError(f"Provenance Violation: {field} missing for {prediction.get('match')}")
        
        return True

    def verify_strategy(
        self, 
        candidates: List[Dict[str, Any]], 
        strategy_name: str
    ) -> Dict[str, Any]:
        """
        Phase 5: Strategy Compatibility Check.
        
        Verifies that chosen candidates meet the structural and content 
        requirements of the specific betting strategy.
        
        Returns:
            Verification result dict with status and details.
        """
        result = {
            "strategy": strategy_name,
            "candidate_count": len(candidates),
            "verified": False,
            "checks_performed": []
        }
        
        if not candidates:
            logger.warning(f"MPIG Phase 5: No candidates provided for {strategy_name}")
            result["verified"] = True
            result["note"] = "No candidates to verify"
            return result
        
        try:
            if strategy_name == "forbidden-fruit":
                self._verify_forbidden_fruit_requirements(candidates)
                result["checks_performed"].append("tier_presence")
                
            elif strategy_name == "accumulator":
                self._verify_accumulator_requirements(candidates)
                result["checks_performed"].append("min_count")
                result["checks_performed"].append("field_integrity")
                
            elif strategy_name == "value-hunter":
                # Add verification for value hunter if needed in future
                logger.debug(f"MPIG Phase 5: No specific checks for {strategy_name}")
            else:
                logger.debug(f"MPIG Phase 5: No specific checks for {strategy_name}")
            
            result["verified"] = True
            logger.info(f"MPIG Phase 5: Strategy compatibility verified for {strategy_name}")
            return result
            
        except IntegrityError as e:
            result["error"] = str(e)
            raise e

    def _verify_forbidden_fruit_requirements(self, candidates: List[Dict[str, Any]]):
        """Verify Forbidden Fruit strategy requirements."""
        missing_tier = [c for c in candidates if "tier" not in c]
        if missing_tier:
            raise IntegrityError(
                f"Forbidden Fruit requires tiered candidates. "
                f"{len(missing_tier)}/{len(candidates)} candidates missing 'tier' field. "
                f"Examples: {missing_tier[:3]}"
            )

    def _verify_accumulator_requirements(self, candidates: List[Dict[str, Any]]):
        """Verify Accumulator strategy requirements."""
        # Check minimum candidate count
        if len(candidates) < 2:
            raise IntegrityError(
                f"Accumulator requires at least 2 candidates, got {len(candidates)}"
            )
        
        # Check required fields
        required_fields = ["confidence", "market"]
        for candidate in candidates:
            missing = [f for f in required_fields if f not in candidate]
            if missing:
                raise IntegrityError(
                    f"Accumulator candidate missing required fields: {missing} for "
                    f"match {candidate.get('match', 'unknown')}"
                )

def verify_system(
    matches: List[Dict[str, Any]], 
    markets: List[str], 
    strategy_name: str, 
    dry_run: bool = False
) -> Dict[str, Any]:
    """Convenience wrapper for system-level MPIG."""
    guard = MPIG(dry_run=dry_run)
    return guard.verify_system(matches, markets, strategy_name)

def verify_prediction(
    prediction: Dict[str, Any], 
    dry_run: bool = False
) -> bool:
    """
    Runtime wrapper for single prediction integrity.
    
    Returns:
        True if prediction passes all verification checks.
        
    Raises:
        IntegrityError: If prediction fails any check
    """
    guard = MPIG(dry_run=dry_run)
    return guard.verify_prediction(prediction)

def verify_strategy(
    candidates: List[Dict[str, Any]], 
    strategy_name: str, 
    dry_run: bool = False
) -> Dict[str, Any]:
    """Convenience wrapper for Phase 5 strategy verification."""
    guard = MPIG(dry_run=dry_run)
    return guard.verify_strategy(candidates, strategy_name)
