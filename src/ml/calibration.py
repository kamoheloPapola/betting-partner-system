"""
Market Calibration Module.

Implements betting-safe confidence calibration with:
1. Isotonic Regression (primary) with Platt Scaling fallback
2. Sharpness gates (AUC, STD)
3. Explicit ECE validation

Core principle: The model predicts likelihood, the pipeline decides trust.
"""
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.config import DATA_DIR, MODELS_DIR

logger = logging.getLogger(__name__)

# === CALIBRATION CONSTANTS ===
MIN_CAL_SAMPLES = 200       # Minimum samples for isotonic regression
MIN_UNIQUE_PROBS = 20       # Minimum unique probability values
ECE_BINS = 10               # Number of bins for ECE calculation
ECE_THRESHOLD = 0.03        # Maximum acceptable ECE (3%)

# === SHARPNESS GATE CONSTANTS ===
MIN_AUC = 0.55              # Minimum ROC-AUC for discrimination
MIN_STD = 0.06              # Minimum probability std for sharpness

# === PATHS ===
CALIBRATOR_DIR = MODELS_DIR / "calibrators"
DAMPENING_ALPHA_FILE = DATA_DIR / "calibration" / "dampening_alphas.json"
DEFAULT_DAMPENING_ALPHA = 0.80
DAMPENING_ALPHA_STEP = 0.05

_dampening_alphas: Dict[str, float] = {}
_dampening_alphas_loaded = False
_raw_probability_mode_logged = False


def _ensure_calibrator_directory(path: Path) -> None:
    """Create a calibrator output directory when a save operation needs it."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(
            f"Failed to create calibrator directory '{path}': {exc}"
        ) from exc


def log_raw_probability_mode_once() -> None:
    """Log raw-probability mode once when no calibrator artifacts are configured."""
    global _raw_probability_mode_logged

    if _raw_probability_mode_logged:
        return
    has_calibrator_artifacts = any(
        path.is_file()
        for pattern in ("*.pkl", "*.joblib")
        for path in CALIBRATOR_DIR.glob(pattern)
    )
    if not has_calibrator_artifacts:
        logger.info("Calibration: no calibrator artifacts loaded - using raw probabilities")
    _raw_probability_mode_logged = True


def _load_dampening_alphas() -> None:
    """Load persisted per-market dampening alphas once per process."""
    global _dampening_alphas_loaded
    global _dampening_alphas

    if _dampening_alphas_loaded:
        return

    _dampening_alphas = {}
    if DAMPENING_ALPHA_FILE.exists():
        try:
            with open(DAMPENING_ALPHA_FILE, encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                for market, alpha in raw.items():
                    if isinstance(alpha, (int, float)):
                        _dampening_alphas[str(market)] = float(np.clip(alpha, 0.0, 1.0))
        except Exception as e:
            logger.warning(f"Failed to load dampening alpha file {DAMPENING_ALPHA_FILE}: {e}")

    _dampening_alphas_loaded = True


def _save_dampening_alphas() -> None:
    DAMPENING_ALPHA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DAMPENING_ALPHA_FILE, "w", encoding="utf-8") as f:
        json.dump(_dampening_alphas, f, indent=2, sort_keys=True)


def get_dampening_alpha(market: str) -> float:
    """Return current dampening alpha for a market."""
    _load_dampening_alphas()
    return float(_dampening_alphas.get(market, DEFAULT_DAMPENING_ALPHA))


def set_dampening_alpha(market: str, alpha: float) -> float:
    """Set and persist dampening alpha for a market."""
    _load_dampening_alphas()
    bounded = float(np.clip(alpha, 0.0, 1.0))
    _dampening_alphas[market] = bounded
    _save_dampening_alphas()
    return bounded


def adjust_dampening_alpha(
    market: str,
    direction: str,
    step: float = DAMPENING_ALPHA_STEP,
) -> Tuple[float, float]:
    """
    Nudge market dampening alpha toward 1.0 (less dampening) or 0.0 (more).

    Returns:
        Tuple of (old_alpha, new_alpha).
    """
    old_alpha = get_dampening_alpha(market)
    if direction == "less_dampening":
        new_alpha = min(1.0, old_alpha + step)
    elif direction == "more_dampening":
        new_alpha = max(0.0, old_alpha - step)
    else:
        raise ValueError(f"Unknown dampening direction: {direction}")

    if new_alpha != old_alpha:
        set_dampening_alpha(market, new_alpha)
    return old_alpha, new_alpha


def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = ECE_BINS) -> float:
    """
    Calculate Expected Calibration Error.
    
    ECE = Σ (|bin_count|/n) * |avg_confidence - avg_accuracy|
    """
    if len(y_true) == 0:
        return 0.0
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        bin_count = bin_mask.sum()
        
        if bin_count == 0:
            continue
        
        bin_confidence = y_prob[bin_mask].mean()
        bin_accuracy = y_true[bin_mask].mean()
        bin_weight = bin_count / len(y_true)
        
        ece += bin_weight * abs(bin_confidence - bin_accuracy)
    
    return ece


def apply_binary_calibrator(calibrator: Any, calibrator_type: str, p_raw: np.ndarray) -> np.ndarray:
    """Apply a fitted binary calibrator to raw probabilities."""
    p = np.asarray(p_raw, dtype=float)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    if calibrator_type == "isotonic":
        return np.asarray(calibrator.predict(p), dtype=float)
    if calibrator_type == "platt":
        return np.asarray(calibrator.predict_proba(p.reshape(-1, 1))[:, 1], dtype=float)
    raise ValueError(f"Unknown calibrator_type: {calibrator_type}")


def fit_best_binary_calibrator(
    p_raw: np.ndarray,
    y_true: np.ndarray,
) -> Optional[Dict[str, Any]]:
    """
    Fit isotonic + Platt on validation data and select the lower-ECE calibrator.

    Returns None when calibration cannot be fit reliably.
    """
    probs = np.asarray(p_raw, dtype=float).reshape(-1)
    labels = np.asarray(y_true, dtype=int).reshape(-1)

    if probs.size == 0 or labels.size == 0 or probs.size != labels.size:
        return None

    mask = np.isfinite(probs) & np.isfinite(labels)
    probs = probs[mask]
    labels = labels[mask]
    if probs.size == 0:
        return None

    # Binary calibrators need both classes.
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        return None

    candidates: List[Dict[str, Any]] = []

    # Candidate 1: Isotonic Regression.
    try:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(probs, labels)
        p_iso = apply_binary_calibrator(iso, "isotonic", probs)
        candidates.append(
            {
                "calibrator": iso,
                "type": "isotonic",
                "ece": float(calculate_ece(labels, p_iso)),
            }
        )
    except Exception as exc:
        logger.debug("Isotonic calibration fit failed: %s", exc)

    # Candidate 2: Platt Scaling.
    try:
        platt = LogisticRegression(solver="lbfgs", max_iter=1000)
        platt.fit(probs.reshape(-1, 1), labels)
        p_platt = apply_binary_calibrator(platt, "platt", probs)
        candidates.append(
            {
                "calibrator": platt,
                "type": "platt",
                "ece": float(calculate_ece(labels, p_platt)),
            }
        )
    except Exception as exc:
        logger.debug("Platt calibration fit failed: %s", exc)

    if not candidates:
        return None

    # Lower ECE wins; ties prefer isotonic for monotonic non-parametric flexibility.
    candidates.sort(key=lambda item: (item["ece"], 0 if item["type"] == "isotonic" else 1))
    winner = candidates[0]
    winner["n_samples"] = int(probs.size)
    winner["fitted_at"] = datetime.now().isoformat()
    return winner


def save_binary_calibrator_artifact(path: Path, payload: Dict[str, Any]) -> None:
    """Persist a calibrated post-hoc artifact to disk."""
    _ensure_calibrator_directory(path.parent)
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)


def load_binary_calibrator_artifact(path: Path) -> Optional[Dict[str, Any]]:
    """Load a post-hoc calibration artifact from disk."""
    log_raw_probability_mode_once()
    if not path.exists():
        return None
    try:
        with open(path, "rb") as handle:
            data = pickle.load(handle)
        if isinstance(data, dict) and "calibrator" in data and "type" in data:
            return data
    except Exception as exc:
        logger.warning("Failed to load calibrator artifact %s: %s", path, exc)
    return None


class MarketCalibrator:
    """
    Per-market isotonic regression calibrator with Platt fallback.
    
    Usage:
        calibrator = MarketCalibrator()
        calibrator.fit('goals_u25', p_raw_train, y_train)
        p_cal = calibrator.calibrate('goals_u25', p_raw)
    """
    
    def __init__(self):
        self.calibrators: Dict[str, Any] = {}
        self.calibrator_types: Dict[str, str] = {}  # 'isotonic' or 'platt'
        self.disabled_markets: Set[str] = set()
        self.validation_metrics: Dict[str, Dict[str, float]] = {}
        self.fit_timestamps: Dict[str, str] = {}
    
    def fit(self, market: str, p_raw: np.ndarray, y_true: np.ndarray) -> bool:
        """
        Fit calibrator for a specific market.
        
        Returns True if fit succeeded, False if market was disabled.
        """
        p_raw = np.asarray(p_raw)
        y_true = np.asarray(y_true).astype(int)
        
        # Guard: Probability rank uniqueness (isotonic requires some spread).
        n_unique = np.unique(p_raw).size
        if n_unique < MIN_UNIQUE_PROBS:
            logger.warning(
                "%s: low unique probabilities (%s < %s). Platt may be selected.",
                market,
                n_unique,
                MIN_UNIQUE_PROBS,
            )

        best = fit_best_binary_calibrator(p_raw=p_raw, y_true=y_true)
        if best is None:
            logger.error("%s: calibration fit failed - market disabled", market)
            self.disabled_markets.add(market)
            return False

        self.calibrators[market] = best["calibrator"]
        self.calibrator_types[market] = str(best["type"])
        self.fit_timestamps[market] = str(best.get("fitted_at") or datetime.now().isoformat())
        p_cal = apply_binary_calibrator(self.calibrators[market], self.calibrator_types[market], p_raw)
        self.validation_metrics[market] = self._compute_validation(p_cal, y_true)
        logger.info(
            "%s: selected %s calibrator (n=%s, ECE=%.4f)",
            market,
            self.calibrator_types[market],
            len(p_raw),
            self.validation_metrics[market]["ece"],
        )
        return True
    
    def _fit_platt(self, market: str, p_raw: np.ndarray, y_true: np.ndarray) -> bool:
        """Fallback: Platt scaling (logistic regression on raw probs)."""
        try:
            # Reshape for sklearn
            X = p_raw.reshape(-1, 1)
            
            platt = LogisticRegression(solver='lbfgs', max_iter=1000)
            platt.fit(X, y_true)
            
            self.calibrators[market] = platt
            self.calibrator_types[market] = "platt"
            self.fit_timestamps[market] = datetime.now().isoformat()
            
            p_cal = platt.predict_proba(X)[:, 1]
            self.validation_metrics[market] = self._compute_validation(p_cal, y_true)
            
            logger.info(f"{market}: Platt calibrator fit (n={len(p_raw)}, ECE={self.validation_metrics[market]['ece']:.4f})")
            return True
            
        except Exception as e:
            logger.error(f"{market}: Platt fit also failed - {e}")
            self.disabled_markets.add(market)
            return False
    
    def _compute_validation(self, p_cal: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """Compute validation metrics for calibrated probabilities."""
        ece = calculate_ece(y_true, p_cal)
        
        # Reliability slope (should be ~1 for perfect calibration)
        bins = np.linspace(0, 1, ECE_BINS + 1)
        bin_centers = []
        bin_hit_rates = []
        
        for i in range(ECE_BINS):
            mask = (p_cal >= bins[i]) & (p_cal < bins[i + 1])
            if mask.sum() >= 10:  # Need enough samples
                bin_centers.append(p_cal[mask].mean())
                bin_hit_rates.append(y_true[mask].mean())
        
        if len(bin_centers) >= 3:
            slope = np.polyfit(bin_centers, bin_hit_rates, 1)[0]
        else:
            slope = 0.0
        
        return {
            'ece': ece,
            'reliability_slope': slope,
            'mean_conf': float(p_cal.mean()),
            'mean_hit': float(y_true.mean()),
            'std': float(np.std(p_cal)),
            'n_samples': len(p_cal)
        }
    
    def calibrate(self, market: str, p_raw: float) -> Optional[float]:
        """
        Apply calibration to raw probability.
        
        Returns None if market is disabled.
        """
        if market in self.disabled_markets:
            return None
        
        if market not in self.calibrators:
            logger.warning(f"{market}: No calibrator - returning raw")
            return p_raw
        
        cal = self.calibrators[market]
        cal_type = self.calibrator_types[market]
        try:
            return float(apply_binary_calibrator(cal, cal_type, np.array([p_raw]))[0])
        except Exception as exc:
            logger.warning("%s: calibration apply failed (%s) - returning raw", market, exc)
            return p_raw
    
    def is_market_disabled(self, market: str) -> bool:
        """Check if a market has been disabled due to calibration failure."""
        return market in self.disabled_markets

    def has_calibrator(self, market: str) -> bool:
        """Check whether a calibrator exists for a market."""
        return market in self.calibrators
    
    def get_validation(self, market: str) -> Optional[Dict[str, float]]:
        """Get validation metrics for a market."""
        return self.validation_metrics.get(market)
    
    def save(self, version: str = "v1"):
        """Save all calibrators to disk."""
        _ensure_calibrator_directory(CALIBRATOR_DIR)
        for market, cal in self.calibrators.items():
            path = CALIBRATOR_DIR / f"{market}_{version}.pkl"
            with open(path, 'wb') as f:
                pickle.dump({
                    'calibrator': cal,
                    'type': self.calibrator_types[market],
                    'timestamp': self.fit_timestamps.get(market),
                    'validation': self.validation_metrics.get(market)
                }, f)
            logger.info(f"Saved calibrator: {path}")
        
        # Save disabled markets list
        disabled_path = CALIBRATOR_DIR / f"disabled_markets_{version}.json"
        with open(disabled_path, 'w') as f:
            json.dump(list(self.disabled_markets), f)
    
    def load(self, version: str = "v1"):
        """Load calibrators from disk."""
        for path in CALIBRATOR_DIR.glob(f"*_{version}.pkl"):
            market = path.stem.replace(f"_{version}", "")
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                self.calibrators[market] = data['calibrator']
                self.calibrator_types[market] = data['type']
                self.fit_timestamps[market] = data.get('timestamp')
                self.validation_metrics[market] = data.get('validation', {})
                logger.info(f"Loaded calibrator: {market}")
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")
        
        # Load disabled markets
        disabled_path = CALIBRATOR_DIR / f"disabled_markets_{version}.json"
        if disabled_path.exists():
            with open(disabled_path) as f:
                self.disabled_markets = set(json.load(f))


class SharpnessGate:
    """
    Disable markets that fail discrimination tests.
    
    A market is disabled if:
    - ROC-AUC < 0.55 (no discrimination)
    - Probability std < 0.06 (spiky, not discriminating)
    """
    
    def __init__(self, min_auc: float = MIN_AUC, min_std: float = MIN_STD):
        self.min_auc = min_auc
        self.min_std = min_std
        self.market_status: Dict[str, str] = {}
        self.market_metrics: Dict[str, Dict[str, float]] = {}
    
    def check_market(self, market: str, p_cal: np.ndarray, y_true: np.ndarray) -> str:
        """
        Check if market passes sharpness gate.
        
        Returns 'ACTIVE' or 'DISABLED'.
        """
        p_cal = np.asarray(p_cal)
        y_true = np.asarray(y_true).astype(int)
        
        # Calculate metrics
        try:
            auc = roc_auc_score(y_true, p_cal)
        except Exception:
            auc = 0.5  # Default if calculation fails
        
        std = float(np.std(p_cal))
        
        self.market_metrics[market] = {
            'auc': auc,
            'std': std,
            'n_samples': len(p_cal)
        }
        
        # Gate logic
        if auc < self.min_auc:
            logger.warning(f"{market}: AUC={auc:.3f} < {self.min_auc} - DISABLED")
            self.market_status[market] = "DISABLED"
            return "DISABLED"
        
        if std < self.min_std:
            logger.warning(f"{market}: STD={std:.4f} < {self.min_std} - DISABLED")
            self.market_status[market] = "DISABLED"
            return "DISABLED"
        
        logger.info(f"{market}: ACTIVE (AUC={auc:.3f}, STD={std:.4f})")
        self.market_status[market] = "ACTIVE"
        return "ACTIVE"
    
    def get_status(self, market: str) -> str:
        """Get current status of a market."""
        return self.market_status.get(market, "UNKNOWN")
    
    def get_metrics(self, market: str) -> Optional[Dict[str, float]]:
        """Get sharpness metrics for a market."""
        return self.market_metrics.get(market)


# === SINGLETON INSTANCES ===
_calibrator: Optional[MarketCalibrator] = None
_sharpness_gate: Optional[SharpnessGate] = None


def get_calibrator() -> MarketCalibrator:
    """Get singleton calibrator instance."""
    global _calibrator
    if _calibrator is None:
        _calibrator = MarketCalibrator()
        try:
            _calibrator.load()
        except Exception:
            pass
    return _calibrator


def get_sharpness_gate() -> SharpnessGate:
    """Get singleton sharpness gate instance."""
    global _sharpness_gate
    if _sharpness_gate is None:
        _sharpness_gate = SharpnessGate()
    return _sharpness_gate


def apply_soft_cap(prob: float, confidence: float, market: str) -> float:
    """
    Apply confidence-conditioned soft ceiling to volatile markets (Cards/Corners).
    
    Principles:
    - Allows honest differentiation even at low confidence.
    - Differentiated ceilings: Corners (0.15), Cards (0.18).
    - Clamped to [0.72, 0.90] (Strict order: Compute → Clamp → Compare).
    - Case-insensitive defensive matching for CLI/Strategy compatibility.
    """
    m = market.lower()
    if m.startswith(("corn", "corner")):
        raw_ceiling = 0.75 + 0.15 * confidence
    elif m.startswith(("card", "cards")):
        raw_ceiling = 0.70 + 0.18 * confidence
    else:
        return prob
        
    # Strict clamping order
    ceiling = min(max(raw_ceiling, 0.72), 0.82)
    
    if prob > ceiling:
        logger.info(
            f"[CAL] Soft cap applied | market={market} "
            f"p={prob:.3f} → {ceiling:.3f} (conf={confidence:.2f})"
        )
        return ceiling
            
    return prob
