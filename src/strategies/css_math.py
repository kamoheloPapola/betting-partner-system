"""
CSS Math Engine.

Implements Combo Survivability Score (CSS) calculations including
market weights, log-space stability, and temporal correlation decay.

CSS Formula:
    CSS = (Product(s_i)) * CorrPenalty * (1 - JointFailureRisk)
"""
import logging
import math
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.config import DATA_DIR
from src.config.thresholds import Thresholds

# Define public API
__all__ = ["calculate_css", "count_correlated_pairs", "refresh_market_weights", "load_market_weights"]

logger = logging.getLogger(__name__)

# Default values for fallback
DEFAULT_VOLATILITY = 0.20
DEFAULT_WEIGHT = 0.85
EPSILON_FLOOR = 1e-10

# Market Stability Constants (Weights)
# Higher = more stable market
MARKET_WEIGHTS: Dict[str, float] = {
    'TG_U1.5': 1.00,
    f'CORNERS_U{Thresholds.CORNERS_U11_LINE}': 0.95,
    f'CORNERS_O{Thresholds.CORNERS_O75_LINE}': 0.90,
    'DC': 0.88,
    '1X2': 0.75,
    f'CARDS_U{Thresholds.CARDS_U55_LINE}': 0.85,
    f'CARDS_O{Thresholds.CARDS_O25_LINE}': 0.80
}
CSS_WEIGHTS_FILE: Path = DATA_DIR / "monitoring" / "css_market_weights.json"
_weights_loaded = False

# Historical Calibrated Volatility (Empirical)
# Mapping market types to historical variance/volatility proxy
# Lower is better. Derived from typical Brier/ECE.
HISTORICAL_VOLATILITY: Dict[str, float] = {
    'TG_U1.5': 0.12,
    f'CORNERS_U{Thresholds.CORNERS_U11_LINE}': 0.15,
    f'CORNERS_O{Thresholds.CORNERS_O75_LINE}': 0.22,
    f'CARDS_U{Thresholds.CARDS_U55_LINE}': 0.18,
    f'CARDS_O{Thresholds.CARDS_O25_LINE}': 0.25,
    '1X2': 0.30
}


def load_market_weights(force: bool = False) -> Dict[str, float]:
    """Load persisted CSS weights and merge over defaults."""
    global _weights_loaded
    if _weights_loaded and not force:
        return MARKET_WEIGHTS

    if CSS_WEIGHTS_FILE.exists():
        try:
            with open(CSS_WEIGHTS_FILE, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for key, val in data.items():
                    if key in MARKET_WEIGHTS and isinstance(val, (int, float)):
                        MARKET_WEIGHTS[key] = float(min(max(val, 0.0), 1.0))
        except Exception as e:
            logger.warning("Failed to load persisted CSS weights from %s: %s", CSS_WEIGHTS_FILE, e)

    _weights_loaded = True
    return MARKET_WEIGHTS


def _save_market_weights() -> None:
    CSS_WEIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CSS_WEIGHTS_FILE, "w", encoding="utf-8") as f:
        json.dump(MARKET_WEIGHTS, f, indent=2, sort_keys=True)


def _normalize_market_label(value: str) -> str:
    return "".join(ch for ch in value.upper() if ch.isalnum())


def _resolve_weight_key(market: str) -> str | None:
    """Map evaluation market labels to css_math MARKET_WEIGHTS keys."""
    norm = _normalize_market_label(market)
    corners_u = f"CORNERS_U{Thresholds.CORNERS_U11_LINE}"
    corners_o = f"CORNERS_O{Thresholds.CORNERS_O75_LINE}"
    cards_u = f"CARDS_U{Thresholds.CARDS_U55_LINE}"
    cards_o = f"CARDS_O{Thresholds.CARDS_O25_LINE}"

    alias_map = {
        "HOMEWIN": "1X2",
        "AWAYWIN": "1X2",
        "DRAW": "1X2",
        "ONEXTWO": "1X2",
        "1X2": "1X2",
        "DOUBLECHANCE": "DC",
        "DC": "DC",
        "TGU15": "TG_U1.5",
        _normalize_market_label(corners_u): corners_u,
        _normalize_market_label(corners_o): corners_o,
        _normalize_market_label(cards_u): cards_u,
        _normalize_market_label(cards_o): cards_o,
    }
    if norm in alias_map:
        return alias_map[norm]

    for key in MARKET_WEIGHTS:
        if _normalize_market_label(key) in norm:
            return key
    return None


def refresh_market_weights(
    market_accuracy: Dict[str, Dict[str, Any]],
    min_settled_bets: int = 200,
) -> Dict[str, float]:
    """
    Re-rank market weights using live market accuracy once sample-size is sufficient.
    """
    load_market_weights()

    scored: Dict[str, Dict[str, float]] = {}
    for market, metrics in market_accuracy.items():
        if not isinstance(metrics, dict):
            continue
        count = int(metrics.get("count", 0))
        accuracy = metrics.get("accuracy")
        if count < min_settled_bets or not isinstance(accuracy, (int, float)):
            continue

        weight_key = _resolve_weight_key(market)
        if not weight_key:
            continue

        bucket = scored.setdefault(weight_key, {"weighted_sum": 0.0, "count": 0.0})
        bucket["weighted_sum"] += float(accuracy) * count
        bucket["count"] += count

    if not scored:
        return {}

    avg_accuracy = [
        (weight_key, vals["weighted_sum"] / vals["count"])
        for weight_key, vals in scored.items()
        if vals["count"] > 0
    ]
    if not avg_accuracy:
        return {}

    avg_accuracy.sort(key=lambda x: x[1], reverse=True)
    high = max(MARKET_WEIGHTS.values())
    low = min(MARKET_WEIGHTS.values())
    span = high - low

    updates: Dict[str, float] = {}
    n = len(avg_accuracy)
    if n == 1:
        updates[avg_accuracy[0][0]] = round(high, 4)
    else:
        for idx, (weight_key, _) in enumerate(avg_accuracy):
            rank_ratio = idx / (n - 1)
            updates[weight_key] = round(high - (span * rank_ratio), 4)

    changed = False
    for weight_key, new_weight in updates.items():
        if MARKET_WEIGHTS.get(weight_key) != new_weight:
            changed = True
            MARKET_WEIGHTS[weight_key] = new_weight

    if changed:
        _save_market_weights()
        logger.info("Refreshed CSS market weights for %s eligible markets", len(updates))

    return updates


def calculate_css(
    legs: List[Dict[str, Any]], 
    corr_count: float = 0.0, 
    simulation_override: bool = False
) -> Tuple[float, Dict[str, float]]:
    """
    Compute the Combo Survivability Score (CSS).
    
    Uses log-space calculation for numerical stability with historical
    volatility lookup and dynamic market weights.
    
    Args:
        legs: List of selection dictionaries with 'confidence' and 'market_name'.
        corr_count: Pre-computed correlation count (from count_correlated_pairs).
        simulation_override: If True, disables volatility/correlation penalties.
        
    Returns:
        Tuple of (css_score, breakdown_dict).
    """
    log_s_product = 0.0
    product_fail_risk = 1.0
    
    for leg in legs:
        p = leg.get('confidence', 0.0)
        m_name = leg.get('market_name', '')
        
        if simulation_override:
            v = 0.0
            w = 1.0
        else:
            # Historical Volatility Lookup
            v_hist = _lookup_volatility(m_name)
            
            # Combine confidence with historical volatility
            v = 0.5 * (1.0 - p) + 0.5 * v_hist
            
            # Determine weight
            w = _lookup_weight(m_name)
        
        # Log-Space Stability Calculation
        # s_i = p * w * (1 - v)
        s_i = max(EPSILON_FLOOR, p * w * (1.0 - v))
        log_s_product += math.log(s_i)
        
        # Joint Failure Risk Component
        if not simulation_override:
            product_fail_risk *= (1.0 - p)

    # Convert back from log space
    product_s_i = math.exp(log_s_product)

    if simulation_override:
        corr_penalty = 1.0
        survivability_boost = 1.0
    else:
        # Temporal Correlation Decay
        corr_penalty = math.exp(-Thresholds.CSS_CORR_DECAY * corr_count)
        survivability_boost = 1.0 - product_fail_risk
    
    # Final CSS
    css = product_s_i * corr_penalty * survivability_boost
    
    return css, {
        "s_i_product": product_s_i,
        "corr_penalty": corr_penalty,
        "survivability": survivability_boost
    }


def _lookup_volatility(market_name: str) -> float:
    """Lookup historical volatility for a market type."""
    for key, val in HISTORICAL_VOLATILITY.items():
        if key in market_name:
            return val
    return DEFAULT_VOLATILITY


def _lookup_weight(market_name: str) -> float:
    """Lookup stability weight for a market type."""
    load_market_weights()
    for key, val in MARKET_WEIGHTS.items():
        if key in market_name:
            return val
    return DEFAULT_WEIGHT


def count_correlated_pairs(legs: List[Dict[str, Any]]) -> float:
    """
    Identify correlated selection pairs with Temporal Decay.
    
    Correlation conditions:
    - Same League: Baseline correlation
    - Temporal Proximity: exp(-0.3 * hours_apart)
    
    Args:
        legs: List of selection dictionaries with 'league' and 'date'.
        
    Returns:
        Total correlation score (float, not integer count).
    """
    total_correlation = 0.0
    n = len(legs)
    
    for i in range(n):
        for j in range(i + 1, n):
            l1 = legs[i]
            l2 = legs[j]
            
            # Correlation Condition: Same League
            if l1.get('league') == l2.get('league'):
                decay = _calculate_temporal_decay(l1.get('date'), l2.get('date'))
                total_correlation += decay
                    
    return total_correlation


def _calculate_temporal_decay(d1: Any, d2: Any) -> float:
    """Calculate temporal decay factor between two dates."""
    try:
        if isinstance(d1, str): 
            d1 = datetime.fromisoformat(d1.replace('Z', '+00:00'))
        if isinstance(d2, str): 
            d2 = datetime.fromisoformat(d2.replace('Z', '+00:00'))
        
        if d1 and d2:
            # Ensure both are naive or both are aware
            if d1.tzinfo != d2.tzinfo:
                d1 = d1.replace(tzinfo=None)
                d2 = d2.replace(tzinfo=None)
                
            diff = abs((d1 - d2).total_seconds()) / 3600.0
            # Decay factor: 1.0 at 0hr, 0.5 at ~2.3hr, 0.1 at ~7hr
            return math.exp(-Thresholds.CSS_TEMPORAL_DECAY * diff)
        else:
            return 1.0  # Fallback if dates missing
    except Exception:
        return 1.0  # Fallback on parse error
