"""
Selection Gate.

Production-grade prediction filter implementing 7 rigorous statistical gates
to enforce discipline and quality control on model outputs.
"""
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from src.config.thresholds import Thresholds
from src.strategies.drift_guard import DriftGuardrail

# Define public API
__all__ = ["SelectionGate"]

logger = logging.getLogger(__name__)

# Default Double Chance baseline (high probability market)
DC_BASELINE_OVERRIDE = 0.70

# Cached drift status keyed by league — avoids repeated checks per league
_DRIFT_STATUS_CACHE: Dict[str, str] = {}
class SelectionGate:
    """
    Production-Grade Selection Gate (v1.0).
    
    Enforces statistical discipline by filtering predictions through 7 rigorous gates:
    1. Model Eligibility
    2. Probability Floor
    3. Edge Validation
    4. Market Drift
    5. Correlation Block
    6. Daily Limit
    7. Confidence Ranking
    """
    
    # Gate 2: Probability Floors (Hard Constraints)
    PROB_FLOORS: Dict[str, float] = {
        'goals': Thresholds.GATE_PROB_GOALS,
        'over_1_5': Thresholds.GATE_PROB_O15,
        'corners': Thresholds.GATE_PROB_CORNERS,
        'cards': Thresholds.GATE_PROB_CARDS,
        'double_chance': Thresholds.GATE_PROB_DC,
        '1x2': Thresholds.GATE_PROB_1X2,
        'default': Thresholds.GATE_PROB_DEFAULT
    }

    # Gate 3: Baselines for Edge Calc (Prob - Baseline)
    # Conservative implied probabilities for "Fair" lines
    BASELINES: Dict[str, float] = {
        'home_win': Thresholds.BASE_HOME_WIN,
        'away_win': Thresholds.BASE_AWAY_WIN,
        'draw': Thresholds.BASE_DRAW,
        'over_2_5': Thresholds.BASE_O25,
        'under_2_5': Thresholds.BASE_U25,
        'btts_yes': Thresholds.BASE_BTTS,
        'double_chance': Thresholds.BASE_DC,
        'home_under_1_5': Thresholds.BASE_TEAM_U15,
        'away_under_1_5': Thresholds.BASE_TEAM_U15,
        'corners_under_11_5': Thresholds.BASE_CORN_U11,
        'cards_over_2_5': Thresholds.BASE_CARD_O25,
        'cards_under_5_5': Thresholds.BASE_CARD_U55
    }
    
    # Gate 6: Daily Limits
    MAX_DAILY_SELECTIONS: int = Thresholds.MAX_DAILY_SELECTIONS
    
    def __init__(self, registry: Optional[Any] = None) -> None:
        self.registry = registry

    def _get_market_type(self, market: str) -> str:
        """Map market name to market type category."""
        m = market.lower()
        if 'card' in m:
            return 'cards'
        if 'corn' in m:
            return 'corners'
        if 'double' in m:
            return 'double_chance'
        if 'over_1_5' in m or 'o1.5' in m:
            return 'over_1_5'
        if m in ['home_win', 'away_win', 'draw']:
            return '1x2'
        if 'goals' in m or 'over' in m or 'under' in m or 'btts' in m:
            return 'goals'
        return 'default'

    def process(self, predictions: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        Run the gauntlet of selection gates.
        
        Args:
            predictions: List of prediction dictionaries.
            
        Returns:
            Tuple of (passed_predictions, rejection_stats).
        """
        passed: List[Dict[str, Any]] = []
        stats: Dict[str, int] = {
            'TOTAL_INPUT': len(predictions),
            'GATE_1_MODEL_ELIGIBILITY': 0,
            'GATE_2_PROB_FLOOR': 0,
            'GATE_3_LOW_EDGE': 0,
            'GATE_4_DRIFT_BLOCKED': 0,
            'GATE_5_CORRELATION_BLOCK': 0,
            'GATE_6_DAILY_LIMIT': 0,
            'GATE_7_CONFIDENCE_CUT': 0,
            'PASSED': 0
        }
        
        # Phase A: Individual Checks
        stage_1 = self._apply_individual_gates(predictions, stats)
            
        # Phase B: Collection Checks
        stage_2 = self._apply_collection_gates(stage_1, stats)
        
        # Gate 6: Daily Exposure Limit
        if len(stage_2) > self.MAX_DAILY_SELECTIONS:
            rejected = stage_2[self.MAX_DAILY_SELECTIONS:]
            stats['GATE_6_DAILY_LIMIT'] += len(rejected)
            
            # Log rejected picks for transparency
            if rejected:
                logger.info(
                    f"Daily limit reached. Rejected {len(rejected)} lower-ranked picks",
                    extra={
                        'limit': self.MAX_DAILY_SELECTIONS,
                        'rejected_matches': [p.get('match_id') for p in rejected[:5]]  # First 5
                    }
                )
            
            passed = stage_2[:self.MAX_DAILY_SELECTIONS]
        else:
            passed = stage_2
            
        stats['PASSED'] = len(passed)
        
        return passed, stats

    def _apply_individual_gates(
        self, 
        predictions: List[Dict[str, Any]], 
        stats: Dict[str, int]
    ) -> List[Dict[str, Any]]:
        """Apply Gates 1-4 (individual prediction checks)."""
        stage_1: List[Dict[str, Any]] = []
        min_edge = Thresholds.GATE_MIN_EDGE
        
        for p in predictions:
            market = p.get('market', 'unknown')
            prob = p.get('probability', 0.0)
            
            # GATE 1: Model Eligibility
            if prob is None or pd.isna(prob):
                stats['GATE_1_MODEL_ELIGIBILITY'] += 1
                p['rejection_reason'] = 'MODEL_ERROR'
                continue
                
            # GATE 2: Probability Floor
            m_type = self._get_market_type(market)
            floor = self.PROB_FLOORS.get(m_type, self.PROB_FLOORS['default'])
            
            if prob < floor:
                stats['GATE_2_PROB_FLOOR'] += 1
                p['rejection_reason'] = f"LOW_CONFIDENCE ({prob:.2f} < {floor})"
                continue
                
            # GATE 3: Edge Validation
            base = self.BASELINES.get(market, 0.50)
            if 'double_chance' in market: 
                base = DC_BASELINE_OVERRIDE
            
            edge = prob - base
            
            if edge < min_edge:
                stats['GATE_3_LOW_EDGE'] += 1
                p['rejection_reason'] = f"LOW_EDGE ({edge:.2%} < {min_edge:.0%})"
                continue
                
            # GATE 4: Market Drift (ECE-based detection)
            # Check if the drift guardrail has flagged degraded model performance
            # Cache is keyed by league — a STOP in one league does not block others.
            global _DRIFT_STATUS_CACHE
            league_key = str(p.get('league', 'GLOBAL'))
            cached = _DRIFT_STATUS_CACHE.get(league_key)
            if cached is None:
                try:
                    from src.config import DATA_DIR
                    league_file = DATA_DIR / "drift" / f"{league_key}_drift_status.json"
                    if not league_file.exists():
                        # No league file — inherit global drift status.
                        drift_guard = DriftGuardrail()
                        global_status = drift_guard.check_drift()
                        cached = global_status
                        logger.debug(
                            f"No league drift file for {league_key} — inheriting global status: {cached}"
                        )
                    else:
                        drift_guard = DriftGuardrail()
                        cached = drift_guard.check_drift(league=league_key)
                    _DRIFT_STATUS_CACHE[league_key] = cached
                except Exception as e:
                    logger.warning(f"Drift check failed for {league_key}: {e}, assuming OK")
                    cached = "OK"
                    _DRIFT_STATUS_CACHE[league_key] = cached
            if cached not in ("OK", "GO", "WATCH", "WARN"):
                # STOP, FAIL, CRITICAL all block predictions for this league only
                stats['GATE_4_DRIFT_BLOCKED'] += 1
                p['rejection_reason'] = f"DRIFT_BLOCKED (league={league_key}, status={cached})"
                continue
            
            # Passed Individual Gates
            p['gate_score'] = prob * (1 + edge)
            stage_1.append(p)
            
        return stage_1

    def _apply_collection_gates(
        self, 
        stage_1: List[Dict[str, Any]], 
        stats: Dict[str, int]
    ) -> List[Dict[str, Any]]:
        """Apply Gates 5-7 (collection-level checks)."""
        # Sort by gate score (Gate 7: Confidence Ranking)
        stage_1.sort(key=lambda x: x['gate_score'], reverse=True)
        
        # GATE 5: Correlation & Duplication (one pick per match)
        seen_matches: Set[str] = set()
        stage_2: List[Dict[str, Any]] = []
        
        for p in stage_1:
            mid = p.get('match_id')
            if mid in seen_matches:
                stats['GATE_5_CORRELATION_BLOCK'] += 1
                continue
            
            seen_matches.add(mid)
            stage_2.append(p)
            
        return stage_2

