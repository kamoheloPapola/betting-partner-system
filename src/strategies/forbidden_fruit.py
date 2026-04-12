"""
Forbidden Fruit Strategy Engine.

Market-agnostic decision block implementing tiered selection,
CSS-based slip construction, and drift guardrail enforcement.
This is the core strategy orchestrator for competitive selection.
"""
import itertools
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.config.thresholds import Thresholds
from src.core.constants import MATCH_SEPARATOR
from src.ml.registry import ModelRegistry
from src.strategies.css_math import calculate_css, count_correlated_pairs
from src.strategies.drift_guard import DriftGuardrail
from src.strategies.edge_engine import EdgeEngine
from src.utils.standings import StandingsManager

# === CALIBRATION PIPELINE IMPORTS ===
from src.ml.calibration import get_calibrator, get_sharpness_gate, apply_soft_cap
from src.monitoring.confidence_drift import get_drift_monitor
from src.monitoring.rejection_logger import get_rejection_logger, RejectionReason
from src.ml.confidence import get_confidence_calculator, get_action_tier




# Define public API
__all__ = ["ForbiddenFruitEvaluator", "ForbiddenFruitEngine"]

logger = logging.getLogger(__name__)


def log_prediction(prediction: dict) -> None:
    """Persist a prediction as a JSONL record without affecting prediction flow."""
    try:
        log_path = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "predictions"
            / "predictions_log.jsonl"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": f"{datetime.now(timezone.utc).isoformat()}",
            **prediction,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")
    except Exception as exc:
        logger.warning("Failed to log prediction: %s", exc)

# === Strategy Constants ===
# Gate relaxation for surging underdogs
MOMENTUM_GATE_RELAXATION = 0.06

# Underdog detection thresholds
UNDERDOG_WIN_PROB_MAX = 0.35
UNDERDOG_U15_MIN_CONF = 0.60

# Slip construction constants
MIN_SLIP_PROBABILITY = 0.65
MIN_SLIP_CONFIDENCE = 0.32
MIN_CSS_SCORE = 0.18

# Market-specific confidence thresholds (Option A)
# These markets are structurally noisier; allowing lower entry gates for slip construction.
MIN_SLIP_CONFIDENCE_MAP = {
    "CORNERS_O7.5": 0.25,
    "CARDS_U5.5": 0.28,
}
MAX_CORRELATED_PAIRS = 1

# Market priority weights (higher = preferred in slip construction)
# Based on backtest analysis: CORNERS_U11.5 = 90.9% win rate vs HOME_TG_U1.5 = 37.5%
MARKET_WEIGHTS: Dict[str, float] = {
    'CORNERS_U11.5': 1.15,  # +15% boost (best performing)
    'CORNERS_O7.5': 1.10,   # +10% boost
    'BTTS_NO': 1.10,        # +10% boost (low variance)
    'GOALS_U3.5': 1.10,     # +10% boost (low variance, ~75% hit rate)
    'CARDS_U5.5': 1.05,     # +5% boost (stable)
    'CARDS_U4.5': 1.05,     # +5% boost
    'CARDS_O2.5': 1.00,     # Neutral
    'TG_U1.5': 0.90,        # -10% penalty (underperforming)
}

# League-specific adjustments
LEAGUE_ADJUSTMENTS: Dict[str, Dict[str, float]] = {
    'BL1': {'o15': 0.05, 'u25': -0.05, 'o75': 0.0},
    'FL1': {'o15': 0.06, 'u25': -0.06, 'o75': 0.0},
    'PD': {'o15': -0.02, 'u25': 0.02, 'o75': 0.0}
}

# Valid markets for slip construction
VALID_SLIP_MARKETS = frozenset([
    'TG_U1.5', 'CORNERS_U11.5', 'CORNERS_O7.5',
    'CARDS_O2.5', 'CARDS_U5.5', 'CARDS_U4.5',
    'BTTS_NO', 'GOALS_U3.5'  # New low-variance markets
])

VALUE_MARKET_MAP = {
    'u25': 'goals_u25',
    'o25': 'goals_o25',
    'btts': 'btts_yes',
    'btts_no': 'btts_no',
}


class ForbiddenFruitEvaluator:
    """
    Implements the 'Forbidden Fruit 3.0' Market-Agnostic Decision Block.
    
    Evaluates predictions against quality gates and returns tiered candidates.
    """
    
    def __init__(self) -> None:
        # Configuration: Conservative Integrity Gates
        self.HARD_GATES: Dict[str, float] = {
            '1X2': Thresholds.STRAT_GATE_1X2,
            'CORNERS': Thresholds.STRAT_GATE_CORNERS,
            'TEAM_CORNERS': Thresholds.STRAT_GATE_CORNERS,
            '1X2_CORNERS': Thresholds.STRAT_GATE_1X2,
            'CARDS': Thresholds.STRAT_GATE_CARDS,
            'CARDS_O25': Thresholds.STRAT_GATE_CARDS_O25,
            'UND_U15': Thresholds.STRAT_GATE_UND_U15,
            'BTTS': Thresholds.STRAT_GATE_BTTS,
            'GOALS_O15': Thresholds.STRAT_GATE_GOALS_O15,
            'GOALS_O25': Thresholds.STRAT_GATE_GOALS_O25,
            'GOALS_U25': Thresholds.STRAT_GATE_GOALS_U25,
            'DC_1X': Thresholds.STRAT_GATE_DC,
            'DC_X2': Thresholds.STRAT_GATE_DC,
            'DC_12': Thresholds.STRAT_GATE_DC,
            'DOUBLE_CHANCE': 0.75, # Risk Wrapper Gate
            'CARDS_U55': Thresholds.STRAT_GATE_CARDS_U55,
            'CORNERS_O75': Thresholds.STRAT_GATE_CORNERS_O75,
            'GOALS_U35': 0.70  # Conservative default for U3.5
        }
        self.GLOBAL_MIN_CONF = Thresholds.GLOBAL_MIN_CONF
        
        # Tempo & Optimization Thresholds
        self.TEMPO_THRESHOLDS: Dict[str, float] = {
            'O15_STABLE': Thresholds.TEMPO_O15_STABLE,
            'O15_LOW_RISK': Thresholds.TEMPO_O15_LOW_RISK,
            'O75_HIGH': Thresholds.TEMPO_O75_HIGH,
            'O75_SLOW': Thresholds.TEMPO_O75_SLOW
        }
        
    def evaluate_decision(
        self, 
        preds: Dict[str, float], 
        is_fallback: bool = False, 
        coverage_status: str = "UNKNOWN", 
        league: Optional[str] = None, 
        standings_context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Structural Decision Block.
        
        Returns ALL qualified candidates for a match, tagged with Tier.
        """
        qualified = self.evaluate(preds, is_fallback, coverage_status, league, standings_context)
        if not qualified:
            return []
            
        dissonance = any(m['dissonance'] for m in qualified)
        
        results = []
        for q in qualified:
            # Determine Tier for this specific market
            tier = 3
            is_anchor = q['type'] in ['1X2', '1X2_CORNERS', 'CARDS_O25', 'GOALS_O15', 'GOALS_O25']
            
            if is_anchor and q['conf'] >= Thresholds.TIER_1_CONF and not dissonance:
                tier = 1
            elif q['conf'] >= Thresholds.TIER_2_CONF:
                tier = 2
                
            results.append({
                'market_name': q['market_name'],
                'confidence': q['conf'],
                'type': q['type'],
                'tier': tier,
                'dissonance': dissonance,
                'is_fallback': is_fallback,
                'coverage_status': coverage_status,
                'tempo_status': q['tempo_status']
            })
            
        return results

    def evaluate(
        self, 
        preds: Dict[str, float], 
        is_fallback: bool = False, 
        coverage_status: str = "UNKNOWN", 
        league: Optional[str] = None, 
        standings_context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Identify all eligible markets matching strict quality gates.
        
        Returns a list of qualified candidates.
        """
        # Parse inputs to potentials (standings aware)
        all_markets = self._parse_inputs(preds, standings_context)
        
        if not all_markets:
             return []
        
        # Audit Assertion: All probabilities must be valid
        validated_markets = []
        for m in all_markets:
            if not (0.0 <= m['conf'] <= 1.0):
                logger.error(
                    "Probability violation in %s: %s - skipping",
                    m['market_name'],
                    m['conf'],
                )
                continue
            validated_markets.append(m)
        all_markets = validated_markets

        if not all_markets:
            return []
        
        # Dissonance Flagging
        dissonance = self._check_dissonance(all_markets)
        
        # Tempo Suppression
        tempo_status, suppress = self._evaluate_tempo(preds, league)
        if suppress:
            return []
            
        # Gate Evaluation
        qualified = []
        for m in all_markets:
            gate = self._get_adjusted_gate(m, standings_context)
            
            if m['conf'] >= gate:
                qualified.append({
                    'market_name': m['market_name'],
                    'type': m['type'],
                    'conf': m['conf'],
                    'direction': m.get('direction'),
                    'dissonance': dissonance,
                    'is_fallback': is_fallback,
                    'coverage_status': coverage_status,
                    'tempo_status': tempo_status
                })
                
        return qualified

    def _check_dissonance(self, all_markets: List[Dict[str, Any]]) -> bool:
        """Check if Goals and Corners 1X2 disagree on winner."""
        g_win = next((m for m in all_markets if m['type'] == '1X2'), None)
        c_win = next((m for m in all_markets if m['type'] == '1X2_CORNERS'), None)
        
        if g_win and c_win and g_win['direction'] != c_win['direction']:
            return True
        return False

    def _evaluate_tempo(
        self, 
        preds: Dict[str, float], 
        league: Optional[str]
    ) -> tuple[str, bool]:
        """Evaluate match tempo and determine suppression."""
        p_o15 = preds.get('over_1_5', 0.0)
        p_o75 = preds.get('corners_over_7_5', 0.0)
        p_u25 = preds.get('under_2_5', 1.0 - preds.get('over_2_5', 0.5))

        # Apply league adjustments
        if league in LEAGUE_ADJUSTMENTS:
            adj = LEAGUE_ADJUSTMENTS[league]
            p_o15 = np.clip(p_o15 + adj.get('o15', 0.0), 0.0, 1.0)
            p_u25 = np.clip(p_u25 + adj.get('u25', 0.0), 0.0, 1.0)
            p_o75 = np.clip(p_o75 + adj.get('o75', 0.0), 0.0, 1.0)
        
        tempo_status = "NORMAL"
        suppress = False
        
        if p_o15 >= self.TEMPO_THRESHOLDS['O15_STABLE']:
            tempo_status = "STABLE"
        elif p_o15 <= self.TEMPO_THRESHOLDS['O15_LOW_RISK']:
            tempo_status = "LOW_EVENT"
            suppress = True
            
        if p_o75 >= self.TEMPO_THRESHOLDS['O75_HIGH']:
            tempo_status = f"{tempo_status}/HIGH_TEMPO" if tempo_status != "NORMAL" else "HIGH_TEMPO"
        elif p_o75 <= self.TEMPO_THRESHOLDS['O75_SLOW']:
            suppress = True

        return tempo_status, suppress

    def _get_adjusted_gate(
        self, 
        market: Dict[str, Any], 
        standings_context: Optional[Dict[str, Any]]
    ) -> float:
        """Get gate with momentum-based adjustment."""
        gate = self.HARD_GATES.get(market['type'], 1.0)
        
        if not standings_context:
            return gate
            
        # Momentum harnessing for select markets
        momentum_markets = ['CORNERS', 'CARDS', 'CARDS_O25', 'BTTS', 'GOALS_O15', 'GOALS_O25', 'GOALS_U25']
        if market['type'] not in momentum_markets:
            return gate
            
        # Check for surging underdog
        for side in ['HOME', 'AWAY']:
            s_data = standings_context.get(side, {})
            if s_data.get('status_band') == "LOW" and s_data.get('momentum_band') == "STRONG_POSITIVE":
                return max(self.GLOBAL_MIN_CONF, gate - MOMENTUM_GATE_RELAXATION)
        
        return gate

    def _parse_inputs(
        self, 
        preds: Dict[str, float], 
        standings: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Parse prediction dictionary into market candidates."""
        markets: List[Dict[str, Any]] = []
        
        # 1X2
        ph = preds.get('home_win', 0)
        pa = preds.get('away_win', 0)
        fav_side = "HOME" if ph > pa else "AWAY"
        fav_conf = ph if fav_side == "HOME" else pa
        
        # Trap Detection: Top team in strong decline
        if standings:
            fav_data = standings.get(fav_side, {})
            if fav_data.get('status_band') == "TOP" and fav_data.get('momentum_band') == "STRONG_NEGATIVE":
                # Collect for summary - detail at DEBUG
                logger.debug(f"Momentum Gate: Skipping {fav_side} WIN due to Strong Negative momentum")
                # Note: Cannot increment _gate_warnings from Evaluator - handled at Engine level
            else:
                markets.append({
                    'type': '1X2', 'direction': fav_side,
                    'market_name': f"{fav_side}_WIN", 'conf': fav_conf
                })
        else:
            markets.append({
                'type': '1X2', 'direction': fav_side,
                'market_name': f"{fav_side}_WIN", 'conf': fav_conf
        })
        
        # Corners Under
        if preds.get('corn_u11'):
            markets.append({
                'type': 'CORNERS', 'direction': 'UNDER',
                'market_name': 'CORNERS_U11.5', 'conf': preds['corn_u11']
            })
            
        # Corners Over
        if preds.get('corners_over_7_5'):
            markets.append({
                'type': 'CORNERS_O75', 'direction': 'OVER',
                'market_name': 'CORNERS_O7.5', 'conf': preds['corners_over_7_5']
            })
            
        # Cards Markets
        if preds.get('card_u45'):
             markets.append({
                'type': 'CARDS', 'direction': 'UNDER',
                'market_name': 'CARDS_U4.5', 'conf': preds['card_u45']
            })
            
        if preds.get('cards_over_2_5'):
             markets.append({
                'type': 'CARDS_O25', 'direction': 'OVER',
                'market_name': 'CARDS_O2.5', 'conf': preds['cards_over_2_5']
            })
            
        if preds.get('card_u55'):
             markets.append({
                'type': 'CARDS_U55', 'direction': 'UNDER',
                'market_name': 'CARDS_U5.5', 'conf': preds['card_u55']
            })
           
        # Underdog U1.5 (Context Aware)
        und_conf = pa if fav_side == "HOME" else ph
        und_side = "AWAY" if fav_side == "HOME" else "HOME"
        p_und_u15 = preds.get(f"{und_side.lower()}_under_1_5", 0.0)
        
        if und_conf < UNDERDOG_WIN_PROB_MAX and p_und_u15 > UNDERDOG_U15_MIN_CONF:
            if self._validate_underdog_standings(standings, und_side, fav_side):
                markets.append({
                    'type': 'UND_U15', 'direction': 'UNDER',
                    'market_name': f"{und_side}_TG_U1.5", 'conf': p_und_u15
                })

        # Standard Goal Bands
        if preds.get('over_1_5'):
            markets.append({
                'type': 'GOALS_O15', 'direction': 'OVER',
                'market_name': 'GOALS_O1.5', 'conf': preds['over_1_5']
            })
            
        if preds.get('over_2_5'):
            markets.append({
                'type': 'GOALS_O25', 'direction': 'OVER',
                'market_name': 'GOALS_O2.5', 'conf': preds['over_2_5']
            })

        if preds.get('under_2_5'):
            markets.append({
                'type': 'GOALS_U25', 'direction': 'UNDER',
                'market_name': 'GOALS_U2.5', 'conf': preds['under_2_5']
            })

        if preds.get('goals_under_3_5'):
            markets.append({
                'type': 'GOALS_U35', 'direction': 'UNDER',
                'market_name': 'GOALS_U3.5', 'conf': preds['goals_under_3_5']
            })

        # 1X2 Corners (Analytical)
        if preds.get('corners_home_win'):
            ch = preds['corners_home_win']
            ca = preds['corners_away_win']
            fav_c_side = "HOME" if ch > ca else "AWAY"
            fav_c_conf = ch if fav_c_side == "HOME" else ca
            
            markets.append({
                'type': '1X2_CORNERS', 'direction': fav_c_side,
                'market_name': f'{fav_c_side}_CORNERS_WIN', 'conf': fav_c_conf
            })
            
        # Double Chance (Risk Wrapper)
        # Only ingest if "Draw Uncertainty is High" OR "Momentum Divergence Flags Instability"
        # We assume DC keys exist from Derived Engine (dc_1x, dc_x2, dc_12)
        p_draw = preds.get('draw', 0.0)
        DRAW_UNCERTAINTY_THRESHOLD = 0.26 # If draw > 26%, consider DC
        
        for k_dc, name_dc, comp_side in [('dc_1x', '1X', 'HOME'), ('dc_x2', 'X2', 'AWAY')]:
            conf_dc = preds.get(k_dc, 0.0)
            if conf_dc > 0:
                # 1. Check Draw Risk
                is_draw_risky = p_draw > DRAW_UNCERTAINTY_THRESHOLD
                
                # 2. Check Momentum Instability (if available)
                # If we are betting ON this side (e.g. 1X -> Home), allow if Home is NOT Strong Positive
                # OR if Underdog (Opposition) is Strong Positive.
                is_unstable = False
                if standings:
                     side_data = standings.get(comp_side, {})
                     opp_side = "AWAY" if comp_side == "HOME" else "HOME"
                     opp_data = standings.get(opp_side, {})
                     
                     # Instability: Side is shaky OR Opp is surging
                     if side_data.get('momentum_band') in ['NEUTRAL', 'NEGATIVE'] or \
                        opp_data.get('momentum_band') == 'STRONG_POSITIVE':
                         is_unstable = True

                # Gate: Use DC only if specific risk profile exists
                # If everything is perfect, we'd bet 1X2. If terrible, we pass.
                # DC is for the "Almost" zone.
                if is_draw_risky or is_unstable:
                    markets.append({
                        'type': 'DOUBLE_CHANCE', 'direction': comp_side,
                        'market_name': f"{comp_side}_DC", 'conf': conf_dc
                    })

        # BTTS Yes/No
        if preds.get('btts_yes'):
            markets.append({
                'type': 'BTTS', 'direction': 'YES',
                'market_name': 'BTTS_YES', 'conf': preds['btts_yes']
            })
        if preds.get('btts_no'):
            markets.append({
                'type': 'BTTS', 'direction': 'NO',
                'market_name': 'BTTS_NO', 'conf': preds['btts_no']
            })
            
        return markets

    def _validate_underdog_standings(
        self, 
        standings: Optional[Dict[str, Any]], 
        und_side: str, 
        fav_side: str
    ) -> bool:
        """Validate underdog market against standings."""
        if not standings:
            return True
            
        und_band = standings.get(und_side, {}).get('status_band')
        fav_band = standings.get(fav_side, {}).get('status_band')
        
        # Reject if "underdog" is actually a TOP team vs LOW/MID favorite
        if und_band == "TOP" and fav_band in ["LOW", "MID"]:
            logger.warning(
                f"Rejecting UND_U15: {und_side} is TOP band but model treats as underdog vs {fav_side} ({fav_band})."
            )
            return False
            
        return True


class ForbiddenFruitEngine:
    """
    Strategy Orchestrator for Competitive Selection.
    
    Combines ForbiddenFruitEvaluator with CSS-based slip construction
    and drift guardrail enforcement.
    """
    
    def __init__(self) -> None:
        self.evaluator = ForbiddenFruitEvaluator()
        self.drift_guard = DriftGuardrail()
        self.edge_engine = EdgeEngine()
        self.standings = StandingsManager()
        self._registry = ModelRegistry()
        self._cached_drift_status: Optional[str] = None
        # Warning collection for consolidated output
        self._gate_warnings: Dict[str, int] = {}

    def _get_drift_status(self) -> str:
        if self._cached_drift_status is None:
            self._cached_drift_status = self.drift_guard.check_drift()
        return self._cached_drift_status

    def analyze_match(
        self, 
        match: Dict[str, Any], 
        predictions: Dict[str, float], 
        is_fallback: bool = False, 
        provenance: Optional[Dict[str, Any]] = None,
        precomputed_confidences: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Structural Analysis: Identify ALL tiered candidates for a match.
        
        Includes standings context for robust gating.
        """
        league_code = match.get('league') or match.get('competition')
        coverage_status = self._registry.get_coverage_status(league_code) if league_code else "UNKNOWN"
        
        # 🛡️ Phase 10: Strategy Gate Enforcement (CRITICAL FIX 2026-01-13)
        # Must CALL check_drift() to get current status from persisted file
        drift_status = self._get_drift_status()
        
        if drift_status == "STOP":
            logger.error(f"[{league_code}] DRIFT STOP HARD-BLOCK: Strategy halted due to drift alert!")
            return []
        elif drift_status == "WATCH":
            logger.warning(f"[{league_code}] DRIFT WATCH: Proceeding with caution")

        # Fetch Standings Context
        standings_context = self._fetch_standings_context(match, league_code)

        decisions = self.evaluator.evaluate_decision(
            predictions, 
            is_fallback=is_fallback, 
            coverage_status=coverage_status, 
            league=league_code,
            standings_context=standings_context
        )
        
        confidence_calc = get_confidence_calculator()
        
        candidates = []
        for d in decisions:
            market_name = d['market_name']
            p_cal = d['confidence']  # This is the probability
            
            # === CONFIDENCE POLICY: Canonical vs Estimated ===
            if precomputed_confidences and market_name in precomputed_confidences and precomputed_confidences[market_name] is not None:
                confidence = precomputed_confidences[market_name]
                source = "precomputed"
            else:
                # Fallback estimation (Audit Requirement: Never modify precomputed)
                conf_result = confidence_calc.calculate(
                    prob=p_cal,
                    market=market_name,
                    league=league_code or 'UNKNOWN',
                    mu=p_cal * 3,  # Approximate mu from prob
                    variance=0.5,  # Moderate default
                    h2h_count=match.get('h2h_match_count', 0),
                    days_since_h2h=match.get('days_since_last_h2h', 0),
                    missing_features=[],
                    referee_known=True
                )
                confidence = conf_result.confidence
                source = "estimated"

            if bool(predictions.get("ensemble_divergence", False)):
                divergence_pct = float(predictions.get("divergence_pct", 0.0) or 0.0)
                confidence *= 0.90
                logger.info(
                    "[FF] %s confidence reduced 10%% due to ensemble divergence (%.2f%%)",
                    market_name,
                    divergence_pct,
                )
            
            # Mandatory Audit-First Log
            logger.debug(f"[FF] {market_name} confidence={confidence:.2f} ({source})")
            
            # === STEP 2.5: Apply Soft Cap (Confidence-Aware) ===
            p_cal = apply_soft_cap(p_cal, confidence, market_name)
            
            candidate = {
                "match": f"{match['home_team']}{MATCH_SEPARATOR}{match['away_team']}",
                "market": market_name,
                "market_name": market_name,
                "type": d['type'],
                "probability": p_cal,  # Renamed for clarity internally
                "confidence": confidence, # Now using the orthogonal signal!
                "action_tier": get_action_tier(confidence),
                "league": league_code or 'Unknown',
                "id": match.get('match_id'),
                "date": match.get('date'),
                "tier": d['tier'],
                "is_fallback": is_fallback,
                "coverage_status": coverage_status,
                "tempo_status": d['tempo_status'],
                "provenance": provenance,
                "conf_components": [] # Components not passed for precomputed
            }
            if standings_context:
                candidate['standings'] = standings_context
            
            candidates.append(candidate)
        
        return candidates

    def _fetch_standings_context(
        self, 
        match: Dict[str, Any], 
        league_code: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Fetch standings context for a match."""
        if not league_code:
            return None
            
        try:
            m_date = match.get('date')
            if isinstance(m_date, str):
                m_date = datetime.fromisoformat(m_date.replace('Z', '+00:00'))
            
            if not m_date:
                return None
                
            season = m_date.year if m_date.month >= 6 else m_date.year - 1
            table = self.standings.get_table(league_code, season, m_date)
            
            h_team = match.get('home_team')
            a_team = match.get('away_team')
            
            if table and h_team in table and a_team in table:
                return {'HOME': table[h_team], 'AWAY': table[a_team]}
        except Exception as e:
            logger.warning(f"Could not fetch standings for context layer: {e}")
            
        return None

    def construct_slip(
        self,
        all_candidates: List[Dict[str, Any]],
        min_probability: float = MIN_SLIP_PROBABILITY,
        max_selections: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Build the final slip using the Combo Survivability Score (CSS).
        
        Strictly enforces structural stability and penalizes volatility.
        """
        if not all_candidates or max_selections < 2:
            return []

        # Hard Filter: Probability & Confidence Gates (Market-Specific)
        valid_candidates = []
        for c in all_candidates:
            market = c.get('market') or c.get('market_name', '')
            prob = c.get('probability', c.get('confidence', 0.0))
            conf = c.get('confidence', prob)
            
            # Lookup market-specific threshold (Option A)
            threshold = MIN_SLIP_CONFIDENCE_MAP.get(market, MIN_SLIP_CONFIDENCE)
            
            if prob >= min_probability and conf >= threshold:
                logger.debug(f"[FF] {market} passed confidence threshold ({threshold})")
                valid_candidates.append(c)
            else:
                logger.debug(f"[FF] {market} REJECTED: prob={prob:.2f}, conf={conf:.2f} (threshold={threshold})")
        
        # Market Filtering
        pool = self._filter_valid_markets(valid_candidates)
        if not pool:
            return []

        # Combinatorial Generation (2 to 4 legs)
        best_slip, best_css, best_details = self._find_best_combination(
            pool,
            max_selections=max_selections,
        )
        
        # Binding Enforcement & Logging
        if best_slip:
            best_slip = self.edge_engine.rank_opportunities(best_slip)
            self._log_slip_breakdown(best_slip, best_css, best_details)

        return best_slip

    def _filter_valid_markets(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter candidates to valid slip markets."""
        pool = []
        
        for c in candidates:
            m_name = c.get('market_name') or c.get('market')
            
            # Check if market is in valid set
            is_valid = any(vm in m_name for vm in VALID_SLIP_MARKETS)
            
            if is_valid:
                # Registry coverage check
                league_code = c.get('league')
                if league_code:
                    if self._registry.get_coverage_status(league_code) == "UNKNOWN":
                        logger.warning(
                            "Skipping market '%s': league '%s' has no registered model",
                            m_name,
                            league_code,
                        )
                        continue
                pool.append(c)
                
        return pool

    def _find_best_combination(
        self, 
        pool: List[Dict[str, Any]],
        max_selections: int = 4,
    ) -> tuple[List[Dict[str, Any]], float, Dict[str, float]]:
        """Find the best slip combination using CSS."""
        best_slip: List[Dict[str, Any]] = []
        best_css = -1.0
        best_details: Dict[str, float] = {}
        max_combo_size = min(max_selections, 4)
        
        for r in range(2, max_combo_size + 1):
            for combo in itertools.combinations(pool, r):
                legs = list(combo)
                
                # Max 1 per match
                match_ids = set(l.get('id') or l.get('match') for l in legs)
                if len(match_ids) != len(legs):
                    continue
                
                # Correlated pair limit
                corr_count = count_correlated_pairs(legs)
                if corr_count > MAX_CORRELATED_PAIRS:
                    continue
                    
                # Calculate CSS
                css, details = calculate_css(legs, corr_count)
                
                # Apply market weights to boost preferred markets
                weighted_css = css
                for leg in legs:
                    m_name = leg.get('market_name') or leg.get('market', '')
                    for market_key, weight in MARKET_WEIGHTS.items():
                        if market_key in m_name:
                            weighted_css *= weight
                            break

                if weighted_css < MIN_CSS_SCORE:
                    continue
                    
                if weighted_css > best_css:
                    best_css = weighted_css
                    best_slip = legs
                    best_details = details
                    
        return best_slip, best_css, best_details

    def _log_slip_breakdown(
        self, 
        slip: List[Dict[str, Any]], 
        css: float, 
        details: Dict[str, float]
    ) -> None:
        """Log CSS breakdown for the winning slip."""
        from src.config.thresholds import css_risk_band
        
        # Concise summary at INFO level
        risk = css_risk_band(css)
        logger.info(f"Slip Score: {css:.2f} | Risk: {risk} | Legs: {len(slip)}")
        
        # Detailed breakdown at DEBUG level
        logger.debug(f"[CSS BREAKDOWN] Suggested Slip (Score: {css:.4f})")
        logger.debug(f"  > Stability Prod (s_i): {details.get('s_i_product', 0):.4f}")
        logger.debug(f"  > Correlation Penalty:  {details.get('corr_penalty', 0):.4f}")
        logger.debug(f"  > Survivability Boost:  {details.get('survivability', 0):.4f}")
        logger.debug(f"  > Joint Failure Risk:   {1.0 - details.get('survivability', 0):.4f}")
        logger.debug("  > Individual Legs:")
        for leg in slip:
            p = leg.get('confidence', 0.0)
            v = 1.0 - p
            logger.debug(f"    - {leg.get('market_name')} @ {p:.2%}, Vol={v:.2f}")

    def annotate_candidates_with_edge(
        self,
        candidates: List[Dict[str, Any]],
        odds_by_market: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Attach price-aware value fields to candidates and rank them."""
        annotated = []
        for candidate in candidates:
            market_name = candidate.get("market") or candidate.get("market_name")
            market_odds = (odds_by_market or {}).get(market_name)
            annotated.append(
                self.edge_engine.annotate_opportunity(
                    candidate,
                    odds=market_odds,
                )
            )

        return self.edge_engine.rank_opportunities(annotated)

    def evaluate_value_opportunities(
        self,
        match: Dict[str, Any],
        raw_predictions: Dict[str, float],
        odds: Optional[Dict[str, float]] = None,
        h2h_data: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Return ranked value opportunities with no stake sizing or bet logging."""
        calibrator = get_calibrator()
        sharpness = get_sharpness_gate()
        drift_monitor = get_drift_monitor()
        rejection_logger = get_rejection_logger()
        confidence_calc = get_confidence_calculator()

        league = match.get('league', 'UNKNOWN')
        match_str = f"{match.get('home_team')}{MATCH_SEPARATOR}{match.get('away_team')}"
        h2h_data = h2h_data or {}
        opportunities = []

        for pred_key, market in VALUE_MARKET_MAP.items():
            if pred_key not in raw_predictions:
                continue

            p_raw = raw_predictions[pred_key]

            if sharpness.get_status(market) == "DISABLED":
                rejection_logger.log_rejection(
                    match_str,
                    market,
                    p_raw,
                    RejectionReason.SHARPNESS_DISABLED,
                )
                continue

            p_cal = calibrator.calibrate(market, p_raw)
            if p_cal is None:
                rejection_logger.log_rejection(
                    match_str,
                    market,
                    p_raw,
                    RejectionReason.CALIBRATOR_DISABLED,
                )
                continue

            h2h_count = h2h_data.get('count', 5)
            days_since_h2h = h2h_data.get('days_ago', 90)
            missing_features = h2h_data.get('missing', [])

            conf_result = confidence_calc.calculate(
                prob=p_cal,
                market=market,
                league=league,
                mu=p_cal * 3,
                variance=max(0.5, p_cal * (1 - p_cal) * 2),
                h2h_count=h2h_count,
                days_since_h2h=days_since_h2h,
                missing_features=missing_features,
                referee_known=True,
            )

            confidence = conf_result.confidence
            action_tier = conf_result.action_tier
            p_cal = apply_soft_cap(p_cal, confidence, market)

            if action_tier == 'NO_BET':
                rejection_logger.log_rejection(
                    match_str,
                    market,
                    p_raw,
                    RejectionReason.LOW_TIER,
                    f"confidence={confidence:.3f}",
                )
                continue

            market_odds = (odds or {}).get(market)
            evaluation = self.edge_engine.evaluate_opportunity(
                probability=p_cal,
                odds=market_odds,
            )
            if not evaluation["priced"]:
                rejection_logger.log_rejection(
                    match_str,
                    market,
                    p_raw,
                    RejectionReason.ODDS_SANITY,
                    evaluation["value_reason"],
                )
                continue

            if not evaluation["passes_value_threshold"]:
                rejection_logger.log_rejection(
                    match_str,
                    market,
                    p_raw,
                    RejectionReason.EV_FAIL,
                    (
                        f"{evaluation['value_reason']}: "
                        f"edge={evaluation['edge']:.3f} ev={evaluation['ev']:.3f}"
                    ),
                )
                continue

            if not drift_monitor.check_drift(market):
                rejection_logger.log_rejection(
                    match_str,
                    market,
                    p_raw,
                    RejectionReason.DRIFT_FREEZE,
                )
                continue

            opportunities.append(
                {
                    'match': match_str,
                    'market': market,
                    'p_raw': round(p_raw, 4),
                    'p_cal': round(p_cal, 4),
                    'probability': round(p_cal, 4),
                    'confidence': round(confidence, 4),
                    'action_tier': action_tier,
                    'league': league,
                    'conf_components': conf_result.components,
                    **evaluation,
                }
            )

        ranked_opportunities = self.edge_engine.rank_opportunities(opportunities)
        for prediction in ranked_opportunities:
            log_prediction(
                {
                    **prediction,
                    "home_team": match.get("home_team"),
                    "away_team": match.get("away_team"),
                    "match_date": match.get("date"),
                }
            )
        return ranked_opportunities

    def calibrated_selection(
        self,
        match: Dict[str, Any],
        raw_predictions: Dict[str, float],
        odds: Optional[Dict[str, float]] = None,
        bankroll: float = 1000.0,
        h2h_data: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Legacy compatibility wrapper for value-filtered opportunities.

        The `bankroll` argument is ignored intentionally: probability intelligence
        ends at pricing and edge evaluation.
        """
        _ = bankroll
        return self.evaluate_value_opportunities(
            match,
            raw_predictions,
            odds=odds,
            h2h_data=h2h_data,
        )

