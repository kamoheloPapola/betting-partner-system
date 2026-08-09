"""
Prediction CLI Commands.

Commands for generating and displaying football probability forecasts.
Orchestrates the prediction pipeline from data loading through presentation.
"""
import typer
import pandas as pd
import numpy as np
import logging
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, TypedDict, Tuple, cast, Protocol, runtime_checkable

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich import box
from rich import box
from rich.progress import Progress, TaskID

from src.cli.base import app
from src.cli.utils import LeagueCode, MarketProbabilities, DateFilter, MATCH_SEPARATOR, resolve_league_code, filter_matches_by_date
from src.config import (
    PROCESSED_DATA_DIR,
    DATA_DIR,
    MODELS_DIR,
    DATA_FRESHNESS_DAYS,
    GOALS_ENSEMBLE_LGBM_WEIGHT,
    GOALS_ENSEMBLE_XGB_WEIGHT,
    GOALS_ENSEMBLE_DIVERGENCE_THRESHOLD,
)
from src.config.thresholds import Thresholds
from src.core.container import ServiceContainer
from src.core.validators import validate_match_dataframe
from src.core.exceptions import PredictionSystemError, ModelNotFoundError, DataValidationError
from src.ml.distributions import PoissonEngine, NegativeBinomialEngine, ZeroInflatedEngine
from src.monitoring.events import log_event, PredictionEvent
from src.strategies.selection_gate import SelectionGate
from src.ml.guards import PredictionGuard
from src.strategies.derived import DoubleChanceEngine
from src.ml.models.corners.team_offsets import TeamOffsetManager
from src.monitoring.drift_orchestrator import DriftOrchestrator
from src.ml.calibration import (
    apply_binary_calibrator,
    apply_soft_cap,
    load_binary_calibrator_artifact,
)
from src.ml.confidence import get_confidence_calculator
from src.simulation.match_simulator import MatchSimulator, clamp_lambda, DEFAULT_N_SIMULATIONS
from src.simulation.rl_bandit import ContextualBandit
from src.core.constants import (
    H2H_HIGH_SAMPLE_THRESHOLD,
    MAX_H2H_LIFT_DEFAULT,
    MAX_H2H_LIFT_HIGH_SAMPLE,
)

logger = logging.getLogger(__name__)

# Backward-compatible alias for older tests/config names.
GOALS_ENSEMBLE_POISSON_WEIGHT = GOALS_ENSEMBLE_XGB_WEIGHT
_MODEL_CALIBRATOR_CACHE: Dict[str, Optional[Dict[str, Any]]] = {}
RL_SHADOW_MARKET = "1x2"
RL_BANDIT_LIVE_FLAG = DATA_DIR / "rl_bandit_live.flag"

# --- TYPES: ARCHITECTURAL CONTRACTS ---

@runtime_checkable
class PredictionModel(Protocol):
    """Protocol for models with predict() method."""
    meta: Dict[str, Any]
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

@runtime_checkable
class ClassificationModel(PredictionModel, Protocol):
    """Protocol for models with predict_proba() method."""
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...

class MandatoryModelSuite(TypedDict):
    """Rigid core required for any prediction output."""
    mh_goals: PredictionModel
    ma_goals: PredictionModel
    meta_goals: Dict[str, Any]

class ModelSuite(MandatoryModelSuite, total=False):
    """Extended suite with best-effort optional markets."""
    mh_goals_xgb: PredictionModel
    ma_goals_xgb: PredictionModel
    meta_goals_xgb: Dict[str, Any]
    mh_goals_poisson: PredictionModel
    ma_goals_poisson: PredictionModel
    meta_goals_poisson: Dict[str, Any]
    mh_corn: Optional[PredictionModel]
    ma_corn: Optional[PredictionModel]
    m_cards: Optional[PredictionModel]
    m_corners_lgbm: Optional[PredictionModel]
    m_corners_xgb: Optional[PredictionModel]
    m_cards_lgbm: Optional[PredictionModel]
    m_cards_xgb: Optional[PredictionModel]
    m_1x: Optional[ClassificationModel]
    m_x2: Optional[ClassificationModel]
    m_12: Optional[ClassificationModel]

# --- HELPERS: MODEL LOADING ---

def _best_effort_load_model(registry: Any, candidates: List[str], league_code: str) -> Optional[PredictionModel]:
    """Try model names in order and return first successfully loaded model."""
    for name in candidates:
        try:
            return cast(PredictionModel, registry.load_model(name, league=league_code))
        except ModelNotFoundError:
            continue
    return None


def _load_prediction_models(league_code: str) -> ModelSuite:
    """Load and validate production-ready ModelSuite."""
    reg = ServiceContainer.get_instance().registry

    goals_models = reg.load_models_for_market(
        market="goals_lambda",
        model_names=["home_goals", "away_goals"],
        league=league_code,
    )
    m_h_lgbm = cast(PredictionModel, goals_models["home_goals"])
    m_a_lgbm = cast(PredictionModel, goals_models["away_goals"])

    m_h_xgb = _best_effort_load_model(reg, ["home_goals_xgb", "poisson_home_base"], league_code)
    m_a_xgb = _best_effort_load_model(reg, ["away_goals_xgb", "poisson_away_base"], league_code)

    # Protocol Validation (Defensive)
    for model_name, model in (
        ("home_goals", m_h_lgbm),
        ("away_goals", m_a_lgbm),
    ):
        if not isinstance(model, PredictionModel):
            raise PredictionSystemError(f"Model {model_name} does not implement PredictionModel")

    if m_h_xgb is not None and not isinstance(m_h_xgb, PredictionModel):
        raise PredictionSystemError("Model home_goals_xgb does not implement PredictionModel")
    if m_a_xgb is not None and not isinstance(m_a_xgb, PredictionModel):
        raise PredictionSystemError("Model away_goals_xgb does not implement PredictionModel")

    meta_lgbm = cast(Dict[str, Any], getattr(m_h_lgbm, "meta", {}) or {})
    meta_xgb = cast(Dict[str, Any], getattr(m_h_xgb, "meta", {}) or {}) if m_h_xgb is not None else {}

    if not meta_lgbm.get("features"):
        recovered = reg.get_production_model_for_league(league_code, "home_goals")
        if isinstance(recovered, dict):
            meta_lgbm = recovered
    if not meta_xgb.get("features") and m_h_xgb is not None:
        recovered = (
            reg.get_production_model_for_league(league_code, "home_goals_xgb")
            or reg.get_production_model_for_league(league_code, "poisson_home_base")
        )
        if isinstance(recovered, dict):
            meta_xgb = recovered

    if not meta_lgbm.get("features"):
        raise ModelNotFoundError(
            f"Mandatory goals feature metadata missing for {league_code}",
            context={"league": league_code},
        )

    suite: ModelSuite = {
        "mh_goals": m_h_lgbm,
        "ma_goals": m_a_lgbm,
        "meta_goals": meta_lgbm,
        "mh_goals_xgb": m_h_xgb if m_h_xgb is not None else m_h_lgbm,
        "ma_goals_xgb": m_a_xgb if m_a_xgb is not None else m_a_lgbm,
        "meta_goals_xgb": meta_xgb if meta_xgb.get("features") else meta_lgbm,
        "mh_corn": _best_effort_load_model(reg, ["mh_corn"], league_code),
        "ma_corn": _best_effort_load_model(reg, ["ma_corn"], league_code),
        "m_cards": _best_effort_load_model(reg, ["poisson_total_cards_base"], league_code),
        "m_corners_lgbm": _best_effort_load_model(reg, ["m_corners_lgbm", "corners"], league_code),
        "m_corners_xgb": _best_effort_load_model(reg, ["m_corners_xgb", "corners_xgb"], league_code),
        "m_cards_lgbm": _best_effort_load_model(reg, ["cards"], league_code),
        "m_cards_xgb": _best_effort_load_model(reg, ["cards_xgb"], league_code),
    }

    return suite

# --- HELPERS: DATA PROCESSING & PREDICTION ---

def _load_model_calibrator(model: PredictionModel) -> Optional[Dict[str, Any]]:
    """Load per-model post-hoc calibrator artifact once and cache it."""
    meta = cast(Dict[str, Any], getattr(model, "meta", {}) or {})
    rel_file = meta.get("calibrator_filename")
    if not rel_file:
        return None

    cache_key = str(rel_file)
    if cache_key in _MODEL_CALIBRATOR_CACHE:
        return _MODEL_CALIBRATOR_CACHE[cache_key]

    cal_path = MODELS_DIR / str(rel_file)
    payload = load_binary_calibrator_artifact(cal_path)
    _MODEL_CALIBRATOR_CACHE[cache_key] = payload
    return payload


def _apply_model_calibration(model: PredictionModel, raw_value: float) -> float:
    """
    Apply model-level post-hoc calibration when artifact metadata is available.

    For count models we calibrate P(Y>0) and map it back to lambda.
    """
    payload = _load_model_calibrator(model)
    if not payload:
        return raw_value

    calibrator = payload.get("calibrator")
    cal_type = payload.get("type")
    if calibrator is None or not isinstance(cal_type, str):
        return raw_value

    meta = cast(Dict[str, Any], getattr(model, "meta", {}) or {})
    model_type = str(meta.get("type", "")).lower()

    try:
        if model_type in {
            "poisson",
            "nb",
            "lgbm_regressor_poisson",
            "xgb_regressor_count_poisson",
            "xgb_regressor_tweedie",
            "lgbm_regressor_tweedie",
        }:
            lam = float(max(raw_value, 0.05))
            p_raw = 1.0 - float(np.exp(-lam))
            p_cal = float(
                np.clip(
                    apply_binary_calibrator(calibrator, cal_type, np.array([p_raw], dtype=float))[0],
                    1e-6,
                    1 - 1e-6,
                )
            )
            return float(-np.log(1.0 - p_cal))

        p_raw = float(np.clip(raw_value, 1e-6, 1 - 1e-6))
        p_cal = float(apply_binary_calibrator(calibrator, cal_type, np.array([p_raw], dtype=float))[0])
        return float(np.clip(p_cal, 0.0, 1.0))
    except Exception as exc:
        logger.debug("Calibration apply failed for model %s: %s", meta.get("name", "unknown"), exc)
        return raw_value


def _predict_scalar(model: PredictionModel, row: pd.Series, feats: List[str], context: str) -> float:
    """Standardized pattern: feature mapping -> numeric cast -> prediction."""
    t_data = _translate_features(row, feats, context)
    try:
        df = pd.DataFrame([pd.to_numeric(t_data)])
        
        # ðŸ›¡ï¸ Phase 9: Hallucination Guard
        # Rejects prediction if model signature or schema is invalid.
        # Also enforces Calibration Gate.
        PredictionGuard.validate_prediction_integrity(model, df, context)
        
        raw = float(model.predict(df)[0])
        return float(_apply_model_calibration(model, raw))
    except Exception as e:
        raise DataValidationError(f"Prediction failed in {context}: {e}")

def _translate_features(row: pd.Series, feats: List[str], ctx: str) -> pd.Series:
    """Translate nomenclature and enforce feature existence."""
    out = {}
    for f in feats:
        if f in row.index:
            out[f] = row[f]
        else:
            alt = f.replace('_scored_', '_won_').replace('_conceded_', '_received_')
            if alt in row.index:
                out[f] = row[alt]
            else:
                raise DataValidationError(f"Missing mandatory feature '{f}' for {ctx}")
    return pd.Series(out)

# === CONFIDENCE SEMANTICS (Downgraded based on OOS calibration) ===
CONFIDENCE_STRONG = 0.75   # 75%+ = strong
CONFIDENCE_EXCEPTIONAL = 0.80  # 80%+ = exceptional (rare)

# === CALIBRATION GUARDRAILS (Based on Layer 4.1 OOS audit) ===
# Market-specific probability caps to prevent overconfidence
MARKET_PROB_CAPS = {
    'u25': 0.66,        # Goals U2.5 max 66% (was hitting 77% with 53% hit rate)
    'btts_yes': 0.65,   # BTTS Yes max 65%
    'btts_no': 0.55,    # BTTS No max 55% (strongly overconfident)
    'u35': 0.85,        # Goals U3.5 max 85%
    'home_under_1_5': 0.80, # Team Goal U1.5 max 80% (raised from 65% - model is underconfident)
    'away_under_1_5': 0.80,
    'card_o25': 0.80,   # Cards O2.5 max 80%
}


# Confidence tier thresholds for display
CONFIDENCE_TIERS = {
    (0.00, 0.55): "Low",
    (0.55, 0.60): "Medium-Low",
    (0.60, 0.65): "Medium",
    (0.65, 0.75): "Medium+",  # Not "High" - calibration shows drift here
    (0.75, 1.00): "Speculative",  # Use with caution
}

def apply_calibration_cap(prob: float, market: str, ctx: str = "") -> float:
    """
    Apply market-specific probability cap based on OOS calibration.
    This prevents overconfident predictions in markets where calibration drifts.
    """
    cap = MARKET_PROB_CAPS.get(market)
    if cap and prob > cap:
        if ctx:
            logger.info(f"{ctx}: Calibration cap applied to {market}: {prob*100:.1f}% -> {cap*100:.1f}%")
        return cap
    return prob

def _poisson_implied(market: str, lh: float, la: float) -> float | None:
    """Compute Poisson-implied probability for a market given goal lambdas."""
    from scipy.stats import poisson as _poisson
    if market == 'u25':
        return sum(
            _poisson.pmf(h, lh) * _poisson.pmf(a, la)
            for h in range(5) for a in range(5) if h + a <= 2
        )
    if market == 'u35':
        return sum(
            _poisson.pmf(h, lh) * _poisson.pmf(a, la)
            for h in range(5) for a in range(5) if h + a <= 3
        )
    if market == 'btts_yes':
        return (1 - _poisson.pmf(0, lh)) * (1 - _poisson.pmf(0, la))
    if market == 'btts_no':
        btts_yes = (1 - _poisson.pmf(0, lh)) * (1 - _poisson.pmf(0, la))
        return 1.0 - btts_yes
    return None


def apply_lambda_aware_adjustment(
    prob: float,
    market: str,
    home_lambda: float,
    away_lambda: float,
    ctx: str = "",
    strength: float = 0.25,
) -> float:
    """Soft-pull market probability toward Poisson-implied value at given lambdas."""
    implied = _poisson_implied(market, home_lambda, away_lambda)
    if implied is None:
        return prob
    adjusted = prob + strength * (implied - prob)
    if ctx and abs(adjusted - prob) > 0.01:
        logger.debug(
            f"{ctx}: lambda_aware {market}: {prob*100:.1f}% -> {adjusted*100:.1f}%"
            f" (implied={implied*100:.1f}%, lh={home_lambda:.3f}, la={away_lambda:.3f})"
        )
    return adjusted


def get_confidence_tier(prob: float) -> str:
    """Get the confidence tier label for a probability."""
    for (low, high), tier in CONFIDENCE_TIERS.items():
        if low <= prob < high:
            return tier
    return "Speculative"


def enforce_h2h_lift_cap(raw: float, adjusted: float, h2h_matches: int, market: str = "unknown", ctx: str = "") -> float:
    """
    Centralized H2H lift cap enforcement with detailed logging.
    
    Rules:
    - Cap positive lift to 5pp (default) or 10pp (high sample)
    - Negative lift (confidence reduction) is always allowed
    - Log clipping events for future analysis
    """
    if adjusted <= raw:
        return adjusted  # Negative lift always allowed
    
    max_lift = MAX_H2H_LIFT_HIGH_SAMPLE if h2h_matches >= H2H_HIGH_SAMPLE_THRESHOLD else MAX_H2H_LIFT_DEFAULT
    lift = adjusted - raw
    
    if lift > max_lift:
        logger.info(
            f"{ctx}: H2H LIFT CLIPPED | market={market} raw_lift={lift:+.3f} applied_lift={max_lift:+.3f} sample_size={h2h_matches}"
        )
        return raw + max_lift
        
    return adjusted

def assert_confidence_bounds(raw_prob: float, final_prob: float, ctx: str) -> None:
    """
    Hard assertion: final probability should never exceed raw by more than 10pp.
    If this fires, the system has a bug.
    """
    lift = final_prob - raw_prob
    if lift > MAX_H2H_LIFT_HIGH_SAMPLE + 1e-6:
        raise ValueError(
            f"{ctx}: Confidence inflation violation. "
            f"raw={raw_prob:.3f}, final={final_prob:.3f}, lift={lift:.3f}, "
            f"max_allowed={MAX_H2H_LIFT_HIGH_SAMPLE:.3f}"
        )

def _validate_market_probs(p: Dict[str, Any]) -> MarketProbabilities:
    """Validate and convert dict to MarketProbabilities."""
    required = ['home', 'draw', 'away', 'u25', 'o25', 'btts', 'over_1_5']
    missing = [k for k in required if k not in p]
    if missing:
        raise DataValidationError(f"Missing probability keys: {missing}")
    return cast(MarketProbabilities, p)


def _goal_ensemble_weights() -> Tuple[float, float]:
    """Return normalized (lgbm, xgb) ensemble weights."""
    lgbm_weight = float(GOALS_ENSEMBLE_LGBM_WEIGHT)
    xgb_weight = float(GOALS_ENSEMBLE_XGB_WEIGHT)
    total = lgbm_weight + xgb_weight
    if total <= 0:
        logger.warning(
            "Invalid goal ensemble weights (lgbm=%s, xgb=%s). Falling back to 0.6/0.4.",
            lgbm_weight,
            xgb_weight,
        )
        return 0.6, 0.4
    return lgbm_weight / total, xgb_weight / total


def _divergence_pct(a: float, b: float) -> float:
    """Relative divergence between two lambdas in [0, +inf)."""
    denom = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / denom


def _calc_goals(match: pd.Series, suite: ModelSuite, ctx: str, simulator: "MatchSimulator", league: str | None = None, rl_weights: Dict[str, float] | None = None, use_simulator: bool = True) -> Dict[str, Any]:
    lgbm_feats = cast(List[str], suite['meta_goals']['features'])
    lh_lgbm = _predict_scalar(suite['mh_goals'], match, lgbm_feats, f"{ctx}:GoalsH:LGBM")
    la_lgbm = _predict_scalar(suite['ma_goals'], match, lgbm_feats, f"{ctx}:GoalsA:LGBM")

    xgb_home_model = cast(Optional[PredictionModel], suite.get("mh_goals_xgb") or suite.get("mh_goals_poisson"))
    xgb_away_model = cast(Optional[PredictionModel], suite.get("ma_goals_xgb") or suite.get("ma_goals_poisson"))
    xgb_meta = cast(
        Dict[str, Any],
        suite.get("meta_goals_xgb", suite.get("meta_goals_poisson", suite["meta_goals"])),
    )
    xgb_feats = cast(List[str], xgb_meta.get("features", lgbm_feats))

    if xgb_home_model is not None and xgb_away_model is not None:
        lh_xgb = _predict_scalar(xgb_home_model, match, xgb_feats, f"{ctx}:GoalsH:XGB")
        la_xgb = _predict_scalar(xgb_away_model, match, xgb_feats, f"{ctx}:GoalsA:XGB")
    else:
        # Backward-compatible fallback for tests/suites that only provide one goals model pair.
        lh_xgb = lh_lgbm
        la_xgb = la_lgbm

    lgbm_home_lambda = float(clamp_lambda(lh_lgbm))
    lgbm_away_lambda = float(clamp_lambda(la_lgbm))
    xgb_home_lambda = float(clamp_lambda(lh_xgb))
    xgb_away_lambda = float(clamp_lambda(la_xgb))

    lgbm_w, xgb_w = _goal_ensemble_weights()
    model_home_lambda = float(
        clamp_lambda((lgbm_w * lgbm_home_lambda) + (xgb_w * xgb_home_lambda))
    )
    model_away_lambda = float(
        clamp_lambda((lgbm_w * lgbm_away_lambda) + (xgb_w * xgb_away_lambda))
    )

    home_divergence = _divergence_pct(lgbm_home_lambda, xgb_home_lambda)
    away_divergence = _divergence_pct(lgbm_away_lambda, xgb_away_lambda)
    divergence_pct = max(home_divergence, away_divergence)
    ensemble_divergence = divergence_pct > GOALS_ENSEMBLE_DIVERGENCE_THRESHOLD

    if ensemble_divergence:
        match_id = match.get("match_id", "unknown")
        logger.warning(
            "%s (%s): GOALS_ENSEMBLE_DIVERGENCE %.1f%% | home lgbm=%.3f xgb=%.3f | away lgbm=%.3f xgb=%.3f",
            match_id,
            ctx,
            divergence_pct * 100.0,
            lgbm_home_lambda,
            xgb_home_lambda,
            lgbm_away_lambda,
            xgb_away_lambda,
        )

    sim_entropy: Optional[float] = None
    sim_tail_mass: Optional[float] = None
    sim_match_type: Optional[str] = None
    sim_n: Optional[int] = None
    sim_top_scorelines: List[Dict[str, float | str]] = []

    if use_simulator:
        sim_res = simulator.simulate(model_home_lambda, model_away_lambda, rl_weights=rl_weights)
        res = {
            'home_win': sim_res.home_win_prob,
            'draw': sim_res.draw_prob,
            'away_win': sim_res.away_win_prob,
            'over_2_5': sim_res.over_2_5,
            'under_2_5': sim_res.under_2_5,
            'over_1_5': sim_res.over_1_5,
            'under_3_5': sim_res.under_3_5,
            'btts_yes': sim_res.btts_prob,
            'btts_no': 1.0 - sim_res.btts_prob,
            'home_under_1_5': sim_res.home_under_1_5_prob,
            'away_under_1_5': sim_res.away_under_1_5_prob,
            'projected_home_goals': sim_res.expected_home_goals,
            'projected_away_goals': sim_res.expected_away_goals,
        }
        sim_entropy = sim_res.entropy
        sim_tail_mass = sim_res.tail_mass
        sim_match_type = sim_res.match_type
        sim_n = sim_res.n_simulations
        sim_top_scorelines = [
            {"score": f"{home}-{away}", "probability": prob}
            for (home, away), prob in sim_res.scoreline_probs.items()
        ]
    else:
        res = PoissonEngine().calculate_probabilities(model_home_lambda, model_away_lambda)
        # Normalise key name: PoissonEngine uses btts_yes, downstream code uses btts_yes too
        # but sim path used btts_prob attribute - res dict is already correct
        sim_entropy = None
        sim_tail_mass = None
        sim_match_type = None
        sim_n = None
        sim_top_scorelines = []
    
    # === H2H ADJUSTMENT FOR GOALS ===
    h2h_match_count = match.get('h2h_match_count', 0)
    h2h_goals_o25_rate = match.get('h2h_goals_o25_rate')
    
    o25_raw = res['over_2_5']
    o15_raw = res['over_1_5']
    o35_raw = 1.0 - res.get('under_3_5', 0.8)
    btts_raw = res['btts_yes']

    o25 = o25_raw
    btts = btts_raw
    
    # Apply H2H O2.5 rate adjustment for Goals markets
    if h2h_match_count >= 6 and h2h_goals_o25_rate is not None and not pd.isna(h2h_goals_o25_rate):
        if abs(o25 - h2h_goals_o25_rate) > 0.15:
            h2h_weight = min(h2h_match_count / 5, 0.5)
            o25_adjusted = (1 - h2h_weight) * o25 + h2h_weight * h2h_goals_o25_rate
            
            # === CENTRALIZED LIFT CAP ===
            o25_adjusted = enforce_h2h_lift_cap(o25_raw, o25_adjusted, h2h_match_count, market="goals_o25", ctx=ctx)
            
            if o25_adjusted != o25:
                logger.info(f"{ctx}: H2H Goals O2.5 - {o25_raw*100:.0f}% -> {o25_adjusted*100:.0f}% (h2h_rate={h2h_goals_o25_rate*100:.0f}%)")
            o25 = o25_adjusted
    
    # === GOAL COHERENCY SCALING ===
    if o25_raw <= 1e-6:
        logger.error(f"{ctx}: raw_o25 near zero ({o25_raw:.6f}) - skipping scaling")
        f = 1.0
    else:
        f = min(o25 / o25_raw, 1.0)
        
    o35 = o35_raw * f
    o15 = o15_raw * max(f, 0.85)
    btts = btts_raw * max(f, 0.9)

    # === HARD VALIDATIONS ===
    if not (o15 >= o25 >= o35):
        raise ValueError(
            f"{ctx}: Goal coherency violation. "
            f"Expected O1.5 >= O2.5 >= O3.5 but got "
            f"O1.5={o15:.3f}, O2.5={o25:.3f}, O3.5={o35:.3f}, scaling_factor={f:.3f}"
        )
    assert_confidence_bounds(o25_raw, o25, ctx)

    # European Handicap +2 (Underdog) calculation
    eh_prob = 0.0
    from src.ml.handicap import EuropeanHandicap
    underdog = "home" if model_home_lambda < model_away_lambda else "away"
    eh_prob = EuropeanHandicap.win_probability(model_home_lambda, model_away_lambda, 2, underdog)
    
    # === LAMBDA-AWARE SOFT ADJUSTMENT ===
    u25_pre_cap = apply_lambda_aware_adjustment(1.0 - o25, 'u25', model_home_lambda, model_away_lambda, ctx)
    u35_pre_cap = apply_lambda_aware_adjustment(1.0 - o35, 'u35', model_home_lambda, model_away_lambda, ctx)
    btts_pre_cap = apply_lambda_aware_adjustment(btts, 'btts_yes', model_home_lambda, model_away_lambda, ctx)
    # Layer 4.1b - League-specific BTTS cap (BL1 lambda inflation guard)
    # BL1 models learn historically accurate high-scoring patterns but
    # overestimate BTTS. Cap at 0.58 until sample grows beyond 50 resolved rows.
    # Reviewed: 2026-04-09. Revisit when BL1 btts_yes resolved N >= 50.
    BTTS_LEAGUE_CAPS = {"BL1": 0.58}
    if league and league in BTTS_LEAGUE_CAPS:
        btts_pre_cap = min(btts_pre_cap, BTTS_LEAGUE_CAPS[league])

    # === APPLY CALIBRATION CAPS (Layer 4.1 OOS guardrails) ===
    u25_capped = apply_calibration_cap(u25_pre_cap, 'u25', ctx)
    o25_final = 1.0 - u25_capped
    o15_final = max(o15, o25_final)
    if o15_final != o15:
        logger.warning(
            f"{ctx}: o15 re-anchored post u25-cap "
            f"(o15={o15:.3f} < o25_final={o25_final:.3f})"
        )
    u35_capped = apply_calibration_cap(u35_pre_cap, 'u35', ctx)
    btts_capped = apply_calibration_cap(btts_pre_cap, 'btts_yes', ctx)
    # btts_no is derived from btts_yes - btts_yes is the calibrated market
    btts_no_capped = 1.0 - btts_capped
    
    # New: Team Goal Caps
    h_u15_capped = apply_calibration_cap(res['home_under_1_5'], 'home_under_1_5', ctx)
    a_u15_capped = apply_calibration_cap(res['away_under_1_5'], 'away_under_1_5', ctx)
    
    return {
        'home': res['home_win'], 'draw': res['draw'], 'away': res['away_win'],
        'u25': u25_capped, 'o25': o25_final, 
        'u35': u35_capped,
        'btts': btts_capped,
        'btts_no': btts_no_capped,
        'over_1_5': o15_final, 
        'home_under_1_5': h_u15_capped, 
        'away_under_1_5': a_u15_capped,
        'eh_plus_2': eh_prob,
        'underdog': underdog,
        'expected_home_goals': float(res.get('projected_home_goals', model_home_lambda)),
        'expected_away_goals': float(res.get('projected_away_goals', model_away_lambda)),
        'goal_model_home_lambda': model_home_lambda,
        'goal_model_away_lambda': model_away_lambda,
        'ensemble_divergence': ensemble_divergence,
        'divergence_pct': float(divergence_pct * 100.0),
        'mc_entropy': sim_entropy,
        'mc_tail_mass': sim_tail_mass,
        'mc_match_type': sim_match_type,
        'mc_n_simulations': sim_n,
        'mc_top_scorelines': sim_top_scorelines,
    }

def _calc_corners(
    match: pd.Series, 
    suite: ModelSuite, 
    ctx: str, 
    league: str, 
    intensity_boost: bool = False,
    warning_collector: Optional[Dict[str, set]] = None
) -> Tuple[Dict[str, Optional[float]], Dict[str, Any]]:
    res_none = {'corn_u11': None, 'corn_o75': None, 'corn_1x2_h': None, 'corn_1x2_d': None, 'corn_1x2_a': None}
    attr = {}
    if not (suite.get('mh_corn') and suite.get('ma_corn')): return res_none, attr
    
    try:
        mu_h_raw = _predict_scalar(suite['mh_corn'], match, suite['mh_corn'].meta['features'], f"{ctx}:CornH")
        mu_a_raw = _predict_scalar(suite['ma_corn'], match, suite['ma_corn'].meta['features'], f"{ctx}:CornA")
        corners_divergence = False
        corners_divergence_pct = 0.0

        corners_lgbm = cast(Optional[PredictionModel], suite.get("m_corners_lgbm"))
        corners_xgb = cast(Optional[PredictionModel], suite.get("m_corners_xgb"))
        total_lgbm = None
        total_xgb = None

        if corners_lgbm is not None:
            lgbm_feats = cast(List[str], cast(Dict[str, Any], getattr(corners_lgbm, "meta", {}) or {}).get("features", []))
            if lgbm_feats:
                try:
                    total_lgbm = float(clamp_lambda(_predict_scalar(corners_lgbm, match, lgbm_feats, f"{ctx}:CornersTotal:LGBM")))
                except Exception as exc:
                    logger.warning("%s: Failed specialist corners LGBM prediction: %s", ctx, exc)
        if corners_xgb is not None:
            xgb_feats = cast(List[str], cast(Dict[str, Any], getattr(corners_xgb, "meta", {}) or {}).get("features", []))
            if xgb_feats:
                try:
                    total_xgb = float(clamp_lambda(_predict_scalar(corners_xgb, match, xgb_feats, f"{ctx}:CornersTotal:XGB")))
                except Exception as exc:
                    logger.warning("%s: Failed specialist corners XGB prediction: %s", ctx, exc)

        if total_lgbm is not None or total_xgb is not None:
            lgbm_w, xgb_w = _goal_ensemble_weights()
            if total_lgbm is None:
                total_lgbm = float(mu_h_raw + mu_a_raw)
            if total_xgb is None:
                total_xgb = float(total_lgbm)
            specialist_total = float(clamp_lambda((lgbm_w * total_lgbm) + (xgb_w * total_xgb)))
            base_total = max(float(mu_h_raw + mu_a_raw), 1e-9)
            specialist_scale = specialist_total / base_total
            mu_h_raw *= specialist_scale
            mu_a_raw *= specialist_scale

            corners_divergence_pct = _divergence_pct(total_lgbm, total_xgb)
            corners_divergence = corners_divergence_pct > GOALS_ENSEMBLE_DIVERGENCE_THRESHOLD
            if corners_divergence:
                match_id = match.get("match_id", "unknown")
                logger.warning(
                    "%s (%s): CORNERS_ENSEMBLE_DIVERGENCE %.1f%% | lgbm=%.3f xgb=%.3f",
                    match_id,
                    ctx,
                    corners_divergence_pct * 100.0,
                    total_lgbm,
                    total_xgb,
                )
        
        # Action 1/4: Offset Attribution
        offset_manager = TeamOffsetManager()
        off_h_obj = offset_manager.get_offset(match.get('home_team'), league)
        off_a_obj = offset_manager.get_offset(match.get('away_team'), league)
        
        # Verify offsets are loaded (graceful degradation with warning)
        if off_h_obj is None or off_a_obj is None:
            if warning_collector is not None:
                if off_h_obj is None: warning_collector['missing_offsets'].add(match.get('home_team'))
                if off_a_obj is None: warning_collector['missing_offsets'].add(match.get('away_team'))
            else:
                logger.warning(f"{ctx}: Team offsets not found - using defaults (home:{off_h_obj is not None}, away:{off_a_obj is not None})")
        
        off_h = off_h_obj.home_corner_bias if off_h_obj else 0.0
        off_a = off_a_obj.away_corner_bias if off_a_obj else 0.0
        
        v_h_raw = mu_h_raw + (getattr(suite['mh_corn'], 'alpha_', 0) * mu_h_raw**2) if hasattr(suite['mh_corn'], 'alpha_') else mu_h_raw * 1.3
        v_a_raw = mu_a_raw + (getattr(suite['ma_corn'], 'alpha_', 0) * mu_a_raw**2) if hasattr(suite['ma_corn'], 'alpha_') else mu_a_raw * 1.3
        res_raw = NegativeBinomialEngine().calculate_probabilities(mu_h_raw, v_h_raw, mu_a_raw, v_a_raw)
        
        mu_h = mu_h_raw * np.exp(off_h)
        mu_a = mu_a_raw * np.exp(off_a)
        
        # === H2H ADJUSTMENT FOR CORNERS (2026-01-13 Fix) ===
        # If we have H2H data, blend the model prediction with H2H history
        h2h_match_count = match.get('h2h_match_count', 0)
        h2h_avg_corners = match.get('h2h_avg_corners')
        
        if h2h_match_count >= 2 and h2h_avg_corners is not None and not pd.isna(h2h_avg_corners):
            # Weight H2H more heavily with more prior meetings (max 40% at 5+ meetings)
            h2h_weight = min(h2h_match_count / 5, 0.4)
            
            # Blend total corners expectation
            mu_total_model = mu_h + mu_a
            mu_total_blended = (1 - h2h_weight) * mu_total_model + h2h_weight * h2h_avg_corners
            
            # Cap upward H2H lift to 2.0 corners (consistent with goals/cards lift cap policy)
            # Negative lift (H2H history lower than model) is always allowed.
            MAX_H2H_CORNERS_LIFT = 2.0
            cap_active = False
            if mu_total_blended > mu_total_model:
                capped_total = min(mu_total_blended, mu_total_model + MAX_H2H_CORNERS_LIFT)
                cap_active = capped_total < mu_total_blended
                mu_total_blended = capped_total
            
            # Scale individual means proportionally
            scale = mu_total_blended / mu_total_model if mu_total_model > 0 else 1.0
            mu_h = mu_h * scale
            mu_a = mu_a * scale
            
            logger.info(
                f"{ctx}: H2H corner adjustment - total {mu_total_model:.1f} -> {mu_total_blended:.1f} "
                f"(h2h_avg={h2h_avg_corners:.1f}, h2h_count={h2h_match_count}, weight={h2h_weight:.2f}, "
                f"cap={'active' if cap_active else 'inactive'})"
            )
            
            attr['h2h_corner_adjustment'] = {
                'h2h_match_count': h2h_match_count,
                'h2h_avg_corners': h2h_avg_corners,
                'h2h_weight': h2h_weight,
                'mu_total_original': mu_total_model,
                'mu_total_blended': mu_total_blended
            }
        
        v_h = mu_h + (getattr(suite['mh_corn'], 'alpha_', 0) * mu_h**2) if hasattr(suite['mh_corn'], 'alpha_') else mu_h * 1.3
        v_a = mu_a + (getattr(suite['ma_corn'], 'alpha_', 0) * mu_a**2) if hasattr(suite['ma_corn'], 'alpha_') else mu_a * 1.3
        
        # Action 2: Apply Intensity Variance Boost (Production Promotion)
        v_mult = Thresholds.INTENSITY_VARIANCE_MULT if intensity_boost else 1.0
        v_h *= v_mult
        v_a *= v_mult
        
        res = NegativeBinomialEngine().calculate_probabilities(mu_h, v_h, mu_a, v_a)
        
        # Attribution Stamp
        attr['corn_1x2_h_attribution'] = {
            'raw_prob': res_raw['corners_home_win'],
            'adj_prob': res['corners_home_win'],
            'variance_mult': v_mult,
            'intensity_boost': intensity_boost
        }
        attr["ensemble_divergence"] = corners_divergence
        attr["divergence_pct"] = float(corners_divergence_pct * 100.0)
        
        # === ISOTONIC CALIBRATION (applied before soft cap) ===
        corn_o75_raw = res['corners_over_7_5']
        corn_u11_raw_nb = res['corners_under_11_5']

        # Load and apply corn_o75 calibrator if available
        cal_o75_file = suite['mh_corn'].meta.get('calibrator_corn_o75_filename') if suite.get('mh_corn') else None
        if cal_o75_file:
            cal_o75_path = MODELS_DIR / cal_o75_file
            cal_o75 = load_binary_calibrator_artifact(cal_o75_path)
            if cal_o75 and cal_o75.get('calibrator'):
                try:
                    corn_o75_raw = float(apply_binary_calibrator(
                        cal_o75['calibrator'], cal_o75['type'],
                        np.array([corn_o75_raw], dtype=float)
                    )[0])
                except Exception as exc:
                    logger.debug("%s: corn_o75 calibration apply failed: %s", ctx, exc)

        # Load and apply corn_u11 calibrator if available
        cal_u11_file = suite['mh_corn'].meta.get('calibrator_corn_u11_filename') if suite.get('mh_corn') else None
        if cal_u11_file:
            cal_u11_path = MODELS_DIR / cal_u11_file
            cal_u11 = load_binary_calibrator_artifact(cal_u11_path)
            if cal_u11 and cal_u11.get('calibrator'):
                try:
                    corn_u11_raw_nb = float(apply_binary_calibrator(
                        cal_u11['calibrator'], cal_u11['type'],
                        np.array([corn_u11_raw_nb], dtype=float)
                    )[0])
                except Exception as exc:
                    logger.debug("%s: corn_u11 calibration apply failed: %s", ctx, exc)

        # === SOFT CAP (Confidence-Aware) for O7.5 ===
        conf_calc = get_confidence_calculator()
        conf_res = conf_calc.calculate(
            prob=corn_o75_raw,
            market='corn_o75',
            league=league,
            mu=mu_h + mu_a,
            variance=v_h + v_a,
            h2h_count=match.get('h2h_match_count', 0),
            days_since_h2h=match.get('days_since_last_h2h', 0)
        )
        corn_o75 = apply_soft_cap(corn_o75_raw, conf_res.confidence, 'corn_o75')

        corn_u11_raw = corn_u11_raw_nb
        conf_u11 = get_confidence_calculator().calculate(
            prob=corn_u11_raw,
            market='corn_u11',
            league=league,
            mu=mu_h + mu_a,
            variance=v_h + v_a,
            h2h_count=match.get('h2h_match_count', 0),
            days_since_h2h=match.get('days_since_last_h2h', 0)
        )
        corn_u11 = apply_soft_cap(corn_u11_raw, conf_u11.confidence, 'corn_u11')
        
        return {
            'corn_u11': corn_u11, 
            'corn_o75': corn_o75,
            'conf_o75': conf_res.confidence,
            'corn_1x2_h': res['corners_home_win'], 
            'corn_1x2_d': res['corners_draw'], 
            'corn_1x2_a': res['corners_away_win']
        }, attr
    except (DataValidationError, ValueError, KeyError, AssertionError) as e:
        logger.warning(f"Corner calculation failed for {ctx}: {e}")
        return res_none, attr
    except Exception as e:
        logger.error(f"Unexpected error in corner calc for {ctx}: {e}", exc_info=True)
        return res_none, attr

def _calc_cards(
    match: pd.Series, 
    suite: ModelSuite, 
    ctx: str, 
    league: str, 
    intensity_boost: bool = False,
    warning_collector: Optional[Dict[str, set]] = None
) -> Tuple[Dict[str, Optional[float]], Dict[str, Any]]:
    res_none = {'card_u45': None, 'card_o25': None, 'card_u55': None}
    attr = {}
    try:
        m_legacy = cast(Optional[PredictionModel], suite.get("m_cards"))
        m_lgbm = cast(Optional[PredictionModel], suite.get("m_cards_lgbm"))
        m_xgb = cast(Optional[PredictionModel], suite.get("m_cards_xgb"))

        cards_divergence = False
        cards_divergence_pct = 0.0
        mu_lgbm_raw = None
        mu_xgb_raw = None

        if m_lgbm is not None:
            feats = cast(List[str], cast(Dict[str, Any], getattr(m_lgbm, "meta", {}) or {}).get("features", []))
            if feats:
                mu_lgbm_raw = _predict_scalar(m_lgbm, match, feats, f"{ctx}:Cards:LGBM")
        if m_xgb is not None:
            feats = cast(List[str], cast(Dict[str, Any], getattr(m_xgb, "meta", {}) or {}).get("features", []))
            if feats:
                mu_xgb_raw = _predict_scalar(m_xgb, match, feats, f"{ctx}:Cards:XGB")

        if mu_lgbm_raw is not None or mu_xgb_raw is not None:
            if mu_lgbm_raw is None:
                mu_lgbm_raw = float(mu_xgb_raw)
            if mu_xgb_raw is None:
                mu_xgb_raw = float(mu_lgbm_raw)

            lgbm_w, xgb_w = _goal_ensemble_weights()
            mu_raw = float(clamp_lambda((lgbm_w * mu_lgbm_raw) + (xgb_w * mu_xgb_raw)))
            cards_divergence_pct = _divergence_pct(float(mu_lgbm_raw), float(mu_xgb_raw))
            cards_divergence = cards_divergence_pct > GOALS_ENSEMBLE_DIVERGENCE_THRESHOLD
            if cards_divergence:
                match_id = match.get("match_id", "unknown")
                logger.warning(
                    "%s (%s): CARDS_ENSEMBLE_DIVERGENCE %.1f%% | lgbm=%.3f xgb=%.3f",
                    match_id,
                    ctx,
                    cards_divergence_pct * 100.0,
                    float(mu_lgbm_raw),
                    float(mu_xgb_raw),
                )
        else:
            if not m_legacy:
                return res_none, attr
            mu_raw = _predict_scalar(m_legacy, match, m_legacy.meta['features'], f"{ctx}:Cards")

        # Action 1/4: Offset Attribution
        offset_manager = TeamOffsetManager()
        off_h_obj = offset_manager.get_offset(match.get('home_team'), league)
        off_a_obj = offset_manager.get_offset(match.get('away_team'), league)
        
        # Verify offsets are loaded (graceful degradation with warning)
        if off_h_obj is None or off_a_obj is None:
            if warning_collector is not None:
                if off_h_obj is None: warning_collector['missing_card_offsets'].add(match.get('home_team'))
                if off_a_obj is None: warning_collector['missing_card_offsets'].add(match.get('away_team'))
            else:
                logger.warning(f"{ctx}: Card offsets not found - using defaults")
        
        off_h = off_h_obj.home_card_bias if off_h_obj else 0.0
        off_a = off_a_obj.away_card_bias if off_a_obj else 0.0

        
        res_raw = ZeroInflatedEngine().calculate_probabilities(mu_raw, pi_zero=0.03)
        
        mu = mu_raw * np.exp(off_h + off_a)
        
        # === H2H ADJUSTMENT ===
        # If we have H2H data, use BOTH the average cards AND the O2.5 rate
        h2h_match_count = match.get('h2h_match_count', 0)
        h2h_avg_cards = match.get('h2h_avg_cards')
        h2h_o25_rate = match.get('h2h_cards_o25_rate')  # This is more reliable than avg!
        
        if h2h_match_count >= 2 and h2h_avg_cards is not None and not pd.isna(h2h_avg_cards):
            # Weight H2H more heavily with more prior meetings (max 50% at 5+ meetings)
            h2h_weight = min(h2h_match_count / 5, 0.5)
            
            # Blend mu with H2H average
            mu_blended = (1 - h2h_weight) * mu + h2h_weight * h2h_avg_cards
            
            logger.info(f"{ctx}: H2H adjustment applied - mu {mu:.2f} -> {mu_blended:.2f} (h2h_avg={h2h_avg_cards:.2f}, h2h_count={h2h_match_count}, weight={h2h_weight:.2f})")
            mu = mu_blended
            
            # === CRITICAL: Use H2H O2.5 rate for probability capping ===
            if h2h_o25_rate is not None and not pd.isna(h2h_o25_rate) and h2h_match_count >= 3:
                model_o25_prob = 1.0 - ZeroInflatedEngine().calculate_probabilities(mu, pi_zero=0.03).get('cards_under_2_5', 0.23)
                
                if model_o25_prob > h2h_o25_rate + 0.20:
                    blend_weight = min(h2h_match_count / 5, 0.6)
                    capped_o25 = (1 - blend_weight) * model_o25_prob + blend_weight * h2h_o25_rate
                    
                    # === CENTRALIZED LIFT CAP ===
                    raw_o25_prob = 1.0 - res_raw.get('cards_under_2_5', 0.23)
                    capped_o25 = enforce_h2h_lift_cap(raw_o25_prob, capped_o25, h2h_match_count, market="card_o25", ctx=ctx)
                    
                    logger.info(f"{ctx}: H2H Cards O2.5 - model {model_o25_prob*100:.0f}% -> {capped_o25*100:.0f}% (h2h_rate={h2h_o25_rate*100:.0f}%)")
                    
                    attr['h2h_rate_cap'] = {
                        'model_o25': model_o25_prob,
                        'h2h_o25_rate': h2h_o25_rate,
                        'capped_o25': capped_o25
                    }

            
            attr['h2h_adjustment'] = {
                'h2h_match_count': h2h_match_count,
                'h2h_avg_cards': h2h_avg_cards,
                'h2h_o25_rate': h2h_o25_rate,
                'h2h_weight': h2h_weight,
                'mu_original': mu_raw * np.exp(off_h + off_a),
                'mu_blended': mu
            }
        else:
            # Log when H2H is NOT applied
            logger.debug(f"{ctx}: No H2H adjustment (h2h_count={h2h_match_count}, h2h_avg={h2h_avg_cards})")
        
        # Action 2: Apply Intensity Damping (Variance scaling for Zero-Inflated)
        v_mult = Thresholds.INTENSITY_VARIANCE_MULT if intensity_boost else 1.0
        res = ZeroInflatedEngine().calculate_probabilities(mu, pi_zero=0.03, var_mult=v_mult)
        
        # === APPLY H2H RATE CAP IF TRIGGERED ===
        card_o25 = res.get('cards_over_2_5', 1.0 - res['cards_under_4_5'])
        if 'h2h_rate_cap' in attr:
            card_o25 = attr['h2h_rate_cap']['capped_o25']
            logger.warning(f"{ctx}: Cards O2.5 CAPPED from {res.get('cards_over_2_5', 0)*100:.0f}% to {card_o25*100:.0f}% based on H2H rate")

        
        # Attribution Stamp
        attr['card_u55_attribution'] = {
            'raw_prob': res_raw.get('cards_under_5_5'),
            'adj_prob': res.get('cards_under_5_5'),
            'variance_mult': v_mult,
            'intensity_boost': intensity_boost
        }
        attr["ensemble_divergence"] = cards_divergence
        attr["divergence_pct"] = float(cards_divergence_pct * 100.0)
        
        # === APPLY CALIBRATION CAPS (Production Hardening) ===
        card_o25 = apply_calibration_cap(card_o25, 'card_o25', ctx)
        card_u45 = apply_calibration_cap(res['cards_under_4_5'], 'card_u45', ctx)
        
        # === SOFT CAP (Confidence-Aware) for U5.5 ===
        card_u55_raw = res.get('cards_under_5_5')
        if card_u55_raw:
            conf_calc = get_confidence_calculator()
            conf_res = conf_calc.calculate(
                prob=card_u55_raw,
                market='card_u55',
                league=league,
                mu=mu,
                variance=mu * 1.3,
                h2h_count=match.get('h2h_match_count', 0),
                days_since_h2h=match.get('days_since_last_h2h', 0)
            )
            card_u55 = apply_soft_cap(card_u55_raw, conf_res.confidence, 'card_u55')
        else:
            card_u55 = None

        return {
            'card_u45': card_u45, 
            'card_o25': card_o25,
            'card_u55': card_u55,
            'conf_u55': conf_res.confidence if card_u55_raw else None
        }, attr
    except (DataValidationError, ValueError, KeyError, AssertionError) as e:
        logger.warning(f"Card calculation failed for {ctx}: {e}")
        return res_none, attr
    except Exception as e:
        logger.error(f"Unexpected error in card calc for {ctx}: {e}", exc_info=True)
        return res_none, attr

def _calc_dc(probs: Dict[str, float], ctx: str) -> Dict[str, float]:
    """Derive Double Chance from 1X2 Probabilities."""
    try:
        # Use probabilities from Goals engine (Poisson 1X2)
        # Assuming 'home', 'draw', 'away' are in probs
        if not all(k in probs for k in ['home', 'draw', 'away']):
             return {'dc_1x': 0.0, 'dc_x2': 0.0, 'dc_12': 0.0}
             
        return DoubleChanceEngine.calculate(
            prob_home=probs['home'],
            prob_draw=probs['draw'],
            prob_away=probs['away'],
            strict=True
        )
    except Exception as e:
        logger.warning(f"DC calculation failed for {ctx}: {e}")
        return {'dc_1x': 0.0, 'dc_x2': 0.0, 'dc_12': 0.0}

# --- CORE ORCHESTRATION ---

@app.command(name="show-predictions")
def show_predictions(
    date: str = typer.Option("today", help="Date filter: 'yesterday', 'today', 'tomorrow', 'week', 'weekend', 'month', or day name e.g. 'friday', 'saturday'"),
    league: Optional[str] = typer.Option(None, "--league", "-l", help="Filter by league code or name."),
    all: bool = typer.Option(False, "--all", "-a", help="Show all future predictions."),
    tz: str = typer.Option("LOCAL", help="Timezone for date filtering (e.g. 'Europe/London', 'America/New_York')"),
    simulate: bool = typer.Option(
        True,
        "--simulate/--no-simulate",
        help="Use Monte Carlo simulation for goal markets (enabled by default)",
    ),
    use_rl_weights: bool = typer.Option(
        RL_BANDIT_LIVE_FLAG.exists(),
        "--use-rl-weights/--no-use-rl-weights",
        help="Run contextual bandit simulator weights in shadow mode and log the output only",
    ),
) -> None:
    """
    Display production match predictions with strict quality gates.
    
    Side Effects:
        - Loads ML models and processes features.
        - Prints multiple tables and panels to the console via _render_output.
        - Logs prediction events to the monitoring system.
    """
    console = Console()
    ALL_LEAGUES = ["PL", "BL1", "FL1", "SA", "PD"]
    if logging.getLogger().getEffectiveLevel() == logging.WARNING:
        os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
        logging.getLogger(__name__).setLevel(logging.ERROR)
        for logger_name in (
            "src.ml.registry",
            "src.monitoring",
            "src.predictions",
            "src.strategies",
            "src.core",
        ):
            logging.getLogger(logger_name).setLevel(logging.CRITICAL)
        warnings.filterwarnings(
            "ignore",
            message="Could not find the number of physical cores.*",
            module="joblib.externals.loky.backend.context",
        )
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module=r"joblib\.externals\.loky\.backend\.context",
        )
    try:
        # 1. Data Loading
        target_leagues = ALL_LEAGUES if (all and not league) else [league or "PL"]
        rendered_any = False

        for current_league in target_leagues:
            lg_val = resolve_league_code(current_league).value if current_league else None
            df = ServiceContainer.get_instance().pipeline.run(league=lg_val)
            validate_match_dataframe(
                df,
                context=f"show_predictions[{lg_val or 'PL'}]",
            )

            date_filter = date
            df_target = filter_matches_by_date(df, date_filter, show_all=False, user_timezone=tz)
            if df_target.empty:
                filter_label = date_filter.value if isinstance(date_filter, DateFilter) else str(date_filter)
                console.print(
                    f"[yellow][!] No matches found for filter: {filter_label} ({lg_val or current_league})[/yellow]"
                )
                continue

            # 2. Prediction Engine (Flattened)
            results = _run_predict_loop(
                df_target,
                use_simulator=simulate,
                use_rl_weights=use_rl_weights,
            )
            if not results:
                console.print(f"[yellow]No valid predictions generated for {lg_val or current_league}.[/yellow]")
                continue

            # 3. Strategy & Presentation
            gated, stats = SelectionGate().process(_prepare_bets(results))
            _render_output(results, gated, stats, console)
            rendered_any = True

            # Persist predictions to CSV for resolver
            _persist_predictions(results, current_league)

        if not rendered_any:
            return
        
    except Exception as e:
        logger.error("Prediction workflow failed", exc_info=True)
        raise typer.Exit(1)

def _is_high_intensity(match: pd.Series, league: str) -> bool:
    """Feature-driven high-intensity detector used for variance damping/boost."""
    cup_keywords = ['CUP', 'POKAL', 'COPA', 'TROPHY', 'SUPER']
    if any(k in str(league).upper() for k in cup_keywords):
        return True

    match_str = f"{match.get('home_team', '')} vs {match.get('away_team', '')}".upper()
    if any(k in match_str for k in cup_keywords):
        return True

    # Optional direct rivalry signals if available in upstream feature rows.
    for key in ('is_derby', 'derby_flag', 'is_rivalry'):
        val = match.get(key)
        if isinstance(val, (bool, np.bool_)) and val:
            return True
        if isinstance(val, (int, float)) and not pd.isna(val) and float(val) > 0:
            return True
        if isinstance(val, str) and val.strip().lower() in {'true', '1', 'yes', 'y'}:
            return True

    # Data-driven H2H intensity cues from historical match behavior.
    h2h_count = pd.to_numeric(pd.Series([match.get('h2h_match_count', 0)]), errors='coerce').iloc[0]
    h2h_cards_o25_rate = pd.to_numeric(pd.Series([match.get('h2h_cards_o25_rate')]), errors='coerce').iloc[0]
    h2h_avg_cards = pd.to_numeric(pd.Series([match.get('h2h_avg_cards')]), errors='coerce').iloc[0]
    days_since_last_h2h = pd.to_numeric(pd.Series([match.get('days_since_last_h2h')]), errors='coerce').iloc[0]

    if pd.isna(h2h_count) or h2h_count < 3:
        return False

    if not pd.isna(h2h_cards_o25_rate) and float(h2h_cards_o25_rate) >= 0.70:
        return True

    if not pd.isna(h2h_avg_cards) and float(h2h_avg_cards) >= 5.0:
        return True

    if (
        not pd.isna(days_since_last_h2h)
        and float(days_since_last_h2h) <= 45
        and (
            (not pd.isna(h2h_cards_o25_rate) and float(h2h_cards_o25_rate) >= 0.60)
            or (not pd.isna(h2h_avg_cards) and float(h2h_avg_cards) >= 4.5)
        )
    ):
        return True

    return False

def _run_predict_loop(
    df: pd.DataFrame,
    use_simulator: bool = True,
    use_rl_weights: bool = False,
) -> List[Dict[str, Any]]:
    """Flattened prediction loop with explicit Progress and consolidated warnings."""
    all_preds = []
    
    # Global warning collector to reduce noise
    warning_collector = {
        'missing_offsets': set(),
        'missing_card_offsets': set()
    }
    
    from src.cli.app import console
    from src.monitoring.drift_monitor import FeatureDriftMonitor
    from src.ml.registry import ModelRegistry
    
    feature_monitor = FeatureDriftMonitor()
    shadow_registry = ModelRegistry()
    logged_smart_routing = set()
    shadow_registry._logged_smart_routing = logged_smart_routing
    shadow_log: List[Dict[str, Any]] = []
    rl_bandit = ContextualBandit() if use_rl_weights else None
    
    with Progress(console=console) as progress:
        for lg in df['league'].unique():
            try:
                suite = _load_prediction_models(lg)
                live_registry = getattr(ServiceContainer.get_instance(), "_registry", None)
                if live_registry is not None:
                    live_registry._logged_smart_routing = logged_smart_routing
                matches = df[df['league'] == lg]
                
                # Fetch drift state once per league run — league-scoped first, global fallback
                try:
                    orchestrator = DriftOrchestrator()
                    league_state_file = (
                        Path(__file__).resolve().parents[3]
                        / "data" / "drift" / f"{lg}_drift_status.json"
                    )
                    if league_state_file.exists():
                        # Use per-league state — SA STOP must not block PL
                        league_status = orchestrator.evaluate_league_drift(lg)
                        if league_status == DriftOrchestrator.GO:
                            drift_state = "OK"
                        elif league_status == DriftOrchestrator.WATCH:
                            drift_state = "WATCH"
                        else:
                            drift_state = "STOP"
                        logger.debug("[%s] Using league-scoped drift state: %s", lg, drift_state)
                    else:
                        # No league file yet — fall back to global state (fails open)
                        orchestrator.load_global_state()
                        if orchestrator.global_status == DriftOrchestrator.GO:
                            drift_state = "OK"
                        elif orchestrator.global_status == DriftOrchestrator.WATCH:
                            drift_state = "WATCH"
                        else:
                            drift_state = "STOP"
                        logger.debug("[%s] No league drift file — using global state: %s", lg, drift_state)
                except Exception:
                    logger.error(
                        "[%s] DRIFT GATE FAIL-CLOSED: Could not load drift state — blocking predictions.",
                        lg,
                    )
                    drift_state = "STOP"

                # === CRITICAL GATE: DRIFT STOP HARD-BLOCK (league-scoped) ===
                if drift_state == "STOP":
                    logger.error(
                        "[%s] DRIFT STOP HARD_BLOCK: League-scoped drift STOP — predictions blocked.", lg
                    )
                    console.print(
                        f"[red bold]âŒ DRIFT STOP: {lg} predictions blocked. "
                        f"Run 'inspect-drift' or 'check-drift --league {lg}' to diagnose.[/red bold]"
                    )
                    continue  # Skip this league only — other leagues unaffected
                task_id = progress.add_task(f"[cyan]Predicting {lg}...", total=len(matches))
                simulator = MatchSimulator(
                    n_simulations=DEFAULT_N_SIMULATIONS,
                    seed=42,
                    league=lg,
                )
                
                for _, match in matches.iterrows():
                    try:
                        # Action 2: Detect High Intensity Fixtures
                        is_intensity = _is_high_intensity(match, lg)
                        
                        # Calculation Logic with consolidated warnings
                        p, attr = _calculate_probabilities_v2(
                            match,
                            suite,
                            lg,
                            is_intensity,
                            warning_collector,
                            simulator=simulator,
                            use_simulator=use_simulator,
                            rl_bandit=rl_bandit,
                        )
                        
                        # Inject drift state into nested attribution dicts only.
                        # Some attribution keys (e.g. ensemble_divergence) are scalars.
                        for key, value in attr.items():
                            if isinstance(value, dict):
                                value["drift_state"] = drift_state

                        # Traceability Snapshots
                        import hashlib
                        import json
                        input_snap = match.replace({np.nan: None}).to_dict()
                        snap_str = json.dumps(input_snap, sort_keys=True, default=str)
                        feat_hash = hashlib.md5(snap_str.encode()).hexdigest()

                        drift_eval = feature_monitor.check_live_prediction(
                            input_snapshot=input_snap, 
                            context_label=f"{lg} - {match.get('match_id')}"
                        )

                        # Replay Capability: enrich snapshot
                        replay_context = {
                            'data_pipeline_version': _get_pipeline_version_hash(),
                            'prediction_timestamp': datetime.now().isoformat(),
                            'external_context': {
                                'odds_snapshot': {
                                    k: match.get(k) for k in match.index
                                    if 'odds' in str(k).lower() or 'bet365' in str(k).lower()
                                    or 'pinnacle' in str(k).lower()
                                },
                            },
                        }

                        res = {
                            'match': f"{match['home_team']}{MATCH_SEPARATOR}{match['away_team']}",
                            'home_team': match['home_team'],
                            'away_team': match['away_team'],
                            'league': lg, 
                            'time': match['date'], 
                            'kickoff_utc': match.get('kickoff_utc'),
                            'match_id': match['match_id'], 
                            'h2h_match_count': match.get('h2h_match_count', 0),
                            'days_since_last_h2h': match.get('days_since_last_h2h', 0),
                            'attribution_stamp': attr,
                            'model_version': suite['mh_goals'].meta.get('version', 'unknown'),
                            'feature_hash': feat_hash,
                            'input_snapshot': input_snap,
                            'feature_drift_report': drift_eval,
                            'replay_context': replay_context,
                            **p
                        }
                        all_preds.append(res)
                        if rl_bandit is not None:
                            _run_rl_shadow_simulation(match, lg, res, rl_bandit)
                        _log_evt(lg, match, res, suite)

                        # Shadow Model Evaluation (Step 3)
                        _run_shadow_predictions(shadow_registry, match, lg, res, shadow_log)
                    except Exception as e:
                        logger.error(f"Prediction failed for {match.get('match_id')}: {e}")
                    finally:
                        progress.advance(task_id)
                        
            except ModelNotFoundError as e:
                logger.warning(f"Skipping league {lg}: {e.message}")
            
    # Print consolidated warnings after the progress bar is done
    if warning_collector['missing_offsets']:
        logger.warning(f"Consolidated: Team corner offsets missing for teams: {sorted(list(warning_collector['missing_offsets']))}")
    if warning_collector['missing_card_offsets']:
        logger.warning(f"Consolidated: Team card offsets missing for teams: {sorted(list(warning_collector['missing_card_offsets']))}")

    # Persist shadow predictions log
    if shadow_log:
        _persist_shadow_log(shadow_log)
            
    return all_preds


def _get_pipeline_version_hash() -> str:
    """Generate a deterministic hash of the feature pipeline source for replay."""
    import hashlib
    try:
        pipeline_path = Path(__file__).parent.parent.parent / 'features' / 'pipeline.py'
        if pipeline_path.exists():
            content = pipeline_path.read_bytes()
            return hashlib.sha256(content).hexdigest()[:12]
    except Exception as exc:
        logger.debug("Could not compute pipeline version hash: %s", exc)
    return "unknown"


def _run_shadow_predictions(
    registry: 'ModelRegistry',
    match: pd.Series,
    league: str,
    active_result: Dict[str, Any],
    shadow_log: List[Dict[str, Any]],
) -> None:
    """Run shadow models in parallel for comparison. Results are logged, NEVER bet on."""
    try:
        shadow_models = registry.get_shadow_models("poisson_home_base", league=league)
        if not shadow_models:
            return

        for shadow_meta in shadow_models:
            shadow_version = shadow_meta.get('version', 'unknown')
            logger.debug(f"Shadow model {shadow_version} evaluated for {match.get('match_id')}")
            shadow_log.append({
                'match_id': match.get('match_id'),
                'league': league,
                'shadow_version': shadow_version,
                'active_home': active_result.get('home'),
                'active_away': active_result.get('away'),
                'timestamp': datetime.now().isoformat(),
            })
    except Exception as e:
        logger.debug(f"Shadow prediction skipped: {e}")


def _persist_shadow_log(shadow_log: List[Dict[str, Any]]) -> None:
    """Persist shadow prediction comparisons to disk."""
    import json
    shadow_dir = DATA_DIR / "shadow_predictions"
    shadow_dir.mkdir(parents=True, exist_ok=True)
    filepath = shadow_dir / f"shadow_{datetime.now().strftime('%Y-%m-%d')}.jsonl"
    try:
        with open(filepath, 'a', encoding='utf-8') as f:
            for entry in shadow_log:
                f.write(json.dumps(entry, default=str) + '\n')
        logger.info(f"Persisted {len(shadow_log)} shadow predictions to {filepath}")
    except Exception as e:
        logger.error(f"Failed to persist shadow log: {e}")


def _persist_predictions(
    results: List[Dict[str, Any]],
    league: str,
) -> None:
    """
    Persist prediction results to data/predictions/ as a CSV file.
    One row per market per match. Called after _run_predict_loop completes.
    The resolver reads these files to resolve WON/LOST/VOID outcomes.
    """
    if not results:
        return

    PRED_DIR = DATA_DIR / "predictions"
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    # Markets to persist — must match resolver's MARKET_ALIASES targets
    # Keys are the keys in the res dict from _run_predict_loop
    MARKET_MAP = {
        'u25':             'goals_under_2_5',
        'o25':             'goals_over_2_5',
        'home_under_1_5':  'home_under_1_5',
        'away_under_1_5':  'away_under_1_5',
        'corn_u11':        'corn_u11',
        'corn_o75':        'corn_o75',
        'corn_home_u25':   'corn_home_u25',
        'corn_home_o25':   'corn_home_o25',
        'corn_home_u35':   'corn_home_u35',
        'corn_home_o35':   'corn_home_o35',
        'corn_home_u45':   'corn_home_u45',
        'corn_home_o45':   'corn_home_o45',
        'corn_home_u55':   'corn_home_u55',
        'corn_home_o55':   'corn_home_o55',
        'corn_away_u25':   'corn_away_u25',
        'corn_away_o25':   'corn_away_o25',
        'corn_away_u35':   'corn_away_u35',
        'corn_away_o35':   'corn_away_o35',
        'corn_away_u45':   'corn_away_u45',
        'corn_away_o45':   'corn_away_o45',
        'corn_away_u55':   'corn_away_u55',
        'corn_away_o55':   'corn_away_o55',
        'card_u45':        'cards_u4.5',
        'card_o25':        'cards_o2.5',
    }

    rows = []
    prediction_date = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

    for res in results:
        if res.get('league', '').upper() != league.upper():
            continue

        match_hash = str(res.get('match_id', ''))
        if not match_hash:
            continue

        kickoff = res.get('time', '')
        if hasattr(kickoff, 'isoformat'):
            kickoff_str = kickoff.isoformat()
        else:
            kickoff_str = str(kickoff)

        base = {
            'match_hash':       match_hash,
            'prediction_date':  prediction_date,
            'league':           res.get('league', league),
            'home_team':        res.get('home_team', ''),
            'away_team':        res.get('away_team', ''),
            'kickoff_date_utc': kickoff_str,
        }

        for res_key, market_name in MARKET_MAP.items():
            prob = res.get(res_key)
            if prob is None or not isinstance(prob, (int, float)):
                continue
            if pd.isna(prob):
                continue
            rows.append({
                **base,
                'market':               market_name,
                'predicted_probability': float(prob),
            })

    if not rows:
        logger.warning("No prediction rows to persist for league %s", league)
        return

    df = pd.DataFrame(rows)
    today = datetime.now().strftime('%Y%m%d')
    filename = PRED_DIR / f"predictions_{league}_{today}.csv"

    # Append to existing file if present (same-day re-runs)
    if filename.exists():
        existing = pd.read_csv(filename, encoding='utf-8')
        # Dedup on match_hash + market — keep newest
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=['match_hash', 'market'], keep='last'
        )
        df = combined

    tmp = filename.with_suffix('.tmp')
    df.to_csv(tmp, index=False, encoding='utf-8')
    tmp.replace(filename)
    logger.info(
        "Persisted %d prediction rows for %s to %s",
        len(df), league, filename.name
    )


def _run_rl_shadow_simulation(
    match: pd.Series,
    league: str,
    active_result: Dict[str, Any],
    bandit: ContextualBandit,
) -> None:
    """Log the delta between bandit-weighted and baseline MC output for monitoring."""
    try:
        home_lambda = float(active_result["goal_model_home_lambda"])
        away_lambda = float(active_result["goal_model_away_lambda"])
        context = bandit.context_key(league, RL_SHADOW_MARKET)
        weights = bandit.get_weights(context)

        # Baseline - no RL weights
        baseline = MatchSimulator(
            n_simulations=DEFAULT_N_SIMULATIONS,
            seed=42,
        ).simulate(home_lambda, away_lambda)

        # Bandit-weighted
        weighted = MatchSimulator(
            n_simulations=DEFAULT_N_SIMULATIONS,
            seed=42,
        ).simulate(home_lambda, away_lambda, rl_weights=weights)

        logger.info(
            "[RL-DIFF] match_id=%s context=%s weights=%s home_delta=%.4f draw_delta=%.4f away_delta=%.4f over_2_5_delta=%.4f btts_delta=%.4f",
            match.get("match_id"),
            context,
            weights,
            weighted.home_win_prob - baseline.home_win_prob,
            weighted.draw_prob - baseline.draw_prob,
            weighted.away_win_prob - baseline.away_win_prob,
            weighted.over_2_5 - baseline.over_2_5,
            weighted.btts_prob - baseline.btts_prob,
        )
    except Exception as exc:
        logger.debug(
            "RL diff simulation skipped for %s: %s",
            match.get("match_id"),
            exc,
        )

def _calculate_probabilities_v2(
    match: pd.Series, 
    suite: ModelSuite, 
    league: str, 
    intensity_boost: bool,
    warning_collector: Dict[str, set],
    simulator: "MatchSimulator",
    use_simulator: bool = True,
    rl_bandit=None,
) -> Tuple[MarketProbabilities, Dict[str, Any]]:
    """Local probability calculation that passes the warning collector."""
    p = {}
    attr = {}
    ctx = f"{match.get('home_team')}{MATCH_SEPARATOR}{match.get('away_team')}"
    
    # 1. Goals (Mandatory)
    rl_weights = None
    if rl_bandit is not None:
        context = rl_bandit.context_key(league, RL_SHADOW_MARKET)
        rl_weights = rl_bandit.get_weights(context)
    p.update(_calc_goals(match, suite, ctx, simulator=simulator, league=league, rl_weights=rl_weights, use_simulator=use_simulator))
    
    # 2. Secondary Markets (Best Effort)
    res_corn, attr_corn = _calc_corners(match, suite, ctx, league, intensity_boost, warning_collector)
    res_card, attr_card = _calc_cards(match, suite, ctx, league, intensity_boost, warning_collector)
    
    p.update(res_corn)
    p.update(res_card)
    p.update(_calc_dc(p, ctx))
    
    attr.update(attr_corn)
    attr.update(attr_card)
    
    return _validate_market_probs(p), attr

def _log_evt(lg: str, match: pd.Series, res: Dict[str, Any], suite: ModelSuite) -> None:
    log_event(PredictionEvent(
        timestamp=datetime.now(), event_type="prediction_generated",
        league=lg, match_id=res['match_id'], confidence=max(res['home'], res['away']),
        model_version=suite['mh_goals'].meta.get('version', 'unknown'),
        metadata={
            'home': match['home_team'], 
            'away': match['away_team'],
            'attribution_stamp': res.get('attribution_stamp')
        }
    ))

# --- PRESENTATION LAYER: HELPERS ---

def _format_corner_1x2(p: Dict[str, Any]) -> str:
    """Format corner 1X2 prediction for display."""
    h = p.get('corn_1x2_h') or 0
    a = p.get('corn_1x2_a') or 0
    
    if h == 0 and a == 0:
        return "-"
    
    winner = "H" if h > a else "A"
    prob = max(h, a)
    return f"{winner} {prob:.0%}"

def _safe_prob(p: Dict[str, Any], key: str) -> float:
    """Extract probability with explicit None handling and diagnostic logging."""
    val = p.get(key)
    if val is None:
        logger.debug(f"Missing probability key: {key}")
        return 0.0
    return float(val)


# --- UI ENHANCEMENT HELPERS ---

LEAGUE_PREFIXES = {
    "PL": "[PL]",
    "BL1": "[BL1]",
    "SA": "[SA]",
    "FL1": "[FL1]",
    "PD": "[PD]",
}


def _get_league_prefix(lg: str) -> str:
    """Get an ASCII-safe league prefix for console titles."""
    return LEAGUE_PREFIXES.get(lg, f"[{lg}]")


def _color_confidence(prob: float) -> str:
    """
    Color-code probability based on confidence level.
    
    - Green: High confidence (>70%)
    - Yellow: Medium (55-70%)
    - Dim: Low (<55%)
    """
    pct = f"{prob:.0%}"
    if prob >= 0.70:
        return f"[bold green]{pct}[/bold green]"
    elif prob >= 0.55:
        return f"[yellow]{pct}[/yellow]"
    else:
        return f"[dim]{pct}[/dim]"


def _format_edge(prob: float, implied_odds: float = 2.0) -> str:
    """
    Format edge percentage.
    
    Edge = Model Prob - Implied Market Prob
    Assumes 1/implied_odds as market probability if no odds available.
    """
    market_prob = 1 / implied_odds if implied_odds > 0 else 0.5
    edge = prob - market_prob
    
    if edge > 0.05:
        return f"[bold green]+{edge:.0%}[/bold green]"
    elif edge > 0:
        return f"[green]+{edge:.0%}[/green]"
    elif edge > -0.05:
        return f"[dim]{edge:.0%}[/dim]"
    else:
        return f"[red]{edge:.0%}[/red]"


def _get_terminal_width() -> int:
    """Get terminal width for responsive layout."""
    import shutil
    try:
        return shutil.get_terminal_size().columns
    except Exception:
        return 120  # Default

# --- PRESENTATION LAYER: MAIN ---

def _prepare_bets(preds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Transform raw predictions into a flat pool for SelectionGate.
    
    Args:
        preds: List of prediction dictionaries including market probabilities.
        
    Returns:
        Flattened list of potential bet candidates.
    """
    pool = []
    for p in preds:
        m_id = p['match_id']
        pool.extend([
            {**p, 'market': 'home_win', 'probability': _safe_prob(p, 'home'), 'selection': 'HOME', 'match_id': m_id},
            {**p, 'market': 'away_win', 'probability': _safe_prob(p, 'away'), 'selection': 'AWAY', 'match_id': m_id},
            {**p, 'market': 'goals_over_2_5', 'probability': _safe_prob(p, 'o25'), 'selection': 'O2.5', 'match_id': m_id},
            {**p, 'market': 'btts_yes', 'probability': _safe_prob(p, 'btts'), 'selection': 'BTTS', 'match_id': m_id},
            
            # Expanded Markets (User Request)
            {**p, 'market': 'home_goals_under_1_5', 'probability': _safe_prob(p, 'home_under_1_5'), 'selection': 'H U1.5', 'match_id': m_id},
            {**p, 'market': 'away_goals_under_1_5', 'probability': _safe_prob(p, 'away_under_1_5'), 'selection': 'A U1.5', 'match_id': m_id},
            {**p, 'market': 'cards_over_2_5', 'probability': _safe_prob(p, 'card_o25'), 'selection': 'Cards O2.5', 'match_id': m_id},
        ])
        
        if p.get('corn_o75'):
            pool.append({**p, 'market': 'corners_over_7_5', 'probability': _safe_prob(p, 'corn_o75'), 'selection': 'Corn O7.5', 'match_id': m_id})
        
        if p.get('corn_u11'):
            pool.append({**p, 'market': 'corners_under_11_5', 'probability': _safe_prob(p, 'corn_u11'), 'selection': 'Corn U11.5', 'match_id': m_id})
            
        if p.get('card_u55'):
            pool.append({**p, 'market': 'cards_under_5_5', 'probability': _safe_prob(p, 'card_u55'), 'selection': 'Cards U5.5', 'match_id': m_id})
            
        best_dc = max(p.get('dc_1x', 0), p.get('dc_x2', 0), p.get('dc_12', 0))
        pool.append({**p, 'market': 'double_chance', 'probability': best_dc, 'selection': 'DC', 'match_id': m_id})
    return pool

def _render_output(
    preds: List[Dict[str, Any]], 
    gated: List[Dict[str, Any]], 
    stats: Dict[str, Any], 
    console: Console
) -> None:
    """
    Unified rendering gateway for prediction output.
    
    Side Effects:
        - Prints league-specific prediction tables to the console.
        - Prints gated selections table to the console if candidates exist.
    """
    # 1. Prediction Tables
    term_width = _get_terminal_width()
    is_narrow = term_width < 110
    gate = SelectionGate()
    
    for lg in sorted({p['league'] for p in preds}):
        p_lg = [p for p in preds if p['league'] == lg]
        gated_lg = [b for b in gated if b.get('league') == lg]
        lg_n = LeagueCode(lg).full_name if lg in LeagueCode.__members__ else lg
        lg_prefix = _get_league_prefix(lg)
        t = Table(title=f"{lg_prefix} [bold cyan]{lg_n} ({lg})[/bold cyan]", box=box.ROUNDED, show_lines=True)
        
        # Define Thresholds Local Shortcuts
        TH_1X2 = Thresholds.GATE_PROB_1X2
        TH_DC = Thresholds.GATE_PROB_DC
        TH_GOALS = Thresholds.GATE_PROB_GOALS
        TH_BTTS = Thresholds.STRAT_GATE_BTTS
        TH_CORN_1X2 = 0.60 # Implicit for market direction
        TH_CORN_U11 = Thresholds.GATE_PROB_CORNERS
        TH_CORN_O75 = Thresholds.STRAT_GATE_CORNERS_O75
        TH_CARD_O25 = Thresholds.STRAT_GATE_CARDS_O25
        TH_CARD_U55 = Thresholds.STRAT_GATE_CARDS_U55
        TH_TEAM_U15 = Thresholds.STRAT_GATE_UND_U15

        def _c(val: float, th: float) -> str:
            """Colorize percentage with confidence colors."""
            if val >= 0.70:
                return f"[bold green]{val:.0%}[/bold green]"
            elif val >= th:
                return f"[green]{val:.0%}[/green]"
            elif val >= 0.55:
                return f"[yellow]{val:.0%}[/yellow]"
            else:
                return f"[dim]{val:.0%}[/dim]"

        def _p(val: float) -> str:
            """Format probability as int for compact display."""
            return f"{val * 100:.0f}"

        # Balanced columns - Corners are priority, hide BTTS/DC if very narrow
        t.add_column("Date", style="dim", width=6)
        t.add_column("Match", width=24)
        t.add_column("1X2", justify="center")
        t.add_column("xG", justify="center", style="dim", no_wrap=True)
        t.add_column("T", justify="center", width=5, no_wrap=True)
        t.add_column("Goals O/U2.5", justify="center")
        t.add_column("C-1X2", justify="center", no_wrap=True)
        t.add_column("Corners", justify="center", no_wrap=True)
        t.add_column("Cards", justify="center", no_wrap=True)
        t.add_column("TG U1.5", justify="center", no_wrap=True)
        
        # Stats tracking for summary
        high_conf_count = sum(
            1
            for bet in gated_lg
            if float(bet.get("probability", 0.0) or 0.0)
            >= gate.PROB_FLOORS.get(
                gate._get_market_type(str(bet.get("market", ""))),
                gate.PROB_FLOORS["default"],
            )
        )
        total_edge = 0.0
        
        for p in p_lg:
            # Format Team U1.5
            t_u15_h = _safe_prob(p, 'home_under_1_5')
            t_u15_a = _safe_prob(p, 'away_under_1_5')
            if t_u15_h > t_u15_a:
                 t_u15_str = f"H:{_c(t_u15_h, TH_TEAM_U15)}"
            else:
                 t_u15_str = f"A:{_c(t_u15_a, TH_TEAM_U15)}"
            
            if max(t_u15_h, t_u15_a) < 0.30: t_u15_str = "-"

            # Format Cards - Show only the most confident market
            c_o25 = _safe_prob(p, 'card_o25')
            c_u55 = _safe_prob(p, 'card_u55')
            
            if c_o25 >= c_u55:
                card_str = f"O2.5:{_c(c_o25, TH_CARD_O25)}"
            else:
                card_str = f"U5.5:{_c(c_u55, TH_CARD_U55)}"

            # 1X2 String (Expanded)
            ph, pd_prob, pa = _safe_prob(p, 'home'), _safe_prob(p, 'draw'), _safe_prob(p, 'away')
            s_1x2 = f"H:{_p(ph)} D:{_p(pd_prob)} A:{_p(pa)}"
            # Highlight best
            m1x2 = max(ph, pd_prob, pa)
            if m1x2 >= TH_1X2:
                if ph == m1x2: s_1x2 = s_1x2.replace(f"H:{_p(ph)}", f"[green]H:{_p(ph)}[/green]")
                elif pd_prob == m1x2: s_1x2 = s_1x2.replace(f"D:{_p(pd_prob)}", f"[green]D:{_p(pd_prob)}[/green]")
                else: s_1x2 = s_1x2.replace(f"A:{_p(pa)}", f"[green]A:{_p(pa)}[/green]")

            # xG and match type
            xg_h = p.get('expected_home_goals')
            xg_a = p.get('expected_away_goals')
            if xg_h is not None and xg_a is not None:
                s_xg = f"{xg_h:.1f}-{xg_a:.1f}"
            else:
                s_xg = "-"
            mc_type = p.get('mc_match_type', '')
            type_map = {'high_uncertainty': '[yellow]UNC[/yellow]', 'balanced': '[cyan]BAL[/cyan]', 'predictable': '[dim]PRD[/dim]'}
            s_type = type_map.get(mc_type, '-')

            
            # DC String
            best_dc = max(p.get('dc_1x', 0), p.get('dc_x2', 0), p.get('dc_12', 0))
            if best_dc == p.get('dc_1x'): dc_lbl = "1X"
            elif best_dc == p.get('dc_x2'): dc_lbl = "X2"
            else: dc_lbl = "12"
            s_dc = f"{dc_lbl} {_c(best_dc, TH_DC)}"
            
            # EH+2 (Underdog) - New Column Logic
            peh2 = p.get('eh_plus_2', 0.0)
            underdog = p.get('underdog', '')
            s_handicap = "-"
            if peh2 > 0:
                team_short = "H" if underdog == "home" else "A"
                s_handicap = f"{team_short}+2 {_c(peh2, 0.75)}"
            
            # Goals String — explicit threshold labels
            po25, pu25 = _safe_prob(p, 'o25'), _safe_prob(p, 'u25')
            s_goals = f"O2.5:{_c(po25, TH_GOALS)} U2.5:{_c(pu25, TH_GOALS)}"
            
            # U3.5 (new low-variance market) - inline append
            pu35 = _safe_prob(p, 'u35')
            if pu35 > 0.70:
                s_goals += f" [bold green]U3.5:{pu35:.0%}[/bold green]"
            elif pu35 > 0.60:
                s_goals += f" U3.5:{pu35:.0%}"
            
            # BTTS String - Show both Yes and No for forecast visibility
            pbtts_yes = _safe_prob(p, 'btts')
            pbtts_no = _safe_prob(p, 'btts_no')
            s_btts = f"Y:{_c(pbtts_yes, TH_BTTS)}"
            
            # Corn 1X2 - Manual check
            ch, ca = p.get('corn_1x2_h', 0), p.get('corn_1x2_a', 0)
            if not ch or not ca or (ch == 0 and ca == 0):
                s_corn_1x2 = "-"
            else:
                winner = "H" if ch > ca else "A"
                prob = max(ch, ca)
                s_corn_1x2 = f"{winner} {_c(prob, TH_CORN_1X2)}"
            
            # Corners - Show only the most confident market (O7.5 vs U11.5)
            c_o75 = _safe_prob(p, 'corn_o75')
            c_u11 = _safe_prob(p, 'corn_u11')
            
            if c_o75 >= c_u11:
                s_corn_str = f"O7.5:{_c(c_o75, TH_CORN_O75)}"
            else:
                s_corn_str = f"U11.5:{_c(c_u11, TH_CORN_U11)}"
            
            # Track stats for summary
            best_prob = max(ph, pd_prob, pa, po25, pbtts_yes)
            total_edge += (best_prob - 0.5)  # Simple edge calculation

            # Build row based on terminal width
            t.add_row(
                p['time'].strftime('%m-%d'), p['match'],
                s_1x2, s_xg, s_type,
                s_goals, s_corn_1x2, s_corn_str, card_str, t_u15_str
            )
        
        console.print(t)
        
        # Summary row
        avg_edge = (total_edge / len(p_lg)) if p_lg else 0
        edge_color = "green" if avg_edge > 0.05 else "yellow" if avg_edge > 0 else "dim"
        console.rule(style="dim")
        # Market breakdown from gated selections for this league
        market_counts: Dict[str, int] = {}
        for b in gated_lg:
            sel = str(b.get('selection', ''))
            # Normalise to market type: "H U1.5" -> "U1.5", "Cards O2.5" -> "Cards",
            # "Corn U11.5" -> "Corners", "DC" -> "DC", "O2.5" -> "Goals O2.5"
            sel_upper = sel.upper()
            if 'U1.5' in sel_upper:
                mkt = 'U1.5'
            elif 'CORN' in sel_upper:
                mkt = 'Corners'
            elif 'CARD' in sel_upper:
                mkt = 'Cards'
            elif 'DC' in sel_upper:
                mkt = 'DC'
            elif 'O2.5' in sel_upper or 'O25' in sel_upper:
                mkt = 'Goals O2.5'
            elif 'HANDICAP' in sel_upper or 'AH' in sel_upper:
                mkt = 'Handicap'
            else:
                mkt = sel.split()[0] if sel else 'Other'
            market_counts[mkt] = market_counts.get(mkt, 0) + 1
        market_str = " | ".join(f"{k}:{v}" for k, v in sorted(market_counts.items()))
        console.print(
            f"[bold]Summary:[/bold] {len(p_lg)} matches | "
            f"[green]High Conf: {high_conf_count}[/green] | "
            f"[{edge_color}]Avg Edge: {avg_edge:+.1%}[/{edge_color}]"
            + (f" | Markets: {market_str}" if market_str else "")
        )
        console.print(f"[dim italic]Legend: [bold green]Green[/bold green]=High(>70%) [yellow]Yellow[/yellow]=Medium(55-70%) [dim]Gray[/dim]=Low(<55%)[/dim italic]")
        console.print("[dim]Note: Double Chance is shown for forecast context only.[/dim]\n")
    
    # 2. Gated Selections
    if gated:
        gt = Table(title="[bold green]GATED SELECTIONS (High Value)[/bold green]", box=box.HEAVY_EDGE)
        gt.add_column("Date", style="dim"); gt.add_column("Match"); gt.add_column("Selection", style="bold cyan")
        gt.add_column("Prob", justify="right")
        gt.add_column("CI ±", justify="right", style="dim")
        gt.add_column("Edge", justify="right")
        gt.add_column("Score", justify="right")
        for b in gated:
            prob = float(b['probability'])
            n = 50000
            ci = 1.645 * (prob * (1 - prob) / n) ** 0.5
            edge = prob - 0.50
            edge_color = "green" if edge >= 0.20 else "yellow" if edge >= 0.10 else "dim"
            gt.add_row(
                b['time'].strftime('%m-%d %H:%M'),
                b['match'],
                b['selection'],
                f"{prob:.1%}",
                f"{ci*100:.1f}%",
                f"[{edge_color}]{edge:+.1%}[/{edge_color}]",
                f"{b['gate_score']:.3f}",
            )
        console.print(gt)
        
# --- CLI ENTRY POINTS ---

@app.command(name="predict")
def predict(
league: str = typer.Argument(..., help="League Code (e.g. PL, BL1)")) -> None:
    """
    End-to-end prediction workflow with local data orchestration.
    
    Side Effects:
        - Orchestrates fixture fetching if data is stale.
        - Delegates rendering to show_predictions.
    """
    from src.cli.commands.data import fetch_upcoming
    from src.config import DATA_FRESHNESS_DAYS
    import time
    
    code = resolve_league_code(league)
    if not code: raise typer.Exit(1)
    
    # Orchestration Layer: Enforce data freshness
    path = PROCESSED_DATA_DIR / "matches" / f"{code.value}_upcoming.csv"
    if not (path.exists() and (time.time() - path.stat().st_mtime) < 86400 * DATA_FRESHNESS_DAYS):
        fetch_upcoming(league=code)
       
    show_predictions(date="weekend", league=league, all=False)

