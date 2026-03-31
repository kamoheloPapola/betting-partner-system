"""
Grid Search for Optimal Alpha Values.

Runs systematic hyperparameter search per league to find optimal
regularization values. Validates adoption criteria before updating.

Usage:
    python -m scripts.grid_search_alpha --league PL
"""
import argparse
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from src.backtest.engine import Backtester
from src.config.alpha_config import (
    GRID_SEARCH_ALPHAS,
    MIN_DELTA_BRIER,
    CALIBRATION_SLOPE_RANGE,
    ALPHA_MAP,
    DEFAULT_ALPHA
)
from src.config import DATA_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def calculate_calibration_slope(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate calibration slope via logistic regression.
    
    A slope of 1.0 indicates perfect calibration.
    < 1.0 = overconfident, > 1.0 = underconfident
    """
    from sklearn.linear_model import LogisticRegression
    
    # Convert to binary (for 1x2, use "home_win" as the target)
    if len(y_pred) == 0:
        return 1.0
    
    # Clip to avoid log(0)
    y_pred_clipped = np.clip(y_pred, 0.01, 0.99)
    
    # Logit transform
    logit_pred = np.log(y_pred_clipped / (1 - y_pred_clipped)).reshape(-1, 1)
    
    try:
        lr = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
        lr.fit(logit_pred, y_true)
        slope = lr.coef_[0][0]
    except Exception:
        slope = 1.0
    
    return float(slope)


def run_grid_search(
    league: str,
    test_season: int = 2024,
    model_types: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """
    Run grid search for a single league.
    
    Returns dict mapping model_type -> {best_alpha, brier, slope, delta}
    """
    if model_types is None:
        model_types = ["poisson_home_base", "poisson_away_base"]
    
    results = {}
    
    # Get current baseline
    current_alpha = ALPHA_MAP.get(league, {}).get(model_types[0], DEFAULT_ALPHA)
    
    for model_type in model_types:
        logger.info(f"Grid search for {league}/{model_type}")
        
        best_alpha = current_alpha
        best_brier = 999.0
        best_slope = 1.0
        
        for alpha in GRID_SEARCH_ALPHAS:
            logger.info(f"  Testing alpha={alpha}")
            
            try:
                # Run backtest with this alpha
                tester = Backtester()
                # TODO: Inject alpha into backtest engine
                # For now, this is a placeholder for the structure
                bt_results = tester.run(
                    train_seasons=list(range(2018, test_season)),
                    test_season=test_season,
                    league=league
                )
                
                if bt_results.empty:
                    continue
                
                brier = bt_results['brier_home'].mean()
                
                # Calculate calibration slope
                y_true = (bt_results['actual'] == 'HOME_WIN').astype(int).values
                y_pred = bt_results['prob_home'].values
                slope = calculate_calibration_slope(y_true, y_pred)
                
                logger.info(f"    Brier={brier:.4f}, Slope={slope:.3f}")
                
                if brier < best_brier:
                    best_brier = brier
                    best_alpha = alpha
                    best_slope = slope
                    
            except Exception as e:
                logger.warning(f"    Failed: {e}")
                continue
        
        # Check adoption criteria
        delta = (ALPHA_MAP.get(league, {}).get(model_type, DEFAULT_ALPHA) - best_brier)
        
        adoption_pass = True
        rejection_reason = None
        
        if delta < MIN_DELTA_BRIER:
            adoption_pass = False
            rejection_reason = f"ΔBrier {delta:.4f} < {MIN_DELTA_BRIER}"
        
        if not (CALIBRATION_SLOPE_RANGE[0] <= best_slope <= CALIBRATION_SLOPE_RANGE[1]):
            adoption_pass = False
            rejection_reason = f"Slope {best_slope:.3f} outside [{CALIBRATION_SLOPE_RANGE[0]}, {CALIBRATION_SLOPE_RANGE[1]}]"
        
        results[model_type] = {
            "best_alpha": best_alpha,
            "brier": best_brier,
            "slope": best_slope,
            "delta": delta,
            "adopt": adoption_pass,
            "rejection_reason": rejection_reason
        }
        
        if adoption_pass:
            logger.info(f"✓ ADOPT: {model_type} alpha={best_alpha} (Brier={best_brier:.4f})")
        else:
            logger.info(f"✗ REJECT: {model_type} - {rejection_reason}")
    
    return results


def save_results(results: Dict[str, Dict], league: str) -> None:
    """Save grid search results to CSV."""
    output_dir = DATA_DIR / "grid_search"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rows = []
    for model_type, data in results.items():
        rows.append({
            "league": league,
            "model_type": model_type,
            **data
        })
    
    df = pd.DataFrame(rows)
    path = output_dir / f"grid_search_{league}.csv"
    df.to_csv(path, index=False)
    logger.info(f"Results saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Grid search for optimal alpha values")
    parser.add_argument("--league", type=str, required=True, help="League code (e.g., PL)")
    parser.add_argument("--test-season", type=int, default=2024, help="Season to test on")
    args = parser.parse_args()
    
    results = run_grid_search(args.league, args.test_season)
    save_results(results, args.league)
    
    # Print summary
    print("\n=== GRID SEARCH SUMMARY ===")
    for model_type, data in results.items():
        status = "✓ ADOPT" if data['adopt'] else "✗ REJECT"
        print(f"{status}: {model_type} -> alpha={data['best_alpha']} (Brier={data['brier']:.4f})")


if __name__ == "__main__":
    main()
