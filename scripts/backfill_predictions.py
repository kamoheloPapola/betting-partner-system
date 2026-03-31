import pandas as pd
import logging
from pathlib import Path
from rich.console import Console
from datetime import datetime

from src.core.container import ServiceContainer
from src.cli.utils import LeagueCode, resolve_league_code, MATCH_SEPARATOR
from src.cli.commands.prediction import _run_predict_loop, _prepare_bets, _load_prediction_models 
from src.config import DATA_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def backfill(leagues: list[str], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    container = ServiceContainer.get_instance()
    
    all_preds_df = []
    
    for league_str in leagues:
        logger.info(f"Processing {league_str}...")
        code = resolve_league_code(league_str)
        if not code:
            logger.warning(f"Invalid league: {league_str}")
            continue
            
        # Load historical matches (using the pipeline logic from backtest/app)
        # We want *completed* matches to calibrate against actuals
        # pipeline.run() usually loads everything if we don't filter?
        # Let's verify pipeline behavior or simpler: load processed matches directly?
        # But we need FEATURES. So we must run pipeline.
        
        # Determine seasons to load? Usually data/raw/[league]/*.csv
        # Pipeline usually picks up what's available.
        
        try:
            df = container.pipeline.run(league=code.value)
            
            # Filter for completed matches (have scores/corners)
            # We assume 'home_score' exists for completed matches
            if 'home_score' in df.columns:
                df = df.dropna(subset=['home_score', 'home_corners']).copy()
            else:
                 logger.warning(f"No score data found for {league_str}")
                 continue
                 
            if df.empty:
                 logger.warning(f"No completed matches found for {league_str}")
                 continue
            
            logger.info(f"Generating predictions for {len(df)} matches in {league_str}...")
            
            # Run prediction loop
            # _run_predict_loop expects df with features
            preds = _run_predict_loop(df)
            
            if preds:
                # Convert to simple log format expected by calibration
                # ID, LEAGUE, PROBS...
                # Calibration expects: match_id, league, P_under_11_5, etc.
                
                # We need to extract the specific corner probabilities from the complex dict
                # The _run_predict_loop returns a list of dicts with keys like 'corn_u11', 'corn_o75'
                
                rows = []
                for p in preds:
                    rows.append({
                        'match_id': p['match_id'],
                        'league': p['league'],
                        'P_under_11_5': p.get('corn_u11'),
                        'P_over_7_5': p.get('corn_o75')
                    })
                
                pd.DataFrame(rows).to_csv(output_dir / f"predictions_corners_backfill_{league_str}.csv", index=False)
                logger.info(f"Saved {len(rows)} predictions for {league_str}")
                
        except Exception as e:
            logger.error(f"Failed to backfill {league_str}: {e}", exc_info=True)

if __name__ == "__main__":
    leagues = ['PD', 'SA', 'BL1', 'PL']
    out_path = Path("data/temp_backfill_logs")
    backfill(leagues, out_path)
    print(f"Backfill complete. Logs in {out_path}")
