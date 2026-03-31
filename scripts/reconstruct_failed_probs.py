import pandas as pd
import numpy as np
import logging
import pickle
from pathlib import Path
from typing import Dict, Any, List

# Core imports from project
from src.core.container import ServiceContainer
from src.ml.distributions import PoissonEngine, NegativeBinomialEngine, ZeroInflatedEngine
from src.strategies.selection_gate import SelectionGate
from src.config.thresholds import Thresholds
from src.cli.utils import MATCH_SEPARATOR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Reconstructor")

def reconstruct():
    # 1. Load data
    df = pd.read_csv("failed_matches_features_full.csv")
    
    # 2. Teams to investigate
    matches = [
        {"id": "4719e5144c1122b1", "league": "BL1", "home": "SC Freiburg", "away": "Hamburger SV"},
        {"id": "1d206c2b7fa9ed5e", "league": "SA", "home": "Udinese", "away": "Pisa"}
    ]
    
    # 3. Offsets from Step 0
    OFFSETS = {
        "SC Freiburg": {"home_corn": -0.24299, "away_corn": -0.22672},
        "Udinese": {"home_corn": -0.37227, "away_corn": 0.01145},
        "Hamburger SV": {"home_corn": 0.0, "away_corn": 0.0},
        "Pisa": {"home_corn": 0.0, "away_corn": 0.0}
    }

    container = ServiceContainer.get_instance()
    reg = container.registry
    
    results = []

    for m_info in matches:
        match_id = m_info["id"]
        league = m_info["league"]
        home_team = m_info["home"]
        away_team = m_info["away"]
        
        row = df[df['match_id'] == match_id].iloc[0]
        
        # Load Models
        m_goals_h = reg.load_model("poisson_home_base", league=league)
        m_goals_a = reg.load_model("poisson_away_base", league=league)
        m_corn_h = reg.load_model("nb_home_corners_base", league=league) or reg.load_model("poisson_home_corners_base", league=league)
        m_corn_a = reg.load_model("nb_away_corners_base", league=league) or reg.load_model("poisson_away_corners_base", league=league)
        m_cards = reg.load_model("poisson_total_cards_base", league=league)
        
        # --- CARDS RECONSTRUCTION ---
        card_feats = m_cards.meta['features']
        card_input = pd.DataFrame([pd.to_numeric(row[card_feats])])
        card_mu_raw = float(m_cards.predict(card_input)[0])
        
        # Probability Under 5.5
        # prediction.py uses ZeroInflatedEngine with pi_zero=0.1
        cards_res = ZeroInflatedEngine().calculate_probabilities(card_mu_raw, pi_zero=0.1)
        prob_cards_u55 = cards_res.get('cards_under_5_5', 0)
        
        # --- CORNERS RECONSTRUCTION ---
        corn_feats_h = m_corn_h.meta['features']
        corn_feats_a = m_corn_a.meta['features']
        
        corn_input_h = pd.DataFrame([pd.to_numeric(row[corn_feats_h])])
        corn_input_a = pd.DataFrame([pd.to_numeric(row[corn_feats_a])])
        
        mu_h_raw = float(m_corn_h.predict(corn_input_h)[0])
        mu_a_raw = float(m_corn_a.predict(corn_input_a)[0])
        
        # Variance calc from prediction.py
        v_h = mu_h_raw + (getattr(m_corn_h, 'alpha_', 0) * mu_h_raw**2) if hasattr(m_corn_h, 'alpha_') else mu_h_raw * 1.3
        v_a = mu_a_raw + (getattr(m_corn_a, 'alpha_', 0) * mu_a_raw**2) if hasattr(m_corn_a, 'alpha_') else mu_a_raw * 1.3
        
        corn_res_raw = NegativeBinomialEngine().calculate_probabilities(mu_h_raw, v_h, mu_a_raw, v_a)
        
        # Simulate with offsets (if they had been applied)
        offset_h = OFFSETS[home_team]["home_corn"]
        offset_a = OFFSETS[away_team]["away_corn"]
        
        mu_h_adj = mu_h_raw * np.exp(offset_h)
        mu_a_adj = mu_a_raw * np.exp(offset_a)
        
        v_h_adj = mu_h_adj + (getattr(m_corn_h, 'alpha_', 0) * mu_h_adj**2) if hasattr(m_corn_h, 'alpha_') else mu_h_adj * 1.3
        v_a_adj = mu_a_adj + (getattr(m_corn_a, 'alpha_', 0) * mu_a_adj**2) if hasattr(m_corn_a, 'alpha_') else mu_a_adj * 1.3
        
        corn_res_adj = NegativeBinomialEngine().calculate_probabilities(mu_h_adj, v_h_adj, mu_a_adj, v_a_adj)
        
        results.append({
            "match": f"{home_team} vs {away_team}",
            "market": "Cards U5.5",
            "mu_raw": card_mu_raw,
            "prob_raw": prob_cards_u55,
            "threshold": Thresholds.STRAT_GATE_CARDS_U55,
            "diff": prob_cards_u55 - Thresholds.STRAT_GATE_CARDS_U55
        })
        
        results.append({
            "match": f"{home_team} vs {away_team}",
            "market": "Corners 1X2 (H)",
            "mu_h_raw": mu_h_raw,
            "mu_a_raw": mu_a_raw,
            "prob_raw": corn_res_raw['corners_home_win'],
            "prob_adj": corn_res_adj['corners_home_win'],
            "threshold": Thresholds.STRAT_GATE_CORNERS,
            "offset_h": offset_h,
            "offset_a": offset_a
        })

    print("\n--- RECONSTRUCTION RESULTS ---")
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    reconstruct()
