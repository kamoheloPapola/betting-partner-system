import pandas as pd
import numpy as np
import logging
from src.core.container import ServiceContainer
from src.ml.distributions import PoissonEngine, NegativeBinomialEngine, ZeroInflatedEngine
from src.strategies.forbidden_fruit import ForbiddenFruitEngine, ForbiddenFruitEvaluator
from src.config.thresholds import Thresholds

logging.basicConfig(level=logging.INFO)

def audit():
    df = pd.read_csv("failed_matches_features_full.csv")
    engine = ForbiddenFruitEngine()
    evaluator = ForbiddenFruitEvaluator()
    
    matches_meta = [
        {"id": "4719e5144c1122b1", "league": "BL1", "home": "SC Freiburg", "away": "Hamburger SV"},
        {"id": "1d206c2b7fa9ed5e", "league": "SA", "home": "Udinese", "away": "Pisa"}
    ]

    for m_info in matches_meta:
        match_id = m_info["id"]
        league = m_info["league"]
        row = df[df['match_id'] == match_id].iloc[0]
        match_dict = row.to_dict()
        match_dict['league'] = league
        
        # 1. Re-calculate raw probs (as done in prediction.py)
        container = ServiceContainer.get_instance()
        reg = container.registry
        m_h_corn = reg.load_model("nb_home_corners_base", league=league) or reg.load_model("poisson_home_corners_base", league=league)
        m_a_corn = reg.load_model("nb_away_corners_base", league=league) or reg.load_model("poisson_away_corners_base", league=league)
        m_cards = reg.load_model("poisson_total_cards_base", league=league)
        
        # Probs Re-calc
        mu_h = float(m_h_corn.predict(pd.DataFrame([pd.to_numeric(row[m_h_corn.meta['features']])]))[0])
        mu_a = float(m_a_corn.predict(pd.DataFrame([pd.to_numeric(row[m_a_corn.meta['features']])]))[0])
        v_h = mu_h + (getattr(m_h_corn, 'alpha_', 0) * mu_h**2) if hasattr(m_h_corn, 'alpha_') else mu_h * 1.3
        v_a = mu_a + (getattr(m_a_corn, 'alpha_', 0) * mu_a**2) if hasattr(m_a_corn, 'alpha_') else mu_a * 1.3
        corn_res = NegativeBinomialEngine().calculate_probabilities(mu_h, v_h, mu_a, v_a)
        
        mu_c = float(m_cards.predict(pd.DataFrame([pd.to_numeric(row[m_cards.meta['features']])]))[0])
        card_res = ZeroInflatedEngine().calculate_probabilities(mu_c, pi_zero=0.1)
        
        preds = {
            'corners_home_win': corn_res['corners_home_win'],
            'corners_away_win': corn_res['corners_away_win'],
            'corn_u11': corn_res['corners_under_11_5'],
            'card_u55': card_res['cards_under_5_5'],
            'over_1_5': 0.8, # Placeholder to avoid tempo suppression
            'corners_over_7_5': 0.6,
            'home_win': 0.4, 'away_win': 0.3 # Placeholder
        }
        
        print(f"\nAUDIT: {m_info['home']} vs {m_info['away']}")
        print(f"Features (Home Corners Rolling 5): {row.get('home_rolling_corners_scored_5')}")
        print(f"Features (Away Corners Rolling 5): {row.get('away_rolling_corners_scored_5')}")
        print(f"Features (Home Cards Rolling 5): {row.get('home_rolling_cards_scored_5')}")
        print(f"Features (Away Cards Rolling 5): {row.get('away_rolling_cards_scored_5')}")
        
        # Analyze Decision
        candidates = engine.analyze_match(match_dict, preds)
        print("Qualified Candidates in Slip Pool:")
        for c in candidates:
            print(f"  - {c['market']} ({c['confidence']:.1%}) Tier {c['tier']}")
            
        # Check Gate specifically for Corner 1X2
        gate = evaluator._get_adjusted_gate({'type': '1X2_CORNERS'}, engine._fetch_standings_context(match_dict, league))
        print(f"Adjusted Gate for 1X2_CORNERS: {gate:.2f}")

if __name__ == "__main__":
    audit()
