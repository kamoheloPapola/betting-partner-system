"""
Betting Market Definitions.

Defines predicate functions for determining betting outcomes based on
match statistics (goals, corners, cards). Each market category contains:
- Predicate lambdas that return boolean Series from a DataFrame
- A 'dependencies' list specifying required columns

Usage:
    from src.config.markets import MARKET_DEFINITIONS
    predicates = MARKET_DEFINITIONS['goals']
    over_25 = predicates['goals_over_2_5'](df)
"""
from typing import Any, Callable, Dict, List

import pandas as pd

from src.config.thresholds import Thresholds

# Define public API
__all__ = ["MARKET_DEFINITIONS", "MarketPredicate"]

# Type alias for market predicate functions
MarketPredicate = Callable[[pd.DataFrame], pd.Series]


def _generate_team_corner_markets() -> Dict[str, Any]:
    """
    Generate over/under predicates for team-specific corner lines.
    
    Creates predicates for home/away corners at 2.5, 3.5, 4.5, 5.5 lines.
    Uses default argument binding to capture loop variables correctly.
    """
    markets: Dict[str, Any] = {'dependencies': ['home_corners', 'away_corners']}
    
    for team in ['home', 'away']:
        col = f'{team}_corners'
        for line in [2.5, 3.5, 4.5, 5.5, 6.5]:
            line_str = str(line).replace('.', '_')
            # Default argument binding prevents late binding issues
            markets[f'{team}_corners_under_{line_str}'] = (
                lambda df, c=col, l=line: df[c] < l
            )
            markets[f'{team}_corners_over_{line_str}'] = (
                lambda df, c=col, l=line: df[c] > l
            )
    
    return markets


MARKET_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    # --- Match Result (1X2) ---
    '1x2': {
        'home_win': lambda df: df['home_goals'] > df['away_goals'],
        'draw': lambda df: df['home_goals'] == df['away_goals'],
        'away_win': lambda df: df['away_goals'] > df['home_goals'],
        'dependencies': ['home_goals', 'away_goals']
    },
    
    # --- Goals Markets ---
    'goals': {
        'goals_over_2_5': lambda df: (df['home_goals'] + df['away_goals']) > Thresholds.GOALS_O2_LINE,
        'goals_under_2_5': lambda df: (df['home_goals'] + df['away_goals']) <= Thresholds.GOALS_O2_LINE,
        'home_goals_under_1_5': lambda df: df['home_goals'] < 1.5,
        'away_goals_under_1_5': lambda df: df['away_goals'] < 1.5,
        'dependencies': ['home_goals', 'away_goals']
    },
    
    # --- Both Teams to Score ---
    'btts': {
        'btts_yes': lambda df: (df['home_goals'] > 0) & (df['away_goals'] > 0),
        'btts_no': lambda df: ~((df['home_goals'] > 0) & (df['away_goals'] > 0)),
        'dependencies': ['home_goals', 'away_goals']
    },
    
    # --- Total Corners ---
    'corners': {
        'corners_under_11_5': lambda df: (df['home_corners'] + df['away_corners']) <= Thresholds.CORNERS_U11_LINE,
        'corners_over_9_5': lambda df: (df['home_corners'] + df['away_corners']) > Thresholds.CORNERS_O9_LINE,
        'corners_over_7_5': lambda df: (df['home_corners'] + df['away_corners']) > Thresholds.CORNERS_O75_LINE,
        'corners_home_win': lambda df: df['home_corners'] > df['away_corners'],
        'corners_draw': lambda df: df['home_corners'] == df['away_corners'],
        'corners_away_win': lambda df: df['away_corners'] > df['home_corners'],
        'dependencies': ['home_corners', 'away_corners']
    },
    
    # --- Team-Specific Corners (generated) ---
    'team_corners': _generate_team_corner_markets(),
    
    # --- Cards Markets ---
    'cards': {
        'home_cards_over_1_5': lambda df: df['home_cards'] > 1.5,
        'away_cards_over_1_5': lambda df: df['away_cards'] > 1.5,
        'cards_under_4_5': lambda df: (df['home_cards'] + df['away_cards']) < 4.5,
        'cards_under_5_5': lambda df: (df['home_cards'] + df['away_cards']) <= Thresholds.CARDS_U55_LINE,
        'cards_over_2_5': lambda df: (df['home_cards'] + df['away_cards']) > Thresholds.CARDS_O25_LINE,
        'cards_over_3_5': lambda df: (df['home_cards'] + df['away_cards']) > 3.5,
        'dependencies': ['home_cards', 'away_cards']
    },

    # --- Double Chance Markets ---
    'double_chance': {
        'home_dc': lambda df: (df['home_goals'] > df['away_goals']) | (df['home_goals'] == df['away_goals']),
        'away_dc': lambda df: (df['away_goals'] > df['home_goals']) | (df['home_goals'] == df['away_goals']),
        'dependencies': ['home_goals', 'away_goals']
    },
}

