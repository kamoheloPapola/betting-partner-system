"""
ML Model Configurations.

Centralized configuration for all trainable models:
- Poisson regression for goals
- Negative Binomial for corners

Defines model types, feature sets, and training parameters.
Alpha values are now dynamically loaded from alpha_config.py.
"""
from typing import List, Dict, Any
from enum import Enum

class ModelType(str, Enum):
    POISSON = "poisson"
    NB = "nb"

class FeatureSet(str, Enum):
    BASE = "base"
    ENHANCED_XG = "enhanced_xg"  # Phase-5 LightGBM features — loads from models/feature_columns.json


class TrainingMode(str, Enum):
    DEBUG = "debug"
    PRODUCTION = "production"

# Note: Alpha is now resolved dynamically at training time via alpha_config.get_alpha()
# The 'params' dict here serves as fallback/default structure.
MODEL_CONFIGS: List[Dict[str, Any]] = [
    # --- BASE MODELS ---
    {
        'name': 'poisson_home_base',
        'type': ModelType.POISSON,
        'target': 'home_score',
        'params': {},  # Alpha set dynamically
        'feature_set': FeatureSet.BASE,
        'required_columns': ['home_score']
    },
    {
        'name': 'poisson_away_base',
        'type': ModelType.POISSON,
        'target': 'away_score',
        'params': {},  # Alpha set dynamically
        'feature_set': FeatureSet.BASE,
        'required_columns': ['away_score']
    },
    {
        'name': 'nb_home_corners_base',
        'type': ModelType.NB,
        'target': 'home_corners',
        'params': {},
        'feature_set': FeatureSet.BASE,
        'required_columns': ['home_corners']
    },
    {
        'name': 'nb_away_corners_base',
        'type': ModelType.NB,
        'target': 'away_corners',
        'params': {},
        'feature_set': FeatureSet.BASE,
        'required_columns': ['away_corners']
    },
    {
        'name': 'poisson_total_cards_base',
        'type': ModelType.POISSON,
        'target': 'match_total_cards',
        'params': {},  # Alpha set dynamically
        'feature_set': FeatureSet.BASE,
        'required_columns': ['match_total_cards']
    },
    

]

