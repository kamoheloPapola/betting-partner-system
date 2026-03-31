"""
Model Configuration Registry.

Defines the configuration for all ML models in the prediction system,
including model type, target variable, hyperparameters, and feature requirements.

Usage:
    from src.config.models import MODEL_CONFIGS, MODEL_CONFIGS_BY_NAME, ModelType
    
    # Get all configs
    for cfg in MODEL_CONFIGS:
        print(cfg['name'])
    
    # Lookup by name
    home_model = MODEL_CONFIGS_BY_NAME.get('poisson_home_base')
"""
from enum import Enum
from typing import Any, Dict, List, TypedDict

# Define public API
__all__ = [
    "MODEL_CONFIGS",
    "MODEL_CONFIGS_BY_NAME",
    "ModelConfig",
    "ModelType",
    "FeatureSet",
]


class ModelType(str, Enum):
    """Supported model architectures."""
    POISSON = "poisson"
    NEGATIVE_BINOMIAL = "nb"


class FeatureSet(str, Enum):
    """Available feature sets for model training."""
    BASE = "base"


class ModelConfig(TypedDict):
    """Type definition for model configuration entries."""
    name: str
    type: str  # ModelType value
    target: str
    params: Dict[str, Any]
    feature_set: str  # FeatureSet value
    required_columns: List[str]


MODEL_CONFIGS: List[ModelConfig] = [
    # =========================================================================
    # BASE MODELS - Core prediction models using standard features
    # =========================================================================
    {
        'name': 'poisson_home_base',
        'type': ModelType.POISSON.value,
        'target': 'home_score',
        'params': {'alpha': 0.01},
        'feature_set': FeatureSet.BASE.value,
        'required_columns': ['home_score']
    },
    {
        'name': 'poisson_away_base',
        'type': ModelType.POISSON.value,
        'target': 'away_score',
        'params': {'alpha': 0.01},
        'feature_set': FeatureSet.BASE.value,
        'required_columns': ['away_score']
    },
    {
        'name': 'nb_home_corners_base',
        'type': ModelType.NEGATIVE_BINOMIAL.value,
        'target': 'home_corners',
        'params': {},
        'feature_set': FeatureSet.BASE.value,
        'required_columns': ['home_corners']
    },
    {
        'name': 'nb_away_corners_base',
        'type': ModelType.NEGATIVE_BINOMIAL.value,
        'target': 'away_corners',
        'params': {},
        'feature_set': FeatureSet.BASE.value,
        'required_columns': ['away_corners']
    },
    {
        'name': 'poisson_total_cards_base',
        'type': ModelType.POISSON.value,
        'target': 'match_total_cards',
        'params': {'alpha': 0.05},
        'feature_set': FeatureSet.BASE.value,
        'required_columns': ['match_total_cards']
    }
]

# O(1) lookup by model name
MODEL_CONFIGS_BY_NAME: Dict[str, ModelConfig] = {
    cfg['name']: cfg for cfg in MODEL_CONFIGS
}


def validate_model_configs() -> None:
    """
    Validate model configuration integrity.
    
    Raises:
        ValueError: If any configuration is invalid.
    """
    names_seen = set()
    valid_types = {t.value for t in ModelType}
    valid_feature_sets = {f.value for f in FeatureSet}
    
    for cfg in MODEL_CONFIGS:
        name = cfg['name']
        
        # Check unique names
        if name in names_seen:
            raise ValueError(f"Duplicate model name: {name}")
        names_seen.add(name)
        
        # Check valid type
        if cfg['type'] not in valid_types:
            raise ValueError(f"Invalid model type '{cfg['type']}' for {name}")
        
        # Check valid feature set
        if cfg['feature_set'] not in valid_feature_sets:
            raise ValueError(f"Invalid feature set '{cfg['feature_set']}' for {name}")
        
        # Check required_columns is not empty
        if not cfg['required_columns']:
            raise ValueError(f"Empty required_columns for {name}")


# Run validation on module load
validate_model_configs()

