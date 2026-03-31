"""
Strategy Configuration.

Defines model requirements for each betting strategy. Used by the
Model Pre-flight Integrity Guard (MPIG) to validate that all necessary
models are trained and production-ready before strategy execution.
"""
from enum import Enum
from typing import Dict, FrozenSet, List

# Define public API
__all__ = [
    "Strategy",
    "STRATEGY_REQUIREMENTS",
    "get_required_models",
    "get_strategies_for_model",
]


class Strategy(str, Enum):
    """Supported betting strategies."""
    FORBIDDEN_FRUIT = "forbidden-fruit"
    ACCUMULATOR = "accumulator"


# Strategy -> Required model names (must exist in MODEL_CONFIGS)
STRATEGY_REQUIREMENTS: Dict[Strategy, List[str]] = {
    Strategy.FORBIDDEN_FRUIT: [
        "poisson_home_base", 
        "poisson_away_base", 
        "nb_home_corners_base", 
        "nb_away_corners_base", 
        "poisson_total_cards_base"
    ],
    Strategy.ACCUMULATOR: [
        "poisson_home_base", 
        "poisson_away_base",
        "nb_home_corners_base", 
        "nb_away_corners_base"
    ],
}


def get_required_models(strategy: Strategy) -> List[str]:
    """
    Get the list of required model names for a strategy.
    
    Args:
        strategy: The strategy to get requirements for.
        
    Returns:
        List of model names required by the strategy.
        
    Raises:
        KeyError: If strategy is not configured.
    """
    return STRATEGY_REQUIREMENTS[strategy]


def get_strategies_for_model(model_name: str) -> List[Strategy]:
    """
    Find all strategies that depend on a given model.
    
    Args:
        model_name: The model name to search for.
        
    Returns:
        List of strategies that require this model.
    """
    return [
        strategy 
        for strategy, models in STRATEGY_REQUIREMENTS.items() 
        if model_name in models
    ]


def _validate_strategy_requirements() -> None:
    """
    Validate that all model names in requirements exist in MODEL_CONFIGS.
    
    Raises:
        ValueError: If any referenced model doesn't exist.
    """
    # Lazy import to avoid circular dependency
    from src.config.models import MODEL_CONFIGS_BY_NAME
    
    for strategy, models in STRATEGY_REQUIREMENTS.items():
        for model in models:
            if model not in MODEL_CONFIGS_BY_NAME:
                raise ValueError(
                    f"Strategy '{strategy.value}' requires unknown model '{model}'. "
                    f"Please add it to MODEL_CONFIGS in src/config/models.py"
                )


# Run validation on module load
_validate_strategy_requirements()

