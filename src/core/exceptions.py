"""
System Exception Hierarchy.

Defines domain-specific exceptions with structured context for debugging
and error handling throughout the prediction system. All exceptions inherit
from PredictionSystemError and support a context dictionary for metadata.

Usage:
    from src.core.exceptions import DataValidationError
    
    raise DataValidationError(
        "Missing required column",
        context={"column": "home_score", "file": "matches.csv"}
    )
"""
from typing import Any, Dict, Optional

# Define public API
__all__ = [
    "PredictionSystemError",
    "ModelNotFoundError",
    "DataValidationError",
    "InsufficientDataError",
    "ConfigurationError",
    "APIError",
    "FeatureEngineeringError",
]


class PredictionSystemError(Exception):
    """
    Base exception with context for the Betting Partner System.
    
    Attributes:
        message: Human-readable error description.
        context: Optional dictionary with structured error metadata.
    """
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        self.message = message
        self.context = context or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        """Format error with context if available."""
        if self.context:
            ctx_str = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{self.message} [{ctx_str}]"
        return self.message
    
    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return f"{self.__class__.__name__}(message={self.message!r}, context={self.context})"


class ModelNotFoundError(PredictionSystemError):
    """Raised when a required prediction model cannot be found or loaded."""
    pass


class DataValidationError(PredictionSystemError):
    """Raised when input data (CSV, API response) fails schema or quality checks."""
    pass


class InsufficientDataError(PredictionSystemError):
    """Raised when a league/market has too few samples for valid training or prediction."""
    pass


class ConfigurationError(PredictionSystemError):
    """Raised when environment variables or config thresholds are invalid."""
    pass


class APIError(PredictionSystemError):
    """Raised when an external API call fails (network, auth, rate limit, etc.)."""
    pass


class FeatureEngineeringError(PredictionSystemError):
    """Raised when feature pipeline transformation or calculation fails."""
    pass

