"""
Evaluation Configuration.

Defines calibration parameters and thresholds for model
quality assessment (Expected Calibration Error, sample sizing, etc.).
"""
from dataclasses import dataclass

# Define public API
__all__ = ["CalibrationConfig", "DEFAULT_CALIBRATION_CONFIG"]


@dataclass(frozen=True)
class CalibrationConfig:
    """
    Configuration for calibration evaluation.
    
    Attributes:
        ECE_NUM_BINS: Number of bins for ECE calculation (default: 10, standard in literature).
        MIN_BIN_SIZE: Minimum samples per bin for valid ECE.
        MIN_SAMPLE_SIZE: Minimum matches for league evaluation.
        ECE_BOOST_THRESHOLD: ECE below this value indicates model underconfidence/high quality.
        ECE_ELIGIBLE_THRESHOLD: ECE below this value is acceptable for betting consideration.
        ECE_EXCLUDE_THRESHOLD: ECE above this value indicates poor calibration.
    """
    # Core Parameters
    ECE_NUM_BINS: int = 10
    MIN_BIN_SIZE: int = 5
    MIN_SAMPLE_SIZE: int = 30
    
    # Thresholds (ordered: BOOST < ELIGIBLE < EXCLUDE)
    ECE_BOOST_THRESHOLD: float = 0.02
    ECE_ELIGIBLE_THRESHOLD: float = 0.05
    ECE_EXCLUDE_THRESHOLD: float = 0.10

    def __post_init__(self) -> None:
        """Validate that thresholds are logically ordered."""
        # Use object.__setattr__ since dataclass is frozen
        if not (self.ECE_BOOST_THRESHOLD < self.ECE_ELIGIBLE_THRESHOLD < self.ECE_EXCLUDE_THRESHOLD):
            raise ValueError(
                f"Thresholds must be ordered: BOOST ({self.ECE_BOOST_THRESHOLD}) < "
                f"ELIGIBLE ({self.ECE_ELIGIBLE_THRESHOLD}) < EXCLUDE ({self.ECE_EXCLUDE_THRESHOLD})"
            )
        if self.ECE_NUM_BINS < 2:
            raise ValueError(f"ECE_NUM_BINS must be >= 2, got {self.ECE_NUM_BINS}")
        if self.MIN_BIN_SIZE < 1:
            raise ValueError(f"MIN_BIN_SIZE must be >= 1, got {self.MIN_BIN_SIZE}")
        if self.MIN_SAMPLE_SIZE < 1:
            raise ValueError(f"MIN_SAMPLE_SIZE must be >= 1, got {self.MIN_SAMPLE_SIZE}")


# Pre-instantiated default configuration
DEFAULT_CALIBRATION_CONFIG = CalibrationConfig()

