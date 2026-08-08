import os
from .thresholds import Thresholds
from pathlib import Path
from dotenv import load_dotenv

# Load env vars
load_dotenv()

# Base Paths
# Assuming src/config/__init__.py, so .parent.parent.parent is root
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR = Path(os.getenv("MODELS_DIR", "/app/data/models"))

# App Config
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
ENV = os.getenv("ENV", "development")
DATA_FRESHNESS_DAYS = 7


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# Specialist-model ensemble configuration (LightGBM + XGBoost).
SPECIALIST_ENSEMBLE_LGBM_WEIGHT = _env_float("SPECIALIST_ENSEMBLE_LGBM_WEIGHT", 0.6)
SPECIALIST_ENSEMBLE_XGB_WEIGHT = _env_float("SPECIALIST_ENSEMBLE_XGB_WEIGHT", 0.4)
SPECIALIST_ENSEMBLE_DIVERGENCE_THRESHOLD = _env_float("SPECIALIST_ENSEMBLE_DIVERGENCE_THRESHOLD", 0.20)

# Backward-compatible aliases used by existing prediction code/tests.
GOALS_ENSEMBLE_LGBM_WEIGHT = SPECIALIST_ENSEMBLE_LGBM_WEIGHT
GOALS_ENSEMBLE_XGB_WEIGHT = SPECIALIST_ENSEMBLE_XGB_WEIGHT
GOALS_ENSEMBLE_POISSON_WEIGHT = SPECIALIST_ENSEMBLE_XGB_WEIGHT
GOALS_ENSEMBLE_DIVERGENCE_THRESHOLD = SPECIALIST_ENSEMBLE_DIVERGENCE_THRESHOLD

def get_required_env(name: str) -> str:
    """Read a required environment variable or fail loudly."""
    value = os.getenv(name)
    if value:
        return value
    raise RuntimeError(f"{name} not set - check your environment or .env file")

# API Keys
FOOTBALL_DATA_API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY")
ODDS_API_KEY = os.getenv("ODDS_API_KEY")

# Training & Leagues
DEFAULT_TRAINING_LEAGUES = ['PL', 'PD', 'SA', 'BL1', 'FL1']
SUPPORTED_LEAGUES = ['PL', 'PD', 'SA', 'BL1', 'FL1', 'DED', 'PPL', 'T1', 'E0', 'SC0'] # More leagues stone
