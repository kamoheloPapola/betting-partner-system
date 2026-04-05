import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import DATA_DIR, MODELS_DIR
from src.models.train_probability_models import ProbabilityModelTrainer, TRACKED_LEAGUES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train specialist models for selected leagues.")
    parser.add_argument(
        "--leagues",
        nargs="+",
        default=list(TRACKED_LEAGUES),
        help="League codes to train (default: PL BL1 FL1 SA PD)",
    )
    parser.add_argument(
        "--full-retrain",
        action="store_true",
        help="Force training from scratch even if prior checkpoints exist.",
    )
    return parser.parse_args()


def normalize_leagues(raw_leagues: List[str]) -> List[str]:
    allowed = set(TRACKED_LEAGUES)
    leagues = [str(league).upper() for league in raw_leagues]
    invalid = [league for league in leagues if league not in allowed]
    if invalid:
        raise ValueError(
            f"Unsupported league codes: {', '.join(invalid)}. "
            f"Allowed: {', '.join(sorted(allowed))}"
        )
    return leagues


def run_training(*, leagues: List[str], full_retrain: bool) -> None:
    print("=" * 60)
    print("STARTING FULL BATCH TRAINING")
    print("=" * 60)
    print(f"Leagues: {' '.join(leagues)}")
    print(f"Mode: {'FULL RETRAIN' if full_retrain else 'WARM START'}")

    start_global = time.time()
    features_csv = DATA_DIR / "features" / "feature_matrix.csv"
    trainer = ProbabilityModelTrainer(features_path=features_csv, models_dir=MODELS_DIR)
    trainer.run(tracked_leagues=leagues, full_retrain=full_retrain)
    print("Updating RL bandit state...")
    try:
        subprocess.run(["python", "scripts/update_bandit.py"], check=True)
        print("Bandit update complete.")
    except Exception as exc:
        print(f"BANDIT UPDATE FAILED (non-fatal): {exc}")

    duration = time.time() - start_global
    print("\n" + "=" * 60)
    print(f"BATCH COMPLETE in {duration:.1f}s")
    print("Status: SUCCESS")
    print("=" * 60)


if __name__ == "__main__":
    try:
        args = parse_args()
        selected_leagues = normalize_leagues(args.leagues)
        run_training(leagues=selected_leagues, full_retrain=bool(args.full_retrain))
    except Exception as exc:
        print(f"BATCH FAILED: {exc}")
        sys.exit(1)
