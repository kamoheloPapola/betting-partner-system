import json
import logging
import sklearn
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
)

from src.config import DATA_DIR
from src.ml.registry import ModelRegistry

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

EXCLUDE_COLS = [
    "match_hash", "date", "match_date", "league", "home_team", "away_team", 
    "home_score", "away_score", "home_goals_ht", "away_goals_ht",
    "home_corners", "away_corners", "home_cards", "away_cards",
    "home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target",
    "home_yellow_cards", "away_yellow_cards", "home_red_cards", "away_red_cards",
    "home_fouls", "away_fouls", "home_total_shots", "away_total_shots", "referee",
    "status", "result"
]

# Model name → (filename, type, target) metadata for registration
_PHASE5_MODEL_MAP = [
    ("match_outcome", "match_outcome_model.pkl",  "lgbm_classifier",        "outcome"),
    ("home_goals",    "home_goals_model.pkl",     "lgbm_regressor_poisson", "home_goals"),
    ("away_goals",    "away_goals_model.pkl",     "lgbm_regressor_poisson", "away_goals"),
    ("home_goals_xgb", "home_goals_xgb_v1.joblib", "xgb_regressor_count_poisson", "home_goals"),
    ("away_goals_xgb", "away_goals_xgb_v1.joblib", "xgb_regressor_count_poisson", "away_goals"),
    ("corners",       "corners_model.pkl",        "lgbm_regressor_poisson", "total_corners"),
    ("corners_xgb",   "corners_xgb_v1.joblib",    "xgb_regressor_tweedie",  "total_corners"),
    ("cards",         "cards_model.pkl",          "lgbm_regressor_poisson", "total_cards"),
    ("cards_xgb",     "cards_xgb_v1.joblib",      "xgb_regressor_tweedie",  "total_cards"),
]


class ProbabilityModelTrainer:
    """
    Trains probability models for match outcomes, goals, corners, and cards.
    Separates models to capture different statistical processes in football interactions.
    """
    def __init__(self, features_path: Path, models_dir: Path):
        self.features_path = features_path
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.registry = ModelRegistry()
        
    # Poisson target cap: extreme scorelines (e.g. 8-0) destabilize log-link learning
    GOAL_TARGET_CAP = 6

    def load_and_engineer_data(self) -> pd.DataFrame:
        logger.info(f"Loading features from {self.features_path}")
        df = pd.read_csv(self.features_path)
        
        # 1. Target Engineering
        # Cap individual goal targets to stabilize Poisson learning
        df["home_goals"] = df["home_score"].clip(upper=self.GOAL_TARGET_CAP)
        df["away_goals"] = df["away_score"].clip(upper=self.GOAL_TARGET_CAP)
        
        # Track capped rate — warn if cap is too aggressive (>3%)
        n_total = len(df)
        capped_home = (df["home_score"] > self.GOAL_TARGET_CAP).sum()
        capped_away = (df["away_score"] > self.GOAL_TARGET_CAP).sum()
        capped_total = capped_home + capped_away
        capped_rate = capped_total / (2 * n_total) if n_total > 0 else 0
        
        if capped_total > 0:
            level = logger.warning if capped_rate > 0.03 else logger.info
            level(
                f"Poisson target capping: {capped_home} home, {capped_away} away "
                f"scores clipped to {self.GOAL_TARGET_CAP} "
                f"(capped_rate={capped_rate:.2%})"
            )
            if capped_rate > 0.03:
                logger.warning(
                    f"Capped rate {capped_rate:.2%} exceeds 3% threshold — "
                    f"consider raising GOAL_TARGET_CAP from {self.GOAL_TARGET_CAP}"
                )
        
        if "home_corners" in df.columns and "away_corners" in df.columns:
            df["total_corners"] = df["home_corners"] + df["away_corners"]
        else:
            df["total_corners"] = np.nan
            
        if "home_cards" in df.columns and "away_cards" in df.columns:
            df["total_cards"] = df["home_cards"] + df["away_cards"]
        else:
            df["total_cards"] = np.nan
            
        # Outcome: 0 = away win, 1 = draw, 2 = home win
        conditions = [
            df["home_score"] < df["away_score"],
            df["home_score"] == df["away_score"],
            df["home_score"] > df["away_score"]
        ]
        df["outcome"] = np.select(conditions, [0, 1, 2], default=np.nan)
        
        # 2. Extract starting year from season string (e.g., "2023-24" -> 2023)
        df["season_year"] = df["season"].astype(str).str[:4].astype(int)
        
        # 3. Cast objects to category for LightGBM
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype("category")
                
        return df

    def get_time_splits(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train = df[df["season_year"] <= 2021].copy()
        val = df[df["season_year"] == 2022].copy()
        test = df[df["season_year"] >= 2023].copy()
        logger.info(f"Temporal Split - Train (<=2021): {len(train)}, Val (2022): {len(val)}, Test (>=2023): {len(test)}")
        return train, val, test

    def select_features(self, df: pd.DataFrame) -> List[str]:
        # Drop identifiers, raw leakages, and our newly engineered targets
        features = [
            c for c in df.columns 
            if c not in EXCLUDE_COLS 
            and not c.startswith("total_") 
            and c not in ("outcome", "season_year", "season", "home_goals", "away_goals")
        ]
        logger.info(f"Selected {len(features)} predictive features.")
        return features

    def train_outcome_model(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[Any, Dict, np.ndarray]:
        logger.info("Training Match Outcome Model (LGBMClassifier + Isotonic Calibration)...")
        base_model = LGBMClassifier(
            n_estimators=600,
            learning_rate=0.02,
            max_depth=6,
            subsample=0.8,
            random_state=42,
            verbose=-1
        )
        
        # Professional probability calibration
        calibrated_model = CalibratedClassifierCV(estimator=base_model, method="isotonic", cv=3)
        calibrated_model.fit(X_train, y_train)
        
        probs = calibrated_model.predict_proba(X_test)
        preds = calibrated_model.predict(X_test)
        
        acc = accuracy_score(y_test, preds)
        ll = log_loss(y_test, probs)
        
        # Multi-class Brier score
        y_test_one_hot = pd.get_dummies(y_test).reindex(columns=[0, 1, 2], fill_value=0).values
        brier = np.mean(np.sum((probs - y_test_one_hot)**2, axis=1))
        
        metrics = {
            "accuracy": round(acc, 4),
            "log_loss": round(ll, 4),
            "brier_score": round(brier, 4)
        }
        logger.info(f"Outcome Metrics on Test: {metrics}")
        
        # Extract mean feature importance from CV estimators
        importances = np.mean([est.estimator.feature_importances_ for est in calibrated_model.calibrated_classifiers_], axis=0)
        return calibrated_model, metrics, importances

    def train_count_model(self, name: str, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[Any, Dict]:
        logger.info(f"Training {name} Model (LGBMRegressor Poisson)...")
        model = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=5,
            objective="poisson",
            subsample=0.8,
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, y_train)
        
        # LightGBM Poisson can output slightly negative — enforce lambda floor
        preds = np.maximum(0.05, model.predict(X_test))
        mae = mean_absolute_error(y_test, preds)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        
        metrics = {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4)
        }
        logger.info(f"{name} Metrics on Test: {metrics}")
        
        return model, metrics

    def train_count_model_xgb(
        self,
        name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        *,
        objective: str,
    ) -> Tuple[Any, Dict]:
        logger.info("Training %s Model (XGBRegressor %s)...", name, objective)
        params: Dict[str, Any] = {
            "n_estimators": 600,
            "learning_rate": 0.03,
            "max_depth": 5,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "objective": objective,
            "random_state": 42,
            "n_jobs": 4,
            "tree_method": "hist",
            "enable_categorical": True,
            "verbosity": 0,
        }
        if objective == "count:poisson":
            params["eval_metric"] = "poisson-nloglik"
        if objective == "reg:tweedie":
            params["tweedie_variance_power"] = 1.3
            params["eval_metric"] = "rmse"

        model = XGBRegressor(**params)
        model.fit(X_train, y_train)

        preds = np.maximum(0.05, model.predict(X_test))
        mae = mean_absolute_error(y_test, preds)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        metrics = {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
        }
        logger.info("%s XGB Metrics on Test: %s", name, metrics)
        return model, metrics

    def _register_model(
        self,
        name: str,
        filename: str,
        model_type: str,
        target: str,
        metrics: Dict[str, Any],
        features: List[str],
    ) -> None:
        """Register a trained Phase-5 model in the manifest."""
        # Always use a fixed '1.0.0' for Phase-5 batch; bump manually on retrain.
        version = "1.0.0"
        metadata: Dict[str, Any] = {
            "filename": filename,
            "status": "productive",
            "type": model_type,
            "target": target,
            "features": features,
            "metrics": metrics,
            "training_date": datetime.now().isoformat(),
            "league": None,  # global model
            "temporal_validated": True,
            # sklearn_version auto-filled by register_model() via setdefault()
        }
        self.registry.register_model(name, version, metadata)

    def run(self):
        df = self.load_and_engineer_data()
        
        df_valid = df.dropna(subset=["outcome"]).copy()
        if len(df_valid) == 0:
            logger.error("No valid outcomes to train on. Aborting.")
            return
            
        train, val, test = self.get_time_splits(df_valid)
        features = self.select_features(train)
        
        # We will use Train + Val for fitting, and evaluate on Test
        X_train_full = pd.concat([train[features], val[features]])
        y_train_out = pd.concat([train["outcome"], val["outcome"]])
        
        X_test = test[features]
        y_test_out = test["outcome"]
        
        report = {}
        
        # 1. Match Outcome
        outcome_model, outcome_metrics, outcome_imp = self.train_outcome_model(X_train_full, y_train_out, X_test, y_test_out)
        joblib.dump(outcome_model, self.models_dir / "match_outcome_model.pkl")
        report["match_outcome"] = outcome_metrics
        self._register_model(
            name="match_outcome",
            filename="match_outcome_model.pkl",
            model_type="lgbm_classifier",
            target="outcome",
            metrics=outcome_metrics,
            features=features,
        )
        
        # Export Feature Importance
        imp_df = pd.DataFrame({"feature": features, "importance": outcome_imp})
        imp_df = imp_df.sort_values("importance", ascending=False)
        imp_df.to_csv(self.models_dir / "feature_importance.csv", index=False)
        
        # 2. Home Goals (Separate Poisson model — captures home advantage)
        if "home_goals" in df.columns and not df["home_goals"].isna().all():
            mask_train = X_train_full.index.intersection(df_valid.dropna(subset=["home_goals"]).index)
            mask_test = X_test.index.intersection(df_valid.dropna(subset=["home_goals"]).index)
            
            home_goals_model, home_goals_metrics = self.train_count_model(
                "Home Goals", 
                X_train_full.loc[mask_train], df_valid.loc[mask_train, "home_goals"], 
                X_test.loc[mask_test], df_valid.loc[mask_test, "home_goals"]
            )
            joblib.dump(home_goals_model, self.models_dir / "home_goals_model.pkl")
            report["home_goals"] = home_goals_metrics
            self._register_model(
                name="home_goals",
                filename="home_goals_model.pkl",
                model_type="lgbm_regressor_poisson",
                target="home_goals",
                metrics=home_goals_metrics,
                features=features,
            )

            home_goals_xgb_model, home_goals_xgb_metrics = self.train_count_model_xgb(
                "Home Goals XGB",
                X_train_full.loc[mask_train],
                df_valid.loc[mask_train, "home_goals"],
                X_test.loc[mask_test],
                df_valid.loc[mask_test, "home_goals"],
                objective="count:poisson",
            )
            joblib.dump(home_goals_xgb_model, self.models_dir / "home_goals_xgb_v1.joblib")
            report["home_goals_xgb"] = home_goals_xgb_metrics
            self._register_model(
                name="home_goals_xgb",
                filename="home_goals_xgb_v1.joblib",
                model_type="xgb_regressor_count_poisson",
                target="home_goals",
                metrics=home_goals_xgb_metrics,
                features=features,
            )

        # 3. Away Goals (Separate Poisson model — captures away scoring patterns)
        if "away_goals" in df.columns and not df["away_goals"].isna().all():
            mask_train = X_train_full.index.intersection(df_valid.dropna(subset=["away_goals"]).index)
            mask_test = X_test.index.intersection(df_valid.dropna(subset=["away_goals"]).index)
            
            away_goals_model, away_goals_metrics = self.train_count_model(
                "Away Goals", 
                X_train_full.loc[mask_train], df_valid.loc[mask_train, "away_goals"], 
                X_test.loc[mask_test], df_valid.loc[mask_test, "away_goals"]
            )
            joblib.dump(away_goals_model, self.models_dir / "away_goals_model.pkl")
            report["away_goals"] = away_goals_metrics
            self._register_model(
                name="away_goals",
                filename="away_goals_model.pkl",
                model_type="lgbm_regressor_poisson",
                target="away_goals",
                metrics=away_goals_metrics,
                features=features,
            )

            away_goals_xgb_model, away_goals_xgb_metrics = self.train_count_model_xgb(
                "Away Goals XGB",
                X_train_full.loc[mask_train],
                df_valid.loc[mask_train, "away_goals"],
                X_test.loc[mask_test],
                df_valid.loc[mask_test, "away_goals"],
                objective="count:poisson",
            )
            joblib.dump(away_goals_xgb_model, self.models_dir / "away_goals_xgb_v1.joblib")
            report["away_goals_xgb"] = away_goals_xgb_metrics
            self._register_model(
                name="away_goals_xgb",
                filename="away_goals_xgb_v1.joblib",
                model_type="xgb_regressor_count_poisson",
                target="away_goals",
                metrics=away_goals_xgb_metrics,
                features=features,
            )

        # 4. Total Corners
        if "total_corners" in df.columns and not df["total_corners"].isna().all():
            mask_train = X_train_full.index.intersection(df_valid.dropna(subset=["total_corners"]).index)
            mask_test = X_test.index.intersection(df_valid.dropna(subset=["total_corners"]).index)
            
            corners_model, corners_metrics = self.train_count_model(
                "Corners", 
                X_train_full.loc[mask_train], df_valid.loc[mask_train, "total_corners"], 
                X_test.loc[mask_test], df_valid.loc[mask_test, "total_corners"]
            )
            joblib.dump(corners_model, self.models_dir / "corners_model.pkl")
            report["corners"] = corners_metrics
            self._register_model(
                name="corners",
                filename="corners_model.pkl",
                model_type="lgbm_regressor_poisson",
                target="total_corners",
                metrics=corners_metrics,
                features=features,
            )

            corners_xgb_model, corners_xgb_metrics = self.train_count_model_xgb(
                "Corners XGB",
                X_train_full.loc[mask_train],
                df_valid.loc[mask_train, "total_corners"],
                X_test.loc[mask_test],
                df_valid.loc[mask_test, "total_corners"],
                objective="reg:tweedie",
            )
            joblib.dump(corners_xgb_model, self.models_dir / "corners_xgb_v1.joblib")
            report["corners_xgb"] = corners_xgb_metrics
            self._register_model(
                name="corners_xgb",
                filename="corners_xgb_v1.joblib",
                model_type="xgb_regressor_tweedie",
                target="total_corners",
                metrics=corners_xgb_metrics,
                features=features,
            )

        # 5. Total Cards
        if "total_cards" in df.columns and not df["total_cards"].isna().all():
            mask_train = X_train_full.index.intersection(df_valid.dropna(subset=["total_cards"]).index)
            mask_test = X_test.index.intersection(df_valid.dropna(subset=["total_cards"]).index)
            
            cards_model, cards_metrics = self.train_count_model(
                "Cards", 
                X_train_full.loc[mask_train], df_valid.loc[mask_train, "total_cards"], 
                X_test.loc[mask_test], df_valid.loc[mask_test, "total_cards"]
            )
            joblib.dump(cards_model, self.models_dir / "cards_model.pkl")
            report["cards"] = cards_metrics
            self._register_model(
                name="cards",
                filename="cards_model.pkl",
                model_type="lgbm_regressor_poisson",
                target="total_cards",
                metrics=cards_metrics,
                features=features,
            )

            cards_xgb_model, cards_xgb_metrics = self.train_count_model_xgb(
                "Cards XGB",
                X_train_full.loc[mask_train],
                df_valid.loc[mask_train, "total_cards"],
                X_test.loc[mask_test],
                df_valid.loc[mask_test, "total_cards"],
                objective="reg:tweedie",
            )
            joblib.dump(cards_xgb_model, self.models_dir / "cards_xgb_v1.joblib")
            report["cards_xgb"] = cards_xgb_metrics
            self._register_model(
                name="cards_xgb",
                filename="cards_xgb_v1.joblib",
                model_type="xgb_regressor_tweedie",
                target="total_cards",
                metrics=cards_xgb_metrics,
                features=features,
            )
            
        # Output Metadata Artifacts
        metadata = {
            "training_seasons": {"train_and_val": "<=2022", "test": ">=2023"},
            "feature_count": len(features),
            "models_trained": list(report.keys()),
            "dataset_rows": len(df_valid)
        }
        
        # Extract Drift Baselines for live monitoring
        logger.info("Extracting feature distributions for drift baselines...")
        baselines = {}
        # compute stats for continuous features
        desc = X_train_full.describe(percentiles=[.25, .5, .75]).fillna(0).to_dict()
        for ft, stats in desc.items():
            baselines[ft] = {
                "mean": stats.get("mean", 0.0),
                "std": stats.get("std", 0.0),
                "25%": stats.get("25%", 0.0),
                "50%": stats.get("50%", 0.0),
                "75%": stats.get("75%", 0.0),
            }
            
        with open(self.models_dir / "feature_baselines.json", "w") as f:
            json.dump(baselines, f, indent=2)
            
        with open(self.models_dir / "feature_columns.json", "w") as f:
            json.dump(features, f, indent=2)
            
        with open(self.models_dir / "training_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        with open(self.models_dir / "training_report.json", "w") as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Artifacts successfully saved to {self.models_dir}")
        logger.info("═══ Probability Models Training — DONE ═══")

if __name__ == "__main__":
    features_csv = DATA_DIR / "features" / "feature_matrix.csv"
    
    # Store at root-level models directory as specified
    models_out = Path("models")
    
    trainer = ProbabilityModelTrainer(features_path=features_csv, models_dir=models_out)
    trainer.run()
