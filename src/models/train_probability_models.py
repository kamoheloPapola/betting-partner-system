import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
)
from xgboost import XGBRegressor

from src.config import DATA_DIR, MODELS_DIR
from src.ml.registry import ModelRegistry

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TRACKED_LEAGUES: Tuple[str, ...] = ("PL", "BL1", "FL1", "SA", "PD")
MODEL_VERSION = "1.0.0"

EXCLUDE_COLS = [
    "match_hash",
    "date",
    "match_date",
    "league",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "home_goals_ht",
    "away_goals_ht",
    "home_corners",
    "away_corners",
    "home_cards",
    "away_cards",
    "home_shots",
    "away_shots",
    "home_shots_on_target",
    "away_shots_on_target",
    "home_yellow_cards",
    "away_yellow_cards",
    "home_red_cards",
    "away_red_cards",
    "home_fouls",
    "away_fouls",
    "home_total_shots",
    "away_total_shots",
    "referee",
    "status",
    "result",
]

# Features unavailable at serving time (ref assignments are often unknown pre-match).
SERVING_UNAVAILABLE_FEATURES = {
    "home_referee",
    "away_referee",
    "referee_card_rate_10",
    "referee_avg_yellows",
    "referee_avg_reds",
    "referee_avg_fouls",
    "card_pressure",
}


class ProbabilityModelTrainer:
    """
    Trains probability models for match outcomes, goals, corners, and cards.
    Trains separate specialist artifacts per league.
    """

    GOAL_TARGET_CAP = 6

    def __init__(self, features_path: Path, models_dir: Path):
        self.features_path = features_path
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.registry = ModelRegistry()
        self._serving_columns_cache: Dict[str, set[str]] = {}

    def load_and_engineer_data(self) -> pd.DataFrame:
        logger.info("Loading features from %s", self.features_path)
        df = pd.read_csv(self.features_path)

        # Cap extreme goals to stabilize count-model fitting.
        df["home_goals"] = df["home_score"].clip(upper=self.GOAL_TARGET_CAP)
        df["away_goals"] = df["away_score"].clip(upper=self.GOAL_TARGET_CAP)

        n_total = len(df)
        capped_home = (df["home_score"] > self.GOAL_TARGET_CAP).sum()
        capped_away = (df["away_score"] > self.GOAL_TARGET_CAP).sum()
        capped_total = capped_home + capped_away
        capped_rate = capped_total / (2 * n_total) if n_total > 0 else 0.0

        if capped_total > 0:
            level = logger.warning if capped_rate > 0.03 else logger.info
            level(
                "Poisson target capping: %s home, %s away scores clipped to %s (capped_rate=%.2f%%)",
                capped_home,
                capped_away,
                self.GOAL_TARGET_CAP,
                capped_rate * 100.0,
            )
            if capped_rate > 0.03:
                logger.warning(
                    "Capped rate %.2f%% exceeds 3%% threshold - consider raising GOAL_TARGET_CAP from %s",
                    capped_rate * 100.0,
                    self.GOAL_TARGET_CAP,
                )

        if "home_corners" in df.columns and "away_corners" in df.columns:
            df["total_corners"] = df["home_corners"] + df["away_corners"]
        else:
            df["total_corners"] = np.nan

        if "home_cards" in df.columns and "away_cards" in df.columns:
            df["total_cards"] = df["home_cards"] + df["away_cards"]
        else:
            df["total_cards"] = np.nan

        # Outcome: 0=away win, 1=draw, 2=home win.
        conditions = [
            df["home_score"] < df["away_score"],
            df["home_score"] == df["away_score"],
            df["home_score"] > df["away_score"],
        ]
        df["outcome"] = np.select(conditions, [0, 1, 2], default=np.nan)

        # "2023-24" -> 2023
        df["season_year"] = df["season"].astype(str).str[:4].astype(int)

        # Cast objects to category for tree models.
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype("category")

        return df

    @staticmethod
    def get_time_splits(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train = df[df["season_year"] <= 2021].copy()
        val = df[df["season_year"] == 2022].copy()
        test = df[df["season_year"] >= 2023].copy()
        logger.info(
            "Temporal Split - Train (<=2021): %s, Val (2022): %s, Test (>=2023): %s",
            len(train),
            len(val),
            len(test),
        )
        return train, val, test

    def _get_serving_columns(self, league_code: str) -> Optional[set[str]]:
        """
        Best-effort discovery of columns available at serving time for a league.
        """
        if league_code in self._serving_columns_cache:
            cached = self._serving_columns_cache[league_code]
            return cached if cached else None

        try:
            from src.core.container import ServiceContainer

            serving_df = ServiceContainer.get_instance().pipeline.run(league=league_code)
            if serving_df is None or serving_df.empty:
                logger.warning(
                    "Serving schema discovery returned no rows for %s; skipping serving-safe feature filter.",
                    league_code,
                )
                self._serving_columns_cache[league_code] = set()
                return None

            cols = {str(col) for col in serving_df.columns}
            self._serving_columns_cache[league_code] = cols
            logger.info(
                "Loaded serving schema for %s with %d columns for feature safety filtering.",
                league_code,
                len(cols),
            )
            return cols
        except Exception as exc:
            logger.warning(
                "Serving schema discovery failed for %s; continuing without serving-safe filter: %s",
                league_code,
                exc,
            )
            self._serving_columns_cache[league_code] = set()
            return None

    def select_features(self, df: pd.DataFrame, league_code: str) -> List[str]:
        features = [
            c
            for c in df.columns
            if c not in EXCLUDE_COLS
            and c not in SERVING_UNAVAILABLE_FEATURES
            and pd.api.types.is_numeric_dtype(df[c])
            and not c.startswith("total_")
            and c not in ("outcome", "season_year", "season", "home_goals", "away_goals")
        ]

        serving_cols = self._get_serving_columns(league_code)
        if serving_cols:
            before = len(features)
            features = [
                f
                for f in features
                if f in serving_cols
                or f.replace("_scored_", "_won_").replace("_conceded_", "_received_") in serving_cols
            ]
            logger.info(
                "Serving-safe feature filter retained %d/%d features for %s.",
                len(features),
                before,
                league_code,
            )

        logger.info("Selected %d predictive features for %s.", len(features), league_code)
        return features

    @staticmethod
    def _append_league_suffix(filename: str, league: str) -> str:
        path = Path(filename)
        return f"{path.stem}_{league}{path.suffix}"

    @staticmethod
    def _dump_json(path: Path, payload: Any) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def _target_masks(
        X_train_full: pd.DataFrame,
        X_test: pd.DataFrame,
        df_scope: pd.DataFrame,
        target: str,
    ) -> Tuple[pd.Index, pd.Index]:
        valid_idx = df_scope.dropna(subset=[target]).index
        mask_train = X_train_full.index.intersection(valid_idx)
        mask_test = X_test.index.intersection(valid_idx)
        return mask_train, mask_test

    @staticmethod
    def _require_non_empty_masks(mask_train: pd.Index, mask_test: pd.Index, target: str, league: str) -> None:
        if len(mask_train) == 0 or len(mask_test) == 0:
            raise RuntimeError(
                f"Insufficient non-null target rows for {target} in {league}: "
                f"train={len(mask_train)}, test={len(mask_test)}"
            )

    def train_outcome_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Tuple[Any, Dict[str, Any], np.ndarray]:
        logger.info("Training Match Outcome Model (LGBMClassifier + Isotonic Calibration)...")
        base_model = LGBMClassifier(
            n_estimators=600,
            learning_rate=0.02,
            max_depth=6,
            subsample=0.8,
            random_state=42,
            verbose=-1,
        )

        calibrated_model = CalibratedClassifierCV(estimator=base_model, method="isotonic", cv=3)
        calibrated_model.fit(X_train, y_train)

        probs = calibrated_model.predict_proba(X_test)
        preds = calibrated_model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        ll = log_loss(y_test, probs)

        y_test_one_hot = pd.get_dummies(y_test).reindex(columns=[0, 1, 2], fill_value=0).values
        brier = np.mean(np.sum((probs - y_test_one_hot) ** 2, axis=1))

        metrics: Dict[str, Any] = {
            "accuracy": round(float(acc), 4),
            "log_loss": round(float(ll), 4),
            "brier_score": round(float(brier), 4),
        }
        logger.info("Outcome Metrics on Test: %s", metrics)

        importances = np.mean(
            [est.estimator.feature_importances_ for est in calibrated_model.calibrated_classifiers_],
            axis=0,
        )
        return calibrated_model, metrics, importances

    def train_count_model(
        self,
        name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Tuple[Any, Dict[str, Any]]:
        logger.info("Training %s Model (LGBMRegressor Poisson)...", name)
        model = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=5,
            objective="poisson",
            subsample=0.8,
            random_state=42,
            verbose=-1,
        )
        model.fit(X_train, y_train)

        preds = np.maximum(0.05, model.predict(X_test))
        mae = mean_absolute_error(y_test, preds)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))

        # Calibration proxy for count models: Brier on event Y>0.
        y_event = (np.asarray(y_test, dtype=float) > 0.0).astype(float)
        p_event = np.clip(1.0 - np.exp(-preds), 1e-6, 1.0 - 1e-6)
        calibration_score = float(brier_score_loss(y_event, p_event))

        metrics: Dict[str, Any] = {
            "mae": round(float(mae), 4),
            "rmse": round(rmse, 4),
            "calibration_score": round(calibration_score, 4),
        }
        logger.info("%s Metrics on Test: %s", name, metrics)
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
    ) -> Tuple[Any, Dict[str, Any]]:
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

        y_event = (np.asarray(y_test, dtype=float) > 0.0).astype(float)
        p_event = np.clip(1.0 - np.exp(-preds), 1e-6, 1.0 - 1e-6)
        calibration_score = float(brier_score_loss(y_event, p_event))

        metrics: Dict[str, Any] = {
            "mae": round(float(mae), 4),
            "rmse": round(rmse, 4),
            "calibration_score": round(calibration_score, 4),
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
        league: Optional[str],
        train_size: int,
        test_size: int,
    ) -> str:
        metadata: Dict[str, Any] = {
            "filename": filename,
            "status": "productive",
            "type": model_type,
            "target": target,
            "features": features,
            "metrics": metrics,
            "training_date": datetime.now().isoformat(),
            "league": league,
            "temporal_validated": True,
            "train_size": int(train_size),
            "test_size": int(test_size),
        }
        self.registry.register_model(name, MODEL_VERSION, metadata)

        manifest_key = f"{name}_v{MODEL_VERSION}_{league if league is not None else 'None'}"
        self.registry.set_active_model(name, manifest_key, league=league)
        return manifest_key

    def _train_league(self, league: str, df_scope: pd.DataFrame) -> Dict[str, Any]:
        train, val, test = self.get_time_splits(df_scope)
        if train.empty or val.empty or test.empty:
            raise RuntimeError(
                f"Insufficient temporal split rows for {league}: "
                f"train={len(train)}, val={len(val)}, test={len(test)}"
            )

        features = self.select_features(train, league)
        if not features:
            raise RuntimeError(f"No predictive features selected for {league}.")

        X_train_full = pd.concat([train[features], val[features]])
        y_train_out = pd.concat([train["outcome"], val["outcome"]])
        X_test = test[features]
        y_test_out = test["outcome"]

        report: Dict[str, Dict[str, Any]] = {}
        models_trained: List[str] = []

        # 1) Match outcome
        outcome_model, outcome_metrics, outcome_imp = self.train_outcome_model(
            X_train_full,
            y_train_out,
            X_test,
            y_test_out,
        )
        outcome_file = self._append_league_suffix("match_outcome_model.pkl", league)
        joblib.dump(outcome_model, self.models_dir / outcome_file)
        report["match_outcome"] = outcome_metrics
        models_trained.append("match_outcome")
        self._register_model(
            name="match_outcome",
            filename=outcome_file,
            model_type="lgbm_classifier",
            target="outcome",
            metrics=outcome_metrics,
            features=features,
            league=league,
            train_size=len(X_train_full),
            test_size=len(X_test),
        )

        # 2) Home goals
        home_train_idx, home_test_idx = self._target_masks(X_train_full, X_test, df_scope, "home_goals")
        self._require_non_empty_masks(home_train_idx, home_test_idx, "home_goals", league)
        home_goals_model, home_goals_metrics = self.train_count_model(
            f"Home Goals [{league}]",
            X_train_full.loc[home_train_idx],
            df_scope.loc[home_train_idx, "home_goals"],
            X_test.loc[home_test_idx],
            df_scope.loc[home_test_idx, "home_goals"],
        )
        home_goals_file = self._append_league_suffix("home_goals_model.pkl", league)
        joblib.dump(home_goals_model, self.models_dir / home_goals_file)
        report["home_goals"] = home_goals_metrics
        models_trained.append("home_goals")
        self._register_model(
            name="home_goals",
            filename=home_goals_file,
            model_type="lgbm_regressor_poisson",
            target="home_goals",
            metrics=home_goals_metrics,
            features=features,
            league=league,
            train_size=len(home_train_idx),
            test_size=len(home_test_idx),
        )

        home_goals_xgb_model, home_goals_xgb_metrics = self.train_count_model_xgb(
            f"Home Goals XGB [{league}]",
            X_train_full.loc[home_train_idx],
            df_scope.loc[home_train_idx, "home_goals"],
            X_test.loc[home_test_idx],
            df_scope.loc[home_test_idx, "home_goals"],
            objective="count:poisson",
        )
        home_goals_xgb_file = self._append_league_suffix("home_goals_xgb_v1.joblib", league)
        joblib.dump(home_goals_xgb_model, self.models_dir / home_goals_xgb_file)
        report["home_goals_xgb"] = home_goals_xgb_metrics
        models_trained.append("home_goals_xgb")
        self._register_model(
            name="home_goals_xgb",
            filename=home_goals_xgb_file,
            model_type="xgb_regressor_count_poisson",
            target="home_goals",
            metrics=home_goals_xgb_metrics,
            features=features,
            league=league,
            train_size=len(home_train_idx),
            test_size=len(home_test_idx),
        )

        # 3) Away goals
        away_train_idx, away_test_idx = self._target_masks(X_train_full, X_test, df_scope, "away_goals")
        self._require_non_empty_masks(away_train_idx, away_test_idx, "away_goals", league)
        away_goals_model, away_goals_metrics = self.train_count_model(
            f"Away Goals [{league}]",
            X_train_full.loc[away_train_idx],
            df_scope.loc[away_train_idx, "away_goals"],
            X_test.loc[away_test_idx],
            df_scope.loc[away_test_idx, "away_goals"],
        )
        away_goals_file = self._append_league_suffix("away_goals_model.pkl", league)
        joblib.dump(away_goals_model, self.models_dir / away_goals_file)
        report["away_goals"] = away_goals_metrics
        models_trained.append("away_goals")
        self._register_model(
            name="away_goals",
            filename=away_goals_file,
            model_type="lgbm_regressor_poisson",
            target="away_goals",
            metrics=away_goals_metrics,
            features=features,
            league=league,
            train_size=len(away_train_idx),
            test_size=len(away_test_idx),
        )

        away_goals_xgb_model, away_goals_xgb_metrics = self.train_count_model_xgb(
            f"Away Goals XGB [{league}]",
            X_train_full.loc[away_train_idx],
            df_scope.loc[away_train_idx, "away_goals"],
            X_test.loc[away_test_idx],
            df_scope.loc[away_test_idx, "away_goals"],
            objective="count:poisson",
        )
        away_goals_xgb_file = self._append_league_suffix("away_goals_xgb_v1.joblib", league)
        joblib.dump(away_goals_xgb_model, self.models_dir / away_goals_xgb_file)
        report["away_goals_xgb"] = away_goals_xgb_metrics
        models_trained.append("away_goals_xgb")
        self._register_model(
            name="away_goals_xgb",
            filename=away_goals_xgb_file,
            model_type="xgb_regressor_count_poisson",
            target="away_goals",
            metrics=away_goals_xgb_metrics,
            features=features,
            league=league,
            train_size=len(away_train_idx),
            test_size=len(away_test_idx),
        )

        # 4) Total corners
        corners_train_idx, corners_test_idx = self._target_masks(X_train_full, X_test, df_scope, "total_corners")
        self._require_non_empty_masks(corners_train_idx, corners_test_idx, "total_corners", league)
        corners_model, corners_metrics = self.train_count_model(
            f"Corners [{league}]",
            X_train_full.loc[corners_train_idx],
            df_scope.loc[corners_train_idx, "total_corners"],
            X_test.loc[corners_test_idx],
            df_scope.loc[corners_test_idx, "total_corners"],
        )
        corners_file = self._append_league_suffix("corners_model.pkl", league)
        joblib.dump(corners_model, self.models_dir / corners_file)
        report["corners"] = corners_metrics
        models_trained.append("corners")
        self._register_model(
            name="corners",
            filename=corners_file,
            model_type="lgbm_regressor_poisson",
            target="total_corners",
            metrics=corners_metrics,
            features=features,
            league=league,
            train_size=len(corners_train_idx),
            test_size=len(corners_test_idx),
        )

        corners_xgb_model, corners_xgb_metrics = self.train_count_model_xgb(
            f"Corners XGB [{league}]",
            X_train_full.loc[corners_train_idx],
            df_scope.loc[corners_train_idx, "total_corners"],
            X_test.loc[corners_test_idx],
            df_scope.loc[corners_test_idx, "total_corners"],
            objective="reg:tweedie",
        )
        corners_xgb_file = self._append_league_suffix("corners_xgb_v1.joblib", league)
        joblib.dump(corners_xgb_model, self.models_dir / corners_xgb_file)
        report["corners_xgb"] = corners_xgb_metrics
        models_trained.append("corners_xgb")
        self._register_model(
            name="corners_xgb",
            filename=corners_xgb_file,
            model_type="xgb_regressor_tweedie",
            target="total_corners",
            metrics=corners_xgb_metrics,
            features=features,
            league=league,
            train_size=len(corners_train_idx),
            test_size=len(corners_test_idx),
        )

        # 5) Total cards
        cards_train_idx, cards_test_idx = self._target_masks(X_train_full, X_test, df_scope, "total_cards")
        self._require_non_empty_masks(cards_train_idx, cards_test_idx, "total_cards", league)
        cards_model, cards_metrics = self.train_count_model(
            f"Cards [{league}]",
            X_train_full.loc[cards_train_idx],
            df_scope.loc[cards_train_idx, "total_cards"],
            X_test.loc[cards_test_idx],
            df_scope.loc[cards_test_idx, "total_cards"],
        )
        cards_file = self._append_league_suffix("cards_model.pkl", league)
        joblib.dump(cards_model, self.models_dir / cards_file)
        report["cards"] = cards_metrics
        models_trained.append("cards")
        self._register_model(
            name="cards",
            filename=cards_file,
            model_type="lgbm_regressor_poisson",
            target="total_cards",
            metrics=cards_metrics,
            features=features,
            league=league,
            train_size=len(cards_train_idx),
            test_size=len(cards_test_idx),
        )

        cards_xgb_model, cards_xgb_metrics = self.train_count_model_xgb(
            f"Cards XGB [{league}]",
            X_train_full.loc[cards_train_idx],
            df_scope.loc[cards_train_idx, "total_cards"],
            X_test.loc[cards_test_idx],
            df_scope.loc[cards_test_idx, "total_cards"],
            objective="reg:tweedie",
        )
        cards_xgb_file = self._append_league_suffix("cards_xgb_v1.joblib", league)
        joblib.dump(cards_xgb_model, self.models_dir / cards_xgb_file)
        report["cards_xgb"] = cards_xgb_metrics
        models_trained.append("cards_xgb")
        self._register_model(
            name="cards_xgb",
            filename=cards_xgb_file,
            model_type="xgb_regressor_tweedie",
            target="total_cards",
            metrics=cards_xgb_metrics,
            features=features,
            league=league,
            train_size=len(cards_train_idx),
            test_size=len(cards_test_idx),
        )

        # League-specific artifacts.
        imp_df = pd.DataFrame({"feature": features, "importance": outcome_imp})
        imp_df = imp_df.sort_values("importance", ascending=False)
        imp_df.to_csv(self.models_dir / f"feature_importance_{league}.csv", index=False)

        baselines: Dict[str, Dict[str, float]] = {}
        desc = X_train_full.describe(percentiles=[0.25, 0.5, 0.75]).fillna(0).to_dict()
        for ft, stats in desc.items():
            baselines[ft] = {
                "mean": float(stats.get("mean", 0.0)),
                "std": float(stats.get("std", 0.0)),
                "25%": float(stats.get("25%", 0.0)),
                "50%": float(stats.get("50%", 0.0)),
                "75%": float(stats.get("75%", 0.0)),
            }

        league_meta = {
            "league": league,
            "training_seasons": {"train_and_val": "<=2022", "test": ">=2023"},
            "feature_count": len(features),
            "models_trained": models_trained,
            "dataset_rows": len(df_scope),
            "train_rows": len(X_train_full),
            "test_rows": len(X_test),
        }

        self._dump_json(self.models_dir / f"feature_baselines_{league}.json", baselines)
        self._dump_json(self.models_dir / f"feature_columns_{league}.json", features)
        self._dump_json(self.models_dir / f"training_metadata_{league}.json", league_meta)
        self._dump_json(self.models_dir / f"training_report_{league}.json", report)

        # Backward-compatible canonical files use PL defaults.
        if league == "PL":
            imp_df.to_csv(self.models_dir / "feature_importance.csv", index=False)
            self._dump_json(self.models_dir / "feature_baselines.json", baselines)
            self._dump_json(self.models_dir / "feature_columns.json", features)

        return {
            "metrics": report,
            "metadata": league_meta,
        }

    def run(self) -> None:
        df = self.load_and_engineer_data()

        if "league" not in df.columns:
            raise RuntimeError("Feature matrix is missing required 'league' column for multi-league training.")

        df_valid = df.dropna(subset=["outcome"]).copy()
        if len(df_valid) == 0:
            logger.error("No valid outcomes to train on. Aborting.")
            return

        all_reports: Dict[str, Dict[str, Any]] = {}
        all_metadata: Dict[str, Dict[str, Any]] = {}
        available_leagues = set(df_valid["league"].astype(str).unique())

        for league in TRACKED_LEAGUES:
            if league not in available_leagues:
                logger.warning("League %s missing from feature matrix. Skipping.", league)
                continue

            league_df = df_valid[df_valid["league"].astype(str) == league].copy()
            logger.info("=== Training specialist models for %s (%d rows) ===", league, len(league_df))
            league_result = self._train_league(league, league_df)
            all_reports[league] = league_result["metrics"]
            all_metadata[league] = league_result["metadata"]

        if not all_reports:
            raise RuntimeError("No leagues were trained. Aborting.")

        combined_metadata = {
            "generated_at": datetime.now().isoformat(),
            "tracked_leagues": list(TRACKED_LEAGUES),
            "leagues": all_metadata,
        }
        combined_report = {
            "generated_at": datetime.now().isoformat(),
            "leagues": all_reports,
        }
        self._dump_json(self.models_dir / "training_metadata.json", combined_metadata)
        self._dump_json(self.models_dir / "training_report.json", combined_report)

        logger.info("Artifacts successfully saved to %s", self.models_dir)
        logger.info("=== Probability Models Training DONE (Multi-league) ===")


if __name__ == "__main__":
    features_csv = DATA_DIR / "features" / "feature_matrix.csv"

    # Persist artifacts directly to the runtime registry directory.
    models_out = MODELS_DIR

    trainer = ProbabilityModelTrainer(features_path=features_csv, models_dir=models_out)
    trainer.run()

