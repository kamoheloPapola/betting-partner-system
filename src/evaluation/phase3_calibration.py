"""Causal PL evaluation and calibration protocol for FFPS Phase 3.

This module is evaluation-only.  It reuses the Phase 1 temporal auditor and
isolated trainer contract, invokes the actual live probability functions, and
keeps every model/calibrator artifact beneath a versioned Phase 3 run folder.
The final-test entry point refuses to run until a byte-hashed experiment
specification has been written and all referenced artifacts still match it.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence, cast
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

import src.cli.commands.prediction as prediction_module
import src.ml.confidence.calculator as confidence_module
from src.cli.commands.prediction import ModelSuite, PredictionModel, _calculate_probabilities_v2, _is_high_intensity
from src.config import GOALS_ENSEMBLE_DIVERGENCE_THRESHOLD, GOALS_ENSEMBLE_LGBM_WEIGHT, GOALS_ENSEMBLE_XGB_WEIGHT, Thresholds
from src.evaluation.phase1_pl_1x2 import (
    AuditedModel,
    FEATURE_MATRIX_PATH,
    LEAGUE,
    MATCHES_PATH,
    Phase1IntegrityError,
    PointInTimeFeatureAuditor,
    active_goal_contract,
    assert_feature_contract,
    assert_phase0_inputs_unchanged,
    bootstrap_confidence_intervals,
    json_safe,
    multiclass_metrics,
    package_versions,
    reliability_tables,
    row_ids_hash,
    sha256_file,
    sha256_json,
    utc_now,
    write_json,
)
from src.ml.calibration import ECE_BINS, MIN_CAL_SAMPLES, MIN_UNIQUE_PROBS, apply_binary_calibrator, save_binary_calibrator_artifact
from src.ml.confidence.calculator import ConfidenceCalculator
from src.ml.confidence.calibration_penalty import get_bucket
from src.ml.distributions import NegativeBinomialEngine, ZeroInflatedEngine
from src.ml.models.corners.team_offsets import MIN_MATCHES_FOR_OFFSET, SHRINKAGE_FACTOR, TeamOffsets, _normalize_team_name
from src.models.train_probability_models import ProbabilityModelTrainer
from src.simulation.match_simulator import DEFAULT_N_SIMULATIONS, MatchSimulator


TRAIN_END_EXCLUSIVE = date(2022, 8, 1)
VALIDATION_END_EXCLUSIVE = date(2023, 8, 1)
CALIBRATION_END_EXCLUSIVE = date(2024, 8, 1)
BOOTSTRAP_SEED = 20260811
BOOTSTRAP_SAMPLES = 2000
CALIBRATION_SELECTION_FRACTION = 0.70
MIN_CLASS_COUNT = 20
RELIABILITY_BINS = 10

PHASE1_RUN = Path("baselines/phase1/phase1-pl-1x2-79110f9a-20240801-v1")
PHASE2_RUN = Path("baselines/phase2/phase2-count-markets-79110f9a-20260811-v1")
PHASE2_CLOSURE = Path("baselines/phase2/closures/phase2-count-markets-79110f9a-20260811-closure-v1")
PRODUCTION_MANIFEST = Path("src/ml/models/manifest.json")
PROCESSED_MATCH_DIR = Path("data/processed/matches")

MODEL_SPECS = (
    ("home_goals", "home_goals", "lgbm", "poisson"),
    ("away_goals", "away_goals", "lgbm", "poisson"),
    ("home_goals_xgb", "home_goals", "xgb", "count:poisson"),
    ("away_goals_xgb", "away_goals", "xgb", "count:poisson"),
    ("home_corners", "home_corners", "lgbm", "poisson"),
    ("away_corners", "away_corners", "lgbm", "poisson"),
    ("total_corners_lgbm", "total_corners", "lgbm", "poisson"),
    ("total_corners_xgb", "total_corners", "xgb", "reg:tweedie"),
    ("total_cards_lgbm", "total_cards", "lgbm", "poisson"),
    ("total_cards_xgb", "total_cards", "xgb", "reg:tweedie"),
)

HOOK_MARKETS = (
    "goals_u25",
    "goals_u35",
    "goals_btts_yes",
    "corners_o75",
    "corners_u11_5",
)

BINARY_MARKETS = (
    "goals_u25",
    "goals_o25",
    "goals_u35",
    "goals_btts_yes",
    "goals_btts_no",
    "corners_o75",
    "corners_u11_5",
    "cards_o25",
    "cards_u45",
    "cards_u55",
)

DERIVED_CALIBRATION = {
    "goals_o25": "goals_u25",
    "goals_btts_no": "goals_btts_yes",
}

OUTPUT_KEYS = {
    "goals_u25": "u25",
    "goals_o25": "o25",
    "goals_u35": "u35",
    "goals_btts_yes": "btts",
    "goals_btts_no": "btts_no",
    "corners_o75": "corn_o75",
    "corners_u11_5": "corn_u11",
    "cards_o25": "card_o25",
    "cards_u45": "card_u45",
    "cards_u55": "card_u55",
}

BOOKMAKER_TOKENS = ("odds", "bookmaker", "implied_probability", "market_price", "opening_price", "closing_price")
TARGET_COLUMNS = ("home_score", "away_score", "home_corners", "away_corners", "home_cards", "away_cards")


class Phase3IntegrityError(RuntimeError):
    """Raised when a Phase 3 chronology, lineage, or freeze invariant fails."""


@dataclass(frozen=True)
class FourPeriodPolicy:
    train_end_exclusive: date = TRAIN_END_EXCLUSIVE
    validation_end_exclusive: date = VALIDATION_END_EXCLUSIVE
    calibration_end_exclusive: date = CALIBRATION_END_EXCLUSIVE
    timestamp_semantics: str = "date_start_utc"
    same_day_source_events: str = "excluded"

    def partition_name(self, fixture_date: pd.Timestamp) -> str:
        value = pd.Timestamp(fixture_date).date()
        if value < self.train_end_exclusive:
            return "training"
        if value < self.validation_end_exclusive:
            return "validation"
        if value < self.calibration_end_exclusive:
            return "calibration"
        return "final_test"

    def prediction_time(self, fixture_date: pd.Timestamp) -> pd.Timestamp:
        value = pd.Timestamp(fixture_date).normalize()
        return value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")


@dataclass
class Phase3Partitions:
    training: pd.DataFrame
    validation: pd.DataFrame
    calibration: pd.DataFrame
    final_test: pd.DataFrame
    exclusions: list[dict[str, Any]]
    manifest: dict[str, Any]
    source_evidence: dict[str, dict[str, Any]]


def canonical_hash_file(path: Path) -> str:
    return sha256_file(path)


def aggregate_directory(path: Path) -> str:
    records = [
        {"path": item.as_posix(), "sha256": sha256_file(item)}
        for item in sorted(path.rglob("*"))
        if item.is_file()
    ]
    return sha256_json(records)


def git_source_state() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()

    status = git("status", "--porcelain=v1", "--untracked-files=all")
    diff = subprocess.run(["git", "diff", "--binary", "HEAD"], check=True, capture_output=True).stdout
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def protected_snapshot() -> dict[str, Any]:
    from scripts.phase0_baseline import data_inventory, model_inventory

    models = model_inventory()
    data = data_inventory()
    phase1_index = json.loads((PHASE1_RUN / "artifact_index.json").read_text(encoding="utf-8"))
    return {
        "captured_at": utc_now(),
        "phase0_evidence_aggregate_sha256": aggregate_directory(Path("baselines/phase0")),
        "phase1_evidence_aggregate_sha256": aggregate_directory(PHASE1_RUN),
        "phase2_evidence_aggregate_sha256": aggregate_directory(PHASE2_RUN),
        "phase2_closure_aggregate_sha256": aggregate_directory(PHASE2_CLOSURE),
        "phase0_golden_sha256": sha256_file(Path("baselines/phase0/golden_predictions.json")),
        "phase1_artifact_aggregate_sha256": phase1_index["aggregate_sha256"],
        "phase2_controlled_golden_sha256": sha256_file(PHASE2_RUN / "controlled_goldens.json"),
        "active_manifest_sha256": models["manifest"]["sha256"],
        "active_model_aggregate_sha256": models["active_artifacts_aggregate_sha256"],
        "protected_data_aggregate_sha256": data["protected_aggregate_sha256"],
        "feature_order_aggregate_sha256": models["active_feature_order_aggregate_sha256"],
    }


def _outcome_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        "H": int((frame["home_score"] > frame["away_score"]).sum()),
        "D": int((frame["home_score"] == frame["away_score"]).sum()),
        "A": int((frame["home_score"] < frame["away_score"]).sum()),
    }


def _event_labels(frame: pd.DataFrame, market: str) -> np.ndarray:
    hg = frame["home_score"].to_numpy(dtype=float)
    ag = frame["away_score"].to_numpy(dtype=float)
    goals = hg + ag
    corners = frame["home_corners"].to_numpy(dtype=float) + frame["away_corners"].to_numpy(dtype=float)
    cards = frame["home_cards"].to_numpy(dtype=float) + frame["away_cards"].to_numpy(dtype=float)
    mapping = {
        "goals_u25": goals <= 2.0,
        "goals_o25": goals > 2.0,
        "goals_u35": goals <= 3.0,
        "goals_btts_yes": (hg > 0.0) & (ag > 0.0),
        "goals_btts_no": (hg <= 0.0) | (ag <= 0.0),
        "corners_o75": corners > 7.5,
        "corners_u11_5": corners < 11.5,
        "cards_o25": cards > 2.5,
        "cards_u45": cards < 4.5,
        "cards_u55": cards < 5.5,
    }
    return np.asarray(mapping[market], dtype=int)


def load_phase3_partitions(
    *,
    policy: FourPeriodPolicy | None = None,
    reveal_final_outcomes: bool = False,
) -> tuple[Phase3Partitions, list[str], dict[str, dict[str, Any]]]:
    """Load and audit four disjoint PL periods.

    Before the experiment is frozen callers use ``reveal_final_outcomes=False``;
    the returned final frame then has every outcome/count target removed.
    """

    policy = policy or FourPeriodPolicy()
    assert_phase0_inputs_unchanged()
    features, contracts = active_goal_contract()
    assert_feature_contract(features)
    if any(any(token in feature.lower() for token in BOOKMAKER_TOKENS) for feature in features):
        raise Phase3IntegrityError("Bookmaker-derived field entered the active feature order")

    matrix = pd.read_csv(FEATURE_MATRIX_PATH, low_memory=False)
    raw = pd.read_csv(MATCHES_PATH, low_memory=False)
    required = {"match_hash", "date", "league", "home_team", "away_team", *TARGET_COLUMNS, *features}
    missing = sorted(required - set(matrix.columns))
    if missing:
        raise Phase3IntegrityError(f"Feature matrix lacks required Phase 3 columns: {missing}")
    matrix["date"] = pd.to_datetime(matrix["date"], errors="coerce").dt.normalize()
    if matrix["date"].isna().any():
        raise Phase3IntegrityError("Unparseable feature-matrix dates")
    pl = matrix[matrix["league"].astype(str).eq(LEAGUE)].copy()
    pl = pl.sort_values(["date", "match_hash"], kind="mergesort").reset_index(drop=True)
    if pl["match_hash"].astype(str).duplicated().any():
        raise Phase3IntegrityError("Duplicate PL match hashes")

    auditor = PointInTimeFeatureAuditor(raw)
    indexes: MutableMapping[str, list[int]] = defaultdict(list)
    exclusions: list[dict[str, Any]] = []
    evidence_by_id: dict[str, dict[str, Any]] = {}
    audited_counts: MutableMapping[str, int] = defaultdict(int)
    for index, row in pl.iterrows():
        name = policy.partition_name(pd.Timestamp(row["date"]))
        evidence = auditor.audit_row(row, features, cast(Any, policy), include_source_ids=False)
        audited_counts[name] += 1
        reasons: list[str] = []
        if evidence["missing_features"]:
            reasons.append("insufficient_history_or_missing_active_feature")
        if any(pd.isna(row[column]) for column in TARGET_COLUMNS):
            reasons.append("missing_required_outcome")
        if reasons:
            exclusions.append(
                {
                    "match_hash": str(row["match_hash"]),
                    "fixture_date": pd.Timestamp(row["date"]).date().isoformat(),
                    "partition": name,
                    "reasons": reasons,
                    "missing_features": evidence["missing_features"],
                }
            )
            continue
        indexes[name].append(index)
        evidence_by_id[str(row["match_hash"])] = evidence

    frames = {name: pl.loc[rows].copy().reset_index(drop=True) for name, rows in indexes.items()}
    for name in ("training", "validation", "calibration", "final_test"):
        if name not in frames or frames[name].empty:
            raise Phase3IntegrityError(f"Approved {name} partition is empty")
    id_sets = {name: set(frame["match_hash"].astype(str)) for name, frame in frames.items()}
    names = list(id_sets)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            if id_sets[left] & id_sets[right]:
                raise Phase3IntegrityError(f"Partitions overlap: {left}/{right}")

    if frames["training"]["date"].max().date() >= policy.train_end_exclusive:
        raise Phase3IntegrityError("Training crossed its cutoff")
    if not (
        frames["validation"]["date"].min().date() >= policy.train_end_exclusive
        and frames["validation"]["date"].max().date() < policy.validation_end_exclusive
    ):
        raise Phase3IntegrityError("Validation boundary violation")
    if not (
        frames["calibration"]["date"].min().date() >= policy.validation_end_exclusive
        and frames["calibration"]["date"].max().date() < policy.calibration_end_exclusive
    ):
        raise Phase3IntegrityError("Calibration boundary violation")
    if frames["final_test"]["date"].min().date() < policy.calibration_end_exclusive:
        raise Phase3IntegrityError("Final-test boundary violation")

    def describe(name: str, frame: pd.DataFrame) -> dict[str, Any]:
        value: dict[str, Any] = {
            "name": name,
            "rows": len(frame),
            "date_min": frame["date"].min().date().isoformat(),
            "date_max": frame["date"].max().date().isoformat(),
            "match_hashes_sha256": row_ids_hash(frame["match_hash"].astype(str)),
            "match_hashes": frame["match_hash"].astype(str).tolist(),
        }
        value["label_distribution"] = "sealed_until_final_evaluation" if name == "final_test" else _outcome_counts(frame)
        if name != "final_test":
            value["binary_event_counts"] = {
                market: {"positive": int(_event_labels(frame, market).sum()), "negative": int(len(frame) - _event_labels(frame, market).sum())}
                for market in BINARY_MARKETS
            }
        return value

    manifest = {
        "schema_version": 1,
        "league": LEAGUE,
        "source": {
            "feature_matrix": {"path": FEATURE_MATRIX_PATH.as_posix(), "sha256": sha256_file(FEATURE_MATRIX_PATH)},
            "raw_matches": {"path": MATCHES_PATH.as_posix(), "sha256": sha256_file(MATCHES_PATH)},
        },
        "temporal_policy": {
            "timestamp_semantics": policy.timestamp_semantics,
            "same_day_source_events": policy.same_day_source_events,
            "training": f"date < {policy.train_end_exclusive.isoformat()}",
            "validation": f"{policy.train_end_exclusive.isoformat()} <= date < {policy.validation_end_exclusive.isoformat()}",
            "calibration": f"{policy.validation_end_exclusive.isoformat()} <= date < {policy.calibration_end_exclusive.isoformat()}",
            "final_test": f"date >= {policy.calibration_end_exclusive.isoformat()}",
        },
        "feature_causality": {
            "audited_rows": dict(audited_counts),
            "active_feature_count": len(features),
            "active_feature_order_sha256": sha256_json(features),
            "strictly_prior_source_events": True,
            "same_day_source_events_admitted": False,
        },
        "partitions": {name: describe(name, frame) for name, frame in frames.items()},
        "exclusions": {
            "count": len(exclusions),
            "by_partition": {name: sum(item["partition"] == name for item in exclusions) for name in frames},
            "records_sha256": sha256_json(exclusions),
        },
        "final_outcomes_revealed": reveal_final_outcomes,
    }
    final_frame = frames["final_test"]
    if not reveal_final_outcomes:
        final_frame = final_frame.drop(columns=list(TARGET_COLUMNS))
    return (
        Phase3Partitions(
            training=frames["training"],
            validation=frames["validation"],
            calibration=frames["calibration"],
            final_test=final_frame,
            exclusions=exclusions,
            manifest=manifest,
            source_evidence=evidence_by_id,
        ),
        features,
        contracts,
    )


def _target(frame: pd.DataFrame, target: str) -> pd.Series:
    if target == "home_goals":
        return frame["home_score"].clip(upper=ProbabilityModelTrainer.GOAL_TARGET_CAP).astype(float)
    if target == "away_goals":
        return frame["away_score"].clip(upper=ProbabilityModelTrainer.GOAL_TARGET_CAP).astype(float)
    if target == "total_corners":
        return (frame["home_corners"] + frame["away_corners"]).astype(float)
    if target == "total_cards":
        return (frame["home_cards"] + frame["away_cards"]).astype(float)
    return frame[target].astype(float)


def _trainer_without_registry() -> ProbabilityModelTrainer:
    return cast(ProbabilityModelTrainer, ProbabilityModelTrainer.__new__(ProbabilityModelTrainer))


def _fit_count_member(
    trainer: ProbabilityModelTrainer,
    role: str,
    target: str,
    family: str,
    objective: str,
    x_fit: pd.DataFrame,
    y_fit: pd.Series,
    x_score: pd.DataFrame,
    y_score: pd.Series,
) -> tuple[PredictionModel, dict[str, Any]]:
    if family == "lgbm":
        return cast(
            tuple[PredictionModel, dict[str, Any]],
            trainer.train_count_model(role, x_fit, y_fit, x_score, y_score, existing_model_path=None, full_retrain=True),
        )
    return cast(
        tuple[PredictionModel, dict[str, Any]],
        trainer.train_count_model_xgb(
            role,
            x_fit,
            y_fit,
            x_score,
            y_score,
            objective=objective,
            existing_model_path=None,
            full_retrain=True,
        ),
    )


def train_isolated_base_bundle(
    partitions: Phase3Partitions,
    features: Sequence[str],
    model_dir: Path,
) -> tuple[dict[str, PredictionModel], dict[str, PredictionModel], dict[str, Any]]:
    """Validate fixed configs on validation, then refit unchanged configs on train+validation."""

    if model_dir.exists():
        raise Phase3IntegrityError(f"Evaluation model directory already exists: {model_dir}")
    model_dir.mkdir(parents=True)
    trainer = _trainer_without_registry()
    x_train = partitions.training[list(features)].copy()
    x_val = partitions.validation[list(features)].copy()
    refit = pd.concat([partitions.training, partitions.validation], ignore_index=True)
    x_refit = refit[list(features)].copy()
    selection_models: dict[str, PredictionModel] = {}
    final_models: dict[str, PredictionModel] = {}
    artifacts: list[dict[str, Any]] = []
    validation_metrics: dict[str, Any] = {}

    for role, target, family, objective in MODEL_SPECS:
        selection_model, metrics = _fit_count_member(
            trainer,
            f"Phase3 selection {role}",
            target,
            family,
            objective,
            x_train,
            _target(partitions.training, target),
            x_val,
            _target(partitions.validation, target),
        )
        selection_models[role] = selection_model
        validation_metrics[role] = metrics
        final_model, final_metrics = _fit_count_member(
            trainer,
            f"Phase3 refit {role}",
            target,
            family,
            objective,
            x_refit,
            _target(refit, target),
            x_val,
            _target(partitions.validation, target),
        )
        model_type = "lgbm_regressor_poisson" if family == "lgbm" else (
            "xgb_regressor_count_poisson" if objective == "count:poisson" else "xgb_regressor_tweedie"
        )
        metadata = {
            "name": role,
            "version": "phase3-evaluation-only-v1",
            "type": model_type,
            "target": target,
            "league": LEAGUE,
            "features": list(features),
            "feature_order_sha256": sha256_json(list(features)),
            "evaluation_only": True,
            "production_eligible": False,
            "warm_start": False,
            "calibrator_filename": None,
            "metrics": final_metrics,
        }
        setattr(final_model, "meta", metadata)
        path = model_dir / f"{role}.evalmodel"
        joblib.dump(final_model, path)
        params = cast(dict[str, Any], final_model.get_params())
        artifacts.append(
            {
                "role": role,
                "target": target,
                "family": family,
                "objective": objective,
                "path": path.as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "features": list(features),
                "feature_order_sha256": sha256_json(list(features)),
                "parameters": json_safe(params),
                "fit_rows": len(refit),
                "fit_row_ids_sha256": row_ids_hash(refit["match_hash"].astype(str)),
                "calibration_rows_used": 0,
                "final_test_rows_used": 0,
                "warm_start": False,
                "evaluation_only": True,
                "production_eligible": False,
            }
        )
        final_models[role] = final_model

    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "league": LEAGUE,
        "evaluation_only": True,
        "production_eligible": False,
        "registry_used": False,
        "warm_start": False,
        "configuration_policy": "Existing ProbabilityModelTrainer family/hyperparameters retained; no alternatives selected.",
        "selection_fit": "training only",
        "selection_score": "validation only",
        "final_refit": "training + validation after configuration freeze",
        "calibration_in_base_fit": False,
        "final_test_in_base_fit": False,
        "trainer": "src.models.train_probability_models.ProbabilityModelTrainer.train_count_model/train_count_model_xgb",
        "feature_order": list(features),
        "feature_order_sha256": sha256_json(list(features)),
        "ensemble": {
            "lgbm_weight": float(GOALS_ENSEMBLE_LGBM_WEIGHT),
            "xgb_weight": float(GOALS_ENSEMBLE_XGB_WEIGHT),
            "divergence_threshold": float(GOALS_ENSEMBLE_DIVERGENCE_THRESHOLD),
        },
        "validation_metrics": validation_metrics,
        "artifacts": artifacts,
        "artifacts_aggregate_sha256": sha256_json([{"role": item["role"], "sha256": item["sha256"]} for item in artifacts]),
    }
    return selection_models, final_models, manifest


def load_base_bundle(manifest: Mapping[str, Any]) -> dict[str, PredictionModel]:
    models: dict[str, PredictionModel] = {}
    for item in cast(Sequence[Mapping[str, Any]], manifest["artifacts"]):
        path = Path(str(item["path"]))
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise Phase3IntegrityError(f"Base-model artifact mismatch: {path}")
        model = cast(PredictionModel, joblib.load(path))
        meta = cast(dict[str, Any], getattr(model, "meta", {}) or {})
        if meta.get("feature_order_sha256") != manifest["feature_order_sha256"]:
            raise Phase3IntegrityError(f"Base-model feature lineage mismatch: {item['role']}")
        models[str(item["role"])] = model
    if set(models) != {item[0] for item in MODEL_SPECS}:
        raise Phase3IntegrityError("Incomplete Phase 3 base-model bundle")
    return models


def binary_reliability(y_true: np.ndarray, probability: np.ndarray, *, n_bins: int = RELIABILITY_BINS) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probability, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict[str, Any]] = []
    ece = 0.0
    for index in range(n_bins):
        lower, upper = float(edges[index]), float(edges[index + 1])
        mask = (p >= lower) & (p < upper if index < n_bins - 1 else p <= upper)
        count = int(mask.sum())
        mean_prediction = float(p[mask].mean()) if count else None
        observed_rate = float(y[mask].mean()) if count else None
        gap = abs(mean_prediction - observed_rate) if count else None
        contribution = (count / len(p)) * gap if count else 0.0
        ece += contribution
        bins.append(
            {
                "bin": index,
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_prediction": mean_prediction,
                "observed_rate": observed_rate,
                "absolute_gap": gap,
                "ece_contribution": contribution,
            }
        )
    return {
        "method": f"{n_bins} fixed-width bins over [0,1]; final bin includes 1.0",
        "ece": float(ece),
        "bins": bins,
    }


def binary_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(probability, dtype=float).reshape(-1)
    if len(y) == 0 or len(y) != len(p):
        raise Phase3IntegrityError("Invalid binary metric arrays")
    if not np.all(np.isfinite(p)) or np.any((p < 0.0) | (p > 1.0)):
        raise Phase3IntegrityError("Binary probability outside [0,1]")
    clipped = np.clip(p, 1e-15, 1.0 - 1e-15)
    reliability = binary_reliability(y, p)
    event_rate = float(y.mean())
    mean_probability = float(p.mean())
    return {
        "sample_size": len(y),
        "positive_outcomes": int(y.sum()),
        "negative_outcomes": int(len(y) - y.sum()),
        "event_rate": event_rate,
        "mean_predicted_probability": mean_probability,
        "calibration_in_the_large": mean_probability - event_rate,
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped))),
        "ece": reliability["ece"],
        "reliability": reliability,
    }


def paired_bootstrap(
    fixture_ids: Sequence[str],
    y_true: np.ndarray,
    raw: np.ndarray,
    calibrated: np.ndarray,
    *,
    seed: int = BOOTSTRAP_SEED,
    samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    if len(set(fixture_ids)) != len(fixture_ids):
        raise Phase3IntegrityError("Paired comparison fixture IDs are not unique")
    if not (len(fixture_ids) == len(y_true) == len(raw) == len(calibrated)):
        raise Phase3IntegrityError("Paired comparison arrays are not aligned")
    rng = np.random.default_rng(seed)
    distributions: dict[str, list[float]] = {"brier": [], "log_loss": [], "ece": []}
    for _ in range(samples):
        indexes = rng.integers(0, len(fixture_ids), size=len(fixture_ids))
        raw_metrics = binary_metrics(y_true[indexes], raw[indexes])
        calibrated_metrics = binary_metrics(y_true[indexes], calibrated[indexes])
        for metric in distributions:
            distributions[metric].append(float(calibrated_metrics[metric] - raw_metrics[metric]))
    point_raw = binary_metrics(y_true, raw)
    point_cal = binary_metrics(y_true, calibrated)
    return {
        "method": "fixture-aligned nonparametric paired bootstrap percentile interval",
        "confidence_level": 0.95,
        "samples": samples,
        "seed": seed,
        "fixture_ids_sha256": row_ids_hash(fixture_ids),
        "deltas": {
            metric: {
                "point": float(point_cal[metric] - point_raw[metric]),
                "lower": float(np.quantile(values, 0.025)),
                "upper": float(np.quantile(values, 0.975)),
            }
            for metric, values in distributions.items()
        },
    }


class IdentityIsotonic:
    """In-memory identity used only to expose the existing live hook input."""

    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float)


def identity_calibrator_payloads() -> dict[str, dict[str, Any]]:
    return {market: {"calibrator": IdentityIsotonic(), "type": "isotonic", "identity_trace": True} for market in HOOK_MARKETS}


def _fit_method(method: str, probability: np.ndarray, labels: np.ndarray) -> Any:
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(probability, labels)
        return model
    if method == "platt":
        model = LogisticRegression(solver="lbfgs", max_iter=1000)
        model.fit(probability.reshape(-1, 1), labels)
        return model
    raise Phase3IntegrityError(f"Unsupported calibrator method: {method}")


def fit_evaluation_calibrator(
    *,
    market: str,
    fixture_ids: Sequence[str],
    probability: np.ndarray,
    labels: np.ndarray,
    base_bundle_sha256: str,
    calibration_predictions_sha256: str,
    output_path: Path,
) -> dict[str, Any]:
    """Chronologically select isotonic/Platt inside calibration, then refit on all calibration rows."""

    p = np.asarray(probability, dtype=float)
    y = np.asarray(labels, dtype=int)
    positives, negatives = int(y.sum()), int(len(y) - y.sum())
    unique_count = int(np.unique(p).size)
    eligibility = {
        "samples": len(p),
        "positive": positives,
        "negative": negatives,
        "unique_probabilities": unique_count,
        "probability_min": float(p.min()),
        "probability_max": float(p.max()),
        "minimum_samples": MIN_CAL_SAMPLES,
        "minimum_unique_probabilities": MIN_UNIQUE_PROBS,
        "minimum_class_count": MIN_CLASS_COUNT,
    }
    if len(p) < MIN_CAL_SAMPLES or unique_count < MIN_UNIQUE_PROBS or min(positives, negatives) < MIN_CLASS_COUNT:
        return {"market": market, "status": "INSUFFICIENT CALIBRATION EVIDENCE", "eligibility": eligibility}

    split = int(math.floor(len(p) * CALIBRATION_SELECTION_FRACTION))
    if split <= 0 or split >= len(p):
        raise Phase3IntegrityError("Invalid internal calibration selection split")
    development_p, selection_p = p[:split], p[split:]
    development_y, selection_y = y[:split], y[split:]
    if np.unique(development_y).size < 2 or np.unique(selection_y).size < 2:
        return {"market": market, "status": "INSUFFICIENT CALIBRATION EVIDENCE", "eligibility": eligibility}

    candidates: list[dict[str, Any]] = []
    for method in ("isotonic", "platt"):
        candidate = _fit_method(method, development_p, development_y)
        predicted = apply_binary_calibrator(candidate, method, selection_p)
        metrics = binary_metrics(selection_y, predicted)
        candidates.append({"method": method, "selection_metrics": {key: metrics[key] for key in ("brier", "log_loss", "ece")}})
    candidates.sort(key=lambda item: (item["selection_metrics"]["ece"], 0 if item["method"] == "isotonic" else 1))
    selected_method = str(candidates[0]["method"])
    calibrator = _fit_method(selected_method, p, y)
    input_hash = sha256_json([float(value) for value in p])
    label_hash = sha256_json([int(value) for value in y])
    row_hash = row_ids_hash(fixture_ids)
    payload = {
        "schema_version": 1,
        "calibrator": calibrator,
        "type": selected_method,
        "market": market,
        "evaluation_only": True,
        "production_eligible": False,
        "base_bundle_sha256": base_bundle_sha256,
        "calibration_predictions_sha256": calibration_predictions_sha256,
        "calibration_input_probability_sha256": input_hash,
        "calibration_label_sha256": label_hash,
        "calibration_row_ids_sha256": row_hash,
        "eligibility": eligibility,
        "selection_policy": {
            "split": "first 70% development / final 30% method selection, chronological",
            "criterion": "lowest 10-bin ECE; exact ties prefer isotonic",
            "candidates": candidates,
            "selected": selected_method,
            "refit": "selected method refit on complete calibration period",
        },
        "final_test_rows_used": 0,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_binary_calibrator_artifact(output_path, payload)
    return {
        "market": market,
        "status": "FITTED_EVALUATION_ONLY",
        "type": selected_method,
        "path": output_path.as_posix(),
        "sha256": sha256_file(output_path),
        "base_bundle_sha256": base_bundle_sha256,
        "calibration_predictions_sha256": calibration_predictions_sha256,
        "calibration_input_probability_sha256": input_hash,
        "calibration_label_sha256": label_hash,
        "calibration_row_ids_sha256": row_hash,
        "eligibility": eligibility,
        "selection_policy": payload["selection_policy"],
    }


def load_evaluation_calibrator(
    item: Mapping[str, Any],
    *,
    expected_bundle_sha256: str,
    expected_calibration_predictions_sha256: str,
) -> dict[str, Any]:
    path = Path(str(item["path"]))
    if not path.is_file() or sha256_file(path) != item["sha256"]:
        raise Phase3IntegrityError(f"Calibrator artifact hash mismatch: {path}")
    payload = cast(dict[str, Any], joblib.load(path))
    if not payload.get("evaluation_only") or payload.get("production_eligible"):
        raise Phase3IntegrityError(f"Unsafe calibrator eligibility metadata: {path}")
    if payload.get("base_bundle_sha256") != expected_bundle_sha256:
        raise Phase3IntegrityError(f"Calibrator base-bundle lineage mismatch: {path}")
    if payload.get("calibration_predictions_sha256") != expected_calibration_predictions_sha256:
        raise Phase3IntegrityError(f"Calibrator calibration-period lineage mismatch: {path}")
    for key in ("calibration_input_probability_sha256", "calibration_label_sha256", "calibration_row_ids_sha256"):
        if payload.get(key) != item.get(key):
            raise Phase3IntegrityError(f"Calibrator {key} mismatch: {path}")
    return payload


class CausalContextBuilder:
    """Reconstruct live H2H and team-offset inputs from strictly earlier rows."""

    def __init__(self, match_dir: Path = PROCESSED_MATCH_DIR) -> None:
        frames = [
            pd.read_csv(path, low_memory=False)
            for path in sorted(match_dir.glob("PL_*.csv"))
            if "upcoming" not in path.name.lower()
        ]
        if not frames:
            raise Phase3IntegrityError("No local PL history is available for causal context")
        history = pd.concat(frames, ignore_index=True)
        history["date"] = pd.to_datetime(history["date"], errors="coerce").dt.normalize()
        history = history[history["date"].notna()].copy()
        history["_home_norm"] = history["home_team"].map(lambda value: _normalize_team_name(str(value), LEAGUE))
        history["_away_norm"] = history["away_team"].map(lambda value: _normalize_team_name(str(value), LEAGUE))
        for column in (
            "home_score", "away_score", "home_corners", "away_corners",
            "home_total_cards", "away_total_cards", "match_total_cards", "total_cards",
        ):
            if column in history:
                history[column] = pd.to_numeric(history[column], errors="coerce")
        if "match_total_cards" in history:
            history["total_cards"] = (
                history["match_total_cards"]
                if "total_cards" not in history
                else history["total_cards"].fillna(history["match_total_cards"])
            )
        history["_source_id"] = history.apply(
            lambda row: str(row.get("match_id") or f"{row['date'].date()}:{row['_home_norm']}:{row['_away_norm']}"),
            axis=1,
        )
        self.history = history.sort_values(["date", "_source_id"], kind="mergesort").reset_index(drop=True)

    def _prior(self, fixture_date: pd.Timestamp) -> pd.DataFrame:
        return self.history[self.history["date"] < pd.Timestamp(fixture_date).normalize()].copy()

    @staticmethod
    def _offset_for(prior: pd.DataFrame, team: str) -> TeamOffsets | None:
        home = prior[prior["_home_norm"].eq(team)]
        away = prior[prior["_away_norm"].eq(team)]
        count = len(home) + len(away)
        if count < MIN_MATCHES_FOR_OFFSET:
            return None
        league_home_corners = float(prior["home_corners"].mean())
        league_away_corners = float(prior["away_corners"].mean())
        if "home_total_cards" in prior and prior["home_total_cards"].notna().any():
            league_home_cards = float(prior["home_total_cards"].mean())
            league_away_cards = float(prior["away_total_cards"].mean())
            raw_home_card = float(home["home_total_cards"].mean() - league_home_cards)
            raw_away_card = float(away["away_total_cards"].mean() - league_away_cards)
        else:
            league_total = float(prior["total_cards"].mean())
            league_home_cards = league_away_cards = league_total / 2.0
            raw_home_card = float(home["total_cards"].mean() / 2.0 - league_home_cards)
            raw_away_card = float(away["total_cards"].mean() / 2.0 - league_away_cards)
        shrink = SHRINKAGE_FACTOR * min(count / 200.0, 1.0)
        return TeamOffsets(
            team_name=team, league=LEAGUE,
            home_corner_bias=float(home["home_corners"].mean() - league_home_corners) * shrink,
            away_corner_bias=float(away["away_corners"].mean() - league_away_corners) * shrink,
            home_card_bias=raw_home_card * shrink,
            away_card_bias=raw_away_card * shrink,
            match_count=count,
        )

    def augment(self, row: pd.Series) -> tuple[pd.Series, dict[str, Any], dict[tuple[str, str], TeamOffsets]]:
        fixture_date = pd.Timestamp(row["date"]).normalize()
        prior = self._prior(fixture_date)
        home = _normalize_team_name(str(row["home_team"]), LEAGUE)
        away = _normalize_team_name(str(row["away_team"]), LEAGUE)
        pair = prior[
            (prior["_home_norm"].eq(home) & prior["_away_norm"].eq(away))
            | (prior["_home_norm"].eq(away) & prior["_away_norm"].eq(home))
        ]
        augmented = row.copy()
        total_goals = pair["home_score"] + pair["away_score"]
        total_corners = pair["home_corners"] + pair["away_corners"]
        if "home_total_cards" in pair and pair["home_total_cards"].notna().any():
            total_cards = pair["home_total_cards"] + pair["away_total_cards"]
        else:
            total_cards = pair["total_cards"]
        augmented["h2h_match_count"] = int(len(pair))
        augmented["h2h_goals_o25_rate"] = float((total_goals > 2.5).mean()) if len(pair) else np.nan
        augmented["h2h_avg_corners"] = float(total_corners.mean()) if len(pair) else np.nan
        augmented["h2h_avg_cards"] = float(total_cards.mean()) if len(pair) else np.nan
        augmented["h2h_cards_o25_rate"] = float((total_cards > 2.5).mean()) if len(pair) else np.nan
        offsets: dict[tuple[str, str], TeamOffsets] = {}
        for team in (home, away):
            value = self._offset_for(prior, team)
            if value is not None:
                offsets[(team, LEAGUE)] = value
        evidence = {
            "prediction_time": fixture_date.date().isoformat(), "strictly_prior": True,
            "same_day_excluded": True,
            "max_source_event_time": prior["date"].max().date().isoformat() if len(prior) else None,
            "prior_rows": len(prior),
            "prior_source_ids_sha256": row_ids_hash(prior["_source_id"].astype(str)),
            "h2h_source_ids_sha256": row_ids_hash(pair["_source_id"].astype(str)),
            "h2h_rows": len(pair),
        }
        if evidence["max_source_event_time"] and evidence["max_source_event_time"] >= evidence["prediction_time"]:
            raise Phase3IntegrityError("Causal context admitted a same-day or future event")
        return augmented, evidence, offsets


class _FixedOffsetManager:
    def __init__(self, offsets: Mapping[tuple[str, str], TeamOffsets]) -> None:
        self.offsets = dict(offsets)

    def get_offset(self, team: str, league: str) -> TeamOffsets | None:
        return self.offsets.get((_normalize_team_name(str(team), league), str(league).upper()))


class _CapturingSimulator(MatchSimulator):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.calls: list[dict[str, Any]] = []

    def simulate(self, home_xg: float, away_xg: float, rl_weights: dict[str, float] | None = None) -> Any:
        result = super().simulate(home_xg, away_xg, rl_weights=rl_weights)
        self.calls.append({"home_xg": float(home_xg), "away_xg": float(away_xg), "result": result})
        return result


class _CapturingNegativeBinomial(NegativeBinomialEngine):
    calls: list[dict[str, Any]] = []

    def calculate_probabilities(self, mu_home: float, var_home: float, mu_away: float, var_away: float) -> dict[str, float]:
        result = super().calculate_probabilities(mu_home, var_home, mu_away, var_away)
        type(self).calls.append({"mu_home": float(mu_home), "mu_away": float(mu_away), "var_home": float(var_home), "var_away": float(var_away), "result": result})
        return result


class _CapturingZeroInflated(ZeroInflatedEngine):
    calls: list[dict[str, Any]] = []

    def calculate_probabilities(self, mu_total: float, pi_zero: float = 0.03, var_mult: float = 1.0) -> dict[str, float]:
        result = super().calculate_probabilities(mu_total, pi_zero=pi_zero, var_mult=var_mult)
        type(self).calls.append({"mu_total": float(mu_total), "pi_zero": float(pi_zero), "var_mult": float(var_mult), "adjusted_mu": float(mu_total * var_mult), "result": result})
        return result


class _AuditedModelView(AuditedModel):
    def __init__(self, role: str, model: PredictionModel) -> None:
        super().__init__(role, model)
        self.meta = dict(self.meta)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)


def confidence_baselines(frame: pd.DataFrame) -> dict[str, float]:
    mapping = {
        "corn_o75": "corners_o75",
        "corn_u11": "corners_u11_5",
        "card_u55": "cards_u55",
    }
    return {key: float(_event_labels(frame, market).mean()) for key, market in mapping.items()}


def derive_validation_hce(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    """Freeze confidence HCE buckets using validation only, never calibration/final."""
    output: dict[str, dict[str, float]] = {}
    market_map = {"corn_o75": "corners_o75", "corn_u11": "corners_u11_5", "card_u55": "cards_u55"}
    for confidence_market, event_market in market_map.items():
        grouped: MutableMapping[str, list[tuple[float, int]]] = defaultdict(list)
        for record in records:
            p = float(cast(Mapping[str, Any], record["probability_stages"])[event_market]["p_final_live"])
            y = int(cast(Mapping[str, Any], record["actuals"])[event_market])
            grouped[get_bucket(p)].append((p, y))
        output[confidence_market] = {
            bucket: float(abs(np.mean([item[0] for item in values]) - np.mean([item[1] for item in values])))
            for bucket, values in grouped.items()
        }
    return output


def _fixture_seed(row: pd.Series, bundle_sha256: str, n_simulations: int) -> int:
    material = f"{row['match_hash']}|{pd.Timestamp(row['date']).date()}|{bundle_sha256}|{n_simulations}|phase3-v1"
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big", signed=False)


def _actuals(row: pd.Series) -> dict[str, Any]:
    frame = pd.DataFrame([row])
    return {
        "1x2": "H" if float(row["home_score"]) > float(row["away_score"]) else (
            "A" if float(row["home_score"]) < float(row["away_score"]) else "D"
        ),
        **{market: int(_event_labels(frame, market)[0]) for market in BINARY_MARKETS},
        "total_goals": float(row["home_score"] + row["away_score"]),
        "total_corners": float(row["home_corners"] + row["away_corners"]),
        "total_cards": float(row["home_cards"] + row["away_cards"]),
    }


def run_fixture_live_path(
    row: pd.Series,
    models: Mapping[str, PredictionModel],
    bundle_manifest: Mapping[str, Any],
    context_builder: CausalContextBuilder,
    *,
    calibrator_payloads: Mapping[str, Mapping[str, Any]] | None = None,
    baselines: Mapping[str, float] | None = None,
    hce_buckets: Mapping[str, Mapping[str, float]] | None = None,
    n_simulations: int = DEFAULT_N_SIMULATIONS,
) -> dict[str, Any]:
    """Invoke unchanged live goal/corner/card math with causal evaluation dependencies."""
    augmented, context_evidence, offsets = context_builder.augment(row)
    wrapped = {role: _AuditedModelView(role, model) for role, model in models.items()}
    payloads = dict(calibrator_payloads or {})
    if payloads:
        wrapped["home_corners"].meta["calibrator_corn_o75_filename"] = "phase3_corners_o75.pkl"
        wrapped["home_corners"].meta["calibrator_corn_u11_filename"] = "phase3_corners_u11_5.pkl"
    features = cast(Sequence[str], bundle_manifest["feature_order"])
    suite: ModelSuite = {
        "mh_goals": wrapped["home_goals"], "ma_goals": wrapped["away_goals"],
        "meta_goals": {"features": list(features)},
        "mh_goals_xgb": wrapped["home_goals_xgb"], "ma_goals_xgb": wrapped["away_goals_xgb"],
        "meta_goals_xgb": {"features": list(features)},
        "mh_corn": wrapped["home_corners"], "ma_corn": wrapped["away_corners"],
        "m_corners_lgbm": wrapped["total_corners_lgbm"], "m_corners_xgb": wrapped["total_corners_xgb"],
        "m_cards_lgbm": wrapped["total_cards_lgbm"], "m_cards_xgb": wrapped["total_cards_xgb"],
        "m_cards": None,
    }
    goal_stages: dict[str, dict[str, float]] = {}
    soft_cap_inputs: dict[str, float] = {}
    original_goal_apply = prediction_module._apply_goals_cal
    original_soft_cap = prediction_module.apply_soft_cap

    def goal_apply(cals: dict[str, Any], market: str, probability: float) -> float:
        after = original_goal_apply(cals, market, probability)
        goal_stages[market] = {"p_pre_calibration_hook": float(probability), "p_after_calibrator": float(after)}
        return after

    def soft_cap(probability: float, confidence: float, market: str) -> float:
        soft_cap_inputs[market] = float(probability)
        return float(original_soft_cap(probability, confidence, market))

    def load_corner(path: Path) -> Mapping[str, Any] | None:
        key = "corners_o75" if "o75" in str(path) else "corners_u11_5"
        return payloads.get(key)

    def baseline(market: str, league: str) -> float:
        return float((baselines or {}).get(market, 0.5))

    def hce(league: str, market: str, probability: float, days_since_update: int = 0, k: float = 5.0) -> float:
        error = float((hce_buckets or {}).get(market, {}).get(get_bucket(probability), 0.10))
        return float(np.clip(math.exp(-error * k), 0.0, 1.0))

    goal_payloads = {
        key: value for key, value in payloads.items()
        if key in {"goals_u25", "goals_u35", "goals_btts_yes"}
    }
    _CapturingNegativeBinomial.calls = []
    _CapturingZeroInflated.calls = []
    seed = _fixture_seed(row, str(bundle_manifest["artifacts_aggregate_sha256"]), n_simulations)
    simulator = _CapturingSimulator(n_simulations=n_simulations, league=LEAGUE, seed=seed)
    manager = _FixedOffsetManager(offsets)
    warnings: dict[str, set[Any]] = {"missing_offsets": set(), "missing_card_offsets": set()}
    with ExitStack() as stack:
        stack.enter_context(patch.object(prediction_module, "_load_goals_calibrators", return_value=goal_payloads))
        stack.enter_context(patch.object(prediction_module, "_apply_goals_cal", side_effect=goal_apply))
        stack.enter_context(patch.object(prediction_module, "load_binary_calibrator_artifact", side_effect=load_corner))
        stack.enter_context(patch.object(prediction_module, "apply_soft_cap", side_effect=soft_cap))
        stack.enter_context(patch.object(prediction_module, "NegativeBinomialEngine", _CapturingNegativeBinomial))
        stack.enter_context(patch.object(prediction_module, "ZeroInflatedEngine", _CapturingZeroInflated))
        stack.enter_context(patch.object(prediction_module, "TeamOffsetManager", return_value=manager))
        stack.enter_context(patch.object(confidence_module, "get_baseline_probability", side_effect=baseline))
        stack.enter_context(patch.object(confidence_module, "calculate_hce_penalty", side_effect=hce))
        result, attribution = _calculate_probabilities_v2(
            augmented, suite, LEAGUE, _is_high_intensity(augmented, LEAGUE), warnings,
            simulator=simulator, use_simulator=True, rl_bandit=None,
        )
    if len(simulator.calls) != 1 or len(_CapturingNegativeBinomial.calls) != 2 or len(_CapturingZeroInflated.calls) < 2:
        raise Phase3IntegrityError("Live-path instrumentation did not observe the expected engines")
    if any(len(item.calls) != 1 for item in wrapped.values()):
        raise Phase3IntegrityError("Each isolated base-model member must execute exactly once")

    sim_result = simulator.calls[0]["result"]
    corner_distribution = _CapturingNegativeBinomial.calls[-1]
    card_distribution = _CapturingZeroInflated.calls[-1]
    stages = {
        "goals_u25": {**goal_stages["goals_u25"], "p_raw_distribution_or_simulation": float(sim_result.under_2_5), "p_final_live": float(result["u25"])},
        "goals_o25": {"p_raw_distribution_or_simulation": float(sim_result.over_2_5), "p_pre_calibration_hook": 1.0 - goal_stages["goals_u25"]["p_pre_calibration_hook"], "p_after_calibrator": 1.0 - goal_stages["goals_u25"]["p_after_calibrator"], "p_final_live": float(result["o25"])},
        "goals_u35": {**goal_stages["goals_u35"], "p_raw_distribution_or_simulation": float(sim_result.under_3_5), "p_final_live": float(result["u35"])},
        "goals_btts_yes": {**goal_stages["goals_btts_yes"], "p_raw_distribution_or_simulation": float(sim_result.btts_prob), "p_final_live": float(result["btts"])},
        "goals_btts_no": {"p_raw_distribution_or_simulation": float(1.0 - sim_result.btts_prob), "p_pre_calibration_hook": 1.0 - goal_stages["goals_btts_yes"]["p_pre_calibration_hook"], "p_after_calibrator": 1.0 - goal_stages["goals_btts_yes"]["p_after_calibrator"], "p_final_live": float(result["btts_no"])},
        "corners_o75": {"p_raw_distribution_or_simulation": float(corner_distribution["result"]["corners_over_7_5"]), "p_pre_calibration_hook": float(corner_distribution["result"]["corners_over_7_5"]), "p_after_calibrator": float(soft_cap_inputs["corn_o75"]), "p_final_live": float(result["corn_o75"])},
        "corners_u11_5": {"p_raw_distribution_or_simulation": float(corner_distribution["result"]["corners_under_11_5"]), "p_pre_calibration_hook": float(corner_distribution["result"]["corners_under_11_5"]), "p_after_calibrator": float(soft_cap_inputs["corn_u11"]), "p_final_live": float(result["corn_u11"])},
        "cards_o25": {"p_raw_distribution_or_simulation": float(card_distribution["result"]["cards_over_2_5"]), "p_final_live": float(result["card_o25"])},
        "cards_u45": {"p_raw_distribution_or_simulation": float(card_distribution["result"]["cards_under_4_5"]), "p_final_live": float(result["card_u45"])},
        "cards_u55": {"p_raw_distribution_or_simulation": float(card_distribution["result"]["cards_under_5_5"]), "p_final_live": float(result["card_u55"])},
    }
    return cast(dict[str, Any], json_safe({
        "fixture": {"match_hash": str(row["match_hash"]), "date": pd.Timestamp(row["date"]).date().isoformat(), "home_team": str(row["home_team"]), "away_team": str(row["away_team"]), "league": LEAGUE},
        "actuals": _actuals(row),
        "probability_stages": stages,
        "one_x_two": {"H": float(result["home"]), "D": float(result["draw"]), "A": float(result["away"])},
        "goal_lambdas": {"home": float(result["goal_model_home_lambda"]), "away": float(result["goal_model_away_lambda"]), "mc_home": float(result["expected_home_goals"]), "mc_away": float(result["expected_away_goals"])},
        "count_means": {"corner_home": corner_distribution["mu_home"], "corner_away": corner_distribution["mu_away"], "corner_total": corner_distribution["mu_home"] + corner_distribution["mu_away"], "card_model_total": card_distribution["mu_total"], "card_distribution_total": card_distribution["adjusted_mu"]},
        "card_intensity": {"flagged": bool(_is_high_intensity(augmented, LEAGUE)), "multiplier": float(card_distribution["var_mult"])},
        "rng": {"seed": seed, "n_simulations": n_simulations},
        "temporal_evidence": context_evidence,
        "feature_values_sha256": sha256_json({feature: float(augmented[feature]) for feature in features}),
        "model_call_hashes": {role: member.calls[0]["input_sha256"] for role, member in wrapped.items()},
        "calibrator_markets_inserted": sorted(payloads),
        "attribution": attribution,
    }))


def run_period_predictions(
    frame: pd.DataFrame, models: Mapping[str, PredictionModel], bundle_manifest: Mapping[str, Any],
    context_builder: CausalContextBuilder, *, calibrator_payloads: Mapping[str, Mapping[str, Any]] | None = None,
    baselines: Mapping[str, float] | None = None, hce_buckets: Mapping[str, Mapping[str, float]] | None = None,
    n_simulations: int = DEFAULT_N_SIMULATIONS,
) -> list[dict[str, Any]]:
    return [
        run_fixture_live_path(row, models, bundle_manifest, context_builder, calibrator_payloads=calibrator_payloads, baselines=baselines, hce_buckets=hce_buckets, n_simulations=n_simulations)
        for _, row in frame.sort_values(["date", "match_hash"], kind="mergesort").iterrows()
    ]


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(json_safe(record), sort_keys=True, separators=(",", ":")) + "\n")
