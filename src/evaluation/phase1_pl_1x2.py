"""Point-in-time Premier League 1X2 validation for FFPS Phase 1.

This module is deliberately evaluation-only.  It reuses the frozen feature
matrix, the existing goal-model trainer methods, ``_calc_goals`` and
``MatchSimulator`` while refusing production registry writes, warm starts,
calibrators, bookmaker features, target-derived features, and non-causal rows.
"""

from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence, cast
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

import src.cli.commands.prediction as prediction_module
from src.cli.commands.prediction import ModelSuite, PredictionModel, _calc_goals
from src.config import (
    GOALS_ENSEMBLE_DIVERGENCE_THRESHOLD,
    GOALS_ENSEMBLE_LGBM_WEIGHT,
    GOALS_ENSEMBLE_XGB_WEIGHT,
)
from src.features.engineering import FeatureDefaults
from src.models.train_probability_models import ProbabilityModelTrainer
from src.simulation.match_simulator import DEFAULT_N_SIMULATIONS, MatchSimulator, clamp_lambda


LEAGUE = "PL"
TRAIN_END_EXCLUSIVE = date(2023, 8, 1)
HOLDOUT_START = date(2024, 8, 1)
GOAL_TARGET_CAP = ProbabilityModelTrainer.GOAL_TARGET_CAP

PHASE0_MANIFEST_PATH = Path("baselines/phase0/baseline_manifest.json")
PRODUCTION_MANIFEST_PATH = Path("src/ml/models/manifest.json")
FEATURE_MATRIX_PATH = Path("data/features/feature_matrix.csv")
MATCHES_PATH = Path("data/processed/matches.csv")
PRODUCTION_MODELS_DIR = Path("src/ml/models")

ACTIVE_GOAL_SLOTS = (
    "home_goals_PL",
    "away_goals_PL",
    "home_goals_xgb_PL",
    "away_goals_xgb_PL",
)

BOOKMAKER_TOKENS = (
    "odds",
    "bookmaker",
    "implied_probability",
    "market_price",
    "opening_price",
    "closing_price",
)

TARGET_TOKENS = {
    "home_score",
    "away_score",
    "home_goals",
    "away_goals",
    "result",
    "outcome",
    "match_total_goals",
}

GLOBAL_FALLBACK_PREFIXES = ("std_", "global_", "last_season_")


class Phase1IntegrityError(RuntimeError):
    """Raised when the evaluation cannot prove a required invariant."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def row_ids_hash(values: Iterable[str]) -> str:
    return sha256_json(sorted(str(value) for value in values))


def read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


@dataclass(frozen=True)
class TemporalPolicy:
    """Conservative date-only prediction-time policy."""

    train_end_exclusive: date = TRAIN_END_EXCLUSIVE
    holdout_start: date = HOLDOUT_START
    timestamp_semantics: str = "date_start_utc"
    same_day_source_events: str = "excluded"

    def prediction_time(self, fixture_date: pd.Timestamp) -> pd.Timestamp:
        normalized = pd.Timestamp(fixture_date).normalize()
        if normalized.tzinfo is None:
            return normalized.tz_localize("UTC")
        return normalized.tz_convert("UTC")

    def partition_name(self, fixture_date: pd.Timestamp) -> str:
        value = pd.Timestamp(fixture_date).date()
        if value < self.train_end_exclusive:
            return "training"
        if value < self.holdout_start:
            return "validation"
        return "final_holdout"


@dataclass(frozen=True)
class TeamEvent:
    match_hash: str
    event_date: pd.Timestamp
    league: str
    goals_scored: float
    goals_conceded: float
    corners_scored: float
    corners_conceded: float
    points: float


@dataclass
class PartitionedRows:
    training: pd.DataFrame
    validation: pd.DataFrame
    final_holdout: pd.DataFrame
    exclusions: list[dict[str, Any]]
    manifest: dict[str, Any]


def phase0_manifest() -> dict[str, Any]:
    if not PHASE0_MANIFEST_PATH.is_file():
        raise Phase1IntegrityError(f"Phase 0 manifest missing: {PHASE0_MANIFEST_PATH}")
    return read_json(PHASE0_MANIFEST_PATH)


def _phase0_protected_hash(path: Path) -> str:
    baseline = phase0_manifest()
    candidates: list[dict[str, Any]] = []
    if path == PRODUCTION_MANIFEST_PATH:
        candidates.append(cast(dict[str, Any], baseline["models"]["manifest"]))
    candidates.extend(cast(list[dict[str, Any]], baseline["data"]["protected_artifacts"]))
    candidates.extend(cast(list[dict[str, Any]], baseline["models"]["active_artifacts"]))
    target = path.as_posix()
    for item in candidates:
        if str(item.get("path")) == target:
            return str(item["sha256"])
    raise Phase1IntegrityError(f"Phase 0 has no protected hash for {target}")


def assert_phase0_inputs_unchanged() -> dict[str, Any]:
    """Verify the Phase 1 source data and production model boundary."""

    checked: list[dict[str, Any]] = []
    for path in (PRODUCTION_MANIFEST_PATH, FEATURE_MATRIX_PATH, MATCHES_PATH):
        if not path.is_file():
            raise Phase1IntegrityError(f"Protected Phase 0 input missing: {path}")
        expected = _phase0_protected_hash(path)
        actual = sha256_file(path)
        if actual != expected:
            raise Phase1IntegrityError(
                f"Protected Phase 0 input changed: {path} expected={expected} actual={actual}"
            )
        checked.append({"path": path.as_posix(), "sha256": actual, "passed": True})
    return {"passed": True, "checked": checked}


def production_goal_snapshot() -> dict[str, Any]:
    manifest = read_json(PRODUCTION_MANIFEST_PATH)
    active = cast(dict[str, str], manifest.get("active_models", {}))
    artifacts: list[dict[str, Any]] = []
    for slot in ACTIVE_GOAL_SLOTS:
        manifest_key = active.get(slot)
        if not manifest_key:
            raise Phase1IntegrityError(f"Production active pointer missing: {slot}")
        metadata = cast(dict[str, Any], manifest.get(manifest_key, {}))
        filename = metadata.get("filename")
        if not filename:
            raise Phase1IntegrityError(f"Production model metadata has no filename: {manifest_key}")
        path = PRODUCTION_MODELS_DIR / str(filename)
        if not path.is_file():
            raise Phase1IntegrityError(f"Production model missing: {path}")
        artifacts.append(
            {
                "slot": slot,
                "manifest_key": manifest_key,
                "path": path.as_posix(),
                "sha256": sha256_file(path),
                "feature_order_sha256": sha256_json(metadata.get("features", [])),
            }
        )
    return {
        "manifest": {
            "path": PRODUCTION_MANIFEST_PATH.as_posix(),
            "sha256": sha256_file(PRODUCTION_MANIFEST_PATH),
        },
        "artifacts": artifacts,
        "aggregate_sha256": sha256_json(artifacts),
    }


def assert_production_snapshot_unchanged(before: Mapping[str, Any]) -> dict[str, Any]:
    after = production_goal_snapshot()
    if canonical_json(before) != canonical_json(after):
        raise Phase1IntegrityError("Production goal manifest or active artifacts changed during Phase 1")
    return {"passed": True, "before": json_safe(before), "after": json_safe(after)}


def active_goal_contract() -> tuple[list[str], dict[str, dict[str, Any]]]:
    manifest = read_json(PRODUCTION_MANIFEST_PATH)
    active = cast(dict[str, str], manifest.get("active_models", {}))
    contracts: dict[str, dict[str, Any]] = {}
    feature_orders: list[list[str]] = []
    for slot in ACTIVE_GOAL_SLOTS:
        manifest_key = active.get(slot)
        if not manifest_key:
            raise Phase1IntegrityError(f"Missing active goal pointer: {slot}")
        metadata = cast(dict[str, Any], manifest.get(manifest_key, {}))
        features = [str(item) for item in cast(list[Any], metadata.get("features", []))]
        if not features:
            raise Phase1IntegrityError(f"Missing feature order for active goal model: {slot}")
        feature_orders.append(features)
        contracts[slot] = {
            "manifest_key": manifest_key,
            "type": metadata.get("type"),
            "target": metadata.get("target"),
            "filename": metadata.get("filename"),
            "features": features,
            "feature_order_sha256": sha256_json(features),
        }
    first = feature_orders[0]
    if any(order != first for order in feature_orders[1:]):
        raise Phase1IntegrityError("The four active PL goal models do not share one exact feature order")
    assert_feature_contract(first)
    return first, contracts


def assert_feature_contract(features: Sequence[str]) -> None:
    normalized = [str(feature).lower() for feature in features]
    bookmaker = [
        feature
        for feature in normalized
        if any(token in feature for token in BOOKMAKER_TOKENS)
    ]
    targets = [feature for feature in normalized if feature in TARGET_TOKENS]
    global_fallbacks = [
        feature
        for feature in normalized
        if feature.startswith(GLOBAL_FALLBACK_PREFIXES)
        or any(f"_{prefix}" in feature for prefix in GLOBAL_FALLBACK_PREFIXES)
    ]
    if bookmaker:
        raise Phase1IntegrityError(f"Bookmaker-derived active features are prohibited: {bookmaker}")
    if targets:
        raise Phase1IntegrityError(f"Target-derived active features are prohibited: {targets}")
    if global_fallbacks:
        raise Phase1IntegrityError(
            "Phase 1 cannot prove full-history global fallback causality for active features: "
            f"{global_fallbacks}"
        )


class PointInTimeFeatureAuditor:
    """Reconstruct active features from strictly prior raw events."""

    def __init__(self, raw_matches: pd.DataFrame) -> None:
        raw = raw_matches.copy()
        date_column = "match_date" if "match_date" in raw.columns else "date"
        raw["event_date"] = pd.to_datetime(raw[date_column], errors="coerce").dt.normalize()
        raw = raw.dropna(subset=["event_date", "match_hash", "home_team", "away_team"])
        raw = raw.sort_values(["event_date", "match_hash"], kind="mergesort")
        self.events: MutableMapping[str, list[TeamEvent]] = defaultdict(list)
        for row in raw.itertuples(index=False):
            event_date = pd.Timestamp(row.event_date)
            match_hash = str(row.match_hash)
            league = str(row.league)
            for side, opponent in (("home", "away"), ("away", "home")):
                team = str(getattr(row, f"{side}_team"))
                goals_scored = float(getattr(row, f"{side}_goals"))
                goals_conceded = float(getattr(row, f"{opponent}_goals"))
                corners_scored = float(getattr(row, f"{side}_corners"))
                corners_conceded = float(getattr(row, f"{opponent}_corners"))
                points = 3.0 if goals_scored > goals_conceded else 1.0 if goals_scored == goals_conceded else 0.0
                self.events[team].append(
                    TeamEvent(
                        match_hash=match_hash,
                        event_date=event_date,
                        league=league,
                        goals_scored=goals_scored,
                        goals_conceded=goals_conceded,
                        corners_scored=corners_scored,
                        corners_conceded=corners_conceded,
                        points=points,
                    )
                )
        for team_events in self.events.values():
            team_events.sort(key=lambda item: (item.event_date, item.match_hash))

    def prior_events(self, team: str, prediction_time: pd.Timestamp) -> list[TeamEvent]:
        events = self.events.get(str(team), [])
        cutoff = pd.Timestamp(prediction_time).tz_localize(None).normalize()
        dates = [event.event_date for event in events]
        prior = events[:bisect_left(dates, cutoff)]
        if any(event.event_date >= cutoff for event in prior):
            raise Phase1IntegrityError(f"Non-causal source event admitted for {team} at {cutoff.date()}")
        return prior

    @staticmethod
    def _form_rating(events: Sequence[TeamEvent]) -> float:
        if not events:
            return float(FeatureDefaults.FORM_NEUTRAL)
        alpha = 1.0 - float(FeatureDefaults.FORM_DECAY_DEFAULT)
        rating = float(events[0].points)
        for event in events[1:]:
            rating = (alpha * float(event.points)) + ((1.0 - alpha) * rating)
        return rating

    def audit_row(
        self,
        row: pd.Series,
        features: Sequence[str],
        policy: TemporalPolicy,
        *,
        include_source_ids: bool,
    ) -> dict[str, Any]:
        fixture_date = pd.Timestamp(row["date"]).normalize()
        prediction_time = policy.prediction_time(fixture_date)
        all_source_events: list[TeamEvent] = []
        side_evidence: dict[str, Any] = {}
        comparisons: list[dict[str, Any]] = []
        for side in ("home", "away"):
            team = str(row[f"{side}_team"])
            prior = self.prior_events(team, prediction_time)
            all_source_events.extend(prior)
            side_evidence[side] = {
                "team": team,
                "prior_match_count": len(prior),
                "max_source_date": prior[-1].event_date.date().isoformat() if prior else None,
                "source_match_hashes": [event.match_hash for event in prior] if include_source_ids else None,
                "source_match_hashes_sha256": row_ids_hash(event.match_hash for event in prior),
            }
            expected_form = self._form_rating(prior)
            self._compare_feature(
                row,
                f"{side}_form_rating",
                expected_form,
                comparisons,
                allow_missing=False,
            )
            for window in (3, 5, 10):
                recent = prior[-window:]
                for metric in (
                    "goals_scored",
                    "goals_conceded",
                    "corners_scored",
                    "corners_conceded",
                ):
                    expected = (
                        float(np.mean([getattr(event, metric) for event in recent]))
                        if recent
                        else math.nan
                    )
                    self._compare_feature(
                        row,
                        f"{side}_rolling_{metric}_{window}",
                        expected,
                        comparisons,
                        allow_missing=True,
                    )

        for feature, expected in (
            ("home_team_availability_pct", 1.0),
            ("away_team_availability_pct", 1.0),
            ("precipitation_mm", 0.0),
            ("wind_speed_kmh", 0.0),
        ):
            if feature in features:
                self._compare_feature(row, feature, expected, comparisons, allow_missing=False)

        source_tracker_columns = [
            column
            for column in row.index
            if "_max_source_date" in str(column) and (str(column).startswith("home_") or str(column).startswith("away_"))
        ]
        for column in source_tracker_columns:
            source_date = pd.to_datetime(row[column], errors="coerce")
            if pd.notna(source_date) and pd.Timestamp(source_date).normalize() >= fixture_date:
                raise Phase1IntegrityError(
                    f"Timeline tracker {column} is not before fixture {row['match_hash']}: {source_date}"
                )

        unique_sources = sorted(
            {(event.event_date.isoformat(), event.match_hash) for event in all_source_events}
        )
        max_source = max((event.event_date for event in all_source_events), default=None)
        if max_source is not None and max_source >= fixture_date:
            raise Phase1IntegrityError(f"Source cutoff violation for fixture {row['match_hash']}")
        selected_values = {feature: json_safe(row[feature]) for feature in features}
        missing = [feature for feature in features if pd.isna(row[feature])]
        return {
            "match_hash": str(row["match_hash"]),
            "prediction_time": prediction_time.isoformat(),
            "fixture_date": fixture_date.date().isoformat(),
            "strict_cutoff_passed": True,
            "same_day_sources_excluded": True,
            "source_rows_count": len(unique_sources),
            "source_rows_sha256": sha256_json(unique_sources),
            "max_source_date": max_source.date().isoformat() if max_source is not None else None,
            "sides": side_evidence,
            "selected_feature_values": selected_values,
            "selected_feature_values_sha256": sha256_json(selected_values),
            "missing_features": missing,
            "comparison_count": len(comparisons),
        }

    @staticmethod
    def _compare_feature(
        row: pd.Series,
        feature: str,
        expected: float,
        comparisons: list[dict[str, Any]],
        *,
        allow_missing: bool,
    ) -> None:
        if feature not in row.index:
            raise Phase1IntegrityError(f"Required active feature absent from frozen matrix: {feature}")
        actual = row[feature]
        if pd.isna(expected) and pd.isna(actual) and allow_missing:
            comparisons.append({"feature": feature, "status": "both_missing"})
            return
        if pd.isna(actual) or not np.isclose(float(actual), float(expected), atol=1e-10, rtol=0.0):
            raise Phase1IntegrityError(
                f"Point-in-time feature mismatch for {row.get('match_hash')} {feature}: "
                f"expected={expected!r} actual={actual!r}"
            )
        comparisons.append({"feature": feature, "status": "matched"})


def load_and_partition(
    policy: TemporalPolicy | None = None,
    *,
    include_holdout_source_ids: bool = True,
) -> tuple[PartitionedRows, list[str], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load, prove, and partition the frozen PL feature matrix."""

    policy = policy or TemporalPolicy()
    assert_phase0_inputs_unchanged()
    features, contracts = active_goal_contract()
    matrix = pd.read_csv(FEATURE_MATRIX_PATH, low_memory=False)
    raw = pd.read_csv(MATCHES_PATH, low_memory=False)
    required = {"match_hash", "date", "league", "home_team", "away_team", "home_score", "away_score", *features}
    missing_columns = sorted(required - set(matrix.columns))
    if missing_columns:
        raise Phase1IntegrityError(f"Frozen feature matrix lacks required columns: {missing_columns}")
    matrix["date"] = pd.to_datetime(matrix["date"], errors="coerce").dt.normalize()
    if matrix["date"].isna().any():
        raise Phase1IntegrityError("Frozen feature matrix contains unparseable dates")
    pl = matrix[matrix["league"].astype(str).eq(LEAGUE)].copy()
    pl = pl.sort_values(["date", "match_hash"], kind="mergesort").reset_index(drop=True)
    if pl.empty:
        raise Phase1IntegrityError("No Premier League rows found")
    if pl["match_hash"].duplicated().any():
        duplicates = pl.loc[pl["match_hash"].duplicated(), "match_hash"].tolist()
        raise Phase1IntegrityError(f"Duplicate PL match hashes: {duplicates[:10]}")

    auditor = PointInTimeFeatureAuditor(raw)
    partition_rows: dict[str, list[int]] = {"training": [], "validation": [], "final_holdout": []}
    exclusions: list[dict[str, Any]] = []
    holdout_sources: dict[str, dict[str, Any]] = {}
    audit_counts: dict[str, int] = defaultdict(int)
    for index, row in pl.iterrows():
        partition = policy.partition_name(pd.Timestamp(row["date"]))
        evidence = auditor.audit_row(
            row,
            features,
            policy,
            include_source_ids=include_holdout_source_ids and partition == "final_holdout",
        )
        audit_counts[partition] += 1
        if partition == "final_holdout":
            holdout_sources[str(row["match_hash"])] = evidence
        reasons: list[str] = []
        if evidence["missing_features"]:
            reasons.append("insufficient_history_or_missing_active_feature")
        if pd.isna(row["home_score"]) or pd.isna(row["away_score"]):
            reasons.append("missing_final_score")
        if reasons:
            exclusions.append(
                {
                    "match_hash": str(row["match_hash"]),
                    "fixture_date": pd.Timestamp(row["date"]).date().isoformat(),
                    "partition": partition,
                    "reasons": reasons,
                    "missing_features": evidence["missing_features"],
                }
            )
            continue
        partition_rows[partition].append(index)

    parts = {
        name: pl.loc[indexes].copy().reset_index(drop=True)
        for name, indexes in partition_rows.items()
    }
    for name, frame in parts.items():
        if frame.empty:
            raise Phase1IntegrityError(f"Approved {name} partition is empty")
    if parts["training"]["date"].max().date() >= policy.train_end_exclusive:
        raise Phase1IntegrityError("Training partition crossed its exclusive cutoff")
    if not (
        parts["validation"]["date"].min().date() >= policy.train_end_exclusive
        and parts["validation"]["date"].max().date() < policy.holdout_start
    ):
        raise Phase1IntegrityError("Validation partition violates the declared boundary")
    if parts["final_holdout"]["date"].min().date() < policy.holdout_start:
        raise Phase1IntegrityError("Final holdout crossed its lower boundary")
    train_ids = set(parts["training"]["match_hash"].astype(str))
    val_ids = set(parts["validation"]["match_hash"].astype(str))
    holdout_ids = set(parts["final_holdout"]["match_hash"].astype(str))
    if train_ids & val_ids or train_ids & holdout_ids or val_ids & holdout_ids:
        raise Phase1IntegrityError("Dataset partitions overlap")

    def describe(name: str, frame: pd.DataFrame) -> dict[str, Any]:
        return {
            "name": name,
            "rows": len(frame),
            "date_min": frame["date"].min().date().isoformat(),
            "date_max": frame["date"].max().date().isoformat(),
            "match_hashes_sha256": row_ids_hash(frame["match_hash"].astype(str)),
            "match_hashes": frame["match_hash"].astype(str).tolist(),
            "label_distribution": outcome_counts(frame),
        }

    manifest = {
        "schema_version": 1,
        "league": LEAGUE,
        "market": "1X2",
        "source": {
            "feature_matrix": {"path": FEATURE_MATRIX_PATH.as_posix(), "sha256": sha256_file(FEATURE_MATRIX_PATH)},
            "raw_matches": {"path": MATCHES_PATH.as_posix(), "sha256": sha256_file(MATCHES_PATH)},
        },
        "temporal_policy": {
            "timestamp_semantics": policy.timestamp_semantics,
            "same_day_source_events": policy.same_day_source_events,
            "training": f"date < {policy.train_end_exclusive.isoformat()}",
            "validation": f"{policy.train_end_exclusive.isoformat()} <= date < {policy.holdout_start.isoformat()}",
            "final_holdout": f"date >= {policy.holdout_start.isoformat()}",
        },
        "feature_causality": {
            "audited_rows": dict(audit_counts),
            "active_feature_count": len(features),
            "active_feature_order_sha256": sha256_json(features),
            "recomputed_from_strictly_prior_raw_events": True,
            "same_day_team_events_admitted": False,
            "global_fallback_features_used": False,
            "static_fallbacks": {
                "home_team_availability_pct": 1.0,
                "away_team_availability_pct": 1.0,
                "precipitation_mm": 0.0,
                "wind_speed_kmh": 0.0,
            },
        },
        "partitions": {
            "training": describe("training", parts["training"]),
            "validation": describe("validation", parts["validation"]),
            "final_holdout": describe("final_holdout", parts["final_holdout"]),
        },
        "exclusions": {
            "count": len(exclusions),
            "by_partition": {
                name: sum(item["partition"] == name for item in exclusions)
                for name in partition_rows
            },
            "records_sha256": sha256_json(exclusions),
        },
    }
    return (
        PartitionedRows(
            training=parts["training"],
            validation=parts["validation"],
            final_holdout=parts["final_holdout"],
            exclusions=exclusions,
            manifest=manifest,
        ),
        features,
        contracts,
        holdout_sources,
    )


def outcome_index(home_score: float, away_score: float) -> int:
    if home_score > away_score:
        return 0
    if home_score == away_score:
        return 1
    return 2


def outcome_label(home_score: float, away_score: float) -> str:
    return ("H", "D", "A")[outcome_index(home_score, away_score)]


def outcome_counts(frame: pd.DataFrame) -> dict[str, int]:
    result = {"H": 0, "D": 0, "A": 0}
    for row in frame[["home_score", "away_score"]].itertuples(index=False):
        result[outcome_label(float(row.home_score), float(row.away_score))] += 1
    return result


def _trainer_without_registry() -> ProbabilityModelTrainer:
    """Allocate the existing trainer without its registry-writing constructor."""

    return cast(ProbabilityModelTrainer, ProbabilityModelTrainer.__new__(ProbabilityModelTrainer))


def assert_isolated_training_policy(*, warm_start: bool, calibrators_enabled: bool) -> None:
    if warm_start:
        raise Phase1IntegrityError("Warm start is forbidden for the isolated Phase 1 bundle")
    if calibrators_enabled:
        raise Phase1IntegrityError("Calibration is forbidden for the isolated Phase 1 slice")


def _model_metadata(
    *,
    name: str,
    model_type: str,
    target: str,
    features: Sequence[str],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "name": name,
        "version": "phase1-evaluation-only-v1",
        "type": model_type,
        "target": target,
        "league": LEAGUE,
        "features": list(features),
        "feature_order_sha256": sha256_json(list(features)),
        "metrics": json_safe(metrics),
        "evaluation_only": True,
        "production_eligible": False,
        "warm_start": False,
        "calibrators_enabled": False,
        "calibrator_filename": None,
    }


def train_isolated_goal_bundle(
    partitions: PartitionedRows,
    features: Sequence[str],
    output_dir: Path,
    *,
    warm_start: bool = False,
    calibrators_enabled: bool = False,
) -> dict[str, Any]:
    """Train the four live-path goal members without registry or production writes."""

    assert_isolated_training_policy(warm_start=warm_start, calibrators_enabled=calibrators_enabled)
    output_dir.mkdir(parents=True, exist_ok=False)
    trainer = _trainer_without_registry()
    x_train = partitions.training[list(features)].copy()
    x_validation = partitions.validation[list(features)].copy()
    y = {
        "home": partitions.training["home_score"].clip(upper=GOAL_TARGET_CAP).astype(float),
        "away": partitions.training["away_score"].clip(upper=GOAL_TARGET_CAP).astype(float),
    }
    y_validation = {
        "home": partitions.validation["home_score"].clip(upper=GOAL_TARGET_CAP).astype(float),
        "away": partitions.validation["away_score"].clip(upper=GOAL_TARGET_CAP).astype(float),
    }
    specs = (
        ("home_goals", "home", "lgbm_regressor_poisson", "lgbm"),
        ("away_goals", "away", "lgbm_regressor_poisson", "lgbm"),
        ("home_goals_xgb", "home", "xgb_regressor_count_poisson", "xgb"),
        ("away_goals_xgb", "away", "xgb_regressor_count_poisson", "xgb"),
    )
    artifact_records: list[dict[str, Any]] = []
    for name, side, model_type, family in specs:
        if family == "lgbm":
            model, metrics = trainer.train_count_model(
                f"Phase 1 {name}",
                x_train,
                y[side],
                x_validation,
                y_validation[side],
                existing_model_path=None,
                full_retrain=True,
            )
        else:
            model, metrics = trainer.train_count_model_xgb(
                f"Phase 1 {name}",
                x_train,
                y[side],
                x_validation,
                y_validation[side],
                objective="count:poisson",
                existing_model_path=None,
                full_retrain=True,
            )
        metadata = _model_metadata(
            name=name,
            model_type=model_type,
            target=f"{side}_goals",
            features=features,
            metrics=metrics,
        )
        setattr(model, "meta", metadata)
        artifact_path = output_dir / f"{name}.evalmodel"
        joblib.dump(model, artifact_path)
        try:
            recorded_artifact_path = artifact_path.resolve().relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            recorded_artifact_path = artifact_path.name
        params = cast(dict[str, Any], model.get_params())
        parameter_keys = (
            "n_estimators",
            "learning_rate",
            "max_depth",
            "objective",
            "subsample",
            "colsample_bytree",
            "random_state",
            "n_jobs",
            "tree_method",
            "enable_categorical",
            "eval_metric",
            "verbosity",
            "verbose",
        )
        artifact_records.append(
            {
                "role": name,
                "family": family,
                "type": model_type,
                "target": f"{side}_goals",
                "path": recorded_artifact_path,
                "sha256": sha256_file(artifact_path),
                "size_bytes": artifact_path.stat().st_size,
                "features": list(features),
                "feature_order_sha256": sha256_json(list(features)),
                "training_rows": len(x_train),
                "validation_rows": len(x_validation),
                "training_row_ids_sha256": row_ids_hash(partitions.training["match_hash"].astype(str)),
                "validation_row_ids_sha256": row_ids_hash(partitions.validation["match_hash"].astype(str)),
                "target_cap": GOAL_TARGET_CAP,
                "metrics_on_validation": json_safe(metrics),
                "parameters": json_safe({key: params.get(key) for key in parameter_keys if key in params}),
                "warm_start": False,
                "calibrators_enabled": False,
                "evaluation_only": True,
                "production_eligible": False,
            }
        )
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "league": LEAGUE,
        "market": "1X2-via-goal-lambdas",
        "evaluation_only": True,
        "production_eligible": False,
        "registry_used": False,
        "warm_start": False,
        "calibrators_enabled": False,
        "trainer_reuse": {
            "path": "src/models/train_probability_models.py",
            "symbols": [
                "ProbabilityModelTrainer.train_count_model",
                "ProbabilityModelTrainer.train_count_model_xgb",
            ],
            "constructor_bypassed_to_prevent_registry_side_effects": True,
        },
        "feature_order": list(features),
        "feature_order_sha256": sha256_json(list(features)),
        "ensemble": {
            "lgbm_weight": float(GOALS_ENSEMBLE_LGBM_WEIGHT),
            "xgb_weight": float(GOALS_ENSEMBLE_XGB_WEIGHT),
            "divergence_threshold": float(GOALS_ENSEMBLE_DIVERGENCE_THRESHOLD),
        },
        "artifacts": artifact_records,
        "artifacts_aggregate_sha256": sha256_json(
            [{"role": item["role"], "sha256": item["sha256"]} for item in artifact_records]
        ),
    }


def load_evaluation_bundle(bundle_manifest: Mapping[str, Any]) -> dict[str, PredictionModel]:
    if not bundle_manifest.get("evaluation_only") or bundle_manifest.get("production_eligible"):
        raise Phase1IntegrityError("Refusing to load a bundle that is not explicitly evaluation-only")
    if bundle_manifest.get("warm_start") or bundle_manifest.get("calibrators_enabled"):
        raise Phase1IntegrityError("Evaluation bundle violates clean-training policy")
    models: dict[str, PredictionModel] = {}
    expected_features = cast(list[str], bundle_manifest["feature_order"])
    for artifact in cast(list[dict[str, Any]], bundle_manifest["artifacts"]):
        path = Path(str(artifact["path"]))
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            raise Phase1IntegrityError(f"Evaluation artifact hash mismatch: {path}")
        model = cast(PredictionModel, joblib.load(path))
        metadata = cast(dict[str, Any], getattr(model, "meta", {}) or {})
        if metadata.get("features") != expected_features:
            raise Phase1IntegrityError(f"Evaluation model feature order mismatch: {artifact['role']}")
        if metadata.get("calibrator_filename"):
            raise Phase1IntegrityError(f"Evaluation model unexpectedly references a calibrator: {artifact['role']}")
        models[str(artifact["role"])] = model
    expected_roles = {"home_goals", "away_goals", "home_goals_xgb", "away_goals_xgb"}
    if set(models) != expected_roles:
        raise Phase1IntegrityError(f"Evaluation goal bundle roles mismatch: {sorted(models)}")
    return models


class AuditedModel:
    """Transparent prediction-call recorder used around an existing model."""

    def __init__(self, role: str, model: PredictionModel) -> None:
        self.role = role
        self.model = model
        self.meta = cast(dict[str, Any], getattr(model, "meta", {}) or {})
        self.features = self.meta.get("features")
        self.feature_names_in_ = getattr(model, "feature_names_in_", np.asarray(self.features))
        self.calls: list[dict[str, Any]] = []

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        result = np.asarray(self.model.predict(frame), dtype=float)
        self.calls.append(
            {
                "columns": frame.columns.tolist(),
                "input_sha256": sha256_json(frame.iloc[0].to_dict()),
                "output": result.tolist(),
            }
        )
        return result


class AuditedMatchSimulator(MatchSimulator):
    """MatchSimulator subclass that records calls while preserving its math."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.audit_calls: list[dict[str, Any]] = []

    def simulate(
        self,
        home_xg: float,
        away_xg: float,
        rl_weights: dict[str, float] | None = None,
    ) -> Any:
        self.audit_calls.append(
            {"home_xg": float(home_xg), "away_xg": float(away_xg), "rl_weights": rl_weights}
        )
        return super().simulate(home_xg, away_xg, rl_weights=rl_weights)


def fixture_seed(
    *,
    match_hash: str,
    prediction_time: str,
    bundle_hash: str,
    n_simulations: int,
) -> int:
    material = f"{match_hash}|{prediction_time}|{bundle_hash}|{n_simulations}|phase1-pl-1x2-v1"
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big", signed=False)


def predict_fixture_live_path(
    row: pd.Series,
    models: Mapping[str, PredictionModel],
    bundle_manifest: Mapping[str, Any],
    source_evidence: Mapping[str, Any],
    *,
    n_simulations: int = DEFAULT_N_SIMULATIONS,
) -> dict[str, Any]:
    """Invoke the existing `_calc_goals` and MatchSimulator path for one row."""

    features = cast(list[str], bundle_manifest["feature_order"])
    if list(source_evidence.get("missing_features", [])):
        raise Phase1IntegrityError(f"Cannot predict an excluded row: {row['match_hash']}")
    wrapped = {role: AuditedModel(role, model) for role, model in models.items()}
    suite: ModelSuite = {
        "mh_goals": wrapped["home_goals"],
        "ma_goals": wrapped["away_goals"],
        "meta_goals": {"features": features},
        "mh_goals_xgb": wrapped["home_goals_xgb"],
        "ma_goals_xgb": wrapped["away_goals_xgb"],
        "meta_goals_xgb": {"features": features},
    }
    prediction_time = str(source_evidence["prediction_time"])
    seed = fixture_seed(
        match_hash=str(row["match_hash"]),
        prediction_time=prediction_time,
        bundle_hash=str(bundle_manifest["artifacts_aggregate_sha256"]),
        n_simulations=n_simulations,
    )
    simulator = AuditedMatchSimulator(n_simulations=n_simulations, league=LEAGUE, seed=seed)
    context = f"Phase1:{row['home_team']} vs {row['away_team']}:{row['match_hash']}"
    with patch.object(prediction_module, "_load_goals_calibrators", return_value={}) as calibrator_loader:
        result = _calc_goals(
            row,
            suite,
            context,
            simulator=simulator,
            league=LEAGUE,
            rl_weights=None,
            use_simulator=True,
        )
    if calibrator_loader.call_count != 1:
        raise Phase1IntegrityError("Goal-calibrator isolation hook was not exercised exactly once")
    if len(simulator.audit_calls) != 1:
        raise Phase1IntegrityError("MatchSimulator was not called exactly once")
    if any(len(model.calls) != 1 for model in wrapped.values()):
        raise Phase1IntegrityError("Each of the four goal members must be called exactly once")

    member_raw = {role: float(model.calls[0]["output"][0]) for role, model in wrapped.items()}
    member_clamped = {role: float(clamp_lambda(value)) for role, value in member_raw.items()}
    probabilities = {
        "H": float(result["home"]),
        "D": float(result["draw"]),
        "A": float(result["away"]),
    }
    probability_sum = sum(probabilities.values())
    if not np.isclose(probability_sum, 1.0, atol=1e-12, rtol=0.0):
        raise Phase1IntegrityError(
            f"1X2 probabilities do not normalize for {row['match_hash']}: {probability_sum}"
        )
    if not (0.05 <= float(result["goal_model_home_lambda"]) <= 4.0):
        raise Phase1IntegrityError("Home ensemble lambda left production clamp range")
    if not (0.05 <= float(result["goal_model_away_lambda"]) <= 4.0):
        raise Phase1IntegrityError("Away ensemble lambda left production clamp range")

    return {
        "schema_version": 1,
        "fixture": {
            "match_hash": str(row["match_hash"]),
            "fixture_date": pd.Timestamp(row["date"]).date().isoformat(),
            "league": LEAGUE,
            "home_team": str(row["home_team"]),
            "away_team": str(row["away_team"]),
            "final_home_score": float(row["home_score"]),
            "final_away_score": float(row["away_score"]),
            "actual_1x2": outcome_label(float(row["home_score"]), float(row["away_score"])),
        },
        "prediction_time": prediction_time,
        "temporal_evidence": json_safe(source_evidence),
        "feature_values_sha256": source_evidence["selected_feature_values_sha256"],
        "model_bundle_sha256": bundle_manifest["artifacts_aggregate_sha256"],
        "member_model_hashes": {
            str(item["role"]): str(item["sha256"])
            for item in cast(list[dict[str, Any]], bundle_manifest["artifacts"])
        },
        "rng": {
            "algorithm": "numpy.random.default_rng",
            "fixture_seed": seed,
            "n_simulations": n_simulations,
            "fixture_specific": True,
            "production_seeding_changed": False,
        },
        "pathway": {
            "entry": "src.evaluation.phase1_pl_1x2.predict_fixture_live_path",
            "goal_function": "src.cli.commands.prediction._calc_goals",
            "simulator": "src.simulation.match_simulator.MatchSimulator.simulate",
            "member_calls": {role: len(model.calls) for role, model in wrapped.items()},
            "match_outcome_classifier_present": False,
            "match_outcome_classifier_used": False,
            "goal_calibrators_loaded": False,
            "model_level_calibrators_present": False,
            "rl_weights": None,
        },
        "member_raw_lambdas": member_raw,
        "member_clamped_lambdas": member_clamped,
        "ensemble": {
            "lgbm_weight": float(GOALS_ENSEMBLE_LGBM_WEIGHT),
            "xgb_weight": float(GOALS_ENSEMBLE_XGB_WEIGHT),
            "home_lambda": float(result["goal_model_home_lambda"]),
            "away_lambda": float(result["goal_model_away_lambda"]),
            "divergence": bool(result["ensemble_divergence"]),
            "divergence_pct": float(result["divergence_pct"]),
        },
        "output": {
            "probabilities": probabilities,
            "probability_sum": probability_sum,
            "predicted_1x2": max(probabilities, key=probabilities.get),
            "expected_home_goals_mc": float(result["expected_home_goals"]),
            "expected_away_goals_mc": float(result["expected_away_goals"]),
            "entropy": float(result["mc_entropy"]),
            "match_type": str(result["mc_match_type"]),
            "tail_mass": float(result["mc_tail_mass"]),
        },
    }


def canonical_prediction_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], json_safe(record))


def prediction_records_hash(records: Sequence[Mapping[str, Any]]) -> str:
    ordered = sorted(records, key=lambda item: (str(item["fixture"]["fixture_date"]), str(item["fixture"]["match_hash"])))
    payload = "".join(canonical_json(canonical_prediction_record(record)) + "\n" for record in ordered)
    return sha256_bytes(payload.encode("utf-8"))


def run_holdout_predictions(
    holdout: pd.DataFrame,
    models: Mapping[str, PredictionModel],
    bundle_manifest: Mapping[str, Any],
    source_evidence: Mapping[str, Mapping[str, Any]],
    *,
    n_simulations: int = DEFAULT_N_SIMULATIONS,
    reverse_order: bool = False,
) -> list[dict[str, Any]]:
    frame = holdout.iloc[::-1] if reverse_order else holdout
    records: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        match_hash = str(row["match_hash"])
        evidence = source_evidence.get(match_hash)
        if evidence is None:
            raise Phase1IntegrityError(f"Missing temporal provenance for holdout row: {match_hash}")
        records.append(
            predict_fixture_live_path(
                row,
                models,
                bundle_manifest,
                evidence,
                n_simulations=n_simulations,
            )
        )
    records.sort(key=lambda item: (item["fixture"]["fixture_date"], item["fixture"]["match_hash"]))
    return records


def records_to_arrays(records: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(
        [{"H": 0, "D": 1, "A": 2}[str(record["fixture"]["actual_1x2"])] for record in records],
        dtype=int,
    )
    probabilities = np.asarray(
        [
            [
                float(record["output"]["probabilities"]["H"]),
                float(record["output"]["probabilities"]["D"]),
                float(record["output"]["probabilities"]["A"]),
            ]
            for record in records
        ],
        dtype=float,
    )
    return labels, probabilities


def multiclass_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    if len(labels) == 0:
        raise Phase1IntegrityError("Cannot evaluate an empty prediction set")
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (len(labels), 3):
        raise Phase1IntegrityError(f"Expected an N x 3 probability matrix, got {probabilities.shape}")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-12, rtol=0.0):
        raise Phase1IntegrityError("Probability rows do not normalize")
    clipped = np.clip(probabilities, 1e-15, 1.0)
    one_hot = np.eye(3, dtype=float)[labels]
    return {
        "accuracy": float(np.mean(np.argmax(probabilities, axis=1) == labels)),
        "log_loss": float(-np.mean(np.log(clipped[np.arange(len(labels)), labels]))),
        "multiclass_brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
    }


def reliability_tables(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_bins: int = 10,
) -> dict[str, Any]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    predicted = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    correct = (predicted == labels).astype(float)

    def bins_for(values: np.ndarray, outcomes: np.ndarray) -> tuple[list[dict[str, Any]], float]:
        bins: list[dict[str, Any]] = []
        ece = 0.0
        for index in range(n_bins):
            lower = float(edges[index])
            upper = float(edges[index + 1])
            mask = (values >= lower) & (values < upper if index < n_bins - 1 else values <= upper)
            count = int(mask.sum())
            mean_prediction = float(values[mask].mean()) if count else None
            observed_rate = float(outcomes[mask].mean()) if count else None
            gap = abs(mean_prediction - observed_rate) if count else None
            contribution = (count / len(values)) * gap if count else 0.0
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
        return bins, float(ece)

    top_bins, top_ece = bins_for(confidence, correct)
    class_tables: dict[str, Any] = {}
    class_eces: list[float] = []
    for class_index, label in enumerate(("H", "D", "A")):
        bins, class_ece = bins_for(probabilities[:, class_index], (labels == class_index).astype(float))
        class_tables[label] = {"ece": class_ece, "bins": bins}
        class_eces.append(class_ece)
    return {
        "methodology": {
            "binning": "10 fixed-width bins over [0,1]",
            "top_label_ece": "sum_b (n_b/N) * abs(mean confidence_b - accuracy_b)",
            "classwise_ece": "mean of one-vs-rest ECE for H, D, A",
            "empty_bins": "retained with null rates and zero contribution",
        },
        "top_label": {"ece": top_ece, "bins": top_bins},
        "classwise": {"mean_ece": float(np.mean(class_eces)), "classes": class_tables},
    }


def bootstrap_confidence_intervals(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    seed: int,
    samples: int = 2000,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {"accuracy": [], "log_loss": [], "multiclass_brier": []}
    for _ in range(samples):
        indices = rng.integers(0, len(labels), size=len(labels))
        sample = multiclass_metrics(labels[indices], probabilities[indices])
        for metric in values:
            values[metric].append(sample[metric])
    return {
        "method": "fixture-level nonparametric bootstrap percentile interval",
        "confidence_level": 0.95,
        "samples": samples,
        "seed": seed,
        "intervals": {
            metric: {
                "lower": float(np.quantile(metric_values, 0.025)),
                "upper": float(np.quantile(metric_values, 0.975)),
            }
            for metric, metric_values in values.items()
        },
    }


def evaluate_predictions(
    records: Sequence[Mapping[str, Any]],
    training: pd.DataFrame,
    *,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    labels, probabilities = records_to_arrays(records)
    system_metrics = multiclass_metrics(labels, probabilities)
    reliability = reliability_tables(labels, probabilities)
    system_metrics["top_label_ece"] = float(reliability["top_label"]["ece"])
    system_metrics["classwise_ece"] = float(reliability["classwise"]["mean_ece"])

    uniform = np.full_like(probabilities, 1.0 / 3.0)
    training_labels = np.asarray(
        [outcome_index(float(row.home_score), float(row.away_score)) for row in training.itertuples(index=False)],
        dtype=int,
    )
    frequencies = np.bincount(training_labels, minlength=3).astype(float)
    frequencies /= frequencies.sum()
    frequency_probs = np.tile(frequencies, (len(labels), 1))
    baselines = {
        "uniform": {
            "probabilities": {label: float(uniform[0, index]) for index, label in enumerate(("H", "D", "A"))},
            "metrics": multiclass_metrics(labels, uniform),
        },
        "training_frequency": {
            "probabilities": {label: float(frequencies[index]) for index, label in enumerate(("H", "D", "A"))},
            "metrics": multiclass_metrics(labels, frequency_probs),
            "source_rows": len(training_labels),
            "source_row_ids_sha256": row_ids_hash(training["match_hash"].astype(str)),
        },
    }
    for baseline in baselines.values():
        baseline_probs = uniform if baseline is baselines["uniform"] else frequency_probs
        baseline_reliability = reliability_tables(labels, baseline_probs)
        baseline["metrics"]["top_label_ece"] = float(baseline_reliability["top_label"]["ece"])
        baseline["metrics"]["classwise_ece"] = float(baseline_reliability["classwise"]["mean_ece"])

    metrics = {
        "schema_version": 1,
        "league": LEAGUE,
        "market": "1X2",
        "sample_size": len(labels),
        "date_min": min(str(record["fixture"]["fixture_date"]) for record in records),
        "date_max": max(str(record["fixture"]["fixture_date"]) for record in records),
        "metric_definitions": {
            "accuracy": "mean(argmax(p_H,p_D,p_A) == observed class)",
            "log_loss": "-mean(log(probability assigned to observed class))",
            "multiclass_brier": "mean(sum_k((p_k - y_k)^2))",
            "top_label_ece": reliability["methodology"]["top_label_ece"],
            "classwise_ece": reliability["methodology"]["classwise_ece"],
        },
        "system": system_metrics,
        "baselines": baselines,
        "confidence_intervals": bootstrap_confidence_intervals(
            labels,
            probabilities,
            seed=bootstrap_seed,
        ),
        "diagnostics": {
            "mean_home_lambda": float(np.mean([record["ensemble"]["home_lambda"] for record in records])),
            "mean_away_lambda": float(np.mean([record["ensemble"]["away_lambda"] for record in records])),
            "mean_expected_home_goals_mc": float(np.mean([record["output"]["expected_home_goals_mc"] for record in records])),
            "mean_expected_away_goals_mc": float(np.mean([record["output"]["expected_away_goals_mc"] for record in records])),
            "ensemble_divergence_count": int(sum(bool(record["ensemble"]["divergence"]) for record in records)),
        },
        "bookmaker_metrics": None,
        "roi_metrics": None,
    }
    reliability_artifact = {
        "schema_version": 1,
        "league": LEAGUE,
        "market": "1X2",
        "sample_size": len(labels),
        **reliability,
    }
    return metrics, reliability_artifact


def verify_prediction_reproduction(
    first: Sequence[Mapping[str, Any]],
    second: Sequence[Mapping[str, Any]],
    *,
    tolerance: float = 0.0,
) -> dict[str, Any]:
    first_by_id = {str(item["fixture"]["match_hash"]): item for item in first}
    second_by_id = {str(item["fixture"]["match_hash"]): item for item in second}
    if set(first_by_id) != set(second_by_id):
        raise Phase1IntegrityError("Reproduction fixture set changed")
    max_difference = 0.0
    mismatches: list[dict[str, Any]] = []
    for match_hash in sorted(first_by_id):
        left = first_by_id[match_hash]
        right = second_by_id[match_hash]
        for label in ("H", "D", "A"):
            difference = abs(
                float(left["output"]["probabilities"][label])
                - float(right["output"]["probabilities"][label])
            )
            max_difference = max(max_difference, difference)
            if difference > tolerance:
                mismatches.append({"match_hash": match_hash, "field": label, "difference": difference})
        for field in ("home_lambda", "away_lambda"):
            difference = abs(float(left["ensemble"][field]) - float(right["ensemble"][field]))
            max_difference = max(max_difference, difference)
            if difference > tolerance:
                mismatches.append({"match_hash": match_hash, "field": field, "difference": difference})
    if mismatches:
        raise Phase1IntegrityError(f"Prediction reproduction failed: {mismatches[:10]}")
    return {
        "passed": True,
        "fixture_count": len(first_by_id),
        "tolerance": tolerance,
        "max_absolute_difference": max_difference,
        "first_output_sha256": prediction_records_hash(first),
        "second_output_sha256": prediction_records_hash(second),
        "second_run_order": "reverse chronological input; canonical output sort",
        "run_order_independent": True,
    }


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda item: (str(item["fixture"]["fixture_date"]), str(item["fixture"]["match_hash"])))
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in ordered:
            handle.write(canonical_json(canonical_prediction_record(record)) + "\n")


def write_exclusions_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in sorted(records, key=lambda item: (str(item["partition"]), str(item["fixture_date"]), str(item["match_hash"]))):
            handle.write(canonical_json(json_safe(record)) + "\n")


def live_path_evidence() -> dict[str, Any]:
    source = Path("src/cli/commands/prediction.py").read_text(encoding="utf-8")
    start = source.index("def _calc_goals(")
    end = source.index("\ndef _calc_corners(", start)
    function_source = source[start:end]
    if "match_outcome" in function_source:
        raise Phase1IntegrityError("Unexpected match_outcome reference found inside _calc_goals")
    required_fragments = (
        "_predict_scalar(suite['mh_goals']",
        "_predict_scalar(suite['ma_goals']",
        'suite.get("mh_goals_xgb")',
        'suite.get("ma_goals_xgb")',
        "simulator.simulate(model_home_lambda, model_away_lambda",
        "'home': res['home_win'], 'draw': res['draw'], 'away': res['away_win']",
    )
    missing = [fragment for fragment in required_fragments if fragment not in function_source]
    if missing:
        raise Phase1IntegrityError(f"Live-path implementation changed; missing evidence fragments: {missing}")
    return {
        "passed": True,
        "path": "src/cli/commands/prediction.py",
        "symbol": "_calc_goals",
        "function_source_sha256": sha256_bytes(function_source.encode("utf-8")),
        "required_fragments_present": list(required_fragments),
        "match_outcome_reference_present": False,
        "probability_source": "MatchSimulator.simulate home_win_prob/draw_prob/away_win_prob",
    }


def package_versions() -> dict[str, str]:
    import lightgbm
    import platform
    import sklearn
    import xgboost

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
        "xgboost": xgboost.__version__,
        "joblib": joblib.__version__,
    }

