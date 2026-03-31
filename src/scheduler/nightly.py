"""
Nightly automation pipeline.

Sequence:
1. Fetch latest season data.
2. Build feature matrix.
3. Retrain stale production models (older than N days).
4. Promote retrained models if they improved.
5. Generate performance report.

Run:
    python -m src.scheduler.nightly
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, cast

import pandas as pd

from src.config import DATA_DIR
from src.features.build_feature_matrix import FeatureMatrixBuilder
from src.fetch.fetch_latest_season_csvs import SeasonFetcher
from src.ml.registry import ModelRegistry
from src.ml.trainer import ModelTrainer
from src.ml.training.data_validator import filter_historical_matches, load_feature_file
from src.ml.training.feature_selector import select_features
from src.ml.training.model_configs import MODEL_CONFIGS, ModelType
from src.monitoring.alerter import Alerter
from src.monitoring.performance_engine import PerformanceTracker
from src.monitoring.telemetry import capture_alert, capture_exception, init_sentry

logger = logging.getLogger(__name__)

FEATURE_MATRIX_PATH = DATA_DIR / "features" / "feature_matrix.csv"
ROLLING_BRIER_WINDOW = 30
AUTO_RETRAIN_TRIGGER_MULTIPLIER = 1.15
AUTO_RETRAIN_PROMOTION_MULTIPLIER = 1.05

# Canonical market used to monitor each model family.
MODEL_MARKET_MAP: Dict[str, str] = {
    "poisson_home_base": "home_win",
    "poisson_away_base": "away_win",
    "nb_home_corners_base": "corners_home_win",
    "nb_away_corners_base": "corners_away_win",
    "poisson_total_cards_base": "cards_over_2_5",
}


@dataclass(frozen=True)
class StaleModelTarget:
    model_name: str
    league: str
    manifest_key: str
    trained_at: datetime


@dataclass(frozen=True)
class RetrainOutcome:
    target: StaleModelTarget
    before_key: Optional[str]
    after_key: Optional[str]
    improved: bool


@dataclass(frozen=True)
class AutoRetrainTarget:
    model_name: str
    league: str
    market: str
    manifest_key: str
    version: str
    baseline_brier: float
    rolling_brier: float
    sample_size: int


@dataclass(frozen=True)
class AutoRetrainOutcome:
    target: AutoRetrainTarget
    before_key: Optional[str]
    after_key: Optional[str]
    promoted: bool
    old_brier: float
    new_brier: float
    reason: str


def _run_pipeline_step(step_name: str, step_fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run one pipeline step and emit telemetry if it fails."""
    try:
        return step_fn(*args, **kwargs)
    except Exception as exc:
        capture_exception(
            exc,
            context={
                "pipeline_step": step_name,
                "step_args_count": len(args),
                "step_kwargs": list(kwargs.keys()),
            },
        )
        capture_alert(
            event_name="daily_pipeline_step_failed",
            message=f"Daily pipeline failed at step: {step_name}",
            level="error",
            context={
                "pipeline_step": step_name,
                "error": str(exc),
            },
        )
        raise


def _configure_logging(level: int = logging.INFO) -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )


def _default_season_token(now: Optional[datetime] = None) -> str:
    utc_now = now or datetime.now(timezone.utc)
    season_start_year = utc_now.year if utc_now.month >= 7 else utc_now.year - 1
    next_year = season_start_year + 1
    return f"{season_start_year % 100:02d}{next_year % 100:02d}"


def _parse_datetime(value: Any) -> Optional[datetime]:
    if not value or not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def _config_map() -> Dict[str, Dict[str, Any]]:
    return {str(config["name"]): config for config in MODEL_CONFIGS}


def _iter_productive_models(registry: ModelRegistry) -> Dict[tuple[str, str], tuple[str, Dict[str, Any], datetime]]:
    latest: Dict[tuple[str, str], tuple[str, Dict[str, Any], datetime]] = {}
    for key, meta in registry.manifest.items():
        if not isinstance(meta, dict):
            continue
        if meta.get("status") != "productive":
            continue

        model_name = str(meta.get("name", "")).strip()
        if not model_name:
            continue
        league = str(meta.get("league") or "Global")
        trained_at = _parse_datetime(meta.get("trained_at")) or _parse_datetime(meta.get("registered_at"))
        if trained_at is None:
            continue

        scope = (model_name, league)
        current = latest.get(scope)
        if current is None or trained_at > current[2]:
            latest[scope] = (str(key), meta, trained_at)

    return latest


def find_stale_models(
    registry: ModelRegistry,
    stale_days: int = 7,
    as_of: Optional[datetime] = None,
) -> List[StaleModelTarget]:
    cfg = _config_map()
    cutoff = (as_of or datetime.now(timezone.utc)) - timedelta(days=stale_days)
    stale: List[StaleModelTarget] = []

    for (model_name, league), (manifest_key, _, trained_at) in _iter_productive_models(registry).items():
        if model_name not in cfg:
            continue
        if trained_at <= cutoff:
            stale.append(
                StaleModelTarget(
                    model_name=model_name,
                    league=league,
                    manifest_key=manifest_key,
                    trained_at=trained_at,
                )
            )

    stale.sort(key=lambda item: (item.league, item.model_name))
    return stale


def fetch_latest_data(season: str, force: bool = False) -> None:
    logger.info("Step 1/5: fetching latest data for season=%s force=%s", season, force)
    SeasonFetcher().fetch_season(season=season, force=force)


def build_feature_matrix() -> Dict[str, Any]:
    logger.info("Step 2/5: building feature matrix")
    result = FeatureMatrixBuilder().build()
    if result.get("status") != "success":
        raise RuntimeError(f"Feature matrix build failed: {result}")
    return result


def load_training_dataframe(path: Path = FEATURE_MATRIX_PATH) -> pd.DataFrame:
    df = load_feature_file(path)
    if "status" in df.columns and "date" in df.columns:
        df = filter_historical_matches(df)
    if df.empty:
        raise ValueError("Feature matrix is empty after historical filtering.")
    return df


def _latest_productive_key(
    registry: ModelRegistry,
    model_name: str,
    league: str,
) -> Optional[str]:
    latest_key: Optional[str] = None
    latest_ts: Optional[datetime] = None

    for key, meta in registry.manifest.items():
        if not isinstance(meta, dict):
            continue
        if meta.get("name") != model_name:
            continue
        if str(meta.get("league") or "Global") != league:
            continue
        if meta.get("status") != "productive":
            continue

        ts = _parse_datetime(meta.get("registered_at")) or _parse_datetime(meta.get("trained_at"))
        if ts is None:
            continue
        if latest_ts is None or ts > latest_ts:
            latest_ts = ts
            latest_key = str(key)
    return latest_key


def _extract_brier(meta: Optional[Dict[str, Any]]) -> float:
    if not meta:
        return float("inf")
    metrics = meta.get("metrics")
    if isinstance(metrics, dict):
        value = metrics.get("brier_score")
        if isinstance(value, (int, float)):
            return float(value)
    value = meta.get("brier_score")
    if isinstance(value, (int, float)):
        return float(value)
    return float("inf")


def _extract_ece(meta: Optional[Dict[str, Any]]) -> float:
    if not meta:
        return float("inf")
    metrics = meta.get("metrics")
    if isinstance(metrics, dict):
        value = metrics.get("ece")
        if isinstance(value, (int, float)):
            return float(value)
        value = metrics.get("calibration_score")
        if isinstance(value, (int, float)):
            return float(value)
    value = meta.get("ece")
    if isinstance(value, (int, float)):
        return float(value)
    return float("inf")


def _is_improved(new_meta: Optional[Dict[str, Any]], old_meta: Optional[Dict[str, Any]]) -> bool:
    if old_meta is None:
        return True

    new_brier = _extract_brier(new_meta)
    old_brier = _extract_brier(old_meta)
    if new_brier != float("inf") and old_brier != float("inf"):
        return new_brier < old_brier

    new_ece = _extract_ece(new_meta)
    old_ece = _extract_ece(old_meta)
    return new_ece < old_ece


def load_recent_evaluations() -> pd.DataFrame:
    """Load reconciled evaluations used for rolling Brier monitoring."""
    try:
        tracker = PerformanceTracker()
        return tracker._load_all_evaluations()  # Reuse canonical evaluation loader.
    except Exception as exc:
        logger.warning("Failed to load evaluations for auto-retrain checks: %s", exc)
        return pd.DataFrame()


def _extract_baseline_brier(meta: Optional[Dict[str, Any]]) -> float:
    if not meta:
        return float("inf")
    metrics = meta.get("metrics")
    if isinstance(metrics, dict):
        value = metrics.get("baseline_brier")
        if isinstance(value, (int, float)):
            return float(value)
    value = meta.get("baseline_brier")
    if isinstance(value, (int, float)):
        return float(value)
    return _extract_brier(meta)


def _rolling_brier_for_scope(
    evals_df: pd.DataFrame,
    *,
    league: str,
    market: str,
    window: int = ROLLING_BRIER_WINDOW,
) -> tuple[float, int]:
    required = {"market", "predicted_probability", "actual_outcome"}
    if evals_df.empty or not required.issubset(set(evals_df.columns)):
        return float("inf"), 0

    scoped = evals_df.copy()
    if "league" in scoped.columns:
        scoped = scoped[scoped["league"].astype(str) == league]
    scoped = scoped[scoped["market"].astype(str) == market]
    if scoped.empty:
        return float("inf"), 0

    if "resolved_date" in scoped.columns:
        scoped["__ts"] = pd.to_datetime(scoped["resolved_date"], errors="coerce")
        scoped = scoped.sort_values("__ts")
    elif "prediction_date" in scoped.columns:
        scoped["__ts"] = pd.to_datetime(scoped["prediction_date"], errors="coerce")
        scoped = scoped.sort_values("__ts")

    scoped["__prob"] = pd.to_numeric(scoped["predicted_probability"], errors="coerce")
    scoped["__actual"] = pd.to_numeric(scoped["actual_outcome"], errors="coerce")
    valid = scoped.dropna(subset=["__prob", "__actual"]).tail(window)
    sample_size = int(len(valid))
    if sample_size == 0:
        return float("inf"), 0

    brier = float(((valid["__prob"] - valid["__actual"]) ** 2).mean())
    return brier, sample_size


def find_auto_retrain_targets(
    registry: ModelRegistry,
    evals_df: pd.DataFrame,
    *,
    window: int = ROLLING_BRIER_WINDOW,
) -> List[AutoRetrainTarget]:
    """Identify productive model scopes that breach rolling-Brier guardrails."""
    config_names = set(_config_map().keys())
    targets: List[AutoRetrainTarget] = []

    for (model_name, league), (manifest_key, meta, _) in _iter_productive_models(registry).items():
        if model_name not in config_names:
            continue

        market = MODEL_MARKET_MAP.get(model_name)
        if not market:
            continue

        baseline_brier = _extract_baseline_brier(meta)
        if baseline_brier == float("inf"):
            continue

        rolling_brier, sample_size = _rolling_brier_for_scope(
            evals_df,
            league=league,
            market=market,
            window=window,
        )
        if sample_size < window:
            continue

        if rolling_brier > baseline_brier * AUTO_RETRAIN_TRIGGER_MULTIPLIER:
            targets.append(
                AutoRetrainTarget(
                    model_name=model_name,
                    league=league,
                    market=market,
                    manifest_key=manifest_key,
                    version=str(meta.get("version", "unknown")),
                    baseline_brier=baseline_brier,
                    rolling_brier=rolling_brier,
                    sample_size=sample_size,
                )
            )

    targets.sort(key=lambda item: (item.league, item.model_name))
    return targets


def _log_auto_retrain_event(
    registry: ModelRegistry,
    *,
    event_type: str,
    model_name: str,
    league: str,
    version: str,
    brier_score: float,
    train_size: int,
) -> None:
    """Best-effort event write for auto-retrain lifecycle."""
    try:
        registry.HISTORY_DB().write_event(
            model_name=model_name,
            league=league,
            version=version,
            brier_score=None if brier_score == float("inf") else float(brier_score),
            ece=None,
            train_size=int(train_size),
            event_type=event_type,
        )
    except Exception as exc:
        logger.warning("Failed to write auto-retrain event (%s): %s", event_type, exc)


def _capture_auto_retrain_failure(
    target: AutoRetrainTarget,
    *,
    reason: str,
    error: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "reason": reason,
        "model": target.model_name,
        "league": target.league,
        "market": target.market,
        "version": target.version,
        "baseline_brier": target.baseline_brier,
        "rolling_brier": target.rolling_brier,
        "sample_size": target.sample_size,
    }
    if error:
        payload["error"] = error
    if context:
        payload.update(context)

    capture_alert(
        event_name="auto_retrain_failed",
        message="Auto-retrain failed",
        level="warning",
        context=payload,
    )


def auto_retrain_underperforming_models(
    *,
    registry: ModelRegistry,
    trainer: ModelTrainer,
    feature_df: pd.DataFrame,
    evals_df: pd.DataFrame,
    alerter: Optional[Alerter] = None,
) -> List[AutoRetrainOutcome]:
    """Retrain and conditionally promote models that breach rolling-Brier thresholds."""
    targets = find_auto_retrain_targets(registry=registry, evals_df=evals_df)
    if not targets:
        logger.info("No auto-retrain targets found from rolling-Brier checks.")
        return []

    logger.info("Step 6/6: auto-retraining %d underperforming model scopes", len(targets))
    config_by_name = _config_map()
    notifier = alerter or Alerter()
    outcomes: List[AutoRetrainOutcome] = []

    for target in targets:
        _log_auto_retrain_event(
            registry,
            event_type="auto_retrain_triggered",
            model_name=target.model_name,
            league=target.league,
            version=target.version,
            brier_score=target.rolling_brier,
            train_size=target.sample_size,
        )

        config = config_by_name.get(target.model_name)
        if config is None:
            _capture_auto_retrain_failure(target, reason="missing_model_config")
            _log_auto_retrain_event(
                registry,
                event_type="auto_retrain_config_missing",
                model_name=target.model_name,
                league=target.league,
                version=target.version,
                brier_score=target.rolling_brier,
                train_size=target.sample_size,
            )
            outcomes.append(
                AutoRetrainOutcome(
                    target=target,
                    before_key=None,
                    after_key=None,
                    promoted=False,
                    old_brier=target.baseline_brier,
                    new_brier=float("inf"),
                    reason="missing_model_config",
                )
            )
            continue

        scoped_df = feature_df if target.league == "Global" else feature_df[feature_df["league"] == target.league]
        if scoped_df.empty:
            notifier.send_alert(
                "Auto-retrain failed: no training data for underperforming model",
                context={
                    "model": target.model_name,
                    "league": target.league,
                    "market": target.market,
                },
                severity="WARNING",
            )
            _capture_auto_retrain_failure(target, reason="no_training_rows")
            _log_auto_retrain_event(
                registry,
                event_type="auto_retrain_data_missing",
                model_name=target.model_name,
                league=target.league,
                version=target.version,
                brier_score=target.rolling_brier,
                train_size=0,
            )
            outcomes.append(
                AutoRetrainOutcome(
                    target=target,
                    before_key=None,
                    after_key=None,
                    promoted=False,
                    old_brier=target.baseline_brier,
                    new_brier=float("inf"),
                    reason="no_training_rows",
                )
            )
            continue

        features = select_features(
            scoped_df,
            str(config["target"]),
            feature_set=config["feature_set"],
        )
        params = dict(config.get("params", {}))
        if config["type"] == ModelType.POISSON:
            from src.config.alpha_config import get_alpha

            params["alpha"] = get_alpha(target.league, target.model_name)

        before_key = _latest_productive_key(registry, target.model_name, target.league)
        old_meta = registry.manifest.get(before_key) if before_key else None
        old_brier = _extract_brier(old_meta if isinstance(old_meta, dict) else None)
        if old_brier == float("inf"):
            old_brier = target.baseline_brier

        try:
            trainer.train_model(
                df=scoped_df,
                target_col=str(config["target"]),
                league=None if target.league == "Global" else target.league,
                model_type=config["type"].value,
                model_name=target.model_name,
                features=features,
                params=params,
                extra_metadata={
                    "feature_set": config["feature_set"].value.upper(),
                    "samples": int(len(scoped_df)),
                    "trained_at": datetime.now(timezone.utc).isoformat(),
                    "auto_retrain": True,
                    "trigger_market": target.market,
                    "trigger_brier": float(target.rolling_brier),
                },
                mode="production",
            )
        except Exception as exc:
            capture_exception(
                exc,
                context={
                    "model": target.model_name,
                    "league": target.league,
                    "market": target.market,
                    "pipeline_step": "auto_retrain_train_model",
                },
            )
            notifier.send_alert(
                "Auto-retrain failed to execute",
                context={
                    "model": target.model_name,
                    "league": target.league,
                    "market": target.market,
                    "error": str(exc),
                },
                severity="WARNING",
            )
            _capture_auto_retrain_failure(
                target,
                reason="training_failed",
                error=str(exc),
            )
            _log_auto_retrain_event(
                registry,
                event_type="auto_retrain_train_failed",
                model_name=target.model_name,
                league=target.league,
                version=target.version,
                brier_score=target.rolling_brier,
                train_size=int(len(scoped_df)),
            )
            outcomes.append(
                AutoRetrainOutcome(
                    target=target,
                    before_key=before_key,
                    after_key=None,
                    promoted=False,
                    old_brier=old_brier,
                    new_brier=float("inf"),
                    reason="training_failed",
                )
            )
            continue

        after_key = _latest_productive_key(registry, target.model_name, target.league)
        new_meta = registry.manifest.get(after_key) if after_key else None
        new_brier = _extract_brier(new_meta if isinstance(new_meta, dict) else None)
        validation_ok = (
            old_brier != float("inf")
            and new_brier != float("inf")
            and new_brier <= old_brier * AUTO_RETRAIN_PROMOTION_MULTIPLIER
        )

        promoted = False
        reason = "validation_failed"
        if validation_ok and after_key:
            registry.set_active_model(
                model_type=target.model_name,
                exact_manifest_key=after_key,
                league=None if target.league == "Global" else target.league,
            )
            promoted = True
            reason = "promoted"
            _log_auto_retrain_event(
                registry,
                event_type="auto_retrain_promoted",
                model_name=target.model_name,
                league=target.league,
                version=str(cast(Dict[str, Any], new_meta).get("version", target.version)) if isinstance(new_meta, dict) else target.version,
                brier_score=new_brier,
                train_size=int(len(scoped_df)),
            )
        else:
            notifier.send_alert(
                "Auto-retrain failed to improve performance",
                context={
                    "model": target.model_name,
                    "league": target.league,
                    "market": target.market,
                    "rolling_brier": round(float(target.rolling_brier), 5),
                    "baseline_brier": round(float(target.baseline_brier), 5),
                    "old_brier": None if old_brier == float("inf") else round(float(old_brier), 5),
                    "new_brier": None if new_brier == float("inf") else round(float(new_brier), 5),
                    "promotion_threshold": round(float(old_brier * AUTO_RETRAIN_PROMOTION_MULTIPLIER), 5)
                    if old_brier != float("inf")
                    else None,
                },
                severity="WARNING",
            )
            _capture_auto_retrain_failure(
                target,
                reason="validation_failed",
                context={
                    "old_brier": None if old_brier == float("inf") else old_brier,
                    "new_brier": None if new_brier == float("inf") else new_brier,
                },
            )
            _log_auto_retrain_event(
                registry,
                event_type="auto_retrain_rejected",
                model_name=target.model_name,
                league=target.league,
                version=str(cast(Dict[str, Any], new_meta).get("version", target.version)) if isinstance(new_meta, dict) else target.version,
                brier_score=new_brier if new_brier != float("inf") else target.rolling_brier,
                train_size=int(len(scoped_df)),
            )

        outcomes.append(
            AutoRetrainOutcome(
                target=target,
                before_key=before_key,
                after_key=after_key,
                promoted=promoted,
                old_brier=old_brier,
                new_brier=new_brier,
                reason=reason,
            )
        )

    return outcomes


def retrain_stale_models(
    registry: ModelRegistry,
    trainer: ModelTrainer,
    stale_targets: List[StaleModelTarget],
    feature_df: pd.DataFrame,
) -> List[RetrainOutcome]:
    logger.info("Step 3/5: retraining %d stale model scopes", len(stale_targets))
    config_by_name = _config_map()
    outcomes: List[RetrainOutcome] = []

    for target in stale_targets:
        config = config_by_name.get(target.model_name)
        if config is None:
            logger.warning("Skipping unknown model config: %s", target.model_name)
            continue

        league = target.league
        scoped_df = feature_df if league == "Global" else feature_df[feature_df["league"] == league]
        if scoped_df.empty:
            logger.warning("No training rows for %s/%s; skipping.", league, target.model_name)
            continue

        features = select_features(
            scoped_df,
            str(config["target"]),
            feature_set=config["feature_set"],
        )
        params = dict(config.get("params", {}))
        if config["type"] == ModelType.POISSON:
            from src.config.alpha_config import get_alpha

            params["alpha"] = get_alpha(league, target.model_name)

        before_key = _latest_productive_key(registry, target.model_name, league)
        trainer.train_model(
            df=scoped_df,
            target_col=str(config["target"]),
            league=None if league == "Global" else league,
            model_type=config["type"].value,
            model_name=target.model_name,
            features=features,
            params=params,
            extra_metadata={
                "feature_set": config["feature_set"].value.upper(),
                "samples": int(len(scoped_df)),
                "trained_at": datetime.now(timezone.utc).isoformat(),
            },
            mode="production",
        )
        after_key = _latest_productive_key(registry, target.model_name, league)
        old_meta = registry.manifest.get(before_key) if before_key else None
        new_meta = registry.manifest.get(after_key) if after_key else None
        improved = _is_improved(new_meta if isinstance(new_meta, dict) else None, old_meta if isinstance(old_meta, dict) else None)
        outcomes.append(
            RetrainOutcome(
                target=target,
                before_key=before_key,
                after_key=after_key,
                improved=improved,
            )
        )

    return outcomes


def promote_if_improved(registry: ModelRegistry, outcomes: List[RetrainOutcome]) -> int:
    logger.info("Step 4/5: evaluating promotions for retrained models")
    promotions = 0
    for outcome in outcomes:
        if not outcome.after_key:
            continue
        if not outcome.improved:
            logger.info(
                "No promotion (not improved): %s/%s",
                outcome.target.league,
                outcome.target.model_name,
            )
            continue

        registry.set_active_model(
            model_type=outcome.target.model_name,
            exact_manifest_key=outcome.after_key,
            league=None if outcome.target.league == "Global" else outcome.target.league,
        )
        promotions += 1
        logger.info(
            "Promoted improved model: %s/%s -> %s",
            outcome.target.league,
            outcome.target.model_name,
            outcome.after_key,
        )
    return promotions


def generate_performance_report() -> Dict[str, Any]:
    logger.info("Step 5/5: generating performance report")
    return PerformanceTracker().generate_report()


def run_nightly(
    *,
    season: Optional[str] = None,
    force_fetch: bool = False,
    stale_days: int = 7,
) -> Dict[str, Any]:
    _configure_logging()
    init_sentry(component="nightly-pipeline")
    registry = ModelRegistry()
    trainer = ModelTrainer(registry=registry)
    season_token = season or _default_season_token()
    logger.info("Nightly pipeline started | season=%s stale_days=%s", season_token, stale_days)

    fetch_latest_data_wrapped = _run_pipeline_step
    fetch_latest_data_wrapped(
        "fetch_latest_data",
        fetch_latest_data,
        season=season_token,
        force=force_fetch,
    )
    matrix_result = _run_pipeline_step("build_feature_matrix", build_feature_matrix)
    stale_targets = _run_pipeline_step(
        "find_stale_models",
        find_stale_models,
        registry=registry,
        stale_days=stale_days,
    )
    feature_df: Optional[pd.DataFrame] = None

    if stale_targets:
        feature_df = _run_pipeline_step("load_training_dataframe", load_training_dataframe)
        outcomes = _run_pipeline_step(
            "retrain_stale_models",
            retrain_stale_models,
            registry=registry,
            trainer=trainer,
            stale_targets=stale_targets,
            feature_df=feature_df,
        )
    else:
        outcomes = []
        logger.info("No stale models found (> %s days old).", stale_days)

    promotions = (
        _run_pipeline_step(
            "promote_if_improved",
            promote_if_improved,
            registry=registry,
            outcomes=outcomes,
        )
        if outcomes
        else 0
    )
    report = _run_pipeline_step("generate_performance_report", generate_performance_report)
    evals_df = _run_pipeline_step("load_recent_evaluations", load_recent_evaluations)
    auto_targets = _run_pipeline_step(
        "find_auto_retrain_targets",
        find_auto_retrain_targets,
        registry=registry,
        evals_df=evals_df,
    )
    if auto_targets:
        if feature_df is None:
            feature_df = _run_pipeline_step("load_training_dataframe", load_training_dataframe)
        auto_outcomes = _run_pipeline_step(
            "auto_retrain_underperforming_models",
            auto_retrain_underperforming_models,
            registry=registry,
            trainer=trainer,
            feature_df=feature_df,
            evals_df=evals_df,
        )
    else:
        auto_outcomes = []
        logger.info("No auto-retrain triggers detected from rolling-Brier checks.")
    auto_promotions = sum(1 for outcome in auto_outcomes if outcome.promoted)

    summary = {
        "status": "ok",
        "season": season_token,
        "feature_matrix": matrix_result,
        "stale_models": len(stale_targets),
        "retrained": len(outcomes),
        "promoted": promotions,
        "auto_retrain_triggered": len(auto_targets),
        "auto_retrained": len(auto_outcomes),
        "auto_promoted": auto_promotions,
        "report_status": report.get("status", "OK") if isinstance(report, dict) else "OK",
    }
    logger.info("Nightly pipeline completed: %s", summary)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="Run nightly ML pipeline automation.")
    parser.add_argument("--season", default=None, help="Season token in YYZZ format (default: inferred).")
    parser.add_argument("--force-fetch", action="store_true", help="Force data fetch even when files are unchanged.")
    parser.add_argument("--stale-days", type=int, default=7, help="Retrain threshold for stale models.")
    args = parser.parse_args(argv)

    return run_nightly(
        season=args.season,
        force_fetch=bool(args.force_fetch),
        stale_days=int(args.stale_days),
    )


if __name__ == "__main__":
    main()
