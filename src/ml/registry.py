"""
Model Registry and Versioning.

Manages the lifecycle, versioning, and retrieval of machine learning models.
Features:
- Singleton registry with in-memory caching
- Semantic Versioning (SemVer)
- Manifest persistence with backup
- Provenance tracking via SHA-256
- Smart routing for League vs Global model selection
"""
import hashlib
import joblib
import json
import logging
import os
import pickle
import shutil
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import s3fs
import sklearn
from sqlalchemy import delete, select
from sqlalchemy.orm import Session
from sklearn.exceptions import InconsistentVersionWarning

from src.config import MODELS_DIR
from src.config.thresholds import Thresholds
from src.core.exceptions import (
    ConfigurationError,
    DataValidationError,
    ModelNotFoundError,
)
from src.db.connection import database_is_configured, get_engine
from src.db.models import ModelManifestEntry
from src.ml.model_db import ModelHistoryDB

# Define public API
__all__ = ["ModelRegistry", "SemVer"]

logger = logging.getLogger(__name__)

# --- LEAGUE PROMOTION CRITERIA (Rule 2025-12-21) ---
PROMOTION_THRESHOLDS = {
    "poisson_home_base": Thresholds.PROMOTION_GOALS,
    "poisson_away_base": Thresholds.PROMOTION_GOALS,
    "nb_home_corners_base": Thresholds.PROMOTION_CORNERS,
    "nb_away_corners_base": Thresholds.PROMOTION_CORNERS,
    "poisson_total_cards_base": Thresholds.PROMOTION_CARDS
}


class SemVer:
    """Helper for Semantic Versioning comparison."""
    
    def __init__(self, v_str: str) -> None:
        self.raw = v_str
        try:
            # Handle "v1.2.3" or "1.2.3"
            clean = v_str.lstrip('v')
            parts = [int(p) for p in clean.split('.')]
            if len(parts) != 3:
                raise ValueError
            self.major, self.minor, self.patch = parts
            self.valid = True
        except Exception:
            self.major, self.minor, self.patch = 0, 0, 0
            self.valid = False

    def __lt__(self, other: 'SemVer') -> bool:
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SemVer):
            return NotImplemented
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)
        
    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def __repr__(self) -> str:
        return f"SemVer({self})"


class ModelRegistry:
    """
    Manages model versioning and metadata using a JSON manifest.
    
    Singleton Pattern: Manifest is cached in memory across all instances for efficiency.
    """
    MANIFEST_FILE = MODELS_DIR / "manifest.json"
    BACKUP_FILE = MODELS_DIR / "manifest.json.bak"
    HISTORY_DB = ModelHistoryDB
    
    _instance: Optional['ModelRegistry'] = None
    _manifest_cache: Optional[Dict[str, Any]] = None

    def __new__(cls) -> 'ModelRegistry':
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._manifest_cache is None:
            self._load_manifest()

    @property
    def manifest(self) -> Dict[str, Any]:
        if self.__class__._manifest_cache is None:
             self._load_manifest()

        cache = self.__class__._manifest_cache
        if cache is None:
            cache = {}
            self.__class__._manifest_cache = cache
        return cache
    
    @manifest.setter
    def manifest(self, value: Dict[str, Any]) -> None:
        self.__class__._manifest_cache = value

    def reload(self) -> None:
        """Forces a manifest reload from disk."""
        self.__class__._manifest_cache = None
        self._load_manifest()

    def _load_manifest(self) -> None:
        if self._manifest_cache is not None:
            return

        if database_is_configured():
            db_manifest = self._load_manifest_from_db()
            if db_manifest is not None:
                self.manifest = db_manifest
                self._save_manifest_file()
                return

        file_manifest = self._load_manifest_from_file()
        if file_manifest is not None:
            self.manifest = file_manifest
            return

        logger.warning("No valid manifest found or recovery failed. Initializing empty Registry.")
        self.manifest = {}

    def _save_manifest(self) -> None:
        self._save_manifest_file()
        self._save_manifest_to_db()

    def _load_manifest_from_file(self) -> Optional[Dict[str, Any]]:
        if self.MANIFEST_FILE.exists():
            try:
                with open(self.MANIFEST_FILE, "r", encoding="utf-8") as f:
                    return cast(Dict[str, Any], json.load(f))
            except Exception as e:
                logger.error(f"Failed to load primary manifest: {e}. Attempting backup recovery...")

        if self.BACKUP_FILE.exists():
            try:
                with open(self.BACKUP_FILE, "r", encoding="utf-8") as f:
                    manifest = cast(Dict[str, Any], json.load(f))
                logger.info("Successfully recovered manifest from backup.")
                return manifest
            except Exception as e:
                logger.error(f"Failed to load backup manifest: {e}")

        return None

    def _save_manifest_file(self) -> None:
        # ensure dir exists
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # Save primary
        with open(self.MANIFEST_FILE, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2)

        # Create backup
        self._save_backup()

    def _save_backup(self) -> None:
        try:
            shutil.copy2(self.MANIFEST_FILE, self.BACKUP_FILE)
        except Exception as e:
            logger.error(f"Failed to create manifest backup: {e}")

    @staticmethod
    def _parse_optional_datetime(value: Any) -> Optional[datetime]:
        if value in {None, ""}:
            return None
        if isinstance(value, datetime):
            return value
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except Exception:
            return None

    @staticmethod
    def _json_blob(value: Any) -> Optional[str]:
        if value is None:
            return None
        try:
            return json.dumps(value, default=str, sort_keys=True)
        except TypeError:
            return json.dumps(str(value))

    def _save_manifest_to_db(self) -> None:
        if not database_is_configured():
            return

        try:
            with Session(get_engine()) as session:
                session.execute(delete(ModelManifestEntry))
                for manifest_key, meta in self.manifest.items():
                    payload: Any = meta if isinstance(meta, dict) else {"__manifest_value__": meta}
                    session.add(
                        ModelManifestEntry(
                            manifest_key=str(manifest_key),
                            model_name=payload.get("name") if isinstance(payload, dict) else None,
                            version=payload.get("version") if isinstance(payload, dict) else None,
                            league=payload.get("league") if isinstance(payload, dict) else None,
                            model_type=payload.get("type") if isinstance(payload, dict) else None,
                            target=payload.get("target") if isinstance(payload, dict) else None,
                            filename=payload.get("filename") if isinstance(payload, dict) else None,
                            mode=payload.get("mode") if isinstance(payload, dict) else None,
                            status=payload.get("status") if isinstance(payload, dict) else None,
                            sklearn_version=payload.get("sklearn_version") if isinstance(payload, dict) else None,
                            train_size=payload.get("train_size") if isinstance(payload, dict) else None,
                            test_size=payload.get("test_size") if isinstance(payload, dict) else None,
                            registered_at=self._parse_optional_datetime(
                                payload.get("registered_at") if isinstance(payload, dict) else None
                            ),
                            features_json=self._json_blob(
                                payload.get("features") if isinstance(payload, dict) else None
                            ),
                            params_json=self._json_blob(
                                payload.get("params") if isinstance(payload, dict) else None
                            ),
                            metrics_json=self._json_blob(
                                payload.get("metrics") if isinstance(payload, dict) else None
                            ),
                            metadata_json=self._json_blob(payload) or "{}",
                        )
                    )
                session.commit()
        except Exception as exc:
            logger.warning("Failed to persist manifest to database: %s", exc)

    def _load_manifest_from_db(self) -> Optional[Dict[str, Any]]:
        try:
            with Session(get_engine()) as session:
                rows = session.execute(
                    select(ModelManifestEntry).order_by(ModelManifestEntry.manifest_key.asc())
                ).scalars().all()
        except Exception as exc:
            logger.warning("Failed to load manifest from database: %s", exc)
            return None

        if not rows:
            return None

        manifest: Dict[str, Any] = {}
        for row in rows:
            payload: Any
            try:
                payload = json.loads(row.metadata_json) if row.metadata_json else {}
            except Exception:
                payload = {}

            if isinstance(payload, dict) and "__manifest_value__" in payload:
                manifest[row.manifest_key] = payload["__manifest_value__"]
            else:
                manifest[row.manifest_key] = payload if isinstance(payload, dict) else {}
        return manifest

    @staticmethod
    def _require_aws_s3_env() -> Dict[str, str]:
        required = {
            "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID"),
            "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY"),
            "AWS_REGION": os.environ.get("AWS_REGION"),
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            missing_list = ", ".join(missing)
            raise EnvironmentError(
                f"Missing required AWS S3 environment variables: {missing_list}"
            )
        return {name: str(value) for name, value in required.items()}

    def _manifest_model_filenames(self, *, pkl_only: bool = False) -> List[str]:
        filenames: set[str] = set()
        for meta in self.manifest.values():
            if not isinstance(meta, dict):
                continue
            filename = meta.get("filename")
            if not isinstance(filename, str):
                continue
            normalized = filename.strip()
            if not normalized:
                continue
            if pkl_only and not normalized.lower().endswith(".pkl"):
                continue
            filenames.add(normalized)
        return sorted(filenames)

    @staticmethod
    def _s3_object_path(bucket: str, prefix: str, filename: str) -> str:
        normalized_bucket = str(bucket).strip().strip("/")
        normalized_prefix = str(prefix).strip().strip("/")
        normalized_filename = str(filename).strip().replace("\\", "/").lstrip("/")
        if normalized_prefix:
            return f"{normalized_bucket}/{normalized_prefix}/{normalized_filename}"
        return f"{normalized_bucket}/{normalized_filename}"

    @classmethod
    def _s3_object_uri(cls, bucket: str, prefix: str, filename: str) -> str:
        return f"s3://{cls._s3_object_path(bucket, prefix, filename)}"

    def _build_s3_filesystem(self) -> s3fs.S3FileSystem:
        env = self._require_aws_s3_env()
        return s3fs.S3FileSystem(
            key=env["AWS_ACCESS_KEY_ID"],
            secret=env["AWS_SECRET_ACCESS_KEY"],
            client_kwargs={"region_name": env["AWS_REGION"]},
        )

    def push_to_s3(self, bucket: str, prefix: str) -> int:
        if not str(bucket).strip():
            raise ValueError("bucket must be non-empty")

        fs = self._build_s3_filesystem()
        uploaded = 0
        for filename in self._manifest_model_filenames(pkl_only=True):
            local_path = MODELS_DIR / filename
            if not local_path.exists():
                raise FileNotFoundError(f"Model artifact listed in manifest is missing locally: {local_path}")
            fs.put(str(local_path), self._s3_object_path(bucket, prefix, filename))
            uploaded += 1

        logger.info(
            "Uploaded %s model artifacts to %s",
            uploaded,
            f"s3://{str(bucket).strip().strip('/')}/{str(prefix).strip().strip('/')}".rstrip("/"),
        )
        return uploaded

    def pull_from_s3(self, bucket: str, prefix: str) -> int:
        if not str(bucket).strip():
            raise ValueError("bucket must be non-empty")

        fs = self._build_s3_filesystem()
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        downloaded = 0
        for filename in self._manifest_model_filenames():
            local_path = MODELS_DIR / filename
            if local_path.exists():
                continue

            remote_path = self._s3_object_path(bucket, prefix, filename)
            if not fs.exists(remote_path):
                raise FileNotFoundError(
                    f"Model artifact listed in manifest is missing in S3: {self._s3_object_uri(bucket, prefix, filename)}"
                )

            local_path.parent.mkdir(parents=True, exist_ok=True)
            fs.get(remote_path, str(local_path))
            downloaded += 1

        logger.info(
            "Downloaded %s model artifacts from %s",
            downloaded,
            f"s3://{str(bucket).strip().strip('/')}/{str(prefix).strip().strip('/')}".rstrip("/"),
        )
        return downloaded

    def _write_lifecycle_event(self, event_type: str, metadata: Dict[str, Any]) -> None:
        """Best-effort write to SQLite lifecycle history."""
        if not isinstance(metadata, dict):
            return
        try:
            self.HISTORY_DB().write_event_from_metadata(metadata, event_type=event_type)
        except Exception as exc:
            logger.warning("Failed to write model lifecycle event (%s): %s", event_type, exc)

    def register_model(self, name: str, version: str, metadata: Dict[str, Any]) -> None:
        """Helper to register a new model version."""
        league_suffix = f"_{metadata['league']}" if 'league' in metadata else ""
        key = f"{name}_v{version}{league_suffix}"
        metadata["registered_at"] = datetime.now().isoformat()
        metadata["name"] = name
        metadata["version"] = version
        metadata.setdefault("sklearn_version", sklearn.__version__)
        
        self.manifest[key] = metadata
        
        # Initialize active_models and shadow_models dicts if not present
        if "active_models" not in self.manifest:
            self.manifest["active_models"] = {}
        if "shadow_models" not in self.manifest:
            self.manifest["shadow_models"] = {}
             
        self._save_manifest()
        self._write_lifecycle_event("train", metadata)
        logger.info(f"Registered model: {key}")

    def set_active_model(self, model_type: str, exact_manifest_key: str, league: Optional[str] = None) -> None:
        """Explicitly pin an active model pointer to prevent auto-upgrades."""
        if "active_models" not in self.manifest:
            self.manifest["active_models"] = {}
            
        pointer_key = f"{model_type}_{league}" if league else model_type
        if exact_manifest_key not in self.manifest:
            raise ValueError(f"Cannot set active model: Key '{exact_manifest_key}' not found in manifest.")
             
        self.manifest["active_models"][pointer_key] = exact_manifest_key
        self._save_manifest()
        selected = self.manifest.get(exact_manifest_key)
        if isinstance(selected, dict):
            self._write_lifecycle_event("promote", selected)
        logger.info(f"Active model pointer pinned: {pointer_key} -> {exact_manifest_key}")

    def rollback_active_model(self, model_type: str, league: Optional[str] = None) -> Optional[str]:
        """
        Roll back active pointer to the previous productive version for a scope.

        Returns:
            Manifest key selected as rollback target, or None if unavailable.
        """
        if "active_models" not in self.manifest:
            self.manifest["active_models"] = {}

        pointer_key = f"{model_type}_{league}" if league else model_type
        current_key = self.manifest["active_models"].get(pointer_key)
        if not current_key:
            logger.warning("Rollback skipped: no active pointer set for %s", pointer_key)
            return None

        scope_league = league if league is not None else "Global"
        candidates: List[tuple[str, Dict[str, Any]]] = []
        for key, meta in self.manifest.items():
            if not isinstance(meta, dict):
                continue
            if meta.get("name") != model_type:
                continue
            if meta.get("status") != "productive":
                continue
            if str(meta.get("league") or "Global") != scope_league:
                continue
            candidates.append((str(key), meta))

        candidates.sort(key=lambda item: item[1].get("registered_at", ""), reverse=True)
        rollback_key = next((key for key, _ in candidates if key != current_key), None)
        if not rollback_key:
            logger.warning("Rollback skipped: no previous productive version for %s", pointer_key)
            return None

        self.manifest["active_models"][pointer_key] = rollback_key
        self._save_manifest()
        selected = self.manifest.get(rollback_key)
        if isinstance(selected, dict):
            self._write_lifecycle_event("rollback", selected)

        logger.info("Active model rollback: %s -> %s", pointer_key, rollback_key)
        return rollback_key

    def set_shadow_model(self, model_type: str, exact_manifest_key: str, league: Optional[str] = None) -> None:
        """Register a shadow model for parallel evaluation (no betting)."""
        if "shadow_models" not in self.manifest:
            self.manifest["shadow_models"] = {}

        pointer_key = f"{model_type}_{league}" if league else model_type
        if exact_manifest_key not in self.manifest:
            raise ValueError(f"Cannot set shadow model: Key '{exact_manifest_key}' not found in manifest.")

        # Shadow models are stored as a list to allow multiple parallel runners
        if pointer_key not in self.manifest["shadow_models"]:
            self.manifest["shadow_models"][pointer_key] = []

        if exact_manifest_key not in self.manifest["shadow_models"][pointer_key]:
            self.manifest["shadow_models"][pointer_key].append(exact_manifest_key)

        self._save_manifest()
        logger.info(f"Shadow model registered: {pointer_key} -> {exact_manifest_key}")

    def get_shadow_models(self, model_type: str, league: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve all shadow models for a given model type, for parallel evaluation."""
        shadow_pointers = self.manifest.get("shadow_models", {})
        pointer_key = f"{model_type}_{league}" if league else model_type
        keys = shadow_pointers.get(pointer_key, [])

        results = []
        for k in keys:
            meta = self.manifest.get(k)
            if meta and isinstance(meta, dict):
                results.append(meta)
            else:
                logger.warning(f"Shadow model key '{k}' not found in manifest.")
        return results

    def update_model_metrics(self, name: str, version: str, metrics: Dict[str, float]) -> None:
        """Updates metrics (e.g., calibration_score) for an existing model."""
        key = f"{name}_v{version}"
        if key in self.manifest:
            if 'metrics' not in self.manifest[key]:
                self.manifest[key]['metrics'] = {}
            self.manifest[key]['metrics'].update(metrics)
            self._save_manifest()
            logger.info(f"Updated metrics for {key}: {metrics}")
        else:
            logger.warning(f"Cannot update metrics: Model {key} not found.")

    def get_next_version(self, name: str, league: Optional[str] = None, increment: str = "minor") -> str:
        """Auto-increments version (MAJOR.MINOR.PATCH)."""
        # Find all versions for this model/league combo
        existing_versions = []
        for meta in self.manifest.values():
            if type(meta) is dict and meta.get('name') == name and meta.get('league') == league:
                v = meta.get('version', '0.0.0')
                existing_versions.append(SemVer(v))
        
        # Filter for valid SemVers
        valid_vers = [v for v in existing_versions if v.valid]
        
        if not valid_vers:
            return "1.0.0"
            
        latest = sorted(valid_vers)[-1]
        
        maj, min, pat = latest.major, latest.minor, latest.patch
        
        if increment == 'major':
            maj += 1; min = 0; pat = 0
        elif increment == 'minor':
            min += 1; pat = 0
        else: # patch
            pat += 1
            
        return f"{maj}.{min}.{pat}"

    def get_model_metadata(self, name: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieves metadata for a specific model version or the latest one."""
        if version:
            # Try specific key lookup first (legacy format)
            key = f"{name}_v{version}"
            if key in self.manifest: return self.manifest[key]
            
            # Search by version field
            for meta in self.manifest.values():
                if type(meta) is dict and meta.get('name') == name and meta.get('version') == version:
                    return meta
            return None
        
        # Find latest version for 'name' using SemVer sort
        relevant = [meta for meta in self.manifest.values() if type(meta) is dict and meta.get('name') == name]
        if not relevant:
            return None
            
        # Sort by registered_at first as baseline
        relevant.sort(key=lambda x: x.get('registered_at', ''), reverse=False)
        
        # Then try strict SemVer sort
        try:
             relevant.sort(key=lambda x: SemVer(x.get('version', '0')), reverse=False)
        except Exception:
             pass
             
        return relevant[-1]

    def get_best_model(self, market: str) -> Optional[Dict[str, Any]]:
        """
        Finds the model with the best metric for a given market.
        Uses lowest Brier score when available (lower is better).
        """
        candidates: List[Dict[str, Any]] = []
        for meta in self.manifest.values():
            if isinstance(meta, dict) and meta.get("name") == market:
                candidates.append(meta)

        if not candidates:
            return None

        def _extract_brier(model_meta: Dict[str, Any]) -> float:
            metrics = model_meta.get("metrics", {})
            if isinstance(metrics, dict):
                val = metrics.get("brier_score")
                if isinstance(val, (int, float)):
                    return float(val)
            direct = model_meta.get("brier_score")
            if isinstance(direct, (int, float)):
                return float(direct)
            return float("inf")

        # Lowest Brier wins; tie-break by latest registration.
        candidates.sort(
            key=lambda x: (_extract_brier(x), x.get("registered_at", "")),
            reverse=False,
        )

        best = candidates[0]
        if _extract_brier(best) == float("inf"):
            # No Brier metadata available: preserve legacy behavior.
            return self.get_model_metadata(market)

        return best

    def get_production_model(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the latest PRODUCTION model.
        Hard rejects 'provisional' or 'debug' models.
        """
        meta = self.get_model_metadata(name)
        if not meta:
            return None
        
        if meta.get('status') != 'productive':
            logger.warning(f"Model {name} found but status is '{meta.get('status')}'. REJECTING.")
            return None

        return cast(Dict[str, Any], meta)

    def get_production_model_for_league(
        self, 
        league: str, 
        model_type: str = "poisson_home_base"
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieves best model optimized for a specific league.
        Balances Local Specificity vs Global Robustness.
        Respects 'active_models' pointers to prevent silent auto-upgrades.
        """
        active_pointers = self.manifest.get("active_models", {})
        
        # 1. Try explicitly pinned League dict pointer
        league_ptr = active_pointers.get(f"{model_type}_{league}")
        if league_ptr and league_ptr in self.manifest:
            return self.manifest[league_ptr]
            
        # 2. Try explicitly pinned Global dict pointer
        global_ptr = active_pointers.get(model_type)
        if global_ptr and global_ptr in self.manifest:
            return self.manifest[global_ptr]

        local_candidates, global_candidates = self._find_model_candidates(league, model_type)
        
        best_local = local_candidates[0] if local_candidates else None
        best_global = global_candidates[0] if global_candidates else None
        
        if not best_local:
            return best_global
            
        if not best_global:
            return best_local
            
        # Compare Scores (Smart Routing)
        return self._select_best_model(best_local, best_global, model_type, league)

    def _find_model_candidates(self, league: str, model_type: str) -> tuple[List[Dict], List[Dict]]:
        """Finds valid Production model candidates for Local and Global scopes."""
        local_candidates = []
        global_candidates = []
        
        for key, meta in self.manifest.items():
            if type(meta) is dict and meta.get('name') == model_type and meta.get('status') == 'productive':
                # SELF-HEALING: Verify file exists
                filename = meta.get('filename')
                if not filename:
                    continue
                model_path = MODELS_DIR / filename
                if not model_path.exists():
                     # Logger pruned for noise reduction
                    continue

                m_lg = meta.get('league')
                if m_lg == league:
                    local_candidates.append(meta)
                elif m_lg in ['Global', None]:
                    global_candidates.append(meta)
        
        # Sort by freshness
        local_candidates.sort(key=lambda x: x.get('registered_at', ''), reverse=True)
        global_candidates.sort(key=lambda x: x.get('registered_at', ''), reverse=True)
        
        return local_candidates, global_candidates

    def _select_best_model(
        self, 
        local_meta: Dict, 
        global_meta: Dict, 
        model_type: str, 
        league: str
    ) -> Dict[str, Any]:
        """Compares Local vs Global model based on Score and Training Size."""
        score_key = 'calibration_score' 
        
        local_score = local_meta.get('metrics', {}).get(score_key, 999.0)
        global_score = global_meta.get('metrics', {}).get(score_key, 999.0)
        self._calibrate_promotion_threshold(league, model_type, local_score, global_score)
        
        # Rule of Specificity:
        # If a local model has passed the Promotion Gate (sufficient data),
        # we trust it over the Global model to capture league-specific nuance,
        # unless it is catastrophically worse (sanity check).
        
        threshold = self._get_promotion_threshold(model_type, league, fallback=1000)
        local_n = local_meta.get('train_size', 0) + local_meta.get('test_size', 0)
        
        if local_n >= threshold:
            # Promoted local model must remain within 1.5x global ECE to win routing.
            promoted_guardrail = global_score * 1.5
            if local_score <= promoted_guardrail:
                logger.debug(
                    "Smart Routing: Selected PROMOTED local model for %s "
                    "(n=%s, local_ece=%.4f, global_ece=%.4f, guardrail=%.4f)",
                    league,
                    local_n,
                    local_score,
                    global_score,
                    promoted_guardrail,
                )
                return local_meta

            logged_smart_routing = getattr(self, "_logged_smart_routing", None)
            if logged_smart_routing is None or league not in logged_smart_routing:
                logger.warning(
                    "Smart Routing: Promoted local model overridden by Global for %s "
                    "(n=%s, local_ece=%.4f > guardrail=%.4f from global_ece=%.4f)",
                    league,
                    local_n,
                    local_score,
                    promoted_guardrail,
                    global_score,
                )
                if logged_smart_routing is not None:
                    logged_smart_routing.add(league)
            return global_meta
            
        # Fallback: If local is under-trained, compare scores
        if local_score < global_score:
             return local_meta
             
        return global_meta

    @staticmethod
    def _league_threshold_key(model_type: str, league: str) -> str:
        return f"{model_type}::{league}"

    @staticmethod
    def _clamp_promotion_threshold(value: int) -> int:
        return max(50, min(500, int(value)))

    def _get_promotion_threshold(
        self,
        model_type: str,
        league: Optional[str] = None,
        fallback: int = Thresholds.MIN_TRAINING_SAMPLES,
    ) -> int:
        if league:
            league_key = self._league_threshold_key(model_type, league)
            league_threshold = PROMOTION_THRESHOLDS.get(league_key)
            if isinstance(league_threshold, (int, float)):
                return self._clamp_promotion_threshold(int(round(league_threshold)))

        base = PROMOTION_THRESHOLDS.get(model_type, fallback)
        if not isinstance(base, (int, float)):
            base = fallback
        return self._clamp_promotion_threshold(int(round(base)))

    def _calibrate_promotion_threshold(
        self,
        league: str,
        model_type: str,
        local_score: float,
        global_score: float,
    ) -> None:
        """
        Auto-tune league promotion gate based on local vs global ECE deltas.

        Rules:
        - local beats global by > 0.02 ECE -> lower threshold by 10%
        - global beats local -> raise threshold by 10%
        - threshold is clamped to [50, 500]
        """
        if not isinstance(local_score, (int, float)) or not isinstance(global_score, (int, float)):
            return

        current = self._get_promotion_threshold(model_type, league)
        gap = float(global_score) - float(local_score)  # Positive means local better.

        if gap > 0.02:
            updated = self._clamp_promotion_threshold(int(round(current * 0.90)))
            reason = "local_better_by_gt_0.02_ece"
        elif gap < 0.0:
            updated = self._clamp_promotion_threshold(int(round(current * 1.10)))
            reason = "global_better"
        else:
            return

        if updated == current:
            return

        threshold_key = self._league_threshold_key(model_type, league)
        PROMOTION_THRESHOLDS[threshold_key] = updated
        logger.info(
            "Promotion threshold auto-calibrated | league=%s model=%s reason=%s threshold=%s->%s",
            league,
            model_type,
            reason,
            current,
            updated,
        )

    def get_fallback_reason(self, league: str, model_type: str) -> Optional[str]:
        """
        Explains WHY a league is using Global fallback for a specific market.
        Used for explicit logging at prediction time.
        """
        local_cands, global_cands = self._find_model_candidates(league, model_type)
        
        if not local_cands:
            return "No league-specific production model exists"
            
        best_local = local_cands[0]
        local_n = best_local.get('train_size', 0) + best_local.get('test_size', 0)
        threshold = self._get_promotion_threshold(
            model_type,
            league,
            fallback=Thresholds.MIN_TRAINING_SAMPLES,
        )
        
        if local_n < threshold:
            return f"Under-trained (n={local_n} < {threshold})"
            
        if global_cands:
            best_global = global_cands[0]
            l_score = best_local.get('metrics', {}).get('calibration_score', 999.0)
            g_score = best_global.get('metrics', {}).get('calibration_score', 999.0)
            
            if l_score >= g_score:
                return f"Sub-optimal (League Score {l_score:.4f} >= Global {g_score:.4f})"
                
        return None

    def get_coverage_status(self, league: str) -> str:
        """
        Classifies league coverage: GLOBAL_ONLY, PARTIAL, FULL.
        FULL requires Goals and Corners to be league-specific.
        """
        goals_h = self.get_production_model_for_league(league, "poisson_home_base")
        goals_a = self.get_production_model_for_league(league, "poisson_away_base")
        corn_h = self.get_production_model_for_league(league, "nb_home_corners_base")
        corn_a = self.get_production_model_for_league(league, "nb_away_corners_base")
        
        def is_local(m: Optional[Dict]) -> bool: 
            return m is not None and m.get('league') == league
        
        has_goals = is_local(goals_h) and is_local(goals_a)
        has_corners = is_local(corn_h) and is_local(corn_a)
        
        if has_goals and has_corners:
            return "FULL"
        elif has_goals or has_corners:
            return "PARTIAL"
        else:
            return "GLOBAL_ONLY"

    def load_models_for_market(
        self,
        market: str,
        model_names: List[str],
        league: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Load multiple required models for a market in one call.

        Raises ModelNotFoundError if any required model cannot be resolved.
        """
        loaded: Dict[str, Any] = {}
        missing: List[str] = []

        for model_name in model_names:
            try:
                loaded[model_name] = self.load_model(model_name, league=league)
            except ModelNotFoundError:
                missing.append(model_name)

        if missing:
            raise ModelNotFoundError(
                f"Missing required models for market '{market}': {missing}",
                context={"market": market, "league": league, "missing_models": missing},
            )

        return loaded

    def load_model(self, name: str, league: Optional[str] = None) -> Any:
        """
        Loads the actual model object from disk.
        If 'league' is provided, tries to find league-specific version first.
        
        Strict Resolution Logic:
        1. Attempt League Specific (if league provided)
        2. Fallback to Global (if allowed) w/ "LEAGUE_FALLBACK" warning
        3. Hard Fail (Raise ModelNotFoundError) if no model found
        """
        meta = None
        
        # 1. League Specific Attempt
        if league and league != "Global":
            meta = self.get_production_model_for_league(league, name)
            if not meta:
                # 2. Global Fallback
                logger.warning(
                    f"LEAGUE_FALLBACK: No production model for '{name}' in '{league}'. Attempting Global fallback."
                )
                meta = self.get_production_model_for_league("Global", name)
                
                if meta:
                    logger.info(f"Fallback successful: Using Global model for {name} (League: {league})")
        else:
            # Direct Global/Generic request
            meta = self.get_production_model(name)
            
        # 3. Hard Fail
        if not meta:
            logger.error(f"MODEL_MISSING: Resolution failed for '{name}' (Target: {league or 'Global'})")
            raise ModelNotFoundError(
                f"Resolution failed for model '{name}' (League: {league or 'Global'}). "
                "Ensure model is trained and promoted.",
                context={"model_name": name, "league": league}
            )
            
        path = MODELS_DIR / meta['filename']
        if not path.exists():
            raise ModelNotFoundError(
                f"Model artifact missing at {path}",
                context={"filename": meta['filename'], "model_name": name, "league": league}
            )
            
        try:
            expected_sklearn = meta.get("sklearn_version")
            if expected_sklearn and expected_sklearn != sklearn.__version__:
                raise ConfigurationError(
                    "Model trained with a different sklearn version. Retrain or install the matching runtime.",
                    context={
                        "model_name": name,
                        "league": league,
                        "expected_sklearn": expected_sklearn,
                        "runtime_sklearn": sklearn.__version__,
                    },
                )

            with warnings.catch_warnings(record=True) as captured_warnings:
                warnings.simplefilter("always")
                if path.suffix.lower() == ".joblib":
                    model = joblib.load(path)
                else:
                    with open(path, 'rb') as f:
                        model = pickle.load(f)

            for warning in captured_warnings:
                if issubclass(warning.category, InconsistentVersionWarning):
                    raise ConfigurationError(
                        "Model trained with a different sklearn version. Retrain or install the matching runtime.",
                        context={
                            "model_name": name,
                            "league": league,
                            "runtime_sklearn": sklearn.__version__,
                            "warning": str(warning.message),
                        },
                    )
             
            # Inject metadata for runtime decisions (e.g. Offsets)
            if hasattr(model, 'meta'):
                model.meta = meta
            else:
                # Dynamically attach if not present
                setattr(model, 'meta', meta)
                
            return model
            
        except (pickle.UnpicklingError, EOFError) as e:
            raise DataValidationError(
                f"Corrupted model file at {path}: {e}",
                context={"filename": str(path)}
            )
        except ConfigurationError:
            raise
        except Exception as e:
             raise DataValidationError(
                f"Failed to load model at {path}: {e}",
                context={"filename": str(path)}
             )

    def get_model_hash(self, filename: str) -> str:
        """Generates a SHA-256 hash for a model file for provenance tracking."""
        path = MODELS_DIR / filename
        if not path.exists():
            return "FILE_MISSING"
        
        sha256_hash = hashlib.sha256()
        with open(path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
