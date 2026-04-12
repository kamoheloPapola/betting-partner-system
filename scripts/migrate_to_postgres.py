from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, TypeVar

import pandas as pd
from sqlalchemy import inspect
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.db.connection import database_is_configured, get_engine
from src.db.models import DriftEvent, ModelManifestEntry, ResolvedPrediction
from src.evaluation.resolve_results import AuthoritativeResolver
from src.ml.registry import ModelRegistry
from src.monitoring.drift_orchestrator import DriftOrchestrator

T = TypeVar("T")


def _parse_datetime(value: Any) -> Optional[datetime]:
    if value in {None, ""}:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        parsed = pd.to_datetime(value, errors="coerce", utc=True)
        if pd.isna(parsed):
            return None
        return parsed.to_pydatetime()


def _to_float(value: Any) -> Optional[float]:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _json_blob(value: Any) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(value, default=str, sort_keys=True)


def _make_event_id(*parts: Any) -> str:
    raw = "|".join("" if part is None else str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _chunked(lst: list[T], size: int) -> Iterator[list[T]]:
    for start in range(0, len(lst), size):
        yield lst[start : start + size]


def _require_database_ready() -> str:
    if not database_is_configured():
        raise RuntimeError("DATABASE_URL is not set. Configure PostgreSQL before running this migration.")

    engine = get_engine()
    table_names = set(inspect(engine).get_table_names())
    required = {"resolved_predictions", "model_manifest_entries", "drift_events"}
    missing = sorted(required - table_names)
    if missing:
        raise RuntimeError(
            "Database schema is incomplete. Run 'python -m src.cli init-db' first. "
            f"Missing tables: {', '.join(missing)}"
        )

    return engine.url.render_as_string(hide_password=True)


def _migrate_global_drift_state() -> int:
    status_path = DriftOrchestrator.DEFAULT_STATUS_FILE
    if not status_path.exists():
        print(f"[skip] Global drift state not found: {status_path}")
        return 0

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    evaluated_at = payload.get("evaluated_at")
    status = payload.get("status")

    with Session(get_engine()) as session:
        session.merge(
            DriftEvent(
                event_id=_make_event_id(
                    DriftOrchestrator.GLOBAL_STATE_EVENT_TYPE,
                    evaluated_at,
                    status,
                ),
                event_type=DriftOrchestrator.GLOBAL_STATE_EVENT_TYPE,
                league="GLOBAL",
                market=None,
                severity=str(status) if status is not None else None,
                metric="global_status",
                value=None,
                threshold=None,
                detected_at=_parse_datetime(evaluated_at),
                payload_json=_json_blob(payload),
            )
        )
        session.commit()

    print(f"[ok] Migrated global drift state from {status_path}")
    return 1


def _alert_source_paths() -> Iterable[Path]:
    primary = DriftOrchestrator.DEFAULT_ALERTS_FILE
    backup = primary.with_suffix(primary.suffix + ".bak")

    if primary.exists() and primary.stat().st_size > 0:
        yield primary
        return

    if backup.exists() and backup.stat().st_size > 0:
        yield backup
        return

    if primary.exists():
        yield primary


def _migrate_drift_alerts() -> int:
    migrated = 0
    source_paths = list(_alert_source_paths())
    if not source_paths:
        print(f"[skip] Drift alerts log not found: {DriftOrchestrator.DEFAULT_ALERTS_FILE}")
        return 0

    with Session(get_engine()) as session:
        for source_path in source_paths:
            if not source_path.exists() or source_path.stat().st_size == 0:
                print(f"[skip] Drift alerts source is empty: {source_path}")
                continue

            with open(source_path, "r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for line_number, row in enumerate(reader, start=2):
                    payload = {
                        "source_path": str(source_path),
                        "source_line": line_number,
                        "raw": row,
                    }
                    session.merge(
                        DriftEvent(
                            event_id=_make_event_id(
                                source_path,
                                line_number,
                                row.get("type"),
                                row.get("league"),
                                row.get("market"),
                                row.get("detected_at"),
                            ),
                            event_type=str(row.get("type") or ""),
                            league=str(row.get("league") or "") or None,
                            market=str(row.get("market") or "") or None,
                            severity=str(row.get("severity") or "") or None,
                            metric=str(row.get("metric") or "") or None,
                            value=_to_float(row.get("value")),
                            threshold=_to_float(row.get("threshold")),
                            detected_at=_parse_datetime(row.get("detected_at")),
                            payload_json=_json_blob(payload),
                        )
                    )
                    migrated += 1
        session.commit()

    joined = ", ".join(str(path) for path in source_paths)
    print(f"[ok] Migrated {migrated} drift alert rows from {joined}")
    return migrated


def _migrate_manifest() -> int:
    manifest_path = ModelRegistry.MANIFEST_FILE
    if not manifest_path.exists():
        print(f"[skip] Manifest not found: {manifest_path}")
        return 0

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Manifest payload is not a JSON object: {manifest_path}")

    rows: list[dict[str, Any]] = []
    for manifest_key, meta in payload.items():
        record: Any = meta if isinstance(meta, dict) else {"__manifest_value__": meta}
        rows.append(
            {
                "manifest_key": str(manifest_key),
                "model_name": record.get("name") if isinstance(record, dict) else None,
                "version": record.get("version") if isinstance(record, dict) else None,
                "league": record.get("league") if isinstance(record, dict) else None,
                "model_type": record.get("type") if isinstance(record, dict) else None,
                "target": record.get("target") if isinstance(record, dict) else None,
                "filename": record.get("filename") if isinstance(record, dict) else None,
                "mode": record.get("mode") if isinstance(record, dict) else None,
                "status": record.get("status") if isinstance(record, dict) else None,
                "sklearn_version": record.get("sklearn_version") if isinstance(record, dict) else None,
                "train_size": record.get("train_size") if isinstance(record, dict) else None,
                "test_size": record.get("test_size") if isinstance(record, dict) else None,
                "registered_at": _parse_datetime(
                    record.get("registered_at") if isinstance(record, dict) else None
                ),
                "features_json": _json_blob(record.get("features") if isinstance(record, dict) else None),
                "params_json": _json_blob(record.get("params") if isinstance(record, dict) else None),
                "metrics_json": _json_blob(record.get("metrics") if isinstance(record, dict) else None),
                "metadata_json": _json_blob(record) or "{}",
            }
        )

    migrated = len(rows)
    if not rows:
        print(f"[ok] Migrated {migrated} manifest entries from {manifest_path}")
        return migrated

    print(f"[bulk] Inserting {migrated} manifest entries...")
    with Session(get_engine()) as session:
        for chunk_index, chunk in enumerate(_chunked(rows, 500), start=1):
            insert_stmt = pg_insert(ModelManifestEntry).values(chunk)
            upsert_stmt = insert_stmt.on_conflict_do_update(
                index_elements=["manifest_key"],
                set_={
                    "model_name": insert_stmt.excluded.model_name,
                    "version": insert_stmt.excluded.version,
                    "league": insert_stmt.excluded.league,
                    "model_type": insert_stmt.excluded.model_type,
                    "target": insert_stmt.excluded.target,
                    "filename": insert_stmt.excluded.filename,
                    "mode": insert_stmt.excluded.mode,
                    "status": insert_stmt.excluded.status,
                    "sklearn_version": insert_stmt.excluded.sklearn_version,
                    "train_size": insert_stmt.excluded.train_size,
                    "test_size": insert_stmt.excluded.test_size,
                    "registered_at": insert_stmt.excluded.registered_at,
                    "features_json": insert_stmt.excluded.features_json,
                    "params_json": insert_stmt.excluded.params_json,
                    "metrics_json": insert_stmt.excluded.metrics_json,
                    "metadata_json": insert_stmt.excluded.metadata_json,
                },
            )
            session.execute(upsert_stmt)
            session.commit()
            inserted = min(chunk_index * 500, migrated)
            print(f"[bulk] Inserted {inserted}/{migrated} rows...")

    print(f"[ok] Migrated {migrated} manifest entries from {manifest_path}")
    return migrated


def _migrate_resolved_predictions() -> int:
    outcomes_path = AuthoritativeResolver.DEFAULT_OUTCOMES_PATH
    if not outcomes_path.exists():
        print(f"[skip] Resolved predictions not found: {outcomes_path}")
        return 0

    try:
        raw = pd.read_csv(outcomes_path)
    except pd.errors.EmptyDataError:
        print(f"[skip] Resolved predictions file is empty: {outcomes_path}")
        return 0

    resolver = AuthoritativeResolver(outcomes_path=outcomes_path)
    repaired = resolver._repair_legacy_outcomes_schema(raw)

    required_columns = set(AuthoritativeResolver.OUTCOME_COLUMNS)
    missing = required_columns - set(repaired.columns)
    if missing:
        raise RuntimeError(
            f"Resolved predictions file missing required columns: {sorted(missing)}"
        )

    repaired = repaired.where(pd.notnull(repaired), None)
    rows: list[dict[str, Any]] = []
    for row in repaired.to_dict(orient="records"):
        prediction_id = row.get("prediction_id")
        match_hash = row.get("match_hash")
        league = row.get("league")
        if not prediction_id or not match_hash or not league:
            continue

        rows.append(
            {
                "prediction_id": str(prediction_id),
                "match_hash": str(match_hash),
                "league": str(league),
                "kickoff_date": _parse_datetime(row.get("kickoff_date")),
                "market": str(row.get("market")) if row.get("market") is not None else None,
                "probability": _to_float(row.get("probability")),
                "outcome": str(row.get("outcome")) if row.get("outcome") is not None else None,
                "resolved_at": _parse_datetime(row.get("resolved_at")),
            }
        )

    # Deduplicate on prediction_id - keep last occurrence (most recent resolution)
    seen: dict[str, dict] = {}
    for row in rows:
        seen[row["prediction_id"]] = row
    rows = list(seen.values())
    migrated = len(rows)

    if not migrated:
        print(f"[ok] Migrated 0 resolved prediction rows from {outcomes_path}")
        return 0

    print(f"[bulk] Inserting {migrated} resolved prediction rows (after dedup)...")
    with Session(get_engine()) as session:
        for chunk_index, chunk in enumerate(_chunked(rows, 500), start=1):
            insert_stmt = pg_insert(ResolvedPrediction).values(chunk)
            upsert_stmt = insert_stmt.on_conflict_do_update(
                index_elements=["prediction_id"],
                set_={
                    "match_hash": insert_stmt.excluded.match_hash,
                    "league": insert_stmt.excluded.league,
                    "kickoff_date": insert_stmt.excluded.kickoff_date,
                    "market": insert_stmt.excluded.market,
                    "probability": insert_stmt.excluded.probability,
                    "outcome": insert_stmt.excluded.outcome,
                    "resolved_at": insert_stmt.excluded.resolved_at,
                },
            )
            session.execute(upsert_stmt)
            session.commit()
            inserted = min(chunk_index * 500, migrated)
            print(f"[bulk] Inserted {inserted}/{migrated} rows...")

    print(f"[ok] Migrated {migrated} resolved prediction rows from {outcomes_path}")
    return migrated


def main() -> int:
    try:
        masked_url = _require_database_ready()
        print(f"Starting PostgreSQL migration into {masked_url}")

        global_state_count = _migrate_global_drift_state()
        drift_alert_count = _migrate_drift_alerts()
        manifest_count = _migrate_manifest()
        resolved_count = _migrate_resolved_predictions()

        print(
            "Migration complete. "
            f"global_state={global_state_count}, "
            f"drift_alerts={drift_alert_count}, "
            f"manifest_entries={manifest_count}, "
            f"resolved_predictions={resolved_count}"
        )
        return 0
    except Exception as exc:
        print(f"Migration failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
