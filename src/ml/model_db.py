"""
SQLite-backed model lifecycle history.

Tracks train/promote/rollback events for production auditing.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import DATA_DIR

logger = logging.getLogger(__name__)

__all__ = ["ModelHistoryDB"]


class ModelHistoryDB:
    """Persist model lifecycle events to SQLite."""

    DB_FILE: Path = DATA_DIR / "models" / "model_history.db"
    BUSY_TIMEOUT_MS = 30_000

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = Path(db_path) if db_path else self.DB_FILE
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.db_path,
            timeout=self.BUSY_TIMEOUT_MS / 1000,
        )
        connection.execute(f"PRAGMA busy_timeout = {self.BUSY_TIMEOUT_MS}")
        return connection

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            journal_mode = conn.execute("PRAGMA journal_mode = WAL").fetchone()
            if not journal_mode or str(journal_mode[0]).lower() != "wal":
                raise RuntimeError(
                    f"SQLite WAL mode unavailable for model history: {self.db_path}"
                )
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS model_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    league TEXT NOT NULL,
                    version TEXT NOT NULL,
                    brier_score REAL,
                    ece REAL,
                    train_size INTEGER,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_model_history_lookup
                ON model_history (model_name, league, timestamp DESC)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_model_history_event_type
                ON model_history (event_type, timestamp DESC)
                """
            )
            conn.commit()

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def write_event(
        self,
        *,
        model_name: str,
        league: str,
        version: str,
        brier_score: Optional[float],
        ece: Optional[float],
        train_size: Optional[int],
        event_type: str,
        timestamp: Optional[str] = None,
    ) -> None:
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO model_history (
                    model_name, league, version, brier_score, ece, train_size, timestamp, event_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model_name,
                    league,
                    version,
                    self._to_float(brier_score),
                    self._to_float(ece),
                    self._to_int(train_size),
                    ts,
                    event_type,
                ),
            )
            conn.commit()

    def write_event_from_metadata(self, metadata: Dict[str, Any], event_type: str) -> None:
        metrics = metadata.get("metrics", {}) if isinstance(metadata.get("metrics"), dict) else {}
        brier_score = metrics.get("brier_score", metadata.get("brier_score"))
        ece = metrics.get("ece")
        if ece is None:
            ece = metrics.get("calibration_score", metadata.get("ece"))

        train_size = metadata.get("train_size")
        if train_size is None and "test_size" in metadata:
            train_size = self._to_int(metadata.get("test_size")) or 0

        self.write_event(
            model_name=str(metadata.get("name", "unknown")),
            league=str(metadata.get("league") or "Global"),
            version=str(metadata.get("version", "unknown")),
            brier_score=self._to_float(brier_score),
            ece=self._to_float(ece),
            train_size=self._to_int(train_size),
            event_type=event_type,
            timestamp=str(metadata.get("event_timestamp") or datetime.now(timezone.utc).isoformat()),
        )

    def fetch_events(
        self,
        *,
        model_name: Optional[str] = None,
        league: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        clauses: List[str] = []
        values: List[Any] = []

        if model_name:
            clauses.append("model_name = ?")
            values.append(model_name)
        if league:
            clauses.append("league = ?")
            values.append(league)
        if event_type:
            clauses.append("event_type = ?")
            values.append(event_type)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        values.append(int(limit))

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""
                SELECT model_name, league, version, brier_score, ece, train_size, timestamp, event_type
                FROM model_history
                {where}
                ORDER BY id DESC
                LIMIT ?
                """,
                values,
            ).fetchall()

        return [dict(row) for row in rows]
