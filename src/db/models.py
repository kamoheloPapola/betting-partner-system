"""SQLAlchemy ORM models for optional operational-data storage."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for optional SQLAlchemy ORM models."""


class ResolvedPrediction(Base):
    """Resolved prediction rows mirroring prediction_outcomes.csv."""

    __tablename__ = "resolved_predictions"
    __table_args__ = (
        Index("idx_resolved_predictions_match_market", "match_hash", "market"),
        Index("idx_resolved_predictions_league_kickoff", "league", "kickoff_date"),
    )

    prediction_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    match_hash: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    league: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    kickoff_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    market: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)
    probability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    outcome: Mapped[Optional[str]] = mapped_column(String(16), nullable=True, index=True)
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True, index=True)


class ModelManifestEntry(Base):
    """
    Manifest rows mirroring manifest.json entries.

    Common fields are extracted into columns, while variable metadata remains
    available through JSON-serialized text blobs.
    """

    __tablename__ = "model_manifest_entries"
    __table_args__ = (
        Index("idx_model_manifest_name_league", "model_name", "league"),
        Index("idx_model_manifest_status", "status"),
        Index("idx_model_manifest_registered_at", "registered_at"),
    )

    manifest_key: Mapped[str] = mapped_column(String(255), primary_key=True)
    model_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    league: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    model_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    target: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    filename: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    mode: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    status: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    sklearn_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    train_size: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    test_size: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    registered_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    features_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    params_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metrics_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")


class DriftEvent(Base):
    """Drift alert rows mirroring drift_alerts.csv entries."""

    __tablename__ = "drift_events"
    __table_args__ = (
        Index("idx_drift_events_detected_at", "detected_at"),
        Index("idx_drift_events_type_league", "event_type", "league"),
        Index("idx_drift_events_severity", "severity"),
    )

    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    event_type: Mapped[str] = mapped_column(String(128), nullable=False)
    league: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    market: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    severity: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    metric: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    threshold: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    detected_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    payload_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
