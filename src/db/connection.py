"""Database connection helpers for optional PostgreSQL support."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

from src.config import DATA_DIR

DEFAULT_SQLITE_DB_PATH: Path = DATA_DIR / "models" / "football_intelligence.db"
create_engine = None


def database_is_configured() -> bool:
    """Return True when DATABASE_URL is explicitly configured."""
    return bool(os.environ.get("DATABASE_URL"))


def _resolve_database_url() -> str:
    raw_url = os.environ.get("DATABASE_URL")
    if raw_url:
        if raw_url.startswith("postgres://"):
            return raw_url.replace("postgres://", "postgresql+psycopg2://", 1)
        if raw_url.startswith("postgresql://") and "+psycopg2" not in raw_url:
            return raw_url.replace("postgresql://", "postgresql+psycopg2://", 1)
        return raw_url

    DEFAULT_SQLITE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return DEFAULT_SQLITE_DB_PATH.resolve().as_uri().replace("file:///", "sqlite:///")


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Return the active SQLAlchemy engine."""
    global create_engine

    if create_engine is None:
        from sqlalchemy import create_engine as sqlalchemy_create_engine

        create_engine = sqlalchemy_create_engine

    database_url = _resolve_database_url()
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite:///") else {}
    return create_engine(
        database_url,
        connect_args=connect_args,
        pool_pre_ping=not database_url.startswith("sqlite:///"),
    )
