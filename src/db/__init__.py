"""Optional SQLAlchemy-backed database helpers."""

from importlib import import_module
from typing import Any

from .connection import database_is_configured, get_engine

_MODEL_EXPORTS = {"Base", "DriftEvent", "ModelManifestEntry", "ResolvedPrediction"}


def __getattr__(name: str) -> Any:
    if name in _MODEL_EXPORTS:
        models = import_module("src.db.models")
        return getattr(models, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "Base",
    "DriftEvent",
    "ModelManifestEntry",
    "ResolvedPrediction",
    "database_is_configured",
    "get_engine",
]
