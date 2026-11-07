"""Prediction adapters for non-CLI consumers."""

from __future__ import annotations

from typing import Any

__all__ = ["Predictor"]


def __getattr__(name: str) -> Any:
    # Keep the public import while avoiding a package-initialization cycle when
    # the model registry imports the provenance submodule.
    if name == "Predictor":
        from src.predictions.predictor import Predictor

        return Predictor
    raise AttributeError(name)
