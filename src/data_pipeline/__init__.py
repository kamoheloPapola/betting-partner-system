"""
Data Pipeline — Canonical Dataset Builder.

Consolidates per-league/season processed match CSVs into a single
unified dataset with coverage metadata.
"""
from .build_matches_dataset import CanonicalDatasetBuilder

__all__ = ["CanonicalDatasetBuilder"]
