"""Execution-only compatibility adapter for the complete Phase 7E-R checkpoint.

The frozen Phase 7E-R target builder identifies fixtures with ``match_id``.
The unchanged Phase 7P prediction recorder consumes the same identity through
its historical ``match_hash`` column name.  This adapter adds that alias while
leaving the feature values, causal bundle, models, and frozen protocols intact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from scripts import run_phase7er_source_refresh as runner
from src.evaluation.phase7e_multi_source_continuity import Phase7EError


_ORIGINAL_BUILD_TARGET_MATRIX = runner._build_target_matrix


def _artifact(path: Path, *, role: str, load_order: int) -> dict[str, Any]:
    if not path.is_file():
        raise Phase7EError(f"checkpoint provenance artifact is missing: {path}")
    return {
        "path": runner.relative(path),
        "sha256": runner.sha256_file(path),
        "size_bytes": path.stat().st_size,
        "role": role,
        "load_order": load_order,
    }


def _checkpoint_artifacts(run_dir: Path) -> list[dict[str, Any]]:
    paths: list[tuple[Path, str]] = [
        (runner.phase7e.PROTECTED_BASE, "protected_historical_base"),
        (
            runner.phase7e.PHASE7D_RUN
            / "raw_provider_snapshot"
            / "football-data-org-pl-2025.source.json",
            "primary_2025_source_identity",
        ),
        (
            runner.phase7e.PHASE7D_RUN
            / "raw_provider_snapshot"
            / "football-data-org-pl-2025.response.json",
            "primary_2025_response",
        ),
        (
            runner.phase7e.PHASE7D_RUN
            / "raw_provider_snapshot"
            / "football-data-org-pl-2026.source.json",
            "primary_2026_source_identity",
        ),
        (
            runner.phase7e.PHASE7D_RUN
            / "raw_provider_snapshot"
            / "football-data-org-pl-2026.response.json",
            "primary_2026_response",
        ),
        (
            run_dir / "raw_supplemental_refresh" / "football-data-co-uk-E0-2627.source.json",
            "supplemental_source_identity",
        ),
        (
            run_dir / "raw_supplemental_refresh" / "football-data-co-uk-E0-2627.csv",
            "supplemental_response",
        ),
        (run_dir / "normalized_supplemental_refresh" / "E0-2627.json", "supplemental_normalized"),
        (run_dir / "multi_source_continuity_snapshot.json", "multi_source_continuity_snapshot"),
    ]
    paths.extend(
        (path, "canonical_match_observation")
        for path in sorted((run_dir / "canonical_match_observations" / "observations").glob("*.json"))
    )
    return [
        _artifact(path, role=role, load_order=index)
        for index, (path, role) in enumerate(paths)
    ]


def build_target_matrix_with_phase7p_identity(
    run_dir: Path, *, prediction_at: str, target_fixture_id: str | None = None
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Expose ``match_id`` as ``match_hash`` for the frozen Phase 7P recorder."""
    matrix, fixture, bundle = _ORIGINAL_BUILD_TARGET_MATRIX(
        run_dir,
        prediction_at=prediction_at,
        target_fixture_id=target_fixture_id,
    )
    if "match_id" not in matrix.columns:
        raise Phase7EError("target feature matrix is missing match_id")
    expected = matrix["match_id"].astype(str)
    if "match_hash" in matrix.columns and not matrix["match_hash"].astype(str).equals(expected):
        raise Phase7EError("match_hash does not preserve the canonical match_id")
    matrix = matrix.copy()
    matrix["match_hash"] = expected
    prior_snapshot = dict(matrix.attrs.get("ffps_data_snapshot", {}))
    snapshot = runner.build_file_snapshot(
        frame=matrix,
        artifacts=_checkpoint_artifacts(run_dir),
        source_type="phase7er_execution_checkpoint_causal_history",
    )
    snapshot["multi_source_continuity_sha256"] = prior_snapshot[
        "multi_source_continuity_sha256"
    ]
    snapshot["causal_bundle"] = prior_snapshot["causal_bundle"]
    matrix.attrs["ffps_data_snapshot"] = snapshot
    target_id = str(fixture["match_id"])
    target_rows = matrix.loc[matrix["match_id"].astype(str).eq(target_id)]
    if len(target_rows) != 1:
        raise Phase7EError("adapted target feature row is not unique")
    return matrix, target_rows.iloc[0], bundle


def install() -> None:
    """Install the compatibility alias for this process only."""
    runner._build_target_matrix = build_target_matrix_with_phase7p_identity
