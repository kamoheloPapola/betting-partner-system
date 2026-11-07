from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts import phase7er_execution_adapter as adapter
from src.evaluation.phase7e_multi_source_continuity import Phase7EError


def test_adapter_adds_identity_alias_without_changing_features(monkeypatch) -> None:
    original = pd.DataFrame(
        [{"match_id": "fixture-1", "feature_a": 1.25, "feature_b": -0.5}]
    )
    original.attrs["ffps_data_snapshot"] = {
        "multi_source_continuity_sha256": "continuity",
        "causal_bundle": {"identity": "causal"},
    }
    bundle = {"multi_source_continuity_sha256": "bundle"}

    def fake_builder(*args, **kwargs):
        return original, original.iloc[0], bundle

    monkeypatch.setattr(adapter, "_ORIGINAL_BUILD_TARGET_MATRIX", fake_builder)
    monkeypatch.setattr(adapter, "_checkpoint_artifacts", lambda run_dir: [])
    matrix, fixture, actual_bundle = adapter.build_target_matrix_with_phase7p_identity(
        Path("unused"), prediction_at="2026-09-05T10:00:00+00:00"
    )

    assert matrix["match_hash"].tolist() == ["fixture-1"]
    assert fixture["match_hash"] == "fixture-1"
    assert matrix[["feature_a", "feature_b"]].equals(original[["feature_a", "feature_b"]])
    assert matrix.attrs["ffps_data_snapshot"]["multi_source_continuity_sha256"] == "continuity"
    assert matrix.attrs["ffps_data_snapshot"]["causal_bundle"] == {"identity": "causal"}
    assert matrix.attrs["ffps_data_snapshot"]["artifacts"] == []
    assert actual_bundle is bundle


def test_adapter_rejects_conflicting_identity_alias(monkeypatch) -> None:
    original = pd.DataFrame([{"match_id": "fixture-1", "match_hash": "fixture-2"}])

    def fake_builder(*args, **kwargs):
        return original, original.iloc[0], {}

    monkeypatch.setattr(adapter, "_ORIGINAL_BUILD_TARGET_MATRIX", fake_builder)

    with pytest.raises(Phase7EError, match="match_hash does not preserve"):
        adapter.build_target_matrix_with_phase7p_identity(
            Path("unused"), prediction_at="2026-09-05T10:00:00+00:00"
        )
