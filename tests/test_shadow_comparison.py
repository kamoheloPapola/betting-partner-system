from __future__ import annotations

import pandas as pd
import pytest

from scripts import shadow_comparison


def test_ece_computation_correctness_with_synthetic_data():
    frame = pd.DataFrame(
        {
            "live_probability": ([0.1] * 10) + ([0.4] * 10) + ([0.8] * 10),
            "actual_outcome": ([0.0] * 10) + ([1.0] * 6) + ([0.0] * 4) + ([1.0] * 9) + ([0.0] * 1),
        }
    )

    ece = shadow_comparison.compute_ece_from_frame(frame, "live_probability")

    assert ece == pytest.approx(0.13333333333333333)


def test_verdict_logic_thresholds():
    assert shadow_comparison.verdict_for_bucket(0.1000, 0.0940) == "PROMOTE"
    assert shadow_comparison.verdict_for_bucket(0.1000, 0.1040) == "HOLD"
    assert shadow_comparison.verdict_for_bucket(0.1000, 0.1080) == "REVERT"


def test_exit_code_logic():
    hold_rows = [
        shadow_comparison.ShadowComparisonRow(
            bucket="PL:home_win",
            live_ece=0.10,
            shadow_ece=0.101,
            delta=-0.001,
            verdict="HOLD",
        )
    ]
    promote_rows = [
        shadow_comparison.ShadowComparisonRow(
            bucket="PL:home_win",
            live_ece=0.10,
            shadow_ece=0.09,
            delta=0.01,
            verdict="PROMOTE",
        )
    ]
    revert_rows = [
        shadow_comparison.ShadowComparisonRow(
            bucket="PL:home_win",
            live_ece=0.10,
            shadow_ece=0.12,
            delta=-0.02,
            verdict="REVERT",
        )
    ]

    assert shadow_comparison.determine_exit_code(promote_rows) == 0
    assert shadow_comparison.determine_exit_code(hold_rows) == 1
    assert shadow_comparison.determine_exit_code(revert_rows) == 2
