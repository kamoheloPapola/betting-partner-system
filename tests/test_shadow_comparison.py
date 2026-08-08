from __future__ import annotations

import subprocess
from datetime import datetime, timezone

import pandas as pd
import pytest

from scripts import shadow_comparison
from scripts import train_batch


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


def test_build_comparison_rows_reads_authoritative_outcomes_csv(tmp_path):
    outcomes_file = tmp_path / "prediction_outcomes.csv"
    log_file = tmp_path / "predictions.log"
    pd.DataFrame(
        [
            {
                "match_hash": "match-1",
                "league": "PL",
                "market": "home_win",
                "probability": 0.70,
                "outcome": "WON",
                "resolved_at": datetime.now(timezone.utc).isoformat(),
            }
        ]
    ).to_csv(outcomes_file, index=False)
    log_file.write_text(
        "[RL-SHADOW] match_id=match-1 context=PL:home_win weights={} "
        "home=0.80 draw=0.10 away=0.10 over_2_5=0.60 btts_yes=0.55 applied=true\n",
        encoding="utf-8",
    )

    rows = shadow_comparison.build_comparison_rows(
        days=30,
        log_file=log_file,
        outcomes_file=outcomes_file,
    )

    assert len(rows) == 1
    assert rows[0].bucket == "PL:home_win"
    assert rows[0].live_ece == pytest.approx(0.30)
    assert rows[0].shadow_ece == pytest.approx(0.20)
    assert rows[0].verdict == "PROMOTE"


def test_train_batch_bandit_failure_is_non_fatal(monkeypatch, capsys):
    calls: list[tuple[list[str], bool]] = []

    class _FakeTrainer:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def run(self, tracked_leagues, full_retrain) -> None:
            calls.append((list(tracked_leagues), bool(full_retrain)))

    def _raise_called_process_error(command, check):
        raise subprocess.CalledProcessError(returncode=1, cmd=command)

    monkeypatch.setattr(train_batch, "ProbabilityModelTrainer", _FakeTrainer)
    monkeypatch.setattr(train_batch.subprocess, "run", _raise_called_process_error)

    train_batch.run_training(leagues=["PL"], full_retrain=False)

    output = capsys.readouterr().out
    assert calls == [(["PL"], False)]
    assert "Updating RL bandit state..." in output
    assert "BANDIT UPDATE FAILED (non-fatal):" in output
    assert "Status: SUCCESS" in output
