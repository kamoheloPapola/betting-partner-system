"""
Drift artifact staleness and consistency tests.

These tests catch the failure mode where a persisted drift state file is:
  - Missing entirely (no nightly has run, or file was deleted)
  - Stale (evaluated_at is absent or too old - indicates a synthetic/patched artifact)
  - Internally inconsistent (re-evaluating stored metrics produces a different status
    than the stored status - indicates the file was manually patched)

These tests are skipped automatically when drift state files do not exist
(e.g. in a clean CI environment before the first nightly run). They are
intended to run on any machine where the nightly pipeline has executed.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.config import DATA_DIR
from src.monitoring.drift_orchestrator import DriftOrchestrator


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Artifact must have been evaluated within this many days or the test fails.
# Set to 2 to give a one-day buffer over the nightly cadence.
STALE_THRESHOLD_DAYS = 2

# Leagues whose per-league drift state files should be validated.
# FL1 is excluded - handoff notes it has no nightly file yet.
ACTIVE_LEAGUES = ["PL", "BL1", "SA", "PD"]

# Capture the real artifact paths at import time. The project test harness
# monkeypatches DriftOrchestrator class defaults per test for write isolation.
GLOBAL_STATE_FILE = DriftOrchestrator.DEFAULT_STATUS_FILE
BASELINE_FILE = DriftOrchestrator.DEFAULT_BASELINE_FILE


def _league_state_file(league: str) -> Path:
    return DATA_DIR / "drift" / f"{league}_drift_status.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_state(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_not_stale(payload: dict, label: str) -> None:
    """Assert evaluated_at is present and within STALE_THRESHOLD_DAYS."""
    evaluated_at = payload.get("evaluated_at")
    assert evaluated_at is not None, (
        f"{label}: 'evaluated_at' is missing - artifact may be synthetic or manually patched. "
        "The PL STOP incident was caused by exactly this condition."
    )
    evaluated_dt = datetime.fromisoformat(str(evaluated_at).replace("Z", "+00:00"))
    if evaluated_dt.tzinfo is None:
        evaluated_dt = evaluated_dt.replace(tzinfo=timezone.utc)
    age = datetime.now(timezone.utc) - evaluated_dt
    assert age < timedelta(days=STALE_THRESHOLD_DAYS), (
        f"{label}: drift artifact is {age.days}d {age.seconds // 3600}h old "
        f"(threshold: {STALE_THRESHOLD_DAYS}d). "
        "Nightly pipeline may not have run, or artifact is from a previous session."
    )


def _assert_status_consistent(
    payload: dict,
    label: str,
    tmp_path: Path,
    *,
    league: str | None = None,
) -> None:
    """
    Re-evaluate stored metrics through DriftOrchestrator and assert the
    computed status matches the persisted status.

    Uses a throwaway status file so the real artifact is never touched.
    """
    metrics = payload.get("metrics", {})
    assert metrics, f"{label}: no 'metrics' in state file - cannot validate consistency"

    if league is not None and metrics.get("inherited_from") == "global":
        global_status = metrics.get("global_status")
        persisted_status = payload.get("status")
        assert persisted_status == global_status, (
            f"{label}: inherited league status '{persisted_status}' does not match "
            f"recorded global status '{global_status}' in stored metrics {metrics}."
        )
        assert metrics.get("sample_size", 0) < metrics.get("minimum_sample_size", 0), (
            f"{label}: inherited drift state should only be used below the minimum "
            f"sample threshold; stored metrics were {metrics}."
        )
        return

    orch = DriftOrchestrator(
        status_file=tmp_path / "throwaway_status.json",
        baseline_file=BASELINE_FILE,
        confidence_state_file=tmp_path / "throwaway_confidence_state.json",
    )
    persisted_status = payload.get("status")
    if league is None:
        computed_status = orch.evaluate_global_drift(current_session_data=metrics)
    else:
        orch._league_status[league] = str(persisted_status)
        orch._evaluate_league_metrics(league, metrics)
        computed_status = orch._league_status.get(league)

    assert computed_status == persisted_status, (
        f"{label}: persisted status '{persisted_status}' does not match "
        f"re-evaluated status '{computed_status}' from stored metrics {metrics}. "
        "The artifact may have been manually patched or is inconsistent."
    )


# ---------------------------------------------------------------------------
# Global drift artifact tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not GLOBAL_STATE_FILE.exists(),
    reason="Global drift state file does not exist - skipping (pre-nightly environment)",
)
def test_global_drift_artifact_evaluated_at_is_present_and_recent():
    """
    The global drift state file must have a non-null evaluated_at timestamp
    that is within STALE_THRESHOLD_DAYS of now.

    A missing evaluated_at is the fingerprint of a synthetic artifact -
    the exact condition that hid the PL STOP drift during the previous session.
    """
    payload = _load_state(GLOBAL_STATE_FILE)
    _assert_not_stale(payload, label="global")


@pytest.mark.skipif(
    not GLOBAL_STATE_FILE.exists(),
    reason="Global drift state file does not exist - skipping (pre-nightly environment)",
)
def test_global_drift_artifact_status_matches_stored_metrics(tmp_path):
    """
    Re-evaluating the metrics stored in the global drift state file through
    DriftOrchestrator must produce the same status that the file records.

    A mismatch means the status field was patched independently of the metrics,
    which breaks the integrity guarantee of the drift state machine.
    """
    payload = _load_state(GLOBAL_STATE_FILE)
    _assert_status_consistent(payload, label="global", tmp_path=tmp_path)


# ---------------------------------------------------------------------------
# Per-league drift artifact tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("league", ACTIVE_LEAGUES)
def test_league_drift_artifact_evaluated_at_is_present_and_recent(league):
    """
    Each active league's drift state file must have a non-null evaluated_at
    timestamp within STALE_THRESHOLD_DAYS.

    Skips automatically if the file does not yet exist (e.g. FL1 pre-nightly).
    """
    state_file = _league_state_file(league)
    if not state_file.exists():
        pytest.skip(f"{league} drift state file does not exist - skipping")

    payload = _load_state(state_file)
    _assert_not_stale(payload, label=league)


@pytest.mark.parametrize("league", ACTIVE_LEAGUES)
def test_league_drift_artifact_status_matches_stored_metrics(league, tmp_path):
    """
    Re-evaluating stored metrics for each league must reproduce the persisted status.

    Skips automatically if the file does not yet exist.
    """
    state_file = _league_state_file(league)
    if not state_file.exists():
        pytest.skip(f"{league} drift state file does not exist - skipping")

    payload = _load_state(state_file)
    _assert_status_consistent(payload, label=league, tmp_path=tmp_path, league=league)
