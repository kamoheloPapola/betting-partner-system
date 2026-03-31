import pandas as pd
import requests

import src.monitoring.reconciler as reconciler_module
from src.monitoring.reconciler import Reconciler


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status={self.status_code}")

    def json(self):
        return self._payload


def test_auto_settle_retries_with_backoff_and_updates_tracker(monkeypatch, tmp_path):
    reconciler = Reconciler()
    reconciler.LABELED_PATH = tmp_path / "results_labeled.csv"
    reconciler.EVALS_DIR = tmp_path / "evaluations"
    reconciler.EVALS_DIR.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(reconciler_module, "FOOTBALL_DATA_ORG_KEY", "test-token")

    attempt_counter = {"count": 0}
    sleep_calls = []

    def fake_get(*args, **kwargs):
        attempt_counter["count"] += 1
        if attempt_counter["count"] < 3:
            raise requests.Timeout("temporary timeout")
        return _FakeResponse(
            {
                "matches": [
                    {
                        "utcDate": "2026-03-30T18:00:00Z",
                        "homeTeam": {"name": "Arsenal"},
                        "awayTeam": {"name": "Chelsea"},
                        "score": {"fullTime": {"home": 2, "away": 1}},
                    }
                ]
            }
        )

    monkeypatch.setattr(reconciler_module.requests, "get", fake_get)
    monkeypatch.setattr(reconciler_module.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    reconciled_dates = []
    monkeypatch.setattr(reconciler, "reconcile_date", lambda pred_date: reconciled_dates.append(pred_date))

    report_calls = []

    class FakeTracker:
        def generate_report(self, last_n_days=None):
            report_calls.append(last_n_days)
            return {"status": "OK"}

    monkeypatch.setattr(reconciler_module, "PerformanceTracker", FakeTracker)

    summary = reconciler.auto_settle(leagues=["PL"], days_back=3)

    assert summary["status"] == "ok"
    assert summary["fixtures_settled"] == 1
    assert sleep_calls == [1.0, 2.0]
    assert reconciled_dates == ["2026-03-30"]
    assert report_calls == [3]

    labeled = pd.read_csv(reconciler.LABELED_PATH)
    assert len(labeled) == 1
    assert labeled.iloc[0]["home_win"] == 1.0
    assert labeled.iloc[0]["draw"] == 0.0
    assert labeled.iloc[0]["away_win"] == 0.0


def test_auto_settle_is_fail_safe_when_api_key_missing(monkeypatch):
    monkeypatch.setattr(reconciler_module, "FOOTBALL_DATA_ORG_KEY", None)
    summary = Reconciler().auto_settle(leagues=["PL"], days_back=1)
    assert summary["status"] == "skipped"
    assert summary["reason"] == "missing_api_key"

