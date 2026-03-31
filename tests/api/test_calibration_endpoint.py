import json
from pathlib import Path

from fastapi.testclient import TestClient

import src.api.main as api_main


client = TestClient(api_main.app)


def _write_reliability_file(
    file_path: Path,
    *,
    generated_at: str,
    date: str,
    market: str,
    league: str,
) -> None:
    payload = {
        "generated_at": generated_at,
        "date": date,
        "market": market,
        "league": league,
        "total_samples": 120,
        "deviation_threshold": 0.10,
        "max_deviation": 0.08,
        "systematic_miscalibration": False,
        "buckets": [],
    }
    file_path.write_text(json.dumps(payload), encoding="utf-8")


def test_calibration_endpoint_returns_latest_payload_per_market(monkeypatch, tmp_path):
    monitoring_dir = tmp_path / "monitoring"
    monitoring_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(api_main, "DATA_DIR", tmp_path)

    _write_reliability_file(
        monitoring_dir / "reliability_home_win_PL_2026-03-30.json",
        generated_at="2026-03-30T01:00:00",
        date="2026-03-30",
        market="home_win",
        league="PL",
    )
    _write_reliability_file(
        monitoring_dir / "reliability_home_win_PL_2026-03-31.json",
        generated_at="2026-03-31T01:00:00",
        date="2026-03-31",
        market="home_win",
        league="PL",
    )
    _write_reliability_file(
        monitoring_dir / "reliability_away_win_PL_2026-03-31.json",
        generated_at="2026-03-31T01:05:00",
        date="2026-03-31",
        market="away_win",
        league="PL",
    )
    _write_reliability_file(
        monitoring_dir / "reliability_home_win_BL1_2026-03-31.json",
        generated_at="2026-03-31T01:00:00",
        date="2026-03-31",
        market="home_win",
        league="BL1",
    )

    response = client.get("/api/v1/calibration/PL")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "OK"
    assert payload["league"] == "PL"
    assert [entry["market"] for entry in payload["reliability"]] == ["away_win", "home_win"]
    latest_home = payload["reliability"][1]
    assert latest_home["date"] == "2026-03-31"
    assert latest_home["source_file"] == "reliability_home_win_PL_2026-03-31.json"


def test_calibration_endpoint_returns_no_data_when_missing(monkeypatch, tmp_path):
    (tmp_path / "monitoring").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(api_main, "DATA_DIR", tmp_path)

    response = client.get("/api/v1/calibration/SA")

    assert response.status_code == 200
    assert response.json() == {"league": "SA", "status": "NO_DATA", "reliability": []}
