import json
import logging
from pathlib import Path

import pandas as pd
import pytest

import src.ml.calibration as calibration_module
import src.monitoring.performance_engine as performance_engine_module
import src.strategies.css_math as css_math
from src.monitoring.confidence_drift import QuantileStratifier
from src.monitoring.performance_engine import PerformanceTracker


def _configure_isolated_alpha_store(monkeypatch, tmp_path):
    monkeypatch.setattr(calibration_module, "DAMPENING_ALPHA_FILE", tmp_path / "dampening_alphas.json")
    monkeypatch.setattr(calibration_module, "_dampening_alphas", {})
    monkeypatch.setattr(calibration_module, "_dampening_alphas_loaded", False)


def _build_tracker(monkeypatch, tmp_path) -> PerformanceTracker:
    tracker = PerformanceTracker()
    report_dir = tmp_path / "monitoring" / "performance"
    report_dir.mkdir(parents=True, exist_ok=True)
    reliability_dir = tmp_path / "monitoring"
    reliability_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(tracker, "REPORT_DIR", report_dir)
    monkeypatch.setattr(tracker, "RELIABILITY_DIR", reliability_dir)
    monkeypatch.setattr(tracker, "_compute_roi", lambda: {"status": "NO_BET_LOG"})
    return tracker


def test_live_ece_feedback_adds_more_dampening_for_overconfident_market(monkeypatch, tmp_path):
    _configure_isolated_alpha_store(monkeypatch, tmp_path)
    tracker = _build_tracker(monkeypatch, tmp_path)

    evals = pd.DataFrame(
        {
            "market": ["home_win"] * 300,
            "predicted_probability": [0.90] * 300,
            "actual_outcome": [0.0] * 300,
            "model_version": ["v1"] * 300,
            "prediction_date": ["2026-03-30"] * 300,
        }
    )
    monkeypatch.setattr(tracker, "_load_all_evaluations", lambda: evals)

    report = tracker.generate_report()

    assert report["auto_calibration_adjustments"]
    adjustment = report["auto_calibration_adjustments"][0]
    assert adjustment["market"] == "home_win"
    assert adjustment["direction"] == "more_dampening"
    assert adjustment["alpha_before"] == 0.8
    assert adjustment["alpha_after"] == 0.75

    saved = json.loads(calibration_module.DAMPENING_ALPHA_FILE.read_text(encoding="utf-8"))
    assert saved["home_win"] == 0.75


def test_live_ece_feedback_reduces_dampening_for_underconfident_market(monkeypatch, tmp_path):
    _configure_isolated_alpha_store(monkeypatch, tmp_path)
    tracker = _build_tracker(monkeypatch, tmp_path)

    evals = pd.DataFrame(
        {
            "market": ["away_win"] * 300,
            "predicted_probability": [0.10] * 300,
            "actual_outcome": [1.0] * 300,
            "model_version": ["v1"] * 300,
            "prediction_date": ["2026-03-30"] * 300,
        }
    )
    monkeypatch.setattr(tracker, "_load_all_evaluations", lambda: evals)

    report = tracker.generate_report()

    assert report["auto_calibration_adjustments"]
    adjustment = report["auto_calibration_adjustments"][0]
    assert adjustment["market"] == "away_win"
    assert adjustment["direction"] == "less_dampening"
    assert adjustment["alpha_before"] == 0.8
    assert adjustment["alpha_after"] == 0.85

    saved = json.loads(calibration_module.DAMPENING_ALPHA_FILE.read_text(encoding="utf-8"))
    assert saved["away_win"] == pytest.approx(0.85)


def _configure_isolated_css_store(monkeypatch, tmp_path):
    monkeypatch.setattr(css_math, "CSS_WEIGHTS_FILE", tmp_path / "css_market_weights.json")
    monkeypatch.setattr(
        css_math,
        "MARKET_WEIGHTS",
        {
            "TG_U1.5": 1.00,
            "CORNERS_U11.5": 0.95,
            "CORNERS_O7.5": 0.90,
            "DC": 0.88,
            "1X2": 0.75,
            "CARDS_U5.5": 0.85,
            "CARDS_O2.5": 0.80,
        },
    )
    monkeypatch.setattr(css_math, "_weights_loaded", True)


def test_css_weight_refresh_reranks_and_persists_after_200_settled_bets(monkeypatch, tmp_path):
    _configure_isolated_alpha_store(monkeypatch, tmp_path)
    _configure_isolated_css_store(monkeypatch, tmp_path)
    tracker = _build_tracker(monkeypatch, tmp_path)

    evals = pd.DataFrame(
        {
            "market": (["1X2"] * 220) + (["TG_U1.5"] * 220),
            "predicted_probability": ([0.90] * 220) + ([0.90] * 220),
            "actual_outcome": ([1.0] * 220) + ([0.0] * 220),
            "model_version": (["m_1x2"] * 220) + (["m_tg"] * 220),
            "prediction_date": ["2026-03-30"] * 440,
        }
    )
    monkeypatch.setattr(tracker, "_load_all_evaluations", lambda: evals)

    report = tracker.generate_report()

    assert report["css_weight_updates"]
    assert report["css_weight_updates"]["1X2"] == 1.0
    assert report["css_weight_updates"]["TG_U1.5"] == 0.75
    assert css_math.CSS_WEIGHTS_FILE.exists()

    saved = json.loads(css_math.CSS_WEIGHTS_FILE.read_text(encoding="utf-8"))
    assert saved["1X2"] == 1.0
    assert saved["TG_U1.5"] == 0.75


def test_css_weight_refresh_waits_until_200_settled_bets(monkeypatch, tmp_path):
    _configure_isolated_alpha_store(monkeypatch, tmp_path)
    _configure_isolated_css_store(monkeypatch, tmp_path)
    tracker = _build_tracker(monkeypatch, tmp_path)

    evals = pd.DataFrame(
        {
            "market": (["1X2"] * 150) + (["TG_U1.5"] * 150),
            "predicted_probability": ([0.90] * 150) + ([0.90] * 150),
            "actual_outcome": ([1.0] * 150) + ([0.0] * 150),
            "model_version": (["m_1x2"] * 150) + (["m_tg"] * 150),
            "prediction_date": ["2026-03-30"] * 300,
        }
    )
    monkeypatch.setattr(tracker, "_load_all_evaluations", lambda: evals)

    report = tracker.generate_report()

    assert report["css_weight_updates"] == {}
    assert not css_math.CSS_WEIGHTS_FILE.exists()


def test_quantile_stratifier_cold_start_blends_30_70_when_under_100_samples():
    stratifier = QuantileStratifier()
    market = "home_win"

    for _ in range(1000):
        stratifier.add_prediction(market, 0.8, league="SA")
    for _ in range(90):
        stratifier.add_prediction(market, 0.2, league="PL")

    stratifier.compute_percentiles(market, league="PL")
    pct = stratifier.get_percentiles(market, league="PL")

    assert pct is not None
    assert pct["q95"] == pytest.approx(0.62)
    assert pct["q90"] == pytest.approx(0.62)
    assert pct["q80"] == pytest.approx(0.62)


def test_quantile_stratifier_cold_start_blends_70_30_when_under_500_samples():
    stratifier = QuantileStratifier()
    market = "away_win"

    for _ in range(1000):
        stratifier.add_prediction(market, 0.8, league="SA")
    for _ in range(200):
        stratifier.add_prediction(market, 0.2, league="PL")

    stratifier.compute_percentiles(market, league="PL")
    pct = stratifier.get_percentiles(market, league="PL")

    assert pct is not None
    assert pct["q95"] == pytest.approx(0.38)
    assert pct["q90"] == pytest.approx(0.38)
    assert pct["q80"] == pytest.approx(0.38)
    assert stratifier.get_tier(market, 0.40, league="PL") == "TIER_A"


def test_generate_report_writes_reliability_diagram_per_market_per_league(monkeypatch, tmp_path):
    _configure_isolated_alpha_store(monkeypatch, tmp_path)
    tracker = _build_tracker(monkeypatch, tmp_path)

    evals = pd.DataFrame(
        {
            "market": ["home_win", "home_win", "away_win", "away_win"],
            "league": ["PL", "PL", "PL", "SA"],
            "predicted_probability": [0.65, 0.35, 0.80, 0.25],
            "actual_outcome": [1.0, 0.0, 1.0, 0.0],
            "model_version": ["v1", "v1", "v2", "v2"],
            "prediction_date": ["2026-03-30"] * 4,
        }
    )
    monkeypatch.setattr(tracker, "_load_all_evaluations", lambda: evals)

    report = tracker.generate_report()

    assert len(report["reliability_diagrams"]) == 3
    home_pl = next(
        item for item in report["reliability_diagrams"]
        if item["market"] == "home_win" and item["league"] == "PL"
    )
    payload = json.loads(Path(home_pl["path"]).read_text(encoding="utf-8"))
    assert payload["market"] == "home_win"
    assert payload["league"] == "PL"
    assert len(payload["buckets"]) == 10


def test_generate_report_warns_when_reliability_bucket_deviation_is_high(monkeypatch, tmp_path, caplog):
    _configure_isolated_alpha_store(monkeypatch, tmp_path)
    tracker = _build_tracker(monkeypatch, tmp_path)

    evals = pd.DataFrame(
        {
            "market": ["home_win"] * 50,
            "league": ["PL"] * 50,
            "predicted_probability": [0.95] * 50,
            "actual_outcome": [0.0] * 50,
            "model_version": ["v1"] * 50,
            "prediction_date": ["2026-03-30"] * 50,
        }
    )
    monkeypatch.setattr(tracker, "_load_all_evaluations", lambda: evals)

    performance_logger = logging.getLogger(performance_engine_module.__name__)
    monkeypatch.setattr(performance_logger, "propagate", True)

    with caplog.at_level(logging.WARNING, logger=performance_engine_module.__name__):
        report = tracker.generate_report()

    assert report["reliability_diagrams"]
    assert any(
        "Systematic miscalibration detected" in record.getMessage()
        for record in caplog.records
    )
