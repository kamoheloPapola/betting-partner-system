from typer.testing import CliRunner

from src.cli.app import app


runner = CliRunner()


def test_run_daily_pipeline_executes_sequence(monkeypatch):
    calls: list[tuple] = []
    alerts: list[dict] = []

    monkeypatch.setattr(
        "src.cli.commands.data.fetch_latest_season",
        lambda season="2526": calls.append(("fetch_latest_season", season)),
    )
    monkeypatch.setattr(
        "src.cli.commands.data.fetch_upcoming",
        lambda league, skip_validation=False, refresh=False: calls.append(("fetch_upcoming", getattr(league, "value", league))),
    )
    monkeypatch.setattr(
        "src.cli.commands.maintenance.check_drift",
        lambda league=None, lookback=30: calls.append(("check_drift", league, lookback)),
    )
    monkeypatch.setattr(
        "src.cli.commands.prediction.show_predictions",
        lambda date="today", league=None, all=False, tz="LOCAL", simulate=True: calls.append(
            ("show_predictions", date, league, all, tz, simulate)
        ),
    )
    monkeypatch.setattr(
        "src.cli.commands.evaluation.reconcile",
        lambda date_str=None: calls.append(("reconcile", date_str)),
    )
    monkeypatch.setattr(
        "src.cli.commands.maintenance._send_pipeline_failure_alert",
        lambda **kwargs: alerts.append(kwargs) or True,
    )

    result = runner.invoke(
        app,
        [
            "run-daily-pipeline",
            "--league",
            "PL",
            "--season",
            "2526",
            "--lookback",
            "21",
            "--prediction-date",
            "today",
            "--reconcile-date",
            "2026-03-30",
            "--no-simulate",
            "--all",
        ],
    )

    assert result.exit_code == 0
    assert [call[0] for call in calls] == [
        "fetch_latest_season",
        "fetch_upcoming",
        "check_drift",
        "show_predictions",
        "reconcile",
    ]
    assert alerts == []


def test_run_daily_pipeline_alerts_on_failure(monkeypatch):
    calls: list[tuple] = []
    alerts: list[dict] = []

    monkeypatch.setattr(
        "src.cli.commands.data.fetch_latest_season",
        lambda season="2526": calls.append(("fetch_latest_season", season)),
    )

    def _boom(*args, **kwargs):
        raise RuntimeError("fetch failed")

    monkeypatch.setattr("src.cli.commands.data.fetch_upcoming", _boom)
    monkeypatch.setattr(
        "src.cli.commands.maintenance.check_drift",
        lambda league=None, lookback=30: calls.append(("check_drift", league, lookback)),
    )
    monkeypatch.setattr(
        "src.cli.commands.prediction.show_predictions",
        lambda date="today", league=None, all=False, tz="LOCAL", simulate=True: calls.append(
            ("show_predictions", date, league, all, tz, simulate)
        ),
    )
    monkeypatch.setattr(
        "src.cli.commands.evaluation.reconcile",
        lambda date_str=None: calls.append(("reconcile", date_str)),
    )
    monkeypatch.setattr(
        "src.cli.commands.maintenance._send_pipeline_failure_alert",
        lambda **kwargs: alerts.append(kwargs) or True,
    )

    result = runner.invoke(
        app,
        [
            "run-daily-pipeline",
            "--league",
            "PL",
            "--season",
            "2526",
        ],
    )

    assert result.exit_code == 1
    assert [call[0] for call in calls] == ["fetch_latest_season"]
    assert len(alerts) == 1
    assert alerts[0]["failed_step"] == "Data fetch"
    assert alerts[0]["league"] == "PL"
    assert "Pipeline failed at step: Data fetch" in result.stdout
