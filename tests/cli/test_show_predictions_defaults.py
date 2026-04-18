import pandas as pd

from src.cli.commands import prediction as prediction_module
from src.cli.utils import DateFilter
from src.config.leagues import ACTIVE_LEAGUES


class _DummyGate:
    def process(self, bets):
        return [], {}


def _scheduled_match(league: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-04-18T12:00:00Z"),
                "status": "SCHEDULED",
                "league": league,
                "home_team": f"{league} Home",
                "away_team": f"{league} Away",
                "match_id": f"{league}-1",
            }
        ]
    )


def _patch_lightweight_prediction_path(monkeypatch):
    pipeline_calls = []
    filter_calls = []
    persisted_leagues = []

    class DummyPipeline:
        def run(self, league=None):
            pipeline_calls.append(league)
            return _scheduled_match(str(league))

    class DummyContainer:
        pipeline = DummyPipeline()

    def fake_filter(df, date_filter, show_all=False, user_timezone="UTC"):
        filter_calls.append((date_filter, show_all, user_timezone))
        return df

    def fake_predict(df, use_simulator=True, use_rl_weights=False):
        row = df.iloc[0]
        return [
            {
                "league": row["league"],
                "match_id": row["match_id"],
            }
        ]

    monkeypatch.setattr(
        prediction_module.ServiceContainer,
        "get_instance",
        lambda: DummyContainer(),
    )
    monkeypatch.setattr(
        prediction_module,
        "validate_match_dataframe",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(prediction_module, "filter_matches_by_date", fake_filter)
    monkeypatch.setattr(prediction_module, "_run_predict_loop", fake_predict)
    monkeypatch.setattr(prediction_module, "_prepare_bets", lambda results: [])
    monkeypatch.setattr(prediction_module, "SelectionGate", lambda: _DummyGate())
    monkeypatch.setattr(prediction_module, "_render_output", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        prediction_module,
        "_persist_predictions",
        lambda results, league: persisted_leagues.append(league),
    )

    return pipeline_calls, filter_calls, persisted_leagues


def test_show_predictions_defaults_to_today_for_all_active_leagues(monkeypatch):
    pipeline_calls, filter_calls, persisted_leagues = _patch_lightweight_prediction_path(monkeypatch)

    prediction_module.show_predictions(
        date="today",
        league=None,
        all=False,
        tz="UTC",
        simulate=False,
        use_rl_weights=False,
    )

    assert pipeline_calls == ACTIVE_LEAGUES
    assert persisted_leagues == ACTIVE_LEAGUES
    assert [(date_filter, show_all) for date_filter, show_all, _ in filter_calls] == [
        ("today", False)
    ] * len(ACTIVE_LEAGUES)


def test_show_predictions_all_flag_expands_date_window_for_selected_league(monkeypatch):
    pipeline_calls, filter_calls, persisted_leagues = _patch_lightweight_prediction_path(monkeypatch)

    prediction_module.show_predictions(
        date="today",
        league="Premier League",
        all=True,
        tz="UTC",
        simulate=False,
        use_rl_weights=False,
    )

    assert pipeline_calls == ["PL"]
    assert persisted_leagues == ["PL"]
    assert filter_calls == [(DateFilter.ALL, True, "UTC")]
