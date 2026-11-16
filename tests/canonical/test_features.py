import pytest
from src.canonical.features import training_features,FeatureEngine,FEATURE_NAMES,RULES
from src.canonical.clock import CanonicalError


def game(mid,day,h=1,a=2,hg=1,ag=0):
    return {"id":mid,"day":day,"league":"PL","home_id":h,"away_id":a,"home_goals":hg,"away_goals":ag}


def test_one_implementation_training_and_query_prefix():
    rows=[game(1,"2026-01-01"),game(2,"2026-01-03",hg=0,ag=0),game(3,"2026-01-05",hg=2,ag=1)]
    vectors,_=training_features(rows)
    _,seed=training_features(rows[:2])
    engine=FeatureEngine(seed,[],asof="2026-01-04T00:00:00Z")
    actual,_=engine.vector({"league":"PL","home_id":1,"away_id":2,"kickoff_utc":"2026-01-05T15:00:00Z"})
    assert actual==vectors[3] and len(actual)==len(FEATURE_NAMES)==17


def test_current_and_future_outcomes_do_not_change_target_features():
    rows=[game(1,"2026-01-01"),game(2,"2026-01-03",hg=0,ag=0),game(3,"2026-01-05",hg=2,ag=1)]
    a,_=training_features(rows)
    rows[1]["home_goals"]=99;rows[2]["home_goals"]=98
    b,_=training_features(rows)
    assert a[2]==b[2] and a[1]==b[1]


def test_elo_draw_equal_teams_home_down_away_up():
    _,seed=training_features([game(1,"2026-01-01",hg=1,ag=1)])
    states={r["team_id"]:r["state"] for r in seed["states"]}
    assert states["1"]["elo"]<1500<states["2"]["elo"]
    assert states["1"]["elo"]+states["2"]["elo"]==pytest.approx(3000)


def test_duplicate_team_day_is_a_failure():
    with pytest.raises(CanonicalError,match="DUPLICATE_TEAM_DATE"):
        training_features([game(1,"2026-01-01"),game(2,"2026-01-01")])


def test_cold_start_and_league_isolation():
    _,seed=training_features([game(1,"2026-01-01")])
    engine=FeatureEngine(seed,[],asof="2026-01-02T00:00:00Z")
    vector,_=engine.vector({"league":"PD","home_id":1,"away_id":2,"kickoff_utc":"2026-01-03T15:00:00Z"})
    assert vector[-3:]==[1500,1500,0] and vector[0]==1.35


def test_verified_finished_observation_needed_before_forecast():
    _,seed=training_features([])
    row={"fixture_id":1,"league":"PL","home_id":1,"away_id":2,"home_goals":1,"away_goals":0,
         "kickoff_utc":"2026-01-01T15:00:00Z","observed_utc":"2026-01-03T00:00:00Z"}
    with pytest.raises(CanonicalError,match="FUTURE_RESULT"):
        FeatureEngine(seed,[row],asof="2026-01-02T00:00:00Z")


def test_source_corners_cannot_be_accessed():
    class Guard(dict):
        def get(self,key,*args):
            if "corner" in key:pytest.fail("Corner accessed")
            return super().get(key,*args)
        def __getitem__(self,key):
            if "corner" in key:pytest.fail("Corner accessed")
            return super().__getitem__(key)
    values,_=training_features([Guard(game(1,"2026-01-01"))])
    assert len(values[1])==17

