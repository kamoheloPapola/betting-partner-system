import json
import hashlib
from datetime import datetime,timezone,timedelta
import pytest

from src.canonical.clock import CanonicalError, utc
from src.canonical.store import Store
from src.canonical.model import Model,MODEL_SCHEMA,PARAMS,TARGET_CAP
from src.canonical.features import FEATURE_NAMES,FEATURE_VERSION,training_features
from src.canonical.markets import MARKET_VERSION
from src.canonical.service import _settled_guard_rows, run_once, health, guardrail
from src.canonical.lifecycle import resume


class Clock:
    def __init__(self):self.at=datetime(2026,8,1,tzinfo=timezone.utc)
    def utcnow(self):return self.at


class Client:
    def __init__(self,records=None,error=None):self.records=records or [];self.error=error;self.receipts=[];self.calls=0
    def fetch(self,pairs):
        self.calls+=1
        if self.error:raise self.error
        return self.records,[]


def fixture(mid,kickoff,status="TIMED",**extra):
    return {"id":mid,"league":"PL","season":2026,"home_id":57,"away_id":61,
      "home_name":"Home","away_name":"Away","kickoff_utc":kickoff,"status":status,**extra}


@pytest.fixture
def setup(tmp_path):
    _,seed=training_features([])
    coeff={"mean":[0.]*17,"scale":[1.]*17,"coef":[0.]*17,"intercept":0.}
    payload={"schema":MODEL_SCHEMA,"feature_names":list(FEATURE_NAMES),"feature_version":FEATURE_VERSION,
      "market_version":MARKET_VERSION,"parameters":dict(PARAMS),"target_cap":TARGET_CAP,
      "coefficients":{"home":dict(coeff),"away":dict(coeff)},"feature_seed":seed,
      "baseline_1x2":{"PL":[.4,.3,.3]}}
    model=Model(payload);path=tmp_path/"model.json"
    path.write_text(json.dumps({"model":payload,"validation":{"status":"PASS_LOCKED_REPLAY","model_id":model.model_id}}))
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    with Store(tmp_path/"store.sqlite3") as store:
        store.register_model(model.model_id,path,sha,"2024-07-01T00:00:00Z")
        store.activate_model(model.model_id,"2024-07-01T00:00:00Z","SYNTHETIC_TEST")
        yield store,Clock(),path,model


def test_fetch_freeze_settle_second_run_is_business_noop(setup):
    store,clock,path,model=setup
    rows=[fixture(1,"2026-07-28T15:00:00Z","FINISHED",home_goals=1,away_goals=0),
          fixture(2,"2026-08-03T15:00:00Z")]
    first=run_once(store,Client(rows),clock=clock)
    assert first["state"]=="SUCCESS" and first["counts"]["forecasts_frozen"]==1
    original=dict(store.db.execute("SELECT * FROM forecasts").fetchone())
    clock.at+=timedelta(minutes=1)
    second=run_once(store,Client(rows),clock=clock)
    assert second["counts"]["forecasts_frozen"]==0 and second["counts"]["results_added"]==0
    assert dict(store.db.execute("SELECT * FROM forecasts").fetchone())==original
    assert health(store,at=clock.at)["state"]=="GREEN"


def test_fetch_failure_preserves_business_state_and_is_visible(setup):
    store,clock,_,_=setup
    notifications=[]
    result=run_once(store,Client(error=CanonicalError("PROVIDER_AUTH_OR_ENTITLEMENT_403")),clock=clock,
                    notifier=lambda code:notifications.append(code))
    assert result["exit_code"]==1 and notifications==["PROVIDER_AUTH_OR_ENTITLEMENT_403"]
    assert store.db.execute("SELECT COUNT(*) FROM fixtures").fetchone()[0]==0
    assert health(store,at=clock.at)["state"]=="RED"


def test_model_tamper_stops_before_network(setup):
    store,clock,path,_=setup;path.write_text("{}")
    client=Client()
    result=run_once(store,client,clock=clock)
    assert result["error_code"]=="MODEL_ARTIFACT_HASH_MISMATCH" and client.calls==0


def test_started_fixture_never_gets_forecast(setup):
    store,clock,_,_=setup
    for status in ("IN_PLAY","SCHEDULED","POSTPONED"):
        result=run_once(store,Client([fixture(1,"2026-08-03T15:00:00Z",status)]),clock=clock)
        assert result["counts"]["forecasts_frozen"]==0
    assert not store.forecasts()


def test_kickoff_during_prediction_is_skipped_not_backdated(setup):
    store,_,_,_=setup
    class MovingClock:
        def __init__(self):self.n=0
        def utcnow(self):
            values=[datetime(2026,8,1,14,59,50,tzinfo=timezone.utc),
                    datetime(2026,8,1,14,59,55,tzinfo=timezone.utc),
                    datetime(2026,8,1,15,0,1,tzinfo=timezone.utc)]
            value=values[min(self.n,2)];self.n+=1;return value
    result=run_once(store,Client([fixture(1,"2026-08-01T15:00:00Z")]),clock=MovingClock())
    assert result["state"]=="SUCCESS" and result["counts"]["skipped_kickoff_during_run"]==1
    assert not store.forecasts()


def test_stale_health_is_red_even_after_previous_success(setup):
    store,clock,_,_=setup
    run_once(store,Client(),clock=clock)
    clock.at+=timedelta(hours=37)
    assert "STALE_DATA" in health(store,at=clock.at)["reasons"]


def seed_bad_forecasts(store, model, first_id, count, *, frozen_at, kickoff, observed_at):
    # Storage-only synthetic replay: all rates deliberately wrong for home winners.
    with store.transaction():
        for offset in range(count):
            day = kickoff + timedelta(days=offset)
            row = fixture(first_id + offset, day.isoformat())
            store.ingest([row], frozen_at)
            assert store.freeze(row, frozen_at, .1, 4., model.model_id, MARKET_VERSION, "x" * 64, None)
            store.ingest([dict(row, status="FINISHED", home_goals=1, away_goals=0)], observed_at)


def test_guard_waits_for_300_eligible_forecasts_and_halt_persists(setup):
    store,clock,_,model=setup
    kickoff = datetime(2025, 1, 3, tzinfo=timezone.utc)
    times = {"frozen_at": "2025-01-01T00:00:00Z", "observed_at": "2026-07-31T00:00:00Z"}
    seed_bad_forecasts(store, model, 1, 299, kickoff=kickoff, **times)
    assert guardrail(store)["state"] == "COLLECTING"
    seed_bad_forecasts(store, model, 300, 1, kickoff=kickoff + timedelta(days=299), **times)
    assert guardrail(store)["state"]=="HALT"
    result=run_once(store,Client([fixture(301,"2026-08-03T15:00:00Z")]),clock=clock)
    assert result["state"]=="HALTED" and result["exit_code"]==3 and store.halted()
    assert result["counts"]["forecasts_frozen"]==0
    # A subsequent empty successful fetch must not auto-clear the halt.
    assert run_once(store,Client(),clock=clock)["state"]=="HALTED"
    assert health(store, at=clock.at)["guardrail"] == {"state": "HALTED"}

    resumed_at = resume(store, "Corrected the cause and reviewed the halt", clock.at)
    clock.at += timedelta(minutes=1)
    result = run_once(store, Client([fixture(302, "2026-08-04T15:00:00Z")]), clock=clock)
    assert result["state"] == "SUCCESS"
    assert result["counts"]["guardrail"] == {
        "state": "COLLECTING", "n": 0, "required": 300,
        "window_started_after_utc": resumed_at["window_started_after_utc"],
    }
    assert result["counts"]["forecasts_frozen"] == 2
    assert store.halted() is None
    status = health(store, at=clock.at)
    assert status["state"] == "YELLOW"
    assert status["warnings"] == ["GUARD_REARMING"]
    assert status["guardrail"] == {**result["counts"]["guardrail"], "state": "GUARD_REARMING"}


def test_guard_rearms_after_300_settled_forecasts_frozen_strictly_after_resume(setup):
    store, clock, _, model = setup
    store.halt({"code": "LIVE_BUG_GUARD", "observed_utc": "2026-08-01T00:00:00Z"})
    resume(store, "Reviewed the previous halt", clock.at)
    observed = datetime(2027, 8, 1, tzinfo=timezone.utc)
    kickoff = datetime(2026, 8, 3, tzinfo=timezone.utc)
    seed_bad_forecasts(store, model, 1, 1, frozen_at=clock.at,
                       kickoff=kickoff, observed_at=observed)
    clock.at += timedelta(seconds=1)
    seed_bad_forecasts(store, model, 2, 299, frozen_at=clock.at,
                       kickoff=kickoff, observed_at=observed)
    clock.at = observed
    result = run_once(store, Client(), clock=clock)
    assert result["state"] == "SUCCESS"
    assert result["counts"]["guardrail"]["state"] == "COLLECTING"
    assert result["counts"]["guardrail"]["n"] == 299

    seed_bad_forecasts(store, model, 301, 1, frozen_at="2026-08-02T00:00:00Z",
                       kickoff=kickoff + timedelta(days=299), observed_at=observed)
    status = health(store, at=clock.at)
    assert status["guardrail"]["state"] == "ARMED"
    assert status["guardrail"]["n"] == 300
    assert "GUARD_REARMING" not in status["warnings"]
    result = run_once(store, Client(), clock=clock)
    assert result["state"] == "HALTED"
    assert result["counts"]["guardrail"]["n"] == 300


def test_guard_settlement_selection_matches_store_forecasts(setup):
    store, clock, _, model = setup
    before = clock.at
    after = before + timedelta(seconds=1)
    kickoff = before + timedelta(days=1)
    observed = kickoff + timedelta(days=1)
    with store.transaction():
        for fixture_id in range(1, 9):
            row = fixture(fixture_id, kickoff)
            frozen = before if fixture_id == 1 else after
            store.ingest([row], frozen)
            assert store.freeze(row, frozen, .1, 4., model.model_id, MARKET_VERSION, "x" * 64, None)
            result = dict(row, status="FINISHED", home_goals=1, away_goals=0)
            if fixture_id in (1, 2, 6, 7):
                if fixture_id == 6:
                    result["kickoff_utc"] = before
                store.ingest([result], observed)
            if fixture_id in (4, 5, 7):
                status = {4: "CANCELLED", 5: "AWARDED", 7: "TIMED"}[fixture_id]
                store.ingest([dict(row, status=status)], observed + timedelta(seconds=1))
            if fixture_id == 8:
                # Current fixture status can be present without a result row.
                store.db.execute("UPDATE fixtures SET status = 'FINISHED' WHERE id = ?", (fixture_id,))

    forecasts = store.forecasts()
    assert [entry["settlement"] for entry in forecasts] == [
        "SETTLED", "SETTLED", "PENDING", "VOID", "VOID",
        "INVALID_RESCHEDULED_BEFORE_FREEZE", "PENDING", "PENDING",
    ]
    for cutoff, expected in ((None, [1, 2]), (utc(before), [2]), (utc(after), [])):
        settled = [entry["fixture_id"] for entry in forecasts
                   if entry["settlement"] == "SETTLED" and (cutoff is None or entry["frozen_utc"] > cutoff)]
        assert [row["fixture_id"] for row in _settled_guard_rows(store, cutoff)] == settled == expected

