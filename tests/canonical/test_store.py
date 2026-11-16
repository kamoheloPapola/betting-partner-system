import sqlite3
import pytest
from src.canonical.store import Store
from src.canonical.clock import CanonicalError, utc

T0="2026-01-01T00:00:00Z"
T1="2026-01-02T15:00:00Z"
T2="2026-01-03T00:00:00Z"
T3="2026-01-04T00:00:00Z"


def fixture(mid=1, **changes):
    data={"id":mid,"league":"PL","season":2025,"home_id":57,"away_id":61,
          "home_name":"Home","away_name":"Away","kickoff_utc":T1,"status":"TIMED"}
    data.update(changes)
    return data


@pytest.fixture
def db(tmp_path):
    with Store(tmp_path/"canonical.sqlite3") as store:
        store.register_model("TEST",tmp_path/"model.json","a"*64,T0)
        store.activate_model("TEST",T0,"SYNTHETIC_STORAGE_TEST")
        with store.transaction():
            store.ingest([fixture()],T0)
        yield store


def freeze(db, at=T0, **kwargs):
    f=dict(db.db.execute("SELECT * FROM fixtures WHERE id=1").fetchone())
    return db.freeze(f,at,1.4,1.1,"TEST","goals-v1","f"*64,kwargs.get("known"))


def test_first_forecast_insert_only_and_idempotent(db):
    assert freeze(db)
    assert not freeze(db,at="2026-01-01T01:00:00Z")
    assert len(db.forecasts())==1
    assert db.forecasts()[0]["frozen_utc"]==utc(T0)


@pytest.mark.parametrize("sql",[
    "UPDATE forecasts SET home_xg=9",
    "DELETE FROM forecasts",
    "INSERT OR REPLACE INTO forecasts SELECT * FROM forecasts",
])
def test_sql_guard_cannot_be_bypassed_by_other_connection(db,sql):
    freeze(db)
    with sqlite3.connect(db.path) as other:
        with pytest.raises(sqlite3.IntegrityError):
            other.execute(sql)
    assert db.forecasts()[0]["home_xg"]==1.4


def test_no_post_kickoff_forecast(db):
    with pytest.raises(sqlite3.IntegrityError):
        freeze(db,at=T2)


def test_unconfirmed_or_inplay_fixture_not_eligible(db):
    with db.transaction():db.ingest([fixture(status="IN_PLAY")],T0)
    with pytest.raises(sqlite3.IntegrityError):freeze(db)


def test_future_known_history_refused(db):
    with pytest.raises(sqlite3.IntegrityError):freeze(db,known=T2)


def test_result_revision_and_asof_history(db):
    freeze(db)
    with db.transaction():
        c=db.ingest([fixture(status="FINISHED",home_goals=2,away_goals=1)],T2)
    assert c["results_added"]==1
    with db.transaction():
        c=db.ingest([fixture(status="FINISHED",home_goals=1,away_goals=1)],T3)
    assert c["results_corrected"]==1
    assert len(db.history(T0))==0
    assert db.history(T2)[0]["home_goals"]==2
    assert db.history(T3)[0]["home_goals"]==1
    assert db.forecasts()[0]["result_revision"]==2


def test_same_result_repeat_no_duplicate_or_churn(db):
    item=fixture(status="FINISHED",home_goals=2,away_goals=1)
    with db.transaction():db.ingest([item],T2)
    with db.transaction():counts=db.ingest([item],T3)
    assert counts=={"fixtures_changed":0,"results_added":0,"results_corrected":0}
    assert db.db.execute("SELECT COUNT(*) FROM results").fetchone()[0]==1


def test_failed_batch_rolls_back_every_business_write(db):
    with pytest.raises(CanonicalError):
        with db.transaction():
            db.ingest([fixture(2),fixture(3,status="FINISHED")],T2)
    assert db.db.execute("SELECT COUNT(*) FROM fixtures").fetchone()[0]==1


def test_reschedule_keeps_first_forecast_and_marks_invalid_earlier_kickoff(db):
    freeze(db)
    with db.transaction():
        db.ingest([fixture(kickoff_utc="2025-12-31T15:00:00Z",status="FINISHED",home_goals=1,away_goals=0)],T2)
    assert db.forecasts()[0]["settlement"]=="INVALID_RESCHEDULED_BEFORE_FREEZE"
    assert db.forecasts()[0]["kickoff_utc"]==utc(T1)


def test_team_rebinding_after_forecast_is_rejected(db):
    freeze(db)
    with pytest.raises(CanonicalError,match="PROTECTED_FIXTURE"):
        with db.transaction():db.ingest([fixture(home_id=99)],T2)


def test_cancelled_never_settled(db):
    freeze(db)
    with db.transaction():db.ingest([fixture(status="CANCELLED")],T2)
    assert db.forecasts()[0]["settlement"]=="VOID"
    assert not db.history(T2)


def test_local_backup_is_consistent_and_exclusive(db,tmp_path):
    freeze(db)
    dest=db.backup(tmp_path/"backup.sqlite3")
    with Store(dest,readonly=True) as other:assert len(other.forecasts())==1
    with pytest.raises(CanonicalError):db.backup(dest)


def test_naive_timestamp_refused():
    with pytest.raises(CanonicalError):utc("2026-01-01T00:00:00")

