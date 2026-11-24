from hashlib import sha256
import sqlite3
import pytest
from src.canonical.provider import project_matches
from src.canonical.store import SCHEMA, Store
from src.canonical.clock import CanonicalError, utc

T0="2026-01-01T00:00:00Z"
T1="2026-01-02T15:00:00Z"
T2="2026-01-03T00:00:00Z"
T3="2026-01-04T00:00:00Z"


def test_reviewed_sql_schema_matches_byte_for_byte():
    # v2 deliberately adds forecast run/waiver provenance and text run keys.
    # Pin the entire SQL string so every other byte keeps the strict parity check.
    assert sha256(SCHEMA.encode("utf-8")).hexdigest() == (
        "19fc67f425c0560c632fcd1ba6f999072503cecdb5b1a590b569636a2d356bd0"
    )


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
    return db.freeze(f,at,1.4,1.1,"TEST","goals-v1","f"*64,kwargs.get("known"), run_id=db.start_run(at))


def test_first_forecast_insert_only_and_idempotent(db):
    assert freeze(db)
    original, = db.forecasts()
    assert not freeze(db,at="2026-01-01T01:00:00Z")
    assert db.forecasts() == [original]
    assert original["frozen_utc"] == utc(T0)
    assert original["run_id"] != db.last_run()["id"]


def test_forecast_insert_without_run_id_is_rejected_by_schema(db):
    db.start_run(T0)
    with pytest.raises(sqlite3.IntegrityError, match=r"NOT NULL constraint failed: forecasts\.run_id"):
        db.db.execute("""INSERT INTO forecasts(fixture_id,frozen_utc,kickoff_utc,home_xg,away_xg,
                         model_id,market_version,feature_hash) VALUES(?,?,?,?,?,?,?,?)""",
                      (1, utc(T0), utc(T1), 1.4, 1.1, "TEST", "goals-v1", "f" * 64))
    assert db.forecasts() == []


def test_store_refuses_old_schema_without_attempting_a_migration(tmp_path):
    path = tmp_path / "v1.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA user_version=1")
    with pytest.raises(CanonicalError, match="UNSUPPORTED_DATABASE_SCHEMA"):
        Store(path)


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


@pytest.mark.parametrize("void_status", ["AWARDED", "CANCELLED"])
def test_live_projection_history_and_settlement_match_pre_refactor_contract(db, void_status):
    assert freeze(db)
    assert db.pending_seasons() == [("PL", 2025)]
    match = {
        "id": 1, "utcDate": T1, "season": {"startDate": "2025-08-01"},
        "homeTeam": {"id": 57, "name": "Home"}, "awayTeam": {"id": 61, "name": "Away"},
    }
    payload = {"competition": {"code": "PL"}, "resultSet": {"count": 1}, "matches": [match]}
    match.update(status="FINISHED", score={"fullTime": {"home": 2, "away": 1}})
    finished, = project_matches(payload, "PL", 2025, T2)
    assert (finished["home_goals"], finished["away_goals"]) == (2, 1)
    with db.transaction():
        assert db.ingest([finished], T2) == {"fixtures_changed": 1, "results_added": 1, "results_corrected": 0}
    settled, = db.forecasts()
    assert (settled["settlement"], settled["result_revision"]) == ("SETTLED", 1)

    match.update(status=void_status, score="must not be read")
    void, = project_matches(payload, "PL", 2025, T3)
    assert "home_goals" not in void and "away_goals" not in void
    with db.transaction():
        assert db.ingest([void], T3) == {"fixtures_changed": 1, "results_added": 0, "results_corrected": 0}
    history, = db.history(T2)
    assert (history["fixture_id"], history["home_goals"], history["away_goals"]) == (1, 2, 1)
    assert db.history(T3) == []
    settlement, = db.forecasts()
    assert (settlement["settlement"], settlement["result_revision"]) == ("VOID", 1)
    assert db.pending_seasons() == []


def test_local_backup_is_consistent_and_exclusive(db,tmp_path):
    freeze(db)
    dest=db.backup(tmp_path/"backup.sqlite3")
    with Store(dest,readonly=True) as other:assert len(other.forecasts())==1
    with pytest.raises(CanonicalError):db.backup(dest)


def test_naive_timestamp_refused():
    with pytest.raises(CanonicalError):utc("2026-01-01T00:00:00")

