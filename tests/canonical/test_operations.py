import os
import sqlite3
import pytest
from src.canonical.clock import CanonicalError
from src.canonical.credentials import save_key,load_key,key_path
from src.canonical.store import Store

@pytest.mark.skipif(os.name!="nt",reason="Windows current-user DPAPI")
def test_protected_key_roundtrip_never_persists_plaintext(tmp_path):
    database=tmp_path/"store.sqlite3";key="SYNTHETIC_NONPROVIDER_KEY_12345678"
    result=save_key(database,key)
    assert result["plaintext_written"] is False
    assert key.encode() not in key_path(database).read_bytes()
    assert load_key(database)==key
    before=key_path(database).read_bytes()
    with pytest.raises(FileExistsError):save_key(database,"OTHER_SYNTHETIC_KEY_1234567890")
    assert key_path(database).read_bytes()==before
    assert not database.exists()


def test_missing_environment_key_writes_nothing(tmp_path):
    with pytest.raises(CanonicalError,match="ENVIRONMENT_KEY"):save_key(tmp_path/"store.sqlite3",None)
    assert list(tmp_path.iterdir())==[]


def test_corrupt_protected_key_fails_closed(tmp_path):
    database=tmp_path/"s.db";key_path(database).write_bytes(b"not a DPAPI blob")
    with pytest.raises(CanonicalError):load_key(database)


def test_overlapping_writer_is_refused_without_partial_business_data(tmp_path):
    db=tmp_path/"s.db"
    with Store(db) as first,Store(db) as second:
        second.db.execute("PRAGMA busy_timeout=30")
        with first.transaction():
            with pytest.raises(sqlite3.OperationalError,match="locked"):
                with second.transaction():pass
        assert first.db.execute("SELECT COUNT(*) FROM fixtures").fetchone()[0]==0


def test_fixture_metadata_history_cannot_be_rewritten(tmp_path):
    with Store(tmp_path/"s.db") as store:
        row={"id":1,"league":"PL","season":2026,"home_id":57,"away_id":61,
          "home_name":"Home","away_name":"Away","kickoff_utc":"2026-08-03T15:00:00Z","status":"TIMED"}
        with store.transaction():store.ingest([row],"2026-08-01T00:00:00Z")
        with pytest.raises(sqlite3.IntegrityError):
            store.db.execute("UPDATE fixture_revisions SET metadata_json='{}'")
        with pytest.raises(sqlite3.IntegrityError):
            store.db.execute("DELETE FROM fixture_revisions")


def test_stalled_run_cannot_appear_healthy(tmp_path):
    from src.canonical.service import health
    with Store(tmp_path/"s.db") as store:
        store.start_run("2026-11-14T00:00:00Z")
        state=health(store,at="2026-11-14T00:16:00Z")
        assert state["state"]=="RED"
        assert "RUN_DID_NOT_FINISH_WITHIN_BOUND" in state["reasons"]


def test_backup_refuses_existing_file_without_changing_it(tmp_path):
    with Store(tmp_path/"s.db") as store:
        dest=tmp_path/"existing.sqlite3";dest.write_bytes(b"OWNER_BYTES")
        with pytest.raises(CanonicalError,match="DESTINATION_EXISTS"):store.backup(dest)
        assert dest.read_bytes()==b"OWNER_BYTES"

