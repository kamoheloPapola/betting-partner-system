import json
import hashlib
import os
from pathlib import Path
import sqlite3
import pytest
from src.canonical.clock import CanonicalError
from src.canonical import lifecycle
from src.canonical.lifecycle import validate_adoption,resume
from src.canonical.model import Model
from src.canonical.store import Store
from src.canonical.cli import main
from test_model import payload


def candidate(tmp_path,**gate):
    p=payload();model=Model(p);path=tmp_path/"model.json"
    path.write_text(json.dumps({"model":p,"validation":{"status":"PASS_LOCKED_REPLAY","model_id":model.model_id,**gate}}))
    return path,hashlib.sha256(path.read_bytes()).hexdigest()


def test_skill_does_not_override_incomplete_identities(tmp_path):
    path,sha=candidate(tmp_path,identity_status="PARTIAL_EXACT_BINDINGS_ONLY")
    with pytest.raises(CanonicalError,match="IDENTITY_FOUNDATION_INCOMPLETE"):validate_adoption(path,sha)


def test_identity_does_not_override_missing_history_bridge(tmp_path):
    path,sha=candidate(tmp_path,identity_status="PROVIDER_IDENTITY_COMPLETE")
    with pytest.raises(CanonicalError,match="HISTORY_BRIDGE_INCOMPLETE"):validate_adoption(path,sha)


def test_cli_refuses_candidate_without_creating_database(tmp_path,capsys):
    path,sha=candidate(tmp_path,identity_status="PARTIAL_EXACT_BINDINGS_ONLY")
    db=tmp_path/"new"/"store.sqlite3"
    assert main(["--database",str(db),"adopt",str(path),"--sha256",sha,"--reason","test"])==2
    assert not db.exists()
    assert json.loads(capsys.readouterr().out)["error_code"]=="LIVE_IDENTITY_FOUNDATION_INCOMPLETE"


def test_model_hash_must_be_supplied(tmp_path):
    path,_=candidate(tmp_path)
    with pytest.raises(CanonicalError,match="EXPECTED_MODEL_SHA256"):validate_adoption(path,"")


@pytest.fixture
def adoption_candidate(tmp_path, monkeypatch):
    path = tmp_path / "model.json"
    model_payload = payload()
    model_payload["baseline_1x2"] = {league: [.4, .3, .3] for league in lifecycle.LEAGUES}
    expected = {}
    references = {}
    for name, filename in lifecycle._FOUNDATION_PROOFS.items():
        source = tmp_path / filename
        raw = json.dumps({"synthetic_reviewed_evidence": name}).encode()
        source.write_bytes(raw)
        expected[name] = hashlib.sha256(raw).hexdigest()
        references[name] = {"path": str(source), "sha256": expected[name]}
    model_payload["training"] = {
        name + "_sha256": expected[name] for name in ("source_manifest", "identity_decisions")
    }
    proofs = []
    for filename, value in (
        ("result.json", {"status": "PASS_LOCKED_REPLAY", "n": 8332, "folds": ["synthetic"] * 22,
                         "E_minus_R": {"upper": -.02}, "margin": .006}),
        ("feature-parity.json", {"status": "PASS"}),
        ("candidate-manifest.json", {}),
    ):
        source = tmp_path / filename
        raw = json.dumps(value).encode()
        source.write_bytes(raw)
        proofs.append({"path": str(source), "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    gate = {
        "status": "PASS_LOCKED_REPLAY",
        "identity_status": "PROVIDER_IDENTITY_COMPLETE",
        "history_status": "VERIFIED_CONTINUOUS_PRIOR_HISTORY",
        "foundation_proofs": references,
        "proofs": proofs,
    }

    def save(*, allow=True):
        model = Model(model_payload)
        raw = json.dumps({"model": model_payload, "validation": {**gate, "model_id": model.model_id}}).encode()
        path.write_bytes(raw)
        sha256 = hashlib.sha256(raw).hexdigest()
        monkeypatch.setattr(lifecycle, "_REVIEWED_ADOPTION_PINS", {sha256: expected} if allow else {})
        return path, sha256

    return save, model_payload, references


def test_adoption_with_independently_reviewed_pins_passes(adoption_candidate):
    save, _, _ = adoption_candidate
    _, pin, _ = validate_adoption(*save())
    assert len(pin["sha256"]) == 64


@pytest.mark.skipif(os.name != "nt", reason="Windows real-path semantics")
def test_adoption_accepts_reviewed_file_via_windows_path_casing(adoption_candidate):
    save, _, references = adoption_candidate
    proof = references["source_manifest"]
    original = Path(proof["path"])
    alternate = original.with_name(original.name.upper())
    proof["path"] = alternate.drive.swapcase() + str(alternate)[len(alternate.drive):]
    assert proof["path"] != str(original)
    assert Path(proof["path"]).samefile(original)
    validate_adoption(*save())


def test_complete_claims_cannot_supply_missing_reviewed_pins(adoption_candidate):
    save, _, _ = adoption_candidate
    with pytest.raises(CanonicalError, match="FOUNDATION_PINS_UNAVAILABLE"):
        validate_adoption(*save(allow=False))


@pytest.mark.parametrize("name", ["source_manifest", "identity_decisions", "history_bridge"])
def test_complete_claims_are_refused_when_reviewed_file_changes(adoption_candidate, name):
    save, _, references = adoption_candidate
    args = save()
    Path(references[name]["path"]).write_text('{"tampered": true}', encoding="utf-8")
    with pytest.raises(CanonicalError, match=f"ADOPTION_FOUNDATION_HASH_MISMATCH: {name}"):
        validate_adoption(*args)


@pytest.mark.parametrize("name", ["source_manifest", "identity_decisions"])
def test_complete_claims_cannot_override_a_mismatched_training_pin(adoption_candidate, name):
    save, model_payload, _ = adoption_candidate
    model_payload["training"][name + "_sha256"] = "0" * 64
    with pytest.raises(CanonicalError, match=f"ADOPTION_TRAINING_INPUT_HASH_MISMATCH: {name}"):
        validate_adoption(*save())


def test_resume_requires_a_reason_and_preserves_original_halt(tmp_path):
    with Store(tmp_path/"s.db") as store:
        store.halt({"code":"SYNTHETIC","n":300})
        with pytest.raises(CanonicalError,match="OWNER_REASON"):resume(store," ","2026-11-14T00:00:00Z")
        assert store.halted()
        resume(store,"investigated fixture correction","2026-11-14T00:00:00Z")
        assert store.halted() is None
        row=store.db.execute("SELECT * FROM operator_decisions").fetchone()
        assert json.loads(row["prior_json"])["n"]==300
        with pytest.raises(sqlite3.IntegrityError):store.db.execute("DELETE FROM operator_decisions")


def test_future_resume_is_refused_without_clearing_halt(tmp_path, monkeypatch):
    now = "2026-11-14T12:00:00.000000Z"
    monkeypatch.setattr(lifecycle, "now_utc", lambda: now)
    with Store(tmp_path / "store.sqlite3") as store:
        halt = {"code": "LIVE_BUG_GUARD", "observed_utc": "2026-11-14T11:59:00.000000Z"}
        store.halt(halt)
        with pytest.raises(CanonicalError, match="RESUME_IN_FUTURE"):
            resume(store, "Inspected failure", "2026-11-14T12:00:00.000001Z")
        assert store.halted() == halt
        assert store.db.execute("SELECT COUNT(*) FROM operator_decisions").fetchone()[0] == 0
        assert resume(store, "Inspected failure", now)["window_started_after_utc"] == now


def test_missing_key_creates_no_database(tmp_path,monkeypatch,capsys):
    monkeypatch.delenv("FOOTBALL_DATA_API_KEY",raising=False)
    monkeypatch.setattr("src.canonical.cli.notify_local",lambda code:{"test_notification":code})
    db=tmp_path/"new"/"store.sqlite3"
    assert main(["--database",str(db),"run"])==2
    assert not db.exists()
    assert "KEY" in capsys.readouterr().out


