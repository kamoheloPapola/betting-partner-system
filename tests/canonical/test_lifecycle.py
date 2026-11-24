import json
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import sqlite3
import pytest
from src.canonical.clock import CanonicalError, canonical
from src.canonical import lifecycle
from src.canonical.lifecycle import validate_adoption,resume
from src.canonical.model import Model
from src.canonical.store import Store
from src.canonical.cli import main
from test_model import payload
from test_team_mapping_consumer import binding, publication, publish


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
def adoption_candidate(tmp_path, monkeypatch, publication):
    path = tmp_path / "model.json"
    model_payload = payload()
    model_payload["baseline_1x2"] = {league: [.4, .3, .3] for league in lifecycle.LEAGUES}
    _, mapping, _, _, config = publication
    mapping["entries"][0] = binding(81, "BARCELONA")
    model_payload["feature_seed"] = config["feature_seed"]
    monkeypatch.setattr(lifecycle, "ROOT", tmp_path)
    expected = {}
    references = {}
    mapping["entries"].append(binding(80, "ESPANYOL"))
    config["feature_seed"]["states"].append({
        **deepcopy(config["feature_seed"]["states"][0]), "team_id": "ESPANYOL",
    })

    def proof(name, raw):
        filename = lifecycle._FOUNDATION_PROOFS[name]
        source = tmp_path / filename
        source.write_bytes(raw)
        expected[name] = hashlib.sha256(raw).hexdigest()
        references[name] = {"path": str(source), "sha256": expected[name]}

    proof("source_manifest", config["source_manifest"].read_bytes())
    proof("identity_decisions", config["identity_decisions"].read_bytes())
    seed_pin = hashlib.sha256((canonical(model_payload["feature_seed"]) + "\n").encode()).hexdigest()
    proof("history_bridge", json.dumps({"status": "CANDIDATE_COMPLETE", "purpose": "FEATURE_AND_RATING_STATE_ONLY",
                                       "candidate_seed_sha256": seed_pin}).encode())
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
    model_payload["training"] = {"source_manifest_sha256": proofs[-1]["sha256"]}
    gate = {
        "status": "PASS_LOCKED_REPLAY",
        "identity_status": "PROVIDER_IDENTITY_COMPLETE",
        "history_status": "VERIFIED_CONTINUOUS_PRIOR_HISTORY",
        "foundation_proofs": references,
        "proofs": proofs,
    }

    def save(*, allow=True, baseline=None, release=None, edit_serving=None):
        publish(publication)
        proof("bridge_baseline_mapping", json.dumps(mapping if baseline is None else baseline).encode())
        if release is not None:
            decision = {"schema_version": 1, "status": "APPROVED", "league": "PD", "provider_id": 81,
                        "legacy_label": "BARCELONA", "binding_disposition": "BIND_LATEST_STATE",
                        "approved_by": "Synthetic owner", "approved_on": "2026-11-22",
                        "history_bridge_sha256": expected["history_bridge"],
                        "baseline_mapping_sha256": expected["bridge_baseline_mapping"],
                        "released_mapping_sha256": config["mapping_sha256"], **release}
            proof("barcelona_release", json.dumps(decision).encode())
        model_payload["serving"] = {
            "source_manifest_sha256": expected["source_manifest"],
            "team_mapping": {
                "mapping_sha256": config["mapping_sha256"],
                "accepted_input_sha256": dict(config["accepted_input_sha256"]),
            },
        }
        if edit_serving is not None:
            edit_serving(model_payload["serving"])
        model = Model(model_payload)
        raw = json.dumps({"model": model_payload, "validation": {**gate, "model_id": model.model_id}}).encode()
        path.write_bytes(raw)
        sha256 = hashlib.sha256(raw).hexdigest()
        monkeypatch.setattr(lifecycle, "_REVIEWED_ADOPTION_PINS", {sha256: expected} if allow else {})
        return path, sha256

    return save, model_payload, references


def test_adoption_with_independently_reviewed_pins_passes(adoption_candidate):
    save, _, _ = adoption_candidate
    model, pin, gate = validate_adoption(*save())
    assert len(pin["sha256"]) == 64
    training = model.payload["training"]
    assert training["source_manifest_sha256"] != gate["foundation_proofs"]["source_manifest"]["sha256"]
    assert "identity_decisions_sha256" not in training


def _restricted_barcelona_baseline(mapping):
    baseline = deepcopy(mapping)
    baseline["entries"][0].update(disposition="OWNER_SEED_WINDOW_ONLY", valid_through="2024-06-02")
    return baseline


def test_production_allowlist_contains_exactly_the_reviewed_v6_tuple():
    assert lifecycle._REVIEWED_ADOPTION_PINS == {
        "29b1fcf21f78ea86efbf653a54a98d5f877c327fe32190f4eb64dc58ce8f6b78": {
            "barcelona_release": "e1d171d6641e4a68327af26c9a4a60d3efd19edf687f92082d087692cc049772",
            "bridge_baseline_mapping": "1b5efbd10f6c09599eb822da15bdaaac01653a3e16045652fc5941a54fa8d5a7",
            "history_bridge": "594efcb2bbe65c6aed7894e18ae80f8fcf333f00b96d339095fe0a5f3823220f",
            "identity_decisions": "709246734bb47d7616651e0f5e5951f72d361c144b0b78357b94548100e1d163",
            "source_manifest": "4fa8fa0fd8c0021a94ad41be9ad2f20b4a9eaa954239907ed95a502a8f0249a3",
        },
    }


@pytest.mark.parametrize("field", [
    "artifact_sha256", "source_manifest", "identity_decisions", "history_bridge",
    "bridge_baseline_mapping", "barcelona_release",
])
def test_adoption_refuses_one_changed_reviewed_tuple_field(adoption_candidate, publication, monkeypatch, field):
    save, _, _ = adoption_candidate
    path, pin = save(baseline=_restricted_barcelona_baseline(publication[1]), release={})
    validate_adoption(path, pin)

    if field == "artifact_sha256":
        changed = path.read_bytes() + b"\n"
        path.write_bytes(changed)
        with pytest.raises(CanonicalError, match="^ADOPTION_FOUNDATION_PINS_UNAVAILABLE$"):
            validate_adoption(path, hashlib.sha256(changed).hexdigest())
    else:
        expected = lifecycle._REVIEWED_ADOPTION_PINS[pin]
        monkeypatch.setattr(lifecycle, "_REVIEWED_ADOPTION_PINS", {pin: {**expected, field: "0" * 64}})
        with pytest.raises(CanonicalError, match=f"^ADOPTION_FOUNDATION_HASH_MISMATCH: {field}$"):
            validate_adoption(path, pin)


@pytest.mark.parametrize("fault, code", [
    ("none", None),
    ("missing-decision", "ADOPTION_MAPPING_RELEASE_DECISION_REQUIRED"),
    ("wrong-receipt", "ADOPTION_MAPPING_RELEASE_DECISION_INVALID"),
    ("wrong-release-pin", "ADOPTION_MAPPING_RELEASE_DECISION_INVALID"),
    ("espanyol-row", "ADOPTION_MAPPING_RELEASE_DELTA_INVALID"),
    ("other-row", "ADOPTION_MAPPING_RELEASE_DELTA_INVALID"),
    ("barcelona-extra-field", "ADOPTION_MAPPING_RELEASE_DELTA_INVALID"),
    ("input-pins", "ADOPTION_MAPPING_RELEASE_DELTA_INVALID"),
])
def test_adoption_accepts_only_an_independently_pinned_barcelona_release(adoption_candidate, publication, fault, code):
    save, _, _ = adoption_candidate
    _, mapping, _, _, _ = publication
    baseline = _restricted_barcelona_baseline(mapping)
    release = {}
    if fault == "missing-decision":
        release = None
    elif fault == "wrong-receipt":
        release["history_bridge_sha256"] = "0" * 64
    elif fault == "wrong-release-pin":
        release["released_mapping_sha256"] = "0" * 64
    elif fault in ("espanyol-row", "other-row", "barcelona-extra-field"):
        target = {"espanyol-row": 80, "other-row": 2, "barcelona-extra-field": 81}[fault]
        next(row for row in mapping["entries"] if row["provider_id"] == target)["extra"] = True
    elif fault == "input-pins":
        mapping["input_sha256"]["proposals"] = "f" * 64
    args = save(baseline=baseline, release=release)
    if code:
        with pytest.raises(CanonicalError, match=code):
            validate_adoption(*args)
    else:
        validate_adoption(*args)


def test_adoption_recomputes_embedded_seed_hash_from_bridge_receipt(adoption_candidate):
    save, model, _ = adoption_candidate
    model["feature_seed"]["states"][0]["state"]["elo"] += 1
    with pytest.raises(CanonicalError, match="ADOPTION_BRIDGE_SEED_HASH_MISMATCH"):
        validate_adoption(*save())


def test_adoption_refuses_any_seed_window_only_binding(adoption_candidate, publication):
    _, mapping, _, _, _ = publication
    mapping["entries"][1].update(disposition="OWNER_SEED_WINDOW_ONLY", valid_through="2024-06-01")
    save, _, _ = adoption_candidate
    with pytest.raises(CanonicalError, match="ADOPTION_SEED_WINDOW_ONLY_BINDING"):
        validate_adoption(*save())


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
def test_complete_claims_cannot_override_a_mismatched_serving_pin(adoption_candidate, name):
    save, _, _ = adoption_candidate

    def edit(serving):
        if name == "source_manifest":
            serving["source_manifest_sha256"] = "0" * 64
        else:
            serving["team_mapping"]["accepted_input_sha256"]["decision_register"] = "0" * 64

    with pytest.raises(CanonicalError, match=f"ADOPTION_SERVING_INPUT_HASH_MISMATCH: {name}"):
        validate_adoption(*save(edit_serving=edit))


def test_adoption_binds_training_pin_to_locked_replay_manifest(adoption_candidate):
    save, model_payload, _ = adoption_candidate
    model_payload["training"]["source_manifest_sha256"] = "0" * 64
    with pytest.raises(CanonicalError, match="ADOPTION_REPLAY_MANIFEST_HASH_MISMATCH"):
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


