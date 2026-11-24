"""Explicit adoption and recovery gates; never auto-promote a model."""
import hashlib
import json
import os
from pathlib import Path
import stat
from .clock import CanonicalError,canonical,now_utc,utc
from .model import read_artifact
from .provider import LEAGUES
from .provider_team_mapping import Refusal, barcelona_release_mapping
from .team_mapping_consumer import MAPPING_PATH, TeamMappingConsumer, _object, _read_pinned, _serving_contract


_RESUME_ACTION = "RESUME_AFTER_REVIEW"
ROOT = Path(__file__).resolve().parents[2]
_FOUNDATION_PROOFS = {
    "source_manifest": "source-authority-manifest.json",
    "identity_decisions": "historical-identity-decisions.json",
    "history_bridge": "history-bridge.json",
    "bridge_baseline_mapping": "provider-team-mapping.json",
    "barcelona_release": "barcelona-release.json",
}
_REQUIRED_FOUNDATION_PROOFS = _FOUNDATION_PROOFS.keys() - {"barcelona_release"}
# Keyed by the complete model artifact digest. Only reviewed evidence digests
# belong here. The candidate cannot supply this allowlist or approve itself.
_REVIEWED_ADOPTION_PINS = {
    "29b1fcf21f78ea86efbf653a54a98d5f877c327fe32190f4eb64dc58ce8f6b78": {
        "barcelona_release": "e1d171d6641e4a68327af26c9a4a60d3efd19edf687f92082d087692cc049772",
        "bridge_baseline_mapping": "1b5efbd10f6c09599eb822da15bdaaac01653a3e16045652fc5941a54fa8d5a7",
        "history_bridge": "594efcb2bbe65c6aed7894e18ae80f8fcf333f00b96d339095fe0a5f3823220f",
        "identity_decisions": "709246734bb47d7616651e0f5e5951f72d361c144b0b78357b94548100e1d163",
        "source_manifest": "4fa8fa0fd8c0021a94ad41be9ad2f20b4a9eaa954239907ed95a502a8f0249a3",
    },
}


def _foundation_path(raw_path, filename, proof_name):
    error = "ADOPTION_FOUNDATION_PATH_INVALID"
    if not isinstance(raw_path, str):
        raise CanonicalError(error, proof_name)
    source = Path(raw_path)
    if not source.is_absolute() or ".." in source.parts:
        raise CanonicalError(error, proof_name)
    try:
        for part in (*reversed(source.parents), source):
            info = part.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                raise CanonicalError(error, proof_name)
        resolved = source.resolve(strict=True)
    except (OSError, ValueError) as exc:
        raise CanonicalError(error, proof_name) from exc
    # Resolve legitimate Windows aliases such as different casing or 8.3 names.
    if not stat.S_ISREG(info.st_mode) or os.path.normcase(resolved.name) != os.path.normcase(filename):
        raise CanonicalError(error, proof_name)
    return resolved


def _verify_foundation(model, gate, artifact_sha256):
    expected = _REVIEWED_ADOPTION_PINS.get(artifact_sha256)
    if expected is None or not _REQUIRED_FOUNDATION_PROOFS <= expected.keys() <= _FOUNDATION_PROOFS.keys():
        raise CanonicalError("ADOPTION_FOUNDATION_PINS_UNAVAILABLE")

    references = gate.get("foundation_proofs")
    if not isinstance(references, dict) or references.keys() != expected.keys():
        raise CanonicalError("ADOPTION_FOUNDATION_PROOFS_MISSING")
    _, serving_pins = _serving_contract(model)
    held = {}
    for name, pin in expected.items():
        reference = references[name]
        if not isinstance(reference, dict) or reference.get("sha256") != pin:
            raise CanonicalError("ADOPTION_FOUNDATION_HASH_MISMATCH", name)
        if name in serving_pins and serving_pins[name] != pin:
            raise CanonicalError("ADOPTION_SERVING_INPUT_HASH_MISMATCH", name)
        source = _foundation_path(reference.get("path"), _FOUNDATION_PROOFS[name], name)
        with source.open("rb") as stream:
            raw = stream.read(16_000_001)
        if len(raw) > 16_000_000 or hashlib.sha256(raw).hexdigest() != pin:
            raise CanonicalError("ADOPTION_FOUNDATION_HASH_MISMATCH", name)
        held[name] = raw
    return held


def _verify_bridge_seed(model, raw):
    receipt = _object(raw, "ADOPTION_BRIDGE_RECEIPT_INVALID")
    if receipt.get("status") != "CANDIDATE_COMPLETE" or receipt.get("purpose") != "FEATURE_AND_RATING_STATE_ONLY":
        raise CanonicalError("ADOPTION_BRIDGE_RECEIPT_INVALID")
    # The bridge writes canonical JSON plus one newline for candidate-seed.json.
    seed_pin = hashlib.sha256((canonical(model.seed) + "\n").encode("utf-8")).hexdigest()
    if receipt.get("candidate_seed_sha256") != seed_pin:
        raise CanonicalError("ADOPTION_BRIDGE_SEED_HASH_MISMATCH")


def _verify_mapping_release(model, held):
    baseline_raw = held["bridge_baseline_mapping"]
    baseline_pin = hashlib.sha256(baseline_raw).hexdigest()
    released_pin = model.payload["serving"]["team_mapping"]["mapping_sha256"]
    if released_pin == baseline_pin:
        return
    if "barcelona_release" not in held:
        raise CanonicalError("ADOPTION_MAPPING_RELEASE_DECISION_REQUIRED")
    decision = _object(held["barcelona_release"], "ADOPTION_MAPPING_RELEASE_DECISION_INVALID")
    expected = {
        "schema_version": 1, "status": "APPROVED", "league": "PD", "provider_id": 81,
        "legacy_label": "BARCELONA", "binding_disposition": "BIND_LATEST_STATE",
        "history_bridge_sha256": hashlib.sha256(held["history_bridge"]).hexdigest(),
        "baseline_mapping_sha256": baseline_pin, "released_mapping_sha256": released_pin,
    }
    if (any(decision.get(key) != value for key, value in expected.items())
            or not all(isinstance(decision.get(key), str) and decision[key].strip()
                       for key in ("approved_by", "approved_on"))):
        raise CanonicalError("ADOPTION_MAPPING_RELEASE_DECISION_INVALID")
    baseline = _object(baseline_raw, "ADOPTION_BRIDGE_MAPPING_INVALID")
    released = _object(_read_pinned(ROOT / MAPPING_PATH, released_pin, "TEAM_MAPPING"), "TEAM_MAPPING_INVALID")
    try:
        expected_mapping = barcelona_release_mapping(baseline)
    except Refusal as exc:
        raise CanonicalError("ADOPTION_BRIDGE_MAPPING_INVALID") from exc
    if canonical(released) != canonical(expected_mapping):
        raise CanonicalError("ADOPTION_MAPPING_RELEASE_DELTA_INVALID")


def validate_adoption(path,sha256):
    if len(sha256)!=64 or any(c not in "0123456789abcdef" for c in sha256):
        raise CanonicalError("EXPECTED_MODEL_SHA256_REQUIRED")
    model,pin,gate=read_artifact(path,sha256)
    # Retrospective skill alone cannot authorize incomplete live identity/history.
    if gate.get("identity_status")!="PROVIDER_IDENTITY_COMPLETE":
        raise CanonicalError("LIVE_IDENTITY_FOUNDATION_INCOMPLETE")
    if gate.get("history_status")!="VERIFIED_CONTINUOUS_PRIOR_HISTORY":
        raise CanonicalError("LIVE_HISTORY_BRIDGE_INCOMPLETE")
    held = _verify_foundation(model, gate, pin["sha256"])
    _verify_bridge_seed(model, held["history_bridge"])
    mapping = TeamMappingConsumer.for_model(ROOT, model, gate)
    _verify_mapping_release(model, held)
    mapping.require_unrestricted_bindings()
    if set(model.payload["baseline_1x2"])!=set(LEAGUES):
        raise CanonicalError("MODEL_LEAGUE_COVERAGE_INCOMPLETE")
    proofs=gate.get("proofs",[])
    required={"result.json","feature-parity.json","candidate-manifest.json"}
    if len(proofs)!=3 or {Path(p["path"]).name for p in proofs}!=required:
        raise CanonicalError("ADOPTION_PROOF_SET_INCOMPLETE")
    for proof in proofs:
        source=Path(proof["path"])
        with source.open("rb") as stream:
            raw=stream.read(16_000_001)
        if len(raw)>16_000_000 or len(raw)!=proof["size_bytes"] or hashlib.sha256(raw).hexdigest()!=proof["sha256"]:
            raise CanonicalError("ADOPTION_PROOF_HASH_MISMATCH")
        if source.name=="result.json":
            result=json.loads(raw)
            if result.get("status")!="PASS_LOCKED_REPLAY" or result.get("n")!=8332 or len(result.get("folds",[]))!=22:
                raise CanonicalError("LOCKED_REPLAY_PROOF_INVALID")
            if not result["E_minus_R"]["upper"] < -result["margin"]:
                raise CanonicalError("LOCKED_REPLAY_MARGIN_FAILED")
        elif source.name=="feature-parity.json" and json.loads(raw).get("status")!="PASS":
            raise CanonicalError("FEATURE_PARITY_PROOF_INVALID")
        elif source.name == "candidate-manifest.json":
            training = model.payload.get("training")
            if not isinstance(training, dict) or training.get("source_manifest_sha256") != proof["sha256"]:
                raise CanonicalError("ADOPTION_REPLAY_MANIFEST_HASH_MISMATCH")
    return model,pin,gate


def adopt(store,path,sha256,reason,at):
    if not isinstance(reason,str) or not reason.strip():raise CanonicalError("OWNER_REASON_REQUIRED")
    model,pin,gate=validate_adoption(path,sha256)
    with store.transaction():
        store.register_model(model.model_id,pin["path"],pin["sha256"],at)
        store.activate_model(model.model_id,at,reason.strip())
    return {"state":"MODEL_ADOPTED","model_id":model.model_id,"artifact":pin}


def rollback(store,reason,at):
    if not isinstance(reason,str) or not reason.strip():raise CanonicalError("OWNER_REASON_REQUIRED")
    current=store.active_model()["id"]
    previous=store.db.execute("""SELECT m.* FROM model_activations a JOIN models m ON m.id=a.model_id
      WHERE a.model_id<>? ORDER BY a.sequence DESC LIMIT 1""",(current,)).fetchone()
    if previous is None:raise CanonicalError("NO_PREVIOUS_MODEL")
    validate_adoption(previous["artifact_path"],previous["artifact_sha256"])
    with store.transaction():store.activate_model(previous["id"],at,"ROLLBACK: "+reason.strip())
    return {"state":"MODEL_ROLLED_BACK","model_id":previous["id"]}


def last_resume_utc(store):
    row = store.db.execute(
        """SELECT observed_utc FROM operator_decisions
           WHERE action = ? ORDER BY sequence DESC LIMIT 1""",
        (_RESUME_ACTION,),
    ).fetchone()
    return utc(row[0]) if row else None


def resume(store,reason,at):
    if not isinstance(reason,str) or not reason.strip():raise CanonicalError("OWNER_REASON_REQUIRED")
    resumed_at = utc(at)
    with store.transaction():
        if resumed_at > now_utc():
            raise CanonicalError("RESUME_IN_FUTURE")
        previous = store.halted()
        if previous is None:
            raise CanonicalError("NO_HALT_TO_REVIEW")
        prior_resume = last_resume_utc(store)
        halted_at = previous.get("observed_utc")
        if ((prior_resume and resumed_at < prior_resume)
                or (halted_at and resumed_at < utc(halted_at))):
            raise CanonicalError("RESUME_BEFORE_GUARD_STATE")
        store.db.execute("INSERT INTO operator_decisions(observed_utc,action,reason,prior_json) VALUES(?,?,?,?)",
                         (resumed_at,_RESUME_ACTION,reason.strip(),canonical(previous)))
        store.db.execute("DELETE FROM metadata WHERE key='halt'")
    return {"state":"HALT_REVIEW_RECORDED",
            "guardrail":"Collecting forecasts frozen after this resume",
            "window_started_after_utc":resumed_at}

