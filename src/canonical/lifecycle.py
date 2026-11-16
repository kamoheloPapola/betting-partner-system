"""Explicit adoption and recovery gates; never auto-promote a model."""
import hashlib
import json
import os
from pathlib import Path
import stat
from .clock import CanonicalError,canonical,now_utc,utc
from .model import read_artifact
from .provider import LEAGUES


_RESUME_ACTION = "RESUME_AFTER_REVIEW"
_FOUNDATION_PROOFS = {
    "source_manifest": "source-authority-manifest.json",
    "identity_decisions": "historical-identity-decisions.json",
    "history_bridge": "history-bridge.json",
}
# Keyed by the complete model artifact digest. Only reviewed evidence digests
# belong here. The candidate cannot supply this allowlist or approve itself.
_REVIEWED_ADOPTION_PINS = {}


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
    if expected is None or set(expected) != set(_FOUNDATION_PROOFS):
        raise CanonicalError("ADOPTION_FOUNDATION_PINS_UNAVAILABLE")

    references = gate.get("foundation_proofs")
    if not isinstance(references, dict) or set(references) != set(_FOUNDATION_PROOFS):
        raise CanonicalError("ADOPTION_FOUNDATION_PROOFS_MISSING")
    training = model.payload.get("training", {})
    for name, filename in _FOUNDATION_PROOFS.items():
        pin = expected[name]
        reference = references[name]
        if not isinstance(reference, dict) or reference.get("sha256") != pin:
            raise CanonicalError("ADOPTION_FOUNDATION_HASH_MISMATCH", name)
        if name in ("source_manifest", "identity_decisions") and training.get(name + "_sha256") != pin:
            raise CanonicalError("ADOPTION_TRAINING_INPUT_HASH_MISMATCH", name)
        source = _foundation_path(reference.get("path"), filename, name)
        with source.open("rb") as stream:
            raw = stream.read(16_000_001)
        if len(raw) > 16_000_000 or hashlib.sha256(raw).hexdigest() != pin:
            raise CanonicalError("ADOPTION_FOUNDATION_HASH_MISMATCH", name)


def validate_adoption(path,sha256):
    if len(sha256)!=64 or any(c not in "0123456789abcdef" for c in sha256):
        raise CanonicalError("EXPECTED_MODEL_SHA256_REQUIRED")
    model,pin,gate=read_artifact(path,sha256)
    # Retrospective skill alone cannot authorize incomplete live identity/history.
    if gate.get("identity_status")!="PROVIDER_IDENTITY_COMPLETE":
        raise CanonicalError("LIVE_IDENTITY_FOUNDATION_INCOMPLETE")
    if gate.get("history_status")!="VERIFIED_CONTINUOUS_PRIOR_HISTORY":
        raise CanonicalError("LIVE_HISTORY_BRIDGE_INCOMPLETE")
    _verify_foundation(model, gate, pin["sha256"])
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

