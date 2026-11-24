"""Assemble a review-only Model E artifact from the pinned v6 evidence."""

import argparse
from datetime import date, timedelta
from hashlib import sha256
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.canonical.clock import CanonicalError, canonical
from src.canonical.history_bridge import SEASON_CLUBS, SEED_DAY
from src.canonical.lifecycle import _FOUNDATION_PROOFS, _verify_bridge_seed, _verify_mapping_release
from src.canonical.model import Model
from src.canonical.provider_team_mapping import APPROVED_INPUTS, encode
from src.canonical.team_mapping_consumer import MAPPING_PATH, TeamMappingConsumer, _object, _read_pinned
from scripts.run_history_bridge import verify_approval_derivation


ARCHIVE = Path("tmp/BL1-T3-offline-dependency-manifest-binding-20261027-01/NEXT-38-system-implementation-01")
APPROVAL = Path("tmp/bridge-v6-approved-2026-11-22")
BRIDGE = Path("tmp/canonical-history-bridge")
BUNDLES = Path("data/canonical/serving")
PINNED_INPUTS = {
    "portable": (ARCHIVE / "replay/portable-unbound-model.json",
                 "577bacf580c78c6d1fc215a55b0961ee7357028c6550190c7b975bd22a2b33b1"),
    "archived": (ARCHIVE / "candidate/model-E.json",
                 "5f405f86c9bcbe4613cbc2a88afcded192ef1f7e2a1b81b8daad8319b05a029c"),
    "plan": (APPROVAL / "bridge-plan.APPROVED_FOR_BRIDGE.json",
             "e2884ae060ada16f3386d6ff2a4edd483a5b55c3273e6430ef21d72c95b90260"),
    "identity_review": (APPROVAL / "identity-review.APPROVED_FOR_BRIDGE.json",
                        "f89a85d1f7445613e99c6dc9a1886370a73d13bc7bec59559aace64704061f8f"),
    "seed": (BRIDGE / "candidate-seed.json", "3e83fd3657b7fed4561a00c119a35f6c1706ad6582de8510842a5443d6f399d2"),
    "history_bridge": (BRIDGE / "history-bridge.json", "594efcb2bbe65c6aed7894e18ae80f8fcf333f00b96d339095fe0a5f3823220f"),
    "mapping": (MAPPING_PATH, "6db652d621c614d4338994655c1672bb7d86214e0ac90037e7964af2a85b8346"),
    "barcelona_release": (Path("tmp/barcelona-v6-signed-release-2026-11-22/barcelona-release.json"),
                          "e1d171d6641e4a68327af26c9a4a60d3efd19edf687f92082d087692cc049772"),
    "authority_base": (ARCHIVE.parent / "NEXT-38d-source-authority-01/source-authority-manifest.json",
                       "73dd69514b7501a49a144bbf51e7bb42470aaa6886ec50c4532c438d7f481cc1"),
    "fixture_manifest": (BRIDGE / "reconstructed-fixture-ids.json",
                         "f6ecb03c7abc19a3c8b81b9c514fe933eeb2f41ffb1d8b3f859e77339efefbbb"),
}
PLAN_APPROVAL_DIFF = {"path": (APPROVAL / "approval-plan-diff.json").as_posix(),
                      "sha256": "3100f46df6c89769bd3161f58fcefe5bac9d0165d957cbd2d17778d0eb23613a"}


def reviewed_readiness(review, receipt):
    """Return readiness labels only after the approved v6 proofs pass."""
    bindings = review.get("seed_bindings")
    if (review.get("schema_version") != 4 or review.get("status") != "APPROVED_FOR_BRIDGE"
            or review.get("blockers") != [] or not isinstance(bindings, list) or not bindings):
        raise CanonicalError("SERVING_IDENTITY_REVIEW_INCOMPLETE")
    for row in bindings:
        if not isinstance(row, dict) or "seed_label" not in row:
            raise CanonicalError("SERVING_IDENTITY_REVIEW_INCOMPLETE")
        label = row["seed_label"]
        if not ((isinstance(label, str) and label.strip()) or
                (label is None and row.get("finding") == "NO_USABLE_STATE"
                 and row.get("reason") in ("NO_SEED_LABEL", "STALE_BEYOND_THRESHOLD"))):
            raise CanonicalError("SERVING_IDENTITY_REVIEW_INCOMPLETE")

    seasons = receipt.get("completed_seasons")
    if (receipt.get("status") != "CANDIDATE_COMPLETE"
            or receipt.get("purpose") != "FEATURE_AND_RATING_STATE_ONLY"
            or not isinstance(seasons, list)):
        raise CanonicalError("SERVING_HISTORY_INCOMPLETE")
    expected = {(league, season) for league in SEASON_CLUBS for season in (2024, 2025)}
    seen = set()
    for row in seasons:
        if not isinstance(row, dict) or not isinstance(row.get("league"), str) or type(row.get("season")) is not int:
            raise CanonicalError("SERVING_HISTORY_INCOMPLETE")
        league, season = row["league"], row["season"]
        key = league, season
        if key not in expected or key in seen:
            raise CanonicalError("SERVING_HISTORY_INCOMPLETE")
        clubs = SEASON_CLUBS[league]
        voids = row.get("accounted_void_fixtures")
        fixtures = row.get("fixtures")
        if (type(row.get("clubs")) is not int or row["clubs"] != clubs
                or type(fixtures) is not int or fixtures != clubs * (clubs - 1)
                or type(voids) is not int or not 0 <= voids <= fixtures):
            raise CanonicalError("SERVING_HISTORY_INCOMPLETE")
        seen.add(key)
    if seen != expected:
        raise CanonicalError("SERVING_HISTORY_INCOMPLETE")
    return {"identity_status": "PROVIDER_IDENTITY_COMPLETE",
            "history_status": "VERIFIED_CONTINUOUS_PRIOR_HISTORY"}


def _encoded(document):
    return (canonical(document) + "\n").encode("utf-8")


def _pin_ref(name):
    path, pin = PINNED_INPUTS[name]
    return {"path": path.as_posix(), "sha256": pin}


def _authority_review(base, schedule, raw_authority, source, review):
    if (schedule.get("status") != "RECONSTRUCTED_FROM_PINNED_METADATA"
            or schedule.get("bridge_receipt") != _pin_ref("history_bridge")
            or schedule.get("plan") != _pin_ref("plan")):
        raise CanonicalError("SERVING_AUTHORITY_DERIVATION_MISMATCH")
    endings = dict.fromkeys(SEASON_CLUBS, "")
    rows = [row for window in schedule["windows"] for row in window["fixtures"]]
    for row in [*rows, *schedule["excluded_fixtures"]]:
        day, league = row["day"], row["league"]
        if "2025-07-01" <= day < "2026-07-01":
            endings[league] = max(endings[league], day)
    additions = {
        "draft_base_authority": _pin_ref("authority_base"),
        "provider_season_max_dates": [
            {"league": league, "season": 2025, "max_date": day} for league, day in sorted(endings.items())
        ],
    }
    if not all(endings.values()) or base.keys() & additions.keys() or encode({**base, **additions}) != raw_authority:
        raise CanonicalError("SERVING_AUTHORITY_DERIVATION_MISMATCH")
    return {
        "serving_review_status": "PENDING_OWNER_REVIEW",
        "base_sha256": additions["draft_base_authority"]["sha256"],
        "serving_authority_sha256": source["sha256"],
        "changes": [{"op": "add", "path": f"/{key}", "value": value} for key, value in additions.items()],
        "metadata_evidence": _pin_ref("fixture_manifest"),
        "bridge_formal_approval": review["formal_approval"],
        "approved_bridge_plan": _pin_ref("plan"),
        "bridge_plan_approval_diff": PLAN_APPROVAL_DIFF,
        "bridge_receipt": _pin_ref("history_bridge"),
    }


def _artifact(archived, portable, seed, serving, validation):
    # The archived binder changed only the seed and its audit. Preserve the
    # archived training record while taking the learned model from the replay.
    original = archived["model"]
    expected = {**portable, "feature_seed": original["feature_seed"],
                "training": {**portable["training"], "identity_audit": original["training"]["identity_audit"]}}
    if canonical(expected) != canonical(original):
        raise CanonicalError("SERVING_FROZEN_MODEL_DIFF")
    model = Model({**portable, "training": original["training"], "feature_seed": seed, "serving": serving})
    model_fields = {key for key in original.keys() | model.payload.keys()
                    if canonical(original.get(key)) != canonical(model.payload.get(key))}
    if model_fields != {"feature_seed", "serving"}:
        raise CanonicalError("SERVING_MODEL_DIFF_OUT_OF_SCOPE")
    document = {"model": model.payload, "validation": {**archived["validation"], **validation,
                "model_id": model.model_id, "scope": "Fixed E replay with approved v6 serving bindings"}}
    return model, document


def assemble(output):
    """Read only pinned metadata and return the complete bundle as bytes."""
    paths = {name: ROOT / path for name, (path, _) in PINNED_INPUTS.items()}
    raw = {name: _read_pinned(paths[name], pin, f"SERVING_{name.upper()}")
           for name, (_, pin) in PINNED_INPUTS.items()}
    documents = {name: _object(value, f"SERVING_{name.upper()}_INVALID") for name, value in raw.items()}
    plan, review = documents["plan"], documents["identity_review"]
    receipt, seed = documents["history_bridge"], documents["seed"]
    readiness = reviewed_readiness(review, receipt)
    review_pin = PINNED_INPUTS["identity_review"][1]
    model_pin = PINNED_INPUTS["portable"][1]
    cutoff = receipt["through_exclusive"]
    if (plan["status"] != "APPROVED_FOR_BRIDGE"
            or plan["identity_review"]["sha256"] != review_pin or receipt["identity_review_sha256"] != review_pin
            or plan["model"]["sha256"] != model_pin or review["model_sha256"] != model_pin
            or receipt["plan_sha256"] != PINNED_INPUTS["plan"][1]
            or receipt["model_id"] != Model(documents["portable"]["model"]).model_id
            or cutoff != review["through_exclusive"] or cutoff != plan["through_exclusive"]
            or receipt["seed_day"] != SEED_DAY
            or date.fromisoformat(seed["last_day"]) + timedelta(days=1) != date.fromisoformat(cutoff)):
        raise CanonicalError("SERVING_BRIDGE_LINEAGE_MISMATCH")
    bindings = {(row["league"], str(row["provider_id"])) for row in review["seed_bindings"]}
    states = {(row["league"], row["team_id"]) for row in seed["states"]}
    if bindings != states or len(bindings) != len(review["seed_bindings"]) or len(states) != len(seed["states"]):
        raise CanonicalError("SERVING_REVIEW_SEED_COVERAGE_MISMATCH")

    source = plan["source_manifest"]
    if review["source_manifest_sha256"] != source["sha256"]:
        raise CanonicalError("SERVING_BRIDGE_LINEAGE_MISMATCH")
    verify_approval_derivation(ROOT, _pin_ref("plan"), review_pin, PLAN_APPROVAL_DIFF)
    register, register_pin = APPROVED_INPUTS["decision_register"]
    baseline_pin = documents["barcelona_release"]["baseline_mapping_sha256"]
    baseline = paths["mapping"].with_name(f"provider-team-mapping.{baseline_pin}.previous.json")
    evidence = {
        "source_manifest": _read_pinned(ROOT / source["path"], source["sha256"], "SERVING_SOURCE_AUTHORITY"),
        "identity_decisions": _read_pinned(register, register_pin, "SERVING_IDENTITY_DECISIONS"),
        "history_bridge": raw["history_bridge"],
        "bridge_baseline_mapping": _read_pinned(baseline, baseline_pin, "SERVING_BASELINE_MAPPING"),
        "barcelona_release": raw["barcelona_release"],
    }
    authority_review = _authority_review(documents["authority_base"], documents["fixture_manifest"],
                                         evidence["source_manifest"], source, review)
    inputs = {name: pin for name, (_, pin) in APPROVED_INPUTS.items()}
    serving = {"source_manifest_sha256": source["sha256"], "team_mapping": {
        "mapping_sha256": PINNED_INPUTS["mapping"][1], "accepted_input_sha256": inputs,
    }}
    proof_paths = {name: Path("proofs") / name / _FOUNDATION_PROOFS[name] for name in evidence}
    references = {name: {"path": str(output / proof_paths[name]), "sha256": sha256(value).hexdigest()}
                  for name, value in evidence.items()}
    archived = documents["archived"]
    replay_proofs = []
    replay_files = {}
    for proof in archived["validation"]["proofs"]:
        original = Path(proof["path"])
        value = _read_pinned(original, proof["sha256"], "SERVING_REPLAY_PROOF")
        if len(value) != proof["size_bytes"]:
            raise CanonicalError("SERVING_REPLAY_PROOF_SIZE_MISMATCH")
        if (original.name == "candidate-manifest.json"
                and proof["sha256"] != archived["model"]["training"]["source_manifest_sha256"]):
            raise CanonicalError("SERVING_REPLAY_MANIFEST_HASH_MISMATCH")
        path = Path("proofs/replay") / original.name
        replay_files[path] = value
        replay_proofs.append({**proof, "path": str(output / path), "relative_path": path.as_posix()})
    identity_path = Path("proofs/identity_review") / paths["identity_review"].name
    model, artifact = _artifact(archived, documents["portable"]["model"], seed, serving, {
        **readiness, "foundation_proofs": references, "proofs": replay_proofs,
        "identity_review": {"path": str(output / identity_path), "sha256": review_pin},
    })
    _verify_bridge_seed(model, evidence["history_bridge"])
    _verify_mapping_release(model, evidence)
    TeamMappingConsumer.for_model(ROOT, model, {"foundation_proofs": {
        "source_manifest": {**references["source_manifest"], "path": str(ROOT / source["path"])},
        "identity_decisions": {**references["identity_decisions"], "path": str(register)},
    }}).require_unrestricted_bindings()

    review_diff = {"archived_model_sha256": PINNED_INPUTS["archived"][1], "changes": []}
    for section in ("model", "validation"):
        before, after = archived[section], artifact[section]
        for field in sorted(before.keys() | after.keys()):
            old, new = before.get(field), after.get(field)
            if canonical(old) != canonical(new):
                if field == "feature_seed":
                    old, new = ({"sha256": sha256(_encoded(item)).hexdigest(),
                                 "states": len(item["states"]), "last_day": item["last_day"]} for item in (old, new))
                review_diff["changes"].append({"field": f"{section}.{field}", "before": old, "after": new})
    review_diff["unchanged_model_fields"] = sorted(archived["model"].keys() - {"feature_seed"})
    review_diff["artifact_sha256"] = sha256(_encoded(artifact)).hexdigest()
    authority_review["serving_artifact_sha256"] = review_diff["artifact_sha256"]
    files = {
        **{proof_paths[name]: value for name, value in evidence.items()},
        **replay_files,
        identity_path: raw["identity_review"],
        Path("proofs/source_manifest/source-authority-base.json"): raw["authority_base"],
        Path("source-authority-review.json"): _encoded(authority_review),
        Path("field-diff.json"): _encoded(review_diff),
    }
    artifact_path = Path("serving-model.json")
    files[artifact_path] = _encoded(artifact)
    manifest = "".join(f"{sha256(value).hexdigest()}  {path.as_posix()}\n"
                       for path, value in sorted(files.items()))
    return {**files, Path("bundle.sha256"): manifest.encode("utf-8")}


def build(output):
    """Write to a new review directory after verifying every input."""
    output = output.resolve()
    if output.parent != ROOT / BUNDLES:
        raise CanonicalError("SERVING_OUTPUT_OUTSIDE_REVIEW_AREA")
    if output.exists():
        raise FileExistsError(output)
    files = assemble(output)
    output.mkdir(parents=True)
    for path, value in files.items():
        destination = output / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(value)
    return sha256(files[Path("serving-model.json")]).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help=f"New versioned bundle directly under {BUNDLES.as_posix()}/")
    args = parser.parse_args(argv)
    try:
        digest = build(args.output)
    except (CanonicalError, OSError, ValueError, KeyError, TypeError) as error:
        print(canonical({"status": "BLOCKED", "error_code": getattr(error, "code", type(error).__name__)}))
        return 1
    print(canonical({"status": "BUILT_FOR_REVIEW", "sha256": digest, "path": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
