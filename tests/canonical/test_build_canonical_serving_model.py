from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest

from scripts import build_canonical_serving_model as builder
from src.canonical.clock import CanonicalError
from src.canonical.model import read_artifact
from test_model import payload


@pytest.fixture
def proofs():
    review = {"schema_version": 4, "status": "APPROVED_FOR_BRIDGE", "blockers": [],
              "seed_bindings": [{"league": "PD", "provider_id": 81, "seed_label": "BARCELONA"}]}
    receipt = {"status": "CANDIDATE_COMPLETE", "purpose": "FEATURE_AND_RATING_STATE_ONLY",
               "completed_seasons": [
                   {"league": league, "season": season, "clubs": clubs,
                    "fixtures": clubs * (clubs - 1), "accounted_void_fixtures": 0}
                   for league, clubs in builder.SEASON_CLUBS.items() for season in (2024, 2025)
               ]}
    return review, receipt


@pytest.mark.parametrize(("fault", "error_code"), [
    ("unapproved-review", "SERVING_IDENTITY_REVIEW_INCOMPLETE"),
    ("hold", "SERVING_IDENTITY_REVIEW_INCOMPLETE"),
    ("missing-blocker-list", "SERVING_IDENTITY_REVIEW_INCOMPLETE"),
    ("unresolved-binding", "SERVING_IDENTITY_REVIEW_INCOMPLETE"),
    ("interrupted", "SERVING_HISTORY_INCOMPLETE"),
    ("missing-season", "SERVING_HISTORY_INCOMPLETE"),
    ("duplicate-replaces-season", "SERVING_HISTORY_INCOMPLETE"),
    ("short-round-robin", "SERVING_HISTORY_INCOMPLETE"),
])
def test_failed_readiness_proof_leaves_no_artifact(tmp_path, monkeypatch, proofs, fault, error_code):
    review, receipt = proofs
    seasons = receipt["completed_seasons"]
    if fault == "unapproved-review":
        review["status"] = "UNAPPROVED"
    elif fault == "hold":
        review["blockers"] = ["Unresolved team"]
    elif fault == "missing-blocker-list":
        review.pop("blockers")
    elif fault == "unresolved-binding":
        review["seed_bindings"][0]["seed_label"] = None
    elif fault == "interrupted":
        receipt["status"] = "OPENING_INTERRUPTED"
    elif fault == "missing-season":
        seasons.pop()
    elif fault == "duplicate-replaces-season":
        seasons[-1] = seasons[0].copy()
    elif fault == "short-round-robin":
        seasons[0]["fixtures"] -= 1

    # Pin the invalid bytes so the tests reach the semantic proof checks.
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    paths = {}
    for name in builder.PINNED_INPUTS:
        document = {"identity_review": review, "history_bridge": receipt}.get(name, {})
        raw = builder._encoded(document)
        path = Path(f"{name}.json")
        (tmp_path / path).write_bytes(raw)
        paths[name] = (path, sha256(raw).hexdigest())
    monkeypatch.setattr(builder, "PINNED_INPUTS", paths)
    output = tmp_path / builder.BUNDLES / "review"
    with pytest.raises(CanonicalError, match=f"^{error_code}$"):
        builder.build(output)
    assert not output.exists()


def test_verified_proofs_derive_labels_and_loader_accepts_reproducible_model(tmp_path, proofs):
    portable = {**payload(), "training": {"identity_audit": {}, "source_manifest_sha256": "a" * 64}}
    archived = {"model": deepcopy(portable), "validation": {"status": "PASS_LOCKED_REPLAY"}}
    archived["model"]["training"]["identity_audit"] = {"status": "PARTIAL_EXACT_BINDINGS_ONLY"}
    readiness = builder.reviewed_readiness(*proofs)
    args = (archived, portable, {**portable["feature_seed"], "last_day": "2026-11-14"},
            {"source_manifest_sha256": "b" * 64}, readiness)
    _, artifact = builder._artifact(*args)
    raw = builder._encoded(artifact)
    assert raw == builder._encoded(builder._artifact(*args)[1])
    path = tmp_path / "serving.json"
    path.write_bytes(raw)
    model, _, validation = read_artifact(path, sha256(raw).hexdigest())
    assert model.payload["training"] == archived["model"]["training"]
    assert validation["identity_status"] == "PROVIDER_IDENTITY_COMPLETE"
    assert validation["history_status"] == "VERIFIED_CONTINUOUS_PRIOR_HISTORY"


def test_artifact_refuses_changed_learned_parameter(proofs):
    portable = {**payload(), "training": {"identity_audit": {}}}
    archived = {"model": deepcopy(portable), "validation": {}}
    archived["model"]["coefficients"]["home"]["intercept"] += 1
    with pytest.raises(CanonicalError, match="^SERVING_FROZEN_MODEL_DIFF$"):
        builder._artifact(archived, portable, {}, {}, builder.reviewed_readiness(*proofs))


@pytest.mark.parametrize("name", ["seed", "mapping", "identity_review"])
def test_builder_refuses_changed_pinned_input(tmp_path, monkeypatch, name):
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    path = tmp_path / f"{name}.json"
    pin = sha256(b"{}\n").hexdigest()
    path.write_bytes(b"{ }\n")
    monkeypatch.setattr(builder, "PINNED_INPUTS", {name: (path, pin)})
    output = tmp_path / builder.BUNDLES / "review"
    with pytest.raises(CanonicalError, match=f"^SERVING_{name.upper()}_PIN_OR_READ_FAILED$"):
        builder.build(output)
    assert not output.exists()
