"""Synthetic end-to-end ordering, publication and forecast-vintage tests."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from src.evaluation import future_lineage_context as c
from src.evaluation import future_lineage_qualification as q
from src.evaluation.future_lineage import SyntheticSession, unique_fixture_outcomes
from test_future_lineage_authorization import AT, HORIZON, authority, sandbox, protected_snapshot
from test_future_lineage_qualification import (
    AS_OF, NAMES, SCIENCE, FixtureFeatures, archive, payload, source_fixture,
)


class Backend:
    execution_mode = "synthetic"

    def __init__(self, store, primary, alternative, frozen):
        self.store, self.primary, self.alternative, self.frozen = store, primary, alternative, frozen
        self.calls = []
        self.bad = None
        self.source_index = None
        self.feature_science = FixtureFeatures()

    def verify_qualification(self, qualification):
        self.calls.append("reconstruct")
        self.source_index = q.verify_fixture_qualification(qualification, self.store,
            self.primary, self.alternative, frozen=self.frozen, names=NAMES,
            science=SCIENCE, features=self.feature_science)

    def features(self, qualification, target, at):
        self.calls.append("features")
        frame, _ = self.feature_science.build(qualification["assessment"], target, at)
        if self.bad == "nonfinite":
            frame.loc[0, "x"] = float("nan")
        if self.bad == "future_feature":
            frame.loc[0, "home_rolling_max_source_date_3"] = pd.Timestamp("2026-10-17T00:00:00Z")
        if self.bad == "schema":
            frame = frame.rename(columns={"x": "wrong"})
        return frame, frame.iloc[0]

    def predict(self, features, at):
        self.calls.append("models")
        frame, fixture = features
        outputs = {label: {"execution_mode": "synthetic", "model_id": "synthetic-" + label,
            "features": fixture[["x"]].tolist(), "probabilities": [0.5, 0.25, 0.25],
            "rng": {"seed": 123, "scope": "synthetic-fixture"}} for label in ("active", "candidate")}
        if self.bad == "rng":
            outputs["candidate"]["rng"] = {"seed": 456}
        if self.bad == "model":
            outputs["candidate"]["model_id"] = "unbound"
        if self.bad == "probability":
            outputs["active"]["probabilities"] = [0.1, 0.1, 0.1]
        return outputs

    def verify_outputs(self, outputs, features, at):
        self.calls.append("replay")
        if self.bad == "replay":
            raise ValueError("SYNTHETIC_PAIRED_REPLAY_FAILED")
        assert all(v["features"] == features[1][["x"]].tolist() for v in outputs.values())


def session_fixture(store, *, mutation=None, operations=("qualify", "collect", "verify"), clock=None,
                    target_at="2026-10-16T12:00:00Z"):
    primary, alternative, frozen, rows_by_year = source_fixture(store, target_at=target_at)
    assessment, _ = q.compose_assessment(store, primary, alternative,
                                         frozen=frozen, names=NAMES, as_of=AS_OF)
    qualification_result = q.qualify_fixture(store, primary, alternative, frozen=frozen,
        names=NAMES, as_of=AS_OF, science=SCIENCE, features=FixtureFeatures(),
        attempt="000001", assessment_number="001")
    protocol = {"execution_mode": "synthetic", "league": "BL1", "season": 2026,
                **HORIZON, "rng": {"seed": 123, "scope": "synthetic-fixture"}}
    fresh_rows = copy.deepcopy(rows_by_year[2026])
    if mutation == "new_finished":
        fresh_rows[-1]["status"] = "FINISHED"
        fresh_rows[-1]["utcDate"] = "2026-10-14T12:00:00Z"
        fresh_rows[-1]["score"]["fullTime"] = {"home": 2, "away": 1}
    elif mutation == "score":
        fresh_rows[0]["score"]["fullTime"]["home"] = 2
    elif mutation == "kickoff":
        fresh_rows[0]["utcDate"] = "2026-08-03T13:00:00Z"
    elif mutation == "identity":
        fresh_rows[0]["homeTeam"]["id"], fresh_rows[0]["awayTeam"]["id"] = 2, 1
    elif mutation == "duplicate":
        fresh_rows.append(copy.deepcopy(fresh_rows[-1]))
    elif mutation == "unknown":
        fresh_rows[-1]["id"] = 999
    elif mutation == "target_changed":
        fresh_rows[-1]["utcDate"] = "2026-10-16T13:00:00Z"
    elif mutation == "out_of_horizon":
        fresh_rows[-1]["utcDate"] = "2027-06-01T12:00:00Z"
    elif mutation == "horizon":
        protocol["end_at"] = "2027-06-01T00:00:00Z"
    fresh = archive(store, "matches", payload(2026, fresh_rows),
        provider="football-data.org:v4", request={
            "url": "https://api.football-data.org/v4/competitions/BL1/matches",
            "params": {"season": 2026}}, acquisition=True)
    qualification_ref = qualification_result["qualification"]
    protocol_ref = store.publish("prospective_protocol.json", protocol, at=AT)
    bindings = {"qualification": qualification_ref, "protocol": protocol_ref,
        "dependency_manifest": store.access.anchor["manifest"],
        "fresh": c.identity(store.repository, store.path("BL1/acquisition/attempt-000001/matches.json"))}
    backend = Backend(store, primary, alternative, frozen)
    session = SyntheticSession(store, operations, bindings, clock or (lambda: AT))
    return session, backend


@pytest.mark.parametrize("mutation", ["new_finished", "score", "kickoff", "identity", "duplicate"])
def test_global_stale_gate_precedes_all_downstream_work(sandbox, mutation):
    session, backend = session_fixture(sandbox, mutation=mutation)
    with pytest.raises(ValueError, match="CAUSAL_INPUTS_STALE|DUPLICATE"):
        # The finished discrepancy is outside these selected dates.
        session.collect(date_from="2026-10-16", date_to="2026-10-16", backend=backend)
    assert backend.calls == []
    assert not sandbox.path("BL1/operation.lock").exists()
    assert not (sandbox.root/"BL1/journal").exists()


def test_synthetic_vertical_slice_publication_replay_and_idempotence(sandbox):
    session, backend = session_fixture(sandbox)
    result = session.collect(date_from="2026-10-16", date_to="2026-10-16", backend=backend)
    assert result["accepted"] == ["BL1_football_data_org_3"] and result["recorded"] == 1
    assert backend.calls.index("reconstruct") < backend.calls.index("features") < backend.calls.index("models")
    assert result["collection_authorization"] is False
    count = backend.calls.count("models")
    again = session.collect(date_from="2026-10-16", date_to="2026-10-16", backend=backend)
    assert not again["accepted"] and again["already_recorded"] == ["BL1_football_data_org_3"]
    assert backend.calls.count("models") == count
    assert session.verify(backend)["recorded"] == 1
    assert len(list((sandbox.root/"BL1/journal").glob("*.json"))) == 1


def test_preparation_simulation_cannot_capture_or_create_permission(sandbox):
    session, backend = session_fixture(sandbox, operations=("qualify",))
    result = session.qualify(lambda: session.load("qualification"))
    assert result["feature_readiness"]
    with pytest.raises(ValueError, match="SIMULATION_SCOPE"):
        session.collect(date_from="2026-10-16", date_to="2026-10-16", backend=backend)
    assert backend.calls == []
    assert not (sandbox.root/"BL1/journal").exists()
    assert not sandbox.path("collection_authorization.json").exists()


@pytest.mark.parametrize("mutation,reason", [
    ("unknown", "UNKNOWN_OR_DUPLICATE_TARGET"), ("target_changed", "TARGET_IDENTITY_CHANGED"),
    ("out_of_horizon", "NOT_ELIGIBLE"), ("horizon", "HORIZON_DRIFT"),
])
def test_global_pass_does_not_make_individual_target_eligible(sandbox, mutation, reason):
    session, backend = session_fixture(sandbox, mutation=mutation)
    day = "2027-06-01" if mutation == "out_of_horizon" else "2026-10-16"
    with pytest.raises(ValueError, match=reason):
        session.collect(date_from=day, date_to=day, backend=backend)
    assert "models" not in backend.calls
    assert not (sandbox.root/"BL1/journal").exists()


@pytest.mark.parametrize("failure,reason", [
    ("nonfinite", "NONFINITE"), ("future_feature", "FUTURE_FEATURE_SOURCE"),
    ("schema", "FEATURE_SCHEMA_MISMATCH"), ("rng", "RNG_MISMATCH"),
    ("model", "MODEL_OR_FEATURE_BINDING"), ("probability", "PROBABILITY_INVALID"),
    ("replay", "PAIRED_REPLAY_FAILED"),
])
def test_each_candidate_requires_valid_features_models_rng_and_replay(sandbox, failure, reason):
    session, backend = session_fixture(sandbox)
    backend.bad = failure
    with pytest.raises(ValueError, match=reason):
        session.collect(date_from="2026-10-16", date_to="2026-10-16", backend=backend)
    assert not (sandbox.root/"BL1/journal").exists()


def test_late_corners_source_blocks_before_features(sandbox):
    session, backend = session_fixture(sandbox)
    metadata = backend.primary["sources"][5]
    bounds = metadata["archived_observation"]
    metadata["retrieved_at"] = bounds["observed_by"] = "2026-10-16T13:00:00Z"
    metadata["archived_observation"] = c.seal(bounds)
    with pytest.raises(ValueError, match="OBSERVATION_LATE"):
        session.collect(date_from="2026-10-16", date_to="2026-10-16", backend=backend)
    assert backend.calls == ["reconstruct"]


def test_expiry_blocks_before_any_feature_model_or_journal_work(sandbox):
    session, backend = session_fixture(sandbox, clock=lambda: "2026-10-16T06:00:00Z")
    with pytest.raises(ValueError, match="PERMISSION_EXPIRED"):
        session.collect(date_from="2026-10-16", date_to="2026-10-16", backend=backend)
    assert not backend.calls and not (sandbox.root/"BL1/journal").exists()


def test_publication_expiry_retains_immutable_orphan_and_never_admits(sandbox):
    ticks = iter([AT, AT, AT, AT, AT, "2026-10-16T06:00:00Z"])
    session, backend = session_fixture(sandbox, clock=lambda: next(ticks))
    with pytest.raises(ValueError, match="PERMISSION_EXPIRED"):
        session.collect(date_from="2026-10-16", date_to="2026-10-16", backend=backend)
    row = sandbox.path("BL1/journal/00000001.json")
    assert row.exists()  # Actual atomic publication occurred before expiry.
    before = row.read_bytes()
    with pytest.raises(ValueError, match="PUBLICATION_INCOMPLETE"):
        session.journal()
    assert row.read_bytes() == before
    assert not sandbox.path("BL1/attempt_outcomes/capture-00000001.json").exists()


def test_interrupted_publication_tmp_is_never_evidence(sandbox):
    session, backend = session_fixture(sandbox)
    directory = sandbox.root/"BL1/journal"
    directory.mkdir()
    (directory/"00000001.json.orphan.tmp").write_bytes(b"partial")
    with pytest.raises(ValueError, match="ORPHAN_PUBLICATION"):
        session.collect(date_from="2026-10-16", date_to="2026-10-16", backend=backend)
    assert "models" not in backend.calls
    assert not sandbox.path("BL1/journal/00000001.json").exists()


def test_tampered_journal_or_missing_completion_receipt_blocks(sandbox):
    session, backend = session_fixture(sandbox)
    session.collect(date_from="2026-10-16", date_to="2026-10-16", backend=backend)
    path = sandbox.path("BL1/journal/00000001.json")
    row = c.read(path)
    row["pair"]["fixture_id"] = "historical-pair-import"
    path.write_text(json.dumps(c.seal(row)), encoding="utf-8")
    with pytest.raises(ValueError, match="PUBLICATION_NOT_PREKICKOFF"):
        session.journal()


def test_fixture_outcome_counts_once_distinct_vintages_remain():
    base = {"league": "BL1", "season": 2026, "provider_fixture_id": 3,
            "home_goals": 1, "away_goals": 0}
    rows = [{**base, "lineage_id": "historical", "vintage_id": "old"},
            {**base, "lineage_id": "future", "vintage_id": "new"}]
    assert unique_fixture_outcomes(rows) == {"unique_fixture_outcomes": 1, "forecast_vintages": 2}
    rows[-1]["home_goals"] = 2
    with pytest.raises(ValueError, match="OUTCOME_CONFLICT"):
        unique_fixture_outcomes(rows)


def test_live_backend_rejected_before_any_effect(sandbox):
    session, backend = session_fixture(sandbox)
    backend.execution_mode = "live"
    with pytest.raises(ValueError, match="LIVE_BACKEND_DENIED"):
        session.collect(date_from="2026-10-16", date_to="2026-10-16", backend=backend)
    assert not backend.calls and not (sandbox.root/"BL1/journal").exists()


def test_kickoff_crossing_during_actual_publication_is_not_admitted(sandbox):
    late = "2026-10-15T12:00:00Z"
    ticks = iter([AT, AT, AT, AT, AT, late])
    session, backend = session_fixture(sandbox, target_at=late, clock=lambda: next(ticks))
    with pytest.raises(ValueError, match="LATE_PUBLICATION"):
        session.collect(date_from="2026-10-15", date_to="2026-10-15", backend=backend)
    row = sandbox.path("BL1/journal/00000001.json")
    assert row.exists()
    receipt = c.read(sandbox.path("BL1/attempt_outcomes/capture-00000001.json"))
    assert receipt["status"] == "LATE_PUBLICATION"
    with pytest.raises(ValueError, match="PUBLICATION_NOT_PREKICKOFF"):
        session.journal()


def test_concurrent_synthetic_writers_publish_one_immutable_pair(sandbox):
    from concurrent.futures import ThreadPoolExecutor
    session, backend = session_fixture(sandbox)
    def collect():
        return session.collect(date_from="2026-10-16", date_to="2026-10-16", backend=backend)
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(lambda _: collect(), range(2)))
    assert sum(len(r["accepted"]) for r in results) == 1
    assert len(session.journal()) == 1


def test_no_network_or_issuer_entrypoint_is_exposed():
    import importlib.util
    path = Path(__file__).resolve().parents[2]/"scripts/run_future_lineage.py"
    spec = importlib.util.spec_from_file_location("future_cli_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(SystemExit) as rejected:
        module.main(["acquire"])
    assert rejected.value.code == 2


def test_t3_missing_session_manifest_blocks_before_backend(sandbox):
    session, backend = session_fixture(sandbox)
    del session.bindings["dependency_manifest"]
    with pytest.raises(ValueError, match="SESSION_DEPENDENCY_BINDING"):
        session.collect(date_from="2026-10-16", date_to="2026-10-16", backend=backend)
    assert not backend.calls and not (sandbox.root / "BL1/journal").exists()


@pytest.mark.parametrize("operation", ["qualify", "collect", "verify"])
def test_t3_manifest_drift_blocks_existing_lifecycle_before_effects(sandbox, operation):
    session, backend = session_fixture(sandbox)
    manifest = sandbox.repository / sandbox.access.anchor["manifest"]["path"]
    manifest.write_bytes(manifest.read_bytes() + b" ")
    with pytest.raises(ValueError, match="REFERENCE_DRIFT"):
        if operation == "qualify":
            session.qualify(lambda: pytest.fail("reconstruction must not execute"))
        elif operation == "collect":
            session.collect(date_from="2026-10-16", date_to="2026-10-16", backend=backend)
        else:
            session.verify(backend)
    assert not backend.calls and not (sandbox.root / "BL1/journal").exists()


def test_t3_dependency_change_during_backend_blocks_publication(sandbox, monkeypatch):
    session, backend = session_fixture(sandbox)
    original = backend.verify_outputs
    def changed(outputs, features, at):
        original(outputs, features, at)
        entry = next(e for e in sandbox.access.manifest["entries"] if "model" in e["roles"])
        path = sandbox.repository / entry["path"]
        path.write_bytes(path.read_bytes() + b"changed synthetic model identity")
    monkeypatch.setattr(backend, "verify_outputs", changed)
    with pytest.raises(ValueError, match="REFERENCE_DRIFT"):
        session.collect(date_from="2026-10-16", date_to="2026-10-16", backend=backend)
    assert "replay" in backend.calls
    assert not (sandbox.root / "BL1/journal").exists()


def test_t3_invalid_binding_rejects_before_covered_legacy_imports(sandbox, monkeypatch):
    import builtins
    session, backend = session_fixture(sandbox)
    manifest = sandbox.repository / sandbox.access.anchor["manifest"]["path"]
    manifest.write_bytes(manifest.read_bytes() + b" ")
    original = builtins.__import__
    forbidden = {"src.evaluation.prospective_collection_next", "src.ml.bundle_activation"}
    def guarded(name, *args, **kwargs):
        if name in forbidden:
            pytest.fail("Covered dependencies must not be imported before approval verification")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(ValueError, match="REFERENCE_DRIFT"):
        session.collect(date_from="2026-10-16", date_to="2026-10-16", backend=backend)
    assert not backend.calls and not sandbox.path("BL1/operation.lock").exists()


@pytest.mark.parametrize("failure,reason", [("user", "APPLICATION_IDENTITY"),
    ("system_owner", "OWNER_OR_INHERITANCE"), ("inherit_only", "PROVISIONER_GRANT_MISSING")])
def test_t3_remediation_security_drift_blocks_effects(sandbox, monkeypatch, failure, reason):
    session, backend = session_fixture(sandbox)
    snapshot = protected_snapshot()
    if failure == "user":
        snapshot["user_sid"] = "S-1-5-21-1-2-3-1002"
    elif failure == "system_owner":
        snapshot["paths"][-1]["owner"] = "S-1-5-18"
    else:
        snapshot["paths"][-1]["aces"][0]["flags"] = 0x0B
    monkeypatch.setattr(c, "_windows_anchor_snapshot", lambda path: snapshot)
    with pytest.raises(ValueError, match=reason):
        session.collect(date_from="2026-10-16", date_to="2026-10-16", backend=backend)
    assert not backend.calls and not sandbox.path("BL1/operation.lock").exists()
    assert not (sandbox.root / "BL1/journal").exists()
