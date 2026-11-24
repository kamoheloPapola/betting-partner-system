from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from src.canonical.clock import CanonicalError
from src.canonical.features import FEATURE_VERSION
from src.canonical.team_mapping_consumer import MAPPING_PATH, RECEIPTS_DIR, TeamMappingConsumer
from src.data_foundation.source_manifest import LEAGUES, UNVERIFIED_PROVIDER


def put(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(document).encode("utf-8")
    path.write_bytes(raw)
    return sha256(raw).hexdigest()


def state(matches=1):
    return {"elo": 1510.0, "for": [2.0], "against": [1.0], "form": 3.0, "matches": matches}


def binding(provider_id, label, **extra):
    return {"league": "PD", "provider_id": provider_id, "legacy_label": label,
            "kind": "BINDING", "disposition": "BIND_LATEST_STATE", **extra}


def fixture(fixture_id=1, home=81, away=2, day="2024-06-02"):
    return {"id": fixture_id, "league": "PD", "home_id": home, "away_id": away,
            "kickoff_utc": f"{day}T18:00:00Z"}


@pytest.fixture
def publication(tmp_path):
    destination = tmp_path / MAPPING_PATH
    decisions = tmp_path / "identity-decisions.json"
    manifest = tmp_path / "source-manifest.json"
    decision_pin = put(decisions, {"schema_version": 1, "recorded_owner_decisions": []})
    manifest_pin = put(manifest, {
        "schema_version": 1,
        "season_scope": {league: {"first": 2024, "last": 2024} for league in LEAGUES},
        # The source files are deliberately absent. The consumer only validates metadata.
        "files": [{"league": "PD", "path": "data/historical/PD/sealed.csv", "sha256": "a" * 64,
                   "filename_season": 2024, "declared_season_start": 2024, "role": "AUTHORITATIVE",
                   "row_count": 1, "min_date": "2024-07-01", "max_date": "2024-07-01"}],
        "known_gaps": [],
        "unverified_seasons": [{"league": league, "declared_season_start": 2024, "status": UNVERIFIED_PROVIDER}
                               for league in LEAGUES if league != "PD"],
    })
    pins = {"decision_register": decision_pin, "proposals": "b" * 64,
            "catalog": "c" * 64, "espanyol_debt": "d" * 64}
    document = {
        "schema_version": 1, "lookup_mode": "CLOSED_WORLD_EXACT",
        "input_sha256": pins, "consumer_approved": False,
        "entries": [
            {**binding(81, "BARCELONA", valid_through="2024-06-02"), "disposition": "OWNER_SEED_WINDOW_ONLY"},
            binding(2, "TEAM TWO"),
            {"league": "PD", "provider_id": 3, "kind": "COLD_START", "disposition": "COLD_START_NO_SUPPORTED_STATE"},
        ],
    }
    kwargs = {
        "root": tmp_path, "accepted_input_sha256": pins, "identity_decisions": decisions,
        "source_manifest": manifest, "source_manifest_sha256": manifest_pin,
        "feature_seed": {"feature_version": FEATURE_VERSION, "last_day": "2024-06-02",
                         "states": [{"league": "PD", "team_id": label, "state": state()}
                                    for label in ("BARCELONA", "TEAM TWO")]},
    }
    receipt_path = destination.parent / RECEIPTS_DIR / ("e" * 32 + ".json")
    receipt = {"result": "SUCCESS", "disposition": "PUBLISHED", "operation": "publish",
               "destination": str(destination), "input_sha256": pins}
    return destination, document, receipt_path, receipt, kwargs


def publish(publication):
    destination, document, receipt_path, receipt, kwargs = publication
    kwargs["mapping_sha256"] = put(destination, document)
    put(receipt_path, {**receipt, "output_sha256": kwargs["mapping_sha256"]})
    return kwargs


@pytest.mark.parametrize(("proof", "code"), [
    ("source_manifest", "TEAM_MAPPING_SOURCE_MANIFEST_MISMATCH"),
    ("identity_decisions", "TEAM_MAPPING_IDENTITY_DECISIONS_MISMATCH"),
])
def test_live_consumer_refuses_foundation_proof_that_disagrees_with_serving_pin(publication, proof, code):
    config = publish(publication)
    model = SimpleNamespace(seed=config["feature_seed"], payload={"serving": {
        "source_manifest_sha256": config["source_manifest_sha256"],
        "team_mapping": {
            "mapping_sha256": config["mapping_sha256"],
            "accepted_input_sha256": config["accepted_input_sha256"],
        },
    }})
    refs = {
        "source_manifest": {"path": str(config["source_manifest"]), "sha256": config["source_manifest_sha256"]},
        "identity_decisions": {"path": str(config["identity_decisions"]),
                               "sha256": config["accepted_input_sha256"]["decision_register"]},
    }
    TeamMappingConsumer.for_model(config["root"], model, {"foundation_proofs": refs})
    refs[proof]["sha256"] = "0" * 64
    with pytest.raises(CanonicalError, match=f"^{code}$"):
        TeamMappingConsumer.for_model(config["root"], model, {"foundation_proofs": refs})


@pytest.mark.parametrize(("artifact", "reason"), [
    ("mapping", "TEAM_MAPPING_PIN_OR_READ_FAILED"),
    ("identity_decisions", "IDENTITY_DECISIONS_PIN_OR_READ_FAILED"),
    ("source_manifest", "SOURCE_MANIFEST_PIN_OR_READ_FAILED"),
])
def test_refuses_hash_mismatch(publication, artifact, reason):
    kwargs = publish(publication)
    target = publication[0] if artifact == "mapping" else kwargs[artifact]
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(CanonicalError, match=f"^{reason}$"):
        TeamMappingConsumer.load(**kwargs)


def test_refuses_to_start_if_mapping_is_missing_or_corrupt(publication):
    kwargs = publish(publication)
    destination, _, receipt_path, receipt, _ = publication
    destination.unlink()
    with pytest.raises(CanonicalError, match="^TEAM_MAPPING_UNAVAILABLE$"):
        TeamMappingConsumer.load(**kwargs)

    raw = b'{"entries": invalid}'
    destination.write_bytes(raw)
    kwargs["mapping_sha256"] = sha256(raw).hexdigest()
    put(receipt_path, {**receipt, "output_sha256": kwargs["mapping_sha256"]})
    with pytest.raises(CanonicalError, match="^TEAM_MAPPING_INVALID$"):
        TeamMappingConsumer.load(**kwargs)


@pytest.mark.parametrize("change", [
    {"result": "DRY_RUN"}, {"result": "FAILED"}, {"operation": "restore"},
    {"disposition": "NOT_PUBLISHED"}, {"destination": "elsewhere"},
    {"output_sha256": "0" * 64}, {"input_sha256": {"decision_register": "0" * 64}},
])
def test_refuses_nonqualifying_publication_receipt(publication, change):
    kwargs = publish(publication)
    _, _, receipt_path, receipt, _ = publication
    put(receipt_path, {**receipt, "output_sha256": kwargs["mapping_sha256"], **change})
    with pytest.raises(CanonicalError, match="^TEAM_MAPPING_PUBLICATION_RECEIPT_MISSING$"):
        TeamMappingConsumer.load(**kwargs)


@pytest.mark.parametrize("change", [{"kind": "COLD_START"}, {"disposition": "ASSUMED_ALIAS"}])
def test_refuses_unknown_dispositions_and_inconsistent_kinds(publication, change):
    publication[1]["entries"][1].update(change)
    with pytest.raises(CanonicalError, match="^TEAM_MAPPING_INVALID$"):
        TeamMappingConsumer.load(**publish(publication))


def test_refuses_mapping_that_disagrees_with_pinned_alias_decision(publication):
    _, document, _, _, kwargs = publication
    decision = {"league": "PD", "provider_id": 2, "legacy_label": "TEAM TWO",
                "status": "APPROVED", "decision_type": "APPROVED_ALIAS"}
    kwargs["accepted_input_sha256"]["decision_register"] = put(
        kwargs["identity_decisions"], {"schema_version": 1, "recorded_owner_decisions": [decision]},
    )
    TeamMappingConsumer.load(**publish(publication))
    document["entries"][1]["legacy_label"] = "DIFFERENT TEAM"
    with pytest.raises(CanonicalError, match="^TEAM_MAPPING_DECISION_MISMATCH$"):
        TeamMappingConsumer.load(**publish(publication))


@pytest.mark.parametrize("kickoff", [None, "random text", "2024-06-02T18:00:00"],
                         ids=["none", "random-text", "timezone-less"])
def test_invalid_kickoff_is_a_clean_mapping_rejection(publication, kickoff):
    consumer = TeamMappingConsumer.load(**publish(publication))
    bad = {**fixture(1), "kickoff_utc": kickoff}
    result = consumer.resolve([bad, fixture(2)])

    assert result["fixtures"][0] == {"fixture_id": 1, "reason": "INVALID_PROVIDER_FIXTURE", "side": None}
    assert result["status"]["rejection_reasons"] == {"INVALID_PROVIDER_FIXTURE": 1}
    assert result["status"]["resolved"] == 1


def test_resolves_exact_pair_and_cold_start_and_counts_rejections(publication):
    consumer = TeamMappingConsumer.load(**publish(publication))
    repeated = fixture(5)
    row_with_legacy_provenance = {**fixture(6), "source_file": "data/processed/matches/PD_2025.csv", "line": 179}
    result = consumer.resolve([
        fixture(), fixture(2, home=3), fixture(3, away=999), fixture(4, day="2024-06-03"),
        repeated, {**repeated, "away_id": 3}, row_with_legacy_provenance,
    ])
    assert result["status"] == {
        "state": "DEGRADED", "resolved": 2, "rejected": 5,
        "rejection_reasons": {"DUPLICATE_PROVIDER_FIXTURE": 2, "NON_PROVIDER_FIXTURE": 1,
                              "TEAM_STATE_AFTER_SEED_WINDOW": 1, "UNKNOWN_TEAM_ID": 1},
    }
    exact, cold, unknown, later, duplicate_first, duplicate_second, historical = result["fixtures"]
    assert exact["home"] == {"provider_id": 81, "disposition": "OWNER_SEED_WINDOW_ONLY", "state": state()}
    assert exact["away"]["state"] == state()
    assert cold["home"] == {
        "provider_id": 3, "disposition": "COLD_START_NO_SUPPORTED_STATE",
        "state": {"elo": 1500.0, "for": [], "against": [], "form": None, "matches": 0},
    }
    assert unknown == {"fixture_id": 3, "reason": "UNKNOWN_TEAM_ID", "side": "away"}
    assert later == {"fixture_id": 4, "reason": "TEAM_STATE_AFTER_SEED_WINDOW", "side": "home"}
    assert duplicate_first == duplicate_second == {"fixture_id": 5, "reason": "DUPLICATE_PROVIDER_FIXTURE", "side": None}
    assert historical["reason"] == "NON_PROVIDER_FIXTURE"
    assert consumer.resolve([]) == {"fixtures": [], "status": {
        "state": "READY", "resolved": 0, "rejected": 0, "rejection_reasons": {},
    }}


def test_seed_window_is_generic_and_inclusive(publication):
    _, document, _, _, _ = publication
    document["entries"][0] = binding(81, "BARCELONA")
    document["entries"][1].update(disposition="OWNER_SEED_WINDOW_ONLY", valid_through="2024-07-15")
    consumer = TeamMappingConsumer.load(**publish(publication))
    on_date, after_date = consumer.resolve([
        fixture(day="2024-07-15"), fixture(2, day="2024-07-16"),
    ])["fixtures"]
    assert on_date["home"]["provider_id"] == 81 and on_date["away"]["provider_id"] == 2
    assert after_date == {"fixture_id": 2, "reason": "TEAM_STATE_AFTER_SEED_WINDOW", "side": "away"}


@pytest.mark.parametrize("disposition, date_fields", [
    ("BIND_LATEST_STATE", {"valid_through": "2024-06-02"}),
    ("OWNER_SEED_WINDOW_ONLY", {}),
    ("OWNER_SEED_WINDOW_ONLY", {"valid_through": "2024-6-2"}),
    ("OWNER_SEED_WINDOW_ONLY", {"valid_through": "2024-02-30"}),
])
def test_valid_through_requires_seed_window_disposition_and_iso_date(publication, disposition, date_fields):
    _, document, _, _, _ = publication
    document["entries"][1] = binding(2, "TEAM TWO", disposition=disposition, **date_fields)
    with pytest.raises(CanonicalError, match="TEAM_MAPPING_INVALID"):
        TeamMappingConsumer.load(**publish(publication))


def test_provider_seed_preserves_replayed_cold_start_state_by_id(publication):
    _, document, _, _, config = publication
    document["entries"][0] = binding(81, "UNUSED LEGACY LABEL")
    config["feature_seed"].update(identity_key="provider_id", states=[
        {"league": "PD", "team_id": "81", "state": state()},
        {"league": "PD", "team_id": "3", "state": state(2)},
    ])
    consumer = TeamMappingConsumer.load(**publish(publication))
    result = consumer.resolve([fixture(1, away=3, day="2024-07-01"), fixture(2, day="2024-07-01")])
    assert result["fixtures"][0]["home"]["state"] == state()
    assert result["fixtures"][0]["away"]["state"] == state(2)
    assert result["fixtures"][1] == {"fixture_id": 2, "reason": "TEAM_STATE_MISSING", "side": "away"}


@pytest.mark.parametrize("ambiguity", ["duplicate_mapping", "duplicate_seed_state", "shared_legacy_state"])
def test_ambiguous_team_only_rejects_affected_fixtures(publication, ambiguity):
    _, document, _, _, kwargs = publication
    if ambiguity == "duplicate_mapping":
        document["entries"].append(binding(2, "OTHER TEAM"))
    elif ambiguity == "duplicate_seed_state":
        kwargs["feature_seed"]["states"].append({"league": "PD", "team_id": "TEAM TWO", "state": state(2)})
    else:
        document["entries"].append(binding(4, "TEAM TWO"))
    consumer = TeamMappingConsumer.load(**publish(publication))
    result = consumer.resolve([fixture(), fixture(2, away=3)])
    assert result["fixtures"][0] == {"fixture_id": 1, "reason": "AMBIGUOUS_TEAM_STATE", "side": "away"}
    assert result["status"]["rejection_reasons"] == {"AMBIGUOUS_TEAM_STATE": 1}
    assert result["status"]["resolved"] == result["status"]["rejected"] == 1


def test_bound_but_missing_state_does_not_become_cold_start(publication):
    kwargs = publish(publication)
    kwargs["feature_seed"]["states"].pop()
    consumer = TeamMappingConsumer.load(**kwargs)
    assert consumer.resolve([fixture()])["fixtures"] == [
        {"fixture_id": 1, "reason": "TEAM_STATE_MISSING", "side": "away"},
    ]


def test_output_bytes_match_under_two_python_hash_seeds(publication):
    kwargs = publish(publication)
    for key in ("root", "identity_decisions", "source_manifest"):
        kwargs[key] = str(kwargs[key])
    inputs = {
        "config": kwargs,
        "fixtures": [fixture(), fixture(2, home=3), fixture(3, away=999), fixture(4, day="2024-06-03")],
    }
    script = """
import json, sys

def offline(event, args):
    if event.startswith("socket."):
        raise RuntimeError("NETWORK_FORBIDDEN")

sys.addaudithook(offline)
from src.canonical.team_mapping_consumer import TeamMappingConsumer
data = json.load(sys.stdin)
consumer = TeamMappingConsumer.load(**data["config"])
sys.stdout.write(json.dumps(consumer.resolve(data["fixtures"]), separators=(",", ":")))
"""
    outputs = []
    for seed in ("101", "202"):
        result = subprocess.run(
            [sys.executable, "-B", "-c", script], input=json.dumps(inputs).encode("utf-8"),
            env={**os.environ, "PYTHONHASHSEED": seed}, cwd=Path(__file__).resolve().parents[2],
            capture_output=True, check=True, timeout=30,
        )
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1]
    assert json.loads(outputs[0])["status"]["rejected"] == 2
