"""Synthetic source closure, composition and archived-observation contracts."""
from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import pandas as pd
import pytest

from src.evaluation import future_lineage_context as c
from src.evaluation import future_lineage_qualification as q
from test_future_lineage_authorization import AT, HORIZON, authority, sandbox

AS_OF = "2026-10-15T08:30:00Z"
RETRIEVED = "2026-10-15T08:00:00Z"
NAMES = {"ALPHA": {"ALPHA"}, "BETA": {"BETA"}}
SCIENCE = {"execution_mode": "synthetic", "ordered_features": ["x"],
           "active_bundle_id": "synthetic-active", "candidate_bundle_id": "synthetic-candidate",
           "active_sha256": c.fingerprint("synthetic-active-model"),
           "candidate_sha256": c.fingerprint("synthetic-candidate-model")}


class FixtureFeatures:
    execution_mode = "synthetic"

    def __init__(self, failure=None):
        self.failure = failure

    def reproduce(self):
        return {"execution_mode": "synthetic", "reproduced": self.failure != "reproduction",
                "code": "SYNTHETIC_REPRODUCTION_FAILURE" if self.failure == "reproduction" else None}

    def build(self, assessment, target, at):
        observation = assessment["composed_observations"][0]
        frame = pd.DataFrame([{"match_id": target["fixture_id"],
            "date": pd.Timestamp(target["kickoff_at"]), "x": float(observation["home_corners"]),
            "home_rolling_max_source_date_3": pd.Timestamp(observation["kickoff_at"])}])
        if self.failure == "nonfinite":
            frame.loc[0, "x"] = float("nan")
        return frame, frame.iloc[0]


def archive(store, stem, payload, *, provider, request, acquisition=False):
    slot = ("BL1/acquisition/attempt-000001/" if acquisition else
            "BL1/qualification/attempt-000001/raw_sources/") + stem
    raw = store.path(slot + ".raw")
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8"))
    ref = c.identity(store.repository, raw)
    metadata = {"execution_mode": "synthetic", "schema_version": "ffps.phase11_raw_source.v1",
        "status": "RETRIEVED", "http_status": 200, "provider": provider, "request": request,
        "started_at": "2026-10-15T07:59:00Z", "retrieved_at": RETRIEVED,
        "credential_headers_present": ["x-auth-token"] if provider == "football-data.org:v4" else [],
        "raw": ref}
    metadata["archived_observation"] = c.seal({
        "basis": "archived_response_observation", "raw": ref, "provider": provider,
        "request_sha256": c.fingerprint(request), "not_before": "2026-10-15T07:59:00Z",
        "observed_by": RETRIEVED})
    metadata_path = store.path(slot + ".json")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return metadata


def match(year, fixture_id, date, *, status="FINISHED", reverse=False):
    return {"id": fixture_id, "utcDate": date, "status": status, "stage": "REGULAR_SEASON",
        "competition": {"code": "BL1"},
        "season": {"id": year, "startDate": f"{year}-08-01", "endDate": f"{year+1}-05-31"},
        "homeTeam": {"id": 2 if reverse else 1}, "awayTeam": {"id": 1 if reverse else 2},
        "lastUpdated": "2026-10-14T07:00:00Z",
        "score": {"fullTime": {"home": 1 if status == "FINISHED" else None,
                                "away": 0 if status == "FINISHED" else None}}}


def payload(year, rows):
    dates = [row["utcDate"][:10] for row in rows]
    return {"competition": {"code": "BL1"}, "filters": {"season": year}, "matches": rows,
            "resultSet": {"count": len(rows), "first": min(dates), "last": max(dates)}}


def source_fixture(store, *, target_at="2026-10-16T12:00:00Z"):
    sources = []
    rows_by_year = {
        2025: [match(2025, 1, "2025-08-01T12:00:00Z")],
        2026: [match(2026, 2, "2026-08-03T12:00:00Z"),
               match(2026, 3, target_at, status="TIMED", reverse=True)]}
    for year, rows in rows_by_year.items():
        teams = {"competition": {"code": "BL1"}, "season": {"startDate": f"{year}-08-01"},
            "teams": [{"id": 1, "name": "Alpha", "shortName": "Alpha"},
                      {"id": 2, "name": "Beta", "shortName": "Beta"}]}
        for kind, value in (("matches", payload(year, rows)), ("teams", teams)):
            sources.append(archive(store, f"primary-{year}-{kind}", value,
                provider="football-data.org:v4", request={
                    "url": f"https://api.football-data.org/v4/competitions/BL1/{kind}",
                    "params": {"season": year}}))
        code, date = ("2526", "01/08/2025") if year == 2025 else ("2627", "03/08/2026")
        csv = ("Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,HC,AC\n" +
               f"D1,{date},Alpha,Beta,1,0,4,2\n").encode()
        sources.append(archive(store, f"statistics-{year}", csv,
            provider="football-data.co.uk:csv:no-www", request={
                "url": f"https://football-data.co.uk/mmz4281/{code}/D1.csv", "params": {}}))
    league = {"details": {"id": 54, "name": "Bundesliga", "selectedSeason": "2026/2027"},
        "fixtures": {"allMatches": [{"id": 20, "home": {"id": "1", "name": "Alpha"},
            "away": {"id": "2", "name": "Beta"}, "status": {
                "finished": True, "utcTime": "2026-08-03T12:00:00Z"}}]}}
    detail = {"general": {"matchId": 20, "leagueId": 54, "leagueName": "Bundesliga",
        "finished": True, "matchTimeUTCDate": "2026-08-03T12:00:00Z",
        "homeTeam": {"id": 1, "name": "Alpha"}, "awayTeam": {"id": 2, "name": "Beta"}},
        "header": {"status": {"scoreStr": "1 - 0"}},
        "content": {"stats": {"Periods": {"All": {"stats": [
            {"stats": [{"title": "Corners", "stats": [4, 2]}]}]}}}}}
    league_source = archive(store, "fotmob-league", league, provider="fotmob:web-api",
        request={"url": "https://www.fotmob.com/api/data/leagues",
                 "params": {"ccode3": "GER", "id": 54, "season": "2026/2027"}})
    detail_source = archive(store, "fotmob-match-20", detail, provider="fotmob:web-api",
        request={"url": "https://www.fotmob.com/api/data/matchDetails", "params": {"matchId": "20"}})
    primary = {"execution_mode": "synthetic", "league": "BL1", "requested_season": 2026,
               "division": "D1", "sources": sources}
    alternative = {"execution_mode": "synthetic", "league": "BL1", "season": 2026,
        "selected_season": "2026/2027", "league_source": league_source,
        "finished_fixture_ids": ["20"], "detail_sources": [detail_source]}
    frozen = pd.DataFrame([{"league": "BL1", "match_date": "2025-08-01",
        "home_team": "ALPHA", "away_team": "BETA", "home_goals": 1, "away_goals": 0,
        "home_corners": 4, "away_corners": 2}])
    return primary, alternative, frozen, rows_by_year


@pytest.fixture
def sources(sandbox):
    return sandbox, source_fixture(sandbox)


def test_complete_composition_preserves_authorities_and_provenance(sources):
    store, (primary, alternative, frozen, _) = sources
    result, index = q.compose_assessment(store, primary, alternative,
                                         frozen=frozen, names=NAMES, as_of=AS_OF)
    assert result["qualified"] and not result["blockers"]
    assert len(result["events"]) == 3 and len(result["upcoming"]) == 1
    assert result["upcoming"][0]["fixture_id"] == "BL1_football_data_org_3"
    assert len(result["historical_overlap"]) == 1
    assert all(result["historical_overlap"][0]["fields"].values())
    assert len(result["cross_provider_overlap"]) == 1 and len(index["sources"]) == 8
    observation = result["composed_observations"][0]
    assert observation["home_corners"] == 4 and observation["away_corners"] == 2
    assert observation["field_provenance"]["home_goals"]["source"] == primary["sources"][3]["raw"]
    assert observation["field_provenance"]["home_corners"]["source"] == primary["sources"][5]["raw"]


@pytest.mark.parametrize("role", range(8))
def test_every_consumed_source_requires_saved_observation(sources, role):
    store, (primary, alternative, frozen, _) = sources
    combined = primary["sources"] + [alternative["league_source"]] + alternative["detail_sources"]
    del combined[role]["archived_observation"]
    with pytest.raises(ValueError, match="OBSERVATION_MISSING"):
        q.compose_assessment(store, primary, alternative, frozen=frozen, names=NAMES, as_of=AS_OF)


@pytest.mark.parametrize("mutation,reason", [
    ("late", "LATE_OR_CONTRADICTORY"), ("contradictory", "LATE_OR_CONTRADICTORY"),
    ("raw", "OBSERVATION_INVALID"), ("scope", "OBSERVATION_INVALID"),
    ("seal", "SEAL_INVALID"),
])
def test_observation_evidence_is_bound_and_fail_closed(sources, mutation, reason):
    store, (primary, _, _, _) = sources
    metadata = primary["sources"][0]
    bounds = metadata["archived_observation"]
    if mutation == "late":
        bounds["observed_by"] = "2026-10-16T13:00:00Z"
        metadata["retrieved_at"] = bounds["observed_by"]
    elif mutation == "contradictory":
        bounds["not_before"] = "2026-10-15T08:01:00Z"
    elif mutation == "raw":
        bounds["raw"] = {**bounds["raw"], "sha256": "0"*64}
    elif mutation == "scope":
        bounds["request_sha256"] = "0"*64
    elif mutation == "seal":
        bounds["record_sha256"] = "0"*64
    if mutation != "seal":
        metadata["archived_observation"] = c.seal(bounds)
    with pytest.raises(ValueError, match=reason):
        q.observation(metadata, store.repository, cutoff=AS_OF)


def test_retrieval_or_provider_update_cannot_replace_observation(sources):
    store, (primary, _, _, _) = sources
    metadata = primary["sources"][0]
    metadata.pop("archived_observation")
    metadata["lastUpdated"] = "2026-01-01T00:00:00Z"
    with pytest.raises(ValueError, match="OBSERVATION_MISSING"):
        q.observation(metadata, store.repository, cutoff=AS_OF)


def test_post_kickoff_observation_cannot_be_backdated(sources):
    store, (primary, _, _, _) = sources
    with pytest.raises(ValueError, match="AFTER_KICKOFF"):
        q.observation(primary["sources"][0], store.repository,
                      cutoff=AS_OF, kickoff="2026-10-15T07:59:30Z")


@pytest.mark.parametrize("mutation,reason", [
    ("missing", "SOURCE_INDEX_INCOMPLETE"), ("duplicate", "DUPLICATE_SOURCE_ROLE"),
    ("extra", "UNDECLARED_SOURCE_ROLE"), ("wrong_provider", "PRIMARY_SOURCE_SCOPE"),
])
def test_exact_source_roles_are_complete_unique_and_scoped(sources, mutation, reason):
    store, (primary, alternative, frozen, _) = sources
    if mutation == "missing":
        primary["sources"].pop()
    elif mutation == "duplicate":
        primary["sources"].append(copy.deepcopy(primary["sources"][0]))
    elif mutation == "extra":
        row = copy.deepcopy(primary["sources"][0])
        row["raw"]["path"] = row["raw"]["path"].replace("primary-2025-matches", "unknown")
        primary["sources"].append(row)
    else:
        primary["sources"][0]["provider"] = "unapproved"
    with pytest.raises(ValueError, match=reason):
        q.compose_assessment(store, primary, alternative, frozen=frozen, names=NAMES, as_of=AS_OF)


@pytest.mark.parametrize("mutation", ["value", "omit", "provenance"])
def test_resealed_incomplete_or_altered_composition_is_rederived(sources, mutation):
    store, (primary, alternative, frozen, _) = sources
    assessment, _ = q.compose_assessment(store, primary, alternative, frozen=frozen,
                                         names=NAMES, as_of=AS_OF)
    if mutation == "value":
        assessment["composed_observations"][0]["home_corners"] += 1
    elif mutation == "omit":
        assessment["composed_observations"] = []
    else:
        assessment["composed_observations"][0]["field_provenance"]["home_goals"]["source"] = primary["sources"][1]["raw"]
    qualification = c.seal({"as_of": AS_OF, "assessment": assessment}, "qualification_sha256")
    with pytest.raises(ValueError, match="COMPLETE_COMPOSITION_REPLAY_MISMATCH"):
        q.verify_composition(qualification, store, primary, alternative, frozen=frozen, names=NAMES)


def test_historical_correction_blocks_without_rewriting_history(sources):
    store, (primary, alternative, frozen, _) = sources
    frozen.loc[0, "home_goals"] = 2
    before = frozen.copy(deep=True)
    result, _ = q.compose_assessment(store, primary, alternative, frozen=frozen, names=NAMES, as_of=AS_OF)
    assert not result["qualified"]
    assert "SOURCE_SEMANTICS_INCOMPATIBLE" in {b["code"] for b in result["blockers"]}
    pd.testing.assert_frame_equal(frozen, before)


def replace_payload(store, metadata, value):
    path = store.repository/metadata["raw"]["path"]
    path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode("utf-8"))
    metadata["raw"] = c.identity(store.repository, path)
    metadata["archived_observation"] = c.seal({
        **metadata["archived_observation"], "raw": metadata["raw"]})


@pytest.mark.parametrize("mutation,reason", [
    ("malformed", "MALFORMED_JSON"), ("count", "FIXTURE_COUNT_MISMATCH"),
    ("bounds", "RESULT_SET_BOUNDARY_MISMATCH"), ("future_update", "FUTURE_PROVIDER_UPDATE"),
    ("missing_corners", "FOTMOB_MISSING_CORNERS"),
    ("missing_detail", "FOTMOB_DETAIL_COVERAGE_INCOMPLETE"),
    ("wrong_team", "FOTMOB_TEAM_IDENTITY_CONFLICT"),
])
def test_unchanged_normalizers_reject_bad_source_payloads(sources, mutation, reason):
    store, (primary, alternative, frozen, _) = sources
    metadata = primary["sources"][3]
    value = json.loads((store.repository/metadata["raw"]["path"]).read_bytes())
    if mutation == "malformed":
        replace_payload(store, metadata, b"not JSON")
    elif mutation == "count":
        value["resultSet"]["count"] += 1
        replace_payload(store, metadata, value)
    elif mutation == "bounds":
        value["resultSet"]["first"] = "2026-01-01"
        replace_payload(store, metadata, value)
    elif mutation == "future_update":
        value["matches"][0]["lastUpdated"] = "2026-10-16T00:00:00Z"
        replace_payload(store, metadata, value)
    elif mutation == "missing_detail":
        alternative["detail_sources"] = []
    else:
        metadata = alternative["detail_sources"][0]
        value = json.loads((store.repository/metadata["raw"]["path"]).read_bytes())
        if mutation == "missing_corners":
            value["content"]["stats"]["Periods"]["All"]["stats"] = []
        else:
            value["general"]["homeTeam"]["id"] = 999
        replace_payload(store, metadata, value)
    with pytest.raises(ValueError, match=reason):
        q.compose_assessment(store, primary, alternative, frozen=frozen, names=NAMES, as_of=AS_OF)


def test_cross_provider_disagreement_blocks_without_imputation(sources):
    store, (primary, alternative, frozen, _) = sources
    metadata = alternative["detail_sources"][0]
    value = json.loads((store.repository/metadata["raw"]["path"]).read_bytes())
    value["content"]["stats"]["Periods"]["All"]["stats"][0]["stats"][0]["stats"][0] = 99
    replace_payload(store, metadata, value)
    result, _ = q.compose_assessment(store, primary, alternative, frozen=frozen, names=NAMES, as_of=AS_OF)
    assert not result["qualified"]
    assert "ALTERNATIVE_PROVIDER_DISAGREEMENT" in {b["code"] for b in result["blockers"]}
    assert result["composed_observations"][0]["home_corners"] == 4


def test_qualification_artifacts_include_reconstructed_features_but_no_permission(sources):
    store, (primary, alternative, frozen, _) = sources
    features = FixtureFeatures()
    result = q.qualify_fixture(store, primary, alternative, frozen=frozen, names=NAMES,
        as_of=AS_OF, science=SCIENCE, features=features, attempt="000001", assessment_number="001")
    qualification = c.read(c.reference(store.repository, result["qualification"]))
    assert result["ready"] and not result["collection_authorization"]
    assert qualification["feature_proofs"][0]["feature_vector_sha256"] == c.fingerprint([4.0])
    rebuilt = q.verify_fixture_qualification(qualification, store, primary, alternative,
        frozen=frozen, names=NAMES, science=SCIENCE, features=features)
    assert len(rebuilt["sources"]) == 8
    assert not store.path("collection_authorization.json").exists()
    assert store.path("BL1/qualification/attempt-000001/assessments/assessment-001/current_source_contract.json").exists()


@pytest.mark.parametrize("failure", ["reproduction", "nonfinite"])
def test_failed_feature_continuity_cannot_produce_ready_contract(sources, failure):
    store, (primary, alternative, frozen, _) = sources
    result = q.qualify_fixture(store, primary, alternative, frozen=frozen, names=NAMES,
        as_of=AS_OF, science=SCIENCE, features=FixtureFeatures(failure),
        attempt="000001", assessment_number="001")
    qualification = c.read(c.reference(store.repository, result["qualification"]))
    assert not result["ready"] and qualification["feature_failures"]
    assert not store.path("BL1/qualification/attempt-000001/assessments/assessment-001/current_source_contract.json").exists()


def test_resealed_feature_proof_is_independently_rebuilt(sources):
    store, (primary, alternative, frozen, _) = sources
    features = FixtureFeatures()
    result = q.qualify_fixture(store, primary, alternative, frozen=frozen, names=NAMES,
        as_of=AS_OF, science=SCIENCE, features=features, attempt="000001", assessment_number="001")
    qualification = c.read(c.reference(store.repository, result["qualification"]))
    qualification["feature_proofs"][0]["feature_vector_sha256"] = "0"*64
    qualification = c.seal(qualification, "qualification_sha256")
    with pytest.raises(ValueError, match="FEATURE_QUALIFICATION_REPLAY_MISMATCH"):
        q.verify_fixture_qualification(qualification, store, primary, alternative,
            frozen=frozen, names=NAMES, science=SCIENCE, features=features)


class T5ArchivedParityTests(unittest.TestCase):
    """Explicit offline replay; no expired sandbox, model inference or publication.

    The dedicated work-unit runner supplies filesystem/network/process isolation.
    Archived observations are historical inputs, not newly qualified sources.
    Derived mutations below are negative tests, never original evidence.
    """

    @classmethod
    def setUpClass(cls):
        import hashlib
        from src.evaluation import phase11_current_source as legacy
        from src.evaluation import phase10_current_data as phase10

        cls.root = Path(__file__).resolve().parents[2]
        base = cls.root / "tmp/BL1-decision-closure-review-20261014-01"
        candidate_bytes = (base / "decision-package.json").read_bytes()
        assert hashlib.sha256(candidate_bytes).hexdigest() == (
            "2615787a20ca275d034c956db255301b4d23f9b2a81414b2ff394a99e2a2d52b")
        candidate = json.loads(candidate_bytes)
        for ref in candidate["bound_artifacts"] + candidate["reviewed_design_documents"]:
            c.reference(cls.root, ref)
        cls.reference = c.read(base / "scientific-reference-values.json")
        cls.names = cls.reference["ordered_features"]
        cls.qualification = c.read(c.reference(
            cls.root, cls.reference["source_snapshot_not_refreshed"]))
        cls.index = c.read(c.reference(cls.root, cls.qualification["source_index"]))
        cls.assessment = cls.qualification["assessment"]
        c.reference(cls.root, cls.qualification["historical_feature_reproduction"]["source"])
        before = c.read(cls.root / "tmp/BL1-administrator-approval-20261015-01/protected-before.json")
        assert c.identity(cls.root, cls.root / "data/processed/matches.csv")["sha256"] == (
            before["files"]["data/processed/matches.csv"])
        dependencies = c.read(base / "dependency-baseline.json")["existing_local_python_files"]
        scientific_paths = {
            "src/evaluation/phase11_current_source.py", "src/evaluation/phase10_current_data.py",
            "src/features/pipeline.py", "src/features/engineering.py",
            "src/features/build_feature_matrix.py", "src/predictions/provenance.py",
        }
        for ref in dependencies:
            if ref["path"] in scientific_paths:
                c.reference(cls.root, ref)
        for metadata in cls.index["sources"]:
            c.reference(cls.root, metadata["raw"])
            assert c.instant(metadata["started_at"]) <= c.instant(metadata["retrieved_at"]) <= (
                c.instant(cls.assessment["as_of"]))
        cls.legacy, cls.phase10 = legacy, phase10
        cls.reproduction = legacy.feature_reproduction("BL1", cls.names)
        cls.matrix, cls.fixture = legacy.target_matrix(
            cls.assessment, cls.assessment["upcoming"][0],
            as_of=cls.assessment["as_of"], source_index=cls.index)
        cls.evidence = {
            "comparison_scope": "ARCHIVED_FEATURE_CONTINUITY_ONLY_NOT_FUTURE_SOURCE_ADMISSION",
            "reference_qualification": cls.reference["source_snapshot_not_refreshed"],
            "feature_matrix": cls.qualification["historical_feature_reproduction"]["source"],
            "history": c.identity(cls.root, cls.root / "data/processed/matches.csv"),
            "source_index": cls.qualification["source_index"],
            "archived_source_references": [m["raw"] for m in cls.index["sources"]],
            "historical_reproduction": cls.reproduction,
            "ordered_features": cls.names,
            "feature_schema_sha256": c.fingerprint(cls.names),
            "reference_target_proof": cls.qualification["feature_proofs"][0],
            "numerical_semantics": "Pinned phase11.feature_reproduction: rtol=0, atol=1e-12, equal_nan=True; frozen all-finite rows define eligibility; no imputation.",
            "per_feature": [{
                "ordinal": i, "name": name, "target_value": float(cls.fixture[name]),
                "eligible_row_differences": cls.reproduction["model_eligible_differing_features"].get(name, 0),
                "all_row_differences": cls.reproduction["all_row_differing_features"].get(name, 0),
                "historical_test": "test_t5_historical_reproduction_matches_archived_result",
                "target_test": "test_t5_target_continuity_matches_archived_proof",
                "negative_tests": ["test_t5_every_missing_feature_blocks", "test_t5_every_nonfinite_feature_blocks"],
                "definition_source": "Pinned FeaturePipeline and FeatureEngineer via phase11.pipeline; ordered schema in scientific-reference-values.json",
            } for i, name in enumerate(cls.names)],
            "saved_observation_envelopes_missing": sum(
                "archived_observation" not in m for m in cls.index["sources"]),
            "limitations": [
                "Historical source metadata lacks the future-lineage saved observation envelope; none is invented.",
                "Shared continuity replay does not prove new-root composition or future source admission.",
                "One archived target proof is replayed; no claim covers every future candidate.",
                "No models are loaded or predictions generated; T3/T10/T11 and independent acceptance remain separate.",
            ],
        }

    def replay(self, *, assessment=None, matrix=None, fixture=None, reproduction=None, names=None):
        parent = self

        class ArchivedFeatures:
            execution_mode = "offline_archived_replay"

            def reproduce(self):
                return copy.deepcopy(parent.reproduction if reproduction is None else reproduction)

            def build(self, assessment, target, at):
                return (parent.matrix.copy(deep=True) if matrix is None else matrix,
                        parent.fixture.copy(deep=True) if fixture is None else fixture)

        return q._feature_continuity(
            self.assessment if assessment is None else assessment,
            self.names if names is None else names, ArchivedFeatures())

    def test_t5_inventory_and_order_match_pinned_contract(self):
        self.assertEqual(len(self.names), 30)
        self.assertEqual(self.names, self.qualification["ordered_features"])
        self.assertEqual(c.fingerprint(self.names), self.reference["feature_schema_sha256"])
        for label in ("active", "candidate"):
            self.assertEqual(self.reference["models"][label]["feature_contract"]["ordered_feature_names"], self.names)

    def test_t5_historical_reproduction_matches_archived_result(self):
        self.assertEqual(self.reproduction, self.qualification["historical_feature_reproduction"])
        self.assertEqual(self.reproduction["model_eligible_rows"], 2258)
        self.assertEqual(self.reproduction["excluded_frozen_nonfinite_rows"], 19)
        for name in self.names:
            with self.subTest(feature=name):
                self.assertEqual(self.reproduction["model_eligible_differing_features"].get(name, 0), 0)

    def test_t5_target_continuity_matches_archived_proof(self):
        reproduction, proofs, failures, ready = self.replay()
        self.assertEqual(reproduction, self.qualification["historical_feature_reproduction"])
        self.assertEqual(proofs, self.qualification["feature_proofs"])
        self.assertEqual(failures, self.qualification["feature_failures"])
        self.assertEqual(ready, self.qualification["feature_readiness"])
        self.evidence["observed_target_proof"] = proofs[0]

    def test_t5_actual_reconstruction_is_deterministic(self):
        repeated = self.legacy.feature_reproduction("BL1", self.names)
        matrix, fixture = self.legacy.target_matrix(
            self.assessment, self.assessment["upcoming"][0],
            as_of=self.assessment["as_of"], source_index=self.index)
        self.assertEqual(repeated, self.reproduction)
        self.assertEqual(fixture[self.names].tolist(), self.fixture[self.names].tolist())
        self.assertEqual(self.replay(matrix=matrix, fixture=fixture, reproduction=repeated), self.replay())

    def test_t5_every_missing_feature_blocks(self):
        for name in self.names:
            with self.subTest(feature=name):
                _, proofs, failures, ready = self.replay(matrix=self.matrix.drop(columns=[name]))
                self.assertFalse(ready)
                self.assertFalse(proofs)
                self.assertEqual(failures[0]["code"], "FEATURE_SCHEMA_MISMATCH")

    def test_t5_every_nonfinite_feature_blocks(self):
        for name in self.names:
            with self.subTest(feature=name):
                fixture = self.fixture.copy(deep=True)
                fixture[name] = float("nan")
                _, proofs, failures, ready = self.replay(fixture=fixture)
                self.assertFalse(ready)
                self.assertFalse(proofs)
                self.assertEqual(failures[0]["code"], "NONFINITE_OR_INSUFFICIENT_HISTORY")

    def test_t5_reordered_inventory_cannot_match_archived_proof(self):
        _, proofs, _, _ = self.replay(names=list(reversed(self.names)))
        self.assertNotEqual(proofs[0]["feature_schema_sha256"], self.reference["feature_schema_sha256"])
        self.assertNotEqual(proofs[0], self.qualification["feature_proofs"][0])

    def test_t5_invalid_inventory_blocks_before_reconstruction(self):
        for names in ([], tuple(self.names), self.names + [self.names[0]]):
            with self.subTest(names=type(names).__name__):
                with self.assertRaisesRegex(ValueError, "FEATURE_SCHEMA_MISMATCH"):
                    self.replay(names=names)

    def test_t5_wrong_target_identity_blocks(self):
        fixture = self.fixture.copy(deep=True)
        fixture["match_id"] = "derived-negative-target"
        _, proofs, failures, ready = self.replay(fixture=fixture)
        self.assertFalse(ready)
        self.assertFalse(proofs)
        self.assertEqual(failures[0]["code"], "FUTURE_FEATURE_TARGET_MISMATCH")

    def test_t5_reproduction_failure_blocks_target_build(self):
        parent = self

        class FailedFeatures:
            def reproduce(self):
                return {"reproduced": False, "code": "FEATURE_SEMANTICS_INCOMPATIBLE"}

            def build(self, *args):
                parent.fail("target build must not follow failed reproduction")

        _, proofs, failures, ready = q._feature_continuity(self.assessment, self.names, FailedFeatures())
        self.assertFalse(ready)
        self.assertFalse(proofs)
        self.assertEqual([x["code"] for x in failures], [
            "FEATURE_SEMANTICS_INCOMPATIBLE", "CURRENT_FEATURE_CONTINUITY_BLOCKED_BY_SOURCE_QUALIFICATION"])

    def test_t5_unqualified_or_missing_target_blocks(self):
        for change in ({"qualified": False}, {"upcoming": []}):
            _, proofs, failures, ready = self.replay(assessment={**self.assessment, **change})
            self.assertFalse(ready)
            self.assertFalse(proofs)
            self.assertEqual(failures[0]["code"], "CURRENT_FEATURE_CONTINUITY_BLOCKED_BY_SOURCE_QUALIFICATION")

    def test_t5_causal_boundary_uses_pinned_inequalities(self):
        column = self.qualification["feature_proofs"][0]["cutoff_columns"][0]
        for value, permitted in ((self.assessment["as_of"], True),
                                 (pd.Timestamp(self.assessment["as_of"]) + pd.Timedelta(1, "ns"), False),
                                 (self.fixture.date, False)):
            fixture = self.fixture.copy(deep=True)
            fixture[column] = value
            _, _, failures, ready = self.replay(fixture=fixture)
            self.assertEqual(ready, permitted)
            if not permitted:
                self.assertEqual(failures[0]["code"], "FUTURE_FEATURE_SOURCE")

    def test_t5_retrospective_target_build_is_rejected(self):
        target = self.assessment["upcoming"][0]
        with self.assertRaisesRegex(ValueError, "RETROSPECTIVE_RECONSTRUCTION"):
            self.legacy.target_matrix(self.assessment, target, as_of=target["kickoff_at"], source_index=self.index)

    def test_t5_historical_metadata_cannot_supply_missing_future_observation_envelope(self):
        for metadata in self.index["sources"]:
            with self.subTest(source=metadata["raw"]["path"]):
                self.assertNotIn("archived_observation", metadata)
                with self.assertRaisesRegex(ValueError, "FUTURE_OBSERVATION_MISSING"):
                    q.observation(metadata, self.root, cutoff=self.assessment["as_of"])

    def test_t5_archived_replay_does_not_enable_synthetic_publication_interface(self):
        class ArchivedFeatures:
            execution_mode = "offline_archived_replay"

        with self.assertRaisesRegex(ValueError, "FUTURE_SYNTHETIC_SCIENCE_REQUIRED"):
            q.qualification_body(self.assessment, self.qualification["source_index"],
                                 science={"execution_mode": "offline_archived_replay"}, features=ArchivedFeatures())

    def test_t5_extraction_preserves_original_synthetic_record_bytes(self):
        import runpy
        original = runpy.run_path(str(self.root / "tmp/BL1-T5-parity-authorization-20261024-01/future_lineage_qualification.py.before"))
        target = {"fixture_id": "derived-synthetic-target", "kickoff_at": "2026-10-16T12:00:00Z"}
        assessment = {"qualified": True, "as_of": AS_OF, "upcoming": [target],
                      "composed_observations": [{"home_corners": 4, "kickoff_at": "2026-08-03T12:00:00Z"}]}
        for failure in (None, "reproduction", "nonfinite"):
            with self.subTest(failure=failure):
                old = original["qualification_body"](assessment, {}, science=SCIENCE, features=FixtureFeatures(failure))
                new = q.qualification_body(assessment, {}, science=SCIENCE, features=FixtureFeatures(failure))
                self.assertEqual(c.canonical(old), c.canonical(new))
