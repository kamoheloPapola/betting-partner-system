"""Phase 7E multi-source continuity primitives.

The module composes fixture/result authority from football-data.org with the
full-time home/away corner fields published in football-data.co.uk CSV files.
It is evaluation-only and deliberately independent of production ingestion.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import os
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence, cast

from src.evaluation.phase7_shadow import sha256_json
from src.evaluation.phase7d_data_continuity import (
    PROVIDER_ID as PRIMARY_PROVIDER,
    canonical_fixture_identity,
    strict_normalize_team,
)


SUPPLEMENTAL_PROVIDER = "football-data.co.uk:csv"
RAW_SCHEMA_VERSION = "ffps.phase7e_supplemental_raw_source.v1"
SUPPLEMENTAL_EVENT_SCHEMA_VERSION = "ffps.phase7e_supplemental_event.v1"
PRIMARY_EVENT_SCHEMA_VERSION = "ffps.phase7e_primary_event_alias_extension.v1"
OBSERVATION_SCHEMA_VERSION = "ffps.phase7e_canonical_match_observation.v1"
REVISION_SCHEMA_VERSION = "ffps.phase7e_observation_revision.v1"
AUTHORITY_SCHEMA_VERSION = "ffps.phase7e_field_authority_matrix.v1"
ALIAS_SCHEMA_VERSION = "ffps.phase7e_team_alias_extension.v1"
CAUSAL_BUNDLE_SCHEMA_VERSION = "ffps.phase7e_multi_source_causal_bundle.v1"
HISTORICAL_CUTOFF = "2025-12-28"
PROSPECTIVE_START = "2026-08-30"
PROSPECTIVE_END = "2026-12-31"


class Phase7EError(RuntimeError):
    """Raised when supplemental evidence cannot be admitted safely."""


FIELD_AUTHORITY_MATRIX: dict[str, Any] = {
    "schema_version": AUTHORITY_SCHEMA_VERSION,
    "policy": "field_authority_composition",
    "fields": {
        "competition": {"primary": PRIMARY_PROVIDER, "fallback": SUPPLEMENTAL_PROVIDER},
        "home_team": {"primary": PRIMARY_PROVIDER, "fallback": SUPPLEMENTAL_PROVIDER},
        "away_team": {"primary": PRIMARY_PROVIDER, "fallback": SUPPLEMENTAL_PROVIDER},
        "kickoff_at": {"primary": PRIMARY_PROVIDER, "fallback": SUPPLEMENTAL_PROVIDER},
        "status": {"primary": PRIMARY_PROVIDER, "fallback": SUPPLEMENTAL_PROVIDER},
        "home_goals": {"primary": PRIMARY_PROVIDER, "fallback": SUPPLEMENTAL_PROVIDER},
        "away_goals": {"primary": PRIMARY_PROVIDER, "fallback": SUPPLEMENTAL_PROVIDER},
        "home_corners": {"primary": SUPPLEMENTAL_PROVIDER, "fallback": None},
        "away_corners": {"primary": SUPPLEMENTAL_PROVIDER, "fallback": None},
    },
    "fallback_rule": "PRIMARY_SOURCE_ABSENT_SUPPLEMENTAL_VERIFIED",
    "implicit_precedence_forbidden": True,
}
FIELD_AUTHORITY_MATRIX_SHA256 = sha256_json(FIELD_AUTHORITY_MATRIX)


TEAM_ALIAS_EXTENSION: dict[str, Any] = {
    "schema_version": ALIAS_SCHEMA_VERSION,
    "scope": "evaluation_only_post_protocol_PL_2026",
    "aliases": [
        {
            "canonical_team": "COVENTRY",
            "primary_provider_team_id": "1076",
            "primary_provider_name": "Coventry City FC",
            "supplemental_provider_name": "Coventry",
            "competition": "PL",
        },
        {
            "canonical_team": "HULL",
            "primary_provider_team_id": "322",
            "primary_provider_name": "Hull City AFC",
            "supplemental_provider_name": "Hull",
            "competition": "PL",
        },
    ],
    "name_similarity_alone_sufficient": False,
    "production_mapping_changed": False,
}
TEAM_ALIAS_EXTENSION_SHA256 = sha256_json(TEAM_ALIAS_EXTENSION)

SUPPLEMENTAL_NORMALIZATION_CONTRACT: dict[str, Any] = {
    "schema_version": "ffps.phase7e_supplemental_normalization_contract.v1",
    "provider": SUPPLEMENTAL_PROVIDER,
    "competition": {"raw": "Div=E0", "canonical": "PL"},
    "completed_status": "FTHG, FTAG, HC and AC must all be non-negative integers",
    "corner_semantics": {
        "HC": "full-time home-team corner kicks",
        "AC": "full-time away-team corner kicks",
        "orientation": "provider HomeTeam/AwayTeam",
        "extra_time_policy": "league match full-time value as published in the season CSV",
    },
    "fixture_join": "date + exact canonical home team + exact canonical away team",
    "date_rule": "provider Date is the match local calendar date; primary kickoff remains authoritative",
    "fuzzy_join_allowed": False,
    "authority_matrix_sha256": FIELD_AUTHORITY_MATRIX_SHA256,
    "team_alias_extension_sha256": TEAM_ALIAS_EXTENSION_SHA256,
}
SUPPLEMENTAL_NORMALIZATION_CONTRACT_SHA256 = sha256_json(
    SUPPLEMENTAL_NORMALIZATION_CONTRACT
)


def _timestamp(value: Any, label: str) -> datetime:
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise Phase7EError(f"{label} is not an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _payload(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): copy.deepcopy(item)
        for key, item in value.items()
        if key not in {"record_fingerprint", "record_integrity_sha256"}
    }


def record_fingerprint(value: Mapping[str, Any]) -> str:
    return sha256_json(_payload(value))


def seal_record(value: Mapping[str, Any]) -> dict[str, Any]:
    sealed = _payload(value)
    sealed["record_fingerprint"] = record_fingerprint(sealed)
    sealed["record_integrity_sha256"] = sha256_json(
        {key: item for key, item in sealed.items() if key != "record_integrity_sha256"}
    )
    return sealed


def verify_seal(value: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if value.get("record_fingerprint") != record_fingerprint(value):
        errors.append("record fingerprint mismatch")
    expected = sha256_json(
        {key: item for key, item in value.items() if key != "record_integrity_sha256"}
    )
    if value.get("record_integrity_sha256") != expected:
        errors.append("record integrity mismatch")
    return errors


def raw_source_identity(
    payload: bytes, *, season_token: str, retrieved_at: str, request_url: str
) -> dict[str, Any]:
    if not re.fullmatch(r"\d{4}", season_token):
        raise Phase7EError("supplemental season token must use YYZZ")
    return {
        "schema_version": RAW_SCHEMA_VERSION,
        "provider": SUPPLEMENTAL_PROVIDER,
        "season_token": season_token,
        "retrieved_at": _timestamp(retrieved_at, "supplemental retrieval").isoformat(),
        "request_url": request_url,
        "response_sha256": hashlib.sha256(payload).hexdigest(),
        "response_size_bytes": len(payload),
        "credentials_retained": False,
    }


def _alias_maps() -> tuple[dict[tuple[str, str], str], dict[str, str]]:
    primary: dict[tuple[str, str], str] = {}
    supplemental: dict[str, str] = {}
    for item in TEAM_ALIAS_EXTENSION["aliases"]:
        primary[(str(item["primary_provider_team_id"]), str(item["primary_provider_name"]))] = str(
            item["canonical_team"]
        )
        supplemental[str(item["supplemental_provider_name"])] = str(item["canonical_team"])
    return primary, supplemental


def normalize_team(
    name: str, *, provider: str, season_start_year: int, provider_team_id: str | None = None
) -> str:
    primary_aliases, supplemental_aliases = _alias_maps()
    if provider == PRIMARY_PROVIDER and provider_team_id is not None:
        alias = primary_aliases.get((str(provider_team_id), str(name)))
        if alias:
            return alias
    if provider == SUPPLEMENTAL_PROVIDER and name in supplemental_aliases:
        return supplemental_aliases[name]
    try:
        return strict_normalize_team(str(name), season=season_start_year)
    except Exception as exc:
        raise Phase7EError(f"unresolved {provider} team identity: {name}") from exc


def normalize_primary_event(
    raw_event: Mapping[str, Any], *, raw_source: Mapping[str, Any]
) -> dict[str, Any]:
    if raw_source.get("provider") != PRIMARY_PROVIDER:
        raise Phase7EError("primary event source identity mismatch")
    season = int(raw_source["season"])
    home_raw = cast(Mapping[str, Any], raw_event.get("homeTeam", {}))
    away_raw = cast(Mapping[str, Any], raw_event.get("awayTeam", {}))
    home = normalize_team(
        str(home_raw.get("name", "")),
        provider=PRIMARY_PROVIDER,
        season_start_year=season,
        provider_team_id=str(home_raw.get("id", "")),
    )
    away = normalize_team(
        str(away_raw.get("name", "")),
        provider=PRIMARY_PROVIDER,
        season_start_year=season,
        provider_team_id=str(away_raw.get("id", "")),
    )
    kickoff = _timestamp(raw_event.get("utcDate"), "primary kickoff")
    status = str(raw_event.get("status", ""))
    score = cast(Mapping[str, Any], cast(Mapping[str, Any], raw_event.get("score", {})).get("fullTime", {}))
    completed = status == "FINISHED"
    home_goals = score.get("home")
    away_goals = score.get("away")
    if completed:
        for label, value in (("home goals", home_goals), ("away goals", away_goals)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise Phase7EError(f"completed primary {label} is invalid")
    elif home_goals is not None or away_goals is not None:
        raise Phase7EError("future primary fixture contains result fields")
    canonical_id = canonical_fixture_identity(
        league="PL",
        season=f"{season}-{str(season + 1)[-2:]}",
        fixture_date=kickoff.date().isoformat(),
        home_team=home,
        away_team=away,
    )
    return seal_record(
        {
            "schema_version": PRIMARY_EVENT_SCHEMA_VERSION,
            "provider": PRIMARY_PROVIDER,
            "provider_event_id": str(raw_event.get("id", "")),
            "canonical_fixture_id": canonical_id,
            "fixture_date": kickoff.date().isoformat(),
            "kickoff_at": kickoff.isoformat(),
            "season_start_year": season,
            "league": "PL",
            "home_team": home,
            "away_team": away,
            "status": status,
            "completed": completed,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "raw_source_sha256": raw_source["response_sha256"],
            "observed_at": _timestamp(raw_source["retrieved_at"], "primary observation").isoformat(),
            "team_alias_extension_sha256": TEAM_ALIAS_EXTENSION_SHA256,
        }
    )


def _decode_csv(payload: bytes) -> tuple[str, str]:
    try:
        return payload.decode("utf-8-sig"), "utf-8-sig"
    except UnicodeDecodeError:
        return payload.decode("cp1252"), "cp1252"


def _match_date(value: str) -> str:
    text = str(value).strip()
    for pattern in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(text, pattern).date().isoformat()
        except ValueError:
            pass
    raise Phase7EError(f"unsupported supplemental match date: {value}")


def _non_negative_int(value: Any, label: str) -> int:
    text = str(value).strip()
    try:
        parsed = int(text)
    except ValueError as exc:
        raise Phase7EError(f"{label} is missing or not an integer") from exc
    if parsed < 0:
        raise Phase7EError(f"{label} is negative")
    return parsed


def normalize_supplemental_csv(
    payload: bytes, *, raw_source: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if raw_source.get("provider") != SUPPLEMENTAL_PROVIDER:
        raise Phase7EError("supplemental source identity mismatch")
    if hashlib.sha256(payload).hexdigest() != raw_source.get("response_sha256"):
        raise Phase7EError("supplemental raw-source hash mismatch")
    text, encoding = _decode_csv(payload)
    reader = csv.DictReader(io.StringIO(text))
    season_token = str(raw_source["season_token"])
    season = 2000 + int(season_token[:2])
    events: list[dict[str, Any]] = []
    for row_number, row in enumerate(reader, start=2):
        if str(row.get("Div", "")).strip() != "E0":
            continue
        required = ("Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HC", "AC")
        if any(not str(row.get(name, "")).strip() for name in required):
            continue
        fixture_date = _match_date(str(row["Date"]))
        home = normalize_team(
            str(row["HomeTeam"]),
            provider=SUPPLEMENTAL_PROVIDER,
            season_start_year=season,
        )
        away = normalize_team(
            str(row["AwayTeam"]),
            provider=SUPPLEMENTAL_PROVIDER,
            season_start_year=season,
        )
        home_goals = _non_negative_int(row["FTHG"], "supplemental home goals")
        away_goals = _non_negative_int(row["FTAG"], "supplemental away goals")
        home_corners = _non_negative_int(row["HC"], "supplemental home corners")
        away_corners = _non_negative_int(row["AC"], "supplemental away corners")
        canonical_id = canonical_fixture_identity(
            league="PL",
            season=f"{season}-{str(season + 1)[-2:]}",
            fixture_date=fixture_date,
            home_team=home,
            away_team=away,
        )
        row_identity = sha256_json({str(key): str(value) for key, value in sorted(row.items())})
        events.append(
            seal_record(
                {
                    "schema_version": SUPPLEMENTAL_EVENT_SCHEMA_VERSION,
                    "provider": SUPPLEMENTAL_PROVIDER,
                    "provider_event_id": f"E0:{season_token}:{row_number}:{row_identity[:16]}",
                    "provider_native_event_id_available": False,
                    "canonical_fixture_id": canonical_id,
                    "fixture_date": fixture_date,
                    "season_start_year": season,
                    "league": "PL",
                    "home_team": home,
                    "away_team": away,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "home_corners": home_corners,
                    "away_corners": away_corners,
                    "completed": True,
                    "raw_source_sha256": raw_source["response_sha256"],
                    "raw_row_sha256": row_identity,
                    "raw_row_number": row_number,
                    "observed_at": _timestamp(
                        raw_source["retrieved_at"], "supplemental observation"
                    ).isoformat(),
                    "source_encoding": encoding,
                    "normalization_contract_sha256": SUPPLEMENTAL_NORMALIZATION_CONTRACT_SHA256,
                }
            )
        )
    return events


def _field(
    *, name: str, value: Any, source: Mapping[str, Any], authority_state: str
) -> dict[str, Any]:
    provider = str(source["provider"])
    policy = cast(Mapping[str, Any], FIELD_AUTHORITY_MATRIX["fields"])[name]
    primary = cast(Mapping[str, Any], policy)["primary"]
    fallback = cast(Mapping[str, Any], policy)["fallback"]
    if provider != primary:
        if provider != fallback or authority_state != FIELD_AUTHORITY_MATRIX["fallback_rule"]:
            raise Phase7EError(f"unauthorized provider for field {name}: {provider}")
    return {
        "value": value,
        "provider": provider,
        "provider_event_id": source["provider_event_id"],
        "raw_source_sha256": source["raw_source_sha256"],
        "observed_at": source["observed_at"],
        "normalization_contract_sha256": (
            source.get("normalization_contract_sha256")
            or source.get("team_alias_extension_sha256")
        ),
        "authority_state": authority_state,
    }


def compose_observation(
    primary: Mapping[str, Any], supplemental: Mapping[str, Any]
) -> dict[str, Any]:
    if verify_seal(primary) or verify_seal(supplemental):
        raise Phase7EError("unverified source record cannot be composed")
    for name in ("fixture_date", "home_team", "away_team"):
        if primary.get(name) != supplemental.get(name):
            raise Phase7EError(f"cross-provider {name} mismatch")
    if primary.get("completed") is not True or supplemental.get("completed") is not True:
        raise Phase7EError("only completed events can become canonical observations")
    if (
        primary.get("home_goals") != supplemental.get("home_goals")
        or primary.get("away_goals") != supplemental.get("away_goals")
    ):
        raise Phase7EError("cross-provider final-score mismatch")
    if primary.get("canonical_fixture_id") != supplemental.get("canonical_fixture_id"):
        raise Phase7EError("cross-provider canonical fixture mismatch")
    primary_state = "PRIMARY_AUTHORITY"
    supplemental_state = "SUPPLEMENTAL_FIELD_AUTHORITY"
    fields = {
        "competition": _field(name="competition", value="PL", source=primary, authority_state=primary_state),
        "home_team": _field(name="home_team", value=primary["home_team"], source=primary, authority_state=primary_state),
        "away_team": _field(name="away_team", value=primary["away_team"], source=primary, authority_state=primary_state),
        "kickoff_at": _field(name="kickoff_at", value=primary["kickoff_at"], source=primary, authority_state=primary_state),
        "status": _field(name="status", value=primary["status"], source=primary, authority_state=primary_state),
        "home_goals": _field(name="home_goals", value=primary["home_goals"], source=primary, authority_state=primary_state),
        "away_goals": _field(name="away_goals", value=primary["away_goals"], source=primary, authority_state=primary_state),
        "home_corners": _field(name="home_corners", value=supplemental["home_corners"], source=supplemental, authority_state=supplemental_state),
        "away_corners": _field(name="away_corners", value=supplemental["away_corners"], source=supplemental, authority_state=supplemental_state),
    }
    available = max(
        _timestamp(primary["observed_at"], "primary observation"),
        _timestamp(supplemental["observed_at"], "supplemental observation"),
    )
    return seal_record(
        {
            "schema_version": OBSERVATION_SCHEMA_VERSION,
            "canonical_fixture_id": primary["canonical_fixture_id"],
            "fixture_date": primary["fixture_date"],
            "kickoff_at": primary["kickoff_at"],
            "season_start_year": primary["season_start_year"],
            "home_team": primary["home_team"],
            "away_team": primary["away_team"],
            "completed": True,
            "availability_at": available.isoformat(),
            "fields": fields,
            "primary_record_fingerprint": primary["record_fingerprint"],
            "supplemental_record_fingerprint": supplemental["record_fingerprint"],
            "field_authority_matrix_sha256": FIELD_AUTHORITY_MATRIX_SHA256,
            "team_alias_extension_sha256": TEAM_ALIAS_EXTENSION_SHA256,
        }
    )


def verify_observation(value: Mapping[str, Any]) -> dict[str, Any]:
    errors = verify_seal(value)
    if value.get("schema_version") != OBSERVATION_SCHEMA_VERSION:
        errors.append("unsupported observation schema")
    if value.get("completed") is not True:
        errors.append("observation is not completed")
    fields = cast(Mapping[str, Any], value.get("fields", {}))
    for name, policy in cast(Mapping[str, Any], FIELD_AUTHORITY_MATRIX["fields"]).items():
        field = cast(Mapping[str, Any], fields.get(name, {}))
        if field.get("value") is None:
            errors.append(f"required field missing: {name}")
            continue
        provider = field.get("provider")
        primary = cast(Mapping[str, Any], policy).get("primary")
        fallback = cast(Mapping[str, Any], policy).get("fallback")
        state = field.get("authority_state")
        if provider != primary and not (
            provider == fallback and state == FIELD_AUTHORITY_MATRIX["fallback_rule"]
        ):
            errors.append(f"unauthorized provider for field {name}")
    for name in ("home_goals", "away_goals", "home_corners", "away_corners"):
        value_field = cast(Mapping[str, Any], fields.get(name, {})).get("value")
        if isinstance(value_field, bool) or not isinstance(value_field, int) or value_field < 0:
            errors.append(f"invalid value for {name}")
    return {"verified": not errors, "errors": errors}


def observation_aggregate(records: Sequence[Mapping[str, Any]]) -> str:
    return sha256_json(sorted(str(value["record_fingerprint"]) for value in records))


def unique_fixture_index(
    records: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Index exact fixture keys and reject ambiguity or reversed duplicates."""
    indexed: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in records:
        key = (
            str(record["fixture_date"]),
            str(record["home_team"]),
            str(record["away_team"]),
        )
        reverse = (key[0], key[2], key[1])
        if key in indexed:
            raise Phase7EError("ambiguous duplicate fixture join")
        if reverse in indexed:
            raise Phase7EError("reversed home/away fixture join")
        indexed[key] = copy.deepcopy(dict(record))
    return indexed


def causal_observations(
    records: Sequence[Mapping[str, Any]], *, target_kickoff_at: str, prediction_at: str
) -> list[dict[str, Any]]:
    target = _timestamp(target_kickoff_at, "target kickoff")
    prediction = _timestamp(prediction_at, "prediction timestamp")
    if prediction >= target:
        raise Phase7EError("prediction does not precede target kickoff")
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        verification = verify_observation(record)
        if not verification["verified"]:
            raise Phase7EError(f"invalid composed observation: {verification['errors']}")
        fixture_id = str(record["canonical_fixture_id"])
        if fixture_id in seen:
            raise Phase7EError("duplicate canonical observation")
        seen.add(fixture_id)
        kickoff = _timestamp(record["kickoff_at"], "observation kickoff")
        available = _timestamp(record["availability_at"], "observation availability")
        if kickoff.date() >= target.date() or available > prediction:
            continue
        selected.append(copy.deepcopy(dict(record)))
    selected.sort(key=lambda item: (str(item["kickoff_at"]), str(item["canonical_fixture_id"])))
    return selected


def build_causal_bundle(
    *,
    protected_base_sha256: str,
    primary_extension_sha256: str,
    supplemental_source_sha256: str,
    observations: Sequence[Mapping[str, Any]],
    target_fixture_id: str,
    target_kickoff_at: str,
    prediction_at: str,
) -> dict[str, Any]:
    included = causal_observations(
        observations,
        target_kickoff_at=target_kickoff_at,
        prediction_at=prediction_at,
    )
    fingerprints = [str(value["record_fingerprint"]) for value in included]
    payload = {
        "protected_base_sha256": protected_base_sha256,
        "primary_extension_sha256": primary_extension_sha256,
        "supplemental_source_sha256": supplemental_source_sha256,
        "field_authority_matrix_sha256": FIELD_AUTHORITY_MATRIX_SHA256,
        "team_alias_extension_sha256": TEAM_ALIAS_EXTENSION_SHA256,
        "ordered_observation_fingerprints": fingerprints,
    }
    maximum = max(
        (str(value["kickoff_at"]) for value in included),
        default=HISTORICAL_CUTOFF + "T00:00:00+00:00",
    )
    return {
        "schema_version": CAUSAL_BUNDLE_SCHEMA_VERSION,
        "target_fixture_id": target_fixture_id,
        "target_kickoff_at": _timestamp(target_kickoff_at, "target kickoff").isoformat(),
        "prediction_at": _timestamp(prediction_at, "prediction timestamp").isoformat(),
        "protected_base_sha256": protected_base_sha256,
        "primary_extension_sha256": primary_extension_sha256,
        "supplemental_source_sha256": supplemental_source_sha256,
        "field_authority_matrix_sha256": FIELD_AUTHORITY_MATRIX_SHA256,
        "team_alias_extension_sha256": TEAM_ALIAS_EXTENSION_SHA256,
        "included_observation_ids": [str(value["canonical_fixture_id"]) for value in included],
        "ordered_observation_fingerprints": fingerprints,
        "maximum_contributing_event_at": maximum,
        "multi_source_continuity_sha256": sha256_json(payload),
        "causal_cutoff_verified": True,
    }


class ImmutableObservationStore:
    """Lock-protected immutable observations with explicit correction revisions."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.observations = self.root / "observations"
        self.revisions = self.root / "revisions"
        self.locks = self.root / ".locks"
        for path in (self.observations, self.revisions, self.locks):
            path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _name(fixture_id: str) -> str:
        prefix = re.sub(r"[^A-Za-z0-9_.-]+", "-", fixture_id).strip("-.")[:48] or "fixture"
        return f"{prefix}-{hashlib.sha256(fixture_id.encode()).hexdigest()[:16]}.json"

    @contextmanager
    def _lock(self, fixture_id: str) -> Iterator[None]:
        path = self.locks / f"{hashlib.sha256(fixture_id.encode()).hexdigest()}.lock"
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise Phase7EError(f"observation is locked: {fixture_id}") from exc
        try:
            os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
            os.close(descriptor)
            yield
        finally:
            try:
                os.close(descriptor)
            except OSError:
                pass
            path.unlink(missing_ok=True)

    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))

    @staticmethod
    def _write(path: Path, value: Mapping[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, path)

    def persist(self, observation: Mapping[str, Any]) -> str:
        verification = verify_observation(observation)
        if not verification["verified"]:
            raise Phase7EError(f"invalid observation: {verification['errors']}")
        fixture_id = str(observation["canonical_fixture_id"])
        path = self.observations / self._name(fixture_id)
        with self._lock(fixture_id):
            if path.exists():
                existing = self._read(path)
                if existing == observation:
                    return "ALREADY_PRESENT"
                raise Phase7EError("conflicting observation requires explicit revision")
            self._write(path, observation)
        return "CREATED"

    def records(self) -> list[dict[str, Any]]:
        return [self._read(path) for path in sorted(self.observations.glob("*.json"))]

    def persist_revision(
        self,
        *,
        superseded_fingerprint: str,
        replacement: Mapping[str, Any],
        changed_fields: Sequence[str],
        reason: str,
        corrected_at: str,
    ) -> dict[str, Any]:
        if not changed_fields or not reason:
            raise Phase7EError("revision changed fields and reason are required")
        current = {str(value["record_fingerprint"]): value for value in self.effective_records()}
        previous = current.get(superseded_fingerprint)
        if previous is None:
            raise Phase7EError("superseded observation is unavailable")
        if previous["canonical_fixture_id"] != replacement.get("canonical_fixture_id"):
            raise Phase7EError("revision cannot change canonical fixture identity")
        if not verify_observation(replacement)["verified"]:
            raise Phase7EError("replacement observation is invalid")
        revision = seal_record(
            {
                "schema_version": REVISION_SCHEMA_VERSION,
                "canonical_fixture_id": previous["canonical_fixture_id"],
                "superseded_record_fingerprint": superseded_fingerprint,
                "replacement_observation": copy.deepcopy(dict(replacement)),
                "changed_fields": sorted({str(value) for value in changed_fields}),
                "reason": reason,
                "corrected_at": _timestamp(corrected_at, "correction timestamp").isoformat(),
            }
        )
        path = self.revisions / f"{revision['record_fingerprint']}.json"
        if path.exists() and self._read(path) == revision:
            return revision
        if path.exists():
            raise Phase7EError("conflicting observation revision")
        self._write(path, revision)
        return revision

    def effective_records(self, *, as_of: str | None = None) -> list[dict[str, Any]]:
        records = {str(value["record_fingerprint"]): value for value in self.records()}
        cutoff = _timestamp(as_of, "effective cutoff") if as_of else None
        revisions = [self._read(path) for path in sorted(self.revisions.glob("*.json"))]
        revisions.sort(key=lambda value: (str(value["corrected_at"]), str(value["record_fingerprint"])))
        for revision in revisions:
            if verify_seal(revision) or revision.get("schema_version") != REVISION_SCHEMA_VERSION:
                raise Phase7EError("invalid observation revision")
            if cutoff and _timestamp(revision["corrected_at"], "correction timestamp") > cutoff:
                continue
            previous = str(revision["superseded_record_fingerprint"])
            if previous not in records:
                raise Phase7EError("revision chain is discontinuous")
            del records[previous]
            replacement = cast(dict[str, Any], revision["replacement_observation"])
            records[str(replacement["record_fingerprint"])] = replacement
        return sorted(
            records.values(),
            key=lambda value: (str(value["kickoff_at"]), str(value["canonical_fixture_id"])),
        )
