"""Phase 7D immutable provider-extension and causal-source primitives.

The protected historical base is never written here.  Provider observations are
normalized with exact repository mappings, persisted append-only, and selected
for a target fixture only when both event and observation timing are causal.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import unicodedata
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence, cast

from src.evaluation.phase7_shadow import canonical_json, sha256_bytes, sha256_json
from src.utils.naming import LEAGUE_ALIASES, LEAGUE_TEAMS, _normalization_candidates


EVENT_SCHEMA_VERSION = "ffps.phase7d_provider_event.v1"
REVISION_SCHEMA_VERSION = "ffps.phase7d_provider_event_revision.v1"
RAW_SOURCE_SCHEMA_VERSION = "ffps.phase7d_raw_source.v1"
CAUSAL_SOURCE_SCHEMA_VERSION = "ffps.phase7d_causal_source_bundle.v1"
PROVIDER_ID = "football-data.org:v4"
LEAGUE = "PL"
HISTORICAL_CUTOFF = "2025-12-28"
PROSPECTIVE_START = "2026-08-30"
PROSPECTIVE_END = "2026-12-31"

NORMALIZATION_CONTRACT = {
    "schema_version": "ffps.phase7d_normalization_contract.v1",
    "provider": PROVIDER_ID,
    "league": LEAGUE,
    "team_mapping": "exact canonical/alias candidates from src.utils.naming; fuzzy matching forbidden",
    "fixture_identity": "CanonicalDatasetBuilder SHA-256 prefix over league|season|date|home|away",
    "season_label": "YYYY-YY from provider season start year",
    "kickoff": "provider utcDate normalized to UTC",
    "completed_status": "FINISHED with integer non-negative fullTime home and away goals",
    "future_status": "non-FINISHED with both fullTime goals absent",
}
NORMALIZATION_CONTRACT_SHA256 = sha256_json(NORMALIZATION_CONTRACT)


class Phase7DError(RuntimeError):
    """Raised when provider continuity cannot be proven safely."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp(value: Any, label: str) -> datetime:
    text = str(value).strip()
    if not text:
        raise Phase7DError(f"{label} is missing")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        result = datetime.fromisoformat(text)
    except ValueError as exc:
        raise Phase7DError(f"{label} is not an ISO-8601 timestamp") from exc
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def _record_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): copy.deepcopy(item)
        for key, item in value.items()
        if key not in {"record_fingerprint", "record_integrity_sha256"}
    }


def record_fingerprint(value: Mapping[str, Any]) -> str:
    return sha256_json(_record_payload(value))


def seal_record(value: Mapping[str, Any]) -> dict[str, Any]:
    result = _record_payload(value)
    result["record_fingerprint"] = record_fingerprint(result)
    result["record_integrity_sha256"] = sha256_json(
        {key: item for key, item in result.items() if key != "record_integrity_sha256"}
    )
    return result


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
    payload: bytes,
    *,
    season: int,
    retrieved_at: str,
    request_url: str,
    response_headers: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    retrieved = _timestamp(retrieved_at, "retrieval timestamp")
    if season < 2000 or season > 2100:
        raise Phase7DError("provider season is invalid")
    safe_headers = {
        str(key).lower(): str(value)
        for key, value in (response_headers or {}).items()
        if str(key).lower() in {"date", "etag", "last-modified", "content-type"}
    }
    return {
        "schema_version": RAW_SOURCE_SCHEMA_VERSION,
        "provider": PROVIDER_ID,
        "competition": LEAGUE,
        "season": season,
        "retrieved_at": retrieved.isoformat(),
        "request_url": request_url,
        "response_sha256": sha256_bytes(payload),
        "response_size_bytes": len(payload),
        "response_headers": safe_headers,
        "credentials_retained": False,
    }


def season_label(season: int) -> str:
    return f"{season}-{(season + 1) % 100:02d}"


def _clean_team(value: str) -> str:
    return unicodedata.normalize("NFKD", value.strip()).encode("ascii", "ignore").decode("ascii").upper()


def strict_normalize_team(name: str, *, league: str = LEAGUE, season: int) -> str:
    """Resolve only exact canonical/alias candidates; never accept fuzzy output."""
    if league not in LEAGUE_TEAMS:
        raise Phase7DError(f"unsupported league mapping: {league}")
    candidates = _normalization_candidates(_clean_team(name))
    resolved: set[str] = set()
    season_maps = LEAGUE_TEAMS[league]
    ordered_seasons = [season, *sorted((value for value in season_maps if value != season), reverse=True)]
    for candidate in candidates:
        for mapping_season in ordered_seasons:
            canonicals = {str(value).upper() for value in season_maps.get(mapping_season, set())}
            aliases = {
                str(key).upper(): str(value).upper()
                for key, value in LEAGUE_ALIASES.get(league, {}).get(mapping_season, {}).items()
            }
            if candidate in canonicals:
                resolved.add(candidate)
            if candidate in aliases:
                resolved.add(aliases[candidate])
    if not resolved:
        raise Phase7DError(f"unresolved canonical team mapping: {name}")
    if len(resolved) != 1:
        raise Phase7DError(f"ambiguous canonical team mapping: {name} -> {sorted(resolved)}")
    return next(iter(resolved))


def canonical_fixture_identity(
    *, league: str, season: str, fixture_date: str, home_team: str, away_team: str
) -> str:
    raw = f"{league}|{season}|{fixture_date}|{home_team}|{away_team}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _goal(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Phase7DError(f"{label} must be a non-negative integer")
    return value


def normalize_provider_event(
    raw_event: Mapping[str, Any], *, raw_source: Mapping[str, Any]
) -> dict[str, Any]:
    if raw_source.get("schema_version") != RAW_SOURCE_SCHEMA_VERSION:
        raise Phase7DError("raw source identity is invalid")
    if raw_source.get("provider") != PROVIDER_ID:
        raise Phase7DError("provider identity mismatch")
    provider_event_id = str(raw_event.get("id", "")).strip()
    if not provider_event_id:
        raise Phase7DError("provider event identity is missing")
    season = int(raw_source["season"])
    kickoff = _timestamp(raw_event.get("utcDate"), "provider kickoff")
    home_raw = str(cast(Mapping[str, Any], raw_event.get("homeTeam", {})).get("name", ""))
    away_raw = str(cast(Mapping[str, Any], raw_event.get("awayTeam", {})).get("name", ""))
    home = strict_normalize_team(home_raw, season=season)
    away = strict_normalize_team(away_raw, season=season)
    if home == away:
        raise Phase7DError("canonical home and away teams collide")
    status = str(raw_event.get("status", "")).upper()
    score = cast(Mapping[str, Any], raw_event.get("score", {}))
    full_time = cast(Mapping[str, Any], score.get("fullTime", {}))
    raw_home = full_time.get("home")
    raw_away = full_time.get("away")
    completed = status == "FINISHED"
    if completed:
        home_goals = _goal(raw_home, "home goals")
        away_goals = _goal(raw_away, "away goals")
    else:
        if raw_home is not None or raw_away is not None:
            raise Phase7DError("future/result leakage: non-finished fixture contains a score")
        home_goals = None
        away_goals = None
    label = season_label(season)
    fixture_date = kickoff.date().isoformat()
    canonical_id = canonical_fixture_identity(
        league=LEAGUE,
        season=label,
        fixture_date=fixture_date,
        home_team=home,
        away_team=away,
    )
    normalized_event = {
        "provider_event_id": provider_event_id,
        "canonical_fixture_id": canonical_id,
        "league": LEAGUE,
        "season": label,
        "season_start_year": season,
        "kickoff_at": kickoff.isoformat(),
        "fixture_date": fixture_date,
        "home_team": home,
        "away_team": away,
        "status": status,
        "completed": completed,
        "home_goals": home_goals,
        "away_goals": away_goals,
    }
    value = {
        "schema_version": EVENT_SCHEMA_VERSION,
        "provider": PROVIDER_ID,
        "normalization_contract_sha256": NORMALIZATION_CONTRACT_SHA256,
        "normalized_event": normalized_event,
        "normalized_event_sha256": sha256_json(normalized_event),
        "raw_evidence": {
            "source_response_sha256": raw_source["response_sha256"],
            "source_response_size_bytes": raw_source["response_size_bytes"],
            "raw_event_sha256": sha256_json(raw_event),
            "retrieved_at": raw_source["retrieved_at"],
            "season": season,
        },
        "trust_assertions": {
            "provider_identity_verified": True,
            "team_mapping_exact_not_fuzzy": True,
            "fixture_identity_canonical": True,
            "future_score_absent": not completed,
            "completed_score_valid": completed,
        },
    }
    return seal_record(value)


def verify_event_record(value: Mapping[str, Any]) -> dict[str, Any]:
    errors = verify_seal(value)
    if value.get("schema_version") != EVENT_SCHEMA_VERSION:
        errors.append("unsupported event schema")
    if value.get("provider") != PROVIDER_ID:
        errors.append("provider identity mismatch")
    if value.get("normalization_contract_sha256") != NORMALIZATION_CONTRACT_SHA256:
        errors.append("normalization-contract drift")
    try:
        event = cast(Mapping[str, Any], value["normalized_event"])
        if value.get("normalized_event_sha256") != sha256_json(event):
            errors.append("normalized event hash mismatch")
        if event.get("league") != LEAGUE:
            errors.append("event league mismatch")
        kickoff = _timestamp(event["kickoff_at"], "event kickoff")
        if event.get("fixture_date") != kickoff.date().isoformat():
            errors.append("fixture date differs from kickoff")
        expected_id = canonical_fixture_identity(
            league=LEAGUE,
            season=str(event["season"]),
            fixture_date=str(event["fixture_date"]),
            home_team=str(event["home_team"]),
            away_team=str(event["away_team"]),
        )
        if event.get("canonical_fixture_id") != expected_id:
            errors.append("canonical fixture identity mismatch")
        completed = event.get("completed") is True
        if completed:
            _goal(event.get("home_goals"), "home goals")
            _goal(event.get("away_goals"), "away goals")
            if event.get("status") != "FINISHED":
                errors.append("completed event status mismatch")
        elif event.get("home_goals") is not None or event.get("away_goals") is not None:
            errors.append("future/result leakage")
        _timestamp(cast(Mapping[str, Any], value["raw_evidence"])["retrieved_at"], "retrieval timestamp")
    except (KeyError, TypeError, ValueError, Phase7DError) as exc:
        errors.append(str(exc))
    return {"verified": not errors, "errors": errors, "record_fingerprint": value.get("record_fingerprint")}


def build_revision_record(
    *,
    superseded_record: Mapping[str, Any],
    replacement_record: Mapping[str, Any],
    changed_fields: Sequence[str],
    reason: str,
    correction_recorded_at: str,
) -> dict[str, Any]:
    old = verify_event_record(superseded_record)
    new = verify_event_record(replacement_record)
    if not old["verified"] or not new["verified"]:
        raise Phase7DError("revision requires two verified event records")
    old_event = cast(Mapping[str, Any], superseded_record["normalized_event"])
    new_event = cast(Mapping[str, Any], replacement_record["normalized_event"])
    if old_event["provider_event_id"] != new_event["provider_event_id"]:
        raise Phase7DError("revision provider event identity changed")
    if not reason or not changed_fields:
        raise Phase7DError("revision reason and changed fields are required")
    return seal_record(
        {
            "schema_version": REVISION_SCHEMA_VERSION,
            "provider": PROVIDER_ID,
            "provider_event_id": old_event["provider_event_id"],
            "superseded_record_fingerprint": superseded_record["record_fingerprint"],
            "replacement_record_fingerprint": replacement_record["record_fingerprint"],
            "changed_fields": sorted({str(value) for value in changed_fields}),
            "reason": reason,
            "correction_recorded_at": _timestamp(
                correction_recorded_at, "correction timestamp"
            ).isoformat(),
        }
    )


def _safe_name(value: str) -> str:
    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")[:48] or "event"
    return f"{prefix}-{sha256_bytes(value.encode('utf-8'))[:12]}"


class ImmutableExtensionStore:
    """Append-only event observations and explicit revision links."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.events = self.root / "events"
        self.revisions = self.root / "revisions"
        self.locks = self.root / ".locks"
        for path in (self.events, self.revisions, self.locks):
            path.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _lock(self, provider_event_id: str) -> Iterator[None]:
        lock = self.locks / f"{_safe_name(provider_event_id)}.lock"
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise Phase7DError(f"extension event is locked: {provider_event_id}") from exc
        try:
            os.close(descriptor)
            yield
        finally:
            try:
                os.close(descriptor)
            except OSError:
                pass
            lock.unlink(missing_ok=True)

    @staticmethod
    def _write_immutable(path: Path, value: Mapping[str, Any]) -> str:
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing == value:
                return "ALREADY_PRESENT"
            raise Phase7DError(f"immutable extension conflict: {path.name}")
        temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        if json.loads(temporary.read_text(encoding="utf-8")) != value:
            temporary.unlink(missing_ok=True)
            raise Phase7DError("extension persistence verification failed")
        os.replace(temporary, path)
        return "CREATED"

    def _event_dir(self, provider_event_id: str) -> Path:
        return self.events / _safe_name(provider_event_id)

    def event_records(self, provider_event_id: str | None = None) -> list[dict[str, Any]]:
        paths = (
            sorted(self._event_dir(provider_event_id).glob("*.json"))
            if provider_event_id is not None
            else sorted(self.events.glob("*/*.json"))
        )
        return [cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8"))) for path in paths]

    def revision_records(self) -> list[dict[str, Any]]:
        return [
            cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
            for path in sorted(self.revisions.glob("*.json"))
        ]

    def persist_event(self, value: Mapping[str, Any]) -> str:
        verification = verify_event_record(value)
        if not verification["verified"]:
            raise Phase7DError(f"event record did not verify: {verification['errors']}")
        event = cast(Mapping[str, Any], value["normalized_event"])
        provider_event_id = str(event["provider_event_id"])
        with self._lock(provider_event_id):
            existing = self.event_records(provider_event_id)
            for record in existing:
                if (
                    record.get("normalized_event_sha256") == value.get("normalized_event_sha256")
                    and cast(Mapping[str, Any], record.get("raw_evidence", {})).get("raw_event_sha256")
                    == cast(Mapping[str, Any], value.get("raw_evidence", {})).get("raw_event_sha256")
                ):
                    return "ALREADY_PRESENT"
            if existing:
                raise Phase7DError(
                    "conflicting provider event requires an explicit append-only revision"
                )
            directory = self._event_dir(provider_event_id)
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{value['record_fingerprint']}.json"
            return self._write_immutable(path, value)

    def persist_revision(
        self, replacement: Mapping[str, Any], revision: Mapping[str, Any]
    ) -> str:
        verification = verify_event_record(replacement)
        if not verification["verified"]:
            raise Phase7DError(f"replacement event did not verify: {verification['errors']}")
        errors = verify_seal(revision)
        if revision.get("schema_version") != REVISION_SCHEMA_VERSION:
            errors.append("unsupported revision schema")
        event = cast(Mapping[str, Any], replacement["normalized_event"])
        provider_event_id = str(event["provider_event_id"])
        if revision.get("provider_event_id") != provider_event_id:
            errors.append("revision provider event mismatch")
        if revision.get("replacement_record_fingerprint") != replacement.get("record_fingerprint"):
            errors.append("revision replacement fingerprint mismatch")
        if errors:
            raise Phase7DError(f"revision did not verify: {errors}")
        with self._lock(provider_event_id):
            existing = self.event_records(provider_event_id)
            fingerprints = {record.get("record_fingerprint") for record in existing}
            if revision.get("superseded_record_fingerprint") not in fingerprints:
                raise Phase7DError("revision does not reference an existing event")
            directory = self._event_dir(provider_event_id)
            directory.mkdir(parents=True, exist_ok=True)
            event_path = directory / f"{replacement['record_fingerprint']}.json"
            revision_path = self.revisions / f"{revision['record_fingerprint']}.json"
            event_status = self._write_immutable(event_path, replacement)
            self._write_immutable(revision_path, revision)
            return event_status

    def effective_records(self, *, as_of: str | None = None) -> list[dict[str, Any]]:
        cutoff = _timestamp(as_of, "effective-record cutoff") if as_of else None
        revisions = self.revision_records()
        allowed_replacements = {
            str(record["replacement_record_fingerprint"]): record
            for record in revisions
            if cutoff is None
            or _timestamp(record["correction_recorded_at"], "correction timestamp") < cutoff
        }
        results: list[dict[str, Any]] = []
        for directory in sorted(path for path in self.events.iterdir() if path.is_dir()):
            records = [
                cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
                for path in sorted(directory.glob("*.json"))
            ]
            if cutoff is not None:
                records = [
                    record
                    for record in records
                    if _timestamp(
                        cast(Mapping[str, Any], record["raw_evidence"])["retrieved_at"],
                        "retrieval timestamp",
                    )
                    < cutoff
                ]
            if not records:
                continue
            replacements = [
                record
                for record in records
                if str(record["record_fingerprint"]) in allowed_replacements
            ]
            if replacements:
                selected = max(
                    replacements,
                    key=lambda record: _timestamp(
                        cast(Mapping[str, Any], record["raw_evidence"])["retrieved_at"],
                        "retrieval timestamp",
                    ),
                )
            else:
                revised_fingerprints = {
                    str(record["replacement_record_fingerprint"])
                    for record in revisions
                }
                initial = [
                    record
                    for record in records
                    if str(record["record_fingerprint"]) not in revised_fingerprints
                ]
                selected = min(
                    initial or records,
                    key=lambda record: _timestamp(
                        cast(Mapping[str, Any], record["raw_evidence"])["retrieved_at"],
                        "retrieval timestamp",
                    ),
                )
            results.append(selected)
        return results


def causal_extension_records(
    records: Sequence[Mapping[str, Any]],
    *,
    target_kickoff_at: str,
    prediction_at: str,
) -> list[dict[str, Any]]:
    target = _timestamp(target_kickoff_at, "target kickoff")
    prediction = _timestamp(prediction_at, "prediction timestamp")
    if prediction >= target:
        raise Phase7DError("prediction does not precede target cutoff")
    selected: list[dict[str, Any]] = []
    seen_provider: set[str] = set()
    seen_canonical: set[str] = set()
    for record in records:
        verification = verify_event_record(record)
        if not verification["verified"]:
            raise Phase7DError(f"extension integrity failure: {verification['errors']}")
        event = cast(Mapping[str, Any], record["normalized_event"])
        provider_id = str(event["provider_event_id"])
        canonical_id = str(event["canonical_fixture_id"])
        if provider_id in seen_provider:
            raise Phase7DError("duplicate provider event in effective extension")
        seen_provider.add(provider_id)
        if canonical_id in seen_canonical:
            raise Phase7DError("duplicate canonical fixture in effective extension")
        seen_canonical.add(canonical_id)
        if event.get("completed") is not True:
            continue
        event_time = _timestamp(event["kickoff_at"], "extension event kickoff")
        observed = _timestamp(
            cast(Mapping[str, Any], record["raw_evidence"])["retrieved_at"],
            "extension observation",
        )
        if event_time.date() >= target.date():
            continue
        if observed >= prediction:
            continue
        selected.append(copy.deepcopy(dict(record)))
    selected.sort(
        key=lambda record: (
            str(cast(Mapping[str, Any], record["normalized_event"])["kickoff_at"]),
            str(cast(Mapping[str, Any], record["normalized_event"])["canonical_fixture_id"]),
        )
    )
    return selected


def build_causal_source_bundle(
    *,
    protected_base_path: str,
    protected_base_sha256: str,
    extension_records: Sequence[Mapping[str, Any]],
    target_fixture_id: str,
    target_kickoff_at: str,
    prediction_at: str,
) -> dict[str, Any]:
    selected = causal_extension_records(
        extension_records,
        target_kickoff_at=target_kickoff_at,
        prediction_at=prediction_at,
    )
    fingerprints = [str(record["record_fingerprint"]) for record in selected]
    included_ids = [
        str(cast(Mapping[str, Any], record["normalized_event"])["canonical_fixture_id"])
        for record in selected
    ]
    maximum = max(
        (
            str(cast(Mapping[str, Any], record["normalized_event"])["kickoff_at"])
            for record in selected
        ),
        default=HISTORICAL_CUTOFF + "T00:00:00+00:00",
    )
    identity_payload = {
        "protected_base_sha256": protected_base_sha256,
        "normalization_contract_sha256": NORMALIZATION_CONTRACT_SHA256,
        "ordered_extension_record_fingerprints": fingerprints,
    }
    return {
        "schema_version": CAUSAL_SOURCE_SCHEMA_VERSION,
        "target_fixture_id": target_fixture_id,
        "target_kickoff_at": _timestamp(target_kickoff_at, "target kickoff").isoformat(),
        "prediction_at": _timestamp(prediction_at, "prediction timestamp").isoformat(),
        "protected_base": {"path": protected_base_path, "sha256": protected_base_sha256},
        "normalization_contract_sha256": NORMALIZATION_CONTRACT_SHA256,
        "included_extension_event_ids": included_ids,
        "ordered_extension_record_fingerprints": fingerprints,
        "extension_snapshot_sha256": sha256_json(fingerprints),
        "maximum_contributing_event_at": maximum,
        "causal_source_bundle_sha256": sha256_json(identity_payload),
        "causal_cutoff_verified": True,
    }


def extension_aggregate(records: Sequence[Mapping[str, Any]]) -> str:
    return sha256_json(sorted(str(record["record_fingerprint"]) for record in records))
