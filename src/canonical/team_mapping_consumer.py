"""Resolve projected provider fixtures against a reviewed Model E seed."""

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import date
from hashlib import sha256
import json
from pathlib import Path
import re
import stat

from src.data_foundation.source_manifest import (
    SourceManifestError,
    _held_bytes,
    _json_object,
    _manifest_entries,
)

from .clock import CanonicalError, utc
from .features import FEATURE_VERSION, _fresh
from .provider import LEAGUES


MAPPING_PATH = Path("data/canonical/provider-team-mapping.json")
RECEIPTS_DIR = "mapping-publication-receipts"
INPUT_PINS = {"proposals", "catalog", "decision_register", "espanyol_debt"}
SEED_WINDOW_ONLY = "OWNER_SEED_WINDOW_ONLY"
BINDINGS = {"BIND_LATEST_STATE", SEED_WINDOW_ONLY}
COLD_STARTS = {"OWNER_WITHHELD_COLD_START", "COLD_START_STALE", "COLD_START_NO_SUPPORTED_STATE"}
PROVIDER_SEED_IDENTITY = "provider_id"
RECEIPT_NAME = re.compile(r"[0-9a-f]{32}\.json")
SHA256 = re.compile(r"[0-9a-f]{64}")


def _has_sha256(value):
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


def _provider_id(value):
    return type(value) is int and value > 0


def _read_pinned(path, expected, kind):
    try:
        return _held_bytes(path, expected)
    except SourceManifestError as error:
        # The shared reader also refuses replaced files, links and reparse points.
        # Keep diagnostic status codes independent of exception text and file contents.
        if isinstance(error.__cause__, FileNotFoundError):
            raise CanonicalError(f"{kind}_UNAVAILABLE") from None
        raise CanonicalError(f"{kind}_PIN_OR_READ_FAILED") from None


def _object(raw, code):
    try:
        value = json.loads(raw, object_pairs_hook=_json_object)
    except (ValueError, RecursionError):
        raise CanonicalError(code) from None
    if not isinstance(value, dict):
        raise CanonicalError(code)
    return value


def _has_publication_receipt(destination, digest, inputs):
    folder = destination.parent / RECEIPTS_DIR
    reparse = stat.FILE_ATTRIBUTE_REPARSE_POINT
    try:
        info = folder.lstat()
        if not stat.S_ISDIR(info.st_mode) or getattr(info, "st_file_attributes", 0) & reparse:
            return False
        for path in sorted(folder.iterdir()):
            if not RECEIPT_NAME.fullmatch(path.name):
                continue
            try:
                info = path.lstat()
                if not stat.S_ISREG(info.st_mode) or getattr(info, "st_file_attributes", 0) & reparse:
                    continue
                record = _object(path.read_bytes(), "PUBLICATION_RECEIPT_INVALID")
                if (record.get("result") == "SUCCESS"
                        and record.get("disposition") == "PUBLISHED"
                        and record.get("operation") == "publish"
                        and isinstance(record.get("destination"), str)
                        and Path(record["destination"]) == destination
                        and record.get("input_sha256") == inputs
                        and record.get("output_sha256") == digest):
                    return True
            except (OSError, ValueError, CanonicalError):
                continue
    except OSError:
        return False
    return False


def _mapping_entries(document, inputs):
    if (type(document.get("schema_version")) is not int or document["schema_version"] != 1
            or document.get("lookup_mode") != "CLOSED_WORLD_EXACT"
            or document.get("input_sha256") != inputs
            or not isinstance(document.get("entries"), list) or not document["entries"]):
        raise CanonicalError("TEAM_MAPPING_INVALID")

    entries = defaultdict(list)
    for row in document["entries"]:
        if not isinstance(row, dict):
            raise CanonicalError("TEAM_MAPPING_INVALID")
        league, team_id = row.get("league"), row.get("provider_id")
        if league not in LEAGUES or not _provider_id(team_id):
            raise CanonicalError("TEAM_MAPPING_INVALID")
        key = (league, team_id)
        disposition = row.get("disposition")
        if isinstance(disposition, str) and disposition in BINDINGS:
            label = row.get("legacy_label")
            if row.get("kind") != "BINDING" or not isinstance(label, str) or not label.strip():
                raise CanonicalError("TEAM_MAPPING_INVALID")
            if disposition == SEED_WINDOW_ONLY:
                try:
                    through = row["valid_through"]
                    if date.fromisoformat(through).isoformat() != through:
                        raise ValueError
                except (KeyError, TypeError, ValueError):
                    raise CanonicalError("TEAM_MAPPING_INVALID") from None
            elif "valid_through" in row:
                raise CanonicalError("TEAM_MAPPING_INVALID")
        elif (not isinstance(disposition, str) or disposition not in COLD_STARTS
              or row.get("kind") != "COLD_START"
              or "legacy_label" in row or "valid_through" in row):
            raise CanonicalError("TEAM_MAPPING_INVALID")
        entries[key].append(row)
    return entries


def _verify_decisions(document, mapping, entries):
    if (type(document.get("schema_version")) is not int or document["schema_version"] != 1
            or not isinstance(document.get("recorded_owner_decisions"), list)):
        raise CanonicalError("IDENTITY_DECISIONS_INVALID")
    for decision in document["recorded_owner_decisions"]:
        if not isinstance(decision, dict) or decision.get("status") != "APPROVED":
            raise CanonicalError("IDENTITY_DECISIONS_INVALID")
        league, team_id = decision.get("league"), decision.get("provider_id")
        if league not in LEAGUES or not _provider_id(team_id):
            raise CanonicalError("IDENTITY_DECISIONS_INVALID")
        rows = entries.get((league, team_id), ())
        action = decision.get("decision_type")
        if action == "APPROVED_ALIAS":
            matches = any(row["kind"] == "BINDING" and row["legacy_label"] == decision.get("legacy_label")
                          for row in rows)
        elif action == "FIXTURE_EXCLUDED_AND_COUNTED":
            debt = mapping.get("espanyol_debt")
            update = debt.get("one_sided_update") if isinstance(debt, dict) else None
            if not isinstance(update, dict) or not isinstance(update.get("revisit_when"), str):
                raise CanonicalError("TEAM_MAPPING_DECISION_MISMATCH")
            original = {key: value for key, value in update.items() if key != "revisit_when"}
            matches = bool(rows) and {**debt, "one_sided_update": original} == decision
        else:
            raise CanonicalError("IDENTITY_DECISIONS_INVALID")
        if not matches:
            raise CanonicalError("TEAM_MAPPING_DECISION_MISMATCH")


def _serving_contract(model):
    try:
        serving = model.payload["serving"]
        mapping = serving["team_mapping"]
        inputs = mapping["accepted_input_sha256"]
        pins = {
            "source_manifest": serving["source_manifest_sha256"],
            "identity_decisions": inputs["decision_register"],
        }
        if not all(map(_has_sha256, (*pins.values(), mapping["mapping_sha256"]))):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise CanonicalError("TEAM_MAPPING_CONFIGURATION_REQUIRED") from None
    return mapping, pins


class TeamMappingConsumer:
    """Read-only resolution against a caller-supplied seed from a verified model."""

    @classmethod
    def for_model(cls, root, model, validation):
        """Take the mapping contract from the hash-verified model artifact."""
        mapping, pins = _serving_contract(model)
        try:
            proofs = validation["foundation_proofs"]
            source = proofs["source_manifest"]
            decisions = proofs["identity_decisions"]
            manifest_path = Path(source["path"])
            decision_path = Path(decisions["path"])
            if source["sha256"] != pins["source_manifest"]:
                raise CanonicalError("TEAM_MAPPING_SOURCE_MANIFEST_MISMATCH")
            if decisions["sha256"] != pins["identity_decisions"]:
                raise CanonicalError("TEAM_MAPPING_IDENTITY_DECISIONS_MISMATCH")
        except (KeyError, TypeError, ValueError):
            raise CanonicalError("TEAM_MAPPING_CONFIGURATION_REQUIRED") from None
        return cls.load(root, mapping_sha256=mapping["mapping_sha256"],
                        accepted_input_sha256=mapping["accepted_input_sha256"],
                        identity_decisions=decision_path, source_manifest=manifest_path,
                        source_manifest_sha256=pins["source_manifest"], feature_seed=model.seed)

    @classmethod
    def load(cls, root, *, mapping_sha256, accepted_input_sha256, identity_decisions,
             source_manifest, source_manifest_sha256, feature_seed):
        """Load the fixed publication path using independently reviewed pins.

        The source manifest is validated as metadata. Fixture rows come from
        the provider projection and this consumer never reads historical CSVs.
        """
        inputs = dict(accepted_input_sha256)
        if (inputs.keys() != INPUT_PINS or not all(map(_has_sha256, inputs.values()))
                or not _has_sha256(mapping_sha256) or not _has_sha256(source_manifest_sha256)):
            raise CanonicalError("TEAM_MAPPING_REVIEWED_PINS_REQUIRED")
        root = Path(root).absolute()
        destination = root / MAPPING_PATH
        raw = _read_pinned(destination, mapping_sha256, "TEAM_MAPPING")
        if not _has_publication_receipt(destination, mapping_sha256, inputs):
            raise CanonicalError("TEAM_MAPPING_PUBLICATION_RECEIPT_MISSING")
        mapping = _object(raw, "TEAM_MAPPING_INVALID")
        entries = _mapping_entries(mapping, inputs)

        decisions = _read_pinned(root / identity_decisions, inputs["decision_register"], "IDENTITY_DECISIONS")
        _verify_decisions(_object(decisions, "IDENTITY_DECISIONS_INVALID"), mapping, entries)

        manifest = _read_pinned(root / source_manifest, source_manifest_sha256, "SOURCE_MANIFEST")
        try:
            _manifest_entries(_object(manifest, "SOURCE_MANIFEST_INVALID"))
        except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
            raise CanonicalError("SOURCE_MANIFEST_INVALID") from None
        return cls(entries, feature_seed)

    def __init__(self, entries, feature_seed):
        if (not isinstance(feature_seed, dict) or feature_seed.get("feature_version") != FEATURE_VERSION
                or not isinstance(feature_seed.get("states"), list)):
            raise CanonicalError("TEAM_MAPPING_SEED_INVALID")
        last_day = feature_seed.get("last_day")
        if last_day is not None:
            try:
                if date.fromisoformat(last_day).isoformat() != last_day:
                    raise ValueError
            except (TypeError, ValueError):
                raise CanonicalError("TEAM_MAPPING_SEED_INVALID") from None
        self._entries = entries
        seed_identity = feature_seed.get("identity_key", "legacy_label")
        if seed_identity not in ("legacy_label", PROVIDER_SEED_IDENTITY):
            raise CanonicalError("TEAM_MAPPING_SEED_INVALID")
        self._provider_seed = seed_identity == PROVIDER_SEED_IDENTITY
        self._states = defaultdict(list)
        for row in feature_seed["states"]:
            try:
                league, label, state = row["league"], row["team_id"], row["state"]
                if league not in LEAGUES or not isinstance(label, str) or not isinstance(state, dict):
                    raise ValueError
                if self._provider_seed and (not label.isascii() or not label.isdecimal() or label[0] == "0"):
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                raise CanonicalError("TEAM_MAPPING_SEED_INVALID") from None
            self._states[league, label].append(deepcopy(state))
        self._binding_counts = Counter(
            (league, row["legacy_label"])
            for (league, _), rows in entries.items()
            for row in rows if row["kind"] == "BINDING"
        )

    def require_unrestricted_bindings(self):
        if any(row["disposition"] == SEED_WINDOW_ONLY for rows in self._entries.values() for row in rows):
            raise CanonicalError("ADOPTION_SEED_WINDOW_ONLY_BINDING")

    def _team(self, league, team_id, day):
        rows = self._entries.get((league, team_id), ())
        if not rows:
            raise CanonicalError("UNKNOWN_TEAM_ID")
        if len(rows) != 1:
            raise CanonicalError("AMBIGUOUS_TEAM_STATE")
        row, = rows
        if row["disposition"] == SEED_WINDOW_ONLY and day > row["valid_through"]:
            raise CanonicalError("TEAM_STATE_AFTER_SEED_WINDOW")
        cold_start = row["kind"] == "COLD_START"
        key = league, str(team_id) if self._provider_seed else row.get("legacy_label")
        states = self._states.get(key, ())
        if cold_start and not states:
            return {"provider_id": team_id, "disposition": row["disposition"], "state": _fresh()}
        if not states:
            raise CanonicalError("TEAM_STATE_MISSING")
        if len(states) != 1 or (not self._provider_seed and self._binding_counts[key] != 1):
            raise CanonicalError("AMBIGUOUS_TEAM_STATE")
        return {"provider_id": team_id, "disposition": row["disposition"], "state": deepcopy(states[0])}

    def resolve(self, fixtures):
        """Return one decision per input and a deterministic status summary.

        Reject every occurrence of a repeated provider fixture ID so a conflicting
        duplicate cannot win by arriving first. Counts refer to input occurrences.
        """
        fixtures = list(fixtures)
        ids = Counter(row.get("id") for row in fixtures
                      if isinstance(row, dict) and _provider_id(row.get("id")))
        decisions = []
        reasons = Counter()
        for row in fixtures:
            fixture_id = row.get("id") if isinstance(row, dict) else None
            fixture_id = fixture_id if _provider_id(fixture_id) else None
            side = None
            try:
                if fixture_id is None:
                    raise CanonicalError("INVALID_PROVIDER_FIXTURE")
                if ids[fixture_id] > 1:
                    raise CanonicalError("DUPLICATE_PROVIDER_FIXTURE")
                if "source_file" in row or "line" in row:
                    raise CanonicalError("NON_PROVIDER_FIXTURE")
                league, home, away = row.get("league"), row.get("home_id"), row.get("away_id")
                if league not in LEAGUES or not _provider_id(home) or not _provider_id(away) or home == away:
                    raise CanonicalError("INVALID_PROVIDER_FIXTURE")
                try:
                    day = utc(row.get("kickoff_utc"))[:10]
                except CanonicalError:
                    raise CanonicalError("INVALID_PROVIDER_FIXTURE") from None
                side = "home"
                home_state = self._team(league, home, day)
                side = "away"
                away_state = self._team(league, away, day)
            except CanonicalError as error:
                reasons[error.code] += 1
                decisions.append({"fixture_id": fixture_id, "reason": error.code, "side": side})
            else:
                decisions.append({"fixture_id": fixture_id, "home": home_state, "away": away_state})

        rejected = sum(reasons.values())
        return {
            "fixtures": decisions,
            "status": {"state": "DEGRADED" if rejected else "READY",
                       "resolved": len(decisions) - rejected, "rejected": rejected,
                       "rejection_reasons": dict(sorted(reasons.items()))},
        }
