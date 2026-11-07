"""Validate local preservation acceptance data without recording an event.

This module has no writer, clock, publisher, or scientific integration. A valid
draft is not acceptance. Completed input requires separately supplied recording,
authorization, and uniqueness context; validation cannot authenticate that context.
"""
from __future__ import annotations

import calendar
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


SCHEMA_VERSION = 'ffps.phase20_local_preservation_acceptance.v1'
DESTINATION = (r'C:\Users\papol\my_systems\betting_partner_system\docs'
               r'\phase20-local-preservation-acceptance-v1.json')
PACKAGE_ID = 'phase20-local-20261019T174318Z-fbd145f179a742358bbea7076c65112d'
PACKAGE_ROOT = r'C:\Users\papol\my_systems\phase20_evidence_archive' + '\\' + PACKAGE_ID
MANIFEST_SHA256 = '3312cd4f868f7bb9bcef26646ce9d971513d546ea332690c498cf7009818d014'
TIMEZONE = 'Africa/Johannesburg'
WRITE_OPERATION = 'EXCLUSIVE_CREATE_ACCEPTANCE_RECORD'
UNIQUENESS_SCOPE = 'PHASE20_LOCAL_PRESERVATION_ACCEPTANCE_DECISIONS'
DRAFT = 'DRAFT_NOT_RECORDED'
RECORDED = 'RECORDED'


class AcceptanceValidationError(ValueError):
    """A coded, fail-closed validation failure."""


@dataclass(frozen=True)
class WriteAuthorization:
    """Caller-authenticated authorization, never inferred from record fields."""

    reference: str
    sha256: str
    destination: str
    operation: str
    schema_version: str
    owner_stated_decision_date: str


@dataclass(frozen=True)
class UniquenessCheck:
    """Caller supplies a complete authoritative snapshot for the named scope.

    A caller must serialize this check with any later publication. This object
    neither reserves an ID nor establishes repository-wide uniqueness.
    """

    existing_ids: frozenset[str]
    scope: str
    complete: bool
    reference: str


@dataclass(frozen=True)
class RecordingContext:
    """Externally established facts about the recording event being validated."""

    event_utc: str
    observed_at_utc: str
    authorization: WriteAuthorization
    uniqueness: UniquenessCheck


@dataclass(frozen=True)
class ValidationResult:
    """Input validity only; never publication, authorization, or package assurance."""

    valid: bool
    record_status: str | None = None
    errors: tuple[str, ...] = ()
    self_seal: str = 'NOT_CHECKED'
    external_digest: str = 'NOT_CHECKED'
    uniqueness: str = 'NOT_ESTABLISHED'
    independent_anchoring: str = 'NOT_ESTABLISHED'


def _require(condition: bool, code: str, path: str = '') -> None:
    if not condition:
        raise AcceptanceValidationError('P20_ACCEPTANCE_' + code + (':' + path if path else ''))


def _contract(recorded: bool) -> dict:
    """Exact v1 field sets; type markers denote variable native scalar fields."""
    return {
        'schema_version': SCHEMA_VERSION,
        'decision': {
            'decision_id': str,
            'decision': 'ACCEPT_WITH_DOCUMENTED_LIMITATIONS',
            'owner_stated_decision_date': str,
            'authority': {
                'name': 'Kamohelo Papola', 'windows_account': r'FRIDAY\papol',
                'basis': 'OWNER_SELF_ATTESTATION', 'independent_verification': 'NOT_ESTABLISHED',
            },
            'write_authorization': {'reference': str, 'sha256': str} if recorded else None,
        },
        'recording': {
            'record_status': RECORDED if recorded else DRAFT,
            'created_at_utc': str if recorded else None,
            'timezone': TIMEZONE, 'acceptance_date': str if recorded else None,
            'acceptance_date_source': 'ACTUAL_LOCAL_RECORDING_EVENT',
        },
        'package': {
            'identifier': PACKAGE_ID, 'root': PACKAGE_ROOT, 'manifest_sha256': MANIFEST_SHA256,
            'payload_file_count': 16, 'payload_bytes': 4552385,
            'historical_discrepancy_count': 10, 'discrepancies_preserved': True,
            'discrepancies_resolved': False, 'discrepancies_restored': False,
            'identity_provenance': 'PREVIOUSLY_REPORTED',
            'independently_reverified_for_acceptance': False,
        },
        'verification': {
            'existing_record_reference': PACKAGE_ROOT + r'\records\single-operator-verification.json',
            'capture_result': 'CAPTURE_PASS', 'verification_result': 'READ_ONLY_VERIFICATION_PASS',
            'provenance': 'PREVIOUSLY_REPORTED', 'classification': 'SINGLE_OPERATOR_VERIFICATION',
            'independence': 'NOT_ESTABLISHED', 'independent_assurance': 'NOT_ESTABLISHED',
        },
        'limitations': {
            'single_volume_loss_risk': 'ACCEPTED',
            'independent_backup': 'NOT_ESTABLISHED; DEFERRED',
            'encryption': 'UNVERIFIED; VERIFICATION DEFERRED',
            'independent_manifest_digest_anchoring': 'NOT_ESTABLISHED; DEFERRED',
            'strong_immutability': 'NOT_ESTABLISHED', 'privileged_local_access_remains': True,
            'disaster_recovery_readiness': 'NOT_ESTABLISHED',
            'independent_verification': 'NOT_ESTABLISHED',
            'acceptance_record_independent_anchoring': 'NOT_ESTABLISHED',
        },
        'retention': {
            'status': 'STARTED' if recorded else 'NOT_STARTED',
            'trigger': 'FORMAL_ACCEPTANCE_DATE_P', 'calendar_months': 12,
            'missing_day_convention': 'LAST_VALID_DAY_OF_TARGET_MONTH',
            'start': str if recorded else None, 'minimum_end': str if recorded else None,
            'review_date': str if recorded else None, 'automatic_deletion': False,
        },
        'destruction': {
            'minimum_expiry_is_authorization': False, 'investigation_closed_required': True,
            'minimum_retention_elapsed_required': True, 'separate_explicit_decision_required': True,
            'authorized': False, 'authorization_reference': None,
        },
        'boundaries': {
            'package_and_historical_receipts_unchanged': True, 'restoration_authorized': False,
            'scientific_boundary_amendment_authorized': False, 'evidence_deletion_authorized': False,
            'phase20_closure_authorized': False, 'acceptance_is_scientific_clearance': False,
            'phase20_status': 'BLOCKED',
        },
        'record_sha256': str if recorded else None,
    }


def _shape(value, expected, path: str = '') -> None:
    if expected is str:
        _require(type(value) is str, 'TYPE', path)
        return
    _require(type(value) is type(expected), 'TYPE', path)
    if type(expected) is dict:
        _require(set(value) == set(expected), 'FIELDS', path)
        for key, child in expected.items():
            _shape(value[key], child, path + '.' + key if path else key)
    else:
        _require(value == expected, 'VALUE', path)


def _native_json(value) -> None:
    if value is None or type(value) in (bool, int):
        return
    if type(value) is str:
        value.encode('utf-8')
        return
    if type(value) is float:
        _require(math.isfinite(value), 'NONFINITE_JSON')
        return
    if type(value) is dict:
        _require(all(type(k) is str for k in value), 'JSON_KEY_TYPE')
        for key, item in value.items():
            _native_json(key)
            _native_json(item)
        return
    if type(value) is list:
        for item in value:
            _native_json(item)
        return
    _require(False, 'JSON_TYPE')


def canonical_bytes(value) -> bytes:
    """Repository canonical JSON for native JSON only, without coercing types.

    Matches src.predictions.provenance on this subset: UTF-8, sorted keys,
    compact separators, unescaped Unicode, no NaN/Infinity, BOM, or newline.
    This is serialization, not schema validation or permission to publish.
    """
    _native_json(value)
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def internal_record_sha256(value: dict) -> str:
    """Compute a seal without assigning it; exclude only the top-level seal key."""
    _require(type(value) is dict, 'ROOT_OBJECT_REQUIRED')
    return hashlib.sha256(canonical_bytes({k: v for k, v in value.items()
                                           if k != 'record_sha256'})).hexdigest()


def _digest(value, path: str) -> None:
    _require(type(value) is str and re.fullmatch(r'[0-9a-f]{64}', value) is not None,
             'DIGEST_FORMAT', path)


def _date(value: str) -> date:
    _require(type(value) is str and re.fullmatch(r'\d{4}-\d{2}-\d{2}', value) is not None,
             'DATE_FORMAT')
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise AcceptanceValidationError('P20_ACCEPTANCE_DATE_INVALID') from exc


def _instant(value: str) -> datetime:
    _require(type(value) is str and re.fullmatch(
        r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|\+00:00)', value
    ) is not None, 'UTC_TIMESTAMP_FORMAT')
    try:
        return datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as exc:
        raise AcceptanceValidationError('P20_ACCEPTANCE_UTC_TIMESTAMP_INVALID') from exc


def add_calendar_months(value: date, months: int) -> date:
    """Calendar addition with the approved last-valid-day convention."""
    _require(type(value) is date and type(months) is int and months >= 0, 'CALENDAR_INPUT')
    year, month = divmod(value.year * 12 + value.month - 1 + months, 12)
    month += 1
    _require(1 <= year <= 9999, 'CALENDAR_RANGE')
    return date(year, month, min(value.day, calendar.monthrange(year, month)[1]))


def destruction_prerequisites_met(*, investigation_closed: bool,
                                 minimum_retention_elapsed: bool,
                                 separate_explicit_decision: bool) -> bool:
    """Policy truth table only; authenticates no decision and authorizes no deletion."""
    values = (investigation_closed, minimum_retention_elapsed, separate_explicit_decision)
    _require(all(type(v) is bool for v in values), 'DESTRUCTION_BOOL_REQUIRED')
    return all(values)


def _validate(value: dict, context: RecordingContext | None) -> ValidationResult:
    _require(type(value) is dict, 'ROOT_OBJECT_REQUIRED')
    _require(type(value.get('recording')) is dict, 'FIELDS', 'recording')
    status = value['recording'].get('record_status')
    _require(type(status) is str and status in (DRAFT, RECORDED), 'RECORD_STATUS')
    recorded = status == RECORDED
    _shape(value, _contract(recorded))
    _native_json(value)
    decision = value['decision']
    ident = decision['decision_id']
    _require(re.fullmatch(r'P20-FORMAL-ACCEPTANCE-[A-Z0-9][A-Z0-9-]{0,95}', ident) is not None,
             'DECISION_ID_FORMAT')
    _date(decision['owner_stated_decision_date'])
    if not recorded:
        _require(context is None, 'DRAFT_RECORDING_CONTEXT')
        return ValidationResult(True, DRAFT, self_seal='NOT_APPLICABLE')

    _require(type(context) is RecordingContext, 'RECORDING_CONTEXT_REQUIRED')
    auth, unique = context.authorization, context.uniqueness
    _require(type(auth) is WriteAuthorization, 'WRITE_AUTHORIZATION_REQUIRED')
    _require(type(auth.reference) is str and bool(auth.reference.strip()), 'AUTHORIZATION_REFERENCE')
    _digest(auth.sha256, 'authorization.sha256')
    _require(type(auth.destination) is str and auth.destination == DESTINATION and
             type(auth.operation) is str and auth.operation == WRITE_OPERATION and
             type(auth.schema_version) is str and auth.schema_version == SCHEMA_VERSION,
             'AUTHORIZATION_SCOPE')
    _date(auth.owner_stated_decision_date)
    _require(decision['owner_stated_decision_date'] == auth.owner_stated_decision_date,
             'OWNER_DATE_CONTEXT_MISMATCH')
    _require(decision['write_authorization'] == {'reference': auth.reference, 'sha256': auth.sha256},
             'AUTHORIZATION_MISMATCH')
    _require(type(unique) is UniquenessCheck, 'UNIQUENESS_CONTEXT_REQUIRED')
    _require(type(unique.complete) is bool and unique.complete and
             type(unique.scope) is str and unique.scope == UNIQUENESS_SCOPE and
             type(unique.reference) is str and bool(unique.reference.strip()) and
             type(unique.existing_ids) is frozenset and
             all(type(i) is str and bool(i.strip()) for i in unique.existing_ids),
             'UNIQUENESS_UNRESOLVED')
    _require(ident not in unique.existing_ids, 'DECISION_ID_REUSED')

    recording, retention = value['recording'], value['retention']
    event = _instant(context.event_utc)
    stamp = _instant(recording['created_at_utc'])
    _require(stamp == event, 'RECORD_EVENT_MISMATCH')
    _require(event <= _instant(context.observed_at_utc), 'RECORD_IN_FUTURE')
    try:
        local_date = event.astimezone(ZoneInfo(TIMEZONE)).date()
    except ZoneInfoNotFoundError as exc:
        raise AcceptanceValidationError('P20_ACCEPTANCE_TIMEZONE_UNAVAILABLE') from exc
    p = _date(recording['acceptance_date'])
    _require(p == local_date, 'P_RECORDING_LOCAL_DATE')
    _require(_date(retention['start']) == p, 'RETENTION_START')
    end = add_calendar_months(p, 12)
    _require(_date(retention['minimum_end']) == end, 'CALENDAR_END')
    _require(_date(retention['review_date']) == end, 'REVIEW_DATE')
    _digest(value['record_sha256'], 'record_sha256')
    _require(value['record_sha256'] == internal_record_sha256(value), 'SELF_SEAL_INVALID')
    return ValidationResult(True, RECORDED, self_seal='PASS', uniqueness='WITHIN_CALLER_SUPPLIED_SCOPE')


def validate_record(value: dict, *, context: RecordingContext | None = None) -> ValidationResult:
    """Fail closed; a successful result describes supplied data, not a real event.

    Context must come from the trusted caller, not be reconstructed from value.
    No clock, files, package receipts, or external authorization are consulted.
    """
    try:
        return _validate(value, context)
    except AcceptanceValidationError as exc:
        return ValidationResult(False, errors=(str(exc),))
    except (UnicodeError, RecursionError, OverflowError) as exc:
        return ValidationResult(False, errors=('P20_ACCEPTANCE_INPUT_UNREPRESENTABLE:' + type(exc).__name__,))


def _object_pairs(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        _require(key not in value, 'DUPLICATE_JSON_KEY', key)
        value[key] = item
    return value


def _nonfinite(value: str):
    raise AcceptanceValidationError('P20_ACCEPTANCE_NONFINITE_JSON:' + value)


def parse_record(payload: bytes) -> dict:
    """Strict UTF-8 JSON boundary; duplicate keys are rejected before dict creation."""
    _require(type(payload) is bytes, 'BYTES_REQUIRED')
    try:
        value = json.loads(payload.decode('utf-8'), object_pairs_hook=_object_pairs,
                           parse_constant=_nonfinite)
    except AcceptanceValidationError:
        raise
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise AcceptanceValidationError('P20_ACCEPTANCE_MALFORMED_JSON') from exc
    _require(type(value) is dict, 'ROOT_OBJECT_REQUIRED')
    return value


def validate_record_bytes(payload: bytes, *, context: RecordingContext | None = None,
                          expected_whole_file_sha256: str | None = None,
                          require_external_digest: bool = False) -> ValidationResult:
    """Validate bytes and optionally an independently supplied exact-byte digest.

    Omitting an optional digest returns NOT_CHECKED, never PASS. Neither this
    digest nor the internal seal upgrades independent anchoring or assurance.
    """
    try:
        _require(type(require_external_digest) is bool, 'EXTERNAL_DIGEST_REQUIREMENT_TYPE')
        value = parse_record(payload)
        result = validate_record(value, context=context)
        if not result.valid:
            return result
        _require(expected_whole_file_sha256 is not None or not require_external_digest,
                 'EXTERNAL_DIGEST_REQUIRED')
        external = 'NOT_CHECKED'
        if expected_whole_file_sha256 is not None:
            _digest(expected_whole_file_sha256, 'external_whole_file_sha256')
            _require(hashlib.sha256(payload).hexdigest() == expected_whole_file_sha256,
                     'EXTERNAL_DIGEST_MISMATCH')
            external = 'PASS'
        return ValidationResult(True, result.record_status, self_seal=result.self_seal,
                                external_digest=external, uniqueness=result.uniqueness)
    except AcceptanceValidationError as exc:
        return ValidationResult(False, errors=(str(exc),))
