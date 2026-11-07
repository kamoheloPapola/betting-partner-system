"""League ownership checks supplement the unchanged Phase 4/5 v1 schemas.

A matching feature list or a successful hash check never transfers ownership.
The immutable league specification supplies the expected artifact join keys.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Any

from src.predictions.provenance import sha256_json, verify_record
from src.ml.training_lineage import verify_lineage_record


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class LeagueModelBundle:
    league: str
    bundle_id: str
    payload: bytes

    @classmethod
    def from_dict(cls, value, *, expected_league):
        from src.predictions.provenance import canonical_bytes
        require(value.get('league') == expected_league, 'bundle league mismatch')
        members = value['members']
        from src.ml.bundle_activation import ROLES
        require(set(members) == set(ROLES), 'incomplete goal bundle')
        names = value['feature_contract']['ordered_feature_names']
        require(names and len(names) == len(set(names)), 'invalid ordered feature schema')
        require(value['feature_contract']['schema_sha256'] == sha256_json(names), 'feature schema hash mismatch')
        for member in members.values():
            require(member['league'] == expected_league, 'member league mismatch')
            require(member['schema_sha256'] == sha256_json(names), 'member feature schema mismatch')
        material = {k:v for k,v in value.items() if k != 'bundle_sha256'}
        require(value['bundle_sha256'] == sha256_json(material), 'bundle fingerprint mismatch')
        return cls(expected_league, value['bundle_id'], canonical_bytes(value))

    def to_dict(self):
        import json
        return json.loads(self.payload)


def verify_prediction(record, *, bundle, root, verify_referenced_artifacts=True):
    value = record.to_dict() if hasattr(record, 'to_dict') else record
    spec = bundle.to_dict()
    result = verify_record(value, root=root, verify_referenced_artifacts=verify_referenced_artifacts)
    errors = list(result['errors'])
    if value['fixture']['league'] != bundle.league:
        errors.append('prediction league mismatch')
    if value['method_provenance']['descriptor']['simulation']['league'] != bundle.league:
        errors.append('simulator league mismatch')
    expected = spec['members']
    for member in value['model_provenance']['members']:
        if member['artifact_sha256'] != expected[member['role']]['sha256']:
            errors.append('prediction artifact/league ownership mismatch')
    for call in value['feature_provenance']['model_inputs']:
        if call['feature_schema_sha256'] != spec['feature_contract']['schema_sha256']:
            errors.append('prediction feature schema mismatch')
    return {**result, 'verified': not errors, 'errors': errors}


def verify_training(record, *, bundle, role, root):
    value = record.to_dict() if hasattr(record, 'to_dict') else record
    result = verify_lineage_record(value, root=root)
    errors = list(result['errors'])
    if value['model_identity'].get('league') != bundle.league:
        errors.append('training lineage league mismatch')
    if value['model_identity'].get('role') != role:
        errors.append('training lineage role mismatch')
    if value['artifact']['artifact_sha256'] != bundle.to_dict()['members'][role]['sha256']:
        errors.append('training lineage artifact join mismatch')
    return {**result, 'assessment_verified': not errors,
            'lineage_verified': not errors and result['lineage_verified'], 'errors': errors}


def verify_field_authority(authority, *, expected_league, source, field):
    require(authority.get('league') == expected_league, 'field authority league mismatch')
    require(authority.get('source') == source and field in authority.get('fields', []), 'field authority scope mismatch')
    require(authority.get('overlap_rows', 0) > 0 and authority.get('mismatches') == 0,
            'field authority requires exact historical overlap')
    return True


def derive_state(*, execution, active_lineage, candidate, shadow, data_ready,
                 promotion=False, activated=False):
    require(not activated or promotion, 'activation requires promotion authority')
    require(not promotion or (execution and candidate and shadow and data_ready), 'promotion prerequisites missing')
    if activated:
        return 'ACTIVE_TRUSTED_BUNDLE'
    if promotion:
        return 'PROMOTION_REVIEW_ELIGIBLE'
    if execution and candidate and shadow and data_ready:
        return 'PROSPECTIVE_COLLECTION_READY'
    if execution and candidate and shadow:
        return 'SHADOW_EVALUATION_READY'
    if candidate:
        return 'CANDIDATE_READY'
    if active_lineage:
        return 'TRAINING_LINEAGE_VERIFIED'
    if execution:
        return 'TRAINING_LINEAGE_UNPROVEN'
    return 'UNASSESSED'
