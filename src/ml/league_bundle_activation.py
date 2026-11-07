"""League-scoped specifications for the existing Phase 8 transition engine.

Phase 9 registers no production pointers. A specification is reviewable evidence;
the Phase 8 journal, quiescence, generation checks and rollback still own writes.
"""
from __future__ import annotations

import copy
from pathlib import Path

from src.ml.bundle_activation import (BundleController, ROOT, require, read_json, sha256_file,
                                     sha256_json, verify_smoke, verify_seal)
from src.ml.league_integrity import LeagueModelBundle, verify_training


class LeagueBundleController(BundleController):
    pointer_schema = 'ffps.active_league_bundle.v1'

    def __init__(self, specification_path: Path, *, league: str, root: Path = ROOT):
        super().__init__(root)
        from src.config.leagues import ACTIVE_LEAGUES
        require(league in ACTIVE_LEAGUES and league != 'PL', 'use the existing PL controller for PL')
        self.league = league
        self.specification_path = specification_path.resolve()
        require(self.specification_path.is_relative_to(self.root/'baselines/phase9'), 'deployment specification escapes Phase 9 evidence')
        self.frozen = read_json(self.specification_path)
        require(self.frozen['league'] == league, 'deployment specification league mismatch')
        material = {k:v for k,v in self.frozen.items() if k != 'specification_sha256'}
        require(self.frozen['specification_sha256'] == sha256_json(material), 'deployment specification fingerprint mismatch')
        self.specification_file_sha256 = sha256_file(self.specification_path)
        self.legacy_id = self.frozen['active']['bundle_id']
        self.candidate_id = self.frozen['candidate']['bundle_id']
        self.pointer = self.models/f'active_{league.lower()}_bundle.json'
        self.history = self.models/f'{league.lower()}_bundle_history'
        self.lock = self.models/f'{league.lower()}_bundle.lock'
        self.pending = self.models/f'{league.lower()}_bundle_pending.json'

    def pointer_scope(self):
        return {'league':self.league,'deployment_specification':str(self.specification_path.relative_to(self.root).as_posix()),
                'deployment_specification_sha256':self.specification_file_sha256}

    def specification(self):
        require(sha256_file(self.specification_path) == self.specification_file_sha256, 'deployment specification changed')
        return copy.deepcopy(self.frozen)

    def bundle(self, bundle_id):
        spec = self.specification()
        require(bundle_id in (self.legacy_id,self.candidate_id), 'unknown league bundle')
        reference = spec['active'] if bundle_id == self.legacy_id else spec['candidate']
        path = self.root/reference['path']
        require(sha256_file(path) == reference['sha256'], 'deployment bundle reference mismatch')
        value = read_json(path)
        bundle = LeagueModelBundle.from_dict(value, expected_league=self.league)
        require(value['bundle_id'] == bundle_id, 'deployment bundle ID mismatch')
        members, lineage = {}, {}
        manifest = read_json(self.manifest)
        for role,item in value['members'].items():
            require(sha256_file(self.root/item['path']) == item['sha256'], 'deployment artifact hash mismatch')
            meta = item.get('meta',item.get('manifest_entry'))
            require(meta.get('league') == self.league, 'deployment model metadata league mismatch')
            if bundle_id == self.legacy_id:
                require(manifest['active_models'][item['registry_slot']] == item['manifest_key'], 'legacy league pointer changed')
                require(manifest[item['manifest_key']] == meta, 'legacy league metadata changed')
                training = {'classification':'LEGACY_TRAINING_LINEAGE_UNPROVEN'}
            else:
                ref = item['lineage']
                require(sha256_file(self.root/ref['path']) == ref['sha256'], 'candidate lineage hash mismatch')
                core = read_json(self.root/ref['path'])
                result = verify_training(core,bundle=bundle,role=role,root=self.root)
                require(result['lineage_verified'], 'candidate lineage not trusted')
                training = {'classification':'proven','lineage_fingerprint':core['training_lineage_fingerprint'], 'lineage_path':ref['path']}
            lineage[role] = training
            members[role] = {'path':item['path'],'sha256':item['sha256'],'size_bytes':item['size_bytes'],'meta':meta,
                'provenance':{'source_location':item['path'],'artifact_sha256':item['sha256'],'size_bytes':item['size_bytes'],
                              'manifest_key':item.get('manifest_key',bundle_id+':'+role),
                              'registry_slot':item.get('registry_slot','ACTIVE_BUNDLE:'+self.league+':'+role),'training_lineage':training}}
        return {'league':self.league,'bundle_id':bundle_id,'aggregate_sha256':value['bundle_sha256'],'members':members,
                'lineage':lineage,'features':value['feature_contract']['ordered_feature_names'],
                'evaluation_specification_sha256':spec['evaluation_specification']['specification_sha256']}

    def smoke(self,bundle,state=None):
        import pandas as pd
        from src.cli.commands.prediction import _calc_goals
        from src.simulation.match_simulator import MatchSimulator, DEFAULT_N_SIMULATIONS
        from src.predictions.provenance import PredictionCapture, mark_output_reproduced, replay_pl_1x2, verify_record
        ref = self.frozen['smoke_reference']
        require(sha256_file(self.root/ref['path']) == ref['sha256'], 'smoke reference drift')
        frozen = read_json(self.root/ref['path'])
        require(frozen['fixture']['league'] == self.league, 'smoke reference league mismatch')
        inputs = frozen['feature_provenance']['model_inputs'][0]
        fixture = pd.Series(dict(zip(inputs['ordered_feature_names'],inputs['feature_values'])))
        for key in ('home_team','away_team','match_hash'):
            fixture[key] = frozen['fixture'][key]
        fixture['date'] = frozen['fixture']['kickoff_identity']
        fixture['league'] = self.league
        simulator = MatchSimulator(league=self.league,seed=42,n_simulations=DEFAULT_N_SIMULATIONS,seed_scope='phase8_deployment_smoke')
        suite = self.load(bundle)
        if state:
            from src.ml.bundle_activation import SLOTS
            for slot in SLOTS.values():
                suite[slot]._ffps_provenance['active_pointer_fingerprint'] = state['activation_fingerprint']
        with PredictionCapture(prediction_timestamp=frozen['fixture']['prediction_timestamp'],data_provenance=frozen['data_provenance']) as capture:
            probabilities = _calc_goals(fixture,suite,'Phase9DeploymentSmoke',simulator,league=self.league,rl_weights=None,use_simulator=True)
            record = capture.finalize(fixture=fixture,league=self.league,probabilities=probabilities,simulator=simulator)
        record = mark_output_reproduced(record,replay_pl_1x2(record,root=self.root)).to_dict()
        require(verify_record(record,root=self.root)['overall_phase4_trusted'], 'league smoke provenance failed')
        require({m['role']:m['artifact_sha256'] for m in record['model_provenance']['members']} == {r:m['sha256'] for r,m in bundle['members'].items()}, 'league smoke loaded artifacts differ')
        return record

    def league_promotion_gate(self,evidence):
        spec = self.specification()
        evaluation = read_json(Path(evidence)/'evaluation_specification.json')
        require(evaluation.get('league') == self.league, 'promotion evidence league mismatch')
        require(evaluation.get('candidate_bundle_id') == self.candidate_id, 'promotion candidate mismatch')
        require(sha256_file(Path(evidence)/'evaluation_specification.json') == spec['evaluation_specification']['sha256'], 'promotion evaluation specification mismatch')
        # No non-PL prospective protocol is admissible until that league has
        # qualified live data and a frozen scientific horizon. Historical results
        # and self-reported finals cannot create this missing scientific authority.
        if spec.get('prospective_contract') is not None:
            from src.evaluation.league_prospective import promotion_gate
            return promotion_gate(self,evidence)
        return {'status':'PROMOTION_NOT_ELIGIBLE','promotion_eligible':False,'classification':'CANDIDATE_INCONCLUSIVE',
                'league':self.league,'candidate_bundle_id':self.candidate_id,
                'specification_fingerprint':evaluation['specification_sha256'],'final_decision_sha256':None,
                'blocking_classification':'LEAGUE_PROSPECTIVE_PROTOCOL_NOT_QUALIFIED'}


def production_controller(league, *, root=ROOT):
    if league == 'PL':
        from src.ml import bundle_activation
        return bundle_activation.BundleController(root)
    from src.config.leagues import ACTIVE_LEAGUES
    if league not in ACTIVE_LEAGUES:
        return None
    models = root/'src/ml/models'
    pointer = models/f'active_{league.lower()}_bundle.json'
    pending = models/f'{league.lower()}_bundle_pending.json'
    history = models/f'{league.lower()}_bundle_history'
    require(not pending.exists(), 'unfinished league transition blocks serving')
    if not pointer.exists():
        require(not any(history.glob('*.json')), 'missing league pointer with transition history')
        return None
    state = read_json(pointer)
    verify_seal(state)
    require(state.get('league') == league, 'production pointer league mismatch')
    spec_path = root/state['deployment_specification']
    require(sha256_file(spec_path) == state['deployment_specification_sha256'], 'production specification hash mismatch')
    return LeagueBundleController(spec_path,league=league,root=root)
