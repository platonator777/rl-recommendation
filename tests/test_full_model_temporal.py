"""Fresh reproduction must train exact temporal models without legacy assets."""
import json
from types import SimpleNamespace

import pytest

from tlab.experiments import full_model_temporal as temporal


def test_temporal_prepare_uses_configured_main_protocol(tmp_path, monkeypatch):
    config = SimpleNamespace(path=tmp_path/'configs/fold.yaml',
                             project={'artifacts_dir':'artifacts/corrected'},
                             data={'processed_dir':'data/fold'},
                             dataset={'cutoff_shift_days':14})
    protocol = tmp_path/'artifacts/corrected/protocol.json'
    protocol.parent.mkdir(parents=True)
    protocol.write_text('{}')
    monkeypatch.setattr(temporal, 'load_config', lambda _: config)
    def stop_at_prepare(_):
        raise RuntimeError('configured protocol accepted')
    monkeypatch.setattr(temporal, 'prepare_dataset', stop_at_prepare)
    with pytest.raises(RuntimeError, match='configured protocol accepted'):
        temporal.prepare_temporal_fold(config.path)


@pytest.mark.parametrize('legacy_source', [False, True])
def test_temporal_exact_source_is_explicit(tmp_path, monkeypatch, legacy_source):
    project = {'artifacts_dir': 'artifacts'}
    if legacy_source:
        project['temporal_e5_source_root'] = 'legacy'
    config = SimpleNamespace(path=tmp_path/'configs/fold.yaml', project=project,
                             dataset={'cutoff_shift_days': 14})
    artifacts = tmp_path/'artifacts'
    artifacts.mkdir()
    (artifacts/'lambda_freeze.json').write_text(json.dumps({'selected_lambda': 1}))
    recipes = {f'{method}:{support}:beta=1': {'update': 100 if support == 'penalized' else 1000}
               for method, support in [('sampled','ordinary'),('exact','ordinary'),('sampled','penalized')]}
    (artifacts/'candidate_freeze.json').write_text(json.dumps({'recipe_freezes': recipes}))
    model_root = temporal._temporal_model_root(config)
    for name in ['sft_strong_seed17', 'rm_fit_seed17']:
        p = model_root/name/'best.pt'
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b'mock checkpoint')
    gate = temporal._temporal_e5_source_dir(config)/'rm_gates.json'
    gate.parent.mkdir(parents=True, exist_ok=True)
    gate.write_text(json.dumps({'fit_q1_predictive_gate_pass': True}))
    monkeypatch.setattr(temporal, 'load_config', lambda _: config)
    calls = []
    def fake_train(*args, **kwargs):
        calls.append(kwargs['method'])
        for update in [100,1000]:
            p = kwargs['run_dir']/'snapshots'/f'update_{update:05d}.pt'
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b'mock checkpoint')
        return {'latest_update':1000}
    monkeypatch.setattr(temporal, 'train_run', fake_train)
    monkeypatch.setattr(temporal, '_resume_model_run', lambda manifest, run, run_fn: run_fn())
    if legacy_source:
        with pytest.raises(FileNotFoundError, match='rl_transfer.json'):
            temporal.run_local_rl(config.path)
        assert 'exact' not in calls
    else:
        result = temporal.run_local_rl(config.path)
        assert calls.count('exact') == 3
        assert len(result['runs']) == 9
        assert all(not row['reused'] for row in result['runs'].values())
        assert result['runs']['sampled_penalized_seed17']['transfer_update_from_main_development'] == 100
