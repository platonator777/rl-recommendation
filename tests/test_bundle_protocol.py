import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import importlib.util

import numpy as np
from tlab.frozen_policy.data import _v5_hash_bucket
from tlab.frozen_policy.reporting import _build_24h_targets,_multilabel_metrics,_evaluate_full_scores
from tlab.policy_diagnostics.c_rm import _hash_bucket

ROOT=Path(__file__).resolve().parents[1]


def reproduction_runner():
    spec=importlib.util.spec_from_file_location('submission_reproduce',ROOT/'scripts/reproduce.py')
    runner=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner

def test_default_workspace_stays_inside_the_bundle(monkeypatch):
    from tlab.paths import workspace_root
    monkeypatch.delenv('TLAB_WORKSPACE',raising=False)
    assert workspace_root()==ROOT/'work'

def test_compatibility_hash_bytes_preserve_published_partitions():
    for uid in [1,17,500,700,1000,99999]:
        policy=hashlib.sha256(bytes.fromhex('76355f666173743a')+str(uid).encode()).hexdigest()
        rm=hashlib.sha256(bytes.fromhex('76345f323a433a726d6465763a')+str(uid).encode()).digest()
        assert _v5_hash_bucket(uid)==int(policy,16)%10
        assert _hash_bucket(uid)==int.from_bytes(rm[:8],'big')%5

def test_positive_targets_start_after_next_item_and_respect_cutoff(tmp_path):
    path=tmp_path/'data/processed_full_model_tminus14';path.mkdir(parents=True)
    (path/'protocol.json').write_text(json.dumps({'boundaries_source_units':{'test_start_raw':200000,'end_raw':300000}}))
    user=SimpleNamespace(split_codes=np.ones(6),recommendable_events=np.array([1,1,1,0,1,1],bool),
        ratio_raw=np.array([1,.5,.49,1,1,1]),raw_item_ids=np.array([10,11,12,13,14,15]))
    panel=[SimpleNamespace(uid=17,user_index=0,position=0)]
    result=_build_24h_targets(tmp_path,panel,np.array([0]),SimpleNamespace(users=[user]),
       {17:np.array([10,20,30,40,86410,86411])})
    assert result['targets'][17]=={11,14}
    assert result['nonempty_contexts']==1

def test_oov_is_miss_and_unreachable_24h_target_stays_in_denominator():
    panel=[SimpleNamespace(uid=1,raw_item_id=999,positive=True,is_new=True,is_repeat=False)]
    result=_evaluate_full_scores(panel,[0],[np.linspace(.8,.2,12)],np.arange(10,22))
    assert result['n']==1 and result['oov_targets']==1 and result['ndcg10']==0
    recall,_,hits=_multilabel_metrics([10,11],{10,999})
    assert recall==.5 and hits==1

def test_rm_fit_nesting_and_user_disjointness():
    def rows(name):
        with (ROOT/'protocol/reward'/name).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
    small,large,dev=[rows(name+'.csv') for name in ['small_fit_rows','large_fit_rows','rm_dev_rows']]
    keys=lambda rr:{(r['uid'],r['position']) for r in rr}
    assert len(small)==21290 and len(large)==100000
    assert keys(small)<=keys(large)
    assert {r['uid'] for r in large}.isdisjoint({r['uid'] for r in dev})

def test_default_runner_is_dry_and_completed_fit_is_not_reused(tmp_path):
    work=tmp_path/'fresh'
    result=subprocess.run([sys.executable,str(ROOT/'scripts/reproduce.py'),'--stage','all','--workspace',str(work)],capture_output=True,text=True)
    assert result.returncode==0 and json.loads(result.stdout)['dry_run']
    assert not work.exists()
    work.mkdir();(work/'execution_receipts.json').write_text('["compact-head"]')
    result=subprocess.run([sys.executable,str(ROOT/'scripts/reproduce.py'),'--stage','compact-head','--workspace',str(work),'--run'],capture_output=True,text=True)
    assert result.returncode==2 and 'fresh workspace' in result.stderr


def test_new_fit_does_not_import_saved_weights_or_model_selection(tmp_path):
    runner=reproduction_runner()
    runner.initialize(tmp_path)
    assert not (tmp_path/'models').exists()
    for name in ['artifacts/residual_support/policy','artifacts/residual_support/protocol/frozen_selection.json',
                 'artifacts/frozen_policy/reporting_selection.json','artifacts/frozen_policy/scaler.json',
                 'artifacts/frozen_policy/reward_scale.json','artifacts/rm_scaling/rm']:
        assert not (tmp_path/name).exists(),name
    assert (tmp_path/'artifacts/panel_tools/evaluation/frozen_primary_validation_panel.jsonl').is_file()
    assert (tmp_path/'artifacts/residual_support/panels/a_rl_train.jsonl').is_file()
    budget_path=tmp_path/'artifacts/policy_diagnostics/protocol/budget.json'
    budget=json.loads(budget_path.read_text())
    assert budget['measured_work']==[] and budget['wall_time_seconds']==0
    assert budget['remaining_new_stage_seconds']==budget['new_stage_cap_seconds']
    # Later stages keep the new run's measurements instead of resetting them.
    budget['measured_work']=[{'name':'new_fit','seconds':17}]
    budget_path.write_text(json.dumps(budget))
    runner.initialize(tmp_path)
    assert json.loads(budget_path.read_text())['measured_work']==budget['measured_work']


def test_saved_model_replay_still_stages_weights(tmp_path,monkeypatch):
    runner=reproduction_runner()
    copied=[]
    monkeypatch.setattr(runner.shutil,'copy2',lambda src,dst:copied.append(Path(src).relative_to(ROOT).as_posix()))
    runner.initialize(tmp_path,evaluation=True)
    assert 'models/actor/strong.pt' in copied
    assert any(name.startswith('models/compact_head/') for name in copied)
    assert 'protocol/policy/reporting_selection.json' in copied


def test_stage_all_is_ordered_and_single_stage_rejects_missing_dependencies(tmp_path):
    runner=reproduction_runner()
    done=set()
    for stage in runner.STAGES:
        assert set(runner.DEPENDENCIES[stage])<=done,stage
        done.add(stage)
    work=tmp_path/'not_initialized'
    result=subprocess.run([sys.executable,str(ROOT/'scripts/reproduce.py'),'--stage','large-rm',
                           '--workspace',str(work),'--run'],capture_output=True,text=True)
    assert result.returncode==2 and 'tabular-policy' in result.stderr
    assert not work.exists()


def test_separate_rm_scaling_stage_accepts_own_state_without_injecting_saved_outputs(tmp_path,monkeypatch):
    runner=reproduction_runner()
    state=tmp_path/'artifacts/rm_scaling/run_status.json'
    state.parent.mkdir(parents=True)
    state.write_text('{"stages": {"rm_fit_seconds": 1}}')
    receipt=tmp_path/'execution_receipts.json'
    receipt.write_text('["tabular-policy", "large-rm"]')
    modes=[]
    monkeypatch.setattr(runner,'initialize',lambda work,evaluation:modes.append(evaluation))
    monkeypatch.setattr(runner,'commands',lambda stage,work:[])
    monkeypatch.setattr(sys,'argv',['reproduce.py','--stage','rm-scaling-policy','--workspace',str(tmp_path),'--run'])
    runner.main()
    assert modes==[False]
    assert json.loads(receipt.read_text())[-1]=='rm-scaling-policy'
    # Re-evaluating a newly trained run must also keep its own artifacts.
    monkeypatch.setattr(sys,'argv',['reproduce.py','--stage','evaluate','--workspace',str(tmp_path),'--run'])
    runner.main()
    assert modes==[False,False]


def test_training_plan_includes_selection_and_comparison_outputs(tmp_path):
    runner=reproduction_runner()
    compact=runner.commands('compact-head',tmp_path)
    assert any('tlab.residual_support.continue_policy' in call for call in compact)
    assert any('--freeze-selection' in call for call in compact)
    assert compact[-1][2]=='tlab.residual_support.evaluation'
    assert runner.commands('matched-sft',tmp_path)[-1][-1]=='--summarize'
