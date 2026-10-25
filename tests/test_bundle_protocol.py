import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import numpy as np
from tlab.frozen_policy.data import _v5_hash_bucket
from tlab.frozen_policy.reporting import _build_24h_targets,_multilabel_metrics,_evaluate_full_scores
from tlab.policy_diagnostics.c_rm import _hash_bucket

ROOT=Path(__file__).resolve().parents[1]

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
