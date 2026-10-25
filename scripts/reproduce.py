"""Dry-run by default. All experiment output belongs to an explicit workspace."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def commands(stage, work):
    py = sys.executable
    def module(m,*args): return [py,'-m',m,*map(str,args)]
    def legacy(s): return [py,str(ROOT/'scripts/full_model_reproduce.py'),'--stage',s,'--workspace',str(work),'--run']
    cfg=lambda name: str(work/'configs'/name)
    if stage in {'prepare','sft','neural-rm','full-model-rl','long-term','horizon','horizon-evaluation'}:
        names={'neural-rm':['rm'],'full-model-rl':['rl-cache','rl-pilot','rl-core','rl-freeze'],
               'long-term':['long-term'],'horizon':['horizon'],'horizon-evaluation':['horizon-evaluation']}
        return [legacy(s) for s in names.get(stage,[stage])]
    if stage=='matched-sft':return [module('tlab.residual_support.sft','--config',cfg('residual_support/sft.yaml'),'--workspace',work,'--prepare')]+[module('tlab.residual_support.sft','--config',cfg('residual_support/sft.yaml'),'--workspace',work,'--method',m) for m in ['dense','last_target']]
    if stage=='tabular-rm':return [module('tlab.residual_support.rm','--config',cfg('residual_support/rm.yaml'),'--workspace',work)]
    if stage=='affinity-rm':return [module('tlab.policy_diagnostics.c_rm')]
    if stage=='large-rm':return [module('tlab.rm_scaling.experiment','--stage','prepare')]
    if stage=='compact-head':return [module('tlab.residual_support.cache','--config',cfg('residual_support/policy.yaml'),'--workspace',work,'--split',s) for s in ['train','development','final']]+[module('tlab.residual_support.train_policy','--config',cfg('residual_support/policy.yaml'),'--workspace',work)]
    if stage=='sample-count':return [module('tlab.policy_diagnostics.run_policy_branch')]
    if stage=='common-cache':return [module('tlab.policy_diagnostics.temporal'),module('tlab.policy_diagnostics.cache_adapter')]
    if stage=='full-model-backtest':return [module('tlab.experiments.full_model_temporal','--config',cfg('full_model_tminus14_reward_scale_fix.yaml'),'--stage',s) for s in ['prepare','train-sft','train-rm','rm-gate','cache','train-rl','validation','final-test']]
    if stage=='tabular-policy':return [module('tlab.frozen_policy.experiment','--stage','all')]
    if stage=='rm-scaling-policy':return [module('tlab.rm_scaling.experiment','--stage',s) for s in ['rewards','policy']]
    if stage=='evaluate':return [[py,str(ROOT/'scripts/check_models.py'),'--workspace',str(work),'--require-external'],module('tlab.rm_scaling.experiment','--stage','report')]
    if stage=='temporal-evaluate':return [[py,str(ROOT/'scripts/check_models.py'),'--workspace',str(work),'--require-external'],[py,'-c','from tlab.rm_scaling.temporal import run_temporal_once; run_temporal_once()']]
    raise ValueError(stage)

STAGES=['prepare','sft','neural-rm','full-model-rl','full-model-backtest','matched-sft','tabular-rm','affinity-rm','compact-head','common-cache','sample-count','tabular-policy','large-rm','rm-scaling-policy','evaluate','temporal-evaluate','long-term','horizon','horizon-evaluation']
DEPENDENCIES={
 'prepare':[], 'sft':['prepare'], 'matched-sft':['full-model-backtest'], 'neural-rm':['prepare'],
 'full-model-rl':['sft','neural-rm'], 'full-model-backtest':['full-model-rl'], 'tabular-rm':['full-model-backtest'], 'affinity-rm':['tabular-rm'],
 'compact-head':['tabular-rm'], 'common-cache':['compact-head'], 'sample-count':['compact-head'], 'tabular-policy':['common-cache','affinity-rm'],
 'large-rm':['affinity-rm'], 'rm-scaling-policy':['large-rm','tabular-policy'],
 'evaluate':['rm-scaling-policy'], 'temporal-evaluate':['evaluate'], 'long-term':['full-model-rl'],
 'horizon':['full-model-rl'], 'horizon-evaluation':['horizon']}

def initialize(work, evaluation=False):
    work.mkdir(parents=True,exist_ok=True)
    # Some historical loaders resolve configurations from workspace/configs.
    shutil.copytree(ROOT/'configs',work/'configs',dirs_exist_ok=True)
    shutil.copytree(ROOT/'src/tlab',work/'src/tlab',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    # The inherited full-model stages resolve their relative configs in the workspace.
    import yaml
    for name in ['full_model_reward_scale_fix.yaml','full_model_tminus14_reward_scale_fix.yaml']:
        path=work/'configs'/name
        config=yaml.safe_load(path.read_text())
        artifact=config['project']['artifacts_dir']
        config['project'].update(panels_source_root=artifact+'/panels',cache_source_root=artifact+'/cache',cache_source_read_only=False)
        config['project'].pop('temporal_model_source_root',None)
        config['project'].pop('temporal_e5_source_root',None)
        path.write_text(yaml.safe_dump(config,sort_keys=False))
    manifest=json.loads((ROOT/'assets_manifest.json').read_text())
    for asset in manifest['assets']:
        runtime=asset.get('runtime_path')
        if asset['status']=='included' and runtime and (evaluation or not asset.get('evaluation_metadata')):
            dst=work/runtime
            if not dst.exists():
                dst.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(ROOT/asset['destination'],dst)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--list-stages',action='store_true')
    p.add_argument('--stage',choices=STAGES+['all'])
    p.add_argument('--workspace',type=Path,default=ROOT/'work')
    p.add_argument('--run',action='store_true')
    a=p.parse_args()
    if a.list_stages:print(json.dumps(DEPENDENCIES,indent=2));return
    if not a.stage:p.error('--stage or --list-stages is required')
    work=a.workspace.resolve()
    if work==ROOT or ROOT.is_relative_to(work):p.error('workspace must be a separate directory')
    stages=STAGES if a.stage=='all' else [a.stage]
    plan=[{'stage':s,'dependencies':DEPENDENCIES[s],'commands':commands(s,work)} for s in stages]
    print(json.dumps({'dry_run':not a.run,'workspace':str(work),'stages':plan},indent=2))
    if not a.run:return
    # A new fit must not silently consume the report's historical completed state.
    receipt=work/'execution_receipts.json'
    done=json.loads(receipt.read_text()) if receipt.exists() else []
    training=set(STAGES)-{'evaluate','temporal-evaluate'}
    if any(s in done for s in stages if s in training):p.error('stage already executed here; use a fresh workspace for a new training run')
    if any(s in training for s in stages) and (work/'artifacts/rm_scaling/run_status.json').exists():
        p.error('historical completed state detected; use a fresh workspace')
    initialize(work,evaluation=all(s in {'evaluate','temporal-evaluate'} for s in stages))
    env=dict(os.environ,TLAB_WORKSPACE=str(work),PYTHONPATH=str(ROOT/'src'),PYTHONIOENCODING='utf-8')
    for row in plan:
        for cmd in row['commands']:subprocess.run(cmd,cwd=work,env=env,check=True)
        if row['stage']=='full-model-backtest':
            for role in ['strong','weak']:
                source=work/f'artifacts/full_model/reward_scale_fix/runs/tminus14/sft_{role}_seed17'
                target=work/'models/actor';target.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source/'best.pt',target/(role+'.pt'))
                if role=='strong':shutil.copy2(source/'run_manifest.json',target/'strong_run_manifest.json')
        done.append(row['stage']);receipt.write_text(json.dumps(done,indent=2))

if __name__=='__main__':main()
