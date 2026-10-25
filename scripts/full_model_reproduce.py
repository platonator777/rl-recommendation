"""Explicit staged research commands. Default prints a plan; --run executes it."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
STAGES=['prepare','sft','rm','baseline-evaluation','rm-evaluation','rl-cache','rl-pilot','rl-core','rl-freeze','analysis','long-term','horizon','horizon-evaluation','temporal']

def initialize(work):
    (work/'configs').mkdir(parents=True,exist_ok=True)
    shutil.copytree(ROOT/'src/tlab',work/'src/tlab',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    import yaml
    for p in (ROOT/'configs').glob('*.yaml'):
        d=yaml.safe_load(p.read_text(encoding='utf-8'))
        if d.get('dataset',{}).get('revision')=='main':
            d['dataset']['revision']='dd6f3a19eef5866e346c3270e098baa641a44948'
        if p.name in ('full_model_reward_scale_fix.yaml','full_model_tminus14_reward_scale_fix.yaml'):
            artifact=d['project']['artifacts_dir']
            d['project']['panels_source_root']=artifact+'/panels'
            d['project']['cache_source_root']=artifact+'/cache'
            d['project']['cache_source_read_only']=False
        if p.name=='full_model_tminus14_reward_scale_fix.yaml':
            # Fresh reproduction trains its own local models in the output tree.
            d['project'].pop('temporal_model_source_root',None)
            d['project'].pop('temporal_e5_source_root',None)
        (work/'configs'/p.name).write_text(yaml.safe_dump(d,sort_keys=False),encoding='utf-8')

def commands(stage,work):
    # All paths below are local to an explicit reproduction workspace.
    cfg='configs/full_model_reward_scale_fix.yaml'; actor='artifacts/reference/runs/sft_strong_seed17/best.pt'
    rm='artifacts/reference/runs/rm_reference_fit_seed17/best.pt'; weak='artifacts/reference/runs/sft_weak_seed17_mb16/best.pt'
    full_model='artifacts/full_model/reward_scale_fix'
    base=['--config',cfg]
    def cmd(m,*a): return [sys.executable,'-m',m,*map(str,a)]
    if stage=='prepare': return [cmd('tlab.data.download',*base),cmd('tlab.data.prepare',*base),cmd('tlab.data.like_index',*base)]
    if stage=='sft': return [
        cmd('tlab.train.sft_reference','--config','configs/reference_sft.yaml','--seed',17,'--max-targets',100000,'--epochs',5,'--learning-rate',.0003,'--validation-anchors-per-user',10,'--run-dir',str(Path(weak).parent),'--phase','weak_reference','--minimum-additional-epochs',5),
        cmd('tlab.train.sft_reference','--config','configs/reference_sft_mb64.yaml','--seed',17,'--max-targets',2000000,'--additional-epochs',10,'--learning-rate',.0001,'--validation-anchors-per-user',10,'--run-dir',str(Path(actor).parent),'--phase','strong_reference','--restart-schedule','--resume',weak,'--minimum-additional-epochs',5)]
    if stage=='rm': return [cmd('tlab.train.reward_model','--config','configs/reference_rm.yaml','--variant',variant,'--seed',17,'--max-targets',200000,'--validation-limit',40000,'--epochs',8,'--batch-size',64,'--run-dir',f'artifacts/reference/runs/rm_reference_{variant}_seed17') for variant in ['fit','audit']]
    if stage=='baseline-evaluation': return [cmd('tlab.evaluation.reference_baselines','--config','configs/reference_sft_mb64.yaml','--seed',17,'--anchors-per-user',10,'--alpha-grid','0,1,10,100,1000','--weak-checkpoint',weak,'--strong-checkpoint',actor,'--output','artifacts/reference/validation_baselines_seed17.json'),cmd('tlab.evaluation.sft_history_diagnostics','--config','configs/reference_sft_mb64.yaml','--checkpoint',actor,'--seed',17,'--per-user-limit',10,'--output','artifacts/reference/sft_history_strong_seed17.json')]
    if stage=='rm-evaluation': return [cmd(m,'--config','configs/reference_rm.yaml','--checkpoint',f'artifacts/reference/runs/rm_reference_{v}_seed17/best.pt','--variant',v,'--seed',17,'--train-limit',200000,'--validation-limit',40000,'--output',f'artifacts/reference/{m.rsplit(".",1)[1]}_{v}_reference_seed17.json') for v in ['fit','audit'] for m in ['tlab.evaluation.rm_diagnostics','tlab.evaluation.seconds_diagnostics']]
    if stage=='rl-cache': return [cmd('tlab.experiments.full_model_matrix',*base,'--mode','panels')]+[cmd('tlab.experiments.full_model_matrix',*base,'--mode','cache','--reward-checkpoint',rm,'--reward-head',head) for head in ['ratio','seconds']]+[[sys.executable,str(ROOT/'scripts/full_model_prepare_protocol.py'),*base]]
    if stage=='rl-pilot': return [cmd('tlab.experiments.full_model_matrix',*base,'--mode','lambda-pilot','--actor-checkpoint',actor),cmd('tlab.experiments.full_model_matrix',*base,'--mode','lambda-freeze')]
    if stage=='rl-core':
        out=[]
        for method in ['sampled','exact']:
            for support in ['ordinary','penalized']:
                for beta in [.1,1.,3.]:
                    name=f'q1_{method}_{support}_b{beta:g}_s17'.replace('.','p')
                    out.append(cmd('tlab.experiments.full_model_matrix',*base,'--mode','train','--actor-checkpoint',actor,'--method',method,'--support-mode',support,'--beta',beta,'--seed',17,'--updates',1000,'--run-dir',f'{full_model}/runs/{name}'))
        return out
    if stage=='rl-freeze':
        # The runner also covers a selected beta other than 1 and keeps
        # confirmation training from evaluating development histories.
        return [cmd('tlab.experiments.full_model_runner',*base,'--stage','freeze')]
    if stage=='analysis': return [cmd('tlab.experiments.full_model_runner',*base,'--stage',s) for s in ['q2','stress','bridge']]+[cmd('tlab.experiments.full_model_analysis',*base,'--mode',m) for m in ['e1','rm-support','frontier','full-cache-curves']]+[[sys.executable,str(ROOT/'scripts/full_model_rescore_endpoint.py'),*base]]
    if stage=='long-term': return [cmd('tlab.evaluation.continuation',*base,'--seed',17,'--train-samples',100000,'--validation-samples',50000)]
    if stage=='horizon': return [[sys.executable,str(ROOT/'scripts/full_model_train_horizon.py'),'--config','configs/reference_rl.yaml','--algorithm','grpo','--horizon','5','--reward-head','ratio','--advantage-mode','centered_corrected','--actor-checkpoint',actor,'--reward-checkpoint',rm,'--seed','17','--beta','1','--updates','1000','--batch-contexts','16','--group-size','8','--update-epochs','1','--run-dir','artifacts/reference/runs/grpo_h5_reference_strong_seed17_beta1_full']]
    if stage=='horizon-evaluation': return [cmd('tlab.experiments.full_model_horizon',*base,'--anchors',1024,'--horizon',5,'--seeds',17,42,73)]
    if stage=='temporal': return [cmd('tlab.experiments.full_model_temporal','--config','configs/full_model_tminus14_reward_scale_fix.yaml','--stage',s) for s in ['prepare','train-sft','train-rm','rm-gate','cache','train-rl','validation','final-test']]
    raise ValueError(stage)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,default=ROOT/'work')
    p.add_argument('--stage',choices=['init',*STAGES],required=True)
    p.add_argument('--run',action='store_true',help='Execute, including long training; otherwise print only')
    args=p.parse_args(); work=args.workspace.resolve()
    if args.stage=='init':
        initialize(work); print(work); return
    calls=commands(args.stage,work)
    print(json.dumps({'cwd':str(work),'commands':calls,'executed':args.run},indent=2))
    if args.run:
        if not (work/'configs/full_model_reward_scale_fix.yaml').is_file(): raise SystemExit('Run --stage init first')
        env=dict(os.environ,PYTHONPATH=str(work/'src'),PYTHONIOENCODING='utf-8')
        for call in calls: subprocess.run(call,cwd=work,env=env,check=True)

if __name__=='__main__': main()
