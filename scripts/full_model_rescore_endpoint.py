"""Add the matched step-1000 penalty comparison to a new working E1 result."""
import argparse
import json
from pathlib import Path
import torch
from tlab.experiments.full_model_matrix import _root_and_processed,_load_cache,create_panels
from tlab.experiments.full_model_analysis import _policy_metrics,_feature_support
from tlab.evaluation.metrics import paired_user_bootstrap

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);a=p.parse_args()
    c,processed,artifacts=_root_and_processed(a.config)
    panels=create_panels(a.config);manifest,cache,_,arrays=_load_cache(a.config,'fit','ratio')
    rows=panels['development']+panels['confirmation'];support=_feature_support(processed)
    results={}
    with torch.no_grad():
        for mode in ['ordinary','penalized']:
            cp=artifacts/'runs'/f'q1_sampled_{mode}_b1_s17/snapshots/update_01000.pt'
            results[mode]=_policy_metrics(a.config,cp,rows,arrays['evaluation'][:len(rows)],support)
    left=results['penalized']['per_user_metrics'];right=results['ordinary']['per_user_metrics']
    differences={m:paired_user_bootstrap({uid:left[uid][m]-right[uid][m] for uid in sorted(set(left)&set(right))},repetitions=2000,seed=27101+i) for i,m in enumerate(['ndcg@10','recall@10','expected_raw_reward','unsupported_mass'])}
    path=artifacts/'e1/support_decomposition.json';data=json.loads(path.read_text(encoding='utf-8'))
    label='full_model normalized sampled penalized beta1 update1000';result=results['penalized']
    data['models'][label]={k:v for k,v in result.items() if k!='per_user_metrics'}
    data['model_rows'][label]=result['per_user_metrics']
    data['matched_penalty_endpoint']={'paired_penalized_minus_ordinary':differences,'note':'New rescore; no selection or checkpoint change'}
    path.write_text(json.dumps(data,indent=2),encoding='utf-8')
    out=path.parent/'penalty_matched_endpoint_supplement.json';out.write_text(json.dumps(data['matched_penalty_endpoint'],indent=2),encoding='utf-8');print(out)
if __name__=='__main__':main()
