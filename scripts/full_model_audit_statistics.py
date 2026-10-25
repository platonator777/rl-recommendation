"""Recalculate user bootstrap from the sufficient statistics in the bundle."""
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def read(rel):return json.loads((ROOT/'results/full_model'/rel).read_text(encoding='utf-8'))
def bootstrap(values,seed=20261016):
    a=np.asarray(values,dtype=float); rng=np.random.default_rng(seed)
    estimates=np.asarray([a[rng.integers(0,len(a),len(a))].mean() for _ in range(2000)])
    lo,hi=np.quantile(estimates,[.025,.975])
    return dict(mean_difference=float(a.mean()),ci_95_low=float(lo),ci_95_high=float(hi),n_users=len(a),repetitions=2000,seed=seed)
def main():
    fix='full_model/reward_scale_fix/'
    e=read(fix+'e1/support_decomposition.json'); users=e['model_rows']
    base=users['SFT strong17']; out={}
    for label in ['full_model normalized sampled ordinary beta1','full_model normalized sampled penalized beta1 update1000','full_model normalized frozen selected candidate']:
        candidate=users[label]; common=sorted(set(base)&set(candidate))
        out[label]={metric:bootstrap([candidate[u][metric]-base[u][metric] for u in common]) for metric in ['ndcg@10','recall@10','expected_raw_reward','unsupported_mass']}
    h5=read('h5_user_aggregates.json')['rows']; groups=defaultdict(list)
    for row in h5:groups[(row['model'],row['source_mode'],row['step'])].append(row['mean'])
    official=read(fix+'e4/h5_context_summary_ci.json')['results']
    hout=[]
    for row in official:
        r=bootstrap(groups[(row['model'],row['source_mode'],row['step'])])
        assert abs(r['mean_difference']-row['mean_append_minus_fixed_rm_ratio'])<1e-10
        hout.append(dict(model=row['model'],source_mode=row['source_mode'],step=row['step'],**r))
    confirmation=read(fix+'confirmation_results.json'); models=confirmation['models']; b=base
    def model(method,support,seed):return next(x['metrics']['per_user_metrics'] for x in models if x['method']==method and x['support_mode']==support and x['seed']==seed)
    pairs=[('sampled_Q1_minus_SFT',('sampled','ordinary'),None),('exact_ordinary_minus_sampled_ordinary',('exact','ordinary'),('sampled','ordinary')),('sampled_penalized_minus_sampled_ordinary',('sampled','penalized'),('sampled','ordinary'))]
    cout={}
    for key,left,right in pairs:
        differences=defaultdict(list)
        for seed in [42,73]:
            l=model(*left,seed); r=model(*right,seed) if right else b
            for u in sorted(set(l)&set(r)):differences[u].append(l[u]['ndcg@10']-r[u]['ndcg@10'])
        r=bootstrap([sum(differences[u])/len(differences[u]) for u in sorted(differences)])
        assert abs(r['mean_difference']-confirmation['primary_holm_family'][key]['mean_difference'])<1e-12
        cout[key]=r
    result=dict(status='passed',note='New deterministic bootstrap seed; means match source. CI Monte Carlo endpoints may differ from historical seeds. Report retains source intervals.',e1=out,h5=hout,confirmation=cout)
    (ROOT/'checks/statistics_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('User means verified; E1, H5 and confirmation bootstraps recalculated (2000 repetitions).')
if __name__=='__main__':main()
