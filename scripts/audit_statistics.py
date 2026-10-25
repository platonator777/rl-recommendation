"""Reaggregate local predictions; preserve user weighting and original bootstrap RNG."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def read(path):
    with (ROOT/path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def user_means(rows,method,column):
    groups=defaultdict(list)
    for r in rows:
        if r['method']==method and r.get(column,'') not in {'','nan','None'}:groups[int(r['uid'])].append(float(r[column]))
    return {u:float(np.mean(v)) for u,v in groups.items()}
def bootstrap(a,b,seed,draws=2000):
    users=sorted(set(a)&set(b));delta=np.array([a[u]-b[u] for u in users],dtype=np.float64)
    if not users:raise ValueError('No paired users for the requested column')
    rng=np.random.default_rng(seed)
    means=np.empty(draws)
    for i in range(draws):means[i]=delta[rng.integers(0,len(delta),size=len(delta))].mean()
    return {'mean_difference':float(delta.mean()),'n_users':len(users),'bootstrap_draws':draws,'seed':seed,
        'ci95_low':float(np.quantile(means,.025)),'ci95_high':float(np.quantile(means,.975)),
        'ci97.5_low':float(np.quantile(means,.0125)),'ci97.5_high':float(np.quantile(means,.9875))}
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'checks/statistics.json');a=p.parse_args()
    results=[];errors=[]
    for prefix in ['results/policy/reward_scaling/','results/temporal/']:
        users=read(prefix+'per_user_metrics.csv')
        for r in read(prefix+'registered_contrasts.csv'):
            left,right=r['contrast'].split(' - ')
            col='24h_positive_ndcg10' if r['metric'].startswith('24h') else 'next_ndcg10'
            if 'temporal' in prefix and col=='24h_positive_ndcg10':col='24h_ndcg10'
            out=bootstrap(user_means(users,left,col),user_means(users,right,col),int(r['seed']),int(r['bootstrap_draws']))
            for key in ['mean_difference','ci95_low','ci95_high','ci97.5_low','ci97.5_high']:
                if r.get(key) and abs(out[key]-float(r[key]))>1e-12:errors.append(f'{prefix}{r["contrast"]}:{key}')
            if out['n_users']!=int(r['n_users']):errors.append('denominator mismatch')
            results.append({'source':prefix,'contrast':r['contrast'],'metric':r['metric'],**out})
    pred=read('results/reward/scaling/rm_dev_predictions.csv')
    rm={}
    for model in ['small_base','small_affinity','large_base','large_affinity']:
        grouped=defaultdict(list)
        for r in pred:grouped[int(r['uid'])].append(float(r['pred_'+model])-float(r['y']))
        rm[model]={u:float(np.abs(v).mean()) for u,v in grouped.items()}
        off=next(r for r in read('results/reward/scaling/rm_dev_metrics.csv') if r['model']==model and r['slice']=='all')
        mae=float(np.mean(list(rm[model].values())))
        if abs(mae-float(off['user_macro_mae']))>1e-12:errors.append(model+':user_macro_mae')
        rmse=float(np.mean([np.sqrt(np.mean(np.square(v))) for v in grouped.values()]))
        if abs(rmse-float(off['user_macro_rmse']))>1e-12:errors.append(model+':user_macro_rmse')
    for r in read('results/reward/scaling/rm_dev_contrasts.csv'):
        left,right=r['contrast'].split(' - ')
        out=bootstrap(rm[left],rm[right],int(r['seed']),int(r['bootstrap_draws']))
        for k in ['mean_difference','ci95_low','ci95_high']:
            if abs(out[k]-float(r[k]))>1e-12:errors.append('RM '+r['contrast']+':'+k)
        results.append({'source':'RM predictions','contrast':r['contrast'],**out})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'passed':not errors,'errors':errors,'contrasts':results,'tolerance':1e-12,
       'aggregation':'Average seed within each user, then users; omit 24h NA; retain OOV misses.'},indent=2),encoding='utf-8')
    print(json.dumps({'passed':not errors,'contrasts':len(results),'errors':errors}))
    if errors:raise SystemExit(1)
if __name__=='__main__':main()
