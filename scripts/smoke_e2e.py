"""Short CPU-only synthetic preparation, actor, residual policy and evaluation check."""
import json, os, sys, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
import polars as pl
import torch
import yaml
from tlab.data.prepare import prepare_dataset
from tlab.models.actor import ItemGPT2Actor,ActorShape
from tlab.residual_support.head import ResidualPolicyHead,ResidualHeadSpec
from tlab.residual_support.objectives import exact_support_statistics,FrozenRewardScale

def main():
    torch.set_num_threads(2);torch.manual_seed(17)
    (ROOT/'work').mkdir(exist_ok=True)
    work=Path(tempfile.mkdtemp(prefix='synthetic_',dir=ROOT/'work'))
    raw=work/'data/raw';raw.mkdir(parents=True)
    rows=[dict(uid=u,item_id=10+(d+u)%12,timestamp=d*86400+u*5,is_organic=0,
       played_ratio_pct=30+(d*7+u)%71,track_length_seconds=120) for u in range(1,25) for d in range(1,61)]
    pl.DataFrame(rows).write_parquet(raw/'listens.parquet')
    cfg={'project':{'seed':17,'artifacts_dir':'artifacts'},
      'dataset':{'files':{'listens':'listens.parquet'},'timestamp_unit_seconds':1,'timestamp_bin_seconds':5,
      'catalog_size':12,'history_length':9,'history_events':8,'train_days':21,'validation_days':14,
      'test_days':7,'rm_validation_days':7,'rl_anchor_days':7},'data':{'source_dir':'data/raw','processed_dir':'data/processed','num_workers':0}}
    path=work/'configs/tiny.yaml';path.parent.mkdir();path.write_text(yaml.safe_dump(cfg))
    prepare_dataset(path)
    prepared=pl.read_parquet(work/'data/processed/listens.parquet')
    seq=prepared.filter(pl.col('token_id')>=3)['token_id'].to_list()[:36]
    inputs=torch.tensor(seq[:32]).reshape(4,8);target=torch.tensor(seq[1:33]).reshape(4,8)[:,-1]
    actor=ItemGPT2Actor(ActorShape(15,n_positions=16,n_layer=1,n_embd=16,n_head=2,dropout=0))
    optim=torch.optim.Adam(actor.parameters(),lr=.001)
    for _ in range(2):
        optim.zero_grad();lp=actor.next_action_log_probs(inputs,torch.ones_like(inputs));loss=-lp[torch.arange(4),target].mean();loss.backward();optim.step()
    with torch.no_grad():ref=actor.next_action_log_probs(inputs,torch.ones_like(inputs))[:,3:];ref=ref-torch.logsumexp(ref,-1,keepdim=True)
    head=ResidualPolicyHead(ResidualHeadSpec(4,8));x=torch.randn(4,12,4)
    zero=head(x);assert torch.count_nonzero(zero)==0
    assert torch.allclose(torch.softmax(ref+zero,-1),ref.exp(),atol=1e-7)
    reward=torch.sigmoid(x[...,0]);mass=torch.full((4,),.7);scale=FrozenRewardScale(.5,.2)
    opt=torch.optim.Adam(head.parameters(),lr=.01)
    first=None
    for _ in range(8):
        opt.zero_grad();q=torch.log_softmax(ref+head(x),-1)
        objective=exact_support_statistics(q,ref,reward,mass,scale,3.)['support_weighted_objective'].mean()
        if first is None:first=float(objective)
        (-objective).backward();opt.step()
    with torch.no_grad():
        q=torch.softmax(ref+head(x),-1)
        assert torch.allclose(q.sum(-1)*mass,.7*torch.ones(4),atol=1e-6)
        assert torch.count_nonzero(head(x))>0
        rankings=q.argsort(-1,descending=True)
        hit=(rankings[:,:10]==(target-3)[:,None]);ndcg=(hit/torch.log2(torch.arange(2,12))).sum(-1).mean()
    result={'passed':True,'kind':'synthetic CPU smoke; not research retraining','prepared_rows':prepared.height,
      'actor_steps':2,'head_steps':8,'zero_head_identity':True,'fixed_support_mass':True,'ndcg10':float(ndcg),
      'initial_objective':first,'final_objective':float(objective)}
    (ROOT/'checks/smoke_e2e.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
