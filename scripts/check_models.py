"""Load bundled models on a fixed tiny input; fail early on missing external inputs."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from tlab.residual_support.head import ResidualPolicyHead,ResidualHeadSpec
from tlab.frozen_policy.training import FrozenPolicyHead,load_scaler,predict_checkpoint
from tlab.frozen_policy.data import Frozen_FEATURE_NAMES
from tlab.rm_scaling.experiment import _predict_v6_checkpoint

def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,default=ROOT/'work');p.add_argument('--require-external',action='store_true')
    p.add_argument('--output',type=Path,default=ROOT/'checks/models.json');a=p.parse_args()
    if a.require_external:
        assets=json.loads((ROOT/'assets_manifest.json').read_text())['assets']
        missing=[x['destination'] for x in assets if x['status']=='external' and x['role'] in {'actor','prepared_data','cache'} and x.get('required_for_evaluation',True) and not (a.workspace/x['destination']).is_file()]
        if missing:
            print(json.dumps({'status':'missing_external_resources','workspace':str(a.workspace),'missing':missing},indent=2));raise SystemExit(2)
    torch.set_num_threads(2);rng=np.random.default_rng(17)
    raw=rng.uniform(0,2,(2,500,54)).astype(np.float32)
    ref=torch.log_softmax(torch.from_numpy(rng.normal(size=(2,500)).astype(np.float32)),-1).numpy()
    arrays={'candidate_features':raw,'reference_conditional_logp':ref}
    scaler=load_scaler(ROOT/'protocol/policy/scaler.json')
    assert scaler.feature_names==tuple(Frozen_FEATURE_NAMES)
    outputs={};counts={'heads':0,'reward_models':0,'masked_heads':0}
    for f in sorted((ROOT/'models').rglob('*.pt')):
        if f.is_relative_to(ROOT/'models/actor'):continue
        saved=torch.load(f,map_location='cpu',weights_only=False);state=saved.get('state_dict',saved.get('head_state'))
        if 'feature_columns' in saved or 'head_state' in saved:
            width=state['network.0.weight'].shape[1]
            head=ResidualPolicyHead(ResidualHeadSpec(width,32));head.load_state_dict(state);head.eval()
            columns=saved.get('feature_columns',saved.get('state_signature',{}).get('feature_columns',list(range(width))))
            with torch.inference_mode():values=head(torch.from_numpy(raw[:,:,columns])).numpy()
        elif f.is_relative_to(ROOT/'models/reward_scaling'):
            masked=bool(saved.get('features_affinity_masked',False))
            values,_=_predict_v6_checkpoint(f,arrays,np.array([0,1]),scaler,mask_affinity=masked,device=torch.device('cpu'))
            if masked:
                changed=raw.copy();changed[:,:,45:54]+=100
                same,_=_predict_v6_checkpoint(f,dict(arrays,candidate_features=changed),np.array([0,1]),scaler,mask_affinity=True,device=torch.device('cpu'))
                np.testing.assert_array_equal(values,same);counts['masked_heads']+=1
        else:
            values,_=predict_checkpoint(f,arrays,np.array([0,1]),scaler,device=torch.device('cpu'))
        assert np.isfinite(values).all()
        outputs[f.relative_to(ROOT).as_posix()]=hashlib.sha256(values.tobytes()).hexdigest();counts['heads']+=1
    import joblib
    for f in sorted((ROOT/'models').rglob('*.joblib')):
        model=joblib.load(f)
        if hasattr(model,'n_jobs'):model.n_jobs=1
        if hasattr(model,'predict'):
            width=getattr(model,'n_features_in_',None)
            if width is not None:
                values=model.predict(np.zeros((2,int(width)),dtype=np.float64));assert np.isfinite(values).all()
                outputs[f.relative_to(ROOT).as_posix()]=hashlib.sha256(np.asarray(values).tobytes()).hexdigest();counts['reward_models']+=1
    zero=FrozenPolicyHead()
    with torch.no_grad():assert torch.count_nonzero(zero(torch.from_numpy(raw)))==0
    result={'passed':True,**counts,'zero_head_identity':True,'outputs_sha256':outputs,'scope':'Fixed synthetic inputs; full GPT inference requires external resources.'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='outputs_sha256'}))
if __name__=='__main__':main()

