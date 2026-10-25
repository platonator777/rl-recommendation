"""Map sampled and exact logit gradients through the trained residual head parameters."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tlab.residual_support.features import POLICY_FEATURE_NAMES
from tlab.residual_support.head import ResidualHeadSpec, ResidualPolicyHead
from tlab.residual_support.scaler import TrainOnlyFeatureScaler
from tlab.residual_support.train_policy import _load_cache
from .protocol import ARTIFACTS, ROOT, record_operation, update_stage


def _scaler()->TrainOnlyFeatureScaler:
    d=json.loads((ROOT/"artifacts/residual_support/policy/scaler.json").read_text(encoding="utf-8"));d.pop("sha256",None)
    for f in ("feature_names","log1p_columns","mean","scale","constant_columns"):d[f]=tuple(d[f])
    return TrainOnlyFeatureScaler(**d)


def _param_grad(g:np.ndarray,x:np.ndarray,h:np.ndarray,w2:np.ndarray)->np.ndarray:
    """Exact reverse-mode linear map from candidate-logit gradients to all head parameters."""
    gb=g.sum()/len(g)
    gw2=np.einsum("bk,bkh->h",g,h)/len(g)
    gh=g[:,:,None]*w2[None,None,:]*(1-h*h)
    gw1=np.einsum("bkh,bkd->hd",gh,x)/len(g)
    gb1=gh.sum(axis=(0,1))/len(g)
    return np.concatenate((gw1.ravel(),gb1.ravel(),gw2.ravel(),np.asarray([gb],dtype=np.float64)))


def run_parameter_probe()->dict[str,Any]:
    started=time.perf_counter();meta,arr,targets=_load_cache(ROOT/"artifacts/residual_support/cache/train",include_full=False)
    selected=sorted(range(len(targets)),key=lambda i:hashlib.sha256((bytes.fromhex('76345f323a425f747261696e3a').decode('ascii')+str(targets[i].event_id)).encode()).digest())[:1024]
    targets=[targets[i] for i in selected];arr={k:v[selected] for k,v in arr.items()}
    scale=json.loads((ROOT/"artifacts/residual_support/rm/reward_scale.json").read_text(encoding="utf-8"));mean=float(scale["mean"]);std=max(float(scale["std"]),float(scale["std_floor"]))
    scaler=_scaler();run=ROOT/"artifacts/residual_support/policy/runs/exact_expected_reward_b0.3_seed17_full"
    manifest=json.loads((run/"run_manifest.json").read_text(encoding="utf-8"));ckpt=run/"last.pt";saved=torch.load(ckpt,map_location="cpu",weights_only=False)
    fit_summary=json.loads((ROOT/"artifacts/residual_support/policy/main_fit_summary.json").read_text(encoding="utf-8"))
    fit_row=next(r for r in fit_summary["runs"] if r["run_name"]==run.name)
    cols=tuple(int(x) for x in saved.get("feature_columns",manifest["feature_columns"]))
    head=ResidualPolicyHead(ResidualHeadSpec(len(cols),32));head.load_state_dict(saved.get("state_dict",saved.get("head_state")));head.eval()
    xraw=np.asarray(arr["candidate_features"]);xscaled=scaler.transform(xraw,POLICY_FEATURE_NAMES)[...,list(cols)]
    ref=np.asarray(arr["reference_conditional_logp"],dtype=np.float64);raw=np.asarray(arr["raw_candidate_rewards"],dtype=np.float64)
    z=(raw-mean)/std;mass=np.asarray(arr["support_mass"],dtype=np.float64)
    q=np.empty(ref.shape,dtype=np.float64);h=np.empty((*ref.shape,32),dtype=np.float64)
    preactivation=np.empty((*ref.shape,32),dtype=np.float64);residual=np.empty(ref.shape,dtype=np.float64)
    for start in range(0,len(ref),64):
        xx=torch.from_numpy(np.ascontiguousarray(xscaled[start:start+64]))
        with torch.no_grad():
            module=head.network
            pre=module[0](xx);act=module[1](pre);out=module[2](act).squeeze(-1)
            logits=torch.from_numpy(ref[start:start+64].astype(np.float32))+out
            q[start:start+64]=torch.softmax(logits,dim=-1).numpy()
            h[start:start+64]=act.numpy()
            preactivation[start:start+64]=pre.numpy();residual[start:start+64]=out.numpy()
    qref=np.exp(ref);qref/=qref.sum(axis=1,keepdims=True)
    all_rows=[]
    for batch in range(8):
        ix=np.arange(batch*32,(batch+1)*32);qb=q[ix];mb=mass[ix];rb=z[ix];pb=qref[ix];xb=xscaled[ix].astype(np.float64);hb=h[ix]
        eq=(qb*rb).sum(axis=1,keepdims=True);gr=-mb[:,None]*qb*(rb-eq)
        logq=np.log(np.maximum(qb,1e-300));logp=np.log(np.maximum(pb,1e-300));kl=(qb*(logq-logp)).sum(axis=1,keepdims=True)
        gkl=.3*mb[:,None]*qb*(logq-logp-kl);gt=gr+gkl
        w2=head.network[2].weight.detach().numpy().astype(np.float64).reshape(-1)
        exact=_param_grad(gt,xb,hb,w2)
        exact_reward=_param_grad(gr,xb,hb,w2);exact_kl=_param_grad(gkl,xb,hb,w2)
        for samples in (4,16,64):
            draws=[];reward_draws=[];norms=[];clipf=[]
            for rep in range(64):
                seed=770000+batch*100000+samples*1000+rep;rng=np.random.default_rng(seed)
                acts=np.stack([rng.choice(qb.shape[1],size=samples,replace=True,p=row/row.sum()) for row in qb])
                sr=np.take_along_axis(rb,acts,axis=1)
                adv=sr-(sr.sum(axis=1,keepdims=True)-sr)/(samples-1)
                gs=np.zeros_like(qb)
                for b in range(len(ix)):np.add.at(gs[b],acts[b],-mb[b]*adv[b]/samples)
                gtotal=gs+gkl
                pg=_param_grad(gtotal,xb,hb,w2);rg=_param_grad(gs,xb,hb,w2)
                norm=float(np.linalg.norm(pg));factor=min(1.,1./max(norm,1e-30))
                draws.append(pg);reward_draws.append(rg);norms.append(norm);clipf.append(factor)
            draws=np.asarray(draws);reward_draws=np.asarray(reward_draws);avg=draws.mean(axis=0)
            cosine=float(np.dot(exact,avg)/(np.linalg.norm(exact)*np.linalg.norm(avg)+1e-12))
            rel=float(np.linalg.norm(avg-exact)/(np.linalg.norm(exact)+1e-12))
            reward_avg=reward_draws.mean(axis=0)
            reward_cosine=float(np.dot(exact_reward,reward_avg)/(np.linalg.norm(exact_reward)*np.linalg.norm(reward_avg)+1e-12))
            reward_rel=float(np.linalg.norm(reward_avg-exact_reward)/(np.linalg.norm(exact_reward)+1e-12))
            clipped=draws*np.asarray(clipf)[:,None]
            per_draw_cosine=np.sum(draws*exact[None,:],axis=1)/(np.maximum(np.linalg.norm(draws,axis=1),1e-30)*np.linalg.norm(exact)+1e-12)
            rms_deviation=float(np.sqrt(np.mean(np.sum((draws-avg[None,:])**2,axis=1))))
            all_rows.append({"batch":batch,"batch_contexts":len(ix),"policy":"trained_last_seed17_exact_b0.3",
                             "samples_per_context":samples,"draws":64,"parameter_count":len(exact),
                             "exact_parameter_gradient_norm":float(np.linalg.norm(exact)),
                             "exact_reward_component_gradient_norm":float(np.linalg.norm(exact_reward)),
                             "exact_KL_component_gradient_norm":float(np.linalg.norm(exact_kl)),
                             "sampled_parameter_gradient_norm_mean":float(np.linalg.norm(draws,axis=1).mean()),
                             "sampled_reward_component_gradient_norm_mean":float(np.linalg.norm(reward_draws,axis=1).mean()),
                             "reward_component_cosine_of_sampled_mean_to_exact":reward_cosine,
                             "reward_component_relative_error_of_sampled_mean":reward_rel,
                             "cosine_of_mean_sampled_to_exact":cosine,"relative_error_of_mean_sampled":rel,
                             "gradient_bias_norm":float(np.linalg.norm(avg-exact)),
                             "negative_cosine_fraction":float(np.mean(per_draw_cosine<0)),
                             "mean_gradient_norm_to_rms_deviation":float(np.linalg.norm(avg)/(rms_deviation+1e-30)),
                             "sampled_norm_sd":float(np.linalg.norm(draws,axis=1).std(ddof=1)),
                             "preclip_over_1_fraction":float(np.mean(np.linalg.norm(draws,axis=1)>1)),
                             "mean_clip_factor_at_1":float(np.mean(clipf)),
                             "postclip_mean_gradient_norm":float(np.linalg.norm(clipped.mean(axis=0)))})
    out=ARTIFACTS/"diagnostics";out.mkdir(parents=True,exist_ok=True)
    path=out/"gradient_parameter_noise.csv"
    with path.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
    residual_centered=residual-np.sum(q*residual,axis=1,keepdims=True)
    residual_within_std=np.sqrt(np.sum(q*residual_centered**2,axis=1))
    parameter_norm=float(np.sqrt(sum(float(torch.sum(p.detach().double()**2)) for p in head.parameters())))
    result={"runtime_seconds":round(time.perf_counter()-started,3),"rows":len(all_rows),"contexts_per_minibatch":32,
            "minibatches":8,"draws_per_sample_size_and_minibatch":64,"gradient_space":"actual ResidualPolicyHead parameters",
            "clip_threshold":1.0,"checkpoint_step":int(fit_row["updates_completed"]),
            "checkpoint_training_seconds":float(fit_row["elapsed_training_and_evaluation_seconds"]),
            "parameter_l2_norm":parameter_norm,"tanh_saturation_fraction_abs_activation_over_0_99":float(np.mean(np.abs(h)>0.99)),
            "preactivation_abs_q99":float(np.quantile(np.abs(preactivation),.99)),
            "residual_rms":float(np.sqrt(np.mean(residual**2))),"residual_within_std_q10_median_q90":[float(np.quantile(residual_within_std,.1)),float(np.median(residual_within_std)),float(np.quantile(residual_within_std,.9))],
            "learning_rate_source":"panel_tools.1 run manifest configuration; see run_manifest.json"}
    (out/"gradient_parameter_noise_summary.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    record_operation("B4_parameter_gradient_noise_and_actual_clip_probe",result["runtime_seconds"],bucket="B",category="gradient_diagnostics",
                     evidence=["artifacts/policy_diagnostics/diagnostics/gradient_parameter_noise.csv"])
    update_stage("B4","complete",["artifacts/policy_diagnostics/diagnostics/gradient_noise.csv","artifacts/policy_diagnostics/diagnostics/gradient_parameter_noise.csv"])
    return result


if __name__=="__main__":print(json.dumps(run_parameter_probe(),indent=2))
