"""Cross-user and per-context probes of the panel_tools.1 head's analytic correction target."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tlab.panel_tools.evaluator import load_actor_checkpoint
from tlab.residual_support.features import POLICY_FEATURE_NAMES
from tlab.residual_support.scaler import TrainOnlyFeatureScaler
from tlab.residual_support.train_policy import _load_cache
from .complete_evaluation import _predict_checkpoint
from .protocol import ARTIFACTS, ROOT, record_operation, update_stage


def _scaler()->TrainOnlyFeatureScaler:
    d=json.loads((ROOT/"artifacts/residual_support/policy/scaler.json").read_text(encoding="utf-8"));d.pop("sha256",None)
    for f in ("feature_names","log1p_columns","mean","scale","constant_columns"):d[f]=tuple(d[f])
    return TrainOnlyFeatureScaler(**d)


def _weighted_metrics(y:np.ndarray,pred:np.ndarray,w:np.ndarray)->dict[str,float|None]:
    y=np.asarray(y,np.float64);pred=np.asarray(pred,np.float64);w=np.asarray(w,np.float64);w=w/w.sum()
    mse=float(np.sum(w*(y-pred)**2));mu=float(np.sum(w*y));var=float(np.sum(w*(y-mu)**2))
    return {"mse":mse,"r2":float(1-mse/var) if var>1e-15 else None}


def _save(path:Path,rows:list[dict[str,Any]])->None:
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def run_expressivity()->dict[str,Any]:
    started=time.perf_counter();scale=json.loads((ROOT/"artifacts/residual_support/rm/reward_scale.json").read_text(encoding="utf-8"));mean=float(scale["mean"]);std=max(float(scale["std"]),float(scale["std_floor"]))
    scaler=_scaler();data={};selected_train=None
    for panel in ("train","development"):
        meta,arr,targets=_load_cache(ROOT/"artifacts/residual_support/cache"/panel,include_full=False)
        if panel=="train":
            selected_train=sorted(range(len(targets)),key=lambda i:hashlib.sha256((bytes.fromhex('76345f323a425f747261696e3a').decode('ascii')+str(targets[i].event_id)).encode()).digest())[:1024]
            targets=[targets[i] for i in selected_train];arr={k:v[selected_train] for k,v in arr.items()}
        data[panel]=(meta,arr,targets)
    train_meta,tr,train_targets=data["train"];dev_meta,dv,dev_targets=data["development"]
    actor_path=ROOT/"models/actor/strong.pt"
    actor,_=load_actor_checkpoint(actor_path,vocab_size=int(train_meta["catalog_size"])+3,n_positions=256,n_layer=4,n_embd=256,n_head=4,device=torch.device("cpu"))
    emb=actor.lm.transformer.wte.weight.detach()[3:].float().cpu().numpy();del actor
    run=ROOT/"artifacts/residual_support/policy/runs/analytic_q_distillation_b0.3_seed17_full"
    manifest=json.loads((run/"run_manifest.json").read_text(encoding="utf-8"));head_path=run/"last.pt";saved=torch.load(head_path,map_location="cpu",weights_only=False)
    cols=tuple(int(x) for x in saved.get("feature_columns",manifest["feature_columns"]))
    scale_target=float(std)*.3
    train_labels={}; arrays_by_panel={"train":tr,"development":dv}
    ztargets={}; pweights={}; feature_mats={}; embedding_mats={}
    for panel,(_,arr,targets) in data.items():
        lp=np.asarray(arr["reference_conditional_logp"],dtype=np.float64);p=np.exp(lp);p/=p.sum(axis=1,keepdims=True)
        z=(np.asarray(arr["raw_candidate_rewards"],dtype=np.float64)-mean)/std
        target=z/.3-np.sum(p*z/.3,axis=1,keepdims=True)
        ztargets[panel]=target;pweights[panel]=p
        feature_mats[panel]=np.asarray(arr["candidate_features"])
        embedding_mats[panel]=emb[np.asarray(arr["support_indices"],dtype=np.int64)]
    # Choose ridge regularization on train-only contexts/candidate holdouts.
    alpha_scores={a:[] for a in (0.1,1.0,10.0,100.0,1000.0)}
    for i,target in enumerate(train_targets):
        k=ztargets["train"].shape[1]
        order=np.asarray(sorted(range(k),key=lambda j:hashlib.sha256(f"{bytes.fromhex('76345f323a42353a74756e653a').decode('ascii')}{target.event_id}:{j}".encode()).digest()))
        tr_ix,va_ix=order[:int(.7*k)],order[int(.7*k):]
        x=embedding_mats["train"][i];y=ztargets["train"][i]
        w=pweights["train"][i]
        for alpha in alpha_scores:
            model=Ridge(alpha=alpha).fit(x[tr_ix],y[tr_ix],sample_weight=w[tr_ix]*len(tr_ix))
            pred=model.predict(x[va_ix]);met=_weighted_metrics(y[va_ix],pred,w[va_ix])
            if met["r2"] is not None:alpha_scores[alpha].append(float(met["r2"]))
    chosen_alpha=max(alpha_scores,key=lambda a:float(np.mean(alpha_scores[a])) if alpha_scores[a] else -1e9)
    rows=[];ctx_rows=[];control_rows=[]
    # Cross-user predictors: constant, item-only, strict additive, and the already trained cross-feature residual MLP.
    item_cols=np.arange(29,35);add_cols=np.r_[np.arange(14),np.arange(29,35)]
    xtr=feature_mats["train"].reshape(-1,45);yd=ztargets["train"].reshape(-1);wtr=pweights["train"].reshape(-1)
    xd=feature_mats["development"].reshape(-1,45);ydv=ztargets["development"].reshape(-1);wd=pweights["development"].reshape(-1)
    # The targets are centered within context. Flattened weights normalize each context to equal user mass.
    wtr=wtr*feature_mats["train"].shape[1]
    wd=wd/len(dev_targets)
    item_model=make_pipeline(StandardScaler(),Ridge(alpha=10.0)).fit(xtr[:,item_cols],yd,ridge__sample_weight=wtr)
    additive=make_pipeline(StandardScaler(),Ridge(alpha=10.0)).fit(xtr[:,add_cols],yd,ridge__sample_weight=wtr)
    pred_cross={"constant_zero":np.zeros_like(ydv),"item_only_linear":item_model.predict(xd[:,item_cols]),
                "strict_additive_linear":additive.predict(xd[:,add_cols])}
    for endpoint in ("last","best"):
        ckpt=run/f"{endpoint}.pt"
        if not ckpt.exists():continue
        qtr,_=_predict_checkpoint(ckpt,manifest,tr,scaler,torch.device("cpu"));qd,_=_predict_checkpoint(ckpt,manifest,dv,scaler,torch.device("cpu"))
        for panel,q in (("train",qtr),("development",qd)):
            target=ztargets[panel];p=pweights[panel];delta=np.log(np.maximum(q,1e-300))-np.asarray(arrays_by_panel[panel]["reference_conditional_logp"],np.float64)
            delta-=np.sum(p*delta,axis=1,keepdims=True)
            y=target.reshape(-1);pred=delta.reshape(-1);w=(p/len(q)).reshape(-1)
            metw=_weighted_metrics(y,pred,w);metu=_weighted_metrics(y,pred,np.full_like(w,1/len(w)))
            rows.append({"panel":panel,"model":"current_analytic_q_distillation_residual_MLP","endpoint":endpoint,
                         "n_contexts":len(q),"feature_dim":len(cols),"target":"reference-centered normalized RM residual / beta=.3",
                         "weighted_mse":metw["mse"],"weighted_r2":metw["r2"],"uniform_mse":metu["mse"],"uniform_r2":metu["r2"],
                         "prediction_centered_with_reference_weights":True,"labels_are_proxy_targets_not_observed_truth":True})
    # Cross-user shared linear controls and heldout-dev candidate scores.
    for name,pred in pred_cross.items():
        for weighting,w in (("reference_conditional",wd),("uniform_candidates",np.full_like(wd,1/len(wd)))):
            met=_weighted_metrics(ydv,pred,w)
            rows.append({"panel":"development","model":name,"endpoint":"shared_train_to_dev",
                         "n_contexts":len(dev_targets),"feature_dim":0 if name=="constant_zero" else (6 if name=="item_only_linear" else 20),
                         "target":"reference-centered normalized RM residual / beta=.3","candidate_weighting":weighting,
                         "mse":met["mse"],"r2":met["r2"],"labels_are_proxy_targets_not_observed_truth":True})
    # Per-context candidate-heldout ridge, with train-only alpha selection and fit-fraction sensitivity.
    rng_seed=93001
    for panel,(_,arr,targets) in data.items():
        for i,target in enumerate(targets):
            k=ztargets[panel].shape[1];x=embedding_mats[panel][i];y=ztargets[panel][i];w=pweights[panel][i]
            order=np.asarray(sorted(range(k),key=lambda j:hashlib.sha256(f"{bytes.fromhex('76345f323a42353a68656c646f75743a').decode('ascii')}{panel}:{target.event_id}:{j}".encode()).digest()))
            for fraction in (.5,.7,.9):
                cut=int(fraction*k);tr_ix,te_ix=order[:cut],order[cut:]
                oracle=Ridge(alpha=chosen_alpha).fit(x[tr_ix],y[tr_ix],sample_weight=w[tr_ix]*len(tr_ix));pred=oracle.predict(x[te_ix])
                const=np.full(len(te_ix),float(np.average(y[tr_ix],weights=w[tr_ix])))
                for candidate_weighting,ww in (("reference_conditional",w[te_ix]),("uniform_candidates",np.ones(len(te_ix)))):
                    met=_weighted_metrics(y[te_ix],pred,ww);base=_weighted_metrics(y[te_ix],const,ww)
                    ctx_rows.append({"panel":panel,"uid":int(target.uid),"fit_fraction":fraction,"n_fit_candidates":len(tr_ix),"n_test_candidates":len(te_ix),
                                     "embedding_dim":int(x.shape[1]),"ridge_alpha_train_selected":chosen_alpha,"candidate_weighting":candidate_weighting,
                                     "ridge_mse":met["mse"],"ridge_r2":met["r2"],"constant_mse":base["mse"],"constant_r2":base["r2"],
                                     "target":"reference-centered normalized RM correction / beta=.3","no_per_context_label_leak_to_test":True})
                # Negative control: shuffle training candidate targets and score against true heldout labels.
                rng=np.random.default_rng(rng_seed+i);yp=y[tr_ix].copy();rng.shuffle(yp)
                if np.isclose(fraction,.7):
                    negative=Ridge(alpha=chosen_alpha).fit(x[tr_ix],yp,sample_weight=w[tr_ix]*len(tr_ix)).predict(x[te_ix])
                    ref_lp=np.asarray(arr["reference_conditional_logp"][i],np.float64)
                    ref_centered=ref_lp-float(np.sum(w*ref_lp))
                    positive_logp=Ridge(alpha=chosen_alpha).fit(x[tr_ix],ref_centered[tr_ix],sample_weight=w[tr_ix]*len(tr_ix)).predict(x[te_ix])
                    for label,yy,pp in (("negative_permuted_correction",y[te_ix],negative),
                                        ("positive_control_centered_reference_logp",ref_centered[te_ix],positive_logp)):
                        for weighting,ww in (("reference_conditional",w[te_ix]),("uniform_candidates",np.ones(len(te_ix)))):
                            met=_weighted_metrics(yy,pp,ww);control_rows.append({"panel":panel,"uid":int(target.uid),"fit_fraction":fraction,
                                "control":label,"candidate_weighting":weighting,"n_fit_candidates":len(tr_ix),"n_test_candidates":len(te_ix),
                                "mse":met["mse"],"r2":met["r2"],"frozen_actor_embedding_dim":int(x.shape[1])})
    # Same-embedding positive and negative controls are summarized separately from proxy-target expressivity.
    out=ARTIFACTS/"diagnostics";out.mkdir(parents=True,exist_ok=True)
    _save(out/"expressivity_cross_user.csv",rows);_save(out/"expressivity_per_context_ridge.csv",ctx_rows);_save(out/"expressivity_embedding_controls.csv",control_rows)
    result={"runtime_seconds":round(time.perf_counter()-started,3),"train_contexts":len(train_targets),"development_contexts":len(dev_targets),
            "ridge_alpha_selected_on_train_only":chosen_alpha,"alpha_train_candidate_split_mean_r2":{str(a):float(np.mean(v)) if v else None for a,v in alpha_scores.items()},
            "per_context_ridge_rows":len(ctx_rows),"embedding_control_rows":len(control_rows),"shared_model_rows":len(rows),
            "target":"centered z(RM proxy)/beta=.3, equivalently centered analytic logit correction",
            "limitations":"all fit targets are the optimized RM proxy; this tests representability of the proxy, not reward truth; per-context ridge is an oracle for new users"}
    (out/"expressivity_summary.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    record_operation("B5_shared_crossfeature_and_percontext_embedding_expressivity",result["runtime_seconds"],bucket="B",category="representability_diagnostics",
                     evidence=["artifacts/policy_diagnostics/diagnostics/expressivity_cross_user.csv","artifacts/policy_diagnostics/diagnostics/expressivity_per_context_ridge.csv","artifacts/policy_diagnostics/diagnostics/expressivity_embedding_controls.csv"])
    update_stage("B5","complete",["artifacts/policy_diagnostics/diagnostics/expressivity_cross_user.csv","artifacts/policy_diagnostics/diagnostics/expressivity_per_context_ridge.csv","artifacts/policy_diagnostics/diagnostics/expressivity_embedding_controls.csv"])
    return result


if __name__=="__main__":print(json.dumps(run_expressivity(),indent=2))
