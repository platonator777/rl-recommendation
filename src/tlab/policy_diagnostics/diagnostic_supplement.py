"""B1-B3/B5 supplement: strata, RM-feature collisions, paired signs, train gaps."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from scipy.stats import rankdata

from tlab.residual_support.features import POLICY_FEATURE_NAMES
from tlab.residual_support.scaler import TrainOnlyFeatureScaler
from tlab.residual_support.train_policy import _load_cache
from .complete_evaluation import PANEL_DIRS, _predict_checkpoint
from .diagnostics import _model_set, _predict_all, _weighted_corr, _weighted_spearman
from .protocol import ARTIFACTS, ROOT, record_operation, update_stage


def _scaler()->TrainOnlyFeatureScaler:
    d=json.loads((ROOT/"artifacts/residual_support/policy/scaler.json").read_text(encoding="utf-8"));d.pop("sha256",None)
    for f in ("feature_names","log1p_columns","mean","scale","constant_columns"):d[f]=tuple(d[f])
    return TrainOnlyFeatureScaler(**d)


def _rows(path:Path,rows:list[dict[str,Any]])->None:
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def _fit_curve_gaps(panel:str,arr:dict[str,Any],targets:list[Any],scaler:TrainOnlyFeatureScaler,
                    scale:dict[str,Any],rows:list[dict[str,Any]],subset:str)->None:
    mean=float(scale["mean"]);std=max(float(scale["std"]),float(scale["std_floor"]))
    lp=np.asarray(arr["reference_conditional_logp"],dtype=np.float64);raw=np.asarray(arr["raw_candidate_rewards"],dtype=np.float64)
    mass=np.asarray(arr["support_mass"],dtype=np.float64);z=(raw-mean)/std
    main=json.loads((ROOT/"artifacts/residual_support/policy/main_fit_summary.json").read_text(encoding="utf-8"))
    runs=[(x["run_name"],x,ROOT/"artifacts/residual_support/policy/runs"/x["run_name"]) for x in main["runs"]]
    cont=ROOT/"artifacts/residual_support/policy_diagnostic/runs/exact_expected_reward_b0.3_seed17_continuation"
    runs.append((cont.name,{"run_name":cont.name,"objective":"exact_expected_reward","beta":.3,"seed":17,"variant":"continuation"},cont))
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for run_name,meta,run_dir in runs:
        manifest_path=run_dir/"run_manifest.json";manifest=json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else meta
        beta=float(meta["beta"])
        if beta not in (1.0,.3):continue
        for endpoint in ("last","best"):
            ckpt=run_dir/f"{endpoint}.pt"
            if not ckpt.exists():continue
            saved=torch.load(ckpt,map_location="cpu",weights_only=False)
            step=int(saved.get("step",saved.get("updates_done",manifest.get("best_checkpoint_step",-1))))
            ep="identity_step0" if endpoint=="best" and step==0 else endpoint
            q,_=_predict_checkpoint(ckpt,manifest,arr,scaler,device);q=np.asarray(q,dtype=np.float64)
            qref=np.exp(lp);qref/=qref.sum(axis=1,keepdims=True)
            logstar=lp+z/beta;logstar-=np.logaddexp.reduce(logstar,axis=1,keepdims=True);qstar=np.exp(logstar)
            klref=np.sum(q*(np.log(np.maximum(q,1e-300))-np.log(np.maximum(qref,1e-300))),axis=1)
            klstar=np.sum(q*(np.log(np.maximum(q,1e-300))-logstar),axis=1)
            j=mass*(np.sum(q*z,axis=1)-beta*klref)
            jstar=mass*(np.sum(qstar*z,axis=1)-beta*np.sum(qstar*(logstar-lp),axis=1))
            jref=mass*np.sum(qref*z,axis=1)
            recovered=float(np.mean(j-jref)/np.mean(jstar-jref)) if abs(float(np.mean(jstar-jref)))>1e-12 else None
            rows.append({"panel":panel,"train_subset":subset,"method":run_name,"objective":meta["objective"],"beta":beta,
                         "seed":int(meta["seed"]),"endpoint":ep,"training_step":step,
                         "training_seconds":manifest.get("elapsed_training_and_evaluation_seconds"),
                         "n_contexts":len(mass),"mean_exact_J":float(j.mean()),"mean_analytic_J":float(jstar.mean()),
                         "mean_exact_gap":float((jstar-j).mean()),"mean_beta_mass_KL_q_to_qstar":float((beta*mass*klstar).mean()),
                         "mean_reference_J":float(jref.mean()),
                         "max_abs_identity_error":float(np.max(np.abs((jstar-j)-beta*mass*klstar))),
                         "mean_KL_q_to_reference":float(klref.mean()),"mean_KL_q_to_qstar":float(klstar.mean()),
                         "recovered_fraction_of_analytic_J":recovered,
                         "mean_abs_logit_residual":float(np.abs(np.log(np.maximum(q,1e-300))-lp).mean())})


def run_supplement()->dict[str,Any]:
    t0=time.perf_counter();models=_model_set();rm=json.loads((ROOT/"artifacts/residual_support/rm/rm_diagnostics.json").read_text(encoding="utf-8"))
    scale=json.loads((ROOT/"artifacts/residual_support/rm/reward_scale.json").read_text(encoding="utf-8"));scaler=_scaler()
    variance=[];pairs=[];collisions=[];gaps=[]
    repeat_col=POLICY_FEATURE_NAMES.index("personal_repeat_count_1000")
    strata_cols={"unsupported":POLICY_FEATURE_NAMES.index("rm_fit_unsupported"),"rare_1_9":POLICY_FEATURE_NAMES.index("rm_fit_support_1_9"),
                 "rare_10_99":POLICY_FEATURE_NAMES.index("rm_fit_support_10_99"),"supported_100_plus":POLICY_FEATURE_NAMES.index("rm_fit_support_100_plus")}
    for panel in ("train","development"):
        cache=ROOT/"artifacts/residual_support/cache"/panel
        meta,arr,targets=_load_cache(cache,include_full=False)
        if panel=="train":
            selected=sorted(range(len(targets)),key=lambda i:hashlib.sha256((bytes.fromhex('76345f323a425f747261696e3a').decode('ascii')+str(targets[i].event_id)).encode()).digest())[:1024]
            targets=[targets[i] for i in selected];arr={key:value[selected] for key,value in arr.items()}
        n,k=arr["support_indices"].shape;x=np.asarray(arr["candidate_features"])
        raw_proxy=np.asarray(arr["raw_candidate_rewards"],dtype=np.float64);p=np.exp(np.asarray(arr["reference_conditional_logp"],dtype=np.float64));p/=p.sum(axis=1,keepdims=True)
        mass=np.asarray(arr["support_mass"],dtype=np.float64)
        predflat=_predict_all(models,x.reshape(n*k,x.shape[-1]));pred={a:b.reshape(n,k).astype(np.float64) for a,b in predflat.items()}
        rewards={"optimized_RM_proxy_raw":raw_proxy,**{f"{name}_raw":val for name,val in pred.items()}}
        candidate_groups={"all":np.ones((n,k),bool),"new":np.asarray(x[...,repeat_col]==0),"repeat":np.asarray(x[...,repeat_col]>0)}
        for name,col in strata_cols.items():candidate_groups[name]=np.asarray(x[...,col]>.5)
        for novelty in ("new","repeat"):
            for support in strata_cols:candidate_groups[f"{novelty}__{support}"]=candidate_groups[novelty]&candidate_groups[support]
        ent=-np.sum(p*np.log(np.maximum(p,1e-300)),axis=1)
        for reward_name,reward in rewards.items():
            sd_by=np.std(reward,axis=1)
            constant=np.ptp(reward,axis=1)<=1e-8
            for group,mask in candidate_groups.items():
                basews=(np.full((n,k),1/k,dtype=np.float64),p)
                for weighting,basew in zip(("uniform_candidates","reference_conditional"),basews):
                    masked=basew*mask
                    denom=masked.sum(axis=1);valid=denom>0
                    if not np.any(valid):continue
                    w=np.divide(masked,denom[:,None],out=np.zeros_like(masked),where=denom[:,None]>0)
                    mu=np.sum(w*reward,axis=1);within_i=np.sum(w*(reward-mu[:,None])**2,axis=1)
                    for context_weighting,cw_all in (("uniform_context",np.ones(n)),("support_mass_weighted_context",mass)):
                        cw=cw_all[valid].astype(np.float64);cw/=cw.sum()
                        vwithin=float(np.dot(cw,within_i[valid]));mu_global=float(np.dot(cw,mu[valid]));vbetween=float(np.dot(cw,(mu[valid]-mu_global)**2));total=vwithin+vbetween
                        variance.append({"panel":panel,"n_contexts":int(valid.sum()),"reward":reward_name,"candidate_group":group,
                                         "candidate_weighting":weighting,"context_weighting":context_weighting,"mean":mu_global,
                                         "within_variance":vwithin,"between_variance":vbetween,"total_variance":total,
                                         "within_fraction":vwithin/total if total>1e-15 else None,
                                         "between_fraction":vbetween/total if total>1e-15 else None,
                                         "decomposition_identity_abs_error":abs(total-vwithin-vbetween),
                                         "conditional_std_q10":float(np.quantile(np.sqrt(within_i[valid]),.1)),
                                         "conditional_std_median":float(np.median(np.sqrt(within_i[valid]))),
                                         "conditional_std_q90":float(np.quantile(np.sqrt(within_i[valid]),.9)),
                                         "conditional_std_z_using_frozen_proxy_scale":float(np.sqrt(vwithin)/max(float(scale["std"]),float(scale["std_floor"]))),
                                         "conditional_std_z_over_beta_0_3":float(np.sqrt(vwithin)/max(float(scale["std"]),float(scale["std_floor"]))/.3),
                                         "conditional_std_z_over_beta_1":float(np.sqrt(vwithin)/max(float(scale["std"]),float(scale["std_floor"]))),
                                         "constant_context_fraction":float(constant[valid].mean()),
                                         "mean_reference_entropy":float(ent[valid].mean()),"mean_support_mass":float(mass[valid].mean())})
        # Exact RM input and reward collisions, ignoring rank/embedding extras.
        for i in range(n):
            rm_unique=np.unique(np.asarray(x[i,:,:35]),axis=0).shape[0]
            rw=np.unique(raw_proxy[i]).size
            collisions.append({"panel":panel,"uid":int(targets[i].uid),"n_support_candidates":k,
                               "unique_exact_35d_RM_inputs":int(rm_unique),"unique_proxy_rewards_exact_float":int(rw),
                               "RM_input_collision_fraction":float(1-rm_unique/k),"proxy_reward_collision_fraction":float(1-rw/k),
                               "policy_rank_and_embedding_excluded":True})
        # Deterministic, outcome-blind action pairs for sign agreement after within-context centering.
        for i,target in enumerate(targets):
            seed=int.from_bytes(hashlib.sha256(f"{bytes.fromhex('76345f323a42323a706169723a').decode('ascii')}{target.event_id}".encode()).digest()[:8],"big")
            rng=np.random.default_rng(seed);a=rng.integers(0,k,size=256);b=rng.integers(0,k,size=256)
            keep=a!=b;a=a[keep];b=b[keep]
            dp=raw_proxy[i,a]-raw_proxy[i,b];da=pred["audit"][i,a]-pred["audit"][i,b]
            valid=(np.abs(dp)>1e-10)&(np.abs(da)>1e-10)
            weights=np.sqrt(p[i,a]*p[i,b]);weights=weights[valid]
            sign_agree=float(np.sum(weights*((dp[valid]*da[valid])>0))/weights.sum()) if weights.sum()>0 else None
            pairs.append({"panel":panel,"uid":int(target.uid),"n_pairs_sampled":int(len(a)),"n_non_tied_pairs":int(valid.sum()),
                          "reference_probability_pair_weighted_sign_agreement":sign_agree,
                          "proxy_constant":bool(np.ptp(raw_proxy[i])<=1e-8),"audit_constant":bool(np.ptp(pred["audit"][i])<=1e-8)})
        _fit_curve_gaps(panel,arr,targets,scaler,scale,gaps,"hash_selected_1024" if panel=="train" else "all_514")
        del predflat,pred,x
    out=ARTIFACTS/"diagnostics";out.mkdir(parents=True,exist_ok=True)
    _rows(out/"variance_by_novelty_support.csv",variance);_rows(out/"rm_input_reward_collisions.csv",collisions)
    _rows(out/"proxy_audit_deterministic_pairs.csv",pairs);_rows(out/"analytic_gap_by_seed_checkpoint.csv",gaps)
    summary={"runtime_seconds":round(time.perf_counter()-t0,3),"train_contexts":1024,"development_contexts":514,
             "variance_rows":len(variance),"collision_contexts":len(collisions),"pair_contexts":len(pairs),"checkpoint_gap_rows":len(gaps),
             "reward_label":"optimized and external audit models predict the logged ratio; neither is ground truth",
             "train_selection":"stable SHA256 on event_id before outcomes","pair_selection":"stable SHA256 on event_id followed by deterministic candidate-index draws"}
    (out/"diagnostic_supplement_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    record_operation("B1_B2_B3_B5_strata_collision_pairs_and_train_checkpoint_gaps",summary["runtime_seconds"],bucket="B",category="diagnostics",
                     evidence=[f"artifacts/policy_diagnostics/diagnostics/{x}" for x in ("variance_by_novelty_support.csv","rm_input_reward_collisions.csv","proxy_audit_deterministic_pairs.csv","analytic_gap_by_seed_checkpoint.csv")])
    update_stage("B1","complete",["artifacts/policy_diagnostics/diagnostics/variance_by_novelty_support.csv","artifacts/policy_diagnostics/diagnostics/rm_input_reward_collisions.csv"])
    update_stage("B2","complete",["artifacts/policy_diagnostics/diagnostics/proxy_audit_deterministic_pairs.csv","artifacts/policy_diagnostics/diagnostics/reward_variance_alignment_beta.csv"])
    update_stage("B3","complete",["artifacts/policy_diagnostics/diagnostics/analytic_gap_by_seed_checkpoint.csv","artifacts/policy_diagnostics/diagnostics/gradient_noise.csv"])
    update_stage("B5","complete",["artifacts/policy_diagnostics/diagnostics/representability_embedding_ridge.csv","artifacts/policy_diagnostics/diagnostics/analytic_gap_by_seed_checkpoint.csv"])
    update_stage("B6","complete",["artifacts/policy_diagnostics/diagnostics/support_coverage.csv","artifacts/policy_diagnostics/evaluation/legacy_completion_slices.csv"])
    return summary


if __name__=="__main__":print(json.dumps(run_supplement(),indent=2))
