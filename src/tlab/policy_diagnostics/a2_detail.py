"""Complete A2 on-policy slice expectations and actual full-list top-10 actions."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import torch

from tlab.data.store import SequenceStore
from tlab.residual_support.features import POLICY_FEATURE_NAMES, candidate_features, history_context_features
from tlab.residual_support.rm import load_priors
from tlab.residual_support.scaler import TrainOnlyFeatureScaler
from tlab.residual_support.train_policy import _load_cache
from .complete_evaluation import PANEL_DIRS, _predict_checkpoint
from .diagnostics import _model_set, _predict_all
from .protocol import ARTIFACTS, ROOT, record_operation, update_stage


def _softmax(logits: np.ndarray) -> np.ndarray:
    x=np.asarray(logits,dtype=np.float64); x=x-np.max(x,axis=1,keepdims=True)
    q=np.exp(x); return q/q.sum(axis=1,keepdims=True)


def _runs() -> list[tuple[str,dict[str,Any],Path]]:
    main=json.loads((ROOT/"artifacts/residual_support/policy/main_fit_summary.json").read_text(encoding="utf-8"))
    runs=[]
    for meta in main["runs"]:
        runs.append((meta["run_name"],meta,ROOT/"artifacts/residual_support/policy/runs"/meta["run_name"]))
    cont=ROOT/"artifacts/residual_support/policy_diagnostic/runs/exact_expected_reward_b0.3_seed17_continuation"
    runs.append((cont.name,{"run_name":cont.name,"objective":"exact_expected_reward","beta":.3,"seed":17,"variant":"continuation"},cont))
    return runs


def run_a2_detail() -> dict[str,Any]:
    started=time.perf_counter(); models=_model_set(); rm_info=json.loads((ROOT/"artifacts/residual_support/rm/rm_diagnostics.json").read_text(encoding="utf-8"))
    scale=json.loads((ROOT/"artifacts/residual_support/rm/reward_scale.json").read_text(encoding="utf-8")); mean=float(scale["mean"]); std=max(float(scale["std"]),float(scale["std_floor"]))
    scaler_data=json.loads((ROOT/"artifacts/residual_support/policy/scaler.json").read_text(encoding="utf-8")); scaler_data.pop("sha256",None)
    for field in ("feature_names","log1p_columns","mean","scale","constant_columns"): scaler_data[field]=tuple(scaler_data[field])
    scaler=TrainOnlyFeatureScaler(**scaler_data); runs=_runs(); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detail_rows=[]; panel_meta={}; panel_targets={}; support_predictions={}
    repeat_col=POLICY_FEATURE_NAMES.index("personal_repeat_count_1000")
    support_cols={"unsupported":POLICY_FEATURE_NAMES.index("rm_fit_unsupported"),
                   "rare_1_9":POLICY_FEATURE_NAMES.index("rm_fit_support_1_9"),
                   "rare_10_99":POLICY_FEATURE_NAMES.index("rm_fit_support_10_99"),
                   "supported_100_plus":POLICY_FEATURE_NAMES.index("rm_fit_support_100_plus")}
    for panel,cache in PANEL_DIRS.items():
        if panel=="train": continue
        meta,arr,targets=_load_cache(cache,include_full=False)
        n,k=arr["support_indices"].shape; x=np.asarray(arr["candidate_features"])
        pred_flat=_predict_all(models,x.reshape(n*k,x.shape[-1]))
        pred={name:value.reshape(n,k) for name,value in pred_flat.items()}
        qref=_softmax(np.asarray(arr["reference_conditional_logp"]))
        mass=np.asarray(arr["support_mass"],dtype=np.float64)
        # Reference and all registered analytic beta tilts.
        policies=[("strong_gpt",None,qref)]
        raw=np.asarray(arr["raw_candidate_rewards"],dtype=np.float64)
        for beta in (3.,1.,.3,.1): policies.append((f"analytic_rm_beta_{beta:g}",None,_softmax(np.asarray(arr["reference_conditional_logp"])+(raw-mean)/(std*beta))))
        for run_name,rm,run_dir in runs:
            manifest_path=run_dir/"run_manifest.json"
            manifest=json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else rm
            for endpoint in ("last","best"):
                ckpt=run_dir/f"{endpoint}.pt"
                if not ckpt.exists():continue
                saved=torch.load(ckpt,map_location="cpu",weights_only=False)
                step=int(saved.get("step",saved.get("updates_done",manifest.get("best_checkpoint_step",-1))))
                kind="identity_step0" if endpoint=="best" and step==0 else endpoint
                q,_=_predict_checkpoint(ckpt,manifest,arr,scaler,device)
                name=f"policy:{run_name}:{kind}"
                policies.append((name,int(rm["seed"]),np.asarray(q,dtype=np.float64)))
        mask_new=np.asarray(x[...,repeat_col]==0)
        reference_proxy=np.sum(qref*raw,axis=1); reference_audit=np.sum(qref*pred["audit"],axis=1)
        for support_name,col in support_cols.items():
            support_mask=np.asarray(x[...,col]>0.5)
            for novelty,novel_mask in (("new",mask_new),("repeat",~mask_new)):
                group_mask=support_mask&novel_mask
                for name,seed,q in policies:
                    share=np.sum(q*group_mask,axis=1)
                    denom=float(share.sum())
                    if denom>0:
                        proxy_raw=float(np.sum(q*raw*group_mask)/denom)
                        audit_raw=float(np.sum(q*pred["audit"]*group_mask)/denom)
                    else: proxy_raw=audit_raw=None
                    detail_rows.append({"panel":panel,"method":name,"seed":seed,"slice":f"{novelty}__{support_name}",
                                        "n_contexts":n,"mean_policy_conditional_share":float(share.mean()),
                                        "mean_support_mass":float(mass.mean()),"mean_policy_full_mass_share":float(np.mean(mass*share)),
                                        "slice_conditional_E_proxy_raw":proxy_raw,"slice_conditional_E_audit_raw":audit_raw,
                                        "mean_q_max":float(np.max(q,axis=1).mean()),"mean_q_top10_mass":float(np.partition(q,-10,axis=1)[:,-10:].sum(axis=1).mean()),
                                        "mean_q_entropy":float((-np.sum(q*np.log(np.maximum(q,1e-300)),axis=1)).mean()),
                                        "mean_conditional_KL_to_reference":float(np.sum(q*(np.log(np.maximum(q,1e-300))-np.log(np.maximum(qref,1e-300))),axis=1).mean()),
                                        "proxy_and_audit_are_raw_ratio":True,"tail":"fixed strong-reference tail; use mass-weighted deltas only"})
        for name,seed,q in policies:
            eq_proxy=np.sum(q*raw,axis=1);eq_audit=np.sum(q*pred["audit"],axis=1)
            eq_z=(eq_proxy-mean)/std
            detail_rows.append({"panel":panel,"method":name,"seed":seed,"slice":"all_candidates",
                                "n_contexts":n,"mean_policy_conditional_share":1.0,
                                "mean_support_mass":float(mass.mean()),"mean_policy_full_mass_share":float(mass.mean()),
                                "slice_conditional_E_proxy_raw":float(eq_proxy.mean()),
                                "slice_conditional_E_proxy_z":float(eq_z.mean()),
                                "slice_conditional_E_audit_raw":float(eq_audit.mean()),
                                "conditional_delta_proxy_raw_vs_reference":float(np.mean(eq_proxy-reference_proxy)),
                                "conditional_delta_proxy_z_vs_reference":float(np.mean(eq_z-(reference_proxy-mean)/std)),
                                "conditional_delta_audit_raw_vs_reference":float(np.mean(eq_audit-reference_audit)),
                                "mass_weighted_delta_proxy_raw_vs_reference":float(np.mean(mass*(eq_proxy-reference_proxy))),
                                "mass_weighted_delta_audit_raw_vs_reference":float(np.mean(mass*(eq_audit-reference_audit))),
                                "mean_q_max":float(np.max(q,axis=1).mean()),
                                "mean_q_top10_mass":float(np.partition(q,-10,axis=1)[:,-10:].sum(axis=1).mean()),
                                "mean_q_entropy":float((-np.sum(q*np.log(np.maximum(q,1e-300)),axis=1)).mean()),
                                "mean_conditional_KL_to_reference":float(np.sum(q*(np.log(np.maximum(q,1e-300))-np.log(np.maximum(qref,1e-300))),axis=1).mean()),
                                "proxy_raw_and_z_are_separate":True,"full_absolute_expectation_includes_unestimated_tail":True,
                                "full_delta_uses_unchanged_reference_tail":True})
        panel_meta[panel]=(meta,arr,targets); panel_targets[panel]={int(t.uid):t for t in targets}; support_predictions[panel]=pred

    # Gather the distinct actions appearing in any historical top-10 list, then score those actual actions.
    top_path=ARTIFACTS/"evaluation/top10_recommendations.csv"
    actions:dict[tuple[str,int],set[int]]={}
    with top_path.open(newline="",encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key=(row["panel"],int(row["uid"]))
            actions.setdefault(key,set()).add(int(row["catalog_index"]))
    store_protocol=json.loads((ROOT/"data/processed_full_model_tminus14/protocol.json").read_text(encoding="utf-8"))
    store=SequenceStore.load_cache(ROOT/"data/processed_full_model_tminus14/sequence_index",expected_source_sha256=store_protocol["processed_listens_sha256"])
    catalog=pl.read_parquet(ROOT/"data/processed_full_model_tminus14/catalog.parquet").sort("token_id")
    item_ids=np.asarray(catalog["item_id"].to_list(),dtype=np.int64)
    priors=load_priors(ROOT/"artifacts/residual_support/rm/priors_for_policy_cache.npz",expected_mapping_sha256=rm_info["catalog"]["raw_token_mapping_sha256"])
    out_actions=ARTIFACTS/"evaluation/top10_action_predictions.csv"; action_preds={}
    with out_actions.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=["panel","uid","catalog_index","raw_item_id","optimized_rm_proxy_raw","optimized_rm_proxy_z","external_audit_raw"])
        writer.writeheader()
        for (panel,uid),indices_set in actions.items():
            if panel not in panel_targets: continue
            target=panel_targets[panel].get(uid)
            if target is None: raise KeyError(f"top-10 context missing from policy cache: {panel}/{uid}")
            indices=np.asarray(sorted(indices_set),dtype=np.int64)
            if np.any((indices<0)|(indices>=len(item_ids))): raise ValueError("top-10 action is outside catalog")
            user=store.users[target.user_index]
            context=history_context_features(user,target.position)
            candidate=candidate_features(user,target.position,item_ids[indices],catalog_indices=indices,priors=priors)
            rm_x=np.column_stack((np.repeat(context[None,:],len(indices),axis=0),candidate)).astype(np.float32)
            proxy=np.clip(models["proxy"].predict(rm_x),0,1); audit=np.clip(models["audit"].predict(rm_x),0,1)
            action_preds[(panel,uid)]={int(idx):(float(r),float(a)) for idx,r,a in zip(indices,proxy,audit)}
            for idx,r,a in zip(indices,proxy,audit):
                writer.writerow({"panel":panel,"uid":uid,"catalog_index":int(idx),"raw_item_id":int(item_ids[idx]),
                                 "optimized_rm_proxy_raw":float(r),"optimized_rm_proxy_z":float((r-mean)/std),"external_audit_raw":float(a)})
    list_acc:dict[tuple[str,str,str,str],dict[int,list[float]]]={}
    with top_path.open(newline="",encoding="utf-8") as f:
        for row in csv.DictReader(f):
            panel=row["panel"]; uid=int(row["uid"]); idx=int(row["catalog_index"])
            values=action_preds[(panel,uid)][idx]
            seed=row["seed"] or ""
            group=(panel,row["method"],row["endpoint"],seed)
            by_uid=list_acc.setdefault(group,{})
            acc=by_uid.setdefault(uid,[0.0,0.0,0.0])
            acc[0]+=values[0];acc[1]+=values[1];acc[2]+=1
    top_rows=[]
    for (panel,method,endpoint,seed),by_uid in list_acc.items():
        per_proxy=np.asarray([v[0]/v[2] for v in by_uid.values()]); per_audit=np.asarray([v[1]/v[2] for v in by_uid.values()])
        n=len(per_audit)
        for label,values in (("optimized_RM_top10_list_mean_raw",per_proxy),("external_audit_top10_list_mean_raw",per_audit)):
            avg=float(values.mean()); se=float(values.std(ddof=1)/np.sqrt(n)) if n>1 else float("nan")
            top_rows.append({"panel":panel,"method":method,"endpoint":endpoint,"seed":seed,"metric":label,"n_lists_users":n,
                             "mean_over_top10_lists":avg,"normal_approx_95pct_ci_lower":avg-1.96*se,"normal_approx_95pct_ci_upper":avg+1.96*se,
                             "list_metric_not_policy_expectation":True,"all_actual_top10_actions_scored":True})
    eval_dir=ARTIFACTS/"evaluation"; eval_dir.mkdir(parents=True,exist_ok=True)
    def write_csv(path:Path,rows:list[dict[str,Any]])->None:
        fields=list(dict.fromkeys(k for r in rows for k in r))
        with path.open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    write_csv(eval_dir/"policy_audit_slices.csv",detail_rows)
    write_csv(eval_dir/"policy_audit_top10.csv",top_rows)
    summary={"runtime_seconds":round(time.perf_counter()-started,3),"n_policy_slice_rows":len(detail_rows),"n_top10_summary_rows":len(top_rows),
             "unique_actual_top10_actions":sum(len(v) for v in actions.values()),"top10_recommendation_source":"A1 actual full-catalog top-10 recommendations",
             "top10_feature_schema":"exact panel_tools.1-compatible 35-column RM schema using frozen policy priors, event prefix and catalog IDs",
             "external_audit_role":"diagnostic only; not used for beta/checkpoint/RM/policy selection",
             "inference":"support slice expectations are conditional on K=500; mass-weighted changes share the unchanged strong-reference tail",
             "top10_uncertainty":"normal-approximation 95% interval over user-level list means; lists are descriptive, panels were previously opened"}
    (eval_dir/"policy_audit_detail_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    record_operation("A2_full_support_slices_and_all_actual_top10_action_predictions",summary["runtime_seconds"],bucket="A",category="evaluation_and_preprocessing",
                     evidence=["artifacts/policy_diagnostics/evaluation/policy_audit_slices.csv","artifacts/policy_diagnostics/evaluation/policy_audit_top10.csv","artifacts/policy_diagnostics/evaluation/top10_action_predictions.csv"])
    update_stage("A","complete",["artifacts/policy_diagnostics/evaluation/legacy_completion.csv","artifacts/policy_diagnostics/diagnostics/external_audit_expectations.csv","artifacts/policy_diagnostics/evaluation/policy_audit_slices.csv","artifacts/policy_diagnostics/evaluation/policy_audit_top10.csv"],
                 note="A1-A3 and A2 complete; policy expectations use the fixed support and unchanged tail, and the separate top-10 metric scores every actual full-catalog list action, including actions outside support.")
    return summary


if __name__=="__main__": print(json.dumps(run_a2_detail(),indent=2))
