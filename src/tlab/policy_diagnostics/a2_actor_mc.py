"""Monte Carlo full-catalog audit expectations for actors with different references."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch

from tlab.data.store import SequenceStore
from tlab.panel_tools.data import history_for_target
from tlab.panel_tools.evaluator import _collate_histories, load_actor_checkpoint
from tlab.residual_support.features import candidate_features, history_context_features
from tlab.residual_support.rm import load_priors
from tlab.residual_support.train_policy import _load_cache
from .complete_evaluation import PANEL_DIRS
from .diagnostics import _model_set
from .protocol import ARTIFACTS, ROOT, record_operation, update_stage


ACTORS={"dense_sft":ROOT/"artifacts/residual_support/sft/runs/dense/last.pt",
        "last_target_sft":ROOT/"artifacts/residual_support/sft/runs/last_target/last.pt",
        "weak_actor":ROOT/"models/actor/weak.pt"}
SAMPLE_COUNT=128


def run_actor_mc()->dict[str,Any]:
    started=time.perf_counter();models=_model_set();rm_info=json.loads((ROOT/"artifacts/residual_support/rm/rm_diagnostics.json").read_text(encoding="utf-8"))
    scale=json.loads((ROOT/"artifacts/residual_support/rm/reward_scale.json").read_text(encoding="utf-8"));mean=float(scale["mean"]);std=max(float(scale["std"]),float(scale["std_floor"]))
    data_protocol=json.loads((ROOT/"data/processed_full_model_tminus14/protocol.json").read_text(encoding="utf-8"))
    store=SequenceStore.load_cache(ROOT/"data/processed_full_model_tminus14/sequence_index",expected_source_sha256=data_protocol["processed_listens_sha256"])
    catalog=pl.read_parquet(ROOT/"data/processed_full_model_tminus14/catalog.parquet").sort("token_id");item_ids=np.asarray(catalog["item_id"].to_list(),dtype=np.int64)
    priors=load_priors(ROOT/"artifacts/residual_support/rm/priors_for_policy_cache.npz",expected_mapping_sha256=rm_info["catalog"]["raw_token_mapping_sha256"])
    support_count=np.asarray(rm_info["rm_fit_item_count"]["counts_by_catalog_item"],dtype=np.int64)
    repeat_col=__import__("tlab.residual_support.features",fromlist=["POLICY_FEATURE_NAMES"]).POLICY_FEATURE_NAMES.index("personal_repeat_count_1000")
    summary_rows=[];slice_rows=[];sample_counts={};action_pred={}
    panel_caches={};panel_targets={}
    for panel,cache in PANEL_DIRS.items():
        if panel=="train":continue
        meta,arr,targets=_load_cache(cache,include_full=True)
        panel_caches[panel]=(meta,arr,targets);panel_targets[panel]={int(t.uid):t for t in targets}
        n=len(targets);sample_counts[panel]={};action_pred[panel]={}
        # Reference is sampled from the full, unchanged 20k distribution.
        lp=np.asarray(arr["reference_logp_full"],dtype=np.float64);lp-=np.logaddexp.reduce(lp,axis=1,keepdims=True)
        refs=np.empty((n,SAMPLE_COUNT),dtype=np.int32)
        for i,target in enumerate(targets):
            rng=np.random.default_rng(int.from_bytes(hashlib.sha256(f"{bytes.fromhex('76345f323a41324d433a').decode('ascii')}{panel}:{target.uid}:strong".encode()).digest()[:8],"big"))
            refs[i]=rng.choice(len(item_ids),size=SAMPLE_COUNT,replace=True,p=np.exp(lp[i]))
        sample_counts[panel]["strong_gpt"]=refs
        # Existing actor checkpoint distributions are sampled directly over the full catalog.
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for method,checkpoint in ACTORS.items():
            actor,_=load_actor_checkpoint(checkpoint,vocab_size=len(item_ids)+3,n_positions=256,n_layer=4,n_embd=256,n_head=4,device=device)
            actor.eval(); samples=np.empty((n,SAMPLE_COUNT),dtype=np.int32)
            for start in range(0,n,16):
                rows=targets[start:start+16];histories=[history_for_target(store,t,200) for t in rows]
                ids,attention=_collate_histories(histories,device)
                with torch.inference_mode():logp=actor.next_action_log_probs(ids,attention)[:,3:].float().cpu().numpy().astype(np.float64)
                logp-=np.logaddexp.reduce(logp,axis=1,keepdims=True)
                for local,target in enumerate(rows):
                    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(f"{bytes.fromhex('76345f323a41324d433a').decode('ascii')}{panel}:{target.uid}:{method}".encode()).digest()[:8],"big"))
                    samples[start+local]=rng.choice(len(item_ids),size=SAMPLE_COUNT,replace=True,p=np.exp(logp[local]))
            del actor
            if torch.cuda.is_available():torch.cuda.empty_cache()
            sample_counts[panel][method]=samples
        # Deduplicate action identities across all sampled policies for each context.
        unions={int(t.uid):set() for t in targets}
        for method,samples in sample_counts[panel].items():
            for i,target in enumerate(targets):unions[int(target.uid)].update(samples[i].tolist())
        for target in targets:
            uid=int(target.uid);indices=np.asarray(sorted(unions[uid]),dtype=np.int64);user=store.users[target.user_index]
            context=history_context_features(user,target.position)
            cand=candidate_features(user,target.position,item_ids[indices],catalog_indices=indices,priors=priors)
            rm_x=np.column_stack((np.repeat(context[None,:],len(indices),axis=0),cand)).astype(np.float32)
            proxy=np.clip(models["proxy"].predict(rm_x),0,1);audit=np.clip(models["audit"].predict(rm_x),0,1)
            repeat=np.asarray(cand[:,__import__("tlab.residual_support.features",fromlist=["CANDIDATE_FEATURE_NAMES"]).CANDIDATE_FEATURE_NAMES.index("personal_repeat_count_1000")]>0)
            count=support_count[indices];support=np.where(count==0,"unsupported",np.where(count<10,"rare_1_9",np.where(count<100,"rare_10_99","supported_100_plus")))
            action_pred[panel][uid]={int(idx):(float(r),float(a),bool(rep),str(sup)) for idx,r,a,rep,sup in zip(indices,proxy,audit,repeat,support)}

    # Per-user Monte Carlo expectation for each full-catalog policy; then macro-average users.
    for panel,methods in sample_counts.items():
        n=len(panel_targets[panel])
        user_result={name:{"proxy":np.empty(n),"audit":np.empty(n),"proxy_se":np.empty(n),"audit_se":np.empty(n),"group":{}} for name in methods}
        # panel_targets are inserted in cache order, so uid->row indices must retain that order.
        uid_to_row={int(t.uid):i for i,t in enumerate(panel_caches[panel][2])}
        for uid,ri in uid_to_row.items():
            for method,samples in methods.items():
                draw=samples[ri];look=[action_pred[panel][uid][int(idx)] for idx in draw]
                r=np.asarray([v[0] for v in look]);a=np.asarray([v[1] for v in look])
                user_result[method]["proxy"][ri]=r.mean();user_result[method]["audit"][ri]=a.mean()
                user_result[method]["proxy_se"][ri]=r.std(ddof=1)/np.sqrt(SAMPLE_COUNT)
                user_result[method]["audit_se"][ri]=a.std(ddof=1)/np.sqrt(SAMPLE_COUNT)
                for novelty,novel_mask in (("new",np.asarray([not v[2] for v in look])),("repeat",np.asarray([v[2] for v in look]))):
                    for support_name in ("unsupported","rare_1_9","rare_10_99","supported_100_plus"):
                        mask=novel_mask&np.asarray([v[3]==support_name for v in look])
                        key=f"{novelty}__{support_name}"
                        agg=user_result[method]["group"].setdefault(key,{"counts":[],"proxy":[],"audit":[]})
                        if mask.any():agg["counts"].append(float(mask.mean()));agg["proxy"].append(float(r[mask].mean()));agg["audit"].append(float(a[mask].mean()))
        for method,d in user_result.items():
            for metric in ("proxy","audit"):
                vals=d[metric];mc_se=float(np.sqrt(np.mean(d[metric+"_se"]**2)/n));avg=float(vals.mean());se=float(vals.std(ddof=1)/np.sqrt(n))
                summary_rows.append({"panel":panel,"method":method,"metric":f"full_catalog_E_{metric}_raw_ratio_MC",
                                    "n_users":n,"samples_per_user":SAMPLE_COUNT,"mean_over_users":avg,
                                    "user_sampling_approx_95_ci_low":avg-1.96*se,"user_sampling_approx_95_ci_high":avg+1.96*se,
                                    "monte_carlo_se_of_macro_mean":mc_se,"policy_reference":"actor-specific full-catalog q",
                                    "tail_cancellation_used":False,"audit_is_diagnostic_only":True})
                if metric=="proxy":
                    zvals=(vals-mean)/std;summary_rows.append({"panel":panel,"method":method,"metric":"full_catalog_E_proxy_z_MC",
                        "n_users":n,"samples_per_user":SAMPLE_COUNT,"mean_over_users":float(zvals.mean()),
                        "monte_carlo_se_of_macro_mean":mc_se/std,"policy_reference":"actor-specific full-catalog q",
                        "tail_cancellation_used":False,"audit_is_diagnostic_only":True})
            for group,data in d["group"].items():
                slice_rows.append({"panel":panel,"method":method,"slice":group,"n_users_with_sampled_actions":len(data["counts"]),
                                   "mean_sampled_policy_mass_share":float(np.mean(data["counts"])) if data["counts"] else None,
                                   "mean_proxy_raw_given_sampled_slice":float(np.mean(data["proxy"])) if data["proxy"] else None,
                                   "mean_audit_raw_given_sampled_slice":float(np.mean(data["audit"])) if data["audit"] else None,
                                   "samples_per_user":SAMPLE_COUNT,"values_are_MC_estimates":True})
    out=ARTIFACTS/"evaluation";out.mkdir(parents=True,exist_ok=True)
    def write(path:Path,rows:list[dict[str,Any]])->None:
        fields=list(dict.fromkeys(k for r in rows for k in r))
        with path.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    write(out/"policy_audit_actor_full_catalog_mc.csv",summary_rows);write(out/"policy_audit_actor_mc_slices.csv",slice_rows)
    result={"runtime_seconds":round(time.perf_counter()-started,3),"samples_per_user_per_policy":SAMPLE_COUNT,
            "policies":["strong_gpt","dense_sft","last_target_sft","weak_actor"],"panels":{p:len(panel_targets[p]) for p in sample_counts},
            "expectations":"full 20k-catalog Monte Carlo action expectations; no fixed-tail cancellation for dense/last-target/weak",
            "metric_uncertainty":"reports user-level normal intervals and Monte Carlo standard error separately",
            "audit_role":"diagnostic only, no selection"}
    (out/"policy_audit_actor_mc_summary.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    record_operation("A2_full_catalog_actor_policy_MC_audit",result["runtime_seconds"],bucket="A",category="preprocessing_and_evaluation",
                     evidence=["artifacts/policy_diagnostics/evaluation/policy_audit_actor_full_catalog_mc.csv","artifacts/policy_diagnostics/evaluation/policy_audit_actor_mc_slices.csv"])
    update_stage("A","complete",["artifacts/policy_diagnostics/evaluation/legacy_completion.csv","artifacts/policy_diagnostics/evaluation/policy_audit_slices.csv","artifacts/policy_diagnostics/evaluation/policy_audit_top10.csv","artifacts/policy_diagnostics/evaluation/policy_audit_actor_full_catalog_mc.csv"],
                 note="A1-A3 complete. Head policies use exact fixed-support expectations; actor baselines use full-catalog action Monte Carlo plus exact observed top-10 list audits; external audit remained diagnostic-only.")
    return result


if __name__=="__main__":print(json.dumps(run_actor_mc(),indent=2))
