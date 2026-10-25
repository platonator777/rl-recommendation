"""User-disjoint, temporally prefix-safe RM representability/affinity experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tlab.data.store import SequenceStore
from tlab.panel_tools.statistics import paired_user_bootstrap
from tlab.residual_support.features import CANDIDATE_FEATURE_NAMES, CONTEXT_FEATURE_NAMES, RM_FEATURE_NAMES
from tlab.residual_support.rm import (_group_event_timestamps, _rm_rows, _sample_positions,
                           build_item_priors)
from .protocol import ARTIFACTS, ROOT, record_operation, update_stage


def _sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()


def _hash_bucket(uid: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{bytes.fromhex('76345f323a433a726d6465763a').decode('ascii')}{uid}".encode()).digest()[:8],"big")%5


def _prefix_item_embedding(listens: Path, catalog_size: int, cutoff: float, output: Path) -> tuple[np.ndarray,dict[str,Any],float]:
    t=time.perf_counter()
    # Only known, unambiguous F_RM events strictly before every C fit/dev target enter affinity geometry.
    grouped=(pl.scan_parquet(listens).filter((pl.col("global_split")=="train") &
              (pl.col("train_subsplit")=="F_RM") & (pl.col("timestamp")<cutoff) &
              (~pl.col("ambiguous_timestamp_block")) & (pl.col("token_id")>=3))
            .select("uid","timestamp","token_id").sort(["uid","timestamp"])
            .group_by("uid",maintain_order=True).agg(pl.col("token_id").alias("tokens"))
            .collect(engine="streaming"))
    blocks=[]; rows=[]; cols=[]; vals=[]; event_count=0; pair_count=0
    def flush()->None:
        if not rows:return
        rr=np.concatenate(rows);cc=np.concatenate(cols);vv=np.concatenate(vals)
        block=coo_matrix((vv,(rr,cc)),shape=(catalog_size,catalog_size),dtype=np.float32).tocsr()
        block.sum_duplicates();blocks.append(block)
        rows.clear();cols.clear();vals.clear()
    for user_no,(_, seq) in enumerate(grouped.iter_rows()):
        x=np.asarray(seq,dtype=np.int32)-3
        x=x[(x>=0)&(x<catalog_size)]
        event_count+=len(x)
        for distance in range(1,6):
            if len(x)<=distance: continue
            a=x[:-distance]; b=x[distance:]
            rows.extend((a,b)); cols.extend((b,a))
            w=1.0/distance
            vals.extend((np.full(len(a),w,dtype=np.float32),np.full(len(b),w,dtype=np.float32)))
            pair_count+=2*len(a)
        if (user_no+1)%4096==0:flush()
    flush()
    if not blocks:
        raise RuntimeError("prefix-safe affinity corpus is empty")
    matrix=blocks[0]
    for block in blocks[1:]:matrix=matrix+block
    matrix.sum_duplicates(); matrix.eliminate_zeros()
    row_sum=np.asarray(matrix.sum(axis=1)).ravel(); col_sum=np.asarray(matrix.sum(axis=0)).ravel(); total=float(matrix.sum())
    coo=matrix.tocoo()
    expected=np.maximum(row_sum[coo.row]*col_sum[coo.col]/max(total,1e-12),1e-12)
    pmi=np.log(np.maximum(coo.data,1e-12)/expected)
    coo.data=np.maximum(pmi,0).astype(np.float32)
    ppmi=coo.tocsr(); ppmi.eliminate_zeros()
    svd=TruncatedSVD(n_components=64,n_iter=5,random_state=17)
    embedding=svd.fit_transform(ppmi).astype(np.float32)
    norm=np.linalg.norm(embedding,axis=1,keepdims=True)
    embedding/=np.maximum(norm,1e-8)
    np.save(output,embedding)
    meta={"events_used":event_count,"directed_weighted_pairs_before_coalescing":pair_count,
          "sparse_nonzeros_after_ppmi":int(ppmi.nnz),"components":64,"explained_variance_ratio_sum":float(svd.explained_variance_ratio_.sum()),
          "embedding_sha256":_sha(output),"source":"F_RM known, non-ambiguous item sequence strictly before minimum C target timestamp",
          "cutoff_exclusive_raw":cutoff,"input_sha256":_sha(listens)}
    return embedding,meta,time.perf_counter()-t


def _affinity_rows(store:SequenceStore, selected:list[tuple[int,int]], item_index:np.ndarray,
                   embeddings:np.ndarray, *, shuffle_seed:int|None=None)->np.ndarray:
    result=np.zeros((len(selected),9),dtype=np.float32)
    rng=np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None
    norms=np.maximum(np.linalg.norm(embeddings,axis=1),1e-8)
    for j,(ui,pos) in enumerate(selected):
        user=store.users[ui]
        start=max(0,pos-1000)
        ids=np.asarray(user.tokens[start:pos],dtype=np.int64)
        ratios=np.asarray(user.ratio_raw[start:pos],dtype=np.float32)
        valid=(ids>=3)&(ids<embeddings.shape[0]+3)
        idx=ids[valid]-3; vals=ratios[valid]
        if rng is not None and len(idx)>1:
            order=rng.permutation(len(idx)); idx=idx[order]; vals=vals[order]
        chosen=int(item_index[j])
        if chosen<0 or chosen>=embeddings.shape[0] or np.linalg.norm(embeddings[chosen])<1e-8:
            result[j,6]=1.0
        vec=embeddings[chosen] if 0<=chosen<embeddings.shape[0] else np.zeros(embeddings.shape[1])
        # last and ordered final-20 are dynamic, but embedding itself is globally prefix-cutoff-safe.
        for slot,hist in ((0,idx[-1:]),(1,idx[-20:])):
            if len(hist):
                mean=embeddings[hist].mean(axis=0)
                denom=max(float(np.linalg.norm(mean))*float(norms[chosen]),1e-8)
                result[j,slot]=float(np.dot(vec,mean)/denom)
            else:result[j,4+slot]=1.0
        for slot,mask in ((2,vals>=.5),(3,(vals<.5)&np.isfinite(vals))):
            hist=idx[mask]
            if len(hist):
                mean=embeddings[hist].mean(axis=0)
                denom=max(float(np.linalg.norm(mean))*float(norms[chosen]),1e-8)
                result[j,slot]=float(np.dot(vec,mean)/denom)
        result[j,6]=float(not np.any(vals>=.5))
        result[j,7]=float(not np.any((vals<.5)&np.isfinite(vals)))
        result[j,8]=float(np.linalg.norm(embeddings[chosen])<1e-8)
    return result


def _metric_rows(y:np.ndarray,pred:np.ndarray,uids:np.ndarray,groups:dict[str,np.ndarray])->list[dict[str,Any]]:
    rows=[]
    for model,x in pred.items():
        for group,mask in {"all":np.ones(len(y),bool),**groups}.items():
            if int(mask.sum())==0: continue
            yy=y[mask]; pp=x[mask]
            by={}
            for uid,err in zip(uids[mask],np.abs(yy-pp)):
                by.setdefault(int(uid),[]).append(float(err))
            per=np.asarray([np.mean(v) for v in by.values()])
            user_rmse=[]
            for uid in np.unique(uids[mask]):
                own=(uids[mask]==uid)
                user_rmse.append(float(np.sqrt(np.mean((yy[own]-pp[own])**2))))
            rows.append({"model":model,"slice":group,"n_rows":int(mask.sum()),"n_users":len(by),
                         "mae":float(np.mean(np.abs(yy-pp))),"rmse":float(np.sqrt(np.mean((yy-pp)**2))),
                         "user_macro_mae":float(per.mean()),"user_macro_rmse":float(np.mean(user_rmse))})
    return rows


def run_c_rm() -> dict[str,Any]:
    overall=time.perf_counter(); processed=ROOT/"data/processed_full_model_tminus14"; listens=processed/"listens.parquet"
    protocol=json.loads((processed/"protocol.json").read_text(encoding="utf-8")); boundaries=protocol["boundaries_source_units"]
    store=SequenceStore.load_cache(processed/"sequence_index",expected_source_sha256=protocol["processed_listens_sha256"])
    catalog=pl.read_parquet(processed/"catalog.parquet").sort("token_id")
    item_ids=np.asarray(catalog["item_id"].to_list(),dtype=np.int64); n_items=len(item_ids)
    raw_to_index={int(x):i for i,x in enumerate(item_ids)}
    timestamps_started=time.perf_counter(); timestamps=_group_event_timestamps(listens,store)
    f_start=min(float(t[0]) for t in timestamps.values() if len(t)); f_end=float(boundaries["rm_validation_start_raw"])
    target_start=f_start+.85*(f_end-f_start)
    selected=_sample_positions(store,timestamps,partition="fit",positions_name="rm_fit_positions",
                               lower=target_start,upper=f_end,max_events=5,seed=17)
    # Match _rm_rows' valid-target filter before assigning user splits and feature rows.
    selected=[(ui,pos) for ui,pos in selected
              if int(store.users[ui].raw_item_ids[pos]) in raw_to_index
              and np.isfinite(float(store.users[ui].ratio_raw[pos]))]
    if not selected: raise RuntimeError("C RM fit pool is empty")
    train_selected=[p for p in selected if _hash_bucket(int(store.users[p[0]].uid))!=0]
    dev_selected=[p for p in selected if _hash_bucket(int(store.users[p[0]].uid))==0]
    if not train_selected or not dev_selected: raise RuntimeError("hashed C RM fit/dev split is empty")
    min_target=min(float(timestamps[int(store.users[ui].uid)][pos]) for ui,pos in selected)
    rm_dir=ARTIFACTS/"rm"; rm_dir.mkdir(parents=True,exist_ok=True)
    timestamp_seconds=time.perf_counter()-timestamps_started
    record_operation("C_timestamp_index_and_user_hash_split",timestamp_seconds,bucket="C",category="preprocessing",
                     evidence=["artifacts/policy_diagnostics/rm/C_split_manifest.json"])

    prep=time.perf_counter()
    priors=build_item_priors(listens,catalog,cutoff_exclusive=min_target,source_split="F_RM_strictly_before_all_C_targets")
    fit_all=_rm_rows(store,selected,priors,raw_to_index)
    split_train=np.asarray([_hash_bucket(int(uid))!=0 for uid in fit_all["uid"]],bool)
    Xtr=fit_all["x"][split_train]; ytr=fit_all["y"][split_train]; uidtr=fit_all["uid"][split_train]
    Xdev=fit_all["x"][~split_train]; y=fit_all["y"][~split_train]; uids=fit_all["uid"][~split_train]
    reptr=fit_all["is_repeat"][split_train]; repdev=fit_all["is_repeat"][~split_train]
    idxtr=fit_all["item_index"][split_train]; idxdev=fit_all["item_index"][~split_train]
    fit_counts=np.bincount(idxtr,minlength=n_items)
    train_support=np.where(fit_counts[idxdev]==0,"unsupported",np.where(fit_counts[idxdev]<10,"rare_1_9",np.where(fit_counts[idxdev]<100,"rare_10_99","supported_100_plus")))
    embedding_path=rm_dir/"prefix_affinity_embeddings.npy"
    embedding,embedding_meta,svd_seconds=_prefix_item_embedding(listens,n_items,min_target,embedding_path)
    train_pos=[selected[i] for i in np.where(split_train)[0]]; dev_pos=[selected[i] for i in np.where(~split_train)[0]]
    afftr=_affinity_rows(store,train_pos,fit_all["item_index"][split_train],embedding)
    affdev=_affinity_rows(store,dev_pos,fit_all["item_index"][~split_train],embedding)
    prep_seconds=time.perf_counter()-prep
    record_operation("C_temporal_priors_affinity_embeddings_and_feature_preprocessing",prep_seconds,bucket="C",category="preprocessing",
                     evidence=["artifacts/policy_diagnostics/rm/prefix_affinity_embeddings.npy"])

    # One factor is added to the refit panel_tools.1 optimized control: the single, prefix-safe affinity feature block.
    fit_started=time.perf_counter()
    opt=HistGradientBoostingRegressor(learning_rate=.05,max_iter=220,max_leaf_nodes=23,min_samples_leaf=35,
                                    l2_regularization=1.0,early_stopping=False,random_state=19).fit(Xtr,ytr)
    hist=HistGradientBoostingRegressor(learning_rate=.05,max_iter=160,max_leaf_nodes=15,min_samples_leaf=40,
                                     l2_regularization=1.0,early_stopping=False,random_state=17).fit(Xtr[:,:14],ytr)
    item_model=HistGradientBoostingRegressor(learning_rate=.05,max_iter=160,max_leaf_nodes=15,min_samples_leaf=40,
                                            l2_regularization=1.0,early_stopping=False,random_state=18).fit(Xtr[:,29:35],ytr)
    addcols=np.r_[np.arange(14),np.arange(29,35)]
    additive=make_pipeline(StandardScaler(),Ridge(alpha=10.0)).fit(Xtr[:,addcols],ytr)
    affinity=HistGradientBoostingRegressor(learning_rate=.05,max_iter=220,max_leaf_nodes=23,min_samples_leaf=35,
                                          l2_regularization=1.0,early_stopping=False,random_state=19).fit(np.column_stack((Xtr,afftr)),ytr)
    fit_seconds=time.perf_counter()-fit_started
    models={"history_only":(hist,Xdev[:,:14]),"item_only":(item_model,Xdev[:,29:35]),
            "strict_additive_linear":(additive,Xdev[:,addcols]),"refit_residual_support_optimized_control":(opt,Xdev),
            "prefix_safe_affinity_extension":(affinity,np.column_stack((Xdev,affdev)))}
    pred={name:np.clip(model.predict(x),0,1) for name,(model,x) in models.items()}
    eval_started=time.perf_counter()
    groups={"personal_new":~repdev,"personal_repeat":repdev}
    for name,label in (("unsupported", "rm_support_unsupported"),("rare_1_9","rm_support_rare_1_9"),
                       ("rare_10_99","rm_support_rare_10_99"),("supported_100_plus","rm_support_100_plus")):
        groups[label]=train_support==name
    metric_rows=_metric_rows(y,pred,uids,groups)
    per_user:dict[str,dict[int,list[float]]]={};per_user_sq:dict[str,dict[int,list[float]]]={}
    for name,p in pred.items():
        d={};sq={}
        for uid,err in zip(uids,y-p):
            d.setdefault(int(uid),[]).append(float(abs(err)));sq.setdefault(int(uid),[]).append(float(err*err))
        per_user[name]={uid:float(np.mean(v)) for uid,v in d.items()}
        per_user_sq[name]={uid:float(np.sqrt(np.mean(v))) for uid,v in sq.items()}
    contrast_rows=[]; base=per_user["refit_residual_support_optimized_control"]
    for model_name in pred:
        if model_name=="refit_residual_support_optimized_control": continue
        for metric,maps in (("per_user_MAE_improvement",per_user),("per_user_RMSE_improvement",per_user_sq)):
            control=maps["refit_residual_support_optimized_control"];candidate=maps[model_name]
            interval=paired_user_bootstrap(control,candidate,n_bootstrap=2000,seed=32000+len(contrast_rows))
            contrast_rows.append({"contrast":f"refit_residual_support_optimized_control - {model_name}","metric":metric,
                                  "positive_favors_candidate":True,"n_users":len(control),"mean_difference":float(np.mean([control[u]-candidate[u] for u in control])),
                                  "ci_lower":float(interval["ci_lower"]),"ci_upper":float(interval["ci_upper"]),"bootstrap_draws":2000})
    # Calibration deciles of the target-relevant RM predictions, user holdout only.
    calibration=[]
    for name,p in pred.items():
        edges=np.unique(np.quantile(p,np.linspace(0,1,11)))
        for j,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
            mask=(p>=lo)&((p<=hi) if j==len(edges)-2 else (p<hi))
            if mask.any(): calibration.append({"model":name,"decile":j+1,"n":int(mask.sum()),"mean_prediction":float(p[mask].mean()),"mean_target":float(y[mask].mean())})
    # Sensitivity: permute paired historical events inside heldout histories, keeping row/item/label fixed.
    aff_shuffle=_affinity_rows(store,dev_pos,fit_all["item_index"][~split_train],embedding,shuffle_seed=42017)
    shuffled_pred=np.clip(affinity.predict(np.column_stack((Xdev,aff_shuffle))),0,1)
    shuffle_rows=[]
    for name,vec in (("original",pred["prefix_safe_affinity_extension"]),("history_event_order_shuffled",shuffled_pred)):
        shuffle_rows.append({"condition":name,"n_rows":len(y),"rmse":float(np.sqrt(np.mean((y-vec)**2))),"mae":float(np.mean(np.abs(y-vec))),
                             "mean_abs_prediction_change_vs_original":0.0 if name=="original" else float(np.mean(np.abs(vec-pred["prefix_safe_affinity_extension"])))})
    eval_seconds=time.perf_counter()-eval_started

    # Persist isolated C artifacts and source provenance.
    def write_csv(path:Path,rows:list[dict[str,Any]])->None:
        if not rows:return
        fields=list(dict.fromkeys(k for row in rows for k in row))
        with path.open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    split_manifest={"schema_version":1,"role":"retrospective hashed user-disjoint split inside historical F_RM fit user pool; not independent test",
        "seed":17,"hash":"SHA256(hex-prefix 76345f323a433a726d6465763a + decimal uid) mod 5; bucket 0 held out; outcome-blind",
        "selected_target_rows":len(selected),"train_rows":int(split_train.sum()),"rmdev_rows":int((~split_train).sum()),
        "train_users":int(len(set(uidtr.tolist()))),"rmdev_users":int(len(set(uids.tolist()))),"fit_pool_user_partition":"existing panel_tools.1 fit partition only",
        "target_time_start":float(min_target),"target_time_end_exclusive":f_end,"feature_prior_cutoff_exclusive":float(min_target),
        "target_ratio_used_in_priors":False,"affinity_embedding_cutoff_exclusive":float(min_target),"affinity_feature_names":["cosine_last","cosine_mean_last20","cosine_positive_history","cosine_negative_history","last_history_missing","history20_missing","positive_history_missing","negative_history_missing","candidate_embedding_missing"],
        "target_id_or_ratio_in_affinity":False,"model_comparison":"all controls fit on same 80% users and same sampled target rows; only affinity extension adds 9 prefix-derived columns to refit optimized RM",
        "source_sha256":protocol["processed_listens_sha256"],"embedding":embedding_meta,
        "old_residual_support_control":"refit exact optimized-HGB feature/parameter architecture on this split; avoids leakage from the historical all-fit model",
        "historical_panel_tools_rm_compatibility":"not directly included as its input/target split and fitting universe do not match this user-hash split; same-split refit panel_tools.1 control is primary"}
    (rm_dir/"C_split_manifest.json").write_text(json.dumps(split_manifest,indent=2)+"\n",encoding="utf-8")
    write_csv(rm_dir/"C_metrics.csv",metric_rows);write_csv(rm_dir/"C_paired_contrasts.csv",contrast_rows)
    write_csv(rm_dir/"C_calibration.csv",calibration);write_csv(rm_dir/"C_shuffle_history_sensitivity.csv",shuffle_rows)
    with (rm_dir/"C_rmdev_predictions.csv").open("w",newline="",encoding="utf-8") as f:
        fields=["uid","label_ratio","is_repeat","fit_support","prediction_model","prediction"]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for i,uid in enumerate(uids):
            for name,p in pred.items(): w.writerow({"uid":int(uid),"label_ratio":float(y[i]),"is_repeat":bool(repdev[i]),"fit_support":str(train_support[i]),"prediction_model":name,"prediction":float(p[i])})
    for name,(model,_) in models.items(): joblib.dump(model,rm_dir/f"C_{name}.joblib",compress=3)
    time_used=time.perf_counter()-overall
    summary={"status":"complete","runtime_seconds":round(time_used,3),"timestamp_index_and_split_seconds":round(timestamp_seconds,3),"preprocessing_seconds":round(prep_seconds,3),
             "fit_seconds":round(fit_seconds,3),"evaluation_seconds":round(eval_seconds,3),"selected_rows":len(selected),
             "train_rows":int(split_train.sum()),"rmdev_rows":int((~split_train).sum()),"train_users":len(set(uidtr.tolist())),"rmdev_users":len(set(uids.tolist())),
             "embedding_events":embedding_meta["events_used"],"embedding_pairs":embedding_meta["directed_weighted_pairs_before_coalescing"],
             "rmdev_primary_metric":"user-macro MAE, paired user bootstrap 95% CI; RMSE and calibration separately", "rows_metrics":len(metric_rows)}
    (rm_dir/"C_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    record_operation("C_RM_control_affinity_model_fit",fit_seconds,bucket="C",category="model_fit",evidence=["artifacts/policy_diagnostics/rm/C_metrics.csv"])
    record_operation("C_user_disjoint_RMdev_predictions_metrics_bootstrap_and_shuffle",eval_seconds,bucket="C",category="evaluation",evidence=["artifacts/policy_diagnostics/rm/C_paired_contrasts.csv","artifacts/policy_diagnostics/rm/C_shuffle_history_sensitivity.csv"])
    update_stage("C","complete",["artifacts/policy_diagnostics/rm/C_split_manifest.json","artifacts/policy_diagnostics/rm/C_metrics.csv","artifacts/policy_diagnostics/rm/C_paired_contrasts.csv","artifacts/policy_diagnostics/rm/C_shuffle_history_sensitivity.csv"])
    return summary


if __name__=="__main__": print(json.dumps(run_c_rm(),indent=2))
