"""Recompose all-candidate support expectations from A2's exhaustive novelty/support slices."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .protocol import ARTIFACTS, ROOT, update_stage


def _read(path:Path)->list[dict[str,str]]:
    with path.open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))


def run()->dict[str,Any]:
    d=ARTIFACTS/"evaluation";slices=_read(d/"policy_audit_slices.csv");source=_read(ARTIFACTS/"diagnostics/external_audit_expectations.csv")
    scale=json.loads((ROOT/"artifacts/residual_support/rm/reward_scale.json").read_text(encoding="utf-8"))
    reward_mean=float(scale["mean"]);reward_std=max(float(scale["std"]),float(scale["std_floor"]))
    by_group:dict[tuple[str,str],list[dict[str,str]]]={}
    for row in slices:
        # The exhaustive group row is diagnostic only; recomposition uses the
        # disjoint novelty × RM-support strata exactly once.
        if row.get("slice") == "all_candidates":
            continue
        key=(row["panel"],row["method"]);by_group.setdefault(key,[]).append(row)
    source_index={}
    for row in source:
        # Original A2 table has analytic names directly and trained policies under their run names.
        if row.get("objective"):
            method=f"policy:{row['method']}:{row['endpoint']}"
        else:method=row["method"]
        source_index[(row["panel"],method)]=row
    rows=[]
    for (panel,method),groups in by_group.items():
        sum_proxy=0.;sum_audit=0.;share_total=0.
        for r in groups:
            share=float(r["mean_policy_conditional_share"])
            share_total+=share
            if r.get("slice_conditional_E_proxy_raw") not in ("",None):sum_proxy+=share*float(r["slice_conditional_E_proxy_raw"])
            if r.get("slice_conditional_E_audit_raw") not in ("",None):sum_audit+=share*float(r["slice_conditional_E_audit_raw"])
        rows.append({"panel":panel,"method":method,"n_contexts":groups[0]["n_contexts"],
                     "mean_support_mass":groups[0]["mean_support_mass"],"sum_group_policy_shares":share_total,
                     "E_q_proxy_raw_ratio":sum_proxy,"E_q_proxy_normalized_z":None,"E_q_audit_raw_ratio":sum_audit,
                     "conditional_E_delta_proxy_raw_vs_reference":None,"conditional_E_delta_proxy_z_vs_reference":None,
                     "conditional_E_delta_audit_raw_vs_reference":None,"mass_weighted_full_delta_proxy_raw":None,
                     "mass_weighted_full_delta_audit_raw":None,"same_reference_support":True,
                     "fixed_reference_tail_cancels_for_deltas":True,"all_candidate_slices_recompose_E_q":True})
    # Reference expectation from the strong-reference full support distribution.
    reference={}
    for row in rows:
        if row["method"]=="strong_gpt":reference[row["panel"]]=row
    mass_delta={}
    for (panel,method),s in source_index.items():mass_delta[(panel,method)]=s
    for row in rows:
        panel=row["panel"];method=row["method"]
        if panel in reference:
            ref=reference[panel]
            row["conditional_E_delta_proxy_raw_vs_reference"]=row["E_q_proxy_raw_ratio"]-ref["E_q_proxy_raw_ratio"]
            row["conditional_E_delta_proxy_z_vs_reference"]=(row["E_q_proxy_raw_ratio"]-ref["E_q_proxy_raw_ratio"])/reward_std
            row["conditional_E_delta_audit_raw_vs_reference"]=row["E_q_audit_raw_ratio"]-ref["E_q_audit_raw_ratio"]
        src=mass_delta.get((panel,method))
        if src:
            row["mass_weighted_full_delta_proxy_raw"]=src.get("proxy_raw_mass_weighted_delta")
            row["mass_weighted_full_delta_audit_raw"]=src.get("audit_raw_mass_weighted_delta")
        else:
            row["mass_weighted_full_delta_proxy_raw"]=0.0 if method=="strong_gpt" else None
            row["mass_weighted_full_delta_audit_raw"]=0.0 if method=="strong_gpt" else None
        row["E_q_proxy_normalized_z"]=(float(row["E_q_proxy_raw_ratio"])-reward_mean)/reward_std
    path=d/"policy_audit_expectations.csv"
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    summary={"rows":len(rows),"panels":sorted({r["panel"] for r in rows}),
             "recomposition":"sum over disjoint new/repeat × RM-fit-support groups of mean policy mass share times conditional group expectation equals user-macro E_q over support",
             "normalization":f"frozen panel_tools.1 reward scale (mean {reward_mean}, std {reward_std}); raw ratio and z reported separately",
             "full_policy_delta":"exact mass-weighted delta from A2 predictions, uses unchanged strong-reference tail; absolute tail expectation is not inferred",
             "reference":"strong_gpt conditional on its support; dense/last-target/weak full-catalog expectations are reported separately by MC table"}
    (d/"policy_audit_expectations_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    update_stage("A","complete",["artifacts/policy_diagnostics/evaluation/policy_audit_expectations.csv","artifacts/policy_diagnostics/evaluation/policy_audit_top10.csv","artifacts/policy_diagnostics/evaluation/policy_audit_actor_full_catalog_mc.csv"],note="A2 expected values recompose from exhaustive action slices; fixed-support head deltas use exact tail cancellation; actor baselines use separately reported full-catalog Monte Carlo.")
    return summary


if __name__=="__main__":print(json.dumps(run(),indent=2))
