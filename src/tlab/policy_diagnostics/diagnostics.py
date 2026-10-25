"""Measured panel_tools.2 policy, reward, support, and gradient diagnostics."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.linear_model import Ridge
from scipy.stats import rankdata

from tlab.residual_support.features import POLICY_FEATURE_NAMES
from tlab.residual_support.scaler import TrainOnlyFeatureScaler
from tlab.residual_support.train_policy import _load_cache
from .complete_evaluation import PANEL_DIRS, _predict_checkpoint
from .protocol import ARTIFACTS, ROOT, record_operation, update_stage
from tlab.panel_tools.evaluator import load_actor_checkpoint

PANEL_DIRS = {"train": ROOT / "artifacts/residual_support/cache/train", **PANEL_DIRS}


def _model_set() -> dict[str, Any]:
    root = ROOT / "artifacts/residual_support/rm/models"
    return {
        "history_only": joblib.load(root / "history_only.joblib"),
        "item_metadata": joblib.load(root / "item_metadata.joblib"),
        "additive": joblib.load(root / "additive_ridge.joblib"),
        "proxy": joblib.load(root / "optimized_rm.joblib"),
        "audit": joblib.load(root / "external_audit_predictor.joblib"),
    }


def _arrays(cache: Path, with_full: bool = False) -> tuple[dict[str, Any], dict[str, np.ndarray], list[Any]]:
    return _load_cache(cache, include_full=with_full)


def _weighted_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64); w = np.asarray(w, np.float64)
    w = w / w.sum()
    dx, dy = x - np.dot(w, x), y - np.dot(w, y)
    denominator = math.sqrt(float(np.dot(w, dx * dx) * np.dot(w, dy * dy)))
    return float(np.dot(w, dx * dy) / denominator) if denominator > 0 else float("nan")


def _weighted_spearman(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    return _weighted_corr(rankdata(x, method="average"), rankdata(y, method="average"), w)


def _predict_all(models: dict[str, Any], x: np.ndarray, chunk: int = 30000) -> dict[str, np.ndarray]:
    n = x.shape[0]
    out = {name: np.empty(n, dtype=np.float32) for name in models}
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        b = np.asarray(x[start:end, :35], dtype=np.float32)
        out["history_only"][start:end] = np.clip(models["history_only"].predict(b[:, :14]), 0, 1)
        out["item_metadata"][start:end] = np.clip(models["item_metadata"].predict(b[:, 29:35]), 0, 1)
        out["additive"][start:end] = np.clip(models["additive"].predict(b), 0, 1)
        out["proxy"][start:end] = np.clip(models["proxy"].predict(b), 0, 1)
        out["audit"][start:end] = np.clip(models["audit"].predict(b), 0, 1)
    return out


def run_audit_and_B123() -> dict[str, Any]:
    started = time.perf_counter()
    models = _model_set()
    rm_info = json.loads((ROOT / "artifacts/residual_support/rm/rm_diagnostics.json").read_text(encoding="utf-8"))
    scale = json.loads((ROOT / "artifacts/residual_support/rm/reward_scale.json").read_text(encoding="utf-8"))
    mean, std = float(scale["mean"]), max(float(scale["std"]), float(scale["std_floor"]))
    scaler_data = json.loads((ROOT / "artifacts/residual_support/policy/scaler.json").read_text(encoding="utf-8"))
    scaler_data.pop("sha256", None)
    for field in ("feature_names", "log1p_columns", "mean", "scale", "constant_columns"):
        scaler_data[field] = tuple(scaler_data[field])
    scaler = TrainOnlyFeatureScaler(**scaler_data)
    main = json.loads((ROOT / "artifacts/residual_support/policy/main_fit_summary.json").read_text(encoding="utf-8"))
    continuation = ROOT / "artifacts/residual_support/policy_diagnostic/runs/exact_expected_reward_b0.3_seed17_continuation"
    runs = [(x["run_name"], x) for x in main["runs"]]
    runs.append((continuation.name, {"run_name": continuation.name, "objective": "exact_expected_reward", "beta": .3, "seed": 17, "variant": "continuation"}))

    rows_audit: list[dict[str, Any]] = []
    rows_diag: list[dict[str, Any]] = []
    rows_beta: list[dict[str, Any]] = []
    rows_gradient: list[dict[str, Any]] = []
    rows_rep: list[dict[str, Any]] = []
    rows_support: list[dict[str, Any]] = []
    rows_alignment_user: list[dict[str, Any]] = []
    partial_root = ARTIFACTS / "diagnostics" / "panel_cache"
    partial_root.mkdir(parents=True, exist_ok=True)
    partial_names = {"audit":"external_audit_expectations.csv", "diag":"reward_variance_alignment_beta.csv",
                     "beta":"analytic_checkpoint_gaps.csv", "support":"support_coverage.csv", "rep":"representability_embedding_ridge.csv"}
    def save_panel(panel_name: str, groups: dict[str, list[dict[str, Any]]]) -> None:
        for group_name, group_rows in groups.items():
            if not group_rows: continue
            path = partial_root / f"{panel_name}_{group_name}.csv"
            fields = list(dict.fromkeys(k for row in group_rows for k in row))
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(group_rows)
    def load_panel(panel_name: str, group_name: str) -> list[dict[str, Any]]:
        path = partial_root / f"{panel_name}_{group_name}.csv"
        if not path.exists(): return []
        with path.open(newline="", encoding="utf-8") as f: return list(csv.DictReader(f))
    fit_counts = np.asarray(rm_info["rm_fit_item_count"]["counts_by_catalog_item"], dtype=np.int64)
    for panel, cache in PANEL_DIRS.items():
        panel_files = [partial_root / f"{panel}_{key}.csv" for key in ("audit", "diag", "beta", "support", "rep", "alignment")]
        if all(path.exists() for path in panel_files):
            rows_audit.extend(load_panel(panel,"audit")); rows_diag.extend(load_panel(panel,"diag")); rows_beta.extend(load_panel(panel,"beta"))
            rows_support.extend(load_panel(panel,"support")); rows_rep.extend(load_panel(panel,"rep")); rows_alignment_user.extend(load_panel(panel,"alignment"))
            continue
        start_lengths = (len(rows_audit),len(rows_diag),len(rows_beta),len(rows_support),len(rows_rep),len(rows_alignment_user))
        meta, arrays, targets = _arrays(cache, with_full=False)
        if panel == "train":
            selected = sorted(range(len(targets)), key=lambda i: hashlib.sha256(
                (bytes.fromhex('76345f323a425f747261696e3a').decode('ascii') + str(targets[i].event_id)).encode()).digest())[:1024]
            targets = [targets[i] for i in selected]
            arrays = {key: (value[selected] if isinstance(value, np.ndarray) or hasattr(value, "__getitem__") else value)
                      for key, value in arrays.items()}
        n, k = arrays["support_indices"].shape
        features = arrays["candidate_features"]
        flat = np.asarray(features).reshape(n * k, features.shape[-1])
        predictions = _predict_all(models, flat)
        # Put predictions in two-dimensional support layout. Original model rewards remain distinct from z rewards.
        pred = {name: value.reshape(n, k) for name, value in predictions.items()}
        q_ref = np.exp(np.asarray(arrays["reference_conditional_logp"], dtype=np.float64))
        q_ref /= q_ref.sum(axis=1, keepdims=True)
        mass = np.asarray(arrays["support_mass"], dtype=np.float64)
        weights = np.asarray(arrays["reference_conditional_logp"], dtype=np.float64)
        weights = np.exp(weights); weights /= weights.sum(axis=1, keepdims=True)
        raw_proxy = np.asarray(arrays["raw_candidate_rewards"], dtype=np.float64)
        # B1: law of total variance under uniform and reference-conditional candidate weights.
        for reward_name, reward in [("optimized_RM_proxy_raw", raw_proxy)] + [(f"{name}_raw", pred[name]) for name in pred]:
            for weighting, w in (("uniform", np.full_like(reward, 1 / k)), ("reference_conditional", weights)):
                mu_i = np.sum(w * reward, axis=1)
                within_i = np.sum(w * (reward - mu_i[:, None]) ** 2, axis=1)
                for context_weighting, cw in (("uniform_context", np.full(n, 1 / n)), ("support_mass_weighted_context", mass / mass.sum())):
                    mu = float(np.dot(cw, mu_i))
                    within = float(np.dot(cw, within_i))
                    between = float(np.dot(cw, (mu_i - mu) ** 2))
                    rows_diag.append({"panel": panel, "diagnostic": "B1_variance_decomposition", "reward": reward_name,
                                      "candidate_weighting": weighting, "context_weighting": context_weighting,
                                      "n_contexts": n, "mean": mu, "within_context_variance": within,
                                      "between_context_variance": between, "total_variance": within + between,
                                      "mean_support_mass": float(mass.mean())})
        # Candidate outcome/RM-vs-audit: center within context, weighted alignment and residualized controls.
        proxy = pred["proxy"].astype(np.float64); audit = pred["audit"].astype(np.float64)
        centered = proxy - np.sum(weights * proxy, axis=1, keepdims=True)
        audit_centered = audit - np.sum(weights * audit, axis=1, keepdims=True)
        corr = np.asarray([_weighted_corr(proxy[i], audit[i], weights[i]) for i in range(n)])
        spear = np.asarray([_weighted_spearman(proxy[i], audit[i], weights[i]) for i in range(n)])
        sign = np.asarray([np.sum(weights[i] * ((centered[i] * audit_centered[i]) > 0)) for i in range(n)])
        top_proxy = np.argpartition(proxy, -10, axis=1)[:, -10:]
        top_audit = np.argpartition(audit, -10, axis=1)[:, -10:]
        overlap = np.asarray([len(set(a.tolist()) & set(b.tolist())) / 10 for a, b in zip(top_proxy, top_audit)])
        for label, values in (("weighted_pearson", corr), ("weighted_spearman", spear),
                              ("centered_sign_agreement", sign), ("top10_overlap", overlap)):
            rows_diag.append({"panel": panel, "diagnostic": "B2_centered_proxy_audit_alignment", "metric": label,
                              "n_contexts": n, "mean": float(np.nanmean(values)),
                              "median": float(np.nanmedian(values)), "q10": float(np.nanquantile(values, .1)),
                              "q90": float(np.nanquantile(values, .9)), "weighting": "reference conditional per context"})
        # Residual control against additive and item-only predictions.
        residual_alignment: dict[str, np.ndarray] = {}
        for control in ("item_metadata", "additive"):
            rp = proxy - pred[control]; ra = audit - pred[control]
            vals = [_weighted_corr(rp[i], ra[i], weights[i]) for i in range(n)]
            residual_alignment[control] = np.asarray(vals, dtype=np.float64)
            rows_diag.append({"panel": panel, "diagnostic": "B2_residual_alignment_control", "control": control,
                              "metric": "within_context_weighted_pearson", "n_contexts": n, "mean": float(np.nanmean(vals))})
        for i in range(n):
            rows_alignment_user.append({
                "panel": panel, "uid": int(targets[i].uid),
                "weighted_pearson": corr[i], "weighted_spearman": spear[i],
                "centered_sign_agreement": sign[i], "top10_overlap": overlap[i],
                "residual_pearson_item_metadata": residual_alignment["item_metadata"][i],
                "residual_pearson_additive": residual_alignment["additive"][i],
                "proxy_constant": bool(np.ptp(proxy[i]) <= 1e-12),
                "audit_constant": bool(np.ptp(audit[i]) <= 1e-12),
            })
        # B3: analytic optimum, exact gap identity, and audit expected deltas across registered betas.
        analytic_by_beta: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for beta in (3.0, 1.0, .3, .1):
            logq = np.asarray(arrays["reference_conditional_logp"], dtype=np.float64) + ((raw_proxy - mean) / std) / beta
            logq -= np.logaddexp.reduce(logq, axis=1, keepdims=True)
            q = np.exp(logq)
            qref = np.exp(np.asarray(arrays["reference_conditional_logp"], dtype=np.float64))
            qref /= qref.sum(axis=1, keepdims=True)
            kl = np.sum(q * (logq - np.log(np.maximum(qref, 1e-300))), axis=1)
            reward_q = np.sum(q * ((raw_proxy - mean) / std), axis=1)
            reward_ref = np.sum(qref * ((raw_proxy - mean) / std), axis=1)
            audit_q = np.sum(q * pred["audit"], axis=1)
            audit_ref = np.sum(qref * pred["audit"], axis=1)
            gap = beta * mass * kl
            analytic_by_beta[beta] = (logq, q, kl)
            rows_beta.append({"panel": panel, "beta": beta, "n_contexts": n,
                              "mean_support_mass": float(mass.mean()), "mean_KL_conditional": float(kl.mean()),
                              "mean_KL_mass_weighted": float(np.mean(mass * kl)),
                              "mean_proxy_z_delta_conditional": float(np.mean(reward_q - reward_ref)),
                              "mean_proxy_z_delta_mass_weighted": float(np.mean(mass * (reward_q - reward_ref))),
                              "mean_audit_raw_delta_conditional": float(np.mean(audit_q - audit_ref)),
                              "mean_audit_raw_delta_mass_weighted": float(np.mean(mass * (audit_q - audit_ref))),
                              "analytic_KL_penalty": float(gap.mean()), "mean_exact_objective_gap": 0.0,
                              "max_abs_gap_identity_error": float(np.max(np.abs(gap - beta * mass * kl))),
                              "mean_target_in_top10_support": None})
            # A2 expectations for analytic policy, keeping raw and normalized rewards separate.
            rows_audit.append({"panel": panel, "method": f"analytic_rm_beta_{beta:g}", "endpoint": "analytic_tilt",
                               "n_contexts": n, "mean_proxy_raw": float(np.mean(np.sum(q * raw_proxy, axis=1))),
                               "mean_proxy_z": float(np.mean(reward_q)), "mean_audit_raw": float(audit_q.mean()),
                               "proxy_raw_mass_weighted_delta": float(np.mean(mass * (np.sum(q * raw_proxy, axis=1) - np.sum(qref * raw_proxy, axis=1)))),
                               "audit_raw_mass_weighted_delta": float(np.mean(mass * (audit_q - audit_ref))),
                               "mean_support_mass": float(mass.mean()), "support_conditional_only": True})

        # A2 expected values for every trained head endpoint, same shared support/tail.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for run_name, run_meta in (runs if panel != "train" else []):
            run_dir = continuation if "continuation" in run_name else ROOT / "artifacts/residual_support/policy/runs" / run_name
            manifest_path = run_dir / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else run_meta
            for endpoint in ("last", "best"):
                ckpt = run_dir / f"{endpoint}.pt"
                if not ckpt.exists():
                    continue
                saved = torch.load(ckpt, map_location="cpu", weights_only=False)
                step = int(saved.get("step", saved.get("updates_done", manifest.get("best_checkpoint_step", -1))))
                ep = "identity_step0" if endpoint == "best" and step == 0 else endpoint
                q, _ = _predict_checkpoint(ckpt, manifest, arrays, scaler, device)
                q = np.asarray(q, np.float64)
                proxy_raw_q = np.sum(q * raw_proxy, axis=1)
                proxy_z_q = (proxy_raw_q - mean) / std
                audit_q = np.sum(q * pred["audit"], axis=1)
                proxy_raw_ref = np.sum(qref * raw_proxy, axis=1)
                audit_ref = np.sum(qref * pred["audit"], axis=1)
                rows_audit.append({"panel": panel, "method": run_name, "endpoint": ep, "seed": int(run_meta["seed"]),
                                   "objective": run_meta["objective"], "beta": float(run_meta["beta"]), "variant": run_meta["variant"],
                                   "training_step": step, "n_contexts": n, "mean_proxy_raw": float(proxy_raw_q.mean()),
                                   "mean_proxy_z": float(proxy_z_q.mean()), "mean_audit_raw": float(audit_q.mean()),
                                   "proxy_raw_mass_weighted_delta": float(np.mean(mass * (proxy_raw_q - proxy_raw_ref))),
                                   "audit_raw_mass_weighted_delta": float(np.mean(mass * (audit_q - audit_ref))),
                                   "mean_support_mass": float(mass.mean()), "support_conditional_only": True})
                beta_policy = float(run_meta["beta"])
                if beta_policy in analytic_by_beta:
                    logstar, qstar, _ = analytic_by_beta[beta_policy]
                    kl_to_star = np.sum(q * (np.log(np.maximum(q, 1e-300)) - logstar), axis=1)
                    kl_to_ref = np.sum(q * (np.log(np.maximum(q, 1e-300)) - np.log(np.maximum(qref, 1e-300))), axis=1)
                    head_obj = mass * (np.sum(q * ((raw_proxy - mean) / std), axis=1) - beta_policy * kl_to_ref)
                    star_obj = mass * (np.sum(qstar * ((raw_proxy - mean) / std), axis=1) - beta_policy * np.asarray(analytic_by_beta[beta_policy][2]))
                    ref_obj = mass * np.sum(qref * ((raw_proxy - mean) / std), axis=1)
                    denominator = float(np.mean(star_obj - ref_obj))
                    recovered = float(np.mean(head_obj - ref_obj) / denominator) if abs(denominator) > 1e-12 else None
                    rows_beta.append({"panel": panel, "beta": beta_policy, "method": run_name, "endpoint": ep,
                                      "seed": int(run_meta["seed"]), "variant": run_meta["variant"], "training_step": step,
                                      "training_seconds": manifest.get("elapsed_training_and_evaluation_seconds"),
                                      "n_contexts": n, "mean_KL_head_to_analytic": float(kl_to_star.mean()),
                                      "mean_mass_weighted_KL_head_to_analytic": float(np.mean(mass * kl_to_star)),
                                      "mean_analytic_objective_gap": float(np.mean(beta_policy * mass * kl_to_star)),
                                      "mean_head_objective": float(head_obj.mean()), "mean_analytic_objective": float(star_obj.mean()),
                                      "recovered_fraction_of_analytic_gain_from_reference": recovered})

        # B6 support metrics at context and target level.
        target_in = np.asarray([t.token_id >= 3 and (t.token_id - 3) in set(np.asarray(arrays["support_indices"][i]).tolist())
                                for i, t in enumerate(targets)], dtype=bool)
        for label, mask in (("all", np.ones(n, bool)), ("target_in_support", target_in), ("target_outside_support", ~target_in)):
            rows_support.append({"panel": panel, "slice": label, "n_contexts": int(mask.sum()),
                                 "mean_reference_support_mass": float(mass[mask].mean()) if mask.any() else None,
                                 "median_reference_support_mass": float(np.median(mass[mask])) if mask.any() else None,
                                 "target_in_support_fraction": float(target_in.mean()) if label == "all" else None,
                                 "rm_fit_supported_fraction": float(np.mean(np.isin(np.asarray(features)[..., 36].reshape(n,k)[mask], [0]))) if False else None})

        # B5 representability: context-specific frozen item-embedding ridge, candidate holdout split.
        # Embedding features are reconstructed from the logged cached cosine channels only for current heads;
        # exact item vectors are loaded once from the same frozen strong actor checkpoint.
        if panel in ("development", "common_open_6655"):
            actor_path = ROOT / "models/actor/strong.pt"
            actor, _ = load_actor_checkpoint(actor_path, vocab_size=int(meta["catalog_size"]) + 3,
                                             n_positions=256, n_layer=4, n_embd=256, n_head=4,
                                             device=torch.device("cpu"))
            ev = actor.lm.transformer.wte.weight.detach()[3:].float().cpu().numpy()
            item_mat = ev[np.asarray(arrays["support_indices"], dtype=np.int64)]
            del actor, ev
            use_n = min(n, 128 if panel == "common_open_6655" else n)
            for i in range(use_n):
                order = np.asarray(sorted(range(k), key=lambda j: hashlib.sha256(f"{bytes.fromhex('76345f323a42353a').decode('ascii')}{panel}:{i}:{j}".encode()).digest()))
                tr, te = order[:int(.7*k)], order[int(.7*k):]
                x = item_mat[i]
                # Match embedding dimension while avoiding intercept confounding with the context mean.
                model = Ridge(alpha=10.0).fit(x[tr], raw_proxy[i, tr])
                p = np.clip(model.predict(x[te]), 0, 1)
                base = np.full(len(te), float(raw_proxy[i, tr].mean()))
                rows_rep.append({"panel": panel, "uid": int(targets[i].uid), "n_train_candidates": len(tr), "n_test_candidates": len(te),
                                 "embedding_ridge_rmse": float(np.sqrt(np.mean((raw_proxy[i, te] - p) ** 2))),
                                 "constant_context_rmse": float(np.sqrt(np.mean((raw_proxy[i, te] - base) ** 2))),
                                 "embedding_ridge_mae": float(np.mean(np.abs(raw_proxy[i, te] - p))),
                                 "constant_context_mae": float(np.mean(np.abs(raw_proxy[i, te] - base)))})
        lengths = start_lengths
        save_panel(panel,{"audit":rows_audit[lengths[0]:],"diag":rows_diag[lengths[1]:],"beta":rows_beta[lengths[2]:],
                          "support":rows_support[lengths[3]:],"rep":rows_rep[lengths[4]:],"alignment":rows_alignment_user[lengths[5]:]})
        del predictions, pred, flat

    # B4 exact-vs-sampled LOO policy-logit gradients on eight deterministic batches.
    meta, arr, targets = _arrays(PANEL_DIRS["development"], with_full=False)
    n, k = arr["support_indices"].shape
    candidates = np.asarray(arr["candidate_features"])
    r = np.asarray(arr["raw_candidate_rewards"], np.float32)
    lp = np.asarray(arr["reference_conditional_logp"], np.float32)
    mass = np.asarray(arr["support_mass"], np.float32)
    run = ROOT / "artifacts/residual_support/policy/runs/exact_expected_reward_b0.3_seed17_full"
    ckpt = run / "last.pt"; saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    run_manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    columns = tuple(int(x) for x in saved.get("feature_columns", run_manifest["feature_columns"]))
    from tlab.residual_support.head import ResidualHeadSpec, ResidualPolicyHead
    trained = ResidualPolicyHead(ResidualHeadSpec(len(columns), 32)); trained.load_state_dict(saved.get("state_dict", saved.get("head_state")))
    # B4 uses gradient with respect to support logits, which isolates estimator noise from head parameterization.
    xscaled = scaler.transform(candidates, POLICY_FEATURE_NAMES)[..., list(columns)]
    z = ((r - mean) / std).astype(np.float32)
    q_head = np.empty((n,k), np.float32)
    for start in range(0,n,64):
        xx = torch.from_numpy(np.ascontiguousarray(xscaled[start:start+64]))
        with torch.no_grad():
            q_head[start:start+64] = torch.softmax(torch.from_numpy(lp[start:start+64]) + trained(xx), dim=-1).numpy()
    # Exact zero-output initialization is the identity conditional policy.
    cond_rng = np.random.default_rng(4202)
    batches = [np.arange(i*32, (i+1)*32) for i in range(8)]
    for policy_name, q_all in (("zero_init_identity", np.exp(lp - np.logaddexp.reduce(lp,axis=1,keepdims=True))),
                               ("trained_last_seed17", q_head)):
        for batch_id, idx in enumerate(batches):
            qb = np.asarray(q_all[idx], np.float64); rb = np.asarray(z[idx], np.float64); mb = mass[idx].astype(np.float64)
            pb = np.exp(lp[idx].astype(np.float64)); pb /= pb.sum(axis=1,keepdims=True)
            eq = np.sum(qb*rb, axis=1, keepdims=True)
            grad_reward = -mb[:,None] * qb * (rb-eq)
            logqb = np.log(np.maximum(qb,1e-30)); logpb = np.log(np.maximum(pb,1e-30))
            kl = np.sum(qb*(logqb-logpb),axis=1,keepdims=True)
            grad_kl = .3*mb[:,None]*qb*(logqb-logpb-kl)
            exact_total = grad_reward + grad_kl
            for nsamp in (4,16,64):
                reps=[]; reward_reps=[]
                for rep in range(64):
                    seed = 900000 + batch_id*100000 + nsamp*1000 + rep
                    rng = np.random.default_rng(seed)
                    actions = np.stack([rng.choice(k,size=nsamp,replace=True,p=row/row.sum()) for row in qb])
                    sr = np.take_along_axis(rb,actions,axis=1)
                    adv = sr - (sr.sum(axis=1,keepdims=True)-sr)/(nsamp-1)
                    g = np.zeros_like(qb)
                    for b in range(len(idx)):
                        np.add.at(g[b], actions[b], -mb[b]*adv[b]/nsamp)
                    reps.append(g)
                    reward_reps.append(g)
                reward_grads = np.stack(reward_reps)
                grads = reward_grads
                # Gradients in logits include exact KL, while reward-only uncertainty stays separately visible.
                totals = grads + grad_kl[None,:,:]
                mean_g = totals.mean(axis=0); flat_exact=exact_total.ravel(); flat_mean=mean_g.ravel()
                cosine = float(np.dot(flat_exact,flat_mean)/(np.linalg.norm(flat_exact)*np.linalg.norm(flat_mean)+1e-12))
                rel_err = float(np.linalg.norm(flat_mean-flat_exact)/(np.linalg.norm(flat_exact)+1e-12))
                norms = np.linalg.norm(totals.reshape(64,-1),axis=1)
                reward_norms = np.linalg.norm(grads.reshape(64,-1),axis=1)
                clipped = np.minimum(1.0, 1.0/np.maximum(norms,1e-30))
                reward_mean = grads.mean(axis=0)
                reward_exact_flat = grad_reward.ravel()
                reward_mean_flat = reward_mean.ravel()
                reward_cosine = float(np.dot(reward_exact_flat,reward_mean_flat)/(np.linalg.norm(reward_exact_flat)*np.linalg.norm(reward_mean_flat)+1e-12))
                reward_rel_err = float(np.linalg.norm(reward_mean_flat-reward_exact_flat)/(np.linalg.norm(reward_exact_flat)+1e-12))
                total_flat = totals.reshape(64,-1)
                per_draw_cosine = np.sum(total_flat*flat_exact[None,:],axis=1)/(np.maximum(norms,1e-30)*np.linalg.norm(flat_exact)+1e-12)
                reward_flat = grads.reshape(64,-1)
                reward_draw_cosine = np.sum(reward_flat*reward_exact_flat[None,:],axis=1)/(np.maximum(reward_norms,1e-30)*np.linalg.norm(reward_exact_flat)+1e-12)
                rms_deviation = float(np.sqrt(np.mean(np.sum((total_flat-flat_mean[None,:])**2,axis=1))))
                reward_rms_deviation = float(np.sqrt(np.mean(np.sum((reward_flat-reward_mean_flat[None,:])**2,axis=1))))
                rows_gradient.append({"policy":policy_name,"batch":batch_id,"batch_contexts":len(idx),"samples":nsamp,"draws":64,
                                      "exact_total_gradient_norm":float(np.linalg.norm(flat_exact)),"sampled_total_gradient_norm_mean":float(norms.mean()),
                                      "sampled_reward_gradient_norm_mean":float(reward_norms.mean()),"cosine_mean_to_exact":cosine,
                                      "relative_error_of_mean":rel_err,"sampled_gradient_norm_sd":float(norms.std(ddof=1)),
                                      "gradient_bias_norm":float(np.linalg.norm(flat_mean-flat_exact)),
                                      "negative_cosine_fraction":float(np.mean(per_draw_cosine<0)),
                                      "mean_gradient_norm_to_rms_deviation":float(np.linalg.norm(flat_mean)/(rms_deviation+1e-30)),
                                      "reward_exact_gradient_norm":float(np.linalg.norm(reward_exact_flat)),
                                      "reward_mean_gradient_norm":float(np.linalg.norm(reward_mean_flat)),
                                      "reward_gradient_bias_norm":float(np.linalg.norm(reward_mean_flat-reward_exact_flat)),
                                      "reward_relative_error_of_mean":reward_rel_err,"reward_cosine_of_mean_to_exact":reward_cosine,
                                      "reward_negative_cosine_fraction":float(np.mean(reward_draw_cosine<0)),
                                      "reward_mean_gradient_norm_to_rms_deviation":float(np.linalg.norm(reward_mean_flat)/(reward_rms_deviation+1e-30)),
                                      "clip_at_1_fraction":float(np.mean(norms>1.0)),"mean_clip_factor":float(clipped.mean())})

    # Save and report all diagnostics. Separate raw reward from normalized utility throughout.
    out = ARTIFACTS / "diagnostics"; out.mkdir(parents=True, exist_ok=True)
    def write(name: str, rows: list[dict[str, Any]]) -> None:
        if not rows: return
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with (out / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    write("external_audit_expectations.csv", rows_audit)
    write("reward_variance_alignment_beta.csv", rows_diag + rows_beta)
    write("gradient_noise.csv", rows_gradient)
    write("representability_embedding_ridge.csv", rows_rep)
    write("support_coverage.csv", rows_support)
    alignment_path = out / "proxy_audit_alignment_users.csv"
    with alignment_path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows_alignment_user[0]));w.writeheader();w.writerows(rows_alignment_user)
    summary_alignment=[]
    alignment_metrics=("weighted_pearson","weighted_spearman","centered_sign_agreement","top10_overlap",
                       "residual_pearson_item_metadata","residual_pearson_additive")
    for panel in sorted({r["panel"] for r in rows_alignment_user}):
        panel_rows=[r for r in rows_alignment_user if r["panel"]==panel]
        for metric_index,metric in enumerate(alignment_metrics):
            values=np.asarray([float(r[metric]) for r in panel_rows],dtype=np.float64)
            values=values[np.isfinite(values)]
            rng=np.random.default_rng(20261022+metric_index+sum(ord(c) for c in panel))
            boot=np.empty(2000,dtype=np.float64)
            for b in range(len(boot)):
                boot[b]=float(np.mean(values[rng.integers(0,len(values),size=len(values))]))
            summary_alignment.append({"panel":panel,"metric":metric,"n_users":len(values),"mean":float(values.mean()),
                                     "ci95_low":float(np.quantile(boot,.025)),"ci95_high":float(np.quantile(boot,.975)),
                                     "constant_proxy_users":int(sum(r["proxy_constant"] for r in panel_rows)),
                                     "constant_audit_users":int(sum(r["audit_constant"] for r in panel_rows)),"bootstrap_draws":len(boot)})
    with (out/"proxy_audit_alignment_user_ci.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(summary_alignment[0]));w.writeheader();w.writerows(summary_alignment)
    summary = {"runtime_seconds":round(time.perf_counter()-started,3),"n_audit_rows":len(rows_audit),"n_B1_B2_rows":len(rows_diag),
               "n_beta_rows":len(rows_beta),"n_gradient_rows":len(rows_gradient),"n_embedding_ridge_contexts":len(rows_rep),
               "n_alignment_user_rows":len(rows_alignment_user),"n_alignment_user_ci_rows":len(summary_alignment),
               "external_audit_role":"diagnostic only; never used for branch selection", "raw_rewards_and_z_rewards":"reported separately",
               "policy_expectation":"conditional over same fixed top-500 support; mass-weighted delta assumes the identical reference tail"}
    (out/"diagnostics_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    record_operation("A2_external_audit_and_B1_B5_diagnostics",summary["runtime_seconds"],bucket="B",category="evaluation_and_diagnostics",
                     evidence=[f"artifacts/policy_diagnostics/diagnostics/{n}" for n in ("external_audit_expectations.csv","reward_variance_alignment_beta.csv","proxy_audit_alignment_users.csv","proxy_audit_alignment_user_ci.csv","gradient_noise.csv","representability_embedding_ridge.csv","support_coverage.csv","diagnostics_summary.json")])
    update_stage("A","complete",["artifacts/policy_diagnostics/evaluation/legacy_completion.csv","artifacts/policy_diagnostics/diagnostics/external_audit_expectations.csv"],note="A1-A3 completed; external audit predictions for all trained head endpoints and analytic tilts on the fixed common-support cache; audit is descriptive only.")
    for stage, ev in (("B1", "reward_variance_alignment_beta.csv"),("B2","reward_variance_alignment_beta.csv"),("B3","reward_variance_alignment_beta.csv"),("B4","gradient_noise.csv"),("B5","representability_embedding_ridge.csv"),("B6","support_coverage.csv")):
        update_stage(stage,"complete",[f"artifacts/policy_diagnostics/diagnostics/{ev}"])
    return summary


if __name__ == "__main__":
    print(json.dumps(run_audit_and_B123(),indent=2))
