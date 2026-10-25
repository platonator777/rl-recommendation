from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from tlab.data.catalog import PAD_ID
from tlab.data.store import SequenceStore
from tlab.models.actor import ActorShape, ItemGPT2Actor
from tlab.panel_tools.data import PanelTarget, build_evaluation_panel, history_for_target
from tlab.panel_tools.evaluation import evaluate_rank_rows


def _collate_histories(histories: Sequence[Sequence[int]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(history) for history in histories)
    ids = torch.full((len(histories), width), PAD_ID, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for row, history in enumerate(histories):
        ids[row, :len(history)] = torch.as_tensor(history, dtype=torch.long, device=device)
        mask[row, :len(history)] = 1
    return ids, mask


def load_actor_checkpoint(
    checkpoint_path: str | Path,
    *,
    vocab_size: int,
    n_positions: int = 256,
    n_layer: int = 4,
    n_embd: int = 256,
    n_head: int = 4,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[ItemGPT2Actor, dict[str, Any]]:
    """Load an actor state without modifying the source checkpoint."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload.get("model") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise ValueError("checkpoint does not contain a model state dictionary")
    shape = ActorShape(vocab_size=vocab_size, n_positions=n_positions, n_layer=n_layer,
                       n_embd=n_embd, n_head=n_head, dropout=0.0)
    actor = ItemGPT2Actor(shape)
    actor.load_state_dict(state, strict=True)
    actor.to(device).eval()
    source_metadata = payload.get("metadata", {})
    if not isinstance(source_metadata, dict):
        source_metadata = {}
    provenance_keys = ("phase", "run_id", "run_start_epoch", "epoch", "global_step", "best_epoch",
                       "sampler_version", "base_learning_rate")
    details = {
        "checkpoint_sha256": _sha256_file(Path(checkpoint_path)),
        "source_step": payload.get("step"),
        "source_metadata": {key: source_metadata[key] for key in provenance_keys if key in source_metadata},
        "actor_shape": {"vocab_size": vocab_size, "n_positions": n_positions, "n_layer": n_layer,
                        "n_embd": n_embd, "n_head": n_head},
    }
    return actor, details


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.inference_mode()
def evaluate_actor_panel(
    actor: ItemGPT2Actor,
    store: SequenceStore,
    catalog_item_ids: Sequence[int],
    panel: Sequence[PanelTarget],
    *,
    history_events: int = 200,
    batch_size: int = 16,
    k: int = 10,
) -> dict[str, object]:
    if not panel or batch_size < 1:
        raise ValueError("evaluation requires panel targets and a positive batch size")
    if actor.vocab_size != len(catalog_item_ids) + 3:
        raise ValueError("actor vocabulary and catalog item count do not match")
    item_ids = np.asarray(catalog_item_ids, dtype=np.int64)
    if len(set(item_ids.tolist())) != len(item_ids):
        raise ValueError("catalog item IDs must be unique")
    device = next(actor.parameters()).device
    actor.eval()
    def score_rows():
        for start in range(0, len(panel), batch_size):
            targets = panel[start:start + batch_size]
            histories = [history_for_target(store, target, history_events) for target in targets]
            input_ids, attention = _collate_histories(histories, device)
            log_probabilities = actor.next_action_log_probs(input_ids, attention)[:, 3:].float().cpu().numpy()
            for target, scores in zip(targets, log_probabilities):
                yield {
                    "uid": target.uid,
                    "scores": scores,
                    "item_ids": item_ids,
                    "target_id": target.raw_item_id,
                    "positive": target.positive,
                    "is_new": target.is_new,
                    "is_repeat": target.is_repeat,
                    "known_catalog": target.token_id >= 3,
                    "event_id": target.event_id,
                }

    result = evaluate_rank_rows(score_rows(), k=k)
    result.update({
        "panel_event_ids_sha256": hashlib.sha256("\n".join(row.event_id for row in panel).encode()).hexdigest(),
        "history_events_max": history_events,
        "score_model": "full-catalog GPT-2 actor",
        "target_selection": "pre-frozen event IDs; OOV targets retained as misses",
        "per_user_primary": result["primary_user_macro"].pop("per_user"),
    })
    return result


def fit_lightgbm_ratio_evaluator(
    train_features: np.ndarray,
    train_ratio: np.ndarray,
    *,
    seed: int = 17,
    n_estimators: int = 300,
    num_leaves: int = 31,
) -> tuple[Any, dict[str, object]]:
    """Fit the optional external/audit evaluator; never label it as ground truth."""
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("LightGBM evaluator is unavailable; install the pinned optional panel_tools dependency separately") from exc
    x = np.asarray(train_features, dtype=np.float32)
    y = np.asarray(train_ratio, dtype=np.float32)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or not len(y):
        raise ValueError("training features and ratio labels must be aligned and non-empty")
    valid = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    if not np.any(valid):
        raise ValueError("no finite training rows remain for the evaluator")
    model = lgb.LGBMRegressor(
        objective="regression", n_estimators=int(n_estimators), num_leaves=int(num_leaves),
        random_state=int(seed), n_jobs=-1, verbosity=-1,
    )
    model.fit(x[valid], y[valid])
    return model, {
        "role": "external/audit evaluator",
        "not_ground_truth": True,
        "backend": "lightgbm.LGBMRegressor",
        "lightgbm_version": lgb.__version__,
        "seed": int(seed),
        "n_train_rows": int(valid.sum()),
        "n_features": int(x.shape[1]),
    }


def evaluate_saved_actor(
    checkpoint_path: str | Path,
    store: SequenceStore,
    catalog_item_ids: Sequence[int],
    *,
    source_sha256: str,
    split: str = "validation",
    seed: int = 17,
    targets_per_user: int = 3,
    history_events: int = 200,
    batch_size: int = 16,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict[str, object]:
    panel = build_evaluation_panel(store, split=split, targets_per_user=targets_per_user, seed=seed,
                                  source_sha256=source_sha256)
    actor, checkpoint = load_actor_checkpoint(checkpoint_path, vocab_size=len(catalog_item_ids) + 3, device=device)
    result = evaluate_actor_panel(actor, store, catalog_item_ids, panel, history_events=history_events,
                                  batch_size=batch_size)
    result.update({"checkpoint": checkpoint, "split": split, "panel_events": len(panel), "seed": seed})
    return result
