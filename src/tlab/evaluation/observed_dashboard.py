from __future__ import annotations

import argparse
import bisect
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tlab.config import load_config
from tlab.data.labels import TimedPairEvent, attribute_new_likes
from tlab.data.manifest import sha256_file, write_json_atomic
from tlab.data.store import SequenceStore


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _bootstrap_mean(values: list[float], seed: int, repetitions: int = 2000) -> dict[str, Any] | None:
    if not values:
        return None
    samples = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(samples), size=(repetitions, len(samples)))
    means = samples[draws].mean(axis=1)
    return {
        "mean": float(samples.mean()),
        "ci_95_low": float(np.quantile(means, 0.025)),
        "ci_95_high": float(np.quantile(means, 0.975)),
        "n_users": len(values),
        "repetitions": repetitions,
    }


def summarize_user_test_events(
    uid: int,
    rows: list[dict[str, Any]],
    likes: list[TimedPairEvent],
    unlikes: list[TimedPairEvent],
    test_start: float,
    test_end: float,
    timestamp_unit_seconds: float = 1.0,
) -> dict[str, Any]:
    """Summarize one user's fixed test window, preserving event maturity rules."""
    if timestamp_unit_seconds <= 0 or test_end <= test_start:
        raise ValueError("invalid test window or timestamp unit")
    day_seconds = 86400.0 / timestamp_unit_seconds
    continue_window = 1800.0 / timestamp_unit_seconds
    relisten_window = 86400.0 / timestamp_unit_seconds
    test_rows = sorted(
        (row for row in rows if test_start <= float(row["timestamp"]) < test_end),
        key=lambda row: (float(row["timestamp"]), int(row["item_id"])),
    )
    valid_rows = [row for row in test_rows if not bool(row.get("ambiguous_timestamp_block", False))]
    active_days = sorted({min(6, int((float(row["timestamp"]) - test_start) // day_seconds)) for row in test_rows})
    daily_seconds = [0.0] * 7
    ratio_values: list[float] = []
    seconds_total = 0.0
    for row in test_rows:
        timestamp = float(row["timestamp"])
        seconds = _finite(row.get("seconds_proxy_raw"))
        if seconds is not None:
            day = min(6, int((timestamp - test_start) // day_seconds))
            daily_seconds[day] += seconds
            seconds_total += seconds
        ratio = _finite(row.get("ratio_raw"))
        if ratio is not None and not bool(row.get("ambiguous_timestamp_block", False)):
            ratio_values.append(ratio)

    valid_times = [float(row["timestamp"]) for row in valid_rows]
    valid_seconds = [_finite(row.get("seconds_proxy_raw")) or 0.0 for row in valid_rows]
    seconds_prefix = [0.0]
    for value in valid_seconds:
        seconds_prefix.append(seconds_prefix[-1] + value)
    nonorganic_times = [
        float(row["timestamp"]) for row in valid_rows if not bool(row.get("is_organic", False))
    ]

    eligible_anchors = [
        row for row in rows
        if test_start <= float(row["timestamp"]) < test_end
        and not bool(row.get("is_organic", False))
        and not bool(row.get("ambiguous_timestamp_block", False))
        and int(row.get("token_id", -1)) >= 3
        and int(row.get("item_id", -1)) >= 0
    ]
    mature_continuation = [
        row for row in eligible_anchors
        if float(row["timestamp"]) + continue_window <= test_end
    ]
    continuation_positive = 0
    continuation_seconds: list[float] = []
    for row in mature_continuation:
        timestamp = float(row["timestamp"])
        next_index = bisect.bisect_right(nonorganic_times, timestamp)
        continuation_positive += int(
            next_index < len(nonorganic_times)
            and nonorganic_times[next_index] <= timestamp + continue_window
        )
        start_index = bisect.bisect_right(valid_times, timestamp)
        stop_index = bisect.bisect_right(valid_times, timestamp + continue_window)
        continuation_seconds.append(seconds_prefix[stop_index] - seconds_prefix[start_index])

    like_listens = [
        TimedPairEvent(uid, int(row["item_id"]), float(row["timestamp"]))
        for row in eligible_anchors
    ]
    like_labels = attribute_new_likes(
        like_listens,
        likes,
        unlikes,
        observation_cutoff=test_end,
        window_seconds=1800,
        timestamp_unit_seconds=timestamp_unit_seconds,
    )
    valid_like_labels = [label for label in like_labels if label.label is not None]
    like_positive = sum(int(label.label == 1) for label in valid_like_labels)
    like_reasons = Counter(label.reason for label in like_labels)

    times_by_item: dict[int, list[float]] = defaultdict(list)
    for row in valid_rows:
        item_id = int(row.get("item_id", -1))
        if item_id >= 0:
            times_by_item[item_id].append(float(row["timestamp"]))
    relisten_by_type: dict[str, list[int]] = {"all": [], "organic": [], "nonorganic": []}
    relisten_anchors = 0
    for row in valid_rows:
        timestamp = float(row["timestamp"])
        if timestamp + relisten_window > test_end:
            continue
        item_id = int(row.get("item_id", -1))
        item_times = times_by_item.get(item_id, [])
        next_index = bisect.bisect_right(item_times, timestamp)
        relistened = int(next_index < len(item_times) and item_times[next_index] <= timestamp + relisten_window)
        relisten_anchors += 1
        relisten_by_type["all"].append(relistened)
        relisten_by_type["organic" if bool(row.get("is_organic", False)) else "nonorganic"].append(relistened)

    replay_excess = [max(0.0, ratio - 1.0) for ratio in ratio_values]
    return {
        "uid": int(uid),
        "test_event_count": len(test_rows),
        "test_active_days": len(active_days),
        "seconds_proxy_total": seconds_total,
        "seconds_proxy_by_day": daily_seconds,
        "eligible_nonorganic_anchors": len(eligible_anchors),
        "mature_continue_anchors": len(mature_continuation),
        "continue_positive": continuation_positive,
        "continue_later_seconds_sum": sum(continuation_seconds),
        "eligible_like_labels": len(valid_like_labels),
        "new_likes": like_positive,
        "like_label_reasons": dict(like_reasons),
        "relisten_mature_anchors": relisten_anchors,
        "relisten_positive": {name: sum(values) for name, values in relisten_by_type.items()},
        "relisten_counts": {name: len(values) for name, values in relisten_by_type.items()},
        "ratio_observations": len(ratio_values),
        "replay_excess_count": sum(value > 0 for value in replay_excess),
        "replay_excess_sum": sum(replay_excess),
    }


def _load_action_events(path: Path) -> dict[int, list[TimedPairEvent]]:
    import pyarrow.parquet as pq

    grouped: dict[int, list[TimedPairEvent]] = defaultdict(list)
    for batch in pq.ParquetFile(path).iter_batches(batch_size=131_072, columns=["uid", "item_id", "timestamp"]):
        uid_col, item_col, timestamp_col = (column.to_pylist() for column in batch.columns)
        for uid, item_id, timestamp in zip(uid_col, item_col, timestamp_col):
            grouped[int(uid)].append(TimedPairEvent(int(uid), int(item_id), int(timestamp)))
    return grouped


def observed_test_dashboard(config_path: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    import pyarrow.dataset as ds
    import polars as pl

    config = load_config(config_path)
    root = config.path.parent.parent
    processed_dir = Path(config.data.get("processed_dir", "data/processed"))
    raw_dir = Path(config.data.get("source_dir", "data/raw"))
    if not processed_dir.is_absolute():
        processed_dir = root / processed_dir
    if not raw_dir.is_absolute():
        raw_dir = root / raw_dir
    protocol_path = processed_dir / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    timestamp_unit = float(protocol["timestamp_unit_seconds"])
    boundaries = protocol["boundaries_source_units"]
    test_start = float(boundaries["test_start_raw"])
    test_end = float(boundaries["end_raw"])
    if abs((test_end - test_start) * timestamp_unit / 86400.0 - 7.0) > 1e-6:
        raise ValueError("observed dashboard expects the fixed seven-day test window")

    store = SequenceStore.load_cache(
        processed_dir / "sequence_index", expected_source_sha256=protocol["processed_listens_sha256"],
    )
    cohort = {
        int(user.uid)
        for user in store.users
        if any(int(split_code) < 2 for split_code in user.split_codes)
    }
    if not cohort:
        raise ValueError("no pre-test user cohort is available")
    likes_by_uid = _load_action_events(raw_dir / "likes.parquet")
    unlikes_by_uid = _load_action_events(raw_dir / "unlikes.parquet")

    dataset = ds.dataset(processed_dir / "listens.parquet", format="parquet")
    scanner = dataset.scanner(
        columns=[
            "uid", "timestamp", "item_id", "is_organic", "token_id", "ambiguous_timestamp_block",
            "ratio_raw", "seconds_proxy_raw",
        ],
        filter=ds.field("global_split") == "test",
        batch_size=65_536,
    )
    summaries: list[dict[str, Any]] = []
    current_uid: int | None = None
    current_rows: list[dict[str, Any]] = []

    def flush_user() -> None:
        nonlocal current_rows, current_uid
        if current_uid is None or current_uid not in cohort:
            current_rows = []
            return
        summaries.append(summarize_user_test_events(
            current_uid,
            current_rows,
            likes_by_uid.get(current_uid, []),
            unlikes_by_uid.get(current_uid, []),
            test_start,
            test_end,
            timestamp_unit,
        ))
        current_rows = []

    for batch in scanner.to_batches():
        columns = [batch.column(index).to_pylist() for index in range(batch.num_columns)]
        for values in zip(*columns):
            uid = int(values[0])
            if current_uid is not None and uid != current_uid:
                flush_user()
            if current_uid != uid:
                current_uid = uid
            if uid not in cohort:
                continue
            current_rows.append({
                "uid": uid,
                "timestamp": float(values[1]),
                "item_id": int(values[2]),
                "is_organic": bool(values[3]),
                "token_id": int(values[4]),
                "ambiguous_timestamp_block": bool(values[5]),
                "ratio_raw": values[6],
                "seconds_proxy_raw": values[7],
            })
    flush_user()

    summary_by_uid = {row["uid"]: row for row in summaries}
    user_count = len(cohort)
    zero_user = {
        uid: summary_by_uid.get(uid, {
            "uid": uid, "test_event_count": 0, "test_active_days": 0, "seconds_proxy_total": 0.0,
            "eligible_nonorganic_anchors": 0, "mature_continue_anchors": 0, "continue_positive": 0,
            "continue_later_seconds_sum": 0.0, "eligible_like_labels": 0, "new_likes": 0,
            "relisten_mature_anchors": 0, "relisten_positive": {"all": 0, "organic": 0, "nonorganic": 0},
            "relisten_counts": {"all": 0, "organic": 0, "nonorganic": 0},
            "ratio_observations": 0, "replay_excess_count": 0, "replay_excess_sum": 0.0,
        })
        for uid in cohort
    }
    user_rows = list(zero_user.values())
    total_seconds = sum(float(row["seconds_proxy_total"]) for row in user_rows)
    user_seconds_per_day = [float(row["seconds_proxy_total"]) / 7.0 for row in user_rows]
    daily_totals = [sum(float(row.get("seconds_proxy_by_day", [0.0] * 7)[day]) for row in summaries) for day in range(7)]
    continuation_n = sum(int(row["mature_continue_anchors"]) for row in user_rows)
    continuation_positive = sum(int(row["continue_positive"]) for row in user_rows)
    continuation_user_rates = [
        int(row["continue_positive"]) / int(row["mature_continue_anchors"])
        for row in summaries if int(row["mature_continue_anchors"]) > 0
    ]
    like_n = sum(int(row["eligible_like_labels"]) for row in user_rows)
    like_positive = sum(int(row["new_likes"]) for row in user_rows)
    like_user_rates = [
        int(row["new_likes"]) / int(row["eligible_like_labels"])
        for row in summaries if int(row["eligible_like_labels"]) > 0
    ]
    relisten_counts = {
        name: sum(int(row.get("relisten_counts", {}).get(name, 0)) for row in user_rows)
        for name in ("all", "organic", "nonorganic")
    }
    relisten_positives = {
        name: sum(int(row.get("relisten_positive", {}).get(name, 0)) for row in user_rows)
        for name in relisten_counts
    }
    ratio_n = sum(int(row["ratio_observations"]) for row in user_rows)
    replay_n = sum(int(row["replay_excess_count"]) for row in user_rows)
    replay_sum = sum(float(row["replay_excess_sum"]) for row in user_rows)
    returned_users = sum(int(row["test_event_count"] > 0) for row in user_rows)
    output = {
        "split": "test",
        "status": "descriptive_post_freeze",
        "selection_use_prohibited": True,
        "observation_window_raw": [test_start, test_end],
        "observation_days": 7,
        "cohort_definition": "users with at least one event before the test window (Train or Validation)",
        "n_users_cohort": user_count,
        "source_hashes": {
            "data_protocol_sha256": sha256_file(protocol_path),
            "processed_listens_sha256": protocol["processed_listens_sha256"],
            "likes_sha256": sha256_file(raw_dir / "likes.parquet"),
            "unlikes_sha256": sha256_file(raw_dir / "unlikes.parquet"),
        },
        "observed": {
            "listening": {
                "seconds_proxy_definition": "ratio_raw * track_length_seconds (uncapped logged proxy)",
                "total_seconds_proxy": total_seconds,
                "mean_seconds_proxy_per_user_day_including_zeros": total_seconds / (user_count * 7),
                "mean_user_7d_seconds_proxy": total_seconds / user_count,
                "mean_user_seconds_proxy_per_day_bootstrap": _bootstrap_mean(user_seconds_per_day, seed=17),
                "seconds_proxy_by_test_day": daily_totals,
                "mean_active_days_per_user_including_zeros": sum(int(row["test_active_days"]) for row in user_rows) / user_count,
                "return_during_test_rate": returned_users / user_count,
                "return_during_test_rate_bootstrap": _bootstrap_mean(
                    [float(row["test_event_count"] > 0) for row in user_rows], seed=18,
                ),
            },
            "continue_30m": {
                "definition": "next strictly later non-organic listen within 30 minutes; full window required",
                "eligible_anchors": continuation_n,
                "positive_anchors": continuation_positive,
                "rate": continuation_positive / continuation_n if continuation_n else None,
                "mean_user_rate_bootstrap": _bootstrap_mean(continuation_user_rates, seed=19),
                "total_follow_on_seconds_proxy": sum(float(row["continue_later_seconds_sum"]) for row in user_rows),
                "mean_follow_on_seconds_proxy_per_anchor": (
                    sum(float(row["continue_later_seconds_sum"]) for row in user_rows) / continuation_n
                    if continuation_n else None
                ),
            },
            "new_like": {
                "definition": "new likes attributed within 30 minutes among eligible, fully observed non-organic listens",
                "eligible_listens": like_n,
                "new_likes": like_positive,
                "likes_per_100_eligible_listens": 100.0 * like_positive / like_n if like_n else None,
                "new_likes_per_user_day_including_zeros": like_positive / (user_count * 7),
                "mean_user_new_likes_per_day_bootstrap": _bootstrap_mean(
                    [float(row["new_likes"]) / 7.0 for row in user_rows], seed=21,
                ),
                "mean_user_rate_bootstrap": _bootstrap_mean(like_user_rates, seed=20),
                "attribution_reason_counts": dict(sum((Counter(row.get("like_label_reasons", {})) for row in summaries), Counter())),
            },
            "relisten_24h": {
                "definition": "same item heard again strictly later within 24 hours; complete future window required",
                "by_anchor_type": {
                    name: {
                        "anchors": relisten_counts[name],
                        "relistened": relisten_positives[name],
                        "rate": relisten_positives[name] / relisten_counts[name] if relisten_counts[name] else None,
                    }
                    for name in relisten_counts
                },
            },
            "replay_excess": {
                "definition": "ratio_raw > 1 on finite, non-ambiguous test listens",
                "eligible_listens": ratio_n,
                "excess_listens": replay_n,
                "replay_excess_rate": replay_n / ratio_n if ratio_n else None,
                "mean_excess_ratio_per_listen": replay_sum / ratio_n if ratio_n else None,
            },
            "repeat_commands": {"status": "not_observable_in_dataset_schema"},
        },
        "prediction_status": {
            "seconds": "NA: RM seconds heads did not beat the strongest simple control",
            "likes": "NA: RM_audit like head did not beat prevalence baseline",
            "continuation": "NA as a candidate policy reward: candidate-aware predictor added no signal over activity-only",
            "online_effect": "not measured",
        },
        "user_sufficient_metrics": user_rows,
    }
    target = Path(output_path) if output_path else root / "artifacts" / "observed_test_dashboard.json"
    if not target.is_absolute():
        target = root / target
    write_json_atomic(target, output)
    output["output_path"] = str(target.resolve())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a descriptive observed-metrics dashboard for the frozen test window")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = observed_test_dashboard(args.config, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
