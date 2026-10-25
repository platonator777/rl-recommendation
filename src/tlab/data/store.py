from __future__ import annotations

import random
import json
import os
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

from tlab.data.catalog import BOS_ID, OOV_ID

SPLIT_CODES = {"train": 0, "validation": 1, "test": 2}
SUBSPLIT_CODES = {"F_RM": 0, "V_RM": 1, "A_RL": 2}


@dataclass
class SFTTargetIndex:
    """Compact, deterministic index of distinct supervised target examples.

    Histories are materialized only when a batch is formed.  This keeps a
    multi-million-target epoch from retaining Python lists of full histories.
    ``positions`` are absolute positions within ``SequenceStore.users``.
    """

    user_indices: array
    positions: array
    history_events: int
    sampling_seed: int
    positive_only: bool = False

    def __len__(self) -> int:
        return len(self.positions)

    def permutation(self, seed: int):
        """Return the fixed NumPy permutation used for this epoch's presentation order."""
        import numpy as np

        order = np.arange(len(self), dtype=np.int64)
        np.random.default_rng(int(seed)).shuffle(order)
        return order

    def example(self, store: "SequenceStore", index: int) -> tuple[list[int], list[bool]]:
        user_index = int(self.user_indices[index])
        position = int(self.positions[index])
        user = store.users[user_index]
        history = user.history(position, self.history_events + 1)
        target = int(user.tokens[position])
        if target < 3:
            raise AssertionError("SFT target index contains a non-catalog token")
        tokens = [*history, target]
        mask = [False] * (len(tokens) - 1) + [True]
        if len(history) > self.history_events + 1:
            raise AssertionError("SFT context exceeds the declared BOS-plus-history contract")
        return tokens, mask

    def target_keys(self):
        """Compact stable IDs for overlap diagnostics across sampled epochs."""
        import numpy as np

        users = np.frombuffer(self.user_indices, dtype=np.uint32)
        positions = np.frombuffer(self.positions, dtype=np.uint32)
        return (users.astype(np.uint64) << np.uint64(32)) | positions.astype(np.uint64)


@dataclass
class UserSequence:
    uid: int
    rm_user_partition: str | None = None
    tokens: array = field(default_factory=lambda: array("i"))
    raw_item_ids: array = field(default_factory=lambda: array("q"))
    split_codes: bytearray = field(default_factory=bytearray)
    subsplit_codes: bytearray = field(default_factory=bytearray)
    train_targets: array = field(default_factory=lambda: array("I"))
    rm_fit_positions: array = field(default_factory=lambda: array("I"))
    rm_validation_positions: array = field(default_factory=lambda: array("I"))
    rl_anchors: array = field(default_factory=lambda: array("I"))
    recommendable_events: bytearray = field(default_factory=bytearray)
    ratio_raw: array = field(default_factory=lambda: array("f"))
    duration_seconds: array = field(default_factory=lambda: array("f"))

    def append(
        self,
        token: int,
        item_id: int,
        timestamp: float,
        split: str,
        train_subsplit: str | None,
        is_organic: bool,
        ratio: float | None,
        duration: float | None,
        ambiguous: bool,
    ) -> None:
        index = len(self.tokens)
        if split not in SPLIT_CODES:
            raise ValueError(f"unexpected global split: {split}")
        stored_token = OOV_ID if ambiguous or token < 3 else int(token)
        self.tokens.append(stored_token)
        self.raw_item_ids.append(-1 if ambiguous else int(item_id))
        self.split_codes.append(SPLIT_CODES[split])
        self.subsplit_codes.append(SUBSPLIT_CODES.get(train_subsplit, 3))
        self.ratio_raw.append(float("nan") if ambiguous or ratio is None else float(ratio))
        self.duration_seconds.append(float("nan") if ambiguous or duration is None else float(duration))
        eligible = index > 0 and not is_organic and not ambiguous
        self.recommendable_events.append(int(eligible))
        if eligible and split == "train" and stored_token >= 3:
            self.train_targets.append(index)
        if eligible and stored_token >= 3 and train_subsplit == "F_RM":
            self.rm_fit_positions.append(index)
        if eligible and stored_token >= 3 and train_subsplit == "V_RM":
            self.rm_validation_positions.append(index)

    def history(self, position: int, history_length: int) -> list[int]:
        if position < 0 or position > len(self.tokens):
            raise IndexError(position)
        start = max(0, position - max(0, history_length - 1))
        return [BOS_ID, *self.tokens[start:position]]


@dataclass
class SequenceStore:
    users: list[UserSequence]
    events: int
    tie_blocks: int
    tied_events: int

    _CACHE_FIELDS = {
        "tokens": ("<i4", "tokens.bin"),
        "raw_item_ids": ("<i8", "raw_item_ids.bin"),
        "split_codes": ("u1", "split_codes.bin"),
        "subsplit_codes": ("u1", "subsplit_codes.bin"),
        "train_targets": ("<u4", "train_targets.bin"),
        "rm_fit_positions": ("<u4", "rm_fit_positions.bin"),
        "rm_validation_positions": ("<u4", "rm_validation_positions.bin"),
        "rl_anchors": ("<u4", "rl_anchors.bin"),
        "recommendable_events": ("u1", "recommendable_events.bin"),
        "ratio_raw": ("<f4", "ratio_raw.bin"),
        "duration_seconds": ("<f4", "duration_seconds.bin"),
    }

    def save_cache(self, directory: str | Path, source_sha256: str) -> None:
        """Persist primitive per-user arrays for fast training restarts."""
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        offsets = {name: 0 for name in self._CACHE_FIELDS}
        files = {name: (target / filename).open("wb") for name, (_, filename) in self._CACHE_FIELDS.items()}
        users_meta = []
        try:
            for user in self.users:
                entry = {"uid": user.uid, "rm_user_partition": user.rm_user_partition, "arrays": {}}
                for name in self._CACHE_FIELDS:
                    values = getattr(user, name)
                    count = len(values)
                    entry["arrays"][name] = [offsets[name], offsets[name] + count]
                    if isinstance(values, bytearray):
                        files[name].write(values)
                    else:
                        values.tofile(files[name])
                    offsets[name] += count
                users_meta.append(entry)
        finally:
            for handle in files.values():
                handle.close()
        manifest = {
            "cache_version": 1,
            "source_sha256": source_sha256,
            "events": self.events,
            "tie_blocks": self.tie_blocks,
            "tied_events": self.tied_events,
            "array_lengths": offsets,
            "fields": {name: {"dtype": dtype, "file": filename} for name, (dtype, filename) in self._CACHE_FIELDS.items()},
            "users": users_meta,
        }
        temp = target / "index.json.tmp"
        temp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(temp, target / "index.json")

    @classmethod
    def load_cache(cls, directory: str | Path, expected_source_sha256: str) -> "SequenceStore":
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("numpy is required to open the persisted sequence cache") from exc
        target = Path(directory)
        manifest = json.loads((target / "index.json").read_text(encoding="utf-8"))
        if manifest.get("cache_version") != 1 or manifest.get("source_sha256") != expected_source_sha256:
            raise ValueError("sequence cache version or source hash does not match")
        for name, details in manifest["fields"].items():
            expected_bytes = int(manifest["array_lengths"][name]) * np.dtype(details["dtype"]).itemsize
            cache_file = target / details["file"]
            if not cache_file.exists() or cache_file.stat().st_size != expected_bytes:
                raise ValueError(f"sequence cache array {name!r} has an unexpected file size")
        maps = {
            name: (
                np.memmap(target / details["file"], dtype=details["dtype"], mode="r", shape=(manifest["array_lengths"][name],))
                if manifest["array_lengths"][name] > 0
                else np.empty(0, dtype=details["dtype"])
            )
            for name, details in manifest["fields"].items()
        }
        users = []
        for entry in manifest["users"]:
            user = UserSequence(uid=int(entry["uid"]), rm_user_partition=entry.get("rm_user_partition"))
            for name in cls._CACHE_FIELDS:
                start, stop = entry["arrays"][name]
                setattr(user, name, maps[name][start:stop])
            users.append(user)
        return cls(users, int(manifest["events"]), int(manifest["tie_blocks"]), int(manifest["tied_events"]))

    @classmethod
    def from_parquet(cls, path: str | Path, batch_size: int = 65536) -> "SequenceStore":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("pyarrow is required to build the lazy sequence index") from exc

        parquet = pq.ParquetFile(path)
        columns = [
            "uid", "item_id", "timestamp", "token_id", "is_organic", "global_split",
            "train_subsplit", "rm_user_partition", "ratio_raw", "duration_raw_seconds", "timestamp_tie_count",
        ]
        user_rows: list[UserSequence] = []
        current_user: UserSequence | None = None
        current_group: list[tuple[int, float, int, int, bool, str, str | None, str | None, float | None, float | None]] = []
        current_key: tuple[int, float] | None = None
        events = tie_blocks = tied_events = 0
        next_progress = 5_000_000

        def flush_group() -> None:
            nonlocal tie_blocks, tied_events
            if not current_group or current_user is None:
                return
            ambiguous = len(current_group) > 1 or any(row[6] == "ambiguous" for row in current_group)
            if ambiguous:
                tie_blocks += 1
                tied_events += len(current_group)
                uid, timestamp = current_group[0][0], current_group[0][1]
                split = current_group[0][5]
                current_user.append(OOV_ID, -1, timestamp, split, None, True, None, None, ambiguous=True)
                return
            uid, timestamp, token_id, item_id, organic, split, train_subsplit, _, ratio, duration = current_group[0]
            current_user.append(
                token=token_id,
                item_id=item_id,
                timestamp=timestamp,
                split=split,
                train_subsplit=train_subsplit,
                is_organic=organic,
                ratio=ratio,
                duration=duration,
                ambiguous=False,
            )
            if current_group[0][6] == "A_RL" and len(current_user.tokens) > 1 and not organic:
                current_user.rl_anchors.append(len(current_user.tokens) - 1)

        for record_batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
            rows = [column.to_pylist() for column in record_batch.columns]
            for (
                uid_raw, item_raw, timestamp_raw, token_raw, organic_raw, split_raw,
                train_subsplit_raw, user_partition_raw, ratio_raw, duration_raw, tie_count_raw,
            ) in zip(*rows):
                uid = int(uid_raw)
                timestamp = float(timestamp_raw)
                key = (uid, timestamp)
                if current_key is not None and key < current_key:
                    raise ValueError("prepared listens must be sorted by uid and timestamp")
                if current_key is not None and key != current_key:
                    flush_group()
                    current_group.clear()
                    if current_user is not None and uid != current_user.uid:
                        user_rows.append(current_user)
                        current_user = None
                if current_user is None:
                    current_user = UserSequence(uid=uid, rm_user_partition=None if user_partition_raw is None else str(user_partition_raw))
                elif current_user.rm_user_partition is None and user_partition_raw is not None:
                    current_user.rm_user_partition = str(user_partition_raw)
                current_key = key
                current_group.append((
                    uid,
                    timestamp,
                    int(token_raw),
                    int(item_raw),
                    bool(organic_raw),
                    str(split_raw),
                    "ambiguous" if int(tie_count_raw or 0) > 1 else str(train_subsplit_raw) if train_subsplit_raw is not None else None,
                    None if user_partition_raw is None else str(user_partition_raw),
                    None if ratio_raw is None else float(ratio_raw),
                    None if duration_raw is None else float(duration_raw),
                ))
                events += 1
                if events >= next_progress:
                    print(f"Indexed {events:,} / {parquet.metadata.num_rows:,} listen rows", flush=True)
                    next_progress += 5_000_000
        flush_group()
        if current_user is not None:
            user_rows.append(current_user)
        return cls(users=user_rows, events=events, tie_blocks=tie_blocks, tied_events=tied_events)

    def sample_sft_windows(
        self,
        max_targets: int,
        history_length: int = 200,
        n_positions: int = 256,
        seed: int = 17,
        positive_only: bool = False,
    ) -> Iterator[tuple[list[int], list[bool]]]:
        """Yield target-stratified windows with full causal history and no duplicate targets.

        With ``positive_only``, supervise only targets whose observed raw completion
        ratio is strictly greater than 0.5. All events remain in the input history.
        """
        if max_targets < 1 or history_length < 1 or n_positions < history_length + 2:
            raise ValueError("invalid SFT target budget or sequence limits")
        if positive_only:
            counts = [sum(user.ratio_raw[position] > 0.5 for position in user.train_targets) for user in self.users]
        else:
            counts = [len(user.train_targets) for user in self.users]
        total = sum(counts)
        if total == 0:
            return
        budget = min(max_targets, total)
        exact = [budget * count / total for count in counts]
        quotas = [min(count, int(value)) for count, value in zip(counts, exact)]
        remaining = budget - sum(quotas)
        fractional_order = sorted(range(len(counts)), key=lambda i: (-(exact[i] - int(exact[i])), self.users[i].uid))
        for user_index in fractional_order:
            if remaining <= 0:
                break
            if quotas[user_index] < counts[user_index]:
                quotas[user_index] += 1
                remaining -= 1

        rng = random.Random(seed)
        target_span = n_positions - history_length - 1
        for user, quota in zip(self.users, quotas):
            if quota <= 0:
                continue
            eligible_targets = (
                [int(position) for position in user.train_targets if user.ratio_raw[position] > 0.5]
                if positive_only else user.train_targets
            )
            if quota == len(eligible_targets):
                selected = [int(position) for position in eligible_targets]
            else:
                selected = sorted(int(eligible_targets[index]) for index in rng.sample(range(len(eligible_targets),), quota))
            selected_by_bucket: dict[int, set[int]] = {}
            for position in selected:
                bucket = (position - 1) // target_span
                selected_by_bucket.setdefault(bucket, set()).add(position)
            train_end = next(
                (i for i, split_code in enumerate(user.split_codes) if split_code != SPLIT_CODES["train"]),
                len(user.tokens),
            )
            for bucket, selected_positions in sorted(selected_by_bucket.items()):
                start = 1 + bucket * target_span
                stop = min(train_end, start + target_span)
                context_start = max(0, start - history_length)
                tokens = [BOS_ID, *user.tokens[context_start:stop]]
                mask = [False] * len(tokens)
                for position in selected_positions:
                    local_target_position = 1 + position - context_start
                    mask[local_target_position] = True
                if len(tokens) > n_positions:
                    raise AssertionError("SFT window exceeded configured n_positions")
                yield tokens, mask

    def sample_sft_target_index(
        self,
        max_targets: int,
        history_events: int = 200,
        seed: int = 17,
        positive_only: bool = False,
    ) -> SFTTargetIndex:
        """Sample distinct target examples with an exact, capped causal history.

        Target quotas match the existing target-stratified sampler: largest-
        remainder allocation proportional to each user's eligible target count,
        followed by uniform sampling without replacement within each user.
        Unlike packed bucket windows, every selected label is trained from the
        same BOS plus up to ``history_events`` preceding events used at serving.
        """
        if max_targets < 1 or history_events < 1:
            raise ValueError("SFT target budget and history_events must be positive")
        if positive_only:
            counts = [
                sum(float(user.ratio_raw[position]) > 0.5 for position in user.train_targets)
                for user in self.users
            ]
        else:
            counts = [len(user.train_targets) for user in self.users]
        total = sum(counts)
        if total == 0:
            return SFTTargetIndex(array("I"), array("I"), history_events, seed, positive_only)

        budget = min(int(max_targets), total)
        exact = [budget * count / total for count in counts]
        quotas = [min(count, int(value)) for count, value in zip(counts, exact)]
        remaining = budget - sum(quotas)
        fractional_order = sorted(
            range(len(counts)),
            key=lambda i: (-(exact[i] - int(exact[i])), self.users[i].uid),
        )
        for user_index in fractional_order:
            if remaining <= 0:
                break
            if quotas[user_index] < counts[user_index]:
                quotas[user_index] += 1
                remaining -= 1

        rng = random.Random(int(seed))
        user_indices = array("I")
        positions = array("I")
        if user_indices.itemsize != 4 or positions.itemsize != 4:
            raise RuntimeError("compact SFT index requires 32-bit unsigned array support")
        for user_index, (user, quota) in enumerate(zip(self.users, quotas)):
            if quota <= 0:
                continue
            if positive_only:
                population = [
                    int(position) for position in user.train_targets
                    if float(user.ratio_raw[position]) > 0.5
                ]
            else:
                population = user.train_targets
            if len(population) != counts[user_index]:
                raise AssertionError("eligible SFT target count changed while building index")
            if quota == len(population):
                selected = [int(position) for position in population]
            else:
                chosen = sorted(rng.sample(range(len(population)), quota))
                selected = [int(population[index]) for index in chosen]
            user_indices.extend([user_index] * len(selected))
            positions.extend(selected)

        if len(positions) != budget:
            raise AssertionError(f"SFT target sampler selected {len(positions)} != {budget}")
        return SFTTargetIndex(user_indices, positions, history_events, int(seed), positive_only)

    def sample_rl_anchors(self, count: int, seed: int = 17) -> list[tuple[int, int]]:
        """Sample anchors uniform over users, then uniform over that user's anchors."""
        if count < 1:
            return []
        eligible_users = [index for index, user in enumerate(self.users) if len(user.rl_anchors) > 0]
        if not eligible_users:
            return []
        rng = random.Random(seed)
        result = []
        for _ in range(count):
            user_index = rng.choice(eligible_users)
            position = rng.choice(self.users[user_index].rl_anchors)
            result.append((user_index, int(position)))
        return result

    def evaluation_targets(
        self,
        split: str,
        per_user_limit: int | None = 10,
        seed: int = 17,
    ) -> list[tuple[int, int]]:
        if split not in SPLIT_CODES:
            raise ValueError(f"unknown evaluation split: {split}")
        targets: list[tuple[int, int]] = []
        for user_index, user in enumerate(self.users):
            positions = [
                position for position, code in enumerate(user.split_codes)
                if code == SPLIT_CODES[split] and user.recommendable_events[position]
            ]
            if per_user_limit is not None and len(positions) > per_user_limit:
                rng = random.Random(f"{seed}:{user.uid}:{split}")
                positions = sorted(rng.sample(positions, per_user_limit))
            targets.extend((user_index, position) for position in positions)
        return targets

    def histories_for(self, anchors: Sequence[tuple[int, int]], history_length: int = 200) -> list[list[int]]:
        return [self.users[user_index].history(position, history_length) for user_index, position in anchors]
