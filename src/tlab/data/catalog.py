from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable

PAD_ID = 0
BOS_ID = 1
OOV_ID = 2
FIRST_ITEM_ID = 3


@dataclass(frozen=True)
class Catalog:
    item_ids: tuple[int, ...]
    training_counts: tuple[int, ...]

    @property
    def item_to_token(self) -> dict[int, int]:
        return {item_id: FIRST_ITEM_ID + index for index, item_id in enumerate(self.item_ids)}

    @property
    def size(self) -> int:
        return len(self.item_ids)

    @property
    def vocabulary_size(self) -> int:
        return FIRST_ITEM_ID + self.size

    @property
    def sha256(self) -> str:
        payload = "".join(f"{item}\t{count}\n" for item, count in zip(self.item_ids, self.training_counts))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def encode(self, item_id: int) -> int:
        return self.item_to_token.get(int(item_id), OOV_ID)


def top_catalog(counts: Iterable[tuple[int, int]], limit: int) -> Catalog:
    """Order by descending training frequency, then ascending raw item ID."""
    if limit < 1:
        raise ValueError("catalog limit must be positive")
    ordered = sorted(((int(item), int(count)) for item, count in counts), key=lambda row: (-row[1], row[0]))[:limit]
    return Catalog(tuple(item for item, _ in ordered), tuple(count for _, count in ordered))


def write_catalog_json(catalog: Catalog, path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            {"special_tokens": {"pad": PAD_ID, "bos": BOS_ID, "oov": OOV_ID},
             "sha256": catalog.sha256,
             "items": [{"item_id": item, "token_id": FIRST_ITEM_ID + i, "training_count": count}
                       for i, (item, count) in enumerate(zip(catalog.item_ids, catalog.training_counts))]},
            handle,
            indent=2,
        )
