from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence


@dataclass
class PopularityBaseline:
    counts: Counter[int] = field(default_factory=Counter)

    def fit(self, item_tokens: Iterable[int]) -> "PopularityBaseline":
        self.counts = Counter(int(token) for token in item_tokens if int(token) >= 3)
        return self

    def score(self, catalog_tokens: Sequence[int]) -> list[float]:
        return [float(self.counts.get(int(token), 0)) for token in catalog_tokens]


@dataclass
class FirstOrderMarkovBaseline:
    alpha: float = 10.0
    item_counts: Counter[int] = field(default_factory=Counter)
    transitions: dict[int, Counter[int]] = field(default_factory=dict)
    total_items: int = 0

    def fit(self, sequences: Iterable[Sequence[int]]) -> "FirstOrderMarkovBaseline":
        self.item_counts = Counter()
        self.transitions = {}
        for sequence in sequences:
            clean = [int(token) for token in sequence]
            self.item_counts.update(token for token in clean if token >= 3)
            for previous, current in zip(clean[:-1], clean[1:]):
                if previous >= 3 and current >= 3:
                    self.transitions.setdefault(previous, Counter())[current] += 1
        self.total_items = sum(self.item_counts.values())
        return self

    def score(self, history: Sequence[int], catalog_tokens: Sequence[int]) -> list[float]:
        previous = next((int(token) for token in reversed(history) if int(token) >= 3), None)
        global_denominator = self.total_items + self.alpha * max(1, len(catalog_tokens))
        transition_counts = self.transitions.get(previous, Counter()) if previous is not None else Counter()
        transition_total = sum(transition_counts.values())
        return [
            (transition_counts.get(int(token), 0) + self.alpha * self.item_counts.get(int(token), 0) / max(1, self.total_items))
            / max(1e-12, transition_total + self.alpha)
            if transition_total or self.total_items
            else 0.0
            for token in catalog_tokens
        ]
