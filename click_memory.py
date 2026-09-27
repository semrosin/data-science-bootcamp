"""Historical query-to-item choices for unseen service searches."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from retrieval import normalize_text


class ClickMemory:
    def __init__(
        self,
        rows: Iterable[dict],
        item_ids: list[str],
        excluded_texts: set[str] | None = None,
    ) -> None:
        positions = {item_id: position for position, item_id in enumerate(item_ids)}
        excluded = excluded_texts or set()
        counts: dict[str, Counter[int]] = defaultdict(Counter)
        total_counts = np.zeros(len(item_ids), dtype=np.int32)
        for row in rows:
            text = normalize_text(row["search_query"])
            position = positions.get(row["item_id"])
            if not text or text in excluded or position is None:
                continue
            counts[text][position] += 1
            total_counts[position] += 1
        self.counts = counts
        self.total_counts = total_counts
        self.texts = sorted(counts)
        self.text_to_index = {text: i for i, text in enumerate(self.texts)}
        self.vectorizer = None
        self.matrix = None
        if self.texts:
            self.vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=1 if len(self.texts) < 10 else 2,
                max_features=100_000,
                dtype=np.float32,
            )
            self.matrix = self.vectorizer.fit_transform(self.texts).tocsr()

    def item_counts(
        self, selected: np.ndarray, query: str, *, exclude_exact: bool = False
    ) -> np.ndarray:
        result = self.total_counts[selected].copy()
        if exclude_exact:
            own = self.counts.get(normalize_text(query), {})
            for index, position in enumerate(selected):
                result[index] -= own.get(int(position), 0)
        return result

    def query_scores(self, query: str, *, exclude_exact: bool = False) -> dict[int, float]:
        query = normalize_text(query)
        if not query or self.vectorizer is None:
            return {}
        query_vector = self.vectorizer.transform([query])
        similarities = (query_vector @ self.matrix.T).toarray().ravel()
        if exclude_exact and query in self.text_to_index:
            similarities[self.text_to_index[query]] = 0
        positive = np.flatnonzero(similarities > 0)
        if len(positive) > 20:
            top = np.argpartition(similarities[positive], -20)[-20:]
            positive = positive[top]
        scores: dict[int, float] = defaultdict(float)
        for index in positive:
            weight = float(similarities[index]) ** 4
            for position, count in self.counts[self.texts[index]].items():
                scores[position] += weight * min(count, 10)
        return dict(scores)
