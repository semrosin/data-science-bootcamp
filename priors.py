"""Learn query-category and geographic priors from historical chosen items.

The category prior transfers service types between similar query texts. The
geographic prior learns which item locations users choose from each search
location. Benchmark item IDs are excluded from geographic transition counts so
that offline validation cannot memorize a tested query-item pair.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from retrieval import normalize_text


def collect_training_counts(rows, corpus_ids: set[str]):
    categories: dict[str, Counter[int]] = defaultdict(Counter)
    locations: dict[int, Counter[int]] = defaultdict(Counter)
    for row in rows:
        text = normalize_text(row["search_query"])
        category = row["item_microcat_id"]
        if text and category is not None:
            categories[text][category] += 1
        search_location = row["search_location_id"]
        item_location = row["item_location_id"]
        if (
            row["item_id"] not in corpus_ids
            and search_location is not None
            and item_location is not None
        ):
            locations[search_location][item_location] += 1
    return categories, locations


class CategoryPrior:
    def __init__(self, counts: dict[str, Counter[int]]):
        if not counts:
            raise ValueError("No query-category training pairs")
        self.counts = counts
        self.texts = sorted(counts)
        self.text_to_index = {text: i for i, text in enumerate(self.texts)}
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1 if len(self.texts) < 10 else 2,
            max_features=100_000,
            dtype=np.float32,
        )
        self.matrix = self.vectorizer.fit_transform(self.texts).tocsc()

    def predict(self, query: str, exclude_exact: bool = False) -> dict[int, float]:
        """Return probabilities for the five strongest service categories.

        ``exclude_exact`` removes the query's own training labels during
        offline evaluation. Inference keeps exact matches when available.
        """
        query = normalize_text(query)
        if not query:
            return {}
        vector = self.vectorizer.transform([query]).tocoo()
        similarities = np.zeros(len(self.texts), dtype=np.float32)
        for column, weight in zip(vector.col, vector.data):
            start, end = self.matrix.indptr[column : column + 2]
            similarities[self.matrix.indices[start:end]] += (
                self.matrix.data[start:end] * weight
            )
        if exclude_exact and query in self.text_to_index:
            similarities[self.text_to_index[query]] = 0
        candidate_indices = np.flatnonzero(similarities > 0)
        if not len(candidate_indices):
            return {}
        if len(candidate_indices) > 20:
            top = np.argpartition(similarities[candidate_indices], -20)[-20:]
            candidate_indices = candidate_indices[top]

        votes: Counter[int] = Counter()
        for index in candidate_indices:
            counts = self.counts[self.texts[index]]
            weight = float(similarities[index]) ** 4
            total = sum(counts.values())
            for category, count in counts.items():
                votes[category] += weight * count / total

        leaders = votes.most_common(5)
        total = sum(vote for _, vote in leaders)
        return {category: vote / total for category, vote in leaders}


class LocationPrior:
    def __init__(
        self,
        item_locations: np.ndarray,
        transitions: dict[int, Counter[int]],
        multiplier: float = 30.0,
        exponent: float = 0.4,
    ):
        self.unique_locations, self.inverse = np.unique(
            item_locations, return_inverse=True
        )
        self.location_to_index = {
            int(location): index for index, location in enumerate(self.unique_locations)
        }
        self.transitions = transitions
        self.multiplier = multiplier
        self.exponent = exponent

    def weights(self, search_location: int) -> np.ndarray:
        """Return one geographic multiplier per corpus item."""
        weights = np.ones(len(self.unique_locations), dtype=np.float32)
        counts = self.transitions.get(search_location, {})
        maximum = max(counts.values(), default=1)
        for destination, count in counts.items():
            position = self.location_to_index.get(destination)
            if position is not None:
                weights[position] = (
                    1 + (self.multiplier - 1) * (count / maximum) ** self.exponent
                )
        # The exact city is a strong match even when history is sparse.
        position = self.location_to_index.get(search_location)
        if position is not None:
            weights[position] = max(weights[position], self.multiplier)
        return weights[self.inverse]
