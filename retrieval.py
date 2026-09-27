"""Local sparse candidate retrieval for short Russian service queries.

The title index uses character n-grams, which tolerate inflection and small
spelling differences. The body index uses words from descriptions and service
parameters, which adds recall when the title is too generic. Both indices are
fitted only on the item corpus; no benchmark labels are used.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def normalize_text(value: str | None) -> str:
    """Make Russian text consistent across queries and item fields."""
    return " ".join((value or "").casefold().replace("ё", "е").split())


class RetrievalIndex:
    def __init__(self, items: Iterable[dict]):
        ids: list[str] = []
        locations: list[int] = []
        titles: list[str] = []
        bodies: list[str] = []
        microcats: list[int] = []
        attributes: list[list[float]] = []
        microcat_rows: dict[int, list[int]] = defaultdict(list)
        seen: set[str] = set()

        for item in items:
            item_id = item["item_id"]
            if item_id in seen:
                continue
            seen.add(item_id)
            ids.append(item_id)
            item_location = item["item_location_id"]
            locations.append(-1 if item_location is None else item_location)
            microcat = item.get("item_microcat_id")
            if microcat is None:
                microcat = -1
            microcats.append(microcat)
            microcat_rows[microcat].append(len(ids) - 1)
            titles.append(normalize_text(item["item_title_raw"]))
            # Long descriptions often contain repeated tags and boilerplate.
            # Capping them reduces memory and keeps the service itself salient.
            description = normalize_text(item["item_description_raw"])[:1400]
            parameters = normalize_text(item["item_infm_params_text"])[:600]
            bodies.append(f"{titles[-1]} {description} {parameters}")
            attributes.append(
                [
                    float(item.get("item_rating_reviews_count") or 0),
                    float(item.get("item_rating") or 0),
                    float(item.get("item_price") or 0),
                    float(bool(item.get("item_is_phone_hidden"))),
                    float(bool(item.get("item_is_message_forbidden"))),
                    float(len(item["item_title_raw"] or "")),
                    float(len(item["item_description_raw"] or "")),
                ]
            )

        if not ids:
            raise ValueError("The item corpus is empty")

        self.item_ids = ids
        self.locations = np.asarray(locations, dtype=np.int64)
        self.microcats = np.asarray(microcats, dtype=np.int64)
        self.attributes = np.nan_to_num(
            np.asarray(attributes, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        self.microcat_rows = {
            category: np.asarray(rows, dtype=np.int32)
            for category, rows in microcat_rows.items()
        }
        self.title_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1 if len(ids) < 10 else 2,
            max_features=180_000,
            dtype=np.float32,
            sublinear_tf=True,
        )
        self.title_matrix = self.title_vectorizer.fit_transform(titles).tocsc()
        self.body_vectorizer = TfidfVectorizer(
            token_pattern=r"(?u)\b\w\w+\b",
            min_df=1 if len(ids) < 10 else 2,
            max_features=180_000,
            dtype=np.float32,
            sublinear_tf=True,
        )
        self.body_matrix = self.body_vectorizer.fit_transform(bodies).tocsc()

    @staticmethod
    def _cosine_scores(matrix, query_vector) -> np.ndarray:
        """Accumulate only postings touched by the query's sparse terms."""
        result = np.zeros(matrix.shape[0], dtype=np.float32)
        query_vector = query_vector.tocoo()
        for column, weight in zip(query_vector.col, query_vector.data):
            start, end = matrix.indptr[column : column + 2]
            result[matrix.indices[start:end]] += matrix.data[start:end] * weight
        return result

    def score_components(self, query: str) -> tuple[np.ndarray, np.ndarray]:
        """Return independent cosine scores for offline weight tuning."""
        query = normalize_text(query)
        title_query = self.title_vectorizer.transform([query])
        body_query = self.body_vectorizer.transform([query])
        title = self._cosine_scores(self.title_matrix, title_query)
        body = self._cosine_scores(self.body_matrix, body_query)
        return title, body

    def rank(
        self,
        title_scores: np.ndarray,
        body_scores: np.ndarray,
        location_id: int,
        limit: int = 50,
        title_weight: float = 0.6,
        location_multiplier: float = 1.7,
        location_weights: np.ndarray | None = None,
        category_probabilities: dict[int, float] | None = None,
        category_boost: float = 4.0,
    ) -> list[str]:
        """Combine lexical evidence and prefer exact-city matches.

        The multiplier is soft: a strong text match outside the city can still
        enter the result, which matters for remote and regional services.
        """
        scores = title_weight * title_scores + (1 - title_weight) * body_scores
        scores = scores.copy()
        if location_weights is None:
            scores[self.locations == location_id] *= location_multiplier
        else:
            scores *= location_weights
        for category, probability in (category_probabilities or {}).items():
            rows = self.microcat_rows.get(category)
            if rows is not None:
                scores[rows] *= 1 + (category_boost - 1) * probability
        matched = np.flatnonzero(scores > 0)
        if not len(matched):
            return []
        if len(matched) > limit:
            top = np.argpartition(scores[matched], -limit)[-limit:]
            matched = matched[top]
        # Item index is a deterministic tie breaker.
        order = np.lexsort((matched, -scores[matched]))
        return [self.item_ids[i] for i in matched[order]]

    def search(self, query: str, location_id: int, limit: int = 50) -> list[str]:
        return self.rank(*self.score_components(query), location_id, limit)
