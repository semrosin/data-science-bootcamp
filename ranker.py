"""Supervised reranking of sparse text candidates using historical choices."""

from __future__ import annotations

import random
from collections.abc import Iterable

import numpy as np
from lightgbm import LGBMClassifier

from priors import CategoryPrior, LocationPrior
from retrieval import RetrievalIndex, normalize_text

FEATURE_NAMES = (
    "title",
    "body",
    "base",
    "adjusted",
    "location_weight",
    "same_location",
    "category_probability",
    "review_log",
    "rating",
    "price_log",
    "phone_hidden",
    "message_forbidden",
    "title_length",
    "description_length",
    "query_words",
    "has_filter",
    "base_rank",
    "filter_body_score",
)
CANDIDATE_LIMIT = 300
FILTER_CANDIDATE_LIMIT = 100


def candidate_features(
    index: RetrievalIndex,
    category_prior: CategoryPrior,
    location_prior: LocationPrior,
    query: str,
    location_id: int,
    filter_text: str,
    *,
    exclude_exact: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the best lexical candidates and finite reranking features."""
    title, body = index.score_components(query)
    base = 0.2 * title + 0.8 * body
    geo = location_prior.weights(location_id)
    category_prob = np.zeros(len(index.item_ids), dtype=np.float32)
    for category, probability in category_prior.predict(
        query, exclude_exact=exclude_exact
    ).items():
        rows = index.microcat_rows.get(category)
        if rows is not None:
            category_prob[rows] = probability
    scores = base * geo * (1 + 3 * category_prob)
    if filter_text:
        filter_vector = index.body_vectorizer.transform([normalize_text(filter_text)])
        filter_scores = index._cosine_scores(index.body_matrix, filter_vector)
    else:
        filter_scores = np.zeros(len(index.item_ids), dtype=np.float32)
    matched = np.flatnonzero(scores > 0)
    if len(matched) > CANDIDATE_LIMIT:
        top = np.argpartition(scores[matched], -CANDIDATE_LIMIT)[-CANDIDATE_LIMIT:]
        matched = matched[top]
    if filter_text:
        filter_candidate_scores = (
            filter_scores * geo * (1 + 3 * category_prob) * np.sqrt(base)
        )
        filter_matches = np.flatnonzero(filter_candidate_scores > 0)
        if len(filter_matches) > FILTER_CANDIDATE_LIMIT:
            top = np.argpartition(
                filter_candidate_scores[filter_matches], -FILTER_CANDIDATE_LIMIT
            )[-FILTER_CANDIDATE_LIMIT:]
            filter_matches = filter_matches[top]
        matched = np.union1d(matched, filter_matches)
    selected = matched[np.lexsort((matched, -scores[matched]))]
    if not len(selected):
        return selected, np.empty((0, len(FEATURE_NAMES)), dtype=np.float32), scores[selected]

    attributes = index.attributes[selected]
    features = np.column_stack(
        [
            title[selected],
            body[selected],
            base[selected],
            np.log1p(scores[selected]),
            geo[selected],
            (index.locations[selected] == location_id).astype(np.float32),
            category_prob[selected],
            np.log1p(np.maximum(attributes[:, 0], 0)),
            attributes[:, 1],
            np.log1p(np.maximum(attributes[:, 2], 0)),
            attributes[:, 3],
            attributes[:, 4],
            np.log1p(np.maximum(attributes[:, 5], 0)),
            np.log1p(np.maximum(attributes[:, 6], 0)),
            np.full(len(selected), len(normalize_text(query).split())),
            np.full(len(selected), bool(filter_text)),
            np.arange(len(selected)) / (CANDIDATE_LIMIT + FILTER_CANDIDATE_LIMIT),
            filter_scores[selected],
        ]
    ).astype(np.float32)
    return selected, features, scores[selected]


def train_ranker(
    index: RetrievalIndex,
    category_prior: CategoryPrior,
    location_prior: LocationPrior,
    pairs: Iterable[tuple[tuple[str, int, str], set[str]]],
) -> LGBMClassifier | None:
    """Train on distinct queries with positives and sampled negatives."""
    rng = random.Random(55)
    training_features: list[np.ndarray] = []
    training_labels: list[np.ndarray] = []
    for (query, location_id, filter_text), relevant in pairs:
        selected, features, _ = candidate_features(
            index, category_prior, location_prior, query, location_id, filter_text
        )
        labels = np.array(
            [index.item_ids[position] in relevant for position in selected], dtype=np.int8
        )
        positive = np.flatnonzero(labels)
        if not len(positive):
            continue
        negative = np.flatnonzero(labels == 0)
        hard = negative[negative < 50]
        other = negative[negative >= 50]
        sampled = rng.sample(other.tolist(), min(100, len(other)))
        chosen = np.r_[positive, hard, sampled]
        training_features.append(features[chosen])
        training_labels.append(labels[chosen])
    if not training_features:
        return None

    features = np.vstack(training_features)
    labels = np.concatenate(training_labels)
    model = LGBMClassifier(
        n_estimators=250,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=100,
        colsample_bytree=0.9,
        subsample=0.8,
        subsample_freq=1,
        reg_lambda=2,
        n_jobs=4,
        verbosity=-1,
    )
    model.fit(features, labels)
    return model


def rank_candidates(
    index: RetrievalIndex,
    category_prior: CategoryPrior,
    location_prior: LocationPrior,
    model: LGBMClassifier,
    query: str,
    location_id: int,
    filter_text: str,
    *,
    limit: int = 50,
    exclude_exact: bool = True,
) -> list[str]:
    selected, features, _ = candidate_features(
        index,
        category_prior,
        location_prior,
        query,
        location_id,
        filter_text,
        exclude_exact=exclude_exact,
    )
    if not len(selected):
        return []
    probabilities = model.predict_proba(features)[:, 1]
    ranking = np.argsort(-probabilities, kind="stable")[:limit]
    return [index.item_ids[selected[position]] for position in ranking]
