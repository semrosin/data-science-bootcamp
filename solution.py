"""Build and check a local candidate-generation submission.

Run ``python solution.py --dataset dataset --output answer.csv`` after placing
the three Parquet files from the assignment in ``dataset/``. Proxy validation
uses historical train pairs, not benchmark labels.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

import pyarrow.parquet as pq

from priors import CategoryPrior, LocationPrior, collect_training_counts
from ranker import rank_candidates, train_ranker
from retrieval import RetrievalIndex, normalize_text

ITEM_COLUMNS = [
    "item_id",
    "item_title_raw",
    "item_description_raw",
    "item_infm_params_text",
    "item_location_id",
    "item_microcat_id",
    "item_rating_reviews_count",
    "item_rating",
    "item_price",
    "item_is_phone_hidden",
    "item_is_message_forbidden",
]
QUERY_COLUMNS = [
    "query_id",
    "search_query",
    "search_location_id",
    "search_infm_params_text",
]
TRAIN_PRIOR_COLUMNS = [
    "search_query",
    "search_location_id",
    "item_id",
    "item_location_id",
    "item_microcat_id",
]
ITEM_ID_PATTERN = re.compile(r"[0-9a-f]{16}\Z")
DEFAULT_SETTING = (0.2, 30.0)
CATEGORY_BOOST = 4.0
LOCATION_EXPONENT = 0.4
RANKER_TRAIN_SIZE = 8000
KNOWN_RANKER_REPLACEMENTS = 10


def iter_parquet_rows(path: Path, columns: list[str]) -> Iterator[dict]:
    """Read large Parquet files in batches instead of materializing a table."""
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=2048, columns=columns):
        yield from batch.to_pylist()


def write_answer(path: Path, predictions: Iterable[tuple[str, list[str]]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["query_id", "answer"])
        for query_id, item_ids in predictions:
            writer.writerow([query_id, " ".join(item_ids)])


def merge_rankings(
    baseline: list[str], reranked: list[str], *, replacements: int, limit: int = 50
) -> list[str]:
    """Keep the first baseline items; replace at most N when baseline has 50."""
    preserved = baseline[: max(0, limit - replacements)]
    result = list(preserved)
    if len(result) >= limit:
        return result[:limit]
    seen = set(result)
    for item_id in [*reranked, *baseline[len(preserved) :]]:
        if item_id not in seen:
            result.append(item_id)
            seen.add(item_id)
            if len(result) >= limit:
                break
    return result


def validate_answer(path: Path, query_ids: set[str], item_ids: set[str]) -> int:
    """Enforce the exact submission contract before delivering the CSV."""
    seen_queries: set[str] = set()
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        if next(reader, None) != ["query_id", "answer"]:
            raise ValueError("CSV must have exactly query_id,answer columns")
        for row in reader:
            if len(row) != 2:
                raise ValueError("CSV row must have exactly two columns")
            query_id, answer_text = row
            if query_id not in query_ids:
                raise ValueError(f"unknown query_id: {query_id}")
            if query_id in seen_queries:
                raise ValueError(f"duplicate query_id: {query_id}")
            seen_queries.add(query_id)
            answer = answer_text.split()
            if len(answer) > 50:
                raise ValueError(f"too many items for {query_id}")
            if len(answer) != len(set(answer)):
                raise ValueError(f"duplicate item_id for {query_id}")
            if any(not ITEM_ID_PATTERN.fullmatch(item) for item in answer):
                raise ValueError(f"invalid item_id format for {query_id}")
            if any(item not in item_ids for item in answer):
                raise ValueError(f"unknown item_id for {query_id}")
    if seen_queries != query_ids:
        raise ValueError(f"missing {len(query_ids - seen_queries)} query IDs")
    return len(seen_queries)


def validation_pairs(
    train_path: Path,
    item_ids: set[str],
    size: int,
    seed: int = 42,
    excluded_texts: set[str] | None = None,
):
    """Sample distinct historical query texts with labels in today's corpus.

    A query may have several chosen items; its Recall@50 denominator uses all
    available items for the selected query/location/filter combination.
    """
    groups: dict[tuple[str, int, str], set[str]] = defaultdict(set)
    columns = [
        "search_query",
        "search_location_id",
        "search_infm_params_text",
        "item_id",
    ]
    for row in iter_parquet_rows(train_path, columns):
        query_text = normalize_text(row["search_query"])
        if row["item_id"] in item_ids and query_text:
            key = (
                query_text,
                row["search_location_id"],
                row["search_infm_params_text"] or "",
            )
            groups[key].add(row["item_id"])

    # Each text appears once in the sample, mirroring unique benchmark texts.
    by_text: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    for key in groups:
        by_text[key[0]].append(key)
    rng = random.Random(seed)
    available = sorted(set(by_text) - (excluded_texts or set()))
    texts = rng.sample(available, min(size, len(available)))
    return [(key, groups[key]) for text in texts for key in [rng.choice(by_text[text])]]


def mean_recall_at_50(
    index: RetrievalIndex,
    pairs: list[tuple[tuple[str, int, str], set[str]]],
    setting: tuple[float, float],
) -> float:
    """Average per-query recall on historical positive pairs."""
    if not pairs:
        raise ValueError("Recall@50 needs at least one labeled query")
    total = 0.0
    for (query, location_id, _filters), relevant in pairs:
        title_scores, body_scores = index.score_components(query)
        found = index.rank(
            title_scores,
            body_scores,
            location_id,
            50,
            title_weight=setting[0],
            location_multiplier=setting[1],
        )
        total += len(set(found) & relevant) / len(relevant)
    return total / len(pairs)


def retrieve_with_priors(
    index: RetrievalIndex,
    category_prior: CategoryPrior,
    location_prior: LocationPrior,
    query: str,
    location_id: int,
    exclude_exact: bool = False,
) -> list[str]:
    """Apply text similarity, geographic behavior, and service type together."""
    title, body = index.score_components(query)
    return index.rank(
        title,
        body,
        location_id,
        title_weight=DEFAULT_SETTING[0],
        location_weights=location_prior.weights(location_id),
        category_probabilities=category_prior.predict(query, exclude_exact),
        category_boost=CATEGORY_BOOST,
    )


def evaluate_with_priors(
    index: RetrievalIndex,
    category_prior: CategoryPrior,
    location_prior: LocationPrior,
    pairs: list[tuple[tuple[str, int, str], set[str]]],
) -> float:
    """Exclude each evaluated query's exact training labels from the prior."""
    if not pairs:
        raise ValueError("Recall@50 needs at least one labeled query")
    total = 0.0
    for (query, location_id, _filters), relevant in pairs:
        found = retrieve_with_priors(
            index, category_prior, location_prior, query, location_id, True
        )
        total += len(set(found) & relevant) / len(relevant)
    return total / len(pairs)


def evaluate_ranker(
    index: RetrievalIndex,
    category_prior: CategoryPrior,
    location_prior: LocationPrior,
    model,
    pairs: list[tuple[tuple[str, int, str], set[str]]],
) -> float:
    total = 0.0
    for (query, location_id, filter_text), relevant in pairs:
        found = rank_candidates(
            index, category_prior, location_prior, model, query, location_id, filter_text
        )
        total += len(set(found) & relevant) / len(relevant)
    return total / len(pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("answer.csv"))
    parser.add_argument("--validation-size", type=int, default=1200)
    args = parser.parse_args()
    if args.validation_size < 0:
        parser.error("--validation-size must be non-negative")
    dataset = args.dataset

    print("Building sparse item index...", flush=True)
    index = RetrievalIndex(
        iter_parquet_rows(dataset / "benchmark_items.parquet", ITEM_COLUMNS)
    )
    print(f"Indexed {len(index.item_ids)} items", flush=True)
    print("Learning query and location priors from train.parquet...", flush=True)
    categories, transitions = collect_training_counts(
        iter_parquet_rows(dataset / "train.parquet", TRAIN_PRIOR_COLUMNS),
        set(index.item_ids),
    )
    category_prior = CategoryPrior(categories)
    location_prior = LocationPrior(
        index.locations,
        transitions,
        multiplier=DEFAULT_SETTING[1],
        exponent=LOCATION_EXPONENT,
    )
    heldout = []
    if args.validation_size:
        heldout = validation_pairs(
            dataset / "train.parquet", set(index.item_ids), args.validation_size, seed=23
        )
    training_pairs = validation_pairs(
        dataset / "train.parquet",
        set(index.item_ids),
        RANKER_TRAIN_SIZE,
        seed=55,
        excluded_texts={key[0] for key, _ in heldout},
    )
    print(f"Training reranker on {len(training_pairs)} distinct query texts...", flush=True)
    model = train_ranker(index, category_prior, location_prior, training_pairs)
    if heldout:
        baseline = evaluate_with_priors(index, category_prior, location_prior, heldout)
        reranked = (
            evaluate_ranker(index, category_prior, location_prior, model, heldout)
            if model is not None
            else baseline
        )
        print(
            f"Proxy Recall@50 on {len(heldout)} unseen texts: "
            f"sparse={baseline:.4f}, reranked={reranked:.4f}",
            flush=True,
        )
        training_pairs = validation_pairs(
            dataset / "train.parquet", set(index.item_ids), RANKER_TRAIN_SIZE, seed=55
        )
        model = train_ranker(index, category_prior, location_prior, training_pairs)

    queries = list(
        iter_parquet_rows(dataset / "benchmark_queries.parquet", QUERY_COLUMNS)
    )
    known_texts = {
        normalize_text(row["search_query"])
        for row in queries
        if normalize_text(row["search_query"]) in category_prior.counts
    }
    known_model = None
    if model is not None and known_texts:
        known_training_pairs = validation_pairs(
            dataset / "train.parquet",
            set(index.item_ids),
            RANKER_TRAIN_SIZE,
            seed=55,
            excluded_texts=known_texts,
        )
        print(
            f"Training known-query reranker on {len(known_training_pairs)} "
            "other query texts...",
            flush=True,
        )
        known_model = train_ranker(
            index, category_prior, location_prior, known_training_pairs
        )
    predictions = []
    for row in queries:
        query = row["search_query"] or ""
        if normalize_text(query) in known_texts:
            baseline = retrieve_with_priors(
                index, category_prior, location_prior, query, row["search_location_id"]
            )
            if known_model is not None:
                reranked = rank_candidates(
                    index,
                    category_prior,
                    location_prior,
                    known_model,
                    query,
                    row["search_location_id"],
                    row["search_infm_params_text"] or "",
                    limit=100,
                    exclude_exact=False,
                )
                selected = merge_rankings(
                    baseline, reranked, replacements=KNOWN_RANKER_REPLACEMENTS
                )
            else:
                selected = baseline
        elif model is not None:
            selected = rank_candidates(
                index,
                category_prior,
                location_prior,
                model,
                query,
                row["search_location_id"],
                row["search_infm_params_text"] or "",
            )
        else:
            selected = retrieve_with_priors(
                index, category_prior, location_prior, query, row["search_location_id"]
            )
        predictions.append((row["query_id"], selected))
    write_answer(args.output, predictions)
    count = validate_answer(
        args.output, {q["query_id"] for q in queries}, set(index.item_ids)
    )
    print(f"Validated {count} query rows in {args.output}", flush=True)


if __name__ == "__main__":
    main()
