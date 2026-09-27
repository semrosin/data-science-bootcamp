"""Evaluate conservative reranking for benchmark-like known query texts."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from priors import CategoryPrior, LocationPrior, collect_training_counts
from ranker import rank_candidates, train_ranker
from retrieval import RetrievalIndex, normalize_text
from solution import (
    ITEM_COLUMNS,
    KNOWN_RANKER_REPLACEMENTS,
    LOCATION_EXPONENT,
    QUERY_COLUMNS,
    RANKER_TRAIN_SIZE,
    TRAIN_PRIOR_COLUMNS,
    iter_parquet_rows,
    merge_rankings,
    retrieve_with_priors,
    validation_pairs,
)


def frequency_bucket(count: int) -> int:
    """Group query texts by their observed number of training choices."""
    return 0 if count == 1 else 1 if count <= 5 else 2


def weighted_recall(
    scores: list[tuple[tuple[int, bool], float]],
    benchmark_counts: Counter[tuple[int, bool]],
) -> float:
    """Weight frequency/filter groups as they occur among known benchmark texts."""
    grouped: dict[tuple[int, bool], list[float]] = defaultdict(list)
    for group, score in scores:
        grouped[group].append(score)
    missing = benchmark_counts.keys() - grouped.keys()
    if missing:
        raise ValueError(f"missing validation groups: {sorted(missing)}")
    total = sum(benchmark_counts.values())
    return sum(
        count / total * float(np.mean(grouped[group]))
        for group, count in benchmark_counts.items()
    )


def compare_pair(index, category_prior, location_prior, model, key, relevant):
    query, location, filter_text = key
    baseline = retrieve_with_priors(
        index, category_prior, location_prior, query, location
    )
    reranked = rank_candidates(
        index,
        category_prior,
        location_prior,
        model,
        query,
        location,
        filter_text,
        limit=100,
        exclude_exact=False,
    )
    combined = merge_rankings(
        baseline, reranked, replacements=KNOWN_RANKER_REPLACEMENTS
    )
    return (
        len(set(baseline) & relevant) / len(relevant),
        len(set(combined) & relevant) / len(relevant),
        len(baseline),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    args = parser.parse_args()
    dataset = args.dataset

    index = RetrievalIndex(
        iter_parquet_rows(dataset / "benchmark_items.parquet", ITEM_COLUMNS)
    )
    item_ids = set(index.item_ids)
    categories, transitions = collect_training_counts(
        iter_parquet_rows(dataset / "train.parquet", TRAIN_PRIOR_COLUMNS), item_ids
    )
    category_prior = CategoryPrior(categories)
    location_prior = LocationPrior(
        index.locations, transitions, multiplier=30, exponent=LOCATION_EXPONENT
    )
    benchmark = list(
        iter_parquet_rows(dataset / "benchmark_queries.parquet", QUERY_COLUMNS)
    )
    known = [
        row
        for row in benchmark
        if normalize_text(row["search_query"]) in category_prior.counts
    ]
    known_texts = {normalize_text(row["search_query"]) for row in known}
    dev = validation_pairs(
        dataset / "train.parquet", item_ids, 800, seed=101, excluded_texts=known_texts
    )
    test = validation_pairs(
        dataset / "train.parquet",
        item_ids,
        800,
        seed=202,
        excluded_texts=known_texts | {key[0] for key, _ in dev},
    )
    reserved = known_texts | {key[0] for key, _ in [*dev, *test]}
    training = validation_pairs(
        dataset / "train.parquet",
        item_ids,
        RANKER_TRAIN_SIZE,
        seed=55,
        excluded_texts=reserved,
    )
    print(f"Training on {len(training)} texts; evaluating {len(dev)} + {len(test)}")
    model = train_ranker(index, category_prior, location_prior, training)
    if model is None:
        raise ValueError("No ranker training pairs")

    frequencies = {
        text: sum(counts.values()) for text, counts in category_prior.counts.items()
    }
    benchmark_counts = Counter(
        (
            frequency_bucket(frequencies[normalize_text(row["search_query"])]),
            bool(row["search_infm_params_text"]),
        )
        for row in known
    )
    for name, pairs in (("dev", dev), ("test", test)):
        base_scores = []
        new_scores = []
        full_base_scores = []
        full_new_scores = []
        for key, relevant in pairs:
            base, new, baseline_size = compare_pair(
                index, category_prior, location_prior, model, key, relevant
            )
            group = (frequency_bucket(frequencies[key[0]]), bool(key[2]))
            base_scores.append((group, base))
            new_scores.append((group, new))
            if baseline_size == 50:
                full_base_scores.append((group, base))
                full_new_scores.append((group, new))
        print(
            name,
            json.dumps(
                {
                    "queries": len(pairs),
                    "baseline": weighted_recall(base_scores, benchmark_counts),
                    "candidate": weighted_recall(new_scores, benchmark_counts),
                    "full_baseline_queries": len(full_base_scores),
                    "full_baseline": weighted_recall(
                        full_base_scores, benchmark_counts
                    ),
                    "full_candidate": weighted_recall(
                        full_new_scores, benchmark_counts
                    ),
                }
            ),
            flush=True,
        )

    exact_keys = {
        (
            normalize_text(row["search_query"]),
            row["search_location_id"],
            row["search_infm_params_text"] or "",
        )
        for row in known
    }
    exact_groups: dict[tuple[str, int, str], set[str]] = defaultdict(set)
    for row in iter_parquet_rows(
        dataset / "train.parquet",
        ["search_query", "search_location_id", "search_infm_params_text", "item_id"],
    ):
        key = (
            normalize_text(row["search_query"]),
            row["search_location_id"],
            row["search_infm_params_text"] or "",
        )
        if key in exact_keys and row["item_id"] in item_ids:
            exact_groups[key].add(row["item_id"])
    direct_scores = []
    for row in known:
        key = (
            normalize_text(row["search_query"]),
            row["search_location_id"],
            row["search_infm_params_text"] or "",
        )
        relevant = exact_groups.get(key)
        if relevant:
            direct_scores.append(
                compare_pair(index, category_prior, location_prior, model, key, relevant)
            )
    direct = np.asarray(direct_scores)
    full_direct = direct[direct[:, 2] == 50]
    print(
        "same_text_location_filter",
        json.dumps(
            {
                "queries": len(direct),
                "baseline": float(direct[:, 0].mean()),
                "candidate": float(direct[:, 1].mean()),
                "full_baseline_queries": len(full_direct),
                "full_baseline": float(full_direct[:, 0].mean()),
                "full_candidate": float(full_direct[:, 1].mean()),
            }
        ),
    )


if __name__ == "__main__":
    main()
