"""Compare click-history signals on two disjoint historical query splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from click_memory import ClickMemory
from priors import CategoryPrior, LocationPrior, collect_training_counts
from ranker import candidate_features, train_ranker
from retrieval import RetrievalIndex
from solution import (
    ITEM_COLUMNS,
    LOCATION_EXPONENT,
    QUERY_COLUMNS,
    RANKER_TRAIN_SIZE,
    TRAIN_PRIOR_COLUMNS,
    iter_parquet_rows,
    validation_pairs,
)

MODES = {
    "baseline": (False, False),
    "count": (False, True),
    "candidates": (True, False),
    "both": (True, True),
}


def evaluate(index, category_prior, location_prior, memory, model, pairs, flags, share):
    totals = {False: [0.0, 0.0, 0], True: [0.0, 0.0, 0]}
    for (query, location_id, filter_text), relevant in pairs:
        selected, features, _ = candidate_features(
            index,
            category_prior,
            location_prior,
            query,
            location_id,
            filter_text,
            click_memory=memory,
            use_history_candidates=flags[0],
            use_history_count=flags[1],
        )
        candidate_ids = {index.item_ids[position] for position in selected}
        probabilities = model.predict_proba(features)[:, 1]
        best = np.argsort(-probabilities, kind="stable")[:50]
        result_ids = {index.item_ids[selected[position]] for position in best}
        group = totals[bool(filter_text)]
        group[0] += len(result_ids & relevant) / len(relevant)
        group[1] += len(candidate_ids & relevant) / len(relevant)
        group[2] += 1
    plain = totals[False]
    filtered = totals[True]
    return {
        "weighted_recall": (1 - share) * plain[0] / plain[2]
        + share * filtered[0] / filtered[2],
        "weighted_ceiling": (1 - share) * plain[1] / plain[2]
        + share * filtered[1] / filtered[2],
        "plain_recall": plain[0] / plain[2],
        "filtered_recall": filtered[0] / filtered[2],
        "plain_ceiling": plain[1] / plain[2],
        "filtered_ceiling": filtered[1] / filtered[2],
        "plain_queries": plain[2],
        "filtered_queries": filtered[2],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    dataset = args.dataset

    print("Building item and query indexes...", flush=True)
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
    dev = validation_pairs(dataset / "train.parquet", item_ids, 800, seed=101)
    test = validation_pairs(
        dataset / "train.parquet",
        item_ids,
        800,
        seed=202,
        excluded_texts={key[0] for key, _ in dev},
    )
    reserved = {key[0] for key, _ in [*dev, *test]}
    training = validation_pairs(
        dataset / "train.parquet",
        item_ids,
        RANKER_TRAIN_SIZE,
        seed=55,
        excluded_texts=reserved,
    )
    memory = ClickMemory(
        iter_parquet_rows(dataset / "train.parquet", ["search_query", "item_id"]),
        index.item_ids,
        excluded_texts=reserved,
    )
    queries = list(iter_parquet_rows(dataset / "benchmark_queries.parquet", QUERY_COLUMNS))
    filter_share = sum(bool(row["search_infm_params_text"]) for row in queries) / len(
        queries
    )
    print(
        "split",
        json.dumps(
            {
                "dev": len(dev),
                "test": len(test),
                "training": len(training),
                "memory_texts": len(memory.texts),
                "benchmark_filter_share": filter_share,
            }
        ),
        flush=True,
    )
    models = {}
    dev_results = {}
    for name, flags in MODES.items():
        print("Training", name, flush=True)
        model = train_ranker(
            index,
            category_prior,
            location_prior,
            training,
            click_memory=memory,
            use_history_candidates=flags[0],
            use_history_count=flags[1],
        )
        models[name] = model
        dev_results[name] = evaluate(
            index, category_prior, location_prior, memory, model, dev, flags, filter_share
        )
        print("dev", name, json.dumps(dev_results[name]), flush=True)

    chosen = max(MODES, key=lambda name: dev_results[name]["weighted_recall"])
    test_results = {}
    for name in ("baseline", chosen):
        if name in test_results:
            continue
        test_results[name] = evaluate(
            index,
            category_prior,
            location_prior,
            memory,
            models[name],
            test,
            MODES[name],
            filter_share,
        )
        print("test", name, json.dumps(test_results[name]), flush=True)
    dev_gain = (
        dev_results[chosen]["weighted_recall"]
        - dev_results["baseline"]["weighted_recall"]
    )
    ceiling_gain = (
        dev_results[chosen]["weighted_ceiling"]
        - dev_results["baseline"]["weighted_ceiling"]
    )
    test_gain = (
        test_results[chosen]["weighted_recall"]
        - test_results["baseline"]["weighted_recall"]
    )
    accept = (
        chosen != "baseline"
        and dev_gain >= 0.005
        and ceiling_gain > 0
        and test_gain >= -0.002
    )
    print(
        "decision",
        json.dumps(
            {
                "chosen": chosen,
                "dev_gain": dev_gain,
                "ceiling_gain": ceiling_gain,
                "test_gain": test_gain,
                "accept": accept,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
