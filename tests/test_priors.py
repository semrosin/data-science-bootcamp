from collections import Counter

import numpy as np

from priors import CategoryPrior, LocationPrior, collect_training_counts
from retrieval import RetrievalIndex


def test_category_prior_uses_related_queries_without_exact_text():
    prior = CategoryPrior(
        {
            "ремонт телевизоров": Counter({7: 3}),
            "починка телевизора": Counter({7: 2}),
            "маникюр": Counter({8: 4}),
        }
    )

    probabilities = prior.predict("ремонт телевизоров", exclude_exact=True)

    assert max(probabilities, key=probabilities.get) == 7
    assert abs(sum(probabilities.values()) - 1) < 1e-6


def test_location_prior_boosts_related_locations_without_losing_exact_match():
    prior = LocationPrior(
        np.array([10, 11, 12]),
        {10: Counter({10: 80, 11: 20})},
        multiplier=30,
        exponent=0.4,
    )

    weights = prior.weights(10)

    assert weights[0] == 30
    assert 1 < weights[1] < 30
    assert weights[2] == 1


def test_location_counts_exclude_benchmark_items():
    rows = [
        {
            "search_query": "ремонт телевизора",
            "item_microcat_id": 7,
            "search_location_id": 10,
            "item_location_id": 99,
            "item_id": "in-corpus",
        },
        {
            "search_query": "ремонт телевизора",
            "item_microcat_id": 7,
            "search_location_id": 10,
            "item_location_id": 11,
            "item_id": "other-item",
        },
    ]

    categories, locations = collect_training_counts(rows, {"in-corpus"})

    assert categories["ремонт телевизора"] == Counter({7: 2})
    assert locations[10] == Counter({11: 1})


def test_retrieval_uses_category_prior_to_break_text_tie():
    items = [
        {
            "item_id": "0000000000000001",
            "item_title_raw": "Ремонт техники",
            "item_description_raw": "",
            "item_infm_params_text": "",
            "item_location_id": 10,
            "item_microcat_id": 7,
        },
        {
            "item_id": "0000000000000002",
            "item_title_raw": "Ремонт техники",
            "item_description_raw": "",
            "item_infm_params_text": "",
            "item_location_id": 10,
            "item_microcat_id": 8,
        },
    ]
    index = RetrievalIndex(items)
    title, body = index.score_components("ремонт техники")

    found = index.rank(
        title,
        body,
        location_id=10,
        category_probabilities={8: 1.0},
        category_boost=4,
    )

    assert found[0] == "0000000000000002"


def test_retrieval_preserves_zero_category_id():
    items = [
        {
            "item_id": "0000000000000001",
            "item_title_raw": "Ремонт техники",
            "item_description_raw": "",
            "item_infm_params_text": "",
            "item_location_id": 10,
            "item_microcat_id": 8,
        },
        {
            "item_id": "0000000000000002",
            "item_title_raw": "Ремонт техники",
            "item_description_raw": "",
            "item_infm_params_text": "",
            "item_location_id": 10,
            "item_microcat_id": 0,
        },
    ]
    index = RetrievalIndex(items)
    title, body = index.score_components("ремонт техники")

    found = index.rank(
        title,
        body,
        location_id=10,
        category_probabilities={0: 1.0},
        category_boost=4,
    )

    assert found[0] == "0000000000000002"
