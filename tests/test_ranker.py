from collections import Counter

import numpy as np

from priors import CategoryPrior, LocationPrior
from ranker import FEATURE_NAMES, candidate_features, rank_candidates
from retrieval import RetrievalIndex


def test_candidate_features_use_filter_text_and_keep_negative_prices_finite():
    items = [
        {
            "item_id": "0000000000000001",
            "item_title_raw": "Repair service",
            "item_description_raw": "Laptop repair",
            "item_infm_params_text": "Laptop",
            "item_location_id": 1,
            "item_microcat_id": 7,
            "item_price": -1,
            "item_rating_reviews_count": 20,
        },
        {
            "item_id": "0000000000000002",
            "item_title_raw": "Repair service",
            "item_description_raw": "Television repair",
            "item_infm_params_text": "Television",
            "item_location_id": 1,
            "item_microcat_id": 7,
        },
    ]
    index = RetrievalIndex(items)
    categories = CategoryPrior({"repair": Counter({7: 1})})
    locations = LocationPrior(index.locations, {})

    selected, features, _scores = candidate_features(
        index, categories, locations, "repair", 1, "Laptop"
    )

    assert len(selected) == 2
    assert np.isfinite(features).all()
    filter_column = FEATURE_NAMES.index("filter_body_score")
    by_id = {index.item_ids[position]: features[row, filter_column] for row, position in enumerate(selected)}
    assert by_id["0000000000000001"] > by_id["0000000000000002"]


def test_rank_candidates_uses_model_scores():
    items = [
        {
            "item_id": f"{number:016x}",
            "item_title_raw": "Repair service",
            "item_description_raw": "Repair service",
            "item_infm_params_text": "",
            "item_location_id": 1,
            "item_microcat_id": 7,
        }
        for number in (1, 2)
    ]
    index = RetrievalIndex(items)
    categories = CategoryPrior({"repair": Counter({7: 1})})
    locations = LocationPrior(index.locations, {})

    class Model:
        def predict_proba(self, features):
            assert len(features) == 2
            return np.array([[0.9, 0.1], [0.1, 0.9]])

    assert rank_candidates(index, categories, locations, Model(), "repair", 1, "", limit=1) == [
        "0000000000000002"
    ]


def test_filter_can_add_candidate_beyond_query_only_top_300():
    items = [
        {
            "item_id": f"{number:016x}",
            "item_title_raw": "Service",
            "item_description_raw": "Service service",
            "item_infm_params_text": "",
            "item_location_id": 1,
            "item_microcat_id": 7,
        }
        for number in range(1, 301)
    ]
    items.extend(
        {
            "item_id": f"{number:016x}",
            "item_title_raw": "Service",
            "item_description_raw": "Laptop",
            "item_infm_params_text": "",
            "item_location_id": 1,
            "item_microcat_id": 7,
        }
        for number in (301, 302)
    )
    index = RetrievalIndex(items)
    categories = CategoryPrior({"service": Counter({7: 1})})
    locations = LocationPrior(index.locations, {})

    without_filter, _, _ = candidate_features(
        index, categories, locations, "service", 1, ""
    )
    with_filter, _, _ = candidate_features(
        index, categories, locations, "service", 1, "Laptop"
    )

    assert 300 not in without_filter
    assert 300 in with_filter
