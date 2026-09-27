from collections import Counter

import numpy as np

from click_memory import ClickMemory
from priors import CategoryPrior, LocationPrior
from ranker import FEATURE_NAMES, candidate_features, rank_candidates, train_ranker
from retrieval import RetrievalIndex


def test_click_history_excludes_the_training_query_itself():
    ids = ["0000000000000001", "0000000000000002"]
    rows = [
        {"search_query": "laptop repair", "item_id": ids[0]},
        {"search_query": "laptop repair", "item_id": ids[0]},
        {"search_query": "laptop repairs", "item_id": ids[0]},
        {"search_query": "laptop service", "item_id": ids[1]},
    ]
    memory = ClickMemory(rows, ids)

    counts = memory.item_counts(np.array([0, 1]), "laptop repair", exclude_exact=True)
    scores = memory.query_scores("laptop repair", exclude_exact=True)

    assert counts.tolist() == [1, 1]
    assert scores[0] > 0
    assert 1 in scores


def test_click_history_omits_heldout_texts_completely():
    ids = ["0000000000000001", "0000000000000002"]
    rows = [
        {"search_query": "laptop repair", "item_id": ids[0]},
        {"search_query": "laptop repairs", "item_id": ids[1]},
    ]
    memory = ClickMemory(rows, ids, excluded_texts={"laptop repair"})

    assert memory.item_counts(np.array([0, 1]), "other repair").tolist() == [0, 1]
    assert 0 not in memory.query_scores("laptop repair")


def test_history_adds_an_item_missed_by_lexical_candidates():
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
    memory = ClickMemory(
        [{"search_query": "service laptop", "item_id": "000000000000012d"}],
        index.item_ids,
    )

    baseline, _, _ = candidate_features(index, categories, locations, "service", 1, "")
    selected, features, _ = candidate_features(
        index,
        categories,
        locations,
        "service",
        1,
        "",
        click_memory=memory,
        use_history_candidates=True,
        use_history_count=True,
    )

    assert 300 not in baseline
    assert 300 in selected
    position = np.flatnonzero(selected == 300)[0]
    assert features[position, FEATURE_NAMES.index("history_score")] > 0
    assert features[position, FEATURE_NAMES.index("history_count_log")] > 0

    class HistoryModel:
        def predict_proba(self, values):
            has_history = values[:, FEATURE_NAMES.index("history_score")] > 0
            positive = np.where(has_history, 0.99, 0.01)
            return np.column_stack([1 - positive, positive])

    assert rank_candidates(
        index,
        categories,
        locations,
        HistoryModel(),
        "service",
        1,
        "",
        click_memory=memory,
        use_history_candidates=True,
        limit=1,
    ) == ["000000000000012d"]
    assert train_ranker(
        index,
        categories,
        locations,
        [(('service', 1, ''), {"000000000000012d"})],
        click_memory=memory,
        use_history_candidates=True,
    ) is not None
