"""The trained ranker must use exact category evidence for known query texts."""

import csv
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import solution
from priors import CategoryPrior, LocationPrior
from ranker import rank_candidates
from retrieval import RetrievalIndex


def test_merge_rankings_preserves_first_40_baseline_items():
    baseline = [f"{number:016x}" for number in range(50)]
    reranked = ["ffffffffffffffff", *reversed(baseline)]

    result = solution.merge_rankings(baseline, reranked, replacements=10)

    assert result[:40] == baseline[:40]
    assert "ffffffffffffffff" in result
    assert len(result) == len(set(result)) == 50


def test_merge_rankings_with_zero_budget_keeps_baseline_exactly():
    baseline = [f"{number:016x}" for number in range(50)]

    assert solution.merge_rankings(
        baseline, ["ffffffffffffffff"], replacements=0
    ) == baseline


def test_ranker_can_keep_exact_query_category(monkeypatch):
    item = {
        "item_id": "0000000000000001",
        "item_title_raw": "Repair service",
        "item_description_raw": "Repair service",
        "item_infm_params_text": "",
        "item_location_id": 1,
        "item_microcat_id": 7,
    }
    index = RetrievalIndex([item])
    categories = CategoryPrior({"repair": Counter({7: 1})})
    locations = LocationPrior(index.locations, {})
    seen = []
    original = categories.predict

    def record(query, exclude_exact=False):
        seen.append(exclude_exact)
        return original(query, exclude_exact=exclude_exact)

    monkeypatch.setattr(categories, "predict", record)

    class Model:
        def predict_proba(self, features):
            return np.array([[0.1, 0.9]])

    assert rank_candidates(
        index, categories, locations, Model(), "repair", 1, "",
        exclude_exact=False,
    ) == [item["item_id"]]
    assert seen == [False]


def test_main_ranks_known_text_with_exact_prior(tmp_path, monkeypatch):
    items = [
        {
            "item_id": f"{number:016x}",
            "item_title_raw": "Repair service",
            "item_description_raw": "Repair service",
            "item_infm_params_text": "",
            "item_location_id": 1,
            "item_microcat_id": 7,
            "item_price": 0,
            "item_rating": 0,
            "item_rating_reviews_count": 0,
            "item_is_phone_hidden": False,
            "item_is_message_forbidden": False,
        }
        for number in (1, 2)
    ]
    queries = [
        {
            "query_id": "AbCdEfGh12345678",
            "search_query": "known repair",
            "search_location_id": 1,
            "search_infm_params_text": "premium",
        },
        {
            "query_id": "AbCdEfGh12345679",
            "search_query": "new repair",
            "search_location_id": 1,
            "search_infm_params_text": "",
        },
    ]
    train = [
        {
            "search_query": "known repair",
            "search_location_id": 1,
            "search_infm_params_text": "premium",
            "item_id": "outside-corpus",
            "item_location_id": 1,
            "item_microcat_id": 7,
        }
    ]
    pq.write_table(pa.Table.from_pylist(items), tmp_path / "benchmark_items.parquet")
    pq.write_table(pa.Table.from_pylist(queries), tmp_path / "benchmark_queries.parquet")
    pq.write_table(pa.Table.from_pylist(train), tmp_path / "train.parquet")
    monkeypatch.setattr(solution, "train_ranker", lambda *args, **kwargs: object())
    original_validation_pairs = solution.validation_pairs
    excluded_samples = []

    def record_validation_pairs(*args, **kwargs):
        excluded_samples.append(kwargs.get("excluded_texts"))
        return original_validation_pairs(*args, **kwargs)

    monkeypatch.setattr(solution, "validation_pairs", record_validation_pairs)
    calls = []

    def record_rank(_index, _categories, _locations, _model, query, location, filter_text, **kwargs):
        calls.append((query, filter_text, kwargs.get("exclude_exact", True)))
        return [items[0]["item_id"]]

    monkeypatch.setattr(solution, "rank_candidates", record_rank)
    output = tmp_path / "answer.csv"
    monkeypatch.setattr(
        "sys.argv",
        ["solution.py", "--dataset", str(tmp_path), "--output", str(output), "--validation-size", "0"],
    )

    solution.main()

    assert calls == [
        ("known repair", "premium", False),
        ("new repair", "", True),
    ]
    assert {"known repair"} in excluded_samples
    with output.open(encoding="utf-8", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 2
