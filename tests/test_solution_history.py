import csv

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import solution
from ranker import FEATURE_NAMES


def test_unseen_query_uses_click_history_in_submission(tmp_path, monkeypatch):
    items = [
        {
            "item_id": "0000000000000001",
            "item_title_raw": "Service",
            "item_description_raw": "Service",
            "item_infm_params_text": "",
            "item_location_id": 1,
            "item_microcat_id": 7,
            "item_price": 0,
            "item_rating": 0,
            "item_rating_reviews_count": 0,
            "item_is_phone_hidden": False,
            "item_is_message_forbidden": False,
        },
        {
            "item_id": "0000000000000002",
            "item_title_raw": "Laptop repairs",
            "item_description_raw": "Laptop repairs",
            "item_infm_params_text": "",
            "item_location_id": 1,
            "item_microcat_id": 7,
            "item_price": 0,
            "item_rating": 0,
            "item_rating_reviews_count": 0,
            "item_is_phone_hidden": False,
            "item_is_message_forbidden": False,
        },
    ]
    queries = [
        {
            "query_id": "AbCdEfGh12345678",
            "search_query": "laptop repairs",
            "search_location_id": 1,
            "search_infm_params_text": "",
        }
    ]
    train = [
        {
            "search_query": "laptop repair",
            "search_location_id": 1,
            "search_infm_params_text": "",
            "item_id": "0000000000000001",
            "item_location_id": 1,
            "item_microcat_id": 7,
        }
    ]
    pq.write_table(pa.Table.from_pylist(items), tmp_path / "benchmark_items.parquet")
    pq.write_table(pa.Table.from_pylist(queries), tmp_path / "benchmark_queries.parquet")
    pq.write_table(pa.Table.from_pylist(train), tmp_path / "train.parquet")

    class HistoryModel:
        def predict_proba(self, features):
            has_history = features[:, FEATURE_NAMES.index("history_score")] > 0
            positive = np.where(has_history, 0.99, 0.01)
            return np.column_stack([1 - positive, positive])

    monkeypatch.setattr(solution, "train_ranker", lambda *args, **kwargs: HistoryModel())
    output = tmp_path / "answer.csv"
    monkeypatch.setattr(
        "sys.argv",
        [
            "solution.py",
            "--dataset",
            str(tmp_path),
            "--output",
            str(output),
            "--validation-size",
            "0",
        ],
    )

    solution.main()

    with output.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["answer"].split()[0] == "0000000000000001"
