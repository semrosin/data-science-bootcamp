import csv

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import solution
from retrieval import RetrievalIndex
from solution import main, validate_answer, write_answer


def test_write_answer_preserves_ids_and_exact_columns(tmp_path):
    path = tmp_path / "answer.csv"
    write_answer(path, [("AbCdEfGh12345678", ["0123456789abcdef", "fedcba9876543210"])])
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    assert rows == [
        ["query_id", "answer"],
        ["AbCdEfGh12345678", "0123456789abcdef fedcba9876543210"],
    ]


def test_validate_answer_rejects_duplicate_or_unknown_items(tmp_path):
    path = tmp_path / "answer.csv"
    write_answer(path, [("AbCdEfGh12345678", ["0123456789abcdef", "0123456789abcdef"])])
    with pytest.raises(ValueError, match="duplicate"):
        validate_answer(path, {"AbCdEfGh12345678"}, {"0123456789abcdef"})

    write_answer(path, [("AbCdEfGh12345678", ["fedcba9876543210"])])
    with pytest.raises(ValueError, match="unknown"):
        validate_answer(path, {"AbCdEfGh12345678"}, {"0123456789abcdef"})


def test_validate_answer_rejects_extra_csv_field(tmp_path):
    path = tmp_path / "answer.csv"
    path.write_text(
        "query_id,answer\nAbCdEfGh12345678,0123456789abcdef,unexpected\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="columns"):
        validate_answer(path, {"AbCdEfGh12345678"}, {"0123456789abcdef"})


def test_validate_answer_rejects_blank_csv_row(tmp_path):
    path = tmp_path / "answer.csv"
    path.write_text(
        "query_id,answer\nAbCdEfGh12345678,0123456789abcdef\n\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="columns"):
        validate_answer(path, {"AbCdEfGh12345678"}, {"0123456789abcdef"})


def test_no_tuning_uses_documented_retrieval_weights(tmp_path, monkeypatch):
    items = [
        {
            "item_id": "0000000000000001",
            "item_title_raw": "General services",
            "item_description_raw": "Television repair",
            "item_infm_params_text": "",
            "item_location_id": 1,
        },
        {
            "item_id": "0000000000000002",
            "item_title_raw": "Television repair",
            "item_description_raw": "Television repair",
            "item_infm_params_text": "",
            "item_location_id": 2,
        },
    ]
    queries = [
        {
            "query_id": "AbCdEfGh12345678",
            "search_query": "television repair",
            "search_location_id": 1,
        }
    ]
    pq.write_table(pa.Table.from_pylist(items), tmp_path / "benchmark_items.parquet")
    pq.write_table(
        pa.Table.from_pylist(queries), tmp_path / "benchmark_queries.parquet"
    )
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

    main()

    with output.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["answer"].split()[0] == "0000000000000001"


def test_negative_validation_size_is_rejected_before_loading_data(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "sys.argv",
        ["solution.py", "--dataset", str(tmp_path), "--validation-size", "-1"],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


def test_mean_recall_at_50_averages_over_queries():
    items = [
        {
            "item_id": "0000000000000001",
            "item_title_raw": "Television repair",
            "item_description_raw": "Television repair",
            "item_infm_params_text": "",
            "item_location_id": 1,
        },
        {
            "item_id": "0000000000000002",
            "item_title_raw": "Plumbing service",
            "item_description_raw": "Plumbing service",
            "item_infm_params_text": "",
            "item_location_id": 2,
        },
        {
            "item_id": "0000000000000003",
            "item_title_raw": "Jazz piano",
            "item_description_raw": "Jazz piano",
            "item_infm_params_text": "",
            "item_location_id": 2,
        },
    ]
    pairs = [
        (("television", 1, ""), {"0000000000000001"}),
        (("plumbing", 2, ""), {"0000000000000002", "0000000000000003"}),
    ]

    assert (
        solution.mean_recall_at_50(RetrievalIndex(items), pairs, (0.15, 60.0)) == 0.75
    )
