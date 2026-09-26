import csv

import pytest

from solution import validate_answer, write_answer


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
