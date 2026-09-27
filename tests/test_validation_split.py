import pyarrow as pa
import pyarrow.parquet as pq

from solution import validation_pairs


def test_validation_pairs_excludes_reserved_query_texts(tmp_path):
    path = tmp_path / "train.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "search_query": "alpha service",
                    "search_location_id": 1,
                    "search_infm_params_text": "",
                    "item_id": "0000000000000001",
                },
                {
                    "search_query": "beta service",
                    "search_location_id": 1,
                    "search_infm_params_text": "",
                    "item_id": "0000000000000002",
                },
            ]
        ),
        path,
    )

    pairs = validation_pairs(
        path,
        {"0000000000000001", "0000000000000002"},
        10,
        excluded_texts={"alpha service"},
    )

    assert pairs == [(("beta service", 1, ""), {"0000000000000002"})]
