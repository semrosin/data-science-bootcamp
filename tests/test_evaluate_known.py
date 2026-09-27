from collections import Counter

import pytest

from evaluate_known import frequency_bucket, weighted_recall


def test_frequency_bucket_separates_rare_and_common_texts():
    assert [frequency_bucket(count) for count in (1, 2, 5, 6)] == [0, 1, 1, 2]


def test_weighted_recall_uses_benchmark_group_share():
    scores = [((0, False), 0.5), ((0, False), 1.0), ((1, True), 0.0)]
    benchmark_counts = Counter({(0, False): 1, (1, True): 3})

    assert weighted_recall(scores, benchmark_counts) == pytest.approx(0.1875)


def test_weighted_recall_rejects_missing_validation_group():
    with pytest.raises(ValueError, match="missing"):
        weighted_recall([((0, False), 1.0)], Counter({(1, True): 1}))
