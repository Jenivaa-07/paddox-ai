import numpy as np
import pandas as pd

from models.rf_fantasy_trainer import ndcg_score_at_k, mean_race_ndcg


def test_ndcg_at_k_is_computed_for_a_single_ranking_group():
    # Perfect ranking: highest relevance receives highest predicted score.
    assert ndcg_score_at_k([10, 5, 1], [0.9, 0.5, 0.1], k=3) == 1.0


def test_ndcg_at_k_handles_short_groups_and_zero_relevance():
    assert ndcg_score_at_k([2], [0.1], k=5) == 1.0
    assert ndcg_score_at_k([0, 0], [0.9, 0.1], k=2) == 0.0


def test_mean_race_ndcg_averages_separate_race_rankings():
    frame = pd.DataFrame(
        {
            "season": [2025, 2025, 2025, 2025],
            "round": [1, 1, 2, 2],
            "truth": [10, 1, 1, 10],
            "score": [0.9, 0.1, 0.9, 0.1],
        }
    )
    # Race 1 is perfectly ranked; race 2 is reversed. The macro average
    # must be the mean of the two independent race-level NDCG values.
    expected = np.mean([
        ndcg_score_at_k([10, 1], [0.9, 0.1], k=2),
        ndcg_score_at_k([1, 10], [0.9, 0.1], k=2),
    ])
    assert mean_race_ndcg(frame, "truth", "score", k=2) == expected
