import pytest

from src.evaluation.metrics import ndcg_at_k, recall_at_k


def test_ranking_metrics_reward_hits_and_order():
    assert recall_at_k(["a", "x"], {"a", "b"}, 2) == 0.5
    assert ndcg_at_k(["a", "x"], {"a"}, 2) == 1.0
    assert ndcg_at_k(["x", "a"], {"a"}, 2) < 1.0


def test_metrics_reject_invalid_k():
    with pytest.raises(ValueError):
        recall_at_k([], {"a"}, 0)