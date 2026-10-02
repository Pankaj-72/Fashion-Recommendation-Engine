import numpy as np
import pytest

from src.retrieval.ann import CandidateIndex


def test_candidate_index_preserves_id_mapping_and_score_order(tmp_path):
    index = CandidateIndex(np.asarray([[1, 0], [0, 1]], dtype="float32"), ["article-a", "article-b"])
    assert index.search([0.9, 0.1], 1)[0]["article_id"] == "article-a"
    index.save(tmp_path)
    restored = CandidateIndex.load(tmp_path)
    assert restored.search([0, 1], 1)[0]["article_id"] == "article-b"


def test_candidate_index_checks_vector_id_alignment():
    with pytest.raises(ValueError):
        CandidateIndex(np.ones((2, 3)), ["only-one"])