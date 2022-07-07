import pytest  # noqa
from jpdr.metrics.hard_pos_neg_scores import HardPosNegScores
import torch


def test_perfect_hard_pos_neg(single_gallery, perfect_query_batch):
    hard_pos_scores, hard_neg_scores = get_hard_pos_neg_scores(
        single_gallery,
        perfect_query_batch
    )
    expected_neg_score = pytest.approx(0)
    expected_pos_score = pytest.approx(1.0)

    assert all(s == expected_pos_score for s in hard_pos_scores)
    assert all(s == expected_neg_score for s in hard_neg_scores)


def test_worst_hard_pos_neg(single_gallery, hardest_negative_query_batch):
    hard_pos_scores, hard_neg_scores = get_hard_pos_neg_scores(
        single_gallery,
        hardest_negative_query_batch
    )
    expected_neg_score = pytest.approx(1.0)
    expected_pos_score = pytest.approx(0)

    assert all(s == expected_pos_score for s in hard_pos_scores)
    assert all(s == expected_neg_score for s in hard_neg_scores)


def get_hard_pos_neg_scores(gallery_batch, query_batch):
    hard_pos_neg_scores = HardPosNegScores()
    queries, query_labels, _, _ = query_batch
    gallery, gallery_labels, _, _ = gallery_batch

    scores = torch.mm(queries, gallery.T)

    hard_pos_neg_scores.update(
        scores=scores,
        query_labels=query_labels,
        gallery_labels=gallery_labels
    )

    return (
        hard_pos_neg_scores.hard_pos_scores,
        hard_pos_neg_scores.hard_neg_scores
    )
