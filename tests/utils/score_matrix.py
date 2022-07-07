import pytest  # noqa

import torch

from jpdr.utils.score_matrix import (
    get_hardest_score_matrix, get_class_mean_score_matrix
)


def test_hardest_score_matrix():
    scores = torch.tensor([
        [0.0, 0.01, 0.4, 0.7, 0.61],
        [0.1, 0.11, 0.5, 0.8, 0.71],
        [0.2, 0.21, 0.6, 0.9, 0.81],
        [0.3, 0.31, 0.7, 1.0, 0.91],
    ])
    gallery_labels = torch.tensor(
        [1, 0, 1, 1, 0]
    )
    query_labels = torch.tensor([
        0,
        1,
        0,
        1,
    ])

    expected_scores = torch.tensor([
        [0.01, 0.7],
        [0.71, 0.1],
        [0.21, 0.9],
        [0.91, 0.3],
    ])
    expected_labels = torch.tensor([0, 1])

    ret_scores, ret_labels = get_hardest_score_matrix(scores, query_labels,
                                                      gallery_labels)
    assert (ret_scores == expected_scores).all()
    assert (ret_labels == expected_labels).all()


def test_mean_score_matrix():
    scores = torch.tensor([
        [2, 5, 7, 4, 9],
        [3, 1, 2, 5, -4],
        [1, 9, 11, 8, -2],
    ])
    gallery_labels = torch.tensor(
        [1, 0, 0, 1, 0]
    )

    expected_scores = torch.tensor([
        [7.0, 3],
        [-0.5, 4],
        [6, 4.5]
    ])
    expected_labels = torch.tensor(
        [0, 1]
    )

    ret_scores, ret_labels = get_class_mean_score_matrix(scores,
                                                         gallery_labels)

    assert (pytest.approx(ret_scores == expected_scores))
    assert (ret_labels == expected_labels).all()
