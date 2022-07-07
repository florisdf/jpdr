import pytest  # noqa

from jpdr.metrics.query_based_pr_curve import QueryBasedPRCurve

import torch


def test_query_based_pr_curve():
    preds = torch.tensor([
        [2.5, 2.0, 3.0],
        [2.0, 1.0, 2.0],
        [2.5, 1.0, 1.0],
        [1.5, 3.0, 2.0],
        [1.0, 3.0, 4.0],
    ])

    query_labels = torch.tensor([
        0,
        1,
        2,
        2,
        1
    ])

    gallery_labels = torch.tensor([
        0, 1, 2
    ])

    expected_precs = [
        torch.tensor([1/2, 0.0, 1.0]),
        torch.tensor([1/3, 0.0, 1.0]),
        torch.tensor([1/3, 0.0, 1.0]),
        torch.tensor([1/2, 0.0, 1.0]),
        torch.tensor([1/2, 0.0, 1.0]),
    ]

    expected_recs = [
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([1.0, 0.0, 0.0]),
    ]

    expected_threshs = [
        torch.tensor([2.5, 3.0]),
        torch.tensor([1.0, 2.0]),
        torch.tensor([1.0, 2.5]),
        torch.tensor([2.0, 3.0]),
        torch.tensor([3.0, 4.0]),
    ]

    pr_curve = QueryBasedPRCurve()
    pr_curve.update(preds, query_labels, gallery_labels)
    res = pr_curve.compute()

    assert all(p == pytest.approx(exp_p) for p, exp_p in zip(res['Precisions'],
                                                             expected_precs))
    assert all(r == pytest.approx(exp_r) for r, exp_r in zip(res['Recalls'],
                                                             expected_recs))
    assert all(t == pytest.approx(exp_t) for t, exp_t in zip(res['Thresholds'],
                                                             expected_threshs))
