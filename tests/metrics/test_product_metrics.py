import pytest  # noqa

from torch import tensor

from jpdr.metrics.product_metrics import ProductMetrics


@pytest.fixture
def prod_metrics():
    return ProductMetrics()


def test_invariant_to_label_frequency(prod_metrics):
    pred_mult = [
        [0, 0, 1, 2, 2, 2],
        [3, 3, 4, 6, 8, 8, 8, 8],
    ]
    gt_mult = [
        [0, 0, 0, 0, 1, 1],
        [3, 4, 4, 4]
    ]

    pred_sing = [
        [0, 1, 2],
        [3, 4, 6, 8],
    ]

    gt_sing = [
        [0, 1],
        [3, 4]
    ]

    prod_metrics.update(pred_mult, gt_mult)
    mult_acc = prod_metrics.compute()

    prod_metrics.update(pred_sing, gt_sing)
    sing_acc = prod_metrics.compute()
    assert mult_acc == sing_acc


def test_works_with_tensors(prod_metrics):
    pred_list = [
        [0, 1, 2],
        [3, 4],
    ]
    gt_list = [
        [0, 1],
        [3, 4],
    ]
    pred_tensor = [tensor(x) for x in pred_list]
    gt_tensor = [tensor(x) for x in gt_list]

    prod_metrics.update(pred_list, gt_list)
    list_acc = prod_metrics.compute()

    prod_metrics.update(pred_tensor, gt_tensor)
    tensor_acc = prod_metrics.compute()

    assert list_acc == tensor_acc


def test_perfect(prod_metrics):
    pred = [
        [0, 1, 2],
        [3, 4]
    ]

    gt = [
        [0, 1, 2],
        [3, 4]
    ]

    prod_metrics.update(pred, gt)
    res = prod_metrics.compute()
    assert res['ProductPrecision'] == 1
    assert res['ProductRecall'] == 1
    assert res['ProductAccuracy'] == 1


def test_worst(prod_metrics):
    pred = [
        [0, 1, 2],
        [3, 4],
    ]

    gt = [
        [3, 4],
        [0, 1, 2],
    ]

    prod_metrics.update(pred, gt)
    res = prod_metrics.compute()
    assert res['ProductPrecision'] == 0
    assert res['ProductRecall'] == 0
    assert res['ProductAccuracy'] == 0


def test_semi(prod_metrics):
    pred = [
        [0, 3],
    ]

    gt = [
        [0, 1, 2],
    ]

    prod_metrics.update(pred, gt)
    res = prod_metrics.compute()
    assert res['ProductPrecision'] == 1/2
    assert res['ProductRecall'] == 1/3
    assert res['ProductAccuracy'] == 1/4


def test_multiple_updates(prod_metrics):
    pred_1 = [
        [0, 1, 2],
        [1, 3],
    ]
    gt_1 = [
        [0, 1],
        [4, 5, 6],
    ]

    pred_2 = [
        [2, 3, 4],
    ]
    gt_2 = [
        [2, 3, 4],
    ]

    pred_cat = [
        *pred_1,
        *pred_2
    ]
    gt_cat = [
        *gt_1,
        *gt_2
    ]

    prod_metrics.update(pred_1, gt_1)
    prod_metrics.update(pred_2, gt_2)
    mult_update_result = prod_metrics.compute()

    prod_metrics.update(pred_cat, gt_cat)
    sing_update_result = prod_metrics.compute()

    assert mult_update_result == sing_update_result


def test_zero_if_empty(prod_metrics):
    res = prod_metrics.compute()
    assert res['ProductPrecision'] == 0
    assert res['ProductRecall'] == 0
    assert res['ProductAccuracy'] == 0
