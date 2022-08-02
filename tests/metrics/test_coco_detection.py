import pytest  # noqa
from jpdr.datasets import GroZi3kTonioni as GroZi3k
from jpdr.metrics.coco_detection import COCO

import copy
import torch


@pytest.fixture
def grozi3k_val():
    return GroZi3k(data_path='data/GroceryProducts_Tonioni',
                   subset='val_query')


@pytest.fixture
def coco_det_val(grozi3k_val):
    gt_coco = grozi3k_val.get_coco_api('labels')
    return COCO(gt_coco, 'labels', 'scores')


def get_pseudo_detections(grozi3k_val, correct=True):
    preds, targets = [], []

    for i in range(10):
        _, target = grozi3k_val[i]
        prediction = copy.deepcopy(target)
        prediction['scores'] = torch.tensor([1.0] * len(prediction['boxes']))
        prediction['labels'] = torch.tensor([1 if correct else 0]
                                            * len(prediction['boxes']))
        preds.append(prediction)
        targets.append(target)

    return preds, targets


@pytest.fixture
def best_detections(grozi3k_val):
    return get_pseudo_detections(grozi3k_val, correct=True)


@pytest.fixture
def worst_detections(grozi3k_val):
    return get_pseudo_detections(grozi3k_val, correct=False)


def test_perfect_detections(coco_det_val, best_detections):
    preds, targets = best_detections
    coco_det_val.update(preds, targets)
    assert all(v == pytest.approx(1.0)
               for _, v in coco_det_val.compute().items())


def test_worst_detections(coco_det_val, worst_detections):
    preds, targets = worst_detections
    coco_det_val.update(preds, targets)
    assert all(v == pytest.approx(0.0)
               for _, v in coco_det_val.compute().items())
