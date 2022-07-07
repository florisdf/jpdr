import io
from contextlib import redirect_stdout
from copy import deepcopy
from itertools import product

import torch

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO as _COCO

import numpy as np

from ..utils.convert_tensor import convert_to_item, convert_to_list


class COCO:
    def __init__(self, coco_gt: _COCO,
                 category_key: str,
                 scores_key: str,
                 metrics=['precision', 'recall'],
                 iou_thrs=[None, 0.5, 0.75, 0.95]):
        self.coco_gt = deepcopy(coco_gt)
        self.coco_eval = COCOeval(coco_gt, iouType="bbox")
        self.category_key = category_key
        self.scores_key = scores_key
        self.metrics = metrics
        self.iou_thrs = iou_thrs

        self.all_results = []
        self.all_img_ids = []

    def update(self, predictions, targets):
        predictions = {
            convert_to_item(target["image_id"]): pred
            for target, pred in zip(targets, predictions)
        }

        results = prepare_for_coco_detection(
            predictions,
            scores_key=self.scores_key,
            category_key=self.category_key
        )
        self.all_results.extend(results)
        self.all_img_ids.extend(list(predictions.keys()))

    def compute_coco_metric(self, metric='precision', iou_thr=None):
        assert metric in ['precision', 'recall']

        values = self.coco_eval.eval[metric]

        iou_thrs = self.coco_eval.params.iouThrs
        if iou_thr is not None:
            values = values[np.where(iou_thr == iou_thrs)[0]]

        iouStr = (
            f'{iou_thrs[0]:0.2f}:{iou_thrs[-1]:0.2f}'
            if iou_thr is None
            else f'{iou_thr:0.2f}'
        )
        value_agg = (
            np.mean(values[values > -1])
            if len(values) > 0
            else -1
        )

        metric_abbr = 'AP' if metric == 'precision' else 'AR'
        metric_name = f'{metric_abbr}@[{iouStr}]'
        return metric_name, value_agg

    def compute(self):
        with redirect_stdout(io.StringIO()):
            coco_dt = (
                _COCO.loadRes(self.coco_gt, self.all_results)
                if len(self.all_results) > 0
                else _COCO()
            )

        img_ids = torch.unique(torch.tensor(self.all_img_ids))

        self.coco_eval.cocoDt = coco_dt
        self.coco_eval.params.imgIds = list(img_ids)
        self.coco_eval.params.maxDets = [1000, 1000, 1000]

        with redirect_stdout(io.StringIO()):
            self.coco_eval.evaluate()

        with redirect_stdout(io.StringIO()):
            self.coco_eval.accumulate()

        self.reset()

        return dict(self.compute_coco_metric(metric, iou_thr)
                    for metric, iou_thr in product(self.metrics,
                                                   self.iou_thrs))

    def reset(self):
        self.all_results = []
        self.all_img_ids = []


def prepare_for_coco_detection(
    predictions, boxes_key="boxes", scores_key="scores", category_key="labels"
):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction[boxes_key]
        boxes = convert_to_xywh(boxes).tolist()
        scores = convert_to_list(prediction[scores_key])
        labels = convert_to_list(prediction[category_key])

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)
