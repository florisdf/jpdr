from copy import deepcopy

import torch
from torch.nn.functional import softmax
from torchvision.ops import boxes as box_ops
import wandb

from jpdr.metrics.coco_detection import COCO as COCOMetric
from jpdr.metrics.product_metrics import ProductMetrics
from jpdr.metrics.agg_pr_curve import AggPRCurve
from jpdr.utils.boxes import make_boxes_batchable
from jpdr.utils.crop_batch import crop_and_batch_boxes


class TrainingSteps:
    def __init__(
        self,
        model,
        ds_val,

        # Loss args
        rpn_box_weight=1.0,
        rpn_objectness_weight=1.0,
        roi_box_weight=1.0,
        roi_classifier_weight=1.0,
        roi_recognition_weight=1.0,

        # Inference postprocessing args
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
    ):
        self.rpn_box_weight = rpn_box_weight
        self.rpn_objectness_weight = rpn_objectness_weight
        self.roi_box_weight = roi_box_weight
        self.roi_classifier_weight = roi_classifier_weight
        self.roi_recognition_weight = roi_recognition_weight

        self.model = model

        self.recog_loss_fn = model.recog_loss_fn

        self.box_score_thresh = box_score_thresh
        self.box_nms_thresh = box_nms_thresh
        self.box_detections_per_img = box_detections_per_img

        self.coco_metric_det = COCOMetric(ds_val.get_coco_api('labels'),
                                          scores_key='det_scores',
                                          category_key='labels',
                                          metrics=['precision'])
        self.coco_metric_recog = COCOMetric(ds_val.get_coco_api('product_ids'),
                                            scores_key='recog_scores',
                                            category_key='product_ids',
                                            metrics=['precision'])
        self.product_metric = ProductMetrics()
        self.agg_pr_metric = AggPRCurve()

        self.n_boxes_per_roi = []
        self.box_det_scores = []
        self.box_recog_scores = []
        self.val_recog_losses = []

    def on_before_training_epoch(self):
        pass

    def on_after_training_epoch(self):
        return {}

    def on_before_validation_epoch(self):
        self.n_boxes_per_roi = []
        self.box_det_scores = []
        self.box_recog_scores = []
        self.val_recog_losses = []

    def aggregate_loss(self, loss_dict, loss_weights):
        keys = list(loss_dict.keys())
        losses = torch.stack([loss_dict[k] for k in keys])
        weights = torch.stack([loss_weights[k] for k in keys]).type_as(losses)
        return torch.sum(torch.mul(weights, losses))

    def get_model_val_output(self, x):
        _, results = self.model(x)
        return results

    def on_validation_step(self, batch):
        self.on_validation_step_detection(batch)
        self.on_validation_step_recognition(batch)

    def on_validation_step_detection(self, batch):
        x, targets = batch
        results = self.get_model_val_output(x)
        results = convert_embeddings_to_ids(
            results, self.recog_loss_fn,
        )
        det_results = filter_boxes(
            results,
            score_thresh=self.box_score_thresh,
            nms_thresh=self.box_nms_thresh,
            detections_per_img=self.box_detections_per_img,
            scores_key='det_scores'
        )
        recog_results = filter_boxes(
            results,
            score_thresh=self.box_score_thresh,
            nms_thresh=self.box_nms_thresh,
            detections_per_img=self.box_detections_per_img,
            scores_key='recog_scores'
        )
        for res in recog_results:
            self.n_boxes_per_roi.extend(
                list(res['roi_idxs'].unique(return_counts=True)[1]
                     .cpu())
            )
            self.box_recog_scores.extend(
                list(res['recog_scores'].cpu())
            )
        for res in det_results:
            self.box_det_scores.extend(
                list(res['det_scores'].cpu())
            )

        self.coco_metric_recog.update(predictions=recog_results,
                                      targets=targets)
        self.coco_metric_det.update(predictions=det_results,
                                    targets=targets)

        pred_labels, true_labels = get_pred_and_true_labels(recog_results,
                                                            targets)
        self.product_metric.update(pred_labels, true_labels)

    def on_validation_step_recognition(self, batch):
        x, targets = batch
        crops, labels = get_recog_crops_and_labels(x, targets)

        for img_crops, img_labels in zip(crops, labels):
            id_embeddings = self.model.get_full_image_embeddings(img_crops)

            # Recognition loss
            recog_loss = self.recog_loss_fn(id_embeddings, img_labels)
            self.val_recog_losses.append(recog_loss)

            # PR metrics
            scores = get_label_scores_from_embeddings(
                id_embeddings, self.recog_loss_fn
            )
            gallery_labels = torch.arange(
                0, scores.shape[1]
            ).type_as(img_labels)
            self.agg_pr_metric.update(scores, img_labels, gallery_labels)

    def on_after_validation_epoch(self):
        log_dict = {}

        log_dict.update(
            compute_and_get_log_dict(self.coco_metric_recog, suffix='_recog')
        )
        log_dict.update(
            compute_and_get_log_dict(self.coco_metric_det, suffix='_detect')
        )
        log_dict.update(
            compute_and_get_log_dict(self.product_metric)
        )
        log_dict.update(
            compute_and_get_log_dict(self.agg_pr_metric)
        )

        log_dict['N boxes per RoI'] = wandb.Histogram(
            self.n_boxes_per_roi
        )
        log_dict['Box recog conf scores'] = wandb.Histogram(
            self.box_recog_scores
        )
        log_dict['Box detect conf scores'] = wandb.Histogram(
            self.box_det_scores
        )
        log_dict['ValLoss/crop_recognition'] = torch.tensor(
            self.val_recog_losses
        ).mean()

        return log_dict


def compute_and_get_log_dict(metric, suffix=''):
    result_dict = metric.compute()
    metric_name = metric.__class__.__name__

    log_dict = {}

    for k, v in result_dict.items():
        name = f'{metric_name}{suffix}/{k}'
        log_dict[name] = v
    return log_dict


def get_pred_and_true_labels(results, det_targets):
    pred_ids = [
        det_result['product_ids']
        for det_result in results
    ]
    true_ids = [
        det_target['product_ids']
        for det_target in det_targets
    ]

    return pred_ids, true_ids


def get_label_scores_from_embeddings(embeddings, recog_loss_fn):
    recog_scores = recog_loss_fn.classifier(
        embeddings
    )
    return softmax(recog_scores, dim=-1)


def convert_embeddings_to_ids(results, recog_loss_fn, is_binary=True):
    for res in results:
        res['det_scores'] = res['scores']
        del res['scores']

        recog_scores = get_label_scores_from_embeddings(res['id_embeddings'],
                                                        recog_loss_fn)

        # Remove background
        recog_scores = recog_scores[:, 1:]

        argmax_idxs = recog_scores.argmax(dim=-1)

        # Set predicted product ids
        res['product_ids'] = argmax_idxs + 1  # +1 to compensate for removed BG

        if is_binary:
            res['recog_scores'] = torch.gather(
                recog_scores, 1, argmax_idxs[:, None]
            )
        else:
            res['recog_scores'] = recog_scores

        for k in res:
            if is_binary:
                # remove all empty dimensions
                res[k] = res[k].squeeze()
            else:
                # For each RoI, keep the box with the best match
                if k == 'boxes':
                    res[k] = torch.gather(
                        res['boxes'], 1,
                        argmax_idxs[:, None, None].expand(-1, -1, 4)
                    ).squeeze()
                elif k in ['product_ids', 'id_embeddings']:
                    continue
                else:
                    res[k] = torch.gather(
                        res[k], 1,
                        argmax_idxs[:, None]
                    ).squeeze()
    return results


def filter_boxes(
    results,
    score_thresh, nms_thresh, detections_per_img,
    scores_key='recog_scores'
):
    results = deepcopy(results)
    for res in results:
        # remove low scoring boxes
        inds = torch.where(res[scores_key] > score_thresh)[0]
        for k in res:
            res[k] = res[k][inds]

        # non-maximum suppression, independently done per label
        keep = box_ops.batched_nms(res['boxes'], res[scores_key],
                                   res['product_ids'],
                                   nms_thresh)
        # keep only topk scoring predictions
        keep = keep[:detections_per_img]
        for k in res:
            res[k] = res[k][keep]

        # remove empty boxes
        keep = box_ops.remove_small_boxes(res['boxes'], min_size=1e-2)
        for k in res:
            res[k] = res[k][keep]

    return results


def get_recog_crops_and_labels(x, targets):
    labels = [t['product_ids'] for t in targets]
    boxes = [t['boxes'] for t in targets]
    crops = []
    for img, img_boxes in zip(x, boxes):
        crop_boxes = make_boxes_batchable(img_boxes, crop_box_size='mean')
        crops.append(crop_and_batch_boxes(img[None, ...],
                                          crop_boxes[None, ...]))
    return crops, labels
