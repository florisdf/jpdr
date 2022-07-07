import torch
from torch import Tensor
from .base_recognition_metric import BaseRecognitionMetric


class QueryBasedPRCurve(BaseRecognitionMetric):
    """
    Computes query-based precision-recall pairs for different thresholds.
    Gallery-items are classified as either *same as query* or *different than
    query*.

    Forward accepts

    - ``preds`` (float tensor): ``(N_q, N_g)`` tensor with similarity scores,
      were ``N_q`` is the number of queries and ``N_g`` is the number of
      gallery items.

    - ``target`` (bool tensor): ``(N_q, N_g)`` indicator tensor containing
      ``True`` for those query-gallery pairs that belong to the same class.

    """
    def __init__(self):
        super().__init__()
        self.precision_per_q = []
        self.recall_per_q = []
        self.threshold_per_q = []

    def _update(self,
                scores: Tensor,
                query_labels: Tensor,
                gallery_labels: Tensor):
        target = query_labels[:, None] == gallery_labels
        p_per_q, r_per_q, th_per_q = _pr_curve_per_q(scores, target)
        self.precision_per_q.extend(p_per_q)
        self.recall_per_q.extend(r_per_q)
        self.threshold_per_q.extend(th_per_q)

    def _compute(self):
        """
        Compute the precision-recall curve

        Returns:
            3-element tuple containing

            precision:
                A list of tensors, one for each query, where element i is the
                precision of predictions with score >= thresholds[i] and the
                last element is 1.
            recall:
                A list of tensors, one for each query, where element i is the
                recall of predictions with score >= thresholds[i] and the last
                element is 0.
            thresholds:
                Thresholds used for computing precision/recall scores
        """
        return {
            'Precisions': self.precision_per_q,
            'Recalls': self.recall_per_q,
            'Thresholds': self.threshold_per_q
        }

    def _reset(self):
        self.precision_per_q = []
        self.recall_per_q = []
        self.threshold_per_q = []


def _pr_curve_per_q(preds: Tensor, target: Tensor):
    tps, fps, thresh = _binary_clf_curve_per_q(preds, target)
    precision = tps / (tps + fps)
    recall = tps / tps.nan_to_num(nan=0).max(dim=1)[0][..., None]

    precision = torch.hstack([
        torch.fliplr(precision),
        torch.ones(len(precision)).type_as(precision)[..., None]
    ])
    recall = torch.hstack([
        torch.fliplr(recall),
        torch.zeros(len(recall)).type_as(recall)[..., None]
    ])
    thresh = torch.fliplr(thresh)

    nan_masks = precision.isnan()

    p_per_q = []
    r_per_q = []
    th_per_q = []

    idxs_with_no_positives = torch.nonzero(~target.any(dim=1),
                                           as_tuple=True)[0]

    for i, (p, r, th, nan_mask) in enumerate(zip(precision, recall, thresh,
                                                 nan_masks)):
        p_per_q.append(p[~nan_mask])

        r = r[~nan_mask]
        if i in idxs_with_no_positives:
            # If a query has no positives, the first recall score will be NaN
            # due to a 0 / 0 division (i.e. tps / max(tps) = 0 / 0). In that
            # case, we define the recall to be 1 (precision will be 0 there)
            assert r[0].isnan()
            r[0] = 1.
        r_per_q.append(r)

        th_per_q.append(th[~nan_mask[:-1]])

    return p_per_q, r_per_q, th_per_q


def _binary_clf_curve_per_q(
    preds: Tensor,
    target: Tensor,
):
    """
    Adapted from
    torchmetrics/functional/classification/precision_recall_curve.py
    """
    idxs = torch.argsort(preds, descending=True)
    preds = torch.gather(preds, 1, idxs)
    target = torch.gather(target, 1, idxs)

    tps = torch.cumsum(target, dim=1).to(torch.float)
    fps = (torch.arange(1, target.size(1) + 1)
           .type_as(tps)
           .repeat(target.size(0), 1)
           - tps)

    # Extract indices associated with distinct values.
    distinct_val_idxs = torch.where(preds[:, :-1] - preds[:, 1:])
    distinct_val_mask = torch.zeros(preds.shape).to(torch.bool)
    distinct_val_mask[distinct_val_idxs] = True
    # "Concatenate" a value for the end of the curve
    distinct_val_mask[:, -1] = True

    tps[~distinct_val_mask] = float('nan')
    fps[~distinct_val_mask] = float('nan')
    preds[~distinct_val_mask] = float('nan')

    # stop when full recall is attained
    last_inds = torch.argmax(
        (tps == tps[:, -1].reshape(-1, 1)).to(torch.float),
        dim=1
    )
    mask = torch.zeros(preds.shape).to(torch.bool)
    mask[torch.arange(0, len(tps)), last_inds] = True
    mask = torch.cumsum(mask, dim=1).roll(1).to(torch.bool)
    mask[:, 0] = False

    tps[mask] = float('nan')
    fps[mask] = float('nan')
    preds[mask] = float('nan')

    return tps, fps, preds
