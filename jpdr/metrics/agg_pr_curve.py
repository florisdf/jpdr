import warnings
from typing import List, Optional, Union

import torch
from torch import tensor, Tensor
from wandb import Histogram

from .query_based_pr_curve import QueryBasedPRCurve


class AggPRCurve(QueryBasedPRCurve):
    """
    Metrics that are aggregated from the PR-curves of multiple queries.
    They are returned as a dict, ready to be logged with Weights and
    Biases.

        - AP: Histogram of the average precisions of all queries.
        - mAP: The mean of the average precisions of all queries.
        - P@maxF1: Histogram of the precisions (corresponding to the highest
              F1-score) of all queries.
        - R@maxF1: Histogram of the recalls (corresponding to the highest
              F1-score) of all queries.
        - T@maxF1: Histogram of the thresholds (corresponding to the
              highest F1-score) of all queries.
        - maxF1: Histogram of the highest F1-scores of all queries.
    """
    def _compute(self):
        ap = _average_precision_compute_with_precision_recall(
            precision=self.precision_per_q, recall=self.recall_per_q,
            num_classes=len(self.precision_per_q)
        )

        mean_ap = ap.mean()

        (p_at_max_f1,
         r_at_max_f1,
         th_at_max_f1,
         max_f1) = get_prtfs_at_max_f1(self.precision_per_q,
                                       self.recall_per_q,
                                       self.threshold_per_q)

        return {
            'AP': Histogram(ap.nan_to_num().cpu()),
            'mAP': mean_ap.cpu(),
            'P@maxF1': Histogram(p_at_max_f1.cpu()),
            'R@maxF1': Histogram(r_at_max_f1.cpu()),
            'T@maxF1': Histogram(th_at_max_f1.cpu()),
            'maxF1': Histogram(max_f1.cpu()),
        }


def get_prtfs_at_max_f1(precisions, recalls, thresholds):
    """
    Return the precision, recall, threshold and F1-score at the max F1-score
    (for each class).
    """
    f1s = [
        2 * (p * r)/(p + r)
        for p, r in zip(precisions, recalls)
    ]
    max_f1_idxs = [torch.argmax(f1) for f1 in f1s]
    prtfs_at_max_f1 = [
        (p[idx], r[idx], th[idx], f1[idx])
        for idx, p, r, th, f1 in zip(
            max_f1_idxs,
            precisions,
            recalls,
            thresholds,
            f1s,
        )
    ]
    (p_at_max_f1,
     r_at_max_f1,
     th_at_max_f1,
     max_f1) = list(zip(*prtfs_at_max_f1))
    return (tensor(p_at_max_f1).nan_to_num(),
            tensor(r_at_max_f1).nan_to_num(),
            tensor(th_at_max_f1).nan_to_num(),
            tensor(max_f1).nan_to_num())


# Copied from torchmetrics.functional.classification.average_precision
def _average_precision_compute_with_precision_recall(
    precision: Tensor,
    recall: Tensor,
    num_classes: int,
    average: Optional[str] = "macro",
    weights: Optional[Tensor] = None,
) -> Union[List[Tensor], Tensor]:
    """Computes the average precision score from precision and recall.

    Args:
        precision: precision values
        recall: recall values
        num_classes: integer with number of classes. Not nessesary to provide
            for binary problems.
        average: reduction method for multi-class or multi-label problems
        weights: weights to use when average='weighted'
    """

    # Return the step function integral
    # The following works because the last entry of precision is
    # guaranteed to be 1, as returned by precision_recall_curve
    if num_classes == 1:
        return -torch.sum((recall[1:] - recall[:-1]) * precision[:-1])

    res = []
    for p, r in zip(precision, recall):
        res.append(-torch.sum((r[1:] - r[:-1]) * p[:-1]))

    # Reduce
    if average in ("macro", "weighted"):
        res = torch.stack(res)
        if torch.isnan(res).any():
            warnings.warn(
                "Average precision score for one or more classes was `nan`. "
                "Ignoring these classes in average",
                UserWarning,
            )
        if average == "macro":
            return res[~torch.isnan(res)].mean()
        weights = torch.ones_like(res) if weights is None else weights
        return (res * weights)[~torch.isnan(res)].sum()
    if average is None or average == "none":
        return res
    allowed_average = ("micro", "macro", "weighted", "none", None)
    raise ValueError(
        f"Expected argument `average` to be one of {allowed_average}"
        f" but got {average}"
    )
