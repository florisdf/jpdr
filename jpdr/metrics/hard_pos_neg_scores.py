import torch
import math
from wandb import Histogram

from .base_recognition_metric import BaseRecognitionMetric


class HardPosNegScores(BaseRecognitionMetric):
    """
    Calculates the similarity scores between each query and the hardest
    negative in the gallery.
    """
    def __init__(self):
        super().__init__()
        self.hard_pos_scores = []
        self.hard_neg_scores = []

    def _update(self, scores: torch.Tensor, query_labels: torch.Tensor,
                gallery_labels: torch.Tensor):
        # Subtract infinity from the positive labels, so we can find the
        # closest negative
        pos_mask = query_labels[:, None] == gallery_labels[None, :]

        hard_neg_scoremat = torch.clone(scores)
        hard_neg_scoremat[pos_mask] -= math.inf
        hardest_neg_scores = hard_neg_scoremat.max(dim=1)[0]

        # Note: an item of hardest_neg_scores will be -inf if there are no
        # negatives for that query
        self.hard_neg_scores.extend(
            hardest_neg_scores[~torch.isinf(hardest_neg_scores)]
        )

        hard_pos_scoremat = torch.clone(scores)
        hard_pos_scoremat[~pos_mask] += math.inf
        hardest_pos_scores = hard_pos_scoremat.min(dim=1)[0]

        # Note: an item of hardest_pos_scores will be +inf if there are no
        # positives for that query
        self.hard_pos_scores.extend(
            hardest_pos_scores[~torch.isinf(hardest_pos_scores)]
        )

    def _compute(self):
        ret = {}

        if len(self.hard_pos_scores) > 0:
            ret['HardPosScores'] = Histogram(
                torch.stack(self.hard_pos_scores).cpu()
            )
        if len(self.hard_neg_scores) > 0:
            ret['HardNegScores'] = Histogram(
                torch.stack(self.hard_neg_scores).cpu()
            )

        return ret
