from typing import List, Union

from torch import Tensor


class ProductMetrics:
    """
    - Product Precision: For each image, the number of unique labels that were
      correctly predicted is divided by the total number of unique labels that
      were predicted. The Product Precision is obtained by calculating the
      mean over all images.

    - Product Accuracy: For each image, the intersection over union of the set
      of predicted labels and the set of true labels present in the image is
      calculated. The Product Accuracy is the mean of these IoUs over all
      images. This metric is sometimes also called *Mean average multi-label
      classification accuracy (mAMCA)*

    - Product Recall: For each image, the number of unique labels that were
      correctly predicted is divided by the total number of ground truth unique
      labels. The Product Recall is obtained by calculating the mean over all
      images.

    Note that *how often* a certain label occurs in the predicted or ground
    truth labels is of no importance, only *which* labels occur.
    """
    def __init__(self):
        self.pp_scores = []
        self.pr_scores = []
        self.pa_scores = []

    def reset(self):
        self.pp_scores = []
        self.pr_scores = []
        self.pa_scores = []

    def update(
        self,
        pred_labels: Union[List[List], List[Tensor]],
        true_labels: Union[List[List], List[Tensor]]
    ):
        """
        Args:
            pred_labels (list): A list of lists containing, for each image, the
                predicted labels of that image.
            true_labels (list): A list of lists containing, for each image, the
                true labels of that image.
        """
        if len(pred_labels) > 0 and isinstance(pred_labels[0], Tensor):
            pred_labels = [t.cpu().numpy() for t in pred_labels]
        if len(true_labels) > 0 and isinstance(true_labels[0], Tensor):
            true_labels = [t.cpu().numpy() for t in true_labels]

        for img_pred_labels, img_true_labels in zip(pred_labels, true_labels):
            img_pred_labels = set(img_pred_labels)
            img_true_labels = set(img_true_labels)

            pred_in_true = {p for p in img_pred_labels
                            if p in img_true_labels}
            pred_or_true = {*img_pred_labels, *img_true_labels}

            if (
                len(img_pred_labels) != 0
                and len(img_true_labels) != 0
                and len(pred_or_true) != 0
            ):
                self.pp_scores.append(len(pred_in_true) / len(img_pred_labels))
                self.pr_scores.append(len(pred_in_true) / len(img_true_labels))
                self.pa_scores.append(len(pred_in_true) / len(pred_or_true))

    def compute(self):
        if len(self.pp_scores) == 0:
            return {
                'ProductPrecision': 0,
                'ProductRecall': 0,
                'ProductAccuracy': 0,
            }

        result = {
            'ProductPrecision': sum(self.pp_scores) / len(self.pp_scores),
            'ProductRecall': sum(self.pr_scores) / len(self.pr_scores),
            'ProductAccuracy': sum(self.pa_scores) / len(self.pa_scores),
        }

        self.reset()

        return result
