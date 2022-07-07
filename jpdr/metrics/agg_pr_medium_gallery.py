import torch

from .agg_query_pr_curve import AggQueryPRCurve


class AggPRMediumGallery(AggQueryPRCurve):
    """
    Calculates the PR curve after aggregation of the score matrix per gallery
    label.

    The score matrix is aggregated by computing the average similarity score
    for each gallery label (the length of the query dimension remains the
    same).
    """
    def _update(self, scores: torch.Tensor, query_labels: torch.Tensor,
                gallery_labels: torch.Tensor):
        scores, new_gallery_labels = get_class_mean_score_matrix(
            scores,
            query_labels,
            gallery_labels
        )
        super()._update(scores, query_labels, new_gallery_labels)

    def _compute(self):
        res = super()._compute()
        return {
            f'{k}_medium': v
            for k, v in res.items()
        }


def prepare_score_matrix_and_labels(scores, gallery_labels):
    g_argsort = gallery_labels.argsort()
    gallery_labels = gallery_labels[g_argsort]
    scores = scores[:, g_argsort]
    unique_labels, g_items_per_label = gallery_labels.unique(
        return_counts=True
    )
    return scores, gallery_labels, unique_labels, g_items_per_label


def get_class_mean_score_matrix(scores, gallery_labels):
    """
    Group the score matrix by class and, for each query, calculate the mean
    score per class.

    Args:
        scores: Score matrix of size $N_q\times N_g$, where $N_q$ is the
            number of queries and $N_g$ is the total number of items in the
            gallery.
        gallery_labels: Array of length $N_g$ containing the labels
            corresponding to each column in the score matrix.
    Returns:
        An aggregated score matrix of size $N_q\times N_c$ where $N_c$ is
        the number of classes in the gallery.
    """
    (scores,
     gallery_labels,
     unique_labels,
     g_items_per_label) = prepare_score_matrix_and_labels(scores,
                                                          gallery_labels)

    num_val_classes = len(unique_labels)

    # Group scores by label and calculate the mean score per label
    class_scores = (
        torch.zeros((len(scores), num_val_classes))
        .type_as(scores)
        .scatter_add_(1, gallery_labels.repeat(len(scores), 1),
                      scores)
    ) / g_items_per_label

    return class_scores, unique_labels
