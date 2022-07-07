import torch

from .agg_query_pr_curve import AggQueryPRCurve
from .agg_query_pr_curve_medium_gallery import prepare_score_matrix_and_labels


class AggPRHardGallery(AggQueryPRCurve):
    """
    Calculates the PR curve after aggregation of the score matrix per gallery
    label.

    The score matrix is aggregated by selecting, per query, the lowest
    positive similarity score and, for each negative label, the highest score.
    """
    def _update(self, scores: torch.Tensor, query_labels: torch.Tensor,
                gallery_labels: torch.Tensor):
        scores, new_gallery_labels = get_hardest_score_matrix(
            scores,
            query_labels,
            gallery_labels
        )
        super()._update(scores, query_labels, new_gallery_labels)

    def _compute(self):
        res = super()._compute()
        return {
            f'{k}_hard': v
            for k, v in res.items()
        }


def get_hardest_score_matrix(scores, query_labels, gallery_labels):
    """
    For each query, group the score matrix by class and, select the
    similarity score of the hardest positive and hardest negatives.

    More specifically, we select the *minimum* score out of the group
    that corresponds to the true class of the query and we select the
    *maximum* score out of all other groups.

    Args:
        scores: Score matrix of shape $N_q\times N_g$, where $N_q$ is the
            number of queries and $N_g$ is the total number of items in the
            gallery.
        query_labels: Tensor of shape $N_q$ with the true query labels
        gallery_labels: Tensor of shape $N_g$ with the gallery labels
    Returns:
        A tuple (score_matrix, labels). The score matrix is of shape
        $N_q\times N_c$ where $N_c$ is the number of classes in the gallery.
        The labels is an array of length $N_c$ containing the respective class
        label of each column in the score matrix.
    """
    assert scores.shape[1] == len(gallery_labels)

    (scores,
     gallery_labels,
     unique_labels,
     g_items_per_label) = prepare_score_matrix_and_labels(scores,
                                                          gallery_labels)

    max_g_per_label = g_items_per_label.max()

    idx = (gallery_labels * max_g_per_label
           + torch.cat([torch.arange(0, n) for n in
                        g_items_per_label]).type_as(gallery_labels))

    class_scores = torch.stack(
        torch.zeros((len(query_labels),
                     len(unique_labels) * max_g_per_label))
        .type_as(scores)
        .fill_(float('nan'))
        .scatter_(1, idx.repeat(len(scores), 1), scores)
        .tensor_split(len(unique_labels), dim=1)
    )

    ret_scores = (class_scores.nan_to_num(nan=0)
                  .max(dim=2)[0].T)

    hard_pos_scores = (class_scores[query_labels,
                                    torch.arange(len(query_labels))]
                       .nan_to_num(nan=float('inf'))
                       .min(dim=1)[0])

    ret_scores[
        torch.arange(len(query_labels)),
        query_labels
    ] = hard_pos_scores

    return ret_scores, unique_labels
