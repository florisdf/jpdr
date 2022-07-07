from torch import Tensor


class BaseRecognitionMetric:
    def _update(self,
                scores: Tensor,
                query_labels: Tensor,
                gallery_labels: Tensor):
        raise NotImplementedError

    def _compute(self):
        raise NotImplementedError

    def _reset(self):
        raise NotImplementedError

    def update(self,
               scores: Tensor,
               query_labels: Tensor,
               gallery_labels: Tensor) -> None:
        if not scores.shape[0] == len(query_labels):
            raise ValueError(
                'First dim of scores should equal the number of queries'
            )
        if not scores.shape[1] == len(gallery_labels):
            raise ValueError(
                'Second dim of scores should equal the number of gallery items'
            )
        return self._update(scores, query_labels, gallery_labels)

    def compute(self):
        res = self._compute()
        self._reset()
        return res
