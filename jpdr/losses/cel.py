import torch
from torch import nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self, num_train_classes: int,
                 embedding_dim: int = None):
        """
        Args:
            embedding_dim (int): The dimension of the embedding.
            num_train_classes (int): The number of training classes.
        """
        super().__init__()
        self.num_train_classes = num_train_classes
        self._classifier = None
        self.embedding_dim = embedding_dim

    @property
    def embedding_dim(self):
        return self._embedding_dim

    @embedding_dim.setter
    def embedding_dim(self, new_value):
        if (
            (self._classifier is None
             or self._embedding_dim != new_value)
            and new_value is not None
        ):
            self._embedding_dim = new_value
            self._classifier = nn.Linear(
                in_features=self.embedding_dim,
                out_features=self.num_train_classes
            )

    def classifier(self, embeddings: torch.Tensor):
        if self.embedding_dim is None and len(embeddings) > 0:
            self.embedding_dim = len(embeddings[0])
            self._classifier = self._classifier.to(embeddings.device)

        if self._classifier is not None:
            return self._classifier(embeddings)
        else:
            raise ValueError('Classifier is not initialized, please enter')

    def forward(self, embeddings, target_labels):
        pred_labels = self.classifier(embeddings)
        return F.cross_entropy(pred_labels, target_labels)
