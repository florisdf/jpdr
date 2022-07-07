import torch
from .split_epoch import TrainingStepsSplitEpoch

from jpdr.utils.crop_batch import crop_and_batch_boxes


class TrainingStepsTwoModels(TrainingStepsSplitEpoch):
    def __init__(
        self,
        *args,
        detector,
        recognizer,
        recog_loss_fn,
        **kwargs
    ):
        super().__init__(
            *args,
            model=detector,
            **kwargs
        )
        self.detector = detector
        self.recognizer = recognizer
        self.recog_loss_fn = recog_loss_fn

    def get_recog_loss(self, x, targets):
        boxes = [tgt['boxes'] for tgt in targets]
        product_ids = torch.hstack([tgt['product_ids'] for tgt in targets])
        crops = crop_and_batch_boxes(x, boxes)
        id_embeddings = self.recognizer(crops)
        return self.recog_loss_fn(id_embeddings, product_ids)

    def on_training_step_recognition(self, batch):
        recog_loss, log_dict = super().on_training_step_recognition(batch)
        log_dict = {
            'recognition_loss': log_dict['roi_recognition_loss']
        }
        return recog_loss, log_dict

    def get_model_val_output(self, x):
        _, results = self.detector(x)

        for img, result in zip(x, results):
            crops = crop_and_batch_boxes(
                img[None, ...],
                result['boxes'].squeeze()[None, ...]
            )
            result['id_embeddings'] = self.recognizer(crops)

        return results
