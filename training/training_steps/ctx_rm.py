import torch
from torch.nn.functional import mse_loss

from jpdr.models import JointRCNN
from jpdr.utils.crop_batch import crop_and_batch_boxes
from jpdr.utils.boxes import (
    sample_boxes, shift_boxes_to_fit_img, make_image_list_boxes_batchable,
)

from .crop_batch import TrainingStepsCropBatch


class TrainingStepsCtxRm(TrainingStepsCropBatch):
    def __init__(
        self,
        *args,

        # Context remover training args
        max_ctx_remover_train_crops=128,
        max_ctx_remover_train_crop_size=256,
        ctx_remover_weight=1.0,

        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.max_ctx_remover_crops = max_ctx_remover_train_crops
        self.max_ctx_remover_crop_size = max_ctx_remover_train_crop_size
        self.ctx_remover_weight = ctx_remover_weight
        self.ctx_remover = self.model.roi_heads.box_predictor.ctx_remover

    def get_detection_loss_dict(self, batch):
        x, targets = batch
        features, proposals, loss_dict, detections = \
            self.model.forward_with_intermediary_results(x, targets,
                                                         use_ctx_remover=False)
        ctx_remover_loss = get_ctx_remover_loss_from_features(
            self.model, self.ctx_remover, x, features, proposals, targets,
            self.max_ctx_remover_crops, self.max_ctx_remover_crop_size
        )
        loss_dict['loss_ctx_remover'] = ctx_remover_loss

        return loss_dict

    def get_loss_weights_detection(self):
        loss_weights = super().get_loss_weights_detection()
        loss_weights['loss_ctx_remover'] = loss_weights
        return loss_weights

    def get_detection_log_dict(self, loss_dict):
        log_dict = super().get_detection_log_dict(loss_dict)
        log_dict['ctx_remover_loss'] = loss_dict['loss_ctx_remover']
        return log_dict

    def get_model_val_output(self, x):
        _, results = self.model(x, use_ctx_remover=True)
        return results


def get_ctx_remover_loss_from_features(
    model: JointRCNN, ctx_remover, det_x, features, proposals, det_targets,
    max_num_crops, max_crop_size, return_crop_size=False
):
    # Pass entire image through model and get box features of detections (=
    #   with context), along with the global features of the images
    images, det_targets, orig_img_sizes = model.prepare_images_for_input(
        det_x, det_targets
    )

    # Make proposal boxes batchable
    proposals = make_image_list_boxes_batchable(proposals, how='mean',
                                                max_size=max_crop_size)
    proposals = [
        shift_boxes_to_fit_img(img, img_boxes)
        for img, img_boxes in zip(images.tensors, proposals)
    ]
    proposals = [
        sample_boxes(boxes, num_samples=min(len(boxes),
                                            max_num_crops//len(proposals)),
                     replacement=False)
        for boxes in proposals
    ]

    # Compute box features on feature map of entire images
    with torch.no_grad():
        box_global_features = model.get_box_global_features_from_features(
            features, proposals, images.image_sizes
        )

    # Create batch of crops
    crop_batch = crop_and_batch_boxes(det_x, proposals)
    crop_size = list(crop_batch.shape[-2:])
    crop_sizes = [crop_size]*len(crop_batch)

    # Pass crop batch through model to get true features without context
    with torch.no_grad():
        crop_features = model.get_backbone_features(crop_batch)
        true_box_features_no_ctx = model.get_global_features_from_features(
            crop_features, crop_sizes
        )

    # Pass the features with context and the global features to the context
    # remover
    pred_box_features_no_ctx = ctx_remover(box_global_features)

    # Compute the loss from the returned predicted features without context and
    # the true features without context
    loss = mse_loss(
        pred_box_features_no_ctx,
        true_box_features_no_ctx
    )
    if return_crop_size:
        return loss, crop_size
    else:
        return loss
