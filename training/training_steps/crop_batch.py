import numpy as np
import wandb

from jpdr.utils.crop_batch import (
    crop_and_batch_boxes, get_crop_boxes_from_targets,
)
from jpdr.utils.boxes import (
    get_overlapping_boxes_mask, get_boxes_outside_img_mask
)
from .split_epoch import TrainingStepsSplitEpoch


class TrainingStepsCropBatch(TrainingStepsSplitEpoch):
    def __init__(
        self,
        *args,

        # Recognition crop box args
        crop_box_size=800,
        crop_box_iou_thresh=0.95,
        crop_box_max_rand_shift=0,
        crop_box_max_out_pct=0.0,
        crop_box_min_tgt_area_pct=0.5,

        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.crop_box_size = crop_box_size
        self.crop_box_max_rand_shift = crop_box_max_rand_shift
        self.crop_box_iou_thresh = crop_box_iou_thresh
        self.crop_box_max_out_pct = crop_box_max_out_pct
        self.crop_box_min_tgt_area_pct = crop_box_min_tgt_area_pct

        self.num_crops_per_img = []

    def on_training_step_recognition(self, batch):
        x, targets = batch

        crops, crop_tgts = create_crop_batch(
            x, targets, self.crop_box_size,
            self.crop_box_max_rand_shift,
            self.crop_box_iou_thresh,
            self.crop_box_max_out_pct,
            self.crop_box_min_tgt_area_pct,
            self.num_crops_per_img,
        )
        return super().on_training_step_recognition((crops, crop_tgts))

    def on_before_training_epoch(self):
        self.num_crops_per_img = []

    def on_after_training_epoch(self):
        return {
            'N crops per img': wandb.Histogram(
                np.array(self.num_crops_per_img)
            ),
        }


def create_crop_batch(
    x, targets, crop_box_size,
    crop_box_max_rand_shift,
    crop_box_iou_thresh,
    crop_box_max_out_pct,
    min_tgt_area_pct,
    num_crops_per_img=None,
    return_crop_boxes=False,
):
    """
    Args:
        crop_box_max_out_pct (float): If a crop box lies outside the image more
            than this relative amount of its area, it will be discarded.
        min_tgt_area_pct (float): If, after cropping, a target has lost more
            than this relative amount of its area, it will be discarded.
    """
    crop_boxes = get_crop_boxes_from_targets(
        targets, crop_box_size, max_rand_shift=crop_box_max_rand_shift
    )

    masks = [
        get_boxes_outside_img_mask(img_boxes, *img.shape[-2:][::-1],
                                   crop_box_max_out_pct)
        for img, img_boxes in zip(x, crop_boxes)
    ]

    masks = [
        mask | get_overlapping_boxes_mask(
            img_boxes,
            crop_box_iou_thresh
        )
        for (mask, img_boxes) in zip(masks, crop_boxes)
    ]

    crop_boxes = [
        img_boxes[~mask]
        for mask, img_boxes in zip(masks, crop_boxes)
    ]

    # Note: each crop box is based on a target because we want the crop boxes
    # to have interesting locations in the image. When a crop box needs to be
    # dropped because it lies too much outside the image, this doesn't mean
    # that the corresponding target needs to be dropped as well. That target
    # can still be useful for other crops. Hence, we don't drop any targets.

    if num_crops_per_img is not None:
        for img_crop_boxes in crop_boxes:
            num_crops_per_img.append(len(img_crop_boxes))
    crops, crop_tgts = crop_and_batch_boxes(
        x, crop_boxes,
        targets, min_tgt_area_pct
    )
    if return_crop_boxes:
        return crops, crop_tgts, crop_boxes
    else:
        return crops, crop_tgts
