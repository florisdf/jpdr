from jpdr.utils.crop_batch import crop_and_batch_boxes
from .split_epoch import TrainingStepsSplitEpoch


class TrainingStepsTaskSpecific(TrainingStepsSplitEpoch):
    def on_training_step_recognition(self, batch):
        x, targets = batch

        crops, crop_tgts = create_recog_batch(x, targets)
        return super().on_training_step_recognition((crops, crop_tgts))

    def get_recog_loss(self, x, targets):
        id_embeddings = self.model.get_full_image_embeddings(x)
        product_ids = targets

        return self.model.get_recog_loss_from_embeddings(
            id_embeddings,
            product_ids
        )


def create_recog_batch(x, targets, return_crop_boxes=False,
                       min_tgt_area_pct=0.95):
    crop_boxes = [tgt['boxes'] for tgt in targets]
    crops, crop_tgts = crop_and_batch_boxes(
        x, crop_boxes,
        targets,
        min_tgt_area_pct=min_tgt_area_pct
    )
    ids = [get_largest_target_id(tgt) for tgt in crop_tgts]
    if return_crop_boxes:
        return crops, ids, crop_boxes
    else:
        return crops, ids


def get_largest_target_id(target, id_key='product_ids'):
    boxes = target['boxes']
    box_areas = (boxes[:, ::2].diff() * boxes[:, 1::2].diff()).flatten()
    i = box_areas.argmax()
    return target['product_ids'][i]
