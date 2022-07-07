import argparse
from collections import OrderedDict
from pathlib import Path

import torch
from torchvision.models import resnet
from torch.optim import SGD
import wandb

from jpdr.models import JointResNetFPN
from jpdr.losses import CrossEntropyLoss

from training.training_loop import (
    TrainingLoopSameEpoch,
    TrainingLoopSplitEpoch
)
from training.training_steps import (
    TrainingStepsTwoModels,
    TrainingStepsSameEpoch,
    TrainingStepsSplitEpoch,
    TrainingStepsCropBatch,
    TrainingStepsCtxRm,
    TrainingStepsTaskSpecific,
)
from data import get_dataloaders


CKPT_DIR = Path('ckpts')


def run_training(
    # Model args
    backbone_name='resnet18',
    pretrained=True,
    trainable_layers=1,
    load_ckpt=None,
    save_unique=False,
    save_last=True,
    save_best=True,

    # Dataset
    dataset='tankstation',

    # RoI args
    roi_output_size=14,
    box_head_out_channels=1024,
    box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,

    # Anchor
    featmap_names=['0', '1', '2', '3'],
    anchor_sizes=((32,), (64,), (128,), (256,), (512,)),

    # Detection batch args
    batch_size_det=2,
    img_size_det=800,

    # Recognition batch args
    batch_size_recog=2,
    img_size_recog=224,
    id_embedding_size=512,

    # K-Fold args
    k_fold_seed=15,
    k_fold_num_folds=5,
    k_fold_val_fold=1,

    # Loss args
    rpn_box_weight=1.0,
    rpn_objectness_weight=1.0,
    roi_box_weight=1.0,
    roi_classifier_weight=1.0,
    roi_recognition_weight=1.0,

    # Dataloader args
    num_workers=8,

    # Optimizer args
    lr=0.01,

    # Train args
    num_epochs=30,

    # Validation args
    val_batch_size=2,

    # Inference postprocessing args
    box_score_thresh=0.05,
    box_nms_thresh=0.5,
    box_detections_per_img=100,

    # Device arg
    device='cuda',

    # Use separate detector and recognizer (baseline)
    use_two_models=False,

    # Split detection and recognition passes
    use_split_detect_recog=False,

    # Recognition crop box args
    use_crop_batch_inputs=False,
    crop_box_size=800,
    crop_box_iou_thresh=0.5,
    crop_box_max_rand_shift=0,
    crop_box_max_out_pct=0.5,
    crop_box_min_tgt_area_pct=0.5,

    # Task-specific training args
    use_task_specific=False,

    # Context remover training args
    use_ctx_remover=False,
    max_ctx_remover_train_crops=128,
    max_ctx_remover_train_crop_size=256,
    ctx_remover_weight=1.0,
):
    if sum(int(x) for x in [
            use_split_detect_recog, use_crop_batch_inputs, use_task_specific,
            use_two_models
    ]) > 1:
        raise ValueError(
            'use_two_models, '
            'use_split_detect_recog, use_crop_batch_inputs and '
            'use_task_specific '
            'are mutually exclusive.'
        )

    dl_train_det, dl_train_recog, dl_val = get_dataloaders(
        dataset,
        img_size_det=img_size_det,
        batch_size_recog=batch_size_recog,
        batch_size_det=batch_size_det,
        val_batch_size=val_batch_size,
        num_workers=num_workers,
        seed=k_fold_seed,
        k=k_fold_num_folds,
        val_fold=k_fold_val_fold,
    )
    ds_recog_train = dl_train_recog.dataset
    ds_det_train = dl_train_det.dataset
    ds_val = dl_val.dataset

    recog_loss_fn = CrossEntropyLoss(
        num_train_classes=len(ds_recog_train.label_to_label_idx),
        embedding_dim=id_embedding_size
    )

    box_labels = set(
        i.item() for ids in ds_det_train.df['target'].apply(
            lambda t: t['labels']
        ).values
        for i in ids
    )
    if 0 not in box_labels:
        box_labels.add(0)
    box_num_classes = len(box_labels)

    device = torch.device(
        'cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu'
    )

    model = JointResNetFPN(
        backbone_name=backbone_name,
        pretrained=pretrained,
        trainable_layers=trainable_layers,
        roi_output_size=roi_output_size,
        box_head_out_channels=box_head_out_channels,
        featmap_names=featmap_names,
        anchor_sizes=anchor_sizes,
        id_embedding_size=id_embedding_size,
        recog_loss_fn=recog_loss_fn,
        box_num_classes=box_num_classes,
        box_fg_iou_thresh=box_fg_iou_thresh,
        box_bg_iou_thresh=box_bg_iou_thresh,
        use_recog_loss_on_forward=not (
            use_split_detect_recog
            or use_crop_batch_inputs
            or use_ctx_remover
        )
    )
    if load_ckpt is not None:
        model.load_state_dict(torch.load(load_ckpt))

    training_step_kwargs = dict(
        model=model,
        ds_val=ds_val,

        # Loss args
        rpn_box_weight=rpn_box_weight,
        rpn_objectness_weight=rpn_objectness_weight,
        roi_box_weight=roi_box_weight,
        roi_classifier_weight=roi_classifier_weight,
        roi_recognition_weight=roi_recognition_weight,

        # Inference postprocessing args
        box_score_thresh=box_score_thresh,
        box_nms_thresh=box_nms_thresh,
        box_detections_per_img=box_detections_per_img,
    )

    TrainingSteps = TrainingStepsSameEpoch
    TrainingLoop = TrainingLoopSameEpoch

    if use_two_models:
        TrainingSteps = TrainingStepsTwoModels
        recognizer = resnet.__dict__[backbone_name](pretrained=pretrained)
        layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][
            :trainable_layers
        ]
        for name, parameter in recognizer.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)
        recognizer = torch.nn.Sequential(
            OrderedDict([
                *(list(recognizer.named_children())[:-1]),
                ('flatten', torch.nn.Flatten()),
                ('fc',  torch.nn.Linear(recognizer.fc.in_features,
                                        id_embedding_size))
            ])
        )
        del training_step_kwargs['model']
        training_step_kwargs.update(
            dict(
                detector=model.to(device),
                recognizer=recognizer.to(device),
                recog_loss_fn=recog_loss_fn.to(device),
            )
        )
    elif use_split_detect_recog:
        TrainingSteps = TrainingStepsSplitEpoch
    elif use_crop_batch_inputs:
        TrainingSteps = TrainingStepsCropBatch
        training_step_kwargs.update(
            dict(
                crop_box_size=crop_box_size,
                crop_box_iou_thresh=crop_box_iou_thresh,
                crop_box_max_rand_shift=crop_box_max_rand_shift,
                crop_box_max_out_pct=crop_box_max_out_pct,
                crop_box_min_tgt_area_pct=crop_box_min_tgt_area_pct,
            )
        )
    elif use_task_specific:
        TrainingSteps = TrainingStepsTaskSpecific

    if use_ctx_remover:
        TrainingSteps = TrainingStepsCtxRm
        extra_kwargs = dict(
            max_ctx_remover_train_crops=max_ctx_remover_train_crops,
            max_ctx_remover_train_crop_size=max_ctx_remover_train_crop_size,
            ctx_remover_weight=ctx_remover_weight,
        )
        training_step_kwargs.update(extra_kwargs)

    training_steps = TrainingSteps(**training_step_kwargs)

    optimizer = SGD(model.parameters(), lr=lr)

    training_loop_kwargs = dict(
        training_steps=training_steps,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        dl_val=dl_val,
        save_unique=save_unique,
        save_last=save_last,
        save_best=save_best,
    )

    if isinstance(training_steps, TrainingStepsSplitEpoch):
        TrainingLoop = TrainingLoopSplitEpoch
        training_loop_kwargs.update(dict(
            dl_train_det=dl_train_det,
            dl_train_recog=dl_train_recog,
        ))
    else:
        assert isinstance(training_steps, TrainingStepsSameEpoch)
        TrainingLoop = TrainingLoopSameEpoch
        training_loop_kwargs.update(dict(
            dl_train=dl_train_det,
        ))

    training_loop = TrainingLoop(
        **training_loop_kwargs
    )
    training_loop.run()


def bool_arg_type(arg):
    arg_lower = arg.lower()
    if arg_lower not in ['true', 'false']:
        raise TypeError(f'Illegal boolean arg value: "{arg}"')
    return arg_lower == 'true'


def int_list_arg_type(arg):
    return [int(s) for s in arg.split(',') if len(s.strip()) > 0]


def str_list_arg_type(arg):
    return [s.strip() for s in arg.split(',') if len(s.strip()) > 0]


def crop_box_size_type(arg):
    try:
        value = int(arg)
        return (value, value)
    except ValueError:
        return arg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--backbone_name', default='resnet18', help='')

    parser.add_argument('--pretrained', default=True, help='',
                        type=bool_arg_type)
    parser.add_argument('--trainable_layers', default=1, help='', type=int)
    parser.add_argument('--load_ckpt', default=None, help='')
    parser.add_argument('--save_unique', action='store_true', help='')
    parser.add_argument('--save_best', action='store_true', help='')
    parser.add_argument('--save_last', action='store_true', help='')

    # Dataset
    parser.add_argument('--dataset', default='tankstation', help='')

    # RoI args
    parser.add_argument('--roi_output_size', default=14, help='', type=int)
    parser.add_argument('--box_head_out_channels', default=1024, help='',
                        type=int)
    parser.add_argument('--box_fg_iou_thresh', default=0.5, help='',
                        type=float)
    parser.add_argument('--box_bg_iou_thresh', default=0.5, help='',
                        type=float)
    parser.add_argument('--box_score_thresh', default=0.05, help='',
                        type=float)
    parser.add_argument('--box_nms_thresh', default=0.5, help='',
                        type=float)
    parser.add_argument('--box_detections_per_img', default=100, help='',
                        type=float)

    # Anchor
    parser.add_argument('--featmap_names', default='0,1,2,3', help='',
                        type=str_list_arg_type)

    # Detection batch args
    parser.add_argument('--batch_size_det', default=2, help='', type=int)
    parser.add_argument('--img_size_det', default=800, help='', type=int)

    # Recognition batch args
    parser.add_argument('--batch_size_recog', default=2, help='', type=int)
    parser.add_argument('--img_size_recog', default=224, help='', type=int)
    parser.add_argument('--id_embedding_size', default=512, help='', type=int)

    # K-Fold args
    parser.add_argument('--k_fold_seed', default=15, help='', type=int)
    parser.add_argument('--k_fold_num_folds', default=5, help='', type=int)
    parser.add_argument('--k_fold_val_fold', default=1, help='', type=int)

    # Loss args
    parser.add_argument('--rpn_box_weight', default=1, help='', type=float)
    parser.add_argument('--rpn_objectness_weight', default=1, help='',
                        type=float)
    parser.add_argument('--roi_box_weight', default=1, help='', type=float)
    parser.add_argument('--roi_classifier_weight', default=1, help='',
                        type=float)
    parser.add_argument('--roi_recognition_weight', default=1, help='',
                        type=float)

    # Dataloader args
    parser.add_argument('--num_workers', default=8, help='', type=int)

    # Optimizer args
    parser.add_argument('--lr', default=0.01, help='', type=float)

    # Context remover training args
    parser.add_argument('--max_ctx_remover_train_crops', default=128, help='',
                        type=int)
    parser.add_argument('--max_ctx_remover_train_crop_size', default=256,
                        help='', type=int)
    parser.add_argument('--ctx_remover_weight', default=1.0, help='',
                        type=float)
    parser.add_argument('--use_ctx_remover', action='store_true', help='')

    # Train args
    parser.add_argument('--num_epochs', default=30, help='', type=int)

    # Log args
    parser.add_argument('--wandb_entity', default='jpdr', help='')
    parser.add_argument('--wandb_project',
                        default='toy_train_separate_forward',
                        help='')

    # Val batch args
    parser.add_argument('--val_batch_size', default=2, help='', type=int)

    # Device arg
    parser.add_argument('--device', default='cuda', help='')

    # Use separate detector and recognizer (baseline)
    parser.add_argument('--use_two_models', action='store_true', help='')

    # Split detection and recognition passes
    parser.add_argument('--use_split_detect_recog',
                        action='store_true', help='')

    # Recognition crop box args
    parser.add_argument('--use_crop_batch_inputs',
                        action='store_true', help='')
    parser.add_argument('--crop_box_size', default=800, help='',
                        type=crop_box_size_type)
    parser.add_argument('--crop_box_iou_thresh', default=0.5, help='',
                        type=float)
    parser.add_argument('--crop_box_max_rand_shift', default=0,
                        help='', type=int)
    parser.add_argument('--crop_box_max_out_pct', default=0.5, help='',
                        type=float)
    parser.add_argument('--crop_box_min_tgt_area_pct', default=0.5, help='',
                        type=float)

    # Task-specific training args
    parser.add_argument('--use_task_specific', action='store_true', help='')

    args = parser.parse_args()

    wandb.init(entity=args.wandb_entity, project=args.wandb_project,
               config=vars(args))
    run_training(
        # Model args
        backbone_name=args.backbone_name,
        pretrained=args.pretrained,
        trainable_layers=args.trainable_layers,
        load_ckpt=args.load_ckpt,
        save_unique=args.save_unique,
        save_best=args.save_best,
        save_last=args.save_last,

        # Dataset
        dataset=args.dataset,

        # RoI args
        roi_output_size=args.roi_output_size,
        box_head_out_channels=args.box_head_out_channels,
        box_fg_iou_thresh=args.box_fg_iou_thresh,
        box_bg_iou_thresh=args.box_bg_iou_thresh,
        box_score_thresh=args.box_score_thresh,
        box_nms_thresh=args.box_nms_thresh,
        box_detections_per_img=args.box_detections_per_img,

        # Anchor
        featmap_names=args.featmap_names,

        # Detection batch args
        batch_size_det=args.batch_size_det,
        img_size_det=args.img_size_det,

        # Recognition batch args
        batch_size_recog=args.batch_size_recog,
        img_size_recog=args.img_size_recog,
        id_embedding_size=args.id_embedding_size,

        # K-Fold args
        k_fold_seed=args.k_fold_seed,
        k_fold_num_folds=args.k_fold_num_folds,
        k_fold_val_fold=args.k_fold_val_fold,

        # Loss args
        rpn_box_weight=args.rpn_box_weight,
        rpn_objectness_weight=args.rpn_objectness_weight,
        roi_box_weight=args.roi_box_weight,
        roi_classifier_weight=args.roi_classifier_weight,
        roi_recognition_weight=args.roi_recognition_weight,

        # Dataloader args
        num_workers=args.num_workers,

        # Val batch args
        val_batch_size=args.val_batch_size,

        # Optimizer args
        lr=args.lr,

        # Train args
        num_epochs=args.num_epochs,

        # Device arg
        device=args.device,

        # Use separate detector and recognizer (baseline)
        use_two_models=args.use_two_models,

        # Split detection and recognition passes
        use_split_detect_recog=args.use_split_detect_recog,

        # Recognition crop box args
        use_crop_batch_inputs=args.use_crop_batch_inputs,
        crop_box_size=args.crop_box_size,
        crop_box_iou_thresh=args.crop_box_iou_thresh,
        crop_box_max_rand_shift=args.crop_box_max_rand_shift,
        crop_box_max_out_pct=args.crop_box_max_out_pct,
        crop_box_min_tgt_area_pct=args.crop_box_min_tgt_area_pct,

        # Task-specific training args
        use_task_specific=args.use_task_specific,

        # Context remover training args
        use_ctx_remover=args.use_ctx_remover,
        max_ctx_remover_train_crops=args.max_ctx_remover_train_crops,
        max_ctx_remover_train_crop_size=args.max_ctx_remover_train_crop_size,
        ctx_remover_weight=args.ctx_remover_weight,
    )
