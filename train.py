import argparse
from collections import OrderedDict
from pathlib import Path

import torch
from torch.optim import SGD
from torchvision.models import resnet
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import wandb

from jpdr.models import JointRCNN
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
    dataset='tonioni',

    # RoI args
    roi_output_size=14,
    box_head_out_channels=1024,
    box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,

    # Anchor
    featmap_names=None,
    anchor_sizes=None,
    use_fpn=True,

    # Detection batch args
    batch_size_det=2,
    img_size_det=800,

    # Recognition batch args
    batch_size_recog=2,
    id_embedding_size=512,

    # K-Fold args
    k_fold_seed=15,
    k_fold_num_folds=5,
    k_fold_val_fold=0,

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

    # Force FPN level
    force_fpn_level=None,

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

    if featmap_names is None:
        featmap_names = (
            ['0', '1', '2', '3'] if use_fpn
            else ['0']
        )
    if anchor_sizes is None:
        anchor_sizes = (
            ((32,), (64,), (128,), (256,), (512,)) if use_fpn
            else ((32, 64, 128, 256, 512),)
        )

    if use_fpn:
        backbone = resnet_fpn_backbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            trainable_layers=trainable_layers
        )
        backbone_out_channels = backbone.out_channels
    else:
        backbone = resnet.__dict__[backbone_name](pretrained=pretrained)
        backbone_out_channels = backbone.fc.in_features
        freeze_layers(backbone, trainable_layers)
        backbone = torch.nn.Sequential(
            OrderedDict([
                *(list(backbone.named_children())[:-1]),
            ])
        )

    model = JointRCNN(
        backbone=backbone,
        backbone_out_channels=backbone_out_channels,
        anchor_sizes=anchor_sizes,
        featmap_names=featmap_names,
        box_head_out_channels=box_head_out_channels,
        id_embedding_size=id_embedding_size,
        recog_loss_fn=recog_loss_fn,
        # RoI parameters
        roi_output_size=roi_output_size,
        # Box parameters
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
        freeze_layers(recognizer, trainable_layers)
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
        force_fpn_level=force_fpn_level,
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


def freeze_layers(model, trainable_layers):
    layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][
        :trainable_layers
    ]
    for name, parameter in model.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)


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
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--backbone_name', default='resnet18',
        help=(
            "Which backbone architecture to use. "
            "Possible values are 'ResNet', 'resnet18', 'resnet34', "
            "'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', "
            "'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2'."
        )
    )

    parser.add_argument(
        '--no_pretrained', action='store_true',
        help='If set, the backbone will be initialized with random weights.',
    )
    parser.add_argument(
        '--trainable_layers', default=1,
        help='The number of unfrozen backbone layers.',
        type=int
    )
    parser.add_argument(
        '--load_ckpt', default=None,
        help='The path to load model checkpoint weights from.'
    )
    parser.add_argument(
        '--save_unique', action='store_true',
        help=(
            'If set, the created checkpoint(s) will get a unique name '
            'containing its WandB run ID.'
        )
    )
    parser.add_argument(
        '--save_best', action='store_true',
        help='If set, save a checkpoint containg the weights with the highest '
        'COCO AP (for the recognition task).'
    )
    parser.add_argument(
        '--save_last', action='store_true',
        help='If set, save a checkpoint containing the weights of the last '
        'epoch.'
    )

    # Dataset
    parser.add_argument(
        '--dataset', default='tonioni',
        help='The dataset to use for training and validation. '
        'To add your own dataset, add it to the `DATASET_REGISTRY` '
        'in `data.py`. See `data.py` for more details.'
    )

    # RoI args
    parser.add_argument(
        '--roi_output_size', default=14,
        help='The output size of the feature maps after RoI pooling.',
        type=int
    )
    parser.add_argument(
        '--box_head_out_channels', default=1024,
        help='The number of dimensions in the RoI embedding '
        '(i.e., the size of the vector to which the RoI feature map is '
        'transformed).',
        type=int
    )
    parser.add_argument(
        '--box_fg_iou_thresh', default=0.5,
        help='The minimum IoU between the proposals and the GT box so that '
        'they can be considered as positive during training of the '
        'classification head.',
        type=float
    )
    parser.add_argument(
        '--box_bg_iou_thresh', default=0.5,
        help='The maximum IoU between the proposals and the GT box so that '
        'they can be considered as negative during training of the '
        'classification head.',
        type=float
    )
    parser.add_argument(
        '--box_score_thresh', default=0.05,
        help='During inference, only return proposals with a classification '
        'score greater than `box_score_thresh`.',
        type=float
    )
    parser.add_argument(
        '--box_nms_thresh', default=0.5,
        help='NMS threshold for the prediction head during inference.',
        type=float
    )
    parser.add_argument(
        '--box_detections_per_img', default=100,
        help='Maximum number of detections per image.',
        type=float
    )

    # Anchor
    parser.add_argument(
        '--featmap_names', default=None,
        help='The names of the feature maps (in the ordered dict of feature '
        'maps returned by the backbone) that will be used for pooling. '
        'If the backbone is not an FPN and simply returns a tensor '
        "(i.e. only a single feature map), set `featmap_names` to `['0']`.",
        type=lambda x: x if x is None else str_list_arg_type(x)
    )
    parser.add_argument(
        '--no_use_fpn',
        action='store_true',
        help="Set this flag if you don't want to use an FPN backbone."
    )

    # Detection batch args
    parser.add_argument(
        '--batch_size_det', default=2,
        help='Batch size to use for detection training.',
        type=int
    )
    parser.add_argument(
        '--img_size_det', default=800,
        help='Image size to use for detection training.',
        type=int
    )

    # Recognition batch args
    parser.add_argument(
        '--batch_size_recog', default=2,
        help='Batch size to use for recognition training.',
        type=int
    )
    parser.add_argument(
        '--id_embedding_size', default=512,
        help='Length of the recognition embedding vector.',
        type=int
    )

    # K-Fold args
    parser.add_argument(
        '--k_fold_seed', default=15,
        help='Seed for the dataset shuffle used to create the K folds.',
        type=int
    )
    parser.add_argument(
        '--k_fold_num_folds', default=5,
        help='The number of folds to use.',
        type=int
    )
    parser.add_argument(
        '--k_fold_val_fold', default=0,
        help='The index of the validation fold. '
        'Should be a value between 0 and k - 1.',
        type=int
    )

    # Loss args
    parser.add_argument(
        '--rpn_box_weight', default=1,
        help='Relative weight of the RPN box regression loss.',
        type=float
    )
    parser.add_argument(
        '--rpn_objectness_weight', default=1,
        help='Relative weight of the RPN objectness loss.',
        type=float
    )
    parser.add_argument(
        '--roi_box_weight', default=1,
        help='Relative weight of the box regression loss.',
        type=float
    )
    parser.add_argument(
        '--roi_classifier_weight', default=1,
        help='Relative weight of the binary classification loss.',
        type=float
    )
    parser.add_argument(
        '--roi_recognition_weight', default=1,
        help='Relative weight of the recognition loss.',
        type=float
    )

    # Dataloader args
    parser.add_argument(
        '--num_workers', default=8,
        help='The number of workers to use for data loading.',
        type=int
    )

    # Optimizer args
    parser.add_argument('--lr', default=0.01, help='The learning rate.',
                        type=float)

    # Train args
    parser.add_argument(
        '--num_epochs', default=500,
        help='The number of epochs to train.',
        type=int
    )

    # Log args
    parser.add_argument(
        '--wandb_entity', help='Weights and Biases entity.'
    )
    parser.add_argument(
        '--wandb_project', help='Weights and Biases project.'
    )

    # Val batch args
    parser.add_argument('--val_batch_size', default=2,
                        help='The validation batch size.', type=int)

    # Device arg
    parser.add_argument('--device', default='cuda',
                        help='The device (cuda/cpu) to use.')

    # Split detection and recognition passes
    parser.add_argument(
        '--use_split_detect_recog',
        action='store_true',
        help='If set, use training procedure 2: "Two-phase training".'
    )

    # Recognition crop box args
    parser.add_argument(
        '--use_crop_batch_inputs',
        action='store_true',
        help='If set, use training procedure 3: "Crop-batch training".'
    )
    parser.add_argument(
        '--crop_box_size', default=800,
        help='The size of the crop boxes to use for Proc. 3.',
        type=crop_box_size_type
    )
    parser.add_argument(
        '--crop_box_iou_thresh', default=0.5,
        help='When crop boxes overlap more than this amount, only one of '
        'them will be kept.',
        type=float
    )
    parser.add_argument(
        '--crop_box_max_rand_shift', default=0,
        help='Set this to a nonzero value to add random shift to the '
        'crop boxes.',
        type=int
    )
    parser.add_argument(
        '--crop_box_max_out_pct', default=0.5,
        help='When a crop boxes falls outside the image with more than this '
        'amount, the crop box will not be used.',
        type=float
    )
    parser.add_argument(
        '--crop_box_min_tgt_area_pct', default=0.5,
        help='If the target (read: product) that belongs to a crop box has '
        'less than this relative amount of its area inside the crop box, '
        'then that crop box will be discarded.',
        type=float
    )

    parser.add_argument(
        '--use_task_specific',
        action='store_true',
        help='If set, use training procedure 4: "Task-specific training".'
    )

    parser.add_argument(
        '--force_fpn_level', default=None,
        help='Forces the FPN level mapper to always choose the given feature '
        'level. If not set, the level will be chosen separately for each '
        'region proposal with the algorithm described in the original FPN '
        'paper.',
        type=int,
    )

    args = parser.parse_args()

    wandb.init(entity=args.wandb_entity, project=args.wandb_project,
               config=vars(args))
    run_training(
        # Model args
        backbone_name=args.backbone_name,
        pretrained=not args.no_pretrained,
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
        use_fpn=not args.no_use_fpn,

        # Detection batch args
        batch_size_det=args.batch_size_det,
        img_size_det=args.img_size_det,

        # Recognition batch args
        batch_size_recog=args.batch_size_recog,
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

        # Split detection and recognition passes
        use_split_detect_recog=args.use_split_detect_recog,

        # Recognition crop box args
        use_crop_batch_inputs=args.use_crop_batch_inputs,
        crop_box_size=args.crop_box_size,
        crop_box_iou_thresh=args.crop_box_iou_thresh,
        crop_box_max_rand_shift=args.crop_box_max_rand_shift,
        crop_box_max_out_pct=args.crop_box_max_out_pct,
        crop_box_min_tgt_area_pct=args.crop_box_min_tgt_area_pct,

        # Task-specific args
        use_task_specific=args.use_task_specific,

        # Force FPN level
        force_fpn_level=args.force_fpn_level,
    )
