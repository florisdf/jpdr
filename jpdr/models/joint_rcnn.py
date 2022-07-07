from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn, Tensor

import torch.nn.functional as F
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss
from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead
from torchvision.ops import boxes as box_ops
from torchvision.ops import MultiScaleRoIAlign


class JointRCNN(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        anchor_sizes: Tuple[Tuple[int]],
        featmap_names: List[str],
        box_head: nn.Module,
        box_head_out_channels: int,
        id_embedding_size,
        recog_loss_fn,
        # RoI parameters,
        roi_output_size,
        # RPN parameters
        aspect_ratios_per_anchor=(0.5, 1.0, 2.0),
        rpn_pre_nms_top_n_train=256, rpn_pre_nms_top_n_test=128,
        rpn_post_nms_top_n_train=256, rpn_post_nms_top_n_test=128,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
        rpn_score_thresh=0.0,
        # Box parameters
        box_num_classes=2,
        box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512, box_positive_fraction=0.25,
        bbox_reg_weights=None,

        use_recog_loss_on_forward=False,
    ):
        """
        Args:
            backbone: The backbone of the detector. Takes a batch of images as
                input and returns a batch of feature maps for the RPN. The RPN
                regions are then 'cropped' out of the feature maps (with RoI
                pooling) and passed through the RoI heads.
            box_head: This is the first stage of the RoI heads. All RoIs will
                pass through this network. Takes a batch of RoIs as input and
                returns a batch of feature vectors that will be used as an
                input to the three predictor networks (i.e. the box regression
                network, the classification network and the recognition
                network). The `box_head` should have an attribute
                `out_channels` with the number of output channels.
            box_head_out_channels (int): The number of output channels of the
                box head.
            anchor_sizes: For backbones without FPN, use e.g. `((32, 64, 128,
                256, 512),)`. For backbone with FPN, use e.g. `((32,), (64,),
                (128,), (256,), (512,))`.
            featmap_names: The names of the feature maps (in the ordered dict
                of feature maps returned by the backbone) that will be used for
                pooling. If the backbone is not an FPN and simply returns a
                tensor (i.e. only a single feature map), set `featmap_names` to
                `['0']`.
            use_recog_loss_on_forward: If True, compute the recognition loss
                when forwarding. The passed-in targets should contain
                `product_ids`.
        """
        super().__init__()

        # RPN
        anchor_sizes = tuple(tuple(size) for size in anchor_sizes)
        aspect_ratios = (aspect_ratios_per_anchor,) * len(anchor_sizes)
        rpn_anchor_generator = AnchorGenerator(
                anchor_sizes, aspect_ratios
        )
        rpn_head = RPNHead(
            backbone.out_channels,
            rpn_anchor_generator.num_anchors_per_location()[0]
        )

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train,
                                 testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train,
                                  testing=rpn_post_nms_top_n_test)

        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh,
            score_thresh=rpn_score_thresh,
        )

        # RoI Heads
        box_predictor = FastRCNNPredictorWithID(box_head_out_channels,
                                                id_embedding_size,
                                                num_classes=box_num_classes)

        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=featmap_names,
            output_size=roi_output_size,
            sampling_ratio=2
        )
        roi_heads = RoIHeadsWithID(
            # Box
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            # score_thresh, nms_thresh, detection_per_img are not used in
            # the postprocessing step of RoIHeadsWithID, so we set them to None
            # to avoid confusion
            None, None, None,
            recog_loss_fn=(
                recog_loss_fn
                if use_recog_loss_on_forward
                else None
            )
        )

        # Transform
        transform = PassThroughRCNNTransform()

        # Initialize RCNN
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads  # type: RoIHeadsWithID
        self.box_roi_pool = box_roi_pool
        self.recog_loss_fn = recog_loss_fn
        self.use_recog_loss_on_forward = use_recog_loss_on_forward

    def prepare_images_for_input(self, images: List[Tensor],
                                 targets: List[Dict[str, Tensor]] = None):
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        return images, targets, original_image_sizes

    def forward(self, images, targets=None, use_ctx_remover=False):
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in
                the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the
                losses. During testing, it returns list[BoxList] contains
                additional fields like `scores`, `labels` and `mask` (for Mask
                R-CNN models).

        """
        features, proposals, losses, detections = \
            self.forward_with_intermediary_results(images, targets,
                                                   use_ctx_remover)
        return losses, detections

    def forward_with_intermediary_results(self, images, targets=None,
                                          use_ctx_remover=False):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if targets is not None:
            check_targets(targets)

        images, targets, orig_img_sizes = self.prepare_images_for_input(
            images, targets
        )

        features, proposals, proposal_losses = self.get_proposals(
            images, targets
        )
        detections, detector_losses = self.get_detections_from_proposals(
            features, proposals, targets, images.image_sizes,
            orig_img_sizes, use_ctx_remover
        )
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        return features, proposals, losses, detections

    def get_proposals(self, images: ImageList,
                      targets: List[Dict[str, Tensor]] = None):
        features = self.get_backbone_features(images.tensors)
        proposals, proposal_losses = self.get_proposals_from_features(
            images, features, targets
        )

        return features, proposals, proposal_losses

    def get_proposals_from_features(self, images: ImageList, features,
                                    targets):
        proposals, proposal_losses = self.rpn(images, features, targets)
        return proposals, proposal_losses

    def get_detections_from_proposals(self, features, proposals, targets,
                                      image_sizes, original_image_sizes,
                                      use_ctx_remover):
        detections, detector_losses = self.roi_heads(
            features, proposals, image_sizes,
            use_ctx_remover=use_ctx_remover,
            targets=targets
        )
        detections = self.transform.postprocess(detections, image_sizes,
                                                original_image_sizes)
        return detections, detector_losses

    def get_full_image_embeddings(self, images: torch.Tensor,
                                  use_ctx_remover=False):
        """
        Return the embeddings corresponding to each image entirely in a batch.
        In other words, the detection model will be used as a recognition
        model.

        We feed the feature map that comes out of the backbone through RoI
        alignment, with the entire image region as 'box'. The output will have
        the same shape as the feature map of each RPN bounding box after RoI
        alginment. Hence, these feature maps can be passed through the RoI ID
        head and as such, a recognition loss can be calculated for the
        detection backbone.

        Args:
            images: A batch of images to convert into embeddings. The
                embeddings will be based on the entire images, like in a
                recognition model.
        """
        image_sizes = [img.shape[-2:] for img in images]  # img shape: NCHW

        # The entire image contains a single product
        boxes = [
            torch.tensor(
                [0, 0, img_shape[1], img_shape[0]]
            )[None, ...]
            for img_shape in image_sizes
        ]

        return self.get_box_embeddings(images, boxes, use_ctx_remover)

    def get_box_embeddings(
        self, images: torch.Tensor, boxes: List[torch.Tensor],
        use_ctx_remover=False
    ):
        """
        Return the recognition embeddings corresponding to the given boxes in
        the image.
        """
        if use_ctx_remover:
            box_features, global_features = \
                self.get_box_features(
                    images, boxes, return_global=True
                )
        else:
            box_features = self.get_box_features(
                images, boxes, return_global=False
            )
            global_features = None
        _, _, embeddings = self.roi_heads.box_predictor(
            box_features,
            global_features
        )

        return embeddings

    def get_box_features(self, images, boxes, return_global=False):
        """
        Return the features that correspond to the boxes in the image by
        passing the image through the backbone and applying RoI Pooling.
        """
        image_sizes = [img.shape[-2:] for img in images]  # img shape: NCHW

        features = self.get_backbone_features(images)

        boxes = [box.type_as(features['0']) for box in boxes]

        # Apply RoI pooling so that we get the same feature map shape,
        # irrespective of the size of the input images
        return self.get_box_features_from_features(
            features, boxes, image_sizes, return_global
        )

    def get_box_features_from_features(self, features, boxes, image_sizes,
                                       return_global=False):
        box_features = self.roi_heads.get_box_features_from_features(
            features, boxes, image_sizes
        )
        if return_global:
            global_features = self.roi_heads.get_global_features_from_features(
                features, image_sizes
            )
            return box_features, global_features
        else:
            return box_features

    def get_box_global_features_from_features(self, features, boxes,
                                              image_sizes):
        """
        Return box features with the global features of the respective images
        concatenated behind them.
        """
        return self.roi_heads.get_box_global_features_from_features(
            features, boxes, image_sizes
        )

    def get_backbone_features(self, images):
        assert isinstance(images, torch.Tensor), (
            'The image batch should be of type Tensor'
        )

        features = self.backbone(images)

        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])

        return features

    def get_recog_loss(self, x, targets):
        # TODO: rename to get_recog_loss_from_box_targets
        boxes = [tgt['boxes'] for tgt in targets]
        product_ids = [tgt['product_ids'] for tgt in targets]
        id_embeddings = self.get_box_embeddings(x, boxes,
                                                use_ctx_remover=False)
        return self.get_recog_loss_from_embeddings(
            id_embeddings,
            product_ids
        )

    def get_recog_loss_from_embeddings(self, id_embeddings, product_ids):
        stacked_prod_ids = torch.hstack(product_ids)
        return self.recog_loss_fn(id_embeddings, stacked_prod_ids)


class FastRCNNPredictorWithID(nn.Module):
    """
    Object/background classification + bounding box regression layers + layers
    for computing an identity embedding.
    """
    def __init__(self, in_channels, id_embedding_size, num_classes=2):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
        self.id_encoder = nn.Linear(in_channels, id_embedding_size)

        self.ctx_remover = ContextRemover(in_channels)

    def forward(self, box_features, box_global_features=None):
        """
        Args:
            box_features (tensor): The box features.
            global_features (bool): The corresponding global feature of each
                box feature. If not `None`, they will be concatenated to the
                box features and passed through the context remover.
        """
        if box_features.dim() == 4:
            assert list(box_features.shape[2:]) == [1, 1]
        box_features = box_features.flatten(start_dim=1)
        scores = self.cls_score(box_features)
        bbox_deltas = self.bbox_pred(box_features)

        if box_global_features is not None:
            no_ctx_features = self.ctx_remover(box_global_features)
            id_embeddings = self.id_encoder(no_ctx_features)
            return scores, bbox_deltas, id_embeddings
        else:
            id_embeddings = self.id_encoder(box_features)
            return scores, bbox_deltas, id_embeddings


class RoIHeadsWithID(RoIHeads):
    """
    Modified from torchvision.models.detection.roi_heads.RoIHeads
    """
    def __init__(self, *args, recog_loss_fn=None, **kwargs):
        """
        Args:
            recog_loss_fn: Loss function to compute the recognition loss. If
                `None`, the recognition loss will not be computed.
        """
        super().__init__(*args, **kwargs)
        self.recog_loss_fn = recog_loss_fn

    def postprocess_detections(self,
                               class_logits,    # type: Tensor
                               box_regression,  # type: Tensor
                               id_embedding,  # type: Tensor
                               proposals,       # type: List[Tensor]
                               image_sizes     # type: List[Tuple[int, int]]
                               ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0]
                           for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)
        pred_id_embs_list = id_embedding.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_id_embs = []
        all_labels = []
        all_roi_idxs = []
        for boxes, scores, id_embs, image_shape in zip(pred_boxes_list,
                                                       pred_scores_list,
                                                       pred_id_embs_list,
                                                       image_sizes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            num_rois = len(boxes)
            roi_idxs = torch.arange(num_rois, device=device)
            roi_idxs = roi_idxs[:, None].expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]
            roi_idxs = roi_idxs[:, 1:]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_id_embs.append(id_embs)
            all_labels.append(labels)
            all_roi_idxs.append(roi_idxs)

        return all_boxes, all_scores, all_id_embs, all_labels, all_roi_idxs

    def forward(self,
                features,      # type: Dict[str, Tensor]
                proposals,     # type: List[Tensor]
                image_sizes,  # type: List[Tuple[int, int]]
                use_ctx_remover,
                targets=None,  # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Args:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_sizes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, (
                    'target boxes must of float type'
                )
                assert t["labels"].dtype == torch.int64, (
                    'target labels must of int64 type'
                )

        if targets is not None:
            proposals, _, labels, regression_targets = \
                self.select_training_samples(proposals, targets)
            if self.recog_loss_fn is not None:
                _, product_ids = self.assign_targets_to_proposals(
                    proposals,
                    [tgt['boxes'] for tgt in targets],
                    [tgt['product_ids'] for tgt in targets],
                )
        else:
            labels = None
            regression_targets = None

        box_features = self.get_box_features_from_features(
            features, proposals, image_sizes
        )

        if use_ctx_remover:
            global_features = self.get_global_features_from_features(
                features, image_sizes
            )
            box_global_features = concat_box_and_global_features(
                box_features, global_features, proposals
            )
            class_logits, box_regression, id_embeddings = \
                self.box_predictor(box_features, box_global_features)
        else:
            class_logits, box_regression, id_embeddings = \
                self.box_predictor(box_features)

        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        if labels is not None and regression_targets is not None:
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)

            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg,
            }

            if self.recog_loss_fn is not None:
                stacked_prod_ids = torch.hstack(product_ids)
                recog_loss = self.recog_loss_fn(id_embeddings,
                                                stacked_prod_ids)
                losses['loss_recognition'] = recog_loss

        boxes, scores, id_embs, labels, roi_idxs = \
            self.postprocess_detections(class_logits,
                                        box_regression,
                                        id_embeddings,
                                        proposals,
                                        image_sizes)
        num_images = len(boxes)
        for i in range(num_images):
            result.append(
                {
                    "boxes": boxes[i],
                    "labels": labels[i],
                    "scores": scores[i],
                    "id_embeddings": id_embs[i],
                    "roi_idxs": roi_idxs[i],
                    "proposals": proposals[i],
                }
            )

        return result, losses

    def get_box_features_from_features(self, features, boxes, image_sizes):
        box_features = self.box_roi_pool(
            features, boxes, image_sizes
        )
        return self.box_head(box_features)

    def get_global_features_from_features(self, features, image_sizes):
        full_image_boxes = [
            torch.tensor([
                [0., 0., w, h]
            ]).type_as(features['0'])
            for (h, w) in image_sizes
        ]
        return self.get_box_features_from_features(features, full_image_boxes,
                                                   image_sizes)

    def get_box_global_features_from_features(self, features, boxes,
                                              image_sizes):
        """
        Return box features with the global features of the respective images
        concatenated behind them.
        """
        box_features_ctx = self.get_box_features_from_features(
            features, [
                box.float()
                for box in boxes
            ], image_sizes
        )
        global_features = self.get_global_features_from_features(
            features, image_sizes
        )
        # Repeat and concat to get the same shape as box_features_ctx
        return concat_box_and_global_features(box_features_ctx,
                                              global_features,
                                              boxes)


def concat_box_and_global_features(box_features, global_features, boxes):
    global_features = torch.cat([
        img_global_features.repeat((img_proposals.shape[0], 1))
        for img_global_features, img_proposals in zip(
            global_features, boxes
        )
    ])
    return torch.cat(
        (box_features, global_features),
        dim=1
    )


class ContextRemover(nn.Module):
    def __init__(self, box_head_out_channels=1024):
        super().__init__()

        in_features = box_head_out_channels * 2
        out_features = box_head_out_channels

        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.fc(x)


class PassThroughRCNNTransform(nn.Module):
    """
    Like GeneralizedRCNNTransform, but simply puts the passed-in images into an
    ImageList, without applying any transforms. This is useful when the
    transformations are already applied in the dataloading pipeline.
    """
    def forward(self, images, targets=None):
        image_sizes = [img.shape[-2:] for img in images]
        image_sizes_list: List[Tuple[int, int]] = []

        for image_size in image_sizes:
            assert len(image_size) == 2
            image_sizes_list.append((image_size[0], image_size[1]))

        image_list = ImageList(images, image_sizes_list)
        return image_list, targets

    def postprocess(self, result, image_sizes, original_image_sizes):
        return result


def check_targets(targets):
    check_target_box_shapes(targets)
    check_degenerate_boxes(targets)


def check_target_box_shapes(targets):
    for target in targets:
        boxes = target["boxes"]
        if isinstance(boxes, torch.Tensor):
            if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                raise ValueError(
                    "Expected target boxes to be a tensor "
                    f"of shape [N, 4], got {boxes.shape}."
                )
        else:
            raise ValueError(
                "Expected target boxes to be of type "
                f"Tensor, got {type(boxes)}."
            )


def check_degenerate_boxes(targets):
    for target_idx, target in enumerate(targets):
        boxes = target["boxes"]
        degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
        if degenerate_boxes.any():
            # print the first degenerate box
            bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
            degen_bb: List[float] = boxes[bb_idx].tolist()
            raise ValueError(
                "All bounding boxes should have positive height and "
                "width."
                f" Found invalid box {degen_bb} for target at index "
                f"{target_idx}."
            )
