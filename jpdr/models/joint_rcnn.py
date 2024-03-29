from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn, Tensor

import torch.nn.functional as F
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import TwoMLPHead
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss
from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead
from torchvision.ops import boxes as box_ops
from torchvision.ops import MultiScaleRoIAlign


class JointRCNN(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        detect_out_channels: int,
        recog_out_channels: int,
        anchor_sizes: Tuple[Tuple[int]],
        featmap_names_detect: List[str],
        featmap_names_recog: List[str],
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

        shared_bbox_head=False,

        use_recog_loss_on_forward=False,
    ):
        """
        Args:
            backbone: The backbone of the detector. Takes a batch of images as
                input and returns a batch of feature maps for the RPN. The RPN
                regions are then 'cropped' out of the feature maps (with RoI
                pooling) and passed through the RoI heads.
            detect_out_channels: The number of output channels in the feature
                map extracted by the backbone for the detection branch.
            recog_out_channels: The number of output channels in the feature
                map extracted by the backbone for the recognition branch.
            box_head_out_channels (int): The number of output channels of the
                box head.
            anchor_sizes: For backbones without FPN, use e.g. `((32, 64, 128,
                256, 512),)`. For backbone with FPN, use e.g. `((32,), (64,),
                (128,), (256,), (512,))`.
            featmap_names_detect: The names of the feature maps (in the ordered
                dict of feature maps returned by the backbone) that will be
                used for pooling in the detection branch.
            featmap_names_recog: Same as `featmap_names_detect` but for the
                recognition branch.
            shared_bbox_head: If True, use the same bbox head for recognition
                and detection.
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
            detect_out_channels,
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

        # Box head
        box_head_detect = TwoMLPHead(
            detect_out_channels * roi_output_size ** 2,
            box_head_out_channels
        )
        if shared_bbox_head:
            box_head_recog = box_head_detect
        else:
            box_head_recog = TwoMLPHead(
                recog_out_channels * roi_output_size ** 2,
                box_head_out_channels
            )

        # RoI Heads
        box_predictor = FastRCNNPredictorWithID(box_head_out_channels,
                                                id_embedding_size,
                                                num_classes=box_num_classes)

        self.featmap_names_detect = featmap_names_detect
        self.featmap_names_recog = featmap_names_recog

        box_roi_pool_detect = MultiScaleRoIAlign(
            featmap_names=featmap_names_detect,
            output_size=roi_output_size,
            sampling_ratio=2
        )
        box_roi_pool_recog = MultiScaleRoIAlign(
            featmap_names=featmap_names_recog,
            output_size=roi_output_size,
            sampling_ratio=2
        )
        roi_heads = RoIHeadsWithID(
            # Box
            box_roi_pool_detect, box_roi_pool_recog,
            box_head_detect, box_head_recog,
            box_predictor,
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

    def forward(self, images, targets=None):
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
            self.forward_with_intermediary_results(images, targets)
        return losses, detections

    def forward_with_intermediary_results(self, images, targets=None):
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
            orig_img_sizes
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
        features = self.backbone_features_to_detect_features(features)
        proposals, proposal_losses = self.rpn(images, features, targets)
        return proposals, proposal_losses

    def get_detections_from_proposals(self, features, proposals, targets,
                                      image_sizes, original_image_sizes):
        detect_features = self.backbone_features_to_detect_features(features)
        recog_features = self.backbone_features_to_recog_features(features)
        detections, detector_losses = self.roi_heads(
            detect_features, recog_features, proposals, image_sizes,
            targets=targets
        )
        detections = self.transform.postprocess(detections, image_sizes,
                                                original_image_sizes)
        return detections, detector_losses

    def get_full_image_embeddings(self, images: torch.Tensor):
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

        return self.get_box_embeddings(images, boxes)

    def get_box_embeddings(
        self, images: torch.Tensor, boxes: List[torch.Tensor],
    ):
        """
        Return the recognition embeddings corresponding to the given boxes in
        the image.
        """
        box_features_detect, box_features_recog = self.get_box_features(
            images, boxes
        )
        _, _, embeddings = self.roi_heads.box_predictor(
            box_features_detect,
            box_features_recog,
        )

        return embeddings

    def get_box_features(self, images, boxes):
        """
        Return the features that correspond to the boxes in the image by
        passing the image through the backbone and applying RoI Pooling.
        """
        image_sizes = [img.shape[-2:] for img in images]  # img shape: NCHW

        features = self.get_backbone_features(images)

        boxes = [box.type_as(features[list(features.keys())[0]]) for box in boxes]

        # Apply RoI pooling so that we get the same feature map shape,
        # irrespective of the size of the input images
        return self.get_box_features_from_features(
            features, boxes, image_sizes
        )

    def backbone_features_to_detect_features(self, features):
        """
        Only keep the backbone features that will be used by the detection
        branch.
        """
        return OrderedDict([
            (k, v) for k, v in features.items()
            if k in self.featmap_names_detect
        ])

    def backbone_features_to_recog_features(self, features):
        """
        Only keep the backbone features that will be used by the recognition
        branch.
        """
        return OrderedDict([
            (k, v) for k, v in features.items()
            if k in self.featmap_names_recog
        ])

    def get_box_features_from_features(self, features, boxes, image_sizes):
        detect_features = self.backbone_features_to_detect_features(features)
        recog_features = self.backbone_features_to_recog_features(features)
        return self.roi_heads.get_box_features_from_features(
            detect_features, recog_features, boxes, image_sizes
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
        id_embeddings = self.get_box_embeddings(x, boxes)
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

    def forward(self, box_features_detect, box_features_recog):
        """
        Args:
            box_features_detect (tensor): The box features to use for
                detection.
            box_features_recog (tensor): The box features to use for
                recognition.
        """
        box_features_detect = flatten_box_features(box_features_detect)
        box_features_recog = flatten_box_features(box_features_recog)

        scores = self.cls_score(box_features_detect)
        bbox_deltas = self.bbox_pred(box_features_detect)

        id_embeddings = self.id_encoder(box_features_recog)
        return scores, bbox_deltas, id_embeddings


def flatten_box_features(box_features):
    if box_features.dim() == 4:
        assert list(box_features.shape[2:]) == [1, 1]
    return box_features.flatten(start_dim=1)


class RoIHeadsWithID(RoIHeads):
    """
    Modified from torchvision.models.detection.roi_heads.RoIHeads
    """
    def __init__(self,
                 box_roi_pool_detect, box_roi_pool_recog,
                 box_head_detect, box_head_recog,
                 box_predictor,
                 # Faster R-CNN training
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 bbox_reg_weights,
                 # Faster R-CNN inference
                 score_thresh,
                 nms_thresh,
                 detections_per_img,
                 recog_loss_fn=None):
        """
        Args:
            recog_loss_fn: Loss function to compute the recognition loss. If
                `None`, the recognition loss will not be computed.
        """
        super().__init__(
            box_head=box_head_detect,
            box_roi_pool=box_roi_pool_detect,
            box_predictor=box_predictor,
            # Faster R-CNN training
            fg_iou_thresh=fg_iou_thresh,
            bg_iou_thresh=bg_iou_thresh,
            batch_size_per_image=batch_size_per_image,
            positive_fraction=positive_fraction,
            bbox_reg_weights=bbox_reg_weights,
            # Faster R-CNN inference
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
            detections_per_img=detections_per_img,
        )
        self.recog_loss_fn = recog_loss_fn

        self.box_head_detect = box_head_detect
        self.box_head_recog = box_head_recog

        self.box_roi_pool_detect = box_roi_pool_detect
        self.box_roi_pool_recog = box_roi_pool_recog

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
                detect_features,      # type: Dict[str, Tensor]
                recog_features,      # type: Dict[str, Tensor]
                proposals,     # type: List[Tensor]
                image_sizes,  # type: List[Tuple[int, int]]
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

        box_features_detect, box_features_recog = \
            self.get_box_features_from_features(
                detect_features, recog_features,
                proposals, image_sizes
            )

        class_logits, box_regression, id_embeddings = \
            self.box_predictor(box_features_detect, box_features_recog)

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

    def get_box_features_from_features(self, detect_features, recog_features,
                                       boxes, image_sizes):
        box_features_detect = self.box_roi_pool_detect(
            detect_features, boxes, image_sizes
        )
        box_features_recog = self.box_roi_pool_recog(
            recog_features, boxes, image_sizes
        )
        return (
            self.box_head_detect(box_features_detect),
            self.box_head_recog(box_features_recog)
        )


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
