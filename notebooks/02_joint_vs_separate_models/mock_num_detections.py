import sys; sys.path.append('../..')  # noqa
from jpdr.models.joint_rcnn import RoIHeadsWithID
from torchvision.models.detection.rpn import RegionProposalNetwork
from torchvision.models.detection.roi_heads import RoIHeads
import functools
import math
import torch


def mock_num_detections(
    num_detections,
):
    filter_proposals = RegionProposalNetwork.filter_proposals
    postprocess_detections = RoIHeads.postprocess_detections
    postprocess_detections_id = RoIHeadsWithID.postprocess_detections

    @functools.wraps(filter_proposals)
    def mock_filter_proposals(self, proposals, objectness, image_shapes,
                              num_anchors_per_level):
        boxes, scores = filter_proposals(self, proposals, objectness,
                                         image_shapes, num_anchors_per_level)
        return mock_boxes_and_scores(boxes, scores, image_shapes,
                                     num_detections)

    @functools.wraps(postprocess_detections)
    def mock_postprocess_detections(self, class_logits, box_regression,
                                    proposals, image_sizes):
        boxes, scores, labels = postprocess_detections(
            self,
            class_logits, box_regression,
            proposals, image_sizes
        )
        boxes, scores = mock_boxes_and_scores(boxes, scores, image_sizes,
                                              num_detections)
        return boxes, scores, labels

    @functools.wraps(postprocess_detections_id)
    def mock_postprocess_detections_id(self, class_logits, box_regression,
                                       id_embedding, proposals, image_sizes):
        boxes, scores, id_embs, labels, roi_idxs = postprocess_detections_id(
            self,
            class_logits, box_regression,
            id_embedding, proposals,
            image_sizes
        )
        boxes, scores = mock_boxes_and_scores(boxes, scores, image_sizes,
                                              num_detections)
        return boxes, scores, id_embs, labels, roi_idxs

    RegionProposalNetwork.filter_proposals = mock_filter_proposals
    RoIHeads.postprocess_detections = mock_postprocess_detections
    RoIHeadsWithID.postprocess_detections = mock_postprocess_detections_id


def mock_boxes_and_scores(boxes, scores, image_shapes, mock_num_boxes):
    mock_boxes = []
    mock_scores = []
    for old_boxes, old_scores, img_shape in zip(boxes, scores,
                                                image_shapes):
        image_height, image_width = img_shape
        new_boxes = get_box_grid(mock_num_boxes, image_width, image_height)
        new_boxes = new_boxes.type_as(old_boxes)
        mock_boxes.append(new_boxes)
        mock_scores.append(torch.ones(mock_num_boxes).type_as(old_scores))

    return mock_boxes, mock_scores


def get_box_grid(num_boxes, image_width, image_height):
    sqrt_n_boxes = math.sqrt(num_boxes)

    dx = image_width/sqrt_n_boxes
    dy = image_height/sqrt_n_boxes
    xs = torch.arange(0, image_width + dx, dx)
    ys = torch.arange(0, image_height + dy, dy)

    x1s = xs[:-1]
    x2s = xs[1:]
    y1s = ys[:-1]
    y2s = ys[1:]

    mins = torch.cartesian_prod(x1s, y1s)
    maxs = torch.cartesian_prod(x2s, y2s)

    return torch.hstack([mins, maxs])[:num_boxes]
