#!/usr/bin/env python
# coding: utf-8

from itertools import product
from random import shuffle
from pathlib import Path

import pandas as pd
from tqdm.notebook import tqdm
import torch

from joint_vs_separate_models import time_model
import sys; sys.path.append('../..')  # noqa
from train import get_joint_model


from jpdr.models.joint_rcnn import RoIHeadsWithID
from torchvision.models.detection.rpn import RegionProposalNetwork
import functools
import math

postprocess_detections = RoIHeadsWithID.postprocess_detections
filter_proposals = RegionProposalNetwork.filter_proposals


def mock_num_detections(num_detections):
    @functools.wraps(filter_proposals)
    def mock_filter_proposals(self, proposals, objectness, image_shapes,
                              num_anchors_per_level):
        boxes, scores = filter_proposals(self, proposals, objectness,
                                         image_shapes, num_anchors_per_level)
        return mock_boxes_and_scores(boxes, scores, image_shapes,
                                     num_detections)

    RegionProposalNetwork.filter_proposals = mock_filter_proposals

    @functools.wraps(postprocess_detections)
    def mock_postprocess_detections(self, class_logits, box_regression,
                                    id_embedding, proposals,
                                    image_sizes):
        boxes, scores, id_embs, labels, roi_idxs = postprocess_detections(
            self,
            class_logits, box_regression,
            id_embedding, proposals,
            image_sizes
        )
        boxes, scores = mock_boxes_and_scores(boxes, scores, image_sizes,
                                              num_detections)
        return boxes, scores, id_embs, labels, roi_idxs

    RoIHeadsWithID.postprocess_detections = mock_postprocess_detections


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


def get_timing_results(
    iters=5,
    grid_dets_per_img=list(range(10, 201, 10)),
    lazy=False,
):
    df_res_path = Path('branched_models_results.pkl')
    if lazy and df_res_path.exists():
        return pd.read_pickle(df_res_path)

    input_size = 800
    grid_branch = [
        'layer1',
        'layer2',
        'layer3',
        'layer4',
        'bbox_head',
        'roi_head'
    ]
    grid_device = [
        torch.device('cuda'),
        torch.device('cpu')
    ]

    results = []

    for branch_layer in tqdm(grid_branch):
        model = get_joint_model(backbone_name='resnet50',
                                num_train_classes=100,
                                use_fpn=False,
                                branch_layer=branch_layer)
        for device in tqdm(grid_device, leave=False):
            for _ in range(iters):
                shuffle(grid_dets_per_img)
                for dets_per_img in grid_dets_per_img:
                    mock_num_detections(dets_per_img)
                    t = time_model(model, device, input_size, dets_per_img)
                    results.append({
                        'Time (s)': t,
                        'Branch layer': branch_layer,
                        'Input size': input_size,
                        'Number of products': dets_per_img,
                        'Device': device.type,
                    })
    df_res = pd.DataFrame(results)
    df_res.to_pickle(df_res_path)

    return df_res
