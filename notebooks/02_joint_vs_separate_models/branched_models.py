#!/usr/bin/env python
# coding: utf-8

from random import shuffle
from pathlib import Path

import pandas as pd
from tqdm.notebook import tqdm
import torch

from joint_vs_separate_models import time_model
import sys; sys.path.append('../..')  # noqa
from train import get_joint_model

from mock_num_detections import mock_num_detections


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
