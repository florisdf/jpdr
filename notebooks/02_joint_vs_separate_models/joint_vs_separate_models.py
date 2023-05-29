import pandas as pd
import torch
from time import time
from torchvision.transforms import functional as F

from mock_num_detections import mock_num_detections


def get_sku_mean_width_heights():
    df = pd.read_pickle('../../.cache/sku110_train.pkl')
    im_width_mean = df['width'].mean()
    im_height_mean = df['height'].mean()

    all_boxes = torch.cat([t['boxes'] for t in df['target']])
    all_box_widths = all_boxes[:, ::2].diff().float()
    all_box_heights = all_boxes[:, 1::2].diff().float()
    box_width_mean = float(all_box_widths.mean())
    box_height_mean = float(all_box_heights.mean())

    return (
        im_width_mean, im_height_mean,
        box_width_mean, box_height_mean
    )


def get_sku_num_boxes_per_img():
    df = pd.read_pickle('../../.cache/sku110_train.pkl')
    return torch.tensor([len(t['boxes']) for t in df['target']]).float()


def random_tfm_crop_boxes(
    orig_im_width, orig_im_height,
    orig_box_width, orig_box_height,
    tfm_im_width, tfm_im_height,
    num_boxes
):
    tfm_x_scale = tfm_im_width / orig_im_width
    tfm_y_scale = tfm_im_height / orig_im_height

    tfm_box_width = int(orig_box_width * tfm_x_scale)
    tfm_box_height = int(orig_box_height * tfm_y_scale)

    crop_boxes_x1 = torch.randint(tfm_im_width - tfm_box_width, (num_boxes, 1))
    crop_boxes_y1 = torch.randint(
        tfm_im_height - tfm_box_height, (num_boxes, 1)
    )

    return torch.hstack([
        crop_boxes_x1,
        crop_boxes_y1,
        crop_boxes_x1 + tfm_box_width,
        crop_boxes_y1 + tfm_box_height,
    ])


def time_separate_det_recog_nets(
    faster_rcnn, rn50,
    device,
    det_input_size,
    recog_input_size,
    dets_per_img,
):
    mock_num_detections(dets_per_img)
    faster_rcnn = faster_rcnn.eval()
    rn50 = rn50.eval()
    faster_rcnn.to(device)
    rn50.to(device)
    x = torch.randn((1, 3, det_input_size, det_input_size)).to(device)

    # Output from get_sku_mean_width_heights()
    im_width_mean = 2435.999269628728
    im_height_mean = 3211.501643335362
    box_width_mean = 120.7902603149414
    box_height_mean = 190.1041259765625

    crop_boxes = random_tfm_crop_boxes(
        im_width_mean, im_height_mean,
        box_width_mean, box_height_mean,
        det_input_size, det_input_size,
        dets_per_img
    )

    start = time()

    with torch.no_grad():
        faster_rcnn(x)

    crops = torch.cat([
        F.center_crop(
            F.resize(
                F.crop(x, top=box[1], left=box[0],
                       height=box[3] - box[1], width=box[2] - box[0]),
                recog_input_size
            ),
            recog_input_size
        )
        for box in crop_boxes
    ])

    with torch.no_grad():
        rn50(crops)

    end = time()

    return end - start


def time_joint_det_recog(
    faster_rcnn, recog_head, device,
    det_input_size, dets_per_img
):
    mock_num_detections(dets_per_img)
    faster_rcnn.to(device)
    recog_head.to(device)
    faster_rcnn = faster_rcnn.eval()
    recog_head = recog_head.eval()
    x = torch.randn((1, 3, det_input_size, det_input_size)).to(device)
    features = torch.randn((dets_per_img, recog_head.in_features)).to(device)

    start = time()

    with torch.no_grad():
        faster_rcnn(x)
        recog_head(features)

    end = time()

    return end - start


def time_model(
    model, device, input_size,
    dets_per_img
):
    model.to(device)
    model = model.eval()
    x = torch.randn((1, 3, input_size, input_size)).to(device)
    start = time()

    with torch.no_grad():
        model(x)

    end = time()

    return end - start
