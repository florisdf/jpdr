from copy import copy

import torch

import jpdr.transforms as T
import jpdr.utils.boxes as B


def get_crop_boxes_from_targets(
    targets, crop_box_size, max_size=None, n=None,
    replace=False, max_rand_shift=0,
):
    tgt_boxes = [tgt['boxes'] for tgt in targets]
    return B.make_image_list_boxes_batchable(
        tgt_boxes, to_int=True, crop_box_size=crop_box_size,
        max_size=max_size, n=n, replace=replace, max_rand_shift=max_rand_shift
    )


def crop_and_batch_boxes(x, crop_boxes, targets=None,
                         min_tgt_area_pct=0.0):
    """
    Crop out all boxes from their respective images and put the crops
    into a batch. If the boxes fall outside the image, the image will be padded
    with zeros until all boxes can be used to crop the image. If the boxes
    don't have the same width and height, the max width and height will be used
    for the batch and the images will be padded with zeros until they have that
    width and height.
    """
    crops = []
    crop_tgts = []
    if targets is None:
        ignore_targets = True
        targets = [None]*len(x)
    else:
        ignore_targets = False

    for img, tgt, img_crop_boxes in zip(x, targets, crop_boxes):
        if len(img_crop_boxes) == 0:
            continue
        boxes_boundary = B.get_boxes_boundary(img_crop_boxes)
        padding = get_padding_to_fit_box(img, boxes_boundary)
        img, tgt = T.pad(img, tgt, padding)
        img_crop_boxes[:, 0::2] += padding[0]
        img_crop_boxes[:, 1::2] += padding[1]

        for crop_box in img_crop_boxes.int():
            if tgt is not None:
                tgt_copy = {
                    k: v.clone() if isinstance(v, torch.Tensor)
                    else copy(v)
                    for k, v in tgt.items()
                }
            else:
                tgt_copy = None
            crop, new_tgt = T.crop(img, tgt_copy,
                                   top=crop_box[1],
                                   left=crop_box[0],
                                   height=crop_box[3] - crop_box[1],
                                   width=crop_box[2] - crop_box[0],
                                   min_new_tgt_area_pct=min_tgt_area_pct)
            h, w = crop.shape[-2:]
            crops.append(crop)
            crop_tgts.append(new_tgt)

    all_boxes = torch.vstack([
        torch.tensor([0, 0, crop.shape[-1], crop.shape[-2]])
        for crop in crops
    ])

    boundary = B.get_boxes_boundary(all_boxes)

    for i in range(len(crops)):
        crops[i], crop_tgts[i] = pad_to_fit_box(
            crops[i], crop_tgts[i], boundary
        )

    crops = torch.stack(crops, dim=0)

    return (crops, crop_tgts) if not ignore_targets else crops


def pad_to_fit_boxes(img, target, boxes):
    boundary = B.get_boxes_boundary(boxes)
    return pad_to_fit_box(img, target, boundary)


def pad_to_fit_box(img, target, box):
    padding = get_padding_to_fit_box(img, box)
    img, target = T.pad(img, target, padding)
    return img, target


def get_padding_to_fit_box(img, box):
    box = [
        coord.item() if isinstance(coord, torch.Tensor)
        else coord
        for coord in box
    ]

    min_x, min_y, max_x, max_y = box
    im_height, im_width = img.shape[-2:]

    pad_left = max(0, -min_x)
    pad_top = max(0, -min_y)
    pad_right = max(0, max_x - im_width)
    pad_bottom = max(0, max_y - im_height)

    pad_left, pad_right = equalize_padding(pad_left, pad_right)
    pad_top, pad_bottom = equalize_padding(pad_top, pad_bottom)

    return [pad_left, pad_top, pad_right, pad_bottom]


def equalize_padding(pad_1, pad_2):
    pad_total = pad_1 + pad_2
    pad_quot = pad_total//2
    pad_rest = pad_total % 2

    if pad_1 > pad_2:
        return pad_quot + pad_rest, pad_quot
    else:
        return pad_quot, pad_quot + pad_rest
