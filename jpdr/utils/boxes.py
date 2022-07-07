import torch
from torchvision.ops import boxes as box_ops


def clip_boxes(boxes: torch.Tensor, width, height):
    """
    Clip the bounding boxes inplace to not cross the image borders. `boxes` is
    assumed to be a tensor where each row is like [x_min, y_min, x_max, y_max].
    """
    x_min, y_min, x_max, y_max = boxes.T
    boxes[:, 0] = torch.clip(x_min, min=0, max=width)
    boxes[:, 1] = torch.clip(y_min, min=0, max=height)
    boxes[:, 2] = torch.clip(x_max, min=0, max=width)
    boxes[:, 3] = torch.clip(y_max, min=0, max=height)

    return boxes


def get_boxes_boundary(boxes):
    min_x = boxes[:, 0].min(dim=0)[0]
    min_y = boxes[:, 1].min(dim=0)[0]
    max_x = boxes[:, 2].max(dim=0)[0]
    max_y = boxes[:, 3].max(dim=0)[0]
    return min_x, min_y, max_x, max_y


def make_boxes_batchable(
    boxes, to_int=True, crop_box_size='max', max_size=None,
    n=None, replace=False, max_rand_shift=0
):
    """
    Return boxes with the same center as the given boxes, but change the width
    and height of each box in such a way that they are equal.

    Args:
        boxes: N x 4 tensor with box coordinates [x_min, y_min, x_max, y_max].
        to_int: if True, the returned coordinates will be integers.
        crop_box_size: How to make the sizes equal. This should be the name of
            a tensor method or a tuple of (width, height). If given as an
            integer, the value will be used for both width and height.
        max_size: The maximum size of the crops. This is only used if
            `crop_box_size` is set to the name of a tensor method.
        n (int): The number of boxes to randomly sample. If `None`, all boxes
            will be used (without shuffling).
        replace: If True, sample with replacement. Only used when n is not
            `None`.
        max_rand_shift: If greater than zero, each box will get an additional
            random shift in the vertical and horizontal dimensions between zero
            and `max_rand_shift`.
    """
    if isinstance(crop_box_size, str):
        if crop_box_size not in ['max', 'min', 'mean', 'median']:
            raise ValueError(
                f'Unsupported value for `crop_box_size`: {crop_box_size}'
            )
    elif isinstance(crop_box_size, tuple):
        if len(crop_box_size) != 2:
            raise ValueError(
                '`crop_box_size` should be a tuple of length 2, got '
                f'length {len(crop_box_size)} instead.'
            )
    elif isinstance(crop_box_size, int):
        crop_box_size = (crop_box_size, crop_box_size)
    else:
        raise ValueError(
            f'Unsupported value for `crop_box_size`: {crop_box_size}'
        )

    if isinstance(crop_box_size, tuple):
        new_width, new_height = crop_box_size
        new_width = torch.tensor(new_width)
        new_height = torch.tensor(new_height)
    else:
        width, height = get_width_height_of_boxes(boxes)
        new_width, new_height = (
            getattr(width, crop_box_size)(),
            getattr(height, crop_box_size)()
        )
        if max_size is not None:
            new_width, new_height = (
                min(new_width, torch.tensor(max_size)),
                min(new_height, torch.tensor(max_size))
            )

    if to_int:
        new_width, new_height = new_width.int(), new_height.int()
        if not new_width % 2 == 0:
            new_width += 1
        if not new_height % 2 == 0:
            new_height += 1

    centers = get_center_of_boxes(boxes)

    if to_int:
        centers = centers.int()

    if max_rand_shift != 0:
        centers += torch.randint(-max_rand_shift, max_rand_shift,
                                 centers.shape)

    new_boxes = create_batchable_boxes_from_width_height_centers(
        new_width, new_height, centers
    )

    if n is not None:
        new_boxes = sample_boxes(new_boxes, n, replacement=replace)

    if to_int:
        new_boxes = new_boxes.int()

    return new_boxes


def make_image_list_boxes_batchable(
    image_list_boxes, *args, **kwargs
):
    """
    Args:
        image_list_boxes (list): List of N x 4 tensors.
    """
    num_boxes_per_img = [len(img_boxes) for img_boxes in image_list_boxes]
    stacked_boxes = torch.vstack(image_list_boxes)
    batchable_boxes = make_boxes_batchable(
        stacked_boxes, *args, **kwargs
    )
    return [
        batchable_boxes[
            slice(
                (num_boxes_per_img[i - 1] if i > 0 else 0),
                num_boxes_per_img[i]
            ),
            :
        ]
        for i in range(len(num_boxes_per_img))
    ]


def create_batchable_boxes_from_width_height_centers(
        width, height, centers
):
    centers = centers.int()
    half_width = int(width / 2)
    half_height = int(height / 2)

    return torch.stack([
        centers[:, 0] - half_width,
        centers[:, 1] - half_height,
        centers[:, 0] + half_width,
        centers[:, 1] + half_height,
    ], dim=1).type_as(centers)


def get_width_height_of_boxes(boxes):
    return (boxes[:, ::2].diff().squeeze(), boxes[:, 1::2].diff().squeeze())


def get_area_of_boxes(boxes):
    w, h = get_width_height_of_boxes(boxes)
    return torch.mul(w, h)


def get_center_of_boxes(boxes):
    width, height = get_width_height_of_boxes(boxes)
    return torch.hstack([
        (boxes[:, 0] + width/2)[:, None],
        (boxes[:, 1] + height/2)[:, None],
    ]).type_as(boxes)


def shift_boxes_to_fit_img(boxes, width, height):
    widths, heights = get_width_height_of_boxes(boxes)
    assert torch.all(widths <= width)
    assert torch.all(heights <= height)

    too_left = boxes[:, 0] < 0
    r_shift = -boxes[too_left, 0]
    boxes[too_left, 0::2] += r_shift[:, None]

    too_right = boxes[:, 2] > width
    l_shift = boxes[too_right, 2] - width
    boxes[too_right, 0::2] -= l_shift[:, None]

    too_high = boxes[:, 1] < 0
    down_shift = -boxes[too_high, 1]
    boxes[too_high, 1::2] += down_shift[:, None]

    too_low = boxes[:, 3] > height
    up_shift = boxes[too_low, 3] - height
    boxes[too_low, 1::2] -= up_shift[:, None]

    return boxes


def drop_boxes_outside_img(boxes, width, height, max_out_pct=0.5):
    is_outside = get_boxes_outside_img_mask(boxes, width, height, max_out_pct)
    return boxes[~is_outside]


def drop_overlapping_boxes(boxes, iou_thresh):
    is_overlapping = get_overlapping_boxes_mask(boxes, iou_thresh)
    return boxes[~is_overlapping]


def get_overlapping_boxes_mask(boxes, iou_thresh):
    boxes = boxes.float()
    idxs = box_ops.nms(boxes,
                       torch.ones(len(boxes)).type_as(boxes),
                       iou_thresh)
    mask = torch.ones(boxes.shape[0]).type_as(boxes)
    mask.scatter_(0, idxs, 0.0)
    return mask.bool()


def get_boxes_outside_img_mask(boxes, width, height, max_out_pct=0.5):
    """
    Return a boolean mask that indicates which boxes are outside an image with
    given width and height.
    """
    widths, heights = get_width_height_of_boxes(boxes)
    assert torch.all(widths <= width)
    assert torch.all(heights <= height)

    box_areas = widths * heights
    max_out = box_areas * max_out_pct
    out_left = -boxes[:, 0] * heights
    out_top = -boxes[:, 1] * widths
    out_right = (boxes[:, 2] - width) * heights
    out_bottom = (boxes[:, 3] - height) * widths

    too_left = out_left > max_out
    too_top = out_top > max_out
    too_right = out_right > max_out
    too_bottom = out_bottom > max_out

    return too_left | too_top | too_right | too_bottom


def generate_uniform_boxes(nrows, ncols, width, height):
    """
    Return uniformly distributed boxes.
    """
    width = int(width / ncols)
    height = int(height / nrows)

    centers = torch.tensor([
        [(i + 0.5)*width, (j + 0.5)*height]
        for i in range(ncols)
        for j in range(nrows)
    ])

    return create_batchable_boxes_from_width_height_centers(
        width, height, centers
    )


def sample_boxes(boxes, num_samples, replacement=False):
    prob = torch.ones(len(boxes))/len(boxes)
    idxs = prob.multinomial(num_samples=num_samples,
                            replacement=replacement)
    return boxes[idxs, :]
