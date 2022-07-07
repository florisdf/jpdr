from typing import Tuple, Dict, Optional, Union, List

import numpy as np
import torch
import torchvision
from torch import nn, Tensor
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T, InterpolationMode
from PIL import Image


class Compose:
    def __init__(self, transforms, bbox_params=None):
        self.transforms = transforms

    def __call__(self, image, bboxes, labels):
        target = {
            'boxes': bboxes,
            'labels': labels
        }
        for t in self.transforms:
            image, target = t(image, target)
        return {
            'image': image,
            'bboxes': target['boxes'].type(torch.float),
            'labels': target['labels'],
        }


class ToTensor(nn.Module):
    def __init__(self, to_dtype=torch.float):
        super().__init__()
        self.dtype = to_dtype

    def forward(
        self, image: Image, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        img = torch.as_tensor(np.array(image))
        img = img.view(image.size[1], image.size[0], len(image.getbands()))
        # put it from HWC to CHW format
        img = img.permute((2, 0, 1))

        if self.dtype is not None:
            img = F.convert_image_dtype(img, self.dtype)

        return img, target


class Normalize(torchvision.transforms.Normalize):
    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = super().forward(image)
        return image, target


class Crop(nn.Module):
    def __init__(self, height, width, how="center", fill=0,
                 padding_mode="constant", min_new_tgt_area_pct=0.0):
        super().__init__()
        self.crop_height = height
        self.crop_width = width
        self.fill = fill
        self.padding_mode = padding_mode
        self.min_new_tgt_area_pct = min_new_tgt_area_pct
        assert how in ["center", "random"]
        self.how = how

    def forward(self, img, target=None):
        width, height = F._get_image_size(img)
        new_height = min(height, self.crop_height)
        new_width = min(width, self.crop_width)

        if new_height != height or new_width != width:
            offset_height = max(height - self.crop_height, 0)
            offset_width = max(width - self.crop_width, 0)

            if self.how == "random":
                r = torch.rand(1)
            elif self.how == "center":
                r = 0.5
            top = int(offset_height * r)
            left = int(offset_width * r)

            img, target = crop(img, target, top, left, new_height,
                               new_width, self.min_new_tgt_area_pct)

        pad_bottom = max(self.crop_height - new_height, 0)
        pad_right = max(self.crop_width - new_width, 0)
        if pad_bottom != 0 or pad_right != 0:
            img, target = pad(img, target, [0, 0, pad_right, pad_bottom],
                              self.fill, self.padding_mode)

        return img, target


def pad(img, target, padding, fill=0, padding_mode="constant"):
    # Taken from the functional_tensor.py pad
    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
    elif len(padding) == 1:
        pad_left = pad_right = pad_top = pad_bottom = padding[0]
    elif len(padding) == 2:
        pad_left = pad_right = padding[0]
        pad_top = pad_bottom = padding[1]
    else:
        pad_left = padding[0]
        pad_top = padding[1]
        pad_right = padding[2]
        pad_bottom = padding[3]

    padding = [pad_left, pad_top, pad_right, pad_bottom]
    img = F.pad(img, padding, fill, padding_mode)
    if target is not None:
        target["boxes"][:, 0::2] += pad_left
        target["boxes"][:, 1::2] += pad_top
        if "masks" in target:
            target["masks"] = F.pad(target["masks"], padding, 0,
                                    "constant")

    return img, target


def get_boxes_areas(boxes):
    return (boxes[:, ::2].diff() * boxes[:, 1::2].diff()).squeeze()


def crop(
    img, target, top, left, height, width,
    min_new_tgt_area_pct
):
    img = F.crop(img, top, left, height, width)
    if target is not None:
        boxes = target["boxes"]
        orig_areas = get_boxes_areas(boxes)
        boxes[:, 0::2] -= left
        boxes[:, 1::2] -= top
        boxes[:, 0::2].clamp_(min=0, max=width)
        boxes[:, 1::2].clamp_(min=0, max=height)

        min_new_areas = min_new_tgt_area_pct * orig_areas
        new_areas = get_boxes_areas(boxes)
        is_valid = (
            (boxes[:, 0] < boxes[:, 2])
            & (boxes[:, 1] < boxes[:, 3])
            & (new_areas >= min_new_areas)
        )

        target["boxes"] = boxes[is_valid]
        target["labels"] = target["labels"][is_valid]
        if "masks" in target:
            target["masks"] = F.crop(target["masks"][is_valid], top, left,
                                     height, width)
        if "product_ids" in target:
            target["product_ids"] = target["product_ids"][is_valid]

    return img, target


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            if target is not None:
                width, height = F._get_image_size(image)
                target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
        return image, target


class RandomShortestSize(nn.Module):
    """
    Resize the image and the boxes (maintaining the aspect ratio) in such a
    way that the image's shortest size is equal to one of the randomly chose
    elements of the list `min_size`. If `min_size` contains only one element
    (or when `min_size` is an integer), every image will have the same shortest
    size.

    Args:
        min_size: A list of shortest sizes to randomly choose from. If given as
            an integer, the shortest size of each image will be resized to that
            size.
        max_size: The maximum size that an image can have.
        interpolation: The interpolation to use for resizing.
    """
    def __init__(
        self,
        min_size: Union[List[int], Tuple[int], int],
        max_size: int = None,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ):
        super().__init__()
        self.min_size = (
            [min_size] if isinstance(min_size, int)
            else list(min_size)
        )
        self.max_size = max_size
        self.interpolation = interpolation

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        orig_width, orig_height = F._get_image_size(image)

        min_size = self.min_size[
            torch.randint(len(self.min_size), (1,)).item()
        ]
        if self.max_size is not None:
            r = min(min_size / min(orig_height, orig_width),
                    self.max_size / max(orig_height, orig_width))
        else:
            r = min_size / min(orig_height, orig_width)

        new_width = int(orig_width * r)
        new_height = int(orig_height * r)

        image = F.resize(image, [new_height, new_width],
                         interpolation=self.interpolation)

        if target is not None:
            target["boxes"][:, 0::2] *= new_width / orig_width
            target["boxes"][:, 1::2] *= new_height / orig_height
            if "masks" in target:
                target["masks"] = F.resize(
                    target["masks"], [new_height, new_width],
                    interpolation=InterpolationMode.NEAREST
                )

        return image, target
