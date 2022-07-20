from pathlib import Path
from copy import deepcopy
import io
from contextlib import redirect_stdout
from pycocotools.coco import COCO
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

from ..utils.convert_tensor import convert_to_item, convert_to_list


class DetectionDataset(Dataset):
    GALLERY_SUBSETS = [
        'val_gallery'
    ]
    QUERY_SUBSETS = [
        'val_query'
    ]

    def __init__(
        self,
        data_path: str,
        transform=None,
        subset: str = 'val_gallery',
    ):
        super().__init__()
        self._coco = None
        self.init_coco_fields()
        self.subset = subset
        self.check_subset()
        self.data_path = Path(data_path)
        self.transform = transform

        all_gallery_df = pd.concat([
            self.get_gallery_df(subset)
            for subset in self.GALLERY_SUBSETS
        ])
        all_query_df = pd.concat([
            self.get_query_df(subset)
            for subset in self.QUERY_SUBSETS
        ])
        self.label_to_label_idx = get_label_to_label_idx(all_gallery_df,
                                                         all_query_df)

        # Also add background label with index 0
        self.label_to_label_idx = {
            k: idx + 1
            for k, idx in self.label_to_label_idx.items()
        }
        bg_label = 'bg_label'
        assert bg_label not in self.label_to_label_idx
        self.label_to_label_idx[bg_label] = 0

        if self.subset in self.GALLERY_SUBSETS:
            gallery_df = self.get_gallery_df(self.subset)
            gallery_df['product_id'] = gallery_df['product_id'].apply(
                lambda label: self.label_to_label_idx[label]
            )
            self.df = gallery_df
        else:
            assert self.subset in self.QUERY_SUBSETS
            query_df = self.get_query_df(self.subset)
            for _, row in query_df.iterrows():
                product_ids = torch.tensor([
                    self.label_to_label_idx[label]
                    for label in row['target']['product_ids']
                ])
                row['target']['product_ids'] = product_ids

            self.df = query_df

    def check_subset(self):
        if self.subset not in [*self.QUERY_SUBSETS, *self.GALLERY_SUBSETS]:
            raise ValueError(
                f'Unsupported subset "{self.subset}"'
            )

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['image'])

        if self.subset == 'val_gallery':
            target = deepcopy(row['product_id'])
            if self.transform is not None:
                img = self.transform(img)
        else:
            target = deepcopy(row['target'])
            if self.transform is not None:
                img, target = self.transform(img, target)

        return img, target

    def init_coco_fields(self):
        self.coco_dataset = {'images': [], 'categories': [], 'annotations': []}
        self.coco_ann_id = 1
        self.coco_categories = set()

    def add_to_coco(self, image, image_id, targets, coco_category_key):
        """
        Based on
        https://github.com/pytorch/vision/blob/5c57f5ec683db58789eaa0bac57447916690422b/references/detection/coco_utils.py#L143
        """
        width, height = F._get_image_size(image)

        self.coco_dataset['images'].append({
            'id': image_id,
            'height': height,
            'width': width,
        })

        bboxes = deepcopy(targets["boxes"])
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()

        img_categories = convert_to_list(targets[coco_category_key])
        areas = convert_to_list(targets['area'])
        iscrowd = convert_to_list(targets['iscrowd'])

        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann['image_id'] = image_id
            ann['bbox'] = bboxes[i]
            ann['category_id'] = img_categories[i]
            self.coco_categories.add(img_categories[i])
            ann['area'] = areas[i]
            ann['iscrowd'] = iscrowd[i]
            ann['id'] = self.coco_ann_id

            self.coco_dataset['annotations'].append(ann)
            self.coco_ann_id += 1

    def get_coco_api(self, coco_category_key):
        self.init_coco_fields()
        for img, target in self:
            image_id = convert_to_item(target['image_id'])
            self.add_to_coco(img, image_id, target, coco_category_key)

        self.coco_dataset['categories'] = [
            {'id': i} for i in sorted(self.coco_categories)
        ]
        coco_ds = COCO()
        coco_ds.dataset = self.coco_dataset

        with redirect_stdout(io.StringIO()):
            coco_ds.createIndex()

        return coco_ds

    def get_query_df(self, subset=None):
        imgs = self.get_query_imgs(subset)
        df_rows = []

        for idx, img in enumerate(imgs):
            with Image.open(img) as im:
                width = im.width
                height = im.height

            df_rows.append({
                'id': idx,
                'image': str(img),
                'width': width,
                'height': height,
                'target': {
                    **self.get_query_img_target(img, subset),
                    'image_id': torch.tensor(idx),
                }
            })

        return pd.DataFrame(df_rows)


def get_label_to_label_idx(gallery_df, query_df):
    q_ids = set(
        i for ids in query_df['target'].apply(
            lambda t: t['product_ids']
        ).values
        for i in ids
    )
    g_ids = set(gallery_df['product_id'].unique())
    all_ids = q_ids.union(g_ids)
    return {
        label: idx
        for idx, label in enumerate(sorted(all_ids))
    }
