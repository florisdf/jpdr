from pathlib import Path

import pandas as pd
from PIL import Image
import torch

from .detection_dataset import DetectionDataset
import jpdr.utils.boxes as B


class GroZi3kOsokin(DetectionDataset):
    GALLERY_SUBSETS = [
        'val_gallery',
    ]
    QUERY_SUBSETS = [
        'train_query', 'val_old_query',
        'val_new_query',
        'all_query',
    ]

    def __init__(
        self,
        data_path: str,
        transform=None,
        subset: str = 'val_gallery',
    ):
        csv_path = data_path / 'classes/grozi.csv'
        self.df_data = pd.read_csv(csv_path)
        super().__init__(data_path=data_path, transform=transform,
                         subset=subset)

    def __len__(self):
        return len(self.df)

    def get_gallery_df(self, subset=None):
        gallery_imgs_path = self.data_path / 'classes/images'
        imgs = list(gallery_imgs_path.glob('*.jpg'))
        df_rows = []

        for img in imgs:
            product_id = int(img.stem)
            df_rows.append({
                'image': img,
                'product_id': product_id,
            })

        return pd.DataFrame(df_rows)

    def get_df_subset(self, subset):
        if subset == 'val_old_query':
            return self.df_data[
                self.df_data['split'] == 'val-old-cl'
            ]
        elif subset == 'val_new_query':
            return self.df_data[
                self.df_data['split'] == 'val-new-cl'
            ]
        elif subset == 'train_query':
            return self.df_data[
                self.df_data['split'] == 'train'
            ]
        elif subset == 'all_query':
            return self.df_data
        else:
            raise ValueError(f'Unknown subset {subset}')

    def get_query_imgs(self, subset):
        df_subset = self.get_df_subset(subset)
        img_ids = df_subset['imageid'].unique()
        imgs = [
            self.data_path / 'src/3264' / f'{img_id}.jpg'
            for img_id in img_ids
        ]
        assert all(img.exists() for img in imgs)
        return imgs

    def get_query_img_target(self, img: str, subset):
        im = Image.open(img)
        image_id = int(Path(img).stem)
        df_subset = self.get_df_subset(subset)

        df_img = df_subset[df_subset['imageid'] == image_id]
        boxes = df_img[['lx', 'ty', 'rx', 'by']].to_numpy()
        boxes[:, ::2] *= im.width
        boxes[:, 1::2] *= im.height

        product_ids = df_img['classid'].to_numpy()

        boxes = torch.FloatTensor(boxes)
        areas = B.get_area_of_boxes(boxes)

        return {
            'boxes': boxes,
            # label == 1 means "product" (i.e. "non-background")
            'labels': torch.LongTensor([1 for _ in range(len(boxes))]),
            'product_ids': product_ids,
            'area': areas,
            'iscrowd': torch.tensor([False]*len(boxes)),
        }
