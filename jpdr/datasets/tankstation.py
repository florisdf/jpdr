import pandas as pd
import xmltodict
import torch
from pathlib import Path

from .detection_dataset import DetectionDataset


class Tankstation(DetectionDataset):
    def __len__(self):
        return len(self.df)

    def get_query_imgs(self):
        return sorted(self.data_path.glob('query/*.jpg'))

    def get_gallery_df(self):
        df_rows = []
        labels = set()

        for img_path in self.data_path.glob('gallery/*/*.jpg'):
            product_id = img_path.parent.name
            if product_id not in labels:
                labels.add(product_id)
                df_rows.append({
                    'image': str(img_path),
                    'product_id': product_id,
                })

        return pd.DataFrame(df_rows)

    def get_query_img_target(self, img: Path):
        xml_path = img.parent / f'{img.stem}.xml'
        xml_dict = xmltodict.parse(xml_path.open().read())

        if not isinstance(xml_dict['annotation']['object'], list):
            xml_dict['annotation']['object'] = [
                xml_dict['annotation']['object']
            ]

        boxes = []
        product_ids = []

        for o in xml_dict['annotation']['object']:
            box = o['bndbox']
            boxes.append([float(box['xmin']), float(box['ymin']),
                          float(box['xmax']), float(box['ymax'])])
            product_ids.append(o['name'])

        boxes = torch.FloatTensor(boxes)
        areas = (boxes[:, 2] - boxes[:, 0])*(boxes[:, 3] - boxes[:, 1])

        return {
            'boxes': boxes,
            # label == 1 means "product" (i.e. "non-background")
            'labels': (
                torch.LongTensor([1 for _ in range(len(boxes))])
            ),
            'product_ids': product_ids,
            'area': areas,
            'iscrowd': torch.tensor([False]*len(boxes)),
        }
