from pathlib import Path
import re

import pandas as pd
import torch

from .detection_dataset import DetectionDataset
import jpdr.utils.boxes as B


class GroZi3k(DetectionDataset):
    def __len__(self):
        return len(self.df)

    def get_gallery_df(self):
        imgs = get_imgs_from_txt('TrainingFiles.txt', self.data_path)
        assert all(f.exists() for f in imgs)
        img_names = get_img_names_from_txt('TrainingFiles.txt', self.data_path)

        df_rows = []

        for img, img_name in zip(imgs, img_names):
            class_name = train_img_to_class_name(img)
            product_id = gallery_img_to_product_id(img)
            df_rows.append({
                'image': img,
                'product_id': product_id,
                'class_name': class_name,
            })

        return pd.DataFrame(df_rows)

    def get_query_imgs(self):
        imgs = get_imgs_from_txt('TestFiles.txt', self.data_path)
        assert all(f.exists() for f in imgs)
        return imgs

    def get_query_df(self):
        df = super().get_query_df()
        return df[
            df['target'].apply(lambda t: 'boxes' in t)
        ].reset_index(drop=True)

    def get_query_img_target(self, img: str):
        annot_csv = test_img_to_tonioni_csv(img, self.data_path)
        if not annot_csv.exists():
            return {}

        csv_lines = [line for line in annot_csv.read_text().split('\n')
                     if len(line) > 0]

        boxes = [get_bbox_from_csv_line(line)
                 for line in csv_lines]
        # class_names = [get_class_name_from_csv_line(line)
        #                for line in csv_lines]
        product_ids = [get_product_id_from_csv_line(line)
                       for line in csv_lines]

        boxes = torch.FloatTensor(boxes)
        areas = B.get_area_of_boxes(boxes)

        return {
            'boxes': boxes,
            # label == 1 means "product" (i.e. "non-background")
            'labels': torch.LongTensor([1 for _ in range(len(boxes))]),
            'product_ids': product_ids,
            # 'class_names': class_names,
            'area': areas,
            'iscrowd': torch.tensor([False]*len(boxes)),
        }


def test_img_to_tonioni_csv(img: str, data_path: Path):
    store_no, img_no = re.match(
        r'.*Testing/store(\d)/images/(\d+)\..*', str(img)
    ).groups()
    return data_path / 'bb' / f's{store_no}_{img_no}.csv'


def get_bbox_from_csv_line(csv_line):
    """Return x_min, x_max, y_min, y_max."""
    return tuple(
        map(int,
            re.match(r'.*,\[(.*), (.*)\],.*,\[(.*), (.*)\],.*,', csv_line)
            .groups())
    )


def get_class_name_from_csv_line(csv_line):
    img = get_img_from_csv_line(csv_line)
    return train_img_to_class_name(img)


def get_product_id_from_csv_line(csv_line):
    img = get_img_from_csv_line(csv_line)
    return gallery_img_to_product_id(img)


def gallery_img_to_product_id(img: str):
    return f'{train_img_to_class_name(img)}/{Path(img).stem}'


def train_img_to_class_name(img: str):
    return class_dir_to_class_name(str(Path(img).parent))


def class_dir_to_class_name(class_dir):
    return (
        class_dir.split('Training/')[-1]
        .split('Testing/')[-1]
        .replace('/', '_')
    )


def get_img_from_csv_line(csv_line):
    return re.match(r'([^,]*),.*', csv_line).group(0).replace('//', '/')


def get_img_names_from_txt(imgs_txt, data_path):
    return (data_path / imgs_txt).read_text().split('\n')[:-1]


def get_imgs_from_txt(imgs_txt, data_path):
    img_names = get_img_names_from_txt(imgs_txt, data_path)
    return [data_path / img for img in img_names]
