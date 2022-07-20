from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jpdr.datasets import GroZi3kTonioni, GroZi3kOsokin, Tankstation
import jpdr.transforms as T
from torch.utils.data import DataLoader


def get_tonioni_datasets(
    train_tfm, val_tfm
):
    data_path = Path(__file__).parent / 'data' / 'GroceryProducts_Tonioni'
    ds_train = GroZi3kTonioni(
        data_path,
        subset='val_query',
        transform=train_tfm,
    )
    ds_val = GroZi3kTonioni(
        data_path,
        subset='val_query',
        transform=val_tfm,
    )
    return ds_train, ds_val


def get_osokin_datasets(
    train_tfm, val_tfm
):
    data_path = Path(__file__).parent / 'data' / 'GroceryProducts_Osokin'
    ds_train = GroZi3kOsokin(
        data_path,
        subset='all_query',
        transform=train_tfm,
    )
    ds_val = GroZi3kOsokin(
        data_path,
        subset='all_query',
        transform=val_tfm,
    )
    return ds_train, ds_val


def get_tankstation_datasets(
    train_tfm, val_tfm
):
    data_path = Path(__file__).parent / 'data' / 'Tankstation'
    ds_train = Tankstation(
        data_path,
        subset='val_query',
        transform=train_tfm,
    )
    ds_val = Tankstation(
        data_path,
        subset='val_query',
        transform=val_tfm,
    )
    return ds_train, ds_val


DATASET_REGISTRY = {
    'tonioni': (
        get_tonioni_datasets,
        dict(mean=[0.4643, 0.4087, 0.3457],
             std=[0.2831, 0.2730, 0.2695])
    ),
    'osokin': (
        get_osokin_datasets,
        dict(mean=[0.4643, 0.4087, 0.3457],
             std=[0.2831, 0.2730, 0.2695])
    ),
    'tankstation': (
        get_tankstation_datasets,
        dict(mean=[0.485, 0.456, 0.406],
             std=[0.229, 0.224, 0.225])
    ),
}


def get_tfms(img_size_det, norm_dict=None, train_crop_how="random"):
    train_tfms = [
        T.RandomShortestSize(int(1.2*img_size_det)),
        T.Crop(img_size_det, img_size_det, how=train_crop_how,
               min_new_tgt_area_pct=0.25),
        T.ToTensor(),
    ]
    val_tfms = [
        T.RandomShortestSize(int(1.2*img_size_det)),
        T.Crop(img_size_det, img_size_det, how="center"),
        T.ToTensor(),
    ]

    if norm_dict is not None:
        train_tfms.append(T.Normalize(**norm_dict))
        val_tfms.append(T.Normalize(**norm_dict))

    train_tfm = DetCompose(train_tfms)
    val_tfm = DetCompose(val_tfms)

    return train_tfm, val_tfm


def get_dataset_from_name(name, img_size_det, normalize=True,
                          train_crop_how="random"):
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f'Unknown dataset "{name}"'
        )
    else:
        get_datasets, norm_dict = DATASET_REGISTRY[name]

    if normalize:
        train_tfm, val_tfm = get_tfms(img_size_det, norm_dict, train_crop_how)
    else:
        train_tfm, val_tfm = get_tfms(img_size_det, None, train_crop_how)

    return get_datasets(train_tfm, val_tfm)


def get_dataloaders(
    name,
    img_size_det=800,
    batch_size_recog=2,
    batch_size_det=2,
    val_batch_size=2,
    num_workers=8,
    seed=15,
    k=5,
    val_fold=0,
    normalize=True,
    shuffle_train=True,
    train_crop_how="random",
):
    ds_train, ds_val = get_dataset_from_name(name, img_size_det, normalize,
                                             train_crop_how)
    ds_train, ds_val = k_fold_trainval_split(ds_train, ds_val, k=k,
                                             val_fold=val_fold, seed=seed)

    dl_train_recog = DataLoader(ds_train, batch_size=batch_size_recog,
                                shuffle=shuffle_train,
                                num_workers=num_workers,
                                collate_fn=collate_fn_detection)
    dl_train_det = DataLoader(ds_train, batch_size=batch_size_det,
                              shuffle=shuffle_train,
                              num_workers=num_workers,
                              collate_fn=collate_fn_detection)

    dl_val = DataLoader(ds_val, batch_size=val_batch_size,
                        num_workers=num_workers,
                        shuffle=False,
                        drop_last=False,
                        collate_fn=collate_fn_detection)

    return dl_train_det, dl_train_recog, dl_val


class DetCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


def collate_fn_detection(data):
    transposed = list(zip(*data))

    x = torch.stack(transposed[0])
    target = transposed[1]

    return x, target


def k_fold_trainval_split(ds_train, ds_val, k=5, val_fold=0, seed=15):
    """
    Shuffle the data in `ds_train`, split it up into into `k` folds and assign
    `k-1` folds to the training dataset and 1 fold to the validation dataset.
    Which fold is used for validation, is determined by `val_fold`. The random
    state used to shuffle the data is set by `seed`.

    Assumes that `ds_train` and `ds_val` are instances of the same class and
    initially also contain the same data. The dataframes inside the datasets
    are changed by this function to contain the train, resp. validation, data.
    """
    assert val_fold < k
    folds = np.array_split(
        ds_train.df.sample(frac=1.0, random_state=seed),
        k
    )
    df_val = folds.pop(val_fold)
    df_train = pd.concat(folds)
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    ds_train.df = df_train
    ds_val.df = df_val
    return ds_train, ds_val
