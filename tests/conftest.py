import pytest

from pathlib import Path
import torch
import torch.nn.functional as F
from jpdr.datasets.detection_dataset import DetectionDataset


@pytest.fixture
def num_train_labels():
    return 15


@pytest.fixture
def num_val_labels():
    return 5


@pytest.fixture
def embedding_dim(num_train_labels, num_val_labels):
    return num_train_labels + num_val_labels


@pytest.fixture
def num_items_per_label():
    return 10


@pytest.fixture
def data_root():
    return 'data'


@pytest.fixture
def data_path(data_root):
    return f'{data_root}/compressed_marb'


@pytest.fixture
def train_imgs(fs, num_train_labels, num_items_per_label, data_path):
    return [
        fs.create_file(f'{data_path}/train/{i}/{i}_{j}.png')
        for i in range(num_train_labels)
        for j in range(num_items_per_label)
    ]


@pytest.fixture
def train_center_radius_files(fs, num_train_labels, num_items_per_label,
                              data_path):
    return [
        fs.create_file(f'{data_path}_center_radius/train/{i}'
                       f'/{i}_{j}_center_radius.csv')
        for i in range(num_train_labels)
        for j in range(num_items_per_label)
    ]


@pytest.fixture
def val_imgs(fs, num_train_labels, num_val_labels, num_items_per_label,
             data_path):
    return [
        fs.create_file(f'{data_path}/val/{i}/{i}_{j}.png',
                       create_missing_dirs=True)
        for i in range(num_train_labels, num_train_labels + num_val_labels)
        for j in range(num_items_per_label)
    ]


@pytest.fixture
def val_center_radius_files(fs, num_train_labels, num_items_per_label,
                            data_path):
    return [
        fs.create_file(f'{data_path}_center_radius/val/{i}'
                       f'/{i}_{j}_center_radius.csv')
        for i in range(num_train_labels)
        for j in range(num_items_per_label)
    ]


@pytest.fixture
def all_imgs(train_imgs, train_center_radius_files, val_imgs,
             val_center_radius_files):
    return [*train_imgs, *val_imgs]


@pytest.fixture
def fake_file_to_stem(fs):
    def _ff_to_stem(fake_file):
        fs.pause()
        stem = Path(fake_file.path).stem
        fs.resume()
        return stem

    return _ff_to_stem


@pytest.fixture
def all_stems(all_imgs, fake_file_to_stem):
    return [
        fake_file_to_stem(fake_file)
        for fake_file in all_imgs
    ]


@pytest.fixture
def corrupt_imgs_file(fs, all_stems):
    fs.pause()
    filename = str(
        Path(__file__).parent.parent
        / 'notebooks/duplicate_regions/corrupt_imgs.txt'
    )
    fs.resume()

    # Arbitrarily mark the first two image as "corrupt"
    corrupt_stems = all_stems[:2]

    return fs.create_file(
        filename,
        contents='\n'.join(corrupt_stems),
        create_missing_dirs=True
    )


@pytest.fixture
def corrupt_stems(corrupt_imgs_file):
    return corrupt_imgs_file.contents.split('\n')


@pytest.fixture
def train_ds(train_imgs, data_root):
    return DetectionDataset(f'{data_root}/', subset="train")


@pytest.fixture
def val_q_ds(val_imgs, data_root):
    return DetectionDataset(f'{data_root}/', subset="val_query")


@pytest.fixture
def val_g_ds(val_imgs, data_root):
    return DetectionDataset(f'{data_root}/', subset="val_gallery")


@pytest.fixture
def num_gallery_refs_per_label():
    return 10


@pytest.fixture
def num_queries_per_label():
    return 20


@pytest.fixture
def single_gallery(num_val_labels, embedding_dim):
    assert embedding_dim >= num_val_labels
    labels = torch.arange(num_val_labels)
    centers = torch.tensor([(0, 0)]*len(labels))
    idxs = torch.arange(len(labels))
    return F.one_hot(labels, embedding_dim).double(), labels, centers, idxs


@pytest.fixture
def double_gallery(single_gallery):
    embeddings, labels, centers, _ = single_gallery

    embeddings = torch.cat([embeddings, embeddings])
    labels = torch.cat([labels, labels])
    centers = torch.cat([centers, centers])
    idxs = torch.arange(len(embeddings))
    return embeddings, labels, centers, idxs


@pytest.fixture
def perfect_query_batch(single_gallery, num_queries_per_label):
    """
    Batch of query embeddings and their labels, such that the distance between
    any positive gallery-query embedding pair will be equal to 0.  The distance
    between any negative gallery-query embedding pair will be equal to sqrt(2).
    """
    embeddings, labels, centers, _ = single_gallery
    labels = labels.repeat(num_queries_per_label)
    centers = centers.repeat(num_queries_per_label, 1)
    embeddings = embeddings.repeat(num_queries_per_label, 1)
    idxs = torch.arange(len(labels))
    return embeddings, labels, centers, idxs


@pytest.fixture
def hardest_positive_query_batch(perfect_query_batch):
    """
    Batch of query embeddings and their labels, such that the distance between
    any positive gallery-query embedding pair will be equal to 2. The distance
    between any negative gallery-query embedding pair will be equal to sqrt(2).
    """
    embeddings, labels, centers, idxs = perfect_query_batch
    embeddings *= -1
    return embeddings, labels, centers, idxs


@pytest.fixture
def hardest_negative_query_batch(perfect_query_batch, num_val_labels):
    """
    Batch of query embeddings and their labels, such that the query embedding
    of each label will have a distance of 0 to the gallery embedding of another
    label. The distance between any positive and any other negative
    gallery-query embedding pair will be equal to sqrt(2).
    """
    embeddings, labels, centers, idxs = perfect_query_batch
    embeddings = embeddings.roll(1, 1)
    last_col = embeddings[:, num_val_labels].clone()
    first_col = embeddings[:, 0].clone()
    embeddings[:, num_val_labels] = first_col
    embeddings[:, 0] = last_col
    return embeddings, labels, centers, idxs
