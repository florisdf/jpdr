import logging

import torch.cuda
from torch.utils.data import DataLoader
from tqdm import tqdm as _tqdm


def calculate_mean_std_of_dataset(
    dataset, batch_size, num_workers, print_every=10000,
    show_progress=True, aggregate=True
):
    """
    Calculates the mean and std of the dataset.

    Args:
        aggregate: If True, aggregate the means and stds of the images by
            calculating the average, else return the mean and std of each
            image.
    """
    dl = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    means = []
    stds = []

    tqdm = _tqdm if show_progress else lambda it, total: it

    logging.info('Calculating mean and std of the dataset...')
    for i, batch in tqdm(enumerate(dl), total=len(dataset)//batch_size):
        t, *_ = batch
        t_view = t.view(len(t), 3, -1).float()

        if torch.cuda.is_available():
            t_view = t_view.cuda()

        means.extend(t_view.mean(axis=-1))
        stds.extend(t_view.std(axis=-1))

        if print_every is not None and i % print_every == 0:
            print(f'Current means: {torch.stack(means).mean(0)}')
            print(f'Current stds: {torch.stack(stds).mean(0)}')

    means, stds = torch.stack(means), torch.stack(stds)
    return (means.mean(0), stds.mean(0)) if aggregate else (means, stds)
