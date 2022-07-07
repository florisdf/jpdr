import torch


def convert_to_item(x):
    if isinstance(x, torch.Tensor):
        return x.item()
    else:
        return x


def convert_to_list(x):
    if isinstance(x, torch.Tensor):
        return x.tolist()
    else:
        return x
