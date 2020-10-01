import torch


def reduction_str(s):
    s = s.lower()
    if s == "mean":
        return torch.mean
    elif s == "sum":
        return torch.sum
    elif s == "none":
        return lambda x, *args, **kwargs: x
    else:
        raise ValueError(f"unknown reduction type {s}")


from .hetero_cross_entropy import hetero_cross_entropy
from .label_smoothing import label_smoothing
