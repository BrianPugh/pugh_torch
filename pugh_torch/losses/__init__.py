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


def __noop_loss_fn(*args, **kwargs):
    """For easier configuration"""
    return 0


functional_loss_lookup = {
    "l1_loss": torch.nn.functional.l1_loss,
    "mse_loss": torch.nn.functional.mse_loss,
    "smooth_l1_loss": torch.nn.functional.smooth_l1_loss,
    "huber_loss": torch.nn.functional.smooth_l1_loss,  # aka
    "soft_margin_loss": torch.nn.functional.soft_margin_loss,
    "noop": __noop_loss_fn,
}


def get_functional_loss(s):
    s = s.lower()
    return functional_loss_lookup[s]
