import torch
import torch.nn.functional as F
from ..helpers import move_dim

from . import reduction_str


def label_smoothing(
    input,
    target,
    weight=None,
    size_average=None,
    reduce=None,
    reduction="mean",
    ignore_index=-100,
    alpha=0.1,
    num_classes=None,
):
    """Computes the smoothed labels and applies cross-entropy loss.

    The smoothed target label is computed as:
        y_ls = (1 - alpha) * y_hot + alpha / n_classes

    Parameters
    ----------
    input : tensor.Tensor
        (B, C, ...) Network logits
    target : tensor.LongTnsor
        (B, ...) Ground truth label indices
    weight : tensor.Tensor
        (C, ...)
    alpha : float
        Smoothing value in range [0,1).
        If this value is 0, then this is the same as vanilla cross-entropy.
        If this value is 1, then you get a uniform distribution, which
        wouldn't be very useful.
        Defaults to 0.1
    num_classes : int
        [optional] Used to compute smoothing amount.
        if not provided, defaults to the number of classes inferred by the ``input``
        shape.
    """

    assert 0.0 <= alpha < 1.0

    if weight is not None:
        raise NotImplementedError

    reducer = reduction_str(reduction)
    if num_classes is None:
        num_classes = input.shape[1]
    uniform = torch.full_like(input, alpha / num_classes)
    one_hot = F.one_hot(target, num_classes=num_classes)
    one_hot = move_dim(one_hot, -1, 1)
    if ignore_index >= 0:
        uniform[:, ignore_index] = 0
        one_hot[:, ignore_index] = 0
    smooth_target = (1 - alpha) * one_hot + uniform
    log_probs = F.log_softmax(input, dim=1)
    ce = (smooth_target * log_probs).sum(dim=1)
    loss = -reducer(ce)
    return loss
