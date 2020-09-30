import pytest
import torch
import torch.nn.functional as F
import numpy as np
from pugh_torch.losses import label_smoothing


def test_f_label_smooth_loss_alpha_0():
    pred = torch.Tensor([-1.2, 2.7, 3]).reshape(1, -1, 1, 1).repeat(2, 1, 5, 5)
    y = torch.LongTensor([2]).reshape(1, 1, 1).repeat(2, 5, 5)

    expected = F.cross_entropy(pred, y).numpy()
    actual = label_smoothing(pred, y, alpha=0).numpy()

    assert np.isclose(expected, actual)


def test_f_label_smooth_loss_default():
    pred = torch.Tensor([-1.2, 2.7, 3]).reshape(1, -1, 1, 1).repeat(2, 1, 5, 5)
    y = torch.LongTensor([2]).reshape(1, 1, 1).repeat(2, 5, 5)
    actual = label_smoothing(pred, y)

    assert np.isclose(0.71293252, actual)
