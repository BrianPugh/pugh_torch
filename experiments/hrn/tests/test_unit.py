import pytest
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from unit import HRNUnit


class DummyCNN(nn.Module):
    def forward(self, x):
        return x


def test_hrnunit_end_to_end():
    dummy_cnn = DummyCNN()

    feat = 27
    unit = HRNUnit(dummy_cnn, feat, 10)

    batch = 2
    channel = 16
    h = 33
    w = 33

    data = torch.rand(batch, channel, h, w)
    y, h = unit(data)

    assert torch.allclose(data, y)
    assert h.shape == (batch, feat)
