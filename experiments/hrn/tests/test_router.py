import pytest
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from router import HRN
from unit import HRNUnit


class DummyCNN(nn.Module):
    def forward(self, x):
        return x


def test_hrn_empty():
    torch.manual_seed(0)
    dummy_cnn = DummyCNN()

    feat = 27
    units = [HRNUnit(dummy_cnn, feat, 10) for _ in range(4)]

    hrn = HRN(units)

    batch = 2
    channel = 16
    h = 33
    w = 33

    data = torch.rand(batch, channel, h, w)
    hashes, routes = hrn(data)

    assert hashes.shape == (batch, feat)
    assert isinstance(routes, list)
    assert len(routes) == batch
