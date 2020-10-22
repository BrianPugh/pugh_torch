import pytest
import torch
from torch import nn
import torch.nn.functional as F

import pugh_torch as pt


def test_batch_linear():
    data = torch.rand(10, 2)
    feat_in = 2
    feat_out = 4

    linear = pt.modules.meta.BatchLinear(feat_in, feat_out)

    weight = linear.weight.clone()
    bias = linear.bias.clone()

    batch_weight = weight[None,]
    batch_bias = bias[None,]

    vanilla_output = linear(data)
    batch_output = linear(data, weight=batch_weight, bias=batch_bias)

    assert batch_output.shape[0] == 1
    assert vanilla_output.shape == batch_output.shape[1:]

    assert (vanilla_output == batch_output[0]).all()

