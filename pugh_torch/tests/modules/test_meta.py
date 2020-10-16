import pytest
import torch
from torch import nn
import torch.nn.functional as F

import pugh_torch as pt


def test_batch_linear():
    # TODO
    feat_in = 2
    feat_out = 4
    layer = pt.modules.meta.BatchLinear(feat_in, feat_out)
