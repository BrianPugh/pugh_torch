import pytest
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from unit import HRNUnit

class DummyCNN(nn.Module):
    def forward(self, x):
        return x

def test_hrnunit():
    dummy_cnn = DummyCNN()

    unit = HRNUnit(dummy_cnn, 300, 10)
