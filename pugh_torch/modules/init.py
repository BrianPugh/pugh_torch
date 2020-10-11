"""
Equivalent names:
    * he == kaming
    * xavier == glorot
Rules of thumb collected from various sources:
    * Use He for ReLU
    * Use xavier for tanh
"""

import torch
from torch import nn


def xavier(m):
    if hasattr(m, "weight"):
        nn.init.xavier_uniform_(m.weight)


def he(m, mode="fan_in", **kwargs):
    if hasattr(m, "weight"):
        nn.init.kaiming_uniform_(m.weight, mode="fan_in", **kwargs)
