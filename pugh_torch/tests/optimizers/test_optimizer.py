import pytest
import torch
from torch import nn
import pugh_torch as pt
import numpy as np


class SimpleOptimizer(pt.optimizers.Optimizer):
    pass


class SimpleOptimizer2(SimpleOptimizer):
    pass


def test_simple_optimizer():
    """Tests registration"""
    pt.optimizers.get("simpleoptimizer")


def test_simple_optimizer2():
    """Tests child registration"""
    pt.optimizers.get("simpleoptimizer2")
