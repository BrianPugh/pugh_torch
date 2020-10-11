import pytest
import torch
from torch import nn
import pugh_torch as pt
import numpy as np


class SimpleInheritedReLU(nn.ReLU, pt.modules.ActivationModule):
    pass


class SimpleInheritedRelUWithInit(nn.ReLU, pt.modules.ActivationModule):
    @torch.no_grad()
    def init_layer(self, m):
        if type(m) == nn.Linear:
            m.weight.fill_(1.0)

    @torch.no_grad()
    def init_first_layer(self, m):
        if type(m) == nn.Linear:
            m.weight.fill_(2.0)


def test_activation_factory_function():
    ones = torch.ones(2)
    fn = pt.modules.Activation("simpleinheritedrelu")
    assert (fn(ones) == 1).all()
    assert (fn(-ones) == 0).all()


def test_activation_factory_function_init_layer_module():
    ones = torch.ones(2)
    fc = nn.Linear(2, 2)
    fn = pt.modules.Activation("simpleinheritedreluwithinit", fc)
    assert (fc.weight == 1).all()


def test_activation_factory_function_init_layer_list():
    ones = torch.ones(2)
    fc = nn.Linear(2, 2)
    fn = pt.modules.Activation("simpleinheritedreluwithinit", [fc])
    assert (fc.weight == 1).all()


def test_activation_factory_function_init_first_layer():
    ones = torch.ones(2)
    fc = nn.Linear(2, 2)
    fn = pt.modules.Activation("simpleinheritedreluwithinit", fc, first=True)
    assert (fc.weight == 2).all()


def test_sine_fn():
    input = np.arange(0, 5, 100)
    expected = np.sin(input)
    input = torch.Tensor(input)
    fn = pt.modules.Activation("sine")
    actual = fn(input).numpy()
    assert np.isclose(expected, actual).all()


def test_sine_first_layer():
    input = np.arange(0, 5, 100)
    expected = np.sin(input)
    fc = nn.Linear(2, 2)
    fn = pt.modules.Activation("sine", fc, first=True)


def test_sine_layer():
    input = np.arange(0, 5, 100)
    expected = np.sin(input)
    fc = nn.Linear(2, 2)
    fn = pt.modules.Activation("sine", fc)
