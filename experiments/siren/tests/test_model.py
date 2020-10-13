import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchtest import assert_vars_change
import numpy as np

import model
from model import SIREN


@pytest.fixture
def rand_inputs():
    return torch.randn(100, 2)


@pytest.fixture
def rand_targets():
    return torch.randn(
        100,
        3,
    )


def test_variables_change(rand_inputs, rand_targets):
    """Basic test just to make sure the graident is making it to all parts
    of the model.
    """

    batch = [rand_inputs, rand_targets]
    model = SIREN()

    # print("Our list of parameters", [np[0] for np in model.named_parameters()])

    # do they change after a training step?
    #  let's run a train step and see
    assert_vars_change(
        model=model,
        loss_fn=F.mse_loss,
        optim=torch.optim.Adam(model.parameters()),
        batch=batch,
        device="cuda:0",
    )

def test_parameterized_fc_2d():
    x = torch.tensor(np.array([
        [1, 2, 3],
        [4, 5, 6],
        ]))  # shape (2, 3)
    weight = torch.tensor(np.array([
        [7],
        [8],
        [9],
        ]))  # shape (3, 1)
    bias = torch.tensor(np.array([
        0.2
        ]))

    layer = model.ParameterizedFC()
    actual = layer(x, weight, bias).numpy()
    assert actual.shape == (2, 1)

    expected = np.array([[50.2], [122.2]])
    assert np.isclose(expected, actual).all()

def test_parameterized_fc_3d():
    x = torch.tensor(np.arange(24).reshape(2,3,4))
    weight = torch.tensor(np.arange(24, 24 + 40).reshape(2, 4, 5))
    bias = torch.tensor(np.array([[0.2] * 5, [0.3] * 5]))

    layer = model.ParameterizedFC()
    actual = layer(x, weight, bias).numpy()
    assert actual.shape == (2, 3, 5)

    expected = np.array([
       [[ 214.2,  220.2,  226.2,  232.2,  238.2],
        [ 718.2,  740.2,  762.2,  784.2,  806.2],
        [1222.2, 1260.2, 1298.2, 1336.2, 1374.2]],

       [[2806.3, 2860.3, 2914.3, 2968.3, 3022.3],
        [3630.3, 3700.3, 3770.3, 3840.3, 3910.3],
        [4454.3, 4540.3, 4626.3, 4712.3, 4798.3]]
    ])

    assert np.isclose(expected, actual).all()
